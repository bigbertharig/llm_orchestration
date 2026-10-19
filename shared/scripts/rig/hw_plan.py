"""The rig's plan: the active controller and the queue of what to do next, in one rig-local file.

`rig observe` reports "now" (GPUs, containers, profiles, leases, guard) and, from this file, "next":
the queue a controller wrote. The rig works through the queue on its own (rig-plan.service), one
item at a time, then waits; loaded models stay loaded unless an item unloads them. So a controller
can disappear and the rig keeps going; a restarted or second controller reads the same file and
picks up. This mirrors the orchestration task queue (queue / processing / complete / failed) in a
single file.

Controller record: one active controller at a time. It heartbeats; another controller may take over
when there is none, when the active one's heartbeat is older than STALE_AFTER_S, or by explicit
handoff (force). Plan edits from a controller that is not active are refused.

Shared report (2026-10-18): the rig and the active controller each keep a copy of this document.
The rig writes it (every write bumps `version`); the controller saves what `rig observe` returns to its
own disk on every heartbeat. If either side crashes it recovers from the other:
- the rig reboots: on the first start of a new boot (`boot_id` changes) the item that was running is
  marked `interrupted` and the plan is held (`hold`); the rig boots neutral and runs nothing until the
  active controller has looked at the report and resumes it (`resume`). A lost network without a reboot
  is not a hold: the rig finishes what it was given.
- the rig lost the file (disk, replacement): the controller's copy has a higher version, and the
  controller writes it back (`restore`), still held, then resumes.
- the controller crashed: it restarts from its own copy and reads the rig's live one.

Item: {id, kind, args, state, added_by, added_at, started_at, ended_at, result}. Kinds:
  load    args {profile, gpus, port, lease, cache_ram}   -> `rig start remote` with that placement
  unload  args {profile}                                  -> `rig stop <profile>`
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from filelock import FileLock

SCHEMA_VERSION = 1
STALE_AFTER_S = 120
KINDS = ("load", "unload")
STATES = ("queued", "running", "done", "failed", "cancelled", "interrupted")
FINISHED = ("done", "failed", "cancelled", "interrupted")
KEEP_FINISHED = 50  # finished items kept in the file, newest last


class PlanError(RuntimeError):
    """kind: invalid | not_controller | not_found | state"""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(t: datetime) -> str:
    return t.isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


class Plan:
    def __init__(self, root: Path):
        self.path = root / "plan.json"
        root.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(self.path) + ".lock", timeout=10)

    # ---------------------------------------------------------------- file

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": SCHEMA_VERSION, "controller": None, "items": [], "version": 0, "hold": None}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("schema_version") != SCHEMA_VERSION:
            raise PlanError("state", f"unsupported plan schema {data.get('schema_version')!r}")
        return data

    def _write(self, data: dict[str, Any]) -> None:
        finished = [i for i in data["items"] if i["state"] in FINISHED]
        drop = {i["id"] for i in finished[:-KEEP_FINISHED]} if len(finished) > KEEP_FINISHED else set()
        data["items"] = [i for i in data["items"] if i["id"] not in drop]
        data["updated_at"] = _iso(_now())
        data["version"] = int(data.get("version") or 0) + 1
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def report(self) -> dict[str, Any]:
        data = self._read()
        ctl = data.get("controller")
        if ctl:
            beat = _parse(ctl.get("heartbeat_at"))
            ctl = {**ctl, "stale": beat is None or (_now() - beat).total_seconds() > STALE_AFTER_S}
        return {"controller": ctl, "items": data["items"],
                "next": [i for i in data["items"] if i["state"] in ("queued", "running")],
                "version": int(data.get("version") or 0), "updated_at": data.get("updated_at"),
                "hold": data.get("hold"), "boot_id": data.get("rig_boot_id"), "booted_at": data.get("booted_at"),
                "resumed": data.get("resumed"), "restored": data.get("restored")}

    # ---------------------------------------------------------------- controller

    def _require_controller(self, data: dict[str, Any], controller: str) -> None:
        ctl = data.get("controller")
        if not ctl or ctl.get("id") != controller:
            active = ctl.get("id") if ctl else "none"
            raise PlanError("not_controller", f"controller {controller!r} is not active (active: {active}); "
                                              "take control first")

    def claim(self, controller: str, host: str, force: bool = False) -> dict[str, Any]:
        """Become the active controller: when there is none, it is us, it is stale, or force (handoff)."""
        controller = str(controller or "").strip()
        if not controller:
            raise PlanError("invalid", "controller id is required")
        with self.lock:
            data = self._read()
            ctl = data.get("controller")
            now = _now()
            if ctl and ctl.get("id") != controller and not force:
                beat = _parse(ctl.get("heartbeat_at"))
                if beat is not None and (now - beat).total_seconds() <= STALE_AFTER_S:
                    raise PlanError("not_controller", f"controller {ctl['id']} ({ctl.get('host')}) is active, "
                                                      f"last heartbeat {ctl.get('heartbeat_at')}; use a handoff")
            previous = ctl.get("id") if ctl and ctl.get("id") != controller else None
            since = ctl["since"] if ctl and ctl.get("id") == controller else _iso(now)
            data["controller"] = {"id": controller, "host": host, "since": since, "heartbeat_at": _iso(now),
                                  **({"took_over_from": previous} if previous else {})}
            self._write(data)
            return data["controller"]

    def heartbeat(self, controller: str, host: str) -> dict[str, Any]:
        with self.lock:
            data = self._read()
            self._require_controller(data, controller)
            data["controller"].update(heartbeat_at=_iso(_now()), host=host)
            self._write(data)
            return data["controller"]

    def release(self, controller: str) -> None:
        with self.lock:
            data = self._read()
            self._require_controller(data, controller)
            data["controller"] = None
            self._write(data)

    def require_active(self, controller: str | None) -> None:
        """For other rig writes (leases, claims): refuse a controller that is not the active one.
        With no controller registered at all, writes stay open (direct operator use, offline lanes)."""
        data = self._read()
        if data.get("controller") is None:
            return
        if not controller:
            raise PlanError("not_controller", f"controller {data['controller']['id']} is active; "
                                              "pass --controller with its id")
        self._require_controller(data, controller)

    # ---------------------------------------------------------------- items

    def add(self, controller: str, kind: str, args: dict[str, Any]) -> dict[str, Any]:
        if kind not in KINDS:
            raise PlanError("invalid", f"kind must be one of {KINDS}")
        if not args.get("profile"):
            raise PlanError("invalid", "args.profile is required")
        with self.lock:
            data = self._read()
            self._require_controller(data, controller)
            item = {"id": f"{kind}-{uuid.uuid4().hex[:12]}", "kind": kind, "args": args, "state": "queued",
                    "added_by": controller, "added_at": _iso(_now()), "started_at": None, "ended_at": None,
                    "result": None}
            data["items"].append(item)
            self._write(data)
            return item

    def cancel(self, controller: str, item_id: str) -> dict[str, Any]:
        with self.lock:
            data = self._read()
            self._require_controller(data, controller)
            item = next((i for i in data["items"] if i["id"] == item_id), None)
            if item is None:
                raise PlanError("not_found", f"no plan item {item_id}")
            if item["state"] != "queued":
                raise PlanError("state", f"{item_id} is {item['state']}; only queued items can be cancelled")
            item.update(state="cancelled", ended_at=_iso(_now()), result="cancelled by " + controller)
            self._write(data)
            return item

    def get(self, item_id: str) -> dict[str, Any] | None:
        return next((i for i in self._read()["items"] if i["id"] == item_id), None)

    # ---------------------------------------------------------------- executor side

    def take_next(self) -> dict[str, Any] | None:
        """Mark the first queued item running and return it (None when nothing is queued, one runs, or the
        plan is held after a reboot)."""
        with self.lock:
            data = self._read()
            if data.get("hold") or any(i["state"] == "running" for i in data["items"]):
                return None
            item = next((i for i in data["items"] if i["state"] == "queued"), None)
            if item is None:
                return None
            item.update(state="running", started_at=_iso(_now()))
            self._write(data)
            return dict(item)

    def finish(self, item_id: str, ok: bool, result: str) -> None:
        with self.lock:
            data = self._read()
            item = next((i for i in data["items"] if i["id"] == item_id), None)
            if item is None:
                return
            item.update(state="done" if ok else "failed", ended_at=_iso(_now()), result=result[-2000:])
            self._write(data)

    def recover(self) -> list[str]:
        """After an executor restart in the same boot: items left running did not finish."""
        with self.lock:
            data = self._read()
            stuck = _interrupt(data, "the plan executor restarted while it ran")
            if stuck:
                self._write(data)
            return stuck

    def boot(self, boot_id: str) -> dict[str, Any] | None:
        """First call in a new boot: interrupt what was running and hold the plan until the controller
        resumes it. Later calls in the same boot do nothing. Returns the new hold, or None."""
        with self.lock:
            data = self._read()
            if data.get("rig_boot_id") == boot_id:
                return None
            stuck = _interrupt(data, "the rig restarted while it ran")
            now = _iso(_now())
            data.update(rig_boot_id=boot_id, booted_at=now,
                        hold={"reason": "boot", "since": now, "interrupted": stuck})
            self._write(data)
            return data["hold"]

    def resume(self, controller: str) -> dict[str, Any]:
        """The active controller has seen the report: clear the hold and let the queue run again."""
        with self.lock:
            data = self._read()
            self._require_controller(data, controller)
            held = data.get("hold")
            data["hold"] = None
            data["resumed"] = {"by": controller, "at": _iso(_now()), "hold": held}
            self._write(data)
            return self.report_of(data)

    def restore(self, controller: str, doc: dict[str, Any]) -> dict[str, Any]:
        """Write back the controller's copy when the rig's is behind (lost or replaced file). Refused when
        the rig's copy is as new or newer. The plan stays held; the controller resumes it after."""
        with self.lock:
            try:
                data = self._read()
            except (PlanError, ValueError):  # unreadable file: anything is newer
                data = {"schema_version": SCHEMA_VERSION, "controller": None, "items": [], "version": 0}
            ctl = data.get("controller")
            if ctl and ctl.get("id") != controller:
                raise PlanError("not_controller", f"controller {ctl['id']} is active; only it restores the plan")
            mine, theirs = int(data.get("version") or 0), int(doc.get("version") or 0)
            if theirs <= mine:
                raise PlanError("state", f"the rig's plan is current (version {mine} >= {theirs}); nothing restored")
            restored = {"schema_version": SCHEMA_VERSION, "controller": doc.get("controller") or ctl,
                        "items": [dict(i) for i in doc.get("items") or []], "version": theirs,
                        "rig_boot_id": data.get("rig_boot_id"), "booted_at": data.get("booted_at")}
            for key in ("stale",):
                (restored["controller"] or {}).pop(key, None)
            stuck = _interrupt(restored, "it was running when the rig lost its plan (restored from the controller)")
            restored["hold"] = data.get("hold") or {"reason": "restored", "since": _iso(_now()), "interrupted": stuck}
            restored["restored"] = {"by": controller, "at": _iso(_now()), "from_version": theirs, "rig_version": mine}
            self._write(restored)
            return self.report_of(restored)

    def report_of(self, data: dict[str, Any]) -> dict[str, Any]:
        with_ctl = data.get("controller")
        return {"version": int(data.get("version") or 0), "hold": data.get("hold"),
                "controller": with_ctl, "items": data.get("items", [])}


def _interrupt(data: dict[str, Any], why: str) -> list[str]:
    """Mark running items interrupted, saying what they were doing; returns their ids."""
    stuck = [i for i in data["items"] if i["state"] == "running"]
    for i in stuck:
        what = f"{i['kind']} {i['args'].get('profile', '')}".strip()
        i.update(state="interrupted", ended_at=_iso(_now()), result=f"interrupted: {why} ({what})")
    return [i["id"] for i in stuck]
