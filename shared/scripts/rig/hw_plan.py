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
STATES = ("queued", "running", "done", "failed", "cancelled")
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
            return {"schema_version": SCHEMA_VERSION, "controller": None, "items": []}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("schema_version") != SCHEMA_VERSION:
            raise PlanError("state", f"unsupported plan schema {data.get('schema_version')!r}")
        return data

    def _write(self, data: dict[str, Any]) -> None:
        finished = [i for i in data["items"] if i["state"] in ("done", "failed", "cancelled")]
        drop = {i["id"] for i in finished[:-KEEP_FINISHED]} if len(finished) > KEEP_FINISHED else set()
        data["items"] = [i for i in data["items"] if i["id"] not in drop]
        data["updated_at"] = _iso(_now())
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
                "next": [i for i in data["items"] if i["state"] in ("queued", "running")]}

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
        """Mark the first queued item running and return it (None when nothing is queued or one runs)."""
        with self.lock:
            data = self._read()
            if any(i["state"] == "running" for i in data["items"]):
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
        """After an executor restart: items left running did not finish; mark them failed."""
        with self.lock:
            data = self._read()
            stuck = [i for i in data["items"] if i["state"] == "running"]
            for i in stuck:
                i.update(state="failed", ended_at=_iso(_now()), result="executor restarted while running")
            if stuck:
                self._write(data)
            return [i["id"] for i in stuck]
