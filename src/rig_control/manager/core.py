"""The rig manager: controller management (docs/MANAGER.md).

It decides who holds which GPUs and what the rig does next. It keeps no state of its own: the lease
and claim record, the active-controller record and the plan queue all live on the rig
(store.RigRecord), and the rig works through the plan by itself (rig-plan.service) and enforces the
hardware rules (one load at a time, health, power, thermal, RAM). So the rig keeps going if this
controller goes away, and this or another controller picks up from `rig observe`.

Shared report (rule 1c): the rig keeps the record and the plan; this controller keeps a copy on its own
disk, saved on every observation. When the rig reboots it holds its plan; the reconciler reviews it
(restoring the plan from the copy if the rig's is behind), cancels stale queued model lifecycle work,
and resumes it for fresh requests. When the rig is unreachable, status answers from the copy and says
since when.

One controller is active at a time. This one takes control when there is none or the active one's
heartbeat is stale, heartbeats while it holds it, and is read-only (writes refused with 409) while
another controller is active. A handoff is an explicit takeover (`take_control(force=True)`).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from collections import Counter, deque
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from ..rig import Rig, RigUnavailable
from . import fit, memory, models, observations, placement
from .errors import ManagerError
from .status import gpu_rows, is_expired
from .store import RigRecord

log = logging.getLogger("rig_control.manager")

OWNER = "orchestration"  # owner of the leases the manager takes for interactive profiles
INTERACTIVE_TTL_S = 300
RECONCILE_EVERY_S = 30
OBSERVE_CACHE_S = 3.0
LOAD_POLL_S = 3.0
DONE_NOT_READY_CHECKS = 3  # observations of a done load before it counts as not serving
# A lease of ours with no rig profile or plan item behind it is released after this long (covers the
# moment between taking the lease and the rig recording the load).
ORPHAN_GRACE_S = 60
# A split load that waits for its cards reserves them with a claim this long, renewed while it waits
# (docs/MANAGER.md, "Reservations"): if this controller goes away mid-wait, the reservation lapses.
WAIT_CLAIM_TTL_S = 180
WAIT_POLL_S = 10.0
RUN_ID = re.compile(r"[A-Za-z0-9_.-]{1,64}")


def _limit(wait_s: float) -> str:
    return "with no time limit" if wait_s < 0 else f"up to {wait_s:.0f} s"


class Manager:
    def __init__(self, rig: Rig, start_timeout_s: float = 900, controller_id: str = "controller",
                 mirror_path: Path | None = None):
        self.rig = rig
        self.controller_id = controller_id
        self.record = RigRecord(rig, controller_id)
        self.start_timeout_s = start_timeout_s
        self.in_control = False
        self.control_note = "not claimed yet"
        self._obs: dict | None = None
        self._obs_at = 0.0
        self._obs_lock = asyncio.Lock()
        # This controller makes one placement decision at a time; the rig refuses overlaps across controllers.
        self._place_lock = asyncio.Lock()
        # Leases a load or unload of this manager is working on right now. The reconciler leaves them alone
        # for the whole operation: between taking a lease and the rig recording the profile, and between the
        # rig unloading a profile and this manager releasing its lease, the lease looks unbacked.
        self._in_flight: set[str] = set()
        # Waits in progress (collecting cards, waiting for named cards): who waits, for which card sizes. They
        # are the demand placement looks at before putting a small model on a bigger card (MANAGER.md, "Runs").
        self._waits: dict[str, dict] = {}
        # Runs already reported (run, state, ended_at), and the interruption each auto-resumed run was resumed for.
        self._runs_seen: set[tuple] = set()
        self._auto_resumed: dict[str, str] = {}
        # This controller's copy of the rig's report, and what it has seen happen to the rig.
        self.mirror_path = mirror_path
        self.mirror: dict | None = self._load_mirror()
        self.rig_down_since: float | None = None
        self.events: deque[dict] = deque(maxlen=50)

    # ------------------------------------------------------------------ control

    async def take_control(self, force: bool = False) -> bool:
        """Become the rig's active controller (force: explicit handoff from another one)."""
        try:
            ctl = await self.record.claim_control(force)
        except ManagerError as exc:
            if exc.extra.get("rig_kind") != "not_controller":
                raise
            self.in_control, self.control_note = False, str(exc)
            return False
        self.in_control = True
        self.control_note = f"active since {ctl.get('since')}" + (
            f" (took over from {ctl['took_over_from']})" if ctl.get("took_over_from") else "")
        log.info("controller %s: %s", self.controller_id, self.control_note)
        self._obs = None
        return True

    async def release_control(self) -> None:
        if self.in_control:
            await self.record.release_control()
            self.in_control, self.control_note = False, "released"

    async def _require_control(self) -> None:
        if self.in_control or await self.take_control():
            return
        raise ManagerError(409, f"read-only: {self.control_note}. This controller is {self.controller_id!r}; "
                                "hand off with POST /v1/manager/controller/claim {\"force\": true}")

    def control(self, obs: dict | None = None) -> dict:
        active = ((obs or self._obs or {}).get("plan") or {}).get("controller")
        return {"this": self.controller_id, "in_control": self.in_control, "note": self.control_note,
                "active": active}

    # ------------------------------------------------------------------ observation

    async def observe(self, fresh: bool = False) -> dict:
        async with self._obs_lock:
            if fresh or self._obs is None or time.monotonic() - self._obs_at > OBSERVE_CACHE_S:
                try:
                    obs = await self.rig.observe()
                except RigUnavailable as exc:
                    if self.rig_down_since is None:
                        self.rig_down_since = time.time()
                        self._event(f"rig unreachable: {str(exc)[:200]}")
                    raise
                if self.rig_down_since is not None:
                    self._event(f"rig back after {time.time() - self.rig_down_since:.0f} s")
                    self.rig_down_since = None
                self._note_boot(obs)
                self._save_mirror(obs)
                self._obs = obs
                self._obs_at = time.monotonic()
            return self._obs

    # ------------------------------------------------------------------ the copy of the rig's report

    def _event(self, text: str) -> None:
        log.warning("rig: %s", text)
        self.events.append({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": text})

    def _load_mirror(self) -> dict | None:
        if not self.mirror_path or not self.mirror_path.is_file():
            return None
        try:
            return json.loads(self.mirror_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("rig report copy unreadable (%s); starting without it", exc)
            return None

    def mirror_plan(self) -> dict:
        return (self.mirror or {}).get("plan") or {}

    def _note_boot(self, obs: dict) -> None:
        new = (obs.get("plan") or {}).get("boot_id")
        old = self.mirror_plan().get("boot_id")
        if new and old and new != old:
            self._event(f"rig rebooted (booted {(obs.get('plan') or {}).get('booted_at')})")

    def _save_mirror(self, obs: dict) -> None:
        """Keep the latest report, except the plan when the rig's copy is behind ours (it lost its file):
        then ours is kept until the reconciler has written it back."""
        plan = obs.get("plan") or {}
        keep = plan
        mine = self.mirror_plan()
        if int(mine.get("version") or 0) > int(plan.get("version") or 0):
            keep = mine
        self.mirror = {"saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "observed_at": obs.get("observed_at"), "rig_plan_version": plan.get("version"),
                       "plan": keep, **{k: obs.get(k) for k in ("mode", "gpu_scan", "gpus", "profiles", "leases", "claims",
                                                                "guard", "containers")}}
        if not self.mirror_path:
            return
        try:
            self.mirror_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.mirror_path.with_name(f".{self.mirror_path.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(self.mirror, indent=1) + "\n", encoding="utf-8")
            tmp.replace(self.mirror_path)
        except OSError as exc:
            log.warning("could not save the rig report copy: %s", exc)

    async def settle_hold(self, obs: dict) -> dict | None:
        """The rig rebooted (or lost its plan) and holds its queue: write back our copy if the rig's is
        behind, report what was interrupted, and resume. Interrupted loads are not retried: their callers
        got the failure, and their leases are released below as unbacked. A boot also cancels model
        lifecycle requests that were still queued before the reboot, so reconnecting the controller cannot
        turn an intentionally empty rig boot into an implicit model load."""
        plan = obs.get("plan") or {}
        hold = plan.get("hold")
        if not hold:
            return None
        mine = self.mirror_plan()
        restored = None
        if int(mine.get("version") or 0) > int(plan.get("version") or 0):
            restored = await self.record.plan_restore(mine)
            self._event(f"rig plan was behind (version {plan.get('version')} < {mine.get('version')}); "
                        f"wrote back this controller's copy")
        current_plan = restored or plan
        items = {i["id"]: i for i in current_plan.get("items", [])}
        stopped = [f"{iid}: {items.get(iid, {}).get('result') or 'interrupted'}" for iid in hold.get("interrupted", [])]
        cancelled = []
        if hold.get("reason") == "boot":
            for item in current_plan.get("items", []):
                if item.get("state") == "queued" and item.get("kind") in {"load", "unload"}:
                    await self.record.plan_cancel(item["id"])
                    cancelled.append(item["id"])
        await self.record.plan_resume()
        self._obs = None
        runs = hold.get("runs_interrupted") or []
        self._event(f"plan held ({hold.get('reason')}) since {hold.get('since')}; resumed"
                    + (f"; interrupted, not retried: {'; '.join(stopped)}" if stopped else "; nothing interrupted")
                    + (f"; stale model requests cancelled: {', '.join(cancelled)}" if cancelled else "")
                    + (f"; runs interrupted: {', '.join(runs)} (rigctl recover RUN or rigctl dismiss RUN)" if runs else ""))
        return {"hold": hold, "restored": bool(restored), "interrupted": stopped, "cancelled": cancelled}

    def rig_report(self) -> dict:
        """Reachability and recent events, for status."""
        return {"reachable": self.rig_down_since is None,
                "down_since": (datetime.fromtimestamp(self.rig_down_since, timezone.utc).isoformat(timespec="seconds")
                               if self.rig_down_since else None),
                "copy_saved_at": (self.mirror or {}).get("saved_at"), "events": list(self.events)[-10:],
                "waits": [dict(w) for w in self._waits.values()]}

    async def leases(self) -> list[dict]:
        return (await self.observe(fresh=True)).get("leases", [])

    async def claims(self) -> list[dict]:
        return (await self.observe(fresh=True)).get("claims", [])

    async def plan(self) -> dict:
        return (await self.observe(fresh=True)).get("plan") or {}

    async def status(self) -> dict:
        """The single report: what the rig is doing now (GPUs, profiles, guard) and next (plan). With the
        rig unreachable, the last copy this controller kept, marked as such."""
        try:
            obs = await self.observe()
        except RigUnavailable:
            if not self.mirror:
                raise
            m = self.mirror
            return {"observed_at": m.get("observed_at"), "mode": m.get("mode"), "summary": {"rig": "unreachable"},
                    "controller": self.control(), "guard": m.get("guard"),
                    "gpu_scan": {k: (m.get("gpu_scan") or {}).get(k) for k in ("ok", "ids", "errors")},
                    "gpus": [], "profiles": m.get("profiles", {}),
                    "plan": m.get("plan") or {}, "jobs": [], "leases": m.get("leases", []),
                    "claims": m.get("claims", []), "rig": self.rig_report(), "from_copy": True}
        leases, claims = obs.get("leases", []), obs.get("claims", [])
        rows = gpu_rows(obs, leases, claims)
        counts: dict[str, int] = {}
        for r in rows:
            counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        return {"observed_at": obs.get("observed_at"), "mode": obs.get("mode"), "summary": counts,
                "controller": self.control(obs), "guard": obs.get("guard"),
                "gpu_scan": {k: (obs.get("gpu_scan") or {}).get(k) for k in ("ok", "ids", "errors")},
                "gpus": rows,
                "profiles": obs.get("profiles", {}), "plan": obs.get("plan") or {}, "jobs": obs.get("jobs", []),
                "last_error": obs.get("last_error"), "state_error": obs.get("state_error"),
                "leases": leases, "claims": claims, "rig": self.rig_report()}

    async def chat_availability(self, profile_id: str | None = None) -> dict:
        """Read-only busy signal for basic Chat. A ready requested profile may continue alone."""
        obs = await self.observe(fresh=True)  # never decide from the stale mirror
        profiles = obs.get("profiles") or {}
        own = profiles.get(profile_id) if profile_id else None
        own_ready = bool(own and own.get("state") == "ready")
        own_lease = own.get("lease") if own_ready else None
        own_gpus = set(own.get("gpus") or []) if own_ready else set()
        blockers = []
        if any(j.get("state") not in ("done", "finished", "failed", "cancelled", "stopped")
               for j in obs.get("jobs") or []):
            blockers.append("active rig job")
        if any(i.get("state") in ("queued", "pending", "waiting", "running")
               for i in (obs.get("plan") or {}).get("items") or []):
            blockers.append("active rig plan")
        if any(pid != profile_id or not own_ready for pid in profiles):
            blockers.append("loaded or loading profile")
        claims = obs.get("claims") or []
        if claims:
            blockers.append("GPU claim")
        if any(lease.get("lease_id") != own_lease for lease in obs.get("leases") or []):
            blockers.append("GPU lease")
        if any(c.get("name") != f"rig-remote-{profile_id}" or not own_ready
               for c in obs.get("containers") or []):
            blockers.append("active runtime")
        if any(int(g.get("used_mb") or 0) > placement.FREE_MAX_USED_MB and g.get("index") not in own_gpus
               for g in obs.get("gpus") or []):
            blockers.append("GPU memory in use")
        if not own_ready and (obs.get("guard") or {}).get("loads_admitted") is False:
            blockers.append("rig load guard")
        chat_claims = [claim for claim in claims if claim.get("purpose") == "chat"]
        chat_cards = {gpu for claim in chat_claims for gpu in claim.get("gpu_ids") or []}
        other_busy_cards = any(
            int(gpu.get("used_mb") or 0) > placement.FREE_MAX_USED_MB
            and gpu.get("index") not in own_gpus | chat_cards
            for gpu in obs.get("gpus") or []
        )
        chat_reservation_only = bool(chat_claims) and len(chat_claims) == len(claims) and not other_busy_cards and set(blockers) <= {"GPU claim", "GPU memory in use"}
        return {"busy": bool(blockers), "reason": ", ".join(dict.fromkeys(blockers)),
                "ready_profiles": [pid for pid, row in profiles.items() if row.get("state") == "ready"],
                "requested_ready": own_ready, "chat_reservation_only": chat_reservation_only,
                "chat_reservations": [{"name": claim.get("name"), "model": claim.get("model"),
                                       "gpu_ids": claim.get("gpu_ids") or []} for claim in chat_claims]}

    async def prepare_chat_profile(self, profile_id: str, profile: dict) -> dict:
        """Prepare one published chat profile without starting a cold load over existing work."""
        availability = await self.chat_availability(profile_id)
        if availability["busy"]:
            raise ManagerError(409, "rig busy: " + availability["reason"], rig_busy=True)
        port = await self.rig.ensure_ready(profile_id, start=lambda: self.load_profile(profile_id, profile))
        obs = await self.observe(fresh=True)
        ready = (obs.get("profiles") or {}).get(profile_id) or {}
        if (ready.get("state") != "ready" or ready.get("model") != profile.get("model")
                or ready.get("port") != port):
            raise ManagerError(409, "loaded profile no longer matches the chosen model")
        return {"profile": profile_id, "state": "ready", "port": port,
                "context_length": ready.get("ctx_size"), "gpu_ids": ready.get("gpus") or []}

    # ------------------------------------------------------------------ leases

    async def preview_lease(self, *, gpus_needed: list[dict] | None = None,
                            gpu_ids: list[int] | None = None, claim: str | None = None) -> dict:
        """Ask rig admission for a current, nonbinding card and port proposal."""
        if bool(gpus_needed) == bool(gpu_ids):
            raise ManagerError(400, "give exactly one of gpus_needed or gpu_ids")
        return await self.record.preview({
            **({"gpu_ids": gpu_ids} if gpu_ids is not None else {"gpus_needed": gpus_needed}),
            "metadata": {"claim": claim} if claim else {},
        })

    async def acquire(self, *, controller: str, owner: str, run_id: str, gpus_needed: list[dict] | None = None,
                      gpu_ids: list[int] | None = None, claim: str | None = None, ttl_s: int = 120,
                      model: str = "", purpose: str = "", lane: str | None = None,
                      requested_by: str = "") -> dict:
        """`controller` is the lease's controller type (interactive, benchmarks, plans, ...)."""
        await self._require_control()
        if bool(gpus_needed) == bool(gpu_ids):
            raise ManagerError(400, "give exactly one of gpus_needed or gpu_ids")
        async with self._place_lock:
            # The rig owns the live scan, placement, and atomic lease admission.
            # The controller sends requirements or an explicit operator pin.
            lease = await self.record.acquire({
                "controller": controller, "owner": owner, "run_id": run_id,
                **({"gpu_ids": gpu_ids} if gpu_ids is not None else {"gpus_needed": gpus_needed}),
                "lane": lane, "ttl_s": int(ttl_s),
                "metadata": {"model": model, "purpose": purpose, "claim": claim,
                             "by_controller": self.controller_id,
                             **({"requested_by": requested_by} if requested_by else {})}})
            self._obs = None  # the record changed
        gpus = lease["gpu_ids"]
        port = lease["metadata"]["port"]
        log.info("lease %s %s/%s GPU %s :%s %s", lease["lease_id"], controller, owner, gpus, port, model)
        return lease

    async def renew(self, lease_id: str, owner: str, ttl_s: int = 120) -> dict:
        await self._require_control()
        return await self.record.renew(lease_id, owner, ttl_s)

    async def release(self, lease_id: str, owner: str = "", force: bool = False) -> None:
        """Release a lease. Unless forced, its runtime must be gone (nothing on its GPUs)."""
        await self._require_control()
        obs = await self.observe(fresh=True)
        lease = next((l for l in obs.get("leases", []) if l.get("lease_id") == lease_id), None)
        if lease is None:
            raise ManagerError(404, f"lease not found: {lease_id}")
        if not force:
            running = [c["name"] for c in obs.get("containers", []) if set(c.get("gpus", [])) & set(lease["gpu_ids"])]
            if running:
                raise ManagerError(409, f"lease {lease_id} still has runtimes on its GPUs: {running}; stop them first")
        await self.record.release(lease_id, owner, force)
        self._obs = None
        log.info("released %s%s", lease_id, " (forced)" if force else "")

    # ------------------------------------------------------------------ claims

    async def add_claim(self, name: str, gpu_ids: list[int], note: str = "", by: str = "", *, owner: str = "",
                        ttl_s: int | None = None, purpose: str = "", model: str = "") -> dict:
        """Take cards out of the pool. They may still be leased: the claim then reserves them for when
        their holder releases (how a split waits for a matching set). ttl_s: lapses unless renewed."""
        await self._require_control()
        claim = await self.record.add_claim(name, gpu_ids, note, by, owner=owner, ttl_s=ttl_s, purpose=purpose,
                                            model=model)
        self._obs = None
        return claim

    async def extend_claim(self, name: str, gpu_ids: list[int], owner: str = "") -> dict:
        await self._require_control()
        existing = next((item for item in await self.claims() if item["name"] == name), None)
        if existing is None:
            raise ManagerError(404, f"no claim {name}")
        if not owner or existing.get("owner") != owner:
            raise ManagerError(403, f"claim {name} belongs to {existing.get('owner') or 'unknown owner'}")
        claim = await self.record.extend_claim(name, gpu_ids, owner)
        self._obs = None
        return claim

    async def renew_claim(self, name: str, owner: str = "", ttl_s: int | None = None) -> dict:
        await self._require_control()
        return await self.record.renew_claim(name, owner, ttl_s)

    async def remove_claim(self, name: str) -> dict:
        await self._require_control()
        claim = await self.record.remove_claim(name)
        self._obs = None
        return claim

    # ------------------------------------------------------------------ operator overrides

    async def kill(self, gpu_ids: list[int], reason: str = "") -> dict:
        """Stop every runtime on these cards, release every lease holding any of them, and drop every
        claim on them; then reconcile, which clears the profiles whose runtimes went."""
        await self._require_control()
        gpus = sorted({int(g) for g in gpu_ids})
        out = await self.record.kill(gpus, reason)
        self._obs = None
        self._event(f"kill GPU {gpus}" + (f" ({reason})" if reason else "") + f": stopped {out.get('stopped')}, "
                    f"released {out.get('released')}, claims dropped {out.get('claims_dropped')}")
        await self.reconcile_once()
        return out

    # ------------------------------------------------------------------ waiting through outages, demand

    async def _observe_for_wait(self, what: str, watch: dict) -> dict | None:
        """A fresh observation for a wait, or None while the rig is unreachable: the wait keeps waiting (the rig
        keeps working through what it was given; a lost network is not a crash). The outage is reported once
        when it starts and once when the rig is back; `watch["had_outage"]` tells the wait a claim it held may
        have lapsed meanwhile."""
        try:
            obs = await self.observe(fresh=True)
        except RigUnavailable as exc:
            if not watch.get("outage"):
                watch["outage"] = time.monotonic()
                self._event(f"{what}: rig unreachable; still waiting ({str(exc)[:120]})")
            return None
        if watch.get("outage"):
            self._event(f"{what}: rig back after {time.monotonic() - watch['outage']:.0f} s; still waiting")
            watch["outage"] = None
            watch["had_outage"] = True
        return obs

    async def _next_obs(self, what: str, watch: dict, deadline: float) -> dict:
        while True:
            obs = await self._observe_for_wait(what, watch)
            if obs is not None:
                return obs
            if time.monotonic() > deadline:
                raise RigUnavailable(504, f"{what}: rig unreachable for the whole wait")
            await asyncio.sleep(WAIT_POLL_S)

    def _demand(self, obs: dict, exclude_owner: str = "") -> dict[int, list[str]]:
        """Card sizes other work is waiting for: waits in progress, and the declared needs of running runs.
        size MiB -> who needs it (a requester's own waits and runs do not count against it)."""
        need: dict[int, list[str]] = {}
        for w in self._waits.values():
            if exclude_owner and w.get("owner") == exclude_owner:
                continue
            for mb in w.get("need", {}):
                need.setdefault(int(mb), []).append(f"wait for {w.get('model')}" + (f" ({w['owner']})" if w.get("owner") else ""))
        for r in (obs.get("plan") or {}).get("runs", []):
            if r.get("state") != "running" or (exclude_owner and exclude_owner in (r.get("run"), r.get("owner"))):
                continue
            for n in r.get("needs") or []:
                need.setdefault(int(n.get("card_mb", 0)), []).append(f"run {r['run']}")
        return need

    # ------------------------------------------------------------------ runs and cancel by owner

    async def runs(self) -> list[dict]:
        return (await self.observe(fresh=True)).get("plan", {}).get("runs", [])

    async def _run(self, run: str) -> dict:
        rec = next((r for r in await self.runs() if r["run"] == run), None)
        if rec is None:
            raise ManagerError(404, f"no run {run}")
        return rec

    async def start_run(self, run: str, argv: list[str], *, owner: str = "", cwd: str | None = None,
                        env: dict | None = None, resume_argv: list[str] | None = None, auto_resume: bool = False,
                        needs: list[dict] | None = None, note: str = "") -> dict:
        """Hand the rig a long run (a campaign, a lane): a `job` plan item the rig starts as its own unit and keeps
        in the shared report, so a reboot shows it as interrupted and the manager can offer to resume it.
        resume_argv: how to continue from a checkpoint (default: argv again). needs: card sizes the run will want
        ([{card_mb, count}]), so placement keeps them free for it."""
        await self._require_control()
        if not RUN_ID.fullmatch(run or ""):
            raise ManagerError(400, "run ids use letters, digits, . _ - (at most 64)")
        if not argv or not all(isinstance(a, str) and a for a in argv):
            raise ManagerError(400, "argv must be a non-empty list of strings")
        if any(r["run"] == run and r["state"] == "running" for r in await self.runs()):
            raise ManagerError(409, f"run {run} is already running")
        args = {"run": run, "argv": list(argv), "owner": owner or run, "cwd": cwd, "env": env or {},
                "resume_argv": list(resume_argv) if resume_argv else None, "auto_resume": bool(auto_resume),
                "needs": needs or [], "note": note}
        item = await self.record.plan_add("job", args)
        self._obs = None
        self._event(f"run {run}: queued ({' '.join(argv)[:160]})")
        return {"item": item, "run": args}

    async def recover_run(self, run: str, *, auto: bool = False) -> dict:
        """Start an interrupted, failed or stopped run again with its resume command (its checkpoint)."""
        rec = await self._run(run)
        if rec["state"] not in ("interrupted", "failed", "stopped"):
            raise ManagerError(409, f"run {run} is {rec['state']}; only an interrupted, failed or stopped run resumes")
        argv = rec.get("resume_argv") or rec["argv"]
        out = await self.start_run(run, argv, owner=rec.get("owner") or run, cwd=rec.get("cwd"), env=rec.get("env"),
                                   resume_argv=rec.get("resume_argv"), auto_resume=bool(rec.get("auto_resume")),
                                   needs=rec.get("needs"), note=rec.get("note") or "")
        self._event(f"run {run}: {'auto-' if auto else ''}resumed after {rec['state']} ({' '.join(argv)[:120]})")
        return out

    async def stop_run(self, run: str, reason: str = "") -> dict:
        """Stop a running run's unit, then clean up everything it owns (cancel by owner)."""
        rec = await self._run(run)
        stopped = await self.record.run_stop(run, reason) if rec["state"] == "running" else rec
        return {"run": stopped, "cleanup": await self.cancel_owner(rec.get("owner") or run, reason or f"stop {run}")}

    async def dismiss_run(self, run: str, reason: str = "") -> dict:
        """An ended run will not be resumed: mark it dismissed and clean up what it still owns."""
        rec = await self._run(run)
        if rec["state"] == "running":
            raise ManagerError(409, f"run {run} is running; stop it instead")
        out = await self.record.run_dismiss(run, reason)
        return {"run": out, "cleanup": await self.cancel_owner(rec.get("owner") or run, reason or f"dismiss {run}")}

    async def cancel_owner(self, who: str, reason: str = "") -> dict:
        """Cancel everything one owner has on the rig (a run id, a lane, a caller's load owner), without time
        limits or guessing: stop its running runs, cancel its queued plan items, unload its models, release its
        leases, drop its claims (which also cancels its waits), and log what was cleaned up.

        Matches leases by owner, run id or `requested_by`; claims by owner or name (and `campaign-<who>`, the
        owner the campaign runner gives its reservations); runs by run id or owner; waits by owner."""
        await self._require_control()
        if not who:
            raise ManagerError(400, "give an owner")
        obs = await self.observe(fresh=True)

        def lease_mine(lease: dict) -> bool:
            md = lease.get("metadata") or {}
            return who in (lease.get("owner"), lease.get("run_id"), md.get("requested_by"))

        leases = [l for l in obs.get("leases", []) if lease_mine(l)]
        lease_ids = {l["lease_id"] for l in leases}
        claims = {c["name"] for c in obs.get("claims", [])
                  if who in (c.get("owner"), c.get("name")) or c.get("owner") == f"campaign-{who}"}
        claims |= {w["claim"] for w in self._waits.values() if w.get("owner") == who and w.get("claim")}
        plan = obs.get("plan") or {}
        items = [i for i in plan.get("items", []) if i["state"] == "queued" and (
            i["args"].get("lease") in lease_ids or who in (i["args"].get("run"), i["args"].get("owner")))]
        runs = [r for r in plan.get("runs", []) if r["state"] == "running" and who in (r.get("run"), r.get("owner"))]
        profiles = [pid for pid, p in obs.get("profiles", {}).items() if p.get("lease") in lease_ids]
        done: dict[str, list] = {"runs_stopped": [], "items_cancelled": [], "profiles_unloaded": [],
                                 "leases_released": [], "claims_dropped": [], "errors": []}

        async def attempt(kind: str, label: str, coro) -> None:
            try:
                await coro
                done[kind].append(label)
            except (ManagerError, RigUnavailable) as exc:
                done["errors"].append(f"{kind} {label}: {exc}")

        for r in runs:
            await attempt("runs_stopped", r["run"], self.record.run_stop(r["run"], reason or f"cancel {who}"))
        for i in items:
            await attempt("items_cancelled", i["id"], self.record.plan_cancel(i["id"]))
        for pid in profiles:
            await attempt("profiles_unloaded", pid, self.rig.stop_profile(pid))
        for l in leases:
            await attempt("leases_released", l["lease_id"], self.record.release(l["lease_id"], l.get("owner", ""), force=True))
        for name in sorted(claims):
            await attempt("claims_dropped", name, self.record.remove_claim(name))
        self._obs = None
        summary = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in done.items() if v and k != "errors") or "nothing"
        self._event(f"cancel {who}" + (f" ({reason})" if reason else "") + f": {summary}"
                    + (f"; errors: {'; '.join(done['errors'])}" if done["errors"] else ""))
        return {"owner": who, **done}

    async def _watch_runs(self, obs: dict) -> None:
        """Report each run that ended (once), and resume an interrupted run that opted in (once per interruption)."""
        for r in (obs.get("plan") or {}).get("runs", []):
            key = (r["run"], r["state"], r.get("ended_at"))
            if r["state"] not in ("interrupted", "failed", "finished") or key in self._runs_seen:
                continue
            self._runs_seen.add(key)
            if r["state"] == "interrupted" and r.get("auto_resume") and self._auto_resumed.get(r["run"]) != r.get("ended_at"):
                self._auto_resumed[r["run"]] = str(r.get("ended_at"))
                await self.recover_run(r["run"], auto=True)
            elif r["state"] == "finished":
                self._event(f"run {r['run']} finished ({r.get('result')})")
            else:
                self._event(f"run {r['run']} {r['state']}: {r.get('result')}; resume it from its checkpoint with "
                            f"`rigctl recover {r['run']}`, or `rigctl dismiss {r['run']}`")

    # ------------------------------------------------------------------ reservations (split waits)

    async def _wait_for_cards(self, gpus: list[int], claim: str | None, wait_s: float, model: str,
                              pid: str, owner: str = "") -> tuple[str | None, str | None]:
        """Wait until every card in `gpus` is free (for the claim, when one is named), reserving them
        meanwhile so single work cannot take them as they free. Returns (claim to load under, the claim
        this wait made, or None). wait_s < 0: no limit. Past wait_s: the reservation is dropped and
        ManagerError 504. Removing the claim the wait holds (or the caller's named claim) cancels it: 409.
        An unreachable rig is waited out (the rig keeps working); a reservation that lapsed during the
        outage is taken again, not read as a cancel."""

        def state(obs: dict, name: str | None) -> tuple[bool, dict]:
            try:
                free, why = placement.availability(obs, obs.get("leases", []), obs.get("claims", []), name)
            except KeyError:
                return False, {g: f"claim {name} is gone" for g in gpus}
            ids = {f["index"] for f in free}
            return all(g in ids for g in gpus), {g: why.get(g) for g in gpus if g not in ids}

        deadline = time.monotonic() + wait_s if wait_s >= 0 else float("inf")
        watch: dict = {}
        obs = await self._next_obs(pid, watch, deadline)
        if claim:
            named = next((c for c in obs.get("claims", []) if c["name"] == claim), None)
            if named and not set(gpus) <= set(named["gpu_ids"]):
                raise ManagerError(409, f"claim {claim} covers GPU {named['gpu_ids']}, not all of {gpus}")
        ready, why = state(obs, claim)
        if ready:
            return claim, None
        size = {g["index"]: int(g["total_mb"]) for g in obs.get("gpus", [])}
        key = f"wait-{pid}-{uuid.uuid4().hex[:6]}"
        need: dict[int, int] = {}
        for g in gpus:
            need[size.get(g, 0)] = need.get(size.get(g, 0), 0) + 1
        self._waits[key] = {"owner": owner, "model": model, "need": need, "gpus": list(gpus), "claim": None,
                            "since": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        own = None

        async def reserve() -> str | None:
            name = f"wait-{pid}"[:64]
            try:
                await self.add_claim(name, gpus, note=f"waiting to load {model}", by=self.controller_id, owner=OWNER,
                                     ttl_s=WAIT_CLAIM_TTL_S, purpose=f"split wait for {pid}", model=model)
                self._waits[key]["claim"] = name
                return name
            except ManagerError as exc:
                # Another claim covers some card: wait without a reservation of our own until it goes.
                self._event(f"{pid}: cannot reserve GPU {gpus} ({exc}); waiting without a reservation")
                return None

        try:
            if not claim:
                own = claim = await reserve()
            self._event(f"{pid}: waiting {_limit(wait_s)} for GPU {gpus}"
                        + (f" (reserved as {own})" if own else "") + f": {why}")
            renewed = time.monotonic()
            while True:
                await asyncio.sleep(WAIT_POLL_S)
                obs = await self._observe_for_wait(pid, watch)
                if obs is None:
                    if time.monotonic() > deadline:
                        raise ManagerError(504, f"GPU {gpus} not free after waiting {wait_s:.0f} s for {pid} "
                                                f"(the rig was unreachable at the end)")
                    continue
                ready, why = state(obs, claim)
                if ready:
                    self._event(f"{pid}: GPU {gpus} free; loading")
                    return claim, own
                if claim and not any(c["name"] == claim for c in obs.get("claims", [])):
                    if own and watch.pop("had_outage", False):
                        # It lapsed while the rig was unreachable (nobody could renew it): take it again.
                        self._event(f"{pid}: reservation {own} lapsed during the outage; taking it again")
                        own = claim = await reserve()
                        renewed = time.monotonic()
                        continue
                    # The reservation went (an operator removed it, or the caller cancelled): stop waiting.
                    self._event(f"{pid}: wait cancelled, claim {claim} was removed")
                    raise ManagerError(409, f"wait for {pid} cancelled: claim {claim} was removed")
                watch.pop("had_outage", None)
                if time.monotonic() > deadline:
                    raise ManagerError(504, f"GPU {gpus} not free after waiting {wait_s:.0f} s for {pid}: "
                                            + "; ".join(f"GPU {g} {w}" for g, w in why.items()), unavailable=why)
                if own and time.monotonic() - renewed > WAIT_CLAIM_TTL_S / 3:
                    try:
                        await self.renew_claim(own, OWNER, WAIT_CLAIM_TTL_S)
                        renewed = time.monotonic()
                    except ManagerError:
                        pass  # gone: the next pass sees it
        except BaseException:
            if own:
                try:
                    await self.remove_claim(own)
                except (ManagerError, RigUnavailable):
                    pass
            raise
        finally:
            self._waits.pop(key, None)

    # ------------------------------------------------------------------ plan

    async def cancel_item(self, item_id: str) -> dict:
        await self._require_control()
        item = await self.record.plan_cancel(item_id)
        self._obs = None
        return item

    async def _wait_item(self, item_id: str, profile_id: str, until_ready: bool) -> dict:
        """Poll the rig until the plan item finishes (or, for a load, the profile serves)."""
        deadline = time.monotonic() + self.start_timeout_s
        not_ready = 0
        watch: dict = {}
        while True:
            obs = await self._observe_for_wait(profile_id, watch)
            if obs is None:
                deadline += LOAD_POLL_S  # the rig keeps loading while unreachable; the outage is not the load's
                await asyncio.sleep(LOAD_POLL_S)
                continue
            item = next((i for i in (obs.get("plan") or {}).get("items", []) if i["id"] == item_id), None)
            prof = obs.get("profiles", {}).get(profile_id)
            if until_ready and prof and prof.get("state") == "ready":
                return item or {}
            if item is None:
                raise RigUnavailable(502, f"plan item {item_id} disappeared from the rig")
            if item["state"] in ("done", "failed", "cancelled") and not (until_ready and item["state"] == "done"):
                return item
            if until_ready and item["state"] == "done":
                # Done but not ready: look again before calling it failed (failing releases the lease,
                # which stops the runtime). A rig whose observe reads the profiles before the plan can
                # report the item done while the profile still reads loading.
                not_ready += 1
                if not_ready < DONE_NOT_READY_CHECKS:
                    await asyncio.sleep(2)
                    continue
                # Still not ready: the runtime went away right after loading (rig guard, crash).
                return {**item, "state": "failed", "result": f"{profile_id} loaded but is not serving "
                                                              f"(rig guard: {(obs.get('guard') or {}).get('why')})"}
            if time.monotonic() > deadline:
                raise RigUnavailable(504, f"{profile_id}: plan item {item_id} still {item['state']} after "
                                          f"{self.start_timeout_s:.0f}s; the rig keeps working on it, retry shortly")
            await asyncio.sleep(LOAD_POLL_S)

    # ------------------------------------------------------------------ interactive profiles

    async def load_profile(self, profile_id: str, profile: dict, claim: str | None = None,
                           gpu_ids: list[int] | None = None) -> None:
        """Load a published profile and wait for it to serve. A profile names no cards: it is planned like any
        model load (the fit picks the cards from what is free and what the history says, and the largest
        context that fits, at least the profile's ctx_size), under the profile's own id so the gateway's
        routing does not change.

        The rig works through its plan by itself; past the start timeout the caller gets a 504 and the
        load keeps going on the rig, with its lease.
        """
        use = profile.get("use", "chat")
        fit_kw = {"profile": profile.get("registry_profile"), "ctx": None, "gpu_ids": gpu_ids,
                  "ctx_min": int(profile.get("ctx_size") or fit.CTX_MIN_DEFAULT),
                  "context_mode": "max" if use == "chat" else "working"}
        plan_ = await self.fit_model(profile["model"], claim=claim, use=use,
                                     requester=profile_id, **fit_kw)
        await self._load_planned(profile["model"], plan_, extra_args=None, claim=claim, use=use,
                                 owner=profile_id, profile_id=profile_id, **fit_kw)

    async def unload_profile(self, profile_id: str) -> dict:
        await self._require_control()
        obs = await self.observe(fresh=True)
        prof = obs.get("profiles", {}).get(profile_id)
        if not prof:
            raise ManagerError(404, f"{profile_id} is not loaded")
        lease_id = prof.get("lease")
        if lease_id:
            self._in_flight.add(lease_id)
        try:
            item = await self.record.plan_add("unload", {"profile": profile_id})
            done = await self._wait_item(item["id"], profile_id, until_ready=False)
            released = None
            if lease_id and any(l.get("lease_id") == lease_id for l in obs.get("leases", [])):
                try:
                    await self.release(lease_id, OWNER, force=True)
                except ManagerError as exc:
                    if exc.status != 404:  # already gone (released by an operator or another controller)
                        raise
                released = lease_id
        finally:
            if lease_id:
                self._in_flight.discard(lease_id)
        return {"unloaded": profile_id, "item": item["id"], "state": done["state"], "released": released}

    # ------------------------------------------------------------------ model loads (any model, any cards)

    async def _rig_model_fit(self, model: str, use: str, *, assume_idle: bool = False,
                             **options) -> dict:
        """Gateway checks the offered model; the rig owns detailed fit and placement."""
        try:
            selected = models.resolve(model, use, options.get("profile"))
        except (KeyError, FileNotFoundError) as exc:
            raise ManagerError(404, str(exc.args[0] if exc.args else exc)) from exc
        request = {"model": selected["model"], "use": use, "assume_idle": assume_idle,
                   **{key: value for key, value in options.items() if value is not None}}
        answer = await self.record.model_fit(request)
        local_sha = models.catalog().get("source_sha256")
        rig_sha = answer.get("source_sha256")
        if not local_sha or not rig_sha or local_sha != rig_sha:
            raise ManagerError(503, "Gateway and rig model libraries differ; refresh both from Benchmarking",
                               gateway_source_sha256=local_sha, rig_source_sha256=rig_sha)
        return answer["fit"]

    async def fit_model(self, model: str, *, cards: int | None = None, gpu_ids: list[int] | None = None,
                        ctx: int | None = None, ctx_min: int = fit.CTX_MIN_DEFAULT, use: str = "benchmark",
                        profile: str | None = None, claim: str | None = None,
                        margin_mb: int = fit.MARGIN_MB_DEFAULT, requester: str = "",
                        batch_size: int | None = None, context_mode: str = "working",
                        engine: str = "auto", placement_policy: str | None = None) -> dict:
        """Request a rig-owned fit preview; no hardware state changes in this call."""
        result = await self._rig_model_fit(
            model, use, cards=cards, gpu_ids=gpu_ids, ctx=ctx, ctx_min=ctx_min,
            profile=profile, claim=claim, margin_mb=margin_mb, batch_size=batch_size,
            context_mode=context_mode, engine=engine, placement_policy=placement_policy)
        if cards or gpu_ids or claim:
            return result
        # Gateway owns work scheduling: preserve a card needed by queued work.
        # The rig still chooses every alternative placement.
        obs = await self.observe(fresh=True)
        demand = self._demand(obs, requester)
        if not demand:
            return result
        idle = await self._rig_model_fit(
            model, use, assume_idle=True, ctx=ctx, ctx_min=ctx_min, profile=profile,
            margin_mb=margin_mb, batch_size=batch_size, context_mode=context_mode,
            engine=engine, placement_policy=placement_policy)
        size = {g["index"]: int(g["total_mb"]) for g in obs.get("gpus", [])}
        natural = {size[g] for g in idle.get("gpus", []) if g in size}
        extra = {size[g] for g in result.get("gpus", []) if g in size} - natural
        kept = {mb: demand[mb] for mb in extra if mb in demand}
        if not kept:
            return result
        excluded = [g for g, mb in size.items() if mb in kept]
        why = "; ".join(f"{mb} MiB cards for {', '.join(sorted(set(who)))}"
                        for mb, who in kept.items())
        try:
            alternative = await self._rig_model_fit(
                model, use, exclude_gpu_ids=excluded, ctx=ctx, ctx_min=ctx_min, profile=profile,
                margin_mb=margin_mb, batch_size=batch_size, context_mode=context_mode,
                engine=engine, placement_policy=placement_policy)
            alternative["chosen_by"] = f"rig: alternative fit keeping {why}"
            return alternative
        except ManagerError as exc:
            if exc.status not in (422, 503):
                raise
        raise ManagerError(409, f"{model}: GPU {result['gpus']} kept free for waiting work ({why}); "
                                "load it later or name the cards", kept_for=kept)

    async def schedule_preview(self, workloads: list[dict], candidate_limit: int = 16) -> dict:
        """Read-only Orchestration alternatives. This does not reserve or load hardware."""
        answer = await self.record.schedule_preview({"workloads": workloads,
                                                     "candidate_limit": candidate_limit})
        local_sha = models.catalog().get("source_sha256")
        rig_sha = answer.get("model_library_sha256")
        if not local_sha or not rig_sha or local_sha != rig_sha:
            raise ManagerError(503, "Gateway and rig model libraries differ; refresh both from Benchmarking",
                               gateway_source_sha256=local_sha, rig_source_sha256=rig_sha)
        return answer

    async def _collect_cards(self, model: str, wait_s: float, use: str, owner: str = "",
                             **fit_kw) -> tuple[str, list[int]]:
        """For a load that names no cards: the cards the harness would choose on an idle rig say how many
        cards of which size the model needs; claim free cards of those sizes one at a time as they free (one
        claim, extended), so single work cannot keep taking them, until the claim holds the set. Returns
        (the claim, its cards). wait_s < 0: no limit. Removing the claim cancels the wait (409); past wait_s
        the claim is dropped and 504. An unreachable rig is waited out; a claim that lapsed during the outage
        is taken again (cards someone else took meanwhile are collected again)."""
        deadline = time.monotonic() + wait_s if wait_s >= 0 else float("inf")
        watch: dict = {}
        r = models.resolve(model, use, fit_kw.get("profile"))
        target = await self._rig_model_fit(model, use, assume_idle=True, **fit_kw)
        obs = await self._next_obs(r["model"], watch, deadline)
        size = {g["index"]: int(g["total_mb"]) for g in obs.get("gpus", [])}
        need = Counter(size[g] for g in target["gpus"])
        slug = re.sub(r"[^a-z0-9]+", "-", r["model"].lower()).strip("-")
        name = f"wait-{slug}-{uuid.uuid4().hex[:6]}"[:64]
        wanted = " + ".join(f"{n} x {mb} MiB" for mb, n in sorted(need.items()))
        self._waits[name] = {"owner": owner, "model": r["model"], "need": dict(need), "claim": name,
                             "since": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        self._event(f"{r['model']}: collecting {wanted} cards as they free, {_limit(wait_s)}, as claim {name}")
        held: list[int] = []
        renewed = time.monotonic()
        try:
            while True:
                if obs is not None:
                    claims = obs.get("claims", [])
                    if held and not any(c["name"] == name for c in claims):
                        if watch.pop("had_outage", False):
                            self._event(f"{r['model']}: claim {name} lapsed during the outage; collecting again")
                            held = []
                        else:
                            self._event(f"{r['model']}: collecting cancelled, claim {name} was removed")
                            raise ManagerError(409, f"wait for {r['model']} cancelled: claim {name} was removed")
                    watch.pop("had_outage", None)
                    free, _ = placement.availability(obs, obs.get("leases", []), claims, None)
                    have = Counter(size[g] for g in held)
                    take = []
                    for g in sorted(free, key=lambda g: g["index"]):
                        mb = int(g["total_mb"])
                        if have[mb] + sum(1 for t in take if size[t] == mb) < need[mb]:
                            take.append(g["index"])
                    if take:
                        if held:
                            await self.extend_claim(name, take, OWNER)
                        else:
                            await self.add_claim(name, take, note=f"collecting cards to load {r['model']}",
                                                 by=self.controller_id, owner=OWNER, ttl_s=WAIT_CLAIM_TTL_S,
                                                 purpose=f"collect {wanted} for {r['model']}", model=r["model"])
                        held = sorted(held + take)
                        renewed = time.monotonic()
                        self._event(f"{r['model']}: claimed GPU {take}, holding {held} of {wanted}")
                    if Counter(size[g] for g in held) == need:
                        if take:
                            obs = await self._next_obs(r["model"], watch, deadline)
                        try:
                            ready, _ = placement.availability(obs, obs.get("leases", []), obs.get("claims", []), name)
                        except KeyError:
                            ready = []  # gone: the next pass decides (lapsed in an outage, or cancelled)
                        if held and {g["index"] for g in ready} >= set(held):
                            self._event(f"{r['model']}: holding GPU {held}; loading")
                            return name, held
                if time.monotonic() > deadline:
                    raise ManagerError(504, f"{r['model']}: only GPU {held} of {wanted} free after {wait_s:.0f} s")
                if held and time.monotonic() - renewed > WAIT_CLAIM_TTL_S / 3:
                    try:
                        await self.renew_claim(name, OWNER, WAIT_CLAIM_TTL_S)
                        renewed = time.monotonic()
                    except (ManagerError, RigUnavailable):
                        pass  # gone or unreachable: the next observation decides
                await asyncio.sleep(WAIT_POLL_S)
                obs = await self._observe_for_wait(r["model"], watch)
        except BaseException:
            if held:
                try:
                    await self.remove_claim(name)
                except (ManagerError, RigUnavailable):
                    pass
            raise
        finally:
            self._waits.pop(name, None)

    async def load_model(self, model: str, *, extra_args: list[str] | None = None, claim: str | None = None,
                         use: str = "benchmark", wait_s: float = 0, force: bool = False, owner: str = "",
                         **fit_kw) -> dict:
        """Fit, lease, queue the load in the rig's plan, wait until it serves, and report what the rig
        actually loaded (context, split, memory per card). The rig re-checks with llama.cpp's own
        projection and refuses a load that would not fit.

        wait_s: when the named cards (gpu_ids) are not all free, reserve them and wait up to this long
        (docs/MANAGER.md, "Reservations"); with no cards named, collect the cards the model needs as they
        free (`_collect_cards`); with a card count instead, retry the fit until it succeeds. wait_s < 0
        waits with no limit; removing the wait's claim cancels it.
        force: operator override, kill whatever runs on the named cards first.
        owner: who the load is for (a run id, a lane): recorded on the lease and the wait, so cancelling that
        owner unloads it and stops its waits, and so placement does not keep cards free from the owner's own
        demand."""
        await self._require_control()
        gpu_ids = sorted({int(g) for g in fit_kw.get("gpu_ids") or []})
        if not gpu_ids and not fit_kw.get("cards") and wait_s != 0:
            try:
                ready = models.resolve(model, use, fit_kw.get("profile"))
                if fit_kw.get("engine") == "strata" or (ready.get("strata") or {}).get("only"):
                    gpu_ids = [0]
                    fit_kw = {**fit_kw, "gpu_ids": gpu_ids}
            except (KeyError, FileNotFoundError):
                pass  # fit_model reports the proper missing-model error below
        if force:
            if not gpu_ids:
                raise ManagerError(400, "force needs explicit gpu_ids")
            # Validate the requested work before the explicit stop command. A typo or
            # disallowed model must never evict a healthy runtime.
            if use not in memory.USES:
                raise ManagerError(400, f"use must be one of {memory.USES}")
            try:
                models.resolve(model, use, fit_kw.get("profile"))
            except (KeyError, FileNotFoundError) as exc:
                raise ManagerError(404, str(exc.args[0] if exc.args else exc)) from exc
            await self.kill(gpu_ids, reason=f"force load {model}")
        own = None
        try:
            if gpu_ids and wait_s != 0:
                r0 = models.resolve(model, use, fit_kw.get("profile"))
                claim, own = await self._wait_for_cards(gpu_ids, claim, wait_s, model,
                                                        models.profile_id(r0["model"], gpu_ids), owner)
                plan_ = await self.fit_model(model, claim=claim, use=use, requester=owner, **fit_kw)
            elif wait_s != 0 and not fit_kw.get("cards"):
                claim, gpus = await self._collect_cards(model, wait_s, use, owner=owner, **fit_kw)
                own = claim
                fit_kw = {**fit_kw, "gpu_ids": gpus}
                plan_ = await self.fit_model(model, claim=claim, use=use, requester=owner, **fit_kw)
            elif wait_s != 0:
                deadline = time.monotonic() + wait_s if wait_s > 0 else float("inf")
                while True:
                    try:
                        plan_ = await self.fit_model(model, claim=claim, use=use, requester=owner, **fit_kw)
                        break
                    except ManagerError as exc:
                        if exc.status not in (409, 503) or time.monotonic() > deadline:
                            raise
                    except RigUnavailable:
                        pass  # unreachable: keep waiting, the rig keeps working
                    await asyncio.sleep(WAIT_POLL_S)
            else:
                plan_ = await self.fit_model(model, claim=claim, use=use, requester=owner, **fit_kw)
            return await self._load_planned(model, plan_, extra_args=extra_args, claim=claim, use=use,
                                            owner=owner, **fit_kw)
        finally:
            if own:  # the lease holds the cards now (or the load failed): the wait's reservation goes
                try:
                    await self.remove_claim(own)
                except ManagerError:
                    pass

    async def _load_planned(self, model: str, plan_: dict, *, extra_args: list[str] | None, claim: str | None,
                            use: str, owner: str = "", profile_id: str | None = None, **fit_kw) -> dict:
        r = models.resolve(model, use, fit_kw.get("profile"))
        if plan_.get("backend") == "strata" and extra_args:
            raise ManagerError(400, "Strata runtime options are fixed by the registered pack")
        rt = r.get("runtime", {})
        gpus = plan_["gpus"]
        pid = profile_id or models.profile_id(r["model"], gpus)  # a published profile keeps its own id
        batch = int(fit_kw.get("batch_size") or rt.get("batch_size", 64))
        spec = {"model": r["model"], "gguf_path": r["gguf_path"], "ctx_size": plan_["ctx"],
                "batch_size": batch, "parallel": int(rt.get("parallel", 1)),
                "n_gpu_layers": plan_.get("n_gpu_layers", "all"), "allow_partial_offload": plan_.get("placement") == "ram",
                 "backend": plan_.get("backend", "llama"),
                 **({"strata": r["strata"]} if plan_.get("backend") == "strata" else {}),
                 "extra_args": [*r.get("extra_args", []), *(extra_args or [])],
                "image": r.get("image"), "use": use, "registry_profile": r["profile"],
                **({"tensor_split": plan_["tensor_split"]} if plan_.get("tensor_split") else {})}
        guard = (await self.observe(fresh=True)).get("guard") or {}
        cache_mib = (0 if plan_.get("placement") in ("ram", "strata") else
                     memory.cache_ram_mib(use, guard.get("mem_available_mb")))
        # Owner llm-gateway, like profile loads: the reconciler renews it while the rig profile lives.
        lease = await self.acquire(controller=models.CONTROLLER_BY_USE.get(use, "interactive"), owner=OWNER,
                                   run_id=f"model-{pid}", gpu_ids=gpus, claim=claim, ttl_s=INTERACTIVE_TTL_S,
                                   requested_by=owner,
                                   model=r["model"], purpose=f"model load {pid} ({use}, ctx {plan_['ctx']})")
        lease_id = lease["lease_id"]
        self._in_flight.add(lease_id)
        started = time.monotonic()
        # Facts every observation of this load carries: what was asked, on which physical cards, and the estimate.
        facts = {"model": r["model"], "source": os.path.basename(r["gguf_path"]), "ubatch": min(batch, 512),
                 "use": use, "placement": plan_.get("placement", "gpu"), "gpus": gpus,
                  "planned_ctx": plan_["ctx"], "tensor_split": plan_.get("tensor_split"),
                 "hw": observations.cards_from(self._obs, gpus), "owner": owner,
                 "card_sizes": [int(g["total_mb"]) for g in (self._obs or {}).get("gpus", []) if g.get("index") in gpus],
                 "estimate": [{k: c.get(k) for k in ("gpu", "weights_mb", "kv_mb", "compute_mb", "projected_mb")}
                              for c in plan_.get("cards") or []], "compute_basis": plan_.get("compute_basis")}
        try:
            try:
                item = await self.record.plan_add("load", {
                    "profile": pid, "spec": spec, "gpus": gpus, "port": lease["metadata"]["port"],
                    "lease": lease_id, "cache_ram": cache_mib})
            except (ManagerError, RigUnavailable):
                await self.release(lease_id, OWNER, force=True)
                raise
            log.info("model load %s queued as %s (GPU %s ctx %s split %s)", pid, item["id"], gpus, plan_["ctx"],
                     plan_.get("tensor_split"))
            done = await self._wait_item(item["id"], pid, until_ready=True)
            obs = await self.observe(fresh=True)
            if done.get("state") in ("failed", "cancelled"):
                if pid in obs.get("profiles", {}):
                    await self.rig.stop_profile(pid)
                await self.release(lease_id, OWNER, force=True)
                why = str(done.get("result") or "")[-1500:]
                try:  # the other side of the fit boundary
                    row = observations.record("load_failed", **facts, state=done["state"], error=why,
                                              oom="out of memory" in why.lower(),
                                              seconds=round(time.monotonic() - started, 1))
                    await self.record.fit_observation(row)
                except (OSError, ManagerError, RigUnavailable) as exc:
                    log.warning("failed-load observation for %s not recorded: %s", pid, exc)
                raise RigUnavailable(502, f"rig did not load {pid} ({done['state']}): {why}")
        finally:
            self._in_flight.discard(lease_id)
        loaded = obs.get("profiles", {}).get(pid, {})
        try:  # what llama.cpp allocated: the next fit of this GGUF plans its compute buffer from it
            # A CPU-offload compute buffer is not a calibration point for a
            # later full-GPU fit of the same GGUF/card count.
            row = (fit.record_observation(
                    facts["source"], r["model"], facts["ubatch"], gpus, loaded.get("fit"),
                                          **{k: v for k, v in facts.items()
                                             if k not in ("source", "model", "ubatch", "gpus")},
                                          seconds=round(time.monotonic() - started, 1),
                                          # nvidia-smi once serving: minus llama.cpp's total is the card overhead
                                          used_mb=[next((g.get("used_mb") for g in obs.get("gpus", [])
                                                         if g.get("index") == i), None) for i in gpus])
                   if plan_.get("placement") not in ("ram", "strata") else None)
            if row:
                await self.record.fit_observation(row)
            elif plan_.get("placement") not in ("ram", "strata"):
                log.info("load %s: the rig reported no measured fit; nothing to calibrate from", pid)
        except (OSError, ManagerError, RigUnavailable) as exc:
            log.warning("fit observation for %s not recorded: %s", pid, exc)
        return {"profile": pid, "model": r["model"], "state": "ready", "port": lease["metadata"]["port"],
                "gpus": gpus, "lease": lease_id, "ctx": loaded.get("ctx_size", plan_["ctx"]),
                "tensor_split": plan_.get("tensor_split"), "estimate": plan_, "measured": loaded.get("fit")}

    # ------------------------------------------------------------------ reconcile

    async def reconcile_once(self) -> dict:
        """Keep control (heartbeat, or take it when the active controller is gone), then renew our
        leases that still have a load, runtime or queued plan item behind them and release the rest.

        Leases of other controllers are only reported (status shows them as stale or lease_idle).
        """
        if self.in_control:
            try:
                await self.record.heartbeat()
            except ManagerError as exc:
                if exc.extra.get("rig_kind") != "not_controller":
                    raise
                self.in_control, self.control_note = False, f"lost control: {exc}"
                log.warning("controller %s: %s", self.controller_id, self.control_note)
        else:
            await self.take_control()
        if not self.in_control:
            return {"in_control": False, "note": self.control_note, "renewed": [], "released": []}

        obs = await self.observe(fresh=True)
        if (obs.get("plan") or {}).get("hold"):
            await self.settle_hold(obs)
            obs = await self.observe(fresh=True)
        try:
            await self._watch_runs(obs)
        except (ManagerError, RigUnavailable) as exc:
            log.warning("reconcile: run check failed: %s", exc)
        # A profile whose runtime is gone (the rig guard freed RAM, or it crashed): clear it on the rig;
        # its lease is then unbacked and released below.
        for pid, prof in obs.get("profiles", {}).items():
            if prof.get("state") == "gone":
                log.info("reconcile: %s runtime is gone on the rig; clearing it", pid)
                await self.rig.stop_profile(pid)
        backed = {p.get("lease") for p in obs.get("profiles", {}).values()
                  if p.get("lease") and p.get("state") != "gone"}
        # A load still waiting in the rig's plan has a lease but no rig profile yet.
        backed |= {i["args"].get("lease") for i in (obs.get("plan") or {}).get("next", []) if i["kind"] == "load"}
        now = datetime.now(timezone.utc)
        renewed, released = [], []
        for lease in obs.get("leases", []):
            if lease.get("owner") != OWNER:
                continue
            lid = lease["lease_id"]
            if lid in self._in_flight:
                continue  # a load or unload of ours is working on it; it settles the lease itself
            if lid in backed:
                if not is_expired(lease, now):
                    await self.record.renew(lid, OWNER, INTERACTIVE_TTL_S)
                    renewed.append(lid)
                continue
            acquired = datetime.fromisoformat(str(lease["acquired_at"]).replace("Z", "+00:00"))
            if (now - acquired).total_seconds() < ORPHAN_GRACE_S:
                continue
            await self.record.release(lid, OWNER, force=True)
            released.append(lid)
            log.info("reconcile: released %s (no rig load, runtime or plan item behind it)", lid)
        if renewed or released:
            self._obs = None
        return {"in_control": True, "renewed": renewed, "released": released}

    async def run_reconciler(self) -> None:
        while True:
            try:
                await self.reconcile_once()
            except Exception as exc:  # keep reconciling; the next pass may succeed
                log.warning("reconcile failed: %s", exc)
            await asyncio.sleep(RECONCILE_EVERY_S)
