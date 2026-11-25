"""The rig's records, kept by the rig: leases and claims (`rig lease`, `rig claim`, hw_state.py), the
active controller and the plan queue (`rig controller`, `rig plan`, hw_plan.py).

The controller holds no state of its own: it reads everything from `rig observe` and asks the rig to
change it. The rig is the single writer, so the records survive a lost network and any controller
can pick up where another left off. Every write carries this controller's id; the rig refuses
writes from a controller that is not the active one.
"""

from __future__ import annotations

import json
from typing import Any

from ..rig import Rig, RigUnavailable
from .errors import KIND_STATUS, ManagerError


class RigRecord:
    def __init__(self, rig: Rig, controller_id: str):
        self.rig = rig
        self.controller_id = controller_id

    async def _call(self, *args: str) -> dict[str, Any]:
        code, out = await self.rig._run(*args, timeout=60)
        try:
            payload = json.loads(out)
        except json.JSONDecodeError as exc:
            raise RigUnavailable(502, f"rig {' '.join(args[:2])} returned no JSON: {out[-300:]}") from exc
        if code != 0:
            raise ManagerError(KIND_STATUS.get(payload.get("kind"), 502), payload.get("error", out[-300:]),
                               rig_kind=payload.get("kind"),
                               **{key: value for key, value in payload.items() if key not in ("error", "kind")})
        return payload

    def _as_me(self) -> list[str]:
        return ["--controller", self.controller_id]

    # ---------------------------------------------------------------- leases and claims

    async def preview(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Read the rig's current admission decision without creating a lease."""
        return await self._call("lease", "preview", "--spec", json.dumps(spec))

    async def model_fit(self, request: dict[str, Any]) -> dict[str, Any]:
        """Ask the rig for its detailed, read-only model placement proposal."""
        return await self._call("model-fit", "--request", json.dumps(request))

    async def schedule_preview(self, request: dict[str, Any]) -> dict[str, Any]:
        """Ask Orchestration for revisioned, read-only workload placement alternatives."""
        return await self._call("schedule-preview", "--request", json.dumps(request))

    async def fit_observation(self, row: dict[str, Any]) -> dict[str, Any]:
        """Send completed load evidence to the rig's calibration ledger."""
        return await self._call("fit-observation", "--row", json.dumps(row))

    async def acquire(self, spec: dict[str, Any]) -> dict[str, Any]:
        return (await self._call("lease", "acquire", "--spec", json.dumps(spec), *self._as_me()))["lease"]

    async def renew(self, lease_id: str, owner: str, ttl_s: int) -> dict[str, Any]:
        return (await self._call("lease", "renew", lease_id, "--owner", owner, "--ttl", str(int(ttl_s)),
                                 *self._as_me()))["lease"]

    async def release(self, lease_id: str, owner: str = "", force: bool = False) -> None:
        args = ["lease", "release", lease_id, "--owner", owner] + (["--force"] if force else [])
        await self._call(*args, *self._as_me())

    async def extend_claim(self, name: str, gpu_ids: list[int], owner: str = "") -> dict[str, Any]:
        args = ["claim", "extend", name, "--gpus", ",".join(str(int(g)) for g in gpu_ids)]
        if owner:
            args += ["--owner", owner]
        return (await self._call(*args, *self._as_me()))["claim"]

    async def add_claim(self, name: str, gpu_ids: list[int], note: str = "", by: str = "", *, owner: str = "",
                        ttl_s: int | None = None, purpose: str = "", model: str = "") -> dict[str, Any]:
        args = ["claim", "add", name, "--gpus", ",".join(str(int(g)) for g in gpu_ids), "--note", note]
        if by:
            args += ["--by", by]
        if owner:
            args += ["--owner", owner]
        if ttl_s:
            args += ["--ttl", str(int(ttl_s))]
        if purpose:
            args += ["--purpose", purpose]
        if model:
            args += ["--model", model]
        return (await self._call(*args, *self._as_me()))["claim"]

    async def renew_claim(self, name: str, owner: str = "", ttl_s: int | None = None) -> dict[str, Any]:
        args = ["claim", "renew", name] + (["--owner", owner] if owner else []) + (
            ["--ttl", str(int(ttl_s))] if ttl_s else [])
        return (await self._call(*args, *self._as_me()))["claim"]

    async def remove_claim(self, name: str) -> dict[str, Any]:
        return (await self._call("claim", "remove", name, *self._as_me()))["released"]

    async def kill(self, gpu_ids: list[int], reason: str = "") -> dict[str, Any]:
        args = ["kill", "--gpus", ",".join(str(int(g)) for g in gpu_ids)] + (["--reason", reason] if reason else [])
        return await self._call(*args, *self._as_me())

    # ---------------------------------------------------------------- controller and plan

    async def claim_control(self, force: bool = False) -> dict[str, Any]:
        args = ["controller", "claim", "--id", self.controller_id] + (["--force"] if force else [])
        return (await self._call(*args))["controller"]

    async def heartbeat(self) -> dict[str, Any]:
        return (await self._call("controller", "heartbeat", "--id", self.controller_id))["controller"]

    async def release_control(self) -> None:
        await self._call("controller", "release", "--id", self.controller_id)

    async def plan_add(self, kind: str, args: dict[str, Any]) -> dict[str, Any]:
        return (await self._call("plan", "add", "--kind", kind, "--args", json.dumps(args), *self._as_me()))["item"]

    async def plan_resume(self) -> dict[str, Any]:
        return (await self._call("plan", "resume", *self._as_me()))["plan"]

    async def plan_restore(self, doc: dict[str, Any]) -> dict[str, Any]:
        return (await self._call("plan", "restore", "--doc", json.dumps(doc), *self._as_me()))["plan"]

    async def run_stop(self, run: str, reason: str = "") -> dict[str, Any]:
        return (await self._call("run", "stop", run, "--reason", reason, *self._as_me()))["run"]

    async def run_dismiss(self, run: str, reason: str = "") -> dict[str, Any]:
        return (await self._call("run", "dismiss", run, "--reason", reason, *self._as_me()))["run"]

    async def plan_cancel(self, item_id: str) -> dict[str, Any]:
        return (await self._call("plan", "cancel", item_id, *self._as_me()))["item"]
