"""Rig claims for orchestration's split reservations (llm-gateway docs/MANAGER.md, "Reservations").

While a split load is waiting for its partner and loading (`waiting_partner` .. `ready_stabilizing` in
gpu_split.py), orchestration's own reservation file keeps its member workers off other work. The rig
claim made here does the same for everyone else: the rig manager will not lease those cards to a
benchmark lane, a campaign or the gateway while the split waits for them. The claim has a TTL that
the launcher renews while it waits, so a dead launcher frees the cards on its own.

The claim goes through the rig manager (rigctl over ssh to the Pi), like benchmark lanes and
campaigns. If the manager cannot be reached, the split goes ahead on orchestration's reservation
alone. RIG_RESERVATIONS=off turns this off.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from typing import Iterable, Optional

HOST = os.getenv("HARNESS_HOST", "pi")
RIGCTL = os.getenv("HARNESS_RIGCTL", "~/llm-gateway/tools/rigctl.py")
OWNER = "plans"
PREFIX = "plans-"            # orchestration's own claims; its workers do not treat them as foreign
TTL_S = 600
RENEW_EVERY_S = 120


def enabled() -> bool:
    return os.getenv("RIG_RESERVATIONS", "on").strip().lower() not in ("off", "0", "false", "no")


def claim_name(group_id: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", f"{PREFIX}split-{group_id}".lower())[:64]


def gpu_ids(members: Iterable[str]) -> list[int]:
    """Worker names like gpu-3 -> GPU ids."""
    ids = set()
    for m in members:
        match = re.search(r"(\d+)$", str(m).strip())
        if match:
            ids.add(int(match.group(1)))
    return sorted(ids)


def _rigctl(*args: str, timeout: float = 30) -> Optional[subprocess.CompletedProcess]:
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", HOST, "python3", RIGCTL,
           *[shlex.quote(str(a)) for a in args]]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None


def reserve(group_id: str, members: Iterable[str], model: str, logger=None) -> bool:
    """Claim the split's cards for its wait. True if the rig holds the claim now."""
    ids = gpu_ids(members)
    if not enabled() or not ids:
        return False
    name = claim_name(group_id)
    proc = _rigctl("claim", name, "--gpus", ",".join(map(str, ids)), "--owner", OWNER, "--ttl", str(TTL_S),
                   "--purpose", f"orchestration split load {group_id}", "--model", model or "",
                   "--note", f"split reservation {group_id}")
    ok = proc is not None and proc.returncode == 0
    if logger:
        if ok:
            logger.info(f"RIG_RESERVATION_HELD group={group_id} claim={name} gpus={ids}")
        else:
            why = (proc.stdout + proc.stderr).strip()[-200:] if proc is not None else "manager unreachable"
            logger.warning(f"RIG_RESERVATION_NOT_HELD group={group_id} claim={name} gpus={ids}: {why}")
    return ok


def renew(group_id: str, logger=None) -> bool:
    if not enabled():
        return False
    proc = _rigctl("renew-claim", claim_name(group_id), "--owner", OWNER, "--ttl", str(TTL_S))
    ok = proc is not None and proc.returncode == 0
    if not ok and logger:
        logger.warning(f"RIG_RESERVATION_RENEW_FAILED group={group_id} claim={claim_name(group_id)}")
    return ok


def release(group_id: str, logger=None) -> None:
    if not enabled():
        return
    proc = _rigctl("unclaim", claim_name(group_id))
    if logger:
        logger.info(f"RIG_RESERVATION_RELEASED group={group_id} claim={claim_name(group_id)} "
                    f"ok={proc is not None and proc.returncode == 0}")
