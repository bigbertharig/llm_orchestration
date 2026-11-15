"""Bounded, on-demand rig health query for the orchestration status API."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone


def _rig_json(*argv: str) -> dict:
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3",
                             "gpu", *argv], text=True, capture_output=True, timeout=8)
    if result.returncode:
        raise RuntimeError(f"rig probe exited {result.returncode}")
    data = json.loads(result.stdout)
    if not isinstance(data, dict):
        raise ValueError("rig probe returned a non-object")
    return data


def sample() -> dict:
    """Read the rig guard heartbeat and current PCIe scan without starting a model."""
    try:
        guard = _rig_json("/home/bryan/llm-orchestration-venv/bin/python",
                          "/home/bryan/llm_orchestration/shared/scripts/rig/hw_guard.py", "status", "--json")
        rig = _rig_json("/usr/local/bin/rig", "status", "--json")
        state = guard.get("state") or {}
        scan = rig.get("gpus") or {}
        sampled_at = state.get("updated_ts")
        age = max(0, time.time() - sampled_at) if isinstance(sampled_at, (int, float)) else None
        return {"sampled_at": datetime.now(timezone.utc).isoformat(), "guard_level": guard.get("level"),
                "guard_age_s": round(age, 1) if age is not None else None,
                "loads_admitted": guard.get("admit"), "guard_reason": guard.get("why"),
                "gpu_scan_ok": scan.get("ok"), "gpu_ids": scan.get("ids", []),
                "gpu_errors": scan.get("errors", []), "mode": rig.get("mode"),
                "ok": bool(age is not None and age <= 15 and guard.get("level") in ("ok", "warn")
                           and scan.get("ok"))}
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        return {"sampled_at": datetime.now(timezone.utc).isoformat(), "ok": False,
                "error": f"rig probe failed: {type(exc).__name__}"}


def current() -> dict:
    return sample()
