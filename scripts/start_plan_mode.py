#!/usr/bin/env python3
"""Start rig in plan-ready mode with the default worker model warmed.

Normal boot is neutral: brain runtime up, workers cold, no auto-default repair.
This wrapper is the explicit old behavior for plan-heavy sessions:
- workers start cold
- gpu-2 gets qwen2.5-coder:7b loaded through a startup meta task
- auto-default keeps returning idle state to that plan-ready target
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REMOTE_HOST = "gpu"
REMOTE_BASE_CONFIG = "/mnt/shared/agents/config.json"
REMOTE_PLAN_CONFIG = "/mnt/shared/agents/config.plan.json"
REMOTE_STARTUP = "/mnt/shared/agents/startup.py"
REMOTE_PREFLIGHT = "/mnt/shared/scripts/runtime_preflight.py"
REMOTE_PYTHON = "/home/bryan/llm-orchestration-venv/bin/python"
REMOTE_LOG = "/mnt/shared/logs/startup-plan.log"
DEFAULT_WORKER = "gpu-2"
DEFAULT_MODEL = "qwen2.5-coder:7b"


def _run_remote(script: str, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", REMOTE_HOST, "bash", "-s"],
        input=script,
        text=True,
        capture_output=True,
        timeout=max(20, timeout),
        check=False,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Start rig in plan-ready startup mode")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    remote_script = """set -e
python3 - <<'PY'
import json
from pathlib import Path
src = Path("__REMOTE_BASE_CONFIG__")
dst = Path("__REMOTE_PLAN_CONFIG__")
cfg = json.loads(src.read_text(encoding="utf-8"))
cfg["worker_mode"] = "cold"
cfg["initial_hot_workers"] = 0
cfg["max_hot_workers"] = 5
cfg["auto_default_enabled"] = True
cfg["auto_default_idle_seconds"] = 90
cfg["auto_default_gpu"] = "__DEFAULT_WORKER__"
cfg["auto_default_model"] = "__DEFAULT_MODEL__"
cfg["startup_warm_workers"] = ["__DEFAULT_WORKER__"]
cfg["startup_meta_tasks"] = [
    {
        "name": "startup_single_default",
        "command": "load_llm",
        "target_model": "__DEFAULT_MODEL__",
        "load_mode": "single",
        "candidate_workers": ["__DEFAULT_WORKER__"],
    }
]
dst.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print(dst)
PY
__REMOTE_PYTHON__ __REMOTE_PREFLIGHT__ \
  --shared-root /mnt/shared \
  --config config.plan.json \
  --clear-orphan-queue-locks \
  --cleanup-stale-runtime-meta \
  --cleanup-test-runtime-containers \
  --fail-on-active-processing \
  --json
pkill -f /mnt/shared/agents/brain.py || true
pkill -f /mnt/shared/agents/gpu.py || true
pkill -f /mnt/shared/agents/startup.py || true
sleep 2
setsid __REMOTE_PYTHON__ __REMOTE_STARTUP__ --config __REMOTE_PLAN_CONFIG__ >> __REMOTE_LOG__ 2>&1 < /dev/null &
sleep 4
python3 - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "/mnt/shared/agents")
from control_policy import write_control_policy
write_control_policy(
    Path("/mnt/shared"),
    "plan_ready",
    reason="plan_mode_requested",
    updated_by="start_plan_mode.py",
)
PY
pgrep -af startup.py || true
pgrep -af brain.py || true
pgrep -af gpu.py || true
"""
    remote_script = (
        remote_script
        .replace("__REMOTE_BASE_CONFIG__", REMOTE_BASE_CONFIG)
        .replace("__REMOTE_PLAN_CONFIG__", REMOTE_PLAN_CONFIG)
        .replace("__REMOTE_PYTHON__", REMOTE_PYTHON)
        .replace("__REMOTE_PREFLIGHT__", REMOTE_PREFLIGHT)
        .replace("__REMOTE_STARTUP__", REMOTE_STARTUP)
        .replace("__REMOTE_LOG__", REMOTE_LOG)
        .replace("__DEFAULT_WORKER__", DEFAULT_WORKER)
        .replace("__DEFAULT_MODEL__", DEFAULT_MODEL)
    )
    proc = _run_remote(remote_script, timeout=args.timeout)
    ok = proc.returncode == 0
    result = {
        "ok": ok,
        "mode": "plans",
        "config": REMOTE_PLAN_CONFIG,
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "").strip(),
        "stderr": (proc.stderr or "").strip(),
    }
    if args.json:
        print(json.dumps(result))
    else:
        print(json.dumps(result, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
