#!/usr/bin/env python3
"""Start rig in benchmark mode with isolated, non-interfering defaults.

Benchmark mode disables auto-default normalization and startup warm workers,
so manual/runtime benchmark model loads are not preempted by system meta tasks.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys


REMOTE_HOST = "gpu"
REMOTE_BASE_CONFIG = "/mnt/shared/agents/config.json"
REMOTE_BENCH_CONFIG = "/mnt/shared/agents/config.benchmark.json"
REMOTE_STARTUP = "/mnt/shared/agents/startup.py"
REMOTE_PYTHON = "/home/bryan/llm-orchestration-venv/bin/python"
REMOTE_LOG = "/mnt/shared/logs/startup-benchmark.log"


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
    ap = argparse.ArgumentParser(description="Start rig in benchmark startup mode")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    remote_script = f"""set -e
python3 - <<'PY'
import json
from pathlib import Path
src = Path("{REMOTE_BASE_CONFIG}")
dst = Path("{REMOTE_BENCH_CONFIG}")
cfg = json.loads(src.read_text(encoding="utf-8"))
cfg["worker_mode"] = "cold"
cfg["initial_hot_workers"] = 0
cfg["max_hot_workers"] = 0
cfg["auto_default_enabled"] = False
cfg["auto_default_idle_seconds"] = 999999
dst.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print(dst)
PY
pkill -f /mnt/shared/agents/brain.py || true
pkill -f /mnt/shared/agents/gpu.py || true
pkill -f /mnt/shared/agents/startup.py || true
sleep 2
nohup {REMOTE_PYTHON} {REMOTE_STARTUP} --config {REMOTE_BENCH_CONFIG} >> {REMOTE_LOG} 2>&1 < /dev/null &
sleep 4
pgrep -af startup.py || true
pgrep -af brain.py || true
pgrep -af gpu.py || true
"""
    proc = _run_remote(remote_script, timeout=args.timeout)
    ok = proc.returncode == 0
    result = {
        "ok": ok,
        "mode": "benchmark",
        "config": REMOTE_BENCH_CONFIG,
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
