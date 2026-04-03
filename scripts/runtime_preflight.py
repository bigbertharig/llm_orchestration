#!/usr/bin/env python3
"""Run rig-side runtime preflight from the operator repo."""

from __future__ import annotations

import argparse
import subprocess
import sys


REMOTE_HOST = "gpu"
REMOTE_PYTHON = "/home/bryan/llm-orchestration-venv/bin/python"
REMOTE_SCRIPT = "/mnt/shared/scripts/runtime_preflight.py"
SHARED_ALIASES = (
    "/mnt/shared",
    "/home/bryan/llm_orchestration/shared",
    "/media/bryan/shared",
)


def _to_runtime_shared_path(path_text: str) -> str:
    value = str(path_text)
    for prefix in SHARED_ALIASES:
        if value == prefix or value.startswith(prefix + "/"):
            return "/mnt/shared" + value[len(prefix):]
    return value


def main() -> int:
    ap = argparse.ArgumentParser(description="Run rig-side runtime preflight")
    ap.add_argument("--shared-root", default="/mnt/shared")
    ap.add_argument("--config", default="config.benchmark.json")
    ap.add_argument("--catalog", default="/mnt/shared/plans/shoulders/benchmarking/models.catalog.json")
    ap.add_argument("--runtime-meta-stale-seconds", type=int, default=180)
    ap.add_argument("--clear-orphan-queue-locks", action="store_true")
    ap.add_argument("--cleanup-stale-runtime-meta", action="store_true")
    ap.add_argument("--cleanup-test-runtime-containers", action="store_true")
    ap.add_argument("--fail-on-active-processing", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cmd = [
        "ssh",
        "-o", "BatchMode=yes",
        REMOTE_HOST,
        REMOTE_PYTHON,
        REMOTE_SCRIPT,
        "--shared-root", args.shared_root,
        "--config", _to_runtime_shared_path(args.config),
        "--catalog", _to_runtime_shared_path(args.catalog),
        "--runtime-meta-stale-seconds", str(args.runtime_meta_stale_seconds),
    ]
    if args.clear_orphan_queue_locks:
        cmd.append("--clear-orphan-queue-locks")
    if args.cleanup_stale_runtime_meta:
        cmd.append("--cleanup-stale-runtime-meta")
    if args.cleanup_test_runtime_containers:
        cmd.append("--cleanup-test-runtime-containers")
    if args.fail_on_active_processing:
        cmd.append("--fail-on-active-processing")
    if args.json:
        cmd.append("--json")

    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr:
        print(proc.stderr.rstrip(), file=sys.stderr)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
