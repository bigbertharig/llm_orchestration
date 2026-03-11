#!/usr/bin/env python3
"""Start rig in custom model mode (isolated startup + explicit runtime targets).

Flow:
1) Start benchmark mode (cold workers, auto-default disabled)
2) Load requested models sequentially via brain-managed meta tasks
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=False,
        timeout=max(20, timeout),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Start custom runtime mode with explicit model targets")
    ap.add_argument("--models", nargs="+", required=True,
                     help="Model IDs to load onto workers (from models.catalog.json)")
    ap.add_argument("--shared-root", default="/mnt/shared")
    ap.add_argument("--config", default="config.benchmark.json",
                     help="Orchestrator config file name (relative to shared/agents/)")
    ap.add_argument("--skip-benchmark-start", action="store_true")
    ap.add_argument("--force-unload-first", action="store_true",
                     help="Unload all currently loaded workers before loading")
    ap.add_argument("--strict-processing-empty", action="store_true")
    ap.add_argument("--load-timeout-seconds", type=int, default=300)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    benchmark_script = script_dir / "start_benchmark_mode.py"

    result: dict[str, object] = {
        "ok": False,
        "mode": "custom",
        "requested": {
            "models": args.models,
        },
    }

    if not args.skip_benchmark_start:
        bench = _run(
            [sys.executable, str(benchmark_script), "--json", "--timeout", str(args.timeout)],
            timeout=args.timeout,
        )
        bench_json: dict[str, object] = {
            "ok": bench.returncode == 0,
            "returncode": bench.returncode,
            "stdout": (bench.stdout or "").strip(),
            "stderr": (bench.stderr or "").strip(),
        }
        if bench.stdout:
            try:
                parsed = json.loads(bench.stdout.strip().splitlines()[-1])
                if isinstance(parsed, dict):
                    bench_json = parsed
            except Exception:
                pass
        result["benchmark_start"] = bench_json
        if bench.returncode != 0:
            result["message"] = "Benchmark-mode startup failed"
            _emit(result, args.json)
            return 1

    prep_cmd = [
        "ssh",
        "-o", "BatchMode=yes",
        "gpu",
        "/home/bryan/llm-orchestration-venv/bin/python",
        "/mnt/shared/scripts/prepare_llm_runtimes.py",
        "--shared-root", args.shared_root,
        "--config", args.config,
        "--clear-orphan-queue-locks",
        "--load-timeout-seconds", str(args.load_timeout_seconds),
        "--models",
    ] + args.models

    if args.strict_processing_empty:
        prep_cmd.append("--strict-processing-empty")
    if args.force_unload_first:
        prep_cmd.append("--force-unload-first")

    prep = _run(prep_cmd, timeout=args.timeout)
    result["prepare_cmd"] = " ".join(prep_cmd)
    result["prepare"] = {
        "ok": prep.returncode == 0,
        "returncode": prep.returncode,
        "stdout": (prep.stdout or "").strip(),
        "stderr": (prep.stderr or "").strip(),
    }

    if prep.returncode != 0:
        result["message"] = "Custom model load failed"
        _emit(result, args.json)
        return 1

    result["ok"] = True
    result["message"] = f"Custom mode active with {len(args.models)} models loaded"
    _emit(result, args.json)
    return 0


def _emit(payload: dict[str, object], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload))
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
