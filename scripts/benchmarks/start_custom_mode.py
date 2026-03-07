#!/usr/bin/env python3
"""Start rig in custom model mode (isolated startup + explicit runtime targets).

Flow:
1) Start benchmark mode (cold workers, auto-default disabled)
2) Prepare/load requested brain/single/split models sequentially
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
    ap.add_argument("--brain-model", default="qwen2.5:32b")
    ap.add_argument("--single-model", default="qwen2.5:7b")
    ap.add_argument("--split-model", default="qwen2.5:14b")
    ap.add_argument("--split-candidate-group", default="pair_4_5")
    ap.add_argument("--keep-alive", default="30m")
    ap.add_argument("--shared-root", default="/mnt/shared")
    ap.add_argument("--skip-benchmark-start", action="store_true")
    ap.add_argument("--no-force-unload-first", action="store_true")
    ap.add_argument("--strict-processing-empty", action="store_true")
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    benchmark_script = script_dir / "start_benchmark_mode.py"

    result: dict[str, object] = {
        "ok": False,
        "mode": "custom",
        "requested": {
            "brain_model": args.brain_model,
            "single_model": args.single_model,
            "split_model": args.split_model,
            "split_candidate_group": args.split_candidate_group,
            "keep_alive": args.keep_alive,
        },
    }

    if not args.skip_benchmark_start:
        bench = _run([sys.executable, str(benchmark_script), "--json", "--timeout", str(args.timeout)], timeout=args.timeout)
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
        "-o",
        "BatchMode=yes",
        "gpu",
        "/home/bryan/llm-orchestration-venv/bin/python",
        "/mnt/shared/scripts/prepare_llm_runtimes.py",
        "--shared-root",
        args.shared_root,
        "--clear-orphan-queue-locks",
        "--brain-model",
        args.brain_model,
        "--single-model",
        args.single_model,
        "--split-model",
        args.split_model,
        "--split-candidate-group",
        args.split_candidate_group,
        "--keep-alive",
        args.keep_alive,
    ]
    if args.strict_processing_empty:
        prep_cmd.append("--strict-processing-empty")
    if not args.no_force_unload_first:
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
    result["message"] = "Custom mode active with requested models loaded"
    _emit(result, args.json)
    return 0


def _emit(payload: dict[str, object], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload))
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
