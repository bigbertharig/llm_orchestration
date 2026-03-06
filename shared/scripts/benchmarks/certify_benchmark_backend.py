#!/usr/bin/env python3
"""Probe a benchmark/backend pair and update benchmark_status.json."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from compatibility import derive_backend_id, load_status, save_status, set_certified_test
from run_lm_eval_task import default_benchmark_python, default_output_dir


def iso_now() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def parse_model_id(model_args: str) -> str:
    for part in str(model_args or "").split(","):
        token = part.strip()
        if token.startswith("model="):
            return token.split("=", 1)[1].strip()
    return ""


def first_line(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    return cleaned.splitlines()[0].strip()


def classify_result(stdout: str, stderr: str, returncode: int) -> tuple[str, str]:
    combined = "\n".join(part for part in [stdout, stderr] if part).strip()
    lines = [line.strip() for line in combined.splitlines() if line.strip()]
    for line in reversed(lines):
        if "required packages ['tenacity'] are not installed" in line:
            return "env_blocked", line
        if "No module named lm_eval" in line:
            return "env_blocked", line
        if "Loglikelihood is not supported for chat completions" in line:
            return "blocked", line
        if "cannot unmarshal array into Go struct field CompletionRequest.prompt" in line:
            return "blocked", line
        if "Permission denied: '/mnt/shared'" in line:
            return "env_blocked", line
        if line.startswith("ModuleNotFoundError:") or line.startswith("PermissionError:"):
            return "env_blocked", line
        if line.startswith("NotImplementedError:"):
            return "blocked", line
    if returncode == 0:
        return "supported", "Probe run completed successfully."
    return "blocked", first_line(stderr) or first_line(stdout) or f"runner exited with code {returncode}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Certify a benchmark/backend pair and update benchmark_status.json")
    ap.add_argument("--id", required=True, help="Benchmark test id")
    ap.add_argument("--model", required=True, help="lm_eval model backend")
    ap.add_argument("--model-args", required=True, help="lm_eval model args")
    ap.add_argument("--catalog", default="benchmark_catalog.json")
    ap.add_argument("--status-path", default=str(Path(__file__).resolve().parent / "benchmark_status.json"))
    ap.add_argument("--python", default=default_benchmark_python(), help="Python interpreter for lm-eval runs.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", default="auto")
    ap.add_argument("--limit", type=float, default=3)
    ap.add_argument("--apply-chat-template", action="store_true")
    ap.add_argument("--output-dir", default=default_output_dir(), help="Output directory for probe harness artifacts.")
    args = ap.parse_args()

    this_dir = Path(__file__).resolve().parent
    runner = this_dir / "run_lm_eval_task.py"
    status_path = Path(args.status_path).expanduser().resolve()
    backend_id = derive_backend_id(args.model, args.model_args, args.apply_chat_template)
    if not backend_id:
        raise SystemExit("Unable to derive backend_id from --model and --model-args")

    cmd = [
        args.python,
        str(runner),
        "--id",
        args.id,
        "--catalog",
        args.catalog,
        "--python",
        args.python,
        "--model",
        args.model,
        "--model-args",
        args.model_args,
        "--device",
        args.device,
        "--batch-size",
        args.batch_size,
        "--output-dir",
        args.output_dir,
        "--limit",
        str(args.limit),
        "--no-record",
        "--allow-uncertified",
        "--skip-compat-check",
    ]
    if args.apply_chat_template:
        cmd.append("--apply-chat-template")

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

    status = load_status(status_path)
    task_name = args.id
    model_id = parse_model_id(args.model_args)
    state, note = classify_result(proc.stdout, proc.stderr, proc.returncode)
    set_certified_test(
        status,
        backend_id=backend_id,
        test_id=args.id,
        task_name=task_name,
        state=state,
        note=note,
        model_id=model_id,
        observed_at=iso_now(),
    )
    save_status(status, status_path)

    result = {
        "backend_id": backend_id,
        "test_id": args.id,
        "state": state,
        "model_id": model_id,
        "note": note,
        "status_path": str(status_path),
    }
    print(json.dumps(result, indent=2))
    if proc.returncode != 0:
        print(proc.stdout, end="")
        print(proc.stderr, end="", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
