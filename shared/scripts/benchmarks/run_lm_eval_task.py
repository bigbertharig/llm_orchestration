#!/usr/bin/env python3
"""Run a single benchmark task from benchmark_catalog.json via lm-eval."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from compatibility import derive_backend_id, find_certified_test, load_status, required_tokenizer


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def default_benchmark_python() -> str:
    configured = os.environ.get("BENCHMARK_PYTHON", "").strip()
    if configured:
        return configured
    candidate = Path.home() / "ml-env" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return sys.executable


def default_output_dir() -> str:
    media_path = Path("/media/bryan/shared/logs/benchmarks")
    if media_path.exists():
        return str(media_path)
    return "/mnt/shared/logs/benchmarks"


def load_tests(path: Path) -> dict[str, dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, str]] = {}
    for item in data.get("tests", []):
        test_id = str(item.get("id", "")).strip()
        if test_id:
            out[test_id] = item
    return out


def lm_eval_task_set(python_bin: str) -> set[str]:
    cmd = [python_bin, "-m", "lm_eval", "ls", "tasks"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip() or "unknown error"
        raise SystemExit(f"Unable to list lm-eval tasks: {msg}")
    tasks = set()
    for raw in proc.stdout.splitlines():
        line = raw.strip()
        if (
            not line
            or line.startswith("-")
            or line.startswith("Available tasks")
            or line.startswith("Total tasks")
        ):
            continue
        # Newer lm-eval prints a table: | task_name | path | mode |
        if line.startswith("|") and line.endswith("|"):
            cols = [c.strip() for c in line.strip("|").split("|")]
            if cols:
                name = cols[0]
                if name and name.lower() != "group" and " " not in name:
                    tasks.add(name)
            continue
        # Fallback for older/simple output formats
        for part in line.split(","):
            name = part.strip()
            if name and " " not in name:
                tasks.add(name)
    return tasks


def parse_model_id(model_args: str) -> str:
    for part in model_args.split(","):
        token = part.strip()
        if token.startswith("model="):
            return token.split("=", 1)[1].strip()
    return ""


def metric_priority(metric: str) -> tuple[int, str]:
    key = metric.strip().lower()
    preferred = [
        "exact_match,none",
        "exact_match",
        "acc,none",
        "acc_norm,none",
        "accuracy,none",
        "accuracy",
        "f1,none",
        "f1",
        "pass@1,none",
        "pass@1",
        "mc2,none",
        "mc2_accuracy",
    ]
    try:
        return preferred.index(key), key
    except ValueError:
        return 9999, key


def select_numeric_metric(result_obj: dict[str, Any]) -> tuple[float, str] | None:
    numeric: list[tuple[str, float]] = []
    for k, v in result_obj.items():
        if isinstance(v, (int, float)):
            name = str(k).strip()
            if name.endswith("_stderr,none") or name.endswith("_stderr"):
                continue
            numeric.append((name, float(v)))
    if not numeric:
        return None
    numeric.sort(key=lambda item: metric_priority(item[0]))
    metric, score = numeric[0]
    return score, metric


def extract_score_from_lm_eval_output(out_dir: Path, task_name: str) -> tuple[float, str]:
    candidates = sorted(out_dir.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        results = data.get("results")
        if not isinstance(results, dict):
            continue

        task_result = results.get(task_name)
        if not isinstance(task_result, dict):
            for k, v in results.items():
                if str(k).strip().lower().startswith(task_name.lower()) and isinstance(v, dict):
                    task_result = v
                    break
        if not isinstance(task_result, dict):
            continue

        picked = select_numeric_metric(task_result)
        if picked is not None:
            return picked

    raise SystemExit(
        f"lm-eval completed but no numeric metric found for task '{task_name}' under output path: {out_dir}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a single lm-eval task by benchmark ID")
    ap.add_argument("--id", required=True, help="Test id from benchmark catalog (example: gsm8k)")
    ap.add_argument("--catalog", default="benchmark_catalog.json", help="Catalog JSON path")
    ap.add_argument(
        "--python",
        default=default_benchmark_python(),
        help="Python interpreter for lm-eval and benchmark helpers.",
    )
    ap.add_argument("--model", default="local-chat-completions", help="lm-eval model backend")
    ap.add_argument(
        "--model-args",
        default=(
            "model=qwen2.5-coder:7b,"
            "base_url=http://localhost:11434/v1/chat/completions,"
            "api_key=ollama,"
            "tokenizer=Qwen/Qwen2.5-Coder-7B-Instruct,"
            "eos_string=<|im_end|>"
        ),
        help="lm-eval model args",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", default="auto")
    ap.add_argument("--limit", type=float, default=None, help="Optional sample cap (debug runs)")
    ap.add_argument("--output-dir", default=default_output_dir())
    ap.add_argument("--suite", default="individual", help="Suite label for benchmark ledger")
    ap.add_argument("--apply-chat-template", action="store_true", help="Pass --apply_chat_template to lm-eval")
    ap.add_argument("--no-record", action="store_true", help="Do not append auto-record to benchmark ledger")
    ap.add_argument("--status-path", default=str(Path(__file__).resolve().parent / "benchmark_status.json"))
    ap.add_argument(
        "--allow-uncertified",
        action="store_true",
        help="Allow execution when the backend/test pair has not been certified yet.",
    )
    ap.add_argument(
        "--skip-compat-check",
        action="store_true",
        help="Bypass compatibility status gating. Intended for certification probes.",
    )
    args = ap.parse_args()

    this_dir = Path(__file__).resolve().parent
    catalog_path = (this_dir / args.catalog).resolve() if not Path(args.catalog).is_absolute() else Path(args.catalog)
    tests = load_tests(catalog_path)

    if args.id not in tests:
        known = ", ".join(sorted(tests.keys()))
        raise SystemExit(f"Unknown test id '{args.id}'. Available: {known}")

    selected = tests[args.id]
    harness = selected.get("harness")
    if harness != "lm_eval":
        raise SystemExit(
            f"Test '{args.id}' uses harness '{harness}', not lm_eval. "
            "This runner only executes lm_eval tests."
        )

    task_name = str(selected.get("task_name", "")).strip()
    if not task_name:
        raise SystemExit(f"Test '{args.id}' has no task_name in catalog.")

    available_tasks = lm_eval_task_set(args.python)
    if task_name not in available_tasks:
        raise SystemExit(
            f"lm-eval task '{task_name}' is not available in this environment. "
            "Run `python3 -m lm_eval --tasks list` to inspect installed task names."
        )

    status = load_status(Path(args.status_path))
    backend_id = derive_backend_id(args.model, args.model_args, args.apply_chat_template)
    if backend_id and not args.skip_compat_check:
        certified = find_certified_test(status, backend_id, args.id)
        if certified is not None:
            state = str(certified.get("state", "")).strip()
            note = str(certified.get("note", "")).strip()
            if state == "blocked":
                raise SystemExit(
                    f"Benchmark test '{args.id}' is blocked for backend '{backend_id}': {note}"
                )
        elif not args.allow_uncertified:
            raise SystemExit(
                f"Benchmark test '{args.id}' is not yet certified for backend '{backend_id}'. "
                "Run certify_benchmark_backend.py first or pass --allow-uncertified."
            )

    model_id = parse_model_id(args.model_args)
    tokenizer = required_tokenizer(status, model_id)
    if tokenizer and "tokenizer=" not in args.model_args:
        sep = "," if args.model_args.strip() else ""
        args.model_args = f"{args.model_args}{sep}tokenizer={tokenizer}"

    out_dir = Path(args.output_dir).expanduser().resolve() / f"{args.id}_{now_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python,
        "-m",
        "lm_eval",
        "run",
        "--model",
        args.model,
        "--model_args",
        args.model_args,
        "--tasks",
        task_name,
        "--device",
        args.device,
        "--batch_size",
        args.batch_size,
        "--output_path",
        str(out_dir),
    ]
    if args.apply_chat_template:
        cmd.append("--apply_chat_template")
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])

    print(f"Running benchmark id={args.id} task={task_name}")
    print("Command:", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    if not args.no_record:
        score, metric = extract_score_from_lm_eval_output(out_dir, task_name)
        model_id = parse_model_id(args.model_args) or args.model
        recorder = this_dir / "record_benchmark_result.py"
        record_cmd = [
            args.python,
            str(recorder),
            "--model",
            model_id,
            "--test-id",
            args.id,
            "--score",
            str(score),
            "--metric",
            metric,
            "--harness",
            "lm_eval",
            "--suite",
            args.suite,
        ]
        print("Auto-recording benchmark result...")
        rec_proc = subprocess.run(record_cmd, check=False)
        if rec_proc.returncode != 0:
            raise SystemExit(rec_proc.returncode)

    print(f"Results written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
