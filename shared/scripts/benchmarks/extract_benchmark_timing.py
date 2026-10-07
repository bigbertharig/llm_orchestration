#!/usr/bin/env python3
"""
Extract benchmark timing data from all history dirs and produce a speed table.

Outputs total wall-clock time per model per benchmark test.
Pipeline: from final_summary.json (duration_seconds) and stage_updates.jsonl (per-test)
Code: from status.json timestamps (run_start → per-task updated_at)
Reasoning: from results_*.json (total_evaluation_time_seconds per task)

Usage:
    python3 extract_benchmark_timing.py [--shared /mnt/shared]
"""

import json
import os
import glob
import sys
from datetime import datetime
from collections import defaultdict

SHARED = "/mnt/shared"
for arg in sys.argv[1:]:
    if arg.startswith("--shared"):
        continue
    SHARED = arg
if "--shared" in sys.argv:
    idx = sys.argv.index("--shared")
    if idx + 1 < len(sys.argv):
        SHARED = sys.argv[idx + 1]

BENCH_ROOT = os.path.join(SHARED, "logs/benchmarks")

# Active models we care about
ACTIVE_MODELS = {
    "qwen2.5-coder:7b", "qwen2.5-coder:14b",
    "llama3.2:3b", "smollm3:3b",
    "gemma-4:e4b", "gemma-4:e2b", "gemma-4:26b-a4b", "gemma-4:31b",
    "qwen3.6:27b", "qwen3.6:35b-a3b",
    "phi-4:14b",
}

# Normalize model IDs from filenames/results to canonical form
MODEL_ALIASES = {
    "qwen2.5-coder__7b": "qwen2.5-coder:7b",
    "qwen2.5-coder__14b": "qwen2.5-coder:14b",
    "qwen2.5-coder__32b": "qwen2.5-coder:32b",
    "llama3.2__3b": "llama3.2:3b",
    "smollm3__3b": "smollm3:3b",
    "gemma-4__e4b": "gemma-4:e4b",
    "gemma-4__e2b": "gemma-4:e2b",
    "gemma-4__26b-a4b": "gemma-4:26b-a4b",
    "gemma-4__31b": "gemma-4:31b",
    "qwen3.6__27b": "qwen3.6:27b",
    "qwen3.6__35b-a3b": "qwen3.6:35b-a3b",
    "phi-4__14b": "phi-4:14b",
}

def normalize_model(model_id):
    """Normalize model ID to canonical short form."""
    m = model_id.strip()
    # Direct match
    if m in ACTIVE_MODELS:
        return m
    # Double-underscore form (from lm-eval sanitized paths)
    if m in MODEL_ALIASES:
        return MODEL_ALIASES[m]
    # Try colon form
    m2 = m.replace("__", ":")
    if m2 in ACTIVE_MODELS:
        return m2
    return m


def fmt_time(seconds):
    """Format seconds as human-readable."""
    if seconds is None:
        return "—"
    s = int(float(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def parse_iso(ts):
    """Parse ISO timestamp, handling various formats."""
    for fmt in [
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S",
    ]:
        try:
            return datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            pass
    return None


def extract_reasoning():
    """Extract timing from bench-reasoning results."""
    # results structure: history/<run_dir>/<task>/<model_sanitized>/results_*.json
    timing = {}  # (model, task) → seconds
    history = os.path.join(BENCH_ROOT, "bench-reasoning/history")
    if not os.path.isdir(history):
        return timing

    for run_dir in os.listdir(history):
        run_path = os.path.join(history, run_dir)
        if not os.path.isdir(run_path):
            continue
        for task in ["gsm8k", "bbh", "drop"]:
            task_path = os.path.join(run_path, task)
            if not os.path.isdir(task_path):
                continue
            for model_dir in os.listdir(task_path):
                model_path = os.path.join(task_path, model_dir)
                if not os.path.isdir(model_path):
                    continue
                model = normalize_model(model_dir)
                if model not in ACTIVE_MODELS:
                    continue
                # Find the newest results file
                result_files = sorted(glob.glob(os.path.join(model_path, "results_*.json")))
                if not result_files:
                    continue
                try:
                    with open(result_files[-1]) as f:
                        data = json.load(f)
                    t = data.get("total_evaluation_time_seconds")
                    n_samples = data.get("n-samples", {})
                    # Get the effective sample count for this task
                    limit = None
                    for k, v in n_samples.items():
                        if isinstance(v, dict):
                            limit = v.get("effective")
                    if t is not None:
                        key = (model, task)
                        # Keep the one with highest limit (most complete run)
                        if key not in timing or (limit and limit > timing[key].get("limit", 0)):
                            timing[key] = {"seconds": t, "limit": limit or 0, "run": run_dir}
                except (json.JSONDecodeError, KeyError):
                    continue
    return timing


def extract_code():
    """Extract timing from bench-code results."""
    timing = {}  # (model, task) → seconds
    history = os.path.join(BENCH_ROOT, "bench-code/history")
    if not os.path.isdir(history):
        return timing

    for run_dir in os.listdir(history):
        run_path = os.path.join(history, run_dir)
        if not os.path.isdir(run_path):
            continue
        status_file = os.path.join(run_path, "status.json")
        if not os.path.isfile(status_file):
            continue
        try:
            with open(status_file) as f:
                status = json.load(f)
            model = normalize_model(status.get("model", ""))
            if model not in ACTIVE_MODELS:
                continue
            if status.get("state") != "completed":
                continue
            run_start = parse_iso(status.get("run_start"))
            if not run_start:
                continue

            tasks = status.get("tasks", {})
            prev_end = run_start
            for task_name in ["humaneval", "mbpp"]:
                task_info = tasks.get(task_name, {})
                task_end = parse_iso(task_info.get("updated_at"))
                if task_end and prev_end:
                    # Make both offset-naive for subtraction
                    if task_end.tzinfo and not prev_end.tzinfo:
                        task_end = task_end.replace(tzinfo=None)
                    elif prev_end.tzinfo and not task_end.tzinfo:
                        prev_end = prev_end.replace(tzinfo=None)
                    delta = (task_end - prev_end).total_seconds()
                    if delta > 0:
                        key = (model, task_name)
                        if key not in timing or delta > timing[key]["seconds"]:
                            # Keep latest/longest run (likely highest quality)
                            timing[key] = {"seconds": delta, "run": run_dir}
                    prev_end = task_end
        except (json.JSONDecodeError, KeyError):
            continue
    return timing


def extract_pipeline():
    """Extract timing from bench-pipeline results."""
    timing = {}  # (model, "pipeline") → seconds
    history = os.path.join(BENCH_ROOT, "bench-pipeline/history")
    if not os.path.isdir(history):
        return timing

    for f in glob.glob(os.path.join(history, "*_final_summary.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
            model = normalize_model(data.get("model", ""))
            if model not in ACTIVE_MODELS:
                continue
            dur = data.get("duration_seconds")
            if dur is not None:
                key = (model, "pipeline")
                # Keep latest (highest duration_seconds as proxy for most complete)
                if key not in timing:
                    timing[key] = {"seconds": dur, "run": os.path.basename(f)}
        except (json.JSONDecodeError, KeyError):
            continue
    return timing


def main():
    print("Extracting benchmark timing data...\n")

    reasoning = extract_reasoning()
    code = extract_code()
    pipeline = extract_pipeline()

    # Merge all timing
    all_timing = {}
    all_timing.update(reasoning)
    all_timing.update(code)
    all_timing.update(pipeline)

    # Collect all tasks
    tasks = ["pipeline", "humaneval", "mbpp", "gsm8k", "bbh", "drop"]

    # Sort models by tier
    tier_order = [
        "gemma-4:e2b", "gemma-4:e4b", "llama3.2:3b", "smollm3:3b",
        "qwen2.5-coder:7b",
        "qwen2.5-coder:14b", "phi-4:14b",
        "gemma-4:26b-a4b", "gemma-4:31b", "qwen3.6:27b", "qwen3.6:35b-a3b",
    ]
    models = [m for m in tier_order if m in ACTIVE_MODELS]

    # Print table
    header = "| Model | pipeline | humaneval | mbpp | gsm8k | bbh | drop |"
    sep =    "|-------|----------|-----------|------|-------|-----|------|"
    print(header)
    print(sep)

    for model in models:
        row = f"| `{model}` |"
        for task in tasks:
            key = (model, task)
            entry = all_timing.get(key)
            if entry:
                row += f" {fmt_time(entry['seconds'])} |"
            else:
                row += " — |"
        print(row)

    # Also dump JSON for machine use
    json_out = {}
    for (model, task), entry in all_timing.items():
        if model not in ACTIVE_MODELS:
            continue
        if model not in json_out:
            json_out[model] = {}
        json_out[model][task] = {
            "seconds": round(float(entry["seconds"]), 1),
            "formatted": fmt_time(entry["seconds"]),
            "run": entry.get("run", ""),
            "limit": entry.get("limit"),
        }

    json_path = os.path.join(SHARED, "logs/benchmarks/benchmark_timing.json")
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2, sort_keys=True)
    print(f"\nJSON written to: {json_path}")


if __name__ == "__main__":
    main()
