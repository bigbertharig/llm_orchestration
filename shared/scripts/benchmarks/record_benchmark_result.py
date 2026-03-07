#!/usr/bin/env python3
"""Append benchmark run records and refresh reference report."""

from __future__ import annotations

import argparse
import json
import subprocess
import uuid
from datetime import datetime
from pathlib import Path


def now_iso() -> str:
    return datetime.now().isoformat()


def normalize_score_pct(score: float) -> float:
    """Normalize heterogeneous benchmark scores to a 0-100 percentage scale.

    Rules:
    - 0..1 values are interpreted as ratios and scaled by 100.
    - 1..100 values are interpreted as already-percent.
    - values outside 0..100 are clipped to keep comparisons bounded.
    """
    if score <= 1.0:
        pct = score * 100.0
    else:
        pct = score
    if pct < 0.0:
        return 0.0
    if pct > 100.0:
        return 100.0
    return pct


def main() -> int:
    ap = argparse.ArgumentParser(description="Record one benchmark result row to JSONL.")
    ap.add_argument("--model", required=True, help="Model id (example: qwen2.5-coder:14b)")
    ap.add_argument("--test-id", required=True, help="Test id from benchmark catalog")
    ap.add_argument("--score", required=True, type=float, help="Numeric score value")
    ap.add_argument("--metric", required=True, help="Metric name (example: accuracy, pass@1)")
    ap.add_argument("--harness", default="", help="Harness used (lm_eval, evalplus, swebench, ...)")
    ap.add_argument("--suite", default="", help="Suite/preset id if applicable")
    ap.add_argument("--run-at", default="", help="Override timestamp (ISO8601). Default: now")
    ap.add_argument("--notes", default="", help="Optional short notes")
    ap.add_argument("--records", default="/mnt/shared/logs/benchmarks/model_benchmark_records.jsonl")
    ap.add_argument("--reference-output", default="/mnt/shared/logs/benchmarks/MODEL_BENCHMARK_REFERENCE.md")
    args = ap.parse_args()

    run_at = args.run_at.strip() or now_iso()
    records_path = Path(args.records).expanduser().resolve()
    records_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_id": str(uuid.uuid4()),
        "run_at": run_at,
        "model": args.model.strip(),
        "test_id": args.test_id.strip(),
        "score": args.score,
        "score_pct": normalize_score_pct(float(args.score)),
        "metric": args.metric.strip(),
        "harness": args.harness.strip(),
        "suite": args.suite.strip(),
        "notes": args.notes.strip()
    }

    with records_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")

    print(f"Recorded result: {records_path}")

    builder = Path(__file__).resolve().parent / "build_benchmark_reference.py"
    cmd = [
        "python3",
        str(builder),
        "--records",
        str(records_path),
        "--output",
        str(Path(args.reference_output).expanduser().resolve())
    ]
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
