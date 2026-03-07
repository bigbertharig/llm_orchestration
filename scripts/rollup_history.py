#!/usr/bin/env python3
"""Build a cross-run summary for a history/ directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


sys.path.insert(0, str(_repo_root() / "shared" / "agents"))

from run_summary import summarize_history_root, write_history_rollup  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Roll up many history/<batch_id> folders")
    parser.add_argument("history_root", help="Path to history directory containing many run folders")
    parser.add_argument(
        "--use-existing-run-summaries",
        action="store_true",
        help="Use existing RUN_SUMMARY.json files when present instead of refreshing each run",
    )
    parser.add_argument("--print-json", action="store_true", help="Print rollup JSON to stdout")
    args = parser.parse_args()

    history_root = Path(args.history_root).resolve()
    if not history_root.exists() or not history_root.is_dir():
        raise SystemExit(f"History directory not found: {history_root}")

    rollup = summarize_history_root(
        history_root,
        refresh_runs=not args.use_existing_run_summaries,
    )
    outputs = write_history_rollup(history_root, rollup)

    print(f"Wrote {outputs['rollup_json']}")
    print(f"Wrote {outputs['rollup_markdown']}")
    print(f"Wrote {outputs['runs_jsonl']}")
    print(f"Wrote {outputs['failures_jsonl']}")
    if args.print_json:
        print(json.dumps(rollup, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
