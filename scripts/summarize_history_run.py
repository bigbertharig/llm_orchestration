#!/usr/bin/env python3
"""Summarize a single history/<batch_id> directory.

Best-effort reducer only. It surfaces the important artifacts and excerpts it
can find without trying to decide what the run means.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


sys.path.insert(0, str(_repo_root() / "shared" / "agents"))

from run_summary import summarize_history_dir, write_run_summary  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a history/<batch_id> folder")
    parser.add_argument("history_dir", help="Path to history/<batch_id> directory")
    parser.add_argument("--print-json", action="store_true", help="Print summary JSON to stdout")
    args = parser.parse_args()

    history_dir = Path(args.history_dir).resolve()
    if not history_dir.exists() or not history_dir.is_dir():
        raise SystemExit(f"History directory not found: {history_dir}")

    summary = summarize_history_dir(history_dir)
    outputs = write_run_summary(history_dir, summary)

    print(f"Wrote {outputs['json']}")
    print(f"Wrote {outputs['markdown']}")
    if args.print_json:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
