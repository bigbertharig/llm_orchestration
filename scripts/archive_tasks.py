#!/usr/bin/env python3
"""Archive completed/failed task files by batch to keep hot task dirs small.

By default, archives non-active batches from:
  shared/tasks/complete
  shared/tasks/failed
into:
  shared/tasks/archive/<batch_id>/<lane>/

Usage examples:
  python scripts/archive_tasks.py --dry-run
  python scripts/archive_tasks.py
  python scripts/archive_tasks.py --batch-id 20260212_163011 --include-active
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable


BASE_DIR = Path(__file__).resolve().parent.parent
SHARED = BASE_DIR / "shared"
TASKS = SHARED / "tasks"
STATE_FILE = SHARED / "brain" / "state.json"
LANES = ("complete", "failed")


def load_active_batches() -> set[str]:
    if not STATE_FILE.exists():
        return set()
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        active = state.get("active_batches", {})
        if isinstance(active, dict):
            return set(active.keys())
    except Exception:
        pass
    return set()


def iter_task_files(lane: str) -> Iterable[Path]:
    lane_dir = TASKS / lane
    if not lane_dir.exists():
        return []
    files = []
    for p in lane_dir.glob("*.json"):
        if p.name.endswith(".heartbeat.json"):
            continue
        files.append(p)
    return files


def read_task(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def archive_one(path: Path, lane: str, batch_id: str, dry_run: bool) -> bool:
    dest_dir = TASKS / "archive" / batch_id / lane
    dest = dest_dir / path.name
    if dry_run:
        print(f"DRY  {path} -> {dest}")
        return True

    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(dest))
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Archive complete/failed task files by batch")
    parser.add_argument("--batch-id", help="Archive only this batch ID")
    parser.add_argument("--include-active", action="store_true", help="Allow archiving active batch tasks")
    parser.add_argument("--dry-run", action="store_true", help="Show moves without changing files")
    args = parser.parse_args()

    active = load_active_batches()

    moved = 0
    skipped_active = 0
    moved_unbatched = 0

    for lane in LANES:
        for task_file in iter_task_files(lane):
            task = read_task(task_file)
            if not task:
                continue

            batch_id = task.get("batch_id") or "_unbatched"

            if args.batch_id and batch_id != args.batch_id:
                continue

            if (not args.include_active) and (batch_id in active):
                skipped_active += 1
                continue

            if archive_one(task_file, lane, batch_id, args.dry_run):
                moved += 1
                if batch_id == "_unbatched":
                    moved_unbatched += 1

    print("-" * 60)
    print(f"Archived files: {moved}")
    print(f"Skipped active-batch files: {skipped_active}")
    print(f"Archived unbatched files: {moved_unbatched}")
    if active:
        print(f"Active batches protected: {sorted(active)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
