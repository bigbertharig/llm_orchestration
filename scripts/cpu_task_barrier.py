#!/usr/bin/env python3
"""
Route CPU-class queue tasks away from GPU workers.

This script rewrites queued tasks with:
  - task_class == "cpu"
  - executor == "worker"
to:
  - executor = "brain"

Run it as a loop on Pi to keep GPU rig from claiming CPU tasks.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def _normalize_batch_summary_command(command: str) -> str:
    """Ensure summary command runs in the venv and uses python3."""
    marker = "/scripts/generate_batch_summary.py"
    if marker not in command:
        return command
    if "source ~/ml-env/bin/activate" in command and "python3 " in command:
        return command
    return (
        "source ~/ml-env/bin/activate && "
        + command.replace("python ", "python3 ", 1)
    )


def route_cpu_tasks(queue_dir: Path, dry_run: bool = False) -> int:
    changed = 0
    for task_file in sorted(queue_dir.glob("*.json")):
        if task_file.name.endswith(".heartbeat.json"):
            continue

        task = _load_json(task_file)
        if not task:
            continue

        if task.get("task_class") != "cpu":
            continue
        if task.get("executor") == "brain":
            continue
        if task.get("status") not in (None, "queued"):
            continue

        task["executor"] = "brain"
        task["cpu_task_rerouted"] = True
        task["cpu_task_rerouted_at"] = datetime.now().isoformat()

        # Fix known summary-task shell env issue when brain auto-generates it.
        if task.get("name") == "batch_summary" and isinstance(task.get("command"), str):
            fixed = _normalize_batch_summary_command(task["command"])
            if fixed != task["command"]:
                task["command"] = fixed
                task["cpu_task_rerouted_summary_fix"] = True
        changed += 1

        if not dry_run:
            _save_json(task_file, task)

    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Route cpu tasks to brain executor")
    parser.add_argument(
        "--queue-dir",
        default="/media/bryan/shared/tasks/queue",
        help="Task queue directory",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=2,
        help="Polling interval in watch mode",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Run continuously",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes without writing files",
    )
    args = parser.parse_args()

    queue_dir = Path(args.queue_dir)
    if not queue_dir.exists():
        print(f"ERROR: queue dir not found: {queue_dir}")
        return 1

    if not args.watch:
        changed = route_cpu_tasks(queue_dir, dry_run=args.dry_run)
        print(f"cpu_task_barrier: rerouted={changed} dry_run={args.dry_run}")
        return 0

    print(
        f"cpu_task_barrier: watching {queue_dir} every {args.interval_seconds}s "
        f"(dry_run={args.dry_run})"
    )
    while True:
        changed = route_cpu_tasks(queue_dir, dry_run=args.dry_run)
        if changed:
            print(
                f"{datetime.now().isoformat()} cpu_task_barrier: rerouted={changed}"
            )
        time.sleep(max(1, args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
