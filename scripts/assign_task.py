#!/usr/bin/env python3
"""Operator override: pin a queued orchestration task to one GPU worker (or unpin it).

    python3 scripts/assign_task.py TASK_ID gpu-3          pin (task id or a unique prefix of it)
    python3 scripts/assign_task.py TASK_ID --clear        unpin
    python3 scripts/assign_task.py --list                 queued tasks and their pins

Sets `target_worker` on the task file in <shared>/tasks/queue/ under the same lock workers use to
claim it. Workers skip a task pinned to another worker (work tasks and meta tasks alike). Only queued
tasks can be pinned: a task already in processing/ has a worker; requeue it first.
Part of the rig overrides in llm-gateway docs/MANAGER.md ("Reservations").
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from filelock import FileLock

DEFAULT_SHARED = [Path("/mnt/shared"), Path("/media/bryan/shared")]


def shared_root(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    for p in DEFAULT_SHARED:
        if (p / "tasks" / "queue").is_dir():
            return p
    sys.exit("no shared drive found (give --shared)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("task", nargs="?")
    ap.add_argument("worker", nargs="?", help="worker name, e.g. gpu-3")
    ap.add_argument("--clear", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--shared")
    a = ap.parse_args()
    queue = shared_root(a.shared) / "tasks" / "queue"
    files = sorted(f for f in queue.glob("*.json"))
    if a.list:
        for f in files:
            try:
                t = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            print(f"{t.get('task_id', f.stem)[:12]:<12} {t.get('task_class', ''):<6} "
                  f"pin={t.get('target_worker') or '-':<7} {str(t.get('name') or t.get('command') or '')[:60]}")
        return 0
    if not a.task or not (a.worker or a.clear):
        ap.error("give TASK_ID and a worker, or --clear, or --list")
    matches = []
    for f in files:
        try:
            tid = str(json.loads(f.read_text(encoding="utf-8")).get("task_id", f.stem))
        except (OSError, ValueError):
            continue
        if tid.startswith(a.task) or f.stem.startswith(a.task):
            matches.append(f)
    if len(matches) != 1:
        sys.exit(f"{len(matches)} queued tasks match {a.task!r}" + ("" if matches else " (already claimed?)"))
    path = matches[0]
    with FileLock(str(path) + ".lock", timeout=10):
        if not path.exists():
            sys.exit(f"{path.name} was claimed meanwhile")
        task = json.loads(path.read_text(encoding="utf-8"))
        if a.clear:
            task.pop("target_worker", None)
            task.pop("target_gpu", None)
        else:
            task["target_worker"] = a.worker
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(task, indent=2), encoding="utf-8")
        tmp.replace(path)
    print(f"{task.get('task_id')}: " + ("unpinned" if a.clear else f"pinned to {a.worker}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
