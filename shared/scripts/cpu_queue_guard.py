#!/usr/bin/env python3
"""Guardrail: detect queued CPU tasks when no CPU workers are alive."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_iso(ts: str):
    if not ts:
        return None
    ts = ts.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def now_utc():
    return datetime.now(timezone.utc)


def load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def count_cpu_tasks(queue_dir: Path) -> int:
    total = 0
    for p in queue_dir.glob("*.json"):
        task = load_json(p)
        if not task:
            continue
        if task.get("executor") == "brain":
            continue
        if task.get("task_class", "cpu") == "cpu":
            total += 1
    return total


def find_live_cpu_workers(heartbeats_dir: Path, stale_seconds: int):
    live = []
    cutoff = now_utc().timestamp() - stale_seconds
    for p in heartbeats_dir.glob("*.json"):
        hb = load_json(p)
        if not hb:
            continue
        if hb.get("worker_type") != "cpu":
            continue
        dt = parse_iso(hb.get("last_updated", ""))
        if not dt:
            continue
        if dt.timestamp() >= cutoff:
            live.append(hb.get("name") or p.stem)
    return sorted(set(live))


def append_log(log_file: Path, message: str):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def main():
    parser = argparse.ArgumentParser(description="Detect cpu-task queue starvation")
    parser.add_argument("--shared-root", default="/media/bryan/shared")
    parser.add_argument("--stale-seconds", type=int, default=45)
    parser.add_argument("--log-file", default="/media/bryan/shared/logs/cpu_queue_guard.log")
    parser.add_argument("--fail-on-alert", action="store_true")
    args = parser.parse_args()

    shared = Path(args.shared_root)
    queue = shared / "tasks" / "queue"
    heartbeats = shared / "heartbeats"
    log_file = Path(args.log_file)

    cpu_tasks = count_cpu_tasks(queue)
    live_workers = find_live_cpu_workers(heartbeats, args.stale_seconds)
    ts = now_utc().isoformat()

    if cpu_tasks > 0 and not live_workers:
        msg = (
            f"{ts} ALERT cpu_tasks_queued={cpu_tasks} cpu_workers_live=0 "
            f"action=bring_up_cpu_agent"
        )
        append_log(log_file, msg)
        print(msg)
        raise SystemExit(2 if args.fail_on_alert else 0)

    msg = (
        f"{ts} OK cpu_tasks_queued={cpu_tasks} "
        f"cpu_workers_live={len(live_workers)} workers={','.join(live_workers) if live_workers else '-'}"
    )
    append_log(log_file, msg)
    print(msg)


if __name__ == "__main__":
    main()
