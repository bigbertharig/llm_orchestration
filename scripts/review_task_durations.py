#!/usr/bin/env python3
"""Review real task durations and every time-out, to tune the time limits from evidence.

    python3 scripts/review_task_durations.py --shared-path /media/bryan/shared
    python3 scripts/review_task_durations.py --shared-path /mnt/shared --history /mnt/shared/plans/arms/<plan>/history

Reads task files in tasks/complete, tasks/failed and tasks/late (and any --history trees). Finished
tasks carry `duration_s`; tasks a time limit moved carry `timed_out` and `timeouts` (written by the brain,
see shared/workspace/brain-behavior.md "Time limits"). Read-only: it reports, it never changes a limit.

The no-progress limits measure the gap since a task's last progress heartbeat, not its total time, so a
total duration below a limit is safe by construction; a duration near or above one is worth a look.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_tasks(paths):
    seen = {}
    for root in paths:
        if not root.exists():
            continue
        files = root.rglob("*.json") if root.is_dir() else [root]
        for f in files:
            if f.name.endswith(".heartbeat.json"):
                continue
            try:
                task = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            if isinstance(task, dict) and task.get("task_id"):
                seen[(task["task_id"], task.get("claim_id"), f.parent.name)] = task
    return list(seen.values())


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * (len(values) - 1) + 0.5))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shared-path", default="/media/bryan/shared")
    ap.add_argument("--history", action="append", default=[], help="extra history tree(s) to scan")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    shared = Path(args.shared_path)
    tasks = load_tasks([shared / "tasks" / lane for lane in ("complete", "failed", "late")]
                       + [Path(h) for h in args.history])

    durations = defaultdict(list)
    timeouts = []
    for t in tasks:
        cls = t.get("task_class", "cpu")
        if isinstance(t.get("duration_s"), (int, float)):
            durations[cls].append(float(t["duration_s"]))
        for to in t.get("timeouts") or []:
            timeouts.append({"task_id": t["task_id"][:8], "name": t.get("name"), "task_class": cls,
                             "status": t.get("status"), **to})

    by_class = {cls: {"n": len(v), "p50_s": round(statistics.median(v), 1), "p95_s": round(pct(v, 0.95), 1),
                      "max_s": round(max(v), 1)} for cls, v in sorted(durations.items())}
    by_limit = defaultdict(list)
    for to in timeouts:
        by_limit[to.get("limit")].append(to)

    if args.json:
        print(json.dumps({"durations_by_class": by_class, "timeouts": timeouts}, indent=1))
        return 0

    print(f"tasks read: {len(tasks)}")
    print("\nDurations of finished tasks (seconds):")
    print(f"  {'class':8} {'n':>6} {'p50':>9} {'p95':>9} {'max':>9}")
    for cls, s in by_class.items():
        print(f"  {cls:8} {s['n']:>6} {s['p50_s']:>9} {s['p95_s']:>9} {s['max_s']:>9}")
    if not by_class:
        print("  none yet (durations are recorded from 2026-10-20 on)")
    print(f"\nTimed out: {len(timeouts)}")
    for limit, rows in sorted(by_limit.items(), key=lambda kv: str(kv[0])):
        observed = [r.get("observed_s") for r in rows if isinstance(r.get("observed_s"), (int, float))]
        print(f"  {limit} (limit {rows[0].get('limit_s')} s): {len(rows)} time-outs, "
              f"observed {min(observed) if observed else '-'}-{max(observed) if observed else '-'} s")
        for r in rows[-10:]:
            print(f"    {r.get('at', '')[:19]} {r['task_id']} {r.get('name') or ''} on {r.get('worker')} "
                  f"-> {r.get('action')} (final status {r.get('status')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
