#!/usr/bin/env python3
"""Clean stale task heartbeat files in tasks/processing.

Stale heartbeat files can remain after orphan recovery or worker crashes and
can make dashboards show UUID-only "tasks". This script removes heartbeats that
no longer have an active processing task file.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def detect_shared_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    for candidate in ("/mnt/shared", "/media/bryan/shared"):
        p = Path(candidate)
        if p.exists():
            return p.resolve()
    raise SystemExit("ERROR: could not detect shared root (checked /mnt/shared, /media/bryan/shared)")


def parse_updated_at(text: str) -> datetime | None:
    value = (text or "").strip()
    if not value:
        return None
    try:
        # Handles standard ISO strings with or without timezone suffix.
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def age_seconds(now: datetime, hb_file: Path) -> float:
    try:
        data = json.loads(hb_file.read_text(encoding="utf-8"))
    except Exception:
        data = {}

    updated = parse_updated_at(str(data.get("updated_at", "")))
    if updated is None:
        return max(0.0, now.timestamp() - hb_file.stat().st_mtime)

    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    return max(0.0, (now.astimezone(timezone.utc) - updated.astimezone(timezone.utc)).total_seconds())


def main() -> int:
    parser = argparse.ArgumentParser(description="Remove stale task heartbeat files from tasks/processing")
    parser.add_argument("--shared-root", default=None, help="Shared root path (auto-detects if omitted)")
    parser.add_argument(
        "--min-age-seconds",
        type=int,
        default=120,
        help="Only remove heartbeat files older than this age (seconds)",
    )
    parser.add_argument("--apply", action="store_true", help="Actually delete files (default is dry-run)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview mode (default behavior unless --apply is set)",
    )
    args = parser.parse_args()

    shared_root = detect_shared_root(args.shared_root)
    processing = shared_root / "tasks" / "processing"
    queue = shared_root / "tasks" / "queue"
    complete = shared_root / "tasks" / "complete"
    failed = shared_root / "tasks" / "failed"

    if not processing.exists():
        raise SystemExit(f"ERROR: processing dir not found: {processing}")

    now = datetime.now(timezone.utc)
    total = 0
    stale = 0
    removed = 0

    for hb_file in sorted(processing.glob("*.heartbeat.json")):
        total += 1
        task_id = hb_file.name.replace(".heartbeat.json", "")
        processing_task = processing / f"{task_id}.json"
        queue_task = queue / f"{task_id}.json"
        complete_task = complete / f"{task_id}.json"
        failed_task = failed / f"{task_id}.json"

        # Keep if task still appears live in processing or has been re-queued.
        if processing_task.exists() or queue_task.exists():
            continue

        # Candidate stale heartbeat: no live task file.
        hb_age = age_seconds(now, hb_file)
        if hb_age < max(0, args.min_age_seconds):
            continue

        stale += 1
        status_hint = "finished" if (complete_task.exists() or failed_task.exists()) else "orphan"
        print(f"stale heartbeat: {hb_file} age={int(hb_age)}s status={status_hint}")
        if args.apply:
            try:
                hb_file.unlink()
                removed += 1
            except OSError as exc:
                print(f"ERROR: failed to remove {hb_file}: {exc}")

    mode = "apply" if args.apply else "dry-run"
    print(
        f"cleanup_task_heartbeats mode={mode} root={shared_root} "
        f"total={total} stale={stale} removed={removed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
