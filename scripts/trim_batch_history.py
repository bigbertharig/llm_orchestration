#!/usr/bin/env python3
"""
Trim historical batch artifacts while preserving summaries.

Behavior:
- Finds batch directories under one or more `history/` roots (supports nested lanes).
- Keeps full raw artifacts for the newest N batches per lane.
- For older batches beyond that window, deletes non-summary artifacts.

Default summary preservation:
- manifest.json
- any markdown file
- output/*.json
- output/*.csv
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_KEEP_PATTERNS = [
    "manifest.json",
    "**/*.md",
    "output/*.json",
    "output/*.csv",
]


@dataclass
class BatchInfo:
    path: Path
    lane: str
    run_dt: datetime


def parse_iso_utc(raw: str | None) -> datetime | None:
    if not raw or not isinstance(raw, str):
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def infer_batch_datetime(batch_dir: Path) -> datetime:
    name = batch_dir.name
    if len(name) == 15 and name[8] == "_":
        try:
            return datetime.strptime(name, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
        except Exception:
            pass

    manifest = batch_dir / "manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        for key in ("created_at", "started_at", "completed_at"):
            dt = parse_iso_utc(data.get(key))
            if dt:
                return dt

    return datetime.fromtimestamp(batch_dir.stat().st_mtime, tz=timezone.utc)


def discover_batches(history_root: Path) -> list[BatchInfo]:
    out: list[BatchInfo] = []
    if not history_root.exists():
        return out

    for manifest in history_root.rglob("manifest.json"):
        batch_dir = manifest.parent
        if not batch_dir.is_dir():
            continue
        # Skip non-batch manifests lacking common artifact directories.
        if not any((batch_dir / n).exists() for n in ("results", "transcripts", "logs", "output")):
            continue

        rel = batch_dir.relative_to(history_root)
        lane_parts = rel.parts[:-1]
        lane = "/".join(lane_parts) if lane_parts else "(root)"
        out.append(BatchInfo(path=batch_dir, lane=lane, run_dt=infer_batch_datetime(batch_dir)))
    return out


def should_keep_file(rel_posix: str, keep_patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(rel_posix, pat) for pat in keep_patterns)


def trim_batch(batch_dir: Path, keep_patterns: list[str], dry_run: bool) -> tuple[int, int, int]:
    files_removed = 0
    bytes_removed = 0
    files_skipped = 0
    for p in sorted(batch_dir.rglob("*"), reverse=True):
        if not p.exists():
            continue
        rel = p.relative_to(batch_dir).as_posix()
        if p.is_dir():
            continue
        if should_keep_file(rel, keep_patterns):
            continue
        try:
            size = p.stat().st_size
        except Exception:
            size = 0
        if not dry_run:
            try:
                p.unlink(missing_ok=True)
            except PermissionError:
                files_skipped += 1
                continue
            except OSError:
                files_skipped += 1
                continue
        files_removed += 1
        bytes_removed += size

    if not dry_run:
        for d in sorted(batch_dir.rglob("*"), reverse=True):
            if d.is_dir():
                try:
                    d.rmdir()
                except OSError:
                    pass
    return files_removed, bytes_removed, files_skipped


def format_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    val = float(n)
    for u in units:
        if val < 1024.0 or u == units[-1]:
            return f"{val:.1f}{u}"
        val /= 1024.0
    return f"{n}B"


def main() -> int:
    parser = argparse.ArgumentParser(description="Trim old batch history while keeping summaries")
    parser.add_argument(
        "--history-root",
        action="append",
        default=[],
        help="History directory to trim (repeatable).",
    )
    parser.add_argument(
        "--older-than-days",
        type=int,
        default=7,
        help="Only trim batches older than this many days (default: 7).",
    )
    parser.add_argument(
        "--keep-recent-raw",
        type=int,
        default=2,
        help="Keep full raw artifacts for newest N batches per lane (default: 2).",
    )
    parser.add_argument(
        "--keep-pattern",
        action="append",
        default=[],
        help="Extra glob pattern to preserve (relative to batch dir).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview only; do not delete.")
    args = parser.parse_args()

    roots = [Path(p).resolve() for p in args.history_root] or [
        Path("/media/bryan/shared/plans/shoulders")
    ]
    keep_patterns = list(DEFAULT_KEEP_PATTERNS) + list(args.keep_pattern)
    cutoff = datetime.now(timezone.utc) - timedelta(days=args.older_than_days)

    all_batches: list[BatchInfo] = []
    for root in roots:
        if root.name == "history":
            all_batches.extend(discover_batches(root))
            continue
        for history in root.rglob("history"):
            all_batches.extend(discover_batches(history))

    grouped: dict[tuple[str, str], list[BatchInfo]] = {}
    for b in all_batches:
        # group by project + lane (project inferred from path under shoulders)
        project = "unknown"
        parts = b.path.parts
        if "shoulders" in parts:
            try:
                project = parts[parts.index("shoulders") + 1]
            except Exception:
                project = "unknown"
        key = (project, b.lane)
        grouped.setdefault(key, []).append(b)

    total_files = 0
    total_bytes = 0
    total_batches = 0
    total_skipped = 0

    for (project, lane), batches in sorted(grouped.items()):
        batches = sorted(batches, key=lambda x: x.run_dt, reverse=True)
        keep_raw = {b.path for b in batches[: max(0, args.keep_recent_raw)]}
        trimmed_here = 0
        for b in batches:
            if b.path in keep_raw:
                continue
            if b.run_dt >= cutoff:
                continue
            removed_files, removed_bytes, skipped_files = trim_batch(
                b.path, keep_patterns, args.dry_run
            )
            if removed_files > 0:
                trimmed_here += 1
                total_batches += 1
                total_files += removed_files
                total_bytes += removed_bytes
                total_skipped += skipped_files
                mode = "DRY-RUN" if args.dry_run else "TRIMMED"
                print(
                    f"{mode} [{project} | {lane}] {b.path} removed={removed_files} "
                    f"reclaimed={format_bytes(removed_bytes)} skipped={skipped_files}"
                )
        if trimmed_here == 0:
            print(f"SKIP [{project} | {lane}] no eligible batches")

    mode = "dry-run estimate" if args.dry_run else "deleted"
    print(
        f"\nDone: batches={total_batches} files={total_files} {mode}={format_bytes(total_bytes)} "
        f"skipped={total_skipped}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
