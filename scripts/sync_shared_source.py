#!/usr/bin/env python3
"""Compare or deploy tracked shared source files to the live shared mount.

The default mode is read-only. --apply copies changed and missing tracked files
atomically. It never deletes live files and never writes protected core or
permission-policy paths.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path


TRACKED_ROOTS = (
    "shared/agents",
    "shared/scripts",
    "shared/workspace",
)
PROTECTED_PREFIXES = (
    "shared/core/",
    "shared/agents/permissions/",
)


@dataclass(frozen=True)
class Difference:
    relative_path: str
    status: str


def repository_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path(__file__).resolve().parent,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("sync_shared_source.py must run from a Git checkout")
    return Path(result.stdout.strip()).resolve()


def tracked_paths(repo: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", *TRACKED_ROOTS],
        cwd=repo,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    paths = [
        item.decode("utf-8")
        for item in result.stdout.split(b"\0")
        if item
    ]
    return [path for path in paths if not is_protected(path)]


def is_protected(relative_path: str) -> bool:
    normalized = relative_path.rstrip("/") + "/"
    return any(normalized.startswith(prefix) for prefix in PROTECTED_PREFIXES)


def live_path(live_root: Path, relative_path: str) -> Path:
    shared_prefix = "shared/"
    if not relative_path.startswith(shared_prefix):
        raise ValueError(f"path is outside shared/: {relative_path}")
    return live_root / relative_path[len(shared_prefix):]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compare_path(source: Path, destination: Path) -> str:
    if source.is_symlink():
        if not destination.is_symlink():
            return "missing" if not os.path.lexists(destination) else "changed"
        return "same" if os.readlink(source) == os.readlink(destination) else "changed"
    if not destination.is_file() or destination.is_symlink():
        return "missing" if not os.path.lexists(destination) else "changed"
    source_mode = stat.S_IMODE(source.stat().st_mode)
    destination_mode = stat.S_IMODE(destination.stat().st_mode)
    if source_mode != destination_mode:
        return "changed"
    return "same" if digest(source) == digest(destination) else "changed"


def differences(repo: Path, live_root: Path) -> list[Difference]:
    rows: list[Difference] = []
    for relative_path in tracked_paths(repo):
        source = repo / relative_path
        destination = live_path(live_root, relative_path)
        if not source.is_file() and not source.is_symlink():
            rows.append(Difference(relative_path, "missing_source"))
            continue
        status_value = compare_path(source, destination)
        if status_value != "same":
            rows.append(Difference(relative_path, status_value))
    return rows


def deploy_one(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(
        f".{destination.name}.sync-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        if source.is_symlink():
            temp_path.symlink_to(os.readlink(source))
        else:
            shutil.copy2(source, temp_path)
        temp_path.replace(destination)
    finally:
        if os.path.lexists(temp_path):
            temp_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare or deploy tracked shared source to the live mount"
    )
    parser.add_argument("--live-root", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    repo = repository_root()
    live_root = Path(args.live_root).resolve()
    if not (live_root / "agents").is_dir() or not (live_root / "workspace").is_dir():
        parser.error(f"live root does not look like the shared mount: {live_root}")

    rows = differences(repo, live_root)
    missing_source = [row for row in rows if row.status == "missing_source"]
    if missing_source:
        for row in missing_source:
            print(f"ERROR\t{row.relative_path}\tmissing_source")
        return 2

    if not rows:
        print("shared source and live tracked files are synchronized")
        return 0

    for row in rows:
        print(f"{row.status.upper()}\t{row.relative_path}")

    if not args.apply:
        print(f"{len(rows)} file(s) differ; rerun with --apply to deploy")
        return 1

    for row in rows:
        relative_path = row.relative_path
        if is_protected(relative_path):
            raise RuntimeError(f"refusing protected path: {relative_path}")
        deploy_one(repo / relative_path, live_path(live_root, relative_path))
    print(f"deployed {len(rows)} tracked file(s) without deleting live files")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
