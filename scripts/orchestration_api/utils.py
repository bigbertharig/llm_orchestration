"""Small readers used by orchestration status and batch controls."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

HEARTBEAT_BAD_S = 120


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return None


def heartbeat_age_seconds(last_updated: str | None) -> int | None:
    if not last_updated:
        return None
    try:
        return max(0, int((datetime.now() - datetime.fromisoformat(last_updated)).total_seconds()))
    except ValueError:
        return None


def file_mtime_iso(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    except OSError:
        return None
