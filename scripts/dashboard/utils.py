"""Utility functions and constants for the dashboard package."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

# Heartbeat thresholds
HEARTBEAT_WARN_S = 60
HEARTBEAT_BAD_S = 120
HEARTBEAT_MAX_S = 600

# Validation patterns
CONFIG_KEY_RE = re.compile(r"^[A-Z0-9_]{1,64}$")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
GITHUB_URL_RE = re.compile(r"^https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?$")

# Config keys that accept file paths for inline content
INLINE_FILE_KEYS = ("QUERY_FILE", "REQUEST_FILE", "INPUT_FILE", "PROMPT_FILE", "CLAIM_FILE")


def load_json(path: Path) -> dict[str, Any] | None:
    """Load JSON file, returning None on error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_config(config_path: Path) -> dict[str, Any]:
    """Load configuration JSON file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_shared_path(config_path: Path, config: dict[str, Any]) -> Path:
    """Resolve the shared path from config relative to config file location."""
    shared = Path(config.get("shared_path", "../"))
    if shared.is_absolute():
        return shared
    return (config_path.resolve().parent / shared).resolve()


def iter_task_files(folder_path: Path):
    """Iterate task JSON files in a folder, skipping heartbeat files."""
    if not folder_path.exists():
        return
    for task_file in folder_path.glob("*.json"):
        if task_file.name.endswith(".heartbeat.json"):
            continue
        yield task_file


def heartbeat_age_seconds(last_updated: str | None) -> int | None:
    """Calculate seconds since last heartbeat update."""
    if not last_updated:
        return None
    try:
        dt = datetime.fromisoformat(last_updated)
        return int((datetime.now() - dt).total_seconds())
    except Exception:
        return None


def parse_iso_datetime(value: str | None) -> datetime | None:
    """Parse ISO datetime string, handling space separator variant."""
    if not value:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    normalized = raw if "T" in raw else raw.replace(" ", "T")
    try:
        return datetime.fromisoformat(normalized)
    except Exception:
        return None


def format_duration_short(total_seconds: int | float | None) -> str:
    """Format duration as short human-readable string (e.g., '1h 23m 45s')."""
    if total_seconds is None:
        return "unknown"
    try:
        seconds = max(0, int(total_seconds))
    except Exception:
        return "unknown"
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def file_mtime_iso(path: Path) -> str | None:
    """Get file modification time as ISO string."""
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    except Exception:
        return None


def sanitize_text(value: Any, *, max_len: int = 12000, single_line: bool = False) -> str:
    """Sanitize text by removing control characters and optionally collapsing whitespace."""
    text = str(value or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_CHAR_RE.sub("", text)
    text = text[:max_len]
    if single_line:
        text = " ".join(text.split())
    return text.strip()


def sanitize_config_value(value: Any) -> Any:
    """Sanitize a single config value (must be scalar JSON type)."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return sanitize_text(value, max_len=4000, single_line=False)
    raise ValueError("config values must be scalar JSON types (string/number/bool/null)")


def sanitize_config_object(raw_cfg: dict[str, Any]) -> dict[str, Any]:
    """Sanitize entire config object, validating keys and values."""
    if len(raw_cfg) > 120:
        raise ValueError("too many config keys (max 120)")
    out: dict[str, Any] = {}
    for k, v in raw_cfg.items():
        if not isinstance(k, str):
            raise ValueError("config keys must be strings")
        key = k.strip().upper()
        if not CONFIG_KEY_RE.fullmatch(key):
            raise ValueError(f"invalid config key: {k!r}")
        out[key] = sanitize_config_value(v)
    return out


def normalize_github_url(url: str) -> str:
    """Normalize GitHub URL to canonical form."""
    normalized = sanitize_text(url, max_len=300, single_line=True)
    m = GITHUB_URL_RE.match(normalized)
    if not m:
        raise ValueError("repo URL must be https://github.com/<owner>/<repo>")
    owner, repo = m.group(1), m.group(2)
    return f"https://github.com/{owner}/{repo}"
