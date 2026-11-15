"""Lookup of outputs for batches executed by orchestration."""

import subprocess
from pathlib import Path
from typing import Any

from .utils import file_mtime_iso

PLAN_SCOPES = ("shoulders", "arms")
STDIO_SNIPPET_LIMIT = 16000
STDIO_TIMEOUT_LIMIT = 8000


def _clip_stdio(text: str, limit: int = STDIO_SNIPPET_LIMIT) -> str:
    """Preserve both head and tail of long output (useful for tracebacks)."""
    if not isinstance(text, str):
        return ""
    if limit <= 0 or len(text) <= limit:
        return text
    head = max(1, int(limit * 0.45))
    tail = max(1, int(limit * 0.45))
    omitted = len(text) - head - tail
    return text[:head] + f"\n\n...[truncated {omitted} chars]...\n\n" + text[-tail:]


def run_shell(cmd: str, timeout_s: int = 120) -> dict[str, Any]:
    """Run shell command and return result dict."""
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": _clip_stdio(proc.stdout),
            "stderr": _clip_stdio(proc.stderr),
            "cmd": cmd,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "ok": False,
            "returncode": -1,
            "stdout": _clip_stdio(e.stdout or "", STDIO_TIMEOUT_LIMIT) if isinstance(e.stdout, str) else "",
            "stderr": _clip_stdio(e.stderr or "", STDIO_TIMEOUT_LIMIT) if isinstance(e.stderr, str) else "",
            "cmd": cmd,
            "error": f"timeout after {timeout_s}s",
        }








def plans_dir_for_scope(shared_path: Path, plan_scope: str) -> Path:
    """Locate an existing batch's scope without preparing a new plan."""
    return shared_path / "plans" / plan_scope












def find_batch_dir(shared_path: Path, batch_id: str) -> tuple[str, Path] | None:
    """Find batch directory by batch ID."""
    if not batch_id:
        return None
    for scope in PLAN_SCOPES:
        scoped_dir = plans_dir_for_scope(shared_path, scope)
        if not scoped_dir.exists():
            continue
        for plan_folder in scoped_dir.iterdir():
            if not plan_folder.is_dir():
                continue
            candidate = plan_folder / "history" / batch_id
            if candidate.exists() and candidate.is_dir():
                return plan_folder.name, candidate
    return None


def collect_batch_outputs(shared_path: Path, batch_id: str, max_files: int = 40) -> dict[str, Any]:
    """Collect output files from a batch directory."""
    found = find_batch_dir(shared_path, batch_id)
    if not found:
        return {"ok": False, "message": f"Batch directory not found for {batch_id}"}

    plan_name, batch_dir = found
    files: list[Path] = []
    preferred = [
        batch_dir / "output",
        batch_dir / "results",
    ]
    for root in preferred:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file():
                files.append(p)

    for extra in [
        batch_dir / "execution_stats.json",
        batch_dir / "manifest.json",
    ]:
        if extra.exists() and extra.is_file():
            files.append(extra)

    # De-duplicate while preserving order.
    dedup: list[Path] = []
    seen: set[Path] = set()
    for p in files:
        if p in seen:
            continue
        seen.add(p)
        dedup.append(p)
    files = dedup[:max_files]

    def to_mnt(path: Path) -> str:
        rel = path.relative_to(shared_path)
        return f"/mnt/shared/{rel.as_posix()}"

    file_rows: list[dict[str, Any]] = []
    for p in files:
        rel = p.relative_to(batch_dir).as_posix()
        try:
            size_b = int(p.stat().st_size)
        except Exception:
            size_b = 0
        preview = ""
        if p.suffix.lower() in {".md", ".txt", ".json", ".csv", ".log"}:
            try:
                preview = p.read_text(encoding="utf-8", errors="replace")[:2000]
            except Exception:
                preview = ""
        file_rows.append(
            {
                "name": p.name,
                "relative_path": rel,
                "path": str(p),
                "mnt_path": to_mnt(p),
                "size_bytes": size_b,
                "updated_at": file_mtime_iso(p),
                "preview": preview,
            }
        )

    return {
        "ok": True,
        "batch_id": batch_id,
        "plan": plan_name,
        "batch_path": str(batch_dir),
        "batch_mnt_path": to_mnt(batch_dir),
        "files": file_rows,
        "message": f"Found {len(file_rows)} output/result files",
    }
