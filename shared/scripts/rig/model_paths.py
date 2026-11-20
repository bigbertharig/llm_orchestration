"""Resolve a model archive path to a verified local hotset copy when available."""

from pathlib import Path

ARCHIVE_MODEL_ROOT = Path("/mnt/shared/models")


def resolve_hotset_gguf(gguf_path: str | Path, config: dict) -> Path:
    path = Path(gguf_path)
    try:
        relative = path.relative_to(ARCHIVE_MODEL_ROOT)
    except ValueError:
        return path
    roots = config.get("model_search_roots") or []
    if not isinstance(roots, list):
        roots = []
    runtime_root = config.get("runtime_model_root")
    for raw in [*roots, runtime_root, str(ARCHIVE_MODEL_ROOT)]:
        if not raw:
            continue
        root = Path(str(raw))
        if root == ARCHIVE_MODEL_ROOT:
            break
        candidate = root / relative
        if candidate.is_file():
            return candidate
    return path
