"""Orchestration's imported model library. No benchmarking checkout is needed at runtime."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


SNAPSHOT_PATH = Path(__file__).with_name("model_library.json")


def load_snapshot(path: Path = SNAPSHOT_PATH) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"orchestration model library unavailable at {path}; run shared/scripts/sync_model_library.py"
        ) from exc
    if data.get("schema") != "orchestration-model-library/1":
        raise RuntimeError(f"unsupported orchestration model library schema in {path}")
    if not isinstance(data.get("registry", {}).get("models"), dict) or not isinstance(data.get("roles", {}).get("roles"), dict):
        raise RuntimeError(f"incomplete orchestration model library at {path}")
    return data


def load_registry() -> dict[str, Any]:
    return load_snapshot()["registry"]


def load_roles() -> dict[str, Any]:
    return load_snapshot()["roles"]


def runtime_image(registry: dict[str, Any] | None = None) -> str:
    return str((registry or load_registry())["runtime_image"])


def resolve(model_id: str, profile: str = "default", registry: dict[str, Any] | None = None) -> dict[str, Any]:
    registry = registry or load_registry()
    models = registry["models"]
    canonical = model_id if model_id in models else next(
        (name for name, entry in models.items() if model_id in entry.get("aliases", [])), None
    )
    if canonical is None:
        raise KeyError(model_id)
    entry = models[canonical]
    profiles = entry["profiles"]
    if profile not in profiles:
        raise KeyError(f"model {canonical} has no profile {profile}")
    merged = copy.deepcopy(profiles["default"])
    if profile != "default":
        for key, value in profiles[profile].items():
            if key == "runtime":
                merged["runtime"] = {**merged.get("runtime", {}), **value}
            else:
                merged[key] = copy.deepcopy(value)
    merged.setdefault("extra_args", [])
    gguf = Path(entry["gguf"])
    return {
        "model": canonical,
        "profile": profile,
        "status": entry["status"],
        **({"in_testing": True} if entry.get("in_testing") else {}),
        "gguf_path": str(gguf if gguf.is_absolute() else Path(registry["archive_root"]) / gguf),
        "schedule": copy.deepcopy(entry.get("schedule", {})),
        **merged,
    }
