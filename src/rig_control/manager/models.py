"""Models for "load model X on N cards" (MANAGER.md, "Model loads").

How each model loads (GGUF, context, batch, extra args per named profile) comes from the Orchestration's own
`model_catalog.json`, which `tools/generate_profiles.py` imports from Benchmarking at startup when available.
The last valid local snapshot remains usable if Benchmarking is unavailable. This module checks
catalog offers and builds the requested load spec; the rig computes detailed fit and placement.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
CATALOG_PATH = Path(os.getenv("MODEL_CATALOG", Path.home() / ".config/llm-orchestration/model_catalog.json"))
# Catalog GGUF paths are written as the rig sees the drive; on the Pi it is mounted elsewhere.
CATALOG_SHARED = "/mnt/shared"
LOCAL_SHARED_CANDIDATES = ("/media/bryan/shared", "/mnt/shared")
# Registry profile per use when the request names none (falls back to `default`).
PROFILE_BY_USE = {"benchmark": "default", "chat": "interactive", "worker": "orchestration"}
# Lease controller type per use (the rig's lease record groups leases by it).
CONTROLLER_BY_USE = {"benchmark": "benchmarks", "chat": "interactive", "worker": "plans"}

_cache: dict[str, Any] = {"key": None, "doc": None}


def catalog() -> dict[str, Any]:
    """The imported catalog, re-read when the file changes. FileNotFoundError until the importer has run."""
    path = Path(os.getenv("MODEL_CATALOG") or CATALOG_PATH)
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    if _cache["key"] != key:
        _cache.update(key=key, doc=json.loads(path.read_text(encoding="utf-8")))
    return _cache["doc"]


def find_model(model: str) -> tuple[str, dict[str, Any]]:
    """Catalog entry for a model id or one of its aliases. KeyError for an unknown model."""
    models = catalog()["models"]
    if model in models:
        return model, models[model]
    for mid, entry in models.items():
        if model in entry.get("aliases", []):
            return mid, entry
    raise KeyError(f"model {model} is not in {CATALOG_PATH.name}; re-run tools/generate_profiles.py if it is new")


def local_path(catalog_path: str) -> str:
    """Where this machine can read a catalog GGUF path (/mnt/shared/... on the rig)."""
    if not catalog_path.startswith(CATALOG_SHARED + "/"):
        return catalog_path
    for root in [os.getenv("SHARED_ROOT")] if os.getenv("SHARED_ROOT") else LOCAL_SHARED_CANDIDATES:
        if root and os.path.isdir(root):
            return root + catalog_path[len(CATALOG_SHARED):]
    return catalog_path


def profile_id(model: str, gpus: list[int]) -> str:
    """Rig profile id for a model load: the model plus its cards, e.g. ornith-1.5-9b-g1-3."""
    slug = re.sub(r"[^A-Za-z0-9.-]+", "-", model).strip("-").lower()
    return f"{slug}-g{'-'.join(str(g) for g in sorted(gpus))}"


def resolve(model: str, use: str, profile: str | None = None) -> dict[str, Any]:
    """Merged settings for the model and use: {model, profile, status, gguf_path, runtime, extra_args, image}.
    KeyError for an unknown model or profile."""
    mid, entry = find_model(model)
    if use != "benchmark" and (entry.get("in_testing") or entry.get("status") != "active"):
        raise KeyError(f"model {mid} is in testing or not approved for routine runtime use")
    name = profile or PROFILE_BY_USE.get(use, "default")
    if name not in entry["profiles"]:
        if profile:
            raise KeyError(f"model {model} has no profile {profile}; profiles: {sorted(entry['profiles'])}")
        name = "default"
    return {**entry["profiles"][name], "model": mid, "profile": name, "status": entry["status"],
            "gguf_path": entry["gguf_path"], "image": catalog()["image"],
            "model_size_bytes": entry.get("model_size_bytes"),
            "ngram_size_bytes": entry.get("ngram_size_bytes", 0),
            "total_size_bytes": (entry["model_size_bytes"] + entry.get("ngram_size_bytes", 0))
            if entry.get("model_size_bytes") is not None else None,
            "strata": entry.get("strata"), "loading_stage": entry.get("loading_stage")}
