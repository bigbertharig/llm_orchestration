"""Resolve a reviewed model role from Orchestration's imported local catalog.

    python3 tools/role_model.py [ROLE]          print the model id (default: single_1060)
    python3 tools/role_model.py ROLE --json     model, profile, gguf_path, gguf_relative

Run tools/sync_model_library.py after benchmarking publishes a new model export.
This helper reads no benchmarking checkout or Python module. Standard library only.
"""

import json
import sys
from pathlib import Path
from .manager import models

CATALOG = Path(__file__).resolve().parents[1] / "model_catalog.json"
DEFAULT_ROLE = "single_1060"


def _catalog() -> dict:
    doc = models.catalog()
    if not isinstance(doc.get("models"), dict) or not isinstance(doc.get("roles"), dict):
        raise ValueError("Orchestration model catalog is incomplete; sync from benchmarking")
    return doc


def _summary(model: str, profile: str, entry: dict) -> dict:
    if profile not in entry.get("profiles", {}):
        raise KeyError(f"model {model} has no profile {profile}")
    gguf = entry.get("gguf_path") or ""
    return {"model": model, "profile": profile, "gguf_path": gguf,
            "gguf_relative": gguf.split("/models/", 1)[1] if "/models/" in gguf else Path(gguf).name}


def resolve(role: str = DEFAULT_ROLE) -> dict:
    """Return canonical model, profile and path for a reviewed role."""
    doc = _catalog()
    spec = doc["roles"][role]
    return _summary(spec["model"], spec.get("profile") or "default", doc["models"][spec["model"]])


def resolve_model(model: str) -> dict:
    """Return canonical model, default profile and path for an ID or alias."""
    doc = _catalog()
    if model in doc["models"]:
        canonical = model
    else:
        canonical = next((mid for mid, entry in doc["models"].items()
                          if model in entry.get("aliases", [])), None)
        if canonical is None:
            raise KeyError(f"unknown model {model}")
    return _summary(canonical, "default", doc["models"][canonical])


if __name__ == "__main__":
    args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    info = resolve(args[0] if args else DEFAULT_ROLE)
    print(json.dumps(info) if "--json" in sys.argv else info["model"])
