#!/usr/bin/env python3
"""Import benchmarking model knowledge into orchestration-owned runtime files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


SHARED = Path(__file__).resolve().parent.parent
AGENTS = SHARED / "agents"
DEFAULT_SOURCE = SHARED / "plans" / "shoulders" / "benchmarking"
SNAPSHOT = AGENTS / "model_library.json"
CATALOG = AGENTS / "models.catalog.json"
ACCESS = AGENTS / "model_access.json"
TUNING = AGENTS / "model_tuning_profiles.json"
SOURCE_NAME = "model_library_export.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rendered(value: dict) -> str:
    return json.dumps(value, indent=2, ensure_ascii=True) + "\n"


def build(source: Path) -> tuple[dict, dict, dict, dict]:
    artifact = read_json(source / SOURCE_NAME)
    if artifact.get("schema") != "gpu-rig-model-library/1":
        raise ValueError("unsupported benchmarking model library export")
    payload = {key: value for key, value in artifact.items() if key != "source_sha256"}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                       ensure_ascii=True).encode("utf-8")).hexdigest()
    if digest != artifact.get("source_sha256"):
        raise ValueError("benchmarking model library export hash mismatch")
    registry, roles, recipes, catalog, tuning = (artifact[key] for key in
        ("registry", "roles", "load_recipes", "models_catalog", "tuning_profiles"))
    resolved_tuning = artifact["resolved_tuning_profiles"]
    if registry.get("schema") != "model-registry/1" or roles.get("schema") != "model-roles/1":
        raise ValueError("unsupported benchmarking registry or roles schema")
    if recipes.get("schema_version") != 1 or not isinstance(recipes.get("recipes"), dict):
        raise ValueError("unsupported benchmarking load-recipes schema")
    if not isinstance(tuning.get("models"), dict):
        raise ValueError("benchmarking tuning profiles have no models")
    if not isinstance(resolved_tuning, dict) or set(resolved_tuning) != set(tuning["models"]):
        raise ValueError("benchmarking resolved tuning profiles are incomplete")
    models = registry.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("benchmarking registry has no models")
    for role, spec in roles["roles"].items():
        if spec.get("model") not in models:
            raise ValueError(f"role {role} names an unknown model")
    for recipe_id, recipe in recipes["recipes"].items():
        if recipe.get("model") not in models:
            raise ValueError(f"recipe {recipe_id} names an unknown model")
    snapshot = {
        "schema": "orchestration-model-library/1",
        "source_sha256": artifact["source_sha256"],
        "registry": registry,
        "roles": roles,
        "load_recipes": recipes,
        "resolved_tuning_profiles": resolved_tuning,
    }
    # The local interactive model list is a planning projection, not a ready roster.
    access = {
        "schema_version": 1,
        "models": sorted(name for name, entry in models.items()
                         if entry.get("status") == "active" and not entry.get("in_testing")
                         and "interactive" in entry.get("profiles", {})),
    }
    return snapshot, catalog, access, tuning


def changes(old: dict, new: dict) -> dict[str, list[str]]:
    before, after = set(old), set(new)
    return {
        "added": sorted(after - before),
        "removed": sorted(before - after),
        "changed": sorted(key for key in before & after if old[key] != new[key]),
    }


def report(snapshot: dict, catalog: dict, access: dict, tuning: dict) -> dict:
    try:
        old = read_json(SNAPSHOT)
        old_catalog = read_json(CATALOG)
        old_access = read_json(ACCESS)
        old_tuning = read_json(TUNING)
    except (OSError, ValueError):
        old, old_catalog, old_access, old_tuning = {}, {}, {}, {}
    old_comparable = {key: value for key, value in old.items() if key != "imported_at"}
    return {
        "source_sha256": snapshot["source_sha256"],
        "current_sha256": old.get("source_sha256"),
        "models": changes(old.get("registry", {}).get("models", {}), snapshot["registry"]["models"]),
        "roles": changes(old.get("roles", {}).get("roles", {}), snapshot["roles"]["roles"]),
        "load_recipes": changes(old.get("load_recipes", {}).get("recipes", {}),
                                snapshot["load_recipes"]["recipes"]),
        "catalog_changed": old_catalog != catalog or CATALOG.read_text(encoding="utf-8") != rendered(catalog),
        "interactive_models_changed": old_access != access,
        "tuning_profiles_changed": old_tuning != tuning,
        "runtime_image_changed": old.get("registry", {}).get("runtime_image") != snapshot["registry"].get("runtime_image"),
        "orchestration_defaults_changed": old.get("registry", {}).get("orchestration_defaults") != snapshot["registry"].get("orchestration_defaults"),
        "pending": (not old.get("imported_at") or old_comparable != snapshot or old_catalog != catalog
                    or old_access != access or old_tuning != tuning or
                    SNAPSHOT.read_text(encoding="utf-8") != rendered({**snapshot, "imported_at": old.get("imported_at")})
                    or CATALOG.read_text(encoding="utf-8") != rendered(catalog)
                    or TUNING.read_text(encoding="utf-8") != rendered(tuning)),
    }


def write_files(files: dict[Path, dict]) -> None:
    staged: dict[Path, Path] = {}
    previous: dict[Path, tuple[bytes | None, str | None]] = {}
    replaced: list[Path] = []
    try:
        for path, data in files.items():
            previous[path] = (None, os.readlink(path)) if path.is_symlink() else (path.read_bytes(), None) if path.is_file() else (None, None)
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent,
                                             prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
                json.dump(data, handle, indent=2, ensure_ascii=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
                os.chmod(handle.name, 0o644)
                staged[path] = Path(handle.name)
        for path, temp in staged.items():
            os.replace(temp, path)
            replaced.append(path)
    except Exception:
        for path in reversed(replaced):
            content, link = previous[path]
            if link is not None:
                path.unlink(missing_ok=True)
                path.symlink_to(link)
            elif content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)
        raise
    finally:
        for temp in staged.values():
            temp.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("sync", "what-new", "check"), default="what-new")
    parser.add_argument("--source", type=Path, default=Path(os.getenv("BENCHMARKING_ROOT", DEFAULT_SOURCE)))
    args = parser.parse_args()
    try:
        snapshot, catalog, access, tuning = build(args.source.resolve())
        summary = report(snapshot, catalog, access, tuning)
        if args.action == "sync" and summary["pending"]:
            snapshot["imported_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            write_files({SNAPSHOT: snapshot, CATALOG: catalog, ACCESS: access, TUNING: tuning})
            summary["pending"] = False
            summary["synced"] = True
        print(json.dumps(summary, indent=2))
        return 1 if args.action == "check" and summary["pending"] else 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"model-library import: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
