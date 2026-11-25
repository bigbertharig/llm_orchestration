#!/usr/bin/env python3
"""Validate benchmarking's library and import changed model data into Manager."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import generate_profiles as gen


def _without_build_time(value: dict) -> dict:
    result = json.loads(json.dumps(value))
    if isinstance(result.get("summary"), dict):
        result["summary"].pop("generated_at", None)
    # The dashboard summary's timestamp can also be nested in older catalogs.
    return result


def changed_keys(before: dict, after: dict) -> dict[str, list[str]]:
    old, new = set(before), set(after)
    return {
        "added": sorted(new - old),
        "removed": sorted(old - new),
        "changed": sorted(key for key in old & new if before[key] != after[key]),
    }


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def inspect(root: Path, catalog: dict, offer: dict) -> dict:
    old_catalog = _read_json(gen.CATALOG_PATH)
    old_offer = _read_json(gen.OUTPUT_PATH)
    source_hash = gen.load_export(root)["source_sha256"]
    old_offers = {item["id"]: item for item in old_offer.get("remote_profiles", [])}
    new_offers = {item["id"]: item for item in offer["remote_profiles"]}
    return {
        "source": str(root),
        "source_sha256": source_hash,
        "registry_updated": catalog.get("registry_updated"),
        "roles_updated": catalog.get("roles_updated"),
        "models": changed_keys(old_catalog.get("models", {}), catalog["models"]),
        "roles": changed_keys(old_catalog.get("roles", {}), catalog["roles"]),
        "load_recipes": changed_keys(old_catalog.get("load_recipes", {}).get("recipes", {}),
                                     catalog["load_recipes"]["recipes"]),
        "offers": changed_keys(old_offers, new_offers),
        "runtime_image_changed": old_catalog.get("image") != catalog.get("image"),
        "catalog_stale": (_without_build_time(old_catalog) != _without_build_time(catalog)
                          or gen.CATALOG_PATH.read_text(encoding="utf-8") != gen.render(catalog)),
        "offers_stale": (old_offer != offer or gen.OUTPUT_PATH.read_text(encoding="utf-8") != gen.render(offer)),
    }


def _atomic_text(path: Path, value: str) -> None:
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n",
                                         prefix=f".{path.name}.", suffix=".tmp",
                                         dir=path.parent, delete=False) as tmp:
            tmp.write(value)
            tmp.flush()
            os.fsync(tmp.fileno())
            temp_name = tmp.name
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="show changes without writing")
    mode.add_argument("--check", action="store_true", help="exit 1 if an import is pending")
    args = parser.parse_args()

    root = gen.benchmarking_root()
    library = gen.load_library(root)
    registry = library.load_registry(root / "model_registry.json")
    roles = library.load_roles(root / "model_roles.json")
    problems = library.validate(registry, roles) + library.generated_drift(registry)
    offer, offer_problems = gen.build(gen.OFFER, library, registry)
    problems += offer_problems
    if problems:
        for problem in problems:
            print(f"problem: {problem}", file=sys.stderr)
        return 2

    try:
        recipes = gen.load_recipes(root, registry)
    except (OSError, ValueError) as exc:
        print(f"problem: {exc}", file=sys.stderr)
        return 2
    try:
        subagents = gen.load_subagent_catalog(root, registry)
    except (OSError, ValueError) as exc:
        print(f"problem: {exc}", file=sys.stderr)
        return 2
    catalog = gen.build_catalog(library, registry, roles, gen._summary(root), recipes, subagents)
    report = inspect(root, catalog, offer)
    pending = report["catalog_stale"] or report["offers_stale"]
    if pending and not (args.dry_run or args.check):
        # Prepare both documents before replacing either. Keep old bytes for rollback.
        originals = {path: path.read_text(encoding="utf-8") if path.exists() else None
                     for path in (gen.CATALOG_PATH, gen.OUTPUT_PATH)}
        try:
            if report["catalog_stale"]:
                _atomic_text(gen.CATALOG_PATH, gen.render(catalog))
            if report["offers_stale"]:
                _atomic_text(gen.OUTPUT_PATH, gen.render(offer))
        except Exception:
            for path, original in originals.items():
                if original is None:
                    path.unlink(missing_ok=True)
                else:
                    _atomic_text(path, original)
            raise
    report["action"] = ("pending" if args.dry_run or args.check else "updated") if pending else "current"
    print(json.dumps(report, indent=2))
    return 1 if args.check and pending else 0


if __name__ == "__main__":
    raise SystemExit(main())
