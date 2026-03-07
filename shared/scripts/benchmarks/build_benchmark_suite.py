#!/usr/bin/env python3
"""Compose benchmark suites from catalog by preset, ids, and tags."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def index_tests(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for test in catalog.get("tests", []):
        test_id = str(test.get("id", "")).strip()
        if test_id:
            out[test_id] = test
    return out


def parse_csv(value: str) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def select_ids_from_tags(tests: dict[str, dict[str, Any]], tags: list[str]) -> list[str]:
    if not tags:
        return []
    wanted = set(tags)
    picked: list[str] = []
    for test_id, test in tests.items():
        test_tags = set(str(t).strip() for t in test.get("tags", []))
        if test_tags.intersection(wanted):
            picked.append(test_id)
    return picked


def preset_ids(presets: dict[str, Any], preset_name: str) -> list[str]:
    for preset in presets.get("presets", []):
        if str(preset.get("id", "")).strip() == preset_name:
            return [str(x).strip() for x in preset.get("include_ids", []) if str(x).strip()]
    raise SystemExit(f"Unknown preset '{preset_name}'.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a benchmark suite file from catalog entries.")
    ap.add_argument("--catalog", default="benchmark_catalog.json")
    ap.add_argument("--presets", default="suite_presets.json")
    ap.add_argument("--preset", default="", help="Preset id from suite_presets.json")
    ap.add_argument("--include-ids", default="", help="Comma-separated test ids to include")
    ap.add_argument("--include-tags", default="", help="Comma-separated tags to include")
    ap.add_argument("--exclude-ids", default="", help="Comma-separated test ids to exclude")
    ap.add_argument("--name", default="", help="Suite name (default: derived from selection)")
    ap.add_argument("--output", required=True, help="Output suite JSON path")
    args = ap.parse_args()

    this_dir = Path(__file__).resolve().parent
    catalog_path = (this_dir / args.catalog).resolve() if not Path(args.catalog).is_absolute() else Path(args.catalog)
    presets_path = (this_dir / args.presets).resolve() if not Path(args.presets).is_absolute() else Path(args.presets)

    catalog = load_json(catalog_path)
    tests_by_id = index_tests(catalog)
    presets = load_json(presets_path)

    chosen_ids: list[str] = []
    if args.preset:
        chosen_ids.extend(preset_ids(presets, args.preset))
    chosen_ids.extend(parse_csv(args.include_ids))
    chosen_ids.extend(select_ids_from_tags(tests_by_id, parse_csv(args.include_tags)))

    if not chosen_ids:
        raise SystemExit("No tests selected. Provide --preset and/or --include-ids and/or --include-tags.")

    exclude = set(parse_csv(args.exclude_ids))
    unique_ids = []
    seen = set()
    for test_id in chosen_ids:
        if test_id in seen or test_id in exclude:
            continue
        if test_id not in tests_by_id:
            raise SystemExit(f"Unknown test id '{test_id}' in selection.")
        seen.add(test_id)
        unique_ids.append(test_id)

    selected = [tests_by_id[test_id] for test_id in unique_ids]
    suite_name = args.name.strip() or args.preset.strip() or "custom_suite"

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": suite_name,
        "generated_at": datetime.now().isoformat(),
        "catalog_version": catalog.get("version", ""),
        "source_catalog": str(catalog_path),
        "selection": {
            "preset": args.preset,
            "include_ids": parse_csv(args.include_ids),
            "include_tags": parse_csv(args.include_tags),
            "exclude_ids": parse_csv(args.exclude_ids)
        },
        "tests": selected
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Suite '{suite_name}' created with {len(selected)} test(s): {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
