#!/usr/bin/env python3
"""Summarize benchmark catalog audit coverage from status and known runners."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from compatibility import load_status


LM_EVAL_TEMPLATED = {
    "gsm8k",
    "drop",
    "bbh",
    "gpqa_diamond",
    "math_500",
    "aime_2024",
    "musr",
}

LM_EVAL_COMPLETIONS = {
    "mmlu",
    "arc_challenge",
    "hellaswag",
    "piqa",
    "winogrande",
    "truthfulqa_mc2",
    "boolq",
    "mmmlu",
}

IMPLEMENTED_LOCAL_CUSTOM = {
    "custom_json_schema_strict",
    "custom_command_safety",
    "custom_ambiguity_handling",
}


def load_catalog(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [item for item in data.get("tests", []) if isinstance(item, dict)]


def classify_certified_entry(entry: dict[str, Any]) -> tuple[str, str]:
    state = str(entry.get("state", "")).strip() or "unknown"
    note = str(entry.get("note", "")).strip()
    if "not available in this environment" in note:
        return "missing_task", note
    if state == "env_blocked":
        return "env_blocked", note
    if state == "blocked":
        return "blocked", note
    if state == "supported":
        return "supported", note
    return state or "unknown", note


def find_certified(status: dict[str, Any], backend_id: str, test_id: str) -> dict[str, Any] | None:
    for row in status.get("certified_tests", []):
        if (
            str(row.get("backend_id", "")).strip() == backend_id
            and str(row.get("test_id", "")).strip() == test_id
        ):
            return row
    return None


def audit_test(status: dict[str, Any], test: dict[str, Any]) -> dict[str, Any]:
    test_id = str(test.get("id", "")).strip()
    harness = str(test.get("harness", "")).strip()
    result = {
        "test_id": test_id,
        "harness": harness,
        "category": str(test.get("category", "")).strip(),
        "audit_state": "not_audited",
        "execution_lane": "",
        "notes": "",
    }

    if harness == "lm_eval":
        if test_id in LM_EVAL_TEMPLATED:
            lane = "ollama_chat_completions_templated"
        elif test_id in LM_EVAL_COMPLETIONS:
            lane = "ollama_completions"
        else:
            lane = ""
        result["execution_lane"] = lane
        if not lane:
            result["audit_state"] = "unknown_profile"
            result["notes"] = "lm_eval task exists in catalog but no execution profile is mapped yet."
            return result
        certified = find_certified(status, lane, test_id)
        if not certified:
            result["audit_state"] = "not_audited"
            result["notes"] = "Mapped lane exists, but no certification result has been recorded yet."
            return result
        audit_state, notes = classify_certified_entry(certified)
        result["audit_state"] = audit_state
        result["notes"] = notes
        return result

    if harness == "local_custom":
        result["execution_lane"] = "local_custom"
        if test_id in IMPLEMENTED_LOCAL_CUSTOM:
            result["audit_state"] = "implemented"
            result["notes"] = "Runnable via run_local_custom_task.py"
        else:
            result["audit_state"] = "missing_runner"
            result["notes"] = "Catalog entry exists, but no local_custom implementation is present yet."
        return result

    result["execution_lane"] = harness
    result["audit_state"] = "missing_runner"
    result["notes"] = f"Harness '{harness}' is defined in the catalog, but no audited runner path is wired yet."
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit benchmark catalog coverage and runner status.")
    ap.add_argument("--catalog", default="benchmark_catalog.json")
    ap.add_argument("--status-path", default=str(Path(__file__).resolve().parent / "benchmark_status.json"))
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    this_dir = Path(__file__).resolve().parent
    catalog_path = (this_dir / args.catalog).resolve() if not Path(args.catalog).is_absolute() else Path(args.catalog)
    status_path = Path(args.status_path).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else this_dir / "benchmark_audit_summary.json"
    )

    status = load_status(status_path)
    tests = load_catalog(catalog_path)
    audited = [audit_test(status, test) for test in tests]
    counts = Counter(row["audit_state"] for row in audited)
    harness_counts: dict[str, dict[str, int]] = {}
    for row in audited:
        harness = row["harness"]
        harness_counts.setdefault(harness, {})
        harness_counts[harness][row["audit_state"]] = harness_counts[harness].get(row["audit_state"], 0) + 1

    payload = {
        "generated_at": datetime.now().isoformat(),
        "catalog_path": str(catalog_path),
        "status_path": str(status_path),
        "counts": dict(sorted(counts.items())),
        "counts_by_harness": harness_counts,
        "tests": audited,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["counts"], indent=2))
    print(f"Wrote audit summary: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
