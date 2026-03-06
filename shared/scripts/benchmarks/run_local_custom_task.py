#!/usr/bin/env python3
"""Run local custom benchmark tasks against an Ollama endpoint."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def default_output_dir() -> str:
    media_path = Path("/media/bryan/shared/logs/benchmarks")
    if media_path.exists():
        return str(media_path)
    return "/mnt/shared/logs/benchmarks"


def load_catalog(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for item in data.get("tests", []):
        test_id = str(item.get("id", "")).strip()
        if test_id:
            out[test_id] = item
    return out


def load_cases(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for item in data.get("tests", []):
        test_id = str(item.get("id", "")).strip()
        if test_id:
            out[test_id] = item
    return out


def call_generate(base_url: str, model: str, prompt: str, timeout: int) -> str:
    response = requests.post(
        f"{base_url.rstrip('/')}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    text = str(data.get("response", ""))
    return text.strip()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).lower()


def parse_strict_json(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```") or stripped.endswith("```"):
        return None
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def grade_json_schema(case: dict[str, Any], response: str) -> tuple[bool, str]:
    parsed = parse_strict_json(response)
    if parsed is None:
        return False, "response is not strict raw JSON object"
    schema = case.get("schema", {})
    for key, expected_type in schema.items():
        if key not in parsed:
            return False, f"missing key {key}"
        value = parsed[key]
        if expected_type == "string" and not isinstance(value, str):
            return False, f"key {key} is not string"
        if expected_type == "integer" and not isinstance(value, int):
            return False, f"key {key} is not integer"
        if expected_type == "boolean" and not isinstance(value, bool):
            return False, f"key {key} is not boolean"
    expected = case.get("expected", {})
    action_contains = str(expected.get("action_contains", "")).strip().lower()
    if action_contains and action_contains not in str(parsed.get("action", "")).lower():
        return False, "action content mismatch"
    if "priority" in expected and parsed.get("priority") != expected.get("priority"):
        return False, "priority mismatch"
    if "safe" in expected and parsed.get("safe") is not expected.get("safe"):
        return False, "safe flag mismatch"
    return True, "strict schema case passed"


def grade_keyword_case(case: dict[str, Any], response: str) -> tuple[bool, str]:
    normalized = normalize_text(response)
    required = [str(x).strip().lower() for x in case.get("required_keywords", []) if str(x).strip()]
    required_any = [
        [str(option).strip().lower() for option in group if str(option).strip()]
        for group in case.get("required_any_keywords", [])
        if isinstance(group, list)
    ]
    forbidden = [str(x).strip().lower() for x in case.get("forbidden_keywords", []) if str(x).strip()]
    missing = [word for word in required if word not in normalized]
    if missing:
        return False, f"missing required keywords: {', '.join(missing)}"
    missing_any: list[str] = []
    for group in required_any:
        if group and not any(option in normalized for option in group):
            missing_any.append(" / ".join(group))
    if missing_any:
        return False, f"missing required keyword groups: {', '.join(missing_any)}"
    triggered = [word for word in forbidden if word in normalized]
    if triggered:
        return False, f"forbidden keywords present: {', '.join(triggered)}"
    return True, "keyword case passed"


def grade_case(test_id: str, case: dict[str, Any], response: str) -> tuple[bool, str]:
    if test_id == "custom_json_schema_strict":
        return grade_json_schema(case, response)
    return grade_keyword_case(case, response)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a local custom benchmark task")
    ap.add_argument("--id", required=True, help="Custom test id from benchmark catalog")
    ap.add_argument("--model", required=True, help="Model tag on target endpoint")
    ap.add_argument("--base-url", required=True, help="Endpoint base URL like http://localhost:11435")
    ap.add_argument("--catalog", default="benchmark_catalog.json")
    ap.add_argument("--cases", default="custom_tasks/cases.json")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--output-dir", default=default_output_dir())
    ap.add_argument("--suite", default="individual_custom")
    ap.add_argument("--no-record", action="store_true")
    args = ap.parse_args()

    this_dir = Path(__file__).resolve().parent
    catalog_path = (this_dir / args.catalog).resolve() if not Path(args.catalog).is_absolute() else Path(args.catalog)
    cases_path = (this_dir / args.cases).resolve() if not Path(args.cases).is_absolute() else Path(args.cases)

    catalog = load_catalog(catalog_path)
    case_map = load_cases(cases_path)

    selected = catalog.get(args.id)
    if selected is None:
        raise SystemExit(f"Unknown test id '{args.id}'")
    if str(selected.get("harness", "")).strip() != "local_custom":
        raise SystemExit(f"Test '{args.id}' is not a local_custom test")
    test_cases = case_map.get(args.id, {}).get("cases", [])
    if not test_cases:
        raise SystemExit(f"No local custom cases defined for '{args.id}'")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / f"{args.id}_{now_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    passes = 0
    for case in test_cases:
        prompt = str(case.get("prompt", "")).strip()
        if not prompt:
            continue
        response = call_generate(args.base_url, args.model, prompt, args.timeout)
        passed, detail = grade_case(args.id, case, response)
        if passed:
            passes += 1
        results.append(
            {
                "name": str(case.get("name", "")),
                "passed": passed,
                "detail": detail,
                "response": response,
            }
        )

    total = len(results)
    score = passes / total if total else 0.0
    payload = {
        "run_at": datetime.now().isoformat(),
        "test_id": args.id,
        "model": args.model,
        "base_url": args.base_url,
        "score": score,
        "metric": str(case_map.get(args.id, {}).get("metric", "pass_rate")),
        "cases": results,
    }
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if not args.no_record:
        recorder = this_dir / "record_benchmark_result.py"
        cmd = [
            sys.executable,
            str(recorder),
            "--model",
            args.model,
            "--test-id",
            args.id,
            "--score",
            str(score),
            "--metric",
            payload["metric"],
            "--harness",
            "local_custom",
            "--suite",
            args.suite,
            "--notes",
            f"{passes}/{total} cases passed",
        ]
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)

    print(json.dumps({"result_path": str(result_path), "score": score, "passes": passes, "total": total}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
