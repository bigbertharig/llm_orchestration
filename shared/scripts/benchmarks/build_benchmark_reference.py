#!/usr/bin/env python3
"""Build a markdown reference report from benchmark result records."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from compatibility import load_status


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                rows.append(data)
        except json.JSONDecodeError:
            continue
    return rows


def sort_key(entry: dict[str, Any]) -> str:
    return str(entry.get("run_at", ""))


def latest_by_model_test(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in sorted(rows, key=sort_key):
        model = str(row.get("model", "")).strip()
        test_id = str(row.get("test_id", "")).strip()
        if not model or not test_id:
            continue
        index[(model, test_id)] = row
    return sorted(index.values(), key=lambda r: (str(r.get("model", "")), str(r.get("test_id", ""))))


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def status_sort_key(entry: dict[str, Any], primary: str) -> tuple[str, str]:
    return (str(entry.get(primary, "")).strip(), str(entry.get("observed_at", "")).strip())


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate benchmark reference markdown from JSONL records.")
    ap.add_argument("--records", default="/mnt/shared/logs/benchmarks/model_benchmark_records.jsonl")
    ap.add_argument("--output", default="/mnt/shared/logs/benchmarks/MODEL_BENCHMARK_REFERENCE.md")
    ap.add_argument("--recent-limit", type=int, default=40)
    ap.add_argument(
        "--status-path",
        default=str(Path(__file__).resolve().parent / "benchmark_status.json"),
        help="Machine-readable benchmark backend/runtime status file.",
    )
    args = ap.parse_args()

    records_path = Path(args.records).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    status_path = Path(args.status_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(records_path)
    status = load_status(status_path)
    rows_sorted = sorted(rows, key=sort_key, reverse=True)
    latest_rows = latest_by_model_test(rows)

    recent_table_rows: list[list[str]] = []
    for row in rows_sorted[: max(1, args.recent_limit)]:
        recent_table_rows.append(
            [
                str(row.get("run_at", "")),
                str(row.get("model", "")),
                str(row.get("test_id", "")),
                str(row.get("score", "")),
                str(row.get("score_pct", "")),
                str(row.get("metric", "")),
                str(row.get("harness", "")),
                str(row.get("suite", "")),
                str(row.get("run_id", ""))
            ]
        )

    latest_table_rows: list[list[str]] = []
    for row in latest_rows:
        latest_table_rows.append(
            [
                str(row.get("model", "")),
                str(row.get("test_id", "")),
                str(row.get("score", "")),
                str(row.get("score_pct", "")),
                str(row.get("metric", "")),
                str(row.get("run_at", "")),
                str(row.get("harness", "")),
                str(row.get("suite", ""))
            ]
        )

    runtime_issue_rows: list[list[str]] = []
    for row in sorted(status.get("runtime_issues", []), key=lambda item: status_sort_key(item, "subject")):
        runtime_issue_rows.append(
            [
                str(row.get("subject", "")),
                str(row.get("state", "")),
                str(row.get("observed_at", "")),
                str(row.get("note", "")),
            ]
        )

    certified_test_rows: list[list[str]] = []
    for row in sorted(status.get("certified_tests", []), key=lambda item: status_sort_key(item, "backend_id")):
        certified_test_rows.append(
            [
                str(row.get("backend_id", "")),
                str(row.get("test_id", "")),
                str(row.get("state", "")),
                str(row.get("model_id", "")),
                str(row.get("observed_at", "")),
                str(row.get("note", "")),
            ]
        )

    runtime_note_rows: list[list[str]] = []
    for row in sorted(status.get("task_runtime_notes", []), key=lambda item: status_sort_key(item, "test_id")):
        runtime_note_rows.append(
            [
                str(row.get("test_id", "")),
                str(row.get("runtime_class", "")),
                str(row.get("observed_at", "")),
                str(row.get("note", "")),
            ]
        )

    generated_at = datetime.now().isoformat()
    content = [
        "# Model Benchmark Reference",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Records file: `{records_path}`",
        f"- Status file: `{status_path}`",
        f"- Total recorded runs: `{len(rows)}`",
        "",
        "## Operational Status",
        "",
        "This section tracks current benchmarkability and backend certification status.",
        "",
        "### Runtime Issues",
        "",
        md_table(
            ["Subject", "State", "Last Observed", "Notes"],
            runtime_issue_rows or [["-", "-", "-", "-"]]
        ),
        "",
        "### Backend/Test Certification",
        "",
        md_table(
            ["Backend", "Test ID", "State", "Probe Model", "Last Observed", "Notes"],
            certified_test_rows or [["-", "-", "-", "-", "-", "-"]]
        ),
        "",
        "### Task Runtime Notes",
        "",
        md_table(
            ["Test ID", "Runtime Class", "Last Observed", "Notes"],
            runtime_note_rows or [["-", "-", "-", "-"]]
        ),
        "",
        "## Latest Score Per Model/Test",
        "",
        md_table(
            ["Model", "Test ID", "Score", "Score %", "Metric", "Last Tested", "Harness", "Suite"],
            latest_table_rows or [["-", "-", "-", "-", "-", "-", "-", "-"]]
        ),
        "",
        "## Recent Runs",
        "",
        md_table(
            ["Run At", "Model", "Test ID", "Score", "Score %", "Metric", "Harness", "Suite", "Run ID"],
            recent_table_rows or [["-", "-", "-", "-", "-", "-", "-", "-", "-"]]
        ),
        ""
    ]
    output_path.write_text("\n".join(content), encoding="utf-8")
    print(f"Wrote reference markdown: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
