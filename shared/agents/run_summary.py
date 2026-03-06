"""History-folder run summary reducer.

Builds best-effort summaries from a plan history run directory regardless of
whether the run completed successfully, failed early, or only produced partial
artifacts.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _load_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    if not path.exists():
        return rows
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except Exception:
        return rows
    return rows


def _count_files(paths: Iterable[Path], suffixes: tuple[str, ...] = (".json", ".md", ".jsonl")) -> int:
    total = 0
    for root in paths:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in suffixes:
                total += 1
    return total


def _snippet_from_text(text: str, max_chars: int = 300) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def _load_text(path: Path, max_chars: int = 300) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return _snippet_from_text(f.read(), max_chars=max_chars)
    except Exception:
        return "unreadable_text"


def _relative_path(base_dir: Path, path: Path) -> str:
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return str(path)


def _artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".md":
        return "markdown"
    if suffix == ".log":
        return "log"
    return suffix.lstrip(".") or "file"


def _compact_json_excerpt(path: Path, payload: Dict[str, Any]) -> str:
    if path.name == "batch_failure.json":
        parts = []
        if payload.get("reason"):
            parts.append(f"reason={_snippet_from_text(payload['reason'], max_chars=180)}")
        if payload.get("source_task"):
            parts.append(f"source_task={payload['source_task']}")
        if payload.get("source_task_id"):
            parts.append(f"source_task_id={payload['source_task_id']}")
        if payload.get("abandoned_tasks") is not None:
            parts.append(f"abandoned_tasks={payload['abandoned_tasks']}")
        return "; ".join(parts) or "batch failure payload present"

    if path.name == "execution_stats.json":
        parts = []
        overall = payload.get("overall")
        if overall:
            parts.append(f"overall={_snippet_from_text(json.dumps(overall), max_chars=180)}")
        outcome = payload.get("outcome")
        if outcome:
            parts.append(f"outcome={_snippet_from_text(json.dumps(outcome), max_chars=180)}")
        return "; ".join(parts) or "execution stats present"

    if path.name in {"runtime_validation.json", "runtime_probe_classification.json"}:
        return _snippet_from_text(json.dumps(payload), max_chars=240)

    if path.name in {
        "brain_strategy.json",
        "analysis_manifest.json",
        "env_manifest.json",
        "repo_context.json",
        "repo_dependency_risk.json",
        "repo_env.json",
        "repo_structure_summary.json",
        "json_data_summary.json",
        "claim_evidence_links.json",
        "slice_complexity.json",
        "final_report.json",
    }:
        return _snippet_from_text(json.dumps(payload), max_chars=240)

    return _snippet_from_text(json.dumps(payload), max_chars=240)


def _summarize_artifact(base_dir: Path, path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return None
    entry: Dict[str, Any] = {
        "path": _relative_path(base_dir, path),
        "name": path.name,
        "suffix": path.suffix,
        "kind": _artifact_kind(path),
    }
    if path.suffix == ".json":
        payload = _load_json(path)
        if payload:
            entry["keys"] = sorted(payload.keys())[:12]
            entry["excerpt"] = _compact_json_excerpt(path, payload)
        else:
            entry["excerpt"] = "unparseable_or_non_dict_json"
        return entry
    if path.suffix == ".md":
        entry["excerpt"] = _load_text(path, max_chars=300)
        return entry
    if path.suffix == ".jsonl":
        rows = _load_jsonl(path)
        entry["line_count"] = len(rows)
        if rows:
            entry["first_event"] = rows[0].get("event")
            entry["last_event"] = rows[-1].get("event")
            event_counts = Counter()
            for row in rows:
                event_counts[str(row.get("event", "unknown")).strip() or "unknown"] += 1
            entry["event_counts"] = dict(event_counts.most_common(6))
        entry["excerpt"] = (
            f"events={entry['line_count']}; first={entry.get('first_event')}; last={entry.get('last_event')}"
        )
        return entry
    if path.suffix == ".log":
        entry["excerpt"] = _load_text(path, max_chars=300)
        return entry
    return None


def summarize_history_dir(
    history_dir: Path,
    *,
    status: Optional[str] = None,
    live_counts: Optional[Dict[str, int]] = None,
    failure_reason: Optional[str] = None,
    refreshed_at: Optional[str] = None,
) -> Dict[str, Any]:
    history_dir = history_dir.resolve()
    batch_id = history_dir.name
    batch_meta = _load_json(history_dir / "batch_meta.json")
    batch_failure = _load_json(history_dir / "results" / "batch_failure.json")
    events = _load_jsonl(history_dir / "logs" / "batch_events.jsonl")
    execution_stats = _load_json(history_dir / "execution_stats.json")

    event_counts = Counter()
    for event in events:
        event_counts[str(event.get("event", "unknown")).strip() or "unknown"] += 1

    artifact_counts = {
        "results_files": _count_files([history_dir / "results"], suffixes=(".json", ".md")),
        "output_files": _count_files([history_dir / "output"], suffixes=(".json", ".md")),
        "log_files": _count_files([history_dir / "logs"], suffixes=(".jsonl", ".log", ".json")),
    }

    inferred_status = status or ""
    if not inferred_status:
        if batch_failure:
            inferred_status = "failed"
        elif execution_stats or (history_dir / "EXECUTION_SUMMARY.md").exists():
            inferred_status = "complete"
        elif event_counts.get("batch_completed", 0) or event_counts.get("batch_complete", 0):
            inferred_status = "complete"
        elif batch_meta or artifact_counts.get("results_files", 0) > 0 or artifact_counts.get("output_files", 0) > 0:
            inferred_status = "partial"
        elif events:
            inferred_status = "running_or_partial"
        else:
            inferred_status = "unknown"

    summary = {
        "batch_id": batch_id,
        "plan_name": batch_meta.get("plan_name") or batch_meta.get("plan") or history_dir.parent.parent.name,
        "history_dir": str(history_dir),
        "status": inferred_status,
        "refreshed_at": refreshed_at or datetime.now().isoformat(),
        "started_at": batch_meta.get("started_at"),
        "event_counts": dict(event_counts),
        "live_counts": dict(live_counts or {}),
        "artifact_counts": artifact_counts,
        "artifact_presence": {
            "batch_meta_json": bool(batch_meta),
            "batch_failure_json": bool(batch_failure),
            "batch_events_jsonl": (history_dir / "logs" / "batch_events.jsonl").exists(),
            "execution_stats_json": bool(execution_stats),
            "execution_summary_md": (history_dir / "EXECUTION_SUMMARY.md").exists(),
        },
        "failure": batch_failure or ({"reason": failure_reason} if failure_reason else None),
    }

    important_artifact_candidates = [
        history_dir / "results" / "batch_failure.json",
        history_dir / "execution_stats.json",
        history_dir / "EXECUTION_SUMMARY.md",
        history_dir / "logs" / "batch_events.jsonl",
        history_dir / "analysis_manifest.json",
        history_dir / "env_manifest.json",
        history_dir / "runtime_validation.json",
        history_dir / "brain_strategy.json",
        history_dir / "repo_context.json",
        history_dir / "repo_dependency_risk.json",
        history_dir / "repo_env.json",
        history_dir / "output" / "final_report.json",
        history_dir / "output" / "final_report.md",
        history_dir / "output" / "runtime_probe_classification.json",
        history_dir / "output" / "repo_structure_summary.json",
        history_dir / "output" / "json_data_summary.json",
        history_dir / "output" / "claim_evidence_links.json",
        history_dir / "output" / "slice_complexity.json",
    ]
    important_artifacts = []
    for candidate in important_artifact_candidates:
        summarized = _summarize_artifact(history_dir, candidate)
        if summarized:
            important_artifacts.append(summarized)
    summary["important_artifacts"] = important_artifacts

    if execution_stats:
        summary["execution_overall"] = execution_stats.get("overall")
        summary["execution_outcome"] = execution_stats.get("outcome")

    return summary


def render_run_summary_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        f"# Run Summary: {summary.get('plan_name') or summary.get('batch_id')}",
        "",
        f"- Batch ID: {summary.get('batch_id')}",
        f"- Status: {summary.get('status')}",
        f"- Refreshed At: {summary.get('refreshed_at')}",
        f"- History Dir: {summary.get('history_dir')}",
        "",
        "## Artifacts",
        "",
    ]

    for key, value in sorted((summary.get("artifact_presence") or {}).items()):
        lines.append(f"- {key}: {value}")

    lines.extend([
        "",
        "## Artifact Counts",
        "",
    ])
    for key, value in sorted((summary.get("artifact_counts") or {}).items()):
        lines.append(f"- {key}: {value}")

    if summary.get("live_counts"):
        lines.extend(["", "## Live Counts", ""])
        for key, value in sorted(summary["live_counts"].items()):
            lines.append(f"- {key}: {value}")

    if summary.get("event_counts"):
        lines.extend(["", "## Event Counts", ""])
        for key, value in sorted(summary["event_counts"].items()):
            lines.append(f"- {key}: {value}")

    important_artifacts = summary.get("important_artifacts") or []
    if important_artifacts:
        lines.extend(["", "## Important Artifacts", ""])
        for artifact in important_artifacts:
            lines.append(f"- {artifact.get('path')} [{artifact.get('kind')}]")
            if artifact.get("excerpt"):
                lines.append(f"- excerpt: {artifact.get('excerpt')}")
            if artifact.get("keys"):
                lines.append(f"- keys: {', '.join(artifact.get('keys', []))}")
            if artifact.get("line_count") is not None:
                lines.append(
                    f"- events: {artifact.get('line_count')} "
                    f"(first={artifact.get('first_event')}, last={artifact.get('last_event')})"
                )
            if artifact.get("event_counts"):
                lines.append(
                    "- event_counts: "
                    + ", ".join(f"{key}={value}" for key, value in artifact.get("event_counts", {}).items())
                )

    failure = summary.get("failure") or {}
    if failure:
        lines.extend([
            "",
            "## Failure",
            "",
            f"- Reason: {failure.get('reason', '')}",
            f"- Source Task: {failure.get('source_task', '')}",
            f"- Source Task ID: {failure.get('source_task_id', '')}",
            f"- Abandoned Tasks: {failure.get('abandoned_tasks', 0)}",
        ])

    return "\n".join(lines) + "\n"


def write_run_summary(history_dir: Path, summary: Dict[str, Any]) -> Dict[str, str]:
    history_dir = history_dir.resolve()
    history_dir.mkdir(parents=True, exist_ok=True)
    summary_json = history_dir / "RUN_SUMMARY.json"
    summary_md = history_dir / "RUN_SUMMARY.md"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write(render_run_summary_markdown(summary))
    return {"json": str(summary_json), "markdown": str(summary_md)}
