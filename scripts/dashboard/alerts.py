"""Alert collection functions."""

from typing import Any

from .utils import format_duration_short, heartbeat_age_seconds, parse_iso_datetime


def collect_recent_batch_failure_alerts(
    failed_tasks: list[dict[str, Any]],
    window_seconds: int = 1800,
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Surface recent batch-fatal signals even after active batch cards clear."""
    best_by_batch: dict[str, tuple[int, dict[str, Any]]] = {}

    for task in failed_tasks:
        batch_id = str(task.get("batch_id") or "").strip()
        if not batch_id:
            continue

        when = (
            task.get("completed_at")
            or task.get("last_attempt_at")
            or task.get("started_at")
            or task.get("created_at")
        )
        age_s = heartbeat_age_seconds(str(when) if when else None)
        if age_s is None or age_s > window_seconds:
            continue

        status = str(task.get("status") or "").strip().lower()
        if status not in {"failed", "abandoned", "error", "blocked_cloud"}:
            continue

        result = task.get("result") if isinstance(task.get("result"), dict) else {}
        error_text = (
            result.get("error")
            or task.get("error")
            or result.get("output")
            or task.get("blocked_reason")
            or ""
        )
        error_text = " ".join(str(error_text).split())
        if not error_text:
            continue

        name = str(task.get("name") or task.get("task_id") or "task")
        error_type = str(result.get("error_type") or "").strip().lower()
        lowered = error_text.lower()

        score = 0
        if error_type == "brain_task_failure":
            score += 6
        if "batch aborted" in lowered:
            score += 2
        if any(x in lowered for x in ("can't open file", "query file not found", "fatal:")):
            score += 4
        if name in {"build_strategy", "identify_people", "execute_searches", "compile_output"}:
            score += 2

        snippet = error_text[:240]
        alert = {
            "severity": "bad",
            "worker": batch_id,
            "age_s": age_s,
            "message": f"Batch {batch_id} failed at {name}: {snippet}",
            "sticky": True,
            "sticky_id": f"batch-failure:{batch_id}",
        }

        prev = best_by_batch.get(batch_id)
        if prev is None or score > prev[0]:
            best_by_batch[batch_id] = (score, alert)

    alerts = [v[1] for v in best_by_batch.values()]
    alerts.sort(key=lambda a: (a.get("age_s") is None, a.get("age_s") or 10**9))
    return alerts[:limit]


def collect_recent_batch_completion_alerts(
    tasks_by_lane: dict[str, list[dict[str, Any]]],
    window_seconds: int = 1800,
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Surface recent completed batches with total runtime."""
    complete_tasks = tasks_by_lane.get("complete", [])
    if not complete_tasks:
        return []

    def count_by_batch(batch_id: str) -> dict[str, int]:
        return {
            "queue": sum(1 for t in tasks_by_lane.get("queue", []) if t.get("batch_id") == batch_id),
            "processing": sum(1 for t in tasks_by_lane.get("processing", []) if t.get("batch_id") == batch_id),
            "complete": sum(1 for t in tasks_by_lane.get("complete", []) if t.get("batch_id") == batch_id),
            "failed": sum(1 for t in tasks_by_lane.get("failed", []) if t.get("batch_id") == batch_id),
            "private": sum(1 for t in tasks_by_lane.get("private", []) if t.get("batch_id") == batch_id),
        }

    # Candidate batches are those with a recent completed task.
    candidate_batch_ids: set[str] = set()
    for task in complete_tasks:
        batch_id = str(task.get("batch_id") or "").strip()
        if not batch_id or batch_id.lower() == "system":
            continue
        when = task.get("completed_at") or task.get("started_at") or task.get("created_at")
        age_s = heartbeat_age_seconds(str(when) if when else None)
        if age_s is not None and age_s <= window_seconds:
            candidate_batch_ids.add(batch_id)

    alerts: list[dict[str, Any]] = []
    for batch_id in sorted(candidate_batch_ids):
        counts = count_by_batch(batch_id)
        # Must actually be done: no active queue/processing/private.
        if counts["queue"] > 0 or counts["processing"] > 0 or counts["private"] > 0:
            continue
        # Skip batches with failures so failure alert remains primary.
        if counts["failed"] > 0:
            continue
        if counts["complete"] <= 0:
            continue

        batch_tasks = [
            t for t in (tasks_by_lane.get("complete", []) + tasks_by_lane.get("failed", []))
            if str(t.get("batch_id") or "").strip() == batch_id
        ]
        if not batch_tasks:
            continue

        start_candidates: list = []
        end_candidates: list = []
        for task in batch_tasks:
            start_dt = parse_iso_datetime(task.get("created_at")) or parse_iso_datetime(task.get("started_at"))
            end_dt = parse_iso_datetime(task.get("completed_at")) or parse_iso_datetime(task.get("last_attempt_at"))
            if start_dt:
                start_candidates.append(start_dt)
            if end_dt:
                end_candidates.append(end_dt)

        if not end_candidates:
            continue
        start_dt = min(start_candidates) if start_candidates else None
        end_dt = max(end_candidates)
        runtime_s = int((end_dt - start_dt).total_seconds()) if start_dt else None
        age_s = heartbeat_age_seconds(end_dt.isoformat())

        alerts.append({
            "severity": "ok",
            "worker": batch_id,
            "age_s": age_s,
            "message": (
                f"Batch {batch_id} complete in {format_duration_short(runtime_s)} "
                f"({counts['complete']} tasks)"
            ),
            "sticky": True,
            "sticky_id": f"batch-complete:{batch_id}",
            "runtime_s": runtime_s,
        })

    alerts.sort(key=lambda a: (a.get("age_s") is None, a.get("age_s") or 10**9))
    return alerts[:limit]
