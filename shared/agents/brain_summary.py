"""Brain batch event-log and run-summary helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from run_summary import summarize_history_dir, write_run_summary


class BrainSummaryMixin:
    def _batch_history_dir(self, batch_id: str, batch_meta: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        meta = batch_meta or self.active_batches.get(batch_id, {}) or {}
        batch_dir = str(meta.get("batch_dir", "")).strip()
        if batch_dir:
            return Path(batch_dir).resolve()

        orchestration_batch_dir = str(meta.get("orchestration_batch_dir", "")).strip()
        if orchestration_batch_dir:
            return Path(orchestration_batch_dir).resolve()

        plan_dir = str(meta.get("plan_dir", "")).strip()
        if plan_dir:
            return (Path(plan_dir).resolve() / "history" / batch_id)
        return None

    def _batch_event_log_path(self, batch_id: str, batch_meta: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        history_dir = self._batch_history_dir(batch_id, batch_meta=batch_meta)
        if history_dir is None:
            return None
        return history_dir / "logs" / "batch_events.jsonl"

    def _event_signature(self, event: str, payload: Dict[str, Any]) -> Tuple[str, ...]:
        return (
            str(event),
            str(payload.get("task_id", "")),
            str(payload.get("task_name", "")),
            str(payload.get("recorded_status", "")),
            str(payload.get("started_at", "")),
            str(payload.get("completed_at", "")),
            str(payload.get("requeued_at", "")),
            str(payload.get("batch_status", "")),
            str(payload.get("reason", "")),
        )

    def _event_index_for_batch(self, batch_id: str) -> set[Tuple[str, ...]]:
        existing = self.batch_event_index.get(batch_id)
        if existing is not None:
            return existing

        index: set[Tuple[str, ...]] = set()
        log_path = self._batch_event_log_path(batch_id)
        if log_path and log_path.exists():
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        if isinstance(row, dict):
                            index.add(self._event_signature(str(row.get("event", "")), row))
            except Exception:
                pass

        self.batch_event_index[batch_id] = index
        return index

    def _append_batch_event(
        self,
        batch_id: str,
        event: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        dedupe: bool = True,
        batch_meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        payload = dict(payload or {})
        log_path = self._batch_event_log_path(batch_id, batch_meta=batch_meta)
        if log_path is None:
            return False

        log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "event": event,
            "batch_id": batch_id,
            "recorded_at": datetime.now().isoformat(),
            **payload,
        }
        signature = self._event_signature(event, entry)
        if dedupe and signature in self._event_index_for_batch(batch_id):
            return False

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        if dedupe:
            self._event_index_for_batch(batch_id).add(signature)
        return True

    def _task_payload(self, task: Dict[str, Any]) -> Dict[str, Any]:
        result = task.get("result", {}) if isinstance(task.get("result"), dict) else {}
        error_text = str(result.get("error", "") or "").strip()
        diagnostic_text = str(result.get("diagnostic", "") or "").strip()
        output_text = str(result.get("output", "") or "").strip()
        summary_text = error_text or diagnostic_text or output_text
        return {
            "task_id": task.get("task_id", ""),
            "task_name": task.get("name", ""),
            "task_class": task.get("task_class", ""),
            "executor": task.get("executor", ""),
            "worker": task.get("assigned_to") or task.get("worker") or "",
            "attempts": task.get("attempts", 0),
            "recorded_status": task.get("status", ""),
            "started_at": task.get("started_at", ""),
            "completed_at": task.get("completed_at", ""),
            "requeued_at": task.get("requeued_at", ""),
            "success": bool(result.get("success", False)),
            "error_type": result.get("error_type", ""),
            "error": error_text[:400],
            "diagnostic": diagnostic_text[:400],
            "summary": summary_text[:400],
            "error_code": str(result.get("error_code", "") or "")[:120],
            "runtime_error_code": str(result.get("runtime_error_code", "") or "")[:120],
            "runtime_error_detail": str(result.get("runtime_error_detail", "") or "")[:400],
        }

    def _iter_batch_tasks(self, batch_id: str, roots: Iterable[Path]) -> Iterable[Dict[str, Any]]:
        for root in roots:
            if not root.exists():
                continue
            for task_file in root.glob("*.json"):
                try:
                    with open(task_file, "r", encoding="utf-8") as f:
                        task = json.load(f)
                except Exception:
                    continue
                if task.get("batch_id") == batch_id:
                    yield task

    def _current_batch_counts(self, batch_id: str) -> Dict[str, int]:
        counts = {"queue": 0, "private": 0, "processing": 0, "complete": 0, "failed": 0}
        roots = {
            "queue": self.queue_path,
            "private": self.private_tasks_path,
            "processing": self.processing_path,
            "complete": self.complete_path,
            "failed": self.failed_path,
        }
        for label, root in roots.items():
            for task in self._iter_batch_tasks(batch_id, [root]):
                if task.get("task_id"):
                    counts[label] += 1
        return counts

    def _refresh_batch_summary(
        self,
        batch_id: str,
        *,
        status: Optional[str] = None,
        failure_reason: Optional[str] = None,
        batch_meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        history_dir = self._batch_history_dir(batch_id, batch_meta=batch_meta)
        if history_dir is None:
            return False

        summary = summarize_history_dir(
            history_dir,
            status=status,
            live_counts=self._current_batch_counts(batch_id),
            failure_reason=failure_reason,
        )
        write_run_summary(history_dir, summary)
        self._append_batch_event(
            batch_id,
            "summary_refreshed",
            {
                "summary_status": summary.get("status", ""),
                "json_path": "RUN_SUMMARY.json",
                "markdown_path": "RUN_SUMMARY.md",
            },
            dedupe=False,
            batch_meta=batch_meta,
        )
        return True

    def _record_batch_interrupted(
        self,
        batch_id: str,
        *,
        reason: str,
        batch_meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        wrote = self._append_batch_event(
            batch_id,
            "batch_interrupted",
            {
                "batch_status": "interrupted",
                "reason": reason,
            },
            dedupe=False,
            batch_meta=batch_meta,
        )
        refreshed = self._refresh_batch_summary(
            batch_id,
            status="partial",
            failure_reason=reason,
            batch_meta=batch_meta,
        )
        return wrote or refreshed

    def _record_resume_handoff(
        self,
        batch_id: str,
        *,
        requested_batch_id: str,
        batch_meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        wrote = self._append_batch_event(
            batch_id,
            "resume_handoff",
            {
                "batch_status": "resumed",
                "requested_batch_id": requested_batch_id,
            },
            dedupe=False,
            batch_meta=batch_meta,
        )
        refreshed = self._refresh_batch_summary(batch_id, batch_meta=batch_meta)
        return wrote or refreshed

    def _reconcile_batch_history(self, batch_id: str, batch_meta: Optional[Dict[str, Any]] = None) -> bool:
        roots = (
            self.queue_path,
            self.processing_path,
            self.complete_path,
            self.failed_path,
        )
        terminal_change = False
        for task in self._iter_batch_tasks(batch_id, roots):
            payload = self._task_payload(task)
            status = str(task.get("status", "")).strip().lower()
            if status == "pending":
                self._append_batch_event(batch_id, "task_released", payload, batch_meta=batch_meta)
            elif status == "processing":
                self._append_batch_event(batch_id, "task_started", payload, batch_meta=batch_meta)
            elif status == "complete" and payload.get("success"):
                if self._append_batch_event(batch_id, "task_succeeded", payload, batch_meta=batch_meta):
                    terminal_change = True
            elif status in {"failed", "blocked_cloud"}:
                if self._append_batch_event(batch_id, "task_failed", payload, batch_meta=batch_meta):
                    terminal_change = True
            elif status == "abandoned":
                if self._append_batch_event(batch_id, "task_abandoned", payload, batch_meta=batch_meta):
                    terminal_change = True

        if terminal_change:
            self._refresh_batch_summary(batch_id, batch_meta=batch_meta)
        return terminal_change
