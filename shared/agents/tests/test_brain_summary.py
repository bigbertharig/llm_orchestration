#!/usr/bin/env python3
"""Tests for brain-owned batch event logging and summary refresh."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from brain_summary import BrainSummaryMixin


class MockBrainSummary(BrainSummaryMixin):
    def __init__(self, root: Path):
        self.queue_path = root / "tasks" / "queue"
        self.processing_path = root / "tasks" / "processing"
        self.complete_path = root / "tasks" / "complete"
        self.failed_path = root / "tasks" / "failed"
        self.private_tasks_path = root / "brain" / "private_tasks"
        self.active_batches = {
            "batch-1": {
                "batch_dir": str(root / "plans" / "demo" / "history" / "batch-1"),
                "plan": "demo",
                "plan_dir": str(root / "plans" / "demo"),
            }
        }
        self.batch_event_index = {}
        for path in (
            self.queue_path,
            self.processing_path,
            self.complete_path,
            self.failed_path,
            self.private_tasks_path,
            root / "plans" / "demo" / "history" / "batch-1",
        ):
            path.mkdir(parents=True, exist_ok=True)


def _write_task(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class BrainSummaryTests(unittest.TestCase):
    def test_reconcile_records_task_events_and_refreshes_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            brain = MockBrainSummary(root)
            history_dir = Path(brain.active_batches["batch-1"]["batch_dir"])
            _write_task(
                brain.queue_path / "released.json",
                {"task_id": "released", "batch_id": "batch-1", "name": "collect", "status": "pending"},
            )
            _write_task(
                brain.processing_path / "running.json",
                {
                    "task_id": "running",
                    "batch_id": "batch-1",
                    "name": "analyze",
                    "status": "processing",
                    "started_at": "2026-03-06T10:00:00",
                    "assigned_to": "gpu-1",
                },
            )
            _write_task(
                brain.complete_path / "done.json",
                {
                    "task_id": "done",
                    "batch_id": "batch-1",
                    "name": "report",
                    "status": "complete",
                    "completed_at": "2026-03-06T10:01:00",
                    "result": {"success": True},
                },
            )

            changed = brain._reconcile_batch_history("batch-1")

            self.assertTrue(changed)
            log_path = history_dir / "logs" / "batch_events.jsonl"
            rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            events = [row["event"] for row in rows]
            self.assertIn("task_released", events)
            self.assertIn("task_started", events)
            self.assertIn("task_succeeded", events)
            self.assertIn("summary_refreshed", events)
            self.assertTrue((history_dir / "RUN_SUMMARY.json").exists())
            self.assertTrue((history_dir / "RUN_SUMMARY.md").exists())

    def test_append_batch_event_dedupes_repeated_task_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            brain = MockBrainSummary(root)

            first = brain._append_batch_event(
                "batch-1",
                "task_started",
                {"task_id": "task-1", "task_name": "analyze", "started_at": "2026-03-06T10:00:00"},
            )
            second = brain._append_batch_event(
                "batch-1",
                "task_started",
                {"task_id": "task-1", "task_name": "analyze", "started_at": "2026-03-06T10:00:00"},
            )

            self.assertTrue(first)
            self.assertFalse(second)

    def test_task_payload_preserves_diagnostic_summary_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            brain = MockBrainSummary(root)

            payload = brain._task_payload(
                {
                    "task_id": "task-1",
                    "name": "load_llm",
                    "status": "failed",
                    "result": {
                        "success": False,
                        "error": "",
                        "diagnostic": "target_model=qwen runtime_error_code=load_failed",
                        "output": "Model failed to load",
                        "error_code": "load_failed",
                        "runtime_error_code": "load_exception",
                        "runtime_error_detail": "model readiness probe timed out",
                    },
                }
            )

            self.assertEqual(payload["error"], "")
            self.assertEqual(payload["diagnostic"], "target_model=qwen runtime_error_code=load_failed")
            self.assertEqual(payload["summary"], "target_model=qwen runtime_error_code=load_failed")
            self.assertEqual(payload["error_code"], "load_failed")
            self.assertEqual(payload["runtime_error_code"], "load_exception")
