#!/usr/bin/env python3
"""Tests for centralized monitor-driven task requeue paths."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from brain_failures import BrainFailureMixin
from brain_monitor import BrainMonitorMixin
from brain_summary import BrainSummaryMixin
from brain_tasks import BrainTaskQueueMixin


class MockBrainMonitor(
    BrainMonitorMixin,
    BrainFailureMixin,
    BrainSummaryMixin,
    BrainTaskQueueMixin,
):
    def __init__(self, root: Path):
        self.queue_path = root / "tasks" / "queue"
        self.processing_path = root / "tasks" / "processing"
        self.complete_path = root / "tasks" / "complete"
        self.failed_path = root / "tasks" / "failed"
        self.private_tasks_path = root / "brain" / "private_tasks"
        self.signals_path = root / "signals"
        self.heartbeat_stale_seconds = 10
        self.force_kill_requeue_seconds = 5
        self.incidents = {}
        self.active_batches = {
            "batch-1": {
                "batch_dir": str(root / "plans" / "demo" / "history" / "batch-1"),
                "plan": "demo",
                "plan_dir": str(root / "plans" / "demo"),
            }
        }
        self.batch_event_index = {}
        self.logger = MagicMock()
        self.logged = []

        for path in (
            self.queue_path,
            self.processing_path,
            self.complete_path,
            self.failed_path,
            self.private_tasks_path,
            self.signals_path,
            root / "plans" / "demo" / "history" / "batch-1",
        ):
            path.mkdir(parents=True, exist_ok=True)

    def log_decision(self, event, message, details):
        self.logged.append({"event": event, "message": message, "details": details})

    def _get_or_create_incident(self, task, result):
        incident_id = str(task.get("incident_id") or "incident-1")
        incident = self.incidents.get(incident_id)
        if incident is None:
            incident = {
                "incident_id": incident_id,
                "history": [],
            }
            self.incidents[incident_id] = incident
        return incident


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class BrainMonitorRequeueTests(unittest.TestCase):
    def test_recover_orphaned_processing_task_uses_shared_requeue_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            brain = MockBrainMonitor(root)
            started_at = (datetime.now() - timedelta(seconds=30)).isoformat()
            task = {
                "task_id": "task-orphan",
                "batch_id": "batch-1",
                "name": "analyze_repo",
                "status": "processing",
                "assigned_to": "gpu-1",
                "started_at": started_at,
            }
            task_file = brain.processing_path / "task-orphan.json"
            _write_json(task_file, task)

            recovered = brain._recover_orphaned_processing_tasks(running_workers={})

            self.assertEqual(recovered, 1)
            queued = json.loads((brain.queue_path / "task-orphan.json").read_text(encoding="utf-8"))
            self.assertEqual(queued["status"], "pending")
            self.assertEqual(queued["requeue_reason"], "orphan_recovered")
            self.assertEqual(queued["orphan_recovered_from"], "gpu-1")
            events_path = (
                Path(brain.active_batches["batch-1"]["batch_dir"]) / "logs" / "batch_events.jsonl"
            )
            events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
            event_names = [row["event"] for row in events]
            self.assertIn("task_retried", event_names)
            self.assertIn("task_released", event_names)

    def test_stuck_force_kill_timeout_uses_shared_requeue_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            brain = MockBrainMonitor(root)
            task_id = "task-stuck"
            processing_task = {
                "task_id": task_id,
                "batch_id": "batch-1",
                "name": "report",
                "status": "processing",
                "assigned_to": "gpu-2",
                "started_at": (datetime.now() - timedelta(minutes=20)).isoformat(),
                "incident_id": "incident-stuck",
            }
            task_file = brain.processing_path / f"{task_id}.json"
            _write_json(task_file, processing_task)
            incident = brain._get_or_create_incident(processing_task, {"success": False})
            incident["stuck_abort_sent_at"] = (datetime.now() - timedelta(seconds=20)).isoformat()
            incident["stuck_force_kill_sent_at"] = (datetime.now() - timedelta(seconds=10)).isoformat()
            incident["stuck_abort_count"] = 2

            stuck_info = {
                "task": processing_task,
                "task_file": task_file,
                "task_id": task_id[:8],
                "assigned_to": "gpu-2",
                "elapsed_min": 20,
                "progress_age_sec": 1200,
                "threshold_sec": 600,
                "name": "report",
            }

            brain._handle_stuck_tasks([stuck_info])

            queued = json.loads((brain.queue_path / f"{task_id}.json").read_text(encoding="utf-8"))
            self.assertEqual(queued["status"], "pending")
            self.assertEqual(queued["requeue_reason"], "force_kill_timeout")
            self.assertEqual(queued["stuck_requeue_count"], 1)
            events_path = (
                Path(brain.active_batches["batch-1"]["batch_dir"]) / "logs" / "batch_events.jsonl"
            )
            events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
            event_names = [row["event"] for row in events]
            self.assertIn("task_retried", event_names)
            self.assertIn("task_released", event_names)


if __name__ == "__main__":
    unittest.main()
