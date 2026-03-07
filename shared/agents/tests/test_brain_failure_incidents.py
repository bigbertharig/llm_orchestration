#!/usr/bin/env python3
"""Tests for incident_id policy across brain requeue paths."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from brain_failures import BrainFailureMixin
from brain_summary import BrainSummaryMixin
from brain_tasks import BrainTaskQueueMixin


class MockBrainFailures(BrainFailureMixin, BrainSummaryMixin, BrainTaskQueueMixin):
    def __init__(self, root: Path):
        self.queue_path = root / "tasks" / "queue"
        self.processing_path = root / "tasks" / "processing"
        self.complete_path = root / "tasks" / "complete"
        self.failed_path = root / "tasks" / "failed"
        self.private_tasks_path = root / "brain" / "private_tasks"
        self.batch_event_index = {}
        self.active_batches = {
            "batch-1": {
                "batch_dir": str(root / "plans" / "demo" / "history" / "batch-1"),
                "plan": "demo",
                "plan_dir": str(root / "plans" / "demo"),
            }
        }
        self.logger = MagicMock()
        self.logged = []
        for path in (
            self.queue_path,
            self.processing_path,
            self.complete_path,
            self.failed_path,
            self.private_tasks_path,
            root / "plans" / "demo" / "history" / "batch-1",
        ):
            path.mkdir(parents=True, exist_ok=True)

    def log_decision(self, event, message, details):
        self.logged.append({"event": event, "message": message, "details": details})


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class BrainFailureIncidentTests(unittest.TestCase):
    def test_recoverable_retry_preserves_incident_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            brain = MockBrainFailures(root)
            task = {
                "task_id": "task-1",
                "batch_id": "batch-1",
                "name": "analyze_repo",
                "status": "failed",
                "incident_id": "inc-existing",
                "attempts": 2,
                "workers_attempted": ["gpu-1"],
            }
            task_file = brain.failed_path / "task-1.json"
            _write_json(task_file, task)

            brain._queue_task_retry(task_file, task, "recoverable_permission_denied")

            queued = json.loads((brain.queue_path / "task-1.json").read_text(encoding="utf-8"))
            self.assertEqual(queued["incident_id"], "inc-existing")
            self.assertEqual(queued["requeue_reason"], "recoverable_permission_denied")
            self.assertEqual(queued["attempts"], 0)
            self.assertEqual(queued["workers_attempted"], [])

    def test_definition_fix_drops_incident_id(self):
        brain = MockBrainFailures(Path(tempfile.mkdtemp()))
        task = {
            "task_id": "task-2",
            "batch_id": "batch-1",
            "name": "prepare_repo",
            "status": "failed",
            "incident_id": "inc-old",
            "definition_error": "missing task_class",
            "command": "python scripts/prepare.py",
        }

        fixed = brain._try_fix_definition_error(task)

        self.assertTrue(fixed)
        self.assertEqual(task["status"], "pending")
        self.assertNotIn("incident_id", task)
        self.assertEqual(task["requeue_reason"], "definition_fix")


if __name__ == "__main__":
    unittest.main()
