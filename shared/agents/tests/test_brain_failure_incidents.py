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
        self.config = {"retry_policy": {"max_attempts": 3}}
        self.max_brain_fix_attempts = 0
        self.model_meta_by_id = {}
        self.gpu_agents = {}
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
        self.aborted_batches = []
        self.escalations = []
        self.saved_state = 0

    def log_decision(self, event, message, details):
        self.logged.append({"event": event, "message": message, "details": details})

    def _get_or_create_incident(self, task, result):
        incident_id = str(task.get("incident_id") or "incident-1")
        return {
            "incident_id": incident_id,
            "history": [],
            "brain_fix_attempts": 0,
            "worker_cycles": 0,
        }

    def _abort_batch(self, batch_id, reason, task):
        self.aborted_batches.append({"batch_id": batch_id, "reason": reason, "task": task.get("name")})

    def emit_cloud_escalation(self, escalation_type, title, details, source_task):
        self.escalations.append(
            {
                "type": escalation_type,
                "title": title,
                "details": details,
                "task": source_task.get("name"),
            }
        )
        return "esc-1"

    def _save_brain_state(self):
        self.saved_state += 1

    def _try_json_repair_escalation(self, task, result, task_file):
        return False

    def _try_fix_missing_module(self, task, result):
        return False

    def _is_recoverable_llm_timeout(self, task, result):
        return False

    def _is_missing_scraped_file(self, result):
        return False

    def _try_fix_permission_denied(self, task, result):
        return False


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

    def test_nonfatal_resource_meta_exhaustion_does_not_abort_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            brain = MockBrainFailures(root)
            task = {
                "task_id": "task-meta",
                "batch_id": "batch-1",
                "name": "load_llm",
                "command": "load_llm",
                "task_class": "meta",
                "executor": "worker",
                "status": "failed",
                "attempts": 3,
                "workers_attempted": ["gpu-5", "gpu-5", "gpu-5"],
                "assigned_to": "gpu-5",
                "result": {
                    "success": False,
                    "error": "load timeout",
                    "error_type": "meta_runtime_failure",
                },
            }
            task_file = brain.failed_path / "task-meta.json"
            _write_json(task_file, task)

            brain.handle_failed_tasks()

            updated = json.loads(task_file.read_text(encoding="utf-8"))
            self.assertEqual(updated["status"], "abandoned")
            self.assertEqual(updated["abandoned_reason"], "resource_meta_exhausted_nonfatal")
            self.assertEqual(updated["result"]["error_type"], "resource_meta_nonfatal")
            self.assertEqual(brain.aborted_batches, [])
            self.assertEqual(brain.escalations, [])
            self.assertTrue(any(x["event"] == "RESOURCE_META_ABANDON" for x in brain.logged))

    def test_deferred_model_load_requeues_without_abort_or_escalation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            brain = MockBrainFailures(root)
            task = {
                "task_id": "task-deferred",
                "batch_id": "batch-1",
                "name": "worker_review_slice_001",
                "task_class": "llm",
                "executor": "worker",
                "llm_model": "qwen2.5:7b",
                "status": "failed",
                "attempts": 3,
                "workers_attempted": ["gpu-2", "gpu-5"],
                "assigned_to": "gpu-5",
                "result": {
                    "success": False,
                    "output": "DEFERRED_MODEL_LOAD: model_unavailable: model 'qwen2.5:7b' is not available; queued_load_llm=True",
                    "reason": "deferred_model_load",
                    "worker": "gpu-5",
                },
            }
            task_file = brain.failed_path / "task-deferred.json"
            _write_json(task_file, task)

            brain.handle_failed_tasks()

            self.assertFalse(task_file.exists())
            queued = json.loads((brain.queue_path / "task-deferred.json").read_text(encoding="utf-8"))
            self.assertEqual(queued["status"], "pending")
            self.assertEqual(queued["requeue_reason"], "deferred_model_load")
            self.assertEqual(queued["attempts"], 0)
            self.assertEqual(queued["workers_attempted"], [])
            self.assertEqual(brain.aborted_batches, [])
            self.assertEqual(brain.escalations, [])
            self.assertTrue(any(x["event"] == "DEFERRED_MODEL_LOAD" for x in brain.logged))


if __name__ == "__main__":
    unittest.main()
