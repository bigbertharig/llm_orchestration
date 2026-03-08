#!/usr/bin/env python3
"""Regression coverage for batch-scoped task completion lookup."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from brain_tasks import BrainTaskQueueMixin


class MockBrainTasks(BrainTaskQueueMixin):
    def __init__(self, root: Path):
        self.queue_path = root / "tasks" / "queue"
        self.processing_path = root / "tasks" / "processing"
        self.complete_path = root / "tasks" / "complete"
        self.failed_path = root / "tasks" / "failed"
        self.private_tasks_path = root / "brain" / "private_tasks"
        self.active_batches = {
            "batch-1": {
                "batch_dir": str(root / "history" / "batch-1"),
            }
        }
        for path in (
            self.queue_path,
            self.processing_path,
            self.complete_path,
            self.failed_path,
            self.private_tasks_path,
            root / "history" / "batch-1" / "logs",
        ):
            path.mkdir(parents=True, exist_ok=True)


class BrainTaskCompletionIndexTests(unittest.TestCase):
    def test_prefers_batch_event_log_over_global_complete_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            brain = MockBrainTasks(root)
            events_log = root / "history" / "batch-1" / "logs" / "batch_events.jsonl"
            events_log.write_text(
                "\n".join(
                    [
                        json.dumps({"event": "task_released", "batch_id": "batch-1", "task_name": "prepare_repo"}),
                        json.dumps({"event": "task_succeeded", "batch_id": "batch-1", "task_name": "prepare_repo"}),
                        json.dumps({"event": "task_succeeded", "batch_id": "batch-1", "task_name": "warm_workers"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            # Leave unrelated global completion artifacts in place; they should not
            # be needed when batch-local history is available.
            (brain.complete_path / "old-task.json").write_text(
                json.dumps({"batch_id": "other-batch", "name": "stale", "result": {"success": True}}),
                encoding="utf-8",
            )

            completed = brain.get_completed_task_ids("batch-1")

            self.assertEqual(completed, {"prepare_repo", "warm_workers"})

    def test_falls_back_to_global_complete_scan_when_batch_log_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            brain = MockBrainTasks(root)
            (brain.complete_path / "prepare_repo.json").write_text(
                json.dumps({"batch_id": "batch-1", "name": "prepare_repo", "result": {"success": True}}),
                encoding="utf-8",
            )
            (brain.complete_path / "failed.json").write_text(
                json.dumps({"batch_id": "batch-1", "name": "warm_workers", "result": {"success": False}}),
                encoding="utf-8",
            )

            completed = brain.get_completed_task_ids("batch-1")

            self.assertEqual(completed, {"prepare_repo"})


if __name__ == "__main__":
    unittest.main()
