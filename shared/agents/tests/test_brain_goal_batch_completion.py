#!/usr/bin/env python3
"""Regression tests for goal-driven batch completion."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from brain_goal import BrainGoalMixin


class StubBrainGoal(BrainGoalMixin):
    def __init__(self, root: Path):
        self.active_batches = {
            "batch-1": {
                "plan": "github_analyzer",
                "goal": {
                    "accepted": 2,
                    "target": 2,
                    "rejected": 0,
                    "status": "complete",
                },
            }
        }
        self.batch_event_index = {"batch-1": {"events": []}}
        self.queue_path = root / "queue"
        self.processing_path = root / "processing"
        self.private_tasks_path = root / "private"
        self.queue_path.mkdir(parents=True, exist_ok=True)
        self.processing_path.mkdir(parents=True, exist_ok=True)
        self.private_tasks_path.mkdir(parents=True, exist_ok=True)
        self.logger = SimpleNamespace(warning=lambda *args, **kwargs: None)
        self.decisions = []
        self.events = []
        self.summary_refreshes = []
        self.saved = False

    def get_private_tasks(self, batch_id: str):
        return []

    def log_decision(self, event_type: str, message: str, details: dict):
        self.decisions.append((event_type, message, details))

    def _append_batch_event(self, batch_id: str, event_type: str, payload: dict, batch_meta=None):
        self.events.append((batch_id, event_type, payload, batch_meta))

    def _refresh_batch_summary(self, batch_id: str, status: str, batch_meta=None):
        self.summary_refreshes.append((batch_id, status, batch_meta))

    def _save_brain_state(self):
        self.saved = True


class BrainGoalBatchCompletionTests(unittest.TestCase):
    def test_check_batch_completion_marks_batch_complete_without_nameerror(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stub = StubBrainGoal(Path(tmpdir))

            stub._check_batch_completion("batch-1")

        self.assertEqual(stub.decisions[0][0], "BATCH_COMPLETE")
        self.assertEqual(stub.events[0][0], "batch-1")
        self.assertEqual(stub.events[0][1], "batch_completed")
        self.assertEqual(stub.events[0][2]["batch_status"], "complete")
        self.assertTrue(stub.events[0][2]["completed_at"])
        self.assertEqual(stub.summary_refreshes[0][1], "complete")
        self.assertTrue(stub.saved)
        self.assertNotIn("batch-1", stub.active_batches)
        self.assertNotIn("batch-1", stub.batch_event_index)


if __name__ == "__main__":
    unittest.main()
