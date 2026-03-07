#!/usr/bin/env python3
"""Tests for interruption and resume-handoff summary lifecycle hooks."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from brain_summary import BrainSummaryMixin


class MockBrainSummaryLifecycle(BrainSummaryMixin):
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
        for path in (
            self.queue_path,
            self.processing_path,
            self.complete_path,
            self.failed_path,
            self.private_tasks_path,
            root / "plans" / "demo" / "history" / "batch-1",
        ):
            path.mkdir(parents=True, exist_ok=True)


class BrainSummaryLifecycleTests(unittest.TestCase):
    def test_record_batch_interrupted_writes_event_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain = MockBrainSummaryLifecycle(Path(tmp))
            history_dir = Path(brain.active_batches["batch-1"]["batch_dir"])

            changed = brain._record_batch_interrupted("batch-1", reason="brain_shutdown")

            self.assertTrue(changed)
            rows = [
                json.loads(line)
                for line in (history_dir / "logs" / "batch_events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertIn("batch_interrupted", [row["event"] for row in rows])
            summary = json.loads((history_dir / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "partial")

    def test_record_resume_handoff_writes_event_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain = MockBrainSummaryLifecycle(Path(tmp))
            history_dir = Path(brain.active_batches["batch-1"]["batch_dir"])

            changed = brain._record_resume_handoff("batch-1", requested_batch_id="resume-batch")

            self.assertTrue(changed)
            rows = [
                json.loads(line)
                for line in (history_dir / "logs" / "batch_events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            events = [row["event"] for row in rows]
            self.assertIn("resume_handoff", events)
            self.assertTrue((history_dir / "RUN_SUMMARY.json").exists())


if __name__ == "__main__":
    unittest.main()
