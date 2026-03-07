#!/usr/bin/env python3
"""Tests for shared meta-task contract validation."""

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
        self.logged = []
        self.events = []
        self.summaries = []
        for path in (
            self.queue_path,
            self.processing_path,
            self.complete_path,
            self.failed_path,
            self.private_tasks_path,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def log_decision(self, event, message, details):
        self.logged.append({"event": event, "message": message, "details": details})

    def _append_batch_event(self, batch_id, event, payload):
        self.events.append({"batch_id": batch_id, "event": event, "payload": payload})
        return True

    def _task_payload(self, task):
        return {"task_id": task.get("task_id"), "task_name": task.get("name")}

    def _refresh_batch_summary(self, batch_id):
        self.summaries.append(batch_id)


class MetaTaskContractTests(unittest.TestCase):
    def test_rejects_unsupported_meta_command_before_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain = MockBrainTasks(Path(tmp))
            task = {
                "task_id": "meta-1",
                "batch_id": "batch-1",
                "name": "bad-meta",
                "task_class": "meta",
                "command": "force_unload_split_llm",
            }

            queued = brain.save_to_public(task)

            self.assertFalse(queued)
            self.assertFalse((brain.queue_path / "meta-1.json").exists())
            failed = json.loads((brain.failed_path / "meta-1.json").read_text(encoding="utf-8"))
            self.assertEqual(failed["result"]["error"], "unsupported_meta_command:force_unload_split_llm")
            self.assertEqual(failed["result"]["error_type"], "meta_task_contract_error")
            self.assertIn("META_TASK_INVALID", [entry["event"] for entry in brain.logged])

    def test_rejects_load_split_llm_with_string_candidate_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain = MockBrainTasks(Path(tmp))
            task = {
                "task_id": "meta-2",
                "batch_id": "batch-1",
                "name": "bad-split-load",
                "task_class": "meta",
                "command": "load_split_llm",
                "target_model": "qwen2.5-coder:14b",
                "candidate_groups": ["pair_4_5"],
            }

            queued = brain.save_to_public(task)

            self.assertFalse(queued)
            failed = json.loads((brain.failed_path / "meta-2.json").read_text(encoding="utf-8"))
            self.assertEqual(failed["result"]["error"], "load_split_llm:invalid_candidate_group")

    def test_rejects_reset_split_runtime_without_target_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain = MockBrainTasks(Path(tmp))
            task = {
                "task_id": "meta-3",
                "batch_id": "batch-1",
                "name": "bad-split-reset",
                "task_class": "meta",
                "command": "reset_split_runtime",
            }

            queued = brain.save_to_public(task)

            self.assertFalse(queued)
            failed = json.loads((brain.failed_path / "meta-3.json").read_text(encoding="utf-8"))
            self.assertEqual(failed["result"]["error"], "reset_split_runtime:missing_target_worker")

    def test_accepts_valid_load_split_llm_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain = MockBrainTasks(Path(tmp))
            task = {
                "task_id": "meta-4",
                "batch_id": "batch-1",
                "name": "good-split-load",
                "task_class": "meta",
                "command": "load_split_llm",
                "target_model": "qwen2.5-coder:14b",
                "candidate_groups": [
                    {
                        "id": "pair_4_5",
                        "members": ["gpu-4", "gpu-5"],
                        "port": 11441,
                    }
                ],
            }

            queued = brain.save_to_public(task)

            self.assertTrue(queued)
            self.assertTrue((brain.queue_path / "meta-4.json").exists())
            self.assertFalse((brain.failed_path / "meta-4.json").exists())


if __name__ == "__main__":
    unittest.main()
