#!/usr/bin/env python3
"""Tests for brain-issued resource task batch attribution."""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if "filelock" not in sys.modules:
    sys.modules["filelock"] = types.SimpleNamespace(FileLock=object)

from brain_resources import BrainResourceMixin


class MockBrainResources(BrainResourceMixin):
    def __init__(self, root: Path):
        self.queue_path = root / "tasks" / "queue"
        self.processing_path = root / "tasks" / "processing"
        self.failed_path = root / "tasks" / "failed"
        self.name = "brain"
        self.logger = type("L", (), {"debug": lambda *a, **k: None})()
        self.last_resource_task_at = {}
        self.resource_task_cooldown_seconds = 0
        self.load_llm_requests = {}
        self.active_batches = {}
        self.logged = []
        for path in (self.queue_path, self.processing_path, self.failed_path):
            path.mkdir(parents=True, exist_ok=True)

    def log_decision(self, event, message, details):
        self.logged.append((event, message, details))

    def save_to_public(self, task):
        out = self.queue_path / f"{task['task_id']}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(task, f, indent=2)
        return True

    def _get_gpu_states(self):
        return {}


class ResourceTaskBatchIdTests(unittest.TestCase):
    def test_single_active_batch_is_used_for_resource_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain = MockBrainResources(Path(tmp))
            brain.active_batches = {
                "batch-1": {"started_at": datetime.now().isoformat()}
            }

            brain._insert_resource_task("load_llm")

            files = list(brain.queue_path.glob("*.json"))
            self.assertEqual(len(files), 1)
            task = json.loads(files[0].read_text(encoding="utf-8"))
            self.assertEqual(task["batch_id"], "batch-1")
            self.assertEqual(task["source_batch_id"], "batch-1")

    def test_explicit_source_batch_id_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain = MockBrainResources(Path(tmp))
            brain.active_batches = {
                "batch-1": {},
                "batch-2": {},
            }

            brain._insert_resource_task(
                "cleanup_split_runtime",
                meta={"group_id": "pair_4_5", "source_batch_id": "batch-9"},
            )

            files = list(brain.queue_path.glob("*.json"))
            task = json.loads(files[0].read_text(encoding="utf-8"))
            self.assertEqual(task["batch_id"], "batch-9")
            self.assertEqual(task["source_batch_id"], "batch-9")
            self.assertEqual(task["group_id"], "pair_4_5")

    def test_multiple_active_batches_without_source_stays_system(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain = MockBrainResources(Path(tmp))
            brain.active_batches = {
                "batch-1": {},
                "batch-2": {},
            }

            brain._insert_resource_task("unload_llm", meta={"candidate_workers": ["gpu-2"]})

            files = list(brain.queue_path.glob("*.json"))
            task = json.loads(files[0].read_text(encoding="utf-8"))
            self.assertEqual(task["batch_id"], "system")
            self.assertNotIn("source_batch_id", task)


if __name__ == "__main__":
    unittest.main()
