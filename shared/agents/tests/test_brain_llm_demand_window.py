#!/usr/bin/env python3
"""Tests for brain LLM demand accounting."""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if "filelock" not in sys.modules:
    sys.modules["filelock"] = types.SimpleNamespace(FileLock=object, Timeout=Exception)

from brain_resources import BrainResourceMixin


class MockBrainDemand(BrainResourceMixin):
    def __init__(self, root: Path):
        self.queue_path = root / "tasks" / "queue"
        self.processing_path = root / "tasks" / "processing"
        self.private_tasks_path = root / "brain" / "private_tasks"
        self.active_batches = {}
        self.default_llm_min_tier = 1
        self.model_tier_by_id = {}
        for path in (self.queue_path, self.processing_path, self.private_tasks_path):
            path.mkdir(parents=True, exist_ok=True)

    def _iter_task_files_json(self, path: Path):
        for task_file in sorted(path.glob("*.json")):
            yield json.loads(task_file.read_text(encoding="utf-8"))


def _write_task(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


class BrainLlmDemandWindowTests(unittest.TestCase):
    def test_ignores_orphaned_private_tasks_from_inactive_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            brain = MockBrainDemand(root)

            _write_task(
                brain.private_tasks_path / "orphan.json",
                {
                    "task_id": "orphan",
                    "batch_id": "old-batch",
                    "task_class": "llm",
                    "llm_model": "qwen2.5-coder:14b",
                    "llm_placement": "split_gpu",
                },
            )

            demand = brain._collect_llm_demand_window_snapshot()

            self.assertEqual(demand["private_llm"], 0)
            self.assertEqual(demand["split_llm"], 0)
            self.assertEqual(demand["total_llm"], 0)

    def test_counts_private_tasks_for_active_batches_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            brain = MockBrainDemand(root)
            brain.active_batches = {"active-batch": {"plan": "demo"}}

            _write_task(
                brain.private_tasks_path / "active.json",
                {
                    "task_id": "active",
                    "batch_id": "active-batch",
                    "task_class": "llm",
                    "llm_model": "qwen2.5:7b",
                    "llm_placement": "single_gpu",
                },
            )
            _write_task(
                brain.private_tasks_path / "orphan.json",
                {
                    "task_id": "orphan",
                    "batch_id": "old-batch",
                    "task_class": "llm",
                    "llm_model": "qwen2.5-coder:14b",
                    "llm_placement": "split_gpu",
                },
            )

            demand = brain._collect_llm_demand_window_snapshot()

            self.assertEqual(demand["private_llm"], 1)
            self.assertEqual(demand["split_llm"], 0)
            self.assertEqual(demand["total_llm"], 1)


if __name__ == "__main__":
    unittest.main()
