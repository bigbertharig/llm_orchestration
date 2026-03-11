#!/usr/bin/env python3
"""Tests for which meta tasks block additional resource decisions."""

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
if "requests" not in sys.modules:
    sys.modules["requests"] = types.SimpleNamespace()

from brain_resources import BrainResourceMixin


def _write_task(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class MockBrainResources(BrainResourceMixin):
    def __init__(self, root: Path):
        self.queue_path = root / "tasks" / "queue"
        self.processing_path = root / "tasks" / "processing"
        self.failed_path = root / "tasks" / "failed"
        self.private_tasks_path = root / "brain" / "private_tasks"
        self.signals_path = root / "signals"
        self.name = "brain"
        self.logger = types.SimpleNamespace(debug=lambda *a, **k: None, warning=lambda *a, **k: None)
        self.gpu_agents = {"gpu-1": {}, "gpu-2": {}, "gpu-3": {}}
        self.gpu_miss_count = {}
        self.missing_gpu_miss_threshold = 3
        self.load_llm_requests = {}
        self.active_batches = {"batch-1": {}}
        self.resource_task_cooldown_seconds = 0
        self.last_resource_task_at = {}
        self.max_hot_workers = 2
        self.min_hot_workers = 0
        self.single_unload_idle_seconds = 60
        self.split_unload_idle_seconds = 60
        self.auto_default_enabled = False
        self.default_llm_min_tier = 1
        self.model_tier_by_id = {"qwen2.5:7b": 1, "qwen2.5-coder:14b": 2}
        self.model_meta_by_id = {
            "qwen2.5:7b": {"placement": "single_gpu", "tier": 1},
            "qwen2.5-coder:14b": {
                "placement": "split_gpu",
                "tier": 2,
                "split_groups": [
                    {"id": "pair_1_3", "members": ["gpu-1", "gpu-3"], "port": 11440},
                ],
            },
        }
        self.logged = []
        self.inserted = []
        self._gpu_states = {}
        self._demand_window = {"total_llm": 0, "split_llm": 0, "min_tier": 1}
        for path in (
            self.queue_path,
            self.processing_path,
            self.failed_path,
            self.private_tasks_path,
            self.signals_path,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def log_decision(self, event, message, details):
        self.logged.append({"event": event, "message": message, "details": details})

    def _insert_resource_task(self, command, meta=None):
        self.inserted.append({"command": command, "meta": meta or {}})

    def _process_recovery_fallback_signals(self):
        return None

    def _check_thermal_recovery_escalation(self, gpu_states):
        return None

    def _monitor_split_health_issues(self, gpu_states):
        return None

    def _monitor_global_load_owner_issues(self, gpu_states):
        return None

    def _handle_missing_gpu_escalations(self, missing, queue_stats):
        return None

    def _collect_llm_demand_window_snapshot(self):
        return dict(self._demand_window)

    def _update_llm_demand_timers(self, demand_window):
        return {"any_llm_idle_s": 0.0, "split_llm_idle_s": 0.0}

    def _check_auto_default_policy(self, **kwargs):
        return {"managed": False, "triggered": False}

    def _get_gpu_states(self):
        return dict(self._gpu_states)

    def _reconcile_split_group_state(self, group, split_model):
        return {
            "group_id": str(group.get("id", "")).strip(),
            "classification": "cold",
        }


class BrainResourceMetaBlockingTests(unittest.TestCase):
    def setUp(self):
        self.gpu_status = [
            {"gpu_id": 1, "util_pct": 0, "mem_used_mb": 0, "power_w": 10},
            {"gpu_id": 2, "util_pct": 0, "mem_used_mb": 0, "power_w": 10},
            {"gpu_id": 3, "util_pct": 0, "mem_used_mb": 0, "power_w": 10},
        ]
        self.queue_stats = {
            "total_pending": 10,
            "cpu": 0,
            "script": 0,
            "llm": 10,
            "llm_split_required": 0,
            "llm_max_tier": 1,
            "llm_model_demand": {"qwen2.5:7b": 10},
            "llm_split_model_demand": {},
            "meta": 0,
            "brain_tasks": 0,
            "worker_tasks": 10,
            "stuck_tasks": 0,
            "thermal_wait_tasks": 0,
            "processing_count": 1,
        }

    def test_unload_meta_does_not_block_load_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain = MockBrainResources(Path(tmp))
            brain._gpu_states = {
                "gpu-1": {"model_loaded": False, "runtime_healthy": True},
                "gpu-2": {"model_loaded": False, "runtime_healthy": True},
                "gpu-3": {"model_loaded": False, "runtime_healthy": True},
            }
            brain._demand_window = {"total_llm": 10, "split_llm": 0, "min_tier": 1}
            _write_task(
                brain.processing_path / "unload.json",
                {"task_id": "unload", "task_class": "meta", "command": "unload_split_llm"},
            )

            brain._make_resource_decisions(self.gpu_status, brain.gpu_agents, self.queue_stats)

            self.assertEqual(brain.inserted, [{"command": "load_llm", "meta": {}}])

    def test_load_meta_blocks_additional_load_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain = MockBrainResources(Path(tmp))
            brain._gpu_states = {
                "gpu-1": {"model_loaded": False, "runtime_healthy": True},
                "gpu-2": {"model_loaded": False, "runtime_healthy": True},
                "gpu-3": {"model_loaded": False, "runtime_healthy": True},
            }
            brain._demand_window = {"total_llm": 10, "split_llm": 1, "min_tier": 1}
            _write_task(
                brain.processing_path / "load.json",
                {"task_id": "load", "task_class": "meta", "command": "load_split_llm"},
            )

            brain._make_resource_decisions(self.gpu_status, brain.gpu_agents, self.queue_stats)

            self.assertEqual(brain.inserted, [])

    def test_pending_split_load_does_not_block_targeted_unloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain = MockBrainResources(Path(tmp))
            brain._gpu_states = {
                "gpu-1": {
                    "model_loaded": True,
                    "runtime_healthy": True,
                    "runtime_placement": "single_gpu",
                    "loaded_model": "qwen2.5:7b",
                    "loaded_tier": 1,
                },
                "gpu-2": {"model_loaded": False, "runtime_healthy": True},
                "gpu-3": {
                    "model_loaded": True,
                    "runtime_healthy": True,
                    "runtime_placement": "single_gpu",
                    "loaded_model": "qwen2.5:7b",
                    "loaded_tier": 1,
                },
            }
            brain._demand_window = {"total_llm": 1, "split_llm": 1, "min_tier": 2}
            split_queue_stats = dict(self.queue_stats)
            split_queue_stats.update(
                {
                    "llm": 1,
                    "llm_split_required": 1,
                    "llm_max_tier": 2,
                    "llm_model_demand": {},
                    "llm_split_model_demand": {"qwen2.5-coder:14b": 1},
                }
            )
            _write_task(
                brain.queue_path / "load_split_retry.json",
                {
                    "task_id": "load-split",
                    "task_class": "meta",
                    "command": "load_split_llm",
                    "target_model": "qwen2.5-coder:14b",
                    "candidate_groups": [
                        {"id": "pair_1_3", "members": ["gpu-1", "gpu-3"], "port": 11440}
                    ],
                },
            )

            brain._make_resource_decisions(self.gpu_status, brain.gpu_agents, split_queue_stats)

            self.assertEqual(
                brain.inserted,
                [
                    {"command": "unload_llm", "meta": {"candidate_workers": ["gpu-1"]}},
                    {"command": "unload_llm", "meta": {"candidate_workers": ["gpu-3"]}},
                ],
            )


if __name__ == "__main__":
    unittest.main()
