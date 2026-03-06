#!/usr/bin/env python3
"""
Targeted tests for centralized runtime authority behavior.

Run:
  python -m pytest shared/agents/tests/test_centralized_runtime_authority.py -v
"""

import json
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

if "filelock" not in sys.modules:
    sys.modules["filelock"] = types.SimpleNamespace(FileLock=object, Timeout=Exception)
if "requests" not in sys.modules:
    sys.modules["requests"] = types.SimpleNamespace()

from brain_resources import BrainResourceMixin
from gpu_tasks import GPUTaskMixin


class MockBrain(BrainResourceMixin):
    def __init__(self, tmpdir: Path):
        self.model_meta_by_id = {
            "model-a": {
                "split_groups": [
                    {"id": "pair_1_2", "members": ["gpu-1", "gpu-2"], "port": 11434}
                ]
            }
        }
        self.signals_path = tmpdir / "signals"
        self.signals_path.mkdir(parents=True, exist_ok=True)
        self.queue_path = tmpdir / "queue"
        self.queue_path.mkdir(parents=True, exist_ok=True)
        self.processing_path = tmpdir / "processing"
        self.processing_path.mkdir(parents=True, exist_ok=True)
        self.name = "brain"
        self.resource_task_cooldown_seconds = 0
        self.last_resource_task_at = {}
        self.load_llm_requests = {}
        self.logged = []
        self.inserted = []
        self.logger = MagicMock()

    def log_decision(self, event, message, details):
        self.logged.append({"event": event, "message": message, "details": details})

    def _insert_resource_task(self, command, meta=None):
        self.inserted.append({"command": command, "meta": meta or {}})


class MockGpuTasks(GPUTaskMixin):
    def __init__(self):
        self.name = "gpu-1"
        self.runtime_group_id = "pair_1_2"
        self.runtime_port = 11434
        self.split_runtime_generation = "gen-current"
        self._reservation_epoch = "epoch-current"
        self.cleanup_calls = []
        self.logger = MagicMock()

    def _read_split_reservation_epoch(self, group_id: str):
        return self._reservation_epoch

    def _touch_meta_task(self, phase="", force=False):
        return None

    def _coordinated_split_failure_cleanup(self, group_id, split_port, reason, task_id=None):
        payload = {
            "group_id": group_id,
            "split_port": split_port,
            "reason": reason,
            "task_id": task_id,
        }
        self.cleanup_calls.append(payload)
        return payload


class CentralizedRuntimeAuthorityTests(unittest.TestCase):
    def test_brain_queues_cleanup_on_critical_issue(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain = MockBrain(Path(tmp))
            gpu_states = {
                "gpu-1": {
                    "split_runtime_generation": "gen-a",
                    "runtime_state": "ready_split",
                    "split_health_issue": {
                        "has_issue": True,
                        "severity": "critical",
                        "issue_code": "invariant_owner_meta_mismatch",
                        "awaiting_brain_decision": True,
                        "runtime_generation": "gen-a",
                        "reservation_epoch": "epoch-a",
                        "split_port": 11434,
                        "group_id": "pair_1_2",
                    },
                }
            }

            brain._monitor_split_health_issues(gpu_states)

            self.assertEqual(len(brain.inserted), 1)
            inserted = brain.inserted[0]
            self.assertEqual(inserted["command"], "cleanup_split_runtime")
            self.assertEqual(inserted["meta"]["group_id"], "pair_1_2")
            self.assertEqual(inserted["meta"]["target_workers"], ["gpu-1", "gpu-2"])
            self.assertEqual(inserted["meta"]["runtime_generation"], "gen-a")
            self.assertEqual(inserted["meta"]["reservation_epoch"], "epoch-a")

    def test_brain_does_not_queue_cleanup_on_single_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain = MockBrain(Path(tmp))
            gpu_states = {
                "gpu-1": {
                    "split_runtime_generation": "gen-a",
                    "runtime_state": "ready_split",
                    "split_health_issue": {
                        "has_issue": True,
                        "severity": "warning",
                        "issue_code": "invariant_reservation_missing",
                        "awaiting_brain_decision": True,
                        "runtime_generation": "gen-a",
                        "reservation_epoch": "epoch-a",
                        "split_port": 11434,
                        "group_id": "pair_1_2",
                    },
                }
            }

            brain._monitor_split_health_issues(gpu_states)

            self.assertEqual(brain.inserted, [])

    def test_cleanup_command_rejects_stale_reservation_epoch(self):
        gpu = MockGpuTasks()
        task = {
            "task_id": "task-1",
            "group_id": "pair_1_2",
            "cleanup_reason": "brain_decision",
            "reservation_epoch": "epoch-old",
            "runtime_generation": "gen-current",
            "split_port": 11434,
        }

        result = gpu._execute_meta_command(task, "cleanup_split_runtime")

        self.assertTrue(result["success"])
        self.assertTrue(result["stale_command"])
        self.assertEqual(gpu.cleanup_calls, [])
        self.assertIn("expected_reservation_epoch=epoch-old", result["output"])

    def test_cleanup_command_rejects_stale_runtime_generation(self):
        gpu = MockGpuTasks()
        task = {
            "task_id": "task-1",
            "group_id": "pair_1_2",
            "cleanup_reason": "brain_decision",
            "reservation_epoch": "epoch-current",
            "runtime_generation": "gen-old",
            "split_port": 11434,
        }

        result = gpu._execute_meta_command(task, "cleanup_split_runtime")

        self.assertTrue(result["success"])
        self.assertTrue(result["stale_command"])
        self.assertEqual(gpu.cleanup_calls, [])
        self.assertIn("expected_generation=gen-old", result["output"])

    def test_cleanup_command_executes_when_fences_match(self):
        gpu = MockGpuTasks()
        task = {
            "task_id": "task-1",
            "group_id": "pair_1_2",
            "cleanup_reason": "brain_decision",
            "reservation_epoch": "epoch-current",
            "runtime_generation": "gen-current",
            "split_port": 11434,
        }

        result = gpu._execute_meta_command(task, "cleanup_split_runtime")

        self.assertTrue(result["success"])
        self.assertEqual(len(gpu.cleanup_calls), 1)
        self.assertEqual(gpu.cleanup_calls[0]["reason"], "brain_command:brain_decision")

    def test_brain_reclaims_matching_stale_global_load_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain = MockBrain(Path(tmp))
            owner_path = brain.signals_path / "model_load.global.json"
            owner = {
                "worker": "gpu-1",
                "pid": 999999,
                "lease_id": "lease-a",
                "heartbeat_at": (datetime.now() - timedelta(minutes=5)).isoformat(),
            }
            owner_path.write_text(json.dumps(owner), encoding="utf-8")

            brain._brain_reclaim_stale_global_load_owner(
                {"owner_worker": "gpu-1", "owner_lease_id": "lease-a"}
            )

            self.assertFalse(owner_path.exists())
            self.assertTrue(any(x["event"] == "GLOBAL_LOAD_OWNER_RECLAIMED" for x in brain.logged))

    def test_brain_does_not_reclaim_when_lease_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain = MockBrain(Path(tmp))
            owner_path = brain.signals_path / "model_load.global.json"
            owner = {
                "worker": "gpu-1",
                "pid": 999999,
                "lease_id": "lease-b",
                "heartbeat_at": (datetime.now() - timedelta(minutes=5)).isoformat(),
            }
            owner_path.write_text(json.dumps(owner), encoding="utf-8")

            brain._brain_reclaim_stale_global_load_owner(
                {"owner_worker": "gpu-1", "owner_lease_id": "lease-a"}
            )

            self.assertTrue(owner_path.exists())
            self.assertFalse(any(x["event"] == "GLOBAL_LOAD_OWNER_RECLAIMED" for x in brain.logged))


if __name__ == "__main__":
    unittest.main()
