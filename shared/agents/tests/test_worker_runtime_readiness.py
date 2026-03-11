#!/usr/bin/env python3
"""Regression tests for worker task claiming against runtime readiness."""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

if "filelock" not in sys.modules:
    sys.modules["filelock"] = types.SimpleNamespace(FileLock=object, Timeout=Exception)
if "requests" not in sys.modules:
    sys.modules["requests"] = types.SimpleNamespace()

import gpu_tasks
from gpu_tasks import GPUTaskMixin
from gpu_workers import GPUWorkerMixin


class DummyLock:
    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class MockClaimGpu(GPUTaskMixin):
    def __init__(self, tmpdir: Path, runtime_state: str):
        self.name = "gpu-2"
        self.logger = MagicMock()
        self.queue_path = tmpdir / "queue"
        self.processing_path = tmpdir / "processing"
        self.queue_path.mkdir(parents=True, exist_ok=True)
        self.processing_path.mkdir(parents=True, exist_ok=True)
        self.thermal_pause_active = False
        self.thermal_pause_until = 0
        self.thermal_pause_reasons = []
        self.runtime_state = runtime_state
        self.runtime_error_code = None
        self.runtime_error_detail = None
        self.model_loaded = True
        self.loaded_model = "qwen2.5:7b"
        self.loaded_tier = 1
        self.model_tier_by_id = {"qwen2.5:7b": 1}
        self.runtime_placement = "single_gpu"
        self.runtime_group_id = None
        self.runtime_port = 11436
        self.runtime_backend = "llama"
        self.split_runtime_owner = False
        self.active_workers = {}
        self.claimed_vram = 0
        self.port = 11436

    def _get_preferred_classes(self):
        return ["llm"]

    def _get_vram_budget(self):
        return 10_000

    def _get_task_vram_cost(self, _task):
        return 1

    def _can_accept_work_task(self):
        return True, ""

    def _attest_runtime_reality(self):
        return {"ok": True, "mismatch_reason": "", "details": {"actual_port_models": ["qwen2.5:7b"]}}

    def _handle_attestation_miss(self, reason):
        return {"action": "soft_fail", "reason": reason}

    def _validate_split_ready_token(self):
        return {"ok": True, "reason": ""}

    def _is_gpu_too_hot(self):
        return False

    def _reserve_task_resources(self, _task, _vram_cost):
        return True

    def _scan_emergency_meta_tasks(self, _commands):
        return []


class MockSpawnGpu(GPUWorkerMixin):
    def __init__(self, runtime_state: str):
        self.name = "gpu-2"
        self.logger = MagicMock()
        self.runtime_state = runtime_state
        self.model_loaded = True
        self.loaded_model = "qwen2.5:7b"
        self.loaded_tier = 1
        self.model_tier_by_id = {"qwen2.5:7b": 1}
        self.runtime_placement = "single_gpu"
        self.runtime_group_id = None
        self.runtime_api_base = "http://127.0.0.1:11436"
        self.runtime_backend = "llama"
        self.gpu_id = 2
        self.config_path = Path("/tmp/config.json")
        self.permissions_file = "/tmp/permissions.json"
        self.active_workers = {}
        self.outbox = []
        self.processing_path = Path("/tmp")
        self.stats = {"tasks_completed": 0, "tasks_failed": 0}
        self.model = "qwen2.5:7b"

    def _llm_task_runtime_compatible(self, task):
        return GPUTaskMixin._llm_task_runtime_compatible(self, task)

    def _get_task_vram_cost(self, _task):
        return 1

    def _remove_task_heartbeat(self, _task_id):
        return None

    def _write_task_heartbeat(self, _task_id, _worker_id, _pid, peak_vram_mb=0):
        return None


class WorkerRuntimeReadinessTests(unittest.TestCase):
    def setUp(self):
        self._orig_lock = gpu_tasks.FileLock
        gpu_tasks.FileLock = DummyLock

    def tearDown(self):
        gpu_tasks.FileLock = self._orig_lock

    def test_claim_tasks_skips_llm_when_runtime_is_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            gpu = MockClaimGpu(Path(tmp), runtime_state="loading_single")
            task = {
                "task_id": "task-1",
                "task_class": "llm",
                "llm_model": "qwen2.5:7b",
                "priority": 5,
            }
            task_file = gpu.queue_path / "task-1.json"
            task_file.write_text(json.dumps(task), encoding="utf-8")

            claimed = gpu.claim_tasks()

            self.assertEqual(claimed, [])
            self.assertTrue(task_file.exists())

    def test_claim_tasks_claims_llm_when_runtime_is_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            gpu = MockClaimGpu(Path(tmp), runtime_state="ready_single")
            task = {
                "task_id": "task-1",
                "task_class": "llm",
                "llm_model": "qwen2.5:7b",
                "priority": 5,
            }
            task_file = gpu.queue_path / "task-1.json"
            task_file.write_text(json.dumps(task), encoding="utf-8")

            claimed = gpu.claim_tasks()

            self.assertEqual(len(claimed), 1)
            self.assertFalse(task_file.exists())
            self.assertTrue((gpu.processing_path / "task-1.json").exists())

    def test_split_runtime_refuses_single_gpu_task_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            gpu = MockClaimGpu(Path(tmp), runtime_state="ready_split")
            gpu.runtime_placement = "split_gpu"
            gpu.runtime_group_id = "pair_1_3"
            gpu.loaded_model = "qwen2.5-coder:14b"
            gpu.loaded_tier = 2
            gpu.model_tier_by_id["qwen2.5-coder:14b"] = 2
            task = {
                "task_id": "task-1",
                "task_class": "llm",
                "llm_model": "qwen2.5:7b",
                "llm_min_tier": 1,
                "llm_placement": "single_gpu",
                "priority": 5,
            }
            task_file = gpu.queue_path / "task-1.json"
            task_file.write_text(json.dumps(task), encoding="utf-8")

            claimed = gpu.claim_tasks()

            self.assertEqual(claimed, [])
            self.assertTrue(task_file.exists())

    def test_spawn_worker_refuses_llm_when_runtime_is_loading(self):
        gpu = MockSpawnGpu(runtime_state="loading_single")
        task = {
            "task_id": "task-1",
            "task_class": "llm",
            "llm_model": "qwen2.5:7b",
        }

        gpu._spawn_worker(task)

        self.assertEqual(len(gpu.active_workers), 0)
        self.assertEqual(len(gpu.outbox), 1)
        self.assertIn("runtime_not_ready:loading_single", gpu.outbox[0].result["error"])


if __name__ == "__main__":
    unittest.main()
