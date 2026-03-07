#!/usr/bin/env python3
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
import gpu_split
from gpu import GPUAgent
from gpu_split import GPUSplitMixin
from gpu_tasks import GPUTaskMixin


class DummyLock:
    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class MockSplitGpu(GPUSplitMixin, GPUTaskMixin):
    def __init__(self, tmpdir: Path):
        self.name = "gpu-3"
        self.logger = MagicMock()
        self.split_state_dir = tmpdir / "split_llm"
        self.split_state_dir.mkdir(parents=True, exist_ok=True)
        self.shared_path = tmpdir
        self.port = 11433
        self.runtime_port = 11441
        self.runtime_group_id = "pair_3_4"
        self.runtime_placement = "split_gpu"
        self.loaded_model = "model-a"
        self.model_loaded = True
        self.loaded_tier = 2
        self.model_tier_by_id = {"model-a": 2}
        self.split_runtime_owner = False
        self.split_runtime_generation = "gen-a"
        self.split_runtime_invariant_failures = 0
        self.split_runtime_port_model_miss_timestamps = []
        self.split_runtime_ready_at = None
        self.active_meta_task = None
        self.active_workers = []
        self.cleanup_calls = []
        self.reported_issues = []
        self.reservation_path = self._split_reservation_path("pair_3_4")
        self._service_calls = 0

    def _split_reservation_path(self, group_id: str) -> Path:
        return self.split_state_dir / f"{group_id}.json"

    def _touch_meta_task(self, phase="", force=False):
        return None

    def _atomic_join_split_reservation(self, reservation, _group_id):
        return reservation, True, ""

    def _is_group_member(self, group):
        members = [str(m).strip() for m in group.get("members", []) if str(m).strip()]
        return self.name in members

    def _service_split_reservations(self):
        self._service_calls += 1
        if self._service_calls == 1 and self.reservation_path.exists():
            self.reservation_path.unlink()

    def _coordinated_split_failure_cleanup(self, group_id, split_port, reason, task_id=None):
        payload = {
            "group_id": group_id,
            "split_port": split_port,
            "reason": reason,
            "task_id": task_id,
        }
        self.cleanup_calls.append(payload)
        return payload

    def _runtime_reset_port_and_state(
        self,
        ports_to_clean,
        reason,
        task_id=None,
        *,
        stop_split_runtime=False,
        stop_local_ollama=False,
    ):
        payload = {
            "ports_to_clean": ports_to_clean,
            "reason": reason,
            "task_id": task_id,
            "stop_split_runtime": stop_split_runtime,
            "stop_local_ollama": stop_local_ollama,
        }
        self.cleanup_calls.append(payload)
        return payload

    def _split_group_has_any_active_work(self, _group_id: str) -> bool:
        return False

    def _read_json_file(self, path: Path):
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _split_runtime_has_expected_owner(self, port, model_id, group_id=None):
        return False

    def _split_runtime_has_any_listener(self, port):
        return True

    def _split_runtime_has_model_loaded(self, port, model_id):
        return True

    def _report_split_health_issue(self, **kwargs):
        self.reported_issues.append(kwargs)

    def _clear_split_health_issue(self):
        return None


class SplitRuntimeHardeningTests(unittest.TestCase):
    def setUp(self):
        self._orig_lock = gpu_tasks.FileLock
        gpu_tasks.FileLock = DummyLock

    def tearDown(self):
        gpu_tasks.FileLock = self._orig_lock

    def test_load_split_reservation_disappearance_forces_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            gpu = MockSplitGpu(Path(tmp))
            task = {
                "task_id": "task-1",
                "target_model": "model-a",
                "candidate_groups": [
                    {"id": "pair_3_4", "members": ["gpu-3", "gpu-4"], "port": 11441}
                ],
            }

            result = gpu._execute_meta_command(task, "load_split_llm")

            self.assertFalse(result["success"])
            self.assertIn("reservation disappeared", result["output"])
            self.assertEqual(len(gpu.cleanup_calls), 1)
            self.assertEqual(
                gpu.cleanup_calls[0]["reason"],
                "dead_split_state:reservation_disappeared",
            )

    def test_owner_loss_invariant_forces_immediate_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            gpu = MockSplitGpu(Path(tmp))
            gpu.reservation_path.write_text(
                json.dumps(
                    {
                        "group_id": "pair_3_4",
                        "status": "ready",
                        "target_model": "model-a",
                        "members": ["gpu-3", "gpu-4"],
                        "port": 11441,
                    }
                ),
                encoding="utf-8",
            )

            gpu._check_split_runtime_invariants()

            self.assertEqual(len(gpu.cleanup_calls), 1)
            self.assertEqual(
                gpu.cleanup_calls[0]["reason"],
                "dead_split_state:owner_meta_mismatch",
            )
            self.assertEqual(gpu.reported_issues, [])

    def test_pending_split_coordination_detects_partner_nudge(self):
        with tempfile.TemporaryDirectory() as tmp:
            gpu = MockSplitGpu(Path(tmp))
            gpu._write_split_partner_nudge("pair_3_4", ["gpu-3", "gpu-4"], "gpu-3")

            self.assertTrue(gpu._has_pending_split_coordination())

    def test_gpu_loop_sleep_shortens_for_split_coordination(self):
        agent = GPUAgent.__new__(GPUAgent)
        agent._has_pending_split_coordination = lambda: True

        self.assertEqual(agent._loop_sleep_seconds(), 0.5)

        agent._has_pending_split_coordination = lambda: False
        self.assertEqual(agent._loop_sleep_seconds(), 5)

    def test_split_warmup_request_budget_is_bounded(self):
        gpu = MockSplitGpu(Path(tempfile.mkdtemp()))
        original_time = gpu_split.time.time
        try:
            gpu_split.time.time = lambda: 100.0
            self.assertEqual(gpu._split_warmup_request_budget_seconds(500.0), 25)
            self.assertEqual(gpu._split_warmup_request_budget_seconds(110.0), 10)
        finally:
            gpu_split.time.time = original_time

    def test_split_runtime_stable_model_presence_accepts_loaded_model(self):
        gpu = MockSplitGpu(Path(tempfile.mkdtemp()))
        original_time = gpu_split.time.time
        original_sleep = gpu_split.time.sleep
        original_requests = gpu_split.requests

        class Response:
            def __init__(self, models):
                self.status_code = 200
                self._models = models

            def json(self):
                return {"models": [{"name": name} for name in self._models]}

        ticks = iter([0, 1, 2, 3, 4, 5])
        gpu_split.time.time = lambda: next(ticks, 999)
        gpu_split.time.sleep = lambda _s: None
        gpu_split.requests = types.SimpleNamespace(
            get=lambda _url, timeout=5: Response(["qwen2.5-coder:14b"])
        )
        try:
            ok = gpu._split_runtime_has_stable_model_presence(11440, "qwen2.5-coder:14b")
            self.assertTrue(ok)
        finally:
            gpu_split.time.time = original_time
            gpu_split.time.sleep = original_sleep
            gpu_split.requests = original_requests


if __name__ == "__main__":
    unittest.main()
