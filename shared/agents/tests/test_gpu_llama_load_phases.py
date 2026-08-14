#!/usr/bin/env python3
"""Tests for explicit single-load heartbeat phases."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gpu_constants import RUNTIME_STATE_LOADING_SINGLE, RUNTIME_STATE_READY_SINGLE
from gpu_llama import GPULlamaMixin


class LoadPhaseStub(GPULlamaMixin):
    def __init__(self):
        self.name = "gpu-2"
        self.model_loaded = False
        self.model = "qwen2.5-coder:7b"
        self.port = 11436
        self.gpu_id = 2
        self.config = {"llama_runtime_image": "llama-runtime:sm61-sm86"}
        self.gpu_config = {"name": "gpu-2"}
        self.worker_num_ctx = 8192
        self.model_tier_by_id = {"qwen2.5-coder:7b": 1}
        self.model_meta_by_id = {"qwen2.5-coder:7b": {"placement": "single_gpu"}}
        self.model_tier = 1
        self.loaded_model = None
        self.loaded_tier = 0
        self.runtime_placement = None
        self.runtime_group_id = None
        self.runtime_port = None
        self.runtime_api_base = None
        self._llama_container_id = None
        self._llama_start_cmd = None
        self._llama_start_time = None
        self.phase_events = []
        self.meta_events = []
        self.logger = types.SimpleNamespace(
            info=lambda *args, **kwargs: None,
            warning=lambda *args, **kwargs: None,
            error=lambda *args, **kwargs: None,
        )

    def _can_accept_load_task(self):
        return True, ""

    def _resolve_gguf_path(self, model_id):
        return f"/models/{model_id}.gguf"

    def _llama_container_name(self):
        return "llama-worker-gpu-2"

    def _probe_existing_loaded_model(self, **_kwargs):
        return False

    def _stop_llama_container(self, _container_name):
        return None

    def _llama_api_base(self):
        return "http://127.0.0.1:11436"

    def _assert_full_gpu_offload(self, _container_name, _target_model):
        return None

    def _run_with_global_model_load_lock(self, phase, fn, max_wait_seconds):
        self.lock_phase = phase
        self.lock_wait = max_wait_seconds
        return fn()

    def _touch_meta_task(self, phase=None):
        self.meta_events.append(phase)

    def _set_runtime_state(self, state, task_id=None, phase=None, error_code=None, error_detail=None):
        self.phase_events.append((state, phase, task_id, error_code, error_detail))
        self.runtime_state = state
        self.runtime_transition_phase = phase
        self.runtime_transition_task_id = task_id

    def _get_container_logs(self, _container_name, tail=20):
        return ""


class LlamaLoadPhaseTests(unittest.TestCase):
    def test_start_llama_clears_stale_loaded_state(self):
        stub = LoadPhaseStub()
        stub.model_loaded = True
        stub.loaded_model = "qwen2.5-coder:7b"
        stub.loaded_tier = 1
        stub.runtime_state = RUNTIME_STATE_READY_SINGLE
        stub.runtime_transition_phase = "load_complete"
        stub.runtime_transition_task_id = "stale-task"
        stub.runtime_consecutive_failures = 4
        stub.runtime_healthy = False

        stub.start_llama()

        self.assertFalse(stub.model_loaded)
        self.assertIsNone(stub.loaded_model)
        self.assertEqual(stub.loaded_tier, 0)
        self.assertEqual(stub.runtime_state, "cold")
        self.assertEqual(stub.runtime_transition_phase, "startup_cold_reset")
        self.assertIsNone(stub.runtime_transition_task_id)
        self.assertEqual(stub.runtime_port, 11436)
        self.assertEqual(stub.runtime_api_base, "http://127.0.0.1:11436")
        self.assertEqual(stub.runtime_consecutive_failures, 0)
        self.assertTrue(stub.runtime_healthy)

    def test_single_load_updates_explicit_runtime_phases(self):
        stub = LoadPhaseStub()

        original_run = __import__("subprocess").run
        original_get = __import__("requests").get
        original_sleep = __import__("time").sleep
        try:
            def fake_run(cmd, capture_output, text, timeout):
                return types.SimpleNamespace(returncode=0, stdout="container-id\n", stderr="")

            responses = [
                types.SimpleNamespace(status_code=503),
                types.SimpleNamespace(status_code=200),
            ]

            def fake_get(url, timeout):
                return responses.pop(0)

            __import__("subprocess").run = fake_run
            __import__("requests").get = fake_get
            __import__("time").sleep = lambda _seconds: None

            stub._llama_load_model(model_id="qwen2.5-coder:7b", task_id="task-1")
        finally:
            __import__("subprocess").run = original_run
            __import__("requests").get = original_get
            __import__("time").sleep = original_sleep

        phases = [phase for state, phase, *_rest in stub.phase_events if state == RUNTIME_STATE_LOADING_SINGLE]
        self.assertIn("preflight_passed", phases)
        self.assertIn("acquiring_global_lock", phases)
        self.assertIn("stopping_existing_container", phases)
        self.assertIn("starting_container", phases)
        self.assertIn("waiting_runtime_ready", phases)
        self.assertIn("validating_runtime_ready", phases)
        self.assertIn("load_llm_waiting_ready", stub.meta_events)
        self.assertEqual(stub.runtime_state, RUNTIME_STATE_READY_SINGLE)
        self.assertEqual(stub.runtime_transition_phase, "load_complete")
        self.assertEqual(stub.loaded_model, "qwen2.5-coder:7b")


if __name__ == "__main__":
    unittest.main()
