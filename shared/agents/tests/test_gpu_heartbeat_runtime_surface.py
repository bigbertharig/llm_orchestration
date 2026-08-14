#!/usr/bin/env python3
"""Tests for backend-neutral GPU heartbeat publication."""

from __future__ import annotations

import json
import sys
import tempfile
from unittest.mock import patch
import unittest
from pathlib import Path
import types

sys.path.insert(0, str(Path(__file__).parent.parent))

from gpu import GPUAgent


class GPUHeartbeatRuntimeSurfaceTests(unittest.TestCase):
    def test_gpu_heartbeat_uses_runtime_neutral_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            heartbeat_path = Path(tmp) / "heartbeat.json"
            agent = GPUAgent.__new__(GPUAgent)

            agent._get_gpu_stats = lambda: {
                "temperature_c": 41,
                "power_draw_w": 55.5,
                "vram_used_mb": 4096,
                "vram_total_mb": 6144,
                "vram_percent": 66,
                "gpu_util_percent": 12,
                "clock_mhz": 1455,
                "throttle_status": "0x0",
            }
            agent._is_resource_constrained = lambda _stats: (False, [])
            agent._check_runtime_health = lambda: {
                "healthy": True,
                "loaded_models": ["qwen2.5:7b"],
                "response_ms": 12,
            }
            agent._get_cpu_temp = lambda: 38
            agent._is_runtime_ready = lambda: True
            agent._get_split_health_issue_heartbeat = lambda: {"has_issue": False}
            agent._get_vram_budget = lambda: 5000
            agent._read_gpu_lease = lambda: {
                "leased": True,
                "lease_id": "supervision-1",
                "controller": "supervision",
                "lane": "supervision",
                "owner": "county-session",
                "gpu_ids": [2],
            }

            agent.active_workers = {}
            agent.active_meta_task = None
            agent.gpu_id = 2
            agent.name = "gpu-2"
            agent.state = "hot"
            agent.runtime_backend = "llama"
            agent.runtime_state = "ready_single"
            agent.runtime_state_updated_at = "2026-03-07T23:30:00"
            agent.runtime_transition_task_id = None
            agent.runtime_transition_phase = None
            agent.runtime_error_code = None
            agent.runtime_error_detail = None
            agent.model_loaded = True
            agent.loaded_model = "qwen2.5:7b"
            agent.loaded_tier = 1
            agent.worker_num_ctx = 2048
            agent.model = "qwen2.5:7b"
            agent.model_tier = 1
            agent.runtime_placement = "single_gpu"
            agent.runtime_group_id = None
            agent.split_runtime_owner = False
            agent.split_runtime_generation = None
            agent.runtime_port = 11436
            agent.runtime_api_base = "http://127.0.0.1:11436"
            agent.pending_global_load_owner_issue = {"has_issue": False}
            agent.runtime_healthy = True
            agent.claimed_vram = 0
            agent.thermal_pause_active = False
            agent.thermal_pause_reasons = []
            agent.thermal_pause_until = 0.0
            agent.thermal_pause_current_seconds = 0
            agent.thermal_pause_attempts = 0
            agent.last_thermal_event = None
            agent.thermal_overheat_started_at = None
            agent.thermal_overheat_sustained_seconds = 0
            agent.thermal_overheat_incident_id = None
            agent.thermal_recovery_reset_count = 0
            agent.thermal_recovery_last_reset_at = None
            agent.env_block_reason = None
            agent.stats = {"tasks_completed": 0, "tasks_failed": 0}
            agent.heartbeat_file = heartbeat_path

            GPUAgent._write_heartbeat(agent)

            payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["runtime_backend"], "llama")
            self.assertTrue(payload["runtime_healthy"])
            self.assertEqual(payload["runtime_api_base"], "http://127.0.0.1:11436")
            self.assertEqual(payload["runtime_health"]["loaded_models"], ["qwen2.5:7b"])
            self.assertTrue(payload["leased"])
            self.assertEqual(payload["lease_controller"], "supervision")
            self.assertEqual(payload["busy_lane"], "SUPERVISING")
            self.assertEqual(
                set(payload).intersection({"runtime_healthy", "runtime_api_base", "runtime_backend"}),
                {"runtime_healthy", "runtime_api_base", "runtime_backend"},
            )

    def test_gpu_heartbeat_reconciles_live_runtime_when_cached_state_is_cold(self):
        with tempfile.TemporaryDirectory() as tmp:
            heartbeat_path = Path(tmp) / "heartbeat.json"
            agent = GPUAgent.__new__(GPUAgent)

            agent._get_gpu_stats = lambda: {
                "temperature_c": 43,
                "power_draw_w": 72.5,
                "vram_used_mb": 9216,
                "vram_total_mb": 24576,
                "vram_percent": 37,
                "gpu_util_percent": 9,
                "clock_mhz": 1410,
                "throttle_status": "0x0",
            }
            agent._is_resource_constrained = lambda _stats: (False, [])
            agent._get_cpu_temp = lambda: 39
            agent._is_runtime_ready = lambda: agent.runtime_state == "ready_single"
            agent._get_split_health_issue_heartbeat = lambda: {"has_issue": False}
            agent._get_vram_budget = lambda: 12000
            agent._read_gpu_lease = lambda: {"leased": False}

            agent.logger = types.SimpleNamespace(
                info=lambda *args, **kwargs: None,
                warning=lambda *args, **kwargs: None,
                error=lambda *args, **kwargs: None,
                debug=lambda *args, **kwargs: None,
            )
            agent.active_workers = {}
            agent.active_meta_task = None
            agent.gpu_id = 2
            agent.name = "gpu-2"
            agent.state = "cold"
            agent.runtime_backend = "llama"
            agent.runtime_state = "cold"
            agent.runtime_state_updated_at = "2026-04-02T20:00:00"
            agent.runtime_transition_task_id = None
            agent.runtime_transition_phase = None
            agent.runtime_error_code = None
            agent.runtime_error_detail = None
            agent.model_loaded = False
            agent.loaded_model = None
            agent.loaded_tier = 0
            agent.worker_num_ctx = 2048
            agent.model = "qwen2.5-coder:7b"
            agent.model_tier = 1
            agent.model_tier_by_id = {"qwen2.5-coder:7b": 1}
            agent.runtime_placement = "single_gpu"
            agent.runtime_group_id = None
            agent.split_runtime_owner = False
            agent.split_runtime_generation = None
            agent.port = 11436
            agent.runtime_port = 11436
            agent.runtime_api_base = "http://127.0.0.1:11436"
            agent.pending_global_load_owner_issue = {"has_issue": False}
            agent.runtime_healthy = True
            agent.runtime_consecutive_failures = 3
            agent.runtime_health_threshold = 6
            agent.runtime_circuit_breaker = 8
            agent.claimed_vram = 0
            agent.thermal_pause_active = False
            agent.thermal_pause_reasons = []
            agent.thermal_pause_until = 0.0
            agent.thermal_pause_current_seconds = 0
            agent.thermal_pause_attempts = 0
            agent.last_thermal_event = None
            agent.thermal_overheat_started_at = None
            agent.thermal_overheat_sustained_seconds = 0
            agent.thermal_overheat_incident_id = None
            agent.thermal_recovery_reset_count = 0
            agent.thermal_recovery_last_reset_at = None
            agent.env_block_reason = None
            agent.stats = {"tasks_completed": 0, "tasks_failed": 0}
            agent.heartbeat_file = heartbeat_path

            class _Response:
                status_code = 200

                @staticmethod
                def json():
                    return {"data": [{"id": "/models/qwen2.5-coder-7b.gguf"}]}

            with patch("gpu_llama.requests.get", return_value=_Response()):
                GPUAgent._write_heartbeat(agent)

            payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["runtime_state"], "ready_single")
            self.assertEqual(payload["runtime_transition_phase"], "runtime_health_reconciled")
            self.assertTrue(payload["model_loaded"])
            self.assertEqual(payload["loaded_model"], "qwen2.5-coder:7b")
            self.assertTrue(payload["capability_ready"])
            self.assertTrue(payload["runtime_healthy"])
            self.assertEqual(payload["runtime_health"]["loaded_models"], ["/models/qwen2.5-coder-7b.gguf"])


if __name__ == "__main__":
    unittest.main()
