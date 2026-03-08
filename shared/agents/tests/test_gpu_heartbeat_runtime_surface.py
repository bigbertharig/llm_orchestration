#!/usr/bin/env python3
"""Tests for backend-neutral GPU heartbeat publication."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
import types

sys.path.insert(0, str(Path(__file__).parent.parent))

if "filelock" not in sys.modules:
    sys.modules["filelock"] = types.SimpleNamespace(FileLock=object, Timeout=Exception)

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
            self.assertEqual(
                set(payload).intersection({"runtime_healthy", "runtime_api_base", "runtime_backend"}),
                {"runtime_healthy", "runtime_api_base", "runtime_backend"},
            )


if __name__ == "__main__":
    unittest.main()
