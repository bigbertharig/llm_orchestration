#!/usr/bin/env python3
"""Regression tests for dashboard worker hold rendering."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.workers import load_worker_rows


class DashboardWorkerRowsTests(unittest.TestCase):
    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _now(self) -> str:
        return datetime.now().isoformat()

    def test_cold_reset_phase_is_not_shown_as_active_hold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shared_path = Path(tmpdir)
            heartbeat = {
                "gpu_id": 2,
                "name": "gpu-2",
                "state": "cold",
                "runtime_state": "cold",
                "runtime_transition_phase": "startup_cold_reset",
                "runtime_transition_task_id": None,
                "model_loaded": False,
                "loaded_model": None,
                "loaded_tier": 0,
                "runtime_placement": "single_gpu",
                "runtime_group_id": None,
                "runtime_port": 11436,
                "runtime_api_base": "http://127.0.0.1:11436",
                "last_updated": self._now(),
                "temperature_c": 31,
                "cpu_temp_c": 62,
                "power_draw_w": 5.99,
                "vram_used_mb": 4,
                "vram_total_mb": 6144,
                "gpu_util_percent": 0,
                "active_workers": 0,
                "active_tasks": [],
                "meta_task_active": False,
            }
            self._write_json(shared_path / "gpus" / "gpu_2" / "heartbeat.json", heartbeat)

            rows = load_worker_rows(shared_path, [])

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["name"], "gpu-2")
            self.assertEqual(rows[0]["holding"], [])

    def test_transition_phase_with_task_id_is_shown_as_hold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shared_path = Path(tmpdir)
            heartbeat = {
                "gpu_id": 2,
                "name": "gpu-2",
                "state": "cold",
                "runtime_state": "loading_single",
                "runtime_transition_phase": "starting_container",
                "runtime_transition_task_id": "task-123",
                "model_loaded": False,
                "loaded_model": None,
                "loaded_tier": 0,
                "runtime_placement": "single_gpu",
                "runtime_group_id": None,
                "runtime_port": 11436,
                "runtime_api_base": "http://127.0.0.1:11436",
                "last_updated": self._now(),
                "temperature_c": 31,
                "cpu_temp_c": 62,
                "power_draw_w": 5.99,
                "vram_used_mb": 4,
                "vram_total_mb": 6144,
                "gpu_util_percent": 0,
                "active_workers": 0,
                "active_tasks": [],
                "meta_task_active": True,
            }
            self._write_json(shared_path / "gpus" / "gpu_2" / "heartbeat.json", heartbeat)

            rows = load_worker_rows(shared_path, [])

            self.assertEqual(len(rows), 1)
            self.assertIn("starting_container", rows[0]["holding"])

    def test_terminal_load_complete_phase_is_not_shown_as_hold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shared_path = Path(tmpdir)
            heartbeat = {
                "gpu_id": 5,
                "name": "gpu-5",
                "state": "hot",
                "runtime_state": "ready_single",
                "runtime_transition_phase": "load_complete",
                "runtime_transition_task_id": "task-456",
                "model_loaded": True,
                "loaded_model": "qwen3.5:4b",
                "loaded_tier": 1,
                "runtime_placement": "single_gpu",
                "runtime_group_id": None,
                "runtime_port": 11439,
                "runtime_api_base": "http://127.0.0.1:11439",
                "last_updated": self._now(),
                "temperature_c": 30,
                "cpu_temp_c": 63,
                "power_draw_w": 3.72,
                "vram_used_mb": 3078,
                "vram_total_mb": 6144,
                "gpu_util_percent": 0,
                "active_workers": 0,
                "active_tasks": [],
                "meta_task_active": False,
            }
            self._write_json(shared_path / "gpus" / "gpu_5" / "heartbeat.json", heartbeat)

            rows = load_worker_rows(shared_path, [])

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["holding"], [])

    def test_lane_reservation_is_normalized_for_dashboard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shared_path = Path(tmpdir)
            heartbeat = {
                "gpu_id": 4,
                "name": "gpu-4",
                "state": "cold",
                "runtime_state": "cold",
                "runtime_transition_phase": "",
                "runtime_transition_task_id": None,
                "model_loaded": False,
                "loaded_model": None,
                "loaded_tier": 0,
                "runtime_placement": "single_gpu",
                "runtime_group_id": None,
                "runtime_port": 11438,
                "runtime_api_base": "http://127.0.0.1:11438",
                "last_updated": self._now(),
                "temperature_c": 31,
                "cpu_temp_c": 62,
                "power_draw_w": 5.99,
                "vram_used_mb": 4,
                "vram_total_mb": 6144,
                "gpu_util_percent": 0,
                "active_workers": 0,
                "active_tasks": [],
                "meta_task_active": False,
                "leased": True,
                "lease_controller": "benchmarks",
                "lease": {
                    "leased": True,
                    "controller": "benchmarks",
                    "lane": "benchmarks",
                    "owner": "bench-suite",
                },
                "busy": True,
                "busy_lane": "BENCHING",
                "busy_owner": "bench-suite",
                "busy_reason": "leased:BENCHING",
                "available_for_lanes": {
                    "BENCHING": False,
                    "TASKING": False,
                    "CHATTING": False,
                    "SUPERVISING": False,
                },
            }
            self._write_json(shared_path / "gpus" / "gpu_4" / "heartbeat.json", heartbeat)

            rows = load_worker_rows(shared_path, [])

            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["busy"])
            self.assertEqual(rows[0]["busy_lane"], "BENCHING")
            self.assertEqual(rows[0]["busy_owner"], "bench-suite")
            self.assertEqual(rows[0]["available_lanes"], [])
            self.assertEqual(
                rows[0]["blocked_lanes"],
                ["BENCHING", "TASKING", "CHATTING", "SUPERVISING"],
            )


if __name__ == "__main__":
    unittest.main()
