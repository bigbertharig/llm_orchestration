#!/usr/bin/env python3
"""Tests for startup idempotence checks."""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import startup


class StartupIdempotenceTests(unittest.TestCase):
    def test_healthy_orchestrator_requires_fresh_brain_and_worker_heartbeats(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            (shared / "brain").mkdir(parents=True, exist_ok=True)
            (shared / "gpus" / "gpu_1").mkdir(parents=True, exist_ok=True)
            now = time.time()
            now_iso = datetime.now().isoformat()

            brain_payload = {
                "last_updated": now_iso,
                "brain_pids": {"pid": 12345},
            }
            gpu_payload = {
                "last_updated": now_iso,
            }
            brain_hb = shared / "brain" / "heartbeat.json"
            gpu_hb = shared / "gpus" / "gpu_1" / "heartbeat.json"
            brain_hb.write_text(json.dumps(brain_payload), encoding="utf-8")
            gpu_hb.write_text(json.dumps(gpu_payload), encoding="utf-8")
            os_utime = __import__("os").utime
            os_utime(brain_hb, (now, now))
            os_utime(gpu_hb, (now, now))

            original_pid_check = startup._pid_is_alive
            startup._pid_is_alive = lambda pid: pid == 12345
            try:
                self.assertTrue(
                    startup._orchestrator_is_healthy(
                        shared,
                        expected_gpu_ids=[1],
                        brain_only=False,
                        stale_seconds=120,
                    )
                )
            finally:
                startup._pid_is_alive = original_pid_check

    def test_missing_worker_heartbeat_prevents_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            (shared / "brain").mkdir(parents=True, exist_ok=True)
            brain_hb = shared / "brain" / "heartbeat.json"
            brain_hb.write_text(
                json.dumps(
                    {
                        "last_updated": datetime.now().isoformat(),
                        "brain_pids": {"pid": 12345},
                    }
                ),
                encoding="utf-8",
            )

            original_pid_check = startup._pid_is_alive
            startup._pid_is_alive = lambda pid: True
            try:
                self.assertFalse(
                    startup._orchestrator_is_healthy(
                        shared,
                        expected_gpu_ids=[1],
                        brain_only=False,
                        stale_seconds=120,
                    )
                )
            finally:
                startup._pid_is_alive = original_pid_check

    def test_brain_only_mode_skips_worker_heartbeat_requirement(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            (shared / "brain").mkdir(parents=True, exist_ok=True)
            brain_hb = shared / "brain" / "heartbeat.json"
            brain_hb.write_text(
                json.dumps(
                    {
                        "last_updated": datetime.now().isoformat(),
                        "brain_pids": {"pid": 12345},
                    }
                ),
                encoding="utf-8",
            )

            original_pid_check = startup._pid_is_alive
            startup._pid_is_alive = lambda pid: True
            try:
                self.assertTrue(
                    startup._orchestrator_is_healthy(
                        shared,
                        expected_gpu_ids=[1, 2, 3],
                        brain_only=True,
                        stale_seconds=120,
                    )
                )
            finally:
                startup._pid_is_alive = original_pid_check

    def test_reclaim_worker_port_stops_named_worker_container_first(self):
        calls = []
        original_is_port_open = startup._is_port_open
        original_run = startup.subprocess.run
        original_sleep = startup.time.sleep

        state = {"checks": 0}

        def fake_is_port_open(_port, host="127.0.0.1"):
            state["checks"] += 1
            return state["checks"] == 1

        def fake_run(cmd, **kwargs):
            calls.append(cmd)

            class Result:
                stdout = ""
                stderr = ""
                returncode = 0

            return Result()

        startup._is_port_open = fake_is_port_open
        startup.subprocess.run = fake_run
        startup.time.sleep = lambda _seconds: None
        try:
            startup._reclaim_worker_port(11437, "gpu-3")
        finally:
            startup._is_port_open = original_is_port_open
            startup.subprocess.run = original_run
            startup.time.sleep = original_sleep

        self.assertEqual(calls[0], ["docker", "rm", "-f", "llama-worker-gpu-3"])
        self.assertEqual(len(calls), 1)

    def test_enqueue_startup_load_prefers_explicit_gpu2_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            agents_dir = shared / "agents"
            agents_dir.mkdir(parents=True, exist_ok=True)
            (agents_dir / "models.catalog.json").write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "id": "qwen2.5-coder:14b",
                                "placement": "split_gpu",
                                "split_groups": [
                                    {"id": "pair_1_3", "members": ["gpu-1", "gpu-3"], "port": 11440}
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            startup._enqueue_startup_load_llm(
                shared_path=shared,
                created_by="startup",
                count=1,
                available_workers=[
                    {"name": "gpu-1", "model": "qwen2.5:7b"},
                    {"name": "gpu-2", "model": "qwen2.5:7b"},
                    {"name": "gpu-3", "model": "qwen2.5:7b"},
                ],
                agents_dir=agents_dir,
                preferred_workers=["gpu-2"],
            )

            queue_files = list((shared / "tasks" / "queue").glob("*.json"))
            self.assertEqual(len(queue_files), 1)
            payload = json.loads(queue_files[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["candidate_workers"], ["gpu-2"])
            self.assertEqual(payload["target_model"], "qwen2.5:7b")

    def test_clear_stale_heartbeats_removes_brain_and_worker_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            brain_hb = shared / "brain" / "heartbeat.json"
            unified_brain_hb = shared / "heartbeats" / "brain.json"
            gpu_hb = shared / "gpus" / "gpu_2" / "heartbeat.json"

            brain_hb.parent.mkdir(parents=True, exist_ok=True)
            unified_brain_hb.parent.mkdir(parents=True, exist_ok=True)
            gpu_hb.parent.mkdir(parents=True, exist_ok=True)

            for hb in (brain_hb, unified_brain_hb, gpu_hb):
                hb.write_text("{}", encoding="utf-8")

            startup._clear_stale_heartbeats(shared, [{"id": 2}])

            self.assertFalse(brain_hb.exists())
            self.assertFalse(unified_brain_hb.exists())
            self.assertFalse(gpu_hb.exists())


if __name__ == "__main__":
    unittest.main()
