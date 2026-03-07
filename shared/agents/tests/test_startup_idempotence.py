#!/usr/bin/env python3
"""Tests for startup idempotence checks."""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
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

            brain_payload = {
                "last_updated": "2026-03-06T17:47:44",
                "brain_pids": {"pid": 12345},
            }
            gpu_payload = {
                "last_updated": "2026-03-06T17:47:40",
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
                        "last_updated": "2026-03-06T17:47:44",
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
                        "last_updated": "2026-03-06T17:47:44",
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


if __name__ == "__main__":
    unittest.main()
