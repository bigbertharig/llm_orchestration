#!/usr/bin/env python3
"""Regression tests for runtime preflight listener ownership checks."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "runtime_preflight.py"
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location("runtime_preflight", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load {SCRIPT_PATH}")
runtime_preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime_preflight)


class RuntimePreflightTests(unittest.TestCase):
    def test_ignores_listener_for_worker_in_managed_transition(self):
        conflicts = runtime_preflight._listener_conflicts(
            [("gpu-2", 11436, None)],
            [],
            {"11436": ["LISTEN 0 512 127.0.0.1:11436 0.0.0.0:*"]},
            [{"name": "llama-worker-gpu-2", "status": "Up 10 seconds"}],
            {
                "gpu-2": {
                    "runtime_state": "loading_single",
                    "meta_task_active": True,
                    "runtime_transition_task_id": "abc123",
                }
            },
        )

        self.assertEqual(conflicts, [])

    def test_flags_listener_for_truly_cold_worker(self):
        conflicts = runtime_preflight._listener_conflicts(
            [("gpu-2", 11436, None)],
            [],
            {"11436": ["LISTEN 0 512 127.0.0.1:11436 0.0.0.0:*"]},
            [{"name": "llama-worker-gpu-2", "status": "Up 10 seconds"}],
            {
                "gpu-2": {
                    "runtime_state": "cold",
                    "meta_task_active": False,
                    "runtime_transition_task_id": "",
                }
            },
        )

        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["type"], "cold_worker_listener_present")


if __name__ == "__main__":
    unittest.main()
