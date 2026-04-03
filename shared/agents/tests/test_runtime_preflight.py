#!/usr/bin/env python3
"""Regression tests for runtime preflight listener ownership checks."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SHARED_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "runtime_preflight.py"
REPO_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "runtime_preflight.py"
sys.path.insert(0, str(SHARED_SCRIPT_PATH.parent))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runtime_preflight = _load_module("shared_runtime_preflight", SHARED_SCRIPT_PATH)
runtime_preflight_wrapper = _load_module("repo_runtime_preflight", REPO_SCRIPT_PATH)


class RuntimePreflightTests(unittest.TestCase):
    def test_normalizes_repo_shared_alias_for_remote_config(self):
        normalized = runtime_preflight_wrapper._to_runtime_shared_path(
            "/home/bryan/llm_orchestration/shared/agents/config.json"
        )

        self.assertEqual(normalized, "/mnt/shared/agents/config.json")

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
