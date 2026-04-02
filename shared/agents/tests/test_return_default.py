#!/usr/bin/env python3
"""Regression tests for return_default transition logic."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "return_default.py"
SPEC = importlib.util.spec_from_file_location("return_default", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load {SCRIPT_PATH}")
return_default = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(return_default)


class ReturnDefaultTransitionTests(unittest.TestCase):
    def _fresh_timestamp(self) -> str:
        return datetime.now().isoformat()

    def test_read_gpu_heartbeat_returns_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared_dir = Path(tmp)
            hb_path = shared_dir / "gpus" / "gpu_2" / "heartbeat.json"
            hb_path.parent.mkdir(parents=True, exist_ok=True)
            hb_path.write_text(
                json.dumps(
                    {
                        "runtime_state_updated_at": self._fresh_timestamp(),
                        "loaded_model": "qwen2.5-coder:7b",
                        "model_loaded": True,
                        "runtime_placement": "single_gpu",
                    }
                ),
                encoding="utf-8",
            )

            result = return_default._read_gpu_heartbeat(shared_dir, "gpu-2")

            self.assertIsInstance(result, dict)
            self.assertEqual(result["loaded_model"], "qwen2.5-coder:7b")

    def test_wait_for_default_state_accepts_clean_cold_without_startup_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared_dir = Path(tmp)
            hb_path = shared_dir / "gpus" / "gpu_2" / "heartbeat.json"
            hb_path.parent.mkdir(parents=True, exist_ok=True)
            hb_path.write_text(
                json.dumps(
                    {
                        "runtime_state_updated_at": self._fresh_timestamp(),
                        "loaded_model": "",
                        "model_loaded": False,
                        "runtime_placement": "",
                    }
                ),
                encoding="utf-8",
            )
            (shared_dir / "tasks" / "queue").mkdir(parents=True, exist_ok=True)
            (shared_dir / "tasks" / "processing").mkdir(parents=True, exist_ok=True)
            (shared_dir / "brain" / "private_tasks").mkdir(parents=True, exist_ok=True)

            result = return_default._wait_for_default_state(
                shared_dir,
                {
                    "initial_hot_workers": 1,
                    "gpus": [
                        {"id": 2, "name": "gpu-2", "model": "qwen2.5-coder:7b"},
                    ],
                },
                timeout_s=1,
                stale_seconds=3600,
            )

            self.assertTrue(result["ok"])
            self.assertEqual(result["reason"], "clean_cold_without_startup_default")

    def test_wait_for_default_state_accepts_expected_default_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared_dir = Path(tmp)
            hb_path = shared_dir / "gpus" / "gpu_2" / "heartbeat.json"
            hb_path.parent.mkdir(parents=True, exist_ok=True)
            hb_path.write_text(
                json.dumps(
                    {
                        "runtime_state_updated_at": self._fresh_timestamp(),
                        "loaded_model": "qwen2.5-coder:7b",
                        "model_loaded": True,
                        "runtime_placement": "single_gpu",
                    }
                ),
                encoding="utf-8",
            )

            result = return_default._wait_for_default_state(
                shared_dir,
                {
                    "initial_hot_workers": 1,
                    "gpus": [
                        {"id": 2, "name": "gpu-2", "model": "qwen2.5-coder:7b"},
                    ],
                },
                timeout_s=1,
                stale_seconds=3600,
            )

            self.assertTrue(result["ok"])
            self.assertEqual(result["reason"], "")

    def test_runtime_probe_default_ready_accepts_expected_loaded_filename(self):
        result = return_default._default_ready_from_runtime_probe(
            {
                "initial_hot_workers": 1,
                "gpus": [
                    {"id": 2, "name": "gpu-2", "model": "qwen2.5-coder:7b"},
                ],
            },
            {
                "runtime_meta_processing_count": 0,
                "worker_scan": [
                    {
                        "name": "gpu-2",
                        "port": 11436,
                        "loaded_model": "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
                    }
                ],
            },
            model_filename_map={
                "qwen2.5-coder:7b": "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
            },
        )

        self.assertTrue(result["ok"])
        self.assertTrue(result["runtime_probe_used"])
        self.assertEqual(result["default_loaded_model"], "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf")

    def test_resolve_wait_timeout_prefers_runtime_profile_budget(self):
        result = return_default._resolve_wait_timeout_seconds(
            {
                "gpus": [
                    {"id": 2, "name": "gpu-2", "model": "qwen2.5-coder:7b"},
                ],
                "llama_single_profiles": {
                    "qwen2.5-coder:7b": {
                        "meta_timeout_seconds": 600,
                    }
                },
            },
            0,
        )

        self.assertEqual(result, 600)

    def test_resolve_wait_timeout_honors_explicit_override(self):
        result = return_default._resolve_wait_timeout_seconds({}, 123)

        self.assertEqual(result, 123)


if __name__ == "__main__":
    unittest.main()
