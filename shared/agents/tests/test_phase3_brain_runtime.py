#!/usr/bin/env python3
"""Phase 3 brain runtime regression tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
import urllib.error
import types
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

if "filelock" not in sys.modules:
    sys.modules["filelock"] = types.SimpleNamespace(FileLock=object, Timeout=Exception)

import brain
import setup
import startup
from brain_core import resolve_model_search_roots


class Phase3BrainRuntimeTests(unittest.TestCase):
    def test_build_config_writes_backend_neutral_runtime_host(self):
        config = setup.build_config(
            assignment={
                "_discovery_mode": "standard",
                "brain": {"model": "qwen2.5:32b", "gpus": [0]},
                "gpus": [],
                "worker_mode": "cold",
            },
            runtime={"host": "http://localhost:11434"},
            system={"hostname": "rig"},
        )

        self.assertEqual(config["runtime_host"], "http://localhost:11434")
        self.assertEqual(config["model_search_roots"], ["/mnt/shared/models"])

    def test_startup_uses_running_llama_brain_container_for_preload_detection(self):
        original_run = startup.subprocess.run

        class Result:
            returncode = 0
            stdout = "true\n"
            stderr = ""

        startup.subprocess.run = lambda *args, **kwargs: Result()
        original_urlopen = startup.urllib.request.urlopen

        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"data":[]}'

        startup.urllib.request.urlopen = lambda *args, **kwargs: Response()
        try:
            loaded = startup.check_loaded_models(
                "http://127.0.0.1:11434",
                runtime_backend="llama",
            )
        finally:
            startup.subprocess.run = original_run
            startup.urllib.request.urlopen = original_urlopen

        self.assertEqual(loaded, {"_llama_brain": {"size_vram": 0, "status": "ready"}})

    def test_startup_detects_loading_llama_brain_container(self):
        original_run = startup.subprocess.run
        original_urlopen = startup.urllib.request.urlopen

        class Result:
            returncode = 0
            stdout = "true\n"
            stderr = ""

        startup.subprocess.run = lambda *args, **kwargs: Result()

        def fake_urlopen(*args, **kwargs):
            raise urllib.error.HTTPError(
                url="http://127.0.0.1:11434/v1/models",
                code=503,
                msg="Loading model",
                hdrs=None,
                fp=None,
            )

        startup.urllib.request.urlopen = fake_urlopen
        try:
            loaded = startup.check_loaded_models(
                "http://127.0.0.1:11434",
                runtime_backend="llama",
            )
        finally:
            startup.subprocess.run = original_run
            startup.urllib.request.urlopen = original_urlopen

        self.assertEqual(loaded, {"_llama_brain": {"size_vram": 0, "status": "loading"}})

    def test_model_search_roots_prefer_explicit_local_hotset(self):
        roots = resolve_model_search_roots(
            {
                "model_search_roots": [
                    "/home/bryan/model-hotset",
                    "/mnt/shared/models",
                ]
            }
        )

        self.assertEqual(
            [str(root) for root in roots],
            ["/home/bryan/model-hotset", "/mnt/shared/models"],
        )

    def test_preloaded_llama_brain_uses_full_ready_budget(self):
        self.assertEqual(
            startup.brain_ready_wait_budget(True, "llama"),
            startup.BRAIN_MAX_WAIT,
        )

    def test_brain_run_reuses_ready_preloaded_runtime(self):
        flag = Path("/tmp/llm-orchestration-flags/brain.ready")
        if flag.exists():
            flag.unlink()

        class StubBrain:
            def __init__(self):
                self.name = "brain"
                self.model = "qwen2.5:32b"
                self.gpus = [0]
                self.config = {"runtime_backend": "llama"}
                self.logger = MagicMock()
                self.running = False
                self.gpu_agents = {}
                self.active_batches = {}
                self.start_runtime_calls = 0
                self.heartbeat_writes = 0

            def _brain_runtime_is_ready(self, timeout_seconds: int = 5) -> bool:
                return True

            def start_runtime(self):
                self.start_runtime_calls += 1

            def stop_runtime(self):
                return None

            def think(self, _prompt):
                raise AssertionError("think should not be called for preloaded runtime")

            def _write_brain_heartbeat(self):
                self.heartbeat_writes += 1

            def _save_brain_state(self):
                return None

        stub = StubBrain()
        brain.Brain.run(stub, model_preloaded=True)

        self.assertEqual(stub.start_runtime_calls, 0)
        self.assertGreaterEqual(stub.heartbeat_writes, 1)
        self.assertTrue(flag.exists())
        flag.unlink()


if __name__ == "__main__":
    unittest.main()
