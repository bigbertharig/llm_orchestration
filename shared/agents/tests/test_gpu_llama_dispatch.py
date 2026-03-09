#!/usr/bin/env python3
"""Regression tests for llama runtime method dispatch."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gpu import GPUAgent
from gpu_llama import GPULlamaMixin


class DispatchStub(GPULlamaMixin):
    def __init__(self):
        self.calls = []

    def _llama_load_model(self, model_id=None, task_id=None):
        self.calls.append(("load", model_id, task_id))
        return "loaded"

    def _llama_unload_model(self, model_id=None, task_id=None):
        self.calls.append(("unload", model_id, task_id))
        return "unloaded"

    def _llama_wait_for_model_ready(self, model_id, max_wait_seconds=90):
        self.calls.append(("ready", model_id, max_wait_seconds))
        return True

    def _llama_wait_for_model_unloaded(self, model_id, max_wait_seconds=30):
        self.calls.append(("unloaded_wait", model_id, max_wait_seconds))
        return True


class LlamaDispatchTests(unittest.TestCase):
    def test_llama_mixin_exposes_active_loader_surface(self):
        stub = DispatchStub()

        self.assertEqual(stub.load_model(model_id="qwen2.5:7b", task_id="t1"), "loaded")
        self.assertEqual(stub.unload_model(model_id="qwen2.5:7b", task_id="t2"), "unloaded")
        self.assertTrue(stub._wait_for_model_ready("qwen2.5:7b", 11))
        self.assertTrue(stub._wait_for_model_unloaded("qwen2.5:7b", 7))

        self.assertEqual(
            stub.calls,
            [
                ("load", "qwen2.5:7b", "t1"),
                ("unload", "qwen2.5:7b", "t2"),
                ("ready", "qwen2.5:7b", 11),
                ("unloaded_wait", "qwen2.5:7b", 7),
            ],
        )

    def test_gpu_agent_prefers_llama_loader_over_legacy_runtime_mixin(self):
        self.assertIs(GPUAgent.load_model, GPULlamaMixin.load_model)
        self.assertIs(GPUAgent.unload_model, GPULlamaMixin.unload_model)


if __name__ == "__main__":
    unittest.main()
