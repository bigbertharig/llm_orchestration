#!/usr/bin/env python3
"""Tests for shared runtime default target inference."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from brain_core import resolve_auto_default_target


class RuntimeDefaultTargetTests(unittest.TestCase):
    def test_prefers_explicit_auto_default_settings(self):
        gpu, model = resolve_auto_default_target(
            {
                "auto_default_gpu": "gpu-4",
                "auto_default_model": "custom:model",
                "gpus": [
                    {"id": 1, "name": "gpu-1", "model": "qwen2.5:7b"},
                    {"id": 4, "name": "gpu-4", "model": "qwen2.5-coder:14b"},
                ],
            }
        )

        self.assertEqual(gpu, "gpu-4")
        self.assertEqual(model, "custom:model")

    def test_derives_model_from_selected_default_gpu(self):
        gpu, model = resolve_auto_default_target(
            {
                "auto_default_gpu": "gpu-3",
                "gpus": [
                    {"id": 2, "name": "gpu-2", "model": "qwen2.5:7b"},
                    {"id": 3, "name": "gpu-3", "model": "qwen2.5-coder:14b"},
                ],
            }
        )

        self.assertEqual(gpu, "gpu-3")
        self.assertEqual(model, "qwen2.5-coder:14b")

    def test_defaults_to_gpu_2_when_present(self):
        gpu, model = resolve_auto_default_target(
            {
                "gpus": [
                    {"id": 1, "name": "gpu-1", "model": "model-a"},
                    {"id": 2, "name": "gpu-2", "model": "model-b"},
                ]
            }
        )

        self.assertEqual(gpu, "gpu-2")
        self.assertEqual(model, "model-b")

    def test_falls_back_to_first_gpu_when_gpu_2_missing(self):
        gpu, model = resolve_auto_default_target(
            {
                "gpus": [
                    {"id": 4, "name": "gpu-4", "model": "model-x"},
                    {"id": 5, "name": "gpu-5", "model": "model-y"},
                ]
            }
        )

        self.assertEqual(gpu, "gpu-4")
        self.assertEqual(model, "model-x")


if __name__ == "__main__":
    unittest.main()
