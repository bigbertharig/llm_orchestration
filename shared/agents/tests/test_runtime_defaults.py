#!/usr/bin/env python3
"""Tests for shared runtime default target inference."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from brain_core import (
    resolve_llama_runtime_profile,
    resolve_auto_default_target,
    resolve_runtime_base_url,
    resolve_runtime_chat_endpoint,
)


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

    def test_runtime_base_url_prefers_backend_neutral_key(self):
        runtime_base = resolve_runtime_base_url(
            {
                "runtime_host": "http://127.0.0.1:18080/",
            }
        )

        self.assertEqual(runtime_base, "http://127.0.0.1:18080")

    def test_runtime_base_url_defaults_when_runtime_host_missing(self):
        runtime_base = resolve_runtime_base_url({})

        self.assertEqual(runtime_base, "http://localhost:11434")

    def test_runtime_chat_endpoint_uses_llama_route(self):
        endpoint = resolve_runtime_chat_endpoint(
            {
                "runtime_backend": "llama",
                "runtime_host": "http://127.0.0.1:11434",
            }
        )

        self.assertEqual(endpoint, "http://127.0.0.1:11434/v1/chat/completions")

    def test_runtime_chat_endpoint_rejects_non_llama_backend(self):
        with self.assertRaises(ValueError):
            resolve_runtime_chat_endpoint(
                {
                    "runtime_backend": "legacy",
                    "runtime_host": "http://127.0.0.1:11434",
                }
            )

    def test_single_llama_profile_merges_defaults_model_and_gpu_override(self):
        profile = resolve_llama_runtime_profile(
            {
                "llama_single_defaults": {
                    "ctx_size": 4096,
                    "batch_size": 128,
                    "parallel": 1,
                    "n_gpu_layers": 999,
                },
                "llama_single_profiles": {
                    "qwen2.5:7b": {
                        "batch_size": 96,
                    }
                },
            },
            model_id="qwen2.5:7b",
            gpu_config={
                "llama_runtime": {
                    "batch_size": 64,
                    "threads": 6,
                }
            },
        )

        self.assertEqual(
            profile,
            {
                "ctx_size": 4096,
                "batch_size": 64,
                "parallel": 1,
                "n_gpu_layers": 999,
                "threads": 6,
            },
        )

    def test_split_llama_profile_uses_model_override(self):
        profile = resolve_llama_runtime_profile(
            {
                "llama_split_defaults": {
                    "ctx_size": 4096,
                    "batch_size": 128,
                    "n_gpu_layers": 999,
                },
                "llama_split_profiles": {
                    "qwen2.5-coder:14b": {
                        "batch_size": 96,
                        "parallel": 1,
                        "tensor_split": "1,1",
                        "extra_args": ["--no-warmup"],
                        "meta_timeout_seconds": 600,
                    }
                },
            },
            model_id="qwen2.5-coder:14b",
            split=True,
        )

        self.assertEqual(
            profile,
            {
                "ctx_size": 4096,
                "batch_size": 96,
                "n_gpu_layers": 999,
                "parallel": 1,
                "tensor_split": "1,1",
                "extra_args": ["--no-warmup"],
                "meta_timeout_seconds": 600,
            },
        )


if __name__ == "__main__":
    unittest.main()
