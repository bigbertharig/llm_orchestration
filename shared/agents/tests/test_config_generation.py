#!/usr/bin/env python3
"""Tests for explicit neutral and warm startup config generation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from setup import build_config


def _assignment(initial_hot_workers: int) -> dict:
    return {
        "_discovery_mode": "standard",
        "brain": {"model": "brain:test", "gpus": [0]},
        "gpus": [
            {"name": "gpu-1", "id": 1, "vram_mb": 6144, "model": "worker:test", "port": 11435},
            {"name": "gpu-2", "id": 2, "vram_mb": 6144, "model": "worker:test", "port": 11436},
            {"name": "gpu-3", "id": 3, "vram_mb": 6144, "model": "worker:test", "port": 11437},
        ],
        "worker_mode": "cold",
        "initial_hot_workers": initial_hot_workers,
    }


class ConfigGenerationTests(unittest.TestCase):
    def test_default_generation_is_explicitly_neutral(self):
        config = build_config(
            _assignment(0),
            {"host": "http://localhost:11434"},
            {"hostname": "test-rig"},
        )

        self.assertEqual(config["llama_runtime_image"], "llama-runtime:b10333")
        self.assertEqual(config["model_catalog_path"], "models.catalog.json")
        self.assertFalse(config["auto_default_enabled"])
        self.assertEqual(config["initial_hot_workers"], 0)
        self.assertEqual(config["startup_warm_workers"], [])
        self.assertEqual(config["startup_meta_tasks"], [])
        self.assertEqual(config["max_hot_workers"], 3)

    def test_warm_generation_prefers_gpu_2_and_matches_requested_count(self):
        config = build_config(
            _assignment(2),
            {"host": "http://localhost:11434"},
            {"hostname": "test-rig"},
        )

        self.assertTrue(config["auto_default_enabled"])
        self.assertEqual(config["startup_warm_workers"], ["gpu-2", "gpu-1"])
        self.assertEqual(
            [task["candidate_workers"] for task in config["startup_meta_tasks"]],
            [["gpu-2"], ["gpu-1"]],
        )


if __name__ == "__main__":
    unittest.main()
