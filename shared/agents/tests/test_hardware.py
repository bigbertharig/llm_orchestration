#!/usr/bin/env python3
"""
Tests for hardware.py — assignment algorithm with various GPU topologies.

Run:
  python -m pytest tests/test_hardware.py -v
  python tests/test_hardware.py  # standalone
"""

import sys
import unittest
from pathlib import Path

# Allow running from agents/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from hardware import suggest_assignment, _find_best_model, scan_system


# Helper to build fake GPU dicts
def gpu(index, name="NVIDIA GTX 1060", vram_mb=6144):
    return {
        "index": index, "name": name, "vram_mb": vram_mb,
        "temp_c": 40, "power_w": 80.0, "power_limit_w": 140.0,
        "clock_mhz": 1800, "throttle": "None",
    }


# Helper to build fake Ollama status
def ollama(models=None):
    if models is None:
        models = []
    return {
        "running": True,
        "host": "http://localhost:11434",
        "available_models": models,
        "loaded_models": [],
    }


def model(name, size_mb):
    return {"name": name, "size_mb": size_mb}


class TestSuggestAssignment(unittest.TestCase):
    """Test the suggest_assignment function with various topologies."""

    # --- 0 GPUs ---

    def test_zero_gpus_cpu_only_mode(self):
        result = suggest_assignment([], ollama())
        self.assertEqual(result["_discovery_mode"], "cpu_only")
        self.assertEqual(result["brain"]["gpus"], [])
        self.assertEqual(result["gpus"], [])

    def test_zero_gpus_with_models(self):
        models = [model("qwen2.5:7b", 4400)]
        result = suggest_assignment([], ollama(models))
        self.assertEqual(result["_discovery_mode"], "cpu_only")
        # No GPU = 0 VRAM = no model fits
        self.assertEqual(result["brain"]["model"], "")

    # --- 1 GPU ---

    def test_one_gpu_single_mode(self):
        gpus = [gpu(0, "NVIDIA RTX 3080", 10240)]
        models = [model("qwen2.5:14b", 8200), model("qwen2.5:7b", 4400)]
        result = suggest_assignment(gpus, ollama(models))

        self.assertEqual(result["_discovery_mode"], "single_gpu")
        self.assertEqual(result["brain"]["gpus"], [0])
        self.assertEqual(result["brain"]["model"], "qwen2.5:14b")
        self.assertEqual(result["gpus"], [])

    def test_one_gpu_small_vram(self):
        gpus = [gpu(0, "NVIDIA GTX 1060", 6144)]
        models = [model("qwen2.5:14b", 8200), model("qwen2.5:7b", 4400)]
        result = suggest_assignment(gpus, ollama(models))

        # 14b doesn't fit (8200 > 6144*0.85=5222), 7b does
        self.assertEqual(result["brain"]["model"], "qwen2.5:7b")

    # --- 2 GPUs ---

    def test_two_gpus_minimal_mode(self):
        gpus = [gpu(0, "NVIDIA RTX 3080", 10240), gpu(1, "NVIDIA GTX 1060", 6144)]
        models = [model("qwen2.5:14b", 8200), model("qwen2.5:7b", 4400)]
        result = suggest_assignment(gpus, ollama(models))

        self.assertEqual(result["_discovery_mode"], "minimal")
        self.assertEqual(result["brain"]["gpus"], [0])  # Bigger GPU
        self.assertEqual(result["brain"]["model"], "qwen2.5:14b")
        self.assertEqual(len(result["gpus"]), 1)
        self.assertEqual(result["gpus"][0]["id"], 1)
        self.assertEqual(result["gpus"][0]["model"], "qwen2.5:7b")

    def test_two_identical_gpus(self):
        gpus = [gpu(0, "NVIDIA GTX 1060", 6144), gpu(1, "NVIDIA GTX 1060", 6144)]
        models = [model("qwen2.5:7b", 4400)]
        result = suggest_assignment(gpus, ollama(models))

        self.assertEqual(result["_discovery_mode"], "minimal")
        # Biggest (first by index since tied) becomes brain
        self.assertEqual(result["brain"]["gpus"], [0])
        self.assertEqual(len(result["gpus"]), 1)

    # --- 3 GPUs ---

    def test_three_gpus_standard_mode(self):
        gpus = [
            gpu(0, "NVIDIA RTX 3080", 10240),
            gpu(1, "NVIDIA GTX 1060", 6144),
            gpu(2, "NVIDIA GTX 1060", 6144),
        ]
        models = [model("qwen2.5:14b", 8200), model("qwen2.5:7b", 4400)]
        result = suggest_assignment(gpus, ollama(models))

        self.assertEqual(result["_discovery_mode"], "standard")
        # Single largest GPU for brain (no identical pair at max VRAM)
        self.assertEqual(result["brain"]["gpus"], [0])
        self.assertEqual(result["brain"]["model"], "qwen2.5:14b")
        self.assertEqual(len(result["gpus"]), 2)

    def test_three_identical_gpus_pairs_brain(self):
        gpus = [
            gpu(0, "NVIDIA GTX 1060", 6144),
            gpu(1, "NVIDIA GTX 1060", 6144),
            gpu(2, "NVIDIA GTX 1060", 6144),
        ]
        models = [model("qwen2.5:7b", 4400)]
        result = suggest_assignment(gpus, ollama(models))

        self.assertEqual(result["_discovery_mode"], "standard")
        # First two identical top-VRAM GPUs paired for brain
        self.assertEqual(result["brain"]["gpus"], [0, 1])
        self.assertEqual(len(result["gpus"]), 1)
        self.assertEqual(result["gpus"][0]["id"], 2)

    # --- 5 GPUs (original rig) ---

    def test_five_identical_gpus(self):
        gpus = [gpu(i, "NVIDIA GTX 1060", 6144) for i in range(5)]
        models = [model("qwen2.5:14b", 8200), model("qwen2.5:7b", 4400)]
        result = suggest_assignment(gpus, ollama(models))

        self.assertEqual(result["_discovery_mode"], "standard")
        # Two paired for brain (identical top-VRAM)
        self.assertEqual(len(result["brain"]["gpus"]), 2)
        self.assertEqual(len(result["gpus"]), 3)

    def test_five_mixed_gpus(self):
        gpus = [
            gpu(0, "NVIDIA RTX 3080", 10240),
            gpu(1, "NVIDIA GTX 1060", 6144),
            gpu(2, "NVIDIA GTX 1060", 6144),
            gpu(3, "NVIDIA RTX 3080", 10240),
            gpu(4, "NVIDIA GTX 1060", 6144),
        ]
        models = [model("qwen2.5:14b", 8200), model("qwen2.5:7b", 4400)]
        result = suggest_assignment(gpus, ollama(models))

        # Two RTX 3080s should be paired for brain
        self.assertEqual(sorted(result["brain"]["gpus"]), [0, 3])
        self.assertEqual(len(result["gpus"]), 3)
        worker_ids = [w["id"] for w in result["gpus"]]
        self.assertEqual(sorted(worker_ids), [1, 2, 4])

    # --- 8 GPUs ---

    def test_eight_gpus(self):
        gpus = [gpu(i, "NVIDIA A100", 81920) for i in range(8)]
        models = [model("llama3:70b", 40000), model("llama3:8b", 4700)]
        result = suggest_assignment(gpus, ollama(models))

        self.assertEqual(result["_discovery_mode"], "standard")
        # Two paired for brain
        self.assertEqual(len(result["brain"]["gpus"]), 2)
        self.assertEqual(len(result["gpus"]), 6)
        self.assertEqual(result["brain"]["model"], "llama3:70b")

    # --- Edge cases ---

    def test_no_ollama(self):
        gpus = [gpu(0, "NVIDIA RTX 3080", 10240)]
        no_ollama = {"running": False, "host": "http://localhost:11434",
                     "available_models": [], "loaded_models": []}
        result = suggest_assignment(gpus, no_ollama)

        self.assertEqual(result["brain"]["model"], "")
        self.assertEqual(result["brain"]["gpus"], [0])

    def test_no_models_fit(self):
        gpus = [gpu(0, "NVIDIA GTX 750", 2048)]
        models = [model("qwen2.5:14b", 8200), model("qwen2.5:7b", 4400)]
        result = suggest_assignment(gpus, ollama(models))

        # Nothing fits in 2048*0.85=1740 MB
        self.assertEqual(result["brain"]["model"], "")

    def test_preferences_override(self):
        gpus = [gpu(0, "NVIDIA RTX 3080", 10240), gpu(1, "NVIDIA GTX 1060", 6144)]
        models = [model("qwen2.5:14b", 8200), model("qwen2.5:7b", 4400)]
        prefs = {"brain_model": "qwen2.5:7b", "worker_model": "qwen2.5:7b", "worker_mode": "cold"}
        result = suggest_assignment(gpus, ollama(models), prefs)

        self.assertEqual(result["brain"]["model"], "qwen2.5:7b")
        self.assertEqual(result["gpus"][0]["model"], "qwen2.5:7b")
        self.assertEqual(result["worker_mode"], "cold")

    def test_port_auto_assignment(self):
        gpus = [gpu(i) for i in range(5)]
        result = suggest_assignment(gpus, ollama())

        ports = [w["port"] for w in result["gpus"]]
        expected = [11435 + i for i in range(len(result["gpus"]))]
        self.assertEqual(ports, expected)

    def test_worker_names_match_gpu_index(self):
        gpus = [gpu(0, vram_mb=10240), gpu(2), gpu(5)]
        result = suggest_assignment(gpus, ollama())

        for w in result["gpus"]:
            self.assertEqual(w["name"], "gpu-%d" % w['id'])


class TestFindBestModel(unittest.TestCase):
    """Test the model selection logic."""

    def test_picks_largest_fitting(self):
        models = [model("small", 2000), model("medium", 4000), model("large", 8000)]
        self.assertEqual(_find_best_model(models, 6000), "medium")  # 6000*0.85=5100

    def test_empty_models(self):
        self.assertIsNone(_find_best_model([], 10000))

    def test_nothing_fits(self):
        models = [model("big", 10000)]
        self.assertIsNone(_find_best_model(models, 5000))

    def test_exact_fit(self):
        models = [model("exact", 4250)]
        # 5000 * 0.85 = 4250 -- exact fit
        self.assertEqual(_find_best_model(models, 5000), "exact")


class TestScanSystem(unittest.TestCase):
    """Test system scanning (runs on any machine)."""

    def test_returns_expected_keys(self):
        result = scan_system()
        self.assertIn("hostname", result)
        self.assertIn("cpu_cores", result)
        self.assertIn("ram_total_mb", result)
        self.assertIn("ram_available_mb", result)
        self.assertGreater(result["cpu_cores"], 0)


if __name__ == "__main__":
    unittest.main()
