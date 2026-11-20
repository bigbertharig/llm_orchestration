"""Rig-owned fit: library status, validated Strata layout, and live card choice."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fit
import model_fit


class Library:
    def __init__(self, status="active", strata=None):
        self.entry = {
            "gguf": "model.gguf", "status": status, "profiles": {
                "default": {"runtime": {"ctx_size": 8192, "batch_size": 64},
                            "context_bounds": {"min": 4096, "max": 16384}}},
            "model_size_bytes": 3 * 1024**3,
        }
        if strata:
            self.entry["strata"] = strata

    def load_snapshot(self):
        return {"source_sha256": "revision", "registry": {
            "models": {"m": self.entry}, "runtime_image": "llama-test"}}

    def resolve(self, model, profile, registry):
        entry = registry["models"][model]
        return {"model": model, "profile": profile, "status": entry["status"],
                "gguf_path": "/fake/model.gguf", **entry["profiles"][profile]}


CARDS = [{"index": 0, "total_mb": 24576, "used_mb": 0},
         {"index": 1, "total_mb": 6144, "used_mb": 0}]


class ModelFitTests(unittest.TestCase):
    def test_candidate_is_available_for_benchmark_only(self):
        lib = Library(status="candidate", strata={"only": True, "validated_gpu_ids": [0],
                                                    "max_context": 8192, "min_host_available_mb": 1024})
        with self.assertRaisesRegex(model_fit.ModelFitError, "not approved"):
            model_fit.preview({"model": "m", "use": "chat"}, library=lib,
                              cards=CARDS, free=CARDS, unavailable={}, host_available=8192)
        result = model_fit.preview({"model": "m", "use": "benchmark"}, library=lib,
                                   cards=CARDS, free=CARDS, unavailable={}, host_available=8192)
        self.assertEqual(result["gpus"], [0])
        self.assertEqual(result["source_sha256"], "revision")

    def test_strata_rejects_unvalidated_or_busy_card(self):
        lib = Library(strata={"only": True, "validated_gpu_ids": [0],
                              "max_context": 8192, "min_host_available_mb": 1024})
        with self.assertRaisesRegex(model_fit.ModelFitError, "validated Strata placement"):
            model_fit.preview({"model": "m", "gpu_ids": [1]}, library=lib,
                              cards=CARDS, free=CARDS, unavailable={}, host_available=8192)
        with self.assertRaisesRegex(model_fit.ModelFitError, "unavailable"):
            model_fit.preview({"model": "m"}, library=lib, cards=CARDS,
                              free=CARDS[1:], unavailable={0: "leased"}, host_available=8192)

    def test_ordinary_fit_uses_live_free_cards(self):
        shape = fit.Shape("m", "llama", 16384, 8, [1024**3], [1024.0],
                          [None], [0], 0, 0)
        with patch.object(fit, "shape_for", return_value=shape):
            result = model_fit.preview({"model": "m", "use": "chat"}, library=Library(),
                                       cards=CARDS, free=CARDS[1:], unavailable={0: "leased"},
                                       host_available=8192)
        self.assertTrue(result["fits"])
        self.assertEqual(result["gpus"], [1])


if __name__ == "__main__":
    unittest.main()
