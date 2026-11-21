"""Rig-owned fit: library status, validated Strata layout, and live card choice."""

from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fit
import model_fit
import observations
import rig
import schedule_preview


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
    def test_rig_backend_timestamps_are_explicit_utc(self):
        self.assertTrue(rig.now_iso().endswith("+00:00"))

    def test_schedule_preview_exposes_revisioned_orchestration_alternatives(self):
        shape = fit.Shape("m", "llama", 16384, 8, [1024**3], [1024.0],
                          [None], [0], 0, 0)
        history = [{"kind": "load", "model": "m", "source": "model.gguf",
                    "gpus": [1], "seconds": 12.0}]
        with patch.object(fit, "shape_for", return_value=shape):
            result = schedule_preview.build_preview(
                {"workloads": [{"id": "model_m", "model": "m", "use": "benchmark"}]},
                library=Library(), cards=CARDS, free=CARDS[1:], unavailable={0: "leased"},
                host_available=8192, loads_admitted=True, guard_level="normal",
                boot_id="boot-a", plan_version=7, mode="bench",
                observed_at="2026-11-21T00:00:00+00:00", load_history=history)
        self.assertEqual(result["schema"], "rig-scheduling-preview/1")
        self.assertEqual(result["constraints"]["hardware_authority"], "orchestration")
        self.assertTrue(result["admission_required_at_execution"])
        self.assertTrue(result["revision"].startswith("sha256:"))
        alternatives = result["workloads"][0]["alternatives"]
        solo = next(row for row in alternatives if row["gpus"] == [1])
        self.assertTrue(solo["currently_available"])
        self.assertEqual((solo["estimated_load_seconds"], solo["load_estimate_basis"]),
                         (12.0, "measured_layout"))
        self.assertFalse(next(row for row in alternatives if row["gpus"] == [0])["currently_available"])

    def test_schedule_preview_revision_changes_with_hardware_state(self):
        shape = fit.Shape("m", "llama", 16384, 8, [1024**3], [1024.0],
                          [None], [0], 0, 0)
        common = dict(library=Library(), cards=CARDS, host_available=8192,
                      loads_admitted=True, guard_level="normal",
                      boot_id="boot-a", plan_version=7, mode="bench",
                      observed_at="2026-11-21T00:00:00+00:00", load_history=[])
        request = {"workloads": [{"id": "m", "model": "m"}]}
        with patch.object(fit, "shape_for", return_value=shape):
            first = schedule_preview.build_preview(request, free=CARDS, unavailable={}, **common)
            second = schedule_preview.build_preview(request, free=CARDS[1:], unavailable={0: "leased"}, **common)
        self.assertNotEqual(first["revision"], second["revision"])

    def test_imported_strata_recipes_keep_tested_gpu_layout(self):
        library = rig.orchestration_model_library
        for model in ("qwen3.6:35b-a3b-xl", "qwen3.8-flash-next:q2_0"):
            result = model_fit.preview({"model": model, "use": "benchmark", "engine": "strata",
                                        "ctx": 8192}, library=library, cards=CARDS, free=CARDS,
                                       unavailable={}, host_available=18000)
            self.assertEqual(result["gpus"], [0])
            self.assertEqual(result["strata"]["validated_gpu_ids"], [0])

    def test_fit_observation_records_rig_local_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(observations, "PATH", str(Path(tmp) / "loads.jsonl")):
                code = rig.cmd_fit_observation(SimpleNamespace(row=json.dumps({
                    "kind": "load", "source": "model.gguf", "model": "m", "n_ctx": 8192})))
                self.assertEqual(code, 0)
                self.assertEqual(list(observations.rows())[0]["n_ctx"], 8192)

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
        with self.assertRaisesRegex(model_fit.ModelFitError, "no capacity"):
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

    def test_excluded_card_is_not_chosen_for_alternative_fit(self):
        shape = fit.Shape("m", "llama", 16384, 8, [1024**3], [1024.0],
                          [None], [0], 0, 0)
        with patch.object(fit, "shape_for", return_value=shape):
            result = model_fit.preview({"model": "m", "exclude_gpu_ids": [1]},
                                       library=Library(), cards=CARDS, free=CARDS,
                                       unavailable={}, host_available=8192)
        self.assertEqual(result["gpus"], [0])


if __name__ == "__main__":
    unittest.main()
