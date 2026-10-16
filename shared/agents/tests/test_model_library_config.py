#!/usr/bin/env python3
"""apply_model_library fills model choices and llama settings from the benchmarking library."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

AGENTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENTS))

import brain_core  # noqa: E402

LIBRARY_PRESENT = (brain_core.MODEL_LIBRARY_DIR / "model_library.py").is_file()


@unittest.skipUnless(LIBRARY_PRESENT, "benchmarking model library not checked out beside the agents")
class ApplyModelLibraryTests(unittest.TestCase):
    def setUp(self):
        self.template = json.loads((AGENTS / "config.template.json").read_text(encoding="utf-8"))
        library = brain_core.load_model_library()
        self.roles = library.load_roles()["roles"]
        self.registry = library.load_registry()

    def test_roles_fill_brain_and_worker_models(self):
        config = brain_core.apply_model_library(copy.deepcopy(self.template))
        self.assertEqual(config["brain"]["model"], self.roles["brain"]["model"])
        worker = self.roles["single_1060"]["model"]
        self.assertTrue(all(gpu["model"] == worker for gpu in config["gpus"]))
        self.assertEqual(config["auto_default_model"], worker)

    def test_library_overrides_stale_config_values(self):
        stale = copy.deepcopy(self.template)
        stale["brain"]["model"] = "qwen2.5:32b"
        stale["llama_single_profiles"] = {"qwen2.5:32b": {"ctx_size": 1}}
        config = brain_core.apply_model_library(stale)
        self.assertEqual(config["brain"]["model"], self.roles["brain"]["model"])
        self.assertNotIn("qwen2.5:32b", config["llama_single_profiles"])

    def test_orchestration_profiles_build_llama_tables(self):
        config = brain_core.apply_model_library(copy.deepcopy(self.template))
        brain = self.roles["brain"]["model"]
        expected = self.registry["models"][brain]["profiles"]["orchestration"]["runtime"]["ctx_size"]
        profile = brain_core.resolve_llama_runtime_profile(config, model_id=brain)
        self.assertEqual(profile["ctx_size"], expected)
        self.assertEqual(config["llama_split_defaults"], self.registry["orchestration_defaults"]["split"])
        split_models = {mid for mid, e in self.registry["models"].items()
                        if "orchestration" in e["profiles"] and e.get("schedule", {}).get("placement") == "split_gpu"}
        self.assertTrue(split_models <= set(config["llama_split_profiles"]))

    def test_warm_start_tasks_get_the_worker_model(self):
        config = copy.deepcopy(self.template)
        config["startup_meta_tasks"] = [{"name": "warm", "command": "load_llm", "candidate_workers": ["gpu-2"]}]
        config = brain_core.apply_model_library(config)
        self.assertEqual(config["startup_meta_tasks"][0]["target_model"], self.roles["single_1060"]["model"])

    def test_missing_library_fails_loudly(self):
        with mock.patch.object(brain_core, "load_model_library", side_effect=ImportError("no library")):
            with self.assertRaises(ImportError):
                brain_core.apply_model_library(copy.deepcopy(self.template))


if __name__ == "__main__":
    unittest.main()
