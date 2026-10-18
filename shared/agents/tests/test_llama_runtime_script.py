#!/usr/bin/env python3
"""Regression tests for llama runtime shell wrapper GPU argument handling."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "llama_runtime" / "run_runtime.sh"


class LlamaRuntimeScriptTests(unittest.TestCase):
    def _dry_run(self, gpus: str, *extra: str) -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "model.gguf"
            model.write_text("stub", encoding="utf-8")
            proc = subprocess.run(
                [
                    "bash",
                    str(SCRIPT),
                    "--name",
                    "test-runtime",
                    "--model",
                    str(model),
                    "--port",
                    "11436",
                    "--gpus",
                    gpus,
                    "--dry-run",
                    *extra,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            return proc.stdout.strip()

    def test_dry_run_quotes_multi_gpu_device_request_for_docker(self):
        out = self._dry_run("device=1,3")
        self.assertIn('--gpus \\"device=1\\,3\\"', out)

    def test_dry_run_keeps_single_gpu_device_request_unquoted(self):
        out = self._dry_run("device=2")
        self.assertIn("--gpus device=2", out)

    def test_dry_run_uses_the_model_library_image(self):
        library = SCRIPT.parents[2] / "plans" / "shoulders" / "benchmarking" / "scripts" / "active" / "model_library.py"
        if not library.is_file():
            self.skipTest("benchmarking model library not checked out")
        expected = subprocess.run(["python3", str(library), "runtime-image"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        out = self._dry_run("device=2")
        self.assertIn(expected, out)

    def test_host_prompt_cache_is_off_by_default(self):
        self.assertIn("--cache-ram 0", self._dry_run("device=2"))

    def test_caller_sets_the_host_prompt_cache(self):
        out = self._dry_run("device=2", "--cache-ram", "2048")
        self.assertIn("--cache-ram 2048", out)
        self.assertNotIn("--cache-ram 0", out)

    def test_cache_in_extra_args_wins_over_the_floor(self):
        out = self._dry_run("device=2", "--extra-arg", "--cache-ram", "--extra-arg", "512")
        self.assertEqual(out.count("--cache-ram"), 1)
        self.assertIn("--cache-ram 512", out)


if __name__ == "__main__":
    unittest.main()
