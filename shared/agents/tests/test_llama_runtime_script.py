#!/usr/bin/env python3
"""Regression tests for llama runtime shell wrapper GPU argument handling."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path("/home/bryan/llm_orchestration/scripts/llama_runtime/run_runtime.sh")


class LlamaRuntimeScriptTests(unittest.TestCase):
    def _dry_run(self, gpus: str) -> str:
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


if __name__ == "__main__":
    unittest.main()
