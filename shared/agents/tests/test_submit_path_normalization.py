#!/usr/bin/env python3
"""
Tests for submit path normalization and preflight path validation.
"""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_SUBMIT = REPO_ROOT / "scripts" / "submit.py"
AGENT_SUBMIT = REPO_ROOT / "shared" / "agents" / "submit.py"


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSubmitPathNormalization(unittest.TestCase):
    def test_normalize_config_paths_for_rig_nested(self):
        scripts_submit = _load_module("scripts_submit", SCRIPTS_SUBMIT)
        cfg = {
            "QUERY_FILE": "/home/bryan/llm_orchestration/shared/plans/a.md",
            "NESTED": {
                "FILES": [
                    "/media/bryan/shared/one.md",
                    "/tmp/no_change.txt",
                ]
            },
        }
        out = scripts_submit._normalize_config_paths_for_rig(cfg)
        self.assertEqual(out["QUERY_FILE"], "/mnt/shared/plans/a.md")
        self.assertEqual(out["NESTED"]["FILES"][0], "/mnt/shared/one.md")
        self.assertEqual(out["NESTED"]["FILES"][1], "/tmp/no_change.txt")

    def test_agent_normalize_config_paths_nested(self):
        agent_submit = _load_module("agent_submit", AGENT_SUBMIT)
        cfg = {
            "QUERY_FILE": "/media/bryan/shared/plans/a.md",
            "NESTED": {
                "FILES": [
                    "/home/bryan/llm_orchestration/shared/two.md",
                ]
            },
        }
        out = agent_submit._normalize_config_paths(cfg)
        self.assertEqual(out["QUERY_FILE"], "/mnt/shared/plans/a.md")
        self.assertEqual(out["NESTED"]["FILES"][0], "/mnt/shared/two.md")

    def test_validate_submission_paths_checks_aliases(self):
        scripts_submit = _load_module("scripts_submit_aliases", SCRIPTS_SUBMIT)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            alias_a = root / "a"
            alias_b = root / "b"
            (alias_a / "plans").mkdir(parents=True)
            target = alias_a / "plans" / "query.md"
            target.write_text("ok\n", encoding="utf-8")

            original_aliases = scripts_submit.SHARED_ALIASES
            scripts_submit.SHARED_ALIASES = (str(alias_a), str(alias_b))
            try:
                cfg = {"QUERY_FILE": str(alias_b / "plans" / "query.md")}
                errors = scripts_submit._validate_submission_config_paths(cfg)
                self.assertEqual(
                    errors,
                    [],
                    msg="expected alias-based path resolution to find existing QUERY_FILE",
                )
            finally:
                scripts_submit.SHARED_ALIASES = original_aliases


class TestSubmitPathValidationCLI(unittest.TestCase):
    def test_invalid_query_file_produces_clean_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            plan_root = Path(temp_dir) / "plan"
            (plan_root / "scripts").mkdir(parents=True)
            (plan_root / "plan.md").write_text(
                "## Tasks\n\n### t1\n- **command**: `echo ok`\n",
                encoding="utf-8",
            )

            cfg = {"QUERY_FILE": "/home/bryan/llm_orchestration/shared/does/not/exist.md"}
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS_SUBMIT),
                    str(plan_root),
                    "--config",
                    json.dumps(cfg),
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
            )

            self.assertEqual(proc.returncode, 1, msg=proc.stdout + proc.stderr)
            self.assertIn("Error: invalid config path values in --config.", proc.stdout)
            self.assertIn("QUERY_FILE", proc.stdout)
            self.assertIn("file not found on any known shared-path alias", proc.stdout)


if __name__ == "__main__":
    unittest.main()
