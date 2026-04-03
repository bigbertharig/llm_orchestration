#!/usr/bin/env python3
"""Regression tests for plan variable substitution in task metadata."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from brain_plan import BrainPlanMixin


class _Logger:
    def warning(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None


class MockBrainPlan(BrainPlanMixin):
    def __init__(self, root: Path):
        self.name = "brain"
        self.logger = _Logger()
        self.default_llm_min_tier = 1
        self.model_tier_by_id = {"qwen2.5-coder:7b": 1}
        self.model_meta_by_id = {"qwen2.5-coder:7b": {"placement": "single_gpu"}}
        self.active_batches = {}
        self.private_tasks_path = root / "brain" / "private_tasks"
        self.public_tasks_path = root / "tasks" / "queue"
        self.failed_tasks_path = root / "tasks" / "failed"
        self.private_tasks_path.mkdir(parents=True, exist_ok=True)
        self.public_tasks_path.mkdir(parents=True, exist_ok=True)
        self.failed_tasks_path.mkdir(parents=True, exist_ok=True)

    def log_decision(self, *_args, **_kwargs):
        return None

    def _cleanup_stale_plan_batches(self, _plan_dir: Path):
        return {"stale_batches": 0}

    def _parse_goal_section(self, _plan_content: str):
        return None

    def _save_brain_state(self):
        return None

    def save_to_private(self, task):
        path = self.private_tasks_path / f"{task['task_id']}.json"
        path.write_text(json.dumps(task), encoding="utf-8")

    def save_to_public(self, task):
        path = self.public_tasks_path / f"{task['task_id']}.json"
        path.write_text(json.dumps(task), encoding="utf-8")

    def save_to_failed(self, task):
        path = self.failed_tasks_path / f"{task['task_id']}.json"
        path.write_text(json.dumps(task), encoding="utf-8")


class BrainPlanVariableSubstitutionTests(unittest.TestCase):
    def test_execute_plan_substitutes_llm_model_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan_dir = root / "plan"
            scripts_dir = plan_dir / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            (scripts_dir / "worker.py").write_text("print('ok')\n", encoding="utf-8")
            (plan_dir / "plan.md").write_text(
                "\n".join(
                    [
                        "## Tasks",
                        "",
                        "### review_task",
                        "- **executor**: worker",
                        "- **task_class**: llm",
                        "- **llm_model**: {WORKER_MODEL}",
                        "- **command**: `python3 scripts/worker.py --model {WORKER_MODEL}`",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            brain = MockBrainPlan(root)
            batch_id = brain.execute_plan(
                str(plan_dir),
                config_overrides={
                    "WORKER_MODEL": "qwen2.5-coder:7b",
                    "RUN_MODE": "fresh",
                },
            )

            self.assertIn(batch_id, brain.active_batches)
            public_files = list(brain.public_tasks_path.glob("*.json"))
            self.assertEqual(len(public_files), 1)
            task = json.loads(public_files[0].read_text(encoding="utf-8"))
            self.assertEqual(task["command"], "python3 scripts/worker.py --model qwen2.5-coder:7b")
            self.assertEqual(task["llm_model"], "qwen2.5-coder:7b")
            self.assertEqual(task["llm_min_tier"], 1)
            self.assertEqual(task["llm_placement"], "single_gpu")
            self.assertNotIn("definition_error", task)


if __name__ == "__main__":
    unittest.main()
