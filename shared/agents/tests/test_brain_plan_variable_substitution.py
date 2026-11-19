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
        self.model_tier_by_id = {"qwen2.5-coder:7b": 1, "reasoner:14b": 2}
        self.model_meta_by_id = {
            "qwen2.5-coder:7b": {"placement": "single_gpu"},
            "reasoner:14b": {"placement": "split_gpu"},
        }
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
    def _gateway_plan(self, root: Path, command: str = "python3 scripts/worker.py") -> Path:
        plan_dir = root / "plan"
        scripts_dir = plan_dir / "scripts"
        scripts_dir.mkdir(parents=True)
        (scripts_dir / "worker.py").write_text("print('ok')\n", encoding="utf-8")
        (plan_dir / "plan.md").write_text(
            "\n".join([
                "## Tasks", "", "### review_task", "- **executor**: worker",
                "- **task_class**: llm", "- **llm_role**: code_review",
                "- **llm_fallback_roles**: reasoning", "- **llm_model**: qwen2.5-coder:7b",
                f"- **command**: `{command}`", "", "### pinned_task",
                "- **executor**: worker", "- **task_class**: llm",
                "- **llm_model**: qwen2.5-coder:7b",
                "- **command**: `python3 scripts/worker.py`",
            ]) + "\n", encoding="utf-8",
        )
        return plan_dir

    def test_gateway_assignment_overrides_only_tagged_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            brain = MockBrainPlan(root)
            plan_dir = self._gateway_plan(root)
            payload = {"schema": 1, "assignments": {
                "review_task": {"model": "reasoner:14b", "role": "reasoning"}
            }}
            brain.execute_plan(str(plan_dir), {"GATEWAY_MODEL_ASSIGNMENTS": json.dumps(payload)})
            tasks = {
                task["name"]: task
                for path in brain.public_tasks_path.glob("*.json")
                for task in [json.loads(path.read_text(encoding="utf-8"))]
            }
            self.assertEqual(tasks["review_task"]["llm_model"], "reasoner:14b")
            self.assertEqual(tasks["review_task"]["llm_min_tier"], 2)
            self.assertEqual(tasks["review_task"]["llm_placement"], "split_gpu")
            self.assertEqual(tasks["pinned_task"]["llm_model"], "qwen2.5-coder:7b")

    def test_gateway_assignment_rejects_unknown_task_before_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            brain = MockBrainPlan(root)
            plan_dir = self._gateway_plan(root)
            payload = {"schema": 1, "assignments": {
                "missing": {"model": "reasoner:14b", "role": "reasoning"}
            }}
            with self.assertRaisesRegex(ValueError, "Unknown gateway model assignment task ID"):
                brain.execute_plan(str(plan_dir), {"GATEWAY_MODEL_ASSIGNMENTS": json.dumps(payload)})
            self.assertEqual(list(brain.public_tasks_path.glob("*.json")), [])

    def test_gateway_assignment_rejects_conflicting_command_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            brain = MockBrainPlan(root)
            plan_dir = self._gateway_plan(root, "python3 scripts/worker.py --model=qwen2.5-coder:7b")
            payload = {"schema": 1, "assignments": {
                "review_task": {"model": "reasoner:14b", "role": "reasoning"}
            }}
            with self.assertRaisesRegex(ValueError, "conflicts with command model flags"):
                brain.execute_plan(str(plan_dir), {"GATEWAY_MODEL_ASSIGNMENTS": json.dumps(payload)})
            self.assertEqual(list(brain.public_tasks_path.glob("*.json")), [])

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
