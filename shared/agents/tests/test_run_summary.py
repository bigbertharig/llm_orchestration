#!/usr/bin/env python3
"""Tests for best-effort history-folder summary reduction."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from run_summary import render_run_summary_markdown, summarize_history_dir


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class RunSummaryTests(unittest.TestCase):
    def test_failed_run_surfaces_failure_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            history_dir = Path(tmp) / "history" / "20260305_014325"
            _write_json(history_dir / "batch_meta.json", {"plan_name": "github_analyzer"})
            _write_json(
                history_dir / "results" / "batch_failure.json",
                {
                    "reason": "fatal analyzer error",
                    "source_task": "analyze_repo",
                    "source_task_id": "task-7",
                    "abandoned_tasks": 3,
                },
            )

            summary = summarize_history_dir(history_dir)

            self.assertEqual(summary["status"], "failed")
            artifact_paths = [artifact["path"] for artifact in summary["important_artifacts"]]
            self.assertIn("results/batch_failure.json", artifact_paths)
            failure_artifact = next(
                artifact for artifact in summary["important_artifacts"] if artifact["path"] == "results/batch_failure.json"
            )
            self.assertIn("fatal analyzer error", failure_artifact["excerpt"])

    def test_complete_run_surfaces_terminal_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            history_dir = Path(tmp) / "history" / "20260215_131551"
            _write_json(history_dir / "batch_meta.json", {"plan_name": "github_analyzer"})
            _write_json(
                history_dir / "execution_stats.json",
                {"overall": {"completed_tasks": 9}, "outcome": {"status": "success"}},
            )
            _write_json(history_dir / "output" / "final_report.json", {"title": "Final report", "issues": []})
            (history_dir / "EXECUTION_SUMMARY.md").write_text("# Success\nall good\n", encoding="utf-8")

            summary = summarize_history_dir(history_dir)

            self.assertEqual(summary["status"], "complete")
            artifact_paths = [artifact["path"] for artifact in summary["important_artifacts"]]
            self.assertIn("execution_stats.json", artifact_paths)
            self.assertIn("EXECUTION_SUMMARY.md", artifact_paths)
            self.assertIn("output/final_report.json", artifact_paths)

    def test_partial_run_surfaces_available_outputs_without_guessing_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            history_dir = Path(tmp) / "history" / "20260304_145415"
            _write_json(history_dir / "batch_meta.json", {"plan_name": "github_analyzer"})
            _write_json(history_dir / "output" / "json_data_summary.json", {"files": 12, "records": 48})
            _write_json(history_dir / "output" / "repo_structure_summary.json", {"dirs": 8, "files": 100})

            summary = summarize_history_dir(history_dir)

            self.assertEqual(summary["status"], "partial")
            artifact_paths = [artifact["path"] for artifact in summary["important_artifacts"]]
            self.assertIn("output/json_data_summary.json", artifact_paths)
            self.assertIn("output/repo_structure_summary.json", artifact_paths)

    def test_markdown_renders_flat_artifact_excerpt_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            history_dir = Path(tmp) / "history" / "20260214_225744"
            _write_json(history_dir / "batch_meta.json", {"plan_name": "github_analyzer"})
            _write_json(
                history_dir / "results" / "batch_failure.json",
                {"reason": "fatal analyzer error", "source_task": "analyze_repo"},
            )

            summary = summarize_history_dir(history_dir)
            markdown = render_run_summary_markdown(summary)

            self.assertIn("## Important Artifacts", markdown)
            self.assertIn("- results/batch_failure.json [json]", markdown)
            self.assertIn("- excerpt: reason=fatal analyzer error", markdown)


if __name__ == "__main__":
    unittest.main()
