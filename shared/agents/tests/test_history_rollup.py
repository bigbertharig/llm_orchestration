#!/usr/bin/env python3
"""Tests for multi-run history rollup."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from run_summary import summarize_history_root, write_history_rollup


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class HistoryRollupTests(unittest.TestCase):
    def test_rollup_summarizes_many_runs_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            history_root = Path(tmp) / "history"
            failed = history_root / "20260305_014325"
            complete = history_root / "20260215_131551"
            partial = history_root / "20260304_145415"

            _write_json(failed / "batch_meta.json", {"plan_name": "github_analyzer"})
            _write_json(
                failed / "results" / "batch_failure.json",
                {"reason": "fatal analyzer error", "source_task": "analyze_repo", "abandoned_tasks": 3},
            )

            _write_json(complete / "batch_meta.json", {"plan_name": "github_analyzer"})
            _write_json(
                complete / "execution_stats.json",
                {"overall": {"completed_tasks": 9}, "outcome": {"status": "success"}},
            )
            (complete / "EXECUTION_SUMMARY.md").parent.mkdir(parents=True, exist_ok=True)
            (complete / "EXECUTION_SUMMARY.md").write_text("# Success\n", encoding="utf-8")

            _write_json(partial / "batch_meta.json", {"plan_name": "github_analyzer"})
            _write_json(partial / "output" / "json_data_summary.json", {"records": 10})

            rollup = summarize_history_root(history_root, refresh_runs=True)

            self.assertEqual(rollup["run_count"], 3)
            self.assertEqual(rollup["status_counts"]["failed"], 1)
            self.assertEqual(rollup["status_counts"]["complete"], 1)
            self.assertEqual(rollup["status_counts"]["partial"], 1)
            self.assertEqual(len(rollup["failures"]), 1)
            self.assertEqual(rollup["failure_source_counts"]["analyze_repo"], 1)

            outputs = write_history_rollup(history_root, rollup)

            self.assertTrue(Path(outputs["rollup_json"]).exists())
            self.assertTrue(Path(outputs["rollup_markdown"]).exists())
            self.assertTrue(Path(outputs["runs_jsonl"]).exists())
            self.assertTrue(Path(outputs["failures_jsonl"]).exists())

    def test_rollup_skips_summary_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            history_root = Path(tmp) / "history"
            _write_json(history_root / "_summary" / "ignored.json", {"ok": True})
            _write_json(history_root / "20260305_014325" / "batch_meta.json", {"plan_name": "demo"})

            rollup = summarize_history_root(history_root, refresh_runs=True)

            self.assertEqual(rollup["run_count"], 1)
