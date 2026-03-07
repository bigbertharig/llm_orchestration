#!/usr/bin/env python3
"""Tests for observation-only split recovery signals."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock
import types

sys.path.insert(0, str(Path(__file__).parent.parent))

if "filelock" not in sys.modules:
    sys.modules["filelock"] = types.SimpleNamespace(FileLock=object, Timeout=Exception)
if "requests" not in sys.modules:
    sys.modules["requests"] = types.SimpleNamespace()

from brain_resources import BrainResourceMixin


class MockBrainResources(BrainResourceMixin):
    def __init__(self, root: Path):
        self.signals_path = root / "signals"
        self.signals_path.mkdir(parents=True, exist_ok=True)
        self.logger = MagicMock()
        self.logged = []
        self.inserted = []
        self.brain_split_failures = {}
        self.brain_quarantined_pairs = {}

    def log_decision(self, event, message, details):
        self.logged.append({"event": event, "message": message, "details": details})

    def _insert_resource_task(self, command, meta=None):
        self.inserted.append({"command": command, "meta": meta or {}})

    def _split_group_members_for_group_id(self, group_id: str):
        if group_id == "pair_1_2":
            return ["gpu-1", "gpu-2"]
        return []


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class SplitRecoveryObservationTests(unittest.TestCase):
    def test_observation_signal_triggers_brain_derived_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            brain = MockBrainResources(root)
            signal_path = brain.signals_path / "gpu-1.recovery_fallback.json"
            _write_json(
                signal_path,
                {
                    "type": "split_recovery_observation",
                    "group_id": "pair_1_2",
                    "worker": "gpu-1",
                    "members": ["gpu-1", "gpu-2"],
                    "issue_code": "verified_cold_failed",
                },
            )

            brain._process_recovery_fallback_signals()

            commands = [item["command"] for item in brain.inserted]
            self.assertIn("unload_split_llm", commands)
            self.assertEqual(commands.count("unload_llm"), 2)
            self.assertFalse(signal_path.exists())

    def test_legacy_fallback_signal_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            brain = MockBrainResources(root)
            signal_path = brain.signals_path / "gpu-1.recovery_fallback.json"
            _write_json(
                signal_path,
                {
                    "type": "split_recovery_fallback",
                    "group_id": "pair_1_2",
                    "worker": "gpu-1",
                    "members": ["gpu-1", "gpu-2"],
                    "issue_code": "verified_cold_failed",
                },
            )

            brain._process_recovery_fallback_signals()

            self.assertEqual(brain.inserted, [])
            self.assertTrue(signal_path.exists())


if __name__ == "__main__":
    unittest.main()
