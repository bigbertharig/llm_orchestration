#!/usr/bin/env python3
"""Tests for brain-owned split quarantine visibility surfaces."""

from __future__ import annotations

import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

if "filelock" not in sys.modules:
    sys.modules["filelock"] = types.SimpleNamespace(FileLock=object, Timeout=Exception)
if "requests" not in sys.modules:
    sys.modules["requests"] = types.SimpleNamespace()

from brain_core import BrainCoreMixin


class MockBrainCore(BrainCoreMixin):
    def __init__(self, root: Path):
        self.active_batches = {}
        self.load_llm_requests = {}
        self.last_resource_task_at = {}
        self.last_any_llm_demand_at = __import__("datetime").datetime.now()
        self.last_split_llm_demand_at = __import__("datetime").datetime.now()
        self.incidents = {}
        self.gpu_missing_escalations = {}
        self.brain_split_failures = {"pair_1_2": [{"timestamp": time.time(), "reason": "cleanup_failed"}]}
        self.brain_quarantined_pairs = {
            "pair_1_2": {
                "until": time.time() + 120,
                "failure_count": 3,
                "reason": "cleanup_failed",
            },
            "expired_pair": {
                "until": time.time() - 10,
                "failure_count": 3,
                "reason": "old_failure",
            },
        }
        self.state_file = root / "brain" / "state.json"
        self.brain_heartbeat_file = root / "brain" / "heartbeat.json"
        self.brain_heartbeat_file_unified = root / "heartbeats" / "brain.json"
        self.logger = MagicMock()
        self.running = True
        self.gpus = [0]
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.brain_heartbeat_file_unified.parent.mkdir(parents=True, exist_ok=True)


class BrainQuarantineVisibilityTests(unittest.TestCase):
    def test_brain_heartbeat_surfaces_active_split_quarantine(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain = MockBrainCore(Path(tmp))

            brain._write_brain_heartbeat()

            hb = json.loads(brain.brain_heartbeat_file.read_text(encoding="utf-8"))
            self.assertIn("split_quarantine", hb)
            self.assertEqual(hb["split_quarantine"]["active_pair_count"], 1)
            self.assertEqual(hb["split_quarantine"]["tracked_failure_groups"], 1)
            self.assertIn("pair_1_2", hb["split_quarantine"]["pairs"])
            self.assertNotIn("expired_pair", hb["split_quarantine"]["pairs"])

    def test_brain_state_persists_quarantine_maps(self):
        with tempfile.TemporaryDirectory() as tmp:
            brain = MockBrainCore(Path(tmp))

            brain._save_brain_state()

            state = json.loads(brain.state_file.read_text(encoding="utf-8"))
            self.assertIn("brain_split_failures", state)
            self.assertIn("brain_quarantined_pairs", state)
            self.assertIn("pair_1_2", state["brain_quarantined_pairs"])


if __name__ == "__main__":
    unittest.main()
