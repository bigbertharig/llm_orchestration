#!/usr/bin/env python3
"""Tests for split recovery observation signal details."""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

if "filelock" not in sys.modules:
    sys.modules["filelock"] = types.SimpleNamespace(FileLock=object, Timeout=Exception)
if "requests" not in sys.modules:
    sys.modules["requests"] = types.SimpleNamespace()

from gpu_split import GPUSplitMixin


class MockGpuSplit(GPUSplitMixin):
    def __init__(self, root: Path):
        self.name = "gpu-1"
        self.signals_path = root / "signals"
        self.signals_path.mkdir(parents=True, exist_ok=True)
        self.runtime_state = "recovering_split"
        self.split_runtime_owner = "gpu-1"
        self.split_runtime_generation = "gen-1"
        self.logger = MagicMock()

    def _split_reservation_path(self, group_id: str) -> Path:
        return self.signals_path / f"{group_id}.reservation.json"

    def _read_split_reservation_epoch(self, group_id: str):
        return "epoch-1"


class SplitRecoverySignalDetailsTests(unittest.TestCase):
    def test_stage_d_signal_includes_verification_details(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gpu = MockGpuSplit(root)
            reservation_path = gpu._split_reservation_path("pair_1_2")
            reservation_path.write_text(json.dumps({"members": ["gpu-1", "gpu-2"]}), encoding="utf-8")

            result = gpu._auto_recovery_stage_d_fallback_tasks(
                "pair_1_2",
                11434,
                verify_result={
                    "checks": {
                        "runtime_state": {"expected": "cold", "actual": "recovering_split", "passed": False},
                        "split_port_clear": {"port": 11434, "has_listener": True, "passed": False},
                    },
                    "duration_ms": 12.5,
                },
            )

            self.assertTrue(result["signal_written"])
            signal_path = gpu.signals_path / "gpu-1.recovery_fallback.json"
            payload = json.loads(signal_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["type"], "split_recovery_observation")
            self.assertEqual(payload["observed_state"]["split_port"], 11434)
            self.assertIn("verification_checks", payload["observed_state"])
            self.assertIn("runtime_state", payload["observed_state"]["verification_checks"])
            self.assertEqual(payload["observed_state"]["verification_duration_ms"], 12.5)


if __name__ == "__main__":
    unittest.main()
