#!/usr/bin/env python3
"""Tests for reusable worker preflight scan helpers."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "scan_workers.py"
SPEC = importlib.util.spec_from_file_location("scan_workers", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load {SCRIPT_PATH}")
scan_workers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scan_workers)


class ScanWorkersTests(unittest.TestCase):
    def test_processing_count_ignores_heartbeat_sidecars(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared_dir = Path(tmp)
            processing_dir = shared_dir / "tasks" / "processing"
            processing_dir.mkdir(parents=True, exist_ok=True)
            (processing_dir / "task-1.json").write_text("{}", encoding="utf-8")
            (processing_dir / "task-1.heartbeat.json").write_text("{}", encoding="utf-8")
            (processing_dir / "task-2.heartbeat.json").write_text("{}", encoding="utf-8")

            original_shared_dir = scan_workers.SHARED_DIR
            scan_workers.SHARED_DIR = shared_dir
            try:
                self.assertEqual(scan_workers._processing_task_count(), 1)
            finally:
                scan_workers.SHARED_DIR = original_shared_dir


if __name__ == "__main__":
    unittest.main()
