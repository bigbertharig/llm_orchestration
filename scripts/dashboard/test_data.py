#!/usr/bin/env python3
"""Tests for dashboard control-plane aggregation."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.data import apply_gpu_leases


class DashboardLeaseOverlayTests(unittest.TestCase):
    def test_lease_overlays_missing_heartbeat_row(self):
        rows = [
            {
                "gpu_id": 4,
                "state": "missing_heartbeat",
                "busy": False,
                "busy_lane": "",
            }
        ]
        apply_gpu_leases(
            rows,
            {
                "leases": [
                    {
                        "lease_id": "supervision-1",
                        "controller": "supervision",
                        "lane": "supervision",
                        "owner": "county-session",
                        "gpu_ids": [4],
                    }
                ]
            },
        )

        self.assertTrue(rows[0]["leased"])
        self.assertEqual(rows[0]["lease_controller"], "supervision")
        self.assertEqual(rows[0]["busy_lane"], "SUPERVISING")
        self.assertEqual(rows[0]["busy_owner"], "county-session")
        self.assertEqual(
            rows[0]["blocked_lanes"],
            ["BENCHING", "TASKING", "CHATTING", "SUPERVISING"],
        )

    def test_duplicate_gpu_lease_fails_visible(self):
        rows = [{"gpu_id": 1}]
        apply_gpu_leases(
            rows,
            {
                "leases": [
                    {"lease_id": "one", "controller": "plans", "gpu_ids": [1]},
                    {"lease_id": "two", "controller": "supervision", "gpu_ids": [1]},
                ]
            },
        )

        self.assertEqual(rows[0]["lease_controller"], "system")
        self.assertEqual(rows[0]["busy_lane"], "CONTROL_ERROR")
        self.assertIn("multiple leases", rows[0]["lease"]["lease_error"])


if __name__ == "__main__":
    unittest.main()
