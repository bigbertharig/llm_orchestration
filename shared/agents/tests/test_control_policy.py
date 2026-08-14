#!/usr/bin/env python3
"""Tests for boot-profile and controller-admission policy."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from control_policy import (
    control_policy_allows_task,
    read_control_policy,
    task_controller,
    write_control_policy,
)


class ControlPolicyTests(unittest.TestCase):
    def test_missing_policy_is_neutral_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            payload = read_control_policy(shared)
            self.assertEqual(payload["boot_profile"], "neutral")
            self.assertEqual(payload["admitted_controllers"], [])
            allowed, reason = control_policy_allows_task(shared, {"batch_id": "plan-1"})
            self.assertFalse(allowed)
            self.assertEqual(reason, "controller_not_admitted:plans")

    def test_plan_ready_defaults_to_plan_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            write_control_policy(
                shared,
                "plan_ready",
                reason="submit_requested",
                updated_by="test",
            )
            payload = read_control_policy(shared)
            self.assertEqual(payload["boot_profile"], "plan_ready")
            self.assertEqual(payload["admitted_controllers"], ["plans"])
            allowed, reason = control_policy_allows_task(shared, {"batch_id": "plan-1"})
            self.assertTrue(allowed)
            self.assertEqual(reason, "")

    def test_profile_and_admission_are_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            write_control_policy(
                shared,
                "custom",
                admitted_controllers=["plans", "supervision"],
                updated_by="test",
            )
            payload = read_control_policy(shared)
            self.assertEqual(payload["boot_profile"], "custom")
            self.assertEqual(payload["admitted_controllers"], ["plans", "supervision"])

    def test_benchmark_runtime_prep_is_classified_without_plan_special_case(self):
        task = {
            "batch_id": "runtime_prep_20260813",
            "task_class": "meta",
            "command": "load_llm",
            "created_by": "runtime-prep",
        }
        self.assertEqual(task_controller(task), "benchmarks")
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            write_control_policy(shared, "benchmark_ready", updated_by="test")
            allowed, reason = control_policy_allows_task(shared, task)
            self.assertTrue(allowed)
            self.assertEqual(reason, "")

    def test_benchmark_profile_blocks_plan_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            write_control_policy(shared, "benchmark_ready", updated_by="test")
            allowed, reason = control_policy_allows_task(shared, {"batch_id": "plan-1"})
            self.assertFalse(allowed)
            self.assertEqual(reason, "controller_not_admitted:plans")

    def test_system_tasks_are_always_admitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            allowed, reason = control_policy_allows_task(
                Path(tmp),
                {"batch_id": "system", "task_class": "meta"},
            )
            self.assertTrue(allowed)
            self.assertEqual(reason, "")

    def test_invalid_persisted_policy_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = Path(tmp)
            path = shared / "brain" / "control_policy.json"
            path.parent.mkdir(parents=True)
            path.write_text('{"schema_version":1,"boot_profile":"bogus"}', encoding="utf-8")
            payload = read_control_policy(shared)
            self.assertEqual(payload["boot_profile"], "neutral")
            self.assertEqual(payload["reason"], "invalid_boot_profile")


if __name__ == "__main__":
    unittest.main()
