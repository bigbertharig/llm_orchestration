#!/usr/bin/env python3
"""Tests for claim tokens and first-result-wins task finalization (task_claims.py)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from task_claims import drop_if_already_recorded, finalize_task, owns_claim, stamp_claim


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data))


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


class TaskClaimsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tasks = Path(self._tmp.name) / "tasks"
        for lane in ("queue", "processing", "complete", "failed"):
            (self.tasks / lane).mkdir(parents=True)

    def tearDown(self):
        self._tmp.cleanup()

    def _claim(self, worker: str, attempts: int = 1) -> dict:
        task = {"task_id": "t1", "name": "demo", "assigned_to": worker, "attempts": attempts,
                "status": "processing"}
        stamp_claim(task)
        _write(self.tasks / "processing" / "t1.json", task)
        _write(self.tasks / "processing" / "t1.heartbeat.json", {"task_id": "t1"})
        return dict(task)

    def test_owner_success_completes_and_cleans_up(self):
        task = self._claim("gpu-1")
        self.assertEqual(finalize_task(self.tasks, task, {"success": True}), "complete")
        self.assertEqual(_read(self.tasks / "complete" / "t1.json")["status"], "complete")
        self.assertFalse((self.tasks / "processing" / "t1.json").exists())
        self.assertFalse((self.tasks / "processing" / "t1.heartbeat.json").exists())

    def test_owner_failure_goes_to_failed(self):
        task = self._claim("gpu-1")
        self.assertEqual(finalize_task(self.tasks, task, {"success": False}), "failed")
        self.assertTrue((self.tasks / "failed" / "t1.json").exists())
        self.assertFalse((self.tasks / "processing" / "t1.json").exists())

    def test_late_success_after_another_result_is_kept_aside(self):
        old = self._claim("cpu-a")
        new = self._claim("gpu-2", attempts=2)           # requeued and re-claimed
        self.assertEqual(finalize_task(self.tasks, new, {"success": True, "output": "new"}), "complete")
        self.assertEqual(finalize_task(self.tasks, old, {"success": True, "output": "old"}), "late")
        self.assertEqual(_read(self.tasks / "complete" / "t1.json")["result"]["output"], "new")
        late = _read(self.tasks / "late" / f"t1.{old['claim_id']}.json")
        self.assertEqual(late["late_reason"], "result_already_recorded")

    def test_stale_success_first_wins_but_leaves_new_owner_alone(self):
        old = self._claim("cpu-a")
        new = self._claim("gpu-2", attempts=2)           # new owner still running
        self.assertEqual(finalize_task(self.tasks, old, {"success": True, "output": "old"}), "complete")
        self.assertEqual(_read(self.tasks / "processing" / "t1.json")["claim_id"], new["claim_id"])
        self.assertTrue((self.tasks / "processing" / "t1.heartbeat.json").exists())
        # The new owner finishing later is the late one, and cleans up its own claim.
        self.assertEqual(finalize_task(self.tasks, new, {"success": True, "output": "new"}), "late")
        self.assertFalse((self.tasks / "processing" / "t1.json").exists())
        self.assertEqual(_read(self.tasks / "complete" / "t1.json")["result"]["output"], "old")

    def test_stale_failure_does_not_queue_a_duplicate_retry(self):
        old = self._claim("cpu-a")
        new = self._claim("gpu-2", attempts=2)
        self.assertEqual(finalize_task(self.tasks, old, {"success": False}), "late")
        self.assertFalse((self.tasks / "failed" / "t1.json").exists())
        self.assertEqual(_read(self.tasks / "processing" / "t1.json")["claim_id"], new["claim_id"])

    def test_failure_after_requeue_to_queue_is_late(self):
        old = self._claim("cpu-a")
        (self.tasks / "processing" / "t1.json").unlink()  # brain moved it back to queue
        _write(self.tasks / "queue" / "t1.json", {"task_id": "t1", "status": "pending"})
        self.assertEqual(finalize_task(self.tasks, old, {"success": False}), "late")
        self.assertFalse((self.tasks / "failed" / "t1.json").exists())

    def test_claims_from_before_claim_ids_match_on_worker_and_attempt(self):
        current = {"task_id": "t1", "assigned_to": "gpu-1", "attempts": 2}
        self.assertTrue(owns_claim(current, {"assigned_to": "gpu-1", "attempts": 2}))
        self.assertFalse(owns_claim(current, {"assigned_to": "gpu-1", "attempts": 1}))
        self.assertFalse(owns_claim(None, {"assigned_to": "gpu-1", "attempts": 2}))

    def test_queued_copy_of_a_recorded_task_is_dropped(self):
        _write(self.tasks / "complete" / "t1.json", {"task_id": "t1", "status": "complete"})
        queued = self.tasks / "queue" / "t1.json"
        _write(queued, {"task_id": "t1", "status": "pending"})
        self.assertTrue(drop_if_already_recorded(self.tasks, queued))
        self.assertFalse(queued.exists())
        other = self.tasks / "queue" / "t2.json"
        _write(other, {"task_id": "t2", "status": "pending"})
        self.assertFalse(drop_if_already_recorded(self.tasks, other))
        self.assertTrue(other.exists())

    def test_each_claim_gets_a_new_id(self):
        task = {"task_id": "t1"}
        first = stamp_claim(task)
        self.assertNotEqual(first, stamp_claim(task))


class RequeueScrubTest(unittest.TestCase):
    def test_requeue_drops_claim_id(self):
        from brain_failures import BrainFailureMixin

        brain = BrainFailureMixin.__new__(BrainFailureMixin)
        task = {"task_id": "t1", "status": "processing", "assigned_to": "gpu-1", "claim_id": "abc",
                "started_at": "x", "attempts": 1}
        brain._prepare_task_for_requeue(task, "orphan_recovered")
        brain._assert_queue_requeue_invariants(task)
        self.assertNotIn("claim_id", task)
        self.assertEqual(task["attempts"], 1)


if __name__ == "__main__":
    unittest.main()
