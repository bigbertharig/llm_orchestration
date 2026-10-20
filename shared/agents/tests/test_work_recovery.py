#!/usr/bin/env python3
"""Tests for work recovery: dead vs unreachable owners and wait-first reassignment (brain_monitor.py)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from test_brain_monitor_requeue import MockBrainMonitor, _write_json  # noqa: E402

STALE = (datetime.now() - timedelta(seconds=60)).isoformat()  # past the mock's 10 s threshold


class WorkRecoveryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.brain = MockBrainMonitor(self.root)
        self.brain.shared_path = self.root
        self.reachable = {}
        self.pings = []

        def probe(host):
            self.pings.append(host)
            return self.reachable.get(host, False)

        self.brain._host_reachable = probe
        _write_json(self.root / "cpus" / "cpu-a" / "heartbeat.json",
                    {"name": "cpu-a", "ip_address": "10.0.0.21", "last_updated": STALE})
        _write_json(self.root / "cpus" / "cpu-b" / "heartbeat.json",
                    {"name": "cpu-b", "ip_address": "10.0.0.22", "last_updated": STALE})

    def tearDown(self):
        self._tmp.cleanup()

    def _processing(self, task_id, owner, batch="batch-1", claim="c1"):
        _write_json(self.brain.processing_path / f"{task_id}.json", {
            "task_id": task_id, "batch_id": batch, "name": task_id, "status": "processing",
            "assigned_to": owner, "started_at": STALE, "claim_id": claim, "attempts": 1,
        })

    def _queued(self, task_id, batch="batch-1"):
        _write_json(self.brain.queue_path / f"{task_id}.json",
                    {"task_id": task_id, "batch_id": batch, "status": "pending"})

    def _events(self, name):
        return [e for e in self.brain.logged if e["event"] == name]

    def _requeued(self, task_id):
        p = self.brain.queue_path / f"{task_id}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def test_owner_whose_host_answers_is_dead_and_requeued_now(self):
        self.reachable["10.0.0.21"] = True
        self._processing("t1", "cpu-a")
        self._queued("t2")                                   # other work does not delay a dead owner
        self.assertEqual(self.brain._recover_orphaned_processing_tasks(running_workers={}), 1)
        task = self._requeued("t1")
        self.assertEqual(task["requeue_reason"], "orphan_recovered")
        self.assertEqual(task["orphan_owner_state"], "dead")
        self.assertNotIn("claim_id", task)

    def test_unreachable_owner_waits_while_batch_work_is_queued(self):
        self._processing("t1", "cpu-a")
        self._queued("t2")
        for _ in range(2):
            self.assertEqual(self.brain._recover_orphaned_processing_tasks(running_workers={}), 0)
        self.assertTrue((self.brain.processing_path / "t1.json").exists())
        self.assertEqual(len(self._events("ORPHAN_UNREACHABLE_WAIT")), 1)   # logged once

    def test_unreachable_owner_waits_while_a_live_worker_runs_batch_work(self):
        self._processing("t1", "cpu-a")
        self._processing("t2", "gpu-1", claim="c2")
        live = {"gpu-1": {"active_task_ids": ["t2"]}}
        self.assertEqual(self.brain._recover_orphaned_processing_tasks(running_workers=live), 0)
        self.assertTrue((self.brain.processing_path / "t1.json").exists())

    def test_unreachable_owner_is_reassigned_when_it_is_the_tail(self):
        self._processing("t1", "cpu-a")
        self._queued("other", batch="batch-2")              # other batches do not count
        self.assertEqual(self.brain._recover_orphaned_processing_tasks(running_workers={}), 1)
        task = self._requeued("t1")
        self.assertEqual(task["requeue_reason"], "unreachable_reassigned_tail")
        self.assertEqual(task["orphan_owner_state"], "unreachable")
        event = self._events("ORPHAN_UNREACHABLE_REASSIGN")[0]
        self.assertEqual(event["details"]["trigger"], "tail")

    def test_unreachable_owner_is_reassigned_at_the_time_cap(self):
        self.brain.unreachable_reassign_after_seconds = 100
        self._processing("t1", "cpu-a")
        self._queued("t2")
        self.brain._unreachable_waits()["t1:c1"] = datetime.now() - timedelta(seconds=101)
        self.assertEqual(self.brain._recover_orphaned_processing_tasks(running_workers={}), 1)
        self.assertEqual(self._requeued("t1")["requeue_reason"], "unreachable_reassigned_time_cap")

    def test_time_cap_marks_the_task_timed_out(self):
        self.brain.unreachable_reassign_after_seconds = 100
        self._processing("t1", "cpu-a")
        self._queued("t2")
        self.brain._unreachable_waits()["t1:c1"] = datetime.now() - timedelta(seconds=150)
        self.brain._recover_orphaned_processing_tasks(running_workers={})
        task = self._requeued("t1")
        self.assertTrue(task["timed_out"])
        self.assertEqual(task["timeouts"][0]["limit"], "work_recovery.unreachable_reassign_after_seconds")
        self.assertGreaterEqual(task["timeouts"][0]["observed_s"], 150)
        self.assertEqual(task["recovery_history"][-1]["claim_id"], "c1")
        self.assertEqual(self._events("TASK_TIMED_OUT")[0]["details"]["worker"], "cpu-a")

    def test_dead_owner_requeue_is_tracked_but_not_timed_out(self):
        self.reachable["10.0.0.21"] = True
        self._processing("t1", "cpu-a")
        self.brain._recover_orphaned_processing_tasks(running_workers={})
        task = self._requeued("t1")
        self.assertNotIn("timed_out", task)
        self.assertEqual(task["recovery_history"][0]["from_worker"], "cpu-a")
        self.assertEqual(task["recovery_history"][0]["owner_state"], "dead")
        self.assertEqual(self._events("TASK_TIMED_OUT"), [])

    def test_stuck_requeue_marks_the_no_progress_timeout(self):
        task = {"task_id": "t9", "batch_id": "batch-1", "name": "slow", "status": "processing",
                "task_class": "llm", "assigned_to": "gpu-2", "started_at": STALE, "claim_id": "c9"}
        task_file = self.brain.processing_path / "t9.json"
        _write_json(task_file, task)
        incident = self.brain._get_or_create_incident(task, {"success": False})
        incident["stuck_abort_sent_at"] = (datetime.now() - timedelta(seconds=20)).isoformat()
        incident["stuck_force_kill_sent_at"] = (datetime.now() - timedelta(seconds=10)).isoformat()
        incident["stuck_abort_count"] = 2
        self.brain._handle_stuck_tasks([{"task": task, "task_file": task_file, "task_id": "t9",
                                         "assigned_to": "gpu-2", "elapsed_min": 20,
                                         "progress_age_sec": 500, "threshold_sec": 420, "name": "slow"}])
        queued = self._requeued("t9")
        self.assertTrue(queued["timed_out"])
        self.assertEqual(queued["timeouts"][0]["limit"], "task_no_progress_thresholds.llm")
        self.assertEqual(queued["timeouts"][0]["observed_s"], 500)
        self.assertEqual(self._events("TASK_TIMED_OUT")[0]["details"]["limit_s"], 420)

    def test_review_script_summarizes_durations_and_timeouts(self):
        import subprocess
        complete = self.root / "tasks" / "complete"
        _write_json(complete / "a.json", {"task_id": "a", "task_class": "llm", "duration_s": 30})
        _write_json(complete / "b.json", {"task_id": "b", "task_class": "llm", "duration_s": 90,
                                          "timed_out": True, "timeouts": [
                                              {"limit": "task_no_progress_thresholds.llm", "limit_s": 420,
                                               "observed_s": 500, "worker": "gpu-2", "action": "x"}]})
        script = Path(__file__).resolve().parents[3] / "scripts" / "review_task_durations.py"
        out = subprocess.run([sys.executable, str(script), "--shared-path", str(self.root), "--json"],
                             capture_output=True, text=True, check=True).stdout
        report = json.loads(out)
        self.assertEqual(report["durations_by_class"]["llm"]["n"], 2)
        self.assertEqual(report["durations_by_class"]["llm"]["max_s"], 90)
        self.assertEqual(report["timeouts"][0]["observed_s"], 500)

    def test_null_cap_never_reassigns_on_time(self):
        self.brain.unreachable_reassign_after_seconds = None
        self._processing("t1", "cpu-a")
        self._queued("t2")
        self.brain._unreachable_waits()["t1:c1"] = datetime.now() - timedelta(days=30)
        self.assertEqual(self.brain._recover_orphaned_processing_tasks(running_workers={}), 0)

    def test_stale_gpu_owner_is_dead_without_a_probe(self):
        self._processing("t1", "gpu-3")
        self._queued("t2")
        self.assertEqual(self.brain._recover_orphaned_processing_tasks(running_workers={}), 1)
        self.assertEqual(self._requeued("t1")["orphan_owner_state"], "dead")
        self.assertEqual(self.pings, [])

    def test_one_probe_per_host_per_cycle(self):
        self._processing("t1", "cpu-a", claim="c1")
        self._processing("t3", "cpu-a", claim="c3")
        self._queued("t2")
        self.brain._recover_orphaned_processing_tasks(running_workers={})
        self.assertEqual(self.pings, ["10.0.0.21"])

    def test_wait_is_forgotten_once_the_task_leaves_processing(self):
        self._processing("t1", "cpu-a")
        self._queued("t2")
        self.brain._recover_orphaned_processing_tasks(running_workers={})
        self.assertIn("t1:c1", self.brain._unreachable_waits())
        (self.brain.processing_path / "t1.json").unlink()   # owner came back and finished
        self.brain._recover_orphaned_processing_tasks(running_workers={})
        self.assertEqual(self.brain._unreachable_waits(), {})


if __name__ == "__main__":
    unittest.main()
