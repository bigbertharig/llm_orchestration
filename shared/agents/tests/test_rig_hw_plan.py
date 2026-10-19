import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

AGENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENTS.parent / "scripts" / "rig"))
import hw_plan  # noqa: E402


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.plan = hw_plan.Plan(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def age_heartbeat(self, seconds):
        data = json.loads(self.plan.path.read_text())
        old = datetime.now(timezone.utc) - timedelta(seconds=seconds)
        data["controller"]["heartbeat_at"] = old.isoformat(timespec="seconds").replace("+00:00", "Z")
        self.plan.path.write_text(json.dumps(data))

    def test_one_active_controller_with_stale_takeover_and_handoff(self):
        self.plan.claim("pi", "10.0.0.2")
        with self.assertRaises(hw_plan.PlanError) as ctx:
            self.plan.claim("desktop", "10.0.0.4")
        self.assertEqual(ctx.exception.kind, "not_controller")
        self.age_heartbeat(hw_plan.STALE_AFTER_S + 5)
        self.assertTrue(self.plan.report()["controller"]["stale"])
        ctl = self.plan.claim("desktop", "10.0.0.4")
        self.assertEqual((ctl["id"], ctl["took_over_from"]), ("desktop", "pi"))
        self.assertEqual(self.plan.claim("pi", "10.0.0.2", force=True)["id"], "pi")  # explicit handoff

    def test_only_the_active_controller_edits_the_plan(self):
        self.plan.require_active(None)  # nobody registered: open
        self.plan.claim("pi", "10.0.0.2")
        with self.assertRaises(hw_plan.PlanError):
            self.plan.add("desktop", "load", {"profile": "coder-7b"})
        with self.assertRaises(hw_plan.PlanError):
            self.plan.require_active(None)
        self.plan.require_active("pi")
        item = self.plan.add("pi", "load", {"profile": "coder-7b", "gpus": [1], "port": 11435})
        self.assertEqual(item["state"], "queued")

    def test_items_run_in_order_one_at_a_time(self):
        self.plan.claim("pi", "h")
        a = self.plan.add("pi", "load", {"profile": "a"})
        b = self.plan.add("pi", "unload", {"profile": "a"})
        first = self.plan.take_next()
        self.assertEqual(first["id"], a["id"])
        self.assertIsNone(self.plan.take_next())  # one running at a time
        self.plan.finish(a["id"], True, "ready")
        self.assertEqual(self.plan.take_next()["id"], b["id"])
        self.plan.finish(b["id"], False, "boom")
        rep = self.plan.report()
        self.assertEqual([i["state"] for i in rep["items"]], ["done", "failed"])
        self.assertEqual(rep["next"], [])

    def test_cancel_and_recover(self):
        self.plan.claim("pi", "h")
        a = self.plan.add("pi", "load", {"profile": "a"})
        b = self.plan.add("pi", "load", {"profile": "b"})
        self.plan.cancel("pi", b["id"])
        self.plan.take_next()
        with self.assertRaises(hw_plan.PlanError):
            self.plan.cancel("pi", a["id"])  # running items are not cancelled
        self.assertEqual(self.plan.recover(), [a["id"]])
        self.assertEqual({i["id"]: i["state"] for i in self.plan.report()["items"]},
                         {a["id"]: "interrupted", b["id"]: "cancelled"})

    def test_a_new_boot_interrupts_and_holds_until_the_controller_resumes(self):
        self.plan.claim("pi", "h")
        self.plan.boot("boot-1")
        self.plan.resume("pi")
        a = self.plan.add("pi", "load", {"profile": "a"})
        b = self.plan.add("pi", "load", {"profile": "b"})
        self.plan.take_next()                             # a is running when the rig goes down
        hold = self.plan.boot("boot-2")
        self.assertEqual(hold["interrupted"], [a["id"]])
        self.assertIsNone(self.plan.boot("boot-2"))       # same boot again: nothing new
        rep = self.plan.report()
        self.assertEqual(rep["hold"]["reason"], "boot")
        self.assertIn("rig restarted", self.plan.get(a["id"])["result"])
        self.assertIsNone(self.plan.take_next())          # held: b waits
        with self.assertRaises(hw_plan.PlanError):
            self.plan.resume("desktop")                   # only the active controller resumes
        self.plan.resume("pi")
        self.assertEqual(self.plan.take_next()["id"], b["id"])

    def test_restore_writes_back_a_newer_controller_copy_only(self):
        self.plan.claim("pi", "h")
        self.plan.add("pi", "load", {"profile": "a"})
        copy = self.plan.report()                          # the controller's copy
        with self.assertRaises(hw_plan.PlanError) as ctx:
            self.plan.restore("pi", copy)                  # the rig is as new: refused
        self.assertEqual(ctx.exception.kind, "state")
        self.plan.path.unlink()                            # the rig lost its file
        self.plan.boot("boot-9")
        rep = self.plan.restore("pi", copy)
        self.assertEqual([i["args"]["profile"] for i in rep["items"]], ["a"])
        self.assertTrue(rep["hold"])                       # still held until the controller resumes
        self.assertGreater(rep["version"], copy["version"])
        self.assertEqual(self.plan.report()["controller"]["id"], "pi")

    def test_bad_items_are_refused(self):
        self.plan.claim("pi", "h")
        for kind, args in (("dance", {"profile": "a"}), ("load", {})):
            with self.assertRaises(hw_plan.PlanError) as ctx:
                self.plan.add("pi", kind, args)
            self.assertEqual(ctx.exception.kind, "invalid")


if __name__ == "__main__":
    unittest.main()
