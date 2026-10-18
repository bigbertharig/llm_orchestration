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
                         {a["id"]: "failed", b["id"]: "cancelled"})

    def test_bad_items_are_refused(self):
        self.plan.claim("pi", "h")
        for kind, args in (("dance", {"profile": "a"}), ("load", {})):
            with self.assertRaises(hw_plan.PlanError) as ctx:
                self.plan.add("pi", kind, args)
            self.assertEqual(ctx.exception.kind, "invalid")


if __name__ == "__main__":
    unittest.main()
