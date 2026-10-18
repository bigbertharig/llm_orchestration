import json
import sys
import tempfile
import unittest
from pathlib import Path

AGENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENTS.parent / "scripts" / "rig"))
import hw_state  # noqa: E402


def spec(gpus, **extra):
    return {"controller": "interactive", "owner": "llm-gateway", "run_id": "r", "gpu_ids": gpus,
            "ports": [11434 + g for g in gpus], "ttl_s": 60, **extra}


class HardwareStateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.root, self.share = base / "local", base / "share"
        (self.share / "gpus").mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, mirror=True):
        return hw_state.HardwareState(self.root, AGENTS, self.share if mirror else None)

    def mirrored(self, name):
        return json.loads((self.share / "gpus" / name).read_text())

    def test_acquire_is_recorded_locally_and_mirrored(self):
        hs = self.make()
        lease = hs.acquire(spec([0], metadata={"port": 11434}))
        self.assertEqual([l["lease_id"] for l in hs.leases()], [lease["lease_id"]])
        self.assertEqual(self.mirrored("leases.json")["leases"][0]["gpu_ids"], [0])

    def test_overlap_is_refused(self):
        hs = self.make()
        hs.acquire(spec([1, 2]))
        with self.assertRaises(hw_state.StateError) as ctx:
            hs.acquire(spec([2]))
        self.assertEqual(ctx.exception.kind, "conflict")

    def test_claimed_cards_need_the_claim(self):
        hs = self.make()
        hs.add_claim("bench", [0], note="3090 for benchmarks")
        with self.assertRaises(hw_state.StateError) as ctx:
            hs.acquire(spec([0]))
        self.assertIn("claimed by bench", str(ctx.exception))
        hs.acquire(spec([0], metadata={"claim": "bench"}))
        with self.assertRaises(hw_state.StateError):
            hs.add_claim("other", [0])
        self.assertEqual(self.mirrored("claims.json")["claims"][0]["name"], "bench")
        hs.remove_claim("bench")
        self.assertEqual(hs.claims(), [])

    def test_release_checks_the_owner(self):
        hs = self.make()
        lid = hs.acquire(spec([3]))["lease_id"]
        with self.assertRaises(hw_state.StateError) as ctx:
            hs.release(lid, owner="someone-else")
        self.assertEqual(ctx.exception.kind, "ownership")
        hs.release(lid, owner="llm-gateway")
        self.assertEqual(hs.leases(), [])
        with self.assertRaises(hw_state.StateError) as ctx:
            hs.renew(lid, "llm-gateway", 60)
        self.assertEqual(ctx.exception.kind, "not_found")

    def test_empty_local_record_is_seeded_from_the_drive_once(self):
        first = self.make()
        lid = first.acquire(spec([4]))["lease_id"]
        # A fresh rig-local root (leases moved off the drive) starts from the drive's copy.
        self.root = Path(self.tmp.name) / "fresh"
        seeded = self.make()
        self.assertEqual([l["lease_id"] for l in seeded.leases()], [lid])

    def test_works_without_the_drive(self):
        hs = self.make(mirror=False)
        hs.acquire(spec([5]))
        self.assertEqual(hs.mirror(), "skipped")
        self.assertEqual(len(hs.leases()), 1)


if __name__ == "__main__":
    unittest.main()
