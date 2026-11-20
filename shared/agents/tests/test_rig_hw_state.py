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
        self.root = base / "local"

    def tearDown(self):
        self.tmp.cleanup()

    def make(self):
        return hw_state.HardwareState(self.root)

    def local_record(self, name):
        return json.loads((self.root / "gpus" / name).read_text())

    def test_acquire_is_recorded_locally(self):
        hs = self.make()
        lease = hs.acquire(spec([0], metadata={"port": 11434}))
        self.assertEqual([l["lease_id"] for l in hs.leases()], [lease["lease_id"]])
        self.assertEqual(self.local_record("leases.json")["leases"][0]["gpu_ids"], [0])

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
        self.assertEqual(self.local_record("claims.json")["claims"][0]["name"], "bench")
        hs.remove_claim("bench")
        self.assertEqual(hs.claims(), [])

    def test_a_claim_reserves_cards_that_are_still_leased(self):
        hs = self.make()
        held = hs.acquire(spec([1]))["lease_id"]
        hs.add_claim("split-1-2", [1, 2], owner="lane-1-2", ttl_s=60, purpose="split load", model="m:14b")
        with self.assertRaises(hw_state.StateError):  # the free partner cannot be taken by single work
            hs.acquire(spec([2]))
        hs.release(held, owner="llm-gateway")
        lease = hs.acquire(spec([1, 2], metadata={"claim": "split-1-2"}))
        self.assertEqual(lease["gpu_ids"], [1, 2])

    def test_a_claim_grows_as_a_collecting_wait_takes_cards(self):
        hs = self.make()
        hs.add_claim("wait-m", [2], owner="llm-gateway", ttl_s=60)
        hs.add_claim("other", [4])
        grown = hs.extend_claim("wait-m", [1], owner="llm-gateway")
        self.assertEqual(grown["gpu_ids"], [1, 2])
        self.assertEqual(self.local_record("claims.json")["claims"][0]["gpu_ids"], [1, 2])
        with self.assertRaises(hw_state.StateError) as ctx:   # a card another claim holds
            hs.extend_claim("wait-m", [4], owner="llm-gateway")
        self.assertEqual(ctx.exception.kind, "conflict")
        with self.assertRaises(hw_state.StateError) as ctx:   # a claim with a different owner
            hs.extend_claim("wait-m", [3], owner="lane-1-2")
        self.assertEqual(ctx.exception.kind, "ownership")
        with self.assertRaises(hw_state.StateError) as ctx:
            hs.extend_claim("gone", [3])
        self.assertEqual(ctx.exception.kind, "not_found")

    def test_claims_with_a_ttl_lapse_unless_renewed(self):
        hs = self.make()
        claim = hs.add_claim("wait", [3], owner="campaign-x", ttl_s=60)
        self.assertIn("expires_at", claim)
        renewed = hs.renew_claim("wait", "campaign-x", 120)
        self.assertEqual(renewed["ttl_s"], 120)
        with self.assertRaises(hw_state.StateError) as ctx:
            hs.renew_claim("wait", "someone-else")
        self.assertEqual(ctx.exception.kind, "ownership")
        # Lapse it: an expired claim no longer blocks leases and is dropped (and logged) on the next write.
        record = json.loads(hs.claims_path.read_text())
        record["claims"][0]["expires_at"] = "2000-01-01T00:00:00Z"
        hs.claims_path.write_text(json.dumps(record))
        self.assertEqual(hs.claims(), [])
        hs.acquire(spec([3]))
        hs.add_claim("other", [4])
        self.assertEqual([c["name"] for c in json.loads(hs.claims_path.read_text())["claims"]], ["other"])
        self.assertIn("expired wait", (self.root / "gpus" / "claims_expired.log").read_text())
        with self.assertRaises(hw_state.StateError) as ctx:
            hs.renew_claim("wait")
        self.assertEqual(ctx.exception.kind, "not_found")

    def test_a_claim_without_ttl_gets_bounded_default_and_kill_drops_claims_on_cards(self):
        hs = self.make()
        claim = hs.add_claim("bench", [0])
        self.assertEqual(claim["ttl_s"], hw_state.DEFAULT_CLAIM_TTL_S)
        self.assertEqual(claim["boot_id"], hw_state.BOOT_ID)
        self.assertIn("expires_at", claim)
        self.assertIn("renewed_at", hs.renew_claim("bench"))
        hs.add_claim("pair", [4, 5])
        dropped = hs.remove_claims_on([5])
        self.assertEqual([c["name"] for c in dropped], ["pair"])
        self.assertEqual([c["name"] for c in hs.claims()], ["bench"])

    def test_previous_boot_and_legacy_claims_are_removed_on_startup(self):
        hs = self.make()
        hs.add_claim("old-boot", [0])
        record = json.loads(hs.claims_path.read_text())
        record["claims"][0]["boot_id"] = "previous-boot"
        record["claims"].append({"name": "legacy", "gpu_ids": [1]})
        hs.claims_path.write_text(json.dumps(record))
        restarted = self.make()
        self.assertEqual(restarted.claims(), [])
        self.assertEqual(self.local_record("claims.json")["claims"], [])

    def test_claim_ttl_cannot_exceed_one_hour(self):
        hs = self.make()
        with self.assertRaises(hw_state.StateError) as ctx:
            hs.add_claim("too-long", [0], ttl_s=hw_state.MAX_CLAIM_TTL_S + 1)
        self.assertEqual(ctx.exception.kind, "invalid")

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

    def test_local_lease_survives_reopen(self):
        first = self.make()
        lid = first.acquire(spec([4]))["lease_id"]
        reopened = self.make()
        self.assertEqual([lease["lease_id"] for lease in reopened.leases()], [lid])

    def test_local_record_does_not_need_a_shared_drive(self):
        first = self.make()
        first.acquire(spec([5]))
        self.assertEqual(len(self.make().leases()), 1)


if __name__ == "__main__":
    unittest.main()
