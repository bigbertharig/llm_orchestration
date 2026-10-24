import sys
import unittest
from pathlib import Path

AGENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENTS.parent / "scripts" / "rig"))
import rig  # noqa: E402


class ModeTests(unittest.TestCase):
    def test_use_picks_the_load_mode(self):
        self.assertEqual(rig.load_mode({"use": "benchmark"}), "bench")
        self.assertEqual(rig.load_mode({"use": "chat"}), "remote")
        self.assertEqual(rig.load_mode({"use": "worker"}), "remote")
        self.assertEqual(rig.load_mode({"id": "published"}), "remote")

    def test_one_mode_at_a_time(self):
        self.assertFalse(rig.mode_blocks("idle", "bench", force=False))
        self.assertFalse(rig.mode_blocks("bench", "bench", force=False))
        self.assertTrue(rig.mode_blocks("remote", "bench", force=False))
        self.assertTrue(rig.mode_blocks("bench", "remote", force=False))
        self.assertTrue(rig.mode_blocks("orchestration", "bench", force=False))
        # force always stops whatever is active first
        self.assertTrue(rig.mode_blocks("bench", "bench", force=True))
        self.assertFalse(rig.mode_blocks("idle", "remote", force=True))

    def test_bench_is_no_longer_a_stack(self):
        self.assertNotIn("bench", rig.STACK_UNITS)
        self.assertEqual(set(rig.LOAD_MODES), {"bench", "remote"})


if __name__ == "__main__":
    unittest.main()
