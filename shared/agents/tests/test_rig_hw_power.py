import json
import sys
import tempfile
import unittest
from pathlib import Path

AGENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENTS.parent / "scripts" / "rig"))
import hw_power  # noqa: E402

GTX_OC = {"index": 1, "name": "NVIDIA GeForce GTX 1060 6GB", "limit_w": 156.0, "default_w": 156.0,
          "min_w": 78.0, "max_w": 182.0}
GTX = {**GTX_OC, "index": 4, "limit_w": 120.0, "default_w": 120.0, "min_w": 60.0, "max_w": 140.0}
TI = {"index": 0, "name": "NVIDIA GeForce RTX 3090 Ti", "limit_w": 450.0, "default_w": 450.0,
      "min_w": 100.0, "max_w": 480.0}


class PowerTests(unittest.TestCase):
    def test_1060s_by_default_limit_and_unknown_cards_keep_their_default(self):
        rules = hw_power.load_rules(Path("/nonexistent/power.json"), Path("/nonexistent/override.json"))
        # 156 W factory-overclocked boards at 120 W, 120 W boards at 100 W, the 3090 Ti untouched
        self.assertEqual([hw_power.planned(c, rules) for c in (TI, GTX_OC, GTX)], [450.0, 120.0, 100.0])

    def test_index_override_wins_and_values_are_clamped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "power.json"
            path.write_text(json.dumps({"by_index": {"0": 400, "4": 20}}))
            rules = hw_power.load_rules(path, Path("/nonexistent/override.json"))
        self.assertEqual(hw_power.planned(TI, rules), 400.0)
        self.assertEqual(hw_power.planned(GTX, rules), 60.0)       # below the card minimum: clamped
        self.assertEqual(hw_power.planned(GTX_OC, rules), 120.0)   # by_model still applies


    def test_temporary_override_until_boot_sits_on_top(self):
        with tempfile.TemporaryDirectory() as tmp:
            override = Path(tmp) / "power_override.json"
            override.write_text(json.dumps({"by_model": {"GTX 1060": {"156": 156, "120": 120}}}))
            rules = hw_power.load_rules(Path("/nonexistent/power.json"), override)
        self.assertEqual([hw_power.planned(c, rules) for c in (TI, GTX_OC, GTX)], [450.0, 156.0, 120.0])

if __name__ == "__main__":
    unittest.main()
