import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

AGENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENTS.parent / "scripts" / "rig"))
import hw_guard  # noqa: E402

LIMITS = hw_guard.load_limits(Path("/nonexistent/guard.json"))


def sample(power=300.0, gpu_temp=60.0, cpu=50.0, mem=12000):
    return {"gpu_power_total_w": power, "gpu_temp_max_c": gpu_temp, "cpu_package_c": cpu, "mem_available_mb": mem}


class LevelTests(unittest.TestCase):
    def test_readings_in_range_are_ok(self):
        self.assertEqual(hw_guard.sample_level(sample(), LIMITS), ("ok", []))

    def test_each_reading_can_raise_the_level(self):
        self.assertEqual(hw_guard.sample_level(sample(power=700), LIMITS)[0], "warn")
        self.assertEqual(hw_guard.sample_level(sample(power=760), LIMITS)[0], "critical")
        self.assertEqual(hw_guard.sample_level(sample(gpu_temp=91), LIMITS)[0], "critical")
        self.assertEqual(hw_guard.sample_level(sample(cpu=92), LIMITS)[0], "warn")
        self.assertEqual(hw_guard.sample_level(sample(mem=3000), LIMITS)[0], "warn")
        level, reasons = hw_guard.sample_level(sample(mem=1000, cpu=96), LIMITS)
        self.assertEqual(level, "critical")
        self.assertEqual(len(reasons), 2)

    def test_power_lines_follow_the_card_limits(self):
        limits = hw_guard.load_limits(Path("/nonexistent/guard.json"))
        sample = {"gpu_power_total_w": 1000.0, "gpu_power_limit_total_w": 1122.0}
        self.assertEqual(hw_guard.power_lines(sample, limits), {"warn": 1122.0, "critical": 1234.2})
        self.assertEqual(hw_guard.sample_level(sample, limits)[0], "ok")
        self.assertEqual(hw_guard.sample_level({**sample, "gpu_power_total_w": 1130.0}, limits)[0], "warn")
        self.assertEqual(hw_guard.sample_level({**sample, "gpu_power_total_w": 1240.0}, limits)[0], "critical")
        # limits unreadable: the fixed fallback lines
        self.assertEqual(hw_guard.power_lines({"gpu_power_total_w": 700.0}, limits), {"warn": 650, "critical": 750})

    def test_missing_sensors_are_skipped(self):
        self.assertEqual(hw_guard.sample_level({"gpu_power_total_w": None}, LIMITS)[0], "ok")

    def test_overrides_merge_over_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guard.json"
            path.write_text(json.dumps({"gpu_power_total_w": {"warn": 500}, "fail_closed": True}))
            limits = hw_guard.load_limits(path)
        self.assertEqual(limits["gpu_power_total_w"]["warn"], 500)
        self.assertFalse(limits["gpu_power_total_w"]["from_limits"])  # a fixed line in guard.json wins
        self.assertTrue(limits["fail_closed"])


class EvaluatorTests(unittest.TestCase):
    def test_rises_at_once_and_clears_only_after_calm_samples(self):
        ev = hw_guard.Evaluator(LIMITS)
        self.assertEqual(ev.push(sample(power=760))[0], "critical")
        levels = [ev.push(sample())[0] for _ in range(12)]
        # the spike stays in the 5-sample window for 4 more samples, then 5 calm samples clear it
        self.assertEqual(levels.index("ok"), 8)
        self.assertEqual(set(levels[:8]), {"critical"})

    def test_a_new_spike_restarts_the_calm_count(self):
        ev = hw_guard.Evaluator(LIMITS)
        ev.push(sample(power=700))
        for _ in range(7):
            ev.push(sample())
        self.assertEqual(ev.push(sample(power=700))[0], "warn")


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "guard_state.json"

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, level, age_s=0.0, reasons=()):
        self.path.write_text(json.dumps({"level": level, "reasons": list(reasons), "updated_ts": time.time() - age_s}))

    def test_only_ok_admits(self):
        self.write("ok")
        self.assertEqual(hw_guard.admission(LIMITS, self.path)[:2], (True, "ok"))
        self.write("warn", reasons=["GPU power 700 W >= warn 650"])
        ok, level, why = hw_guard.admission(LIMITS, self.path)
        self.assertEqual((ok, level), (False, "warn"))
        self.assertIn("700 W", why)

    def test_missing_or_stale_guard_follows_fail_closed(self):
        self.assertEqual(hw_guard.admission(LIMITS, self.path)[:2], (True, "unknown"))
        self.write("ok", age_s=60)
        self.assertEqual(hw_guard.admission(LIMITS, self.path)[:2], (True, "unknown"))
        closed = {**LIMITS, "fail_closed": True}
        self.assertEqual(hw_guard.admission(closed, self.path)[:2], (False, "unknown"))


class MemoryActionTests(unittest.TestCase):
    def test_acts_on_critical_memory_once_per_cooldown(self):
        now = 1000.0
        self.assertTrue(hw_guard.memory_action_due(sample(mem=1500), LIMITS, 0.0, now))
        self.assertFalse(hw_guard.memory_action_due(sample(mem=1500), LIMITS, now - 30, now))
        self.assertTrue(hw_guard.memory_action_due(sample(mem=1500), LIMITS, now - 61, now))

    def test_warn_memory_or_hold_only_does_not_act(self):
        self.assertFalse(hw_guard.memory_action_due(sample(mem=3000), LIMITS, 0.0, 1000.0))
        hold = {**LIMITS, "memory_action": "hold_only"}
        self.assertFalse(hw_guard.memory_action_due(sample(mem=1500), hold, 0.0, 1000.0))
        self.assertFalse(hw_guard.memory_action_due({"mem_available_mb": None}, LIMITS, 0.0, 1000.0))


if __name__ == "__main__":
    unittest.main()
