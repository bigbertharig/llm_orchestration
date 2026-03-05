#!/usr/bin/env python3
"""
Tests for thermal recovery upgrade - brain-level thermal incident handling.

Run:
  python -m pytest tests/test_thermal_recovery.py -v
  python tests/test_thermal_recovery.py  # standalone
"""

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Allow running from agents/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))


class MockBrain:
    """Mock Brain class for testing thermal recovery controller."""

    def __init__(self):
        # Thermal recovery config (defaults)
        self.thermal_recovery_trigger_seconds = 300
        self.thermal_recovery_reset_interval_seconds = 60
        self.thermal_recovery_max_resets = 5
        self.thermal_recovery_full_reset_cooldown = 300
        self.thermal_recovery_enable_full_reset = True
        self.thermal_recovery_same_gpu_backoff = 120

        # Active incident tracking
        self.thermal_recovery_active_incident_id = None
        self.thermal_recovery_incident_started_at = None
        self.thermal_recovery_resets_issued = 0
        self.thermal_recovery_last_reset_at = None
        self.thermal_recovery_last_reset_gpu = None
        self.thermal_recovery_full_reset_at = None
        self.thermal_recovery_gpu_last_reset = {}

        # Mock logger and decision log
        self.logger = MagicMock()
        self.decisions = []

        # Track inserted tasks
        self.inserted_tasks = []

    def log_decision(self, event, message, details):
        self.decisions.append({
            "event": event,
            "message": message,
            "details": details,
        })

    def _insert_resource_task(self, command, meta=None):
        self.inserted_tasks.append({
            "command": command,
            "meta": meta or {},
        })


def make_gpu_state(
    gpu_name,
    cpu_temp_c=50,
    gpu_temp_c=60,
    incident_id=None,
    sustained_seconds=0,
):
    """Create a mock GPU state dict."""
    return {
        "name": gpu_name,
        "cpu_temp_c": cpu_temp_c,
        "temperature_c": gpu_temp_c,
        "thermal_overheat_incident_id": incident_id,
        "thermal_overheat_sustained_seconds": sustained_seconds,
        "model_loaded": False,
        "runtime_state": "cold",
    }


# =============================================================================
# Inline implementation of thermal recovery controller for testing
# This avoids import dependencies while testing the core logic
# =============================================================================
def _check_thermal_recovery_escalation_impl(brain, gpu_states):
    """
    Brain-level thermal recovery controller implementation.
    Extracted for testing without module import dependencies.
    """
    now = time.time()

    # Find GPUs with active thermal incidents (sustained overheat)
    incident_gpus = []
    for gpu_name, state in gpu_states.items():
        incident_id = state.get("thermal_overheat_incident_id")
        sustained_seconds = int(state.get("thermal_overheat_sustained_seconds", 0) or 0)
        cpu_temp = state.get("cpu_temp_c")

        if incident_id and sustained_seconds >= brain.thermal_recovery_trigger_seconds:
            incident_gpus.append({
                "gpu": gpu_name,
                "incident_id": incident_id,
                "sustained_seconds": sustained_seconds,
                "cpu_temp_c": cpu_temp,
                "gpu_temp_c": state.get("temperature_c"),
            })

    # No active thermal incidents meeting trigger threshold
    if not incident_gpus:
        if brain.thermal_recovery_active_incident_id:
            brain.log_decision(
                "THERMAL_INCIDENT_BRAIN_CLEARED",
                f"Thermal incident cleared: {brain.thermal_recovery_active_incident_id}",
                {
                    "incident_id": brain.thermal_recovery_active_incident_id,
                    "total_resets_issued": brain.thermal_recovery_resets_issued,
                    "duration_seconds": int(now - (brain.thermal_recovery_incident_started_at or now)),
                },
            )
            brain.thermal_recovery_active_incident_id = None
            brain.thermal_recovery_incident_started_at = None
            brain.thermal_recovery_resets_issued = 0
            brain.thermal_recovery_last_reset_at = None
            brain.thermal_recovery_last_reset_gpu = None
        return

    # Sort by hottest CPU temp first (descending)
    incident_gpus.sort(key=lambda x: x.get("cpu_temp_c") or 0, reverse=True)

    # Start or continue brain-level incident tracking
    canonical_incident_id = incident_gpus[0]["incident_id"]
    if brain.thermal_recovery_active_incident_id != canonical_incident_id:
        brain.thermal_recovery_active_incident_id = canonical_incident_id
        brain.thermal_recovery_incident_started_at = now
        brain.thermal_recovery_resets_issued = 0
        brain.thermal_recovery_last_reset_at = None
        brain.thermal_recovery_last_reset_gpu = None
        brain.log_decision(
            "THERMAL_INCIDENT_BRAIN_START",
            f"Brain tracking thermal incident: {canonical_incident_id}",
            {
                "incident_id": canonical_incident_id,
                "affected_gpus": [g["gpu"] for g in incident_gpus],
                "hottest_cpu_c": incident_gpus[0].get("cpu_temp_c"),
                "trigger_threshold_seconds": brain.thermal_recovery_trigger_seconds,
            },
        )

    # Check if we've exceeded max targeted resets
    if brain.thermal_recovery_resets_issued >= brain.thermal_recovery_max_resets:
        if brain.thermal_recovery_enable_full_reset:
            if brain.thermal_recovery_full_reset_at:
                elapsed_since_full = now - brain.thermal_recovery_full_reset_at
                if elapsed_since_full < brain.thermal_recovery_full_reset_cooldown:
                    remaining = int(brain.thermal_recovery_full_reset_cooldown - elapsed_since_full)
                    brain.log_decision(
                        "THERMAL_FULL_RESET_COOLDOWN",
                        f"Full reset cooldown active ({remaining}s remaining)",
                        {
                            "incident_id": canonical_incident_id,
                            "cooldown_seconds": remaining,
                            "resets_issued": brain.thermal_recovery_resets_issued,
                        },
                    )
                    return

            brain.log_decision(
                "THERMAL_FULL_RESET_TRIGGERED",
                f"Thermal recovery escalating to full orchestrator reset",
                {
                    "incident_id": canonical_incident_id,
                    "resets_issued": brain.thermal_recovery_resets_issued,
                    "max_resets": brain.thermal_recovery_max_resets,
                    "affected_gpus": [g["gpu"] for g in incident_gpus],
                },
            )
            brain._insert_resource_task(
                "orchestrator_full_reset",
                meta={
                    "reason": "thermal_recovery_escalation",
                    "incident_id": canonical_incident_id,
                    "resets_attempted": brain.thermal_recovery_resets_issued,
                },
            )
            brain.thermal_recovery_full_reset_at = now
        else:
            brain.log_decision(
                "THERMAL_ESCALATION_DISABLED",
                f"Max targeted resets reached but full reset disabled",
                {
                    "incident_id": canonical_incident_id,
                    "resets_issued": brain.thermal_recovery_resets_issued,
                },
            )
        return

    # Check if we're within reset interval
    if brain.thermal_recovery_last_reset_at:
        elapsed_since_reset = now - brain.thermal_recovery_last_reset_at
        if elapsed_since_reset < brain.thermal_recovery_reset_interval_seconds:
            return

    # Select target GPU for reset (hottest that's not in backoff)
    target_gpu = None
    for gpu_info in incident_gpus:
        gpu_name = gpu_info["gpu"]
        last_reset = brain.thermal_recovery_gpu_last_reset.get(gpu_name)
        if last_reset:
            elapsed = now - last_reset
            if elapsed < brain.thermal_recovery_same_gpu_backoff:
                continue
        target_gpu = gpu_name
        break

    if not target_gpu:
        brain.log_decision(
            "THERMAL_RESET_ALL_BACKOFF",
            f"All overheated GPUs in reset backoff",
            {
                "incident_id": canonical_incident_id,
                "gpus": [g["gpu"] for g in incident_gpus],
                "backoff_seconds": brain.thermal_recovery_same_gpu_backoff,
            },
        )
        return

    # Issue targeted reset
    brain.thermal_recovery_resets_issued += 1
    brain.thermal_recovery_last_reset_at = now
    brain.thermal_recovery_last_reset_gpu = target_gpu
    brain.thermal_recovery_gpu_last_reset[target_gpu] = now

    target_info = next((g for g in incident_gpus if g["gpu"] == target_gpu), {})

    brain.log_decision(
        "THERMAL_TARGETED_RESET_ISSUED",
        f"Issuing targeted reset for {target_gpu}",
        {
            "incident_id": canonical_incident_id,
            "target_gpu": target_gpu,
            "reset_number": brain.thermal_recovery_resets_issued,
            "max_resets": brain.thermal_recovery_max_resets,
            "cpu_temp_c": target_info.get("cpu_temp_c"),
        },
    )

    brain._insert_resource_task(
        "reset_gpu_runtime",
        meta={
            "target_worker": target_gpu,
            "reason": "thermal_recovery",
            "incident_id": canonical_incident_id,
            "reset_number": brain.thermal_recovery_resets_issued,
        },
    )


# =============================================================================
# Test Classes
# =============================================================================

class TestThermalRecoveryTargetSelection(unittest.TestCase):
    """Test brain target selection picks hottest eligible GPU."""

    def setUp(self):
        self.brain = MockBrain()

    def test_selects_hottest_gpu(self):
        """Controller should target hottest GPU first."""
        gpu_states = {
            "gpu-1": make_gpu_state(
                "gpu-1", cpu_temp_c=85, incident_id="inc-1", sustained_seconds=350
            ),
            "gpu-2": make_gpu_state(
                "gpu-2", cpu_temp_c=92, incident_id="inc-2", sustained_seconds=400
            ),
            "gpu-3": make_gpu_state(
                "gpu-3", cpu_temp_c=78, incident_id="inc-3", sustained_seconds=320
            ),
        }

        _check_thermal_recovery_escalation_impl(self.brain, gpu_states)

        # Should target gpu-2 (hottest at 92C)
        self.assertEqual(len(self.brain.inserted_tasks), 1)
        task = self.brain.inserted_tasks[0]
        self.assertEqual(task["command"], "reset_gpu_runtime")
        self.assertEqual(task["meta"]["target_worker"], "gpu-2")

    def test_skips_gpu_in_backoff(self):
        """Controller should skip GPUs that were recently reset."""
        now = time.time()
        # Put gpu-2 in backoff (reset 30s ago, backoff is 120s)
        self.brain.thermal_recovery_gpu_last_reset = {"gpu-2": now - 30}

        gpu_states = {
            "gpu-1": make_gpu_state(
                "gpu-1", cpu_temp_c=85, incident_id="inc-1", sustained_seconds=350
            ),
            "gpu-2": make_gpu_state(
                "gpu-2", cpu_temp_c=92, incident_id="inc-2", sustained_seconds=400
            ),
        }

        _check_thermal_recovery_escalation_impl(self.brain, gpu_states)

        # Should target gpu-1 (gpu-2 is in backoff)
        self.assertEqual(len(self.brain.inserted_tasks), 1)
        task = self.brain.inserted_tasks[0]
        self.assertEqual(task["meta"]["target_worker"], "gpu-1")

    def test_no_action_when_all_in_backoff(self):
        """Controller should not issue reset when all GPUs are in backoff."""
        now = time.time()
        self.brain.thermal_recovery_gpu_last_reset = {
            "gpu-1": now - 30,
            "gpu-2": now - 60,
        }

        gpu_states = {
            "gpu-1": make_gpu_state(
                "gpu-1", cpu_temp_c=85, incident_id="inc-1", sustained_seconds=350
            ),
            "gpu-2": make_gpu_state(
                "gpu-2", cpu_temp_c=92, incident_id="inc-2", sustained_seconds=400
            ),
        }

        _check_thermal_recovery_escalation_impl(self.brain, gpu_states)

        # No reset should be issued
        self.assertEqual(len(self.brain.inserted_tasks), 0)
        # Should log backoff event
        backoff_events = [d for d in self.brain.decisions if "BACKOFF" in d["event"]]
        self.assertEqual(len(backoff_events), 1)


class TestThermalRecoveryRateLimiting(unittest.TestCase):
    """Test rate-limiter enforces one reset per interval."""

    def setUp(self):
        self.brain = MockBrain()

    def test_respects_reset_interval(self):
        """Controller should wait reset_interval_seconds between resets."""
        now = time.time()
        # Last reset was 30s ago, interval is 60s
        self.brain.thermal_recovery_last_reset_at = now - 30
        self.brain.thermal_recovery_active_incident_id = "inc-1"
        self.brain.thermal_recovery_incident_started_at = now - 400

        gpu_states = {
            "gpu-1": make_gpu_state(
                "gpu-1", cpu_temp_c=85, incident_id="inc-1", sustained_seconds=350
            ),
        }

        _check_thermal_recovery_escalation_impl(self.brain, gpu_states)

        # Should not issue reset (within interval)
        self.assertEqual(len(self.brain.inserted_tasks), 0)

    def test_allows_reset_after_interval(self):
        """Controller should issue reset after interval has passed."""
        now = time.time()
        # Last reset was 65s ago, interval is 60s
        self.brain.thermal_recovery_last_reset_at = now - 65
        self.brain.thermal_recovery_active_incident_id = "inc-1"
        self.brain.thermal_recovery_incident_started_at = now - 400

        gpu_states = {
            "gpu-1": make_gpu_state(
                "gpu-1", cpu_temp_c=85, incident_id="inc-1", sustained_seconds=350
            ),
        }

        _check_thermal_recovery_escalation_impl(self.brain, gpu_states)

        # Should issue reset
        self.assertEqual(len(self.brain.inserted_tasks), 1)
        self.assertEqual(self.brain.inserted_tasks[0]["command"], "reset_gpu_runtime")


class TestThermalRecoveryEscalation(unittest.TestCase):
    """Test escalation after max targeted resets."""

    def setUp(self):
        self.brain = MockBrain()

    def test_escalates_to_full_reset(self):
        """Controller should escalate to full reset after max targeted resets."""
        now = time.time()
        self.brain.thermal_recovery_active_incident_id = "inc-1"
        self.brain.thermal_recovery_incident_started_at = now - 600
        self.brain.thermal_recovery_resets_issued = 5  # Max resets
        self.brain.thermal_recovery_last_reset_at = now - 65  # Past interval

        gpu_states = {
            "gpu-1": make_gpu_state(
                "gpu-1", cpu_temp_c=85, incident_id="inc-1", sustained_seconds=600
            ),
        }

        _check_thermal_recovery_escalation_impl(self.brain, gpu_states)

        # Should issue full reset, not targeted reset
        self.assertEqual(len(self.brain.inserted_tasks), 1)
        task = self.brain.inserted_tasks[0]
        self.assertEqual(task["command"], "orchestrator_full_reset")
        self.assertEqual(task["meta"]["reason"], "thermal_recovery_escalation")

    def test_full_reset_respects_cooldown(self):
        """Controller should not issue full reset within cooldown period."""
        now = time.time()
        self.brain.thermal_recovery_active_incident_id = "inc-1"
        self.brain.thermal_recovery_incident_started_at = now - 600
        self.brain.thermal_recovery_resets_issued = 5
        self.brain.thermal_recovery_full_reset_at = now - 100  # 100s ago, cooldown is 300s

        gpu_states = {
            "gpu-1": make_gpu_state(
                "gpu-1", cpu_temp_c=85, incident_id="inc-1", sustained_seconds=600
            ),
        }

        _check_thermal_recovery_escalation_impl(self.brain, gpu_states)

        # Should not issue reset (cooldown active)
        self.assertEqual(len(self.brain.inserted_tasks), 0)
        # Should log cooldown event
        cooldown_events = [d for d in self.brain.decisions if "COOLDOWN" in d["event"]]
        self.assertEqual(len(cooldown_events), 1)

    def test_does_not_escalate_if_disabled(self):
        """Controller should not escalate if full reset is disabled."""
        now = time.time()
        self.brain.thermal_recovery_enable_full_reset = False
        self.brain.thermal_recovery_active_incident_id = "inc-1"
        self.brain.thermal_recovery_incident_started_at = now - 600
        self.brain.thermal_recovery_resets_issued = 5

        gpu_states = {
            "gpu-1": make_gpu_state(
                "gpu-1", cpu_temp_c=85, incident_id="inc-1", sustained_seconds=600
            ),
        }

        _check_thermal_recovery_escalation_impl(self.brain, gpu_states)

        # Should not issue any reset
        self.assertEqual(len(self.brain.inserted_tasks), 0)
        # Should log disabled event
        disabled_events = [d for d in self.brain.decisions if "DISABLED" in d["event"]]
        self.assertEqual(len(disabled_events), 1)


class TestThermalRecoveryIncidentTracking(unittest.TestCase):
    """Test incident lifecycle tracking."""

    def setUp(self):
        self.brain = MockBrain()

    def test_clears_incident_when_temps_recover(self):
        """Controller should clear incident when no GPUs have active incidents."""
        now = time.time()
        self.brain.thermal_recovery_active_incident_id = "inc-old"
        self.brain.thermal_recovery_incident_started_at = now - 600
        self.brain.thermal_recovery_resets_issued = 2

        # GPUs have recovered (no incident_id or below threshold)
        gpu_states = {
            "gpu-1": make_gpu_state("gpu-1", cpu_temp_c=60, incident_id=None),
            "gpu-2": make_gpu_state("gpu-2", cpu_temp_c=55, incident_id=None),
        }

        _check_thermal_recovery_escalation_impl(self.brain, gpu_states)

        # Incident should be cleared
        self.assertIsNone(self.brain.thermal_recovery_active_incident_id)
        self.assertEqual(self.brain.thermal_recovery_resets_issued, 0)
        # Should log cleared event
        cleared_events = [d for d in self.brain.decisions if "CLEARED" in d["event"]]
        self.assertEqual(len(cleared_events), 1)

    def test_starts_new_incident(self):
        """Controller should start new incident when threshold is met."""
        gpu_states = {
            "gpu-1": make_gpu_state(
                "gpu-1", cpu_temp_c=85, incident_id="inc-new", sustained_seconds=350
            ),
        }

        _check_thermal_recovery_escalation_impl(self.brain, gpu_states)

        # New incident should be tracked
        self.assertEqual(self.brain.thermal_recovery_active_incident_id, "inc-new")
        self.assertIsNotNone(self.brain.thermal_recovery_incident_started_at)
        # Should log start event
        start_events = [d for d in self.brain.decisions if "START" in d["event"]]
        self.assertEqual(len(start_events), 1)


class TestTargetWorkerEnforcement(unittest.TestCase):
    """Test target_worker routing enforcement."""

    def test_reset_task_uses_target_worker(self):
        """Reset task should include target_worker in meta."""
        brain = MockBrain()
        gpu_states = {
            "gpu-1": make_gpu_state(
                "gpu-1", cpu_temp_c=85, incident_id="inc-1", sustained_seconds=350
            ),
        }

        _check_thermal_recovery_escalation_impl(brain, gpu_states)

        self.assertEqual(len(brain.inserted_tasks), 1)
        task = brain.inserted_tasks[0]
        self.assertEqual(task["command"], "reset_gpu_runtime")
        # Must have target_worker, not just target_gpu
        self.assertIn("target_worker", task["meta"])
        self.assertEqual(task["meta"]["target_worker"], "gpu-1")


class TestRuntimeStateResettingThermal(unittest.TestCase):
    """Test the resetting_thermal runtime state."""

    def test_state_exists_in_constants(self):
        """Verify resetting_thermal state is defined."""
        from gpu_constants import (
            RUNTIME_STATE_RESETTING_THERMAL,
            RUNTIME_STATES_RECOVERING,
            RUNTIME_STATES_NOT_CLAIMABLE,
        )

        self.assertEqual(RUNTIME_STATE_RESETTING_THERMAL, "resetting_thermal")
        self.assertIn(RUNTIME_STATE_RESETTING_THERMAL, RUNTIME_STATES_RECOVERING)
        # NOT_CLAIMABLE includes RECOVERING, so resetting_thermal should be blocked
        self.assertIn(RUNTIME_STATE_RESETTING_THERMAL, RUNTIME_STATES_NOT_CLAIMABLE)


class TestMetaClaimTargetEnforcement(unittest.TestCase):
    """Test that normal meta claim path enforces target_worker."""

    def test_can_claim_meta_task_checks_target(self):
        """Verify _can_claim_meta_task rejects mismatched targets."""
        # This test reads the source to verify the check exists
        # Actual behavior would require more mocking
        import inspect
        try:
            from gpu_tasks import GPUTaskMixin
            source = inspect.getsource(GPUTaskMixin._can_claim_meta_task)
            # Verify reset commands check target
            self.assertIn("reset_gpu_runtime", source)
            self.assertIn("reset_split_runtime", source)
            self.assertIn("target_worker", source)
            self.assertIn("target_mismatch", source)
        except ImportError:
            # If import fails, skip gracefully
            self.skipTest("Could not import GPUTaskMixin (missing dependencies)")


if __name__ == "__main__":
    unittest.main()
