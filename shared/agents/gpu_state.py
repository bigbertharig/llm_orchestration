"""GPU agent runtime state machine mixin.

Extracted from gpu.py to isolate state machine logic for runtime transitions.
"""

import time
from datetime import datetime
from typing import Optional

from gpu_constants import (
    AUTO_RECOVERY_WEDGED_IDLE_THRESHOLD_SECONDS,
    MISMATCH_WEDGE_COOLDOWN_SECONDS,
    RUNTIME_STATE_COLD,
    RUNTIME_STATE_RECOVERING_SINGLE,
    RUNTIME_STATE_RECOVERING_SPLIT,
    RUNTIME_STATE_WEDGED,
    RUNTIME_STATES_READY,
    RUNTIME_STATES_RECOVERING,
    RUNTIME_STATES_TRANSITIONING,
)


class GPUStateMixin:
    """Mixin providing runtime state machine methods."""

    def _set_runtime_state(
        self,
        new_state: str,
        *,
        task_id: Optional[str] = None,
        phase: Optional[str] = None,
        error_code: Optional[str] = None,
        error_detail: Optional[str] = None,
    ):
        """Transition to a new runtime state with full audit trail."""
        old_state = self.runtime_state
        self.runtime_state = new_state
        self.runtime_state_updated_at = datetime.now().isoformat()
        self.runtime_transition_task_id = task_id
        self.runtime_transition_phase = phase
        self.runtime_error_code = error_code
        self.runtime_error_detail = error_detail

        # Keep legacy state field in sync
        if new_state in RUNTIME_STATES_READY:
            self.state = "hot"
        elif new_state == RUNTIME_STATE_COLD:
            self.state = "cold"
        # Transitioning/wedged states keep previous legacy state

        if old_state != new_state:
            self.logger.info(
                f"RUNTIME_STATE_TRANSITION {old_state}->{new_state} "
                f"task={task_id or '-'} phase={phase or '-'} "
                f"error={error_code or '-'}"
            )

    def _is_runtime_ready(self) -> bool:
        """Check if GPU is in a ready state (has loaded runtime)."""
        return self.runtime_state in RUNTIME_STATES_READY

    def _is_runtime_transitioning(self) -> bool:
        """Check if GPU is in a transitioning state (loading/unloading)."""
        return self.runtime_state in RUNTIME_STATES_TRANSITIONING

    def _is_runtime_recovering(self) -> bool:
        """Check if GPU is in a recovery state."""
        return self.runtime_state in RUNTIME_STATES_RECOVERING

    def _is_runtime_wedged(self) -> bool:
        """Check if GPU is in a wedged state requiring reclaim.

        Implements cooldown recovery: if wedged due to mismatch circuit breaker
        and cooldown has elapsed, auto-recover to cold state.
        """
        if self.runtime_state != RUNTIME_STATE_WEDGED:
            return False

        # Check if cooldown has elapsed (auto-recovery)
        cooldown_until = getattr(self, 'runtime_recovery_cooldown_until', None)
        if cooldown_until is not None:
            now = datetime.now().timestamp()
            if now >= cooldown_until:
                # Cooldown elapsed - auto-recover to cold
                self.logger.info(
                    f"WEDGE_COOLDOWN_RECOVERY worker={self.name} "
                    f"cooldown_elapsed transitioning to cold"
                )
                self._set_runtime_state(
                    RUNTIME_STATE_COLD,
                    phase="wedge_cooldown_recovery",
                )
                # Clear mismatch tracking
                self.mismatch_timestamps = []
                self.runtime_mismatch_count = 0
                self.runtime_recovery_cooldown_until = None
                try:
                    self._write_heartbeat()
                except Exception:
                    pass
                return False

        return True

    def _can_accept_load_task(self) -> tuple[bool, str]:
        """Check if GPU can accept a load task. Returns (can_accept, reason)."""
        if self._is_runtime_ready():
            return False, f"already_loaded:{self.runtime_state}"
        if self._is_runtime_transitioning():
            return False, f"transitioning:{self.runtime_state}"
        if self._is_runtime_wedged():
            return False, "wedged_requires_reclaim"
        return True, ""

    def _can_accept_work_task(self) -> tuple[bool, str]:
        """Check if GPU can accept a work task. Returns (can_accept, reason).

        Includes split pair loading lock: if this GPU is a member of any
        split reservation currently loading (status in waiting_partner/joining/loading),
        reject work tasks to prevent interference. Both launcher and follower
        are blocked during split load.

        Hard reject claiming when runtime_state in:
        - loading_single, loading_split, unloading (transitioning)
        - recovering_split, recovering_single (auto-recovery in progress)
        - wedged (requires explicit recovery)
        """
        if self._is_runtime_transitioning():
            return False, f"transitioning:{self.runtime_state}"
        if self._is_runtime_recovering():
            return False, f"recovering:{self.runtime_state}"
        if self._is_runtime_wedged():
            return False, "wedged"

        # Check split pair loading lock (blocks both launcher and follower)
        in_pair_lock, lock_info = self._is_in_split_pair_loading_lock()
        if in_pair_lock:
            return False, self._format_split_pair_lock_reason(lock_info)

        return True, ""

    def _mark_wedged(self, error_code: str, error_detail: str, task_id: Optional[str] = None):
        """Mark GPU as wedged due to failed operation."""
        self._set_runtime_state(
            RUNTIME_STATE_WEDGED,
            task_id=task_id,
            error_code=error_code,
            error_detail=error_detail,
        )
        # Track when we entered wedged state for auto-recovery trigger
        self.wedged_since = time.time()
        self.logger.error(
            f"RUNTIME_WEDGED worker={self.name} error={error_code}: {error_detail}"
        )

    def _should_trigger_auto_recovery(self) -> tuple[bool, str]:
        """Check if auto-recovery should be triggered.

        Auto-recovery triggers when:
        - runtime_state == wedged
        - No active tasks (active_workers == 0, no active_meta_task)
        - Wedged for longer than threshold

        Returns (should_trigger, reason).
        """
        if self.runtime_state != RUNTIME_STATE_WEDGED:
            return False, "not_wedged"

        # Don't trigger if we're already recovering
        if self._is_runtime_recovering():
            return False, "already_recovering"

        # Check for active work
        if self.active_workers:
            return False, "active_workers"
        if self.active_meta_task:
            return False, "active_meta_task"

        # Check how long we've been wedged
        wedged_since = getattr(self, 'wedged_since', None)
        if wedged_since is None:
            # No tracking - set it now and wait
            self.wedged_since = time.time()
            return False, "wedge_tracking_started"

        idle_seconds = time.time() - wedged_since
        if idle_seconds < AUTO_RECOVERY_WEDGED_IDLE_THRESHOLD_SECONDS:
            return False, f"idle_threshold_not_reached:{idle_seconds:.0f}s"

        # Determine recovery type based on runtime placement
        placement = str(getattr(self, 'runtime_placement', '')).strip()
        if placement == "split_gpu":
            return True, "wedged_split_idle"
        else:
            return True, "wedged_single_idle"

    def _mark_recovering(self, recovery_type: str, reason: str, task_id: Optional[str] = None):
        """Mark GPU as entering recovery state."""
        if recovery_type == "split":
            new_state = RUNTIME_STATE_RECOVERING_SPLIT
        else:
            new_state = RUNTIME_STATE_RECOVERING_SINGLE

        self._set_runtime_state(
            new_state,
            task_id=task_id,
            phase=f"auto_recovery:{reason}",
        )
        self.recovery_started_at = time.time()
        self.logger.info(
            f"AUTO_RECOVERY_TRIGGER worker={self.name} type={recovery_type} reason={reason}"
        )
