"""GPU agent constants.

Extracted from gpu.py to centralize configuration values and state definitions.
"""

# =============================================================================
# Task Classification
# =============================================================================
VALID_TASK_CLASSES = ['cpu', 'script', 'llm', 'meta']
DEFAULT_LLM_MIN_TIER = 1

# =============================================================================
# Split Runtime Configuration
# =============================================================================
# Split loads can spend significant time waiting on the global load lock and then
# warming a large model on the shared split runtime.
SPLIT_META_TIMEOUT_SECONDS = 900
SPLIT_LAUNCHER_HEARTBEAT_MAX_AGE_SECONDS = 45
# Fail-fast timeout: if all_joined && all_prepared but member_clean still missing
# after this many seconds, fail the reservation instead of waiting forever
SPLIT_MEMBER_CLEAN_STALL_TIMEOUT_SECONDS = 60

# Split runtime invariant check stabilization
# Grace window after split becomes ready - don't trigger port_model_missing during this period
SPLIT_READY_GRACE_WINDOW_SECONDS = 15
# Require N consecutive port_model_missing events within M seconds to clear
SPLIT_PORT_MODEL_MISS_CONSECUTIVE_COUNT = 3
SPLIT_PORT_MODEL_MISS_WINDOW_SECONDS = 30

# =============================================================================
# Global Model Load Lock
# =============================================================================
GLOBAL_MODEL_LOAD_OWNER_STALE_SECONDS = 45
GLOBAL_MODEL_LOAD_OWNER_HEARTBEAT_INTERVAL = 2

# =============================================================================
# Runtime State Machine
# =============================================================================
RUNTIME_STATE_COLD = "cold"
RUNTIME_STATE_LOADING_SINGLE = "loading_single"
RUNTIME_STATE_READY_SINGLE = "ready_single"
RUNTIME_STATE_LOADING_SPLIT = "loading_split"
RUNTIME_STATE_READY_SPLIT = "ready_split"
RUNTIME_STATE_UNLOADING = "unloading"
RUNTIME_STATE_WEDGED = "wedged"
RUNTIME_STATE_ERROR_RECOVERABLE = "error_recoverable"
# Recovery states - explicit auto-recovery in progress
RUNTIME_STATE_RECOVERING_SPLIT = "recovering_split"
RUNTIME_STATE_RECOVERING_SINGLE = "recovering_single"

RUNTIME_STATES_READY = {RUNTIME_STATE_READY_SINGLE, RUNTIME_STATE_READY_SPLIT}
RUNTIME_STATES_LOADING = {RUNTIME_STATE_LOADING_SINGLE, RUNTIME_STATE_LOADING_SPLIT}
RUNTIME_STATES_TRANSITIONING = {RUNTIME_STATE_LOADING_SINGLE, RUNTIME_STATE_LOADING_SPLIT, RUNTIME_STATE_UNLOADING}
RUNTIME_STATES_RECOVERING = {RUNTIME_STATE_RECOVERING_SPLIT, RUNTIME_STATE_RECOVERING_SINGLE}
# States that block task claiming
RUNTIME_STATES_NOT_CLAIMABLE = (
    RUNTIME_STATES_LOADING |
    RUNTIME_STATES_TRANSITIONING |
    RUNTIME_STATES_RECOVERING |
    {RUNTIME_STATE_WEDGED, RUNTIME_STATE_ERROR_RECOVERABLE}
)
# States that are stable (can accept work or are verified clean)
RUNTIME_STATES_STABLE = RUNTIME_STATES_READY | {RUNTIME_STATE_COLD}

# =============================================================================
# Timing
# =============================================================================
EXTERNAL_HEARTBEAT_INTERVAL = 30  # seconds - filesystem heartbeat to brain
ACTIVE_WORK_HEARTBEAT_INTERVAL = 5  # seconds - faster claims while work is active
INTERNAL_POLL_INTERVAL = 5        # seconds - check worker status internally
SIGNAL_CHECK_INTERVAL = 5         # seconds - check stop/abort signals
META_TASK_HEARTBEAT_INTERVAL = 5  # seconds - keep ownership visible during long meta work

# =============================================================================
# VRAM Budget
# =============================================================================
VRAM_BUDGET_RATIO = 0.8           # Use at most 80% of total VRAM
DEFAULT_CPU_VRAM_COST = 1024      # MB - virtual cost for CPU tasks to limit concurrency

# =============================================================================
# Runtime Mismatch Circuit Breaker
# =============================================================================
# If N mismatches occur within M seconds, worker enters wedged state
MISMATCH_CIRCUIT_BREAKER_COUNT = 3
MISMATCH_CIRCUIT_BREAKER_WINDOW_SECONDS = 300  # 5 minutes
MISMATCH_WEDGE_COOLDOWN_SECONDS = 120  # 2 minutes before auto-recovery attempt

# =============================================================================
# Auto-Recovery Configuration
# =============================================================================
# Wedged state triggers auto-recovery if no active tasks after this period
AUTO_RECOVERY_WEDGED_IDLE_THRESHOLD_SECONDS = 30
# Maximum time for recovery workflow before fallback to targeted meta tasks
AUTO_RECOVERY_TIMEOUT_SECONDS = 60
# Retry cleanup once with this backoff before emitting fallback tasks
AUTO_RECOVERY_CLEANUP_RETRY_DELAY_SECONDS = 5
# VRAM threshold (MB) below which a GPU is considered "cold" for recovery verification
AUTO_RECOVERY_COLD_VRAM_THRESHOLD_MB = 500

# =============================================================================
# Split Pair Quarantine
# =============================================================================
# Failures within rolling window trigger quarantine
SPLIT_QUARANTINE_FAILURE_COUNT = 3
SPLIT_QUARANTINE_FAILURE_WINDOW_SECONDS = 600  # 10 minutes
# Quarantine cooldown period before pair is eligible again
SPLIT_QUARANTINE_COOLDOWN_SECONDS = 900  # 15 minutes

# =============================================================================
# Readiness State Machine + Token
# =============================================================================
# Stability gate configuration - consecutive probes required before ready
READY_STABLE_PROBE_COUNT = 3
READY_STABLE_PROBE_INTERVAL_SECONDS = 2
# Minimum token age before a worker can claim split work
READY_MIN_AGE_SECONDS = 10

# Phased mismatch recovery thresholds
ATTESTATION_MISS_SOFT_FAIL_THRESHOLD = 1  # First miss: soft-fail + requeue
ATTESTATION_MISS_HARD_FAIL_THRESHOLD = 2  # Second consecutive miss: mark failed

# Split reservation statuses (state machine)
SPLIT_RESERVATION_STATUSES = {
    "waiting_partner",  # Initial: launcher waiting for follower to join
    "joining",          # Both members joining
    "loading",          # Launcher starting split runtime
    "warming",          # NEW: Warmup in progress
    "ready_stabilizing",  # NEW: Post-warmup stability gate (probe loop)
    "ready",            # Split runtime ready for work
    "failed",           # Load failed
    "unloading",        # Unload in progress
    "unloaded",         # Unload complete
    "expired",          # Reservation timed out
}

# States where split load is still in progress (block work tasks)
SPLIT_RESERVATION_LOADING_STATES = {
    "waiting_partner",
    "joining",
    "loading",
    "warming",
    "ready_stabilizing",
}
