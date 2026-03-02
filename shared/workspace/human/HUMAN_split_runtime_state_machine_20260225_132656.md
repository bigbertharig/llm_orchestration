# HUMAN: Core GPU Agent Split/Single Runtime State Machine Hardening

## Why This Is Needed

Observed repeatedly during `github_analyzer` verify waves (14B split tasks):
- split pair port remains reachable, but expected split model is missing (`SPLIT_RECONCILE_WEDGED_PORT`)
- brain suppresses `load_split_llm` because only wedged split ports remain
- pair (`gpu-4`/`gpu-5`) falls back to single-GPU `load_llm` tasks while split verify queue remains pending
- result: reduced split capacity, long verify stalls, unpredictable recoveries

Recent example (Feb 25, 2026):
- `pair_4_5` port `11441` reachable without `qwen2.5-coder:14b`
- brain logs `SPLIT_RECONCILE_WEDGED_PORT` and suppresses further split loads

Plan-local mitigations are now in place (preflight reclaim before verify phases), but the root fix belongs in `shared/agents/*.py`.

## Constraints / Context

- `shared/agents/*.py` is currently protected by immutable rules, so this note is for a human-approved core change.
- Goal is fail-closed runtime ownership and clean transitions for both single and split runtimes.

## Desired Behavior (Authoritative State Machine)

A GPU should have explicit runtime states and enforce strict transitions.

### Per-GPU States (minimum)
- `cold`
- `loading_single`
- `ready_single`
- `loading_split`
- `ready_split`
- `unloading`
- `wedged` (requires reclaim/reset before reuse)
- `error_recoverable`

### Split Pair State (brain-visible, atomic)
For split groups (e.g. `pair_4_5`), the pair must also have a derived group state:
- `cold`
- `loading`
- `ready(model)`
- `unloading`
- `wedged`

Brain should consider a split group `ready` only when all checks pass (see below). Port reachability alone is insufficient.

## Required Invariants

1. **No load while loaded**
- If GPU is `ready_single` or `ready_split`, it must not accept `load_llm` or `load_split_llm`.
- Only allowed actions:
  - compatible work task
  - explicit unload task

2. **No work during transitions**
- No LLM work tasks while state is `loading_*` or `unloading`.
- Transition states must be short-lived and visible in heartbeat.

3. **Split transitions are atomic (pair-wise)**
- Split `load`/`unload` must reserve and transition both members together.
- If one member fails transition, mark group `wedged` and recover; do not leave one side apparently `ready_split`.

4. **Ready means verified model state, not process presence**
- A split group is `ready` only if:
  - split endpoint is reachable
  - target model is present on split endpoint
  - both member heartbeats agree on placement/group/model
  - reservation/ownership is active and consistent

5. **Unload returns to a clean baseline**
- `unload_llm` / `unload_split_llm` must verify postconditions:
  - no target model loaded
  - runtime ownership cleared
  - runtime_group_id cleared (for split)
  - stale reservation/lock artifacts cleared or marked terminal
  - no stale split process/port ownership left behind

## Load/Unload Meta-Task Handling Rules

### `load_llm` / `load_split_llm`
Before attempting load, run a worker-side preflight:
- verify worker is not already in `ready_*`
- verify no conflicting meta task is active for this GPU/group
- verify runtime port ownership/process state matches expectation
- reclaim/kill stale owned runtime process if present (explicit log)
- for split load: verify partner readiness + pair reservation lock acquired
- if preflight fails:
  - fail task with structured reason (do not half-load)
  - examples: `preflight_busy`, `stale_port_owner`, `partner_unavailable`, `split_reservation_conflict`, `wedged_requires_reclaim`

Important: do not silently drop the task. Brain needs explicit failure signals.

### `unload_llm` / `unload_split_llm`
Unload must be treated as a cleanup transaction:
- transition to `unloading`
- perform unload/reclaim actions
- verify clean postconditions
- if postconditions fail, set `wedged` and emit explicit error cause

## Work Task Claim Rules (Worker Side)

When GPU has runtime loaded:
- claim only compatible work tasks for the loaded runtime class and placement
- reject all `load_*` meta tasks while loaded

When GPU is `cold`:
- may claim `load_*` meta tasks or compatible cold-start work (if deferred-load path is allowed)
- if deferred-load path is kept, it must still respect the same preflight and state transitions

When GPU/pair is `wedged`:
- do not claim normal work
- prefer reclaim/unload/reset meta task
- if none available, advertise `wedged` in heartbeat so brain does not route split work to it

## Heartbeat / Telemetry Changes Needed

Add explicit fields (or strengthen existing semantics):
- `runtime_state` (per GPU)
- `runtime_state_updated_at`
- `runtime_transition_task_id`
- `runtime_transition_phase` (`preflight`, `loading`, `verify_ready`, `unloading`, `reclaiming`)
- `runtime_error_code` / `runtime_error_detail` (for wedged/error states)

For split members, ensure heartbeat reflects:
- `runtime_placement`
- `runtime_group_id`
- `split_runtime_owner`
- expected `loaded_model`

Brain should trust `runtime_state` over heuristics where possible.

## Brain-Side Changes (Core)

### Split Reconcile / Resource Decisions
Current issue:
- wedged port detection can lead to repeated suppression loops without active recovery.

Required behavior:
- after N consecutive `SPLIT_RECONCILE_WEDGED_PORT` detections for a group:
  - enqueue targeted `unload_split_llm` (or reclaim-reset task)
  - clear stale reservation
  - backoff briefly
  - then allow fresh `load_split_llm`
- do not continue indefinite suppression if demand remains and only wedged groups exist

Suggested knobs:
- `SPLIT_WEDGE_RECONCILE_THRESHOLD`
- `SPLIT_WEDGE_RECLAIM_COOLDOWN_S`
- `SPLIT_WEDGE_MAX_RECLAIMS_PER_HOUR`

### Meta-task arbitration
- Prevent brain from issuing `load_llm` to GPUs that are already `ready_split` unless an explicit unload happened first.
- Prevent simultaneous conflicting meta tasks on same GPU/group.

## Concrete Files Likely Involved (for Human Review)

Protected core agent code (cannot be edited by agent under current rules):
- `/home/bryan/llm_orchestration/shared/agents/gpu.py`
- `/home/bryan/llm_orchestration/shared/agents/brain_*` (resource decision / split reconcile paths)

Key areas to inspect:
- worker task claim eligibility for meta tasks vs loaded runtime state
- split load/unload implementation and postcondition checks
- heartbeat state publishing during transitions
- split reconcile wedged-port handling and retry/suppression policy

## What Has Already Been Implemented (Plan-Local Mitigation)

These changes are already live in plan code, but they are not a substitute for core fixes:
- `github_analyzer` verify waves now run a split preflight before 14B tasks
- preflight can:
  - clear stale split reservations
  - detect wedged idle split groups from heartbeats
  - queue targeted `unload_split_llm` for wedged groups
  - queue targeted `load_split_llm`
  - wait for verified ready split groups before releasing verify work

Files:
- `/home/bryan/llm_orchestration/shared/plans/shoulders/github_analyzer/scripts/preflight_split_capacity.py`
- `/home/bryan/llm_orchestration/shared/plans/shoulders/github_analyzer/plan.md`
- `/home/bryan/llm_orchestration/shared/plans/shoulders/github_analyzer_verify_benchmark/plan.md`

## Acceptance Criteria for Core Fix

1. A GPU with a loaded runtime never claims a `load_*` meta task.
2. Split groups cannot be considered ready solely because the split port responds.
3. `unload_split_llm` reliably returns the pair to a clean state (verified).
4. Wedged split groups are actively reclaimed after threshold, not indefinitely suppressed.
5. Under repeated verify waves, split pairs remain stable across sequential work tasks unless an explicit unload occurs.
6. Brain logs clearly distinguish:
   - `ready`
   - `loading`
   - `wedged`
   - `reclaiming`
   - `cooldown`

## Suggested Implementation Order (Human)

1. Worker runtime state machine + heartbeat fields in `shared/agents/gpu.py`
2. Enforce meta-task claim exclusions when `ready_*`
3. Harden split load/unload postcondition verification
4. Brain split reconcile thresholded reclaim policy
5. Integration test on `github_analyzer_verify_benchmark` (`county-map`) and one smaller repo

---

## Implementation Status (Feb 25, 2026)

**Status: IMPLEMENTED** - All core changes complete, pending integration testing.

### What Was Implemented

#### 1. Worker Runtime State Machine (`gpu.py`)
- Added state constants: `RUNTIME_STATE_COLD`, `RUNTIME_STATE_LOADING_SINGLE`, `RUNTIME_STATE_READY_SINGLE`, `RUNTIME_STATE_LOADING_SPLIT`, `RUNTIME_STATE_READY_SPLIT`, `RUNTIME_STATE_UNLOADING`, `RUNTIME_STATE_WEDGED`, `RUNTIME_STATE_ERROR_RECOVERABLE`
- Added tracking fields: `runtime_state`, `runtime_state_updated_at`, `runtime_transition_task_id`, `runtime_transition_phase`, `runtime_error_code`, `runtime_error_detail`
- Added helper methods: `_set_runtime_state()`, `_is_runtime_ready()`, `_is_runtime_transitioning()`, `_is_runtime_wedged()`, `_can_accept_load_task()`, `_can_accept_work_task()`, `_mark_wedged()`
- State transitions integrated into `load_model()`, `unload_model()`, `_set_split_runtime_loaded()`, `_clear_split_runtime_loaded()`, `_start_split_runtime()`

#### 2. Meta-Task Claim Enforcement (`gpu.py`)
- `_can_claim_meta_task()` now returns `(bool, reason)` tuple with strict state checks
- Preflight rejection for loads when in ready/transitioning/wedged states
- Unload tasks can be claimed even when wedged (for recovery)

#### 3. Work Task Claim Enforcement (`gpu.py`)
- `_can_accept_work_task()` is now enforced in `claim_tasks()` for all llm/script/cpu tasks
- No work tasks claimed during transitions or when wedged

#### 4. Postcondition Verification (`gpu.py`)
- `unload_model()` verifies postconditions and marks as wedged on failure
- `unload_split_llm` handling marks as wedged if listener/model still present after unload
- Reservation status only set to "unloaded" AFTER postconditions pass (uses transitional "unloading" status first)

#### 5. Heartbeat Updates (`gpu.py`)
- All new state machine fields included in heartbeat output

#### 6. Brain Thresholded Reclaim Policy (`brain_resources.py`)
- Added constants: `SPLIT_WEDGE_RECONCILE_THRESHOLD=3`, `SPLIT_WEDGE_RECLAIM_COOLDOWN_S=120`, `SPLIT_WEDGE_MAX_RECLAIMS_PER_HOUR=6`
- Added tracking in `brain.py`: `split_wedge_counts`, `split_wedge_last_reclaim_at`, `split_wedge_reclaims_this_hour`
- Added methods: `_increment_split_wedge_count()`, `_clear_split_wedge_count()`, `_can_reclaim_split_group()`, `_record_split_reclaim_attempt()`, `_should_trigger_wedge_reclaim()`
- `_reconcile_split_group_state()` tracks wedge counts and clears on healthy
- `_make_resource_decisions()` triggers `unload_split_llm` after threshold instead of indefinite suppression

### Post-Review Fixes (Code Review Findings)

Six issues identified and fixed:

1. **High: Scoped orphan-runner cleanup** - `_kill_orphan_ollama_runners()` now requires a `target_port` parameter and only kills runners associated with that port. Prevents accidentally killing healthy runners for other split groups.

2. **High: Restricted split pair prep port reclamation** - `_prepare_local_for_split_pairing()` now only reclaims the port if `is_launcher=True` OR `status != "loading"`. Prevents a late-joining partner from killing a launcher's already-started runtime.

3. **Medium: Work task state guard enforced** - `_can_accept_work_task()` is now called in the claim loop for all llm/script/cpu tasks, enforcing "no work during transitioning/wedged" invariant.

4. **Medium: Reservation status ordering fixed** - `unload_split_llm` now sets reservation to "unloading" first, performs cleanup, verifies postconditions, then sets "unloaded" (or "failed") based on result. Prevents misleading brain reconciliation.

5. **Medium: Orphan runner kill gated by can_reclaim_port** - `_prepare_local_for_split_pairing()` now only calls `_kill_orphan_ollama_runners()` when `can_reclaim_port` is true. Previously, orphan runners were killed unconditionally before the reclaim check, allowing a non-launcher partner to kill a launcher's healthy runtime via the orphan-runner path.

6. **Low: Dead code cleanup in orphan runner logic** - Removed unreachable `or not port_pids` branch in `_kill_orphan_ollama_runners()` (dead code after early return) and fixed misleading comments about "parent PID lineage" checking that wasn't implemented. Comments now accurately describe the actual logic: killing orphan runners that have the target port open (inherited socket).

### Files Modified

- `/home/bryan/llm_orchestration/shared/agents/gpu.py` (~150 lines added/modified)
- `/home/bryan/llm_orchestration/shared/agents/brain.py` (~5 lines added)
- `/home/bryan/llm_orchestration/shared/agents/brain_resources.py` (~80 lines added)

### Next Steps

1. **Integration testing** on `github_analyzer_verify_benchmark` with `county-map` repo
2. **Monitor** for `RUNTIME_STATE_TRANSITION` and `SPLIT_WEDGE_*` log patterns
3. **Validate** acceptance criteria 1-6 under repeated verify waves
4. **Consider refactoring** `gpu.py` (~3500 lines) into smaller modules:
   - `gpu_state.py` - State machine logic
   - `gpu_split.py` - Split runtime handling
   - `gpu_ollama.py` - Ollama management
   - `gpu.py` - Main agent loop and coordination

## Additional Runtime Robustness Issue (Feb 25, 2026): Thermal Pause Causes Misleading Deferred-Load Retries

### Observed Behavior

During `github_analyzer_verify_benchmark` runs:
- multiple `worker_review` slices entered retry loops with errors labeled:
  - `DEFERRED_MODEL_LOAD: worker_timeout ... queued_load_llm=True`
- at the same time, affected GPUs showed:
  - `model_loaded: true`
  - healthy local Ollama
  - `thermal_pause_active: true`

This indicates some retries are not true "model missing" problems.

### Likely Mechanism

Current thermal pause behavior uses `SIGSTOP` / `SIGCONT` on active worker subprocesses.

If a worker is paused while blocked in an Ollama HTTP request:
- the worker process stops advancing
- request timeout clocks continue externally
- the worker later reports a timeout
- worker logic may misclassify it as deferred-model-load timeout and queue `load_llm`

Result:
- misleading error messages
- unnecessary `load_llm` churn
- retries consumed on slices that were really thermally stalled

### Plan-Local Mitigation (Already Implemented)

`worker_review.py` timeout classification was improved to distinguish:
- `thermal_pause_timeout`
- `inference_timeout`

and to avoid queuing `load_llm` when the model is already present on the worker Ollama runtime.

This improves observability and reduces pointless load-meta traffic, but does **not** fully solve the underlying pause behavior.

### Core-Agent Fixes Needed (Human)

1. **Do not pause active LLM worker subprocesses mid-request with `SIGSTOP`**
- Prefer claim throttling only:
  - stop claiming new work during thermal pause
  - let in-flight workers finish naturally
- If pausing remains necessary, it must be coordinated with worker/runtime timeouts

2. **Separate thermal stall from model-load timeout in worker retry policy**
- Timeouts during thermal pause should be tagged/classified distinctly
- Avoid automatic `load_llm` enqueue for thermal/stall timeouts when model is loaded and healthy

3. **Add a thermal-aware timeout budget**
- If a worker is thermally paused, extend or suspend request timeout accounting
- Alternatively, mark the task as `paused_by_thermal` and resume without incrementing retry count

4. **Telemetry / diagnostics**
- Record per-task thermal pause overlap in task heartbeat/result metadata
- This enables debugging of "why did this timeout?" without inferring from global heartbeat snapshots

### Why This Matters

This directly affects reliability of long-running LLM phases:
- otherwise healthy slices can fail after repeated thermal pauses
- benchmark timing and failure signals become noisy
- operators see "deferred load" errors that are actually thermal stalls
