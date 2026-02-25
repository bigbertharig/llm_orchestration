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

