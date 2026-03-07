# Thermal Recovery Upgrade Plan (Core Integration)

Date: 2026-03-04
Owner: orchestration core
Status: In Progress (re-audited 2026-03-05; code paths/tests complete; runtime escalation + brain observability proof still open)

## Goal
Upgrade existing thermal gating so sustained CPU overheat does not deadlock throughput.

Current behavior pauses work under thermal pressure. New behavior must:
1. Detect sustained overheat incidents.
2. Perform targeted runtime resets one GPU at a time.
3. Escalate to full orchestrator reset after bounded attempts.
4. Stay fully inside core orchestration state machines (no sidecar patch scripts).

## Non-Goals
1. No new dashboard-only recovery logic.
2. No hidden auto-retry loops outside brain/gpu state machines.
3. No compatibility shim layers.

## Why This Is Needed
Observed failure mode:
1. CPU temp remains above warning threshold for long periods.
2. `thermal_pause_active` blocks work claiming.
3. Worker runtimes/runner processes can keep CPU hot while queue stalls.
4. Batch appears stuck with high pending LLM tasks and low task completion.

## Design Principles
1. Extend existing thermal logic, do not replace it.
2. Keep one authoritative control loop: Brain decides resets; GPU executes deterministic reset tasks.
3. Fail closed: recovery states are non-claimable for regular work.
4. Rate-limit all recovery actions to prevent thrashing.

## Implementation Audit (2026-03-04, pass 2)

This section reflects direct code inspection in:
- `/home/bryan/llm_orchestration/shared/agents/gpu.py`
- `/home/bryan/llm_orchestration/shared/agents/gpu_thermal.py`
- `/home/bryan/llm_orchestration/shared/agents/gpu_tasks.py`
- `/home/bryan/llm_orchestration/shared/agents/brain.py`
- `/home/bryan/llm_orchestration/shared/agents/brain_resources.py`
- `/home/bryan/llm_orchestration/shared/agents/brain_dispatch.py`
- `/home/bryan/llm_orchestration/shared/agents/config.json`
- `/home/bryan/llm_orchestration/shared/agents/config.template.json`
- `/home/bryan/llm_orchestration/shared/agents/tests/test_thermal_recovery.py`

### Implemented
1. GPU incident fields are in heartbeat (`thermal_overheat_started_at`, `thermal_overheat_sustained_seconds`, `thermal_overheat_incident_id`, `thermal_recovery_reset_count`, `thermal_recovery_last_reset_at`).
2. Thermal incident lifecycle logging is present (`THERMAL_INCIDENT_START`, `THERMAL_INCIDENT_CLEARED`).
3. Brain thermal controller exists (`_check_thermal_recovery_escalation`) and is called from resource decisions.
4. Targeted reset and escalation counters/rate limits are present.
5. `reset_gpu_runtime` and `reset_split_runtime` handlers exist in GPU meta-task executor.
6. `orchestrator_full_reset` command exists and is handled by brain dispatch.
7. `resetting_thermal` runtime state exists and is non-claimable.
8. `thermal_recovery` config block exists in both template and live config.

### Critical Gaps Status (after re-audit)
1. Targeted reset routing mismatch: Fixed.
   - Emergency thermal scan checks `target_worker` first, with `target_gpu` fallback.
   - Normal meta-claim path now also enforces `target_worker`/`target_gpu` for reset tasks.
2. Full reset restart handshake: Fixed.
   - `startup.py` watches `brain/restart_workers.signal`, restarts workers, and consumes the signal file as ack.
3. Full reset cleanup split ports: Fixed.
   - Full reset port sweep includes `11435-11441` (worker + split ports).
4. Full reset stop-wait heartbeat key: Fixed.
   - Stop-wait freshness uses `last_updated` primary, `timestamp` fallback.
5. Test verification: Improved but not final.
   - Latest run: `Ran 13 tests ... OK (skipped=1)`.
   - Remaining skip is dependency-related import gating for one target-enforcement test.
6. Runtime/observability proof: Not complete.
   - Worker-side incident lifecycle evidence now exists in `/media/bryan/shared/logs/startup-manual.log` (`THERMAL_INCIDENT_START` / `THERMAL_INCIDENT_CLEARED`).
   - Recent incidents are short-lived (`duration` mostly `0-5s`) with `resets_issued=0`; targeted reset/escalation path has not been exercised.
   - `brain_decisions.log` still has no `THERMAL_TARGETED_RESET_*`, `THERMAL_FULL_RESET_TRIGGERED`, or `THERMAL_INCIDENT_BRAIN_*` events.
   - Gate C/D still require deliberate sustained-overheat exercise and capture.

### Non-Critical Mismatches (doc vs code)
1. `thermal_recovery_state` field is documented but not implemented.
2. Doc previously referenced `targeted_reset_cooldown_seconds`; code uses `same_gpu_reset_backoff_seconds`.
3. Doc referenced `brain_tasks.py`/`executor.py`; implementation is in `brain_dispatch.py`.

## Core Changes

### 1) GPU thermal incident tracking
Files:
- `/home/bryan/llm_orchestration/shared/agents/gpu_thermal.py`
- `/home/bryan/llm_orchestration/shared/agents/gpu.py`

Add heartbeat fields:
1. `thermal_overheat_started_at`
2. `thermal_overheat_sustained_seconds`
3. `thermal_overheat_incident_id`
4. `thermal_recovery_reset_count`
5. `thermal_recovery_last_reset_at`

Behavior:
1. Start incident timer when CPU temp >= warning threshold continuously.
2. Clear incident when temps recover below resume margin for configured hold period.
3. Persist incident counters in heartbeat for Brain visibility.

### 2) Emergency meta-task claim path during thermal pause
File:
- `/home/bryan/llm_orchestration/shared/agents/gpu_tasks.py`

Current `claim_tasks()` exits early when `thermal_pause_active`.

Change:
1. Keep normal work blocked.
2. Allow explicit emergency meta commands while paused:
   - `reset_gpu_runtime`
   - `reset_split_runtime`
3. Emergency commands run local reset routines and return GPU to verified cold state.

### 3) Brain thermal recovery controller
File:
- `/home/bryan/llm_orchestration/shared/agents/brain_resources.py`

Add policy loop inside `_make_resource_decisions()`:
1. Detect sustained incident (`cpu_overheat_trigger_seconds`, default 300s).
2. Every `targeted_reset_interval_seconds` (default 60s), issue one targeted reset task.
3. Target selection: hottest eligible GPU first.
4. Max targeted resets per incident: `max_targeted_resets_per_incident` (default 5).
5. If still overheated after max resets, issue orchestrator full reset escalation action.

Eligibility filter for targeted reset:
1. GPU heartbeat fresh.
2. Not already in recovery/resetting runtime state.
3. Not currently mid critical meta transition.
4. Not reset within short cooldown window.

### 4) Full reset escalation as first-class orchestration action
Files:
- `/home/bryan/llm_orchestration/shared/agents/brain.py`
- `/home/bryan/llm_orchestration/shared/agents/brain_dispatch.py`

Add action `orchestrator_full_reset`:
1. Quiesce new work release briefly.
2. Execute full reset through controlled brain action.
3. Wait for heartbeat staleness OR explicit stop-ack.
4. Force-clean worker and split ports.
5. Deterministically restart workers.
6. Resume scheduling.

Requirements:
1. Full reset action is deduplicated.
2. Full reset action has cooldown and explicit failure logging.

### 5) Runtime states for recovery
Files:
- `/home/bryan/llm_orchestration/shared/agents/gpu_constants.py`
- `/home/bryan/llm_orchestration/shared/agents/gpu_state.py`

Add/confirm states:
1. `resetting_thermal`
2. Existing recovering states remain non-claimable.

Contract:
1. Any recovery/reset state blocks regular work claim.
2. State exits only on verified cold checks.

### 6) Config policy block
Files:
- `/home/bryan/llm_orchestration/shared/agents/config.template.json`
- `/home/bryan/llm_orchestration/shared/agents/config.json`

Add `thermal_recovery` config:
1. `cpu_overheat_trigger_seconds` (300)
2. `targeted_reset_interval_seconds` (60)
3. `max_targeted_resets_per_incident` (5)
4. `same_gpu_reset_backoff_seconds` (120)
5. `full_reset_cooldown_seconds` (300)
6. `enable_full_reset_escalation` (true)

## Event Logging
Files:
- `/home/bryan/llm_orchestration/shared/agents/brain_resources.py`
- `/home/bryan/llm_orchestration/shared/agents/gpu_thermal.py`

Emit structured events:
1. `THERMAL_INCIDENT_START`
2. `THERMAL_TARGETED_RESET_ISSUED`
3. `THERMAL_TARGETED_RESET_RESULT`
4. `THERMAL_INCIDENT_ESCALATE_FULL_RESET`
5. `THERMAL_INCIDENT_CLEARED`

Each event includes:
1. `incident_id`
2. `gpu_name` (if targeted)
3. temp snapshot
4. reset counters
5. queue/processing counts

## Safety Constraints
1. At most one targeted reset issued per interval.
2. No duplicate targeted reset task for same GPU while pending/processing.
3. No full reset while one is already in progress.
4. Incident cooldown after full reset to avoid loops.

## Remaining Fix Tasks (authoritative)

### Task 1: Unify target routing keys for thermal reset tasks
Files:
- `/home/bryan/llm_orchestration/shared/agents/brain_resources.py`
- `/home/bryan/llm_orchestration/shared/agents/gpu_tasks.py`

Changes:
1. Standardize on one key: `target_worker`.
2. In GPU claim logic, honor `target_worker` first, then optional legacy `target_gpu`.
3. Apply targeting check in both:
   - emergency thermal scan path
   - normal meta claim path
4. Add explicit reject reason logs for target mismatch.

Status: Implemented in both paths.

Exit criteria:
1. Targeted reset task can only be claimed by intended worker in both emergency and normal claim paths.
2. No cross-GPU accidental claiming during thermal pause.

### Task 2: Make full-reset restart deterministic
Files:
- `/home/bryan/llm_orchestration/shared/agents/brain_dispatch.py`
- launcher/startup owner (where worker processes are supervised)

Changes:
1. Replace orphan `restart_workers.signal` file with a real restart action that has a known consumer.
2. If restart remains file-based, add an explicit watcher and ack protocol.
3. Full reset must return `success=false` if restart path is unavailable.

Status: Implemented in `startup.py` via `restart_workers.signal` watcher and `_restart_gpu_workers()`.

Exit criteria:
1. Full reset always transitions system to healthy heartbeating workers without manual intervention.
2. Restart signal file is consumed/cleared as explicit ack.

### Task 3: Full-reset cleanup must include split ports
Files:
- `/home/bryan/llm_orchestration/shared/agents/brain_dispatch.py`

Changes:
1. Expand force-kill set to include split ports (`11440-11441`).
2. Include kill telemetry per port.
3. Verify no leftover listeners before returning success.

Status: Implemented in `brain_dispatch.py` (`11435-11441` sweep).

Exit criteria:
1. No worker/split Ollama listeners remain after full reset cleanup stage.

### Task 4: Strengthen automated verification
Files:
- `/home/bryan/llm_orchestration/shared/agents/tests/test_thermal_recovery.py`
- test runner/CI hook for agents tests

Changes:
1. Remove dependency-based skip gate from core logic tests (or install deps in test environment).
2. Add concrete tests for:
   - target_worker enforcement
   - emergency claim path under thermal pause
   - full reset restart path success/failure behavior
   - split port cleanup during full reset

Status: Partially complete.

Observed:
1. `python3 /media/bryan/shared/agents/tests/test_thermal_recovery.py`
2. Result: `Ran 13 tests ... OK (skipped=1)`.
3. Remaining skipped test: `TestMetaClaimTargetEnforcement.test_can_claim_meta_task_checks_target`
4. Skip reason: `Could not import GPUTaskMixin (missing dependencies)`.

Next required changes for closure:
1. Gate B update completed: import-gated skip removed; full suite now passes without skips.
2. Remaining closure work is Gate C/D runtime evidence, not additional unit test plumbing.

Exit criteria:
1. Thermal test suite runs with meaningful coverage (no blanket skips).

### Task 5: Fix full-reset stop-wait heartbeat key
Files:
- `/home/bryan/llm_orchestration/shared/agents/brain_dispatch.py`

Changes:
1. Stop-wait logic must use `last_updated` as primary freshness field.
2. Allow fallback to `timestamp` only for backward compatibility if needed.
3. Add explicit logging when heartbeat freshness field is missing.

Status: Implemented in `brain_dispatch.py` (`last_updated` primary, `timestamp` fallback).

Exit criteria:
1. Full reset correctly distinguishes active vs stopped GPU workers during wait.

## Rollout Plan

### Phase A: Instrumentation only
1. Add heartbeat incident fields.
2. Add logs/events.
3. No automatic resets yet.

Exit criteria:
1. Sustained incidents visible in heartbeat and logs.
2. No behavior change in task claiming.

### Phase B: Targeted reset automation
1. Enable one-per-minute targeted reset policy.
2. Emergency meta-task claim path active.

Exit criteria:
1. During sustained overheat, resets are issued exactly at policy cadence.
2. Queue resumes without full reset in common cases.

### Phase C: Full reset escalation
1. Enable post-5-reset escalation path.
2. Add cooldown and dedupe protections.

Exit criteria:
1. On persistent overheat, full reset triggers once and recovers heartbeats.
2. No repeated full-reset storms.

## Test Plan

Unit tests:
1. Brain target selection picks hottest eligible GPU.
2. Rate-limiter enforces one reset per interval.
3. Escalation after max targeted resets.
4. Emergency meta-task claim allowed during thermal pause.

Integration tests:
1. Synthetic high CPU temp with queued LLM tasks.
2. Verify targeted reset sequence and queue progress.
3. Verify escalation to full reset when temps remain high.
4. Verify recovery returns to normal scheduling.

Regression checks:
1. Normal thermal pause behavior unchanged when no incident.
2. Split load/unload sequencing remains intact.
3. No duplicate meta-task floods.

## Open Questions
1. Should targeted resets prefer idle hot GPUs first or hottest overall (current proposal: hottest eligible)?
2. Should full reset clear active batch or preserve and continue after restart?
3. Should CPU threshold use warning or critical for incident trigger in production defaults?

## Current Completion Status
1. Incident detection/logging: Implemented.
2. Targeted reset cadence logic: Implemented.
3. Full reset escalation trigger: Implemented.
4. Deterministic restart handshake: Implemented.
5. Split-port cleanup in full reset: Implemented.
6. Targeted reset routing: Implemented in emergency + normal claim paths.
7. Verified test coverage: Complete (`python3 -m unittest discover -s /home/bryan/llm_orchestration/shared/agents/tests -p 'test_*.py' -v` -> `Ran 66 tests ... OK`).
8. Runtime integration proof: Not complete.
9. Observability evidence continuity: Not complete.

## Verification Snapshot (2026-03-05)
Direct checks performed:
1. `gpu_tasks.py`: emergency scan and normal claim paths both enforce `target_worker` with `target_gpu` fallback.
2. `startup.py`: `_check_restart_workers_signal()` + `_restart_gpu_workers()` present and main loop polling active.
3. `brain_dispatch.py`: stop-wait uses `last_updated` then `timestamp`; cleanup sweep covers `11435-11441`.
4. Thermal events present across modules (code-level presence):
   - `THERMAL_INCIDENT_START`
   - `THERMAL_INCIDENT_CLEARED`
   - `THERMAL_TARGETED_RESET_ISSUED`
   - `THERMAL_TARGETED_RESET_RESULT`
   - `THERMAL_FULL_RESET_TRIGGERED`
   - `THERMAL_INCIDENT_BRAIN_CLEARED`
5. Test command run:
   - `python3 -m unittest discover -s /home/bryan/llm_orchestration/shared/agents/tests -p 'test_*.py' -v`
   - Result: `Ran 66 tests ... OK`.
6. Log evidence check:
   - `brain_decisions.log` search for thermal lifecycle events returned no matching lines in current sample window.
   - Runtime/observability gates remain open until incident path is exercised and captured.

## Remaining Required Work (authoritative)
1. Complete Gate B:
   - Completed (`66` tests passing, no skips).
2. Complete Gate C:
   - run a controlled sustained-overheat scenario (or equivalent simulation),
   - verify bounded targeted resets and escalation behavior.
3. Complete Gate D:
   - capture and attach decision-log evidence showing incident_id continuity across:
     - start
     - targeted reset issued/result
     - optional full reset trigger
     - incident cleared.

## Finish Criteria (Must All Pass)

This work is only considered finished when every gate below is satisfied.
No partial completion, no skipped tests, no "looks good" sign-off.

### Gate A: Code Correctness (static requirements)
1. Targeted thermal reset routing is enforced in both claim paths:
   - emergency thermal scan path
   - normal meta-task claim path
2. Routing key contract is explicit:
   - primary key: `target_worker`
   - legacy fallback: `target_gpu` (optional, compatibility only)
3. Full reset cleanup covers both worker and split ports:
   - worker: `11435-11439`
   - split: `11440-11441`
4. Full reset stop-wait heartbeat freshness uses:
   - primary: `last_updated`
   - fallback: `timestamp`
5. Full reset restart has a real consumer and acknowledgment:
   - launcher/startup consumes restart signal
   - signal is removed as ack

### Gate B: Automated Tests (required)
1. Thermal test suite runs with no blanket skips.
2. Required command:
   - `python3 /home/bryan/llm_orchestration/shared/agents/tests/test_thermal_recovery.py`
3. Pass condition:
   - all required tests execute and pass
   - `skipped=0` for core thermal logic tests
4. Required test coverage includes:
   - target-worker routing enforcement (both claim paths)
   - reset interval rate limiting
   - full-reset escalation threshold and cooldown
   - restart signal consumption path
   - split-port cleanup verification
   - heartbeat freshness key behavior

### Gate C: Runtime Validation (integration)
1. Sustained-overheat scenario (simulated or real) produces:
   - incident start log
   - bounded targeted resets (policy cadence)
   - either recovery or one escalated full reset after max attempts
2. After full reset:
   - workers are heartbeating again
   - queue resumes draining
   - no stale listeners remain on `11435-11441`
3. No repeated reset storms:
   - full-reset cooldown is respected
   - no duplicate full-reset tasks issued

Current state (2026-03-05 check):
1. Partially met: incident start/clear observed.
2. Not met: no targeted reset and no full reset observed yet (`resets_issued=0` across sampled incidents).
3. Action required: run a controlled sustained overheat test that exceeds `cpu_overheat_trigger_seconds` (default 300s).

Additional live finding (2026-03-04 ~22:55-22:58 local):
1. `gpu-2` remained `cold` with a queued targeted `load_llm` task pending (`batch_id=system`) while other GPUs kept processing normal slices.
2. Brain repeatedly warned:
   - `load_llm task available for ... but GPUs ['gpu-2'] still cold`
3. Worker logs show repeated claim suppression on `gpu-2`:
   - `TASKS_THERMAL_PAUSED: skip claiming new work ...`
4. Interpretation:
   - Thermal pause gating is active and effective.
   - The specific targeted recovery path for that cold GPU was blocked by thermal pause windows, with no higher-level corrective action observed.

### Gate D: Observability Evidence
1. Decision log contains these events with coherent metadata:
   - `THERMAL_INCIDENT_START`
   - `THERMAL_TARGETED_RESET_ISSUED`
   - `THERMAL_TARGETED_RESET_RESULT`
   - `THERMAL_FULL_RESET_TRIGGERED` (when escalation path is exercised)
   - `THERMAL_INCIDENT_BRAIN_CLEARED`
2. Evidence must include:
   - incident_id continuity
   - reset counters
   - target gpu names
   - queue/processing context at time of action

Current state (2026-03-05 check):
1. Partially met:
   - Worker logs show incident continuity (`incident_id`) and clear events.
2. Not met:
   - No brain-level thermal recovery events recorded for targeted/full-reset path.
3. Action required:
   - Capture one complete brain-side incident lifecycle containing:
     `THERMAL_INCIDENT_BRAIN_START` -> `THERMAL_TARGETED_RESET_ISSUED` -> `THERMAL_TARGETED_RESET_RESULT` -> (`THERMAL_FULL_RESET_TRIGGERED` when applicable) -> `THERMAL_INCIDENT_BRAIN_CLEARED`.

Additional observability gap from live run:
1. Worker-side evidence exists for pause/incident transitions (`TASKS_THERMAL_PAUSED`, `THERMAL_INCIDENT_START/CLEARED`).
2. Brain-side evidence is still absent for the same window despite blocked progress on a targeted `load_llm`.
3. Required follow-up:
   - Add/verify brain decision emission at the point where a targeted thermal-blocked meta task remains pending beyond threshold (before full escalation), so the operator can see why progress is constrained.

### Gate E: Documentation Integrity
1. This document's status section matches observed behavior and test outputs.
2. Any remaining known gap is listed as open and not marked done.
3. No claim of completion is allowed while any gate above is red.
