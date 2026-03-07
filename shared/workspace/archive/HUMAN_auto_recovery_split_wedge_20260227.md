# HUMAN: Auto-Recovery for Wedged Split Runtime Failures

Date: 2026-02-27
Owner: Human follow-up required
Status: **IMPLEMENTED** (2026-02-27)
Scope: GPU runtime state machine + split reservation lifecycle + recovery orchestration

## Implementation Summary

Implemented in:
- `gpu_constants.py` - Added recovery states and quarantine constants
- `gpu_state.py` - Added recovery state checks and auto-recovery trigger detection
- `gpu_split.py` - Added full auto-recovery workflow (Stages A-E) and quarantine tracking
- `brain_resources.py` - Added quarantine-aware split pair filtering
- `gpu.py` - Integrated auto-recovery trigger into main loop

## Problem Statement

Observed failure mode:
- Split load starts (`load_split_llm`) for pair (example: `pair_1_3`), then warmup crashes (`rc=-9`) or port/model invariant fails.
- Reservation transitions to failed, but one or both members remain in stale split ownership state (`runtime_state=wedged`, `loaded_model=qwen2.5-coder:14b`, low VRAM, no active task).
- Workers stop claiming normal 7B tasks because they are no longer in claimable clean state.
- No automatic recovery path consistently returns both members to verified `cold`.

Goal:
- System self-heals without human intervention and restores workers to claimable state after split failures.

## Authoritative Policy

1. Split failure recovery is pair-coordinated, not per-worker best-effort.
2. A worker in `wedged` with no active task is a recovery candidate, never a steady state.
3. No worker may re-enter task claiming until `verified_cold` postconditions pass.
4. Recovery must be explicit in logs and heartbeat state (machine-readable reason codes).

## Required Runtime States

Add/standardize these states:
- `recovering_split`
- `recovering_single`
- `cold` (verified clean)
- `wedged` (transient only; must auto-heal or escalate)

Optional terminal for repeated failures:
- `quarantined_split_pair`

## Recovery Triggers

Any of these MUST trigger auto-recovery workflow:
- Split warmup process exit before ready (`rc != 0`, especially `rc=-9`).
- Split warmup timeout.
- Split invariant clear event (`port_model_missing`, reservation mismatch, runtime liveness failure).
- Local Ollama health fails while `runtime_placement=split_gpu`.
- Worker heartbeat in `wedged` with `active_tasks=[]` older than threshold.

## Auto-Recovery Workflow (Authoritative)

### Stage A: Pair Lock + Mark Recovering
- Acquire split reservation lock.
- Set reservation status: `failed_recoverable` with reason code.
- Mark both members as `recovering_split` in heartbeat context.

### Stage B: Coordinated Hard Cleanup (Both Members)
For each member in pair:
- Stop split listener port process.
- Kill orphan llama runners tied to pair/group port.
- Stop/clear local worker port runtime owner if stale.
- Attempt unload of local model.
- Clear split metadata:
  - `runtime_group_id`
  - `runtime_placement`
  - `loaded_model`
  - `model_loaded`
- Rewrite heartbeat immediately.

### Stage C: Verified Cold Gate
Before returning to scheduler eligibility, verify all conditions:
- `runtime_state == cold`
- `model_loaded == false`
- split port has no model process/listener
- local worker port healthy and idle
- VRAM below cold threshold

If any check fails:
- retry cleanup once with reasoned backoff
- then enqueue targeted recovery meta tasks (Stage D)

### Stage D: Recovery Task Fallback
Emit targeted tasks if direct cleanup failed:
- `unload_split_llm(group_id=<pair>)`
- `unload_llm(candidate_workers=[member_a])`
- `unload_llm(candidate_workers=[member_b])`

These are recovery-only tasks and should not count as normal task failures/retries.

### Stage E: Quarantine (Repeated Failure Protection)
If same pair fails split load >= 3 times in rolling window:
- mark pair `quarantined_split_pair` for cooldown period (example: 10-15 min)
- suppress new `load_split_llm` targeting that pair during cooldown
- continue servicing 7B work on recovered members where possible

## Scheduler / Claiming Rules

Hard reject claiming when:
- `runtime_state in {loading_single, loading_split, unloading, recovering_split, recovering_single, wedged}`

Allow claiming only when:
- `runtime_state in {cold, ready_single, ready_split}`
- and state-specific invariants pass.

## Logging Requirements

Add structured logs with stable event keys:
- `AUTO_RECOVERY_TRIGGER`
- `AUTO_RECOVERY_STAGE_A_LOCKED`
- `AUTO_RECOVERY_STAGE_B_MEMBER_RESET`
- `AUTO_RECOVERY_STAGE_C_VERIFIED_COLD`
- `AUTO_RECOVERY_STAGE_D_TARGETED_FALLBACK`
- `AUTO_RECOVERY_QUARANTINE_ENTER`
- `AUTO_RECOVERY_QUARANTINE_EXIT`

Log fields:
- `group_id`, `members`, `model`, `reason`, `attempt`, `duration_ms`, `result`

## Implementation Targets (Code)

Primary files:
- `/home/bryan/llm_orchestration/shared/agents/gpu_split.py`
- `/home/bryan/llm_orchestration/shared/agents/gpu_state.py`
- `/home/bryan/llm_orchestration/shared/agents/gpu_tasks.py`
- `/home/bryan/llm_orchestration/shared/agents/gpu_constants.py`
- `/home/bryan/llm_orchestration/shared/agents/brain_resources.py` (quarantine-aware targeting)

Likely touchpoints:
- Split runtime startup and warmup error handlers
- Invariant checker clear path
- Wedged detection and cooldown transition logic
- Meta task claim guard logic
- Brain split-pair selection filter (exclude quarantined)

## Acceptance Criteria

1. After induced split warmup crash, both pair members return to `cold` automatically within bounded time.
2. No persistent `wedged` state remains without active recovery attempt.
3. Workers resume normal claiming (7B or split as eligible) without manual reset.
4. Repeated pair failures enter quarantine and avoid repeated batch stalls.
5. Dashboard reflects recovery stage and reason, not ambiguous stale split labels.

## Test Plan (Required)

1. Inject split warmup failure (`rc=-9` equivalent) on one member.
2. Verify pair-coordinated reset events and `verified_cold` completion.
3. Confirm queue progresses with unaffected workers during recovery.
4. Repeat failure 3 times; confirm quarantine and suppression of new split loads for that pair.
5. Confirm quarantine expiry re-enables pair candidate selection.

## Notes

- This is intentionally a single-path design: no silent fallback hacks.
- If `verified_cold` cannot be established, fail loud with explicit machine-readable reason.
