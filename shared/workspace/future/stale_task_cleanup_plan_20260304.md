# Stale Task Cleanup Plan (Execution Checklist)

Date: 2026-03-04
Scope: core orchestration (`shared/agents/*`)
Status: Open (not implemented)

## Goal

Prevent stale queue/processing tasks from poisoning scheduling by adding deterministic lifecycle cleanup with bounded recovery.

## Current State (Verified)

Not implemented yet in core:
1. No `stale_task_policy` block consumed by agents.
2. No stale-task sweeper in `brain_resources.py`.
3. No stale-task event keys (`STALE_SWEEP_*`, `TASK_EXPIRED_UNCLAIMED`, etc.).
4. No dedicated stale cleanup unit test module.

## Required Outcome

1. Unclaimed tasks expire by TTL with explicit terminal reason.
2. Dead processing tasks are failed/recovered deterministically.
3. Hard processing timeouts are enforced with bounded requeue.
4. Invalid queue state is auto-quarantined.
5. Recovery resets are targeted/deduped.

## Phase 1: Policy + Config Wiring

Files:
- `shared/agents/config.template.json`
- `shared/agents/config.json`
- `shared/agents/brain.py` (load defaults)

Implement:
1. Add `stale_task_policy` config block.
2. Load policy into brain state with defaults if missing.
3. Keep feature behind `stale_task_policy.enabled`.

Policy fields:
1. `pending_ttl_seconds` per command (`load_llm`, `unload_llm`, `load_split_llm`, `unload_split_llm`).
2. `processing_hard_timeout_seconds` per command.
3. `processing_lease_missed_heartbeats`.
4. `max_timeout_requeues`.

Exit criteria:
1. Brain starts cleanly with/without the block.
2. Effective policy values visible in brain logs at startup.

## Phase 2: Brain Stale Sweeper Core

Files:
- `shared/agents/brain_resources.py`
- `shared/agents/brain_dispatch.py` (shared fail helper if needed)

Implement:
1. Add `_sweep_stale_tasks(shared_path, state, now)`.
2. Invoke sweep at start of `_make_resource_decisions()`.
3. Sweep `tasks/queue/*.json`:
- expire by pending TTL (`expired_unclaimed`),
- detect invalid queue state (`pending` + `completed_at`/`result`/stale assignment markers).
4. Sweep `tasks/processing/*.json`:
- lease timeout from heartbeat staleness,
- hard wall timeout from `started_at`.
5. For stale processing:
- fail task with structured reason,
- issue targeted reset (`reset_gpu_runtime` or `reset_split_runtime`) using `target_worker`,
- bounded requeue only when budget allows.
6. Add dedupe for recovery task insertion by `(command, target_worker, incident_key)`.

Required helper:
1. Single task terminalization helper that atomically moves queue/processing task to failed with `status`, `completed_at`, `result`.

Exit criteria:
1. Sweep is idempotent across ticks.
2. No partial task-file transitions.
3. No duplicate reset flood for same worker/incident.

## Phase 3: Worker-Side Guardrails

File:
- `shared/agents/gpu_tasks.py`

Implement:
1. Reject obviously invalid task lifecycle state early (`invalid_task_state`) before claim/execution.
2. Preserve existing target enforcement for reset commands in both claim paths.

Exit criteria:
1. Invalid stale task payload cannot execute.
2. Reset routing remains target-safe.

## Phase 4: Telemetry

Files:
- `shared/agents/brain_resources.py`
- `shared/agents/gpu_tasks.py` (if result-side event needed)

Emit events:
1. `STALE_SWEEP_START`
2. `TASK_EXPIRED_UNCLAIMED`
3. `TASK_STALE_PROCESSING`
4. `TASK_TIMEOUT_PROCESSING`
5. `TASK_INVALID_QUEUE_STATE`
6. `STALE_RECOVERY_TASK_ISSUED`
7. `STALE_SWEEP_SUMMARY`

Event payload minimum:
1. `task_id`, `task_name`, `task_class`, `command`
2. `batch_id`
3. `assigned_to`
4. `age_seconds`, `sla_seconds`
5. `action_taken`

Exit criteria:
1. Logs are sufficient to reconstruct every stale decision.

## Phase 5: Tests

File:
- `shared/agents/tests/test_stale_task_cleanup.py`

Minimum tests:
1. Pending meta task expires by TTL and replacement is bounded/deduped.
2. Processing task with stale heartbeat triggers targeted reset.
3. Hard-timeout processing task transitions terminal and bounded requeue applies.
4. Invalid queue task is terminalized with `invalid_queue_state`.
5. Recovery dedupe prevents duplicate reset tasks.
6. Lock/orphan file handling does not crash sweep.

Exit criteria:
1. Tests pass with no skipped core stale-cleanup cases.

## Runtime Validation Checklist

1. Inject synthetic stale `load_llm` in queue and confirm auto-expire.
2. Inject stale processing task and confirm targeted reset appears.
3. Confirm queue resumes draining without manual intervention.
4. Confirm no reset storms from repeated sweeps.

## Definition of Done

All must pass:
1. No stale meta tasks remain pending beyond policy TTL.
2. Stale processing tasks are failed/recovered deterministically.
3. Invalid queue-state tasks are auto-terminalized.
4. Reset/requeue behavior is bounded and deduped.
5. Unit tests for stale cleanup pass.
6. Runtime validation checklist passes with evidence in `brain_decisions.log`.
