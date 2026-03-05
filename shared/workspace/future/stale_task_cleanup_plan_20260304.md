# Stale Task Cleanup Plan (Authoritative)

Date: 2026-03-04  
Scope: core orchestration (`shared/agents/*`)  
Status: Proposed

## Objective

Prevent queue poisoning and worker deadlocks from stale `meta`/`llm` tasks by adding deterministic stale lifecycle handling:
1. Expire unclaimed tasks (`pending` TTL)
2. Detect dead processing tasks (heartbeat lease timeout)
3. Enforce hard runtime timeouts (processing wall timeout)
4. Recover workers safely (targeted reset + controlled requeue)

No silent deletion. Every stale action must produce a terminal status + log event.

---

## Policy (What to enforce)

## 1) Pending TTL (unclaimed)

If `status=pending` for too long:
- Mark task terminal: `expired_unclaimed`
- Write result metadata (`reason`, `age_seconds`, `sla_seconds`)
- For retryable commands, release one replacement task (dedup-safe)

Suggested defaults:
- `unload_llm`: 60s
- `load_llm`: 240s
- `unload_split_llm`: 90s
- `load_split_llm`: 420s
- normal `llm` work task: 300s pending warning, 900s hard expire (configurable)

## 2) Processing lease timeout (heartbeat-based)

If `status=processing` but task heartbeat is stale:
- Mark task terminal: `stale_processing`
- Issue targeted reset for assigned worker
- Requeue original task only if retry budget allows

Lease rule:
- stale if no heartbeat update for `3 * task_heartbeat_interval_seconds`

## 3) Hard processing timeout (wall clock)

If task is processing and `now - started_at > hard_timeout`:
- Mark terminal: `timeout_processing`
- Issue targeted reset
- Requeue once (or fail+escalate after max timeout retries)

Suggested hard timeouts:
- `unload_llm`: 90s
- `load_llm`: 240s
- `unload_split_llm`: 120s
- `load_split_llm`: 480s
- worker review llm tasks: keep existing logic, add config cap if missing

## 4) Queue hygiene invariant

Invalid queue tasks must be auto-cleaned:
- `pending` task with `completed_at` or `result` present
- `pending` task with stale `assigned_to` state from prior lifecycle
- orphan `.lock` with no corresponding task JSON

Action:
- mark/relocate task to `failed/` with reason `invalid_queue_state`
- emit explicit log event

---

## Core Implementation (Where to change)

## A) Brain side stale sweeper

File: `shared/agents/brain_resources.py`

Add function:
- `_sweep_stale_tasks(shared_path, state, now) -> dict`

Responsibilities:
1. Scan `tasks/queue/*.json` and `tasks/processing/*.json`
2. Apply pending TTL and invalid queue-state checks
3. Apply processing lease/hard-timeout checks
4. Emit recovery tasks (`reset_gpu_runtime` / `reset_split_runtime`) targeted via `target_worker`
5. Respect dedupe (no duplicate recovery task per worker+reason)
6. Return summary counters for monitor logs

Call site:
- start of `_make_resource_decisions()` before normal capacity decisions

## B) Recovery task routing consistency

File: `shared/agents/gpu_tasks.py`

Confirm/keep:
- `reset_gpu_runtime` and `reset_split_runtime` are target-enforced in:
  - emergency claim path
  - normal meta claim path

Add:
- reject stale/invalid task states early with `invalid_task_state` reason

## C) Shared constants/config

Files:
- `shared/agents/gpu_constants.py` (or dedicated stale constants block)
- `shared/agents/config.template.json`
- `shared/agents/config.json`

Add config block:
```json
"stale_task_policy": {
  "enabled": true,
  "pending_ttl_seconds": {
    "load_llm": 240,
    "unload_llm": 60,
    "load_split_llm": 420,
    "unload_split_llm": 90
  },
  "processing_hard_timeout_seconds": {
    "load_llm": 240,
    "unload_llm": 90,
    "load_split_llm": 480,
    "unload_split_llm": 120
  },
  "processing_lease_missed_heartbeats": 3,
  "max_timeout_requeues": 1
}
```

## D) Task state transition helper (single authority)

Preferred location:
- `shared/agents/brain_dispatch.py` or shared task utility module used by brain

Add helper:
- `_fail_task_with_reason(task, reason, details)`  
Guarantees:
1. atomic move queue/processing -> failed
2. sets `status`, `completed_at`, `result`
3. writes lock-safe
4. never leaves partially updated task files

---

## Logging/Telemetry Requirements

Add structured events:
- `STALE_SWEEP_START`
- `TASK_EXPIRED_UNCLAIMED`
- `TASK_STALE_PROCESSING`
- `TASK_TIMEOUT_PROCESSING`
- `TASK_INVALID_QUEUE_STATE`
- `STALE_RECOVERY_TASK_ISSUED`
- `STALE_SWEEP_SUMMARY`

Each event should include:
- `task_id`, `task_name`, `task_class`, `command`
- `batch_id`
- `assigned_to` (if any)
- `age_seconds`, `sla_seconds`
- `action_taken` (`failed`, `requeued`, `reset_issued`)

---

## Guardrails

1. Never delete task JSON without terminalizing it.
2. Never requeue indefinitely; enforce max timeout requeues.
3. Never issue untargeted reset from stale sweeper; always `target_worker`.
4. Dedupe recovery tasks by `(command, target_worker, incident_key)`.
5. Stale sweeper must be idempotent each brain tick.

---

## Test Plan (required before rollout)

File: `shared/agents/tests/test_stale_task_cleanup.py`

Minimum tests:
1. pending meta task expires at TTL and replacement is issued once
2. processing task with stale heartbeat triggers targeted reset
3. processing hard-timeout task transitions to failed and requeue obeys budget
4. invalid queue task (`pending` + `completed_at`) is quarantined/finalized
5. dedupe prevents duplicate reset tasks per worker
6. lock/orphan handling does not crash sweep

Runtime validation:
1. Inject synthetic stale `load_llm` in queue -> auto-expire observed
2. Inject synthetic stale processing task -> reset task appears and worker recovers
3. Confirm queue resumes draining without manual intervention

---

## Rollout Sequence

1. Add constants/config + sweep logic behind `stale_task_policy.enabled`
2. Add tests and pass locally
3. Enable in one benchmark batch (`github_analyzer_verify_benchmark`)
4. Validate logs and no false positives
5. Enable for full `github_analyzer`

---

## Definition of Done

All must pass:
1. No stale meta tasks remain pending after SLA windows
2. Stale processing tasks are deterministically failed/recovered
3. No queue entries remain with contradictory state (`pending` + `completed_at`)
4. Reset/requeue behavior is bounded and deduped
5. Tests pass with no skipped core stale-cleanup tests

