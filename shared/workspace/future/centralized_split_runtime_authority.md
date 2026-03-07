# Centralized Runtime Authority

## Context

**Problem**: Shared runtime coordination still has multiple distributed authority paths.

The current system improved some brain-side control, but workers still directly make or shape decisions in several places:

1. **Split runtime cleanup race**
   - A GPU can detect a transient invariant miss and call local split cleanup
   - Another GPU may have just successfully launched the runtime
   - Result: healthy runtimes get torn down by a follower or stale observer

2. **Split quarantine split-brain**
   - GPUs track split-pair failure counts and quarantine locally
   - Brain tracks similar failure/quarantine state separately
   - Result: worker-local and scheduler-authoritative views can diverge

3. **Recovery fallback still worker-shaped**
   - Workers emit recovery fallback signals that already encode requested tasks
   - Brain processes those signals and inserts the requested meta tasks
   - Result: the worker still partially decides remediation, not just detection

4. **Global model-load owner reclaim is worker-local**
   - A worker can detect a stale global owner file and unlink it locally
   - Result: ownership reclamation is still distributed and can race with live owners

5. **Task requeue hygiene is inconsistent**
   - Some brain requeue paths scrub stale ownership/terminal fields correctly
   - Other paths requeue tasks with stale `assigned_to`, `started_at`, `completed_at`, or `result`
   - Result: queue state is not consistently authoritative

**Root cause**: Shared authority is still distributed across workers, fallback signals, and multiple partial brain paths.

**Required direction**: Brain owns all shared authority transitions. Workers only:
- observe
- report
- execute explicit brain-issued commands

---

## Authority Model

### Current (Mixed Authority)

```text
GPU detects issue -> GPU decides or shapes decision -> shared state mutates -> Brain reacts
```

Examples:
- GPU calls `_coordinated_split_failure_cleanup()`
- GPU enters local split quarantine
- GPU unlinks stale global load owner
- GPU emits fallback signal that pre-selects recovery commands
- Multiple brain paths requeue tasks with different scrub behavior

### Proposed (Centralized Authority)

```text
GPU detects issue -> GPU reports observation -> Brain decides -> Brain issues fenced command -> GPU executes
```

**Key principle**: Workers are sensors and executors. Brain is the sole authority for shared coordination state.

That includes:
- split cleanup decisions
- split quarantine state
- recovery strategy selection
- global model-load ownership reclaim
- queue re-entry / requeue scrubbing
- reservation terminal-state transitions when they affect shared orchestration

---

## Non-Negotiable Design Rules

1. **No worker may unilaterally mutate shared authority state**
   - No local quarantine entry
   - No local stale-owner reclaim
   - No local split cleanup except narrow owner-only startup failure cases

2. **Every brain-issued destructive command must be fenced**
   - Use reservation epoch / runtime generation / lease ID
   - A stale cleanup or unload command must no-op against a newer runtime

3. **Workers report observations, not remedies**
   - Report issue code, scope, observed state, severity, epoch
   - Do not report `tasks_needed` as authority-bearing intent

4. **Requeue must use one shared scrub helper**
   - Any path writing `status: pending` into `tasks/queue` must pass through one helper

5. **Timeout escape hatches must be narrow and epoch-aware**
   - Timeout fallback exists only to recover from brain unavailability
   - Timeout actions must verify they still apply to the same generation that was reported

---

## Scope

This plan covers five centralization tracks:

1. Split runtime cleanup centralization
2. Split quarantine centralization
3. Recovery fallback centralization
4. Global model-load owner centralization
5. Task requeue scrub centralization
6. Brain-owned run summary and event ledger

---

## Implementation Progress

### 2026-03-06

- Track 5 started
- Brain now has a shared task requeue normalization helper
- Brain failure retry/fix requeue paths were routed through the helper
- Brain monitor orphan-recovery and force-kill timeout requeue paths were routed through the helper
- Requeue invariants now fail fast if stale ownership or terminal fields remain on queue re-entry
- Track 1 scaffolding started
- GPU heartbeats now publish split health issue observations and split runtime generation state
- Split invariant checks now emit issue reports for brain visibility before any future authority flip
- Brain resource loop now observes and deduplicates split health issue reports for planning/visibility
- Explicit brain-issued `cleanup_split_runtime` meta command path now exists
- Meta-task dedupe now includes split cleanup fencing fields (`target_workers`, epochs, runtime generation)
- Workers reject stale cleanup commands when the runtime generation does not match local state
- Split invariant-triggered cleanup is no longer worker-local
- Brain now issues first-pass cleanup decisions for critical reports and all-member error reports
- Reservation terminal-state cleanup in `_service_split_reservations()` is now report-first
- Only a narrow owner-local dead-runtime exception still performs local cleanup in that path
- Cleanup commands are now fenced by both runtime generation and reservation epoch
- Workers reject stale cleanup commands when either fence no longer matches local state
- Worker-local split-pair quarantine tracking has been removed
- Brain now owns split failure counting/quarantine state for cleanup-driven failures
- Recovery fallback signals are now observation-driven instead of worker-authored task plans
- Brain derives fallback unload actions from observed group/member state
- Global load-owner payloads now carry lease IDs
- Workers now report stale-owner takeover observations to the brain via heartbeat state
- Workers no longer unlink stale global load-owner leases locally
- Brain now reclaims stale global load-owner leases after verifying the observed lease still matches disk state
- Brain and workers now use the same stale-owner timeout constant
- Workers clear stale owner-issue state on successful lease acquire/release
- Idle workers now claim from queue on the fast internal cadence instead of waiting for the idle external heartbeat
- Split reservation creation now writes a partner nudge so the other member stays on a fast coordination loop
- `load_split_llm` now force-cleans local split state if the reservation disappears mid-load
- Split workers now force a local reset when shared owner metadata is missing, instead of lingering in `has_model=True, has_owner=False`

Track 5 status:
- core requeue centralization is landed
- no known queue re-entry paths remain outside the shared helper after routing
  `brain_failures.py` and `brain_monitor.py` through it
- `incident_id` policy is explicit:
  - preserve across retries/requeues for the same work item
  - drop only when the brain rewrites task definition semantics enough to
    start a new incident lineage
- future hardening only:
  - enforce queue-write invariants at a lower shared brain queue layer if more paths appear

Remaining work for Track 1:
- brain-issued fenced `cleanup_split_runtime` commands are active
- expand brain cleanup decision rules beyond first-pass critical/all-member-error handling
- audit remaining split cleanup call sites outside the narrow dead-state local reset paths
- replace scaffolded reservation-epoch derivation with an explicit durable epoch field if needed

Track 2 status:
- landed
- non-cleanup recovery observations increment brain-side pair failure counts
- brain-owned quarantine state is exposed in brain heartbeat/state output

Track 3 status:
- landed
- observation-only recovery signaling is the only accepted path
- fallback observations carry verification-check detail from the failed
  verified-cold stage so brain policy can reason over the observation without
  workers prescribing a remedy

Remaining work for Track 4:
- consider replacing brain-side unlink reclaim with a fully explicit superseding lease model
- decide whether owner issues should travel via heartbeat only or also via explicit observation artifacts

---

## Track 6: Brain-Owned Run Summary And Event Ledger

### Problem

Raw history folders contain the right evidence, but the old summary path was
still structurally wrong:

- `brain_plan.py` inserted `batch_summary` as a terminal task
- plan-task dependencies gated when the summary could run
- fatal aborts bypassed that task path

That meant failure-heavy runs often had no clean summary unless a separate
post-run script was invoked.

### Goal

Make the brain own the per-run event ledger and incremental summary refresh,
while keeping the reducer reusable as a standalone script.

That means:
- no summary task in the plan graph
- append-only batch events under `logs/batch_events.jsonl`
- brain-owned summary refresh on terminal task events and batch terminal states
- the same reducer can still be run manually on any history folder later

### Required Artifacts

Per run:
- `history/{batch_id}/logs/batch_events.jsonl`
- `history/{batch_id}/RUN_SUMMARY.json`
- `history/{batch_id}/RUN_SUMMARY.md`

### Reduction Strategy

Use a reducer library to compress raw runtime artifacts into a small review
surface before asking an LLM to analyze anything.

That means:
- JSON summaries are the machine-readable source
- markdown summaries are the human-readable render
- the brain refreshes those summaries during batch execution
- the standalone CLI can still summarize any history folder after the fact
- the reducer surfaces important existing logs and artifacts, not conclusions
- the LLM still decides what the surfaced artifacts imply

### Recommended Implementation Order

1. Remove `batch_summary` from the plan graph
2. Add brain-owned `batch_events.jsonl` writes from task/batch lifecycle state
3. Refresh `RUN_SUMMARY.json` and `RUN_SUMMARY.md` from the brain using the reducer
4. Keep a tracked CLI entrypoint for manual/offline summarization
5. Optionally add cross-run rollup/query scripts later, built on top of the per-run reducer

### Initial Scope For This Workstream

First useful slice:
- no cross-run rollups yet
- per-run batch event log only
- per-run summary refresh only
- keep the reducer tolerant of missing or inconsistent artifacts

### Progress

Landed:
- reusable reducer in `shared/agents/run_summary.py`
- tracked CLIs in:
  - `scripts/summarize_history_run.py`
  - `scripts/rollup_history.py`
- brain-owned `batch_events.jsonl` writing via `brain_summary.py`
- automatic `RUN_SUMMARY.json` / `RUN_SUMMARY.md` refresh on:
  - terminal task events observed by the brain
  - fatal batch abort
  - normal batch completion
- automatic `batch_summary` task insertion removed from `brain_plan.py`
- monitor-driven force-kill timeout and orphan recovery requeues now also use
  the shared requeue helper and emit the same batch retry/release events
- recoverable retries now preserve `incident_id`; definition rewrites are the
  only requeue path that intentionally drop it
- recovery fallback processing now accepts only observation-only
  `split_recovery_observation` payloads; legacy worker-authored fallback signal
  types are no longer honored
- brain heartbeat/state now expose brain-owned split quarantine state so the
  authoritative quarantine view is externally observable
- Stage D recovery observations now include failed verified-cold check detail
  for richer brain-side recovery policy inputs

Track 6 status:
- landed
- manual-stop summary refresh is landed
- resume-handoff summary refresh is landed
- cross-run rollup ledgers under `history/_summary/` are landed

### Why This Matters

This is a context-reduction layer.

Instead of asking an LLM to read raw history trees and logs first, scripts
shrink the review set to the handful of artifacts and excerpts most likely to
matter. That lowers context cost without moving decision-making into the script.

---

## Test Coverage Added

### Test Coverage Added

- `shared/agents/tests/test_centralized_runtime_authority.py`

Current targeted coverage includes:
- brain queues split cleanup on critical issue reports
- brain does not queue cleanup on single-worker warning reports
- worker rejects stale cleanup commands on reservation-epoch mismatch
- worker rejects stale cleanup commands on runtime-generation mismatch
- worker executes cleanup when both fences match
- brain reclaims stale global load-owner leases only when the observed lease still matches disk state
- brain does not reclaim when the observed lease does not match current disk state

---

## Track 1: Split Runtime Cleanup Centralization

### Problem

Workers can still tear down shared split state based on local observations.

Current hot paths include:
- `gpu_split.py:_check_split_runtime_invariants()`
- `gpu_split.py:_service_split_reservations()`

### Goal

Brain becomes the sole authority for split cleanup decisions.

### Required Changes

#### 1.1 Add heartbeat issue reporting

**Files**:
- `/media/bryan/shared/agents/gpu_constants.py`
- `/media/bryan/shared/agents/gpu.py`
- `/media/bryan/shared/agents/gpu_split.py`

Add heartbeat fields for split health observations:

```python
"split_health_issue": {
    "has_issue": False,
    "severity": None,
    "issue_code": None,
    "issue_detail": None,
    "group_id": None,
    "split_port": None,
    "reservation_epoch": None,
    "runtime_generation": None,
    "detected_at": None,
    "consecutive_detections": 0,
    "awaiting_brain_decision": False,
    "reported_at": None,
}
```

#### 1.2 Add fenced cleanup command

**File**:
- `/media/bryan/shared/agents/gpu_tasks.py`

Add meta command:

```json
{
  "command": "cleanup_split_runtime",
  "command_id": "uuid",
  "group_id": "pair_4_5",
  "cleanup_reason": "brain_decision_reason",
  "target_workers": ["gpu-4", "gpu-5"],
  "reservation_epoch": "uuid-or-counter",
  "runtime_generation": "uuid-or-counter"
}
```

Workers execute cleanup only if the command epoch matches local observed state.

#### 1.3 Move invariant handling to report-only

**File**:
- `/media/bryan/shared/agents/gpu_split.py`

Change `_check_split_runtime_invariants()` so it:
- reports issue
- optionally marks local state degraded
- does **not** call `_coordinated_split_failure_cleanup()` directly

Keep narrow local cleanup only for launcher-owned startup failures:
- process crash during startup
- warmup timeout during owned launch
- exception while starting owned runtime

These are safe because they are owner-local and immediate.

#### 1.4 Move deferred reservation cleanup to brain-controlled path

**File**:
- `/media/bryan/shared/agents/gpu_split.py`

Current `status in {"unloaded", "failed", "expired"}` handling should not directly trigger destructive cleanup unless:
- the worker is executing an explicit brain command, or
- it is owner-local cleanup for a runtime generation the worker still owns and can prove is already dead

#### 1.5 Add brain-side issue monitor and decision rules

**File**:
- `/media/bryan/shared/agents/brain_resources.py`

Add:
- `_monitor_split_health_issues(gpu_states)`
- `_evaluate_split_cleanup_decision(group_id, issues, gpu_states)`
- `_issue_split_cleanup_command(group_id, issues, reason, epoch)`

Decision rules:
1. Any `CRITICAL` issue -> immediate cleanup
2. All members report `ERROR` on same epoch -> cleanup
3. Majority report `ERROR` on same epoch -> cleanup
4. Same issue persists past deadline -> cleanup
5. One warning with healthy peer -> wait

---

## Track 2: Split Quarantine Centralization

### Problem

Both GPU workers and brain track split-pair failures and quarantine independently.

Current split-brain tracking exists in:
- `/media/bryan/shared/agents/gpu_split.py`
- `/media/bryan/shared/agents/brain_resources.py`

### Goal

Brain owns all split failure counting and quarantine decisions.

### Required Changes

#### 2.1 Remove worker-local quarantine authority

**File**:
- `/media/bryan/shared/agents/gpu_split.py`

Deprecate local authority paths:
- `_record_split_failure_for_quarantine()`
- `_enter_split_pair_quarantine()`
- `_is_split_pair_quarantined()`

Workers may still report:
- recent failure event
- local diagnostics
- observed cleanup result

But workers must not decide quarantine status.

#### 2.2 Make brain quarantine authoritative

**File**:
- `/media/bryan/shared/agents/brain_resources.py`

Brain should:
- record split failure events
- maintain rolling windows
- enter/exit quarantine
- publish quarantine state in decisions/logs
- optionally expose current quarantine state in a durable brain-owned file if needed for observability

#### 2.3 Fence quarantine-triggering events

Failures counted toward quarantine must include:
- `group_id`
- `reservation_epoch`
- `runtime_generation`
- `reason`
- `source`

This avoids counting failures from dead generations against a fresh reservation.

---

## Track 3: Recovery Fallback Centralization

### Problem

Workers still shape remediation by emitting fallback signals with `tasks_needed`.

Current path:
- worker writes `*.recovery_fallback.json`
- brain reads signal and inserts requested resource tasks

### Goal

Workers report condition only. Brain chooses remediation.

### Required Changes

#### 3.1 Replace remedy-bearing signals with observation reports

**Files**:
- `/media/bryan/shared/agents/gpu_split.py`
- `/media/bryan/shared/agents/brain_resources.py`

Replace:

```json
{
  "type": "split_recovery_fallback",
  "group_id": "...",
  "worker": "...",
  "tasks_needed": [...]
}
```

With something like:

```json
{
  "type": "split_recovery_observation",
  "group_id": "...",
  "worker": "...",
  "issue_code": "verified_cold_failed",
  "reservation_epoch": "...",
  "runtime_generation": "...",
  "observed_state": {
    "listener_up": true,
    "model_loaded": false,
    "runtime_state": "recovering_split"
  }
}
```

#### 3.2 Brain derives the recovery plan

**File**:
- `/media/bryan/shared/agents/brain_resources.py`

Brain should decide whether to:
- issue `cleanup_split_runtime`
- issue `unload_split_llm`
- issue `unload_llm`
- quarantine the pair
- defer action and re-check

No worker-authored signal should directly dictate command shape.

---

## Track 4: Global Model-Load Owner Centralization

### Problem

A worker can locally decide the global model-load owner is stale and unlink the owner file.

Current path:
- `/media/bryan/shared/agents/gpu_ollama.py:_try_acquire_global_model_load_owner()`

### Goal

Global model-load ownership becomes a fenced lease controlled by brain-owned reclaim policy.

### Required Changes

#### 4.1 Add lease identity

**File**:
- `/media/bryan/shared/agents/gpu_ollama.py`

Owner payload should include:

```python
{
    "worker": self.name,
    "pid": os.getpid(),
    "phase": phase,
    "lease_id": "...",
    "lease_epoch": "...",
    "acquired_at": "...",
    "heartbeat_at": "...",
}
```

#### 4.2 Remove worker-local unlink-on-stale as final authority

Workers may detect:
- stale owner heartbeat
- dead owner pid
- invalid owner payload

But should report the condition to brain instead of unlinking the file directly.

#### 4.3 Add brain-issued owner reclaim command or fenced takeover

Brain should either:
- issue explicit reclaim command, or
- write a newer fenced lease that supersedes the old one

The important rule: a worker must not locally decide to delete shared ownership state because it *appears* stale.

---

## Track 5: Task Requeue Scrub Centralization

### Problem

Multiple brain code paths write tasks back to `queue/` with inconsistent field scrubbing.

Known affected areas include:
- `/media/bryan/shared/agents/brain_failures.py`
- `/media/bryan/shared/agents/brain_monitor.py`
- `/media/bryan/shared/workspace/human/HUMAN_requeue_owner_scrub_core_fix_20260305.md`

### Goal

Any task re-entering `queue/` has one authoritative normalization path.

### Required Changes

#### 5.1 Add one helper for all requeues

**Suggested file**:
- `/media/bryan/shared/agents/brain_failures.py`
  or a shared brain utility module if cleaner

Suggested helper:

```python
def _prepare_task_for_requeue(task: Dict[str, Any], reason: str, *, reset_attempts: bool = False) -> Dict[str, Any]:
    ...
```

Behavior:
1. Set `task["status"] = "pending"`
2. Remove:
   - `assigned_to`
   - `worker`
   - `started_at`
   - `completed_at`
   - `result`
3. Remove or normalize recovery-specific terminal fields where appropriate:
   - `cloud_escalated`
   - `cloud_escalation_id`
   - `blocked_reason`
   - stale `incident_id` only if policy says requeue starts a fresh incident chain
4. Preserve:
   - `attempts` unless caller explicitly resets
   - `workers_attempted`
   - `last_attempt_at`
   - `first_attempted_at`
5. Stamp:
   - `requeued_at`
   - `requeue_reason`

#### 5.2 Route all queue re-entry through the helper

Patch sites:
- `brain_failures.py`
  - worker retry branch
  - missing-module fix requeue
  - brain-fix succeeded requeue
  - any `_queue_task_retry()` path
- `brain_monitor.py`
  - `_recover_orphaned_processing_tasks()`
  - force-kill timeout requeue path

#### 5.3 Add fail-fast invariant

When writing to `tasks/queue/*.json`, brain should assert:
- `status == "pending"`
- no `assigned_to`
- no `started_at`
- no `completed_at`
- no terminal `result`

If violated, fail loudly and log the exact offending path.

---

## Cross-Cutting Requirement: Fencing / Epochs

This is the critical hardening layer for all centralization work.

Without fencing, moving decisions to brain still allows delayed destructive commands to hit newer healthy state.

### Required Identifiers

1. **Reservation epoch**
   - Changes whenever a split reservation is recreated or fundamentally reset

2. **Runtime generation**
   - Changes whenever a shared runtime is started anew on a port

3. **Lease ID / owner epoch**
   - Changes whenever global model-load ownership changes hands

### Rules

1. Every worker observation about shared runtime health includes the current epoch/generation if known
2. Every brain-issued destructive command includes the epoch/generation it applies to
3. Every worker verifies epoch match before executing cleanup/unload/reclaim
4. Any mismatch means:
   - no-op
   - log stale command rejection
   - clear pending issue if superseded

---

## Migration Plan

### Phase 1: Add Report-Only Infrastructure

1. Add heartbeat/report fields for split health observations
2. Add reservation epoch / runtime generation / lease ID fields
3. Add brain monitors for split health and owner-stale reports
4. Add shared requeue scrub helper

This phase is additive and non-breaking.

### Phase 2: Switch Brain to Authority

1. Convert split invariant cleanup sites to report-only
2. Convert worker quarantine to report-only
3. Convert recovery fallback signals to observation-only
4. Convert global owner stale detection to report-only
5. Route all requeue paths through one helper

### Phase 3: Enforce Fencing

1. Require epochs on all destructive commands
2. Reject stale cleanup/unload/reclaim commands
3. Reject direct queue writes that bypass requeue helper invariants

### Phase 4: Remove Legacy Distributed Authority

1. Remove worker-local quarantine structures
2. Remove worker-authored `tasks_needed` recovery signals
3. Remove local stale-owner unlink as a decision path
4. Remove redundant split cleanup call sites no longer needed

---

## Verification

### Split Cleanup

1. Simulate follower-only transient invariant miss during launcher success
2. Verify follower reports issue but does not cleanup locally
3. Verify brain waits or cleans up based on peer state and epoch

### Stale Command Rejection

1. Start runtime generation A
2. Emit cleanup decision for generation A but delay delivery
3. Start runtime generation B
4. Verify delayed cleanup for A is rejected by workers

### Quarantine Consistency

1. Trigger repeated split failures
2. Verify only brain enters quarantine
3. Verify scheduler decisions match quarantine state exactly

### Recovery Fallback

1. Emit worker recovery observation
2. Verify brain derives remediation
3. Verify no worker-chosen `tasks_needed` path exists

### Global Owner Reclaim

1. Simulate stale owner heartbeat
2. Verify worker reports stale-owner observation
3. Verify only brain-authorized reclaim or superseding lease changes ownership

### Requeue Hygiene

1. Trigger retry from failed path
2. Trigger orphan recovery from monitor path
3. Trigger force-kill timeout requeue
4. Verify all queued tasks have:
   - `status: pending`
   - no `assigned_to`
   - no `started_at`
   - no `completed_at`
   - no terminal `result`

---

## Concrete Patch Targets

### Split / Worker Side

- `/media/bryan/shared/agents/gpu_constants.py`
- `/media/bryan/shared/agents/gpu.py`
- `/media/bryan/shared/agents/gpu_split.py`
- `/media/bryan/shared/agents/gpu_tasks.py`
- `/media/bryan/shared/agents/gpu_ollama.py`

### Brain Side

- `/media/bryan/shared/agents/brain_resources.py`
- `/media/bryan/shared/agents/brain_monitor.py`
- `/media/bryan/shared/agents/brain_failures.py`

### Reference Note

- `/media/bryan/shared/workspace/human/HUMAN_requeue_owner_scrub_core_fix_20260305.md`

---

## Recommended Implementation Order

1. Requeue scrub helper
   - smallest, safest, immediate consistency gain

2. Split cleanup report-only path + brain cleanup command
   - highest value for current runtime race

3. Split quarantine centralization
   - remove duplicate authority

4. Recovery fallback observation-only conversion
   - eliminate worker-shaped recovery plans

5. Global owner lease fencing and reclaim centralization
   - same authority pattern, but broader blast radius

---

## Summary

The split cleanup race is not an isolated bug. It is one instance of the same broader architectural problem:

**workers still hold pieces of shared authority.**

The fix is not just "move split cleanup to brain." The real fix is:

- brain owns shared authority transitions
- workers report observations
- workers execute fenced commands
- all queue re-entry is normalized through one authoritative brain helper

That is the full centralization path.

*Created: 2026-03-05*
*Updated: 2026-03-06*
*Status: Planning*
