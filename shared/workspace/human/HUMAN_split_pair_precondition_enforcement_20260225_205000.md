# HUMAN: Split Pair Precondition Enforcement + Claim-Time Runtime Attestation

## Authoritative Lead Policy

`tentative claim -> direct runtime attestation -> if mismatch: release task (no retry penalty) + full worker self-reset -> return to task pool`

This is the authoritative policy for worker runtime integrity. Cached heartbeat/runtime state is advisory; direct Ollama/runtime probes at claim time are the source of truth.

## Status Snapshot (Read This First)

This is the **single active HUMAN implementation doc** for the runtime-integrity track.

### Implemented (verified in code)

- Split pair loading lock blocks both launcher and follower during split load
- Split member clean precondition verification (`member_clean`) before reservation can enter `loading`
- Fail-closed split precondition probe on local `/api/ps` probe errors
- Claim-time runtime attestation for LLM tasks
- Claim-time command-specific attestation for state-changing meta tasks (`load/unload`, split load/unload)
- Mismatch release semantics (requeue without retry increment / worker attempt penalty)
- Full local reset now reclaims stale worker-port listeners and ends in `cold` (no auto-restart)
- Mismatch circuit-breaker transitions worker to `wedged` with cooldown auto-recovery
- Brain split promotion sequencing: targeted `unload_llm` for hot pair members before `load_split_llm`
- `blocked_cloud` now terminalizes the local batch correctly (no zombie batch finalization)
- Startup warm policy prefers non-split GPUs first
- Brain JSON repair artifact discovery now searches `results/worker_review_failures/` with slice/phase metadata matching
- `worker_review.py` repair consumption extended to all phases (extractor, verifier, adjudicator, deep-pass)
- Mismatch cooldown auto-recovery implemented in `_is_runtime_wedged()` (wedged -> cold after cooldown elapsed)
- Benchmark `worker_review.py` synced with main script
- **Repair metadata wiring complete**: Worker launcher sets `WORKER_TASK_JSON_PATH` env var pointing to task JSON in processing/; `worker_review.py` reads repair metadata from task context at runtime (no plan template changes needed)

### Partially implemented / integration gaps (still needs work)

- Contradictions/verify straggler handling (e.g. `slice_008`) still causes 14B timeouts and `blocked_cloud`

### Next Steps (priority order)

1. Add contradictions-slice/verify straggler splitting or stricter complexity caps to reduce 14B timeout escalations

### Fresh-session handoff rule

If resuming in a new session:
- trust the **Status Snapshot** and **Remaining Gaps** sections first
- use historical sections below for rationale/examples, not as the source of current state

## Problem (Observed)

In batch `20260225_203857`, `gpu-3` launched `load_split_llm` for `pair_1_3` and `gpu-1` joined the split reservation while still hot with a local single-GPU model loaded (`qwen2.5:7b`).

Result:
- reservation advanced to `status=loading`
- split runtime on `11440` started and then hung warming
- `gpu-3` stuck in `loading_split`
- `gpu-1` remained `ready_single`

This means the "explicit unload before pairing" invariant is **not actually enforced as a hard precondition**.

Related observed consequence:
- workers can later execute tasks using the local Ollama runtime even when heartbeat/runtime fields say `cold`
- i.e. control-plane state and actual Ollama state diverge

## What Was Implemented Already (and Why It Wasn't Enough)

The recent split-pair loading lock fix correctly prevents the follower from stealing other tasks while the pair is loading.

What it does **not** do:
- verify the follower is cold/unloaded before joining/loading split
- block transition to `loading` when a member is still `ready_single`
- re-attest local Ollama reality at claim time before executing normal work

## Root Cause (Code-Level)

### 1) `prepared` means "port cleanup done", not "member is clean"

`shared/agents/gpu_split.py`
- `_prepare_local_for_split_pairing(...)` only does best-effort:
  - orphan runner cleanup
  - split port reclaim safety checks
- It does **not** enforce/verify:
  - local single-GPU runtime unloaded
  - `runtime_state == cold`
  - `model_loaded == false`
  - local worker Ollama port empty

The returned `prep_result` is then written into reservation `prepared[self.name]`.

### 2) Reservation transitions to `loading` based on `all_prepared`, not `all_members_clean`

`shared/agents/gpu_split.py` in `_service_split_reservations()`
- `all_prepared = all(bool(prepared.get(m)) for m in members if m in joined)`
- If `status in {waiting_partner, joining, loading}` and `all_joined and all_prepared`, launcher can advance to `loading`

This allows `loading` even if a follower is still `ready_single(qwen2.5:7b)`.

### 3) `_join_split_reservation()` runs before clean-state verification

`shared/agents/gpu_split.py`
- `_join_split_reservation(...)` writes `joined[self.name]` immediately
- There is no fail-closed validation before join or before prepared -> loading

### 4) Worker claim path can trust stale agent state over actual local Ollama state

Observed behavior:
- heartbeat/runtime fields reset to `cold`
- local worker Ollama still has models loaded (`/api/ps`)
- worker can still run a `7B` task from that stale local runtime

This means claim/execution paths need a **claim-time runtime attestation** against local Ollama before work starts.

## Robust Fix (Core) — Unified Runtime Integrity Policy

This should be implemented as one coherent policy:
1. split join/load requires verified clean members
2. claim-time task execution requires attested runtime reality
3. any runtime mismatch triggers fail-safe full local reset (no task retry penalty)

### A. Introduce explicit member clean-state verification for split participation

Add a helper in `shared/agents/gpu_split.py` (or `gpu_state.py` if preferred):

- `_verify_local_split_member_clean_precondition(...) -> dict`

Required checks (hard fail if any false):
- `runtime_state == cold` (or explicit transitional split-prep state)
- `model_loaded == false`
- `runtime_placement != split_gpu` (unless rejoining same reservation in a valid recovery path)
- local worker Ollama port (`self.port`) has no loaded models (probe `/api/ps`)
- optional: local worker port listener absent or empty
- optional: VRAM below threshold (telemetry check only at first if too noisy)

Return structured result:
- `ok: bool`
- `reason_code`
- `details` (loaded_model, runtime_state, local_port_models, vram_used_mb)

### B. Do not treat `prepared` as readiness-to-load

Split reservation should track **two distinct concepts**:
- `prepared` = port/reclaim prep done
- `member_ready` (or `member_clean`) = local runtime verified safe for split load

Do not compute `all_ready_to_load` from `prepared`.

### C. Fail closed before reservation enters `loading`

Before launcher sets `waiting_partner/joining -> loading`, require:
- all joined
- all prepared
- all members `member_clean == true`

If not:
- set reservation `status = failed` (or `failed_precondition`)
- set `error` like:
  - `member_not_cold:gpu-1:model_loaded=qwen2.5:7b:runtime_state=ready_single`
- log explicit reason

No split runtime should be started on `11440/11441` until preconditions pass.

### D. Enforce follower-side unload (if auto-unload is part of design)

If the intended design is "explicit unload before pairing" (automatic), then:
- follower must run unload path first
- post-unload verification must pass before `member_clean=true`

If unload fails or verification fails:
- reservation fails precondition (no `loading`)

### E. Logging / Observability

Add explicit logs:
- `SPLIT_JOIN_BLOCKED member=gpu-1 reason=member_not_cold ...`
- `SPLIT_PRECONDITION_FAIL group=pair_1_3 ...`
- `SPLIT_MEMBER_CLEAN_OK member=gpu-1 ...`

This makes it obvious why pairing stalls instead of silently hanging in warmup.

### F. Claim-Time Ollama Attestation (All Task Pickups)

Before a worker executes any claimed task (especially `llm` tasks, and also relevant meta tasks), perform a direct runtime attestation against local Ollama/split runtime state.

Required flow:
1. Worker tentatively claims task
2. Worker probes runtime reality:
   - local Ollama `/api/ps` on worker port
   - split port `/api/ps` when split runtime is expected/relevant
   - reservation state for split tasks/meta
3. Compare actual runtime state to:
   - task requirements (`llm_model`, tier/capability, placement/meta constraints)
   - worker heartbeat/runtime state
4. If match: keep task and execute
5. If mismatch: release task back to queue **without retry increment**, then run full local reset (below)

This prevents:
- "heartbeat says cold, Ollama still loaded" silent execution
- stale split/single runtime confusion after invariant clears
- bad retries caused by hidden local runtime state drift

### G. Mismatch Handling = Full Local Reset (Worker Self-Reset)

On claim-time runtime mismatch, the worker should treat itself as untrusted and perform a full local reset before claiming again.

Required behavior:
- release task back to queue (not a failure retry)
- no increment to task `attempts`
- no `workers_attempted` penalty for this mismatch release
- emit explicit log/event (e.g. `CLAIM_RELEASED_RUNTIME_MISMATCH`)

Full local reset should include:
- stop/kill local worker `ollama serve`
- kill non-brain local `ollama runner` children associated with worker runtime
- stop split runtime if locally owned
- kill local split listeners on member ports if residue exists (as recovery path)
- clear local runtime state fields (`runtime_state`, `model_loaded`, placement/group ownership)
- clear local active/meta task state
- write fresh heartbeat reflecting actual clean state

Optional safety:
- if repeated mismatches (e.g. 3 in N minutes), mark worker `wedged` and stop claiming until cooldown/intervention

### H. Task Accounting Semantics for Mismatch Releases

Claim-time runtime mismatch is **not** task execution failure.

Do not:
- increment retry/attempt count
- mark task failed
- trigger normal retry/escalation logic

Do:
- release/requeue task immediately
- record structured mismatch reason for observability/debugging

## Suggested Implementation Locations

- `shared/agents/gpu_split.py`
  - add precondition verification helper(s)
  - extend reservation schema (`member_clean` / `preconditions`)
  - gate launcher transition to `loading`
- `shared/agents/gpu_tasks.py`
  - if auto-unload-on-split is triggered from meta path, ensure result is verified before proceeding
  - add claim-time runtime attestation before task execution starts
  - add task release path for runtime mismatch (non-retry)
- `shared/agents/gpu_state.py`
  - optional: dedicated runtime state for split precondition/unload (`preparing_split`)
- `shared/agents/gpu_ollama.py` / split helpers (or equivalent mixin locations)
  - full local reset helpers (kill/verify/clear state)
- `shared/agents/brain_failures.py` / task accounting path (if needed)
  - ensure mismatch release does not count as retry/failure

## Acceptance Criteria

1. Repro case: `gpu-1` hot with `qwen2.5:7b`, `gpu-3` claims `load_split_llm pair_1_3`
- Expected: reservation does **not** advance to `loading`
- Split runtime on `11440` is **not** started
- Clear error/log indicates `gpu-1` not clean

2. Clean pair case: both `gpu-1` and `gpu-3` cold
- Expected: reservation advances to `loading`, then `ready`

3. If follower unload is attempted and fails verification
- Expected: reservation fails precondition and exits cleanly (no warmup hang)

4. No more long hangs where:
- launcher is `loading_split`
- follower is still `ready_single`
- reservation `status=loading`

5. Repro case: worker heartbeat says `cold` but local Ollama still has `qwen2.5:7b` loaded
- Expected: on next claim, worker detects mismatch before execution
- task is returned to queue without retry increment
- worker performs full local reset and writes fresh heartbeat

6. No `7B` task executes successfully on a worker whose heartbeat claims `cold` unless the worker first re-attests and updates state.

## Related Docs

- `shared/workspace/human/HUMAN_split_runtime_state_machine_20260225_132656.md`
- `shared/workspace/human/HUMAN_patchwork_consolidation_cleanup_20260225.md`

## Consolidated Remaining Work (Merged From Patchwork Cleanup Doc)

This section is the active backlog for runtime-integrity/consolidation work that is not yet implemented in core.

### Core Runtime / Scheduling (High Priority)

1. Split preflight / wedge recovery should move from plan-local scripts into core brain resource logic
- current patchwork: `shared/plans/shoulders/github_analyzer/scripts/preflight_split_capacity.py`
- target home: `shared/agents/brain_resources.py` + related brain scheduling code

1a. ~~**Explicit brain sequencing rule for split promotion (must be enforced)**~~ **IMPLEMENTED**
- ~~When brain decides to promote a pair (e.g. `gpu-4` + `gpu-5`) from single `7B` work to split `14B` capacity:~~
  ~~1. issue targeted `unload_llm` tasks for the specific member GPUs first~~
  ~~2. wait for verified clean state on both members (not just task completion)~~
  ~~3. only then release `load_split_llm` for that split group~~
- ~~Brain must not release `load_split_llm` first and rely on implicit follower-side cleanup.~~
- ~~This sequencing is required to avoid split warmup hangs caused by members joining while still hot.~~
- See "Implementation Status" section E below.

2. ~~Brain-level JSON repair escalation (first-class retry path)~~ **IMPLEMENTED**
- ~~current patchwork: `worker_review.py` + `repair_worker_review_json.py` subprocess~~
- ~~target home: `shared/agents/brain_failures.py` repair task orchestration~~
- See "Implementation Status" section below.

3. Timeout classification / retry policy must be centralized in core
- current patchwork: analyzer `worker_review.py` distinguishes `thermal_pause_timeout` vs `inference_timeout`
- target home: GPU/brain retry policy so all worker tasks behave consistently

4. Suppress pointless `load_llm` recovery churn system-wide
- current patchwork: analyzer `worker_review.py` checks model presence before queueing `load_llm`
- target home: core retry/recovery classification

5. Capability-based routing still needs end-to-end brain/task emitter alignment
- worker claim side is partially implemented
- brain/task generation/resource logic still encodes placement constraints in places
- see: `HUMAN_capability_based_llm_task_routing_20260225_152228.md`

### Dashboard / Observability (Pending)

6. Dashboard failure counts must be strictly batch-scoped
- avoid cross-batch stale failed artifacts inflating current batch failure count

7. Dashboard processing-lane retry artifacts should not count as failures
- tasks in `processing` may carry stale prior `result.success=false`
- UI/data layer should count by lane/status, not stale result payloads

8. Dashboard batch failure attribution should surface `batch_failure.json` source task cleanly
- expose `source_task`, `source_task_id`, and reason in UI

### Codebase Consolidation (Pending)

9. Analyzer + benchmark script duplication should be reduced/removed
- current issue: manual sync drift across plan repos caused real failures
- target: shared importable module/package or shared script directory

10. Plan-local `brain_strategy.py` timeout/retry tuning should move to shared brain call policy

11. Shared worker error artifact contract (raw LLM failure persistence) should be generalized
- current patchwork only in analyzer `worker_review.py`

### Operational Recovery / Admin Tooling (Pending)

12. Manual live-rig recovery operations should become explicit admin tooling/core recovery paths
- stale task/heartbeat cleanup
- stale split listener cleanup
- targeted worker resets with logging/safety checks

13. ~~Startup warm policy should become placement-aware (brain + gpu-2 hot default)~~ **IMPLEMENTED**
- ~~current startup uses count-based `initial_hot_workers`~~
- ~~desired default policy keeps `gpu-1/3/4/5` free for split pairs~~
- See "Implementation Status" section below.

### Status Notes (Implemented Elsewhere)

Already implemented and verified (do not re-add to active TODO here):
- split pair loading lock (leader + follower busy during split load)
- `blocked_cloud` local batch terminalization
- `TASK_UNFIXABLE` spam prevention via `unfixable_marked`
- thermal pause normal path skips pausing active LLM workers

## Implementation Status (2026-02-25)

**CORE FIX COMPLETE:** All sections A-H from "Robust Fix (Core)" are implemented, plus brain-level sequencing (1a).

The observed bug (split reservation advancing to `loading` while follower still hot) is now fixed at multiple layers:
1. **Worker-side:** Member must verify clean precondition before joining reservation
2. **Reservation-side:** Transition to `loading` requires `all_members_clean`
3. **Brain-side:** Issues targeted `unload_llm` before `load_split_llm` for hot members
4. **Claim-time:** Runtime attestation catches state drift, triggers full reset

Remaining items in "Consolidated Remaining Work" are backlog/cleanup items NOT required for the core fix.

---

### A. Explicit member clean-state verification - IMPLEMENTED

Added `_verify_local_split_member_clean_precondition()` to `gpu_split.py`:
- Checks `runtime_state == cold` (hard fail if not)
- Checks `model_loaded == false` (hard fail if not)
- Checks `runtime_placement != split_gpu` (unless same-group rejoin)
- Probes local worker Ollama port `/api/ps` for loaded models (hard fail if models present)
- Returns structured result: `{ok: bool, reason_code: str, details: dict}`

### B. Separate `prepared` from `member_clean` - IMPLEMENTED

Reservation schema now tracks both concepts:
- `prepared[member]` = port cleanup prep done
- `member_clean[member]` = local runtime verified safe for split load

### C. Fail closed before `loading` transition - IMPLEMENTED

Updated `_service_split_reservations()`:
- Transition to `loading` now requires: `all_joined AND all_prepared AND all_members_clean`
- Added `can_advance_to_loading` gating variable
- Explicit logging when split reservation stalls due to missing clean verification:
  - `SPLIT_RESERVATION_BLOCKED group=... missing_prepared=... missing_clean=...`

### D. Join verification - IMPLEMENTED

Updated `_join_split_reservation()`:
- Calls `_verify_local_split_member_clean_precondition()` BEFORE joining
- If precondition fails: does NOT join, logs `SPLIT_JOIN_BLOCKED`
- If precondition passes: joins AND writes `member_clean[member]`
- Logs `SPLIT_MEMBER_CLEAN_OK` on successful verified join

### F. Claim-Time Ollama Attestation - IMPLEMENTED

Added to `gpu_tasks.py`:

1. `_attest_runtime_reality()` helper:
   - Probes actual Ollama `/api/ps` on relevant port (worker or split)
   - Compares to cached state (model_loaded, loaded_model, runtime_state)
   - Returns `{ok: bool, mismatch_reason: str, details: dict}`

2. Integration in `claim_tasks()`:
   - For LLM tasks, calls `_attest_runtime_reality()` after compatibility checks
   - On mismatch: calls `_release_task_for_mismatch()` then `_full_local_reset()`
   - Immediately returns claimed list (stops claiming for this cycle)

### G. Mismatch Handling = Full Local Reset - IMPLEMENTED

Added to `gpu_tasks.py`:

1. `_release_task_for_mismatch()`:
   - Releases task back to queue WITHOUT incrementing attempts/retry count
   - Does NOT add to `workers_attempted`
   - Records `last_mismatch_release` for observability only
   - Logs `CLAIM_RELEASED_RUNTIME_MISMATCH`

2. `_full_local_reset()`:
   - Stops split runtime if owner
   - Stops local worker Ollama
   - Kills orphan ollama runners for worker port
   - Clears all runtime state fields (model_loaded, loaded_model, etc)
   - Transitions to `RUNTIME_STATE_COLD`
   - Restarts Ollama for next claim cycle
   - Writes fresh heartbeat
   - Tracks `runtime_mismatch_count` for potential circuit breaker
   - Logs `FULL_LOCAL_RESET` and `FULL_LOCAL_RESET_COMPLETE`

### H. Task Accounting Semantics - IMPLEMENTED

- `_release_task_for_mismatch()` explicitly does NOT:
  - Increment `attempts` count
  - Add to `workers_attempted` list
  - Trigger retry/escalation logic
  - Mark task as failed
- Task is simply returned to queue for another worker

### Expected Log Events

New log events for observability:
- `SPLIT_JOIN_BLOCKED group=... member=... reason=...` - member failed clean precondition
- `SPLIT_MEMBER_CLEAN_OK group=... member=...` - member passed clean precondition and joined
- `SPLIT_RESERVATION_BLOCKED group=... missing_prepared=... missing_clean=...` - reservation stalled
- `RUNTIME_ATTESTATION_FAIL task=... worker=... reason=...` - claim-time mismatch detected
- `CLAIM_RELEASED_RUNTIME_MISMATCH task=... worker=... reason=...` - task released without retry
- `FULL_LOCAL_RESET worker=... reason=...` - worker performing full reset
- `FULL_LOCAL_RESET_COMPLETE worker=... mismatch_count=...` - reset completed

### E. Brain-Level Split Promotion Sequencing - IMPLEMENTED

Added to `brain_resources.py`:

1. `_split_group_members_needing_unload()` helper:
   - Checks if any split group members are hot with single-GPU runtimes
   - Returns list of members that need unload before split load can proceed

2. `_has_pending_unload_for_members()` helper:
   - Checks queue and processing for existing unload_llm tasks targeting members
   - Prevents duplicate unload task insertion

3. Updated `_make_resource_decisions()` split load insertion:
   - Before issuing `load_split_llm`, checks if any members need unload
   - If members are hot: issues targeted `unload_llm` for each hot member
   - If unloads already pending: logs "deferred" and waits for completion
   - Only issues `load_split_llm` when all members are verified cold

New log events:
- `RESOURCE_DECISION: Split promotion blocked: members [...] hot - issuing targeted unload_llm first`
- `RESOURCE_DECISION: Split promotion deferred: waiting for member unloads to complete`

### Files Modified

- `shared/agents/gpu_split.py`:
  - Added import for `RUNTIME_STATES_READY`
  - Added `_verify_local_split_member_clean_precondition()` helper
  - Updated `_join_split_reservation()` with precondition verification
  - Updated `_service_split_reservations()` with `all_members_clean` gating

- `shared/agents/gpu_tasks.py`:
  - Added import for `requests`, `RUNTIME_STATES_READY`
  - Added `_attest_runtime_reality()` helper
  - Added `_release_task_for_mismatch()` helper
  - Added `_full_local_reset()` helper
  - Updated `claim_tasks()` with claim-time attestation for LLM tasks

- `shared/agents/brain_resources.py`:
  - Added `_split_group_members_needing_unload()` helper
  - Added `_has_pending_unload_for_members()` helper
  - Updated load_split_llm insertion logic with sequencing checks

### JSON Repair Escalation - IMPLEMENTED (Historical, Superseded by Inline Repair)

This subsection documents the initial spawned-task JSON repair approach.
It is superseded by the later inline brain repair implementation described in:
- `Findings Addressed (2026-02-25 Late) -> Finding #1`
- `Findings Addressed (2026-02-25 Late - Round 2)`

Keep this section as history only; do not implement/restore the spawned-task flow.

Added to `brain_failures.py`:

1. `_is_json_format_failure()` helper:
   - Detects JSON format failure classes: `non_json_response`, `malformed_json`, `repair_failed`
   - Checks error text for common JSON parse error patterns

2. `_find_json_repair_artifacts()` helper:
   - Looks for raw worker response in batch failure dirs
   - Infers phase/schema from task name (extractor, verifier, adjudicator)

3. `_spawn_json_repair_task()` helper:
   - Creates a brain/CPU repair task linked to original task
   - Task includes raw response path and failure context

4. `_try_json_repair_escalation()`:
   - Called before normal retry logic in `handle_failed_tasks()`
   - If JSON failure detected and artifacts exist: spawns repair task
   - Marks original task as `pending_json_repair`
   - Returns True to skip normal retry

New log events:
- `JSON_REPAIR_SPAWNED task=... failure_class=...`
- `JSON_REPAIR_SKIPPED task=...` (no artifacts available)

### Startup Warm Policy Placement-Aware - IMPLEMENTED

Added to `startup.py`:

1. `_get_split_member_gpus()` helper:
   - Reads `models.catalog.json` to find GPUs in split groups
   - Returns set of GPU names that are split pair members

2. Updated `_enqueue_startup_load_llm()`:
   - Takes `available_workers` list and `agents_dir` path
   - Computes placement-aware preferred order:
     - Non-split GPUs first (e.g., gpu-2)
     - Split member GPUs last (e.g., gpu-1, gpu-3, gpu-4, gpu-5)
   - Logs: `Startup warm policy: non-split first [...], split-member last [...]`

This ensures split pairs remain cold on startup, preserving capacity for 14B work.

### Files Modified (Additional)

- `shared/agents/brain_failures.py`:
  - Added `_is_json_format_failure()` helper
  - Added `_find_json_repair_artifacts()` helper
  - Added `_spawn_json_repair_task()` helper
  - Added `_try_json_repair_escalation()` orchestration method
  - Updated `handle_failed_tasks()` to call JSON repair before normal retry

- `shared/agents/startup.py`:
  - Added `_get_split_member_gpus()` helper
  - Updated `_enqueue_startup_load_llm()` with placement-aware logic
  - Updated call site to pass available_workers and agents_dir

---

## Findings Addressed (2026-02-25 Late)

### Finding #1: JSON Repair Stranding Tasks - FIXED

**Problem:** `_try_json_repair_escalation()` spawned a separate repair task with no completion path, leaving original tasks stranded in `pending_json_repair` state indefinitely.

**Fix:** Rewrote `_try_json_repair_escalation()` in `brain_failures.py` to use **inline repair**:
- Attempts JSON repair directly using brain LLM model (not a separate task)
- On success: writes repaired output, clears error, requeues original task for re-run
- On failure: falls through to normal retry/escalation logic
- No separate task, no stranding - repair either works inline or doesn't

### Finding #2: Full Local Reset Missing Stale Ollama - FIXED

**Problem:** `_full_local_reset()` only stopped the tracked `ollama_process`, missing stale Ollama instances left from crashes or previous runs that still hold the worker port.

**Fix:** Updated `_full_local_reset()` in `gpu_tasks.py` to detect and kill stale listeners:
```python
# Kill ANY listener on worker port - catches stale Ollama from crashes
if self.port:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.5)
    try:
        port_in_use = sock.connect_ex(("127.0.0.1", self.port)) == 0
    finally:
        sock.close()
    if port_in_use:
        self._kill_local_listener_on_port(self.port)
```

### Finding #3: Split Clean-Precondition Fail-Open on Probe Exceptions - FIXED

**Problem:** `_verify_local_split_member_clean_precondition()` caught generic exceptions from port probes and logged them as debug, continuing as if precondition passed (fail-open).

**Fix:** Updated exception handling in `gpu_split.py` to fail-closed:
```python
except Exception as e:
    # Fail closed - if we can't probe, we can't guarantee member is clean
    result["ok"] = False
    result["reason_code"] = f"probe_error:{type(e).__name__}"
    self.logger.warning(f"Split precondition probe error (fail-closed): {e}")
    return result
```

### Finding #4: Claim-Time Attestation Only for LLM Tasks - FIXED

**Problem:** Claim-time runtime attestation only ran for `llm` tasks. Meta tasks like `load_llm`, `unload_llm`, and `load_split_llm` modify runtime state and equally need attested reality before execution.

**Fix:** Added attestation for state-changing meta tasks in `gpu_tasks.py`:
```python
# Claim-time attestation for state-changing meta tasks
meta_cmd = str(task.get("command", ""))
if meta_cmd in ("load_llm", "unload_llm", "load_split_llm"):
    attestation = self._attest_runtime_reality()
    if not attestation["ok"]:
        # ... release task, full local reset, return
```

### Finding #5: Split-Promotion Unload Skips Wrong Split Placements - FIXED

**Problem:** `_split_group_members_needing_unload()` skipped unload if `runtime_placement == "split_gpu"` without checking if the GPU was in the *same* split group. GPUs in a different split group still need unload.

**Fix:** Updated check in `brain_resources.py` to verify `runtime_group_id` matches target:
```python
target_group_id = str(group.get("id", "")).strip()
# ...
if str(state.get("runtime_placement", "")) == "split_gpu":
    member_group_id = str(state.get("runtime_group_id", "")).strip()
    if target_group_id and member_group_id == target_group_id:
        # Rejoining same split group - don't require unload
        continue
    # Different split group - still needs unload
```

---

## Findings Addressed (2026-02-25 Late - Round 2)

### Finding #6: JSON Repair Output Path Not Consumed - FIXED

**Problem:** `brain_failures.py` writes `json_repair_output_path` to the task, but `worker_review.py` ignores it, causing repair to have no effect and tasks to fail the same way again.

**Fix:** Two-part fix:

1. Added `_try_use_repaired_json()` helper in `worker_review.py`:
   - Checks for `json_repair_output_path` in task/args
   - Validates repair phase matches current phase
   - Loads repaired JSON and returns it instead of calling model
   - Tracks `json_repair_consumed` to prevent stale reuse

2. Added command-line args to `worker_review.py`:
   - `--json-repair-output-path`: Path to brain-repaired JSON
   - `--json-repair-phase`: Phase the repair was for

3. Updated extractor flow to check for repaired JSON before calling model:
```python
repaired_data, repair_note = _try_use_repaired_json(repair_task, f'{args.analysis_mode}_extractor')
if repaired_data is not None:
    # Use brain-repaired JSON instead of calling model
    llm = normalize_worker_output(repaired_data, snippets)
```

### Finding #7: Meta Task Attestation Not Command-Specific - FIXED

**Problem:** Previous fix added generic attestation for meta tasks, but different commands have different precondition requirements (load_llm needs cold, unload_llm doesn't care about mismatch).

**Fix:** Added `_attest_meta_task_precondition()` helper in `gpu_tasks.py`:
- `load_llm`: Requires cold state, fails if models present when state says cold
- `load_split_llm`: Same as load_llm, plus split-specific checks
- `unload_llm`/`unload_split_llm`: Informational only, allows continue on mismatch

```python
def _attest_meta_task_precondition(self, task: Dict[str, Any]) -> Dict[str, Any]:
    # Different commands have different precondition requirements
    # Returns {"ok": bool, "mismatch_reason": str, "allow_continue": bool}
```

### Finding #8: Full Local Reset Restarts Ollama Immediately - FIXED

**Problem:** `_full_local_reset()` called `start_ollama()` at the end, which can thrash if the mismatch cause persists (thermal/port race).

**Fix:** Three changes to `gpu_tasks.py`:

1. **Removed auto-restart**: Reset now ends in verified cold state without starting Ollama. Re-warm happens through normal task flow when `load_llm` is claimed.

2. **Added circuit breaker**: If N mismatches within M seconds, worker enters wedged state:
```python
if self.runtime_mismatch_count >= MISMATCH_CIRCUIT_BREAKER_COUNT:
    self.runtime_recovery_cooldown_until = now.timestamp() + MISMATCH_WEDGE_COOLDOWN_SECONDS
    self._set_runtime_state(RUNTIME_STATE_WEDGED, phase=f"circuit_breaker:{reason}")
    return
```

3. **Added constants** to `gpu_constants.py`:
```python
MISMATCH_CIRCUIT_BREAKER_COUNT = 3
MISMATCH_CIRCUIT_BREAKER_WINDOW_SECONDS = 300  # 5 minutes
MISMATCH_WEDGE_COOLDOWN_SECONDS = 120  # 2 minutes
```

### Finding #9: JSON Repair Metadata Incomplete - FIXED

**Problem:** Brain repair didn't track which phase/attempt the repair was for, making consumption logic fragile.

**Fix:** Updated `_try_json_repair_escalation()` in `brain_failures.py` to track:
```python
task["json_repair_phase"] = phase
task["json_repair_for_attempt"] = task.get("attempts", 0)
```

### Files Modified (Round 2)

- `shared/plans/shoulders/github_analyzer/scripts/worker_review.py`:
  - Added `_try_use_repaired_json()` helper
  - Added `--json-repair-output-path` and `--json-repair-phase` CLI args
  - Updated extractor flow to consume repaired JSON

- `shared/agents/gpu_tasks.py`:
  - Added `_attest_meta_task_precondition()` for command-specific attestation
  - Updated meta task claim to use command-specific attestation
  - Rewrote `_full_local_reset()` to remove auto-restart and add circuit breaker
  - Added import for new constants

- `shared/agents/gpu_constants.py`:
  - Added `MISMATCH_CIRCUIT_BREAKER_COUNT`
  - Added `MISMATCH_CIRCUIT_BREAKER_WINDOW_SECONDS`
  - Added `MISMATCH_WEDGE_COOLDOWN_SECONDS`

- `shared/agents/brain_failures.py`:
  - Added `json_repair_phase` and `json_repair_for_attempt` tracking

---

## Latest Batch Validation Notes (2026-02-25 Night)

### Batch `20260225_203857` End-State (Verify Benchmark)

- Batch now terminates correctly on `blocked_cloud` and writes:
  - `results/batch_failure.json`
- Observed failure:
  - `worker_review_verify_slice_008` escalated to `blocked_cloud`
  - reason: repeated `14B` verifier + adjudicator timeouts on split runtime (`11440`)
- Downstream benchmark summary tasks were correctly `abandoned` after batch abort (not zombie/stuck finalization).

This confirms:
- `blocked_cloud` local terminalization fix is working
- Remaining issue is long-running `14B` verify straggler handling (throughput/timeout problem), not batch-finalization logic

### Slice `slice_008` Lesson

`slice_008` (`contradictions_01`) is a heavy mixed docs/config/reference slice and is a repeat timeout straggler.

Implication:
- Even with runtime-integrity fixes, contradictions/verify slices can still exceed practical `14B` timeout budgets.
- This needs a manifest/splitting and/or phase-specific verifier strategy fix (not just runtime safety fixes).

Recommended direction (non-core analyzer/manifest work):
- split contradictions slices more aggressively by complexity
- add stricter per-slice caps for docs/reference-heavy verify work
- consider a docs-contradictions verifier mode or chunked verifier pass

---

## Remaining Gaps (Doc Claim vs Runtime Reality)

These are the remaining issues after the latest re-check. Some are core-agent changes (blocked here), and some are plan/script integration gaps.

### 1) Brain JSON repair artifact discovery - FIXED

`brain_failures.py::_find_json_repair_artifacts()` now:
- Searches `results/worker_review_failures/` (primary location)
- Falls back to `results/failures/`, `results/debug/` (legacy locations)
- Matches using JSON sidecar metadata (`slice_id`, `phase`) instead of filename guesses
- Uses most recent match when multiple candidates exist

### 2) Repaired JSON consumption extended to all phases - FIXED

`worker_review.py` now consumes repaired JSON in all phases:
- Extractor flow
- Verifier flow
- Deep-pass flow
- Adjudicator flow

Each phase checks `_try_use_repaired_json()` with phase guard before calling LLM.

### 3) Mismatch cooldown auto-recovery - FIXED

`gpu_state.py::_is_runtime_wedged()` now implements auto-recovery:
- If `runtime_recovery_cooldown_until` is set and elapsed, transitions wedged -> cold
- Clears mismatch tracking on recovery
- Writes fresh heartbeat

### 4) Repaired JSON wiring - FIXED

Implemented worker-side task context approach:

1. `gpu_workers.py::_spawn_worker()` now sets `WORKER_TASK_JSON_PATH` env var pointing to task JSON in `processing/`

2. `worker_review.py` reads repair metadata from task context via `_load_task_context_repair_metadata()`:
   - Checks `WORKER_TASK_JSON_PATH` env var
   - Reads `json_repair_output_path`, `json_repair_phase` from task JSON
   - Falls back to CLI args if env var not set

3. `_get_repair_metadata()` helper provides unified access (task context > CLI args)

This approach:
- Works dynamically per retry without plan template changes
- Works for any future worker script that needs repair metadata
- Keeps retry/recovery semantics in worker/brain path where they belong

Previous gap note (for reference):
- pass repair metadata into `worker_review.py` on retry/requeue path (preferred: worker gets task fields directly or worker launcher injects task metadata)
- if plan templating supports task-field placeholders, wire:
  - `--json-repair-output-path`
  - `--json-repair-phase`
- otherwise move consumption to worker-side task JSON reading instead of CLI-only args

Note:
- benchmark `worker_review.py` copy has been synced to the latest main script so it now contains the repaired-JSON consumption logic, but invocation wiring still needs to be solved.

### 3) Repaired JSON consumption is extractor-first; verifier/adjudicator/deep-pass consumption should be extended (ANALYZER SCRIPT WORK)

Current consumption path is verified in extractor flow.

Remaining work:
- add `_try_use_repaired_json()` checks for verifier/adjudicator/deep-pass phases
- guard with `json_repair_phase`
- emit repair markers per phase for observability

### 4) Mismatch circuit-breaker cooldown semantics are only partially implemented (BLOCKED: `shared/agents/*.py`)

Implemented:
- mismatch count window
- transition to `RUNTIME_STATE_WEDGED`
- `runtime_recovery_cooldown_until` is set

But the cooldown value is not clearly enforced/read in claim/recovery paths.

Consequence:
- safety is mostly okay (`wedged` blocks claims), but doc wording about cooldown/auto-recovery may overstate current behavior.

Robust fix (core):
- either implement explicit cooldown recovery/read path using `runtime_recovery_cooldown_until`
- or simplify policy/doc to `wedged until explicit recovery`

### 5) Unified doc cleanup: old JSON repair spawned-task text is still present (DOC HYGIENE)

The doc still contains older text describing:
- separate spawned repair tasks
- `pending_json_repair` state

That was superseded by inline brain repair.

Fix:
- remove or mark those older paragraphs as superseded to avoid future confusion.
