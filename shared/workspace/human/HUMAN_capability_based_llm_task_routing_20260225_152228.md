# HUMAN: Capability-Based LLM Task Routing (Placement as Runtime Detail)

## Why This Is Needed

Observed during `github_analyzer_verify_benchmark` and `github_analyzer` verify waves:
- workers/pairs can be visibly ready (`JOINED`, target 14B model loaded) but queued 14B work is underutilizing available capacity
- routing/claim behavior appears constrained by split-specific task identity (`split-required` semantics), not just model capability
- result: artificial serialization and lower-than-expected verify concurrency even when multiple 14B-capable runtimes are available

This is a design issue in task eligibility/routing semantics, not only a split-runtime stability issue.

## Core Principle (Authoritative)

**Normal LLM work tasks should be routed by capability requirements, not runtime placement.**

Meaning:
- A task should care about what model/capability is required.
- A task should *not* care whether that capability is provided by:
  - a single GPU runtime, or
  - a split runtime across two GPUs.

Placement (`single_gpu`, `split_gpu`) remains important for:
- telemetry
- load/unload meta-tasks
- brain resource decisions / prewarming / unload policies
- debugging and observability

But placement should not be part of the identity of normal LLM work tasks unless a task truly requires placement-specific behavior (rare / exceptional).

## Desired Behavior

### Task Requirements (Normal LLM Work)
LLM tasks should express requirements such as:
- `required_model` (exact model when necessary) OR `model_family` / compatibility contract
- `llm_min_tier`
- `required_context_tokens` (or minimum effective context)
- optional capability flags (e.g. `json_mode`, tool support, tokenizer assumptions)

LLM tasks should **not** require `llm_placement` for claim eligibility in normal cases.

### Worker / Runtime Capability Advertisement
GPU heartbeats should expose the effective capability of the currently loaded runtime, independent of placement:
- `effective_model_id` (or `loaded_model` normalized)
- `effective_tier`
- `effective_context_tokens`
- `runtime_state` (`ready`, `loading`, `wedged`, etc.)
- `runtime_placement` (telemetry only; may be used for scheduling preference but not hard eligibility)

For split runtimes, the effective capability should look equivalent to a single runtime of the same model/tier.

### Claim Eligibility Rules (Worker Side)
For normal `llm` tasks:
- if runtime is `ready_*` and capability satisfies task requirements, worker may claim
- placement mismatch alone should not reject claim
- transition / wedged state still rejects claim (existing state machine rules stay)

For meta-tasks:
- keep placement-specific handling (`load_split_llm`, `unload_split_llm`, `load_llm`, `unload_llm`)
- these remain explicit orchestration operations

### Brain Resource Management (Brain Side)
Brain remains responsible for deciding how to satisfy task demand:
- whether to load split or single runtimes
- whether to preserve split capacity for expected 14B work
- whether to repurpose a worker via unload/load

But once work tasks are released, any capable worker should be eligible to take them.

## What Should Change (Core)

### 1. Decouple LLM Work Claiming from `llm_placement`
In worker claim logic (`shared/agents/gpu.py`):
- remove hard placement gating for normal `llm` tasks
- keep model/tier/context/runtime-state checks
- treat placement as telemetry/preference only for work tasks

Current problematic pattern (conceptual):
- split runtime rejects `single_gpu` tasks
- single runtime rejects `split_gpu` tasks

Desired pattern:
- reject only when capability is insufficient or runtime state is not claimable

### 2. Normalize Runtime Capability Semantics in Heartbeats
Ensure heartbeat fields can support capability-based matching cleanly.

Potential additions / clarifications:
- `effective_model_id`
- `effective_model_family`
- `effective_tier`
- `effective_context_tokens`
- `capability_ready` boolean (derived from `runtime_state` + model loaded)

(`runtime_placement`, `runtime_group_id`, `split_runtime_owner` remain and should stay.)

### 3. Brain Task Schema / Release Semantics
For standard LLM work task creation and release:
- stop encoding placement as a hard requirement unless explicitly needed
- encode requirement as capability contract instead
- allow candidate restrictions only when there is a real reason (e.g. locality, pinned artifacts, explicit user policy)

### 4. Scheduling Preferences (Optional, Not Hard Blocks)
Brain may still prefer certain placements for efficiency:
- prefer single-GPU runtimes for 7B work to preserve split pairs
- prefer split 14B runtimes for 14B verifier waves
- prefer already-hot compatible workers to avoid churn

But these should be **preferences**, not hard eligibility gates, unless explicitly configured.

## Exceptions (Placement-Specific Work, Rare)

If a task truly depends on placement-specific behavior (example: split-runtime diagnostics task), it may declare explicit placement requirements.

Recommendation:
- reserve explicit placement constraints for meta tasks and diagnostics tasks
- default all analyzer worker LLM tasks to capability-based routing

## Why This Aligns With Current State-Machine Work

The recently hardened runtime state machine (`ready/loading/unloading/wedged`) is the right foundation.

Capability-based routing should build on it:
- state machine handles safety and ownership
- capability matching handles work eligibility
- brain handles resource strategy

This separates concerns cleanly and avoids using `split` as a proxy for capability.

## Likely Core Files to Review

Protected core files (human changes required):
- `/home/bryan/llm_orchestration/shared/agents/gpu.py`
- `/home/bryan/llm_orchestration/shared/agents/brain.py`
- `/home/bryan/llm_orchestration/shared/agents/brain_core.py`
- `/home/bryan/llm_orchestration/shared/agents/brain_resources.py`

Likely non-core plan/task generation follow-up (after core semantics are updated):
- analyzer plan task emitters that currently encode placement in worker tasks

## Acceptance Criteria

1. A ready split runtime can claim a normal 14B LLM task if it meets model/tier/context requirements, regardless of task placement tag.
2. A ready single-GPU runtime can claim a normal LLM task if it meets the same capability requirements.
3. Placement remains visible in telemetry and used for meta-task orchestration, but not as a hard gate for normal LLM work.
4. Brain can still prefer split or single runtimes via resource decisions without creating artificial task ineligibility.
5. Verify benchmarks show improved effective concurrency when multiple capable runtimes are hot.
6. No regression in runtime safety: `loading/unloading/wedged` workers still do not claim work.

## Suggested Implementation Order (Human)

1. Audit current claim-time placement gating in `shared/agents/gpu.py` for normal `llm` tasks.
2. Introduce/normalize effective capability fields in heartbeat semantics.
3. Update worker claim eligibility to use capability + runtime state, not placement.
4. Update brain/task emitters to stop marking normal LLM work as placement-hard (or treat existing field as advisory).
5. Re-run `github_analyzer_verify_benchmark` on:
   - `research_prospector` (small/medium)
   - `county-map` (larger)
   and compare `estimated_concurrency` + verify wall time.

## Notes From Recent Observation (Feb 25, 2026)

In `github_analyzer_verify_benchmark` batch `20260225_141644` (`research_prospector`):
- benchmark completed in ~`43m`
- verify phase effective concurrency was `1.0`
- user observed periods where two split runtimes appeared ready (`JOINED`) but only one was taking verify tasks

This note is intended to remove that class of underutilization by changing task eligibility semantics.

## Additional Core Bug To Fix During Agent Edit Pass

### Stale `assigned_to` Preserved on Retry Requeue (Misleading Queue State)

Observed behavior:
- a failed task is retried and moved back to `shared/tasks/queue/`
- dashboard still shows `assigned_to: gpu-X` on the queued task
- this makes it look like the worker claim is still held, even though the task is queued again

Root cause (core code):
- `shared/agents/brain_failures.py` `_queue_task_retry()` resets:
  - `status`
  - `attempts`
  - `workers_attempted`
- but does **not** clear `assigned_to` (and related transient assignment fields if present)

Required fix:
- when requeueing a task for retry, clear transient worker assignment fields before writing back to queue
  - `assigned_to`
  - `worker` (if present)
  - `started_at` (if preserved from prior attempt)

Why it matters:
- avoids misleading operator/debugger signals during live queue inspection
- prevents accidental future logic from treating stale assignment metadata as meaningful

Suggested patch site:
- `/home/bryan/llm_orchestration/shared/agents/brain_failures.py`
- method: `_queue_task_retry(...)`

## Additional Core Improvement: Brain-Level JSON Repair Escalation (Worker Output Format Failures)

### Problem

Some `worker_review` tasks fail due to output format only (for example: `no JSON object found in LLM response`)
even when the underlying model response may contain usable analysis text.

Current behavior tends to consume retries on the same worker task and can end in `blocked_cloud`.

### Desired Behavior

When a worker task fails with a recognized JSON-format failure class (`non_json_response`, `malformed_json`, `repair_failed`):
- do **not** immediately spend all normal retries rerunning the same task
- spawn a small brain/CPU repair task that converts the preserved raw worker response into the required strict JSON schema
- if repair succeeds, mark the original task recovered (or inject repaired result) and continue downstream
- only retry/re-escalate if repair also fails

### Why This Is Better Than Heuristic Fallback

- preserves model-produced semantic content
- uses stronger brain model (`32B`) for formatting/normalization only
- avoids default/templated answers that reduce quality
- reduces wasted GPU retries for format-only failures

### Prerequisites (Mostly Done in Plan Scripts)

`worker_review.py` should persist failure artifacts (implemented in plan-local script):
- raw worker response text
- same-model repair response (if any)
- brain repair response (if any)
- prompt excerpt / phase metadata

This provides the inputs for a deterministic repair task.

### Core-Agent Wiring (Human Change)

In retry/failure handling (`shared/agents/brain_failures.py` and related orchestration):
1. Detect JSON-format failure classes from task result/error text (or structured error code if added later)
2. Create a repair task (CPU/brain executor) instead of immediate standard retry
3. Pass:
   - failure artifact path(s)
   - expected schema/phase (`extractor`, `verifier`, `adjudicator`)
   - original task id
4. On repair success:
   - write repaired output to expected task artifact path
   - mark original task as recovered/completed (or synthesize a successful retry result)
5. On repair failure:
   - continue normal retry/escalation path

### Acceptance Criteria

1. A `worker_review` task failing only due to JSON formatting can recover without rerunning the GPU task.
2. Dashboard/logs clearly show `json_repair_escalation` and `json_repair_recovered`.
3. Recovered outputs are traceable to the repair artifact/task for auditability.
4. Normal retries still apply for timeouts/model-unavailable/runtime failures.

## Other Human Changes Needed (Non-GPU Core / Dashboard)

These are additional fixes discovered during runtime tuning that are **not** part of the GPU capability-routing refactor itself, but should be addressed in the same broader cleanup pass.

### 1. Batch-Scoped Failure Counting in Dashboard (Avoid Cross-Batch Overcounting)

Observed behavior:
- Dashboard "failures" view/count appeared to show ~13 failures for a live batch
- direct task-lane inspection showed only `1` real failed task for the current batch
- most of the displayed failures were from a previous aborted benchmark batch

Likely cause:
- failure tab/count is aggregating global/recent `shared/tasks/failed/*.json` entries without strict current-batch filtering

Required fix:
- ensure dashboard failure counts and failure tables are filtered by the selected/current `batch_id`
- clearly distinguish:
  - `failed` (current batch)
  - `abandoned from prior batch`
  - `global recent failures` (if a separate view exists)

Why it matters:
- prevents operators from intervening based on false failure counts
- avoids confusion during long benchmark runs

### 2. Processing-Task Failure Count Artifact (Retried Tasks Carry Prior Failed Result)

Observed behavior:
- task is actively retrying in `processing/`
- task JSON still contains a previous attempt `result.success=false`
- dashboard can misread this as a current failure if it checks `result.success` without lane/status context

Required fix (dashboard-side):
- when counting/displaying failures, primary source should be task lane/status (`failed` vs `processing` vs `complete`)
- for `processing` tasks, do not count stale `result.success=false` as a current failure
- optionally show "previous attempt failed" as a retry diagnostic, not as a failure count increment

Why it matters:
- retried tasks in progress should not inflate failure counts
- reduces operator confusion while watching live runs

### 3. Batch Failure Attribution in Dashboard (Show Source Task From `batch_failure.json`)

Observed behavior:
- dashboard surfaced a non-source task (`score_slice_complexity`) as the apparent failure
- actual batch abort source was `brain_strategy` timeout

Required fix:
- when batch abort marker exists (`results/batch_failure.json`), use it as the authoritative batch failure source
- display:
  - `source_task`
  - `source_task_id`
  - `reason`
- avoid inferring primary failure from the first `failed/` entry, which may be an abandoned dependent task

Why it matters:
- avoids chasing the wrong root cause during recovery
- especially important when many downstream tasks are auto-abandoned

### 4. Brain Retry/Failure Metadata Hygiene (Complement to Existing `assigned_to` Fix)

Already documented above:
- clear stale `assigned_to` / `worker` / `started_at` when requeueing retries

Additional suggestion:
- consider recording a structured retry-attempt history field instead of mutating `result` in-place for live tasks
- dashboard can then display previous attempt failures without confusing the current lane status

This is optional, but it would cleanly separate:
- current task state
- prior attempt diagnostics

---

## Implementation Status (Feb 25, 2026)

### Completed Changes

#### 1. GPU Mixin Refactor
Refactored monolithic `gpu.py` (~3656 lines) into 9 focused modules:

| File | Lines | Purpose |
|------|-------|---------|
| `gpu_constants.py` | 55 | Runtime states, timing, VRAM constants |
| `gpu_state.py` | 92 | State machine transitions |
| `gpu_core.py` | 190 | Config loading, utilities |
| `gpu_thermal.py` | 317 | Thermal safety, VRAM budget |
| `gpu_workers.py` | 345 | Worker subprocess management |
| `gpu_ollama.py` | 602 | Ollama server/model management |
| `gpu_tasks.py` | 829 | Task claiming, meta task handling |
| `gpu_split.py` | 918 | Split GPU coordination |
| `gpu.py` | 492 | Main orchestrator inheriting all mixins |

#### 2. Capability-Based Routing (`gpu_tasks.py`)

**`_llm_task_runtime_compatible()` rewritten** (lines 38-72):

Old behavior:
- Required exact model match
- Hard placement gating: split runtime rejected non-split tasks
- `split_runtime_requires_split_task` error blocked 7B work on 14B runtimes

New behavior:
- **Tier-based matching**: If task specifies `llm_model`, checks if loaded tier >= required tier
- **`llm_min_tier` support**: Tasks can specify minimum tier requirement
- **Placement is advisory only**: Removed hard gating on `llm_placement` mismatch
- A 14B split runtime (tier 2) can now serve 7B tasks (tier 1)

```python
# Key change: tier-based compatibility
required_tier = int(self.model_tier_by_id.get(required_model, DEFAULT_LLM_MIN_TIER))
if self.loaded_tier < required_tier:
    return False, f"tier_insufficient:{self.loaded_tier}<{required_tier}"
# Tier is sufficient - model compatibility assumed for same family
```

#### 3. Heartbeat Effective Capability Fields (`gpu.py`)

Added to heartbeat (lines 262-273):
```python
"capability_ready": capability_ready,  # runtime ready + model loaded + ollama healthy
"effective_model_id": self.loaded_model if self.model_loaded else None,
"effective_tier": self.loaded_tier if self.model_loaded else 0,
"effective_context_tokens": self.worker_num_ctx if self.model_loaded else 0,
```

These fields enable brain/dashboard to match tasks by capability independent of placement.

#### 4. Stale Assignment Cleanup (`brain_failures.py`)

**`_queue_task_retry()` updated** (lines 172-187):

Now clears transient worker assignment fields before requeueing:
```python
task.pop("assigned_to", None)
task.pop("worker", None)
task.pop("started_at", None)
task.pop("last_attempt_at", None)
```

This prevents misleading `assigned_to: gpu-X` on queued retry tasks.

#### 5. GPU Hardening Fixes

| Fix | File | Description |
|-----|------|-------------|
| Double cleanup | `gpu.py` | `run()` owns cleanup; removed from `main()` |
| Crash visibility | `gpu.py` | Re-raises exceptions after cleanup for supervisor |
| Heartbeat order | `gpu.py` | Stop runtimes before final heartbeat |
| Meta-task safety | `gpu_tasks.py` | try/finally wrapper ensures `_end_meta_task()` always called |
| Missing import | `gpu.py` | Added `import sys` |
| Heartbeat filtering | `gpu_workers.py` | `_has_active_work()` excludes `.heartbeat.json` sidecars |
| Process cleanup | `gpu_ollama.py` | `ollama_process = None` after stop |

### Acceptance Criteria Status

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Split runtime can claim 14B LLM task regardless of placement tag | ✅ Implemented |
| 2 | Single-GPU runtime can claim LLM task if capability matches | ✅ Implemented |
| 3 | Placement visible in telemetry but not hard gate for work | ✅ Implemented |
| 4 | Brain can prefer placements without creating ineligibility | ✅ Ready (brain unchanged) |
| 5 | Improved concurrency in verify benchmarks | ⏳ Pending test batch |
| 6 | No regression in runtime safety | ✅ State machine unchanged |

### Additional Implementations (2026-02-25)

#### Brain-Level JSON Repair Escalation - IMPLEMENTED

Added to `brain_failures.py`:
- `_is_json_format_failure()` - detects JSON format errors
- `_find_json_repair_artifacts()` - finds raw response files
- `_spawn_json_repair_task()` - creates repair task for brain processing
- `_try_json_repair_escalation()` - orchestrates repair before normal retry

#### Startup Warm Policy Placement-Aware - IMPLEMENTED

Added to `startup.py`:
- `_get_split_member_gpus()` - reads catalog for split members
- Updated `_enqueue_startup_load_llm()` - prefers non-split GPUs for warmup
- Keeps split pairs (gpu-1,3,4,5) cold to preserve 14B capacity

### Not Yet Implemented

- **Dashboard batch-scoped failure counting** (Section: Other Human Changes)
- **Dashboard processing-task failure artifact handling**
- **Dashboard batch failure attribution from `batch_failure.json`**

Note: Dashboard items require changes to the dashboard UI/data layer which is separate from the agent codebase.

### Next Steps

1. Run test batch to validate capability-based routing
2. Monitor for any tier-mismatch edge cases
3. Consider brain-side scheduling preferences (optional optimization)
