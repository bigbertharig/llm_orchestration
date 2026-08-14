# Rig Startup Issues & Fixes — April 30, 2026

Historical record. The `system_mode.json` and benchmark-reservation controls
described below were later replaced by `control_policy.json` and
`gpus/leases.json`. Do not use this file as the current operator runbook.

Session context: Attempting to run the research_prospector cloud search plan
after updating it to the new PLAN_FORMAT.md format with benchmark-backed model
selections. This was the first end-to-end plan run since the benchmarking work
in April.

## Follow-up Hardening — May 1, 2026

Goal: remove temporary Ollama compatibility and make stale startup/runtime
states fail or clean up deterministically.

Progress:
- [done] Add submit-time failure for legacy Ollama plans/scripts.
- [done] Remove `WORKER_OLLAMA_URL` compatibility export from worker runtime.
- [done] Use `write_system_mode()` in plan/benchmark startup wrappers.
- [done] Add startup cleanup for stale split reservations.
- [done] Tighten `kill_plan.py` brain-state cleanup.
- [done] Run focused tests and record results.

Policy decision: old Ollama-era plans must fail immediately with an upgrade
message. Do not keep hidden compatibility shims for `WORKER_OLLAMA_URL`,
`/api/generate`, `call_ollama`, or Ollama service assumptions. Plans/scripts
must use the llama runtime API via `WORKER_API_BASE`.

Implemented changes:
- `/home/bryan/llm_orchestration/scripts/submit.py` and
  `/media/bryan/shared/agents/submit.py` now reject active plan files/scripts
  containing legacy Ollama markers before any task is queued.
- `/media/bryan/shared/agents/worker.py` and
  `/home/bryan/llm_orchestration/shared/agents/worker.py` now export only
  `WORKER_API_BASE` for runtime endpoint discovery.
- `/home/bryan/llm_orchestration/scripts/start_plan_mode.py` and
  `/home/bryan/llm_orchestration/scripts/benchmarks/start_benchmark_mode.py`
  now call `write_system_mode()` instead of manually writing the JSON file.
- `/media/bryan/shared/agents/startup.py` and
  `/home/bryan/llm_orchestration/shared/agents/startup.py` now purge stale
  split reservations during startup when no matching split container/listener
  exists.
- `/home/bryan/llm_orchestration/scripts/kill_plan.py` now also clears
  `batch_queue`/`pending_tasks` and resets idle state when clearing all batches.

Verification:
- `python3 -m py_compile` passed for modified submit/startup/worker/mode/kill
  scripts.
- `python3 -m unittest tests.test_worker_runtime_readiness tests.test_system_mode
  tests.test_split_runtime_hardening` passed: 36 tests.
- Direct split purge smoke test removed a stale `pair_1_3` reservation and
  related lock/log artifacts.
- Old plan dry run against `shared/plans/shoulders/dc_integration` fails
  immediately with the legacy Ollama upgrade message.
- Updated `research_prospector/cloud_search_plan.md` dry run passes the legacy
  Ollama guard and reaches normal plan preview.

## Issue 1: Stale Split Pair State Blocks Worker Scan Loop

**Location**: `/mnt/shared/signals/split_llm/pair_1_3.json`

**Symptom**: Workers gpu-1 and gpu-3 stuck in a tight loop (every ~5s) logging:
```
SPLIT_READY_REJECTED group=pair_1_3 model=qwen2.5-coder:14b port=11440
has_model=False has_owner=True
```
Workers consumed their entire scan cycle on split validation and never scanned
the task queue.

**Root cause**: Split reservation from April 2 (28 days stale) had
`status: "ready"` with a valid-looking ready_token. Workers kept trying to
validate readiness against a Docker container that no longer exists.

**Fix**: Removed stale files:
```
/mnt/shared/signals/split_llm/pair_1_3.json
/mnt/shared/signals/split_llm/pair_1_3.json.lock
/mnt/shared/signals/split_llm/pair_1_3.runtime.log
/mnt/shared/signals/split_llm/pair_4_5.runtime.log
```
Then restarted workers (in-memory state is cached; file removal alone is
insufficient).

**Prevention needed**: Split state should either expire after a configurable TTL,
or startup.py should purge stale split reservations during Phase 2 init. A split
reservation older than the most recent startup timestamp is definitionally stale.

**May 1 update**: startup now purges stale split reservations before agent launch
when the reservation has no live `llama-split-<group>` container and no listener
on the reservation port.

---

## Issue 2: System Mode Defaults to `idle` After Startup

**Location**: `/mnt/shared/brain/system_mode.json`

**Symptom**: `prepare_llm_runtimes.py` queued `load_llm` meta tasks, but workers
never claimed them. Brain monitor showed `meta: 1` sitting in queue indefinitely.
Zero worker log output about task scanning.

**Root cause**: `startup.py` writes `mode: "idle"` after Phase 2 finishes. In the
`system_mode_allows_task()` function in `brain_core.py:266`:
```python
if mode == SYSTEM_MODE_PLANS:
    return True, ""
if mode == SYSTEM_MODE_BENCHMARKS:
    # Only allows specific benchmark meta commands
    ...
return False, "system_mode_idle"  # ← idle blocks everything
```

Workers call this check for every task in the queue. With mode=idle, all
non-system tasks are silently rejected. Workers don't log the rejection (it's
at debug level), making the failure completely silent in normal output.

**Fix**: Set mode to `plans` using the `write_system_mode()` API:
```python
from brain_core import write_system_mode
write_system_mode(Path("/mnt/shared"), "plans",
    reason="research_prospector smoke test", updated_by="operator")
```

**Note**: This is by design — `startup.py` is "neutral boot." The operator or a
wrapper script (like `start_plan_mode.py`) must explicitly set the mode before
work can flow. The issue is that `prepare_llm_runtimes.py` doesn't set the mode
either, so calling it directly after a bare `startup.py` always deadlocks.

---

## Issue 3: Wrong System Mode File Path

**Symptom**: Manually writing `system_mode.json` to `/mnt/shared/system_mode.json`
had no effect. Workers continued rejecting tasks.

**Root cause**: The actual path is `/mnt/shared/brain/system_mode.json`, defined
in `brain_core.py:211`:
```python
def system_mode_file(shared_path: Path) -> Path:
    return shared_path / "brain" / "system_mode.json"
```

**Fix**: Always use `write_system_mode()` API instead of manual file creation.

---

## Issue 4: Stale Lock Files from Previous Sessions

**Location**: `/mnt/shared/gpus/gpu_*/benchmark_reservation.json.lock`

**Symptom**: Leftover lock files from benchmark sessions. Not directly blocking
in this case, but could cause issues with future reservation claiming.

**Fix**: Removed stale `.lock` files.

---

## Issue 5: Brain Container Removed (from April 29 session)

**Symptom**: `docker start llama-brain` would fail because the container was
fully removed (not just stopped) during the previous session's split GPU
benchmark tests on the 3090.

**Fix**: `startup.py` creates a new container automatically. No manual
intervention needed — just restart.

---

## Issue 6: Plan Specified gemma-4:e2b-q8 But Runtime Can't Load It

**Symptom**: After fixing all startup issues and submitting the cloud search plan,
gpu-1 got stuck in `LLAMA_LOAD_WAIT worker=gpu-1 model=gemma-4:e2b-q8` forever.

**Root cause**: The `llm_model: gemma-4:e2b-q8` field in the plan causes the
orchestrator to load that specific model. But the current runtime image
(`llama-runtime:sm61-sm86`) cannot load Gemma 4 GGUF files — requires
`llama-runtime:b8884-candidate`.

The runtime image is a global config setting (`config.json:llama_runtime_image`),
not per-model. Mixing Gemma 4 and Qwen models requires the b8884 image for all
workers.

**Fix**: Changed the plan to use `qwen2.5-coder:7b` for all LLM tasks (smoke
test). The plan documents gemma-4:e2b-q8 as the target model with a note about
the runtime prerequisite.

**Long-term fix**: Promote `llama-runtime:b8884-candidate` to the default runtime
image, or implement per-model image selection in the worker agent.

---

## Issue 7: run_shoulder.py Path Resolution Fails from .submit_runtime Copy

**Location**: `/media/bryan/shared/plans/arms/research_prospector/scripts/run_shoulder.py`

**Symptom**: Smoke test batch `smoke_20260430_161605` failed with all 5
identify_people rounds reporting:
```
Shoulder script not found: /mnt/shared/plans/arms/research_prospector/input/shoulders/research_assistant/scripts/identify_people.py
```
Brain escalated to `blocked_cloud` after 3 retries.

**Root cause**: `submit.py` copies the plan into a `.submit_runtime/` subdirectory
under `input/`, so `{PLAN_PATH}` resolves to:
```
/mnt/shared/plans/arms/research_prospector/input/.submit_runtime/cloud_search_plan_<stamp>/
```
`run_shoulder.py` used `Path(__file__).parent.parent.parent.parent` (3 levels up
from `scripts/`) to find the `plans/` root. From the original location this
reaches `plans/`, but from the `.submit_runtime` copy it only reaches `input/`:
```
scripts/ → cloud_search_plan_<stamp>/ → .submit_runtime/ → input/  ← WRONG
```
The correct root (`plans/`) is 3 more levels up.

**Fix applied**: Replaced hardcoded 3-level traversal with upward walk that
searches for the `shoulders/` directory:
```python
candidate = arm_scripts_dir.parent
for _ in range(10):
    if (candidate / "shoulders").is_dir():
        plans_root = candidate
        break
    candidate = candidate.parent
```
This resolves correctly from both the original location (depth 3) and the
runtime copy (depth 6). Verified with test runs from both paths.

---

## Issue 8: Stale Brain Batch State Blocks Resubmission

**Location**: `/mnt/shared/brain/state.json`

**Symptom**: After killing the failed smoke test batch, resubmitting created new
tasks but the brain's `active_batches` still contained entries from the two
earlier failed attempts (`20260430_160753`, `20260430_161102`). The stale batch
entries could interfere with goal-driven orchestration.

**Fix**: Manually cleared `active_batches`, `batch_queue`, and `pending_tasks`
from `state.json` before resubmitting.

**Prevention needed**: `kill_plan.py` should clean batch entries from brain state,
or the brain should garbage-collect batches that have no active tasks.

---

## Issue 9: Stale Lock Files in Task Queue

**Location**: `/mnt/shared/tasks/queue/*.lock`

**Symptom**: 14 `.lock` files in the task queue directory with no corresponding
`.json` task files. These are orphans from previous sessions.

**Fix**: Removed all `.lock` files from `tasks/queue/`.

---

## Issue 10: WORKER_OLLAMA_URL Env Var Not Set — Scripts Hit Wrong Port

**Location**: `/media/bryan/shared/agents/worker.py`

**Symptom**: Smoke test batch `smoke_20260430_202134` ran identify_people tasks
successfully (path fix worked), but workers showed 0% GPU utilization despite
having qwen2.5-coder:7b loaded. Tasks hung for minutes without producing output.

**Root cause**: All shoulder scripts (`identify_people.py`, `score_person.py`,
`plan_searches.py`, `transform_output.py`, etc.) read the LLM endpoint from
env var `WORKER_OLLAMA_URL`, defaulting to `http://localhost:11434` (Ollama's
legacy port). But `worker.py` only set `WORKER_API_BASE` — the shoulder scripts
never saw it and connected to port 11434 instead of the worker's llama-server
port (e.g. 11437).

Confirmed via `ss -tnp`: all identify_people processes had ESTAB connections to
`127.0.0.1:11434` instead of their assigned worker ports.

**Fix applied**: Added `WORKER_OLLAMA_URL` env var to `worker.py` alongside
the existing `WORKER_API_BASE`:
```python
runtime_api_base = args.api_base or args.runtime_url
if runtime_api_base:
    os.environ["WORKER_API_BASE"] = runtime_api_base
    # Legacy env var used by shoulder scripts
    os.environ["WORKER_OLLAMA_URL"] = runtime_api_base
```

After fix, confirmed connections go to correct ports (11435, 11436, 11437).

**Scope**: This affected every shoulder script that makes LLM calls. The fix
is backward-compatible — new worker.py sets both env vars, old scripts read
whichever one they know about.

**Long-term fix**: Rename all `WORKER_OLLAMA_URL` references in shoulder scripts
to `WORKER_API_BASE` and remove the legacy shim from worker.py.

**May 1 update**: the legacy shim was removed from worker.py. Submit now fails
old Ollama-era plans/scripts immediately, so remaining `WORKER_OLLAMA_URL`
references must be upgraded before those plans can run.

---

## Smoke Test Results

### Batch: `smoke_20260430_203515` (brain batch `20260430_203600`)

**Plan**: `cloud_search_plan.md` with all tasks using `qwen2.5-coder:7b`

**Status**: Running successfully. Full pipeline flowing:
```
identify_people → plan_searches → scrape_person → score_person → (compile_output pending)
```

**What works**:
- Plan format update correct (llm_model, llm_min_tier, llm_placement fields)
- Task creation, dependency resolution, foreach expansion all correct
- `run_shoulder.py` path fix resolves from `.submit_runtime` copy
- `WORKER_OLLAMA_URL` propagation works — connections to correct worker ports
- Goal-driven discovery loop: 10 rounds, 44+ candidates extracted
- Validation pipeline: plan_searches, scrape_person, score_person all execute
- Brain resource management: auto-loads models on cold GPUs as needed

**What doesn't work (expected)**:
- Extraction quality with qwen2.5-coder:7b is terrible — extracts discipline
  names ("Geographic Information Science", "Geospatial Data Science") instead
  of actual people
- All scored candidates auto-rejected (fit_score=5, decision=reject)
- This confirms the March 2026 extraction test findings: extraction needs a
  model with strong reading comprehension, not a code model
- Batch will exhaust max_rejections (25) and compile empty output

### Smoke Test Conclusion

The plan format rewrite is **validated**. The pipeline infrastructure works
end-to-end. The only gap is model quality, which is the expected outcome of
using qwen2.5-coder:7b as a temporary stand-in.

---

## Next Steps

### 1. Runtime Image Upgrade (Required for Real Run)

Switch from `llama-runtime:sm61-sm86` to `llama-runtime:b8884-candidate`:

```bash
# In /mnt/shared/agents/config.json:
# Change: "llama_runtime_image": "llama-runtime:sm61-sm86"
# To:     "llama_runtime_image": "llama-runtime:b8884-candidate"
```

Then restart the rig. This unblocks Gemma 4 GGUF loading on all workers.

### 2. Revert cloud_search_plan.md to Target Models

Change `identify_people` back from `qwen2.5-coder:7b` to `gemma-4:e2b-q8`:
```
- **llm_model**: gemma-4:e2b-q8
```
The plan_searches and score_person tasks stay on `qwen2.5-coder:7b` (correct
per benchmarks).

### 3. Run Real Batch

Resubmit with same query (`gis_university_contacts.md`) and target=5. With
gemma-4:e2b-q8 handling extraction, expect actual person names and meaningful
triage scores.

### 4. Longer-Term Cleanup

- Rename `WORKER_OLLAMA_URL` → `WORKER_API_BASE` in all shoulder scripts
- Remove `call_ollama` function naming (it's llama-server now, not Ollama)
- Add split state TTL or startup purge to prevent Issue 1 recurrence
- Fix `kill_plan.py` to clean brain `active_batches` state (Issue 8)
- Consider making `b8884-candidate` the default runtime image if Gemma 4
  compatibility holds across all existing models

---

## Files Modified This Session

| File | Change |
|------|--------|
| `plans/arms/research_prospector/cloud_search_plan.md` | Full PLAN_FORMAT.md update; benchmark-backed model selections; temporary qwen2.5-coder:7b for smoke test |
| `plans/arms/research_prospector/plan.md` | Full PLAN_FORMAT.md update; benchmark-backed model selections (gemma-4:e2b-q8 + qwen2.5-coder:7b) |
| `workspace/PLAN_FORMAT.md` | Removed Ollama migration notes; added clean LLM Runtime API section |
| `plans/arms/research_prospector/scripts/run_shoulder.py` | Path resolution: hardcoded parent traversal → upward directory walk |
| `agents/worker.py` | Added `WORKER_OLLAMA_URL` env var for shoulder script compatibility |

---

## Operational Lessons

1. **After bare `startup.py`, always set system mode.** Use `start_plan_mode.py`
   or `start_custom_mode.py` instead of bare `startup.py` + manual mode setting.

2. **Split state has no expiry.** Stale split reservations can wedge workers
   indefinitely. Check `/mnt/shared/signals/split_llm/*.json` if workers are
   unresponsive after restart.

3. **Worker task rejection is silent at INFO level.** If workers are running but
   not claiming tasks, the mode check is the most likely culprit. Check
   `/mnt/shared/brain/system_mode.json`.

4. **`prepare_llm_runtimes.py` requires an active mode.** It does not set the
   mode itself. If called after a neutral startup, it will always timeout.

5. **Runtime image compatibility is global.** If a plan uses a Gemma 4 model,
   ALL workers need the b8884 image. There is no per-worker image override.

6. **`run_shoulder.py` path resolution breaks under `.submit_runtime` copy.**
   Any arm script that calculates relative paths from `__file__` must account
   for the extra directory nesting created by `submit.py`'s runtime copy.
   Use upward directory walks instead of hardcoded parent counts.

7. **Always clear brain batch state between failed resubmissions.** Stale
   `active_batches` entries in `state.json` persist across restarts and can
   interfere with goal-driven orchestration.

8. **Shoulder scripts use `WORKER_OLLAMA_URL`, not `WORKER_API_BASE`.** Until
   the rename is done, worker.py must set both env vars. If GPU utilization is
   0% despite models being loaded, check which port the script is connecting to
   with `ss -tnp | grep python`.

9. **Use `ss -tnp` to diagnose LLM connectivity issues.** When scripts appear
   stuck but GPUs show 0% utilization, check network connections — the script
   may be connecting to the wrong port (e.g. Ollama 11434 vs worker 11437).
