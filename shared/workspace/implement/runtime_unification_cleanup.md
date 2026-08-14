# Runtime Unification Cleanup

Goal: make the orchestration system operate as layered components with a stable
runtime control surface.

## August 13, 2026 Audit Update

The latest whole-system review is in
`system_unification_audit_2026-08-13.md`. The highest-priority gap is now the
benchmark campaign scheduler's direct ownership of GPUs, worker ports, and
`llama-campaign-*` containers outside the orchestrator reservation and
heartbeat surfaces. Do not treat runtime unification as complete until
campaigns acquire shared leases and preflight recognizes those leases.

First shared/Plan implementation slice completed in `llm_orchestration`:

- `gpu_leases.py` is the atomic single/group allocation authority;
- `control_policy.py` separates boot profile from controller admission;
- Plan claim/resource paths now exclude controller-leased GPUs;
- dashboard status overlays global leases even without worker heartbeats.

Still open: Plan-owned lease lifecycle, preflight/runtime generation checks,
and a transactional profile transition service.

The April progress and live-validation sections below are retained as a
historical implementation log. References there to `system_mode.json`,
`idle`, and global Plan/benchmark modes describe the superseded control path;
the current contract is `control_policy.json` plus `gpus/leases.json`.

## Target Model

Layers:
- runtime base layer: versioned `llama-runtime` images plus stable wrapper scripts
- control/mode layer: startup, benchmark, custom, submit, kill, return-to-default
- package layer: benchmark harnesses and plan repos
- ownership layer: orchestrator owns worker runtime ports and model lifecycle

Operational rule:
- benchmark runs and plans should depend on stable wrappers and config
- direct `run_runtime.sh` is debug-only
- llama.cpp upgrades should happen by promoting a new runtime image, not by editing agent code

## Problems Found

1. Runtime image selection is hardcoded in agent/runtime code.
2. Docs had stale wrapper paths and stale `status.py` expectations.
3. Shared path assumptions drift between laptop (`/media/bryan/shared`) and rig (`/mnt/shared`).
4. Unmanaged debug containers can interfere with orchestrator-owned worker GPUs.
5. Current llama runtime image is too old for Gemma 4.

## Gemma 4 Result

Observed on April 2, 2026:
- current image: `llama-runtime:sm61-sm86`
- current llama.cpp ref: `b8250`
- direct smoke result: `unknown model architecture: 'gemma4'`

Conclusion:
- Gemma 4 is blocked by runtime support, not tuning

## Cleanup Plan

1. Make runtime image/tag selectable from config instead of hardcoded code defaults.
2. Keep stable and candidate runtime images side by side.
3. Add a smoke/promotion path for runtime images:
   - known-good single-worker model
   - known-good split model
   - target new model family
4. Keep mode wrappers as the operator control surface:
   - `start_plan_mode.py`
   - `start_benchmark_mode.py`
   - `start_custom_mode.py`
   - `submit.py`
   - `kill_plan.py`
5. Keep direct runtime commands debug-only and clearly documented.
6. Add preflight/cleanup handling for unmanaged runtime containers.
7. Normalize path handling and docs around:
   - rig `10.0.0.3` -> `/mnt/shared/...`
   - laptop `10.0.0.2` -> `/media/bryan/shared/...`
8. Run a system certification sweep after cleanup.

## Immediate Implementation Steps

1. Add runtime image config field and thread it through runtime helpers.
2. Update tests and docs to reference the unified runtime image selection path.
3. Build a candidate llama runtime image with newer llama.cpp for Gemma 4 smoke tests.
4. Promote candidate image only after smoke passes on known-good and new model families.

## Historical April Progress Checkpoint

Completed:
- `llama_runtime_image` is now config-driven instead of hardcoded in agent runtime code
- stable smoke path verified with `qwen3.5:4b`
- existing benchmark harness verified on stable image with `custom_command_safety`
- rig-side runtime preflight script added
- repo-side runtime preflight wrapper added
- benchmark/custom mode paths now use runtime cleanup/preflight
- docs updated for rig-local worker port rule and active runtime notes

Current operator commands:
- `python3 ~/llm_orchestration/scripts/runtime_preflight.py --config config.benchmark.json --json`
- `python3 ~/llm_orchestration/scripts/benchmarks/start_benchmark_mode.py --json`
- `python3 ~/llm_orchestration/scripts/benchmarks/start_custom_mode.py --models qwen3.5:4b --force-unload-first --json`

Current next steps:
1. Commit the startup/mode/heartbeat hardening patch set.
2. Fix reasoning benchmark prompt/extraction fit for current worker-tier models before scaling out more medium runs.
3. Run medium benchmark sanity passes on additional worker-tier models through the stabilized benchmark path.
4. Add candidate runtime image build/smoke/promotion path.
5. Defer Gemma 4 runtime upgrade until the current benchmark/runtime path is solid.

New findings during `return_default.py` integration:
- fixed: `_read_gpu_heartbeat()` had unreachable code and effectively returned `None` for all GPUs
- fixed: `runtime_preflight.py` was flagging expected startup listeners as conflicts during managed `load_llm` transitions
- fixed: repo-side `runtime_preflight.py` now normalizes shared-path aliases like `/home/bryan/llm_orchestration/shared/...` to rig paths before proxying
- fixed: `brain_plan.py` now substitutes plan variables into `llm_model`, `llm_min_tier`, and `llm_placement` instead of only command strings
- fixed: `return_default.py` can now fall back to rig-side runtime probe results when heartbeat `model_loaded` state lags behind reality
- fixed: `return_default.py` no longer uses a stale fixed `90s` wait budget by default; it derives wait time from the configured single-model load timeout/profile
- fixed: startup enqueue no longer duplicates `startup_single_default` when a fresh meta heartbeat exists without a sibling task json
- fixed: single-load heartbeat phases are now explicit during startup/load (`waiting_runtime_ready`, etc.) instead of staying stale on `acquiring_global_lock`
- fixed: system execution mode is now explicit via `system_mode.json` with `idle`, `plans`, and `benchmarks`
- fixed: startup now forces `idle`, `submit.py` flips to `plans`, and benchmark startup flips to `benchmarks`
- fixed: brain/worker task claim paths now honor explicit system mode instead of relying only on queue heuristics
- fixed: stale startup meta tasks in `brain/private_tasks` no longer block plan execution after startup config changes
- fixed: startup stale-task purge now includes `brain/private_tasks`, not just public queue/processing
- fixed: startup enqueue blockers now ignore non-system `load_llm` tasks, so preserved plan queue state no longer deadlocks startup recovery
- fixed: worker heartbeat now reconciles against the live llama runtime probe when cached state is stale after reboot/startup recovery
- fixed: startup now flips `system_mode.json` from `idle/startup_in_progress` to `idle/startup_finished` only after the configured startup target is actually ready
- fixed: `start_benchmark_mode.py` no longer crashes while writing benchmark mode because the embedded remote Python block no longer uses a broken f-string/dict literal combination
- fixed: worker startup cold reset now clears stale single-runtime loaded state after container cleanup, so benchmark-mode workers no longer advertise dead runtimes as `ready_single`
- fixed: benchmark mode now allows `runtime_prep_*` benchmark meta tasks (`load_llm`, `unload_llm`, split runtime prep) while still blocking ordinary non-benchmark plan work
- fixed: local custom benchmark runner now falls back to `reasoning_content` when `message.content` is empty, which is required for current `qwen3.5:4b` chat responses on this rig
- fixed: dashboard worker holding labels no longer treat terminal runtime phases like `startup_cold_reset` or `load_complete` as active tasks/holds

## Historical April Live Validation

Validated live on April 2, 2026:
- real host reboot now comes back in `idle`
- default startup on `gpu-2` auto-loads `qwen2.5-coder:7b`
- `/v1/models` returns `200` after the post-reboot warm load completes
- `system_mode.json` now settles to `idle` with reason `startup_finished` after the default startup target is ready
- preserved `github_analyzer` queue state survives reboot
- after startup normalization, mode can be flipped back to `plans` and preserved batch work resumes

Validated plan path:
- real production plan used: `github_analyzer/plan.md`
- active smoke batch: `20260402_195559`
- batch queue/private tasks survived reboot as expected
- the old deadlock is fixed:
  - before: preserved batch `load_llm` blocked startup enqueue, while `idle` blocked the preserved batch task from running
  - now: startup still enqueues/runs `startup_single_default`, then preserved plan work can resume once mode returns to `plans`

Observed startup/load timing:
- single-worker `qwen2.5-coder:7b` load on `gpu-2` can legitimately take around 4 minutes on reboot/recovery paths
- during the long wait:
  - container can already be up
  - VRAM can already be allocated
  - API can still return `503 Loading model`
  - heartbeat phase now shows the worker is in runtime readiness wait instead of a stale earlier phase

Validated live on April 3, 2026:
- `python3 ~/llm_orchestration/scripts/benchmarks/start_benchmark_mode.py --json` now succeeds again and writes `system_mode.json = benchmarks`
- benchmark-mode worker startup now comes up genuinely cold instead of preserving stale `ready_single` state from a killed worker runtime
- `python3 ~/llm_orchestration/scripts/benchmarks/start_custom_mode.py --models qwen3.5:4b --force-unload-first --json` now successfully queues and completes orchestrator-owned benchmark runtime prep
- fresh live worker load validated:
  - worker: `gpu-1`
  - port: `11435`
  - model: `qwen3.5:4b`
  - `/v1/models`: returns `Qwen3.5-4B-Q4_K_M.gguf`
- fresh live benchmark validation:
  - `custom_command_safety` against `qwen3.5:4b` initially recorded a false `0/12` because the harness only read `message.content`
  - direct endpoint probe confirmed the model was responding in `reasoning_content` with empty `content`
  - after the harness fallback fix, rerun recorded `12/12` and the shared reference now shows latest `custom_command_safety = 1.0` for `qwen3.5:4b`
- completed medium saved benchmark campaign:
  - campaign: `gpu1_worker_qwen35_4b_medium_reasoning`
  - run id: `20260403_qwen35_4b_medium_reasoning_v1`
  - worker: `gpu-1`
  - suite: `bench-reasoning`
  - tasks: `gsm8k,bbh,drop`
  - limit: `10`
  - final state: `completed`
  - task results:
    - `gsm8k`: `0.5`
    - `bbh`: `0.0`
    - `drop`: `0.0`
- benchmark interpretation from that run:
  - infrastructure path is now working end to end
  - the next issue is benchmark configuration quality for reasoning-heavy tasks on current qwen worker models
  - `bbh = 0.0` and `drop = 0.0` with `gsm8k = 0.5` strongly suggests prompt/extraction mismatch, not just a broken runtime
  - the current strict worker system prompt likely overconstrains `bbh_cot_fewshot_*` style tasks and may also be hurting `drop`
- dashboard validation after the campaign:
  - live API/backend state matched the actual worker state
  - `gpu-1` correctly showed the active reasoning task
  - `gpu-2/3/4` were cold with no active task
  - the UI bug was specifically that terminal runtime phases were being surfaced as worker holds

## April 3 Historical State

What is solid now:
- startup/default mode separation is working
- plan-mode isolation is working
- benchmark-mode isolation is working
- preserved-queue reboot recovery no longer deadlocks on batch-owned `load_llm`
- real `github_analyzer` plan parsing and task creation work on the production `plan.md` path
- post-reboot heartbeat/runtime surface is repaired when `/v1/models` is already healthy but cached worker state is still `cold`
- idle startup now has an explicit finished state instead of lingering on `startup_in_progress`
- benchmark runtime prep can once again claim/load orchestrator-owned meta tasks inside `benchmarks` mode
- local custom benchmark grading is now compatible with current `qwen3.5:4b` response format on the llama runtime
- dashboard worker cards now distinguish real active work from terminal runtime-state markers
- current live baseline:
  - default/startup baseline:
    - `system_mode.json` = `idle` / `startup_finished`
    - `gpu-2` = `ready_single`
    - `loaded_model` = `qwen2.5-coder:7b`
    - `/v1/models` = `200`
  - benchmark validation baseline:
    - `system_mode.json` = `benchmarks` / `benchmark_mode_requested`
    - benchmark workers start cold
    - validated benchmark worker load:
      - `gpu-1` = `ready_single`
      - `loaded_model` = `qwen3.5:4b`
      - `runtime_port` = `11435`

Operational takeaway:
- plan-side restart/recovery flow is now good enough to proceed
- benchmark-side work has moved from smoke/debug into real saved medium-run validation
- the first saved medium reasoning run completed successfully at the orchestration/runtime layer
- the current blocker is no longer runtime ownership/startup state; it is benchmark prompt/extraction fit for some reasoning suites on worker-tier qwen models
- immediate benchmark strategy should be:
  - inspect raw `bbh`/`drop` outputs from `qwen3.5:4b` and adjust prompt profile / harness behavior if the model is being overconstrained
  - rerun a short reasoning A/B on `qwen3.5:4b` after that adjustment
  - medium sanity runs first (`limit 10-25`) on additional worker-tier targets only after the qwen reasoning result shape looks sane
  - then higher-cost (`limit 50-100`) runs only after results and response formats look sane
