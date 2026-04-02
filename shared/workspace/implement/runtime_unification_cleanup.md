# Runtime Unification Cleanup

Goal: make the orchestration system operate as layered components with a stable
runtime control surface.

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
   - `start_default_mode.py`
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

## Progress Checkpoint

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
1. Finish validating `return_default.py` against a full live restart cycle with the new dynamic wait budget.
2. Run a small `github_analyzer` plan smoke on the stabilized path to flush real plan/runtime bugs.
3. Add candidate runtime image build/smoke/promotion path.
4. Defer Gemma 4 runtime upgrade until the current plan/runtime path is solid.

New findings during `return_default.py` integration:
- fixed: `_read_gpu_heartbeat()` had unreachable code and effectively returned `None` for all GPUs
- fixed: `runtime_preflight.py` was flagging expected startup listeners as conflicts during managed `load_llm` transitions
- fixed: `return_default.py` can now fall back to rig-side runtime probe results when heartbeat `model_loaded` state lags behind reality
- fixed: `return_default.py` no longer uses a stale fixed `90s` wait budget by default; it derives wait time from the configured single-model load timeout/profile
- remaining live issue: default startup load for `qwen2.5-coder:7b` on `gpu-2` is materially slower than the original wrapper assumptions and should be treated as a real system characteristic or tuned down
- likely next bug-finder after this: run a `github_analyzer` smoke batch, since it exercises real submission, sequencing, and verifier/runtime behavior instead of just model loads
