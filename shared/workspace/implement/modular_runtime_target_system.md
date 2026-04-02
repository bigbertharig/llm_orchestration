# Modular Runtime Target System

Purpose: capture the desired end-state for runtime, benchmark, and plan
execution, plus the concrete code paths that currently need cleanup.

## Desired System

The system should behave as four clean layers:

1. Runtime base layer
- versioned `llama-runtime` images
- stable wrapper scripts
- swappable llama.cpp version without agent code edits

2. Control/mode layer
- `start_default_mode.py`
- `start_benchmark_mode.py`
- `start_custom_mode.py`
- `submit.py`
- `kill_plan.py`
- `return_default.py`

3. Package layer
- benchmarks as packages that run on top of the runtime/control layer
- plans as packages that run on top of the runtime/control layer

4. Resource ownership layer
- each GPU owns its own runtime state
- workload classes use explicit reservations
- a benchmark GPU should not claim plan tasks
- a plan GPU should not be stolen by a benchmark launch
- queued work should wait for the GPU/resource owner to become available

## Operational End-State

Normal operation:
- all worker model loads/unloads happen through orchestrator meta tasks
- all mode switches run through one supported control surface
- runtime image upgrades are promoted by config, not code edits
- debug-only direct runtime launches are isolated and never confused with orchestrator-owned runtimes

Future coexistence target:
- benchmarks and plans can run at the same time on different GPUs
- workload scheduling respects GPU reservations and queueing
- if a GPU is busy, competing work waits instead of colliding

## What Exists Already

Good pieces already present:
- config-driven llama runtime tuning profiles:
  - `shared/agents/config*.json`
  - `shared/agents/brain_core.py:resolve_llama_runtime_profile`
- shared runtime helpers:
  - `shared/scripts/llama_runtime/run_runtime.sh`
  - `build_image.sh`
  - `build_and_smoke_test.sh`
- orchestrator-owned worker runtime path:
  - `shared/agents/gpu_llama.py`
  - `shared/agents/gpu_split.py`
- benchmark wrappers:
  - `scripts/benchmarks/start_benchmark_mode.py`
  - `scripts/benchmarks/start_custom_mode.py`
- return/default wrapper:
  - `scripts/return_default.py`
- live status helper:
  - `scripts/status.py`
- split reservation machinery:
  - `shared/agents/gpu_split.py`

## Gaps Found

### 1. Runtime image selection was not layered

Before cleanup:
- `gpu_llama.py` hardcoded `llama-runtime:sm61-sm86`

Now fixed:
- `llama_runtime_image` is config-driven
- single and split launches both resolve image through shared config logic

Still needed:
- candidate/stable promotion workflow
- image certification/smoke tooling

### 2. Mode control is split across repo-side and shared-side paths

Current reality:
- repo-side wrappers live under `~/llm_orchestration/scripts/benchmarks/`
- shared-side runtime helpers live under `shared/scripts/llama_runtime/`

This is acceptable, but only if docs and tests consistently describe the boundary.

Still needed:
- one operator runbook for all mode transitions
- a single preflight gate used by benchmark/custom/default transitions

### 3. Stale state still causes conflicts

Observed classes of failure:
- stale tasks in `tasks/processing/`
- unmanaged debug containers occupying worker GPUs
- ports responding with 503 or refusing while orchestration believes a load is active
- worker restarts during intentional resets

Still needed:
- strong preflight before mode switches and custom loads
- explicit detection of unmanaged worker runtime containers
- explicit port ownership checks
- clearer cleanup of wedged `processing` tasks / heartbeats

### 4. Pathing still drifts across machines

Live path rules:
- rig `10.0.0.3` uses `/mnt/shared/...`
- laptop `10.0.0.2` uses `/media/bryan/shared/...`

Still needed:
- all scripts that can run from either side should normalize shared paths
- tests/docs should stop implying non-existent shared helper paths

### 5. Runtime capability is pinned behind an old llama.cpp ref

Observed:
- current image uses llama.cpp `b8250`
- Gemma 4 GGUFs fail with `unknown model architecture: 'gemma4'`

Conclusion:
- model support upgrades belong in the runtime image promotion flow

## Concrete Code Areas To Change Next

### Preflight / cleanup gate

Likely targets:
- `scripts/benchmarks/start_benchmark_mode.py`
- `scripts/benchmarks/start_custom_mode.py`
- `scripts/return_default.py`
- `shared/scripts/prepare_llm_runtimes.py`

Needed behaviors:
- scan worker ports
- detect unmanaged llama containers
- detect stale processing meta tasks
- detect split reservations left active
- either fail loudly or clean deterministically

### GPU reservation / workload class separation

Likely targets:
- `shared/agents/gpu.py`
- `shared/agents/gpu_tasks.py`
- `shared/agents/gpu_state.py`
- `shared/agents/gpu_split.py`
- brain-side scheduling/resource logic

Needed behaviors:
- explicit GPU mode or reservation state
- benchmark worker exclusion from plan-task claiming
- queued handoff when the GPU becomes available

### Runtime image promotion tooling

Likely targets:
- `shared/scripts/llama_runtime/build_image.sh`
- `shared/scripts/llama_runtime/build_and_smoke_test.sh`
- maybe a new promotion wrapper script

Needed behaviors:
- build candidate image
- smoke known-good single-worker model
- smoke known-good split model
- smoke new target family
- promote by config change only after passing

## Recommended Next Sequence

1. Finish stable smoke verification on the current image.
2. Build the preflight/cleanup gate for mode transitions and custom loads.
3. Add candidate runtime image build/smoke/promotion flow.
4. Upgrade llama.cpp in a candidate image for Gemma 4.
5. After runtime hardening, start the GPU reservation/workload-class separation work.
