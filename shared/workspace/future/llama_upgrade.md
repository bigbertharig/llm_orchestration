# Ollama -> llama.cpp Migration Plan

## Goal

Move runtime ownership from Ollama to `llama-server` so model lifecycle is:

- explicit
- process-based
- easier to observe
- easier to kill and recover
- simpler for split-GPU serving

This is not expected to fix every historical batch failure. A large share of prior failures were plan/config/report issues. The value of this migration is narrower and more important:

- reduce runtime indirection
- remove Ollama-specific load/unload behavior
- simplify split runtime ownership
- make future debugging more direct

## Why This Is Worth Doing

The recent runtime failure cluster is concentrated in model loading and split readiness, not ordinary task execution:

- repeated split preflight timeouts with `ready_groups=[]`
- repeated `load_split_llm_model_qwen2_5_coder_14b_pair_1_3` failures
- repeated `load_llm` escalation failures

The current design spreads runtime state across:

- `ollama serve`
- Ollama API semantics (`/api/generate`, `/api/ps`, `/api/tags`, `keep_alive`)
- local cleanup logic
- split reservation logic
- orphan runner cleanup
- port ownership recovery

`llama-server` will not remove all orchestration complexity, but it removes one whole layer of wrapper behavior and gets runtime control closer to the actual serving process.

## What The Codebase Looks Like Today

Based on current code exploration:

- [gpu_ollama.py](/home/bryan/llm_orchestration/shared/agents/gpu_ollama.py) owns single-GPU Ollama lifecycle, health checks, load, unload, and readiness probing.
- [gpu_split.py](/home/bryan/llm_orchestration/shared/agents/gpu_split.py) is the main complexity sink for split serving, ownership metadata, warmup, readiness validation, orphan cleanup, and recovery.
- [gpu_tasks.py](/home/bryan/llm_orchestration/shared/agents/gpu_tasks.py) assumes Ollama runtime probes via `/api/ps` and dispatches `load_llm`, `unload_llm`, `load_split_llm`, and `unload_split_llm`.
- [gpu.py](/home/bryan/llm_orchestration/shared/agents/gpu.py) still builds worker API URLs around `/api/generate`.
- [brain_core.py](/home/bryan/llm_orchestration/shared/agents/brain_core.py), [brain.py](/home/bryan/llm_orchestration/shared/agents/brain.py), and [startup.py](/home/bryan/llm_orchestration/shared/agents/startup.py) still assume Ollama startup and health semantics.
- [hardware.py](/home/bryan/llm_orchestration/shared/agents/hardware.py), [launch.py](/home/bryan/llm_orchestration/shared/agents/launch.py), [setup.py](/home/bryan/llm_orchestration/shared/agents/setup.py), and [kill_plan.py](/home/bryan/llm_orchestration/scripts/kill_plan.py) also contain Ollama-specific control logic.

That means this migration is real work, but it also means the seam lines are clear.

## Core Design Choice

### Current Model

- start `ollama serve`
- trigger model load indirectly via API request
- infer loaded state via `/api/ps`
- trigger unload indirectly via `keep_alive=0`
- special-case split runtimes with additional ownership and cleanup logic

### Target Model

- start `llama-server` with an explicit GGUF path
- server process equals loaded model
- readiness comes from process health and API health
- unload by killing the server process
- split runtime is one explicit `llama-server --tensor-split ...` process

That is the simplification we want.

## Architecture Decision

Chosen direction:

- keep the orchestration control plane on the host
- run serving runtimes in containers
- run benchmark harnesses in separate containers
- treat plan/task containers as a future selective capability, not Phase 1 scope

This means:

- `llm-orchestrator.service` remains the owner of brain/worker supervision, queue state, heartbeats, and recovery
- `llama-server` runs as one container per runtime
- benchmark execution stays isolated from serving/runtime containers
- we do not move the orchestrator itself into Docker as part of this migration

Why this is the right split:

- it preserves one clear owner for startup and recovery
- it isolates heavy runtime dependencies from the orchestration venv
- it avoids Docker-in-Docker or nested runtime ownership
- it keeps benchmark churn from contaminating serving/runtime stability
- it leaves room to containerize specific plan/task classes later where that actually helps

Not chosen for this migration:

- containerizing the full orchestrator stack
- running serving containers from inside another container
- making all plans use containers immediately

Companion build/runtime spec:

- [llama_runtime_image_spec.md](/home/bryan/llm_orchestration/shared/workspace/future/llama_runtime_image_spec.md)

Current implementation artifacts:

- [Dockerfile](/home/bryan/llm_orchestration/scripts/llama_runtime/Dockerfile)
- [entrypoint.sh](/home/bryan/llm_orchestration/scripts/llama_runtime/entrypoint.sh)
- [build_image.sh](/home/bryan/llm_orchestration/scripts/llama_runtime/build_image.sh)
- [run_runtime.sh](/home/bryan/llm_orchestration/scripts/llama_runtime/run_runtime.sh)
- [stop_runtime.sh](/home/bryan/llm_orchestration/scripts/llama_runtime/stop_runtime.sh)
- [probe_runtime.sh](/home/bryan/llm_orchestration/scripts/llama_runtime/probe_runtime.sh)

## Constraints

- Shared GGUF files under `/mnt/shared/models/` remain the source of truth.
- Local SSD remains hot storage only when explicitly used.
- Brain/worker orchestration model stays the same.
- Worker default remains cold start.
- Migration should be phaseable.
- Rollback must remain possible until cleanup phase is complete.
- Container ownership must remain single-layer:
  - host orchestrator launches containers directly
  - no Docker-in-Docker for runtime control

## Operational Baseline Before Backend Migration

This groundwork is now required, not optional:

- `llm-orchestrator.service` must remain the single default owner of `startup.py`.
- Default recovery paths must restart the service, not spawn detached competing launchers.
- The service-owned startup path must run from `~/llm-orchestration-venv`, not `~/ml-env`.
- Startup warm-up must prove one default worker reaches `ready_single` after service start and after reboot.

Why this belongs in the llama migration plan:

- If runtime ownership is ambiguous, moving from Ollama to `llama-server` will just reproduce the same class of bugs with different processes.
- The service/venv cleanup we just completed is the baseline the llama migration should build on.
- Do not introduce a second launcher path for llama runtimes. Keep one owner.

## Non-Goals

- fixing historical plan placeholder bugs
- fixing report synthesis quality problems
- redesigning the entire brain scheduler
- changing queue/task semantics
- changing dashboard behavior beyond runtime control plumbing

## API Mapping

| Purpose | Ollama | llama-server |
|---|---|---|
| Health | `GET /api/tags` | `GET /health` |
| Loaded model presence | `GET /api/ps` | `GET /v1/models`, `GET /slots`, process alive |
| Chat inference | `POST /v1/chat/completions` | `POST /v1/chat/completions` |
| Raw completion with logprobs | weak/broken via Ollama path | `POST /v1/completions` |
| Load model | generate request + warmup | start process |
| Unload model | generate request with `keep_alive=0` | stop process |
| Split serving | multi-layer Ollama coordination | one process with `--tensor-split` |

## Major Simplifications We Expect

1. No implicit model loading through a generation call.
2. No implicit unloading through `keep_alive`.
3. No Ollama blob cache duplication.
4. No split pair built out of multiple Ollama runtime assumptions.
5. No proxy workaround for logprobs on completions.
6. Fewer ambiguous runtime states because the server process is the model.

## Risks That Still Exist

1. Split orchestration logic will still exist even if the serving backend gets simpler.
2. A single `llama-server` process can still crash or hang under load.
3. GPU targeting flags may differ from assumptions and need proof on this rig.
4. Context window, batching, and memory tuning will still need benchmarking.
5. Brain and worker code may still carry old Ollama assumptions in subtle places.

## Prerequisites

Before changing agent code:

- [x] Verify whether a working host-side `llama-server` binary exists on the rig.
- [ ] Verify single-GPU pinning works on a GTX 1060 with either `--device` or `CUDA_VISIBLE_DEVICES`.
- [ ] Verify `--tensor-split` works for the intended 2x1060 split pair.
- [ ] Verify `/v1/chat/completions` works with the client payloads we already send.
- [ ] Verify `/v1/completions` returns usable logprobs for the benchmark use case.
- [ ] Verify startup time and steady-state memory behavior with `--mlock` on representative models.
- [ ] Record one known-good manual command for:
  - single worker model
  - brain model
  - split model

Observed now:

- Host-side `llama-server` binary is not installed on the rig.
- Host-side `llama-cpp-python` is not installed in the active orchestration venv.
- Local fallback exists in Docker image `bench-knowledge:latest`, which already contains `llama_cpp`.
- Default orchestration startup is now service-owned and proven on `~/llm-orchestration-venv`.

## Manual Validation Commands To Capture

Fill these in once proven on the rig:

- [ ] Brain runtime command
- [x] Single-worker runtime command
- [ ] Split runtime command
- [ ] Health probe command
- [x] Chat inference smoke test command
- [x] Completion-with-logprobs smoke test command
- [ ] Kill/unload command

Phase 0 findings so far:

- Working local fallback runtime for experiments:

```bash
docker run --rm --gpus 'device=1' --network host \
  -v /mnt/shared/models:/mnt/shared/models \
  --entrypoint python3 bench-knowledge:latest \
  -m llama_cpp.server \
  --model /mnt/shared/models/qwen2.5-coder-7b/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf \
  --host 127.0.0.1 --port 18080 \
  --n_gpu_layers -1 --n_ctx 2048 --use_mlock True
```

- Working chat smoke test:

```bash
curl -s http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Reply with exactly: OK"}],"max_tokens":8}'
```

- Working completions + logprobs smoke test:

```bash
curl -s http://127.0.0.1:18080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Say OK","max_tokens":4,"logprobs":5}'
```

- Working model-list probe:

```bash
curl -s http://127.0.0.1:18080/v1/models
```

- Stop/unload for containerized proof:

```bash
docker rm -f <container_id>
```

## Migration Strategy

Do this in controlled phases with hard exit criteria.

Preferred order:

1. runtime proof outside agents
2. single-worker backend
3. brain backend
4. split runtime backend
5. support scripts and dashboard-adjacent controls
6. cleanup

Do not start by rewriting split runtime first.

## Phase 0: Runtime Proof And Baseline

Purpose:
- prove `llama-server` works on this hardware before changing orchestration code
- capture real commands and real constraints

Checklist:
- [ ] Build or install `llama-server`
- [x] Run one single-GPU worker model manually
- [ ] Run brain model manually
- [ ] Run one split model manually with `--tensor-split`
- [ ] Confirm health endpoint behavior
- [x] Confirm model listing behavior
- [x] Confirm inference payload compatibility
- [x] Record startup times
- [x] Record steady-state VRAM usage
- [x] Record any required flags beyond the current plan assumptions

Artifacts to save:
- [x] exact commands
- [x] successful curl examples
- [x] startup timings
- [x] notes on `--device` vs `CUDA_VISIBLE_DEVICES`
- [x] notes on `--mlock`
- [x] notes on split GPU constraints

Exit criteria:
- [x] one single-GPU command works end-to-end
- [ ] one split command works end-to-end
- [ ] we know the exact runtime flags to standardize on

Rollback:
- no code rollback needed; this phase is proof only

### Phase 0 Findings

1. There is no host-side `llama-server` binary on the rig today.
2. There is no host-side `llama-cpp-python` install in the main rig venv.
3. `bench-knowledge:latest` provides a usable local fallback for proving runtime behavior without downloading anything.
4. The `llama-cpp-python` server in that container supports:
   - `/v1/models`
   - `/v1/chat/completions`
   - `/v1/completions` with `logprobs`
5. The original plan assumption about `/health` is wrong for this fallback server. `GET /health` returned `404`.
6. `GET /v1/models` is the reliable readiness probe observed so far.
7. `--use_mlock True` does not block startup, but it emits a warning under the current memlock limit:
   - `warning: failed to mlock ... Try increasing RLIMIT_MEMLOCK`
8. Startup for the 7B single-GPU worker case was slow but successful, roughly on the order of about 60-70 seconds to become reachable.
9. `docker run --gpus 'device=1' ...` correctly restricted the single-GPU case to one 1060.
10. For split proof, using `NVIDIA_VISIBLE_DEVICES` alone was not sufficient for precise device control in this container path.
11. A six-value tensor split mask did steer layer assignment correctly:

```bash
--tensor_split 0 1 0 1 0 0
```

12. That split mask assigned layers to `CUDA1` and `CUDA3` only, which is promising for the target design.
13. The 14B split case did not reach HTTP readiness within the test window, so split is only partially proven.
14. The 32B brain-model case was started on the 3090, but was not carried all the way to HTTP readiness within the test window.

### Current Phase 0 Status

- Single-worker fallback proof: proven
- Logprobs on completions: proven
- Host-side production binary availability: blocked
- Split readiness on target 14B: partially proven only
- Brain-model readiness on 32B: partially proven only
- Service-owned orchestration baseline: proven

### Implications For The Migration

- We should not assume `/health` exists across all candidate runtimes. The production plan needs a concrete runtime choice before hard-coding health semantics.
- The host binary prerequisite is real. Before Phase 1 agent work, we either:
  - install/build a real host-side `llama-server`, or
  - consciously decide that production agents will use a containerized server path
- The split concept is promising, but we do not yet have a fully proven ready command for the 14B pair.
- The plan should continue in this order:
  - finish host/runtime choice
  - finish split readiness proof
  - only then start agent rewrites

## Phase 1: New Runtime Mixin For Single-GPU Serving

Phase 1 implementation status now:

- [x] dedicated runtime image directory created
- [x] host-side build helper created
- [x] host-side run/stop/probe helpers created
- [ ] dedicated image built locally on the rig
- [ ] dedicated image proven with one real single-worker model
- [ ] agent runtime mixin added
- [ ] agent path switched from Ollama to llama

Primary files:
- [gpu_ollama.py](/home/bryan/llm_orchestration/shared/agents/gpu_ollama.py)
- [gpu.py](/home/bryan/llm_orchestration/shared/agents/gpu.py)
- [gpu_core.py](/home/bryan/llm_orchestration/shared/agents/gpu_core.py)

Plan:
- [x] create dedicated image and launcher helpers under `scripts/llama_runtime/`
- [ ] add `gpu_llama.py`
- [ ] implement server start
- [ ] implement server stop
- [ ] implement health check
- [ ] implement readiness check
- [ ] capture process PID and command line in runtime state
- [ ] preserve strong runtime diagnostics on failure
- [ ] remove dependency on Ollama `keep_alive` for worker lifecycle

Notes:
- Server start becomes the load operation.
- Server stop becomes the unload operation.
- Keep the external orchestration contract stable where possible.
- Keep service ownership unchanged. Phase 1 swaps the backend under the existing service/supervisor model; it does not introduce a parallel launcher.
- Use the new wrapper scripts as the canonical command source instead of embedding raw `docker run` strings in multiple places.

Exit criteria:
- [ ] one worker can start a model-backed runtime on demand
- [ ] one worker can unload it by killing the process
- [ ] failure logging shows real command/process/health information

Rollback:
- [ ] keep `gpu_ollama.py` intact until Phase 6
- [ ] allow worker path to switch back to Ollama during transition

### Phase 1 Canonical Commands

Build the dedicated image:

```bash
/home/bryan/llm_orchestration/scripts/llama_runtime/build_image.sh
```

Start one worker runtime:

```bash
/home/bryan/llm_orchestration/scripts/llama_runtime/run_runtime.sh \
  --name llama-worker-gpu1 \
  --model /mnt/shared/models/qwen2.5-coder-7b/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf \
  --port 11436 \
  --gpus device=1
```

Probe readiness:

```bash
/home/bryan/llm_orchestration/scripts/llama_runtime/probe_runtime.sh 11436
```

Stop the runtime:

```bash
/home/bryan/llm_orchestration/scripts/llama_runtime/stop_runtime.sh llama-worker-gpu1
```

## Phase 2: Wire Single-GPU Agent Path

Primary files:
- [gpu.py](/home/bryan/llm_orchestration/shared/agents/gpu.py)
- [gpu_tasks.py](/home/bryan/llm_orchestration/shared/agents/gpu_tasks.py)
- [gpu_workers.py](/home/bryan/llm_orchestration/shared/agents/gpu_workers.py)
- [worker.py](/home/bryan/llm_orchestration/shared/agents/worker.py)

Checklist:
- [ ] replace `/api/generate` assumptions
- [ ] replace `/api/ps` readiness assumptions for single runtimes
- [ ] rename worker URL variables from Ollama naming to llama naming where appropriate
- [ ] keep task contract stable for `load_llm` and `unload_llm`
- [ ] update heartbeats and runtime state fields as needed
- [ ] verify existing observability fields still populate meaningful failure detail

Acceptance tests:
- [ ] `load_llm` succeeds on a cold worker
- [ ] `unload_llm` returns the worker to cold state
- [ ] one real LLM task runs end-to-end on a worker
- [ ] failed startup produces useful batch artifacts

Rollback:
- [ ] revert worker mixin selection to Ollama path

## Phase 3: Brain Runtime Migration

Primary files:
- [brain.py](/home/bryan/llm_orchestration/shared/agents/brain.py)
- [brain_core.py](/home/bryan/llm_orchestration/shared/agents/brain_core.py)
- [brain_dispatch.py](/home/bryan/llm_orchestration/shared/agents/brain_dispatch.py)

Checklist:
- [ ] replace brain startup from Ollama to `llama-server`
- [ ] replace brain health checks
- [ ] replace brain inference endpoint assumptions
- [ ] update env var names and config keys
- [ ] verify brain startup and shutdown behavior

Acceptance tests:
- [ ] brain starts with the intended model
- [ ] brain can process one plan from startup to initial task release
- [ ] restart behavior works cleanly

Rollback:
- [ ] allow brain to continue on Ollama while workers use llama if needed

## Phase 4: Split Runtime Rewrite

Primary files:
- [gpu_split.py](/home/bryan/llm_orchestration/shared/agents/gpu_split.py)
- [gpu_tasks.py](/home/bryan/llm_orchestration/shared/agents/gpu_tasks.py)
- [brain_resources.py](/home/bryan/llm_orchestration/shared/agents/brain_resources.py)

This is the highest-risk phase.

Current pain points in code:
- ownership metadata
- staged startup
- repeated `/api/ps` probing
- owner/follower handoff logic
- orphan cleanup
- postcondition cleanup

Target:
- one explicit split `llama-server` process
- one health/readiness model
- much less orphan-runner recovery logic

Checklist:
- [ ] define canonical split startup command
- [ ] define canonical split readiness criteria
- [ ] update split reservation data model if needed
- [ ] remove dual-Ollama assumptions
- [ ] remove Ollama-specific warmup/load calls
- [ ] keep reservation and ownership logic only where still necessary
- [ ] simplify cleanup around one runtime process
- [ ] keep strong failure artifacts for split launch, readiness, and teardown

Acceptance tests:
- [ ] split runtime loads across target GPUs
- [ ] split runtime serves one inference successfully
- [ ] `load_split_llm` marks ready state correctly
- [ ] `unload_split_llm` kills the real serving process cleanly
- [ ] a failed split start leaves actionable diagnostics

Rollback:
- [ ] keep old split code available until this phase is proven
- [ ] if needed, run workers on llama while split remains on Ollama temporarily

## Phase 5: Support Scripts And Operator Surface

Primary files likely affected:
- [hardware.py](/home/bryan/llm_orchestration/shared/agents/hardware.py)
- [setup.py](/home/bryan/llm_orchestration/shared/agents/setup.py)
- [startup.py](/home/bryan/llm_orchestration/shared/agents/startup.py)
- [launch.py](/home/bryan/llm_orchestration/shared/agents/launch.py)
- [kill_plan.py](/home/bryan/llm_orchestration/scripts/kill_plan.py)
- [batch_runtime_diag.py](/media/bryan/shared/scripts/batch_runtime_diag.py)
- dashboard runtime recovery paths

Checklist:
- [ ] replace health scans using `/api/tags`
- [ ] replace loaded-model scans using `/api/ps`
- [ ] replace kill/unload logic that depends on Ollama semantics
- [ ] keep `llm-orchestrator.service` as the single default startup owner
- [ ] ensure `return_default.py` and any future reset path restart the service instead of launching detached runtimes
- [ ] update setup guidance
- [ ] update runtime diagnostics script
- [ ] update startup readiness checks
- [ ] update launch/cleanup behavior
- [ ] update dashboard reset expectations if needed

Acceptance tests:
- [ ] startup succeeds from clean rig state
- [ ] reboot comes back under the intended backend with no competing launcher
- [ ] kill/cleanup scripts still work
- [ ] hardware/setup checks reflect actual llama runtime state
- [ ] batch runtime diagnostics report the new runtime correctly

Rollback:
- [ ] temporary operator docs can note mixed/hybrid state if needed

## Phase 6: Cleanup

Do this only after earlier phases are stable.

Checklist:
- [ ] remove dead Ollama references from docs
- [ ] archive or remove `gpu_ollama.py` only after full cutover
- [ ] remove benchmark proxy that only exists for Ollama limitations
- [ ] remove old control scripts that are now misleading
- [ ] remove Ollama package/service from rig if no longer needed
- [ ] reclaim `/home/bryan/.ollama/` space after explicit confirmation

Acceptance criteria:
- [ ] no production path depends on Ollama
- [ ] no operator doc points users to Ollama commands
- [ ] no required script assumes Ollama endpoints

## Explicit File Checklist

### Core agent/runtime files

- [ ] [gpu_ollama.py](/home/bryan/llm_orchestration/shared/agents/gpu_ollama.py)
- [ ] [gpu.py](/home/bryan/llm_orchestration/shared/agents/gpu.py)
- [ ] [gpu_split.py](/home/bryan/llm_orchestration/shared/agents/gpu_split.py)
- [ ] [gpu_tasks.py](/home/bryan/llm_orchestration/shared/agents/gpu_tasks.py)
- [ ] [gpu_workers.py](/home/bryan/llm_orchestration/shared/agents/gpu_workers.py)
- [ ] [gpu_core.py](/home/bryan/llm_orchestration/shared/agents/gpu_core.py)
- [ ] [worker.py](/home/bryan/llm_orchestration/shared/agents/worker.py)
- [ ] [brain.py](/home/bryan/llm_orchestration/shared/agents/brain.py)
- [ ] [brain_core.py](/home/bryan/llm_orchestration/shared/agents/brain_core.py)
- [ ] [brain_dispatch.py](/home/bryan/llm_orchestration/shared/agents/brain_dispatch.py)
- [ ] [brain_resources.py](/home/bryan/llm_orchestration/shared/agents/brain_resources.py)

### Support files

- [ ] [hardware.py](/home/bryan/llm_orchestration/shared/agents/hardware.py)
- [ ] [setup.py](/home/bryan/llm_orchestration/shared/agents/setup.py)
- [ ] [startup.py](/home/bryan/llm_orchestration/shared/agents/startup.py)
- [ ] [launch.py](/home/bryan/llm_orchestration/shared/agents/launch.py)
- [ ] [kill_plan.py](/home/bryan/llm_orchestration/scripts/kill_plan.py)
- [ ] [/media/bryan/shared/scripts/batch_runtime_diag.py](/media/bryan/shared/scripts/batch_runtime_diag.py)

### Tests/docs

- [ ] [test_split_runtime_hardening.py](/home/bryan/llm_orchestration/shared/agents/tests/test_split_runtime_hardening.py)
- [ ] tests covering hardware/runtime scanning
- [ ] operator docs
- [ ] dashboard notes if behavior changes

## Decision Points

### Runtime implementation

- host-installed `llama-server`
- containerized `llama-server`

Recommendation:
- use containerized `llama-server` for production agents
- keep host-installed `llama-server` as a future optimization, not the initial cutover path
- do not use `python3 -m llama_cpp.server` as the long-term production runtime form

Locked decision:
- build a dedicated lightweight production runtime image around `llama-server`
- do not reuse `bench-knowledge:latest` as the long-term serving image

### Migration mode

- all-at-once
- hybrid

Recommendation:
- hybrid during transition
- workers first
- brain second
- split last

### Orchestrator placement

- host service
- full orchestrator container

Recommendation:
- keep the orchestrator on the host under `llm-orchestrator.service`
- do not containerize the control plane in this migration

### Benchmark placement

- same runtime environment as serving
- separate benchmark containers

Recommendation:
- keep benchmarks in separate containers
- do not merge benchmark execution into serving/runtime containers

### Image distribution

- shared-drive image export/import
- local registry

Recommendation:
- do not stand up a registry yet
- use local image builds or explicit `docker save` / `docker load` while image count stays small
- revisit registry only after the runtime/benchmark image set stabilizes

### Plan/task execution environment

- per-plan venvs
- selective task containers
- universal task containers

Recommendation:
- keep current plan behavior for now
- allow selective future containerization for heavyweight or conflicting plan/task environments
- do not make universal plan containers part of the llama migration

## Canonical Startup Command Families

These are templates, not final commands. Exact image, binary path, and flags must be recorded during implementation.

Single worker runtime:

```bash
docker run --rm --name llama-worker-gpu2 \
  --gpus "device=2" \
  --network host \
  -v /mnt/shared/models:/mnt/shared/models \
  <llama-image> \
  llama-server \
  --model <gguf-path> \
  --host 127.0.0.1 \
  --port 11436
```

Brain runtime:

```bash
docker run --rm --name llama-brain \
  --gpus <brain-gpu-selection> \
  --network host \
  -v /mnt/shared/models:/mnt/shared/models \
  <llama-image> \
  llama-server \
  --model <gguf-path> \
  --host 127.0.0.1 \
  --port 11434
```

Split runtime:

```bash
docker run --rm --name llama-split-pair-1-3 \
  --gpus all \
  --network host \
  -v /mnt/shared/models:/mnt/shared/models \
  <llama-image> \
  llama-server \
  --model <gguf-path> \
  --host 127.0.0.1 \
  --port 11440 \
  --tensor-split 0 1 0 1 0 0
```

To lock down during execution:

- container image name
- exact `llama-server` binary path inside image
- exact GPU selection flags for brain and split paths
- standard flags for context, batching, threads, GPU layers, and mlock
- canonical stop command
- canonical readiness probe

Locked implementation intent:

- create a dedicated lightweight runtime image for serving
- keep benchmark images separate from serving images
- defer local registry work until later

## Open Questions Still Remaining Before/Alongside Phase 1

These are the only meaningful open questions left. They should be answered during early execution, but they do not all block Phase 1 coding.

1. Dedicated runtime image spec
- final image name
- Dockerfile location
- base image choice
- exact binary path for `llama-server`
- CUDA arch/build settings for the rig (`sm_61` + `sm_86`)

2. Canonical runtime flags
- worker default flags
- brain default flags
- split default flags
- readiness probe endpoint and timeout

3. Remaining proof work running in parallel
- finish 14B split readiness proof
- finish 32B brain readiness proof

4. Production hardening gate
- `mlock` / memlock settings are deferred until production testing
- this is not a Phase 1 coding blocker
- this is a production-readiness gate before full cutover

## Phase Execution Clarification

Locked decision:

- Phase 1 single-worker runtime work may start now
- completion of 14B split proof and 32B brain proof continues in parallel
- full split cutover still waits for split proof
- full brain cutover still waits for brain proof

### Port strategy

- keep 11434-11439 and split ports
- renumber everything

Recommendation:
- keep ports stable unless `llama-server` forces a specific change

## Observability Requirements

These are mandatory during migration:

- [ ] persist exact startup command for each runtime
- [ ] persist PID for each runtime
- [ ] persist startup duration
- [ ] persist health probe failure detail
- [ ] persist stderr tail for startup failure
- [ ] persist split launch diagnostics
- [ ] ensure batch summaries show actual runtime cause, not blank failure entries

## Done Definition

The migration is done when all of the following are true:

- [ ] brain runtime uses `llama-server`
- [ ] single-worker runtime uses `llama-server`
- [ ] split runtime uses one explicit `llama-server` process
- [ ] startup, kill, scan, and diagnostic scripts no longer depend on Ollama APIs
- [ ] operator docs reflect the new runtime truth
- [ ] rollback to Ollama is no longer needed

## Immediate Next Step

Start with Phase 0 and fill in the exact proven commands on the rig. Do not start rewriting the split runtime until the manual single-GPU and split-GPU `llama-server` commands are both known-good.

## Branching Note

If Phase 1 execution continues on a branch or separate worktree, do not leave `main` in a half-migrated dirty state.

Required follow-up:

- either land work in small clean commits back to `main`
- or do a proper cleanup/reconciliation pass before switching active development back to `main`

Do not let the migration proceed with long-lived ambiguous divergence between the branch/worktree and the main operator path.
