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

## What The Codebase Looks Like Now

As of 2026-03-08 after the runtime vocabulary cleanup:

- [gpu_ollama.py](/home/bryan/llm_orchestration/shared/agents/gpu_ollama.py) still exists, but the class inside was renamed to `GPURuntimeMixin` and is now the generic single-runtime mixin entrypoint rather than an Ollama-only implementation surface.
- [gpu_llama.py](/home/bryan/llm_orchestration/shared/agents/gpu_llama.py) is the active llama runtime implementation for containerized `llama-server`.
- [gpu.py](/home/bryan/llm_orchestration/shared/agents/gpu.py), [gpu_tasks.py](/home/bryan/llm_orchestration/shared/agents/gpu_tasks.py), [gpu_workers.py](/home/bryan/llm_orchestration/shared/agents/gpu_workers.py), and [worker.py](/home/bryan/llm_orchestration/shared/agents/worker.py) now use llama-first runtime naming and llama-compatible API assumptions.
- [brain_core.py](/home/bryan/llm_orchestration/shared/agents/brain_core.py), [brain.py](/home/bryan/llm_orchestration/shared/agents/brain.py), and [brain_dispatch.py](/home/bryan/llm_orchestration/shared/agents/brain_dispatch.py) now treat llama as the sole active backend in normal control flow.
- [gpu_split.py](/home/bryan/llm_orchestration/shared/agents/gpu_split.py) remains the highest-complexity runtime file, but its runtime naming and launch path were cleaned to the llama model.
- [hardware.py](/home/bryan/llm_orchestration/shared/agents/hardware.py), [setup.py](/home/bryan/llm_orchestration/shared/agents/setup.py), [startup.py](/home/bryan/llm_orchestration/shared/agents/startup.py), and [launch.py](/home/bryan/llm_orchestration/shared/agents/launch.py) were also cleaned to use runtime terminology as the primary operator surface.

What is still intentionally present:

- some backwards-compat config fallbacks like `config.get("ollama_host")`
- some legacy cleanup commands that explicitly reap old `ollama serve` / `ollama runner` processes during transition
- compatibility wrappers such as `scan_ollama()` in [hardware.py](/home/bryan/llm_orchestration/shared/agents/hardware.py)

That means the active path is now llama-first, but the final cleanup phase still needs to remove the last transition-only compatibility shims.

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
- [smoke_test.sh](/home/bryan/llm_orchestration/scripts/llama_runtime/smoke_test.sh)
- [build_and_smoke_test.sh](/home/bryan/llm_orchestration/scripts/llama_runtime/build_and_smoke_test.sh)

## Verification Pass (2026-03-08)

The earlier cleanup-gap report in this document is now stale.

Current repo verification from this pass:

- targeted grep against active agent/runtime code under [shared/agents](/home/bryan/llm_orchestration/shared/agents) and [scripts](/home/bryan/llm_orchestration/scripts) found no active `ollama`, `/api/ps`, `/api/tags`, `scan_ollama`, or `ollama serve` references in the current control-plane files
- spot checks of [gpu_split.py](/home/bryan/llm_orchestration/shared/agents/gpu_split.py), [brain_core.py](/home/bryan/llm_orchestration/shared/agents/brain_core.py), [startup.py](/home/bryan/llm_orchestration/shared/agents/startup.py), [brain_resources.py](/home/bryan/llm_orchestration/shared/agents/brain_resources.py), and [hardware.py](/home/bryan/llm_orchestration/shared/agents/hardware.py) show llama-only runtime probes and llama-only backend enforcement on the active path
- the older report of active Ollama branches should be treated as historical context, not current repo state

What is still not proven by this verification pass:

- clean-rig operator acceptance from a full reboot
- end-to-end split inference on the live rig during a real orchestrated batch
- shoulder-plan runtime behavior outside the unit/regression surface

## Llama-Only Conversion Progress (2026-03-08)

Completed in the active repo:

- [hardware.py](/home/bryan/llm_orchestration/shared/agents/hardware.py) now rejects non-llama backends and no longer exposes `scan_ollama()`.
- [startup.py](/home/bryan/llm_orchestration/shared/agents/startup.py) preload detection is now llama-only and no longer probes `/api/ps`.
- [brain_core.py](/home/bryan/llm_orchestration/shared/agents/brain_core.py), [brain_dispatch.py](/home/bryan/llm_orchestration/shared/agents/brain_dispatch.py), and [worker.py](/home/bryan/llm_orchestration/shared/agents/worker.py) now require llama-only runtime routing instead of branching to legacy Ollama endpoints.
- [setup.py](/home/bryan/llm_orchestration/shared/agents/setup.py), [launch.py](/home/bryan/llm_orchestration/shared/agents/launch.py), and the config JSON files no longer write or read `ollama_host` in the normal agent path.
- [gpu_split.py](/home/bryan/llm_orchestration/shared/agents/gpu_split.py) and [brain_resources.py](/home/bryan/llm_orchestration/shared/agents/brain_resources.py) now use llama runtime probes and warmup only.
- The agent tests were updated away from legacy `ollama_host`, `runtime_ollama_url`, and `GPUOllamaMixin` naming where those references were just migration leftovers.

Still to finish after this pass:

- sweep the remaining Ollama-specific helper scripts under `scripts/`
- remove or rename any leftover Ollama-specific filenames that still imply dual-backend support
- re-run targeted tests and fix whatever breaks under the stricter llama-only contract

Follow-up after the second cleanup pass:

- the active operator scripts were moved further toward llama-only:
  - [clear_runtime.py](/home/bryan/llm_orchestration/scripts/clear_runtime.py)
  - [dashboard/server.py](/home/bryan/llm_orchestration/scripts/dashboard/server.py)
  - [scan_workers.py](/home/bryan/llm_orchestration/scripts/scan_workers.py)
  - [kill_plan.py](/home/bryan/llm_orchestration/scripts/kill_plan.py)
  - [batch_runtime_diag.py](/home/bryan/llm_orchestration/scripts/batch_runtime_diag.py)
  - [batch_runtime_diag.py](/home/bryan/llm_orchestration/shared/scripts/batch_runtime_diag.py)
  - [brain_watchdog.py](/home/bryan/llm_orchestration/shared/scripts/brain_watchdog.py)
  - [agent-monitor.py](/home/bryan/llm_orchestration/scripts/agent-monitor.py)
  - [chat-brain](/home/bryan/llm_orchestration/scripts/chat-brain)
  - [chat-worker](/home/bryan/llm_orchestration/scripts/chat-worker)
  - [chat_runtime.py](/home/bryan/llm_orchestration/scripts/chat_runtime.py)

Remaining non-agent Ollama references are now mostly in benchmark/history tooling and older model-management scripts, not in the main agent control plane.

## Non-Benchmark Sweep Progress (2026-03-08)

Completed in this pass:

- [quickstart.md](/home/bryan/llm_orchestration/shared/workspace/quickstart.md), [architecture.md](/home/bryan/llm_orchestration/shared/workspace/architecture.md), [PLAN_FORMAT.md](/home/bryan/llm_orchestration/shared/workspace/PLAN_FORMAT.md), and [README.md](/home/bryan/llm_orchestration/README.md) now present llama as the only active runtime path in normal operator docs.
- [scripts/llama_runtime/README.md](/home/bryan/llm_orchestration/scripts/llama_runtime/README.md) now describes the directory as the active runtime surface instead of a migration-only bridge.
- [hardware.py](/home/bryan/llm_orchestration/shared/agents/hardware.py), [NETWORK_SETUP.md](/home/bryan/llm_orchestration/shared/workspace/NETWORK_SETUP.md), [systems_prep.md](/home/bryan/llm_orchestration/shared/workspace/systems_prep.md), and [brain-behavior.md](/home/bryan/llm_orchestration/shared/workspace/brain-behavior.md) no longer describe the old Ollama path as current operator behavior.
- Obsolete tracked helper [clear_ollama.py](/home/bryan/llm_orchestration/scripts/clear_ollama.py) was removed from the repo.

Phase status after this pass:

- Phase 1, core agent/control plane cleanup: effectively done
- Phase 2, operator/runtime tooling cleanup: mostly done
- Phase 3, repo-wide non-benchmark doc/support cleanup: in progress
- Phase 4, benchmark/history-specific cleanup: intentionally deferred for now
- Phase 5, broad validation/testing: deferred until the remaining non-benchmark runtime leftovers are removed

Still intentionally deferred in this pass:

- benchmark-adjacent runtime prep/model-management helpers such as [prepare_llm_runtimes.py](/home/bryan/llm_orchestration/shared/scripts/prepare_llm_runtimes.py) and [manage_model_hotset.py](/home/bryan/llm_orchestration/shared/scripts/manage_model_hotset.py)
- historical/archive material under `shared/workspace/archive/`, `shared/brain/escalations/`, and related old reports
- anything under `shared/plans/` or benchmark-specific trees

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

Current rig status as of 2026-03-07:

- `llm-orchestrator.service` is now the only default startup owner.
- The service startup wrapper points at `~/llm-orchestration-venv/bin/python`.
- Legacy `ollama.service` and `ollama-1060.service` autostarts were removed from the dependency graph and both units were masked to `/dev/null`.
- `/etc/systemd/system/llm-orchestrator.service` no longer has `After=` or `Wants=` edges on `ollama.service`.
- Dashboard fetch failure was traced to the dashboard process being down, not a backend API regression. Manual restart restored `GET /api/status` and the dashboard runtime surface now reads `runtime_api_base` / `runtime_healthy` instead of heartbeat `ollama_*` keys.

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

## Current State (2026-03-07 Late Evening)

- Dashboard API is healthy again at `http://127.0.0.1:8787/api/status`.
- GPU heartbeat payload is now runtime-neutral:
  - `runtime_api_base`
  - `runtime_healthy`
  - `runtime_health`
- Dashboard worker rendering now uses `runtime_api_base` instead of `runtime_ollama_url`.
- Default worker launcher now emits `--api-base ...` to worker subprocesses instead of `--ollama-url ...`.
- `github_analyzer` got past the old `brain_strategy` 404 blocker after patching the shoulder scripts to use backend-neutral API resolution.
- The `brain_strategy` JSON parser was hardened on 2026-03-08:
  - plain balanced-object extraction is now used before failing
  - same-model JSON repair fallback is now used if the brain wraps JSON in prose or markdown
  - manual re-run against failed batch `20260307_234042` completed and wrote `brain_strategy.json`
- `PLAN_FORMAT.md` now explicitly states the structured-output contract:
  - scripts own schema enforcement and JSON recovery
  - prompts should ask for one strict JSON object
  - consumers must tolerate wrapper prose / fenced markdown and recover the first balanced JSON object when possible
- On 2026-03-08 another control-plane stall was isolated:
  - the brain could wedge in uninterruptible disk wait (`D` state) while scanning the global `/mnt/shared/tasks/complete` archive
  - this showed up immediately after `prepare_repo` on batch `20260308_000056`, with heartbeat freezing even though the task itself had succeeded
  - root cause was not llama inference; it was control-plane file scanning against a large historical task archive on the shared drive
  - fix landed in the brain queue/control path:
    - dependency satisfaction now prefers per-batch `logs/batch_events.jsonl` instead of scanning every file in `/mnt/shared/tasks/complete`
    - stale-failure cleanup now uses active-batch event logs instead of a global completion scan
    - goal validation completion discovery now prefers per-batch event logs as well
  - operational cleanup was also applied:
    - historical completion artifacts older than 2026-03-07 were archived out of the hot directory to `/mnt/shared/tasks/archive/complete_pre_20260307`
    - this reduced the live `tasks/complete` hot set to recent artifacts only
- Current active validation rerun is `github_analyzer` batch `20260308_000056` against `code-visualizer`.

What remains:

- Finish cleaning the remaining Ollama compatibility env names and comments in the active runtime path.
- Continue watching `github_analyzer` for the next real blocker after `brain_strategy`.
- Keep the compatibility bridge only where older shoulders still require it; do not add new Ollama-first surfaces.

## 2026-03-08 Runtime Vocabulary Cleanup Consolidation

The protected cleanup pass captured in:

- [PROTECTED_LLAMA_CLEANUP_REPORT.md](/home/bryan/Desktop/Shared/workspace/archive/PROTECTED_LLAMA_CLEANUP_REPORT.md)

has now been folded into this plan.

Summary:

- All targeted protected agent files listed in that report were cleaned on 2026-03-08.
- The codebase now uses a unified runtime vocabulary with llama as the sole active backend in normal operation.
- The dashboard/control layer was also cleaned so the operator surface now says `Clear Runtime` and routes through `clear_runtime.py`.

Files confirmed cleaned in that pass:

- [brain_dispatch.py](/home/bryan/llm_orchestration/shared/agents/brain_dispatch.py)
- [gpu_workers.py](/home/bryan/llm_orchestration/shared/agents/gpu_workers.py)
- [gpu.py](/home/bryan/llm_orchestration/shared/agents/gpu.py)
- [gpu_tasks.py](/home/bryan/llm_orchestration/shared/agents/gpu_tasks.py)
- [gpu_split.py](/home/bryan/llm_orchestration/shared/agents/gpu_split.py)
- [gpu_llama.py](/home/bryan/llm_orchestration/shared/agents/gpu_llama.py)
- [startup.py](/home/bryan/llm_orchestration/shared/agents/startup.py)
- [brain_core.py](/home/bryan/llm_orchestration/shared/agents/brain_core.py)
- [hardware.py](/home/bryan/llm_orchestration/shared/agents/hardware.py)
- [setup.py](/home/bryan/llm_orchestration/shared/agents/setup.py)
- [worker.py](/home/bryan/llm_orchestration/shared/agents/worker.py)
- [launch.py](/home/bryan/llm_orchestration/shared/agents/launch.py)
- [gpu_core.py](/home/bryan/llm_orchestration/shared/agents/gpu_core.py)
- [brain.py](/home/bryan/llm_orchestration/shared/agents/brain.py)
- [brain_resources.py](/home/bryan/llm_orchestration/shared/agents/brain_resources.py)
- [brain_failures.py](/home/bryan/llm_orchestration/shared/agents/brain_failures.py)

Intentionally retained transition shims:

- `config.get("ollama_host")` compatibility reads where older config files may still exist
- explicit cleanup of legacy `ollama serve` / `ollama runner` processes during resets and return-to-default flows
- `scan_ollama()` wrapper in [hardware.py](/home/bryan/llm_orchestration/shared/agents/hardware.py)
- deprecated `--ollama-host` alias in [setup.py](/home/bryan/llm_orchestration/shared/agents/setup.py)
- retained filename [gpu_ollama.py](/home/bryan/llm_orchestration/shared/agents/gpu_ollama.py) even though the class inside is now runtime-generic

This is the current line between "migration complete enough to run llama-first" and "final cleanup complete enough to delete transition scaffolding."

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
- [x] Build or install `llama-server`
- [x] Run one single-GPU worker model manually
- [ ] Run brain model manually
- [ ] Run one split model manually with `--tensor-split`
- [x] Confirm health endpoint behavior
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
8. Startup for the 7B single-GPU worker case was slow but successful.
   - earlier direct-container proofs on the prior storage path were roughly on the order of about 60-70 seconds
   - a clean orchestrator-owned cold load on 2026-03-07 from the current USB-attached shared drive path took about 3m50s to reach `GET /v1/models == 200`
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
- Host-side production binary availability: resolved (dedicated `llama-runtime:sm61-sm86` image built from llama.cpp b4837)
- Split readiness on target 14B: partially proven only
- Brain-model readiness on 32B: partially proven only
- Service-owned orchestration baseline: proven

### Dedicated Image Build And Smoke Test

Completed: dedicated `llama-runtime:sm61-sm86` image built from llama.cpp `b4837` with CUDA architectures `61;86` and static linking (`-DBUILD_SHARED_LIBS=OFF`).

Build fixes applied:
- Removed COPY lines for `.so` files (not produced with static linking)
- Fixed chat-templates path: `/src/llama.cpp/models/templates` (not `/src/llama.cpp/common/chat-templates`)

Smoke test on GPU 1 (GTX 1060 6GB):
```bash
/mnt/shared/scripts/llama_runtime/run_runtime.sh \
  --name llama-smoke-test \
  --model /mnt/shared/models/qwen2.5-coder-7b/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf \
  --port 11500 \
  --gpus device=1
```

Results:
- Container started and reached readiness
- `/v1/chat/completions` smoke test passed
- `/v1/completions` smoke test passed
- Clean shutdown, GPU memory fully released

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
- [x] dedicated image built locally on the rig (`llama-runtime:sm61-sm86`)
- [x] dedicated image proven with one real single-worker model (smoke test passed on GPU 1, qwen2.5-coder-7b)
- [x] agent runtime mixin added (`gpu_llama.py`)
- [x] agent path switched from Ollama to llama (backend-aware dispatchers in gpu.py)

Primary files:
- [gpu_ollama.py](/home/bryan/llm_orchestration/shared/agents/gpu_ollama.py)
- [gpu.py](/home/bryan/llm_orchestration/shared/agents/gpu.py)
- [gpu_core.py](/home/bryan/llm_orchestration/shared/agents/gpu_core.py)

Plan:
- [x] create dedicated image and launcher helpers under `scripts/llama_runtime/`
- [x] add `gpu_llama.py`
- [x] implement server start
- [x] implement server stop
- [x] implement health check
- [x] implement readiness check
- [x] capture process PID and command line in runtime state
- [x] preserve strong runtime diagnostics on failure
- [x] remove dependency on Ollama `keep_alive` for worker lifecycle

Notes:
- Server start becomes the load operation.
- Server stop becomes the unload operation.
- Keep the external orchestration contract stable where possible.
- Keep service ownership unchanged. Phase 1 swaps the backend under the existing service/supervisor model; it does not introduce a parallel launcher.
- Use the new wrapper scripts as the canonical command source instead of embedding raw `docker run` strings in multiple places.

Exit criteria:
- [x] one worker can start a model-backed runtime on demand
- [x] one worker can unload it by killing the process
- [x] failure logging shows real command/process/health information

Rollback:
- [x] keep `gpu_ollama.py` intact until Phase 6
- [x] allow worker path to switch back to Ollama during transition (`runtime_backend` config key)

### Phase 1 Canonical Commands

Build the dedicated image:

```bash
/home/bryan/llm_orchestration/scripts/llama_runtime/build_image.sh
```

Build and prove one worker runtime in one command:

```bash
/home/bryan/llm_orchestration/scripts/llama_runtime/build_and_smoke_test.sh \
  --name llama-worker-gpu1 \
  --model /mnt/shared/models/qwen2.5-coder-7b/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf \
  --port 11436 \
  --gpus device=1
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
- [x] replace `/api/generate` assumptions (worker.py uses `/v1/chat/completions` for llama)
- [x] replace `/api/ps` readiness assumptions for single runtimes (attestation uses `/v1/models`)
- [x] rename worker URL variables from Ollama naming to llama naming where appropriate
- [x] keep task contract stable for `load_llm` and `unload_llm`
- [x] update heartbeats and runtime state fields as needed (`runtime_backend` in heartbeat)
- [x] verify existing observability fields still populate meaningful failure detail

Acceptance tests:
- [x] `load_llm` succeeds on a cold worker
- [x] `unload_llm` returns the worker to cold state (clean `ready_single->unloading->cold`)
- [x] one real LLM task runs end-to-end on a worker (gpu-1, task completed with result)
- [x] failed startup produces useful batch artifacts (runtime_error_code/detail populated)

Rollback:
- [x] revert worker mixin selection to Ollama path (set `runtime_backend: "ollama"` in config.json)

## Phase 3: Brain Runtime Migration

Primary files:
- [brain.py](/home/bryan/llm_orchestration/shared/agents/brain.py)
- [brain_core.py](/home/bryan/llm_orchestration/shared/agents/brain_core.py)
- [brain_dispatch.py](/home/bryan/llm_orchestration/shared/agents/brain_dispatch.py)

Checklist:
- [x] replace brain startup from Ollama to `llama-server`
- [x] replace brain health checks
- [x] replace brain inference endpoint assumptions
- [x] update env var names and config keys
- [x] verify brain startup and shutdown behavior

Current note:

- Brain startup is using `runtime_backend: "llama"` in [config.json](/home/bryan/llm_orchestration/shared/agents/config.json).
- [brain_core.py](/home/bryan/llm_orchestration/shared/agents/brain_core.py) now starts the brain via `/mnt/shared/scripts/llama_runtime/run_runtime.sh` and probes `GET /v1/models`.
- Backend-neutral aliases now exist for the brain path:
  - config prefers `runtime_host` with fallback to `ollama_host`
  - brain shell dispatch exports `BRAIN_API_BASE` / `WORKER_API_BASE`
  - worker execution accepts `WORKER_API_BASE` with fallback to `WORKER_OLLAMA_URL`
- On 2026-03-07, reboot verification showed the rig comes up under `llm-orchestrator.service` with both Ollama units still masked.
- The earlier brain loading issue was storage-path related, not a broken brain model:
  - old path: NFS-backed `/mnt/shared` from the Pi, too slow for clean 32B startup
  - current path: local USB-attached shared drive mounted on the rig at `/mnt/shared`
  - active brain model promoted to local hot-set at `/home/bryan/model-hotset/qwen2.5-coder-32b/`
- Current live proof on 2026-03-07:
  - `llama-brain` is running
  - `GET /v1/models` returns `200`
  - `brain.ready` is present
  - `POST /v1/chat/completions` succeeded with `200`
- Default runtime ownership has been hardened further on 2026-03-07:
  - stale `model_load.global.json` owners are now reclaimed automatically by the worker load path
  - `load_llm` / `load_split_llm` now auto-reset wedged or dirty local runtime state before retrying
  - startup port reclaim now removes the canonical `llama-worker-gpu-*` container before falling back to `lsof` / `fuser`
  - single-worker llama load budget increased from `180s` to `300s` for the default 7B worker path on GTX 1060s
- Observability / cleanup findings from the live rig on 2026-03-07:
  - legacy dashboard fields could still make a real `loading_single` worker look falsely `cold`
  - stale manual llama runtimes can contaminate VRAM readings and make it look like the default worker is loaded when it is not
  - stale split reservation files can keep producing `SPLIT_READY_REJECTED ... has_model=False has_owner=True` noise after the real split runtime is gone
  - cleanup helper added for returning the rig to service-owned default state:
    - [`cleanup_llama_runtime_state.py`](/media/bryan/shared/scripts/cleanup_llama_runtime_state.py)
- Current default config expectation remains:
  - brain: `qwen2.5:32b` on GPU `0`
  - default hot worker target: `qwen2.5:7b` on `gpu-2`
- Current acceptance target for final Phase 3 proof:
  - run [`github_analyzer`](/home/bryan/Desktop/Shared/plans/shoulders/github_analyzer/plan.md)
  - against [`code-visualizer`](/home/bryan/Desktop/Shared/plans/shoulders/code-visualizer)
  - confirm starter-task consumption plus first worker task release under the default runtime state
- Correct plan submission entrypoints:
  - local operator wrapper: [`/home/bryan/llm_orchestration/scripts/submit.py`](/home/bryan/llm_orchestration/scripts/submit.py)
  - rig-side implementation invoked by the wrapper: `/mnt/shared/agents/submit.py`
  - the local wrapper does not queue locally by default; it SSHes to host alias `gpu` and runs the rig-side submit helper there
  - config file paths passed to submit should be rig-visible paths (`/mnt/shared/...`) for files the rig must read directly
  - shared aliases normalized by submit include:
    - `/mnt/shared`
    - `/home/bryan/llm_orchestration/shared`
    - `/media/bryan/shared`
- Clean default-worker proof on 2026-03-07 after stale-runtime cleanup:
  - `llama-worker-gpu-2` reached `/v1/models == 200`
  - heartbeat promoted to:
    - `state=hot`
    - `runtime_state=ready_single`
    - `runtime_transition_phase=load_complete`
    - `model_loaded=True`
    - `loaded_model=qwen2.5:7b`
  - real cold-load time from `/mnt/shared` on the rig was about `3m50s`

Acceptance tests:
- [x] brain starts with the intended model
- [x] brain can process one plan from startup to initial task release
- [x] restart behavior works cleanly

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
- [x] remove dual-Ollama assumptions from the active split code path
- [x] remove Ollama-specific warmup/load calls from the active split code path
- [ ] keep reservation and ownership logic only where still necessary
- [x] simplify naming and cleanup mechanics around one runtime process
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
- [x] replace health/runtime operator surfaces so llama/runtime terminology is primary
- [x] replace loaded-model scans in the active operator surface where the llama path is now primary
- [ ] replace the remaining kill/unload logic that still exists only to reap legacy Ollama transition artifacts
- [x] keep `llm-orchestrator.service` as the single default startup owner
- [x] ensure `return_default.py` restarts the service instead of launching detached runtimes
- [x] update setup guidance to runtime-first / llama-first terminology
- [ ] update runtime diagnostics script
- [x] update startup readiness checks to runtime-first / llama-first terminology
- [x] update launch/cleanup behavior to runtime-first / llama-first terminology
- [x] update dashboard reset expectations and control-page wording

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

- [x] [gpu_ollama.py](/home/bryan/llm_orchestration/shared/agents/gpu_ollama.py)
- [x] [gpu.py](/home/bryan/llm_orchestration/shared/agents/gpu.py)
- [x] [gpu_split.py](/home/bryan/llm_orchestration/shared/agents/gpu_split.py)
- [x] [gpu_tasks.py](/home/bryan/llm_orchestration/shared/agents/gpu_tasks.py)
- [x] [gpu_workers.py](/home/bryan/llm_orchestration/shared/agents/gpu_workers.py)
- [x] [gpu_core.py](/home/bryan/llm_orchestration/shared/agents/gpu_core.py)
- [x] [worker.py](/home/bryan/llm_orchestration/shared/agents/worker.py)
- [x] [brain.py](/home/bryan/llm_orchestration/shared/agents/brain.py)
- [x] [brain_core.py](/home/bryan/llm_orchestration/shared/agents/brain_core.py)
- [x] [brain_dispatch.py](/home/bryan/llm_orchestration/shared/agents/brain_dispatch.py)
- [x] [brain_resources.py](/home/bryan/llm_orchestration/shared/agents/brain_resources.py)
- [x] [brain_failures.py](/home/bryan/llm_orchestration/shared/agents/brain_failures.py)

### Support files

- [x] [hardware.py](/home/bryan/llm_orchestration/shared/agents/hardware.py)
- [x] [setup.py](/home/bryan/llm_orchestration/shared/agents/setup.py)
- [x] [startup.py](/home/bryan/llm_orchestration/shared/agents/startup.py)
- [x] [launch.py](/home/bryan/llm_orchestration/shared/agents/launch.py)
- [ ] [kill_plan.py](/home/bryan/llm_orchestration/scripts/kill_plan.py)
- [ ] [/media/bryan/shared/scripts/batch_runtime_diag.py](/media/bryan/shared/scripts/batch_runtime_diag.py)
- [x] dashboard control-page/runtime reset surface

### Tests/docs

- [x] [test_split_runtime_hardening.py](/home/bryan/llm_orchestration/shared/agents/tests/test_split_runtime_hardening.py)
- [x] tests covering hardware/runtime scanning
- [ ] operator docs
- [x] dashboard notes if behavior changes

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

## 2026-03-07 Runtime Validation Update

Current live conclusions after the direct-attached shared-drive move and llama runtime hardening:

- Worker and split paths were previously "serving but not really offloaded". Logs showed `offloaded 0/... layers to GPU` and large `CPU_Mapped model buffer` allocations.
- The runtime launcher now passes explicit llama device selection based on visible GPUs, and worker/split defaults now force real GPU offload rather than the old `n_gpu_layers=-1` behavior.
- Default worker policy remains `qwen2.5:7b` on `gpu-2`.
- Default split review model remains `qwen2.5-coder:14b`.

Validated full-offload profiles:

- `qwen2.5:7b` on a single GTX 1060 6 GB:
  - use `ctx_size=2048`
  - use `batch_size=64`
  - use `n_gpu_layers=999`
  - this produces about `5.1 GiB` VRAM use and real full offload
- `qwen2.5-coder:14b` split on two GTX 1060 6 GB cards:
  - use `ctx_size=4096`
  - use `batch_size=128`
  - use `n_gpu_layers=999`
  - use `tensor_split=1,1`
  - validated with `49/49` layers offloaded and about `5.3 GiB` / `4.9 GiB` VRAM
- `qwen2.5-coder:32b` brain on the RTX 3090 Ti:
  - `brain_context_tokens=32768` is not viable for full GPU residency
  - `brain_context_tokens=8192` is viable
  - validated with `65/65` layers offloaded, about `18.5 GiB` model buffer on CUDA and `2.0 GiB` KV on CUDA

Code/config updates made:

- `run_runtime.sh` now maps Docker-visible GPUs to explicit llama `--device CUDA...` arguments
- single-worker defaults updated to the validated 7B full-offload profile
- default 7B worker profile now also uses `--no-warmup` because the GTX 1060 path was spending too long inside llama's empty warmup while already holding VRAM
- split defaults/profile updated to the validated 14B full-offload profile
- brain context default reduced to `8192` so the 32B brain can fully reside on the 3090
- brain startup now reuses a ready preloaded llama runtime instead of restarting it during `--model-preloaded` startup
- startup clears stale brain heartbeat files before a fresh launch
- single-worker load now follows `check healthy matching runtime -> reuse -> otherwise reset/reload`
- split member backfill now tolerates already-active split runtimes instead of self-resetting healthy members during `loading` / `ready_stabilizing`
- llama health checks now probe the split runtime port when a worker is in `split_gpu` placement

Focused test status:

- 2026-03-08 focused llama migration suite:
  - `test_runtime_defaults.py`: passing
  - `test_split_runtime_hardening.py`: passing
  - `test_phase3_brain_runtime.py`: passing
  - `test_startup_idempotence.py`: passing
  - `test_brain_llm_demand_window.py`: passing
  - `test_worker_runtime_readiness.py`: passing
  - `test_brain_failure_incidents.py`: passing
  - `shared/plans/shoulders/github_analyzer/tests/test_warm_workers.py`: passing
- 2026-03-08 broader orchestrator unit surface:
  - `python -m unittest discover -s shared/agents/tests -p 'test_*.py'`
  - result: `151` tests passed

Important remaining blocker:

- The brain readiness/ready-flag handoff bug is fixed.
- Fresh restart now publishes `brain.ready` and worker ready flags correctly.
- The next live issue was worker load timing on the real orchestrator path.
- That is now better understood:
  - `gpu-2` was not falsely loaded; it was genuinely still in `load_tensors` during the long `503 Loading model` window
  - after removing a stale manual GPU-2 benchmark runtime and clearing stale split reservation state, the default `gpu-2` worker reached `ready_single` cleanly
  - the real current cold-load expectation for `qwen2.5:7b` from `/mnt/shared` is about `3m50s`, not the older shorter estimate
- Another control-plane issue was also fixed:
  - brain LLM demand accounting was counting orphaned private tasks from inactive old batches
  - that could keep split demand timers artificially alive and cause unnecessary split-hot decisions
  - demand accounting now counts private tasks only for currently active batches
  - regression added:
    - [`test_brain_llm_demand_window.py`](/media/bryan/shared/agents/tests/test_brain_llm_demand_window.py)
- Split false-failure / heartbeat note:
  - the split runtime itself could come up healthy, fully offloaded, and listening on the expected port while the reservation still flipped to `failed`
  - when that happened, member heartbeats mirrored the failed reservation and dropped back to `cold`, which is why the dashboard showed low-trust `cold` / missing `host` even with real VRAM residency
  - the fix is in the split launcher path now:
    - if the expected split runtime is already healthy on the reserved port, reuse it instead of treating `port already in use before launch` as terminal
    - llama offload verification now reads a deeper log tail so it can still see the real `offloaded 49/49 layers to GPU` lines for long startup logs
  - focused regression coverage for split reuse remains in:
    - [`test_split_runtime_hardening.py`](/media/bryan/shared/agents/tests/test_split_runtime_hardening.py)
- Dashboard note:
  - heartbeat updates are healthy
  - the dashboard had been rendering legacy `state` / `host` fields, which made a real `loading_single` runtime look falsely `cold`
  - dashboard rendering is now updated to prefer `runtime_state` and show the configured model while a worker is loading
- Worker claim/readiness note:
  - the next real orchestrated bug was not task release timing; it was worker self-awareness
  - ordinary `llm` tasks were being claimed as soon as a worker looked model-capable, even when the runtime was still in `loading_single`
  - that produced `model_unavailable` failures on the first `worker_review_slice_*` wave, followed by worker resets and sudden VRAM drops
  - fix:
    - ordinary `llm` work now requires `runtime_state` in `RUNTIME_STATES_READY`, not just `model_loaded=True`
    - the same guard is enforced again at worker spawn time for defense in depth
  - live validation after restart:
    - fresh `github_analyzer` batch `20260308_002527` no longer lets cold workers opportunistically grab 7B work
    - `gpu-2` stayed the only ready single-GPU worker and completed `worker_doc_claims_shard_0` and `worker_doc_claims_shard_1`
    - remaining 7B shards stayed queued instead of failing immediately with `model_unavailable`
  - regression added:
    - [`test_worker_runtime_readiness.py`](/media/bryan/shared/agents/tests/test_worker_runtime_readiness.py)
- Split orphan-reclaim note:
  - the next real split blocker was not model fit; it was stale split listeners surviving without owner metadata
  - when `pair_1_3` relaunched, the reservation failed fast with `split port 11440 already in use before launch`
  - the deterministic container name was still present, but shared owner metadata was gone, so owner-based cleanup could not remove it
  - fix:
    - split startup now reclaims the deterministic split container name first, then rechecks the reserved port before failing
    - the rig-side runtime launcher no longer rewrites multi-GPU launches to `--gpus all`; it keeps the explicit `device=...` request and still passes llama-visible ordinals for the remapped container view
  - regression remains in:
    - [`test_split_runtime_hardening.py`](/media/bryan/shared/agents/tests/test_split_runtime_hardening.py)
- Batch `20260308_003554` lesson:
  - this run proved the earlier worker-claim fix held: `worker_doc_claims_shard_*` completed cleanly on `gpu-2`, and the batch reached `brain_strategy`, `runtime_validation`, `build_scope_manifest`, and the review-wave release point
  - the actual fatal event was later and different:
    - a generic `load_llm` resource task on `gpu-5` timed out after the full llama readiness budget
    - the brain treated that background warmup failure like a plan-verification failure and aborted the entire batch
  - fix:
    - generic resource meta tasks `load_llm` / `unload_llm` are now non-fatal when they exhaust retries
    - they are marked abandoned locally instead of cloud-escalating and killing the whole batch
    - this keeps the batch moving on already-hot workers instead of treating optional capacity expansion as a terminal plan failure
  - regression added:
    - [`test_brain_failure_incidents.py`](/media/bryan/shared/agents/tests/test_brain_failure_incidents.py)

Phase read after this update:

- Phase 3 runtime/storage migration: complete and re-verified in repo plus focused tests
- Phase 4 split llama runtime rewrite: live acceptance is now proven both in isolated load testing and in a completed fresh `github_analyzer` batch under the restarted service
- Phase 5 support scripts/operator surface: startup, reboot, submit preflight, dashboard/runtime diagnostics, and fresh full-batch acceptance are now proven on the rig

## Live Rig Acceptance Update (2026-03-08 Late)

This pass moved from full-batch debugging to isolated split-load acceptance first.

What was proven live on the rig:

- local SSD split load for `qwen2.5-coder:14b` succeeded in about `70s` to ready
- shared-drive split load for the same model succeeded in about `109s` to ready from a cold post-reboot state
- the earlier multi-minute split failures were primarily timeout/config/coordination problems, not an unavoidable shared-drive bottleneck
- clean reboot startup worked:
  - brain cold start reached ready in about `73s`
  - startup warm load on the default worker completed in about `53s`

Operational fix landed on the rig:

- `/etc/systemd/system/llm-orchestrator.service` now uses `Restart=on-failure` instead of `Restart=always`
- reason: `startup.py` intentionally exits `0` when it detects an already-healthy orchestrator, and `Restart=always` turned that benign exit into a duplicate relaunch loop
- post-change verification showed `NRestarts=0` and no repeated `skipping duplicate startup` churn after the service restart

Model path state after this pass:

- [models.catalog.json](/home/bryan/llm_orchestration/shared/agents/models.catalog.json) remains pinned to the shared-drive 14B GGUF for live split testing
- the SSD copy was kept on local disk but moved out of the hotset search roots to avoid accidental auto-resolution during shared-path validation:
  - `/home/bryan/model-archive-disabled/qwen2.5-coder-14b/Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf`

What changed after this pass:

- completed fresh `github_analyzer` batch `20260308_194140` against `code-visualizer`
  - batch status: complete
  - split pairs `1/3` and `4/5` both served real 14B review work during the run
  - final report artifacts were produced under `/media/bryan/shared/plans/shoulders/github_analyzer/history/20260308_194140/output/`
- dashboard holding-state cleanup landed so terminal runtime transitions such as `load_complete` and `split_cleared:unload_complete` no longer linger in the Holding column
- the active bottleneck has shifted from runtime ownership/split startup correctness to workload-shape tuning in `github_analyzer`
- next benchmark phase should focus on scaling behavior:
  - docs-task sizing by doc corpus
  - manifest sizing by repo complexity
  - verify promotion by extractor density / contradiction risk
  - conditional backfill/gap instead of fixed extra work
- legacy tuning plans under `shared/plans/shoulders/github_analyzer/` are now marked `ARCHIVE`, and the replacement direction is documented in:
  - [github_analyzer_scaling_tuning_plan_20260308.md](/home/bryan/llm_orchestration/shared/plans/shoulders/github_analyzer/notes/future/github_analyzer_scaling_tuning_plan_20260308.md)

What is still open after this pass:

- confirm the batch-level split preflight/review wave behaves as cleanly as the isolated `load_split_llm` tasks
- decide whether any stale split-reservation cleanup should be folded into startup normalization rather than left to targeted recovery paths
