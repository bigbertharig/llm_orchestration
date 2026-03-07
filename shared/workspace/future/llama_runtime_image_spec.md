# llama-server Runtime Image Spec

## Purpose

Define the dedicated production-serving image for Phase 1 worker runtime migration.

This image is for serving only.

It is not:

- the orchestration control plane image
- the benchmark image
- a general-purpose dev image

## Scope

Use this image for:

- single-worker `llama-server` runtimes
- brain `llama-server` runtime later
- split `llama-server` runtime later

Do not use this image for:

- benchmark harness execution
- plan-specific Python environments
- Docker-in-Docker control flows

## Requirements

The image must provide:

- native `llama-server` binary
- CUDA-enabled build
- support for rig GPU architectures:
  - `sm_61`
  - `sm_86`
- predictable container entrypoint behavior
- small enough footprint to rebuild and load quickly on the rig

The image should not include:

- benchmark harness dependencies
- notebook tooling
- large Python stacks unrelated to serving

## Initial Build Direction

Recommendation:

- build a dedicated lightweight runtime image from llama.cpp
- compile `llama-server` in the image
- keep the binary path explicit and stable

Current implementation artifacts:

- Dockerfile: [scripts/llama_runtime/Dockerfile](/home/bryan/llm_orchestration/scripts/llama_runtime/Dockerfile)
- entrypoint: [scripts/llama_runtime/entrypoint.sh](/home/bryan/llm_orchestration/scripts/llama_runtime/entrypoint.sh)
- build helper: [scripts/llama_runtime/build_image.sh](/home/bryan/llm_orchestration/scripts/llama_runtime/build_image.sh)
- run helper: [scripts/llama_runtime/run_runtime.sh](/home/bryan/llm_orchestration/scripts/llama_runtime/run_runtime.sh)
- stop helper: [scripts/llama_runtime/stop_runtime.sh](/home/bryan/llm_orchestration/scripts/llama_runtime/stop_runtime.sh)
- probe helper: [scripts/llama_runtime/probe_runtime.sh](/home/bryan/llm_orchestration/scripts/llama_runtime/probe_runtime.sh)

Locked choices for the first implementation pass:

- image name: `llama-runtime:sm61-sm86`
- base CUDA images:
  - build: `nvidia/cuda:12.6.3-devel-ubuntu24.04`
  - runtime: `nvidia/cuda:12.6.3-runtime-ubuntu24.04`
- binary path: `/opt/llama/bin/llama-server`
- compile target: native `llama-server`
- CUDA architectures: `61;86`

Still open while proving the image:

- exact `llama.cpp` ref to standardize on long-term
- whether any extra runtime libraries are needed on the rig beyond the current image
- final default values for `--ctx-size`, `--batch-size`, and `--parallel`

## Expected Runtime Contract

One container equals one loaded runtime.

That means:

- start container = start runtime
- container PID/process = runtime owner
- stop container = unload runtime
- readiness = container alive plus HTTP readiness probe

## Canonical Command Families

Single worker:

```bash
/home/bryan/llm_orchestration/scripts/llama_runtime/run_runtime.sh \
  --name llama-worker-gpu2 \
  --model <gguf-path> \
  --port 11436 \
  --gpus device=2
```

Brain:

```bash
/home/bryan/llm_orchestration/scripts/llama_runtime/run_runtime.sh \
  --name llama-brain \
  --model <gguf-path> \
  --port 11434 \
  --gpus <brain-gpu-selection>
```

Split:

```bash
/home/bryan/llm_orchestration/scripts/llama_runtime/run_runtime.sh \
  --name llama-split-pair-1-3 \
  --model <gguf-path> \
  --port 11440 \
  --gpus all \
  --tensor-split 0,1,0,1,0,0
```

## Runtime Flags To Standardize

Need one standard set per runtime class:

- `--ctx-size`
- `--n-gpu-layers`
- `--batch-size`
- `--threads`
- `--parallel`
- `--mlock` policy

`mlock` is deferred until production-readiness testing.

It is not a Phase 1 coding blocker.

## Operational Rules

- host orchestrator launches containers directly
- `llm-orchestrator.service` remains the single startup owner
- restart/recovery flows must restart or replace containers through the host control plane
- no detached side-launchers outside the service-owned orchestration path

## Acceptance For This Spec

Before Phase 1 is considered implemented enough to move on:

- [x] one image is chosen and named
- [x] build source is recorded
- [ ] one single-worker container command works end-to-end from the dedicated image
- [ ] the runtime can be started, probed, and stopped deterministically from the dedicated image
