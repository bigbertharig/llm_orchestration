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

Open implementation choices to lock during build:

- image name
- image tag format
- base CUDA image
- exact compile flags
- exact binary install path

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
docker run --rm --name llama-worker-gpu2 \
  --gpus "device=2" \
  --network host \
  -v /mnt/shared/models:/mnt/shared/models \
  <llama-runtime-image> \
  llama-server \
  --model <gguf-path> \
  --host 127.0.0.1 \
  --port 11436
```

Brain:

```bash
docker run --rm --name llama-brain \
  --gpus <brain-gpu-selection> \
  --network host \
  -v /mnt/shared/models:/mnt/shared/models \
  <llama-runtime-image> \
  llama-server \
  --model <gguf-path> \
  --host 127.0.0.1 \
  --port 11434
```

Split:

```bash
docker run --rm --name llama-split-pair-1-3 \
  --gpus all \
  --network host \
  -v /mnt/shared/models:/mnt/shared/models \
  <llama-runtime-image> \
  llama-server \
  --model <gguf-path> \
  --host 127.0.0.1 \
  --port 11440 \
  --tensor-split 0 1 0 1 0 0
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

- one image is chosen and named
- build source is recorded
- one single-worker container command works end-to-end
- the runtime can be started, probed, and stopped deterministically
