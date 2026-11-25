# LLM Orchestration Engine

Orchestration is the GPU host's hardware and runtime layer. It discovers cards, checks physical admission, owns leases and context reservations, starts model services, and reports their observed state. Gateway is the controller in the active deployment. A complete standalone local-model workflow is Planned. Benchmarking publishes optional model evidence and measured recipes; Harness supplies model-facing tools and agent behavior.

## Project map

The [cross-repo contract](https://github.com/bigbertharig/benchmarking/blob/master/CROSS_REPO_CONTRACT.md) owns the product division. These are accepted
responsibilities; the [cleanup plan](https://github.com/bigbertharig/benchmarking/blob/master/REPOSITORY_CLEANUP_PLAN.md) separates active code from pending migration.
Each product must support a useful standalone workflow with optional integrations.

| Repository | Responsibility |
|---|---|
| [llm-gateway](https://github.com/bigbertharig/llm-gateway) | Dashboard, conversations, remote access, cloud accounts/connectors, connection configuration, operator approvals and work coordination. |
| [llm-harness](https://github.com/bigbertharig/llm-harness) | Agent loop, model-facing tools, bounded execution, context, delegation, execution journals and receipts. |
| [llm_orchestration](https://github.com/bigbertharig/llm_orchestration) | Local executable model profiles, hardware discovery, physical admission, reservations, runtime lifecycle and observation. |
| [benchmarking](https://github.com/bigbertharig/benchmarking) | Evaluation tasks, comparative experiments, measurements, scores, result history and published model evidence. |

Gateway coordinates the connected deployment. Direct clients and standalone callers are
part of the accepted design. Product state stays with its owner; versioned requests,
events, receipts and optional artifact imports connect the products. Private site
configuration supplies addresses, credentials, model files and mounts. The current
checkout dependencies are tracked in the cleanup plan, not a supported installation contract.

Orchestration is NVIDIA/CUDA-focused today. AMD/ROCm and Apple Silicon/Metal backends
are Planned. See [portability](docs/PORTABILITY.md) for backend acceptance.

## Start here

1. [Rig quickstart](shared/workspace/quickstart.md) covers scanning, observing, previewing, and starting admitted services.
2. [Hardware layer](shared/scripts/rig/README.md) describes the rig command, leases, controller ownership, guard, and runtime lifecycle.
3. [System context](shared/workspace/CONTEXT.md) explains code and state locations.
4. [External runtimes](EXTERNAL_REPOS.md) identifies upstream engines and local revisions.
5. [Portability and accelerator backends](docs/PORTABILITY.md) defines the multi-backend architecture.

On the GPU host, `rig scan` writes the versioned inventory in the rig's hardware state directory. `rig observe` is the live report consumed by a controller. `rig lease preview --spec JSON` checks fit without allocating; acquisition checks again under the rig lock. The controller may run on the same machine or a separate host. A shared drive is optional for control; it can hold model archives and benchmark results.

A chat reserves its admitted context and KV capacity until release. Unused tokens inside that reservation remain allocated for capacity accounting; no billing is implied. The rig controls the physical resources and reports observed placement. Gateway still contains some fit logic; moving those remaining calculations behind rig admission is tracked in the Gateway Manager guide.

## Code and deployment

`shared/scripts/rig/` contains the hardware service, guard, lease store, job executor, and systemd units. `shared/scripts/llama_runtime/` launches llama-server with the rig's runtime checks. `shared/model_library/` is the local projection imported from Benchmarking. Each code-running host uses its own checkout; Big Bertha also has a mounted drive for models and results. Private site configuration supplies addresses, mounts, and model paths. GPU count and indices are discovered rather than fixed by this repository.

The old Markdown plan, brain, GPU-worker, CPU-worker, NFS task queue, and private batch API were removed. The active rig `plan` is a hardware action queue for loads, unloads, and recovery; it does not interpret Markdown work plans. Gateway dashboard chat and agent sessions handle user work.

## Local test environment

Use this repository's own virtual environment rather than another project's interpreter:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q shared/scripts/rig/tests
```

## Managed Strata runtime

The rig's `rig start remote <id> --spec ...` accepts a reviewed `backend: strata` profile. It launches the registered pack, verifies service health and host RAM, and stops the engine on unload. The profile's measured requirements still determine admission. See [external runtimes](EXTERNAL_REPOS.md) for upstream license and revision tracking.

## License

MIT.
