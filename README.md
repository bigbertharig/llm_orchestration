# LLM Orchestration Engine

Orchestration is the GPU host's hardware and runtime layer. It discovers cards, checks physical admission, owns leases and context reservations, starts model services, and reports their observed state. Gateway is the controller and user-facing entry point. Benchmarking publishes model evidence and load recipes; Harness supplies model-facing tools and agent behavior.

## Project map

| Repository | Responsibility |
|---|---|
| [llm-gateway](https://github.com/bigbertharig/llm-gateway) | Dashboard, conversations, model API, tool policy, work coordination, and controller requests. |
| [llm_orchestration](https://github.com/bigbertharig/llm_orchestration) | Rig hardware observation, admission, leases, context reservations, and model runtimes. |
| [benchmarking](https://github.com/bigbertharig/benchmarking) | Model and hardware measurements, canonical model library, recipes, evaluation suites, and results. |
| [llm-harness](https://github.com/bigbertharig/llm-harness) | Shared tools, agent loops, context, sessions, delegation, and recovery. |

The four repositories communicate through reviewed requests, receipts, and versioned model-library exports. Benchmarking owns the source model registry. Each runtime keeps a local projection; it is not a second authority. A model can be a candidate for testing without being offered in the main dashboard. Gateway requests a load; Orchestration admits it against live hardware or returns a reason for refusal.

## Start here

1. [Rig quickstart](shared/workspace/quickstart.md) covers scanning, observing, previewing, and starting admitted services.
2. [Hardware layer](shared/scripts/rig/README.md) describes the rig command, leases, controller ownership, guard, and runtime lifecycle.
3. [System context](shared/workspace/CONTEXT.md) explains code and state locations.
4. [External runtimes](EXTERNAL_REPOS.md) identifies upstream engines and local revisions.

On the GPU host, `rig scan` writes the versioned inventory in the rig's hardware state directory. `rig observe` is the live report consumed by a controller. `rig lease preview --spec JSON` checks fit without allocating; acquisition checks again under the rig lock. The controller may run on the same machine or a separate host. A shared drive is optional for control; it can hold model archives and benchmark results.

A chat reserves its admitted context and KV capacity until release. Unused tokens inside that reservation remain allocated for capacity accounting; no billing is implied. The rig controls the physical resources and reports observed placement. Gateway still contains some fit logic; moving those remaining calculations behind rig admission is tracked in the Gateway Manager guide.

## Code and deployment

`shared/scripts/rig/` contains the hardware service, guard, lease store, job executor, and systemd units. `shared/scripts/llama_runtime/` launches llama-server with the rig's runtime checks. `shared/model_library/` is the local projection imported from Benchmarking. Each code-running host uses its own checkout; Big Bertha also has a mounted drive for models and results. Private site configuration supplies addresses, mounts, and model paths. GPU count and indices are discovered rather than fixed by this repository.

The old Markdown plan, brain, GPU-worker, CPU-worker, NFS task queue, and private batch API were removed. The active rig `plan` is a hardware action queue for loads, unloads, and recovery; it does not interpret Markdown work plans. Gateway dashboard chat and agent sessions handle user work.

## Managed Strata runtime

The rig's `rig start remote <id> --spec ...` accepts a reviewed `backend: strata` profile. It launches the registered pack, verifies service health and host RAM, and stops the engine on unload. The profile's measured requirements still determine admission. See [external runtimes](EXTERNAL_REPOS.md) for upstream license and revision tracking.

## License

MIT.
