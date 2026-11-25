# LLM Orchestration Engine

Orchestration owns local-model decisions and execution: model profiles and offers, fit,
placement, admission, scheduling, leases, context reservations, loading, recovery and observed
runtime state. It works through its CLI/API without Gateway. Gateway is an optional dashboard
that makes requests. Benchmarking evidence is optional; Harness owns agent/tool execution.

## Product boundary

When a local-runtime decision is ambiguous, Orchestration owns it. Its control service and
hardware executor are two layers of this product. Orchestration is NVIDIA/CUDA-focused;
other accelerator backends remain Planned. Local models can be registered without an
imported Benchmarking record. See [standalone operations](docs/STANDALONE.md).

## Control service and CLI

[Control-plane setup](docs/CONTROL_PLANE.md) describes the independently installed
`rig-control serve` service and `rig-control request` CLI. `orchestration-control/1` provides
versioned requests with authentication, durable request identity and explicit refusal/recovery
semantics. The service owns reconciliation and heartbeat; it continues without a dashboard.

## Start here

1. [Rig quickstart](shared/workspace/quickstart.md) covers scanning, observing, previewing, and starting admitted services.
2. [Hardware layer](shared/scripts/rig/README.md) describes the rig command, leases, controller ownership, guard, and runtime lifecycle.
3. [System context](shared/workspace/CONTEXT.md) explains code and state locations.
4. [External runtimes](EXTERNAL_REPOS.md) identifies upstream engines and local revisions.
5. [Portability and accelerator backends](docs/PORTABILITY.md) defines the multi-backend architecture.

On the GPU host, `rig scan` writes the versioned inventory in the rig's hardware state directory. `rig observe` is the live report consumed by a controller. `rig lease preview --spec JSON` checks fit without allocating; acquisition checks again under the rig lock. The controller may run on the same machine or a separate host. A shared drive is optional for control; it can hold model archives and benchmark results.

A chat reserves its admitted context and KV capacity until release. Unused tokens inside that reservation remain allocated for capacity accounting; no billing is implied. The rig controls the physical resources and reports observed placement. The local control service owns fit and placement policy; the hardware executor rechecks live admission.

## Code and deployment

`shared/scripts/rig/` contains the hardware service, guard, lease store, job executor, and systemd units. `shared/scripts/llama_runtime/` launches llama-server with the rig's runtime checks. `shared/model_library/` is the local projection imported from Benchmarking. Each code-running host uses its own checkout; Big Bertha also has a mounted drive for models and results. Private site configuration supplies addresses, mounts, and model paths. GPU count and indices are discovered rather than fixed by this repository.

The old Markdown plan, brain, GPU-worker, CPU-worker, NFS task queue, and private batch API were removed. The active rig `plan` is a hardware action queue for loads, unloads, and recovery; it does not interpret Markdown work plans. Gateway dashboard chat and agent sessions handle user work.

## Local test environment

Use this repository's own virtual environment rather than another project's interpreter:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e . -r requirements-dev.txt
.venv/bin/python -m pytest -q tests/control shared/scripts/rig/tests
```

## Managed Strata runtime

The rig's `rig start remote <id> --spec ...` accepts a reviewed `backend: strata` profile. It launches the registered pack, verifies service health and host RAM, and stops the engine on unload. The profile's measured requirements still determine admission. See [external runtimes](EXTERNAL_REPOS.md) for upstream license and revision tracking.

## License

MIT.

## Implementation status

Implemented and tested: versioned inspection, direct lifecycle and CLI wire contracts,
immutable local profile registration, controller-owned load/observe/stop, lease cleanup and
scoped power restoration. The November 25 Pi/GPU handoff passed real NVIDIA load, Gateway
inference, Gateway restart, unload, shared-KV turns and power-changing campaign recovery.
Orchestration passed 268 tests in the extended handoff. Comparative experiment source
moved to Benchmarking with provenance. See [standalone operations](docs/STANDALONE.md)
and the [control-plane guide](docs/CONTROL_PLANE.md).
