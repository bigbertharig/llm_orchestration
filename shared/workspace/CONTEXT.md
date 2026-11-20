# LLM Orchestration context

This repository owns rig-side hardware observation, physical admission, leases and claims, context reservations, runtime execution, and the imported model-data projection. The old brain/worker source remains for a reviewed removal; its startup mode and unit are retired. New user work and model assignment belong in Gateway. Benchmarking owns the canonical model library and test evidence. Harness owns model-facing tools and sessions. See the [cross-repo contract](https://github.com/bigbertharig/benchmarking/blob/master/CROSS_REPO_CONTRACT.md).

## Current entry points

- On the rig, `rig scan` refreshes the versioned hardware inventory. `rig observe` reports current GPUs, leases, services, jobs, guard, and controller state. `rig lease preview --spec JSON` performs the same fit checks as lease acquisition without allocating a lease.
- Gateway is the controller for model loads and dashboard work. Its Manager sends direct rig commands; the rig admits or refuses them and reports the actual service. The Pi plan submit wrapper, dashboard plan-start route, and manual model-load wrappers have been retired.
- The private Pi `scripts/orchestration_api` service reports legacy batch status, diagnostics, activity, and stop controls over a Unix socket. It does not accept new dashboard plan submissions.
- The rig hardware implementation and operator commands are in [shared/scripts/rig/README.md](../scripts/rig/README.md). Gateway's manager contract documents its current controller requests and remaining fit migration.

## Code and state locations

The controller and GPU rig each have their own code checkout. In the Big Bertha deployment the Pi also mounts a shared drive; that drive holds model archives, benchmark results, and legacy batch state. It is optional for a single-host installation. New control commands travel through direct controller-to-rig calls, not through files placed on the shared drive.

The rig keeps authoritative hardware state in its checkout under `shared/brain/hw/`. The retired batch source still describes shared task queues, plan files, heartbeats, and control policy. No new work uses those files for command submission. The obsolete Pi watchdog and shared hardware mirrors have been removed. The [published cross-repo contract](https://github.com/bigbertharig/benchmarking/blob/master/CROSS_REPO_CONTRACT.md) describes the migration boundary; do not treat mirrored files as a second hardware authority.

Model knowledge originates in Benchmarking's versioned export. `python3 shared/scripts/sync_model_library.py what-new|sync|check` imports the reviewed snapshot into Orchestration's local model files. Normal rig runtime does not import Benchmarking Python or use a mounted Benchmarking checkout for model selection.

## Legacy batch engine

The brain/worker startup mode is removed. Its source and older shoulder/arm documents await a reviewed removal; completed batch results and receipts remain. Dashboard chat and Gateway/Harness tools are the replacement for interactive work. Do not create new Pi-to-rig submission wrappers.

Protected instructions under `shared/core/` are deployment-owned and are not changed by this cleanup.
