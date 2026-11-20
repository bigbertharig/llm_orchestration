# Orchestration context

This repository owns rig-side hardware discovery, physical admission, leases and claims, context reservations, runtime execution, and the imported model-library projection. Gateway owns user work, conversations, model assignment, and controller requests. Benchmarking owns the canonical model library and test evidence. Harness owns model-facing tools and agent behavior.

## Current entry points

- `rig scan` refreshes the versioned hardware inventory on demand.
- `rig observe` reports GPUs, leases, claims, services, guard, jobs, and active controller.
- `rig lease preview --spec JSON` checks the current admission rules without reserving; lease acquisition repeats checks under the lock.
- `rig start remote <id> --spec ...` launches a reviewed model service after admission.
- The [rig hardware guide](../scripts/rig/README.md) is the operator and implementation reference.

The controller can run on the GPU host or another local machine. The GPU host keeps authoritative hardware state in its own checkout under `shared/brain/hw/`. Code-running hosts keep separate checkouts. A mounted shared drive is optional for hardware control; Big Bertha uses one for model files and benchmark artifacts. Control requests do not travel through task files on that drive.

Benchmarking exports a versioned model library. `python3 shared/scripts/sync_model_library.py what-new|sync|check` imports the reviewed snapshot into `shared/model_library/`. Runtime code reads that local projection. Candidate and active models share the same source registry; active status governs normal dashboard offers, while candidate status remains available to tests.

The old brain/worker engine, Markdown work plans, NFS task queues, Pi submit wrappers, and private batch API have been removed. The rig's current `plan` queue is limited to hardware actions. Dashboard chat and Gateway-managed agent sessions replace interactive work orchestration.

Protected site instructions and private configuration are not published. See the [cross-repo contract](https://github.com/bigbertharig/benchmarking/blob/master/CROSS_REPO_CONTRACT.md) for request and ownership boundaries.
