# Orchestration context

This repository owns rig-side hardware discovery, physical admission, leases and claims, context reservations, runtime execution, and the imported model-library projection. Gateway owns dashboard/user work, conversations, remote access, cloud accounts, connections, approvals, model assignment and controller requests in the connected deployment. Benchmarking owns its published model-evidence library and comparative evaluations. Orchestration owns executable local profiles; standalone registration/loading remains Planned. Harness owns model-facing tools, bounded agent execution, context and execution journals/receipts.

## Current entry points

- `rig scan` refreshes the versioned hardware inventory on demand.
- `rig observe` reports GPUs, leases, claims, services, guard, jobs, and active controller.
- `rig lease preview --spec JSON` checks the current admission rules without reserving; lease acquisition repeats checks under the lock.
- `rig start remote <id> --spec ...` launches a reviewed model service after admission.
- The [rig hardware guide](../scripts/rig/README.md) is the operator and implementation reference.

The controller can run on the GPU host or another local machine. The GPU host keeps authoritative hardware state in its own checkout under `shared/brain/hw/`. Code-running hosts keep separate checkouts. A mounted shared drive is optional for hardware control; Big Bertha uses one for model files and benchmark artifacts. Control requests do not travel through task files on that drive.

Benchmarking exports a versioned evidence/model library. Its identity is authoritative for that artifact; the Planned standalone workflow also accepts explicit local models and configuration with separate provenance. `python3 shared/scripts/sync_model_library.py what-new|sync|check` imports the reviewed snapshot into `shared/model_library/`. Runtime code reads that local projection. Candidate and active models share the same source registry; active status governs normal dashboard offers, while candidate status remains available to tests.

The old brain/worker engine, Markdown work plans, NFS task queues, Pi submit wrappers, and private batch API have been removed. The rig's current `plan` queue is limited to hardware actions. Dashboard chat and Gateway-managed agent sessions replace interactive work orchestration.

Protected site instructions and private configuration are not published. See the [product boundary](../../README.md#product-boundary) for request and ownership boundaries.

The [implementation status](../../README.md#implementation-status) defines standalone acceptance and the migration sequence. Each repo keeps its own correctness tests; comparative measurements belong to Benchmarking.
