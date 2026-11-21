# Rig model fit contract

`rig model-fit --request JSON` is the read-only hardware preview used by Gateway. The request names a Benchmarking model ID, `use` (`chat`, `worker`, or `benchmark`), and optional profile, context, card count, pinned GPU IDs, engine, and placement policy. The rig reads its imported Benchmarking library, the GGUF header, prior load observations, live GPU state, claims, leases, and available host RAM. It returns `fit` and the library `source_sha256`; a later lease acquisition rechecks admission under the rig lock.

Gateway selects a model for work and checks its own imported library before making the call. It compares the returned source hash with its local projection and refuses a mismatched revision. Gateway may reserve capacity for queued work and ask the rig for another fit with `exclude_gpu_ids`. `assume_idle=true` gives an empty-rig capacity preview for waiting policy; it never authorizes a load. The rig chooses the actual card layout and returns a specific refusal when it cannot fit.

Benchmarking owns the model status and tested load details. `active` models may serve routine Chat and worker requests. `candidate` models can be fit and loaded for explicit benchmarking but are excluded from routine use. A Strata offer also needs `strata.validated_gpu_ids` in the library; the rig only offers those tested cards. Other rigs must validate their own layout before updating that field. The imported library snapshot remains usable if Benchmarking is temporarily unavailable.

`rig fit-observation --row JSON` stores `load` and `load_failed` receipts in rig-local state for the next fit. Gateway also retains its own receipt copy for Benchmarking. The rig's calibration file is not a model-library source and is not committed. A fit preview changes no lease, claim, service, or GPU state.

# Scheduling previews

`rig schedule-preview --request JSON` is the read-only multi-workload companion to `model-fit`.
It accepts unique workload IDs plus model fit intents and returns `rig-scheduling-preview/1` with:

- a revision derived from the boot, plan version, imported model library, hardware inventory, live
  availability, rig mode, and host-memory availability;
- Orchestration-generated placement alternatives and current-availability flags;
- the rig constraint that model loads are serialized; and
- measured layout/model load-time estimates when receipts exist, otherwise a labeled fallback.

This response is advice for Gateway's campaign simulator. It creates no lease or claim and does not
make a future placement authoritative. Every live load is rechecked through normal admission and may
be accepted, revised, delayed, or rejected if the rig has changed.
