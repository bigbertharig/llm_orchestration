# System Unification Audit - 2026-08-13

Scope: `/home/bryan/llm_orchestration`, the shared runtime tree,
`shared/plans/shoulders/benchmarking`, and a read-only review of
`/home/bryan/global map/county-map-private/tools/gpu_rig`.

This audit was read-only against the live rig. No runtime, task, reservation,
campaign, or County Map state was changed.

Implementation boundary: changes from this audit belong only in
`/home/bryan/llm_orchestration`. County Map and benchmarking remain read-only
integration clients while their current work is active.

## Executive Result

The rig has three distinct workload products. They should not be forced through
one scheduler:

| Product | Controller semantics | Primary output |
| --- | --- | --- |
| Benchmarks | Predetermined suite/campaign runner | Comparable model results and runtime evidence |
| Plans | DAG decomposition across local models, scripts, and CPU/GPU workers | Completed plan artifacts without routine cloud-model dependence |
| Supervision | One cloud model supervises one capable local model, optionally two independent local lanes | Reviewed County Map candidate artifacts and handoffs |

Unification belongs in a shared lower control plane: hardware inventory, leases,
runtime lifecycle, model catalog and hot set, health, telemetry, transitions,
and dashboard state. Each product should retain its own controller, state
machine, result contract, and failure policy.

Current priority order:

1. Introduce one per-GPU lease and runtime-ownership contract used by all three
   controllers.
2. Separate boot profile, controller admission, and observed GPU activity; the
   current `system_mode.json` mixes these concepts.
3. Integrate benchmark campaigns and County Map supervision as first-class
   control-plane clients without routing them through the Plan DAG.
4. Make the dashboard render the shared control-plane state plus
   controller-specific progress for every physical GPU.
5. Establish one config schema/generator and one read-only certification tool.

## System Boundaries

### Shared control plane

`llm_orchestration` should own the resource-facing contracts:

- physical GPU and port inventory;
- atomic GPU/group leases with owner and expiry;
- llama runtime launch, probe, stop, and cleanup;
- split-runtime membership and generation;
- model catalog, certified runtime profile, and hot-set location;
- thermal, utilization, VRAM, process, port, and heartbeat telemetry;
- boot-profile transitions and readiness evidence;
- normalized controller/run status for the dashboard;
- an append-only event stream or equivalent transition history.

### Benchmark controller

The benchmark package should continue to own:

- campaign manifests and deterministic suite ordering;
- suite Docker images and harness versions;
- retry/cascade policy;
- progress, raw logs, result recording, and scoreboards.

It should request leases and runtimes from the shared control plane. It should
not become a Plan or emit fake Plan tasks.

### Plan controller

The existing brain/task system should continue to own:

- Plan parsing and DAG dependency release;
- task classification and placement;
- worker retries, results, and batch completion;
- local-model versus script/CPU decomposition.

It should use the same lease/runtime API as the other controllers instead of
implicitly assuming every started worker GPU belongs to Plans.

### Supervision controller

The County Map GPU-rig flow should continue to own:

- cloud-to-local model sessions;
- one bounded assignment per local-brain lane;
- stage prompts, semantic review, and escalation;
- deterministic County Map scripts and validators;
- artifact claims and handoff packaging.

Its base topology is one cloud supervisor to one local brain on GPU 0. A second
independent local-brain lane is allowed. This is deliberately not broad Plan
decomposition. It needs a small adapter that acquires a lease/runtime, publishes
session progress, and releases the lane. That adapter belongs in
`llm_orchestration`; it does not require County Map changes or adoption of the
Plan DAG.

## Initial Evidence Collected

These counts and live observations were captured before the implementation
slice documented later in this file.

- Main agent tests: 209 passed.
- Dashboard tests: 4 passed.
- Benchmark package tests: 49 passed.
- Python compilation passed for first-party orchestration and benchmark code.
  A whole-tree benchmark compile encounters expected Python 2 and invalid
  syntax fixtures inside vendored benchmark repositories.
- Main repo: `main` is five commits ahead of `origin/main` and has a large
  existing uncommitted patch set.
- Benchmark repo: `master` is 29 commits ahead of `origin/master`; generated
  result ledgers are modified by the active campaign.
- County Map has two divergent working copies:
  `/home/bryan/global map` at `2d883514` and
  `/home/bryan/global-map-reconcile` at `2ac46efb`.
- County Map's current contract explicitly defines cloud supervision of one
  `qwen3.6:27b` local brain, with an optional second identical lane.
- No County Map supervision session, lease, or progress adapter exists in the
  orchestration control state or dashboard.
- Live campaign process: `run_campaign.py` was active for
  `20260813_limit5_all` when inspected.
- Live worker ports 11435 and 11438 served campaign models from
  `llama-campaign-*` containers.
- No live GPU worker heartbeats or lane reservation sidecars existed for those
  ports.
- `system_mode.json` reported `idle/startup_in_progress` even though the
  campaign was running.
- The canonical runtime preflight rejected both active campaign listeners as
  `listener_without_managed_container`.
- At the final audit sample, the campaign had 37 completed, 14 failed, one
  running, and eight dependency-waiting blocks.

The 14 failed blocks are not one problem:

- nine `ministral-3:3b` blocks cascaded after its port 11436 runtime vanished;
  the code suite's watchdog recorded 90 seconds of runtime unavailability and
  every following suite lost the same endpoint;
- four otherwise-completed reasoning runs were marked failed because the
  result recorder expected a group-level `n-samples` entry for `bbh` or
  `mmlu_pro`; lm-eval emitted subgroup sample counts instead;
- one `bench-runtime` launch failed before execution because Docker received a
  conflicting GPU device request.

## P0 - Shared GPU Leases

Implementation status: the controller-neutral lease repository, CLI, Plan
exclusion checks, heartbeat fields, brain resource exclusion, and dashboard
overlay now exist in `llm_orchestration`. Plan-owned acquisition around active
work and external-controller adoption remain open.

`shared/scripts/benchmarks/run_campaign.py` currently owns GPU slots only
inside its own process and launches `llama-campaign-*` containers directly.
The orchestrator cannot distinguish that valid benchmark activity from an
unmanaged listener. The same problem would affect a direct County Map local
brain session if it launched outside the existing startup process.

Target lease fields:

```text
lease_id, generation, controller, run_id, lane, owner_host, owner_pid,
gpu_ids, ports, runtime_ids, acquired_at, renewed_at, expires_at, state
```

Required behavior:

1. A controller requests one GPU or an atomic GPU group.
2. The control plane validates GPU, port, active task, split group, runtime,
   thermal, and existing lease state under one lock.
3. Runtime launch and cleanup require the current lease generation.
4. The controller renews while active and releases on every terminal path.
5. Expired leases become reclaimable only after process, port, and runtime
   reconciliation.
6. Startup and preflight recognize all valid controller leases as managed.
7. A controller can never stop a runtime owned by a different live lease.

The old `benchmark_reservation.json` prototype has been superseded on the Plan
side by `gpus/leases.json`. Direct benchmark campaigns do not yet adopt the new
authority and therefore remain invisible to it.

Until every controller adopts the lease contract, do not start or reset the
orchestrator while a direct campaign or supervision session is active.

## P1 - Modes Need Three Orthogonal States

Implementation status: `control_policy.json` now separates `boot_profile` from
`admitted_controllers`, and brain/worker task claiming uses controller
admission. Observed activity remains derived from leases and heartbeats. A
transactional profile transition service and dashboard selector remain open.

The audit originally found the following mode machinery:

- `system_mode.json` supports only `idle`, `plans`, and `benchmarks` and acts as
  a global Plan-task gate;
- `start_plan_mode.py` and `start_benchmark_mode.py` also change startup config
  and restart processes;
- `empty` aliases benchmark mode;
- `custom` means benchmark-isolated startup followed by explicit model loads;
- `default` means return to brain plus one warm worker;
- there is no `supervision` state or dashboard selector.

Replace the overloaded term `mode` with three explicit concepts:

1. `boot_profile`: desired startup/runtime posture.
   Initial profiles: `neutral`, `plan_ready`, `benchmark_ready`,
   `supervision_ready`, and `custom`.
2. `admission_policy`: which controllers may request new leases. This may allow
   more than one controller when their requested GPUs do not conflict.
3. `observed_activity`: derived per GPU from the live lease, runtime, process,
   task/session, and telemetry. Never set this by operator intent.

A boot-profile transition should be transactional:

```text
preflight -> draining -> stopping -> configuring -> starting -> verifying -> active
```

Persist transition ID, phase, target profile, admission policy, timestamps,
failure detail, and readiness checks under a global transition lock. A failed
transition must not publish the target profile as active.

`supervision_ready` should start or preserve the GPU 0 brain runtime with the
certified County Map model/profile and keep worker GPUs cold unless a second
lane is explicitly requested. It should not enable Plan task consumption by
implication.

## P1 - Dashboard Is Plan-Centric

Implementation status: the dashboard now always builds configured GPU rows,
overlays canonical leases, exposes the boot profile/admitted controllers in
the header, and includes the `SUPERVISING` lane. It remains Plan-centric below
that shared summary and has no transition/profile control surface.

The initial audit found that benchmark progress was inferred by scanning
suite-local status files. Remaining gaps are:

- it has no boot profile or admission selector or transition state;
- it does not show transition phase/failure;
- direct campaign GPUs remain invisible unless the campaign acquires a lease;
- benchmark progress is inferred by port from suite-specific files;
- County Map supervision sessions are absent;
- controls submit and manage Plans, plus broad reset/clear operations.

The primary view should always show physical GPUs 0 through 5, even when agents
are stopped. Each row should show:

```text
GPU, health, temperature/utilization/VRAM, lease controller, lane, run/session,
model, runtime state/port, current work, progress, heartbeat age, last error
```

Controller detail views should remain separate:

- Benchmarks: campaign/model/suite/sample progress and result state.
- Plans: batch/DAG/task queues and outputs.
- Supervision: session, assignment, stage, last exchange, validator state, and
  handoff claim.

The dashboard should read normalized control-plane JSON. It should not continue
adding controller-specific filesystem inference to `workers.py`.

## P1 - Controller Integration Gaps

### Benchmarks

- hardcoded GPU/port topology in the campaign runner;
- direct container lifecycle bypasses shared ownership;
- no lease renewal/release protocol;
- runtime failure cascades through dependent suites;
- reasoning result ingestion assumes group-level sample counts;
- Docker GPU selector construction can emit conflicting options.

### Plans

- task admission now uses controller policy and workers exclude leased GPUs;
- Plan scheduling still does not acquire/renew/release its own leases;
- started unleased GPU agents remain eligible for Plan work;
- Plan submission selects `plan_ready` as a side effect;
- shared brain/worker runtime ownership is spread across large modules;
- no controller-neutral run/event contract exists.

### Supervision

- strong workflow docs and deterministic scripts exist, but no runnable
  supervision-session controller exists in `llm_orchestration`;
- no lease type represents local-brain supervision;
- no normalized session/stage/progress record exists;
- County Map docs refer to direct port 11434 rather than a leased runtime
  endpoint;
- the two County Map working copies are divergent, so the adapter must consume
  an explicit operator-supplied repo/work root rather than guessing one.

## P1 - Configuration Drift

The active configs agree on `llama-runtime:b10333`, but campaign manifests
mostly pin `llama-runtime:b8884-candidate`, and the campaign runner still has an
obsolete `llama-runtime:sm61-sm86` fallback. The active config also contains a
duplicate `qwen2.5-coder:7b` profile key and stale generated hardware metadata.

Runtime image selection should be:

```text
explicit run override -> certified model profile -> active control-plane config
```

There should be no code-level image fallback. Introduce a versioned config
schema plus one generator for hardware inventory, certified models, boot
profiles, and controller overrides. Validate unique GPU IDs/ports, profile
keys, model availability, image presence, and split topology before writing.

## P1 - Repository State

The main and benchmark repositories contain many unpushed commits. The main
repo also has a large uncommitted cross-cutting patch, while benchmark result
files are changing. County Map has two divergent checkouts. This is a backup,
rollback, and integration-target risk.

After the active campaign finishes:

1. Keep the County Map and benchmarking repositories read-only from this work.
2. Commit the existing main runtime/mode patch in logical groups.
3. Run the main orchestration suites from a clean worktree.
4. Push only when external network use is explicitly allowed.

## P2 - Health And Certification

Add one read-only `system_audit.py --json` aggregator with stable checks:

- config/schema and boot-profile validity;
- transition and admission state;
- complete GPU inventory, telemetry, and heartbeat freshness;
- lease, runtime, port, container, and process reconciliation;
- queue/processing/private-task consistency;
- stale split state and runtime transitions;
- active campaign reconciliation;
- active supervision-session reconciliation;
- Git state for the main and controller repositories.

Exit codes should distinguish healthy, degraded, active-work conflict, and
invalid configuration. The dashboard should render this same normalized data.

## P2 - Module Boundaries

The largest modules are `gpu_split.py` (3,489 lines), `brain_resources.py`
(3,041), `gpu_tasks.py` (1,953), and `startup.py` (1,469). Size alone is not a
reason to refactor them, but their shared-state responsibilities overlap.

After lease and state contracts have tests, extract cohesive modules for:

- GPU/group leases;
- runtime launch/probe/stop;
- split-runtime state;
- boot-profile transitions;
- normalized telemetry/events;
- Plan task repository.

Keep controller semantics out of the shared runtime layer.

## Changes Applied During This Audit

- Made `shared/scripts/llama_runtime` the tracked, canonical runtime-helper
  directory.
- Removed the stale tracked duplicate under `scripts/llama_runtime`.
- Resolved brain, single-worker, split-worker, and tests against the shared
  helper path.
- Changed `config.template.json` to explicit neutral startup defaults.
- Made `setup.py` generate the same explicit catalog/runtime keys and emit
  warm-load tasks only when warm workers are requested.
- Corrected the top-level topology and coordination documentation.
- Reframed the target architecture around three independent controllers using
  one lower GPU control plane.
- Added an atomic controller-neutral GPU/group lease repository and operator
  CLI. Expired leases fail closed until explicitly reconciled.
- Replaced `system_mode.json` task gating with `control_policy.json`, separating
  boot profile from admitted controllers.
- Migrated Plan brain/worker task admission, GPU claim exclusion, brain resource
  decisions, worker heartbeats, and dashboard status to the new contracts.
- Added dashboard visibility for profile, admissions, global leases, and the
  `SUPERVISING` lane, including GPUs with missing worker heartbeats.

## Recommended Delivery Sequence

### Phase 1 - Protect Active Work

- Let `20260813_limit5_all` finish.
- Do not run startup/reset wrappers during the campaign.
- Capture final status and failed-suite diagnostics.
- Record an explicit County Map work root in supervision session config; do not
  modify or infer a canonical County Map checkout.

### Phase 2 - Establish The Control Plane

- Done: define and test the lease schema/repository and configured GPU inventory.
- Move runtime ownership checks behind lease generations.
- Done: teach the dashboard to recognize all controller leases.
- Teach preflight to reconcile all controller leases.
- Add collision tests across benchmark, Plan, supervision, chat, and split
  loads.

### Phase 3 - Integrate Controllers

- Adapt campaign allocation, renewal, progress, and release.
- Adapt Plan placement to acquire leases instead of assuming worker ownership.
- Add a minimal supervision-session adapter for one or two local-brain lanes.

### Phase 4 - Profiles And Dashboard

- Replace duplicated mode wrappers with one transition service.
- Add boot-profile/admission controls and verified transition state.
- Render all physical GPUs from normalized control-plane data.
- Add separate Benchmark, Plan, and Supervision detail panels.

### Phase 5 - Certification

- Add `system_audit.py` and CI coverage.
- Certify neutral, plan-ready, benchmark-ready, supervision-ready, custom,
  mixed non-conflicting leases, split loads, failure recovery, and reboot.
- Promote the resulting config/runtime pair as the stable baseline.
