# Modular Runtime Target System

Purpose: capture the target architecture, the current implementation status,
and the next cleanup steps for runtime, plan, and benchmark execution.

Latest audit: `system_unification_audit_2026-08-13.md`. It supersedes the
priority ordering below where the two differ. In particular, shared campaign
resource leases now precede further runtime-image work because live campaign
containers and orchestrator startup can currently contend for the same worker
ports.

August 13 implementation update: the main repo now has controller-neutral GPU
leases and separate boot-profile/controller-admission policy. Plan workers and
brain resource decisions honor leases, and the dashboard renders their state.
The historical `system_mode.json` status descriptions below are superseded by
`control_policy.json`; Plan-owned lease acquisition and transactional profile
transitions remain incomplete.

## Target Architecture

The system should behave as four clean layers:

1. Runtime base layer
- versioned `llama-runtime` images
- stable wrapper scripts
- swappable llama.cpp version without agent code edits

2. Control and admission layer
- `control_policy.py`
- `start_plan_mode.py`
- `start_benchmark_mode.py`
- `start_custom_mode.py`
- `submit.py`
- `kill_plan.py`
- `return_default.py`

3. Package layer
- benchmarks as packages that run on top of the runtime/control layer
- plans as packages that run on top of the runtime/control layer

4. Resource ownership layer
- each GPU owns its own runtime state
- controllers acquire atomic leases for physical GPUs or split groups
- a benchmark GPU should not claim plan tasks
- a plan GPU should not be stolen by a benchmark launch
- competing work should wait instead of colliding

## Current Status

### Done enough to treat as working

- runtime image selection is config-driven instead of hardcoded in `gpu_llama.py`
- single-worker and split-worker launches both resolve runtime image/profile through shared config logic
- runtime helpers have one canonical location under `shared/scripts/llama_runtime/`
- normal startup is explicit `neutral`: brain up, worker GPUs cold
- `control_policy.json` separates boot profile from controller admission
- Plan and benchmark wrappers select their profiles explicitly
- brain and worker claim paths honor controller admission
- `gpu_leases.py` is the atomic single- and multi-GPU allocation authority
- worker claims and brain placement exclude leased GPUs
- leases fail closed after expiry until runtime reconciliation and explicit release
- single-load heartbeat phases are explicit during load
- worker heartbeat now reconciles against live `/v1/models` when cached state is stale
- dashboard status overlays leases even when a worker heartbeat is missing
- shared-path normalization has improved in repo-side wrappers such as `runtime_preflight.py`

### Partially done

- the shared authority exists, but each controller still needs complete acquire/renew/release integration
- benchmark/custom/Plan transitions use preflight paths, but there is no transactional transition service
- the operator runbook still spans multiple scripts and documents
- runtime smoke/build helpers exist, but not yet as a complete stable/candidate promotion workflow

### Not done yet

- Plan-owned leases around scheduled runtime work
- benchmark campaign lease integration in the benchmark repository
- supervision lease integration in County Map
- automatic reconciliation for expired leases
- candidate/stable runtime image promotion flow
- end-to-end mixed-controller validation on disjoint GPUs

## Historical Live Baseline

The April 2, 2026 validation below predates `control_policy.json` and leases. It
is retained as recovery evidence, not as the current control contract:

- rig startup returns to:
  - `system_mode.json = idle / startup_finished`
- default warm worker returns to:
  - `gpu-2 = ready_single`
  - `loaded_model = qwen2.5-coder:7b`
  - `/v1/models = 200`
- reboot no longer leaves startup state ambiguous
- reboot no longer leaves the worker heartbeat stale while runtime is already healthy

Plan-side validation status:

- real production `github_analyzer/plan.md` parsed and executed
- preserved queue state survived reboot
- old restart deadlock was fixed
  - before: preserved batch `load_llm` blocked startup enqueue while `idle` blocked that batch task from running
  - now: startup default load runs first, then mode can return to `plans`

## Open Gaps

### 1. Controller lease lifecycles are incomplete

What remains:
- Plans acquire and renew leases for runtime-owning work, then release after reconciliation
- benchmark campaigns acquire canonical leases instead of directly assuming GPU ownership
- supervision sessions acquire canonical leases from County Map without duplicating storage
- each controller publishes stable `owner` and `run_id` values

The benchmark and supervision repository changes are deliberately outside this
repository's current implementation scope.

### 2. Preflight / cleanup gate is not final yet

Current state:
- better than before
- still not the single definitive gate for all mode transitions

Still needed:
- scan worker ports deterministically
- detect unmanaged llama containers
- detect stale processing/meta heartbeats
- detect stale split reservations
- fail loudly or clean deterministically

### 3. Profile transitions are not transactional

Current state:
- policy writes are atomic
- leases are atomic
- startup and wrappers still execute their steps independently

Still needed:
- preflight current processes, ports, runtimes, tasks, and leases
- acquire or validate required resources
- write the desired profile/admission policy
- start or stop components
- verify the resulting state or roll back clearly

Likely targets:
- a new shared transition service in `shared/agents/`
- `scripts/start_plan_mode.py`
- `scripts/benchmarks/start_benchmark_mode.py`
- `scripts/runtime_preflight.py`

### 4. Runtime image promotion flow is still missing

Still needed:
- build candidate image
- smoke known-good single-worker model
- smoke known-good split model
- smoke new target family
- promote by config change only after passing

Likely targets:
- `shared/scripts/llama_runtime/build_image.sh`
- `shared/scripts/llama_runtime/build_and_smoke_test.sh`
- possibly a dedicated promotion wrapper

## Recommended Next Sequence

1. Add Plan-owned lease lifecycle around runtime allocation and task execution.
2. Add lease/process/port reconciliation to the unified preflight.
3. Define the controller integration contract for benchmarks and supervision.
4. Implement a transactional profile transition service.
5. Add candidate runtime image build/smoke/promotion flow.

## Bottom Line

This track is not finished overall.

What is finished enough:
- neutral cold-start behavior
- controller admission policy
- atomic shared GPU lease authority
- Plan-side exclusion and dashboard visibility

What is next:
- controller-owned lease lifecycles
- stronger reconciliation and transitions
- cross-repository controller adoption
- runtime image promotion
