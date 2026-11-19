# LLM Orchestration Engine

Orchestration observes Big Bertha's hardware, enforces admission and reservations, controls model runtimes, and runs the legacy brain/worker batch queue. Gateway coordinates user work and still handles part of fit, placement, and loading. [Work organization](https://github.com/bigbertharig/llm-gateway/blob/main/docs/WORK_ORGANIZATION_DIRECTION.md) records the ownership migration.

For each admitted chat, the rig reserves its configured context capacity, including KV memory, until release. A chat using fewer tokens still holds that reservation; this is capacity accounting, not billing. See the [rig hardware layer](shared/scripts/rig/README.md) for the operational contract.

## Project map

Big Bertha runs across a Pi control host, an isolated GPU rig, and optional CPU workers. Four repositories divide the system's work:

| Repository | Responsibility |
|---|---|
| [llm-gateway](https://github.com/bigbertharig/llm-gateway) | Serves the dashboard, conversations, model API, tool policy, and work coordination. Its Manager also coordinates some model placement and loading. |
| [llm_orchestration](https://github.com/bigbertharig/llm_orchestration) | Observes and controls rig hardware, enforces admission and reservations, manages runtimes, and runs the legacy brain/worker batch engine. |
| [benchmarking](https://github.com/bigbertharig/benchmarking) | Measures models and runtimes; publishes the model library, roles, load recipes, evaluation suites, and results. |
| [llm-harness](https://github.com/bigbertharig/llm-harness) | Builds and tests shared tools, agent loops, context, sessions, delegation, and recovery. Harness selection for dashboard work remains open. |

Benchmarking publishes a validated model library. Gateway imports it, serves model routes, and manages dashboard work. Harness supplies reviewed tool schemas and tests model interactions through Gateway routes. Orchestration enforces rig hardware rules and runs admitted model services. [The work-organization guide](https://github.com/bigbertharig/llm-gateway/blob/main/docs/WORK_ORGANIZATION_DIRECTION.md) tracks the remaining Manager-to-Orchestration migration.

Private site configuration supplies host addresses, credentials, model files, and mounts. The published examples use role placeholders. For another card mix, run a hardware survey and validate fit, context, concurrency, and speed before offering a load recipe. [Porting guide](https://github.com/bigbertharig/llm-gateway/blob/main/docs/PORTABILITY.md).

## Legacy worker task claims

The GPU plan workers serialize each queued task claim with a per-task `FileLock` in `shared/agents/gpu_tasks.py`. Under that lock, a worker rechecks the queue file, writes a processing record, and removes the queued file. The brain monitors claims and heartbeats for recovery. This path depends on the target NFS server and mount honoring file locks; test competing claims and worker loss there. Gateway-managed model services use the rig admission path.

## Read this repo

- [System context](shared/workspace/CONTEXT.md): where components run and which guide owns each topic.
- [Rig hardware layer](shared/scripts/rig/README.md): observation, guard, runtime and admission controls.
- [Operator quickstart](shared/workspace/quickstart.md): the existing plan and batch workflow.
- [Private API](scripts/orchestration_api/README.md): the status/control interface consumed by Gateway.

## Documentation index

[shared/workspace/CONTEXT.md](shared/workspace/CONTEXT.md) is the entry point: what runs where, the modes,
which doc owns what. Then, by topic:

| Doc | For |
|---|---|
| [EXTERNAL_REPOS.md](EXTERNAL_REPOS.md) | outside runtimes and source projects, local revisions, and update checks |
| [shared/workspace/quickstart.md](shared/workspace/quickstart.md) | submit, monitor and kill a plan |
| [shared/workspace/PLAN_FORMAT.md](shared/workspace/PLAN_FORMAT.md) | how plans are written (including goal-driven plans) |
| [shared/workspace/architecture.md](shared/workspace/architecture.md) | system design |
| [shared/workspace/brain-behavior.md](shared/workspace/brain-behavior.md) | the brain loop, task handling, work recovery and its time limits |
| [shared/scripts/rig/README.md](shared/scripts/rig/README.md) | the rig hardware layer: guard, power caps, the plan executor and runs, `rig observe` |
| [shared/workspace/systems_prep.md](shared/workspace/systems_prep.md), [cpu-worker-setup.md](shared/workspace/cpu-worker-setup.md), [NETWORK_SETUP.md](shared/workspace/NETWORK_SETUP.md) | building the rig host, the CPU workers and the network |

Docs use role placeholders for addresses (`<pi-ip>`, `<rig-ip>`, `<worker-ip>`, ...); the real values are
in a local, uncommitted inventory (`shared/workspace/local/inventory.md`). Scripts carry the real values.

The model each role uses comes from benchmarking's registry and roles. Gateway currently requests and admits model work through its Manager. The rig-side
layer enforces hardware rules and reports live state. The target contract moves
complete-capacity reservation and placement decisions here, with Gateway
consuming the resulting route and receipt. Its brain and workers run as one rig
mode among others.

## Machines

- **Controller** (`<controller-host>`): Gateway, dashboard, and model requests. Big Bertha runs this on a Pi; a single-host installation can run Gateway on the GPU host.
- **GPU rig** (`<rig-host>`): its own code checkout and `llama-server` runtimes. Its GPUs are discovered by `rig scan` and `rig observe`; no published card count is a load requirement. Big Bertha also mounts a shared drive for legacy batch work and backup.
- **CPU workers** (`<worker-ip>`): Orange Pi Prime boards (ARM64, 2 GB) that claim `task_class: cpu` work.

## Running it

```bash
# On the Pi: submit a plan
cd ~/llm_orchestration
python3 scripts/submit.py ~/llm_orchestration/shared/plans/<plan_name> --config '{"VAR":"value"}'

# Watch it: the dashboard (served by llm-gateway on the Pi)
#   http://127.0.0.1:8080/dashboard/      (from elsewhere: ssh -N -L 8080:127.0.0.1:8080 pi)
```

The dashboard UI lives only in llm-gateway. This repo's `scripts/orchestration_api` serves
versioned status and plan-control endpoints over a private Unix socket on the Pi. The gateway
reads it through HTTP and reads hardware status through its own `/v1/manager/status` API.
The service definition is `deploy/llm-orchestration-api.service`.

Starting and stopping the orchestration mode on the rig, and its startup order: `shared/workspace/CONTEXT.md`
and `shared/workspace/quickstart.md`.

## Plans

Plans define workflows as markdown files: `plan.md` (tasks with dependencies), `scripts/` (what tasks run)
and `history/` (batch runs). See [PLAN_FORMAT.md](shared/workspace/PLAN_FORMAT.md).

- Dependency-based task execution; goal-driven plans that launch candidates, validate them and spawn
  replacements until a target count is met
- Mixed GPU and CPU workers; hot and cold worker states (models loaded by demand)
- Retries with model rotation; work recovery for lost workers and stuck tasks
- File-based coordination across machines over NFS

## Source and live deployment

Each code-running host keeps its own checkout. The shared drive is optional for the hardware layer; Big Bertha also uses it for legacy batch coordination and as an extra copy. In this deployment, the Pi's `shared/` is a bind mount of that drive:

- source: `~/llm_orchestration/shared`
- live on the Pi: `<shared-drive-root>` (on the Pi, `~/llm_orchestration/shared` is a bind mount of the
  drive, so edits there are live at once)
- live on the GPU rig: `/mnt/shared` holds shared runtime state. The rig's own checkout is a separate
  local copy for the brain and GPU agents; update its tracked code from the Pi checkout with
  `git pull --ff-only pi main`. The Pi checkout is the deployment source and GitHub is the published remote.

Compare source with the live drive, and deploy changed tracked files without deleting runtime files:

```bash
python3 scripts/sync_shared_source.py --live-root <shared-drive-root>            # compare
python3 scripts/sync_shared_source.py --live-root <shared-drive-root> --apply    # deploy
```

The sync always excludes `shared/core/**` and `shared/agents/permissions/**`. Those protected paths contain
site-specific agent instructions and permissions and are provisioned outside Git; their contents are not
required to inspect the published rig hardware layer. Machine-specific configs and runtime state are not tracked.

## Project structure

```
llm_orchestration/
+-- scripts/              # Pi utilities: submit, kill, assign, chat, sync
+-- shared/               # the shared drive's tracked part
    +-- agents/           # brain, GPU agent, CPU agent code
    +-- scripts/rig/      # rig hardware layer (guard, power, plan executor, runs)
    +-- scripts/          # runtime, benchmark and worker scripts used from the drive
    +-- workspace/        # docs, plan format, human escalations
    +-- plans/            # plan definitions (benchmarking lives at plans/shoulders/benchmarking,
    |                     #   a separate repo, moving beside the others)
    +-- core/             # protected agent instructions (root-owned, not in git)
    +-- tasks/ brain/ gpus/ cpus/ signals/ logs/   # runtime state (not in git)
```

## Managed Strata runtime

The rig's `rig start remote <id> --spec ...` path accepts `backend: strata` from the gateway manager. The rig uses the pinned Strata control under `/data/bench-staging/strata-v0.2.10`, launches only a registered pack on GPU 0, verifies API health and at least 4 GiB available host RAM, records the upstream model id, and stops the engine when the profile unloads. The existing rig plan, GPU lease, mode, and guard admission still own the hardware. A second Strata engine is refused. Qwen3.8 requires its prepared expert pack on disk; the load path does not download or repack one.

## License

MIT
