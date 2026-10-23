# LLM Orchestration Engine

Parallel batch work on the local rig: a brain model reads a plan, creates tasks, and GPU and CPU workers
claim and run them through a shared filesystem (no message broker). This repo also holds the rig-side
hardware layer that every mode runs under.

## Start here

[shared/workspace/CONTEXT.md](shared/workspace/CONTEXT.md) is the entry point: what runs where, the modes,
which doc owns what. Then, by topic:

| Doc | For |
|---|---|
| [shared/workspace/quickstart.md](shared/workspace/quickstart.md) | submit, monitor and kill a plan |
| [shared/workspace/PLAN_FORMAT.md](shared/workspace/PLAN_FORMAT.md) | how plans are written (including goal-driven plans) |
| [shared/workspace/architecture.md](shared/workspace/architecture.md) | system design |
| [shared/workspace/brain-behavior.md](shared/workspace/brain-behavior.md) | the brain loop, task handling, work recovery and its time limits |
| [shared/scripts/rig/README.md](shared/scripts/rig/README.md) | the rig hardware layer: guard, power caps, the plan executor and runs, `rig observe` |
| [shared/workspace/systems_prep.md](shared/workspace/systems_prep.md), [cpu-worker-setup.md](shared/workspace/cpu-worker-setup.md), [NETWORK_SETUP.md](shared/workspace/NETWORK_SETUP.md) | building the rig host, the CPU workers and the network |

Docs use role placeholders for addresses (`<pi-ip>`, `<rig-ip>`, `<worker-ip>`, ...); the real values are
in a local, uncommitted inventory (`shared/workspace/local/inventory.md`). Scripts carry the real values.

## The four repos

| Repo | Owns | Start at |
|---|---|---|
| [llm-gateway](https://github.com/bigbertharig/llm-gateway) | the machines and the API: rig manager (leases, placement, model loads, runs), the OpenAI-compatible gateway, the dashboard, `rigctl` | `docs/MANAGER.md` |
| **llm_orchestration** (this one) | parallel batch work (brain, workers, plans), the rig hardware scripts, CPU workers | `shared/workspace/CONTEXT.md` |
| [benchmarking](https://github.com/bigbertharig/benchmarking) | which models are good: model registry and roles, suites, campaigns, results | `NEW_MODEL_INTEGRATION.md`, `MODEL_LIBRARY.md` |
| [llm-harness](https://github.com/bigbertharig/llm-harness) | one model in an interactive loop (no code yet) | `README.md` |

The model each role uses comes from benchmarking's registry and roles; GPU allocation is the rig manager's
(llm-gateway). This repo's brain and workers run as one rig mode among others.

## Machines

- **Pi** (`<pi-ip>`): control plane. Repo checkout, plan submission, llm-gateway, the dashboard; the 4 TB
  shared drive is attached here and exported over NFS.
- **GPU rig** (`<rig-ip>`): an RTX 3090 Ti (brain) and GTX 1060 6 GB worker cards, running `llama-server`
  runtimes. It mounts the shared drive at `/mnt/shared` and keeps its own checkout of this repo.
- **CPU workers** (`<worker-ip>`): Orange Pi Prime boards (ARM64, 2 GB) that claim `task_class: cpu` work.

## Running it

```bash
# On the Pi: submit a plan
cd ~/llm_orchestration
python3 scripts/submit.py ~/llm_orchestration/shared/plans/<plan_name> --config '{"VAR":"value"}'

# Watch it: the dashboard (served by llm-gateway on the Pi)
#   http://127.0.0.1:8080/dashboard/      (from elsewhere: ssh -N -L 8080:127.0.0.1:8080 pi)
```

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

The Git checkout is the canonical source tree. Its `shared/` directory is a versioned deployment image of
the shared drive, not the live mount, except on the Pi:

- source: `~/llm_orchestration/shared`
- live on the Pi: `/media/bryan/shared` (on the Pi, `~/llm_orchestration/shared` is a bind mount of the
  drive, so edits there are live at once)
- live on the GPU rig: `/mnt/shared`; the rig's own checkout is a separate copy on local disk, and its
  brain and GPU agents run from it (`git pull --ff-only pi main` there to update)

Compare source with the live drive, and deploy changed tracked files without deleting runtime files:

```bash
python3 scripts/sync_shared_source.py --live-root /media/bryan/shared            # compare
python3 scripts/sync_shared_source.py --live-root /media/bryan/shared --apply    # deploy
```

The sync always excludes `shared/core/**` and `shared/agents/permissions/**`. Machine-specific configs and
runtime state are not tracked.

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

## License

MIT
