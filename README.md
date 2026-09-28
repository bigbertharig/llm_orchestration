# LLM Orchestration Engine

Distributed task orchestration system for running local LLM workloads across GPU and CPU workers. A local brain coordinates plans while GPU and CPU workers execute tasks through a shared filesystem control plane.

## Architecture

- **Operator host** (10.0.0.2): Repo checkout, plan authoring, dashboard, and operator wrappers
- **GPU rig** (10.0.0.3): NFS source, brain GPU, worker GPUs, and `llama-server` runtimes
- **CPU workers** (10.0.0.10+): Orange Pi Prime cluster (ARM64, 2GB RAM), claim CPU tasks over NFS
- **Brain**: Config-selected coordinator model that interprets plans, creates tasks, monitors workers, and validates results
- **GPU workers**: Config-selected worker models that execute LLM and script tasks
- **Shared drive**: 4TB ext4 storage attached to the GPU rig and NFS-mounted by the other nodes
- **File-based coordination**: Tasks, state, and signals managed via shared filesystem — no message broker needed

See [shared/workspace/architecture.md](shared/workspace/architecture.md) for detailed system design.

## Hardware

- **Operator host**: Control plane at 10.0.0.2
- **GPU rig**: RTX 3090-class brain GPU plus five 6GB worker GPUs at 10.0.0.3
- **CPU workers**: Orange Pi Prime (Allwinner H5, 4-core ARM64, 2GB RAM) running Armbian
- **Network**: 10.0.0.x subnet, gigabit switch, NFS-mounted shared drive on all nodes

## Quick Start

See [shared/workspace/quickstart.md](shared/workspace/quickstart.md) for full setup instructions.

### Installation

```bash
# On the operator host: activate the orchestration venv and install dependencies
source ~/llm-orchestration-venv/bin/activate
pip install -r requirements.txt
```

### Running the System

```bash
# Start the full system (on GPU rig via shared drive)
cd ~/llm_orchestration/shared/agents
source ~/llm-orchestration-venv/bin/activate
python3 startup.py

# Submit a plan (from control plane)
cd ~/llm_orchestration
python3 scripts/submit.py ~/llm_orchestration/shared/plans/<plan_name> --config '{"VAR":"value"}'

# Monitor via web dashboard
cd ~/llm_orchestration/scripts
python3 -m dashboard --port 8787
```

### Interactive Model Access

The optional Pi-side model access API gives LAN gateways a narrow session
contract over the existing runtime loader and GPU leases. It binds to localhost
by default; only the client-facing gateway needs a LAN listener.

```bash
ORCHESTRATOR_API_KEY=replace-me \
python3 scripts/model_access_api.py \
  --shared-root /media/bryan/shared \
  --rig-host 10.0.0.3
```

See
[`shared/workspace/implement/interactive_model_access.md`](shared/workspace/implement/interactive_model_access.md)
for the API contract and current model allowlist.

## Plans

Plans define workflows as markdown files. Each plan contains:
- `plan.md` — Task definitions with dependencies
- `scripts/` — Executable scripts for task processing
- `history/` — Batch execution runs

See [shared/workspace/PLAN_FORMAT.md](shared/workspace/PLAN_FORMAT.md) for plan specifications.

## Key Features

- **Dependency-based task execution**: Tasks run when dependencies complete (Gantt-chart style)
- **Goal-driven plans**: Dynamic task generation — brain launches candidates, validates results, spawns replacements until target count is met
- **Mixed GPU/CPU workers**: GPU workers handle LLM and script tasks; CPU workers (Orange Pi cluster) handle file I/O, scraping, data transforms
- **Hot/cold worker states**: Dynamic LLM model loading based on workload pressure
- **Automatic retries**: Brain retries failed tasks with model rotation and dependency-aware re-queuing
- **Web dashboard**: Real-time monitoring with worker tabs, task lanes, batch progress, and goal tracking
- **File-based coordination**: Works across heterogeneous machines via NFS — no message broker or database
- **Security layers**: Protected core/ (root-owned), executor command filtering, git pre-commit hooks
- **Headless worker provisioning**: Burn SD image, plug in, worker auto-configures hostname and joins cluster

## Project Structure

```
llm_orchestration/
├── scripts/              # RPi utilities (submit.py, dashboard.py, watch.py)
└── shared/               # 4TB ext4 USB, NFS-shared to all nodes
    ├── agents/           # Brain, GPU agent, CPU agent code (synced to git)
    ├── core/             # Protected agent instructions (root-owned, NOT in git)
    ├── workspace/        # Docs, plan format spec, human escalations (synced to git)
    ├── plans/            # Plan definitions and scripts (synced to git)
    ├── logs/             # System logs (synced to git)
    ├── tasks/            # Runtime task queue/processing/complete/failed (not in git)
    ├── brain/            # Brain state and private tasks (not in git)
    ├── gpus/             # GPU agent heartbeats (not in git)
    ├── cpus/             # CPU worker heartbeats (not in git)
    └── signals/          # GPU agent control signals (not in git)
```

## Documentation

- [CONTEXT.md](shared/workspace/CONTEXT.md) — Project overview and orientation
- [PLAN_FORMAT.md](shared/workspace/PLAN_FORMAT.md) — Plan specifications (including goal-driven plans)
- [architecture.md](shared/workspace/architecture.md) — System architecture and design
- [brain-behavior.md](shared/workspace/brain-behavior.md) — Brain loop and task handling
- [quickstart.md](shared/workspace/quickstart.md) — Setup and usage guide

## License

MIT

## Contributing

This is a personal project for orchestrating local LLM workloads. Plans (workflows) are maintained as separate repos.
