# LLM Orchestration Engine

Distributed task orchestration system for running local LLM workloads across GPU and CPU workers. Uses a tiered intelligence hierarchy with a coordinator (brain) and parallel workers, managed from a Raspberry Pi 5 control plane.

## Architecture

- **RPi 5** (10.0.0.2): Control plane, NFS server, internet gateway, Claude Code, plan authoring
- **GPU rig** (10.0.0.3): RTX 3090 Ti, runs brain + GPU workers via Ollama
- **CPU workers** (10.0.0.10+): Orange Pi Prime cluster (ARM64, 2GB RAM), claim CPU tasks over NFS
- **Brain (Qwen 32B)**: Coordinator — interprets plans, creates tasks, monitors workers, validates results
- **GPU workers (Qwen 7B)**: Execute LLM and script tasks on GPU
- **Shared drive**: 4TB ext4 USB on RPi 5, NFS-shared to all nodes
- **File-based coordination**: Tasks, state, and signals managed via shared filesystem — no message broker needed

See [shared/workspace/architecture.md](shared/workspace/architecture.md) for detailed system design.

## Hardware

- **Control plane**: Raspberry Pi 5 with 4TB ext4 USB drive
- **GPU rig**: RTX 3090 Ti (24GB VRAM), connected via ethernet at 10.0.0.3
- **CPU workers**: Orange Pi Prime (Allwinner H5, 4-core ARM64, 2GB RAM) running Armbian
- **Network**: 10.0.0.x subnet, gigabit switch, NFS-mounted shared drive on all nodes

## Quick Start

See [shared/workspace/quickstart.md](shared/workspace/quickstart.md) for full setup instructions.

### Installation

```bash
# On RPi: activate venv and install dependencies
source ~/ml-env/bin/activate
pip install -r requirements.txt
```

### Running the System

```bash
# Start the full system (on GPU rig via shared drive)
cd ~/llm_orchestration/shared/agents
source ~/ml-env/bin/activate
python launch.py

# CPU workers auto-start via systemd (cpu-agent.service)
# Plug in SD card + ethernet → worker joins cluster automatically

# Submit a plan (from RPi)
cd ~/llm_orchestration
python scripts/submit.py shared/plans/<plan_name> --config '{"VAR": "value"}'

# Monitor via web dashboard
python scripts/dashboard.py   # http://localhost:8080
```

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
