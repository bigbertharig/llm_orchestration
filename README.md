# LLM Orchestration Engine

Multi-GPU distributed task orchestration system for running local LLM workloads across multiple GPUs. Uses a tiered intelligence hierarchy with a coordinator (brain) and parallel workers, managed from a Raspberry Pi control plane.

## Architecture

- **RPi 5** (control plane): Internet, Claude Code, plan authoring, GitHub, NFS server
- **Brain (Qwen 14B)**: Coordinator running on GPUs 0+3, interprets plans, creates tasks, monitors workers
- **Workers (Qwen 7B)**: Executors on GPUs 1, 2, 4, claim and run tasks in parallel
- **Shared drive**: 4TB ext4 USB on RPi, NFS-shared to air-gapped GPU rig over direct ethernet
- **File-based coordination**: Tasks, state, and signals managed via shared filesystem

See [shared/workspace/architecture.md](shared/workspace/architecture.md) for detailed system design.

## Hardware

- **Control plane**: RPi 5 with 4TB ext4 USB drive
- **GPU rig**: 5x GTX 1060 6GB, air-gapped, connected via direct ethernet (10.0.0.1 ↔ 10.0.0.2)
- PCIe Gen1 x1 (~250 MB/s per GPU)
- Brain: 12.4 tok/s (14B model split across 2 GPUs)
- Workers: ~22.6 tok/s each, ~68 tok/s combined parallel

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

# Submit a plan (from RPi)
cd ~/llm_orchestration
python scripts/submit.py shared/plans/<plan_name> --config '{"VAR": "value"}'

# Monitor execution
python scripts/watch.py
```

## Plans

Plans define workflows as markdown files. Each plan contains:
- `plan.md` - Task definitions with dependencies
- `scripts/` - Executable scripts for task processing
- `history/` - Batch execution runs

See [shared/plans/PLAN_FORMAT.md](shared/plans/PLAN_FORMAT.md) for plan specifications.

## Key Features

- **Dependency-based task execution**: Tasks run when dependencies complete (Gantt-chart style)
- **Hot/cold worker states**: Dynamic model loading based on workload
- **Automatic retries**: Brain retries failed tasks up to 3 times
- **Graceful degradation**: Stuck tasks detected and aborted (20 min timeout)
- **File-based coordination**: Works across air-gapped machines via shared filesystem
- **Security layers**: Protected core/ (root-owned), executor permissions, git hooks

## Project Structure

```
llm_orchestration/
├── scripts/              # RPi utilities (submit.py, status.py, watch.py)
└── shared/               # 4TB ext4 USB, NFS-shared to GPU rig
    ├── agents/           # Brain and worker code (synced to git)
    ├── core/             # Protected agent instructions (root-owned, NOT in git)
    ├── workspace/        # Docs, ideas, human escalations (synced to git)
    ├── plans/            # Plan definitions (synced to git)
    ├── logs/             # System logs (synced to git)
    ├── tasks/            # Runtime task queue (not in git)
    ├── brain/            # Brain state (not in git)
    ├── gpus/             # GPU agent heartbeats (not in git)
    └── signals/          # GPU agent control signals (not in git)
```

## Documentation

- [CONTEXT.md](shared/workspace/CONTEXT.md) - Project overview and orientation
- [architecture.md](shared/workspace/architecture.md) - System architecture and design
- [brain-behavior.md](shared/workspace/brain-behavior.md) - Brain loop and task handling
- [quickstart.md](shared/workspace/quickstart.md) - Setup and usage guide

## License

MIT

## Contributing

This is a personal project for orchestrating local LLM workloads. Plans (workflows) are maintained as separate repos.
