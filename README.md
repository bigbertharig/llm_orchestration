# LLM Orchestration Engine

Multi-GPU distributed task orchestration system for running local LLM workloads across multiple GPUs. Uses a tiered intelligence hierarchy with a coordinator (brain) and parallel workers.

## Architecture

- **Brain (Qwen 14B)**: Coordinator running on GPUs 0+3, interprets plans, creates tasks, monitors workers
- **Workers (Qwen 7B)**: Executors on GPUs 1, 2, 4, claim and run tasks in parallel
- **File-based coordination**: Tasks, state, and signals managed via shared filesystem

See [docs/architecture.md](docs/architecture.md) for detailed system design.

## Hardware

- 5x GTX 1060 6GB GPUs
- PCIe Gen1 x1 (~250 MB/s per GPU)
- Brain: 12.4 tok/s (14B model split across 2 GPUs)
- Workers: ~22.6 tok/s each, ~68 tok/s combined parallel

## Quick Start

See [docs/quickstart.md](docs/quickstart.md) for full setup instructions.

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up Ollama and load models
# (See docs/quickstart.md for model setup)
```

### Running the System

```bash
# Start brain (on GPU rig)
cd shared/agents
python brain.py

# Start workers (on GPU rig)
python worker.py worker-1 --hot  # Start with model loaded
python worker.py worker-2
python worker.py worker-3

# Submit a plan (from manager/RPi)
python scripts/submit.py <plan_name> --config '{"VAR": "value"}'

# Monitor execution
python scripts/watch.py
```

## Plans

Plans are separate repositories that define workflows. Each plan contains:
- `plan.md` - Task definitions with dependencies
- `scripts/` - Executable scripts for task processing
- `batches/` - Runtime execution logs (not in git)

See [shared/plans/PLAN_FORMAT.md](shared/plans/PLAN_FORMAT.md) for plan specifications.

## Key Features

- **Dependency-based task execution**: Tasks run when dependencies complete (Gantt-chart style)
- **Hot/cold worker states**: Dynamic model loading based on workload
- **Automatic retries**: Brain retries failed tasks up to 3 times
- **Graceful degradation**: Stuck tasks detected and aborted (20 min timeout)
- **File-based coordination**: Works across air-gapped machines via shared filesystem

## Project Structure

```
llm_orchestration/
├── scripts/          # Manager utilities (submit.py, status.py, watch.py)
├── docs/             # Documentation
└── shared/           # Shared filesystem (network drive)
    ├── agents/       # Brain and worker code (THIS IS IN GIT)
    ├── plans/        # Plan repos (separate git repos)
    ├── tasks/        # Runtime task queue (not in git)
    ├── brain/        # Brain state (not in git)
    ├── workers/      # Worker heartbeats (not in git)
    └── logs/         # System logs (not in git)
```

## Recent Improvements

### Logging Enhancements (2026-02)
- **Worker lifecycle**: Start/stop/crash detection with task completion counts
- **Claim loop visibility**: Debug logging shows what workers see in queue
- **Stuck task detection**: Auto-abort tasks running >20 minutes
- **Model loading tracking**: Warns when load_llm tasks aren't claimed

### Resource Management
- **Sequential load_llm**: Only 1 model loads at a time (prevents PCIe contention)
- **Graceful abort + force kill**: Two-tier timeout system for stuck tasks

## Documentation

- [CONTEXT.md](docs/CONTEXT.md) - Project overview and orientation
- [architecture.md](docs/architecture.md) - System architecture and design
- [brain-behavior.md](docs/brain-behavior.md) - Brain loop and task handling
- [quickstart.md](docs/quickstart.md) - Setup and usage guide

## License

MIT

## Contributing

This is a personal project for orchestrating local LLM workloads. Plans (workflows) are maintained as separate repos.
