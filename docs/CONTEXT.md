# LLM Orchestration - Project Context

**Read this first.** This document helps you quickly understand the project and find the right documentation.

---

## What This Project Does (30 seconds)

A **multi-GPU LLM orchestration system** that coordinates local language models across 5x GTX 1060 GPUs. Uses a tiered intelligence hierarchy:

- **Claude** (external) writes plans
- **Brain** (Qwen 14B) interprets plans, creates tasks, monitors GPU agents
- **GPU Agents** (3 agents on GPUs 1, 2, 4) claim tasks, spawn worker subprocesses

**Key concept:** Smart planning happens externally. Local models handle execution and monitoring.

---

## Quick Orientation

| Term | Meaning |
|------|---------|
| **Plan** | Markdown file defining a goal, scripts, and tasks with dependencies |
| **Brain** | Coordinator agent (14B on GPUs 0+3) - fixes task-level issues |
| **GPU Agent** | One per physical GPU (GPUs 1, 2, 4) - claims tasks, spawns worker subprocesses |
| **Worker** | Short-lived subprocess spawned by GPU agent - executes one task and exits |
| **Dependency** | Task A depends_on Task B means B must complete before A runs |
| **Batch** | One execution run of a plan |
| **Escalation** | Problems flow upward: GPU Agent -> Brain -> Claude (see [architecture.md](architecture.md#error-handling--escalation)) |

---

## Documentation Map

### Start Here

| Doc | Purpose |
|-----|---------|
| **[quickstart.md](quickstart.md)** | Check status, start system, submit plans |
| **[architecture.md](architecture.md)** | High-level system overview, hardware, file structure |
| **[PLAN_FORMAT.md](../shared/plans/PLAN_FORMAT.md)** | How to write plans (the authoritative format) |

### Implementation Details

| Doc | Purpose |
|-----|---------|
| [brain-behavior.md](brain-behavior.md) | Brain loop, task handling, phase management, failure recovery |
| [distributed_work_guide.md](distributed_work_guide.md) | End-to-end workflow from planning to execution |

### Reference

| Doc | Purpose |
|-----|---------|
| [llm_benchmark_testing_guide.md](llm_benchmark_testing_guide.md) | How to benchmark GPU performance |
| [systems_analyst_questionnaire.md](systems_analyst_questionnaire.md) | Hardware and config review checklist |
| [future/resource_manager_design.md](future/resource_manager_design.md) | Future: GPU state management, LLM/script switching |

---

## Hardware Summary

| GPU | Role | Notes |
|-----|------|-------|
| 0 | Brain (pair) | Cooler, 2025 MHz |
| 1 | GPU Agent (gpu-1) | Runs hot (76-78C) |
| 2 | GPU Agent (gpu-2) | Runs hot (76-78C) |
| 3 | Brain (pair) | Cooler, 2025 MHz |
| 4 | GPU Agent (gpu-4) | Coolest |

- 6GB VRAM per card
- PCIe Gen1 x1 (~250 MB/s per GPU)
- 140W power limits

---

## File Structure

```
llm_orchestration/
├── config.json                 # RPi config (for submit.py)
│
├── scripts/                    # RPi-only utilities (not on GPU rig)
│   ├── submit.py          # Submit a plan for execution
│   ├── status.py               # Check system status
│   ├── watch.py                # Live monitoring
│   └── gpu-monitor.py          # GPU benchmarking tools
│
├── docs/                       # Documentation (you are here)
│
└── shared/                     # Mounted by BOTH machines
    │
    ├── agents/                 # Agent code (GPU rig runs these)
    │   ├── brain.py            # Brain coordinator
    │   ├── gpu.py              # GPU agent (one per physical GPU)
    │   ├── worker.py           # Worker subprocess (spawned by gpu.py)
    │   ├── executor.py         # Permission-aware command executor
    │   ├── launch.py           # Launcher script
    │   └── config.json         # GPU rig config
    │
    ├── plans/                  # Plan folders
    │   ├── PLAN_FORMAT.md      # Plan specification
    │   └── <plan_name>/
    │       ├── plan.md
    │       ├── scripts/
    │       └── batches/
    │
    ├── tasks/                  # Task queue
    │   ├── queue/              # Ready for workers
    │   ├── processing/
    │   ├── complete/
    │   └── failed/
    │
    ├── brain/                  # Brain state
    │   ├── state.json
    │   └── private_tasks/      # Tasks waiting for dependencies
    │
    ├── gpus/                   # GPU agent state
    │   └── gpu_<id>/
    │       └── heartbeat.json  # GPU agent heartbeat (sole owner, no lock needed)
    │
    ├── signals/                # GPU agent control signals (stop, abort, kill)
    └── logs/                   # Logs
```

**Key insight:** The `shared/` folder is mounted by both RPi and GPU rig. Agent code lives in `shared/agents/` so the air-gapped GPU rig can run it.

---

## Quick Start

See **[quickstart.md](quickstart.md)** for complete instructions.

### Submit a Plan

```bash
python scripts/submit.py <plan_name> \
  --config '{"INPUT_VAR": "/path/to/input"}'
```

### Monitor Execution

```bash
tail -f shared/logs/brain_decisions.log
```

### Check Status

```bash
python scripts/status.py
```

---

## Key Concepts

### Plans

Plans are markdown files that tell the brain what to do. They contain:
- Goal description
- Available scripts with run commands
- Tasks with explicit dependencies

See [PLAN_FORMAT.md](../shared/plans/PLAN_FORMAT.md).

### Dependency-Based Task Release

Tasks specify what they depend on:

```markdown
### aggregate
- **executor**: worker
- **task_class**: cpu
- **command**: `python scripts/combine.py`
- **depends_on**: process
```

The brain releases tasks when all their dependencies complete. This is a **Gantt-chart style** model - flexible ordering based on actual dependencies, not rigid phases.

### Private/Public Task Lists

- Brain holds all tasks in a private list (`shared/brain/private_tasks/`)
- Releases to public queue when dependencies are met
- Workers only see the public queue (`shared/tasks/queue/`)

This keeps workers simple while brain controls all sequencing.

---

## What to Read Next

- **Building a new plan?** → [PLAN_FORMAT.md](../shared/plans/PLAN_FORMAT.md)
- **Understanding the system?** → [architecture.md](architecture.md)
- **Debugging brain behavior?** → [brain-behavior.md](brain-behavior.md)
- **Following work end-to-end?** → [distributed_work_guide.md](distributed_work_guide.md)

---

## Design Principles

**No Backwards Compatibility.** This project maintains ONE clean way of doing things. When the design changes:
- Update the code to the new design
- Update all documentation
- Delete old patterns, don't keep fallbacks "just in case"
- If old plans break, fix them or delete them

Why: Backwards compatibility creates technical debt, confusing code paths, and documentation that lies. A small project like this should stay clean and simple.

**Fail Fast, Fix Smart.** When something is wrong (like a missing required field), fail immediately with a clear error. The brain can then attempt to fix the issue rather than silently guessing.

---

## Current Status

**Working:**
- GPU monitoring and benchmarking
- Task queue with filesystem state
- Brain + GPU agent architecture in `shared/agents/`
- GPU agent model: one agent per physical GPU, spawns worker subprocesses
- Training data collection
- Dependency-based task release (private/public lists)
- GPU agent self-awareness (temp, VRAM, power monitoring via nvidia-smi)
- Resource-aware task claiming (GPU agents back off when constrained)
- Hot/Cold state management (LLM loaded vs empty GPU)
- VRAM budget system (GPU agent tracks internally, limits concurrent workers)
- Definition error detection and auto-fix
- Independent component logging (GPU_LOG_LEVEL, EXECUTOR_LOG_LEVEL, WORKER_LOG_LEVEL, BRAIN_LOG_LEVEL)
- Task memory system (attempts tracking, retry limits)
- Dual-cycle GPU agent loop (30s external heartbeat + 5s internal poll)
- Signal system: brain sends stop/abort/kill signals to GPU agents
- Stuck task detection with 3-level escalation (abort -> force_kill -> manual)
- Foreach task expansion (template tasks expand into N tasks from manifest)
- Auto batch summary task (depends on all other tasks)
- Variable substitution in commands ({BATCH_ID}, {PLAN_PATH}, {BATCH_PATH})
- Launch singleton lock and model preload detection

**Planned:**
- RPi gateway for Claude escalation

---

*Last Updated: February 2026*
