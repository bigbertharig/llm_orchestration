# System Architecture

High-level overview of the LLM orchestration system. For implementation details, see [brain-behavior.md](brain-behavior.md).

---

## What This System Does

A **multi-GPU LLM orchestration system** that coordinates local language models across 5 GTX 1060 GPUs. Uses a tiered intelligence hierarchy where smart planning happens externally (Claude) and local models handle execution and monitoring.

**Key concept:** Local-first AI with tiered intelligence. Simple tasks run on local 7B models, coordination runs on local 14B brain, complex planning comes from external Claude.

---

## Intelligence Hierarchy

```
┌─────────────────────────────────────────────────────────────────────┐
│  CLAUDE (Planner)                                                   │
│  • Writes plans with goals, scripts, and workflow guidance          │
│  • Prepares environment setup in plan                               │
│  • External, accessed via RPi gateway (future)                      │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ writes plan.md
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  BRAIN (Qwen 14B on GPUs 0+3)                                       │
│  • Interprets plans using LLM reasoning                             │
│  • Creates and sequences tasks                                      │
│  • Monitors workers, evaluates outputs                              │
│  • Handles failures and retries                                     │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ assigns tasks
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  WORKERS (Qwen 7B on GPUs 1, 2, 4)                                  │
│  • Claim and execute tasks                                          │
│  • Run shell commands and scripts                                   │
│  • Report results                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

See [brain-behavior.md](brain-behavior.md) for detailed brain loop and task handling.

### Error Handling & Escalation

Each layer handles problems at its level. Unresolvable issues escalate upward.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  CLAUDE (Planner)                                                       │
│                                                                         │
│  Handles: Plan-level failures escalated from brain                      │
│  Actions: Rewrite plan, fix scripts, adjust approach                    │
│  Status:  Future - currently plan failures are terminal                 │
└─────────────────────────────────────────────────────────────────────────┘
                            ▲ escalate plan failures (future)
                            │
┌─────────────────────────────────────────────────────────────────────────┐
│  BRAIN (Coordinator)                                                    │
│                                                                         │
│  Handles: Task-level failures from workers                              │
│  Actions:                                                               │
│    • Retry failed tasks (up to 3 attempts)                              │
│    • Fix task definition errors (infer missing fields)                  │
│    • Reassign tasks from dead workers                                   │
│    • Evaluate output quality, request rework                            │
│  Escalates: Plan-level issues it cannot fix                             │
└─────────────────────────────────────────────────────────────────────────┘
                            ▲ report task failures
                            │
┌─────────────────────────────────────────────────────────────────────────┐
│  WORKERS (Executors)                                                    │
│                                                                         │
│  Handles: Command execution                                             │
│  Actions: Run command, report success/failure with output               │
│  Escalates: All failures go to brain (workers don't retry)              │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key principle:** Intelligence matches responsibility. Workers are simple - they just execute and report. Brain has enough intelligence to diagnose and fix task-level issues. Claude has the full context to fix plan-level problems.

| Failure Type | Who Handles | Example |
|--------------|-------------|---------|
| Command fails | Brain retries | Script exits non-zero, timeout |
| Missing task field | Brain infers | `task_class` missing, brain infers from command |
| Worker dies | Brain reassigns | Heartbeat timeout, task reset to queue |
| Output poor quality | Brain re-queues | Evaluation score < 3 |
| Plan malformed | Claude (future) | Missing sections, invalid structure |
| Script bug | Claude (future) | Repeated failures across retries |

---

## Hardware

| GPU | Role | Notes |
|-----|------|-------|
| 0 | Brain (pair with 3) | Cooler, 2025 MHz |
| 1 | Worker 1 | Runs hot (76-78C) |
| 2 | Worker 2 | Runs hot (76-78C) |
| 3 | Brain (pair with 0) | Cooler, 2025 MHz |
| 4 | Worker 3 | Coolest |

**Constraints:**
- PCIe Gen1 x1 via ASMedia switch (~250 MB/s per GPU)
- 6GB VRAM per card
- 140W power limit per card
- Models must load sequentially (PCIe bottleneck)

**Performance:**
- Brain (14B split): 12.4 tok/s
- Workers (7B each): ~22.6 tok/s
- Combined workers: ~68 tok/s parallel

---

## Physical Architecture

```
┌──────────────────────────────────┐     ethernet     ┌──────────────────────────────────┐
│         Manager (RPi)            │◄────────────────►│           GPU Rig                │
│  • Internet access               │                  │  • Air-gapped (no internet)      │
│  • Runs Claude Code              │                  │  • 5x GTX 1060 6GB               │
│  • Prepares plans                │                  │  • Runs brain + workers          │
│  • Submits to shared folder      │                  │  • Executes plans                │
└──────────────────────────────────┘                  └──────────────────────────────────┘
                    │                                              │
                    └──────────── shared/ folder ──────────────────┘
                                 (mounted by both)
```

**Communication is file-based only.** No network APIs between machines.

---

## File Structure

**Key insight:** The `shared/` folder is mounted by both RPi and GPU rig. Agent code lives in `shared/agents/` so the GPU rig can run it.

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
├── docs/                       # Documentation
│   ├── CONTEXT.md              # Start here
│   ├── architecture.md         # This file
│   └── ...
│
└── shared/                     # Mounted by BOTH machines
    │
    ├── agents/                 # Agent code (GPU rig runs these)
    │   ├── brain.py            # Brain coordinator
    │   ├── worker.py           # Worker executor
    │   ├── launch.py           # Launcher script
    │   └── config.json         # GPU rig config
    │
    ├── plans/                  # Plan folders
    │   ├── PLAN_FORMAT.md      # Plan specification
    │   └── <plan_name>/
    │       ├── plan.md         # Plan definition
    │       ├── scripts/        # Plan scripts
    │       └── batches/        # Execution runs (logs per batch)
    │
    ├── tasks/                  # Task queue (file-based)
    │   ├── queue/              # Ready for workers to claim
    │   ├── processing/         # Being worked on
    │   ├── complete/           # Finished (with result)
    │   └── failed/             # Failed (for retry logic)
    │
    ├── brain/                  # Brain state
    │   ├── state.json          # Active batches, status
    │   └── private_tasks/      # Tasks waiting for dependencies
    │
    ├── workers/                # Worker status
    │   └── <worker_id>/
    │       ├── heartbeat.json
    │       └── status.json
    │
    ├── signals/                # Worker control signals
    │
    └── logs/                   # Logs
        ├── brain_decisions.log
        └── training_samples.jsonl
```

---

## Task Flow

```
1. Claude writes plan.md with tasks and dependencies
2. User submits: python scripts/submit.py <plan_name> --config '{...}'
3. Brain parses plan.md directly (no LLM needed for parsing)
4. Brain creates all tasks, stores in private_tasks/
5. Brain releases tasks with no dependencies to queue/
6. Workers claim and execute tasks, write results to complete/
7. Brain checks completed tasks, releases dependent tasks
8. When all tasks complete, batch is done
```

**Dependency model:** Tasks specify `depends_on: [task1, task2]`. Brain releases a task to the public queue only when all its dependencies have completed successfully. This is like a Gantt chart - flexible ordering based on actual dependencies, not rigid phases.

---

## Plan Format

Plans are markdown files that tell the brain what to do. See [PLAN_FORMAT.md](../shared/plans/PLAN_FORMAT.md) for the complete specification.

Key sections:
- **Goal** - What success looks like
- **Inputs** - Configuration variables
- **Available Scripts** - Tools with run commands
- **Workflow** - Execution order (init → processing → aggregate)

---

## Task Classes

Tasks are classified by what resources they need:

| Class | Needs LLM? | Needs GPU? | Example |
|-------|------------|------------|---------|
| `cpu` | No | No | File manipulation, aggregation |
| `script` | No | Yes | Whisper transcription, embeddings |
| `llm` | Yes | Yes | Text generation, reasoning |

Brain uses task classes to decide which workers can claim which tasks.

---

## Hot and Cold Workers

Workers can be in two states:

| State | LLM Loaded? | Can Handle | Use Case |
|-------|-------------|------------|----------|
| **Hot** | Yes | `cpu`, `script`, `llm` | Ready for any task type |
| **Cold** | No | `cpu`, `script` | GPU compute without LLM overhead |

**Startup behavior:**
- All 3 workers start immediately (GPUs 1, 2, 4)
- 1 worker starts **hot** (LLM loaded, ready for `llm` tasks)
- 2 workers start **cold** (immediately available for `script` tasks)

**Dynamic transitions:**
- **Heat up**: Brain sends `load_llm` task when `llm` tasks are queued
- **Cool down**: Brain sends `unload_llm` task when only `script` tasks remain

**Why this matters:**
- `script` tasks (like Whisper transcription) don't need LLM - cold workers handle them immediately
- Avoids loading 3 LLMs when most work is GPU compute
- Brain can heat up workers on demand when `llm` tasks appear

---

## Task Types

| Type | Handler | Description |
|------|---------|-------------|
| `execute_plan` | Brain | Read plan.md, generate tasks |
| `shell` | Workers | Run shell commands |
| `decide` | Brain | Complex reasoning |
| `generate` | Workers | Text generation (needs hot worker) |
| `parse` | Workers | Extract structured data |

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| File-based queue | Simple, debuggable, works across air-gap |
| Brain/worker split | 14B for coordination, 7B for throughput |
| Qwen 2.5 | Best reasoning/size ratio for 6GB cards |
| Claude as planner | Smart work upfront, dumb execution locally |
| Private/public task lists | Brain controls sequencing, workers stay simple |

---

## State & Recovery

**All state lives in the filesystem.** Nothing critical is stored only in memory.

### Multi-Plan Execution

Multiple plans can be submitted and run concurrently:

```
Plan A (video processing)  ──┐
                             ├──► Brain's task pool ──► Workers
Plan B (document indexing) ──┘
```

- Brain tracks active batches in `state.json`
- Tasks from all batches mix in the public queue
- Workers don't know which plan a task belongs to - they just execute
- Each batch's output stays isolated in its own `batches/{batch_id}/` folder

### Graceful Recovery

The system survives restarts, crashes, and power loss:

| State | Location | On Restart |
|-------|----------|------------|
| Active batches | `brain/state.json` | Brain reloads and continues monitoring |
| Pending tasks | `brain/private_tasks/` | Released when dependencies met |
| Ready tasks | `tasks/queue/` | Workers claim them |
| In-progress tasks | `tasks/processing/` | Brain detects stale tasks, re-queues |
| Completed tasks | `tasks/complete/` | Already done, triggers dependency release |
| Failed tasks | `tasks/failed/` | Brain retries or leaves for review |

**Recovery flow:**
1. Brain starts, loads `state.json` → knows which batches were active
2. Scans `private_tasks/` → finds unreleased tasks
3. Scans `complete/` → knows what finished while down
4. Releases newly-unblocked tasks to queue
5. Detects orphaned tasks in `processing/` (worker died) → re-queues them
6. Resumes normal loop

**Task state is never lost.** If a worker crashes mid-task:
- The task stays in `processing/`
- Brain detects it's orphaned (no heartbeat from worker)
- Brain moves it back to `queue/`
- Another worker picks it up and starts fresh

The *work* on that specific task run is lost, but the task naturally returns to queue. This is why scripts should be **idempotent** - safe to restart from scratch if interrupted.

---

## Related Docs

| Doc | Purpose |
|-----|---------|
| [CONTEXT.md](CONTEXT.md) | Project entry point, quick orientation |
| [brain-behavior.md](brain-behavior.md) | Brain loop, task handling, failure recovery |
| [PLAN_FORMAT.md](../shared/plans/PLAN_FORMAT.md) | How to write plans |
| [future/resource_manager_design.md](future/resource_manager_design.md) | Future: GPU resource management design |

---

*Last Updated: February 2026*
