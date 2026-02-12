# System Architecture

High-level overview of the LLM orchestration system. For implementation details, see [brain-behavior.md](brain-behavior.md).

---

## What This System Does

A **multi-GPU LLM orchestration system** that coordinates local language models across multiple NVIDIA GPUs. Uses a tiered intelligence hierarchy where smart planning happens externally (Claude) and local models handle execution and monitoring.

**Key concept:** Local-first AI with tiered intelligence. Simple tasks run on local worker models, coordination runs on a local brain model, complex planning comes from external Claude.

---

## Intelligence Hierarchy

```
┌─────────────────────────────────────────────────────────────────────┐
│  CLOUD LLM (Planner) — e.g. Claude                                  │
│  • Writes plans with goals, scripts, and workflow guidance          │
│  • Prepares environment setup in plan                               │
│  • External, accessed via control plane                             │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ writes plan.md
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  BRAIN (local LLM on dedicated GPU(s) — per config.json)            │
│  • Interprets plans using LLM reasoning                             │
│  • Creates and sequences tasks                                      │
│  • Monitors workers, evaluates outputs                              │
│  • Handles failures and retries                                     │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ assigns tasks
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  GPU AGENTS (optional — one per worker GPU, per config.json)         │
│  • One agent per physical GPU - owns the hardware                    │
│  • Dual-cycle loop: 30s external (heartbeat/claim) + 5s internal    │
│  • Claim tasks from queue within VRAM budget                        │
│  • Spawn short-lived worker subprocesses per task                   │
│  • Monitor GPU resources (temp, VRAM, power) via nvidia-smi         │
│  • Hot/Cold state: LLM loaded vs empty GPU                          │
│  • Self-regulate task claiming based on resource health             │
│  • Handle brain signals: stop, abort, kill                          │
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
│  GPU AGENTS (Executors)                                                 │
│                                                                         │
│  Handles: Task claiming, worker management, resource monitoring         │
│  Actions:                                                               │
│    • Claim tasks within VRAM budget, spawn worker subprocesses          │
│    • Monitor GPU health (temp, VRAM, power) every 5s internal cycle    │
│    • Back off from GPU tasks when resource-constrained                  │
│    • Handle meta tasks directly (load/unload LLM)                      │
│    • Respond to brain signals (stop, abort, kill)                      │
│  Escalates: All failures go to brain (GPU agents don't retry)           │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key principle:** Intelligence matches responsibility. GPU agents are simple - they claim, spawn, and report. Brain has enough intelligence to diagnose and fix task-level issues. Claude has the full context to fix plan-level problems.

| Failure Type | Who Handles | Example |
|--------------|-------------|---------|
| Command fails | Brain retries | Script exits non-zero, timeout |
| Missing task field | Brain infers | `task_class` missing, brain infers from command |
| GPU agent dies | Brain detects | Heartbeat stale, tasks stuck in processing |
| Stuck task (>20 min) | Brain escalates | Abort signal -> force kill -> manual |
| Output poor quality | Brain re-queues | Evaluation score < 3 |
| Plan malformed | Claude (future) | Missing sections, invalid structure |
| Script bug | Claude (future) | Repeated failures across retries |

---

## Hardware

Hardware is auto-discovered by `setup.py` and stored in `config.json`. The system adapts to whatever GPUs are available:

| Mode | GPUs | Brain | Workers |
|------|------|-------|---------|
| `cpu_only` | 0 | CPU | None |
| `single_gpu` | 1 | Gets the GPU | None |
| `minimal` | 2 | Largest GPU | Other GPU |
| `standard` | 3+ | Largest GPU(s) | All remaining |

Run `python setup.py` on any rig to scan hardware and generate `config.json`. See `config.template.json` for the schema.

Models are stored on the shared drive and loaded into Ollama on demand. See [NETWORK_SETUP.md](NETWORK_SETUP.md) and [systems_prep.md](systems_prep.md) for rig-specific details.

---

## Physical Architecture

```
┌──────────────────────────────────┐     ethernet     ┌──────────────────────────────────┐
│       Control Plane              │                  │           GPU Rig                │
│  • Internet access               │◄────────────────►│  • Air-gapped (no internet)      │
│  • Runs cloud LLM (Claude Code)  │                  │  • NVIDIA GPUs (per config.json) │
│  • Prepares plans                │                  │  • Runs brain + workers          │
│  • Submits to shared folder      │                  │  • Executes plans                │
│  • GitHub for version control    │                  │  • Ollama for local LLM inference │
└──────────────────────────────────┘                  └──────────────────────────────────┘
           │                                                      │
           │  USB                                                 │ NFS mount
           ▼                                                      │
┌──────────────────────────────────┐                              │
│     Shared Drive (ext4)          │◄─────────────────────────────┘
│  Mounted on control plane        │
│  bind-mounted → repo/shared      │
│  NFS-exported to GPU rig         │
└──────────────────────────────────┘
```

**Communication is file-based only.** No network APIs between machines. The shared drive is the sole communication channel.

---

## File Structure

**Key insight:** The `shared/` folder lives on an external drive, bind-mounted into the git repo on the control plane, and NFS-exported to the GPU rig.

```
llm_orchestration/                 # Git repo on RPi (~/llm_orchestration)
├── .claude/settings.json          # Claude Code permissions (full access, core/ denied)
├── .gitignore                     # Selective sync rules
├── requirements.txt               # Python dependencies
│
├── scripts/                       # RPi-only utilities
│   ├── submit.py                  # Submit a plan for execution
│   ├── status.py                  # Check system status
│   ├── watch.py                   # Live monitoring
│   └── gpu-monitor.py             # GPU benchmarking tools
│
└── shared/                        # External drive, bind-mounted here
    │                              # NFS-shared to GPU rig via ethernet
    │
    ├── agents/                    # Agent code (GPU rig runs these)
    │   ├── brain.py               # Brain coordinator
    │   ├── gpu.py                 # GPU agent (one per physical GPU)
    │   ├── worker.py              # Worker subprocess (spawned by gpu.py)
    │   ├── executor.py            # Permission-aware command executor
    │   ├── launch.py              # Launcher script (brain + GPU agents)
    │   ├── config.json            # GPU rig config (not in git)
    │   └── permissions/           # Agent permission rules (not in git)
    │       ├── brain.json         # Brain: full access except core/
    │       └── worker.json        # Worker: full access except core/
    │
    ├── core/                      # PROTECTED - root-owned, read-only (not in git)
    │   ├── SYSTEM.md              # Agent system prompt (read on startup)
    │   ├── RULES.md               # 13 immutable rules
    │   └── ESCALATION_POLICY.md   # Worker -> Brain -> Cloud -> Human escalation
    │
    ├── workspace/                 # Agent-writable working area (synced to git)
    │   ├── CONTEXT.md             # Project context (read this first)
    │   ├── architecture.md        # This file
    │   ├── implement/             # Priority tasks (do these first)
    │   ├── future/                # Ideas parking lot
    │   ├── human/                 # Stuck items needing human review
    │   └── archive/               # Completed work and historical lessons
    │
    ├── plans/                     # Plan folders (synced to git)
    │   ├── PLAN_FORMAT.md         # Plan specification
    │   └── <plan_name>/
    │       ├── plan.md            # Plan definition
    │       ├── scripts/           # Plan scripts
    │       └── history/           # Batch execution runs
    │
    ├── tasks/                     # Task queue - runtime only (not in git)
    │   ├── queue/                 # Ready for workers to claim
    │   ├── processing/            # Being worked on
    │   ├── complete/              # Finished (with result)
    │   └── failed/                # Failed (for retry logic)
    │
    ├── brain/                     # Brain state - runtime only (not in git)
    │   ├── state.json             # Active batches, status
    │   └── private_tasks/         # Tasks waiting for dependencies
    │
    ├── gpus/                      # GPU agent heartbeats (not in git)
    ├── signals/                   # GPU agent control signals (not in git)
    └── logs/                      # Logs (synced to git for backup)
```

---

## Task Flow

```
1. Claude writes plan.md with tasks and dependencies
2. User submits: python scripts/submit.py <plan_name> --config '{...}'
3. Brain parses plan.md directly (no LLM needed for parsing)
4. Brain creates all tasks, stores in private_tasks/
5. Brain releases tasks with no dependencies to queue/
6. GPU agents claim tasks from queue (within VRAM budget)
7. GPU agents spawn worker subprocesses to execute each task
8. Workers print result JSON to stdout, GPU agent collects and writes to complete/
9. Brain checks completed tasks, releases dependent tasks
10. Brain auto-inserts batch_summary task (depends on all others)
11. When all tasks complete, batch is done
```

**Dependency model:** Tasks specify `depends_on: [task1, task2]`. Brain releases a task to the public queue only when all its dependencies have completed successfully. This is like a Gantt chart - flexible ordering based on actual dependencies, not rigid phases.

---

## Plan Format

Plans are markdown files that tell the brain what to do. See [PLAN_FORMAT.md](PLAN_FORMAT.md) for the complete specification.

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
| `meta` | No | No | Worker management (load/unload LLM) |

### Resource-Aware Task Claiming

GPU agents adjust task claiming based on **two independent factors**:

#### Factor 1: GPU Health (Resource Constraints)

| GPU Health | Thresholds (per config.json) | Behavior |
|------------|-----------|----------|
| **Healthy** | Below all resource_limits thresholds | Normal task claiming (see Factor 2) |
| **Constrained** | Any threshold exceeded (temp, VRAM, or power) | Only accept `meta`, `cpu` tasks |

When constrained, GPU agents:
- Still accept `meta` tasks (brain can tell them to unload LLM to free VRAM)
- Still accept `cpu` tasks (no GPU usage)
- Reject `llm` and `script` tasks (would worsen resource pressure)
- Auto-recover when resources improve (checked every 5s internal cycle)

Hot GPUs naturally back off from heavy work, then resume when cooled.

#### Factor 2: Model State (Hot vs Cold)

GPU agents track whether their LLM is loaded in VRAM:

| State | LLM Loaded? | Can Handle (when healthy) | Use Case |
|-------|-------------|---------------------------|----------|
| **Hot** | Yes | `meta`, `llm`, `cpu` | LLM loaded - card full, no room for script tasks |
| **Cold** | No | `meta`, `script`, `cpu`, `llm` | GPU compute available - can load LLM or run scripts |

**Key principle:** When an LLM is loaded, the GPU card is considered FULL. Script tasks (like Whisper) need VRAM that the LLM is occupying, so hot GPU agents skip script tasks entirely.

#### Combined Behavior Matrix

How both factors interact to determine task claiming:

| GPU Health | Model State | Tasks Claimed | Why |
|------------|-------------|---------------|-----|
| Healthy | Hot | `meta`, `llm`, `cpu` | LLM loaded - no room for scripts |
| Healthy | Cold | `meta`, `script`, `cpu`, `llm` | GPU free - can do anything |
| Constrained | Hot | `meta`, `cpu` | Overloaded - back off all GPU work |
| Constrained | Cold | `meta`, `cpu` | Overloaded - back off all GPU work |

**Startup behavior:**
- GPU agents start per `config.json` (one per worker GPU)
- All start Cold (no LLM loaded)
- Brain manages LLM loading on demand via meta tasks
- GPU agents are immediately available for `script` and `cpu` tasks

**Dynamic transitions:**
- **Heat up**: Brain sends `load_llm` meta task when `llm` tasks are queued
- **Cool down**: Brain sends `unload_llm` meta task when only `script` tasks remain
- **Resource recovery**: Brain can send `unload_llm` to overheated GPU agents to free VRAM

**Why this matters:**
- `script` tasks (like Whisper transcription) don't need LLM - cold GPU agents handle them immediately
- Avoids loading LLMs on all GPUs when most work is GPU compute
- Brain can heat up GPU agents on demand when `llm` tasks appear
- Brain can help overloaded GPU agents recover by unloading their models

---

## GPU Agent Self-Awareness

GPU agents monitor their own GPU resources and adjust behavior to prevent overheating, OOM errors, and power issues.

### Monitored Resources

Every internal cycle (every 5 seconds), GPU agents query nvidia-smi for:
- **Temperature** (C) - Prevents thermal throttling
- **VRAM usage** (MB and %) - Prevents OOM crashes
- **Power draw** (W) - Stays under PSU limits
- **GPU utilization** (%) - General health metric
- **Clock speed** (MHz) - Detects throttling
- **Throttle status** - Thermal/power/sync flags

Per-worker VRAM tracking is also done via nvidia-smi `--query-compute-apps` to get per-PID memory usage.

### Resource Thresholds

Configured in `config.json` under `resource_limits`:
```json
"resource_limits": {
  "max_temp_c": 80,
  "gpu_temp_warning_c": 75,
  "gpu_temp_critical_c": 90,
  "max_vram_percent": 95,
  "max_power_w": 140
}
```
Adjust these values based on your hardware's thermal and power characteristics.

### Self-Regulation Behavior

When any threshold is exceeded, GPU agent automatically:
1. Logs warning with specific constraint reasons
2. Switches to claiming only `meta` and `cpu` tasks
3. Rejects `llm` and `script` tasks (GPU-intensive)
4. Continues internal monitoring cycle
5. Auto-recovers when resources drop below thresholds

**Example constraint scenarios:**
- A GPU exceeds `max_temp_c` -> Only claims meta/cpu tasks until temp drops
- A GPU VRAM exceeds `max_vram_percent` -> Only claims meta/cpu tasks, can accept brain's unload_llm command
- A GPU power exceeds `max_power_w` -> Backs off GPU work until power draw decreases

### Heartbeat Format

Each GPU agent is the sole owner of its heartbeat file (no FileLock needed):

```
shared/gpus/gpu_<id>/heartbeat.json
```

```json
{
  "gpu_id": 1,
  "name": "gpu-1",
  "state": "cold",
  "model_loaded": false,
  "last_updated": "2026-02-07T10:05:00",
  "temperature_c": 78,
  "power_draw_w": 125.3,
  "vram_used_mb": 4200,
  "vram_total_mb": 6144,
  "vram_percent": 68,
  "gpu_util_percent": 85,
  "clock_mhz": 1950,
  "throttle_status": "None",
  "claimed_vram_mb": 2048,
  "budget_available_mb": 2867,
  "active_workers": 2,
  "active_tasks": [
    {
      "worker_id": "gpu-1-w0-d5de16d7",
      "task_id": "d5de16d7",
      "task_class": "script",
      "task_name": "transcribe",
      "vram_estimate_mb": 1024,
      "peak_vram_mb": 800,
      "pid": 12345,
      "started_at": "2026-02-07T10:04:30"
    }
  ],
  "stats": {
    "tasks_completed": 5,
    "tasks_failed": 0
  }
}
```

Brain and monitoring tools read these heartbeats to understand system health, GPU agent state, and active tasks.

---

## GPU Agent Architecture

Each GPU agent owns one physical GPU and manages concurrent worker subprocesses:

| Config Field | Example | Purpose |
|-------------|---------|---------|
| `id` | 1 | Physical GPU index (CUDA_VISIBLE_DEVICES) |
| `name` | gpu-1 | Agent name for logging and signals |
| `model` | (per config) | LLM model (for hot state) |
| `port` | 11435 | Dedicated Ollama port |

**Key boundary:** A GPU is either:
- **Hot** (LLM loaded): Claims meta, llm, cpu tasks. LLM fills VRAM so no script tasks.
- **Cold** (no LLM): Claims meta, script, cpu, llm tasks. Can run multiple script workers concurrently within VRAM budget.

### VRAM Budget System

The GPU agent tracks VRAM budget internally (no external coordination needed):

1. Total budget = `vram_mb` from config x 80%
2. Each claimed task has a VRAM cost estimate
3. GPU agent tracks `claimed_vram` as sum of active workers' estimates
4. New tasks are only claimed if their cost fits within remaining budget
5. When a worker subprocess finishes, its VRAM estimate is released

| Task Class | VRAM Cost |
|-----------|-----------|
| `llm` | Full GPU VRAM - one task at a time |
| `script` | Per-task `vram_estimate_mb` field, default 1024 MB |
| `cpu` | Virtual cost of 1024 MB (limits concurrency) |
| `meta` | 0 MB (handled directly by GPU agent, no subprocess) |

### Dual-Cycle Loop

GPU agents run two nested cycles for efficient I/O:

| Cycle | Interval | Actions |
|-------|----------|---------|
| **Internal** | 5 seconds | Check signals, collect finished workers, update VRAM tracking |
| **External** | 30 seconds | Flush results to filesystem, write heartbeat, claim new tasks |

The external cycle also triggers immediately when all workers finish and results are pending (no waiting for next 30s tick).

### Worker Subprocesses

Workers are short-lived subprocesses, not long-running agents:

```
GPU Agent (gpu.py)           Worker Subprocess (worker.py)
  |                             |
  |-- spawns subprocess ------->|
  |                             |-- executes task
  |-- polls proc.poll() -----  |-- prints JSON to stdout
  |                             |-- exits
  |<-- reads stdout JSON -------|
  |-- writes result to outbox
  |-- releases VRAM budget
```

Each worker subprocess:
- Receives task JSON as a command-line argument
- Uses `PermissionExecutor` for sandboxed shell execution
- Sends LLM requests to parent GPU agent's Ollama port
- Logs training samples for future fine-tuning
- Prints result as JSON to stdout (GPU agent reads this)
- Exits immediately after completing its single task

---

## Logging System

The system uses independent logging levels for each component, allowing targeted debugging without noise from other components.

### Log Level Control

Each component has its own environment variable:
- **GPU_LOG_LEVEL** - GPU agent loop, claiming, worker management (default: INFO)
- **EXECUTOR_LOG_LEVEL** - Permission checking in worker subprocesses (default: INFO)
- **WORKER_LOG_LEVEL** - Worker subprocess task execution (default: INFO)
- **BRAIN_LOG_LEVEL** - Coordination and decisions (default: INFO)

### Log Levels by Component

| Component | INFO | WARNING | ERROR | DEBUG |
|-----------|------|---------|-------|-------|
| **GPU Agent** | Lifecycle, claims, worker spawn/finish, cycles | Resource constraints | Crashes, signal errors | VRAM tracking, queue visibility |
| **Executor** | BLOCKED/NEEDS_APPROVAL actions | Permission failures | Fatal errors | All ALLOWED actions |
| **Worker** | Task start/finish | Execution issues | Parse errors | Training sample details |
| **Brain** | Decisions, lifecycle, monitoring | Stuck tasks, missing GPUs | Fatal errors, crashes | Dependencies, task analysis |

### Usage Examples

```bash
# Debug GPU agent task claiming and VRAM budget
GPU_LOG_LEVEL=DEBUG python shared/agents/gpu.py gpu-1

# Debug brain decision-making
BRAIN_LOG_LEVEL=DEBUG python shared/agents/brain.py

# Debug permission issues in worker subprocesses
EXECUTOR_LOG_LEVEL=DEBUG python shared/agents/gpu.py gpu-1
```

See [archive/LOG_IMPROVEMENTS.md](archive/LOG_IMPROVEMENTS.md) for detailed logging documentation.

---

## Signal System

Brain communicates with GPU agents via signal files in `shared/signals/`:

| Signal | File Pattern | Purpose | GPU Agent Response |
|--------|-------------|---------|-------------------|
| **stop** | `gpu-1.stop` | Graceful shutdown | Finish current work, flush results, exit |
| **abort** | `gpu-1.abort` | Cancel specific task | Terminate the worker subprocess for that task |
| **kill** | `gpu-1.kill` | Force kill specific task | SIGKILL the worker subprocess (after abort fails) |

### Stuck Task Escalation

Brain detects tasks stuck in `processing/` for more than 20 minutes and escalates:

1. **First detection** -- Send abort signal to GPU agent (graceful terminate)
2. **After 2 minutes** -- Abort signal still present, escalate to force kill signal
3. **After force kill** -- If still stuck, log for manual intervention

This prevents runaway tasks from blocking the system indefinitely.

---

## Foreach Task Expansion

Plans can define template tasks that expand into N individual tasks at runtime:

```markdown
### transcribe
- **executor**: worker
- **task_class**: script
- **command**: `python scripts/transcribe.py {ITEM.path} --output {BATCH_PATH}/results/`
- **depends_on**: scan
- **foreach**: {BATCH_PATH}/manifest.json:videos
```

When the `scan` task completes and produces `manifest.json`, the brain:
1. Reads the JSON file at the specified path
2. Navigates to the `videos` array
3. Creates one task per array item (e.g., `transcribe_v1`, `transcribe_v2`, ...)
4. Substitutes `{ITEM.field}` placeholders with item values
5. Updates downstream tasks to depend on ALL expanded tasks

This allows plans to handle variable-length input (e.g., N video files) without knowing the count at plan-writing time.

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
| Brain/worker split | Larger model for coordination, smaller for throughput |
| Cloud LLM as planner | Smart work upfront, simpler execution locally |
| Private/public task lists | Brain controls sequencing, workers stay simple |
| Dynamic hardware discovery | `setup.py` adapts to any GPU configuration |

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
- GPU agents don't know which plan a task belongs to - they just claim and execute
- Each batch's output stays isolated in its own `history/{batch_id}/` folder

### Graceful Recovery

The system survives restarts, crashes, and power loss:

| State | Location | On Restart |
|-------|----------|------------|
| Active batches | `brain/state.json` | Brain reloads and continues monitoring |
| Pending tasks | `brain/private_tasks/` | Released when dependencies met |
| Ready tasks | `tasks/queue/` | Workers claim them |
| In-progress tasks | `tasks/processing/` | Brain detects stuck tasks (>20 min), sends abort/kill signals |
| Completed tasks | `tasks/complete/` | Already done, triggers dependency release |
| Failed tasks | `tasks/failed/` | Brain retries or leaves for review |

**Recovery flow:**
1. Brain starts, loads `state.json` → knows which batches were active
2. Scans `private_tasks/` → finds unreleased tasks
3. Scans `complete/` → knows what finished while down
4. Releases newly-unblocked tasks to queue
5. Detects stuck tasks in `processing/` (>20 min) -> sends abort/kill signals
6. Resumes normal loop

**Task state is never lost.** If a GPU agent crashes mid-task:
- The task stays in `processing/`
- Brain detects it as stuck (>20 min in processing)
- Brain sends abort/kill signals, task eventually returns to `failed/`
- Brain's retry logic moves it back to `queue/` (up to 3 attempts)
- Another GPU agent picks it up and starts fresh

The *work* on that specific task run is lost, but the task naturally retries. This is why scripts should be **idempotent** - safe to restart from scratch if interrupted.

---

## Related Docs

| Doc | Purpose |
|-----|---------|
| [CONTEXT.md](CONTEXT.md) | Project entry point, quick orientation |
| [brain-behavior.md](brain-behavior.md) | Brain loop, task handling, failure recovery |
| [PLAN_FORMAT.md](PLAN_FORMAT.md) | How to write plans |
| [future/resource_manager_design.md](future/resource_manager_design.md) | Future: GPU resource management design |

---

## Security Model

The system uses defense-in-depth to protect core agent instructions:

| Layer | Protection |
|-------|-----------|
| **Filesystem** | `core/` is root-owned (chmod 644) - agent process can read but not write |
| **Executor** | Permission system blocks writes to `**/core/**` in both brain.json and worker.json |
| **Bash patterns** | Blocked: chmod/chown on core, sudo, SYSTEM.md/RULES.md references, .ssh, passwd |
| **Git hook** | Pre-commit rejects changes to `shared/core/` (only `--no-verify` from terminal) |
| **Workspace isolation** | Agents write freely to workspace/ but cannot touch core/ or permissions/ |

**Escalation pattern:** When agents encounter problems they can't solve after 3+ attempts, they write a `HUMAN_{topic}.md` file to `workspace/human/` and stop working on that issue.

---

*Last Updated: February 2026*
