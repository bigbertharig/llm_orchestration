# Brain Behavior

Implementation details for the brain agent (`shared/agents/brain.py`). For high-level architecture, see [architecture.md](architecture.md).

---

## Overview

The brain is the central coordinator. It:
- Parses plan.md and creates tasks with dependencies
- Releases tasks when their dependencies complete
- Monitors GPU agent health and output quality
- Handles failures and retries
- Manages LLM loading/unloading via meta tasks
- Detects stuck tasks with escalating intervention (abort -> kill)

---

## The Brain Loop

```python
while running:
    # 1. Check for brain tasks (execute_plan, decide, brain-targeted shell)
    claim_brain_tasks()

    # 2. Check completed tasks, release dependent tasks
    check_and_release_tasks()

    # 3. Handle failed tasks (retry logic, definition error fixes)
    handle_failed_tasks()

    # 4. Monitor system (GPU agents, stuck tasks, resource decisions)
    monitor_system()

    # 5. Persist brain state
    save_brain_state()

    sleep(poll_interval)
```

---

## Private and Public Task Lists

*Implemented in `shared/agents/brain.py`*

The brain maintains two task lists to handle dependencies:

| List | Location | Purpose |
|------|----------|---------|
| **Private** | `shared/brain/private_tasks/` | Tasks waiting for dependencies |
| **Public** | `shared/tasks/queue/` | Tasks ready for workers |

### How It Works

1. Brain reads plan, parses the `## Tasks` section directly
2. Each task specifies its dependencies via `depends_on`
3. All tasks start in the private list
4. Brain releases tasks when their dependencies are met:
   - Tasks with `depends_on: none` are released immediately
   - Other tasks wait until all their dependencies complete
5. Workers only see the public queue

### Why This Design

- Workers stay simple - just grab and execute
- Brain controls all sequencing logic
- Gantt-chart style flexibility - no rigid phases
- Clear visibility into what's waiting vs ready

---

## Dependency-Based Task Release

*Implemented in `shared/agents/brain.py`*

Tasks specify their dependencies explicitly:

```markdown
### aggregate
- **executor**: worker
- **task_class**: cpu
- **command**: `python scripts/combine.py`
- **depends_on**: process
```

### Dependency Checking

```python
def check_and_release_tasks(self):
    """Release tasks whose dependencies are now met."""
    # Get completed task names
    completed = set()
    for task_file in (self.tasks_path / "complete").glob("*.json"):
        task = json.loads(task_file.read_text())
        completed.add(task.get("name", task["task_id"]))

    # Check each private task
    for task in self.get_private_tasks():
        depends_on = task.get("depends_on", [])

        # Check if all dependencies are complete
        deps_met = all(dep in completed for dep in depends_on)

        if deps_met:
            # Release to public queue
            self.save_to_public(task)
```

### Batch Completion

A batch is complete when:
- No private tasks remain (all released)
- No public/processing tasks remain (all completed or failed)

---

## Task Creation

The brain parses the plan's `## Tasks` section directly (no LLM needed):

```python
def parse_plan_md(self, plan_content: str) -> List[Dict]:
    """
    Parse a structured plan.md file into task definitions.

    Expected format for each task:
    ### task_id
    - **executor**: brain|worker
    - **task_class**: cpu|script|llm
    - **command**: `shell command here`
    - **depends_on**: task1, task2
    - **foreach**: {BATCH_PATH}/manifest.json:videos  (optional)
    """
    tasks = []

    # Split by ### to get task sections
    sections = re.split(r'\n### ', plan_content)

    for section in sections[1:]:  # Skip header before first ###
        lines = section.strip().split('\n')
        task_id = lines[0].strip()
        task = {"id": task_id, "executor": "worker", "task_class": None,
                "command": "", "depends_on": [], "foreach": None}

        for line in lines[1:]:
            if line.startswith('- **executor**:'):
                task["executor"] = line.split(':', 1)[1].strip()
            elif line.startswith('- **task_class**:'):
                task["task_class"] = line.split(':', 1)[1].strip().lower()
            elif line.startswith('- **command**:'):
                match = re.search(r'`([^`]+)`', line)
                if match:
                    task["command"] = match.group(1)
            elif line.startswith('- **depends_on**:'):
                deps = line.split(':', 1)[1].strip()
                if deps.lower() != 'none':
                    task["depends_on"] = [d.strip() for d in deps.split(',')]
            elif line.startswith('- **foreach**:'):
                task["foreach"] = line.split(':', 1)[1].strip()

        if task["command"]:
            tasks.append(task)

    return tasks
```

**Variable substitution:** Commands support `{BATCH_ID}`, `{PLAN_PATH}`, `{BATCH_PATH}` placeholders, plus any config overrides passed at submission time.

**Auto summary task:** Brain automatically inserts a `batch_summary` task that depends on all other tasks in the batch. This runs `generate_batch_summary.py` to create a batch report.

### Task Schema

```json
{
  "task_id": "uuid",
  "batch_id": "abc123",
  "name": "process",
  "type": "shell",
  "command": "source ~/ml-env/bin/activate && python ...",
  "priority": 5,
  "status": "pending",
  "executor": "worker",
  "task_class": "cpu",
  "depends_on": ["init"],
  "assigned_to": null,
  "created_at": "2026-02-07T10:00:00"
}
```

---

## GPU Agent Monitoring

### Heartbeats

Each GPU agent is the sole owner of its heartbeat file (no FileLock needed):

```
shared/gpus/gpu_<id>/heartbeat.json
```

The heartbeat contains GPU stats, VRAM budget, and active worker subprocess info:

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
  "claimed_vram_mb": 2048,
  "budget_available_mb": 2867,
  "active_workers": 1,
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
  "stats": {"tasks_completed": 5, "tasks_failed": 0}
}
```

The brain reads these to understand GPU agent health, model state, and task assignment.

### Health Checks

The brain uses two methods to detect running GPU agents:

```python
def _get_running_gpus():
    """Check which GPU agents are running."""
    for gpu_name in gpu_agents:
        # Method 1: Check process via pgrep
        result = subprocess.run(["pgrep", "-f", f"gpu.py.*{gpu_name}"], ...)

        # Method 2: Check heartbeat freshness (fallback)
        if heartbeat age < 60 seconds:
            running[gpu_name] = ...
```

### Stuck Task Detection

Brain detects tasks stuck in `processing/` for more than 20 minutes:

```python
def _detect_stuck_tasks():
    """Find tasks in processing state for too long."""
    for task_file in processing_path.glob("*.json"):
        elapsed = now() - task["started_at"]
        if elapsed > 1200:  # 20 minutes
            stuck_tasks.append(task_info)
```

### Stuck Task Escalation

When stuck tasks are found, brain escalates through 3 levels:

```python
def _handle_stuck_tasks(stuck_tasks):
    for task in stuck_tasks:
        if kill_signal_exists:
            # Level 3: Already force-killed, needs manual intervention
            logger.error("Task still stuck after force kill")
        elif abort_signal_exists and abort_age > 120:
            # Level 2: Abort failed, escalate to force kill
            _force_kill_worker_task(worker, task_id)
        else:
            # Level 1: First detection, send graceful abort
            _send_abort_signal(worker, task_id, reason)
```

### Missing GPU Agent Detection

Brain uses a 3-consecutive-miss tolerance to avoid false positives:

```python
# Track consecutive misses per GPU agent
for gpu_name in gpu_agents:
    if gpu_name not in running:
        gpu_miss_count[gpu_name] += 1
    else:
        gpu_miss_count[gpu_name] = 0

# Only report after 3 consecutive misses
truly_missing = [g for g in missing if gpu_miss_count[g] >= 3]
```

---

## Output Evaluation

The brain evaluates worker output quality:

```python
def evaluate_output(task: dict, result: dict) -> dict:
    """Brain evaluates worker output."""

    if not result.get("success"):
        return {"acceptable": False, "rating": 1, "retry": True}

    prompt = f"""Evaluate this output:

Task: {task["command"]}
Output: {result["output"][:1000]}

Rate 1-5 and explain issues.
Return JSON: {{"acceptable": bool, "rating": int, "issues": [], "feedback": str}}
"""

    response = think(prompt)
    evaluation = parse_json(response)

    return evaluation
```

### Rating Scale

| Rating | Meaning | Action |
|--------|---------|--------|
| 5 | Perfect | Accept, use for training |
| 4 | Good | Accept |
| 3 | Acceptable | Accept with note |
| 2 | Poor | Retry if attempts remain |
| 1 | Failed | Retry or abandon |

---

## Failure Handling

The brain is responsible for handling task-level failures. This is part of the tiered intelligence model - workers just execute and report, brain diagnoses and fixes. See [architecture.md](architecture.md#error-handling--escalation) for the full escalation model.

The brain distinguishes between two types of failures:

| Error Type | Cause | Brain's Response |
|------------|-------|------------------|
| `worker` | Worker crashed, timeout, script error | Retry up to 3 times |
| `definition` | Missing/invalid required fields in task | Infer and fix if possible |

### Definition Errors

When a task has a definition issue (e.g., missing `task_class`), the brain uses its intelligence to fix it:

1. **Detects the error** during task creation
2. **Sends task to `failed/`** immediately with `error_type: "definition"`
3. **Attempts auto-fix** on next `handle_failed_tasks()` cycle
4. **Re-queues if fixed**, or leaves in `failed/` for manual intervention

```python
def _try_fix_definition_error(task: dict) -> bool:
    """Attempt to fix a task with a definition error."""
    error = task.get("definition_error", "")

    if "missing task_class" in error:
        # Infer task_class from command keywords
        command = task.get("command", "").lower()

        if any(kw in command for kw in ["whisper", "transcrib", "embed", "cuda", "gpu"]):
            task["task_class"] = "script"  # GPU compute
        elif any(kw in command for kw in ["ollama", "generate", "llm"]):
            task["task_class"] = "llm"     # Needs LLM model
        else:
            task["task_class"] = "cpu"     # Default

        task["fix_applied"] = f"inferred task_class='{task['task_class']}'"
        return True

    return False
```

**Log entries:**
- `TASK_DEFINITION_ERROR` - Task created with definition error
- `TASK_FIXED` - Definition error fixed, task re-queued
- `TASK_UNFIXABLE` - Could not auto-fix, needs manual intervention

### Worker Failures (Retry)

The brain uses task memory fields (`attempts`, `workers_attempted`) set by the GPU agent when claiming:

```python
max_attempts = config["retry_policy"]["max_attempts"]  # default: 3

def handle_failed_tasks():
    for task in failed_tasks:
        attempts = task.get("attempts", 0)
        workers = task.get("workers_attempted", [])

        if attempts < max_attempts:
            task["status"] = "pending"
            save_to_queue(task)
            log_decision("RETRY", f"Attempt {attempts}/{max_attempts}, workers: {workers}")
        else:
            log_decision("ABANDON", f"Abandoned after {attempts} attempts by {workers}")
```

### Escalation to Cloud Brain (Active)

Some failures are beyond the local brain's ability to fix. These should escalate to cloud brain for replanning:

| Escalate When | Example | Action |
|---------------|---------|---------------|
| Same task fails on 3 workers | Script bug, missing dependency | Cloud brain rewrites script |
| Plan parsing fails | Malformed plan structure | Cloud brain rewrites plan |
| Definition error unfixable | Can't infer required field | Cloud brain fixes plan |
| All workers unhealthy | System problem | Alert human |

**Current behavior:** On `execute_plan` startup failure, local brain writes an escalation request to `shared/brain/escalations/` with directed context (`details.context`) to guide cloud triage.
The failed task result includes `escalated: true` and `escalation_id`.

---

## Brain State

The brain persists its state:

```
shared/brain/
  state.json          # Active batches, status
  private_tasks/      # Tasks not yet released
  escalations/        # Pending cloud escalation requests
```

### state.json

```json
{
  "pid": 12345,
  "started_at": "2026-02-07T10:00:00",
  "status": "active",
  "active_batches": {
    "20260207_100000": {
      "plan": "invoice_processor",
      "plan_dir": "/path/to/shared/plans/invoice_processor",
      "batch_dir": "/path/to/shared/plans/invoice_processor/history/20260207_100000",
      "started_at": "2026-02-07T10:00:00",
      "config": {"INPUT_FOLDER": "/data/invoices"},
      "total_tasks": 5
    }
  }
}
```

Batch IDs are timestamp-based (`YYYYMMDD_HHMMSS`) for chronological ordering. Multiple batches can be active simultaneously.

---

## Singleton Behavior

Only one brain should run at a time:

```python
def start_brain():
    """Start brain, checking for existing instance."""
    state = read_brain_state()

    if state and state["status"] == "active":
        if process_alive(state["pid"]):
            raise RuntimeError("Brain already running")

    # Start brain
    write_brain_state({"pid": os.getpid(), "status": "active", ...})
```

When submitting a new plan to a running brain, queue it:

```python
def submit_plan(plan_name: str, config: dict):
    """Submit plan to brain."""
    if brain_is_running():
        # Queue for existing brain
        queue_plan(plan_name, config)
    else:
        # Start brain with this plan
        start_brain(plan_name, config)
```

---

## Logging

### Decision Log

All brain decisions are logged for monitoring and debugging:

```
shared/logs/brain_decisions.log
```

```json
{"timestamp": "...", "type": "TASK_RELEASED", "message": "Released task: process", "details": {...}}
{"timestamp": "...", "type": "TASK_CREATED", "message": "Created task: init", "details": {...}}
{"timestamp": "...", "type": "BATCH_COMPLETE", "message": "Batch abc123 complete", "details": {...}}
```

### Training Samples

Prompt/response pairs are logged for future fine-tuning:

```
shared/logs/training_samples.jsonl
```

See [architecture.md](architecture.md) for training sample schema.

---

## Priority System

Higher priority = claimed first by workers.

| Priority | Use Case |
|----------|----------|
| 10 | Meta tasks: load_llm, unload_llm (resource management) |
| 5 | Normal work tasks |
| 1 | Auto-inserted batch_summary task (runs last) |

---

## Resource Management

The brain monitors GPU agents and controls LLM model loading/unloading.

### GPU Agent Heartbeats

GPU agents write heartbeat files every 30 seconds (sole owner, no lock needed):

```
shared/gpus/gpu_<id>/heartbeat.json
```

The brain reads these files to get GPU state, model loading status, and active task info. See [architecture.md](architecture.md#heartbeat-format) for the full heartbeat schema.

### Model Load/Unload Decisions

The brain monitors the queue and GPU agent states:

1. **Need to load LLM?**
   - If `llm` tasks are waiting AND no GPU agents have models loaded
   - Brain inserts a `load_llm` meta task
   - Tracks request to detect when GPU agents don't pick it up (warns after 60s)

2. **Need to unload LLM?**
   - If ONLY `script` tasks are waiting AND some GPU agents have models loaded
   - Brain inserts an `unload_llm` meta task

3. **Deduplication:**
   - Brain checks both queue AND processing for existing meta tasks before inserting new ones
   - Prevents duplicate load/unload requests

GPU agents claim and handle meta tasks directly (no worker subprocess needed). This centralizes all resource decisions in the brain.

### Task Classes

| Class | Who Claims | Description |
|-------|------------|-------------|
| `cpu` | Any GPU agent | CPU-only tasks |
| `script` | GPU agents (when healthy + cold) | GPU compute without LLM |
| `llm` | GPU agents (when healthy) | Needs LLM model loaded |
| `meta` | GPU agents (always, handled directly) | Load/unload LLM, no subprocess |

**Resource-aware claiming:** GPU agents automatically back off from `llm` and `script` tasks when resource thresholds are exceeded (per `config.json` `resource_limits`). They still accept `meta` tasks so the brain can help them recover.

---

## Related Docs

| Doc | Purpose |
|-----|---------|
| [architecture.md](architecture.md) | High-level system overview |
| [PLAN_FORMAT.md](PLAN_FORMAT.md) | How to write plans |
| [resource_manager_design.md](resource_manager_design.md) | Future GPU management |

---

*Last Updated: February 2026*
