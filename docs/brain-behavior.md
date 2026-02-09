# Brain Behavior

Implementation details for the brain agent (`shared/agents/brain.py`). For high-level architecture, see [architecture.md](architecture.md).

---

## Overview

The brain is the central coordinator. It:
- Parses plan.md and creates tasks with dependencies
- Releases tasks when their dependencies complete
- Monitors worker health and output quality
- Handles failures and retries

---

## The Brain Loop

```python
while running:
    # 1. Check for new plans
    claim_pending_plans()

    # 2. Check completed tasks, release dependent tasks
    check_and_release_tasks()

    # 3. Check worker health
    check_worker_heartbeats()

    # 4. Handle failures
    process_failed_tasks()

    # 5. Monitor system resources
    monitor_system()

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
    """
    tasks = []

    # Split by ### to get task sections
    sections = re.split(r'\n### ', plan_content)

    for section in sections[1:]:  # Skip header before first ###
        lines = section.strip().split('\n')
        task_id = lines[0].strip()
        task = {"id": task_id, "executor": "worker", "task_class": None, "command": "", "depends_on": []}

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

        if task["command"]:
            tasks.append(task)

    return tasks
```

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

## Worker Monitoring

### Heartbeats

Workers write heartbeat files every 30 seconds:

```
shared/workers/{worker_id}/
  heartbeat.json    # Last check-in time
  status.json       # Current state
```

```json
{
  "worker_id": "worker-1",
  "gpu_id": 1,
  "timestamp": "2026-02-07T10:05:00",
  "status": "busy",
  "current_task": "task-uuid",
  "gpu_memory_used_mb": 4200
}
```

### Health Checks

```python
def check_worker_health():
    """Detect unhealthy workers."""
    for worker_id in get_workers():
        heartbeat = read_heartbeat(worker_id)
        age = now() - heartbeat["timestamp"]

        if age > timedelta(minutes=2):
            handle_dead_worker(worker_id)
```

### Handling Dead Workers

```python
def handle_dead_worker(worker_id: str):
    """Recover from worker failure."""
    # Find task assigned to this worker
    task = find_task_by_worker(worker_id)

    if task:
        # Reset task to pending
        task["status"] = "pending"
        task["assigned_to"] = None
        task["retry_count"] = task.get("retry_count", 0) + 1
        save_task(task)

        log_decision("WORKER_FAILED", f"{worker_id} offline, task reset")
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

```python
MAX_RETRIES = 3

def handle_worker_failure(task: dict):
    """Retry worker failures up to MAX_RETRIES."""
    retries = task.get("retry_count", 0)

    if retries < MAX_RETRIES:
        task["retry_count"] = retries + 1
        task["status"] = "pending"
        save_to_queue(task)
        log_decision("RETRY", f"Task retry {retries + 1}/{MAX_RETRIES}")
    else:
        log_decision("ABANDON", f"Task abandoned after {MAX_RETRIES} retries")
```

### Escalation to Claude (Future)

Some failures are beyond the brain's ability to fix. These should escalate to Claude for replanning:

| Escalate When | Example | Future Action |
|---------------|---------|---------------|
| Same task fails on 3 workers | Script bug, missing dependency | Claude rewrites script |
| Plan parsing fails | Malformed plan structure | Claude rewrites plan |
| Definition error unfixable | Can't infer required field | Claude fixes plan |
| All workers unhealthy | System problem | Alert human |

**Current behavior:** Plan-level failures are terminal. The batch fails and requires manual intervention.

**Future behavior:** Brain writes escalation request to `shared/brain/escalations/`. RPi gateway picks it up and sends to Claude for replanning. Claude submits corrected plan.

---

## Brain State

The brain persists its state:

```
shared/brain/
  state.json          # Active batches, status
  private_tasks/      # Tasks not yet released
  escalations/        # Pending escalations
```

### state.json

```json
{
  "pid": 12345,
  "started_at": "2026-02-07T10:00:00",
  "status": "active",
  "current_batch": "abc123",
  "current_plan": "batch_processor",
  "stats": {
    "tasks_total": 50,
    "tasks_complete": 32,
    "tasks_failed": 1,
    "tasks_pending": 17
  }
}
```

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
| 10 | Urgent (health checks, resource management) |
| 7 | Resource tasks (load/unload LLM) |
| 5 | Normal work |
| 3 | Background/batch |
| 1 | Low priority cleanup |

---

## Resource Management

The brain monitors workers and controls LLM model loading/unloading.

### Worker Heartbeats

Workers write heartbeat files every poll cycle:

```
shared/workers/{worker_name}/heartbeat.json
```

```json
{
  "worker_id": "worker-1",
  "gpu_id": 1,
  "timestamp": "2026-02-07T10:05:00",
  "model_loaded": true,
  "model": "qwen2.5:7b",
  "port": 11435
}
```

### Model Load/Unload Decisions

The brain monitors the queue and worker states:

1. **Need to load LLM?**
   - If `llm` tasks are waiting AND no workers have models loaded
   - Brain inserts a `load_llm` resource task

2. **Need to unload LLM?**
   - If ONLY `script` tasks are waiting AND some workers have models loaded
   - Brain inserts an `unload_llm` resource task

Workers claim and execute these resource tasks like any other task. This centralizes all resource decisions in the brain.

### Task Classes

| Class | Who Claims | Description |
|-------|------------|-------------|
| `cpu` | Any worker | CPU-only tasks |
| `script` | GPU workers | GPU compute without LLM |
| `llm` | GPU workers | Needs LLM model loaded |
| `resource` | GPU workers | Model load/unload (brain-inserted) |

See [future/resource_manager_design.md](future/resource_manager_design.md) for future enhancements.

---

## Related Docs

| Doc | Purpose |
|-----|---------|
| [architecture.md](architecture.md) | High-level system overview |
| [PLAN_FORMAT.md](../shared/plans/PLAN_FORMAT.md) | How to write plans |
| [resource_manager_design.md](resource_manager_design.md) | Future GPU management |

---

*Last Updated: February 2026*
