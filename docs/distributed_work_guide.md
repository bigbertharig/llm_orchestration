# Distributed Work Guide

How work flows through the orchestration system: from Claude planning to worker execution.

---

## The Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. PLANNING (Claude)                                               │
│     - Analyze the goal                                              │
│     - Write plan.md with scripts and tasks                          │
│     - Prepare environment setup in run commands                     │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. SUBMISSION (User)                                               │
│     python scripts/submit.py <plan_folder> --config '{...}'         │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. INTERPRETATION (Brain)                                          │
│     - Read and parse plan.md                                        │
│     - Create all tasks with dependencies                            │
│     - Release tasks when dependencies are met                       │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. EXECUTION (Workers)                                             │
│     - Claim tasks from public queue                                 │
│     - Run shell commands                                            │
│     - Report results                                                │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  5. MONITORING (Brain)                                              │
│     - Check for completed tasks                                     │
│     - Release newly-unblocked tasks                                 │
│     - Handle failures and retries                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Planning with Claude

When you need to process something, discuss with Claude to create a plan.

**You:** "I need to process 100 PDF invoices and extract totals into a spreadsheet"

**Claude:** "I'll create a plan with tasks and dependencies:
1. **scan** - Find all PDFs, create a manifest
2. **extract** - Extract totals (depends on scan)
3. **combine** - Merge into final CSV (depends on extract)

Let me write the plan and scripts..."

Claude then creates:
- `shared/plans/invoice_processor/plan.md`
- `shared/plans/invoice_processor/scripts/scan.py`
- `shared/plans/invoice_processor/scripts/extract.py`
- `shared/plans/invoice_processor/scripts/combine.py`

**For plan format details, see [PLAN_FORMAT.md](../shared/plans/PLAN_FORMAT.md).**

---

## Step 2: Submitting a Plan

```bash
python scripts/submit.py shared/plans/invoice_processor \
  --config '{"INPUT_FOLDER": "/data/invoices", "OUTPUT_FILE": "/data/totals.csv"}'
```

This creates an `execute_plan` task that the brain picks up.

---

## Step 3: Brain Interpretation

The brain reads the plan and manages task execution:

1. **Parses plan.md** - Extracts tasks from the `## Tasks` section
2. **Creates all tasks** - With their dependency relationships
3. **Holds tasks privately** - In `shared/brain/private_tasks/`
4. **Releases tasks** - To public queue when dependencies are met

**For brain behavior details, see [brain-behavior.md](brain-behavior.md).**

---

## Step 4: Worker Execution

Workers are simple - they poll the queue, claim tasks, and execute:

```python
while running:
    task = claim_available_task()
    if task:
        result = execute(task["command"])
        report_result(task, result)
    sleep(poll_interval)
```

Workers don't know about plans or dependencies. They just execute shell commands and report results.

---

## Step 5: Brain Monitoring

The brain continuously monitors execution:

- **Watches for completions** - Checks `shared/tasks/complete/`
- **Releases dependent tasks** - When dependencies are met
- **Handles failures** - Retries up to 3 times, then abandons
- **Logs decisions** - To `shared/logs/brain_decisions.log`

---

## Task Types

| Type | Handler | Example |
|------|---------|---------|
| `execute_plan` | Brain | Read plan, create tasks |
| `shell` | Workers | Run a shell command |
| `decide` | Brain | Complex reasoning |
| `generate` | Workers | Text generation |
| `parse` | Workers | Extract structured data |

Workers also distinguish between:
- **Script tasks** - Direct GPU/CPU compute (transcribe, embed, etc.)
- **LLM tasks** - Need the language model loaded

---

## Monitoring Execution

### Watch brain decisions
```bash
tail -f shared/logs/brain_decisions.log
```

### Check task status
```bash
python scripts/status.py
```

### View failed tasks
```bash
ls shared/tasks/failed/
cat shared/tasks/failed/<task_id>.json
```

### Batch output
```
shared/plans/<plan_name>/batches/<batch_id>/
  manifest.json   # Created by init task
  results/        # Processing output
  output/         # Final aggregated output
  logs/           # Batch-specific logs
```

---

## Best Practices

### Writing Good Plans

1. **Be precise in run commands** - Brain copies them directly
2. **Include full environment setup** - Workers don't assume anything
3. **Use explicit dependencies** - `depends_on` controls ordering
4. **Keep tasks generic** - Don't reference specific workers or GPUs

### Task Sizing

| Size | Approach |
|------|----------|
| Small (1 file) | Single task handles it |
| Medium (10-100 items) | Multiple parallel tasks |
| Large (1000+ items) | Workers claim from manifest file |

### Error Handling

- Scripts should exit non-zero on failure
- Brain retries up to 3 times
- After 3 failures, task is abandoned
- Check `shared/tasks/failed/` for issues

---

## Related Docs

| Doc | Purpose |
|-----|---------|
| [PLAN_FORMAT.md](../shared/plans/PLAN_FORMAT.md) | **Authoritative plan format specification** |
| [architecture.md](architecture.md) | System overview and file structure |
| [brain-behavior.md](brain-behavior.md) | Brain loop and task handling details |

---

*Last Updated: February 2026*
