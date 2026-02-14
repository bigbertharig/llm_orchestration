# Plan Format

Plans are markdown files written by a smart LLM (Claude) to guide the brain in orchestrating work across GPU workers.

---

## Intelligence Hierarchy

| Role | Model | Responsibility |
|------|-------|----------------|
| **Planner** | Claude (external) | Writes plans, prepares scripts, sets up environment |
| **Brain** | Qwen brain model (local, currently qwen2.5:32b) | Interprets plans, creates tasks, monitors workers, evaluates output |
| **Workers** | Qwen 7B (local) | Execute assigned tasks, report results |

**Key principle:** Claude does the smart work upfront. The brain interprets and orchestrates. Workers execute.

---

## What a Plan Contains

A plan tells the brain:

1. **Goal** - What success looks like
2. **Inputs** - Configuration values provided at submission
3. **Available Scripts** - What tools exist and how to use them
4. **Workflow** - The order of operations and dependencies
5. **Outputs** - What will be produced

The brain reads this as an LLM and generates shell tasks. The plan should be precise enough that the brain reliably generates correct commands, but flexible enough that it can make sensible decisions about parallelism, worker assignment, and error handling.

---

## Plan Location

```
shared/plans/<plan_name>/
  plan.md           # The plan template (immutable)
  scripts/          # Scripts the brain can invoke
  lib/              # Shared libraries for scripts (optional)

  history/          # Execution history (all runs, success and failures)
    {batch_id}/
      # Working data (created during execution)
      manifest.json
      logs/
      output/
      results/

      # Auto-generated summary (created at end of run)
      EXECUTION_SUMMARY.md     # Human-readable lessons learned
      execution_stats.json     # Machine-readable statistics

    AGGREGATE_LEARNINGS.md   # Optional: Insights across all runs
```

**Note:** The `history/` folder contains complete records of every execution attempt. Success runs show what worked well, failed runs show what broke and why. Each run captures lessons about worker performance, brain interventions, VRAM usage, and system behavior.

---

## Plan Structure

Plans have two sections:

1. **Header** - Goal, inputs, outputs, script descriptions
2. **Tasks** - Concrete task definitions the brain parses directly

```markdown
# Plan: <Name>

## Goal

One or two sentences describing what this plan accomplishes.

## Inputs

- **VAR_NAME**: Description and format

## Outputs

- What files/data will be produced
- Where they will be written

## Available Scripts

### scripts/init_script.py

- **Purpose**: What this script does
- **GPU**: no
- **Run command**: `source ~/ml-env/bin/activate && python {PLAN_PATH}/scripts/init_script.py --batch-id {BATCH_ID} --input {INPUT_VAR}`
- **Output**: Description of what it produces

### scripts/process_script.py

- **Purpose**: What this script does
- **GPU**: yes
- **VRAM estimate**: infer first run, then set fixed from measured usage
- **Run command**: `source ~/ml-env/bin/activate && python {PLAN_PATH}/scripts/process_script.py --batch-id {BATCH_ID}`
- **Output**: Description

### scripts/aggregate_script.py

- **Purpose**: What this script does
- **GPU**: no
- **Run command**: `source ~/ml-env/bin/activate && python {PLAN_PATH}/scripts/aggregate_script.py --batch-id {BATCH_ID}`
- **Output**: Description

## Tasks

Define tasks with explicit dependencies. The brain parses this section directly.

### init
- **executor**: worker
- **task_class**: cpu
- **command**: `source ~/ml-env/bin/activate && python {PLAN_PATH}/scripts/init_script.py --batch-id {BATCH_ID} --input {INPUT_VAR}`
- **depends_on**: none
- **requires**: {INPUT_VAR}
- **produces**: {BATCH_PATH}/manifest.json

### process
- **executor**: worker
- **task_class**: script
- **vram_policy**: infer
- **command**: `source ~/ml-env/bin/activate && python {PLAN_PATH}/scripts/process_script.py --batch-id {BATCH_ID}`
- **depends_on**: init
- **requires**: {BATCH_PATH}/manifest.json
- **produces**: {BATCH_PATH}/results/{ITEM.id}/processed.json

### aggregate
- **executor**: brain
- **task_class**: cpu
- **command**: `source ~/ml-env/bin/activate && python {PLAN_PATH}/scripts/aggregate_script.py --batch-id {BATCH_ID}`
- **depends_on**: process
- **requires**: {BATCH_PATH}/results/{ITEM.id}/processed.json
- **produces**: {BATCH_PATH}/output/final.json

## Notes

Any additional guidance, constraints, or edge cases.
```

---

## Task Format

The brain parses the `## Tasks` section directly. Each task is defined as:

```markdown
### task_id
- **executor**: brain|worker
- **task_class**: cpu|script|llm
- **command**: `shell command here`
- **depends_on**: task1, task2
- **requires**: path1, path2
- **produces**: path3, path4
- **foreach**: {BATCH_PATH}/manifest.json:items  (optional)
- **batch_size**: 4  (optional, foreach only)
- **vram_policy**: default|infer|fixed  (optional, script tasks)
- **vram_estimate_mb**: 2400  (optional, script tasks)
```

| Field | Values | Description |
|-------|--------|-------------|
| `task_id` | Any unique name | Used for dependency references |
| `executor` | `brain` or `worker` | Who runs this task (routing control) |
| `task_class` | `cpu`, `script`, or `llm` | **Required.** What resources the task needs (see below) |
| `command` | Shell command in backticks | What to execute |
| `depends_on` | Comma-separated task IDs or `none` | Tasks that must complete first |
| `requires` | Comma-separated file paths/patterns | **Required.** Inputs this task expects to read |
| `produces` | Comma-separated file paths/patterns | **Required.** Outputs this task is responsible for writing |
| `foreach` | `path:jsonpath` | **Optional.** Expand to N tasks (see Foreach Expansion below) |
| `batch_size` | Integer >= 1 | **Optional.** Group foreach items into micro-batch tasks (default `1`) |
| `vram_policy` | `default`, `infer`, `fixed` | **Optional (script).** How brain sets script VRAM estimate |
| `vram_estimate_mb` | Integer MB | **Optional (script).** Explicit VRAM estimate used when `fixed` |

**Important:** `executor`, `task_class`, `command`, `depends_on`, `requires`, and `produces` are required in every task. `foreach`, `batch_size`, `vram_policy`, and `vram_estimate_mb` are optional. If `task_class` is missing, the task goes to `failed/` immediately. The brain will attempt to auto-fix by inferring the class from command keywords, but plans should always specify it explicitly.

### Executor Routing (Brain vs Worker)

Use `executor` to control ownership of each task:

- `executor: worker`
  - Default for parallel per-item work and high-throughput stages.
  - Claimed by GPU/CPU workers based on `task_class`.
- `executor: brain`
  - Reserved for orchestration-heavy steps, cross-item reasoning, and final synthesis.
  - Claimed directly by the brain loop (never by workers).

Recommended split:
1. Initial setup/validation: `brain`
2. Per-item processing (`foreach`): `worker`
3. Mid-run quality gate/reconciliation: `brain`
4. Final summary/decision: `brain`

Rule of thumb:
- If task logic is mostly "run this command N times", use `worker`.
- If task logic is mostly "decide/aggregate/reconcile across many tasks", use `brain`.

### Dataflow Contracts

Use `requires` and `produces` to make task handoffs explicit and machine-checkable.

- `requires` should list concrete files (or explicit patterns) that must exist before the task can run.
- `produces` should list the outputs the task guarantees when successful.
- For `foreach` tasks, use `{ITEM.field}` in paths to bind per-item artifacts.
- Downstream tasks should `require` outputs from upstream tasks. If an output is never consumed, remove it or document why.
- Keep contracts tight: list only files that matter for orchestration and validation.

### Task Classes

| Class | Needs | Examples | VRAM Default |
|-------|-------|----------|--------------|
| `cpu` | CPU only | File I/O, data transforms, aggregation | 1000 MB (nominal) |
| `script` | GPU compute (no LLM) | Transcription, embeddings, ffmpeg | **Per-script estimate** |
| `llm` | GPU + LLM model | Text generation, parsing, summarization | 5000 MB (not stackable) |
| `meta` | Worker management | load_llm, unload_llm | 0 MB |

Workers prioritize tasks they're optimized for:
- **GPU workers**: Prefer `llm` (if model loaded) or `script`, fall back to `cpu`
- **CPU workers** (RPi): Only claim `cpu` tasks

**VRAM Management:**
- **cpu tasks**: Nominal 1GB estimate prevents workers from claiming unlimited CPU tasks (limits to ~4-5 concurrent)
- **llm tasks**: Fixed 5GB estimate, not stackable - if GPU has LLM loaded, it's dedicated to that
- **script tasks**: **Planner must estimate per-script** - these vary widely and are the priority for VRAM optimization
- **meta tasks**: Zero VRAM (model load/unload commands)

**Note:** There's also a `meta` task class used internally by the brain for model loading/unloading. Plans should not use this class - the brain inserts these tasks automatically based on queue state.

### VRAM Estimation (Script Tasks)

**Priority: Script tasks are the focus for VRAM optimization.**

**Policy options (per task):**
- `default`: no explicit estimate is attached; workers use their script default
- `infer`: brain LLM infers one estimate once per task definition, then applies it to all expanded tasks
- `fixed`: planner provides explicit `vram_estimate_mb`, which brain passes through directly

For `task_class: script`, omitted `vram_policy` defaults to `infer`.

When documenting scripts in the `## Available Scripts` section, include a `VRAM estimate` field for GPU-based scripts. This helps the brain allocate tasks efficiently and enables workers to claim multiple script tasks concurrently if VRAM allows.

**Common Script VRAM Estimates:**

| Tool/Library | VRAM Estimate | Notes |
|--------------|---------------|-------|
| Whisper (tiny) | 800 MB | Fast, lower quality |
| Whisper (base) | 1000 MB | Standard |
| Whisper (medium) | 1500 MB | Better accuracy |
| Whisper (large) | 2500 MB | Best quality |
| Sentence-Transformers | 500 MB | Embedding generation |
| ffmpeg (NVENC) | 800 MB | GPU-accelerated video encoding |
| OCR (Tesseract GPU) | 600 MB | Text extraction |
| Custom CUDA script | Estimate based on model size | Document in script |

**Estimation Guidelines:**
- Check the model size being loaded (transformers, torch models)
- Add 20-30% overhead for CUDA operations and buffers
- Test on GPU rig and document actual usage (see logging below)
- Be conservative - overestimating is safer than OOM crashes

**VRAM Logging:**
Workers log VRAM usage before and after script tasks to track actual vs estimated usage. This data helps tune future estimates.

### Estimation Workflow (Recommended)

Use a two-pass approach for script tasks:

1. Start with `vram_policy: infer`.
2. Run a small pilot batch and collect `max_vram_used_mb` from completed tasks.
3. Set `vram_policy: fixed` with:
   - `vram_estimate_mb = ceil(p95(max_vram_used_mb) * 1.2)`
4. Re-test and adjust if OOM or under-utilization appears.

This keeps plans dynamic at first, then stable and deterministic once measured on your actual hardware.

### Foreach Expansion

Use `foreach` to expand a single task definition into N tasks - one per item in a data source. This enables parallel processing across multiple workers.

**Format:** `path:jsonpath` where path is a JSON file and jsonpath navigates to an array.

```markdown
### process
- **foreach**: {BATCH_PATH}/manifest.json:items
- **batch_size**: 4
- **command**: `python script.py --id {ITEM.id}`
- **depends_on**: init
```

**Item substitution:** Use `{ITEM.field}` to insert values from each item. For simple arrays (not objects), use `{ITEM}`.

**Per-item dependencies:** `depends_on` can also use `{ITEM.field}` to chain matching items across foreach stages.
Example:
`add_topics` task can use `depends_on: init, transcribe_whisper_{ITEM.id}` so each topic task is released as soon as its own transcript finishes.

**Dependency behavior:** Tasks depending on a foreach task automatically wait for ALL expanded tasks. If `process` expands to 100 tasks, `aggregate` with `depends_on: process` waits for all 100.

**Batching behavior (`batch_size > 1`):**
- Brain groups foreach items into micro-batch tasks (e.g., `process_batch_0001_0004`).
- Each micro-batch runs item commands sequentially in one claimed task.
- Micro-batch dependencies are the union of dependencies across items in that batch.
- Use conservative batch sizes for tasks with highly variable per-item runtime.

**Typical pattern:**
1. `init` (executor: brain) - creates manifest listing items to process
2. `process` (foreach) - expands to N worker tasks
3. `aggregate` (executor: brain) - runs after all processing complete

### How the Brain Handles Dependencies

The brain maintains two task lists:

1. **Private list** - All tasks from the plan, waiting for dependencies (`shared/brain/private_tasks/`)
2. **Public queue** - Tasks ready for workers (`shared/tasks/queue/`)

Workflow:
1. Brain reads plan, parses all tasks from `## Tasks` section
2. Tasks with no dependencies are released immediately to public queue
3. Brain monitors completed tasks
4. When a task completes, brain checks which private tasks now have all dependencies met
5. Newly-ready tasks are released to public queue
6. Workers only see and claim tasks from the public queue

This is a **Gantt-chart style** dependency model - flexible ordering based on actual dependencies, not rigid phases.

---

## Environment Setup

**The planner (Claude) is responsible for environment setup.**

When writing the plan, include complete run commands with all required setup. The brain copies these commands - it doesn't figure out environment variables on its own.

**Simple script (no special GPU libs):**
```markdown
- **Run command**: `source ~/ml-env/bin/activate && python {PLAN_PATH}/scripts/process.py --batch-id {BATCH_ID}`
```

**GPU script needing CUDA libraries:**
```markdown
- **Run command**:
  ```
  export LD_LIBRARY_PATH="$HOME/ml-env/lib/python3.13/site-packages/nvidia/cublas/lib:$HOME/ml-env/lib/python3.13/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH" && \
  source ~/ml-env/bin/activate && \
  python {PLAN_PATH}/scripts/gpu_process.py --batch-id {BATCH_ID}
  ```
```

**Note:** Scripts that need GPU access should auto-detect their GPU via `CUDA_VISIBLE_DEVICES`, which is set by the worker process. Plans should not specify GPU IDs - this keeps plans generic and allows workers to be added/removed without changing plans.

---

## Variables

The brain substitutes these when generating tasks:

| Variable | Description |
|----------|-------------|
| `{PLAN_PATH}` | Absolute path to the plan folder |
| `{BATCH_ID}` | Timestamp-based execution ID (format: YYYYMMDD_HHMMSS) |
| `{BATCH_PATH}` | `{PLAN_PATH}/history/{BATCH_ID}` |
| Custom inputs | Values passed when submitting the plan |

**Note:** Plans should not reference specific GPUs or workers. Tasks are generic - any available worker claims them. This allows the system to scale without changing plans.

---

## Run Modes (Fresh vs Resume)

Every operational plan should define how to run in both modes:

1. `fresh`: create new batch state and process from scratch
2. `resume`: continue from existing batch state without resetting progress

System behavior note:
- `RUN_MODE=fresh` now triggers plan-scoped cleanup of stale queued/processing/private tasks from older active batches of the same plan before launching the new batch.
- Cleanup does not touch unrelated plans, so multiple different plans can run in parallel.

Recommended submission inputs:

- `RUN_MODE`: `fresh` or `resume`
- `RESUME_BATCH_ID`: required when `RUN_MODE=resume`

Submission contract for execute-plan tasks:

```json
{
  "type": "execute_plan",
  "plan_path": "/path/to/plan",
  "config": {
    "RUN_MODE": "resume",
    "RESUME_BATCH_ID": "e49bc95c"
  }
}
```

### Resume Safety Rules

Plans must be explicit about resume behavior:

1. Resume must never overwrite an existing `history/{batch_id}/manifest.json`.
2. Resume must not regenerate already completed outputs unless explicitly requested.
3. Resume scripts should be idempotent and skip work already marked complete.
4. If resume state is missing or corrupted, fail fast with a clear error.

### Implementation Patterns

Use one of these patterns:

1. Single plan with mode-aware scripts:
- `init` checks `RUN_MODE`; on `resume`, validates existing batch and no-ops.

2. Separate plans:
- `plan_fresh.md` includes init/create-manifest.
- `plan_resume.md` starts from existing manifest/state and only schedules continuation tasks.

For critical pipelines, pattern 2 is usually safer and easier to audit.

---

## Worker Environment Variables

Workers set environment variables that scripts can use to access worker-specific resources:

| Variable | Example | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | `2` | GPU index assigned to this worker |
| `WORKER_OLLAMA_URL` | `http://localhost:11436` | URL to worker's Ollama instance for LLM calls |

**Usage in scripts:**
```python
import os

# Auto-detect GPU (for CUDA-based libraries)
gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))

# Call worker's LLM for inference
ollama_url = os.environ.get("WORKER_OLLAMA_URL", "http://localhost:11434")
response = requests.post(f"{ollama_url}/api/generate", json={...})
```

**Why this matters:** Each worker runs its own Ollama instance on a dedicated port. Scripts that need LLM inference should use `WORKER_OLLAMA_URL` rather than hardcoding `localhost:11434` (which is the brain's Ollama, not available to workers).

---

## Script Guidelines

Scripts in the plan folder should:

1. **Accept `--batch-id`** - So they know which batch folder to use
2. **Accept input/output paths as arguments** - Don't hardcode paths
3. **Be self-contained** - Import what they need, use plan's `lib/` folder
4. **Exit 0 on success, non-zero on failure** - Brain uses exit code to detect success
5. **Write progress to stdout** - Brain logs this for monitoring
6. **Be idempotent when possible** - Safe to retry on failure
7. **Support resume semantics** - If run against existing batch state, skip completed work instead of resetting it

**VRAM Tracking:**
Workers automatically log VRAM usage before and after executing script tasks. This data is collected in task completion results, allowing the brain and monitoring tools to compare estimated vs actual VRAM usage. Use this data to tune VRAM estimates in future plan updates.

### Script Output Convention

Scripts should write output to predictable locations within the batch folder:

```
{BATCH_PATH}/
  manifest.json           # Created by init (list of items to process)
  results/                # Processing output (one file per item)
    item_001.json
    item_002.json
  output/                 # Aggregate output (final combined result)
    final.json
  logs/
    events.log            # Progress and debug info
```

The folder names (`results/`, `output/`) are conventions - use whatever makes sense for your plan.

---

## Example Plans

### Example 1: Batch File Processor (Simple)

```markdown
# Plan: Batch File Processor

## Goal

Process a folder of input files, transform each one, and combine results into a summary.

## Inputs

- **INPUT_FOLDER**: Path to folder containing files to process
- **FILE_PATTERN**: Glob pattern for files (e.g., "*.json")

## Outputs

- Processed files in `{BATCH_PATH}/results/`
- Combined summary in `{BATCH_PATH}/output/summary.json`

## Available Scripts

### scripts/scan.py

- **Purpose**: List all files matching pattern, create manifest
- **Task class**: script
- **GPU**: no
- **Run command**: `source ~/ml-env/bin/activate && python {PLAN_PATH}/scripts/scan.py --input {INPUT_FOLDER} --pattern "{FILE_PATTERN}" --batch-id {BATCH_ID}`
- **Output**: `{BATCH_PATH}/manifest.json` listing files to process

### scripts/process.py

- **Purpose**: Process files from the manifest
- **Task class**: script
- **GPU**: no
- **Run command**: `source ~/ml-env/bin/activate && python {PLAN_PATH}/scripts/process.py --batch-id {BATCH_ID}`
- **Output**: Result files in `{BATCH_PATH}/results/`
- **Note**: Script claims items from manifest internally until none remain

### scripts/combine.py

- **Purpose**: Combine all processed results into summary
- **Task class**: script
- **GPU**: no
- **Run command**: `source ~/ml-env/bin/activate && python {PLAN_PATH}/scripts/combine.py --batch-id {BATCH_ID}`
- **Output**: `{BATCH_PATH}/output/summary.json`

## Tasks

### scan
- **executor**: worker
- **task_class**: cpu
- **command**: `source ~/ml-env/bin/activate && python {PLAN_PATH}/scripts/scan.py --input {INPUT_FOLDER} --pattern "{FILE_PATTERN}" --batch-id {BATCH_ID}`
- **depends_on**: none

### process
- **executor**: worker
- **task_class**: cpu
- **vram_policy**: infer
- **command**: `source ~/ml-env/bin/activate && python {PLAN_PATH}/scripts/process.py --batch-id {BATCH_ID}`
- **depends_on**: scan

### combine
- **executor**: worker
- **task_class**: cpu
- **command**: `source ~/ml-env/bin/activate && python {PLAN_PATH}/scripts/combine.py --batch-id {BATCH_ID}`
- **depends_on**: process

## Notes

- CPU-only plan (`task_class: cpu`), can run on any worker including RPi
- Safe to retry - scripts skip already-processed items
```

### Example 2: GPU Compute Batch (With CUDA)

```markdown
# Plan: Embedding Generator

## Goal

Generate vector embeddings for a set of text documents.

## Inputs

- **DOCS_PATH**: Path to folder containing text files

## Outputs

- Embeddings in `{BATCH_PATH}/output/vectors.json`

## Available Scripts

### scripts/init.py

- **Purpose**: Scan documents, create manifest
- **Task class**: script
- **GPU**: no
- **Run command**: `source ~/ml-env/bin/activate && python {PLAN_PATH}/scripts/init.py --docs {DOCS_PATH} --batch-id {BATCH_ID}`
- **Output**: `{BATCH_PATH}/manifest.json`

### scripts/embed.py

- **Purpose**: Generate embeddings for documents using GPU (sentence-transformers)
- **Task class**: script
- **GPU**: yes
- **VRAM estimate**: 500 MB
- **Run command**: `export LD_LIBRARY_PATH="$HOME/ml-env/lib/python3.13/site-packages/nvidia/cublas/lib:$HOME/ml-env/lib/python3.13/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH" && source ~/ml-env/bin/activate && python {PLAN_PATH}/scripts/embed.py --batch-id {BATCH_ID}`
- **Output**: Embedding files in `{BATCH_PATH}/results/`
- **Note**: Script auto-detects available GPU via CUDA_VISIBLE_DEVICES set by worker

### scripts/merge.py

- **Purpose**: Merge all embeddings into single file
- **Task class**: script
- **GPU**: no
- **Run command**: `source ~/ml-env/bin/activate && python {PLAN_PATH}/scripts/merge.py --batch-id {BATCH_ID}`
- **Output**: `{BATCH_PATH}/output/vectors.json`

## Tasks

### init
- **executor**: worker
- **task_class**: cpu
- **command**: `source ~/ml-env/bin/activate && python {PLAN_PATH}/scripts/init.py --docs {DOCS_PATH} --batch-id {BATCH_ID}`
- **depends_on**: none

### embed
- **executor**: worker
- **task_class**: script
- **vram_policy**: fixed
- **vram_estimate_mb**: 500
- **command**: `export LD_LIBRARY_PATH="$HOME/ml-env/lib/python3.13/site-packages/nvidia/cublas/lib:$HOME/ml-env/lib/python3.13/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH" && source ~/ml-env/bin/activate && python {PLAN_PATH}/scripts/embed.py --batch-id {BATCH_ID}`
- **depends_on**: init

### merge
- **executor**: worker
- **task_class**: cpu
- **command**: `source ~/ml-env/bin/activate && python {PLAN_PATH}/scripts/merge.py --batch-id {BATCH_ID}`
- **depends_on**: embed
```

---

## Script Adaptation Layer

**External scripts may need small adjustments to work with the orchestration system.**

Plans often include scripts written for standalone use or other contexts. The planner (Claude) is responsible for identifying integration gaps and documenting them so the brain can handle mismatches.

### Common Adaptations

| Issue | Example | Resolution |
|-------|---------|------------|
| Hardcoded GPU IDs | Script requires `--gpu 1` | Make optional, auto-detect from `CUDA_VISIBLE_DEVICES` |
| Hardcoded paths | Script uses `/data/input` | Accept as argument, use `{PLAN_PATH}` variables |
| Missing batch awareness | Script doesn't accept `--batch-id` | Add argument or wrapper script |
| Environment assumptions | Script expects specific virtualenv | Include full activation in run command |

### Documenting Adaptations

If a plan uses scripts that need adaptation, document them in the `## Notes` section:

```markdown
## Notes

### Script Adaptation Required

| Script | Issue | Fix |
|--------|-------|-----|
| `process.py` | Requires `--gpu` argument | Make optional; auto-detect from `CUDA_VISIBLE_DEVICES` |
```

### Brain's Role

When a task fails due to a script mismatch the brain can detect:
1. Brain identifies the error type (missing argument, path issue, etc.)
2. Brain attempts to infer the fix from context (e.g., add missing argument)
3. If fixed, task is re-queued
4. If unfixable, escalate to Claude for script rewrite (future)

**Principle:** External plans provide domain logic (what to do). The orchestration system provides execution context (which GPU, batch paths, etc.). The planner documents gaps; the brain bridges them.

---

## What the Brain Does

1. **Reads plan.md** - Parses the `## Tasks` section directly (no LLM needed)
2. **Creates batch** - Generates timestamp-based batch ID, creates `history/{batch_id}/` folders
3. **Creates all tasks** - One task per entry in `## Tasks` section
4. **Holds tasks privately** - In `shared/brain/private_tasks/`
5. **Releases ready tasks** - Tasks with no pending dependencies go to public queue
6. **Monitors completions** - Watches for tasks finishing in `shared/tasks/complete/`
7. **Releases dependent tasks** - When a task completes, check what's now unblocked
8. **Handles failures** - Retry or escalate based on retry count
9. **Generates execution summary** - Auto-inserts summary task at end of every plan
10. **Reports completion** - When all tasks complete, batch is done

### Important: Batch ID Ownership

By default, `execute_plan` creates a new timestamp batch id. If you need true resume behavior, your submission+scripts must explicitly target an existing batch id and avoid reinitializing state.

---

## Automatic Execution Summary

**The brain automatically inserts a final summary task for every plan.** You don't need to add this to your plan - it's handled automatically.

This summary task:
- Runs last (depends on all other tasks)
- Analyzes all completed and failed tasks
- Generates EXECUTION_SUMMARY.md and execution_stats.json
- Writes both to `history/{batch_id}/`
- Tracks what worked, what failed, and lessons learned

The summary captures:
- **Overall results**: Total tasks, success rate, duration
- **Task class performance**: Success rates and VRAM usage by class (cpu, script, llm)
- **Worker performance**: Which workers completed how many tasks, resource constraints
- **Brain interventions**: Auto-fixes, model management, retries
- **Escalations**: Issues that needed human intervention
- **Failures analysis**: Root causes and recommended fixes
- **VRAM tracking**: Estimated vs actual usage for tuning future runs
- **Lessons learned**: Actionable insights for next execution

This enables the system to learn from both successful and failed runs.

---

## Submitting a Plan

```bash
python scripts/submit.py <plan_name> \
  --config '{"INPUT_VAR": "/path/to/input", "OTHER_VAR": "value"}'
```

Examples:
```bash
# Simple batch processor
python scripts/submit.py batch_processor \
  --config '{"INPUT_FOLDER": "/data/files", "FILE_PATTERN": "*.json"}'

# GPU embedding job
python scripts/submit.py embedding_generator \
  --config '{"DOCS_PATH": "/data/documents"}'

# Resume an existing batch
python scripts/submit.py video_zim_batch \
  --config '{"RUN_MODE": "resume", "RESUME_BATCH_ID": "e49bc95c"}'
```

This creates an `execute_plan` task that the brain picks up.

---

*Last Updated: February 2026*
