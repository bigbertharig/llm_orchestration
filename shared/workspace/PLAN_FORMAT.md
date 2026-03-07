# Plan Format

Plans are markdown files that tell the brain how to orchestrate work.

This document defines:
- the stable plan structure the parser expects
- the task fields that are valid
- the execution contract that plans should rely on
- generic design patterns that are safe to reuse across many plan types

This is a shared format spec, not a plan template for one specific workflow.

---

## Core Purpose

A plan tells the brain:
1. What the plan is trying to accomplish
2. What inputs are provided at submission time
3. What scripts are available
4. What outputs should be produced
5. What tasks exist and how they depend on each other

The brain reads the markdown and generates executable tasks.

Plans should be:
- precise enough that commands are deterministic
- generic enough that the same format works for many workflows
- explicit about dependencies and outputs

Important boundary:
- Generic infrastructure checks that must pass before any plan starts belong in the submit wrapper, not in the plan graph.
- Only put setup steps in `## Tasks` when they are specific to that plan's workflow.
- Reuse shared runtime prep/health flows instead of re-creating them in each plan.
  - Canonical runtime prep script: `/mnt/shared/scripts/prepare_llm_runtimes.py`
  - This script includes stale launch-lock cleanup, deterministic load order, and split-runtime recovery.
  - Plans should not duplicate force-unload/restart/load-scan logic unless they truly require custom behavior.

---

## Hard Contract

### Required Sections

A standard plan should contain:
1. `# Plan: <Name>`
2. `## Objective`
3. `## Inputs`
4. `## Outputs`
5. `## Available Scripts`
6. `## Tasks`

A plan may also contain:
- `## Notes`
- `## Goal` for goal-driven execution

### Variable Substitution

The brain substitutes only:
- built-in variables such as `{BATCH_ID}`, `{BATCH_PATH}`, `{PLAN_PATH}`
- submission config keys defined by the user at submit time

Important rules:
- If a command references `{MISSING_VAR}`, it remains literal and can break the run.
- Text like `(default: ...)` in `## Inputs` is documentation only unless the command or script actually enforces that default.
- If you want dynamic behavior such as `auto`, the called script must explicitly support it.
- Wrapper-level preflight behavior is controlled by the submit tool, not by plan variables, unless the submit tool explicitly exposes a switch.

### Task Fields Required In Every Task

Every task must define:
- `executor`
- `task_class`
- `command`
- `depends_on`

If `task_class` is missing, the task can fail immediately.
Plans should always specify it explicitly.

### Standard Task Fields

Valid task fields are:
- `executor`
- `task_class`
- `command`
- `depends_on`
- `requires`
- `produces`
- `foreach`
- `batch_size`
- `llm_min_tier`
- `llm_model`
- `llm_placement`
- `vram_policy`
- `vram_estimate_mb`

Only use fields that apply to the task type.

### Field Validity By Task Type

Fields valid for all task types:
- `executor`
- `task_class`
- `command`
- `depends_on`
- `requires`
- `produces`
- `foreach`
- `batch_size`

Fields valid only for `llm` tasks:
- `llm_min_tier`
- `llm_model`
- `llm_placement`

Brain-owned LLM tasks:
- A task may be `executor: brain` with `task_class: llm` when it requires the brain model.
- Do not add `llm_model` to brain-owned tasks.
- Do not pass fixed brain model ids or fixed brain Ollama URLs in the task command.
- The shared runtime is the source of truth for brain model selection and endpoint binding.

Fields valid only for `script` tasks:
- `vram_policy`
- `vram_estimate_mb`

### Executor Values

Valid values:
- `brain`
- `worker`

Use:
- `worker` for parallel or throughput-oriented execution
- `brain` for orchestration-heavy, centralized, or synthesis steps

Important boundary:
- `executor: brain` means "run this with the active brain runtime."
- Plans must not choose the concrete brain model or brain endpoint.
- Brain tasks should consume runtime-injected `BRAIN_MODEL` and `BRAIN_OLLAMA_URL`.
- Plans may still choose worker-side model demand when that is part of the workflow contract.

### Task Class Values

Valid values:
- `cpu`
- `script`
- `llm`
- `brain`
- `meta`

Use:
- `cpu` for normal CPU-only scripting
- `script` for non-LLM tasks that may need GPU or explicit VRAM tracking
- `llm` for model-backed tasks
- `brain` only for tasks that are logically owned by the brain itself
- `meta` for runtime control operations (model load/unload/reset commands)

### Known Meta Commands

`meta` tasks use `command` to specify the operation. Current known commands:

- `load_llm`
- `unload_llm`
- `load_split_llm`
- `unload_split_llm`
- `cleanup_split_runtime`
- `reset_gpu_runtime`
- `reset_split_runtime`

Important rules:
- Plans and shared scripts must use only commands implemented in `/mnt/shared/agents/gpu_tasks.py`.
- Do not invent new meta command names in plan files or helper scripts.
- If you need new runtime-control behavior, land the agent support first, then use it from plans or shared helpers.

Required payload shape by command:
- `load_llm`: optional `target_model`; may include `candidate_workers`
- `unload_llm`: optional `candidate_workers`
- `load_split_llm`: requires `target_model` and `candidate_groups`
- `unload_split_llm`: should include `group_id` when unloading a specific split runtime
- `cleanup_split_runtime`: should include `group_id`; may include `target_workers`
- `reset_gpu_runtime`: should include `target_worker`
- `reset_split_runtime`: should include `target_worker`; include `group_id` when known

`candidate_groups` for `load_split_llm` must be full group objects, not bare ids:

```json
[
  {
    "id": "pair_4_5",
    "members": ["gpu-4", "gpu-5"],
    "port": 11441
  }
]
```

Example:

```markdown
### load_pair_4_5
- **executor**: worker
- **task_class**: meta
- **command**: `load_split_llm`
- **depends_on**: none
```

### Foreach Expansion

`foreach` expands one task template into many tasks.

Format:
- `path:jsonpath`

Example:
- `{BATCH_PATH}/manifest.json:items`

Rules:
- `batch_size` groups multiple expanded items into one task
- If a valid `foreach` source expands to zero items, the brain may treat the template task as a no-op success
- A task depending on a `foreach` task waits for all expanded tasks

### Dependency Semantics

`depends_on` must be:
- `none`
- or a comma-separated list of task ids

A task is not released until all dependencies are complete.

---

## Plan Location

```text
shared/plans/<plan_name>/
  plan.md
  scripts/
  lib/                # optional

  history/
    {batch_id}/
      output/
      results/
      logs/
      batch_events.jsonl
      RUN_SUMMARY.md
      RUN_SUMMARY.json
      EXECUTION_SUMMARY.md      # optional legacy artifact
      execution_stats.json      # optional legacy artifact
```

This layout is conventional, not magical. A plan may add more files, but commands should reference them explicitly.

---

## Standard Plan Structure

```markdown
# Plan: <Name>

## Objective

Short description of what the plan does.

## Inputs

- **INPUT_A**: Description
- **INPUT_B**: Description

## Outputs

- {BATCH_PATH}/output/result.json

## Available Scripts

### scripts/prepare.py
- **Purpose**: Prepare input artifacts
- **GPU**: no
- **Run command**: `python {PLAN_PATH}/scripts/prepare.py --batch-id {BATCH_ID}`
- **Output**: Writes an input manifest

### scripts/process.py
- **Purpose**: Process one or more items
- **GPU**: yes
- **Run command**: `python {PLAN_PATH}/scripts/process.py --batch-id {BATCH_ID}`
- **Output**: Writes per-item results

## Tasks

### prepare
- **executor**: brain
- **task_class**: cpu
- **command**: `python {PLAN_PATH}/scripts/prepare.py --batch-id {BATCH_ID}`
- **depends_on**: none
- **produces**: {BATCH_PATH}/manifest.json

### process
- **executor**: worker
- **task_class**: llm
- **llm_model**: model_id_here
- **command**: `python {PLAN_PATH}/scripts/process.py --batch-id {BATCH_ID} --item-id {ITEM.id}`
- **depends_on**: prepare
- **foreach**: {BATCH_PATH}/manifest.json:items
- **batch_size**: 1

### aggregate
- **executor**: brain
- **task_class**: cpu
- **command**: `python {PLAN_PATH}/scripts/aggregate.py --batch-id {BATCH_ID}`
- **depends_on**: process
- **requires**: {BATCH_PATH}/results/*.json
- **produces**: {BATCH_PATH}/output/final.json
```

---

## Task Field Reference

### Task Definition Shape

```markdown
### task_id
- **executor**: brain|worker
- **task_class**: cpu|script|llm|brain
- **command**: `shell command here`
- **depends_on**: task_a, task_b
- **requires**: path_a, path_b
- **produces**: path_c, path_d
- **foreach**: {BATCH_PATH}/manifest.json:items
- **batch_size**: 1
- **llm_min_tier**: 1
- **llm_model**: model_id
- **llm_placement**: single_gpu|split_gpu
- **vram_policy**: default|infer|fixed
- **vram_estimate_mb**: 2048
```

### Field Meanings

| Field | Meaning |
|---|---|
| `task_id` | Unique identifier for dependency references |
| `executor` | Which actor owns the task (`brain` or `worker`) |
| `task_class` | What kind of resources the task needs |
| `command` | The shell command to execute |
| `depends_on` | Prior tasks that must complete first |
| `requires` | Documentation-only contract for expected inputs |
| `produces` | Documentation-only contract for expected outputs |
| `foreach` | Expand this task from a manifest source |
| `batch_size` | Group multiple `foreach` items into one task |
| `llm_min_tier` | Minimum model capability tier |
| `llm_model` | Preferred model id |
| `llm_placement` | Placement constraint for LLM tasks |
| `vram_policy` | How VRAM is estimated for script tasks |
| `vram_estimate_mb` | Explicit VRAM estimate when fixed |

### Documentation-Only Fields

The following fields document intent but do not automatically enforce correctness by themselves:
- `requires`
- `produces`
- default values written in `## Inputs`

Use them anyway. They make plans easier to reason about and easier to audit.

---

## LLM Task Rules

### LLM Placement

Valid values:
- `single_gpu`
- `split_gpu`

### Split LLM Rule

Tasks with `llm_placement: split_gpu` should be single-model tasks.

If a workflow needs multiple model roles, decompose it into separate tasks and pass artifacts between them.

Good pattern:
1. one task produces intermediate JSON
2. a later split-GPU task consumes that JSON

Avoid mixed-model command lines in one split task.

---

## Recommended Patterns

These are not required by the parser, but they are good general-purpose patterns.

### Standard Submission Preflight

Use the submit wrapper for checks that should apply to every plan before queueing:
- worker heartbeat vs live-runtime consistency checks
- reset-to-default recovery if the rig is in a bad state
- fail-fast validation that shared infrastructure is healthy

This keeps generic infrastructure protection out of individual plan graphs.

### Manifest-Preserving Transforms

When one script transforms a manifest into another:
- preserve downstream-critical item metadata
- do not silently drop fields that later tasks rely on
- if you replace fields, replace them with an equivalent representation

Examples of execution-shaping metadata:
- line-range segment metadata
- complexity scores
- routing hints
- chunk metadata

### Explicit `auto` Mode

If a script supports heuristic or adaptive behavior:
- use an explicit symbolic value such as `auto`
- document it in `## Inputs`
- pass it intentionally in the executable command path

Two safe patterns:
1. manual default, with `auto` as an explicit override
2. explicit `auto` hardcoded in the command path

If `auto` is hardcoded in the actual command, remove or rewrite any stale input knob documentation so the plan prose matches the runtime interface.

### Secondary Manifest Expansion

When one logical phase expands into more scheduler-visible tasks later:
1. add a manifest-builder task that writes the expanded task manifest
2. point the downstream `foreach` task at that new manifest
3. if partial outputs are produced, add an explicit merge task afterward

This pattern is generic and applies to any workload that needs fan-out after an earlier stage.

### Fail-Closed Merge Pattern

If a merge task reconstructs canonical outputs from split tasks:
- fail closed by default when expected partial outputs are missing or inconsistent
- do not silently write partial canonical outputs unless the plan explicitly allows that mode
- keep the merged output schema consistent with the non-split path

### Constrained Item Safety

If a plan introduces per-item constraints that make later generic transforms unsafe:
- preserve those constraints
- or mark the item as intentionally non-transformable
- do not silently ignore the constraint

---

## Goal-Driven Plans

Plans may include an optional `## Goal` section for dynamic, result-driven execution.

Use this when the plan should stop after reaching a target quality or count, rather than processing a fixed list only once.

### Goal Section Format

```markdown
## Goal
- **type**: target_count
- **target**: {TARGET_COUNT}
- **tolerance**: 2
- **tracked_task**: validate_item
- **discovery_tasks**: discover_items, score_items
- **max_attempts_multiplier**: 5
```

### Goal Fields

| Field | Meaning |
|---|---|
| `type` | Goal mode, currently `target_count` |
| `target` | Desired accepted count |
| `tolerance` | Acceptable undershoot / overshoot range |
| `tracked_task` | Task whose completion drives acceptance counting |
| `discovery_tasks` | Ordered discovery chain, if needed |
| `max_attempts_multiplier` | Circuit breaker for retries / candidate expansion |

### When To Use Goal-Driven Plans

Use goal-driven execution when:
- output quality varies
- some candidates will be rejected
- you want a target number of acceptable results

Do not use it when every item must always be processed exactly once.

---

## Anti-Patterns

Avoid these mistakes in any plan:

1. Unresolved placeholders in commands
- If `{MISSING_VAR}` appears in the command, the run can fail immediately.

2. Stale input knobs
- Do not keep an input documented if the command path no longer uses it.

3. Hidden defaults
- If behavior depends on a special mode such as `auto`, make that visible in the command path or in the explicit submission config.

4. Dropping manifest metadata
- If a later task depends on per-item fields, do not rebuild the manifest and silently discard them.

5. Partial canonical merges by default
- If split tasks feed a merge step, do not silently produce incomplete final artifacts unless the plan explicitly opts into partial behavior.

6. Split-GPU mixed-model tasks
- If a task requires split placement, keep the model contract simple and explicit.

---

## Practical Notes

- Use generic examples in shared format docs.
- Keep plan-specific tuning knobs in the plan repo, not in the shared format spec.
- If a repo needs environment bootstrap, include it in the plan command; the format doc should stay generic.
- If a plan uses unusual task fields or conventions, document them in that plan’s `## Notes` section, not here.
