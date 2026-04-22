# Plan Creation Guide

Purpose: define the practical workflow for designing a new plan on another machine so it ports cleanly to this rig with minimal rewrite.

Use this guide with:
- [PLAN_FORMAT.md](PLAN_FORMAT.md) for the exact parser contract
- [quickstart.md](quickstart.md) for operator submission and monitoring
- [workload_transfer_guide.md](workload_transfer_guide.md) for external handoff expectations

This guide is intentionally biased toward script-heavy workloads that shard, process, and aggregate large datasets.

---

## When To Use A Plan

Use a real `plan.md` when you need:
1. dependency management between stages
2. parallel fan-out from a manifest
3. rerunnable batch history under `history/{batch_id}/`
4. clear operator submission through `submit.py` or dashboard
5. one workflow that may later grow from "just scripts" into a larger staged pipeline

Do not force a plan if the job is only:
1. one script
2. one input
3. one output
4. no staging
5. no need for fan-out or batch history

Rule of thumb:
1. one-off local utility -> plain script
2. repeated shard/process/merge workflow -> plan
3. reusable generic pattern -> shoulder
4. project-specific implementation -> arm

---

## Think In Work Units

Before writing scripts, decide what one schedulable unit of work is.

Good work units are:
1. independent
2. bounded in RAM/VRAM
3. deterministic
4. cheap to retry
5. large enough that scheduler overhead is small relative to useful work

Bad work units are:
1. one giant all-day script
2. items with wildly unpredictable memory usage
3. tasks that need hidden state from a previous shell session
4. tasks that write to the same output file concurrently

If the work can be divided many ways, prefer the unit that makes retries and partial reruns cheap.

---

## Design Goal

Write scripts on the source machine as if they will later run inside this shape:

1. `prepare`:
   - validate inputs
   - normalize paths/config
   - build a deterministic manifest of work items
2. `process`:
   - consume one manifest item or one batch of items
   - write one output artifact per item/batch
3. `aggregate`:
   - combine outputs
   - compute final datasets/reports
   - fail clearly if required intermediate results are missing

If the scripts already fit that pattern before they arrive here, converting them into a rig plan is fast.

---

## How To Structure CPU Work

If you want to use the CPU side of the rig well, structure tasks like this:

1. make each task independent and file-driven
2. partition the input into many bounded shards
3. keep per-task memory predictable
4. write one output per shard or shard-batch
5. merge only at the end

Good CPU workload patterns:
1. parquet partition transforms
2. spatial joins split by tile/region
3. geometry cleanup and normalization
4. bounded graph/path computations on regional subgraphs
5. feature extraction, indexing, filtering, and aggregation

Best structure for CPU tasks:
1. `prepare.py` builds a manifest of many small-to-medium work items
2. `process.py` handles one item or a small batch of items
3. `aggregate.py` merges outputs into final artifacts

What usually works best:
1. many more items than workers
2. no shared mutable state between workers
3. outputs written independently, then merged
4. retryable tasks that do not require re-running the whole dataset

Avoid for CPU plans:
1. one monolithic process that holds the whole dataset in memory
2. one shared sqlite/db file written by many workers at once
3. shard definitions based only on row counts when geometry/network complexity varies a lot
4. expensive global merge steps in the middle of the pipeline

CPU planning rule:
1. split by data locality and bounded cost
2. process in parallel
3. aggregate late

---

## How To Structure GPU Work

If you want to use GPUs well, structure tasks differently.

GPU tasks are a good fit when the script has a real GPU-backed implementation and the speedup is material.

Good GPU workload patterns:
1. CUDA/RAPIDS-backed dataframe operations
2. GPU spatial kernels
3. batched matrix/vector or tensor-heavy processing
4. expensive scoring/inference passes that clearly benefit from VRAM

Best structure for GPU tasks:
1. make each task heavy enough to justify GPU startup and transfer overhead
2. keep data transfer boundaries explicit
3. batch enough work per task to keep the GPU busy
4. avoid tiny GPU tasks with large setup cost
5. write deterministic outputs exactly like CPU tasks do

Use `task_class: script` when:
1. the task is not LLM work
2. it really does need GPU or explicit VRAM accounting

When designing GPU tasks, document:
1. expected VRAM per task
2. whether the task can run on any GPU or only certain cards
3. whether work items can be batched safely
4. whether CPU fallback exists or not

Avoid for GPU plans:
1. GPU versions of work that are still dominated by disk or network I/O
2. tiny shards that spend more time loading data than computing
3. hidden assumptions about fixed ports, fixed devices, or machine-local CUDA layout
4. mixed CPU/GPU logic inside one giant script with no resource boundary

GPU planning rule:
1. make GPU tasks coarse enough to amortize overhead
2. keep VRAM assumptions explicit
3. isolate GPU-heavy steps from CPU-heavy steps

---

## CPU vs GPU Decision Rule

When deciding whether a stage should be `cpu` or `script`, ask:

1. is the bottleneck compute, memory, disk I/O, or data transfer?
2. does a real GPU implementation already exist?
3. is the work large enough per task to justify GPU overhead?
4. can the task fit predictably in available VRAM?
5. would more CPU shards be simpler and almost as fast?

Choose `cpu` when:
1. the work is mostly file I/O, parsing, filtering, joins, graph traversal, or moderate geometry processing
2. the task scales cleanly by sharding
3. the GPU path would add complexity without clear throughput gain

Choose `script` when:
1. the task has a genuine GPU-backed implementation
2. task size is large enough to benefit
3. VRAM needs are estimable
4. the stage is clearly separable from CPU stages

Default rule:
1. if unsure, start with `cpu`
2. only promote a stage to GPU when measurement justifies it

---

## Portable Script Contract

Scripts intended for future plan use should follow these rules from the start.

### 1. Explicit Inputs

Every script should take explicit CLI args for:
1. input root(s)
2. output root
3. temp/work directory if needed
4. shard/item identifiers
5. config path

Avoid:
1. hardcoded absolute paths
2. hidden machine-local config
3. dependence on current working directory

### 2. Deterministic Outputs

Each process step should write outputs to deterministic locations.

Good:
- `results/{shard_id}.parquet`
- `logs/{shard_id}.log`
- `manifests/work_items.json`

Bad:
- random temp names with no index
- mixed outputs dumped into one directory with no stable naming

### 3. Idempotent Reruns

Each script should make reruns safe.

Choose one policy explicitly:
1. skip existing valid outputs
2. overwrite existing outputs
3. require `--force` to overwrite

Do not leave rerun behavior ambiguous.

### 4. Fail Fast

Scripts should exit non-zero with clear errors when:
1. required files are missing
2. schemas are wrong
3. a shard definition is invalid
4. an output write fails

### 5. No Hidden Cross-Step State

The output of one stage should be files on disk, not in-memory assumptions.

Good:
- `manifest.json`
- `results/shard_00017.parquet`
- `output/final_summary.json`

Bad:
- "run this second script in the same shell after the first one"
- hidden local sqlite/temp state outside the batch folder

---

## Parallelism Rules

To take advantage of the rig, structure scripts so the scheduler has room to work.

Good parallel structure:
1. at least dozens of work items for medium jobs
2. hundreds or thousands of work items for very large jobs when each item is cheap enough
3. item runtime variance kept reasonably bounded
4. one task writes one result target

Use `foreach` when:
1. items are independent
2. failures can be retried item-by-item
3. per-item outputs can be written safely

Use `batch_size` when:
1. each item is too small to justify a task on its own
2. startup overhead dominates compute time
3. several nearby items can be processed together without blowing memory

Bad parallel structure:
1. 3 huge tasks for a cluster with many workers
2. 50,000 tiny tasks that each do one second of work
3. one aggregate task that secretly redoes all the expensive work

The target is not "as many tasks as possible." The target is enough tasks to keep workers busy without drowning in overhead.

---

## Recommended Plan Shape

For large data workloads, start with this layout:

```text
shared/plans/arms/<plan_name>/
  plan.md
  README.md
  scripts/
    prepare.py
    process.py
    aggregate.py
  lib/                  # optional shared helpers
  input/                # optional static configs/templates
  history/              # runtime-created
```

Recommended batch artifact layout:

```text
history/{batch_id}/
  logs/
  output/
  results/
  manifest.json
  RUN_SUMMARY.md
  RUN_SUMMARY.json
```

Use `{BATCH_PATH}` for runtime outputs. Do not write real run outputs outside the batch history tree unless that is the deliberate final destination and the script documents it clearly.

---

## Standard Build Sequence

Build plans in this order.

### 1. Build The Scripts First

On the source machine, get these working locally:
1. `prepare.py`
2. `process.py`
3. `aggregate.py`

Verify:
1. `prepare.py` writes a valid manifest
2. `process.py` can run on one shard from that manifest
3. `aggregate.py` succeeds from multiple shard outputs

### 2. Freeze The Manifest Schema

Before writing `plan.md`, make the manifest contract explicit.

Recommended fields:

```json
{
  "items": [
    {
      "id": "tile_001",
      "input_path": "/shared/input/tile_001.parquet",
      "output_path": "results/tile_001.parquet"
    }
  ]
}
```

Keep item ids:
1. stable
2. unique
3. safe for filenames

### 3. Decide Task Class Correctly

Use:
1. `cpu` for normal CPU-only scripts
2. `script` for non-LLM tasks that may need GPU or explicit VRAM accounting
3. `llm` only for model-backed tasks
4. `meta` only for runtime-control operations already supported by the agents

For data aggregation, geometry overlap, parquet processing, routing, and travel-time work, default to `cpu` unless you have a real GPU-backed implementation with clear performance value.

### 4. Write `plan.md`

Use the exact section structure from [PLAN_FORMAT.md](PLAN_FORMAT.md):
1. `# Plan`
2. `## Objective`
3. `## Inputs`
4. `## Outputs`
5. `## Available Scripts`
6. `## Tasks`

### 5. Test With Small Inputs First

Before full ingestion on this rig:
1. use a bounded sample dataset
2. generate a small manifest
3. verify one full batch completes
4. only then scale shard count or batch size

---

## Canonical Data-Processing Pattern

This is the default shape for parallel script workloads:

```markdown
# Plan: Example Data Pipeline

## Objective

Process a large dataset in parallel and aggregate deterministic outputs.

## Inputs

- `INPUT_ROOT`: shared path to source data
- `CONFIG_PATH`: shared path to config file

## Outputs

- `{BATCH_PATH}/output/final.parquet`
- `{BATCH_PATH}/output/report.json`

## Available Scripts

### scripts/prepare.py
- **Purpose**: validate inputs and emit work manifest
- **GPU**: no
- **Run command**: `python {PLAN_PATH}/scripts/prepare.py --input-root {INPUT_ROOT} --config {CONFIG_PATH} --batch-path {BATCH_PATH}`
- **Output**: `{BATCH_PATH}/manifest.json`

### scripts/process.py
- **Purpose**: process one shard
- **GPU**: no
- **Run command**: `python {PLAN_PATH}/scripts/process.py --batch-path {BATCH_PATH} --item-id {ITEM.id}`
- **Output**: `{BATCH_PATH}/results/{ITEM.id}.parquet`

### scripts/aggregate.py
- **Purpose**: combine shard outputs and emit final outputs
- **GPU**: no
- **Run command**: `python {PLAN_PATH}/scripts/aggregate.py --batch-path {BATCH_PATH}`
- **Output**: `{BATCH_PATH}/output/final.parquet`

## Tasks

### prepare
- **executor**: brain
- **task_class**: cpu
- **command**: `python {PLAN_PATH}/scripts/prepare.py --input-root {INPUT_ROOT} --config {CONFIG_PATH} --batch-path {BATCH_PATH}`
- **depends_on**: none
- **produces**: {BATCH_PATH}/manifest.json

### process
- **executor**: worker
- **task_class**: cpu
- **command**: `python {PLAN_PATH}/scripts/process.py --batch-path {BATCH_PATH} --item-id {ITEM.id}`
- **depends_on**: prepare
- **foreach**: {BATCH_PATH}/manifest.json:items
- **batch_size**: 1
- **produces**: {BATCH_PATH}/results/{ITEM.id}.parquet

### aggregate
- **executor**: brain
- **task_class**: cpu
- **command**: `python {PLAN_PATH}/scripts/aggregate.py --batch-path {BATCH_PATH}`
- **depends_on**: process
- **requires**: {BATCH_PATH}/results/*.parquet
- **produces**: {BATCH_PATH}/output/final.parquet, {BATCH_PATH}/output/report.json
```

This is the baseline to copy for future script-heavy imports.

---

## Choosing `foreach` And `batch_size`

Use `foreach` when the job can be broken into independent work items.

Examples:
1. one parquet partition per item
2. one tile/bbox per item
3. one county/region per item
4. one road-network subgraph per item

Use `batch_size` to group small items into one task when per-task overhead is too high.

Good pattern:
1. tiny shards -> `batch_size: 10` or higher
2. heavy shards -> `batch_size: 1`

Start conservative. Increase grouping only after measuring runtime and overhead.

---

## CPU Cluster Guidance

For CPU worker use, design the plan so the work can run on weaker nodes.

Good fit for the RPis:
1. independent shard transforms
2. bounded parquet scans
3. metadata extraction
4. light geometry filtering
5. preprocessing and manifest generation

Poor fit for the RPis:
1. giant in-memory joins
2. heavy graph routing on very large networks
3. large geometry-overlap jobs requiring high RAM
4. tasks with intense local temp-disk churn

If the work is heavy but still CPU-bound:
1. use more shards
2. keep shard memory bounded
3. avoid one giant aggregate step until the end

If you want CPU work to spread well across heterogeneous machines:
1. prefer shards defined by bounded cost, not just equal file counts
2. make every shard self-describing in the manifest
3. avoid assuming local caches already exist
4. keep temp storage use explicit and bounded
5. make it safe for a slow worker to finish later without blocking unrelated shards

---

## Import Checklist For Plans Built Elsewhere

Before copying a new workload here, verify:

1. `plan.md` exists and follows [PLAN_FORMAT.md](PLAN_FORMAT.md)
2. every referenced script exists under `scripts/`
3. all paths are passed by args, not hardcoded to the source machine
4. the manifest schema is stable and documented
5. rerun behavior is explicit
6. outputs land under `{BATCH_PATH}` unless documented otherwise
7. a small known-good sample run exists
8. one sample submit config JSON exists
9. CPU vs GPU assumptions are documented
10. estimated RAM and temp-disk needs are documented

If those are true, rig alignment should be quick.

---

## Quick Alignment On Arrival

When a plan folder arrives here, do this:

1. read `plan.md`
2. compare task fields against [PLAN_FORMAT.md](PLAN_FORMAT.md)
3. check that commands only reference available scripts and valid placeholders
4. check that all source/input paths are rig-visible shared paths
5. run the scripts manually on a tiny sample if needed
6. submit a small batch through `~/llm_orchestration/scripts/submit.py`
7. inspect `history/{batch_id}/RUN_SUMMARY.md`

If the imported scripts already follow this guide, the usual remaining work is only:
1. path cleanup
2. placeholder cleanup
3. small manifest/output alignment

---

## What To Avoid

Avoid these anti-patterns:

1. writing scripts first as one giant monolith with no shard boundary
2. hiding defaults in code but leaving unresolved placeholders in `plan.md`
3. writing outputs outside the batch tree with no audit trail
4. assuming local disks, local env vars, or local service ports
5. mixing orchestration logic into the processing scripts
6. inventing unsupported `meta` commands
7. treating `requires` and `produces` as enforcement instead of documentation
8. making GPU tasks so small that transfer/setup dominates compute
9. making CPU tasks so large that only one machine can finish them reliably

---

## Bottom Line

The portable workflow is:

1. write shard-friendly scripts elsewhere
2. choose work units that match CPU or GPU economics
3. make them deterministic and explicit
4. give them a manifest-driven shape
5. wrap them in the standard `prepare -> process -> aggregate` plan pattern
6. bring them here and do a small path/placeholder alignment pass

That is the shortest path from "built on another machine" to "runs cleanly on the rig."
