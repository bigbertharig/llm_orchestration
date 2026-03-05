# GPU Rig Workload Transfer Guide

Purpose: define the exact contract for external teams to deliver a plan folder that drops into `shared/plans/arms/` or `shared/plans/shoulders/` and runs with minimal local integration work.

Audience: external project teams preparing their own plan package, and rig operators validating intake.

---

## Primary Goal

External project team should build and deliver a self-contained plan folder that:
1. conforms to rig `plan.md` contract,
2. includes working scripts,
3. emits expected artifacts,
4. can be submitted immediately via wrapper submit.

If they do this, local team should not need to rewrite their shoulder/arm.

---

## Arms vs Shoulders (Where it goes)

1. `shared/plans/shoulders/<plan_name>`
- reusable plan contract/template
- benchmark or generalized workflow definitions
- reference implementation allowed, but usually generic

2. `shared/plans/arms/<plan_name>`
- project-specific implementation
- concrete scripts and domain logic
- directly runnable by operators

Rule of thumb:
1. If it is project-specific and ready to run now, deliver as an `arm`.
2. If it is a reusable pattern or scaffold used by many projects, deliver as a `shoulder`.

---

## What This System Can Do

1. Orchestrate multi-stage workflows using a dependency graph (`plan.md`).
2. Run CPU, script, LLM, and brain-owned tasks in one pipeline.
3. Split large workloads into parallel worker tasks (`foreach` + `batch_size`).
4. Dynamically load/unload worker models (`load_llm`, `unload_llm`, split variants).
5. Route tasks by capability (`llm_model`, tier, placement).
6. Persist full run artifacts under a per-batch history folder.

---

## Core Concepts

1. `Brain`: parses plans, releases tasks, handles orchestration/resource decisions.
2. `Workers`: execute queued tasks (CPU/script/LLM) based on eligibility.
3. `Plans`: markdown contracts with deterministic commands + dependencies.
4. `Task lanes`: `queue/`, `processing/`, `complete/`, `failed/`.
5. `Batch`: one execution instance with isolated outputs under `history/{batch_id}`.

---

## Workload Types That Fit Well

1. Large repository analysis and review.
2. Document extraction/classification at scale.
3. Dataset processing + staged summarization.
4. Multi-pass QA/verification pipelines.
5. Any workload that benefits from decomposition into independent shards.

---

## Required Folder Contract (External Delivery)

External team must provide this structure:

```text
shared/plans/arms/<project_name>/            # or shoulders/<project_name> for generic templates
  plan.md
  scripts/
    <all referenced .py/.sh scripts>
  lib/                                       # optional
  input/                                     # optional static inputs/templates
  history/                                   # created at runtime
```

Hard requirements:
1. Every command referenced in `plan.md` exists and is runnable.
2. No command depends on files outside this folder except repo input path and shared runtime lanes.
3. Scripts fail fast with clear non-zero exit on invalid input.
4. All produced artifacts are written under `{BATCH_PATH}`.

---

## External Handoff Package (What to send)

Minimum handoff from external team:
1. plan folder (`plan.md`, `scripts/`, optional `lib/`, optional `input/`)
2. one sample submit config JSON
3. expected output artifact list
4. one known-good test dataset/repo path
5. short runbook (`README.md` inside plan folder)

Recommended:
1. `--dry-run` proof output from submit parser
2. one successful batch `EXECUTION_SUMMARY.md`
3. runtime estimate by phase

---

## Standard Migration Flow (External Project -> Rig)

1. External team builds plan folder locally against this contract.
2. External team runs parser dry-run and fixes unresolved placeholders.
3. External team runs at least one small successful batch.
4. Copy folder into `shared/plans/arms/<project_name>` (or `shoulders/`).
5. Local operator submits with wrapper submit.
6. Local operator validates first run and archives benchmark notes.

---

## Plan Format (Required Contract)

Use `workspace/PLAN_FORMAT.md` as source of truth.

Minimum required sections:
1. `# Plan: <Name>`
2. `## Objective`
3. `## Inputs`
4. `## Outputs`
5. `## Available Scripts`
6. `## Tasks`

Required task fields:
1. `executor`
2. `task_class`
3. `command`
4. `depends_on`

Valid task classes:
1. `cpu`
2. `script`
3. `llm`
4. `brain`

LLM-only fields:
1. `llm_model`
2. `llm_min_tier`
3. `llm_placement`

Parallel expansion fields:
1. `foreach`
2. `batch_size`

Important rules:
1. Never leave unresolved placeholders (for example `{MISSING_VAR}`) in commands.
2. If a variable is optional, provide a literal command default instead of unresolved placeholder.
3. Keep generic infra preflight out of plan graph; wrapper submit handles it.

---

## Task Decomposition Pattern (Recommended)

1. `prepare` task:
   - validate inputs, normalize paths, generate manifest(s).
2. `extract` task family (parallel):
   - broad first-pass worker tasks.
3. `verify` task family (parallel/split):
   - deeper/expensive checks (often higher-tier model).
4. `aggregate/summarize` tasks:
   - merge artifacts, compute metrics, final outputs.

Keep tasks:
1. small enough to finish reliably,
2. independent where possible,
3. explicit about produced artifacts.

For large workloads, prefer:
1. `build_manifest` CPU stage that emits deterministic `slices`/`items`.
2. `worker_pass_1` broad extract pass on cheap model/class.
3. `worker_pass_2` verify/adjudicate pass on expensive model/class.
4. `merge`/`normalize` stage that rebuilds canonical outputs.

---

## Model/Resource Strategy

1. Keep workers cold by default.
2. Let brain/meta tasks warm capacity when demand appears.
3. Use lower-tier models for high-volume first pass.
4. Use higher-tier models only where they add value (verify/adjudication/synthesis).
5. Prefer capability-based routing over hard worker pinning.

For exact live model/tier/port assignments, check:
1. `shared/agents/config.json`

---

## Submission and Operations

Default submission path (authoritative):
```bash
python3 ~/llm_orchestration/scripts/submit.py \
  ~/llm_orchestration/shared/plans/arms/<plan_name> \
  --config '{"KEY":"VALUE"}'
```

Why this path:
1. submit-time worker preflight scan/reset
2. placeholder validation
3. path translation and rig proxy behavior

Monitor:
```bash
tail -f ~/llm_orchestration/shared/logs/brain_decisions.log
```

Dashboard:
1. `http://127.0.0.1:8787/`
2. Controls: `http://127.0.0.1:8787/controls`

---

## Artifact Contract (What to Persist)

Every plan should persist:
1. normalized input/context artifact,
2. per-shard worker outputs,
3. merged/normalized output,
4. benchmark/runtime summary,
5. final report (if applicable).

Keep output paths under:
1. `{BATCH_PATH}/output/`
2. `{BATCH_PATH}/results/`

Also persist:
1. manifest(s) used to create parallel tasks
2. any split/chunk mapping files needed for deterministic merge

---

## Reliability Requirements

1. Fail fast on invalid input/config.
2. Use explicit preflight checks for required tools/artifacts.
3. Keep retries bounded.
4. Log clear machine-readable failure reasons.
5. Avoid hidden fallback behavior; make recovery explicit.

Do not ship:
1. hardcoded absolute paths to developer machines,
2. implicit dependencies on global virtualenv state,
3. scripts that silently skip work and still exit `0`.

---

## New Project Onboarding Checklist

1. Plan repo created (`plan.md`, `scripts/`).
2. Inputs/outputs documented and deterministic.
3. No unresolved placeholders in task commands.
4. `foreach` manifests tested with sample data.
5. First benchmark run completed.
6. Runtime bottlenecks identified from batch artifacts.
7. Shard/model placement tuned.
8. Full workload run completed with reproducible outputs.

## Intake Acceptance Checklist (Operator Side)

Accept external plan only if all are true:
1. Folder contract is complete (`plan.md` + referenced scripts exist).
2. `python3 ~/llm_orchestration/scripts/submit.py <plan> --dry-run ...` succeeds.
3. First real batch completes without parser/CLI arg errors.
4. Outputs match declared `## Outputs`.
5. No out-of-tree writes outside `{BATCH_PATH}` and task lanes.
6. Runtime errors are actionable (not silent/opaque).

---

## Related Docs

1. `workspace/quickstart.md`
2. `workspace/PLAN_FORMAT.md`
3. `workspace/architecture.md`
4. `workspace/brain-behavior.md`
5. `workspace/distributed_work_guide.md`
