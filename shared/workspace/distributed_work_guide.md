# Distributed Work Guide

How work moves through the current orchestration system, from plan creation to
runtime review.

For operator commands, use [quickstart.md](quickstart.md). For structure and
contracts, use [architecture.md](architecture.md), [PLAN_FORMAT.md](PLAN_FORMAT.md),
and [brain-behavior.md](brain-behavior.md).

---

## Workflow

1. Planning:
   - a plan folder is created under `shared/plans/arms/` or `shared/plans/shoulders/`
   - `plan.md` defines tasks, dependencies, and required artifacts
2. Submission:
   - operator submits through `scripts/submit.py` or the dashboard `Start Plan` action
   - normal runs do not use direct `/mnt/shared/agents/submit.py`
3. Brain interpretation:
   - brain parses `plan.md`
   - all tasks start in `shared/brain/private_tasks/`
   - ready tasks are released to `shared/tasks/queue/`
4. Worker execution:
   - GPU or CPU workers claim eligible tasks
   - workers manage local execution and report results
   - brain-issued `meta` tasks handle runtime changes like model load/unload
5. Brain monitoring:
   - brain watches completion/failure lanes
   - retries, requeues, quarantines, and shared runtime cleanup stay brain-owned
   - brain refreshes per-run summary artifacts during execution
6. Review:
   - inspect `history/<batch_id>/RUN_SUMMARY.json` or `RUN_SUMMARY.md`
   - for older or partial runs, use the history summary tools to reduce context first

---

## Submission Path

Preferred CLI:

```bash
python3 ~/llm_orchestration/scripts/submit.py \
  ~/llm_orchestration/shared/plans/<shoulder-or-arm>/<plan_name> \
  --config '{"KEY":"VALUE"}'
```

Equivalent operator UI:

- dashboard `Start Plan`

This creates an `execute_plan` task that the brain claims and expands into the
real task graph.

---

## Runtime Ownership

- brain owns queue truth, dependency release, retries, requeues, quarantine,
  and shared runtime authority
- workers own local process execution, probes, and child cleanup
- workers report observations upward; the brain decides shared-state changes

That keeps the brain authoritative without turning it into a per-step worker
controller.

---

## Batch Artifacts

Each batch writes under:

```text
shared/plans/<plan_name>/history/<batch_id>/
  logs/
    batch_events.jsonl
  output/
  results/
  RUN_SUMMARY.json
  RUN_SUMMARY.md
```

Legacy artifacts like `EXECUTION_SUMMARY.md` or `execution_stats.json` may
still exist for older runs, but the runtime-owned summary path is now
`RUN_SUMMARY.*`.

---

## Review Path

For one batch:

```bash
python3 ~/llm_orchestration/scripts/summarize_history_run.py \
  ~/llm_orchestration/shared/plans/<plan>/history/<batch_id>
```

For many batches:

```bash
python3 ~/llm_orchestration/scripts/rollup_history.py \
  ~/llm_orchestration/shared/plans/<plan>/history
```

These reducers shrink the evidence surface before an LLM reads it. They do not
replace review or make fix decisions.

---

## Related Docs

| Doc | Purpose |
|-----|---------|
| [quickstart.md](quickstart.md) | operator commands and recovery flow |
| [PLAN_FORMAT.md](PLAN_FORMAT.md) | authoritative plan format |
| [architecture.md](architecture.md) | system layout and authority boundary |
| [brain-behavior.md](brain-behavior.md) | runtime coordinator behavior |
| [history_summary_tools.md](history_summary_tools.md) | one-run and many-run reducers |
