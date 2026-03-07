# History Summary Tools

## Goal

Reduce LLM context load when reviewing one batch or many batches.

Instead of reading whole history trees directly, run summary reducers first and
let the LLM read the reduced outputs.

The same reducer library powers:

- brain-owned per-run summary refresh during runtime
- a one-run CLI
- a many-run rollup CLI

## Tool Split

Use these tools for different scopes:

- `scripts/summarize_history_run.py`
  - one `history/<batch_id>/` folder
  - answers: what artifacts exist for this single run

- `scripts/rollup_history.py`
  - one `history/` root containing many run folders
  - answers: what patterns exist across many runs

## Runtime-Owned Artifacts

The brain now owns per-run lifecycle artifacts:

- `history/<batch_id>/logs/batch_events.jsonl`
- `history/<batch_id>/RUN_SUMMARY.json`
- `history/<batch_id>/RUN_SUMMARY.md`

These are refreshed during batch execution and on terminal batch lifecycle
events.

## One-Run Summary

Input:

- one `history/<batch_id>/` directory

Outputs:

- `RUN_SUMMARY.json`
- `RUN_SUMMARY.md`

What it surfaces:

- inferred coarse run state like `complete`, `failed`, or `partial`
- artifact presence and artifact counts
- important artifacts that already exist
- short excerpts or key fields from those artifacts

Examples of important artifacts:

- `results/batch_failure.json`
- `execution_stats.json`
- `EXECUTION_SUMMARY.md`
- `brain_strategy.json`
- `runtime_validation.json`
- `output/final_report.json`
- `output/runtime_probe_classification.json`
- `output/repo_structure_summary.json`
- `output/json_data_summary.json`

CLI:

```bash
python3 scripts/summarize_history_run.py \
  shared/plans/shoulders/github_analyzer/history/20260305_014325
```

Optional stdout JSON:

```bash
python3 scripts/summarize_history_run.py \
  shared/plans/shoulders/github_analyzer/history/20260305_014325 \
  --print-json
```

## Many-Run Rollup

Input:

- one `history/` directory containing many `history/<batch_id>/` folders

Outputs under `history/_summary/`:

- `ROLLUP_SUMMARY.json`
- `ROLLUP_SUMMARY.md`
- `runs.jsonl`
- `failures.jsonl`

What it surfaces:

- total run counts
- status counts across runs
- plan counts
- artifact-presence counts
- failure source-task counts
- per-run reduced summaries in `runs.jsonl`
- one normalized failure line per failed run in `failures.jsonl`

CLI:

```bash
python3 scripts/rollup_history.py \
  shared/plans/shoulders/github_analyzer/history
```

If you want to reuse already-written per-run summaries instead of refreshing
them first:

```bash
python3 scripts/rollup_history.py \
  shared/plans/shoulders/github_analyzer/history \
  --use-existing-run-summaries
```

Optional stdout JSON:

```bash
python3 scripts/rollup_history.py \
  shared/plans/shoulders/github_analyzer/history \
  --print-json
```

## What These Tools Do Not Do

- decide what to fix
- replace LLM review
- require batches to have completed normally
- depend on a terminal summary task in the plan graph

These tools compress runtime evidence into a smaller review surface. The LLM
still reads that reduced surface and decides what matters.
