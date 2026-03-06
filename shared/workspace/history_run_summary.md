# History Run Summary

## Goal

Reduce LLM context load when reviewing a finished, failed, or interrupted batch.

Instead of reading an entire `history/<batch_id>/` tree, run one reducer over
the folder and read the generated summary artifacts first.

This is an external analysis tool. It does not change batch execution logic.

## Inputs

- one `history/<batch_id>/` directory

The reducer is best-effort. It should still produce useful output if the run
ended cleanly, failed early, or stopped partway through.

## Outputs

The reducer writes two files into the same history folder:

- `RUN_SUMMARY.json`
- `RUN_SUMMARY.md`

## What It Surfaces

- inferred coarse run state like `complete`, `failed`, or `partial`
- artifact presence and artifact counts
- important run artifacts that already exist
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

## What It Does Not Do

- decide what to fix
- replace LLM review
- require the batch to have completed normally
- depend on a terminal summary task inside the orchestration graph

The script compresses the run into a smaller review surface. The LLM still
reads that reduced surface and decides what matters.

## CLI

Tracked entrypoint:

- `scripts/summarize_history_run.py`

Example:

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
