# Lessons

What this repo has learned from running the rig, kept here so it carries across sessions and machines, and
so benchmarking can read it. This is one half of how the four repos share knowledge without depending on
each other:

- **In:** benchmarking keeps its lessons in the model library (registry and roles). This repo's importer,
  `tools/generate_profiles.py`, reads the library and writes this repo's own `model_catalog.json` and
  `remote_profiles.json`.
- **Out:** this folder. Benchmarking (or anyone) reads it when it wants to; this repo never writes into
  another repo, and nothing here needs another repo to run.

## Files

| File | What | Written by |
|---|---|---|
| `load_observations.jsonl` | manager loads and failures (`load`, `load_failed`), full-depth memory growth (`depth_memory`), and comparable prompt/decode speed (`performance`); each row identifies its model, GGUF, card layout and relevant runtime settings | the manager, on every load (`src/llm_gateway/manager/observations.py`) |

The manager also plans from this file (compute by card count, overhead and growth by card layout, OOM ceilings and measured placement speed:
`docs/MANAGER.md`, "Learning from loads"). It started on 2026-10-25 from the loads in the benchmarking
capacity records up to then; those rows carry `from_record` (the record's file name in benchmarking).

Append only. A row that turns out wrong is corrected by a later row, not edited. Commit it like any other
file.
