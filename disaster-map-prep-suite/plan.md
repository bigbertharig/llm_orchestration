# Plan: Disaster Map Prep Suite

## Objective
Create rerunnable disaster aggregate outputs for map/chat usage from existing `events + event_areas`, then verify outputs quickly before long full-history runs.

## Inputs
1. `COUNTY_MAP_DATA_ROOT` (path to `county-map-data`)
2. `PLAN_PATH` (this folder path)
3. `HAZARDS` (optional comma list)
4. `MIN_YEAR` (optional, defaults to `2000` for smoke; use lower for full runs)

## Outputs
1. `global/disasters/{hazard}/aggregates/admin2/yearly.parquet`
2. `global/disasters/{hazard}/aggregates/admin2/rolling_10y.parquet` (when requested)
3. `global/disasters/{hazard}/aggregates/admin2/rolling_20y.parquet` (when requested)
4. `global/qa/disaster_aggregate_build_report.csv`

## Available Scripts
1. `scripts/build_disaster_aggregates.py` - main aggregate builder
2. `scripts/verify_aggregate_outputs.py` - output QA checks
3. `scripts/run_smoke_suite.py` - bounded end-to-end smoke run

## Tasks

### Task: smoke_build
- executor: worker
- task_class: script
- command: `python {PLAN_PATH}/scripts/run_smoke_suite.py --data-root {COUNTY_MAP_DATA_ROOT} --config {PLAN_PATH}/input/disaster_aggregation_config.json --min-year {MIN_YEAR}`
- depends_on: `[]`

### Task: full_hazard_build (optional long run)
- executor: worker
- task_class: script
- command: `python {PLAN_PATH}/scripts/build_disaster_aggregates.py --all-hazards --data-root {COUNTY_MAP_DATA_ROOT} --config {PLAN_PATH}/input/disaster_aggregation_config.json --min-year {MIN_YEAR} --windows 10 20`
- depends_on: `[]`

### Task: verify_outputs
- executor: worker
- task_class: script
- command: `python {PLAN_PATH}/scripts/verify_aggregate_outputs.py --data-root {COUNTY_MAP_DATA_ROOT}`
- depends_on: `["smoke_build"]`

