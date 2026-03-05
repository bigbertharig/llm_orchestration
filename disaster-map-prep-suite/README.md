# Disaster Map Prep Suite

Single-folder workload package for preparing disaster aggregates for county-map.

This package is intentionally self-contained (scripts + inputs + plan) so it can be dropped into rig plan locations without extra restructuring.

## Folder Layout

```text
disaster-map-prep-suite/
  plan.md
  README.md
  input/
    disaster_aggregation_config.json
    sample_submit_config.json
  scripts/
    build_disaster_aggregates.py
    run_smoke_suite.py
    verify_aggregate_outputs.py
  history/           # runtime artifacts if you choose to persist here
```

## What It Builds

From `events + event_areas`, script builds:
- `global/disasters/{hazard}/aggregates/admin2/yearly.parquet`
- `global/disasters/{hazard}/aggregates/admin2/rolling_10y.parquet` (optional)
- `global/disasters/{hazard}/aggregates/admin2/rolling_20y.parquet` (optional)
- `global/qa/disaster_aggregate_build_report.csv`

## Quick Start (Local or Rig)

Smoke run:

```bash
python scripts/run_smoke_suite.py \
  --data-root /path/to/county-map-data \
  --config ./input/disaster_aggregation_config.json \
  --min-year 2000
```

One hazard:

```bash
python scripts/build_disaster_aggregates.py \
  --hazard earthquakes \
  --data-root /path/to/county-map-data \
  --config ./input/disaster_aggregation_config.json \
  --min-year 2000 \
  --windows 10
```

Long run:

```bash
python scripts/build_disaster_aggregates.py \
  --all-hazards \
  --data-root /path/to/county-map-data \
  --config ./input/disaster_aggregation_config.json \
  --min-year 1900 \
  --windows 10 20
```

## Rig Branch Sync Workflow

Use this when the suite is developed in `feature/disaster-map-prep-suite` and mirrored into rig plans.

1. Sync plan files from branch into rig plan location:
```bash
/home/bryan/llm_orchestration/scripts/sync_disaster_map_prep_suite.sh
```
2. Submit with wrapper submit path (default, recommended):
```bash
python3 /home/bryan/llm_orchestration/scripts/submit.py \
  /media/bryan/shared/plans/arms/disaster-map-prep-suite \
  --config '{"COUNTY_MAP_DATA_ROOT":"/data/county-map-data","MIN_YEAR":"2000"}'
```

Notes:
1. Sync script exports from `origin/feature/disaster-map-prep-suite` directly, so it does not require switching local branches.
2. Re-run the sync script any time the feature branch is updated.
3. Plan content should be edited in `disaster-map-prep-suite/` (source), not in the mirrored `shared/plans/arms/` copy.

## Notes

1. Heavy hazards (notably floods/hurricanes) can take longer for rolling windows.
2. Start with yearly-only for heavy hazards, then add rolling windows on bigger rigs.
3. Wildfires are disabled in config until event file shape is unified.
