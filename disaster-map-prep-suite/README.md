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

## Notes

1. Heavy hazards (notably floods/hurricanes) can take longer for rolling windows.
2. Start with yearly-only for heavy hazards, then add rolling windows on bigger rigs.
3. Wildfires are disabled in config until event file shape is unified.

