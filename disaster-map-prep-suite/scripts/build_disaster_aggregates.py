"""
Build reusable disaster aggregates from events + event_areas.

This is the standard pipeline for producing hazard aggregates that can be rerun
as data density improves over time.

What it does:
1. Loads hazard events from `global/disasters/{hazard}/...`
2. Loads `global/disasters/event_areas/{hazard}.parquet`
3. Joins by event loc_id and deduplicates event/location pairs
4. Builds admin2 yearly aggregates
5. Builds rolling window aggregates (10y/20y by default)
6. Writes outputs to `global/disasters/{hazard}/aggregates/admin2/`

Output files per hazard:
- `yearly.parquet`
- `rolling_10y.parquet` (if requested)
- `rolling_20y.parquet` (if requested)

Output schema (core):
- `loc_id`
- `year` (yearly) or `window_end_year` (rolling)
- `event_count`
- `source`
- hazard metrics from config (max/sum/avg/etc)

Notes:
- Admin2 is the base truth level for disaster aggregation.
- Parent rollups (admin1/admin0) should be done after this using existing
  metadata-aware rollup tools or level-specific recompute.
- Wildfires are intentionally disabled in config until the event file shape is
  standardized across sources.

Usage examples:
  python data_converters/scripts/build_disaster_aggregates.py --all-hazards
  python data_converters/scripts/build_disaster_aggregates.py --hazard earthquakes
  python data_converters/scripts/build_disaster_aggregates.py --hazard hurricanes --windows 10 20
  python data_converters/scripts/build_disaster_aggregates.py --all-hazards --min-year 1900
  python data_converters/scripts/build_disaster_aggregates.py --all-hazards --allow-partial-windows
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


# Rig-friendly defaults:
# - Prefer explicit CLI args
# - Fall back to env vars
# - Finally use local plan folder defaults
PLAN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path.cwd() / "county-map-data"
DEFAULT_CONFIG_PATH = PLAN_ROOT / "input" / "disaster_aggregation_config.json"


@dataclass
class MetricSpec:
    output: str
    source: str
    agg: str


def _load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def _safe_year_series(df: pd.DataFrame, year_col: str) -> pd.Series:
    years = pd.to_numeric(df[year_col], errors="coerce")
    return years


def _build_yearly_agg(
    merged: pd.DataFrame,
    metric_specs: List[MetricSpec],
    source_label: str,
) -> pd.DataFrame:
    # Base aggregation: deduped event-area links count as exposure events.
    group_cols = ["affected_loc_id", "year"]
    parts: List[pd.DataFrame] = []

    base = (
        merged.groupby(group_cols, as_index=False)
        .agg(event_count=("event_loc_id", "nunique"))
        .rename(columns={"affected_loc_id": "loc_id"})
    )
    parts.append(base)

    for spec in metric_specs:
        if spec.source not in merged.columns:
            continue

        if spec.agg == "sum":
            sub = (
                merged.groupby(group_cols, as_index=False)
                .agg(**{spec.output: (spec.source, "sum")})
                .rename(columns={"affected_loc_id": "loc_id"})
            )
        elif spec.agg == "max":
            sub = (
                merged.groupby(group_cols, as_index=False)
                .agg(**{spec.output: (spec.source, "max")})
                .rename(columns={"affected_loc_id": "loc_id"})
            )
        elif spec.agg == "min":
            sub = (
                merged.groupby(group_cols, as_index=False)
                .agg(**{spec.output: (spec.source, "min")})
                .rename(columns={"affected_loc_id": "loc_id"})
            )
        elif spec.agg == "avg":
            sub = (
                merged.groupby(group_cols, as_index=False)
                .agg(**{spec.output: (spec.source, "mean")})
                .rename(columns={"affected_loc_id": "loc_id"})
            )
        else:
            # Unknown aggregation type: ignore metric, keep pipeline running.
            continue

        parts.append(sub)

    out = parts[0]
    for sub in parts[1:]:
        out = out.merge(sub, on=["loc_id", "year"], how="left")

    out["source"] = source_label
    out = out.sort_values(["loc_id", "year"]).reset_index(drop=True)
    return out


def _build_event_lookup(
    events: pd.DataFrame,
    key_cols: List[str],
    year_col: str,
    metric_cols: List[str],
    canonical_col: str,
) -> Tuple[pd.DataFrame, dict]:
    """
    Build a robust key lookup for joining event_areas to events.

    Supports:
    - direct keys (loc_id, event_id, storm_id, ...)
    - loc_id aliases:
      - drop first segment (e.g., BRA-FLOOD-DFO-2 -> FLOOD-DFO-2)
      - drop first two segments (e.g., XOO-EQ-NOAA-SIG-1 -> NOAA-SIG-1)
    """
    key_cols = [c for c in key_cols if c in events.columns]
    metric_cols = [c for c in metric_cols if c in events.columns]
    if not key_cols:
        return pd.DataFrame(columns=["event_loc_id", "year"] + metric_cols), {"aliases_built": 0, "ambiguous_keys": 0}

    # Canonical id should uniquely identify an event row.
    if canonical_col not in events.columns:
        canonical_col = key_cols[0]

    base_cols = [canonical_col, year_col] + metric_cols
    select_cols: List[str] = []
    for c in base_cols + key_cols:
        if c not in select_cols:
            select_cols.append(c)
    work = events[select_cols].copy()
    work = work.rename(columns={year_col: "year", canonical_col: "_canonical_id"})
    work["_canonical_id"] = work["_canonical_id"].astype(str)

    parts: List[pd.DataFrame] = []
    alias_count = 0

    key_iter = ["_canonical_id" if c == canonical_col else c for c in key_cols]

    for key_col in key_iter:
        s = work[key_col].astype("string")
        direct = work[["_canonical_id", "year"] + metric_cols].copy()
        direct["event_loc_id"] = s
        parts.append(direct)

        # Create loc-like aliases for dashed IDs.
        d1 = s.str.split("-", n=1).str[-1]
        d1_mask = d1.notna() & (d1 != s)
        if d1_mask.any():
            a1 = work.loc[d1_mask, ["_canonical_id", "year"] + metric_cols].copy()
            a1["event_loc_id"] = d1[d1_mask]
            parts.append(a1)
            alias_count += int(d1_mask.sum())

        d2 = d1.str.split("-", n=1).str[-1]
        d2_mask = d2.notna() & (d2 != d1)
        if d2_mask.any():
            a2 = work.loc[d2_mask, ["_canonical_id", "year"] + metric_cols].copy()
            a2["event_loc_id"] = d2[d2_mask]
            parts.append(a2)
            alias_count += int(d2_mask.sum())

    lookup = pd.concat(parts, ignore_index=True)
    lookup["event_loc_id"] = lookup["event_loc_id"].astype("string").str.strip()
    lookup = lookup[lookup["event_loc_id"].notna() & (lookup["event_loc_id"] != "")]

    # Keep only unambiguous keys to avoid false multi-match joins.
    key_uniques = lookup.groupby("event_loc_id")["_canonical_id"].nunique()
    valid_keys = key_uniques[key_uniques == 1].index
    ambiguous_keys = int((key_uniques > 1).sum())
    lookup = lookup[lookup["event_loc_id"].isin(valid_keys)]

    # One lookup row per key after ambiguity filtering.
    lookup = lookup.drop_duplicates(subset=["event_loc_id"], keep="first")
    lookup = lookup.drop(columns=["_canonical_id"], errors="ignore")
    return lookup, {"aliases_built": alias_count, "ambiguous_keys": ambiguous_keys}


def _build_rolling_agg(
    yearly: pd.DataFrame,
    metric_specs: List[MetricSpec],
    window_years: int,
    allow_partial: bool = False,
) -> pd.DataFrame:
    rows: List[dict] = []
    if yearly.empty:
        return pd.DataFrame()

    # Split metrics by rule for rolling behavior.
    sum_metrics = [m.output for m in metric_specs if m.agg == "sum" and m.output in yearly.columns]
    max_metrics = [m.output for m in metric_specs if m.agg == "max" and m.output in yearly.columns]
    min_metrics = [m.output for m in metric_specs if m.agg == "min" and m.output in yearly.columns]
    avg_metrics = [m.output for m in metric_specs if m.agg == "avg" and m.output in yearly.columns]

    for loc_id, loc_df in yearly.groupby("loc_id"):
        loc_df = loc_df.sort_values("year")
        years = loc_df["year"].dropna().astype(int).tolist()
        if not years:
            continue
        unique_years = sorted(set(years))

        for end_year in unique_years:
            start_year = end_year - window_years + 1
            sub = loc_df[(loc_df["year"] >= start_year) & (loc_df["year"] <= end_year)].copy()
            if sub.empty:
                continue
            if not allow_partial and sub["year"].nunique() < window_years:
                continue

            rec = {
                "loc_id": loc_id,
                "window_start_year": int(start_year),
                "window_end_year": int(end_year),
                "window_years": int(window_years),
                "years_observed": int(sub["year"].nunique()),
                "event_count": int(sub["event_count"].sum()),
                "source": str(sub["source"].iloc[0]) if "source" in sub.columns and len(sub) else None,
            }

            for col in sum_metrics:
                rec[col] = float(sub[col].sum(skipna=True))
            for col in max_metrics:
                rec[col] = float(sub[col].max(skipna=True))
            for col in min_metrics:
                rec[col] = float(sub[col].min(skipna=True))
            for col in avg_metrics:
                # Weighted by yearly event_count to approximate event-level rolling mean.
                valid = sub[["event_count", col]].dropna()
                if valid.empty or valid["event_count"].sum() == 0:
                    rec[col] = np.nan
                else:
                    rec[col] = float((valid[col] * valid["event_count"]).sum() / valid["event_count"].sum())

            rows.append(rec)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values(["loc_id", "window_end_year"]).reset_index(drop=True)
    return out


def _hazard_paths(data_root: Path, hazard: str, event_file: str) -> Tuple[Path, Path, Path]:
    hazards_root = data_root / "global" / "disasters"
    event_path = hazards_root / hazard / event_file
    areas_path = hazards_root / "event_areas" / f"{hazard}.parquet"
    out_dir = hazards_root / hazard / "aggregates" / "admin2"
    return event_path, areas_path, out_dir


def _to_metric_specs(metric_rows: Iterable[dict]) -> List[MetricSpec]:
    specs: List[MetricSpec] = []
    for row in metric_rows:
        output = str(row.get("output", "")).strip()
        source = str(row.get("source", "")).strip()
        agg = str(row.get("agg", "")).strip().lower()
        if not output or not source or agg not in {"sum", "max", "min", "avg"}:
            continue
        specs.append(MetricSpec(output=output, source=source, agg=agg))
    return specs


def build_hazard(
    data_root: Path,
    hazard: str,
    hazard_cfg: dict,
    windows: List[int],
    min_year: int | None,
    max_year: int | None,
    allow_partial_windows: bool,
) -> dict:
    if not hazard_cfg.get("enabled", True):
        return {"hazard": hazard, "status": "skipped", "reason": "disabled in config"}

    event_file = hazard_cfg.get("event_file", "")
    if not event_file:
        return {"hazard": hazard, "status": "skipped", "reason": "no event_file in config"}

    event_id_col = hazard_cfg.get("event_id_col", "loc_id")
    year_col = hazard_cfg.get("year_col", "year")
    source_label = hazard_cfg.get("source_label", f"{hazard}_events")
    metric_specs = _to_metric_specs(hazard_cfg.get("metrics", []))

    event_path, areas_path, out_dir = _hazard_paths(data_root, hazard, event_file)
    if not event_path.exists():
        return {"hazard": hazard, "status": "skipped", "reason": f"missing events file: {event_path}"}
    if not areas_path.exists():
        return {"hazard": hazard, "status": "skipped", "reason": f"missing event_areas file: {areas_path}"}

    # Load events
    events = pd.read_parquet(event_path)
    if event_id_col not in events.columns:
        return {"hazard": hazard, "status": "error", "reason": f"event_id_col '{event_id_col}' not found"}
    if year_col not in events.columns:
        return {"hazard": hazard, "status": "error", "reason": f"year_col '{year_col}' not found"}

    # Keep needed columns (including multiple key candidates for robust join).
    key_candidates = [event_id_col, "loc_id", "event_id", "storm_id", "eruption_id", "dfo_id"]
    key_candidates = [c for c in key_candidates if c in events.columns]
    metric_sources = []
    for m in metric_specs:
        if m.source in events.columns and m.source not in metric_sources:
            metric_sources.append(m.source)
    keep_cols = []
    for c in key_candidates + [year_col] + metric_sources:
        if c not in keep_cols:
            keep_cols.append(c)
    events = events[keep_cols].copy()

    # Normalize year and metric numeric types.
    events["year"] = _safe_year_series(events.rename(columns={year_col: "year"}), "year")
    if year_col != "year":
        events = events.drop(columns=["year"], errors="ignore")
    events = events.rename(columns={year_col: "year"})
    events = events[events["year"].notna()].copy()
    events["year"] = events["year"].astype(int)
    if min_year is not None:
        events = events[events["year"] >= int(min_year)]
    if max_year is not None:
        events = events[events["year"] <= int(max_year)]
    for col in metric_sources:
        events[col] = pd.to_numeric(events[col], errors="coerce")

    # Build robust lookup from event IDs to event year/metrics.
    lookup, lookup_stats = _build_event_lookup(
        events=events,
        key_cols=key_candidates,
        year_col="year",
        metric_cols=metric_sources,
        canonical_col=event_id_col,
    )
    if lookup.empty:
        return {"hazard": hazard, "status": "skipped", "reason": "event lookup empty after key normalization"}

    # Load areas and dedupe event/location pairs (ignore impact_type duplication).
    areas = pd.read_parquet(areas_path, columns=["event_loc_id", "affected_loc_id"])
    areas = areas.dropna(subset=["event_loc_id", "affected_loc_id"])
    areas = areas.drop_duplicates(subset=["event_loc_id", "affected_loc_id"])

    merged = areas.merge(lookup, on="event_loc_id", how="left")
    unmatched = int(merged["year"].isna().sum())
    merged = merged[merged["year"].notna()].copy()
    if merged.empty:
        return {"hazard": hazard, "status": "skipped", "reason": "no matched event/year rows after join"}

    yearly = _build_yearly_agg(merged, metric_specs, source_label=source_label)

    out_dir.mkdir(parents=True, exist_ok=True)
    yearly_path = out_dir / "yearly.parquet"
    yearly.to_parquet(yearly_path, index=False)

    rolling_written = []
    for window in windows:
        rdf = _build_rolling_agg(
            yearly=yearly,
            metric_specs=metric_specs,
            window_years=int(window),
            allow_partial=allow_partial_windows,
        )
        if rdf.empty:
            continue
        path = out_dir / f"rolling_{int(window)}y.parquet"
        rdf.to_parquet(path, index=False)
        rolling_written.append(str(path))

    return {
        "hazard": hazard,
        "status": "ok",
        "events_file": str(event_path),
        "areas_file": str(areas_path),
        "yearly_file": str(yearly_path),
        "rolling_files": ";".join(rolling_written),
        "events_rows": int(len(events)),
        "event_lookup_rows": int(len(lookup)),
        "event_lookup_aliases": int(lookup_stats.get("aliases_built", 0)),
        "event_lookup_ambiguous_keys": int(lookup_stats.get("ambiguous_keys", 0)),
        "areas_rows": int(len(areas)),
        "joined_rows": int(len(merged)),
        "unmatched_area_rows": unmatched,
        "yearly_rows": int(len(yearly)),
        "year_start": int(yearly["year"].min()),
        "year_end": int(yearly["year"].max()),
        "loc_count": int(yearly["loc_id"].nunique()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reusable disaster aggregates from events + event_areas.")
    parser.add_argument("--config", default=None, help="Path to disaster aggregation config JSON.")
    parser.add_argument("--data-root", default=None, help="Path to county-map-data root.")
    parser.add_argument(
        "--report-path",
        default=None,
        help="CSV build report path. Defaults to {data_root}/global/qa/disaster_aggregate_build_report.csv.",
    )
    parser.add_argument("--hazard", action="append", help="Hazard(s) to build. Can be repeated.")
    parser.add_argument("--all-hazards", action="store_true", help="Build all enabled hazards from config.")
    parser.add_argument("--windows", type=int, nargs="*", default=None, help="Rolling windows in years (e.g., 10 20).")
    parser.add_argument("--min-year", type=int, default=None, help="Optional lower year bound.")
    parser.add_argument("--max-year", type=int, default=None, help="Optional upper year bound.")
    parser.add_argument("--allow-partial-windows", action="store_true", help="Allow rolling outputs with partial history.")
    args = parser.parse_args()

    import os
    if args.data_root:
        data_root = Path(args.data_root)
    elif os.getenv("COUNTY_MAP_DATA_ROOT"):
        data_root = Path(os.getenv("COUNTY_MAP_DATA_ROOT"))
    else:
        data_root = DEFAULT_DATA_ROOT

    if args.config:
        config_path = Path(args.config)
    else:
        config_path = Path(os.getenv("DISASTER_AGG_CONFIG", str(DEFAULT_CONFIG_PATH)))

    cfg = _load_config(config_path)

    hazards_cfg: Dict[str, dict] = cfg.get("hazards", {})
    if not hazards_cfg:
        raise ValueError("No hazards found in config.")

    windows = args.windows if args.windows is not None else cfg.get("windows", [10, 20])
    windows = [int(w) for w in windows if int(w) > 0]

    selected: List[str]
    if args.all_hazards:
        selected = list(hazards_cfg.keys())
    elif args.hazard:
        selected = [h for h in args.hazard if h in hazards_cfg]
        unknown = [h for h in args.hazard if h not in hazards_cfg]
        if unknown:
            print(f"Skipping unknown hazards: {unknown}")
    else:
        # Reasonable default for one-shot runs.
        selected = ["earthquakes", "volcanoes", "tsunamis", "hurricanes", "tornadoes", "floods", "landslides"]

    print("=" * 72)
    print("Build Disaster Aggregates")
    print("=" * 72)
    print(f"Data root: {data_root}")
    print(f"Config:    {config_path}")
    print(f"Hazards:   {selected}")
    print(f"Windows:   {windows}")
    print(f"Year cut:  min={args.min_year}, max={args.max_year}")
    print(f"Partial windows: {args.allow_partial_windows}")

    reports: List[dict] = []
    for hazard in selected:
        print(f"\n[{hazard}]")
        rep = build_hazard(
            data_root=data_root,
            hazard=hazard,
            hazard_cfg=hazards_cfg[hazard],
            windows=windows,
            min_year=args.min_year,
            max_year=args.max_year,
            allow_partial_windows=args.allow_partial_windows,
        )
        reports.append(rep)
        print(f"  status: {rep.get('status')}")
        if rep.get("status") == "ok":
            print(f"  yearly rows: {rep.get('yearly_rows')}, locs: {rep.get('loc_count')}, years: {rep.get('year_start')}-{rep.get('year_end')}")
        else:
            print(f"  reason: {rep.get('reason')}")

    report_df = pd.DataFrame(reports)
    if args.report_path:
        report_path = Path(args.report_path)
    else:
        report_path = data_root / "global" / "qa" / "disaster_aggregate_build_report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(report_path, index=False)

    ok = int((report_df["status"] == "ok").sum()) if not report_df.empty else 0
    skipped = int((report_df["status"] == "skipped").sum()) if not report_df.empty else 0
    errors = int((report_df["status"] == "error").sum()) if not report_df.empty else 0
    print("\n" + "-" * 72)
    print(f"Done. ok={ok}, skipped={skipped}, errors={errors}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
