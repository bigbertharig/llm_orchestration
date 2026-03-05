"""
Quick QA checks for generated disaster aggregate outputs.

Checks:
1. expected output files exist
2. required columns exist
3. row counts and year ranges are sensible

Usage:
  python scripts/verify_aggregate_outputs.py --data-root /path/to/county-map-data
  python scripts/verify_aggregate_outputs.py --data-root /path/to/county-map-data --hazard earthquakes --hazard floods
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd


def run_checks(data_root: Path, hazards: List[str]) -> pd.DataFrame:
    rows = []
    base = data_root / "global" / "disasters"
    for hazard in hazards:
        agg_dir = base / hazard / "aggregates" / "admin2"
        yearly = agg_dir / "yearly.parquet"
        rolling10 = agg_dir / "rolling_10y.parquet"
        rolling20 = agg_dir / "rolling_20y.parquet"

        rec = {
            "hazard": hazard,
            "yearly_exists": yearly.exists(),
            "rolling10_exists": rolling10.exists(),
            "rolling20_exists": rolling20.exists(),
            "yearly_rows": None,
            "year_start": None,
            "year_end": None,
            "required_cols_ok": False,
            "status": "missing",
            "message": "",
        }

        if yearly.exists():
            try:
                df = pd.read_parquet(yearly)
                rec["yearly_rows"] = int(len(df))
                if "year" in df.columns and len(df):
                    rec["year_start"] = int(df["year"].min())
                    rec["year_end"] = int(df["year"].max())
                required = {"loc_id", "year", "event_count", "source"}
                rec["required_cols_ok"] = required.issubset(set(df.columns))
                if rec["required_cols_ok"] and len(df) > 0:
                    rec["status"] = "ok"
                else:
                    rec["status"] = "warning"
                    rec["message"] = "yearly exists but required columns/rows are incomplete"
            except Exception as exc:
                rec["status"] = "error"
                rec["message"] = f"failed reading yearly parquet: {exc}"
        else:
            rec["message"] = "missing yearly.parquet"

        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify disaster aggregate outputs quickly.")
    parser.add_argument("--data-root", required=True, help="Path to county-map-data root.")
    parser.add_argument("--hazard", action="append", help="Hazard(s) to verify. Repeatable.")
    parser.add_argument("--report-json", default=None, help="Optional JSON report output path.")
    args = parser.parse_args()

    hazards = args.hazard or [
        "earthquakes", "volcanoes", "tsunamis", "hurricanes",
        "tornadoes", "floods", "landslides", "wildfires",
    ]
    report = run_checks(Path(args.data_root), hazards)
    print(report.to_string(index=False))

    if args.report_json:
        out = Path(args.report_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_dict(orient="records"), indent=2), encoding="utf-8")
        print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()

