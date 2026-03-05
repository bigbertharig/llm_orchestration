"""
Fast smoke suite for disaster aggregate generation.

Runs lightweight checks with bounded settings so operators can validate
pipeline health before long full-history runs.

Default smoke profile:
- hazards: earthquakes, volcanoes, tsunamis, tornadoes, floods, landslides, hurricanes
- min_year: 2000
- windows: 10 for lighter hazards, yearly-only for heavy hazards
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


LIGHT_HAZARDS = ["earthquakes", "volcanoes", "tsunamis", "tornadoes"]
HEAVY_HAZARDS = ["floods", "hurricanes", "landslides"]


def run_cmd(cmd: list[str]) -> int:
    print(" ".join(cmd))
    return subprocess.call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded smoke tests for disaster aggregate builds.")
    parser.add_argument("--data-root", required=True, help="Path to county-map-data root.")
    parser.add_argument("--config", required=True, help="Path to disaster aggregation config.")
    parser.add_argument("--min-year", type=int, default=2000, help="Lower year bound for smoke.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    build_script = script_dir / "build_disaster_aggregates.py"
    verify_script = script_dir / "verify_aggregate_outputs.py"

    # 1) lighter hazards with rolling window
    for hazard in LIGHT_HAZARDS:
        code = run_cmd([
            sys.executable, str(build_script),
            "--hazard", hazard,
            "--data-root", args.data_root,
            "--config", args.config,
            "--min-year", str(args.min_year),
            "--windows", "10",
        ])
        if code != 0:
            raise SystemExit(code)

    # 2) heavier hazards yearly-only
    for hazard in HEAVY_HAZARDS:
        code = run_cmd([
            sys.executable, str(build_script),
            "--hazard", hazard,
            "--data-root", args.data_root,
            "--config", args.config,
            "--min-year", str(args.min_year),
            "--windows",
        ])
        if code != 0:
            raise SystemExit(code)

    # 3) quick output verification
    code = run_cmd([
        sys.executable, str(verify_script),
        "--data-root", args.data_root,
    ])
    raise SystemExit(code)


if __name__ == "__main__":
    main()

