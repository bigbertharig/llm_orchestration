#!/usr/bin/env python3
"""GPU power limits: set each card's board-power cap at boot. Rig-side, runs as root (rig-power.service).

    hw_power.py apply [--dry-run]     set every card to its planned limit (needs root)
    hw_power.py show [--json]         current, default, planned limit per card
    hw_power.py check --gpus 1,3|all [--repair]
                                      exit 0 when those cards run at their planned limit; --repair
                                      re-applies once (sudo -n systemctl restart rig-power.service)

`nvidia-smi -pl` does not survive a reboot, so the rig sets the caps at every boot, after
nvidia-persistenced, and run_runtime.sh checks the cards of every load against the plan before
starting it (the double check: a load on a card at the wrong limit is refused). Caps go by card model
and default limit, not index (indices moved when the risers were re-cabled): GTX 1060s with a 120 W
default run at 100 W, measured 2026-08-09 to keep about 98% of decode speed for 17% less energy and
5 C cooler (benchmarking results/runtime_experiments/gtx1060_power_cap_100w_20260809.json,
gpu4_120w_vs_100w_sustained_decode_20260809.json); the factory-overclocked ones (156 W default) run
at 120 W (operator, 2026-10-18). Cards without a rule keep their default (the 3090 Ti: no cap measured
yet). A planned value outside the card's min/max range is clamped.

Override in rig-local `brain/hw/power.json`: {"by_model": {"GTX 1060": {"156": 120, "120": 100}},
"by_index": {"0": 400}}. A by_model value is watts, or watts per card default limit; by_index wins.
A temporary override until the next boot goes in /run/rig/power_override.json (same shape, read on
top): /run is cleared at boot, so a crash or reboot brings the caps back (used for stress tests).
The rig guard reads the resulting limits every sample: its power warn line is their sum.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import os

LOCAL = Path(__file__).resolve().parents[2]
# Rig-local hardware state, whichever copy of this file runs (rig checkout or the shared drive's copy).
_HW = Path(os.getenv("RIG_HW_DIR") or "/home/bryan/llm_orchestration/shared/brain/hw")
CONFIG_PATH = (_HW if _HW.is_dir() else LOCAL / "brain" / "hw") / "power.json"
OVERRIDE_PATH = Path(os.getenv("RIG_POWER_OVERRIDE") or "/run/rig/power_override.json")
DEFAULT_RULES = {"by_model": {"GTX 1060": {"156": 120, "120": 100}}, "by_index": {}}


def load_rules(path: Path = CONFIG_PATH, override: Path | None = None) -> dict:
    rules = json.loads(json.dumps(DEFAULT_RULES))
    for f in (path, OVERRIDE_PATH if override is None else override):
        if f.is_file():
            for key, value in json.loads(f.read_text(encoding="utf-8")).items():
                if isinstance(value, dict) and isinstance(rules.get(key), dict):
                    rules[key].update(value)
    return rules


def cards() -> list[dict]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,power.limit,power.default_limit,power.min_limit,power.max_limit",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=True).stdout
    rows = []
    for line in out.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) == 6 and p[0].isdigit():
            rows.append({"index": int(p[0]), "name": p[1], "limit_w": float(p[2]), "default_w": float(p[3]),
                         "min_w": float(p[4]), "max_w": float(p[5])})
    return rows


def planned(card: dict, rules: dict) -> float:
    """The limit this card should run at (pure)."""
    want = rules.get("by_index", {}).get(str(card["index"]))
    if want is None:
        rule = next((w for model, w in rules.get("by_model", {}).items() if model in card["name"]), None)
        if isinstance(rule, dict):  # watts per factory default limit (overclocked boards differ)
            rule = rule.get(f"{card['default_w']:.0f}")
        want = rule
    if want is None:
        want = card["default_w"]
    return max(card["min_w"], min(card["max_w"], float(want)))


def cmd_show(args) -> int:
    rules = load_rules()
    rows = [{**c, "planned_w": planned(c, rules)} for c in cards()]
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for r in rows:
            print(f"GPU{r['index']} {r['name']:<28} now {r['limit_w']:.0f} W  planned {r['planned_w']:.0f} W  "
                  f"(default {r['default_w']:.0f}, range {r['min_w']:.0f}-{r['max_w']:.0f})")
        print(f"sum planned {sum(r['planned_w'] for r in rows):.0f} W (the guard's power warn line)")
    return 0


def cmd_apply(args) -> int:
    rules, failed = load_rules(), 0
    for c in cards():
        want = planned(c, rules)
        if abs(c["limit_w"] - want) < 0.5:
            print(f"GPU{c['index']} {c['name']}: {want:.0f} W (already)")
            continue
        cmd = ["nvidia-smi", "-i", str(c["index"]), "-pl", f"{want:.0f}"]
        if args.dry_run:
            print("would run: " + " ".join(cmd))
            continue
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        ok = proc.returncode == 0
        failed += not ok
        print(f"GPU{c['index']} {c['name']}: {c['limit_w']:.0f} -> {want:.0f} W {'ok' if ok else 'FAILED'}"
              + ("" if ok else f": {(proc.stdout + proc.stderr).strip()[-200:]}"))
    return 1 if failed else 0


def mismatches(gpus: str, rules: dict) -> list[str]:
    wanted = None if gpus == "all" else {int(g) for g in gpus.split(",") if g.strip()}
    return [f"GPU{c['index']} {c['name']} at {c['limit_w']:.0f} W, planned {planned(c, rules):.0f} W"
            for c in cards() if (wanted is None or c["index"] in wanted)
            and abs(c["limit_w"] - planned(c, rules)) >= 0.5]


def cmd_check(args) -> int:
    rules = load_rules()
    try:
        bad = mismatches(args.gpus, rules)
    except (OSError, subprocess.SubprocessError) as exc:
        # Same rule as the guard: no reading admits with a warning (a broken driver fails the load anyway).
        print(f"warning: cannot read GPU power limits ({exc}); not checked", file=sys.stderr)
        return 0
    if bad and args.repair and os.getenv("HW_POWER_REPAIR", "on") != "off":
        print("power limits off plan (" + "; ".join(bad) + "); re-applying with rig-power.service", file=sys.stderr)
        subprocess.run(["sudo", "-n", "systemctl", "restart", "rig-power.service"], capture_output=True,
                       text=True, timeout=60)
        bad = mismatches(args.gpus, rules)
    if bad:
        print("power limits off plan: " + "; ".join(bad) + ". Fix: sudo systemctl restart rig-power.service "
              "(scripts/rig/hw_power.py)", file=sys.stderr)
        return 2
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("apply"); s.add_argument("--dry-run", action="store_true"); s.set_defaults(fn=cmd_apply)
    s = sub.add_parser("show"); s.add_argument("--json", action="store_true"); s.set_defaults(fn=cmd_show)
    s = sub.add_parser("check"); s.add_argument("--gpus", required=True); s.add_argument("--repair", action="store_true")
    s.set_defaults(fn=cmd_check)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
