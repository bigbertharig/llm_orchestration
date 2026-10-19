#!/home/bryan/llm-orchestration-venv/bin/python
"""Rig hardware guard: always-on power, thermal and memory watch, and load admission. Rig-side.

    hw_guard.py run                 the service loop (rig-guard.service)
    hw_guard.py status [--json]     current level, reasons, latest sample
    hw_guard.py admit [--wait S]    exit 0 when a model load may start; waits while level is warn

Every 2 s it samples CPU package temperature, per-GPU temperature, power (instant draw, which catches
the spikes the averaged reading misses) and utilization, plus RAM and swap. The worst reading over
the last window sets the level:

    ok        loads may start
    warn      loads wait (a new load adds power, heat and memory)
    critical  loads are refused; the event log gets a process snapshot

A level drops only after the readings stay below it for `clear_samples` samples. For power and
temperature the guard gates new loads only (stopping or throttling running work would change
benchmark measurements). For memory it also acts: when available RAM reads critical, it stops the
llama runtime holding the most host RAM, at most once per cooldown. On 2026-10-17 RAM filled with
nothing freeing it (earlyoom had been disarmed by extra swap) and the rig hung until a hard reset;
losing one runtime is the better outcome. Limits are in DEFAULT_LIMITS, overridable in the
rig-local guard.json. Total GPU power follows the cards' own limits (`nvidia-smi` power.limit, read
every sample): warn at their sum, the most the cards are set to draw together, critical 10% above it
(spikes past the limits). Explicit warn/critical numbers in guard.json replace that. The rig runs this whether or not the controller (the Pi) is reachable;
`rig observe` reports the state for the controller's benefit.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path

LOCAL = Path(__file__).resolve().parents[2]
# The rig's hardware state lives in one rig-local place, whichever copy of this file runs (the rig
# checkout, or the shared drive's copy that run_runtime.sh callers use). RIG_HW_DIR overrides.
_RIG_LOCAL_HW = Path("/home/bryan/llm_orchestration/shared/brain/hw")
HW_DIR = Path(os.getenv("RIG_HW_DIR") or (_RIG_LOCAL_HW if _RIG_LOCAL_HW.is_dir() else LOCAL / "brain" / "hw"))
STATE_PATH = Path(os.getenv("RIG_GUARD_STATE", str(HW_DIR / "guard_state.json")))
EVENTS_PATH = HW_DIR / "guard_events.log"
LIMITS_PATH = HW_DIR / "guard.json"
sys.path.insert(0, str(LOCAL / "scripts"))
from startup_thermal_monitor import cpu_temperatures, process_snapshot  # noqa: E402  (same sensors)

# Resets on 2026-10-15/16 came at 650-750 W sampled GPU draw (3090 busy plus 1060s, dual-PSU wiring
# suspect). The PSU wiring was fixed after that (operator, 2026-10-18), so power now follows the cards'
# limits; 650/750 W remain only as the fallback when the limits cannot be read.
# The startup monitor aborts at CPU package 95 C; earlyoom kills near 10% free memory.
DEFAULT_LIMITS = {
    "interval_s": 2.0,
    "window_samples": 5,          # worst reading over the last 10 s sets the level
    "clear_samples": 5,           # readings must stay below a level this long before it drops
    "stale_after_s": 15,          # admission treats an older state as "guard not running"
    "fail_closed": False,         # guard not running: False admits loads (with a warning), True refuses
    # from_limits: warn = sum of the cards' power limits, critical = that + critical_over_limits_pct.
    # warn/critical here are the fallback (limits unreadable) or, set in guard.json, fixed lines.
    "gpu_power_total_w": {"from_limits": True, "critical_over_limits_pct": 10, "warn": 650, "critical": 750},
    "gpu_temp_c": {"warn": 83, "critical": 90},
    "cpu_package_c": {"warn": 90, "critical": 95},
    "mem_available_mb": {"warn": 4096, "critical": 2048},   # lower is worse
    "memory_action": "stop_largest_runtime",   # or "hold_only"
    "memory_action_cooldown_s": 60,            # let RAM recover before stopping another runtime
}
LEVELS = ("ok", "warn", "critical")


def load_limits(path: Path = LIMITS_PATH) -> dict:
    limits = json.loads(json.dumps(DEFAULT_LIMITS))
    if path.is_file():
        for key, value in json.loads(path.read_text(encoding="utf-8")).items():
            if isinstance(value, dict) and isinstance(limits.get(key), dict):
                if key == "gpu_power_total_w" and ("warn" in value or "critical" in value):
                    limits[key]["from_limits"] = False  # fixed lines configured: they win
                limits[key].update(value)
            else:
                limits[key] = value
    return limits


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


# ------------------------------------------------------------------ sampling

def _num(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def gpu_sample() -> list[dict]:
    fields = "index,temperature.gpu,power.draw,power.limit,power.draw.instant,utilization.gpu"
    for query in (fields, fields.replace(",power.draw.instant", "")):
        try:
            out = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5, check=True).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        gpus = []
        for line in out.splitlines():
            p = [x.strip() for x in line.split(",")]
            if not p or not p[0].isdigit():
                continue
            instant = _num(p[4]) if len(p) == 6 else None
            gpus.append({"index": int(p[0]), "temp_c": _num(p[1]), "power_w": _num(p[2]),
                         "power_limit_w": _num(p[3]), "power_instant_w": instant, "util_pct": _num(p[-1])})
        if gpus:
            return gpus
    return []


def memory_sample() -> dict:
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        info[key] = int(rest.split()[0]) // 1024 if rest.split() else 0
    return {"mem_available_mb": info.get("MemAvailable"), "mem_total_mb": info.get("MemTotal"),
            "swap_free_mb": info.get("SwapFree"), "swap_total_mb": info.get("SwapTotal")}


def take_sample() -> dict:
    package_c, _ = cpu_temperatures()
    gpus = gpu_sample()
    power = [g["power_instant_w"] if g["power_instant_w"] is not None else g["power_w"] for g in gpus]
    return {"at": _now(), "cpu_package_c": package_c, "gpus": gpus,
            "gpu_power_total_w": round(sum(p for p in power if p is not None), 1) if gpus else None,
            "gpu_power_limit_total_w": (round(sum(g["power_limit_w"] for g in gpus), 1)
                                        if gpus and all(g.get("power_limit_w") for g in gpus) else None),
            "gpu_temp_max_c": max((g["temp_c"] for g in gpus if g["temp_c"] is not None), default=None),
            **memory_sample()}


# ------------------------------------------------------------------ evaluation (pure)

def power_lines(sample: dict, limits: dict) -> dict:
    """{warn, critical} for total GPU power: from the cards' limits unless fixed numbers are configured."""
    lim = limits["gpu_power_total_w"]
    total = sample.get("gpu_power_limit_total_w")
    if lim.get("from_limits") and total:
        return {"warn": total, "critical": round(total * (1 + lim.get("critical_over_limits_pct", 10) / 100), 1)}
    return {"warn": lim["warn"], "critical": lim["critical"]}


def sample_level(sample: dict, limits: dict) -> tuple[str, list[str]]:
    """Level of one sample, with the readings that set it."""
    level, reasons = "ok", []

    def bump(new: str, why: str) -> None:
        nonlocal level
        reasons.append(why)
        if LEVELS.index(new) > LEVELS.index(level):
            level = new

    for key, label in (("gpu_power_total_w", "GPU power {v} W"), ("gpu_temp_max_c", "GPU temp {v} C"),
                       ("cpu_package_c", "CPU package {v} C")):
        value = sample.get(key)
        lim = power_lines(sample, limits) if key == "gpu_power_total_w" else \
            limits["gpu_temp_c" if key == "gpu_temp_max_c" else key]
        if value is None:
            continue
        for name in ("critical", "warn"):
            if value >= lim[name]:
                bump(name, label.format(v=value) + f" >= {name} {lim[name]}")
                break
    mem, lim = sample.get("mem_available_mb"), limits["mem_available_mb"]
    if mem is not None:
        for name in ("critical", "warn"):
            if mem <= lim[name]:
                bump(name, f"available memory {mem} MB <= {name} {lim[name]}")
                break
    return level, reasons


class Evaluator:
    """Worst level over a window, dropping a level only after `clear_samples` calm samples."""

    def __init__(self, limits: dict):
        self.limits = limits
        self.window: deque = deque(maxlen=int(limits["window_samples"]))
        self.level = "ok"
        self.calm = 0

    def push(self, sample: dict) -> tuple[str, list[str]]:
        self.window.append(sample_level(sample, self.limits))
        worst = max((lvl for lvl, _ in self.window), key=LEVELS.index)
        reasons = next(rs for lvl, rs in reversed(self.window) if lvl == worst)  # latest reading at that level
        if LEVELS.index(worst) >= LEVELS.index(self.level):
            self.level, self.calm = worst, 0
        else:
            self.calm += 1
            if self.calm >= int(self.limits["clear_samples"]):
                self.level, self.calm = worst, 0
        return self.level, reasons


# ------------------------------------------------------------------ memory action

def memory_action_due(sample: dict, limits: dict, last_action_ts: float, now: float) -> bool:
    """Act on the current reading (not the window): RAM is critical right now and the cooldown passed."""
    mem = sample.get("mem_available_mb")
    return (limits.get("memory_action") == "stop_largest_runtime" and mem is not None
            and mem <= limits["mem_available_mb"]["critical"]
            and now - last_action_ts >= float(limits["memory_action_cooldown_s"]))


def runtime_memory() -> list[dict]:
    """Every llama-server on the rig, largest host RAM (anonymous + swap) first, with its container."""
    found = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            if (proc / "comm").read_text().strip() != "llama-server":
                continue
            status, cgroup = (proc / "status").read_text(), (proc / "cgroup").read_text()
        except OSError:
            continue

        def kb(key: str) -> int:
            m = re.search(rf"^{key}:\s+(\d+)", status, re.M)
            return int(m.group(1)) if m else 0

        cid = re.search(r"docker-([0-9a-f]{12,64})\.scope|/docker/([0-9a-f]{12,64})", cgroup)
        found.append({"pid": int(proc.name), "host_mb": (kb("RssAnon") + kb("VmSwap")) // 1024,
                      "container": (cid.group(1) or cid.group(2)) if cid else None})
    return sorted(found, key=lambda r: -r["host_mb"])


def stop_largest_runtime() -> str:
    candidates = [r for r in runtime_memory() if r["container"]]
    if not candidates:
        return "no llama runtime container to stop"
    target = candidates[0]
    name = subprocess.run(["docker", "inspect", "-f", "{{.Name}}", target["container"]],
                          capture_output=True, text=True, timeout=15).stdout.strip().lstrip("/")
    proc = subprocess.run(["docker", "stop", "-t", "5", target["container"]],
                          capture_output=True, text=True, timeout=60)
    others = ", ".join(f"{r['pid']}:{r['host_mb']}MB" for r in candidates[1:]) or "none"
    result = "stopped" if proc.returncode == 0 else f"FAILED to stop ({proc.stderr.strip()[:200]})"
    return f"{result} {name or target['container'][:12]} (pid {target['pid']}, {target['host_mb']} MB host RAM); others: {others}"


# ------------------------------------------------------------------ state and admission

def read_state(path: Path = STATE_PATH) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def admission(limits: dict | None = None, state_path: Path = STATE_PATH) -> tuple[bool, str, str]:
    """(may a load start now, level, why). Stale or missing state follows `fail_closed`."""
    limits = limits or load_limits()
    st = read_state(state_path)
    age = None if st is None else time.time() - float(st.get("updated_ts", 0))
    if st is None or age is None or age > float(limits["stale_after_s"]):
        why = "guard not running (no fresh guard_state.json)"
        return (not limits["fail_closed"], "unknown", why)
    level = st.get("level", "unknown")
    return (level == "ok", level, "; ".join(st.get("reasons", [])) or level)


def run(limits: dict) -> int:
    ev = Evaluator(limits)
    since, last = _now(), None
    last_action_ts, last_action = 0.0, None
    while True:
        started = time.time()
        sample = take_sample()
        level, reasons = ev.push(sample)
        if memory_action_due(sample, limits, last_action_ts, started):
            last_action_ts = started
            last_action = f"{_now()} {stop_largest_runtime()}"
            HW_DIR.mkdir(parents=True, exist_ok=True)
            with EVENTS_PATH.open("a", encoding="utf-8") as fh:
                fh.write(f"{last_action} [available memory {sample.get('mem_available_mb')} MB]\n")
                fh.write(process_snapshot() + "\n\n")
        if level != last:
            since = _now()
            HW_DIR.mkdir(parents=True, exist_ok=True)
            with EVENTS_PATH.open("a", encoding="utf-8") as fh:
                fh.write(f"{since} level {last or 'start'} -> {level}: {'; '.join(reasons) or 'all readings in range'}\n")
                if level == "critical":
                    fh.write(process_snapshot() + "\n\n")
            last = level
        _write_json(STATE_PATH, {"level": level, "since": since, "reasons": reasons, "sample": sample,
                                 "last_memory_action": last_action,
                                 "updated_at": _now(), "updated_ts": time.time(),
                                 "limits": {k: v for k, v in limits.items() if isinstance(v, dict)}})
        time.sleep(max(0.2, float(limits["interval_s"]) - (time.time() - started)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    s = sub.add_parser("status")
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("admit")
    s.add_argument("--wait", type=float, default=0, help="seconds to wait while the level is warn")
    args = ap.parse_args()
    limits = load_limits()
    if args.cmd == "run":
        return run(limits)
    if args.cmd == "status":
        st = read_state()
        ok, level, why = admission(limits)
        if args.json:
            print(json.dumps({"admit": ok, "level": level, "why": why, "state": st}, indent=2))
        else:
            smp = (st or {}).get("sample", {})
            print(f"guard {level} (loads {'admitted' if ok else 'held'}): {why}")
            if smp:
                print(f"  GPU {smp.get('gpu_power_total_w')} W total, max {smp.get('gpu_temp_max_c')} C; "
                      f"CPU {smp.get('cpu_package_c')} C; {smp.get('mem_available_mb')} MB available")
        return 0
    deadline = time.time() + args.wait
    while True:
        ok, level, why = admission(limits)
        if ok:
            if level == "unknown":
                print(f"guard: admitting without a guard: {why}", file=sys.stderr)
            return 0
        if level == "critical" or time.time() >= deadline:
            print(f"guard {level}: load not admitted: {why}", file=sys.stderr)
            return 3
        time.sleep(2)


if __name__ == "__main__":
    raise SystemExit(main())
