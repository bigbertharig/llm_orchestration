#!/usr/bin/env python3
"""Return the rig to its default startup state.

This is the reusable local wrapper for the existing dashboard "return default"
behavior. It clears worker/split Ollama runtimes, restarts the remote agents,
and waits for GPU heartbeats to return.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SHARED_DIR = BASE_DIR / "shared"
CONFIG_PATH = SHARED_DIR / "agents" / "config.json"

sys.path.insert(0, str(SHARED_DIR / "agents"))
from brain_core import resolve_auto_default_target


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return data


def _heartbeat_time_seconds(data: dict, fallback_mtime: float) -> float:
    for key in ("last_updated", "runtime_state_updated_at"):
        raw = data.get(key)
        if not raw:
            continue
        try:
            return datetime.fromisoformat(str(raw)).timestamp()
        except Exception:
            continue
    return fallback_mtime


def _wait_for_heartbeats(shared_dir: Path, gpu_ids: list[int], timeout_s: int, stale_seconds: int) -> dict:
    deadline = time.time() + max(1, timeout_s)
    last_missing: list[str] = []
    last_stale: list[str] = []
    while time.time() < deadline:
        now = time.time()
        missing: list[str] = []
        stale: list[str] = []
        for gpu_id in gpu_ids:
            hb_path = shared_dir / "gpus" / f"gpu_{gpu_id}" / "heartbeat.json"
            if not hb_path.exists():
                missing.append(f"gpu-{gpu_id}:missing")
                continue
            try:
                with open(hb_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                missing.append(f"gpu-{gpu_id}:unreadable:{exc}")
                continue
            ts = _heartbeat_time_seconds(data if isinstance(data, dict) else {}, hb_path.stat().st_mtime)
            age = now - ts
            if age > stale_seconds:
                stale.append(f"gpu-{gpu_id}:{age:.1f}s")
        if not missing and not stale:
            return {
                "ok": True,
                "missing": [],
                "stale": [],
            }
        last_missing = missing
        last_stale = stale
        time.sleep(2)
    return {
        "ok": False,
        "missing": last_missing,
        "stale": last_stale,
    }


def _read_gpu_heartbeat(shared_dir: Path, gpu_name: str) -> dict | None:
    try:
        gpu_id = int(str(gpu_name).replace("gpu-", ""))
    except Exception:
        return None
    hb_path = shared_dir / "gpus" / f"gpu_{gpu_id}" / "heartbeat.json"
    if not hb_path.exists():
        return None
    try:
        with open(hb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _wait_for_default_state(shared_dir: Path, config: dict, timeout_s: int, stale_seconds: int) -> dict:
    deadline = time.time() + max(1, timeout_s)
    auto_default_gpu, auto_default_model = resolve_auto_default_target(config)
    initial_hot_workers = max(0, int(config.get("initial_hot_workers", 0) or 0))
    last_result = {
        "ok": False,
        "reason": "timeout",
        "default_gpu": auto_default_gpu,
        "default_model": auto_default_model,
        "split_loaded": [],
        "hot_workers": [],
    }

    gpu_names = []
    for gpu in config.get("gpus", []) or []:
        if not isinstance(gpu, dict):
            continue
        gpu_name = str(gpu.get("name") or f"gpu-{gpu.get('id')}").strip()
        if gpu_name:
            gpu_names.append(gpu_name)

    while time.time() < deadline:
        now = time.time()
        split_loaded: list[str] = []
        hot_workers: list[str] = []
        default_ready = initial_hot_workers == 0
        stale_default = False

        for gpu_name in gpu_names:
            hb = _read_gpu_heartbeat(shared_dir, gpu_name)
            if not hb:
                if gpu_name == auto_default_gpu:
                    stale_default = True
                continue
            runtime_updated = _heartbeat_time_seconds(hb, now - stale_seconds - 1)
            if now - runtime_updated > stale_seconds:
                if gpu_name == auto_default_gpu:
                    stale_default = True
                continue
            placement = str(hb.get("runtime_placement", "")).strip()
            loaded_model = str(hb.get("loaded_model", "")).strip()
            model_loaded = bool(hb.get("model_loaded", False))
            if placement == "split_gpu":
                split_loaded.append(gpu_name)
            if model_loaded:
                hot_workers.append(gpu_name)
            if gpu_name == auto_default_gpu and initial_hot_workers > 0:
                default_ready = model_loaded and loaded_model == auto_default_model and placement != "split_gpu"

        ok = not split_loaded and not stale_default and default_ready
        last_result = {
            "ok": ok,
            "reason": "" if ok else ("default_gpu_stale" if stale_default else "not_default_ready"),
            "default_gpu": auto_default_gpu,
            "default_model": auto_default_model,
            "split_loaded": split_loaded,
            "hot_workers": hot_workers,
        }
        if ok:
            return last_result
        time.sleep(2)

    return last_result


def _run_clear_ollama(timeout_s: int) -> tuple[bool, dict]:
    script_path = BASE_DIR / "scripts" / "clear_ollama.py"
    proc = subprocess.run(
        [sys.executable, str(script_path), "--json"],
        capture_output=True,
        text=True,
        timeout=max(10, timeout_s),
    )
    parsed: dict = {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "").strip(),
        "stderr": (proc.stderr or "").strip(),
    }
    if proc.stdout:
        try:
            candidate = json.loads(proc.stdout.strip().splitlines()[-1])
            if isinstance(candidate, dict):
                parsed.update(candidate)
        except Exception:
            pass
    return proc.returncode == 0, parsed


def _run_remote_restart(timeout_s: int) -> tuple[bool, dict]:
    remote_script = (
        "set -e\n"
        "pkill -f /mnt/shared/agents/brain.py || true\n"
        "pkill -f /mnt/shared/agents/gpu.py || true\n"
        "pkill -f /mnt/shared/agents/startup.py || true\n"
        "sleep 2\n"
        "nohup /home/bryan/ml-env/bin/python /mnt/shared/agents/startup.py "
        "--config /mnt/shared/agents/config.json "
        ">> /mnt/shared/logs/startup-manual.log 2>&1 < /dev/null &\n"
        "sleep 4\n"
        "pgrep -af startup.py || true\n"
        "pgrep -af brain.py || true\n"
        "pgrep -af gpu.py || true\n"
    )
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "gpu", "bash", "-s"],
        input=remote_script,
        capture_output=True,
        text=True,
        timeout=max(20, timeout_s),
    )
    return (
        proc.returncode == 0,
        {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "").strip(),
            "stderr": (proc.stderr or "").strip(),
            "cmd": "ssh gpu bash -s (stdin: kill agents, restart startup.py)",
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Return rig to default startup state")
    parser.add_argument("--timeout", type=int, default=240, help="Remote restart timeout seconds per restart attempt")
    parser.add_argument("--wait", type=int, default=30, help="Heartbeat wait timeout seconds per restart attempt")
    parser.add_argument("--heartbeat-stale-seconds", type=int, default=60, help="Heartbeat freshness threshold")
    parser.add_argument("--max-restarts", type=int, default=10, help="Maximum restart/rescan attempts before failing")
    parser.add_argument("--rescan-interval", type=int, default=30, help="Seconds to wait between failed restart attempts")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary")
    args = parser.parse_args()

    config = _load_json(CONFIG_PATH)
    gpu_ids: list[int] = []
    for gpu in config.get("gpus", []) or []:
        if isinstance(gpu, dict) and gpu.get("id") is not None:
            gpu_ids.append(int(gpu["id"]))

    clear_ok, clear_summary = _run_clear_ollama(timeout_s=min(args.timeout, 180))
    attempt_summaries: list[dict] = []
    restart_ok = False
    heartbeat_summary: dict = {"ok": False, "missing": [], "stale": []}

    for attempt in range(1, max(1, args.max_restarts) + 1):
        restart_ok, restart_summary = _run_remote_restart(timeout_s=args.timeout)
        heartbeat_summary = _wait_for_heartbeats(
            SHARED_DIR,
            gpu_ids,
            timeout_s=max(1, args.wait),
            stale_seconds=args.heartbeat_stale_seconds,
        )
        attempt_summary = {
            "attempt": attempt,
            "restart": restart_summary,
            "heartbeats": heartbeat_summary,
        }
        attempt_summaries.append(attempt_summary)
        if restart_ok and bool(heartbeat_summary.get("ok")):
            break
        if attempt < max(1, args.max_restarts):
            time.sleep(max(0, args.rescan_interval))

    default_summary = _wait_for_default_state(
        SHARED_DIR,
        config,
        timeout_s=max(1, args.wait),
        stale_seconds=args.heartbeat_stale_seconds,
    )
    final_attempt = attempt_summaries[-1] if attempt_summaries else {"restart": {"ok": False}, "heartbeats": {"ok": False}}
    ok = (
        clear_ok
        and bool(final_attempt.get("restart", {}).get("ok"))
        and bool(final_attempt.get("heartbeats", {}).get("ok"))
        and bool(default_summary.get("ok"))
    )
    result = {
        "ok": ok,
        "message": "Returned system to default startup state" if ok else "Failed to return system to default startup state",
        "clear_ollama": clear_summary,
        "restart": final_attempt.get("restart"),
        "heartbeats": final_attempt.get("heartbeats"),
        "default_state": default_summary,
        "restart_attempts": attempt_summaries,
    }

    if args.json:
        print(json.dumps(result))
    else:
        print(json.dumps(result, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
