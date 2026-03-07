#!/usr/bin/env python3
"""Preflight scan worker heartbeats against live Ollama port state.

If requested, this can trigger a return-to-default reset and wait for the rig
to stabilize before allowing a plan submission to continue.
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
TRANSITIONAL_RUNTIME_STATES = {
    "loading_single",
    "loading_split",
    "unloading",
    "recovering_single",
    "recovering_split",
}


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


def _load_gpu_specs(config: dict) -> list[dict]:
    specs: list[dict] = []
    for gpu in config.get("gpus", []) or []:
        if not isinstance(gpu, dict):
            continue
        if gpu.get("id") is None or gpu.get("port") is None:
            continue
        specs.append(
            {
                "id": int(gpu["id"]),
                "name": str(gpu.get("name") or f"gpu-{gpu['id']}"),
                "port": int(gpu["port"]),
            }
        )
    return specs


def _load_heartbeat(gpu_id: int) -> tuple[dict | None, str | None, float | None]:
    hb_path = SHARED_DIR / "gpus" / f"gpu_{gpu_id}" / "heartbeat.json"
    if not hb_path.exists():
        return None, "missing", None
    try:
        with open(hb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None, "invalid_json_object", None
        ts = _heartbeat_time_seconds(data, hb_path.stat().st_mtime)
        return data, None, ts
    except Exception as exc:
        return None, str(exc), None


def _fetch_live_port_models(ports: list[int], timeout_s: int) -> tuple[dict[int, dict], str | None]:
    port_json = json.dumps([int(p) for p in ports])
    remote_script = (
        "python3 - <<'PY'\n"
        "import json, subprocess\n"
        f"ports = {port_json}\n"
        "out = {}\n"
        "for port in ports:\n"
        "    url = f'http://localhost:{port}/api/ps'\n"
        "    try:\n"
        "        proc = subprocess.run(['curl', '-sS', '--max-time', '2', url], capture_output=True, text=True, timeout=5)\n"
        "    except Exception as exc:\n"
        "        out[str(port)] = {'ok': False, 'error': str(exc), 'models': []}\n"
        "        continue\n"
        "    if proc.returncode != 0:\n"
        "        out[str(port)] = {'ok': False, 'error': (proc.stderr or '').strip() or f'curl_rc={proc.returncode}', 'models': []}\n"
        "        continue\n"
        "    raw = (proc.stdout or '').strip()\n"
        "    try:\n"
        "        data = json.loads(raw) if raw else {}\n"
        "    except Exception as exc:\n"
        "        out[str(port)] = {'ok': False, 'error': f'invalid_json:{exc}', 'models': [], 'raw': raw[:400]}\n"
        "        continue\n"
        "    models = []\n"
        "    for item in (data.get('models') or []):\n"
        "        if isinstance(item, dict) and item.get('name'):\n"
        "            models.append(str(item.get('name')))\n"
        "    out[str(port)] = {'ok': True, 'error': None, 'models': models}\n"
        "print(json.dumps(out))\n"
        "PY\n"
    )
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "gpu", "bash", "-s"],
        input=remote_script,
        capture_output=True,
        text=True,
        timeout=max(10, timeout_s),
    )
    if proc.returncode != 0:
        return {}, (proc.stderr or "").strip() or f"ssh_rc={proc.returncode}"
    stdout = (proc.stdout or "").strip()
    if not stdout:
        return {}, "empty_remote_output"
    try:
        payload = json.loads(stdout.splitlines()[-1])
    except Exception as exc:
        return {}, f"invalid_remote_json:{exc}"
    live: dict[int, dict] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            try:
                live[int(key)] = value if isinstance(value, dict) else {"ok": False, "error": "invalid_port_payload", "models": []}
            except Exception:
                continue
    return live, None


def _processing_task_count() -> int:
    processing_dir = SHARED_DIR / "tasks" / "processing"
    try:
        return sum(
            1
            for path in processing_dir.glob("*.json")
            if path.is_file() and not path.name.endswith(".heartbeat.json")
        )
    except Exception:
        return 0


def _analyze_once(specs: list[dict], stale_seconds: int, remote_timeout: int) -> dict:
    now = time.time()
    live_ports, remote_error = _fetch_live_port_models([spec["port"] for spec in specs], timeout_s=remote_timeout)
    issues: list[dict] = []
    workers: list[dict] = []

    if remote_error:
        issues.append({"scope": "system", "reason": "remote_live_check_failed", "detail": remote_error})

    processing_count = _processing_task_count()
    if processing_count > 0:
        issues.append({"scope": "system", "reason": "processing_tasks_present", "detail": processing_count})

    for spec in specs:
        hb, hb_error, hb_ts = _load_heartbeat(spec["id"])
        live_info = live_ports.get(spec["port"], {"ok": False, "error": "missing_live_result", "models": []})
        live_models = list(live_info.get("models") or [])
        worker_summary = {
            "gpu_id": spec["id"],
            "name": spec["name"],
            "port": spec["port"],
            "heartbeat_ok": hb is not None,
            "live_ok": bool(live_info.get("ok")),
            "live_models": live_models,
        }
        if hb is None:
            worker_summary["status"] = "issue"
            worker_summary["issue"] = hb_error
            issues.append({"scope": spec["name"], "reason": "heartbeat_missing_or_invalid", "detail": hb_error})
            workers.append(worker_summary)
            continue

        age = max(0.0, now - float(hb_ts or now))
        runtime_state = str(hb.get("runtime_state") or "")
        model_loaded = bool(hb.get("model_loaded"))
        state = str(hb.get("state") or "")
        active_tasks = list(hb.get("active_tasks") or [])
        meta_active = bool(hb.get("meta_task_active"))
        worker_summary.update(
            {
                "heartbeat_age_seconds": round(age, 1),
                "state": state,
                "runtime_state": runtime_state,
                "model_loaded": model_loaded,
                "loaded_model": hb.get("loaded_model"),
                "active_tasks": active_tasks,
                "meta_task_active": meta_active,
            }
        )
        if age > stale_seconds:
            issues.append({"scope": spec["name"], "reason": "stale_heartbeat", "detail": round(age, 1)})
        if runtime_state in TRANSITIONAL_RUNTIME_STATES and not active_tasks and not meta_active:
            issues.append({"scope": spec["name"], "reason": "transitional_state_without_work", "detail": runtime_state})

        heartbeat_expects_runtime = model_loaded or runtime_state.startswith("ready") or state == "hot"
        heartbeat_expects_cold = (runtime_state == "cold") and (not model_loaded) and state == "cold"

        if heartbeat_expects_runtime and not live_models:
            issues.append(
                {
                    "scope": spec["name"],
                    "reason": "heartbeat_hot_but_port_empty",
                    "detail": {
                        "runtime_state": runtime_state,
                        "state": state,
                        "loaded_model": hb.get("loaded_model"),
                    },
                }
            )
        elif heartbeat_expects_cold and live_models:
            issues.append(
                {
                    "scope": spec["name"],
                    "reason": "heartbeat_cold_but_port_loaded",
                    "detail": {
                        "runtime_state": runtime_state,
                        "state": state,
                        "live_models": live_models,
                    },
                }
            )

        worker_summary["status"] = "ok"
        workers.append(worker_summary)

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "workers": workers,
        "processing_count": processing_count,
    }


def _run_return_default(timeout_s: int) -> tuple[bool, dict]:
    script_path = BASE_DIR / "scripts" / "return_default.py"
    proc = subprocess.run(
        [sys.executable, str(script_path), "--json", "--timeout", str(timeout_s), "--wait", str(min(timeout_s, 90))],
        capture_output=True,
        text=True,
        timeout=max(timeout_s + 15, 30),
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
                parsed = candidate
        except Exception:
            pass
    return proc.returncode == 0, parsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan workers for heartbeat/live-state mismatches")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Path to shared/agents/config.json")
    parser.add_argument("--heartbeat-stale-seconds", type=int, default=90, help="Heartbeat freshness threshold")
    parser.add_argument("--remote-timeout", type=int, default=30, help="Live API check timeout seconds")
    parser.add_argument("--fix", action="store_true", help="If issues are found, run return_default.py and rescan")
    parser.add_argument("--fix-timeout", type=int, default=240, help="Reset timeout seconds when --fix is used")
    parser.add_argument("--post-fix-wait", type=int, default=120, help="Seconds to wait for a clean rescan after reset")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary")
    args = parser.parse_args()

    config = _load_json(Path(args.config))
    specs = _load_gpu_specs(config)
    initial = _analyze_once(specs, stale_seconds=args.heartbeat_stale_seconds, remote_timeout=args.remote_timeout)
    result = {
        "ok": bool(initial.get("ok")),
        "fixed": False,
        "initial": initial,
        "after_fix": None,
        "reset": None,
    }

    if (not initial.get("ok")) and args.fix:
        reset_ok, reset_summary = _run_return_default(timeout_s=args.fix_timeout)
        result["reset"] = reset_summary
        if reset_ok:
            deadline = time.time() + max(5, args.post_fix_wait)
            while time.time() < deadline:
                followup = _analyze_once(specs, stale_seconds=args.heartbeat_stale_seconds, remote_timeout=args.remote_timeout)
                result["after_fix"] = followup
                if followup.get("ok"):
                    result["ok"] = True
                    result["fixed"] = True
                    break
                time.sleep(3)
        else:
            result["after_fix"] = _analyze_once(specs, stale_seconds=args.heartbeat_stale_seconds, remote_timeout=args.remote_timeout)

    if result["after_fix"] is not None and not result["after_fix"].get("ok"):
        result["ok"] = False

    result["message"] = (
        "Worker preflight clean"
        if result["ok"] and not result["fixed"]
        else "Worker preflight fixed by reset"
        if result["ok"] and result["fixed"]
        else "Worker preflight failed"
    )

    if args.json:
        print(json.dumps(result))
    else:
        print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
