#!/usr/bin/env python3
"""Brain watchdog: dynamic GPU scan, brain heartbeat, and stale-map recovery."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log_line(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{now_iso()} {message}\n")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def scan_gpu_ids() -> list[int]:
    # Reuse the same discovery path as setup.py/startup.py.
    agents_dir = Path("/mnt/shared/agents")
    if agents_dir.exists():
        sys.path.insert(0, str(agents_dir))
    try:
        from hardware import scan_gpus  # type: ignore
    except Exception:
        scan_gpus = None

    if scan_gpus:
        try:
            gpus = scan_gpus()
            ids = []
            for gpu in gpus:
                idx = gpu.get("index")
                if isinstance(idx, int):
                    ids.append(idx)
            return sorted(set(ids))
        except Exception:
            pass

    # Fallback only if shared hardware module is unavailable.
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except Exception:
        return []
    if p.returncode != 0:
        return []
    ids: list[int] = []
    for line in p.stdout.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            ids.append(int(s))
        except ValueError:
            continue
    return sorted(set(ids))


def scan_gpu_telemetry() -> tuple[dict[int, dict[str, Any]], int | None]:
    """Return live GPU telemetry and max CPU temp using shared hardware module when available."""
    telemetry: dict[int, dict[str, Any]] = {}
    max_cpu_c: int | None = None

    agents_dir = Path("/mnt/shared/agents")
    if agents_dir.exists():
        sys.path.insert(0, str(agents_dir))
    try:
        from hardware import scan_gpus, scan_cpu_temps  # type: ignore
    except Exception:
        scan_gpus = None
        scan_cpu_temps = None

    if scan_gpus:
        try:
            for g in scan_gpus():
                gid = g.get("index")
                if not isinstance(gid, int):
                    continue
                telemetry[gid] = {
                    "gpu_temp_c": g.get("temp_c"),
                    "power_w": g.get("power_w"),
                    "vram_total_mb": g.get("vram_mb"),
                }
        except Exception:
            pass

    if scan_cpu_temps:
        try:
            temps = scan_cpu_temps()
            vals = [int(t.get("temp_c")) for t in temps if isinstance(t, dict) and isinstance(t.get("temp_c"), int)]
            if vals:
                max_cpu_c = max(vals)
        except Exception:
            pass

    # Fill util + used memory via nvidia-smi when available.
    try:
        p = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if p.returncode == 0:
            for line in p.stdout.splitlines():
                parts = [x.strip() for x in line.split(",")]
                if len(parts) < 6:
                    continue
                try:
                    gid = int(parts[0])
                except Exception:
                    continue
                row = telemetry.setdefault(gid, {})
                try:
                    row["gpu_util"] = int(float(parts[1]))
                except Exception:
                    pass
                try:
                    row["vram_used_mb"] = int(float(parts[2]))
                except Exception:
                    pass
                try:
                    row["vram_total_mb"] = int(float(parts[3]))
                except Exception:
                    pass
                try:
                    row["power_w"] = float(parts[4])
                except Exception:
                    pass
                try:
                    row["gpu_temp_c"] = int(float(parts[5]))
                except Exception:
                    pass
    except Exception:
        pass

    return telemetry, max_cpu_c


def scan_loaded_models(runtime_host: str = "http://localhost:11434", runtime_backend: str = "llama") -> list[str]:
    out: list[str] = []
    backend = str(runtime_backend or "llama").strip()
    if backend == "llama":
        try:
            req = urllib.request.Request(f"{runtime_host}/v1/models")
            with urllib.request.urlopen(req, timeout=3) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 503:
                return ["_llama_loading"]
            return out
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            return out

        for model in payload.get("data", []) if isinstance(payload, dict) else []:
            if not isinstance(model, dict):
                continue
            name = model.get("id")
            if isinstance(name, str) and name.strip():
                out.append(name.strip())
    return sorted(set(out))


def list_live_worker_names(shared: Path, stale_seconds: int) -> list[str]:
    gpus_dir = shared / "gpus"
    now = datetime.now()
    out: list[str] = []
    for hb in sorted(gpus_dir.glob("gpu_*/heartbeat.json")):
        data = load_json(hb, {})
        ts = data.get("last_updated")
        if not isinstance(ts, str):
            continue
        try:
            age = int((now - datetime.fromisoformat(ts)).total_seconds())
        except Exception:
            continue
        if age <= stale_seconds:
            gid = data.get("gpu_id")
            if isinstance(gid, int):
                out.append(f"gpu-{gid}")
    return sorted(set(out))


def configured_workers(config: dict[str, Any]) -> list[dict[str, Any]]:
    items = config.get("gpus")
    if not isinstance(items, list):
        return []
    return [x for x in items if isinstance(x, dict)]


def _default_worker_model(config: dict[str, Any]) -> str:
    for g in configured_workers(config):
        m = g.get("model")
        if isinstance(m, str) and m.strip():
            return m.strip()
    return "qwen2.5:7b"


def _default_permissions(config: dict[str, Any]) -> str:
    for g in configured_workers(config):
        p = g.get("permissions")
        if isinstance(p, str) and p.strip():
            return p.strip()
    return "worker.json"


def _vram_for_gpu(config: dict[str, Any], gpu_id: int) -> int:
    for g in configured_workers(config):
        if g.get("id") == gpu_id and isinstance(g.get("vram_mb"), int):
            return int(g["vram_mb"])
    return 6144


def reconcile_config(config: dict[str, Any], detected_gpu_ids: list[int]) -> tuple[dict[str, Any], bool]:
    new_cfg = dict(config)
    brain_ids = set(new_cfg.get("brain", {}).get("gpus", []))
    worker_ids = [gid for gid in detected_gpu_ids if gid not in brain_ids]
    worker_ids = sorted(set(worker_ids))

    existing_by_id: dict[int, dict[str, Any]] = {}
    for g in configured_workers(new_cfg):
        gid = g.get("id")
        if isinstance(gid, int):
            existing_by_id[gid] = dict(g)

    model = _default_worker_model(new_cfg)
    permissions = _default_permissions(new_cfg)
    desired_workers: list[dict[str, Any]] = []
    for gid in worker_ids:
        current = existing_by_id.get(gid, {})
        merged = dict(current)
        merged["id"] = gid
        merged["name"] = f"gpu-{gid}"
        if not merged.get("model"):
            merged["model"] = model
        if not merged.get("permissions"):
            merged["permissions"] = permissions
        if not isinstance(merged.get("vram_mb"), int):
            merged["vram_mb"] = _vram_for_gpu(new_cfg, gid)
        merged["port"] = int(11434 + gid)
        desired_workers.append(merged)

    desired_workers = sorted(desired_workers, key=lambda x: int(x["id"]))
    new_cfg["gpus"] = desired_workers

    # Keep this explicit and dynamic, because brain defaults to 1 when missing.
    new_cfg["max_hot_workers"] = len(desired_workers)
    if "initial_hot_workers" in new_cfg:
        ihw = int(new_cfg.get("initial_hot_workers", 0))
        new_cfg["initial_hot_workers"] = min(max(0, ihw), len(desired_workers))

    changed = json.dumps(new_cfg, sort_keys=True) != json.dumps(config, sort_keys=True)
    return new_cfg, changed


def brain_known_workers(shared: Path) -> list[str]:
    state = load_json(shared / "brain" / "state.json", {})
    running = state.get("running_gpu_agents")
    if not isinstance(running, list):
        return []
    return sorted({str(x) for x in running if isinstance(x, str)})


def latest_brain_agent_counts(shared: Path) -> tuple[int | None, int | None]:
    """Read the latest monitor line and return (running, total) agent counts."""
    candidates = [
        shared / "logs" / "brain-manual.log",
        shared / "logs" / "brain_decisions.log",
    ]
    pattern = re.compile(r"Agents:\s*(\d+)\s*/\s*(\d+)")
    for path in candidates:
        if not path.exists():
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for line in reversed(lines[-400:]):
            if "MONITOR" not in line or "Agents:" not in line:
                continue
            m = pattern.search(line)
            if not m:
                continue
            return int(m.group(1)), int(m.group(2))
    return None, None


def find_brain_pids() -> list[int]:
    try:
        p = subprocess.run(
            ["pgrep", "-f", "agents/brain.py"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except Exception:
        return []
    if p.returncode != 0:
        return []
    out: list[int] = []
    for line in p.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(int(line))
        except ValueError:
            continue
    return sorted(set(out))


def restart_brain(config_path: Path, shared: Path, dry_run: bool) -> bool:
    pids = find_brain_pids()
    log_path = shared / "logs" / "brain_watchdog.log"

    if dry_run:
        log_line(log_path, f"DRY RUN restart brain pids={pids}")
        return True

    for pid in pids:
        try:
            os.kill(pid, 15)
        except ProcessLookupError:
            pass
        except PermissionError:
            log_line(log_path, f"failed to signal brain pid={pid}: permission denied")

    deadline = time.time() + 20
    while time.time() < deadline:
        if not find_brain_pids():
            break
        time.sleep(0.5)

    candidates = [
        "/home/bryan/llm-orchestration-venv/bin/python",
    ]
    py = next((candidate for candidate in candidates if Path(candidate).exists()), "python3")

    out_log = shared / "logs" / "brain-manual.log"
    out_log.parent.mkdir(parents=True, exist_ok=True)
    with open(out_log, "ab") as lf:
        cmd = [py, "/mnt/shared/agents/brain.py", "--config", str(config_path)]
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    log_line(log_path, f"restarted brain pid={proc.pid} cmd={' '.join(cmd)}")
    return True


def write_brain_heartbeat(
    shared: Path,
    detected_gpu_ids: list[int],
    brain_gpu_stats: dict[int, dict[str, Any]],
    cpu_temp_c: int | None,
    configured_model: str | None,
    loaded_models: list[str],
    expected_worker_names: list[str],
    live_worker_names: list[str],
    known_worker_names: list[str],
    mismatch_count: int,
) -> None:
    now = now_iso()
    data = {
        "agent_name": "brain-watchdog",
        "host": socket.gethostname(),
        "last_updated": now,
        "detected_gpu_ids": detected_gpu_ids,
        "detected_gpu_count": len(detected_gpu_ids),
        "cpu_temp_c": cpu_temp_c,
        "configured_model": configured_model,
        "loaded_models": loaded_models,
        "loaded_model": loaded_models[0] if loaded_models else configured_model,
        "brain_gpus": {str(k): v for k, v in sorted(brain_gpu_stats.items())},
        "expected_worker_names": expected_worker_names,
        "live_worker_names": live_worker_names,
        "brain_known_worker_names": known_worker_names,
        "mismatch_count": mismatch_count,
        "brain_pids": find_brain_pids(),
    }
    save_json(shared / "brain" / "heartbeat.json", data)

    # Publish a worker-style unified heartbeat so consumers can treat brain
    # telemetry the same way as other agents.
    unified = {
        "worker_type": "brain",
        "name": "brain",
        "hostname": socket.gethostname(),
        "state": "online",
        "last_updated": now,
        "cpu_temp_c": cpu_temp_c,
        "gpu_id": None,
        "configured_model": configured_model,
        "loaded_models": loaded_models,
        "loaded_model": loaded_models[0] if loaded_models else configured_model,
        "brain_gpu_ids": sorted(brain_gpu_stats.keys()),
        "brain_gpus": {str(k): v for k, v in sorted(brain_gpu_stats.items())},
        "stats": {
            "detected_gpu_count": len(detected_gpu_ids),
            "expected_worker_count": len(expected_worker_names),
            "live_worker_count": len(live_worker_names),
            "brain_known_worker_count": len(known_worker_names),
            "mismatch_count": mismatch_count,
        },
    }
    save_json(shared / "heartbeats" / "brain.json", unified)


def run_once(args: argparse.Namespace) -> int:
    shared = Path(args.shared_path)
    config_path = Path(args.config)
    log_path = shared / "logs" / "brain_watchdog.log"
    state_path = shared / "brain" / "brain_watchdog_state.json"
    wd_state = load_json(state_path, {"mismatch_count": 0, "last_restart_at": None})

    config = load_json(config_path, {})
    if not config:
        log_line(log_path, f"missing or invalid config at {config_path}")
        return 1

    detected = scan_gpu_ids()
    gpu_stats, cpu_temp_c = scan_gpu_telemetry()
    runtime_host = str(config.get("runtime_host") or "http://localhost:11434")
    runtime_backend = str(config.get("runtime_backend", "llama"))
    loaded_models = scan_loaded_models(runtime_host, runtime_backend)
    configured_model = None
    brain_cfg = config.get("brain")
    if isinstance(brain_cfg, dict):
        m = brain_cfg.get("model")
        if isinstance(m, str) and m.strip():
            configured_model = m.strip()
    new_cfg, cfg_changed = reconcile_config(config, detected)
    if cfg_changed and args.sync_config:
        save_json(config_path, new_cfg)
        config = new_cfg
        log_line(
            log_path,
            f"synced config to detected GPUs={detected} workers={[g.get('name') for g in configured_workers(config)]} "
            f"max_hot_workers={config.get('max_hot_workers')}",
        )

    expected_names = [str(g.get("name")) for g in configured_workers(config) if g.get("name")]
    expected_names = sorted(set(expected_names))
    live_names = list_live_worker_names(shared, args.worker_heartbeat_stale_seconds)
    known_names = brain_known_workers(shared)
    running_count, total_count = latest_brain_agent_counts(shared)

    mismatch = False
    reason = ""
    if total_count is not None and total_count != len(expected_names):
        mismatch = True
        reason = f"brain_total={total_count} expected_total={len(expected_names)}"
    elif running_count is not None and len(live_names) > 0 and running_count < len(live_names):
        mismatch = True
        reason = f"brain_running={running_count} live_workers={len(live_names)}"

    if mismatch:
        wd_state["mismatch_count"] = int(wd_state.get("mismatch_count", 0)) + 1
    else:
        wd_state["mismatch_count"] = 0

    write_brain_heartbeat(
        shared=shared,
        detected_gpu_ids=detected,
        brain_gpu_stats={gid: gpu_stats.get(gid, {}) for gid in config.get("brain", {}).get("gpus", []) if isinstance(gid, int)},
        cpu_temp_c=cpu_temp_c,
        configured_model=configured_model,
        loaded_models=loaded_models,
        expected_worker_names=expected_names,
        live_worker_names=live_names,
        known_worker_names=known_names,
        mismatch_count=int(wd_state.get("mismatch_count", 0)),
    )

    if mismatch:
        log_line(
            log_path,
            f"brain-worker mismatch ({reason}); live={live_names}; known={known_names}; "
            f"mismatch_count={wd_state['mismatch_count']}/{args.mismatch_cycles}",
        )

    if int(wd_state.get("mismatch_count", 0)) >= args.mismatch_cycles:
        ok = restart_brain(config_path=config_path, shared=shared, dry_run=args.dry_run)
        if ok:
            wd_state["last_restart_at"] = now_iso()
            wd_state["mismatch_count"] = 0

    save_json(state_path, wd_state)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Brain watchdog for dynamic GPU worker detection")
    ap.add_argument("--shared-path", default="/mnt/shared")
    ap.add_argument("--config", default="/mnt/shared/agents/config.json")
    ap.add_argument("--interval-seconds", type=int, default=20)
    ap.add_argument("--worker-heartbeat-stale-seconds", type=int, default=90)
    ap.add_argument("--mismatch-cycles", type=int, default=3)
    ap.add_argument("--sync-config", action="store_true", default=True)
    ap.add_argument("--no-sync-config", dest="sync_config", action="store_false")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.once:
        return run_once(args)

    while True:
        try:
            run_once(args)
        except Exception as e:
            log_line(Path(args.shared_path) / "logs" / "brain_watchdog.log", f"watchdog exception: {e}")
        time.sleep(max(5, int(args.interval_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
