#!/usr/bin/env python3
"""Capture CPU/GPU/runtime diagnostics for a specific batch over time."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

LANES = ("queue", "processing", "complete", "failed")
PROC_HINTS = (
    "llama",
    "llama-server",
    "llama-brain",
    "llama-worker",
    "llama-split",
    "brain.py",
    "gpu.py",
    "worker.py",
    "cpu_agent.py",
    "python /mnt/shared/plans/shoulders/github_analyzer/scripts/",
)


def _run(cmd: list[str], timeout: float = 5.0) -> tuple[bool, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:
        return False, str(exc)
    if p.returncode != 0:
        err = (p.stderr or p.stdout or "").strip()
        return False, err
    return True, p.stdout


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _batch_task_files(shared_path: Path, batch_id: str) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {k: [] for k in LANES}
    for lane in LANES:
        lane_dir = shared_path / "tasks" / lane
        if not lane_dir.exists():
            continue
        for p in lane_dir.glob("*.json"):
            data = _load_json(p)
            if not isinstance(data, dict):
                continue
            if str(data.get("batch_id", "")).strip() == batch_id:
                out[lane].append(p)
    return out


def _resolve_batch_path(shared_path: Path, batch_id: str) -> Path | None:
    for lane in ("processing", "queue", "complete", "failed"):
        lane_dir = shared_path / "tasks" / lane
        if not lane_dir.exists():
            continue
        for p in sorted(lane_dir.glob("*.json"), reverse=True):
            data = _load_json(p)
            if not isinstance(data, dict):
                continue
            if str(data.get("batch_id", "")).strip() != batch_id:
                continue
            batch_path = str(data.get("batch_path", "")).strip()
            if batch_path:
                p = Path(batch_path)
                if str(p).startswith("/mnt/shared/"):
                    rel = str(p)[len("/mnt/shared/") :]
                    return shared_path / rel
                return p
    return None


def _task_counts(shared_path: Path, batch_id: str) -> dict[str, int]:
    files = _batch_task_files(shared_path, batch_id)
    return {lane: len(paths) for lane, paths in files.items()}


def _process_snapshot() -> list[dict]:
    ok, out = _run(
        ["ps", "-eo", "pid,ppid,pcpu,pmem,comm,args", "--sort=-pcpu", "--no-headers"],
        timeout=6.0,
    )
    if not ok:
        return [{"error": out}]

    rows: list[dict] = []
    for raw in out.splitlines():
        parts = raw.strip().split(None, 5)
        if len(parts) < 6:
            continue
        pid, ppid, pcpu, pmem, comm, args = parts
        args_l = args.lower()
        comm_l = comm.lower()
        if not any(h in args_l or h in comm_l for h in PROC_HINTS):
            continue
        rows.append(
            {
                "pid": int(pid),
                "ppid": int(ppid),
                "cpu_percent": float(pcpu),
                "mem_percent": float(pmem),
                "comm": comm,
                "args": args[:240],
            }
        )
    return rows[:40]


def _gpu_snapshot() -> dict:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    ok, out = _run(cmd, timeout=6.0)
    if not ok:
        return {"error": out}

    gpus: list[dict] = []
    for line in out.splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 8:
            continue
        gpus.append(
            {
                "index": int(cols[0]),
                "name": cols[1],
                "temp_c": float(cols[2]),
                "gpu_util_percent": float(cols[3]),
                "mem_util_percent": float(cols[4]),
                "mem_used_mb": float(cols[5]),
                "mem_total_mb": float(cols[6]),
                "power_w": float(cols[7]),
            }
        )
    return {"gpus": gpus}


def _cpu_temp_snapshot() -> dict:
    ok, out = _run(["bash", "-lc", "sensors 2>/dev/null"], timeout=6.0)
    if not ok:
        return {"error": "sensors unavailable"}
    temps = [ln.strip() for ln in out.splitlines() if "°C" in ln or " C" in ln]
    return {"sensor_lines": temps[:80]}


def _execution_context(shared_path: Path) -> dict:
    shared_text = str(shared_path)
    on_rig = shared_text.startswith("/mnt/shared")
    hostname_ok, hostname_out = _run(["hostname"], timeout=3.0)
    hostname = hostname_out.strip() if hostname_ok else ""
    return {
        "hostname": hostname,
        "shared_path": shared_text,
        "rig_side": on_rig,
        "warning": ""
        if on_rig
        else (
            "Running off-rig. Runtime port probes and GPU telemetry may be misleading or unavailable. "
            "Prefer running this script on the GPU rig with --shared-path /mnt/shared."
        ),
    }


def _probe_models(api_base: str) -> dict:
    api_base = str(api_base or "").strip().rstrip("/")
    if not api_base:
        return {"reachable": False, "status_code": None, "models": [], "error": "missing_api_base"}
    try:
        req = urllib.request.Request(f"{api_base}/v1/models")
        with urllib.request.urlopen(req, timeout=3) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            models = []
            if isinstance(payload, dict):
                for item in payload.get("data", []):
                    if isinstance(item, dict):
                        model_id = str(item.get("id", "")).strip()
                        if model_id:
                            models.append(model_id)
            return {
                "reachable": True,
                "status_code": resp.status,
                "models": models,
                "error": None,
            }
    except urllib.error.HTTPError as exc:
        return {
            "reachable": exc.code == 503,
            "status_code": exc.code,
            "models": [],
            "error": exc.reason or str(exc),
        }
    except Exception as exc:
        return {
            "reachable": False,
            "status_code": None,
            "models": [],
            "error": str(exc),
        }


def _runtime_snapshot(shared_path: Path) -> dict:
    runtime: dict[str, object] = {"brain": {}, "gpus": []}

    brain_hb = _load_json(shared_path / "brain" / "heartbeat.json")
    if isinstance(brain_hb, dict):
        brain_probe = _probe_models("http://127.0.0.1:11434")
        runtime["brain"] = {
            "state": str(brain_hb.get("state", "")).strip(),
            "last_updated": str(brain_hb.get("last_updated", "")).strip(),
            "active_batches": int(brain_hb.get("active_batches", 0) or 0),
            "brain_gpus": brain_hb.get("brain_gpus", []),
            "pid": int(((brain_hb.get("brain_pids") or {}).get("pid") or 0)),
            "probe": brain_probe,
        }
    else:
        runtime["brain"] = {"error": "missing_brain_heartbeat"}

    gpus_dir = shared_path / "gpus"
    if gpus_dir.exists():
        for hb_path in sorted(gpus_dir.glob("gpu_*/heartbeat.json")):
            hb = _load_json(hb_path)
            if not isinstance(hb, dict):
                continue
            api_base = str(hb.get("runtime_api_base", "")).strip()
            if api_base.startswith("http://localhost:"):
                api_base = api_base.replace("http://localhost:", "http://127.0.0.1:", 1)
            runtime["gpus"].append(
                {
                    "name": str(hb.get("name", "")).strip(),
                    "state": str(hb.get("state", "")).strip(),
                    "runtime_state": str(hb.get("runtime_state", "")).strip(),
                    "last_updated": str(hb.get("last_updated", "")).strip(),
                    "model_loaded": bool(hb.get("model_loaded", False)),
                    "loaded_model": str(hb.get("loaded_model", "")).strip(),
                    "configured_model": str(hb.get("configured_model", "")).strip(),
                    "runtime_placement": str(hb.get("runtime_placement", "")).strip(),
                    "runtime_group_id": str(hb.get("runtime_group_id", "")).strip(),
                    "runtime_port": hb.get("runtime_port"),
                    "meta_task_active": bool(hb.get("meta_task_active", False)),
                    "meta_task_id": str(hb.get("meta_task_id", "")).strip(),
                    "meta_task_phase": str(hb.get("meta_task_phase", "")).strip(),
                    "runtime_error_code": str(hb.get("runtime_error_code", "")).strip(),
                    "runtime_error_detail": str(hb.get("runtime_error_detail", "")).strip(),
                    "probe": _probe_models(api_base) if api_base else {"reachable": False, "status_code": None, "models": [], "error": "missing_api_base"},
                }
            )

    return runtime


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-id", required=True)
    ap.add_argument("--shared-path", default="/home/bryan/llm_orchestration/shared")
    ap.add_argument("--duration-seconds", type=int, default=90)
    ap.add_argument("--interval-seconds", type=float, default=5.0)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    shared_path = Path(args.shared_path).resolve()
    exec_ctx = _execution_context(shared_path)
    batch_path = _resolve_batch_path(shared_path, args.batch_id)
    if args.output:
        out_path = Path(args.output).resolve()
    elif batch_path:
        out_dir = batch_path / "diagnostics"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"runtime_diag_{stamp}.json"
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path.cwd() / f"runtime_diag_{args.batch_id}_{stamp}.json"

    start = time.time()
    interval = max(1.0, float(args.interval_seconds))
    duration = max(1, int(args.duration_seconds))
    samples: list[dict] = []

    while True:
        now = time.time()
        elapsed = now - start
        if elapsed > duration:
            break
        samples.append(
            {
                "ts": datetime.now().isoformat(),
                "elapsed_seconds": round(elapsed, 1),
                "task_counts": _task_counts(shared_path, args.batch_id),
                "processes": _process_snapshot(),
                "gpu": _gpu_snapshot(),
                "runtime": _runtime_snapshot(shared_path),
                "cpu_temps": _cpu_temp_snapshot(),
            }
        )
        time.sleep(interval)

    payload = {
        "batch_id": args.batch_id,
        "created_at": datetime.now().isoformat(),
        "execution_context": exec_ctx,
        "shared_path": str(shared_path),
        "batch_path": str(batch_path) if batch_path else "",
        "duration_seconds": duration,
        "interval_seconds": interval,
        "sample_count": len(samples),
        "samples": samples,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if exec_ctx["warning"]:
        print(f"WARNING: {exec_ctx['warning']}")
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
