#!/usr/bin/env python3
"""Capture CPU/GPU/runtime diagnostics for a specific batch over time."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

LANES = ("queue", "processing", "complete", "failed")
PROC_HINTS = (
    "ollama",
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


def _ollama_snapshot() -> dict:
    ok, out = _run(["ollama", "ps"], timeout=6.0)
    if not ok:
        return {"error": out}
    lines = [ln.rstrip() for ln in out.splitlines() if ln.strip()]
    return {"lines": lines[:80]}


def _cpu_temp_snapshot() -> dict:
    ok, out = _run(["bash", "-lc", "sensors 2>/dev/null"], timeout=6.0)
    if not ok:
        return {"error": "sensors unavailable"}
    temps = [ln.strip() for ln in out.splitlines() if "°C" in ln or " C" in ln]
    return {"sensor_lines": temps[:80]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-id", required=True)
    ap.add_argument("--shared-path", default="/home/bryan/llm_orchestration/shared")
    ap.add_argument("--duration-seconds", type=int, default=90)
    ap.add_argument("--interval-seconds", type=float, default=5.0)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    shared_path = Path(args.shared_path).resolve()
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
                "ollama": _ollama_snapshot(),
                "cpu_temps": _cpu_temp_snapshot(),
            }
        )
        time.sleep(interval)

    payload = {
        "batch_id": args.batch_id,
        "created_at": datetime.now().isoformat(),
        "shared_path": str(shared_path),
        "batch_path": str(batch_path) if batch_path else "",
        "duration_seconds": duration,
        "interval_seconds": interval,
        "sample_count": len(samples),
        "samples": samples,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
