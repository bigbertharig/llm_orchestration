"""Worker and GPU status loading functions."""

import subprocess
from pathlib import Path
from typing import Any

from .utils import (
    HEARTBEAT_BAD_S,
    HEARTBEAT_MAX_S,
    HEARTBEAT_WARN_S,
    heartbeat_age_seconds,
    load_json,
)


def classify_thermal_cause(reasons: Any) -> str:
    """Classify thermal event cause as cpu, gpu, mixed, or none."""
    parts: list[str] = []
    if isinstance(reasons, list):
        parts = [str(x) for x in reasons]
    elif isinstance(reasons, str):
        parts = [reasons]
    text = " ".join(parts).upper()
    has_cpu = "CPU" in text
    has_gpu = "GPU" in text
    if has_cpu and has_gpu:
        return "mixed"
    if has_cpu:
        return "cpu"
    if has_gpu:
        return "gpu"
    return "none"


def load_worker_rows(shared_path: Path, processing_tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Load worker status rows from heartbeat files."""
    by_worker: dict[str, list[str]] = {}
    for t in processing_tasks:
        worker = t.get("assigned_to")
        if worker:
            by_worker.setdefault(worker, []).append(t.get("name") or t.get("task_id") or "(task)")

    rows: list[dict[str, Any]] = []

    # GPU workers
    for hb_file in sorted((shared_path / "gpus").glob("gpu_*/heartbeat.json")):
        hb = load_json(hb_file)
        if not hb:
            continue
        name = hb.get("name") or hb_file.parent.name
        held = by_worker.get(name, [])[:]
        for at in hb.get("active_tasks", []) or []:
            if isinstance(at, dict):
                n = at.get("task_name") or at.get("task_id")
                if n and n not in held:
                    held.append(n)
        last_thermal_event = hb.get("last_thermal_event") if isinstance(hb.get("last_thermal_event"), dict) else None
        thermal_reasons = hb.get("thermal_reasons") if isinstance(hb.get("thermal_reasons"), list) else []
        thermal_cause = "none"
        if last_thermal_event and isinstance(last_thermal_event.get("reasons"), list):
            thermal_cause = classify_thermal_cause(last_thermal_event.get("reasons"))
        elif thermal_reasons:
            thermal_cause = classify_thermal_cause(thermal_reasons)
        rows.append({
            "name": name,
            "gpu_id": hb.get("gpu_id"),
            "type": "gpu",
            "state": hb.get("state", "?"),
            "host": hb.get("hostname", "-"),
            "updated_at": hb.get("last_updated"),
            "age_s": heartbeat_age_seconds(hb.get("last_updated")),
            "gpu_temp_c": hb.get("temperature_c"),
            "cpu_temp_c": hb.get("cpu_temp_c"),
            "gpu_util": hb.get("gpu_util_percent"),
            "power_w": hb.get("power_draw_w"),
            "vram_used_mb": hb.get("vram_used_mb"),
            "vram_total_mb": hb.get("vram_total_mb"),
            "holding": held,
            "thermal_event_type": last_thermal_event.get("type") if last_thermal_event else None,
            "thermal_event_detail": " ".join(last_thermal_event.get("reasons") or []) if last_thermal_event else "",
            "thermal_cause": thermal_cause,
        })

    # CPU workers
    for hb_file in sorted((shared_path / "cpus").glob("*/heartbeat.json")):
        hb = load_json(hb_file)
        if not hb:
            continue
        name = hb.get("name") or hb_file.parent.name
        held = by_worker.get(name, [])[:]
        active_task_id = hb.get("active_task_id")
        if active_task_id:
            exists = any(active_task_id in x for x in held)
            if not exists:
                held.append(active_task_id)
        rows.append({
            "name": name,
            "gpu_id": None,
            "type": "cpu",
            "state": hb.get("state", "?"),
            "host": hb.get("hostname", "-"),
            "updated_at": hb.get("last_updated"),
            "age_s": heartbeat_age_seconds(hb.get("last_updated")),
            "gpu_temp_c": None,
            "cpu_temp_c": hb.get("cpu_temp_c"),
            "gpu_util": None,
            "power_w": None,
            "vram_used_mb": None,
            "vram_total_mb": None,
            "holding": held,
            "thermal_event_type": None,
            "thermal_event_detail": "",
            "thermal_cause": "cpu",
        })

    # Drop expired heartbeats so dead workers disappear from live tables.
    rows = [
        r for r in rows
        if (r.get("age_s") is None or r.get("age_s") <= HEARTBEAT_MAX_S)
    ]

    # Add heartbeat status classification
    for row in rows:
        age = row.get("age_s")
        if age is None:
            row["heartbeat_status"] = "missing"
        elif age >= HEARTBEAT_BAD_S:
            row["heartbeat_status"] = "stale_bad"
        elif age >= HEARTBEAT_WARN_S:
            row["heartbeat_status"] = "stale_warn"
        else:
            row["heartbeat_status"] = "ok"

    rows.sort(key=lambda r: (r["type"], r["name"]))
    return rows


def load_brain_state(shared_path: Path) -> dict[str, Any]:
    """Load brain state from state.json."""
    return load_json(shared_path / "brain" / "state.json") or {}


def load_brain_heartbeat(shared_path: Path) -> dict[str, Any]:
    """Load brain heartbeat, preferring unified path over legacy."""
    return (
        load_json(shared_path / "heartbeats" / "brain.json")
        or load_json(shared_path / "brain" / "heartbeat.json")
        or {}
    )


def load_gpu_telemetry() -> dict[int, dict[str, Any]]:
    """Best-effort live GPU telemetry keyed by GPU index."""
    result: dict[int, dict[str, Any]] = {}
    query_cmd = [
        "nvidia-smi",
        "--query-gpu=index,temperature.gpu,utilization.gpu,power.draw,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]

    def parse_output(text: str) -> None:
        nonlocal result
        for line in text.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            try:
                gid = int(parts[0])
                temp_c = int(float(parts[1]))
                util_pct = int(float(parts[2]))
                power_w = float(parts[3])
                mem_used = int(float(parts[4]))
                mem_total = int(float(parts[5]))
            except Exception:
                continue
            result[gid] = {
                "gpu_temp_c": temp_c,
                "gpu_util": util_pct,
                "power_w": power_w,
                "vram_used_mb": mem_used,
                "vram_total_mb": mem_total,
            }

    # First preference: local nvidia-smi (dashboard running on GPU rig).
    try:
        proc = subprocess.run(
            query_cmd,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            parse_output(proc.stdout)
    except Exception:
        pass

    if result:
        return result

    # Fallback: dashboard on Pi/CPU host, query GPU rig via SSH.
    try:
        proc = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "gpu", *query_cmd],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            parse_output(proc.stdout)
    except Exception:
        pass

    return result
