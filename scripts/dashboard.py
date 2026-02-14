#!/usr/bin/env python3
"""Local web dashboard for orchestration status."""

import argparse
import json
import re
import shlex
import subprocess
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

HEARTBEAT_WARN_S = 60
HEARTBEAT_BAD_S = 120
HEARTBEAT_MAX_S = 600


def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_shared_path(config_path: Path, config: dict[str, Any]) -> Path:
    shared = Path(config.get("shared_path", "../"))
    if shared.is_absolute():
        return shared
    return (config_path.resolve().parent / shared).resolve()


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def iter_task_files(folder_path: Path):
    if not folder_path.exists():
        return
    for task_file in folder_path.glob("*.json"):
        if task_file.name.endswith(".heartbeat.json"):
            continue
        yield task_file


def task_sort_key(task: dict[str, Any]) -> str:
    return (
        task.get("last_attempt_at")
        or task.get("completed_at")
        or task.get("started_at")
        or task.get("created_at")
        or ""
    )


def to_task_view(task: dict[str, Any]) -> dict[str, Any]:
    result = task.get("result") if isinstance(task.get("result"), dict) else {}
    error_text = (
        result.get("output")
        or task.get("error")
        or task.get("blocked_reason")
        or ""
    )
    error_text = " ".join(str(error_text).split())
    return {
        "task_id": task.get("task_id"),
        "name": task.get("name"),
        "executor": task.get("executor") or "worker",
        "task_class": task.get("task_class") or "-",
        "type": task.get("type"),
        "batch_id": task.get("batch_id"),
        "assigned_to": task.get("assigned_to"),
        "status": task.get("status"),
        "attempts": task.get("attempts") or task.get("retry_count") or 0,
        "created_at": task.get("created_at"),
        "started_at": task.get("started_at"),
        "completed_at": task.get("completed_at"),
        "error": error_text[:220],
    }


def list_tasks(shared_path: Path) -> dict[str, list[dict[str, Any]]]:
    lanes = {
        "queue": [],
        "processing": [],
        "complete": [],
        "failed": [],
    }
    for lane in lanes:
        folder = shared_path / "tasks" / lane
        for task_file in iter_task_files(folder) or []:
            task = load_json(task_file)
            if not task:
                continue
            status = str(task.get("status") or "").strip().lower()
            # Some failed tasks can be materialized under complete/.
            # Rebucket by explicit status so dashboard failure counts stay accurate.
            if status in {"failed", "blocked_cloud", "abandoned", "error"}:
                lanes["failed"].append(task)
            else:
                lanes[lane].append(task)
    for lane in lanes:
        lanes[lane].sort(key=task_sort_key, reverse=True)
    return lanes


def list_private_tasks(shared_path: Path) -> list[dict[str, Any]]:
    private_dir = shared_path / "brain" / "private_tasks"
    rows: list[dict[str, Any]] = []
    for task_file in iter_task_files(private_dir) or []:
        task = load_json(task_file)
        if task:
            rows.append(task)
    rows.sort(key=task_sort_key, reverse=True)
    return rows


def heartbeat_age_seconds(last_updated: str | None) -> int | None:
    if not last_updated:
        return None
    try:
        dt = datetime.fromisoformat(last_updated)
        return int((datetime.now() - dt).total_seconds())
    except Exception:
        return None


def classify_thermal_cause(reasons: Any) -> str:
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
    by_worker: dict[str, list[str]] = {}
    for t in processing_tasks:
        worker = t.get("assigned_to")
        if worker:
            by_worker.setdefault(worker, []).append(t.get("name") or t.get("task_id") or "(task)")

    rows: list[dict[str, Any]] = []

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
    return load_json(shared_path / "brain" / "state.json") or {}


def load_brain_heartbeat(shared_path: Path) -> dict[str, Any]:
    # Prefer unified worker-style heartbeat path, fallback to legacy brain path.
    return (
        load_json(shared_path / "heartbeats" / "brain.json")
        or load_json(shared_path / "brain" / "heartbeat.json")
        or {}
    )


def file_mtime_iso(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    except Exception:
        return None


def load_gpu_telemetry() -> dict[int, dict[str, Any]]:
    """Best-effort live GPU telemetry keyed by GPU index."""
    result: dict[int, dict[str, Any]] = {}
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,temperature.gpu,utilization.gpu,power.draw,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except Exception:
        return result

    if proc.returncode != 0:
        return result

    for line in proc.stdout.splitlines():
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
    return result


def count_by_batch(tasks: dict[str, list[dict[str, Any]]], batch_id: str) -> dict[str, int]:
    return {
        "queue": sum(1 for t in tasks["queue"] if t.get("batch_id") == batch_id),
        "processing": sum(1 for t in tasks["processing"] if t.get("batch_id") == batch_id),
        "complete": sum(1 for t in tasks["complete"] if t.get("batch_id") == batch_id),
        "failed": sum(1 for t in tasks["failed"] if t.get("batch_id") == batch_id),
        "private": sum(1 for t in tasks["private"] if t.get("batch_id") == batch_id),
    }


def lane_view(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return [to_task_view(t) for t in items[:limit]]


def is_system_meta_task(task: dict[str, Any]) -> bool:
    name = str(task.get("name") or "").lower()
    return name.startswith("load_llm") or name.startswith("load_worker_model")


def extract_stage_item(name: str | None) -> tuple[str | None, str | None]:
    if not name:
        return None, None
    m = re.match(
        r"^(?P<stage>.+?)_(?P<item>(contact|chunk|item|prospect|person|day)_[0-9A-Za-z]+)$",
        name,
    )
    if not m:
        return None, None
    return m.group("stage"), m.group("item")


def build_batch_chain(tasks_by_lane: dict[str, list[dict[str, Any]]], batch_id: str) -> dict[str, Any]:
    # Flatten all task lanes for this batch into one lookup by name.
    by_name: dict[str, dict[str, Any]] = {}
    lane_of: dict[str, str] = {}
    for lane, tasks in tasks_by_lane.items():
        for t in tasks:
            if t.get("batch_id") != batch_id:
                continue
            name = t.get("name")
            if not name:
                continue
            by_name[name] = t
            lane_of[name] = lane

    # Build item-stage status map from task names like stage_contact_0019.
    item_stage_status: dict[str, dict[str, str]] = {}
    stage_types: dict[str, str] = {}
    stages: set[str] = set()
    edges: set[tuple[str, str]] = set()
    global_edges: set[tuple[str, str]] = set()

    for name, task in by_name.items():
        stage, item = extract_stage_item(name)
        if stage and item:
            stages.add(stage)
            item_stage_status.setdefault(item, {})[stage] = lane_of.get(name, "unknown")
            executor = str(task.get("executor", "worker")).lower()
            task_class = str(task.get("task_class", "")).lower()
            if executor == "brain":
                stage_type = "brain"
            elif task_class == "script":
                stage_type = "gpu"
            elif task_class in {"cpu", "llm", "meta"}:
                stage_type = task_class
            else:
                stage_type = "-"
            if stage_type != "-":
                prev = stage_types.get(stage)
                # Keep strongest signal if mixed data appears.
                if prev is None or prev == "-" or stage_type == "brain":
                    stage_types[stage] = stage_type

            for dep in task.get("depends_on", []) or []:
                dep_stage, dep_item = extract_stage_item(dep)
                if dep_stage and (dep_item == item):
                    edges.add((dep_stage, stage))
        else:
            for dep in task.get("depends_on", []) or []:
                if dep in by_name:
                    global_edges.add((dep, name))

    # Topological-ish order from per-item edges. Fallback to alpha.
    if stages:
        indeg = {s: 0 for s in stages}
        children = {s: set() for s in stages}
        for a, b in edges:
            if a in stages and b in stages and b not in children[a]:
                children[a].add(b)
                indeg[b] += 1
        ready = sorted([s for s in stages if indeg[s] == 0])
        ordered = []
        while ready:
            s = ready.pop(0)
            ordered.append(s)
            for ch in sorted(children[s]):
                indeg[ch] -= 1
                if indeg[ch] == 0:
                    ready.append(ch)
            ready.sort()
        if len(ordered) != len(stages):
            stage_order = sorted(stages)
        else:
            stage_order = ordered
    else:
        # Fallback chain view for global stages (no per-item suffix yet),
        # so users still see declared order like build_strategy -> execute_searches.
        global_nodes = {
            n
            for n in by_name.keys()
            if n
            and n != "batch_summary"
            and not n.startswith("load_llm")
            and not n.startswith("load_worker_model")
        }
        if global_nodes:
            indeg = {s: 0 for s in global_nodes}
            children = {s: set() for s in global_nodes}
            for a, b in global_edges:
                if a in global_nodes and b in global_nodes and b not in children[a]:
                    children[a].add(b)
                    indeg[b] += 1
            ready = sorted([s for s in global_nodes if indeg[s] == 0])
            ordered = []
            while ready:
                s = ready.pop(0)
                ordered.append(s)
                for ch in sorted(children[s]):
                    indeg[ch] -= 1
                    if indeg[ch] == 0:
                        ready.append(ch)
                ready.sort()
            stage_order = ordered if len(ordered) == len(global_nodes) else sorted(global_nodes)
            for s in stage_order:
                t = by_name.get(s, {})
                executor = str(t.get("executor", "worker")).lower()
                task_class = str(t.get("task_class", "")).lower()
                if executor == "brain":
                    stage_types[s] = "brain"
                elif task_class == "script":
                    stage_types[s] = "gpu"
                elif task_class in {"cpu", "llm", "meta"}:
                    stage_types[s] = task_class
                else:
                    stage_types[s] = "-"
        else:
            stage_order = []

    items = sorted(item_stage_status.keys())
    rows = []
    for item in items:
        rows.append({
            "item": item,
            "stages": {s: item_stage_status[item].get(s, "-") for s in stage_order},
        })

    return {
        "stage_order": stage_order,
        "stage_types": {s: stage_types.get(s, "-") for s in stage_order},
        "rows": rows,
        "row_count": len(items),
    }


def summarize(shared_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    lanes = list_tasks(shared_path)
    private_tasks = list_private_tasks(shared_path)
    lanes["private"] = private_tasks

    brain = load_brain_state(shared_path)
    brain_hb = load_brain_heartbeat(shared_path)
    brain_state_mtime = file_mtime_iso(shared_path / "brain" / "state.json")
    hb_age = heartbeat_age_seconds(brain_hb.get("last_updated"))
    mtime_age = heartbeat_age_seconds(brain_state_mtime)
    # Fallback when heartbeat writing is stale but brain state is active.
    if brain_state_mtime and (
        not brain_hb.get("last_updated")
        or hb_age is None
        or (hb_age > HEARTBEAT_MAX_S and mtime_age is not None and mtime_age <= HEARTBEAT_MAX_S)
    ):
        brain_hb = dict(brain_hb)
        brain_hb["last_updated"] = brain_state_mtime
        brain_hb["last_updated_source"] = "state_file_mtime"

    gpu_telemetry = load_gpu_telemetry()
    configured_brain_model = config.get("brain", {}).get("model") if isinstance(config.get("brain"), dict) else None
    active_batches = brain.get("active_batches", {}) if isinstance(brain.get("active_batches"), dict) else {}

    workers = load_worker_rows(shared_path, lanes["processing"])
    brain_holding = []
    for t in lanes.get("processing", []):
        if str(t.get("executor", "worker")).lower() == "brain":
            brain_holding.append(f"{t.get('name', t.get('task_id', '-'))} [processing]")
    for t in lanes.get("queue", []):
        if str(t.get("executor", "worker")).lower() == "brain":
            brain_holding.append(f"{t.get('name', t.get('task_id', '-'))} [queue]")
    brain_gpu_ids = list(config.get("brain", {}).get("gpus", []))
    gpu_by_id = {}
    for w in workers:
        if w.get("type") == "gpu":
            gpu_by_id[w.get("gpu_id")] = w

    brain_gpus = []
    for gpu_id in brain_gpu_ids:
        matched = gpu_by_id.get(gpu_id)
        if matched:
            row = dict(matched)
            row["note"] = "heartbeat"
            row["holding"] = brain_holding[:]
            row["model"] = (
                brain_hb.get("loaded_model")
                or (brain_hb.get("loaded_models") or [None])[0]
                or configured_brain_model
                or "-"
            )
            brain_gpus.append(row)
        else:
            updated_at = brain_hb.get("last_updated")
            telem = gpu_telemetry.get(gpu_id, {})
            hb_gpus = brain_hb.get("brain_gpus") if isinstance(brain_hb.get("brain_gpus"), dict) else {}
            hb_gpu = hb_gpus.get(str(gpu_id), {}) if isinstance(hb_gpus.get(str(gpu_id), {}), dict) else {}
            hb_age_s = heartbeat_age_seconds(updated_at if isinstance(updated_at, str) else None)
            gpu_temp = telem.get("gpu_temp_c")
            if gpu_temp is None:
                gpu_temp = hb_gpu.get("gpu_temp_c")
            gpu_util = telem.get("gpu_util")
            if gpu_util is None:
                gpu_util = hb_gpu.get("gpu_util")
            power_w = telem.get("power_w")
            if power_w is None:
                power_w = hb_gpu.get("power_w")
            vram_used = telem.get("vram_used_mb")
            if vram_used is None:
                vram_used = hb_gpu.get("vram_used_mb")
            vram_total = telem.get("vram_total_mb")
            if vram_total is None:
                vram_total = hb_gpu.get("vram_total_mb")
            brain_gpus.append({
                "model": (
                    brain_hb.get("loaded_model")
                    or (brain_hb.get("loaded_models") or [None])[0]
                    or configured_brain_model
                    or "-"
                ),
                "name": f"brain-gpu-{gpu_id}",
                "gpu_id": gpu_id,
                "type": "gpu",
                "state": "online" if brain_hb else "no_heartbeat",
                "host": brain_hb.get("host") or brain_hb.get("hostname", "-"),
                "updated_at": updated_at if isinstance(updated_at, str) else None,
                "age_s": hb_age_s,
                "gpu_temp_c": gpu_temp,
                "cpu_temp_c": brain_hb.get("cpu_temp_c"),
                "gpu_util": gpu_util,
                "power_w": power_w,
                "vram_used_mb": vram_used,
                "vram_total_mb": vram_total,
                "holding": brain_holding[:],
                "note": "brain heartbeat + nvidia-smi" if brain_hb else "configured brain GPU has no worker heartbeat",
            })

    alerts: list[dict[str, Any]] = []
    for w in workers:
        hb = w.get("heartbeat_status")
        if hb in {"stale_warn", "stale_bad", "missing"}:
            sev = "warn" if hb == "stale_warn" else "bad"
            age = w.get("age_s")
            msg = f"{w.get('name')} heartbeat {hb}"
            if age is not None:
                msg += f" ({age}s old)"
            alerts.append({"severity": sev, "message": msg, "worker": w.get("name"), "age_s": age})
        thermal_type = w.get("thermal_event_type")
        thermal_cause = w.get("thermal_cause")
        if thermal_type == "critical_shutdown":
            cause_label = thermal_cause if thermal_cause and thermal_cause != "none" else "unknown"
            detail = (w.get("thermal_event_detail") or "")[:120]
            message = f"thermal shutdown ({cause_label})"
            if detail:
                message += f": {detail}"
            alerts.append({
                "severity": "bad",
                "message": message,
                "worker": w.get("name"),
                "age_s": w.get("age_s"),
            })

    expected_worker_names: set[str] = set()
    for w in config.get("workers", []) if isinstance(config.get("workers"), list) else []:
        name = w.get("name") if isinstance(w, dict) else None
        if name:
            expected_worker_names.add(name)
    if not expected_worker_names:
        brain_ids = set(brain_gpu_ids)
        for g in config.get("gpus", []) if isinstance(config.get("gpus"), list) else []:
            if not isinstance(g, dict):
                continue
            gid = g.get("id")
            gname = g.get("name")
            if gname and gid not in brain_ids:
                expected_worker_names.add(gname)
    seen_worker_names = {w.get("name") for w in workers if w.get("type") == "gpu"}
    for name in sorted(expected_worker_names - seen_worker_names):
        alerts.append({
            "severity": "bad",
            "message": f"missing configured GPU worker heartbeat: {name}",
            "worker": name,
            "age_s": None,
        })

    batches: dict[str, Any] = {}
    batch_chains: dict[str, Any] = {}
    for batch_id, meta in active_batches.items():
        counts = count_by_batch(lanes, batch_id)
        batch_info: dict[str, Any] = {
            "plan": meta.get("plan"),
            "started_at": meta.get("started_at"),
            "total_hint": meta.get("total_tasks"),
            "counts": counts,
        }
        # Include goal progress if this is a goal-driven batch
        goal = meta.get("goal")
        if goal and isinstance(goal, dict):
            batch_info["goal"] = {
                "target": goal.get("target", 0),
                "tolerance": goal.get("tolerance", 0),
                "accepted": goal.get("accepted", 0),
                "rejected": goal.get("rejected", 0),
                "in_flight": len(goal.get("in_flight_ids", [])),
                "pool_remaining": goal.get("candidates_total", 0) - goal.get("next_index", 0),
                "status": goal.get("status", ""),
            }
        batches[batch_id] = batch_info
        batch_chains[batch_id] = build_batch_chain(lanes, batch_id)

    # Keep lane tables focused on currently active batches so historical
    # completes don't swamp the dashboard.
    active_batch_ids = set(active_batches.keys())
    if active_batch_ids:
        lane_source = {
            "queue": [t for t in lanes["queue"] if t.get("batch_id") in active_batch_ids],
            "processing": [
                t for t in lanes["processing"]
                if t.get("batch_id") in active_batch_ids or is_system_meta_task(t)
            ],
            "private": [t for t in lanes["private"] if t.get("batch_id") in active_batch_ids],
            "complete": [t for t in lanes["complete"] if t.get("batch_id") in active_batch_ids],
            "failed": [t for t in lanes["failed"] if t.get("batch_id") in active_batch_ids],
        }
    else:
        lane_source = lanes

    return {
        "generated_at": datetime.now().isoformat(),
        # Keep tab/card counts aligned with the same filtered lane source
        # that drives the visible list tables.
        "counts": {k: len(v) for k, v in lane_source.items()},
        "counts_all": {k: len(v) for k, v in lanes.items()},
        "active_batches": batches,
        "batch_chains": batch_chains,
        "workers": workers,
        "brain_gpus": brain_gpus,
        "alerts": alerts,
        "lanes": {
            "queue": lane_view(lane_source["queue"], 50),
            "processing": lane_view(lane_source["processing"], 50),
            "private": lane_view(lane_source["private"], 80),
            "complete": lane_view(lane_source["complete"], 50),
            "failed": lane_view(lane_source["failed"], 50),
        },
    }


def run_shell(cmd: str, timeout_s: int = 120) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
            "cmd": cmd,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "ok": False,
            "returncode": -1,
            "stdout": (e.stdout or "")[-2000:] if isinstance(e.stdout, str) else "",
            "stderr": (e.stderr or "")[-2000:] if isinstance(e.stderr, str) else "",
            "cmd": cmd,
            "error": f"timeout after {timeout_s}s",
        }


def shoulders_dir(shared_path: Path) -> Path:
    return shared_path / "plans" / "shoulders"


def shoulder_plan_dir(shared_path: Path, plan_name: str) -> Path:
    return shoulders_dir(shared_path) / plan_name


def discover_plans(shared_path: Path) -> list[str]:
    plans_dir = shoulders_dir(shared_path)
    out: list[str] = []
    if not plans_dir.exists():
        return out
    for p in sorted(plans_dir.iterdir()):
        if not p.is_dir():
            continue
        has_scripts = (p / "scripts").exists()
        has_root_plan = (p / "plan.md").exists()
        has_input_starter = False
        input_dir = p / "input"
        if input_dir.exists():
            has_input_starter = any(
                md.is_file() and (md.name == "plan.md" or md.name.endswith("_plan.md"))
                for md in input_dir.glob("*.md")
            )
        if has_scripts and (has_root_plan or has_input_starter):
            out.append(p.name)
    return out


def discover_plan_starters(shared_path: Path, plan_name: str) -> list[str]:
    plan_dir = shoulder_plan_dir(shared_path, plan_name)
    starters: set[str] = set()
    if not plan_dir.exists():
        return []
    for md in plan_dir.glob("*.md"):
        if md.is_file() and (md.name == "plan.md" or md.name.endswith("_plan.md")):
            starters.add(md.name)
    input_dir = plan_dir / "input"
    if input_dir.exists():
        for md in input_dir.rglob("*.md"):
            if md.is_file() and (md.name == "plan.md" or md.name.endswith("_plan.md")):
                starters.add(str(Path("input") / md.relative_to(input_dir)))
    return sorted(starters)


def discover_plan_input_files(shared_path: Path, plan_name: str, limit: int = 200) -> list[str]:
    plan_dir = shoulder_plan_dir(shared_path, plan_name)
    input_dir = plan_dir / "input"
    if not input_dir.exists():
        return []
    out: list[str] = []
    for p in sorted(input_dir.rglob("*")):
        if len(out) >= limit:
            break
        if not p.is_file():
            continue
        if ".submit_runtime" in p.parts:
            continue
        rel = p.relative_to(shared_path)
        out.append(f"/mnt/shared/{rel.as_posix()}")
    return out


def find_batch_dir(shared_path: Path, batch_id: str) -> tuple[str, Path] | None:
    if not batch_id:
        return None
    plans_dir = shoulders_dir(shared_path)
    if not plans_dir.exists():
        return None
    for plan_dir in plans_dir.iterdir():
        if not plan_dir.is_dir():
            continue
        candidate = plan_dir / "history" / batch_id
        if candidate.exists() and candidate.is_dir():
            return plan_dir.name, candidate
    return None


def discover_recent_batches(
    shared_path: Path,
    active_batches: dict[str, Any],
    limit: int = 80,
) -> list[dict[str, Any]]:
    plans_dir = shoulders_dir(shared_path)
    out: list[dict[str, Any]] = []
    if not plans_dir.exists():
        return out

    active_set = set(active_batches.keys())
    for plan_dir in plans_dir.iterdir():
        if not plan_dir.is_dir():
            continue
        hist = plan_dir / "history"
        if not hist.exists():
            continue
        for batch_dir in hist.iterdir():
            if not batch_dir.is_dir():
                continue
            out.append(
                {
                    "batch_id": batch_dir.name,
                    "plan": plan_dir.name,
                    "updated_at": file_mtime_iso(batch_dir),
                    "active": batch_dir.name in active_set,
                }
            )
    out.sort(key=lambda x: str(x.get("updated_at") or ""), reverse=True)
    return out[:limit]


def collect_batch_outputs(shared_path: Path, batch_id: str, max_files: int = 40) -> dict[str, Any]:
    found = find_batch_dir(shared_path, batch_id)
    if not found:
        return {"ok": False, "message": f"Batch directory not found for {batch_id}"}

    plan_name, batch_dir = found
    files: list[Path] = []
    preferred = [
        batch_dir / "output",
        batch_dir / "results",
    ]
    for root in preferred:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file():
                files.append(p)

    for extra in [
        batch_dir / "execution_stats.json",
        batch_dir / "manifest.json",
    ]:
        if extra.exists() and extra.is_file():
            files.append(extra)

    # De-duplicate while preserving order.
    dedup: list[Path] = []
    seen: set[Path] = set()
    for p in files:
        if p in seen:
            continue
        seen.add(p)
        dedup.append(p)
    files = dedup[:max_files]

    def to_mnt(path: Path) -> str:
        rel = path.relative_to(shared_path)
        return f"/mnt/shared/{rel.as_posix()}"

    file_rows: list[dict[str, Any]] = []
    for p in files:
        rel = p.relative_to(batch_dir).as_posix()
        try:
            size_b = int(p.stat().st_size)
        except Exception:
            size_b = 0
        preview = ""
        if p.suffix.lower() in {".md", ".txt", ".json", ".csv", ".log"}:
            try:
                preview = p.read_text(encoding="utf-8", errors="replace")[:2000]
            except Exception:
                preview = ""
        file_rows.append(
            {
                "name": p.name,
                "relative_path": rel,
                "path": str(p),
                "mnt_path": to_mnt(p),
                "size_bytes": size_b,
                "updated_at": file_mtime_iso(p),
                "preview": preview,
            }
        )

    return {
        "ok": True,
        "batch_id": batch_id,
        "plan": plan_name,
        "batch_path": str(batch_dir),
        "batch_mnt_path": to_mnt(batch_dir),
        "files": file_rows,
        "message": f"Found {len(file_rows)} output/result files",
    }


def default_plan_config(shared_path: Path, plan_name: str) -> dict[str, Any]:
    if plan_name == "research_assistant":
        return {
            "QUERY_FILE": "/mnt/shared/plans/shoulders/research_assistant/input/query.md",
            "TARGET_COUNT": "20",
            "TARGET_TOLERANCE": "0",
            "SECONDARY_TARGET_COUNT": "20",
            "SEARCH_DEPTH": "basic",
            "OUTPUT_FORMAT": "both",
            "RUN_MODE": "fresh",
        }
    if plan_name == "dc_integration":
        return {
            "ZIM_PATH": "/mnt/shared/path/to/archive.zim",
            "SOURCE_ID": "source_name",
            "OUTPUT_FOLDER": "/mnt/shared/plans/shoulders/dc_integration/output",
            "RUN_MODE": "fresh",
        }
    return {"RUN_MODE": "fresh"}


def plan_input_help(shared_path: Path, plan_name: str, starter_file: str | None = None) -> list[dict[str, str]]:
    plan_dir = shoulder_plan_dir(shared_path, plan_name)
    if starter_file:
        plan_md = plan_dir / starter_file
    else:
        plan_md = plan_dir / "plan.md"
    if not plan_md.exists() and not starter_file:
        starters = discover_plan_starters(shared_path, plan_name)
        if starters:
            plan_md = plan_dir / starters[0]
    if not plan_md.exists():
        return []
    try:
        text = plan_md.read_text(encoding="utf-8")
    except Exception:
        return []

    lines = text.splitlines()
    in_inputs = False
    out: list[dict[str, str]] = []
    for line in lines:
        if line.strip().lower() == "## inputs":
            in_inputs = True
            continue
        if in_inputs and line.startswith("## "):
            break
        if not in_inputs:
            continue
        m = re.match(r"^\s*-\s+\*\*(.+?)\*\*:\s*(.+)\s*$", line)
        if not m:
            continue
        key = m.group(1).strip()
        desc = m.group(2).strip()
        options = ", ".join(re.findall(r"`([^`]+)`", desc))
        out.append({
            "key": key,
            "description": desc,
            "options": options,
        })
    return out


HTML = """<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>LLM Orchestration Dashboard</title>
  <style>
    :root {
      --bg: #08131f;
      --panel: #0f2234;
      --panel2: #122a3f;
      --line: #24445f;
      --text: #eaf4ff;
      --muted: #9db4c8;
      --ok: #38d89c;
      --warn: #ffcc66;
      --bad: #ff6b6b;
      --cpu: #5db4ff;
      --gpu: #ffa94d;
      --llm: #f5a97f;
      --meta: #cba6f7;
      --script: #74c7ec;
      --taskcpu: #94e2d5;
    }
    body {
      margin: 0;
      font-family: \"IBM Plex Sans\", \"Segoe UI\", sans-serif;
      background: radial-gradient(1200px 600px at 10% 0%, #12304a 0%, var(--bg) 60%);
      color: var(--text);
    }
    .wrap { max-width: 1400px; margin: 0 auto; padding: 14px; }
    h1 { margin: 0 0 10px; font-size: 24px; }
    h3 { margin: 2px 0 10px; font-size: 16px; }
    .meta { color: var(--muted); margin-bottom: 10px; font-size: 13px; }
    .grid { display: grid; gap: 10px; }
    .g5 { grid-template-columns: repeat(5, minmax(0, 1fr)); }
    .g2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .card {
      background: linear-gradient(180deg, var(--panel), var(--panel2));
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px;
      overflow: hidden;
    }
    .k { color: var(--muted); font-size: 12px; }
    .v { font-size: 24px; font-weight: 700; line-height: 1.1; }
    .ok { color: var(--ok); }
    .warn { color: var(--warn); }
    .bad { color: var(--bad); }
    .section { margin-top: 10px; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; table-layout: fixed; }
    th, td { text-align: left; padding: 5px 6px; border-bottom: 1px solid var(--line); vertical-align: top; word-break: break-word; }
    th { color: var(--muted); font-weight: 600; }
    tr:last-child td { border-bottom: 0; }
    .pill { display: inline-block; padding: 2px 7px; border-radius: 999px; font-size: 11px; border: 1px solid var(--line); }
    .pill.gpu { background: rgba(255,169,77,.15); color: var(--gpu); }
    .pill.cpu { background: rgba(93,180,255,.15); color: var(--cpu); }
    .pill.llm { background: rgba(245,169,127,.16); color: var(--llm); }
    .pill.meta { background: rgba(203,166,247,.16); color: var(--meta); }
    .pill.script { background: rgba(116,199,236,.16); color: var(--script); }
    .pill.taskcpu { background: rgba(148,226,213,.16); color: var(--taskcpu); }
    .pill.brain { background: rgba(255,107,107,.14); color: #ff9a9a; }
    .pill.worker { background: rgba(74,214,109,.14); color: #7be495; }
    .mono { font-family: \"JetBrains Mono\", \"Consolas\", monospace; }
    .clip {
      display: inline-block;
      max-width: 100%;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      vertical-align: top;
    }
    .copyable { cursor: copy; }
    .copy-toast {
      position: fixed;
      right: 14px;
      bottom: 14px;
      background: #16324a;
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 6px 10px;
      font-size: 12px;
      z-index: 9999;
      opacity: 0;
      transition: opacity .18s ease;
      pointer-events: none;
    }
    .copy-toast.show { opacity: 1; }
    .lane-title { display: flex; justify-content: space-between; align-items: baseline; }
    .lane-count { color: var(--muted); font-size: 12px; }
    .chain-box { margin-top: 10px; padding: 8px; border: 1px solid var(--line); border-radius: 8px; }
    .chain-head { display:flex; justify-content:space-between; align-items:center; gap:8px; margin-bottom:6px; }
    .chain-controls { display:flex; gap:6px; align-items:center; }
    .small-btn {
      background: #16324a;
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 3px 8px;
      cursor: pointer;
      font-size: 11px;
    }
    .small-btn:disabled { opacity: 0.45; cursor: default; }
    .chip { display:inline-block; padding:2px 6px; border-radius:999px; font-size:11px; border:1px solid var(--line); }
    .chip.queue { color:#f9c74f; background:rgba(249,199,79,.15); }
    .chip.processing { color:#4cc9f0; background:rgba(76,201,240,.15); }
    .chip.private { color:#cba6f7; background:rgba(203,166,247,.15); }
    .chip.complete { color:#4ad66d; background:rgba(74,214,109,.15); }
    .chip.failed { color:#ff6b6b; background:rgba(255,107,107,.15); }
    .chip.missing { color:#9db4c8; background:rgba(157,180,200,.12); }
    .tabs { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }
    .plan-scope { margin-top: 8px; }
    .filter-bar { display:flex; gap:8px; flex-wrap:wrap; margin: 6px 0 8px; align-items:center; }
    .filter-field { display:flex; gap:4px; align-items:center; }
    .filter-field input, .filter-field select {
      background: #0d1f31;
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 4px 6px;
      font-size: 12px;
    }
    .tab-btn {
      background: #16324a;
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 6px 10px;
      cursor: pointer;
      font-size: 12px;
    }
    .tab-btn.active {
      border-color: #4f9bd1;
      background: #204565;
    }
    @media (max-width: 1180px) {
      .g5, .g2 { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"tabs\" style=\"margin-bottom:8px;\">
      <a class=\"tab-btn active\" href=\"/\">Dashboard</a>
      <a class=\"tab-btn\" href=\"/controls\">Controls</a>
    </div>
    <h1>LLM Orchestration Dashboard</h1>
    <div class=\"meta\" id=\"meta\">Loading...</div>

    <div class=\"grid g5\" id=\"countCards\"></div>

    <div class=\"section card plan-scope\">
      <h3>Plan Scope</h3>
      <div id=\"planScope\"></div>
    </div>

    <div class=\"section card\">
      <h3>Alerts</h3>
      <div id=\"alerts\"></div>
    </div>

    <div class=\"section card\">
      <h3>Brain GPU Status</h3>
      <div id=\"brainGpus\"></div>
    </div>

    <div class=\"section card\">
      <h3>Workers</h3>
      <div class=\"tabs\" id=\"workerTabs\"></div>
      <div id=\"workerTable\"></div>
    </div>

    <div class=\"section card\">
      <h3>Active Batches</h3>
      <div id=\"batches\"></div>
      <div id=\"batchChains\"></div>
    </div>

    <div class=\"section card\">
      <div class=\"tabs\" id=\"laneTabs\"></div>
      <div id=\"laneTable\"></div>
    </div>
  </div>

  <script>
    const fmt = (x) => x === null || x === undefined || x === '' ? '-' : x;
    const esc = (x) => String(x)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/\"/g, '&quot;');
    const oneLine = (x) => String(fmt(x)).replace(/\\s+/g, ' ').trim() || '-';
    function truncCell(value, maxLen = 60, mono = false) {
      const raw = oneLine(value);
      const shown = raw.length > maxLen ? `${raw.slice(0, Math.max(1, maxLen - 1))}…` : raw;
      const cls = mono ? 'mono clip' : 'clip';
      return `<span class=\"${cls} copyable\" title=\"${esc(raw)}\" data-copy=\"${esc(raw)}\">${esc(shown)}</span>`;
    }
    let copyToastTimer = null;
    function showCopyToast(msg) {
      let el = document.getElementById('copyToast');
      if (!el) {
        el = document.createElement('div');
        el.id = 'copyToast';
        el.className = 'copy-toast';
        document.body.appendChild(el);
      }
      el.textContent = msg;
      el.classList.add('show');
      if (copyToastTimer) clearTimeout(copyToastTimer);
      copyToastTimer = setTimeout(() => el.classList.remove('show'), 900);
    }
    async function copyText(text) {
      const val = String(text || '');
      if (!val) return;
      try {
        await navigator.clipboard.writeText(val);
      } catch (_) {
        const ta = document.createElement('textarea');
        ta.value = val;
        ta.style.position = 'fixed';
        ta.style.left = '-9999px';
        document.body.appendChild(ta);
        ta.focus();
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
      }
      showCopyToast('Copied full text');
    }
    function bindCopyables(containerSelector) {
      document.querySelectorAll(`${containerSelector} .copyable`).forEach(el => {
        el.addEventListener('click', () => copyText(el.getAttribute('data-copy') || ''));
      });
    }
    const laneOrder = ['queue', 'processing', 'private', 'complete', 'failed'];
    let activeLane = 'processing';
    let activeWorkerTab = 'cpu';
    const chainState = {};
    const batchState = { batch: '', plan: '', sort: 'started_desc' };
    const laneState = { taskClass: '', task: '', worker: '', executor: '', batch: '', error: '', sort: 'task_asc' };
    let latestStatus = null;
    let visibleBatchIds = null;

    function typeBadge(type) {
      return `<span class=\"pill ${type}\">${type}</span>`;
    }

    function taskType(cls) {
      const norm = (cls || '').toLowerCase();
      if (norm === 'cpu') return 'cpu';
      if (norm === 'llm') return 'llm';
      if (norm === 'script') return 'gpu';
      if (norm === 'meta') return 'meta';
      return norm || '-';
    }
    function classBadge(cls, executor = '') {
      if (String(executor || '').toLowerCase() === 'brain') {
        return `<span class=\"pill brain\">BRAIN</span>`;
      }
      const type = taskType(cls);
      const map = { cpu: 'taskcpu', llm: 'llm', gpu: 'script', meta: 'meta', brain: 'brain' };
      const key = map[type] || 'script';
      return `<span class=\"pill ${key}\">${String(type).toUpperCase()}</span>`;
    }
    function executorBadge(executor) {
      const e = (executor || 'worker').toLowerCase() === 'brain' ? 'brain' : 'worker';
      return `<span class=\"pill ${e}\">${e}</span>`;
    }

    function tempClass(cpu) {
      if (cpu === null || cpu === undefined) return '';
      if (cpu >= 85) return 'bad';
      if (cpu >= 75) return 'warn';
      return 'ok';
    }

    function hbClass(age) {
      if (age === null || age === undefined) return 'bad';
      if (age >= 120) return 'bad';
      if (age >= 60) return 'warn';
      return 'ok';
    }

    function table(headers, rows) {
      if (!rows.length) return '<div class=\"k\">(none)</div>';
      return `
        <table>
          <thead><tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr></thead>
          <tbody>
            ${rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join('')}</tr>`).join('')}
          </tbody>
        </table>
      `;
    }

    function renderCountCards(counts) {
      const items = [
        ['Queue', counts.queue, ''],
        ['Processing', counts.processing, 'ok'],
        ['Private', counts.private, 'warn'],
        ['Complete', counts.complete, 'ok'],
        ['Failed', counts.failed, counts.failed > 0 ? 'bad' : ''],
      ];
      return items.map(([k, v, cls]) => `
        <div class=\"card\"><div class=\"k\">${k}</div><div class=\"v ${cls}\">${v}</div></div>
      `).join('');
    }

    function renderPlanScope(activeBatches) {
      const plans = [...new Set(activeBatches.map(b => b.plan).filter(Boolean))].sort();
      if (batchState.plan && !plans.some(p => String(p).toLowerCase() === batchState.plan)) {
        batchState.plan = '';
      }
      const html = `
        <div class=\"filter-bar\">
          <label class=\"filter-field k\">Selected Plan
            <select id=\"globalPlanScope\">
              <option value=\"\">all active plans</option>
              ${plans.map(p => `<option value=\"${String(p).toLowerCase()}\" ${batchState.plan === String(p).toLowerCase() ? 'selected' : ''}>${p}</option>`).join('')}
            </select>
          </label>
          <span class=\"k\">Applies to Active Batches and task lanes.</span>
        </div>
      `;
      const el = document.getElementById('planScope');
      el.innerHTML = html;
      const sel = document.getElementById('globalPlanScope');
      if (!sel) return;
      const rerender = () => {
        batchState.plan = sel.value || '';
        if (latestStatus) refreshFromData(latestStatus);
      };
      sel.addEventListener('change', rerender);
      sel.addEventListener('input', rerender);
    }

    function renderAlerts(alerts) {
      if (!alerts || !alerts.length) {
        document.getElementById('alerts').innerHTML = '<div class=\"k\">(none)</div>';
        return;
      }
      const rows = alerts.slice(0, 40).map(a => [
        `<span class=\"${a.severity === 'bad' ? 'bad' : 'warn'}\">${a.severity}</span>`,
        fmt(a.worker),
        truncCell(a.message, 100, false),
        fmt(a.age_s),
      ]);
      document.getElementById('alerts').innerHTML = table(['Severity', 'Worker', 'Message', 'HB s'], rows);
      bindCopyables('#alerts');
    }

    function isSystemMetaTaskName(taskName) {
      const n = String(taskName || '').toLowerCase();
      return n.startsWith('load_llm') || n.startsWith('load_worker_model');
    }

    function renderTaskLane(targetId, items) {
      const classes = [...new Set(items.map(t => taskType(t.task_class)).filter(Boolean))].sort();
      const executors = [...new Set(items.map(t => (t.executor || 'worker').toLowerCase()).filter(Boolean))].sort();
      const sort = laneState.sort || 'task_asc';
      const showRuntime = activeLane === 'processing';
      const showCompletedAt = activeLane === 'complete';
      const showQueuedAt = activeLane === 'queue';

      let filtered = items.filter(t => {
        const cls = taskType(t.task_class);
        const task = (t.name || '').toLowerCase();
        const worker = (t.assigned_to || '').toLowerCase();
        const executor = (t.executor || 'worker').toLowerCase();
        const batch = (t.batch_id || '').toLowerCase();
        const err = (t.error || '').toLowerCase();
        if (visibleBatchIds && !visibleBatchIds.has(t.batch_id || '') && !isSystemMetaTaskName(t.name)) return false;
        if (laneState.taskClass && cls !== laneState.taskClass) return false;
        if (laneState.task && !task.includes(laneState.task.toLowerCase())) return false;
        if (laneState.worker && !worker.includes(laneState.worker.toLowerCase())) return false;
        if (laneState.executor && executor !== laneState.executor.toLowerCase()) return false;
        if (laneState.batch && !batch.includes(laneState.batch.toLowerCase())) return false;
        if (laneState.error && !err.includes(laneState.error.toLowerCase())) return false;
        return true;
      });

      filtered.sort((a, b) => {
        const aval = (key) => String(a[key] || '').toLowerCase();
        const bval = (key) => String(b[key] || '').toLowerCase();
        if (sort === 'task_asc') return aval('name').localeCompare(bval('name'));
        if (sort === 'task_desc') return bval('name').localeCompare(aval('name'));
        if (sort === 'worker_asc') return aval('assigned_to').localeCompare(bval('assigned_to'));
        if (sort === 'worker_desc') return bval('assigned_to').localeCompare(aval('assigned_to'));
        if (sort === 'executor_asc') return aval('executor').localeCompare(bval('executor'));
        if (sort === 'executor_desc') return bval('executor').localeCompare(aval('executor'));
        if (sort === 'batch_asc') return aval('batch_id').localeCompare(bval('batch_id'));
        if (sort === 'batch_desc') return bval('batch_id').localeCompare(aval('batch_id'));
        if (sort === 'try_desc') return (Number(b.attempts || 0) - Number(a.attempts || 0));
        if (sort === 'try_asc') return (Number(a.attempts || 0) - Number(b.attempts || 0));
        return 0;
      });

      const runtimeText = (startedAt) => {
        if (!startedAt) return '-';
        const dt = new Date(startedAt);
        if (Number.isNaN(dt.getTime())) return '-';
        const sec = Math.max(0, Math.floor((Date.now() - dt.getTime()) / 1000));
        const h = Math.floor(sec / 3600);
        const m = Math.floor((sec % 3600) / 60);
        const s = sec % 60;
        if (h > 0) return `${h}h ${m}m ${s}s`;
        if (m > 0) return `${m}m ${s}s`;
        return `${s}s`;
      };
      const completedAtText = (completedAt) => {
        if (!completedAt) return '-';
        const raw = String(completedAt).trim();
        // Prefer local display always (same basis as the top clock).
        // Normalize common ISO-ish variants before parsing.
        const normalized = raw.includes('T') ? raw : raw.replace(' ', 'T');
        const dt = new Date(normalized);
        if (Number.isNaN(dt.getTime())) return '-';
        return dt.toLocaleString(undefined, { hour12: false });
      };
      const queuedAtValue = (t) =>
        t.stale_requeued_at ||
        t.requeued_at ||
        t.last_attempt_at ||
        t.created_at ||
        '';

      const rows = filtered.map(t => [
        classBadge(t.task_class, t.executor),
        truncCell(t.name, 42, false),
        truncCell(t.assigned_to, 24, false),
        executorBadge(t.executor),
        `<span class=\"mono\">${fmt(t.batch_id)}</span>`,
        fmt(t.attempts),
        ...(showQueuedAt ? [ `<span class=\"mono\">${completedAtText(queuedAtValue(t))}</span>` ] : []),
        ...(showRuntime ? [ `<span class=\"mono\">${runtimeText(t.started_at)}</span>` ] : []),
        ...(showCompletedAt ? [ `<span class=\"mono\">${completedAtText(t.completed_at || t.started_at)}</span>` ] : []),
        truncCell(t.error, 96, true)
      ]);
      const sortArrow = (ascKey, descKey) => sort === ascKey ? ' ↑' : (sort === descKey ? ' ↓' : '');
      const controls = `
        <div class=\"filter-bar\">
          <span class=\"k\">showing ${filtered.length} of ${items.length}</span>
          <label class=\"filter-field k\">Type
            <select id=\"laneFilterClass\">
              <option value=\"\">all</option>
              ${classes.map(c => `<option value=\"${c}\" ${laneState.taskClass === c ? 'selected' : ''}>${c}</option>`).join('')}
            </select>
          </label>
          <label class=\"filter-field k\">Task <input id=\"laneFilterTask\" value=\"${laneState.task}\" placeholder=\"contains\" /></label>
          <label class=\"filter-field k\">Worker <input id=\"laneFilterWorker\" value=\"${laneState.worker}\" placeholder=\"contains\" /></label>
          <label class=\"filter-field k\">Executor
            <select id=\"laneFilterExecutor\">
              <option value=\"\">all</option>
              ${executors.map(e => `<option value=\"${e}\" ${laneState.executor === e ? 'selected' : ''}>${e}</option>`).join('')}
            </select>
          </label>
          <label class=\"filter-field k\">Batch <input id=\"laneFilterBatch\" value=\"${laneState.batch}\" placeholder=\"contains\" /></label>
          <label class=\"filter-field k\">Error <input id=\"laneFilterError\" value=\"${laneState.error}\" placeholder=\"contains\" /></label>
          <span class=\"k\">Click headers to sort</span>
        </div>
      `;
      const headers = [
        `<th data-sort=\"class\">Type${sortArrow('class_asc', 'class_desc')}</th>`,
        `<th data-sort=\"task\">Task${sortArrow('task_asc', 'task_desc')}</th>`,
        `<th data-sort=\"worker\">Worker${sortArrow('worker_asc', 'worker_desc')}</th>`,
        `<th data-sort=\"executor\">Executor${sortArrow('executor_asc', 'executor_desc')}</th>`,
        `<th data-sort=\"batch\">Batch${sortArrow('batch_asc', 'batch_desc')}</th>`,
        `<th data-sort=\"try\">Try${sortArrow('try_asc', 'try_desc')}</th>`,
        ...(showQueuedAt ? ['<th>Queued At</th>'] : []),
        ...(showRuntime ? ['<th>Runtime</th>'] : []),
        ...(showCompletedAt ? ['<th>Completed At</th>'] : []),
        `<th>Error</th>`
      ].join('');
      const laneTable = rows.length
        ? `<table><thead><tr>${headers}</tr></thead><tbody>${rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table>`
        : '<div class=\"k\">(none)</div>';
      document.getElementById(targetId).innerHTML = controls + laneTable;
      const bind = (id, key) => {
        const el = document.getElementById(id);
        if (!el) return;
        el.addEventListener('input', () => { laneState[key] = el.value; renderTaskLane(targetId, items); });
        el.addEventListener('change', () => { laneState[key] = el.value; renderTaskLane(targetId, items); });
      };
      document.querySelectorAll('#laneTable th[data-sort]').forEach(th => {
        th.style.cursor = 'pointer';
        th.addEventListener('click', () => {
          const key = th.getAttribute('data-sort');
          const current = laneState.sort || 'task_asc';
          const nextMap = {
            class: ['class_asc', 'class_desc'],
            task: ['task_asc', 'task_desc'],
            worker: ['worker_asc', 'worker_desc'],
            executor: ['executor_asc', 'executor_desc'],
            batch: ['batch_asc', 'batch_desc'],
            try: ['try_asc', 'try_desc'],
          };
          const pair = nextMap[key] || ['task_asc', 'task_desc'];
          laneState.sort = current === pair[0] ? pair[1] : pair[0];
          renderTaskLane(targetId, items);
        });
      });
      bind('laneFilterClass', 'taskClass');
      bind('laneFilterTask', 'task');
      bind('laneFilterWorker', 'worker');
      bind('laneFilterExecutor', 'executor');
      bind('laneFilterBatch', 'batch');
      bind('laneFilterError', 'error');
      bindCopyables(`#${targetId}`);
    }

    function renderBatches(activeBatches) {
      const plans = [...new Set(activeBatches.map(b => b.plan).filter(Boolean))].sort();
      const sort = batchState.sort || 'started_desc';
      let filtered = activeBatches.filter(b => {
        const bid = (b.id || '').toLowerCase();
        const plan = (b.plan || '').toLowerCase();
        if (batchState.batch && !bid.includes(batchState.batch.toLowerCase())) return false;
        if (batchState.plan && plan !== batchState.plan.toLowerCase()) return false;
        return true;
      });
      filtered.sort((a, b) => {
        if (sort === 'batch_asc') return String(a.id || '').localeCompare(String(b.id || ''));
        if (sort === 'batch_desc') return String(b.id || '').localeCompare(String(a.id || ''));
        if (sort === 'plan_asc') return String(a.plan || '').localeCompare(String(b.plan || ''));
        if (sort === 'plan_desc') return String(b.plan || '').localeCompare(String(a.plan || ''));
        if (sort === 'stage_asc') return Number(a.stage_rank || 0) - Number(b.stage_rank || 0);
        if (sort === 'stage_desc') return Number(b.stage_rank || 0) - Number(a.stage_rank || 0);
        if (sort === 'done_desc') return Number(b.done_pct || 0) - Number(a.done_pct || 0);
        if (sort === 'done_asc') return Number(a.done_pct || 0) - Number(b.done_pct || 0);
        if (sort === 'complete_desc') return Number(b.complete || 0) - Number(a.complete || 0);
        if (sort === 'complete_asc') return Number(a.complete || 0) - Number(b.complete || 0);
        if (sort === 'queue_desc') return Number(b.queue || 0) - Number(a.queue || 0);
        if (sort === 'queue_asc') return Number(a.queue || 0) - Number(b.queue || 0);
        if (sort === 'processing_desc') return Number(b.processing || 0) - Number(a.processing || 0);
        if (sort === 'processing_asc') return Number(a.processing || 0) - Number(b.processing || 0);
        if (sort === 'private_desc') return Number(b.private || 0) - Number(a.private || 0);
        if (sort === 'private_asc') return Number(a.private || 0) - Number(b.private || 0);
        if (sort === 'failed_desc') return Number(b.failed || 0) - Number(a.failed || 0);
        if (sort === 'failed_asc') return Number(a.failed || 0) - Number(b.failed || 0);
        if (sort === 'started_asc') return String(a.started_raw || '').localeCompare(String(b.started_raw || ''));
        return String(b.started_raw || '').localeCompare(String(a.started_raw || ''));
      });
      const rows = filtered.map(b => [
        `<span class=\"mono\">${b.id}</span>`,
        fmt(b.plan),
        fmt(b.stage),
        `<span class=\"mono\">${fmt(b.complete)}/${fmt(b.total)}</span>`,
        fmt(b.goal),
        fmt(b.queue),
        fmt(b.processing),
        fmt(b.failed),
        fmt(b.private),
        fmt(b.started)
      ]);
      visibleBatchIds = new Set(filtered.map(b => b.id));
      const sortArrow = (ascKey, descKey) => sort === ascKey ? ' ↑' : (sort === descKey ? ' ↓' : '');
      const controls = `
        <div class=\"filter-bar\">
          <span class=\"k\">showing ${filtered.length} of ${activeBatches.length}</span>
          <label class=\"filter-field k\">Batch <input id=\"batchFilterBatch\" value=\"${batchState.batch}\" placeholder=\"contains\" /></label>
          <span class=\"k\">Plan scope: ${batchState.plan ? batchState.plan : 'all active plans'}</span>
          <span class=\"k\">Click headers to sort</span>
        </div>
      `;
      const headers = [
        `<th data-sort=\"batch\">Batch${sortArrow('batch_asc', 'batch_desc')}</th>`,
        `<th data-sort=\"plan\">Plan${sortArrow('plan_asc', 'plan_desc')}</th>`,
        `<th data-sort=\"stage\">Stage${sortArrow('stage_asc', 'stage_desc')}</th>`,
        `<th data-sort=\"complete\">Complete${sortArrow('complete_asc', 'complete_desc')}</th>`,
        `<th>Goal</th>`,
        `<th data-sort=\"queue\">Queue${sortArrow('queue_asc', 'queue_desc')}</th>`,
        `<th data-sort=\"processing\">Processing${sortArrow('processing_asc', 'processing_desc')}</th>`,
        `<th data-sort=\"failed\">Failed${sortArrow('failed_asc', 'failed_desc')}</th>`,
        `<th data-sort=\"private\">Private${sortArrow('private_asc', 'private_desc')}</th>`,
        `<th data-sort=\"started\">Started${sortArrow('started_asc', 'started_desc')}</th>`,
      ].join('');
      const batchTable = rows.length
        ? `<table><thead><tr>${headers}</tr></thead><tbody>${rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table>`
        : '<div class=\"k\">(none)</div>';
      document.getElementById('batches').innerHTML = controls + batchTable;
      const bind = (id, key) => {
        const el = document.getElementById(id);
        if (!el) return;
        const rerender = () => {
          batchState[key] = el.value;
          renderBatches(activeBatches);
          if (latestStatus) renderBatchChains(latestStatus, visibleBatchIds);
        };
        el.addEventListener('input', rerender);
        el.addEventListener('change', rerender);
      };
      document.querySelectorAll('#batches th[data-sort]').forEach(th => {
        th.style.cursor = 'pointer';
        th.addEventListener('click', () => {
          const key = th.getAttribute('data-sort');
          const current = batchState.sort || 'started_desc';
          const nextMap = {
            batch: ['batch_asc', 'batch_desc'],
            plan: ['plan_asc', 'plan_desc'],
            stage: ['stage_asc', 'stage_desc'],
            complete: ['complete_asc', 'complete_desc'],
            queue: ['queue_asc', 'queue_desc'],
            processing: ['processing_asc', 'processing_desc'],
            failed: ['failed_asc', 'failed_desc'],
            private: ['private_asc', 'private_desc'],
            started: ['started_asc', 'started_desc'],
          };
          const pair = nextMap[key] || ['started_asc', 'started_desc'];
          batchState.sort = current === pair[0] ? pair[1] : pair[0];
          renderBatches(activeBatches);
          if (latestStatus) renderBatchChains(latestStatus, visibleBatchIds);
        });
      });
      bind('batchFilterBatch', 'batch');
      // Plan scope is controlled by the global selector (renderPlanScope).
    }

    function renderLaneTabs(counts, lanes) {
      const labels = {
        queue: 'Queue',
        processing: 'Processing',
        private: 'Private',
        complete: 'Complete',
        failed: 'Failed'
      };
      const html = laneOrder.map(l => {
        const cls = l === activeLane ? 'tab-btn active' : 'tab-btn';
        const count = counts[l] ?? 0;
        return `<button class=\"${cls}\" data-lane=\"${l}\">${labels[l]} (${count})</button>`;
      }).join('');
      const tabs = document.getElementById('laneTabs');
      tabs.innerHTML = html;
      tabs.querySelectorAll('button[data-lane]').forEach(btn => {
        btn.addEventListener('click', () => {
          activeLane = btn.getAttribute('data-lane');
          renderLaneTabs(counts, lanes);
          renderTaskLane('laneTable', lanes[activeLane] || []);
        });
      });
      renderTaskLane('laneTable', lanes[activeLane] || []);
    }

    function renderWorkerTabs(workers) {
      const gpuWorkers = workers.filter(w => w.type === 'gpu');
      const cpuWorkers = workers.filter(w => w.type === 'cpu');
      const tabs = [
        { key: 'gpu', label: 'GPU', count: gpuWorkers.length },
        { key: 'cpu', label: 'CPU', count: cpuWorkers.length },
      ];
      const html = tabs.map(t => {
        const cls = t.key === activeWorkerTab ? 'tab-btn active' : 'tab-btn';
        return `<button class=\"${cls}\" data-wtab=\"${t.key}\">${t.label} (${t.count})</button>`;
      }).join('');
      const container = document.getElementById('workerTabs');
      container.innerHTML = html;
      container.querySelectorAll('button[data-wtab]').forEach(btn => {
        btn.addEventListener('click', () => {
          activeWorkerTab = btn.getAttribute('data-wtab');
          renderWorkerTabs(workers);
        });
      });
      const active = activeWorkerTab === 'gpu' ? gpuWorkers : cpuWorkers;
      renderWorkerTable(active, activeWorkerTab);
    }

    function renderWorkerTable(workers, type) {
      if (type === 'gpu') {
        const vram = (w) => (w.vram_used_mb !== null && w.vram_total_mb !== null)
          ? `${w.vram_used_mb}/${w.vram_total_mb}` : '-';
        const rows = workers.map(w => [
          fmt(w.name),
          fmt(w.state),
          fmt(w.host),
          `<span class=\"${tempClass(w.cpu_temp_c)}\">${fmt(w.cpu_temp_c)}</span>`,
          fmt(w.gpu_temp_c),
          fmt(w.gpu_util),
          fmt(w.power_w),
          vram(w),
          fmt(w.thermal_cause && w.thermal_cause !== 'none' ? w.thermal_cause : '-'),
          truncCell((w.holding || []).slice(0,2).join(' | ') || '-', 64, true),
          `<span class=\"${hbClass(w.age_s)}\">${fmt(w.age_s)}</span>`
        ]);
        document.getElementById('workerTable').innerHTML = table(
          ['Name', 'State', 'Host', 'CPU C', 'GPU C', 'GPU %', 'W', 'VRAM', 'Thermal', 'Holding', 'HB s'],
          rows
        );
      } else {
        const rows = workers.map(w => [
          fmt(w.name),
          fmt(w.state),
          fmt(w.host),
          `<span class=\"${tempClass(w.cpu_temp_c)}\">${fmt(w.cpu_temp_c)}</span>`,
          fmt(w.thermal_cause && w.thermal_cause !== 'none' ? w.thermal_cause : '-'),
          truncCell((w.holding || []).slice(0,2).join(' | ') || '-', 64, true),
          `<span class=\"${hbClass(w.age_s)}\">${fmt(w.age_s)}</span>`
        ]);
        document.getElementById('workerTable').innerHTML = table(
          ['Name', 'State', 'Host', 'CPU C', 'Thermal', 'Holding', 'HB s'],
          rows
        );
      }
      bindCopyables('#workerTable');
    }

    function laneChip(lane) {
      if (!lane || lane === '-') return '<span class=\"chip missing\">-</span>';
      return `<span class=\"chip ${lane}\">${lane}</span>`;
    }

    function renderBatchChains(data, allowedBatchIds = null) {
      const perPage = 15;
      const out = [];
      const chains = data.batch_chains || {};
      const laneRank = { '-': 0, queue: 1, private: 2, processing: 3, complete: 4, failed: 5 };
      Object.entries(chains).forEach(([batchId, chain]) => {
        if (allowedBatchIds && !allowedBatchIds.has(batchId)) return;
        const stages = chain.stage_order || [];
        const stageTypes = chain.stage_types || {};
        if (!stages.length) return;
        const totalRows = chain.row_count || (chain.rows || []).length;
        if (!chainState[batchId]) {
          chainState[batchId] = { collapsed: true, page: 1, sortKey: 'item', sortDir: 'asc' };
        }
        const st = chainState[batchId];
        const totalPages = Math.max(1, Math.ceil(totalRows / perPage));
        if (st.page > totalPages) st.page = totalPages;
        const sortKey = st.sortKey || 'item';
        const sortDir = st.sortDir || 'asc';
        const direction = sortDir === 'desc' ? -1 : 1;
        const sortedRows = [...(chain.rows || [])].sort((a, b) => {
          if (sortKey === 'item') {
            return direction * String(a.item || '').localeCompare(String(b.item || ''));
          }
          const ar = laneRank[(a.stages || {})[sortKey] || '-'] || 0;
          const br = laneRank[(b.stages || {})[sortKey] || '-'] || 0;
          if (ar !== br) return direction * (ar - br);
          return String(a.item || '').localeCompare(String(b.item || ''));
        });
        const start = (st.page - 1) * perPage;
        const end = start + perPage;

        const visibleRows = sortedRows.slice(start, end).map(r => {
          const cols = [ `<span class=\"mono\">${r.item}</span>` ];
          stages.forEach(s => cols.push(laneChip((r.stages || {})[s])));
          return cols;
        });
        const arrow = (key) => (sortKey === key ? (sortDir === 'asc' ? ' ↑' : ' ↓') : '');
        const headers = [
          `<th data-batch=\"${batchId}\" data-sort=\"item\">Item${arrow('item')}</th>`,
          ...stages.map(s => {
            const stype = stageTypes[s] || '-';
            const badge = stype && stype !== '-' ? classBadge(stype) : '<span class=\"k\">-</span>';
            return `<th data-batch=\"${batchId}\" data-sort=\"${s}\">${s}${arrow(s)}<div style=\"margin-top:4px\">${badge}</div></th>`;
          })
        ];
        const chainTable = visibleRows.length
          ? `
            <table>
              <thead><tr>${headers.join('')}</tr></thead>
              <tbody>
                ${visibleRows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join('')}</tr>`).join('')}
              </tbody>
            </table>
          `
          : '<div class=\"k\">(none)</div>';
        const collapsed = st.collapsed;
        const body = collapsed
          ? ''
          : `${chainTable}
             <div class=\"k\">showing ${visibleRows.length} of ${totalRows}</div>`;

        out.push(`
          <div class=\"chain-box\" data-batch=\"${batchId}\">
            <div class=\"chain-head\">
              <div class=\"k\">${batchId} chain: ${stages.join(' -> ')}</div>
              <div class=\"chain-controls\">
                <button class=\"small-btn\" data-action=\"toggle\" data-batch=\"${batchId}\">${collapsed ? 'Expand' : 'Collapse'}</button>
                <button class=\"small-btn\" data-action=\"prev\" data-batch=\"${batchId}\" ${collapsed || st.page <= 1 ? 'disabled' : ''}>Prev</button>
                <span class=\"k\">Page ${st.page}/${totalPages}</span>
                <button class=\"small-btn\" data-action=\"next\" data-batch=\"${batchId}\" ${collapsed || st.page >= totalPages ? 'disabled' : ''}>Next</button>
              </div>
            </div>
            <div>${body}</div>
          </div>
        `);
      });
      const container = document.getElementById('batchChains');
      container.innerHTML = out.join('') || '<div class=\"k\">(no itemized dependency chains detected for current batch/plan filter)</div>';
      container.querySelectorAll('button[data-action]').forEach(btn => {
        btn.addEventListener('click', () => {
          const action = btn.getAttribute('data-action');
          const batchId = btn.getAttribute('data-batch');
          const st = chainState[batchId] || { collapsed: true, page: 1, sortKey: 'item', sortDir: 'asc' };
          const total = (chains[batchId]?.row_count) || 0;
          const pages = Math.max(1, Math.ceil(total / perPage));
          if (action === 'toggle') st.collapsed = !st.collapsed;
          if (action === 'prev' && st.page > 1) st.page -= 1;
          if (action === 'next' && st.page < pages) st.page += 1;
          chainState[batchId] = st;
          renderBatchChains(data, allowedBatchIds);
        });
      });
      container.querySelectorAll('th[data-sort]').forEach(th => {
        th.style.cursor = 'pointer';
        th.addEventListener('click', () => {
          const batchId = th.getAttribute('data-batch');
          const key = th.getAttribute('data-sort');
          const st = chainState[batchId] || { collapsed: true, page: 1, sortKey: 'item', sortDir: 'asc' };
          if (st.sortKey === key) {
            st.sortDir = st.sortDir === 'asc' ? 'desc' : 'asc';
          } else {
            st.sortKey = key;
            st.sortDir = 'asc';
          }
          st.page = 1;
          chainState[batchId] = st;
          renderBatchChains(data, allowedBatchIds);
        });
      });
    }

    function refreshFromData(data) {
      latestStatus = data;
      document.getElementById('meta').textContent = `Updated ${new Date(data.generated_at).toLocaleTimeString()}`;
      document.getElementById('countCards').innerHTML = renderCountCards(data.counts);

      const batchRows = Object.entries(data.active_batches).map(([id, b]) => {
        const c = b.counts;
        const total = Math.max(b.total_hint || 0, c.queue + c.processing + c.private + c.complete + c.failed);
        let stage = 'idle';
        let stageRank = 0;
        if (c.failed > 0) { stage = 'failed'; stageRank = 4; }
        else if (c.processing > 0) { stage = 'processing'; stageRank = 3; }
        else if (c.queue > 0 || c.private > 0) { stage = 'queued'; stageRank = 2; }
        else if (total > 0 && c.complete >= total) { stage = 'complete'; stageRank = 5; }
        else if (c.complete > 0) { stage = 'partial'; stageRank = 1; }
        // Goal progress string
        let goalStr = '-';
        if (b.goal) {
          const g = b.goal;
          const statusBadge = g.status === 'complete' ? '\\u2705' : g.status === 'exhausted' ? '\\u26a0' : g.status === 'draining' ? '\\u23f3' : '';
          goalStr = `${g.accepted}/${g.target} ${statusBadge} (${g.in_flight} fly, ${g.rejected} rej, ${g.pool_remaining} pool)`;
        }
        return {
          id: id,
          plan: b.plan || '',
          stage: stage,
          stage_rank: stageRank,
          done_pct: total > 0 ? (c.complete / total) : 0,
          complete: c.complete,
          total: total,
          queue: c.queue,
          processing: c.processing,
          failed: c.failed,
          private: c.private,
          goal: goalStr,
          started_raw: b.started_at || '',
          started: fmt(b.started_at ? new Date(b.started_at).toLocaleTimeString() : '-')
        };
      });
      renderPlanScope(batchRows);
      renderBatches(batchRows);
      renderBatchChains(data, visibleBatchIds);

      const brainRows = (data.brain_gpus || []).map(w => {
        const vram = (w.vram_used_mb !== null && w.vram_total_mb !== null)
          ? `${w.vram_used_mb}/${w.vram_total_mb}`
          : '-';
        const thermal = w.thermal_cause && w.thermal_cause !== 'none' ? w.thermal_cause : '-';
        return [
          fmt(w.model),
          fmt(w.name),
          fmt(w.state),
          vram,
          `<span class=\"${tempClass(w.cpu_temp_c)}\">${fmt(w.cpu_temp_c)}</span>`,
          fmt(w.gpu_temp_c),
          fmt(w.gpu_util),
          fmt(w.power_w),
          fmt(thermal),
          truncCell((w.holding || []).slice(0,2).join(' | ') || '-', 64, true),
          `<span class=\"${hbClass(w.age_s)}\">${fmt(w.age_s)}</span>`
        ];
      });
      document.getElementById('brainGpus').innerHTML = table(
        ['Model', 'Name', 'State', 'VRAM', 'CPU C', 'GPU C', 'GPU %', 'W', 'Thermal', 'Holding', 'HB'],
        brainRows
      );
      bindCopyables('#brainGpus');

      renderWorkerTabs(data.workers);

      renderAlerts(data.alerts || []);
      renderLaneTabs(data.counts, data.lanes);
    }

    async function refresh() {
      const res = await fetch('/api/status');
      const data = await res.json();
      refreshFromData(data);
    }

    refresh().catch(console.error);
    setInterval(() => refresh().catch(console.error), 2000);
  </script>
</body>
</html>
"""

CONTROLS_HTML = """<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>LLM Orchestration Controls</title>
  <style>
    :root {
      --bg: #08131f;
      --panel: #0f2234;
      --panel2: #122a3f;
      --line: #24445f;
      --text: #eaf4ff;
      --muted: #9db4c8;
      --warn: #ffcc66;
      --bad: #ff6b6b;
      --ok: #38d89c;
    }
    body { margin: 0; font-family: \"IBM Plex Sans\", \"Segoe UI\", sans-serif; background: radial-gradient(1200px 600px at 10% 0%, #12304a 0%, var(--bg) 60%); color: var(--text); }
    .wrap { max-width: 1100px; margin: 0 auto; padding: 14px; }
    .tabs { display: flex; gap: 8px; margin-bottom: 10px; }
    .tab-btn { background: #16324a; color: var(--text); border: 1px solid var(--line); border-radius: 8px; padding: 6px 10px; text-decoration: none; }
    .tab-btn.active { border-color: #4f9bd1; background: #204565; }
    .card { background: linear-gradient(180deg, var(--panel), var(--panel2)); border: 1px solid var(--line); border-radius: 10px; padding: 12px; margin-bottom: 10px; }
    h1 { margin: 0 0 10px; font-size: 24px; }
    h3 { margin: 0 0 8px; font-size: 16px; }
    .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    .start-grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(320px, 420px); gap: 10px; align-items: start; }
    .field-grid { display: grid; gap: 8px; }
    .field-row { display: grid; grid-template-columns: 180px minmax(0, 1fr); gap: 8px; align-items: center; }
    .field-key { color: var(--muted); font-size: 12px; font-family: \"JetBrains Mono\", monospace; }
    .field-help { color: var(--muted); font-size: 12px; line-height: 1.35; }
    select, input, textarea, button {
      background: #0d1f31;
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 13px;
    }
    button { cursor: pointer; background: #204565; }
    button.danger { background: #5a2323; border-color: #7a2d2d; }
    button.warn { background: #4d3b1a; border-color: #6a5120; }
    textarea { width: 100%; min-height: 120px; font-family: \"JetBrains Mono\", monospace; }
    #result { white-space: pre-wrap; font-family: \"JetBrains Mono\", monospace; font-size: 12px; color: var(--muted); }
    .k { color: var(--muted); font-size: 12px; }
    .ok { color: var(--ok); }
    .bad { color: var(--bad); }
    table { width: 100%; border-collapse: collapse; font-size: 12px; table-layout: fixed; }
    th, td { text-align: left; padding: 5px 6px; border-bottom: 1px solid var(--line); vertical-align: top; word-break: break-word; }
    th { color: var(--muted); font-weight: 600; }
    .mono { font-family: "JetBrains Mono", monospace; }
    .stack { display: grid; gap: 8px; }
    .tiny { font-size: 11px; color: var(--muted); }
    .preview { white-space: pre-wrap; background: #0b1a2a; border: 1px solid var(--line); border-radius: 8px; padding: 10px; font-family: "JetBrains Mono", monospace; font-size: 12px; max-height: 320px; overflow: auto; }
    @media (max-width: 980px) {
      .start-grid { grid-template-columns: 1fr; }
      .field-row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"tabs\">
      <a class=\"tab-btn\" href=\"/\">Dashboard</a>
      <a class=\"tab-btn active\" href=\"/controls\">Controls</a>
    </div>
    <h1>Control Panel</h1>

    <div class=\"card\">
      <h3>Batch Actions</h3>
      <div class=\"row\">
        <select id=\"killBatch\"></select>
        <button class=\"danger\" onclick=\"killPlan()\">Kill selected batch</button>
        <button class=\"danger\" onclick=\"killAllActive()\">Kill all active</button>
      </div>
      <div class=\"k\">Kills selected/all active batches without shutting down workers.</div>
      <div class=\"row\">
        <select id=\"resumeBatch\"></select>
        <button onclick=\"resumePlan()\">Resume selected batch</button>
      </div>
      <div class=\"k\">Resubmits selected batch with `RUN_MODE=resume`.</div>
      <div style=\"margin-top:10px;\">
        <table>
          <thead>
            <tr><th>Batch</th><th>Plan</th><th>Started</th><th>Actions</th></tr>
          </thead>
          <tbody id=\"activeBatchRows\"></tbody>
        </table>
      </div>
    </div>

    <div class=\"card\">
      <h3>Return To Default</h3>
      <div class=\"row\">
        <button class=\"warn\" onclick=\"returnDefault()\">Reset to brain + 1 hot worker</button>
      </div>
      <div class=\"k\">Stops duplicate/orphan agents and restarts via startup defaults.</div>
    </div>

    <div class=\"card\">
      <h3>Start Plan (Simplified)</h3>
      <div class=\"row\" style=\"margin-bottom:8px;\">
        <select id=\"planName\"></select>
        <select id=\"starterFile\"></select>
      </div>
      <div class=\"start-grid\">
        <div>
          <div class=\"row\" style=\"margin-bottom:6px;\">
            <button onclick=\"applyFormToJson()\">Apply form to JSON</button>
            <button onclick=\"loadJsonToForm()\">Load JSON into form</button>
          </div>
          <div id=\"planConfigForm\" class=\"field-grid\"></div>
          <div class=\"k\" style=\"margin-top:8px;\">Advanced JSON (optional)</div>
          <textarea id=\"planConfig\">{
  "QUERY_FILE": "/mnt/shared/plans/shoulders/research_assistant/input/query.md",
  "TARGET_COUNT": "20",
  "TARGET_TOLERANCE": "0",
  "SECONDARY_TARGET_COUNT": "20",
  "SEARCH_DEPTH": "basic",
  "OUTPUT_FORMAT": "both",
  "RUN_MODE": "fresh"
}</textarea>
          <div class=\"row\" style=\"margin-top:8px;\">
            <button onclick=\"startPlan()\">Submit plan</button>
          </div>
          <div class=\"k\">Uses `/mnt/shared/agents/submit.py` on GPU host.</div>
        </div>
        <div>
          <div class=\"k\">Input settings (with options)</div>
          <div id=\"planInputs\" style=\"margin-top:8px;\"></div>
        </div>
      </div>
    </div>

    <div class=\"card\">
      <h3>Batch Outputs</h3>
      <div class=\"row\">
        <select id=\"outputBatch\"></select>
        <button onclick=\"loadBatchOutputs()\">Load outputs</button>
      </div>
      <div class=\"k\">Browse recent batch output files and quick previews.</div>
      <div id=\"outputMeta\" class=\"tiny\" style=\"margin-top:8px;\"></div>
      <div id=\"outputFiles\" style=\"margin-top:8px;\"></div>
      <div id=\"outputPreview\" class=\"preview\" style=\"margin-top:8px;\">(select a file preview)</div>
    </div>

    <div class=\"card\">
      <h3>Result</h3>
      <div id=\"result\">(waiting)</div>
    </div>
  </div>

  <script>
    let planDefaults = {};
    let planStarters = {};
    let planDefaultStarter = {};
    let planInputs = {};
    let planInputFiles = {};
    let outputFiles = [];

    async function api(path, payload) {
      const res = await fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload || {})
      });
      return await res.json();
    }

    function showResult(data) {
      const ok = data.ok ? '[ok]' : '[error]';
      const text = [
        ok + ' ' + (data.message || ''),
        data.cmd ? ('cmd: ' + data.cmd) : '',
        data.stdout ? ('\\nstdout:\\n' + data.stdout) : '',
        data.stderr ? ('\\nstderr:\\n' + data.stderr) : '',
      ].filter(Boolean).join('\\n');
      const el = document.getElementById('result');
      el.textContent = text || '(no output)';
      el.className = data.ok ? 'ok' : 'bad';
    }

    function fmtTs(ts) {
      if (!ts) return '-';
      const d = new Date(ts);
      if (Number.isNaN(d.getTime())) return String(ts);
      return d.toLocaleString();
    }

    function parseOptions(optionsText) {
      const raw = String(optionsText || '').trim();
      if (!raw) return [];
      return raw
        .split(/[|,/]/)
        .map(x => x.trim())
        .filter(Boolean);
    }

    function getCurrentConfig() {
      try {
        const obj = JSON.parse(document.getElementById('planConfig').value || '{}');
        return (obj && typeof obj === 'object' && !Array.isArray(obj)) ? obj : {};
      } catch (e) {
        return {};
      }
    }

    function readFormConfig() {
      const out = {};
      document.querySelectorAll('[data-config-key]').forEach(el => {
        const key = el.getAttribute('data-config-key');
        if (!key) return;
        out[key] = String(el.value ?? '').trim();
      });
      return out;
    }

    function applyFormToJson() {
      const cfg = readFormConfig();
      document.getElementById('planConfig').value = JSON.stringify(cfg, null, 2);
    }

    function loadJsonToForm() {
      renderConfigForm(document.getElementById('planName').value, getCurrentConfig());
    }

    function renderConfigForm(planName, overrideCfg) {
      const cfg = (overrideCfg && typeof overrideCfg === 'object') ? overrideCfg : (planDefaults[planName] || getCurrentConfig());
      const starterFile = document.getElementById('starterFile').value;
      const perStarter = planInputs[planName] || {};
      const helpList = perStarter[starterFile] || [];
      const helpByKey = {};
      helpList.forEach(x => { helpByKey[x.key] = x; });
      const keys = Object.keys(cfg);
      const inputFiles = planInputFiles[planName] || [];
      const rows = keys.map(key => {
        const v = String(cfg[key] ?? '');
        const help = helpByKey[key] || {};
        const opts = parseOptions(help.options || '');
        const desc = help.description ? `<div class="field-help">${help.description}</div>` : '';
        const hint = help.options ? `<div class="field-help">Options: ${help.options}</div>` : '<div class="field-help">(free text)</div>';
        if (opts.length) {
          const options = opts.map(o => `<option value="${o}" ${o === v ? 'selected' : ''}>${o}</option>`).join('');
          return `<div class="field-row"><div class="field-key">${key}</div><div>${desc}${hint}<select data-config-key="${key}">${options}</select></div></div>`;
        }
        if (key.endsWith('_FILE')) {
          const listId = `list_${key.replace(/[^a-zA-Z0-9_]/g, '_')}`;
          const dlist = inputFiles.map(p => `<option value="${p}"></option>`).join('');
          return `<div class="field-row"><div class="field-key">${key}</div><div>${desc}${hint}<input data-config-key="${key}" value="${v}" list="${listId}" /><datalist id="${listId}">${dlist}</datalist></div></div>`;
        }
        return `<div class="field-row"><div class="field-key">${key}</div><div>${desc}${hint}<input data-config-key="${key}" value="${v}" /></div></div>`;
      }).join('');
      document.getElementById('planConfigForm').innerHTML = rows || '<div class="k">(no config keys)</div>';
    }

    function renderActiveBatches(rows) {
      const tbody = document.getElementById('activeBatchRows');
      if (!rows.length) {
        tbody.innerHTML = '<tr><td colspan="4" class="k">(no active batches)</td></tr>';
        return;
      }
      tbody.innerHTML = rows.map(r => `
        <tr>
          <td class="mono">${r.batch_id || '-'}</td>
          <td>${r.plan || '-'}</td>
          <td>${fmtTs(r.started_at)}</td>
          <td class="row">
            <button class="danger" onclick="killBatchInline('${r.batch_id}')">Kill</button>
            <button onclick="resumeBatchInline('${r.batch_id}')">Resume</button>
            <button onclick="viewBatchInline('${r.batch_id}')">Outputs</button>
          </td>
        </tr>
      `).join('');
    }

    function fillBatchSelect(selectId, ids) {
      const el = document.getElementById(selectId);
      el.innerHTML = '';
      ids.forEach(b => {
        const o = document.createElement('option');
        o.value = b;
        o.textContent = b;
        el.appendChild(o);
      });
      if (!el.options.length) {
        const o = document.createElement('option');
        o.value = '';
        o.textContent = '(none)';
        el.appendChild(o);
      }
    }

    async function refreshOptions() {
      const res = await fetch('/api/control/options');
      const data = await res.json();
      planDefaults = data.plan_defaults || {};
      planStarters = data.plan_starters || {};
      planDefaultStarter = data.plan_default_starter || {};
      planInputs = data.plan_inputs || {};
      planInputFiles = data.plan_input_files || {};
      renderActiveBatches(data.active_batches_meta || []);

      const ids = [];
      const seen = new Set();
      (data.active_batches || []).forEach(b => {
        if (!seen.has(b)) { ids.push(b); seen.add(b); }
      });
      (data.recent_batches || []).forEach(r => {
        const b = r.batch_id;
        if (b && !seen.has(b)) { ids.push(b); seen.add(b); }
      });
      fillBatchSelect('killBatch', ids);
      fillBatchSelect('resumeBatch', ids);
      fillBatchSelect('outputBatch', ids);

      const plan = document.getElementById('planName');
      const starter = document.getElementById('starterFile');
      plan.innerHTML = '';
      (data.plans || []).forEach(p => {
        const o = document.createElement('option');
        o.value = p;
        o.textContent = p;
        plan.appendChild(o);
      });
      plan.onchange = () => applyPlanDefault(plan.value);
      starter.onchange = () => {
        renderPlanInputHelp(plan.value, starter.value);
        renderConfigForm(plan.value);
      };
      if (plan.value) applyPlanDefault(plan.value);
    }

    function applyPlanDefault(planName) {
      const defaults = planDefaults[planName];
      if (defaults) {
        document.getElementById('planConfig').value = JSON.stringify(defaults, null, 2);
      }
      const starterSelect = document.getElementById('starterFile');
      starterSelect.innerHTML = '';
      const starters = planStarters[planName] || [];
      starters.forEach(s => {
        const o = document.createElement('option');
        o.value = s;
        o.textContent = s;
        starterSelect.appendChild(o);
      });
      if (starters.length) {
        starterSelect.value = planDefaultStarter[planName] || starters[0];
      }
      renderPlanInputHelp(planName, starterSelect.value);
      renderConfigForm(planName, defaults || {});
    }

    function renderPlanInputHelp(planName, starterFile) {
      const perStarter = planInputs[planName] || {};
      const inputList = perStarter[starterFile] || [];
      const help = {};
      inputList.forEach(x => { help[x.key] = x; });
      const defaults = planDefaults[planName] || {};
      const src = Object.keys(defaults).length ? defaults : help;
      const keys = Object.keys(src);
      const rows = keys.map(key => {
        const h = help[key] || {};
        const opts = h.options ? `<div class=\"field-help\">Options: ${h.options}</div>` : '';
        const desc = h.description ? `<div class=\"field-help\">${h.description}</div>` : '';
        return `
          <div class=\"field-row\">
            <div class=\"field-key\">${key}</div>
            <div>${desc}${opts || '<div class=\"field-help\">(no explicit options listed)</div>'}</div>
          </div>
        `;
      }).join('');
      document.getElementById('planInputs').innerHTML = rows
        ? `<div class=\"field-grid\">${rows}</div>`
        : '<div class=\"k\">(no input metadata found)</div>';
    }

    async function killBatchInline(batchId) {
      if (!batchId) return;
      showResult(await api('/api/control/kill_plan', { batch_id: batchId }));
      await refreshOptions();
    }

    async function resumeBatchInline(batchId) {
      if (!batchId) return;
      showResult(await api('/api/control/resume_plan', { batch_id: batchId }));
      await refreshOptions();
    }

    async function viewBatchInline(batchId) {
      document.getElementById('outputBatch').value = batchId;
      await loadBatchOutputs();
    }

    async function killPlan() {
      const batchId = document.getElementById('killBatch').value;
      if (!batchId) return;
      showResult(await api('/api/control/kill_plan', { batch_id: batchId }));
      await refreshOptions();
    }

    async function killAllActive() {
      showResult(await api('/api/control/kill_all_active', {}));
      await refreshOptions();
    }

    async function returnDefault() {
      showResult(await api('/api/control/return_default', {}));
      await refreshOptions();
    }

    async function resumePlan() {
      const batchId = document.getElementById('resumeBatch').value;
      if (!batchId) return;
      showResult(await api('/api/control/resume_plan', { batch_id: batchId }));
      await refreshOptions();
    }

    async function startPlan() {
      const planName = document.getElementById('planName').value;
      const starterFile = document.getElementById('starterFile').value;
      applyFormToJson();
      const configText = document.getElementById('planConfig').value;
      showResult(await api('/api/control/start_plan', { plan_name: planName, starter_file: starterFile, config_json: configText }));
      await refreshOptions();
    }

    function showOutputPreview(idx) {
      const row = outputFiles[idx];
      const box = document.getElementById('outputPreview');
      if (!row) {
        box.textContent = '(no preview)';
        return;
      }
      const header = [
        `file: ${row.relative_path}`,
        `mnt: ${row.mnt_path}`,
        `updated: ${fmtTs(row.updated_at)}`,
        `size: ${row.size_bytes} bytes`,
        '',
      ].join('\\n');
      box.textContent = header + (row.preview || '(preview unavailable for this file type)');
    }

    async function loadBatchOutputs() {
      const batchId = document.getElementById('outputBatch').value;
      if (!batchId) return;
      const data = await api('/api/control/batch_outputs', { batch_id: batchId });
      showResult(data);
      if (!data.ok) {
        document.getElementById('outputMeta').textContent = '';
        document.getElementById('outputFiles').innerHTML = '';
        document.getElementById('outputPreview').textContent = '(no output)';
        return;
      }
      outputFiles = data.files || [];
      document.getElementById('outputMeta').textContent =
        `Batch ${data.batch_id} | Plan ${data.plan} | ${outputFiles.length} files | ${data.batch_mnt_path}`;
      const rows = outputFiles.map((f, i) => `
        <tr>
          <td class="mono">${f.relative_path}</td>
          <td>${f.size_bytes}</td>
          <td>${fmtTs(f.updated_at)}</td>
          <td><button onclick="showOutputPreview(${i})">Preview</button></td>
        </tr>
      `).join('');
      document.getElementById('outputFiles').innerHTML = `
        <table>
          <thead><tr><th>File</th><th>Bytes</th><th>Updated</th><th></th></tr></thead>
          <tbody>${rows || '<tr><td colspan="4" class="k">(no files)</td></tr>'}</tbody>
        </table>
      `;
      showOutputPreview(0);
    }

    refreshOptions().catch(console.error);
  </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    shared_path: Path
    config: dict[str, Any]

    def _send_json(self, data: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _gpu_ssh(self, remote_cmd: str, timeout_s: int = 180) -> dict[str, Any]:
        cmd = ["ssh", "-o", "BatchMode=yes", "gpu", "bash", "-lc", remote_cmd]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            return {
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
                "cmd": " ".join(cmd),
            }
        except subprocess.TimeoutExpired as e:
            return {
                "ok": False,
                "returncode": -1,
                "stdout": (e.stdout or "")[-2000:] if isinstance(e.stdout, str) else "",
                "stderr": (e.stderr or "")[-2000:] if isinstance(e.stderr, str) else "",
                "cmd": " ".join(cmd),
                "error": f"timeout after {timeout_s}s",
            }

    def _control_options(self) -> dict[str, Any]:
        brain = load_brain_state(self.shared_path)
        active = brain.get("active_batches", {}) if isinstance(brain.get("active_batches"), dict) else {}
        plans = discover_plans(self.shared_path)
        recent_batches = discover_recent_batches(self.shared_path, active)
        plan_defaults = {p: default_plan_config(self.shared_path, p) for p in plans}
        plan_starters = {p: discover_plan_starters(self.shared_path, p) for p in plans}
        plan_input_files = {p: discover_plan_input_files(self.shared_path, p) for p in plans}
        plan_default_starter = {p: ("plan.md" if "plan.md" in plan_starters[p] else (plan_starters[p][0] if plan_starters[p] else "")) for p in plans}
        plan_inputs = {
            p: {starter: plan_input_help(self.shared_path, p, starter) for starter in plan_starters[p]}
            for p in plans
        }
        active_meta = []
        for batch_id, meta in active.items():
            if not isinstance(meta, dict):
                continue
            active_meta.append(
                {
                    "batch_id": batch_id,
                    "plan": meta.get("plan", ""),
                    "started_at": meta.get("started_at", ""),
                }
            )
        active_meta.sort(key=lambda x: str(x.get("started_at") or ""), reverse=True)
        return {
            "ok": True,
            "active_batches": sorted(active.keys()),
            "active_batches_meta": active_meta,
            "resumable_batches": sorted(active.keys()),
            "recent_batches": recent_batches,
            "plans": plans,
            "plan_defaults": plan_defaults,
            "plan_starters": plan_starters,
            "plan_input_files": plan_input_files,
            "plan_default_starter": plan_default_starter,
            "plan_inputs": plan_inputs,
        }

    def _kill_plan(self, batch_id: str) -> dict[str, Any]:
        if not batch_id:
            return {"ok": False, "message": "batch_id is required"}
        local_kill = Path(__file__).resolve().parent / "kill_plan.py"
        cmd = f"python3 {local_kill} {batch_id} --keep-workers --keep-models --no-default-warm"
        out = run_shell(cmd, timeout_s=240)
        out["message"] = f"Killed batch {batch_id}" if out.get("ok") else f"Failed to kill batch {batch_id}"
        return out

    def _kill_all_active(self) -> dict[str, Any]:
        local_kill = Path(__file__).resolve().parent / "kill_plan.py"
        cmd = f"python3 {local_kill} --keep-workers --keep-models --no-default-warm"
        out = run_shell(cmd, timeout_s=300)
        out["message"] = "Killed all active batches" if out.get("ok") else "Failed to kill all active batches"
        return out

    def _return_default(self) -> dict[str, Any]:
        # Pass commands via stdin so pkill can't match its own shell's cmdline.
        script = (
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
        try:
            proc = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "gpu", "bash", "-s"],
                input=script,
                capture_output=True,
                text=True,
                timeout=240,
            )
            return {
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
                "cmd": "ssh gpu bash -s (stdin: kill agents, restart startup.py)",
                "message": "Returned system to default startup state",
            }
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "message": "Returned system to default startup state",
                "error": "timeout after 240s",
                "cmd": "ssh gpu bash -s",
            }

    def _start_plan(self, plan_name: str, config_json: str, starter_file: str = "") -> dict[str, Any]:
        if not plan_name:
            return {"ok": False, "message": "plan_name is required"}
        plans = set(discover_plans(self.shared_path))
        if plan_name not in plans:
            return {"ok": False, "message": f"unknown plan: {plan_name}"}

        try:
            cfg_obj = json.loads(config_json) if config_json.strip() else {}
            if not isinstance(cfg_obj, dict):
                raise ValueError("config JSON must be an object")
        except Exception as e:
            return {"ok": False, "message": f"invalid config JSON: {e}"}

        cfg_payload = json.dumps(cfg_obj, separators=(",", ":"))
        plan_path = f"/mnt/shared/plans/shoulders/{plan_name}"
        cfg_tmp = f"/tmp/dashboard_submit_{plan_name}.json"
        starter_arg = starter_file.strip()
        starter_opt = f" --plan-file {shlex.quote(starter_arg)}" if starter_arg else ""
        script = (
            "set -e\n"
            f"cat > {cfg_tmp} <<'__CFG__'\n"
            f"{cfg_payload}\n"
            "__CFG__\n"
            f"python3 /mnt/shared/agents/submit.py {plan_path}{starter_opt} --config \"$(cat {cfg_tmp})\"\n"
            f"rm -f {cfg_tmp}\n"
        )
        try:
            proc = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "gpu", "bash", "-s"],
                input=script,
                capture_output=True,
                text=True,
                timeout=120,
            )
            out = {
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
                "cmd": "ssh gpu bash -s (stdin: write config JSON to temp file, run submit.py)",
            }
        except subprocess.TimeoutExpired as e:
            out = {
                "ok": False,
                "returncode": -1,
                "stdout": (e.stdout or "")[-2000:] if isinstance(e.stdout, str) else "",
                "stderr": (e.stderr or "")[-2000:] if isinstance(e.stderr, str) else "",
                "cmd": "ssh gpu bash -s",
                "error": "timeout after 120s",
            }
        display_starter = starter_arg or "plan.md"
        out["message"] = (
            f"Submitted plan {plan_name} ({display_starter})"
            if out.get("ok")
            else f"Failed to submit plan {plan_name} ({display_starter})"
        )
        return out

    def _resume_plan(self, batch_id: str) -> dict[str, Any]:
        if not batch_id:
            return {"ok": False, "message": "batch_id is required"}

        brain = load_brain_state(self.shared_path)
        active = brain.get("active_batches", {}) if isinstance(brain.get("active_batches"), dict) else {}
        meta = active.get(batch_id) if isinstance(active.get(batch_id), dict) else {}
        if not meta:
            return {"ok": False, "message": f"batch_id not found in active_batches: {batch_id}"}

        plan_name = str(meta.get("plan") or "").strip()
        if not plan_name:
            plan_dir = str(meta.get("plan_dir") or "").strip()
            if plan_dir:
                plan_name = Path(plan_dir).name
        if not plan_name:
            return {"ok": False, "message": f"could not resolve plan name for batch {batch_id}"}

        cfg = meta.get("config") if isinstance(meta.get("config"), dict) else {}
        cfg = dict(cfg)
        cfg["RUN_MODE"] = "resume"
        cfg["BATCH_ID"] = batch_id
        cfg_json = json.dumps(cfg)

        out = self._start_plan(plan_name, cfg_json)
        if out.get("ok"):
            out["message"] = f"Resumed batch {batch_id} ({plan_name})"
        else:
            out["message"] = f"Failed to resume batch {batch_id} ({plan_name})"
        return out

    def _batch_outputs(self, batch_id: str) -> dict[str, Any]:
        if not batch_id:
            return {"ok": False, "message": "batch_id is required"}
        return collect_batch_outputs(self.shared_path, batch_id)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            self._send_json(summarize(self.shared_path, self.config))
            return
        if parsed.path == "/api/control/options":
            self._send_json(self._control_options())
            return
        if parsed.path in ("/", "/index.html"):
            self._send_html(HTML)
            return
        if parsed.path == "/controls":
            self._send_html(CONTROLS_HTML)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        payload = self._read_json_body()

        if parsed.path == "/api/control/kill_plan":
            self._send_json(self._kill_plan(str(payload.get("batch_id", "")).strip()))
            return
        if parsed.path == "/api/control/kill_all_active":
            self._send_json(self._kill_all_active())
            return
        if parsed.path == "/api/control/return_default":
            self._send_json(self._return_default())
            return
        if parsed.path == "/api/control/resume_plan":
            self._send_json(self._resume_plan(str(payload.get("batch_id", "")).strip()))
            return
        if parsed.path == "/api/control/batch_outputs":
            self._send_json(self._batch_outputs(str(payload.get("batch_id", "")).strip()))
            return
        if parsed.path == "/api/control/start_plan":
            self._send_json(
                self._start_plan(
                    str(payload.get("plan_name", "")).strip(),
                    str(payload.get("config_json", "")).strip(),
                    str(payload.get("starter_file", "")).strip(),
                )
            )
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local orchestration dashboard")
    default_config = Path(__file__).resolve().parent.parent / "shared" / "agents" / "config.json"
    parser.add_argument("--config", default=str(default_config), help="Path to config.json")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    shared_path = resolve_shared_path(config_path, config)

    class BoundHandler(DashboardHandler):
        pass

    BoundHandler.shared_path = shared_path
    BoundHandler.config = config

    server = ThreadingHTTPServer((args.host, args.port), BoundHandler)
    print(f"Dashboard: http://{args.host}:{args.port}")
    print(f"Shared path: {shared_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
