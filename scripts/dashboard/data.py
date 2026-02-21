"""Task loading and dashboard data aggregation."""

from datetime import datetime
from pathlib import Path
from typing import Any

from .alerts import collect_recent_batch_completion_alerts, collect_recent_batch_failure_alerts
from .chains import build_batch_chain
from .plans import find_batch_dir
from .utils import (
    HEARTBEAT_MAX_S,
    file_mtime_iso,
    heartbeat_age_seconds,
    iter_task_files,
    load_json,
)
from .workers import load_brain_heartbeat, load_brain_state, load_gpu_telemetry, load_worker_rows

_TASK_CACHE: dict[str, tuple[float, dict[str, Any] | None]] = {}


def task_sort_key(task: dict[str, Any]) -> str:
    """Get sort key for task ordering."""
    return (
        task.get("last_attempt_at")
        or task.get("completed_at")
        or task.get("started_at")
        or task.get("created_at")
        or ""
    )


def to_task_view(task: dict[str, Any]) -> dict[str, Any]:
    """Convert task to view format for display."""
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


def lane_view(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Convert task list to view format with limit."""
    return [to_task_view(t) for t in items[:limit]]


def _load_task_cached(task_file: Path) -> dict[str, Any] | None:
    """Load task file with mtime cache to avoid reparsing unchanged JSON."""
    cache_key = str(task_file)
    try:
        mtime = task_file.stat().st_mtime
    except Exception:
        _TASK_CACHE.pop(cache_key, None)
        return None
    cached = _TASK_CACHE.get(cache_key)
    if cached and cached[0] == mtime:
        return cached[1]
    task = load_json(task_file)
    _TASK_CACHE[cache_key] = (mtime, task)
    return task


def _task_batch_id(task: dict[str, Any]) -> str:
    return str(task.get("batch_id") or "").strip()


def list_tasks(shared_path: Path, focus_batch_ids: set[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    """List tasks from all lanes."""
    lanes = {
        "queue": [],
        "processing": [],
        "complete": [],
        "failed": [],
    }
    focus_set = {b for b in (focus_batch_ids or set()) if b}
    for lane in lanes:
        if lane in {"complete", "failed"} and not focus_set:
            continue
        folder = shared_path / "tasks" / lane
        for task_file in iter_task_files(folder) or []:
            task = _load_task_cached(task_file)
            if not task:
                continue
            batch_id = _task_batch_id(task)
            if focus_set and lane in {"complete", "failed"} and batch_id not in focus_set:
                continue
            status = str(task.get("status") or "").strip().lower()
            # Some failed tasks can be materialized under complete/.
            # Rebucket by explicit status so dashboard failure counts stay accurate.
            if status in {"failed", "blocked_cloud", "abandoned", "error"}:
                if focus_set and batch_id not in focus_set:
                    continue
                lanes["failed"].append(task)
            else:
                lanes[lane].append(task)
    for lane in lanes:
        lanes[lane].sort(key=task_sort_key, reverse=True)
    return lanes


def list_private_tasks(shared_path: Path, focus_batch_ids: set[str] | None = None) -> list[dict[str, Any]]:
    """List private tasks from brain directory."""
    private_dir = shared_path / "brain" / "private_tasks"
    focus_set = {b for b in (focus_batch_ids or set()) if b}
    rows: list[dict[str, Any]] = []
    for task_file in iter_task_files(private_dir) or []:
        task = _load_task_cached(task_file)
        if task:
            if focus_set and _task_batch_id(task) not in focus_set:
                continue
            rows.append(task)
    rows.sort(key=task_sort_key, reverse=True)
    return rows


def count_by_batch(tasks: dict[str, list[dict[str, Any]]], batch_id: str) -> dict[str, int]:
    """Count tasks per lane for a batch."""
    return {
        "queue": sum(1 for t in tasks["queue"] if t.get("batch_id") == batch_id),
        "processing": sum(1 for t in tasks["processing"] if t.get("batch_id") == batch_id),
        "complete": sum(1 for t in tasks["complete"] if t.get("batch_id") == batch_id),
        "failed": sum(1 for t in tasks["failed"] if t.get("batch_id") == batch_id),
        "private": sum(1 for t in tasks["private"] if t.get("batch_id") == batch_id),
    }


def is_system_meta_task(task: dict[str, Any]) -> bool:
    """Check if task is a system meta task (model loading)."""
    name = str(task.get("name") or "").lower()
    return name.startswith("load_llm") or name.startswith("load_worker_model")


def _resolve_batch_plan(shared_path: Path, batch_id: str, active_batches: dict[str, Any]) -> str:
    """Resolve plan name from active metadata or history folder lookup."""
    meta = active_batches.get(batch_id) if isinstance(active_batches.get(batch_id), dict) else {}
    if meta:
        plan_name = str(meta.get("plan") or "").strip()
        if plan_name:
            return plan_name
    found = find_batch_dir(shared_path, batch_id)
    if found:
        return str(found[0] or "")
    return ""


def summarize(
    shared_path: Path,
    config: dict[str, Any],
    selected_batch_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Generate full dashboard summary data."""
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
    active_batch_ids = set(str(bid) for bid in active_batches.keys())
    selected_set = {str(b).strip() for b in (selected_batch_ids or []) if str(b).strip()}
    focus_batch_ids = active_batch_ids | selected_set

    lanes = list_tasks(shared_path, focus_batch_ids=focus_batch_ids)
    private_tasks = list_private_tasks(shared_path, focus_batch_ids=focus_batch_ids)
    lanes["private"] = private_tasks

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

    # Keep recent batch-fatal/completion events visible even if batch cards collapsed.
    alerts.extend(collect_recent_batch_failure_alerts(lanes["failed"]))
    alerts.extend(collect_recent_batch_completion_alerts(lanes))

    batches: dict[str, Any] = {}
    batch_chains: dict[str, Any] = {}
    engaged_batch_ids: set[str] = set()
    candidate_batch_ids: set[str] = set()
    candidate_batch_ids.update(active_batch_ids)
    candidate_batch_ids.update(selected_set)
    for lane in ("queue", "processing", "private", "complete", "failed"):
        for t in lanes.get(lane, []):
            bid = _task_batch_id(t)
            if bid:
                candidate_batch_ids.add(bid)

    for batch_id in sorted(candidate_batch_ids):
        meta = active_batches.get(batch_id) if isinstance(active_batches.get(batch_id), dict) else {}
        counts = count_by_batch(lanes, batch_id)
        is_engaged = (counts["queue"] + counts["processing"] + counts["private"] + counts["failed"]) > 0
        # Also show batches that have completed or failed tasks (not just actively running)
        has_any_tasks = (counts["complete"] + counts["failed"]) > 0 or is_engaged
        if not has_any_tasks:
            # Hide stale/idle leftovers that linger in brain state.
            continue
        plan_name = str(meta.get("plan") or "").strip() if isinstance(meta, dict) else ""
        if not plan_name:
            plan_name = _resolve_batch_plan(shared_path, batch_id, active_batches)
        batch_info: dict[str, Any] = {
            "plan": plan_name or None,
            "started_at": meta.get("started_at") if isinstance(meta, dict) else None,
            "total_hint": meta.get("total_tasks") if isinstance(meta, dict) else None,
            "counts": counts,
            "priority": meta.get("priority", "normal") if isinstance(meta, dict) else "normal",
            "preemptible": meta.get("preemptible", True) if isinstance(meta, dict) else True,
            "is_active": batch_id in active_batch_ids,
        }
        # Include goal progress if this is a goal-driven batch
        goal = meta.get("goal") if isinstance(meta, dict) else None
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
        engaged_batch_ids.add(batch_id)

    # Keep lane tables focused on engaged or user-selected batches.
    lane_focus_ids = engaged_batch_ids | selected_set
    if lane_focus_ids:
        lane_source = {
            "queue": [t for t in lanes["queue"] if t.get("batch_id") in lane_focus_ids],
            "processing": [
                t for t in lanes["processing"]
                if t.get("batch_id") in lane_focus_ids or is_system_meta_task(t)
            ],
            "private": [t for t in lanes["private"] if t.get("batch_id") in lane_focus_ids],
            "complete": [t for t in lanes["complete"] if t.get("batch_id") in lane_focus_ids],
            "failed": [t for t in lanes["failed"] if t.get("batch_id") in lane_focus_ids],
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
        "tracked_batch_ids": sorted(lane_focus_ids),
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
