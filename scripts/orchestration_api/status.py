"""Bounded read-only rig state for the gateway and diagnostic tools."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any

from .utils import HEARTBEAT_BAD_S, heartbeat_age_seconds, load_json
from . import hardware


def _age(value: Any, path: Path) -> int | None:
    age = heartbeat_age_seconds(value) if isinstance(value, str) else None
    if age is not None:
        return age
    try:
        return max(0, int(datetime.now().timestamp() - path.stat().st_mtime))
    except OSError:
        return None


def _worker(path: Path, *, name: str | None = None) -> dict[str, Any]:
    data = load_json(path) or {}
    age = _age(data.get("last_updated") or data.get("updated_at") or data.get("timestamp"), path)
    heartbeat_status = "missing" if not data else "stale" if age is None or age > HEARTBEAT_BAD_S else "ok"
    fresh = heartbeat_status == "ok"
    return {
        "name": name or str(data.get("name") or path.parent.name),
        "gpu_id": data.get("gpu_id"),
        "state": (data.get("state") or data.get("status") or "online") if fresh else heartbeat_status,
        "reported_state": data.get("state") or data.get("status"),
        "loaded_model": data.get("loaded_model") if fresh else None,
        "last_reported_model": data.get("loaded_model"),
        "model_loaded": bool(data.get("model_loaded")) if fresh else False,
        "runtime_state": data.get("runtime_state") if fresh else None,
        "busy": bool(data.get("busy") or data.get("active_tasks")) if fresh else False,
        "busy_lane": data.get("busy_lane") if fresh else None,
        "busy_owner": data.get("busy_owner") if fresh else None,
        "age_s": age,
        "heartbeat_status": heartbeat_status,
        "gpu_temp_c": data.get("gpu_temp_c") if fresh else None,
    }


def snapshot(shared: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Read current state and heartbeat files without scanning tasks or invoking subprocesses."""
    state = load_json(shared / "brain" / "state.json") or {}
    brain_path = shared / "heartbeats" / "brain.json"
    if not brain_path.is_file():
        brain_path = shared / "brain" / "heartbeat.json"
    active = state.get("active_batches") if isinstance(state.get("active_batches"), dict) else {}
    batches = {
        str(key): {
            "plan": value.get("plan"),
            "display_name": value.get("display_name"),
            "started_at": value.get("started_at"),
        }
        for key, value in active.items() if isinstance(value, dict)
    }
    policy = load_json(shared / "brain" / "control_policy.json") or {}
    recent = _recent(shared)
    task_active = bool(batches)
    hardware_state = hardware.current()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware_state,
        "task_mode": "active" if task_active else "idle",
        "brain": _worker(brain_path, name="brain") if task_active else None,
        "workers": ([_worker(path) for path in sorted((shared / "gpus").glob("gpu_*/heartbeat.json"))]
                    if task_active else []),
        "active_batches": batches,
        "active_batch_count": len(batches),
        "recent_batches": [{key: row.get(key) for key in ("batch_id", "plan", "status", "completed_at")}
                           for row in recent],
        "control_policy": {"boot_profile": policy.get("boot_profile")},
        "visible_gpu_count": len(hardware_state.get("gpu_ids") or []),
    }


def _recent(shared: Path) -> list[dict[str, Any]]:
    path = shared / "brain" / "recent_batches.json"
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(rows, list):
        return []
    cutoff = datetime.now() - timedelta(hours=24)
    result = []
    for row in rows[:100]:
        if not isinstance(row, dict):
            continue
        try:
            completed = datetime.fromisoformat(str(row.get("completed_at")))
            if completed.replace(tzinfo=None) >= cutoff:
                result.append(row)
        except ValueError:
            continue
    return result


def activity(shared: Path, batch_id: str, after: int = 0, limit: int = 100) -> dict[str, Any]:
    """Page through one current or last-day job journal using byte offsets."""
    state = load_json(shared / "brain" / "state.json") or {}
    active = state.get("active_batches") if isinstance(state.get("active_batches"), dict) else {}
    meta = active.get(batch_id) if isinstance(active.get(batch_id), dict) else None
    if meta is None:
        meta = next((row for row in _recent(shared) if row.get("batch_id") == batch_id), None)
    if not meta:
        raise ValueError("batch is neither active nor recent")
    directory = Path(str(meta.get("batch_dir") or "")).resolve()
    if not directory.is_relative_to((shared / "plans").resolve()):
        raise ValueError("batch directory is outside the plans root")
    path = directory / "logs" / "batch_events.jsonl"
    if not path.resolve().is_relative_to((shared / "plans").resolve()):
        raise ValueError("batch journal is outside the plans root")
    if after < 0 or limit < 1 or limit > 100:
        raise ValueError("invalid activity page")
    if not path.is_file():
        if after:
            raise ValueError("activity cursor exceeds journal size")
        return {"batch_id": batch_id, "events": [], "next_after": 0, "has_more": False}
    events = []
    with path.open("rb") as stream:
        stream.seek(0, 2)
        end = stream.tell()
        if after > end:
            raise ValueError("activity cursor exceeds journal size")
        stream.seek(after)
        if after and _previous_byte(stream, after) != b"\n":
            stream.readline()
        start = stream.tell()
        while len(events) < limit and stream.tell() - start < 262_144:
            line = stream.readline(16_384)
            if not line:
                break
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("batch_id") == batch_id:
                events.append({key: row.get(key) for key in
                               ("event", "recorded_at", "task_id", "task_name", "worker", "recorded_status", "batch_status")})
        cursor = stream.tell()
    return {"batch_id": batch_id, "events": events, "next_after": cursor, "has_more": cursor < end}


def _previous_byte(stream, position: int) -> bytes:
    stream.seek(position - 1)
    value = stream.read(1)
    stream.seek(position)
    return value


def diagnostics(shared: Path, config: dict[str, Any]) -> dict[str, Any]:
    status = snapshot(shared, config)
    expected = {str(row.get("name") or f"gpu-{row.get('id')}") for row in config.get("gpus", [])
                if isinstance(row, dict)}
    observed = {row["name"] for row in status["workers"]}
    hardware_state = status["hardware"]
    checks = [{"name": "hardware", "ok": hardware_state["ok"],
               "reason": hardware_state.get("error") or hardware_state.get("guard_reason")}]
    # Task-mode workers are only required while a legacy prepared batch is active.
    if status["active_batch_count"]:
        checks.append({"name": "brain_heartbeat", "ok": status["brain"]["heartbeat_status"] == "ok",
                       "age_s": status["brain"]["age_s"]})
        checks.extend({"name": f"worker:{row['name']}", "ok": row["heartbeat_status"] == "ok",
                       "age_s": row["age_s"]} for row in status["workers"])
        checks.extend({"name": f"worker:{name}", "ok": False, "reason": "missing heartbeat"}
                      for name in sorted(expected - observed))
    return {"generated_at": status["generated_at"], "ok": all(row["ok"] for row in checks),
            "checks": checks}
