"""Worker and GPU status loading functions."""

import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .utils import (
    HEARTBEAT_BAD_S,
    HEARTBEAT_MAX_S,
    HEARTBEAT_WARN_S,
    heartbeat_age_seconds,
    load_json,
)

BENCHMARK_STATUS_MAX_AGE_S = 6 * 3600
BENCHMARK_IDLE_CLEAR_AGE_S = 5 * 60
TASK_NICKNAME_PATHS = (
    Path("/media/bryan/shared/plans/shoulders/benchmarking/docker/task_nicknames.json"),
    Path("/home/bryan/Desktop/shared/plans/shoulders/benchmarking/docker/task_nicknames.json"),
    Path("/mnt/shared/plans/shoulders/benchmarking/docker/task_nicknames.json"),
)
_TASK_NICKNAME_CACHE: dict[str, str] = {}
_TASK_NICKNAME_CACHE_MTIME: float = -1.0


def _count_task_samples(output_dir: Any) -> int | None:
    text = str(output_dir or "").strip()
    if not text:
        return None
    path = Path(text)
    if not path.exists() or not path.is_dir():
        return None
    for cand in sorted(path.glob("*.jsonl")):
        name = cand.name
        if name.endswith(".raw.jsonl"):
            continue
        if name.endswith("_eval_results.jsonl"):
            continue
        try:
            with cand.open("r", encoding="utf-8", errors="ignore") as f:
                return sum(1 for _ in f)
        except Exception:
            return None
    return None


def _parse_limit_int(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        num = int(float(text))
    except Exception:
        return None
    if num <= 0:
        return None
    return num


def _read_tail_text(path: Path, max_bytes: int = 350_000) -> str:
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            if size <= 0:
                return ""
            start = max(0, size - max_bytes)
            f.seek(start)
            data = f.read()
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _short_bbh_name(task_name: str) -> str:
    short = str(task_name or "").strip()
    if short.startswith("bbh_cot_fewshot_"):
        short = short[len("bbh_cot_fewshot_"):]
    nick = _task_nickname(short) or _task_nickname(task_name)
    if nick:
        return nick
    return short.replace("_", " ")


def _task_nickname(task_name: str) -> str:
    key = str(task_name or "").strip()
    if not key:
        return ""
    nick_map = _load_task_nicknames()
    return str(nick_map.get(key) or "")


def _load_task_nicknames() -> dict[str, str]:
    global _TASK_NICKNAME_CACHE, _TASK_NICKNAME_CACHE_MTIME
    cfg_path = None
    for cand in TASK_NICKNAME_PATHS:
        if cand.exists():
            cfg_path = cand
            break
    if cfg_path is None:
        return {}
    try:
        mtime = cfg_path.stat().st_mtime
    except Exception:
        return {}
    if _TASK_NICKNAME_CACHE and mtime == _TASK_NICKNAME_CACHE_MTIME:
        return _TASK_NICKNAME_CACHE
    raw = load_json(cfg_path)
    mapping = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            kk = str(k or "").strip()
            vv = str(v or "").strip()
            if kk and vv:
                mapping[kk] = vv
    _TASK_NICKNAME_CACHE = mapping
    _TASK_NICKNAME_CACHE_MTIME = mtime
    return _TASK_NICKNAME_CACHE


def _status_log_path(status_path: Path) -> Path | None:
    worker_bucket = status_path.parent.parent
    run_root = worker_bucket.parent
    log_path = run_root / f"{worker_bucket.name}.log"
    if log_path.exists():
        return log_path
    return None


def _file_mtime_dt(path: Path | None) -> datetime | None:
    if path is None:
        return None
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except Exception:
        return None


def _live_task_progress_from_log(status_path: Path, limit: Any) -> tuple[str, str]:
    """Return (task_label, sample_progress) from run log tail when available."""
    log_path = _status_log_path(status_path)
    if log_path is None:
        return ("", "")
    tail = _read_tail_text(log_path)
    if not tail:
        return ("", "")
    run_task = ""
    for m in re.finditer(r"--- Running task:\s*([a-z0-9_]+)\s*---", tail):
        run_task = str(m.group(1))
    live_task = ""
    for m in re.finditer(r"Building contexts for ([a-z0-9_]+) on rank", tail):
        live_task = str(m.group(1))
    if not live_task and not run_task:
        for m in re.finditer(r"Task:\s+([a-z0-9_]+)\s+\(", tail):
            live_task = str(m.group(1))
    if not live_task and not run_task:
        return ("", "")
    req_seen = None
    req_total = None
    # Progress bars are emitted with carriage returns; parse the last X/Y pair.
    for m in re.finditer(r"Requesting API:[^\n\r]*?(\d+)/(\d+)", tail):
        try:
            req_seen = int(m.group(1))
            req_total = int(m.group(2))
        except Exception:
            continue
    if isinstance(req_seen, int) and isinstance(req_total, int) and req_total > 0:
        base = run_task or live_task
        if base.startswith("bbh_cot_fewshot_"):
            base = "bbh"
        if base == "bbh":
            return ("bbh", f"{min(req_seen, req_total)}/{req_total}")
        nick = _task_nickname(base)
        if nick:
            base = nick
        else:
            base = base.replace("_", " ")
        return (base, f"{min(req_seen, req_total)}/{req_total}")

    lim = _parse_limit_int(limit)
    if live_task.startswith("bbh_cot_fewshot_"):
        if lim:
            return (f"bbh - {_short_bbh_name(live_task)}", f"{lim}/{lim}")
        return (f"bbh - {_short_bbh_name(live_task)}", "")
    nick = _task_nickname(live_task)
    if nick:
        if lim:
            return (nick, f"{lim}/{lim}")
        return (nick, "")
    if lim:
        return (live_task.replace("_", " "), f"{lim}/{lim}")
    return (live_task.replace("_", " "), "")


def _split_pairing_holds(shared_path: Path) -> dict[str, list[str]]:
    """Build synthetic dashboard hold labels for split-load partner GPUs."""
    holds: dict[str, list[str]] = {}
    split_dir = shared_path / "signals" / "split_llm"
    if not split_dir.exists():
        return holds
    for res_file in sorted(split_dir.glob("pair_*.json")):
        res = load_json(res_file)
        if not isinstance(res, dict):
            continue
        if str(res.get("status") or "") != "loading":
            continue
        members = [str(x) for x in (res.get("members") or []) if x]
        joined = res.get("joined") if isinstance(res.get("joined"), dict) else {}
        launcher = str(res.get("launcher") or "")
        group_id = str(res.get("group_id") or res_file.stem)
        if len(members) != 2:
            continue
        for worker in members:
            if worker == launcher:
                continue
            if worker not in joined:
                continue
            partner = members[0] if members[1] == worker else members[1]
            partner_short = str(partner).replace("gpu-", "")
            label = f"Pairing {partner_short}"
            holds.setdefault(worker, []).append(label)
    return holds


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


def _is_active_runtime_holding_phase(phase: str) -> bool:
    """Return True only for in-progress runtime transition phases."""
    text = str(phase or "").strip()
    if not text:
        return False
    terminal_markers = (
        "load_complete",
        "split_load_complete",
        "unload_complete",
    )
    if text in terminal_markers:
        return False
    if text.startswith("split_cleared"):
        return False
    return True


def _parse_iso(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _runtime_port(runtime: Any) -> int:
    text = str(runtime or "").strip()
    if not text:
        return 0
    m = re.search(r":(\d+)(?:/|$)", text)
    if not m:
        return 0
    try:
        return int(m.group(1))
    except Exception:
        return 0


def _task_status_meta(status: dict[str, Any]) -> dict[str, Any]:
    tasks = status.get("tasks") if isinstance(status.get("tasks"), dict) else {}
    ordered = [
        str(x).strip()
        for x in (status.get("tasks_requested") or [])
        if str(x).strip()
    ]
    if not ordered:
        ordered = [str(k).strip() for k in tasks.keys() if str(k).strip()]
    total = len(ordered) if ordered else len(tasks)
    if total <= 0:
        return {"current_task": "", "progress": "", "active": False}
    terminal = 0
    terminal_states = {
        "completed",
        "passed",
        "evaluated",
        "failed",
        "error",
        "error_missing_samples",
        "generated_partial",
    }
    current = ""
    current_idx = 0
    sample_progress = ""
    for name in ordered:
        task_data = tasks.get(name) if isinstance(tasks.get(name), dict) else {}
        state = str((task_data or {}).get("state") or "").strip().lower()
        if state in terminal_states:
            terminal += 1
            continue
        if not current:
            current = name
            current_idx = terminal + 1
            if state == "running":
                current = f"{name} (running)"
                lim = _parse_limit_int(status.get("limit"))
                seen = _count_task_samples((task_data or {}).get("output_dir"))
                if lim is not None and seen is not None:
                    sample_progress = f"{min(seen, lim)}/{lim}"
            elif state == "failed":
                current = f"{name} (failed)"
    if not current and terminal < total:
        current = "(in-progress)"
        current_idx = terminal + 1
    progress_label = f"{terminal}/{total} tasks"
    if current_idx > 0 and total > 0:
        progress_label = f"task {current_idx}/{total}"
    return {
        "current_task": current,
        "progress": progress_label,
        "active": terminal < total,
        "current_index": current_idx,
        "task_total": total,
        "sample_progress": sample_progress,
    }


def _status_candidates(shared_path: Path) -> list[tuple[str, Path]]:
    logs_root = shared_path / "logs" / "benchmarks"
    if not logs_root.exists():
        return []
    out: list[tuple[str, Path]] = []
    for p in (logs_root / "bench-reasoning" / "history").glob("**/status.json"):
        out.append(("reasoning", p))
    for p in (logs_root / "bench-pipeline" / "history").glob("**/*_status.json"):
        out.append(("pipeline", p))
    for p in (logs_root / "bench-code" / "history").glob("**/status.json"):
        out.append(("code", p))
    return out


def _active_benchmark_by_port(shared_path: Path) -> dict[int, dict[str, Any]]:
    """Map runtime port -> active benchmark suite/task/progress via shared status files."""
    best: dict[int, dict[str, Any]] = {}
    newest_seen_ts: dict[int, float] = {}
    now = datetime.now(timezone.utc)
    for suite, path in _status_candidates(shared_path):
        status = load_json(path)
        if not isinstance(status, dict):
            continue
        port = _runtime_port(status.get("runtime") or status.get("runtime_base"))
        if port <= 0:
            continue

        updated = _parse_iso(status.get("updated_at") or status.get("time") or status.get("run_start"))
        if not updated:
            updated = _parse_iso(status.get("run_start"))
        log_path = _status_log_path(path)
        log_updated = _file_mtime_dt(log_path)
        freshness = updated
        if log_updated and (freshness is None or log_updated > freshness):
            freshness = log_updated
        age_s = int((now - freshness).total_seconds()) if freshness else None
        cur_ts = freshness.timestamp() if freshness else 0.0
        prev_latest = newest_seen_ts.get(port)
        if prev_latest is None or cur_ts > prev_latest:
            newest_seen_ts[port] = cur_ts

        current_task = ""
        progress = ""
        is_active = False
        if suite == "pipeline":
            completed = int(status.get("completed_stages") or 0)
            total = int(status.get("total_stages") or 0)
            current_task = str(status.get("current_test") or "").strip()
            progress = f"{completed}/{total} tests" if total > 0 else ""
            is_active = bool(total > 0 and completed < total)
        else:
            meta = _task_status_meta(status)
            current_task = str(meta.get("current_task") or "").strip()
            progress = str(meta.get("progress") or "").strip()
            is_active = bool(meta.get("active"))
            current_idx = int(meta.get("current_index") or 0)
            task_total = int(meta.get("task_total") or 0)
            sample_progress = str(meta.get("sample_progress") or "").strip()
            limit = str(status.get("limit") or "").strip()
            if sample_progress:
                if current_idx > 0 and task_total > 0:
                    progress = f"{sample_progress} | task {current_idx}/{task_total}"
                else:
                    progress = sample_progress
            elif limit:
                progress = f"{progress} | limit {limit}" if progress else f"limit {limit}"
            live_task, live_progress = _live_task_progress_from_log(path, status.get("limit"))
            if live_task:
                if live_progress:
                    current_task = f"{live_task} - {live_progress}"
                    progress = ""
                else:
                    current_task = live_task
            if current_task and current_idx > 0 and task_total > 0:
                # Include both positions: suite index and in-suite task progress.
                if " - " in current_task:
                    left, right = current_task.split(" - ", 1)
                    current_task = f"{left} {current_idx}/{task_total} - {right}"
                else:
                    current_task = f"{current_task} {current_idx}/{task_total}"

        # keep stale zombie rows out of the active worker view
        if age_s is not None and age_s > BENCHMARK_STATUS_MAX_AGE_S:
            is_active = False
        if not is_active:
            continue

        prev = best.get(port)
        prev_ts = prev.get("updated_ts") if isinstance(prev, dict) else None
        if prev is not None and isinstance(prev_ts, (int, float)) and prev_ts > cur_ts:
            continue
        best[port] = {
            "suite": suite,
            "current_task": current_task,
            "progress": progress,
            "age_s": age_s,
            "updated_ts": cur_ts,
        }
    # If a newer status exists on the same runtime port (even terminal), drop older
    # active entries so workers do not show stale "holding" labels forever.
    for port in list(best.keys()):
        entry = best.get(port) or {}
        entry_ts = entry.get("updated_ts")
        latest_ts = newest_seen_ts.get(port)
        if not isinstance(entry_ts, (int, float)) or not isinstance(latest_ts, (int, float)):
            continue
        if latest_ts > entry_ts:
            best.pop(port, None)
    return best


def load_worker_rows(shared_path: Path, processing_tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Load worker status rows from heartbeat files."""
    by_worker: dict[str, list[str]] = {}
    for t in processing_tasks:
        worker = t.get("assigned_to")
        if worker:
            by_worker.setdefault(worker, []).append(t.get("name") or t.get("task_id") or "(task)")

    rows: list[dict[str, Any]] = []
    pairing_holds = _split_pairing_holds(shared_path)
    benchmark_by_port = _active_benchmark_by_port(shared_path)

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
        for label in pairing_holds.get(str(name), []):
            if label not in held:
                held.append(label)
        last_thermal_event = hb.get("last_thermal_event") if isinstance(hb.get("last_thermal_event"), dict) else None
        thermal_reasons = hb.get("thermal_reasons") if isinstance(hb.get("thermal_reasons"), list) else []
        thermal_cause = "none"
        if last_thermal_event and isinstance(last_thermal_event.get("reasons"), list):
            thermal_cause = classify_thermal_cause(last_thermal_event.get("reasons"))
        elif thermal_reasons:
            thermal_cause = classify_thermal_cause(thermal_reasons)
        loaded_model = str(hb.get("loaded_model") or "").strip()
        configured_model = str(hb.get("configured_model") or "").strip()
        runtime_placement = str(hb.get("runtime_placement") or "").strip()
        runtime_group_id = str(hb.get("runtime_group_id") or "").strip()
        runtime_state = str(hb.get("runtime_state") or "").strip()
        runtime_phase = str(hb.get("runtime_transition_phase") or "").strip()
        runtime_api_base = str(hb.get("runtime_api_base") or "").strip()
        model_loaded = bool(hb.get("model_loaded"))
        if model_loaded and loaded_model:
            if runtime_placement == "split_gpu":
                host_display = f"{loaded_model} [split"
                if runtime_group_id:
                    host_display += f" {runtime_group_id}"
                host_display += "]"
            else:
                host_display = loaded_model
        elif model_loaded:
            host_display = "[loaded]"
        elif runtime_state.startswith("loading"):
            host_display = "[loading]"
        else:
            host_display = "-"

        if runtime_state and runtime_state != "cold":
            state_display = runtime_state
        else:
            state_display = hb.get("state", "?")

        if _is_active_runtime_holding_phase(runtime_phase) and runtime_phase not in held:
            held.append(runtime_phase)

        active_suite = ""
        active_task = ""
        active_progress = ""
        try:
            port_key = int(hb.get("runtime_port") or 0)
        except Exception:
            port_key = 0
        if port_key > 0 and port_key in benchmark_by_port:
            bench = benchmark_by_port.get(port_key) or {}
            bench_age_s = bench.get("age_s")
            gpu_util = hb.get("gpu_util_percent")
            idle_runtime = (
                not (hb.get("active_tasks") or [])
                and int(hb.get("active_workers") or 0) <= 0
                and str(runtime_state or "").startswith("ready")
            )
            util_pct = 0
            try:
                util_pct = int(float(gpu_util))
            except Exception:
                util_pct = 0
            if not (
                idle_runtime
                and util_pct <= 1
                and isinstance(bench_age_s, int)
                and bench_age_s > BENCHMARK_IDLE_CLEAR_AGE_S
            ):
                active_suite = str(bench.get("suite") or "").strip()
                active_task = str(bench.get("current_task") or "").strip()
                active_progress = str(bench.get("progress") or "").strip()
                if active_task:
                    task_label = active_task
                    if active_progress:
                        task_label = f"{active_task} ({active_progress})"
                    if task_label not in held:
                        held.insert(0, task_label)

        rows.append({
            "name": name,
            "gpu_id": hb.get("gpu_id"),
            "type": "gpu",
            "state": state_display,
            "model_loaded": hb.get("model_loaded"),
            "loaded_model": hb.get("loaded_model"),
            "loaded_tier": hb.get("loaded_tier"),
            "runtime_placement": hb.get("runtime_placement"),
            "runtime_group_id": hb.get("runtime_group_id"),
            "host": host_display,
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
            "active_suite": active_suite,
            "active_task": active_task,
            "active_progress": active_progress,
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
