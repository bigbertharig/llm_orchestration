#!/usr/bin/env python3
"""GPU worker watchdog: restart missing workers and escalate on flapping."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def run(cmd: str, timeout: int = 60) -> tuple[int, str, str]:
    p = subprocess.run(cmd, shell=True, executable="/bin/bash", capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def read_local_cpu_temp_c() -> int | None:
    temp_path = Path("/sys/class/thermal/thermal_zone0/temp")
    if not temp_path.exists():
        return None
    try:
        raw = temp_path.read_text(encoding="utf-8").strip()
        return int(int(raw) / 1000)
    except Exception:
        return None


def read_gpu_rig_cpu_temp_c(shared: Path) -> int | None:
    """Read CPU temp from the host that owns worker processes."""
    if str(shared).startswith("/mnt/shared"):
        return read_local_cpu_temp_c()
    rc, out, _ = run("ssh -o BatchMode=yes gpu \"cat /sys/class/thermal/thermal_zone0/temp\"", timeout=10)
    if rc != 0:
        return None
    txt = (out or "").strip()
    if not txt:
        return None
    try:
        return int(int(txt) / 1000)
    except Exception:
        return None


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def expected_workers(cfg: dict[str, Any]) -> list[str]:
    workers = []
    ws = cfg.get("workers") if isinstance(cfg.get("workers"), list) else []
    for w in ws:
        if isinstance(w, dict) and w.get("name"):
            workers.append(str(w["name"]))
    if workers:
        return sorted(set(workers))

    brain_ids = set(cfg.get("brain", {}).get("gpus", []))
    gs = cfg.get("gpus") if isinstance(cfg.get("gpus"), list) else []
    for g in gs:
        if not isinstance(g, dict):
            continue
        gid = g.get("id")
        name = g.get("name")
        if name and gid not in brain_ids:
            workers.append(str(name))
    return sorted(set(workers))


def worker_age_s(shared: Path, worker_name: str) -> int | None:
    try:
        gid = int(worker_name.split("-")[-1])
    except Exception:
        return None
    hb = shared / "gpus" / f"gpu_{gid}" / "heartbeat.json"
    if not hb.exists():
        return None
    d = load_json(hb, {})
    ts = parse_ts(d.get("last_updated"))
    if not ts:
        return None
    return int((datetime.now() - ts).total_seconds())


def llm_queue_count(shared: Path) -> int:
    qdir = shared / "tasks" / "queue"
    n = 0
    for p in qdir.glob("*.json"):
        d = load_json(p, {})
        if d.get("task_class") == "llm":
            n += 1
    return n


def active_split_meta_by_worker(shared: Path) -> dict[str, dict[str, Any]]:
    """Return workers currently assigned an in-flight load_split_llm meta task."""
    processing_dir = shared / "tasks" / "processing"
    active: dict[str, dict[str, Any]] = {}
    for p in processing_dir.glob("*.json"):
        task = load_json(p, {})
        if not isinstance(task, dict):
            continue
        if str(task.get("command", "")).strip() != "load_split_llm":
            continue
        if str(task.get("status", "")).strip() != "processing":
            continue
        worker = str(task.get("assigned_to", "")).strip()
        if not worker:
            continue
        started_at = parse_ts(task.get("started_at"))
        age_s = int((datetime.now() - started_at).total_seconds()) if started_at else None
        active[worker] = {
            "task_id": str(task.get("task_id", "")).strip() or p.stem,
            "started_at": task.get("started_at"),
            "age_s": age_s,
            "group_ids": [str(g.get("id", "")).strip() for g in (task.get("candidate_groups") or []) if isinstance(g, dict)],
            "target_model": str(task.get("target_model", "")).strip(),
        }
    return active


def active_split_reservations_by_worker(shared: Path) -> dict[str, dict[str, Any]]:
    """Return workers participating in active split reservations."""
    split_dir = shared / "signals" / "split_llm"
    active: dict[str, dict[str, Any]] = {}
    active_statuses = {"waiting_partner", "joining", "loading"}
    for p in split_dir.glob("*.json"):
        reservation = load_json(p, {})
        if not isinstance(reservation, dict):
            continue
        status = str(reservation.get("status", "")).strip()
        if status not in active_statuses:
            continue
        launcher = str(reservation.get("launcher", "")).strip()
        members = [str(m).strip() for m in (reservation.get("members") or []) if str(m).strip()]
        updated = parse_ts(reservation.get("updated_at")) or parse_ts(reservation.get("created_at"))
        age_s = int((datetime.now() - updated).total_seconds()) if updated else None
        info = {
            "group_id": str(reservation.get("group_id", "")).strip(),
            "status": status,
            "launcher": launcher,
            "members": members,
            "target_model": str(reservation.get("target_model", "")).strip(),
            "age_s": age_s,
        }
        participants = set(members)
        if launcher:
            participants.add(launcher)
        for worker in participants:
            active[worker] = info
    return active


def escalate(shared: Path, kind: str, title: str, details: dict[str, Any]) -> Path:
    esc_dir = shared / "brain" / "escalations"
    esc_dir.mkdir(parents=True, exist_ok=True)
    eid = f"gpu_watchdog_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    payload = {
        "escalation_id": eid,
        "created_at": now_iso(),
        "status": "pending",
        "target": "cloud_brain",
        "source": "gpu_worker_watchdog",
        "type": kind,
        "title": title,
        "details": details,
        "recommended_action": "Investigate GPU worker flapping and startup ownership",
    }
    path = esc_dir / f"{eid}.json"
    save_json(path, payload)
    return path


def append_log(log_path: Path, line: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{now_iso()} {line}\n")


def classify_thermal_cause(text: str) -> str:
    upper = text.upper()
    has_cpu = bool(re.search(r"\bCPU\b", upper))
    has_gpu = bool(re.search(r"\bGPU\b", upper))
    if has_cpu and has_gpu:
        return "mixed"
    if has_cpu:
        return "cpu"
    if has_gpu:
        return "gpu"
    return "unknown"


def parse_worker_gid(worker_name: str) -> int | None:
    try:
        return int(worker_name.split("-")[-1])
    except Exception:
        return None


def latest_thermal_signal_for_worker(shared: Path, worker_name: str) -> dict[str, Any]:
    gid = parse_worker_gid(worker_name)
    if gid is None:
        return {
            "worker": worker_name,
            "cause": "unknown",
            "source": "none",
            "event": None,
            "detail": "invalid worker name",
        }

    hb_path = shared / "gpus" / f"gpu_{gid}" / "heartbeat.json"
    hb = load_json(hb_path, {}) if hb_path.exists() else {}
    last_event = hb.get("last_thermal_event") if isinstance(hb.get("last_thermal_event"), dict) else {}
    if last_event:
        reasons = last_event.get("reasons")
        if isinstance(reasons, list):
            reason_text = " ".join(str(x) for x in reasons)
        else:
            reason_text = str(reasons or "")
        cause = classify_thermal_cause(reason_text)
        if cause != "unknown":
            return {
                "worker": worker_name,
                "cause": cause,
                "source": "heartbeat",
                "event": last_event.get("type"),
                "detail": reason_text[:300],
            }

    log_path = shared / "logs" / f"{worker_name}-manual.log"
    if log_path.exists():
        try:
            lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            lines = []
        for line in reversed(lines[-300:]):
            if "THERMAL SAFETY SHUTDOWN:" in line:
                tail = line.split("THERMAL SAFETY SHUTDOWN:", 1)[-1].strip()
                return {
                    "worker": worker_name,
                    "cause": classify_thermal_cause(tail),
                    "source": "worker_log",
                    "event": "critical_shutdown",
                    "detail": tail[:300],
                }
            if "THERMAL_PAUSE_ENTER:" in line:
                tail = line.split("THERMAL_PAUSE_ENTER:", 1)[-1].strip()
                return {
                    "worker": worker_name,
                    "cause": classify_thermal_cause(tail),
                    "source": "worker_log",
                    "event": "pause_enter",
                    "detail": tail[:300],
                }

    return {
        "worker": worker_name,
        "cause": "unknown",
        "source": "none",
        "event": None,
        "detail": "no thermal signal found",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared-path", default="/media/bryan/shared")
    ap.add_argument("--stale-seconds", type=int, default=45)
    ap.add_argument("--flap-window-seconds", type=int, default=180)
    ap.add_argument("--min-llm-queue", type=int, default=1)
    ap.add_argument("--block-restart-cpu-c", type=int, default=90)
    ap.add_argument("--cpu-cause-ratio-block", type=float, default=0.5)
    ap.add_argument("--split-meta-grace-seconds", type=int, default=420)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    shared = Path(args.shared_path)
    cfg = load_json(shared / "agents" / "config.json", {})
    if not cfg:
        raise SystemExit("missing or invalid config.json")

    log_path = shared / "logs" / "gpu_worker_watchdog.log"
    state_path = shared / "brain" / "gpu_watchdog_state.json"
    state = load_json(state_path, {"workers": {}})

    workers = expected_workers(cfg)
    if not workers:
        append_log(log_path, "no expected workers resolved; skipping")
        return 0

    q_llm = llm_queue_count(shared)
    stale = []
    for w in workers:
        age = worker_age_s(shared, w)
        if age is None or age >= args.stale_seconds:
            stale.append({"name": w, "age_s": age})

    if not stale:
        append_log(log_path, f"ok workers={workers} llm_queue={q_llm}")
        return 0

    append_log(log_path, f"stale workers={stale} llm_queue={q_llm}")
    if q_llm < args.min_llm_queue:
        append_log(log_path, f"skip restart: llm_queue {q_llm} < {args.min_llm_queue}")
        return 0

    stale_names = [x["name"] for x in stale]
    split_inflight = active_split_meta_by_worker(shared)
    split_reservations = active_split_reservations_by_worker(shared)
    restart_candidates = []
    split_deferred = []
    for w in stale_names:
        split_meta = split_inflight.get(w)
        split_res = split_reservations.get(w)
        split_info = split_meta or split_res
        if not split_info:
            restart_candidates.append(w)
            continue
        split_age = split_info.get("age_s")
        if split_age is None or split_age <= args.split_meta_grace_seconds:
            split_deferred.append({"worker": w, "split": split_info})
            continue
        restart_candidates.append(w)
    if split_deferred:
        append_log(log_path, f"defer restart for split load workers={split_deferred}")
    if not restart_candidates:
        append_log(log_path, "skip restart: all stale workers are in active split load windows")
        return 0

    stale_names = restart_candidates
    stale = [x for x in stale if x.get("name") in set(stale_names)]
    thermal_signals = [latest_thermal_signal_for_worker(shared, name) for name in stale_names]
    cause_counts: dict[str, int] = {}
    for sig in thermal_signals:
        cause = str(sig.get("cause") or "unknown")
        cause_counts[cause] = cause_counts.get(cause, 0) + 1
    append_log(
        log_path,
        f"thermal classification stale_workers={stale_names} causes={cause_counts} "
        f"signals={json.dumps(thermal_signals)[:1200]}",
    )
    thermal_log = shared / "logs" / "thermal_events.log"
    for sig in thermal_signals:
        append_log(
            thermal_log,
            (
                f"THERMAL_EVENT worker={sig.get('worker')} cause={sig.get('cause')} "
                f"event={sig.get('event')} source={sig.get('source')} detail={sig.get('detail')}"
            ),
        )

    cpu_c = read_gpu_rig_cpu_temp_c(shared)
    if cpu_c is not None and cpu_c >= args.block_restart_cpu_c:
        esc = escalate(
            shared,
            "gpu_restart_blocked_high_cpu_temp",
            "Watchdog blocked GPU restart due to high CPU temp",
            {
                "cpu_temp_c": cpu_c,
                "threshold_c": args.block_restart_cpu_c,
                "stale_workers": stale,
                "thermal_causes": cause_counts,
                "thermal_signals": thermal_signals,
                "llm_queue": q_llm,
            },
        )
        append_log(log_path, f"blocked restart: cpu={cpu_c}C threshold={args.block_restart_cpu_c}C escalated={esc}")
        return 1

    cpu_like_count = int(cause_counts.get("cpu", 0)) + int(cause_counts.get("mixed", 0))
    cpu_like_ratio = (cpu_like_count / len(stale_names)) if stale_names else 0.0
    if cpu_like_ratio >= args.cpu_cause_ratio_block:
        cpu_threshold = int(cfg.get("resource_limits", {}).get("cpu_temp_warning_c", args.block_restart_cpu_c))
        if cpu_c is None:
            append_log(
                log_path,
                (
                    f"skip restart: cpu/mixed thermal dominant ratio={cpu_like_ratio:.2f} "
                    f"but cpu temp unavailable"
                ),
            )
            return 0
        if cpu_c >= cpu_threshold:
            append_log(
                log_path,
                (
                    f"skip restart: cpu/mixed thermal dominant ratio={cpu_like_ratio:.2f} "
                    f"cpu={cpu_c}C threshold={cpu_threshold}C"
                ),
            )
            return 0

    signal_by_worker = {
        str(sig.get("worker")): sig for sig in thermal_signals if isinstance(sig, dict)
    }

    # Escalate immediately if worker recently restarted and disappeared again.
    # Suppress flap escalation when root cause is clearly CPU thermal, since this
    # is host cooling pressure rather than a GPU worker fault.
    for w in stale_names:
        sig = signal_by_worker.get(w, {})
        cause = str(sig.get("cause") or "unknown")
        if cause == "cpu":
            append_log(
                log_path,
                f"suppress flap escalation: worker={w} cause=cpu event={sig.get('event')} source={sig.get('source')}",
            )
            continue
        last_restart = parse_ts(state.get("workers", {}).get(w, {}).get("last_restart_at"))
        if last_restart is None:
            continue
        delta = int((datetime.now() - last_restart).total_seconds())
        if delta <= args.flap_window_seconds:
            esc = escalate(
                shared,
                "gpu_worker_flap",
                f"GPU worker flapping after restart: {w}",
                {
                    "worker": w,
                    "seconds_since_restart": delta,
                    "stale_workers": stale,
                    "thermal_causes": cause_counts,
                    "thermal_signals": thermal_signals,
                    "llm_queue": q_llm,
                    "flap_window_seconds": args.flap_window_seconds,
                },
            )
            append_log(log_path, f"escalated flap: {w} file={esc}")

    restart_script_local = Path("/mnt/shared/scripts/restart_gpu_agents.py")
    if restart_script_local.exists() and str(shared).startswith("/mnt/shared"):
        # Running on GPU rig directly.
        cmd = f"python3 /mnt/shared/scripts/restart_gpu_agents.py --gpus {','.join(stale_names)}"
    else:
        # Running from control host; restart remotely on gpu host.
        cmd = f"ssh -o BatchMode=yes gpu \"python3 /mnt/shared/scripts/restart_gpu_agents.py --gpus {','.join(stale_names)}\""
    if args.dry_run:
        append_log(log_path, f"dry-run restart cmd={cmd}")
        return 0

    rc, out, err = run(cmd, timeout=180)
    append_log(log_path, f"restart rc={rc} workers={stale_names}")
    if out.strip():
        append_log(log_path, f"restart stdout={out.strip()[:1200]}")
    if err.strip():
        append_log(log_path, f"restart stderr={err.strip()[:1200]}")

    if rc != 0:
        esc = escalate(
            shared,
            "gpu_restart_failed",
            "GPU worker auto-restart failed",
            {
                "workers": stale_names,
                "thermal_causes": cause_counts,
                "thermal_signals": thermal_signals,
                "llm_queue": q_llm,
                "returncode": rc,
                "stderr": err[-1500:],
            },
        )
        append_log(log_path, f"escalated restart failure file={esc}")
        return 1

    st_workers = state.setdefault("workers", {})
    ts = now_iso()
    for w in stale_names:
        info = st_workers.setdefault(w, {})
        info["last_restart_at"] = ts
        info["restart_count"] = int(info.get("restart_count", 0)) + 1
    save_json(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
