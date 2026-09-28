#!/usr/bin/env python3
"""Runtime preflight for mode transitions and manual load operations."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prepare_llm_runtimes import (
    RUNTIME_META_COMMANDS,
    cleanup_stale_runtime_meta_tasks,
    cleanup_test_runtime_containers,
    clear_orphan_queue_locks,
    collect_stale_runtime_meta_tasks,
    list_test_runtime_containers,
    load_config,
    scan_worker_ports,
)


DEFAULT_MODELS_CATALOG = "/mnt/shared/plans/shoulders/benchmarking/models.catalog.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_split_ports(catalog: dict) -> list[int]:
    ports: set[int] = set()
    for model in catalog.get("models", []) or []:
        if not isinstance(model, dict):
            continue
        for group in model.get("split_groups", []) or []:
            if not isinstance(group, dict):
                continue
            port = group.get("port")
            try:
                if port is not None:
                    ports.add(int(port))
            except Exception:
                continue
    return sorted(ports)


def _list_processing_tasks(shared_root: Path) -> tuple[list[dict], list[dict]]:
    runtime_meta: list[dict] = []
    non_runtime: list[dict] = []
    processing_dir = shared_root / "tasks" / "processing"
    for task_path in sorted(processing_dir.glob("*.json")):
        if task_path.name.endswith(".heartbeat.json"):
            continue
        try:
            task = _load_json(task_path)
        except Exception:
            continue
        command = str(task.get("command") or "").strip()
        entry = {
            "task_id": str(task.get("task_id") or task_path.stem),
            "command": command,
            "batch_id": str(task.get("batch_id") or ""),
            "assigned_to": str(task.get("assigned_to") or ""),
        }
        if command in RUNTIME_META_COMMANDS:
            runtime_meta.append(entry)
        else:
            non_runtime.append(entry)
    return runtime_meta, non_runtime


def _active_listeners_for_ports(ports: list[int]) -> dict[str, list[str]]:
    listeners: dict[str, list[str]] = {}
    if not ports:
        return listeners
    proc = subprocess.run(
        ["ss", "-ltnp"],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return listeners
    lines = (proc.stdout or "").splitlines()
    for port in ports:
        marker = f":{port}"
        matches = [line.strip() for line in lines if marker in line]
        if matches:
            listeners[str(port)] = matches
    return listeners


def _list_llama_containers() -> list[dict]:
    proc = subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}|{{.Status}}"],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    rows: list[dict] = []
    for line in (proc.stdout or "").splitlines():
        name, _, status = line.partition("|")
        name = name.strip()
        status = status.strip()
        if not name:
            continue
        if name.startswith("llama-") or name.startswith("test-"):
            rows.append({"name": name, "status": status})
    return rows


def _load_worker_heartbeats(shared_root: Path) -> dict[str, dict]:
    heartbeats: dict[str, dict] = {}
    gpus_dir = shared_root / "gpus"
    if not gpus_dir.exists():
        return heartbeats
    for hb_path in sorted(gpus_dir.glob("gpu_*/heartbeat.json")):
        try:
            data = _load_json(hb_path)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        name = str(data.get("name") or "").strip()
        if not name:
            try:
                gpu_id = int(hb_path.parent.name.replace("gpu_", ""))
                name = f"gpu-{gpu_id}"
            except Exception:
                continue
        heartbeats[name] = data
    return heartbeats


def _listener_conflicts(
    worker_scan: list[tuple[str, int, str | None]],
    split_ports: list[int],
    listeners: dict[str, list[str]],
    containers: list[dict],
    heartbeats: dict[str, dict],
) -> list[dict]:
    conflicts: list[dict] = []
    running_names = {row["name"] for row in containers if "Up " in row.get("status", "") or row.get("status", "").startswith("Up")}
    split_port_set = {int(p) for p in split_ports}
    for name, port, loaded in worker_scan:
        if not port:
            continue
        port_key = str(port)
        if port_key not in listeners:
            continue
        expected_name = f"llama-worker-{name}"
        hb = heartbeats.get(name) or {}
        runtime_state = str(hb.get("runtime_state") or "").strip()
        runtime_transition_task_id = str(hb.get("runtime_transition_task_id") or "").strip()
        meta_task_active = bool(hb.get("meta_task_active"))
        managed_transition = bool(runtime_transition_task_id) or meta_task_active or runtime_state.startswith("loading_") or runtime_state.startswith("unloading_")
        if loaded:
            if expected_name not in running_names:
                conflicts.append(
                    {
                        "port": port,
                        "type": "listener_without_managed_container",
                        "worker": name,
                        "loaded_model": loaded,
                    }
                )
        elif not managed_transition:
            conflicts.append(
                {
                    "port": port,
                    "type": "cold_worker_listener_present",
                    "worker": name,
                    "loaded_model": "",
                    "runtime_state": runtime_state,
                }
            )
    for port in split_port_set:
        port_key = str(port)
        if port_key in listeners:
            conflicts.append(
                {
                    "port": port,
                    "type": "split_listener_present",
                }
            )
    return conflicts


def _active_split_reservation_files(shared_root: Path) -> list[str]:
    split_dir = shared_root / "signals" / "split_llm"
    if not split_dir.exists():
        return []
    return sorted(p.name for p in split_dir.glob("pair_*.json"))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Runtime preflight for rig mode transitions")
    ap.add_argument("--shared-root", default="/mnt/shared")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--catalog", default=DEFAULT_MODELS_CATALOG)
    ap.add_argument("--runtime-meta-stale-seconds", type=int, default=180)
    ap.add_argument("--clear-orphan-queue-locks", action="store_true")
    ap.add_argument("--cleanup-stale-runtime-meta", action="store_true")
    ap.add_argument("--cleanup-test-runtime-containers", action="store_true")
    ap.add_argument("--fail-on-active-processing", action="store_true")
    ap.add_argument("--json", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    shared_root = Path(args.shared_root).expanduser().resolve()
    config = load_config(shared_root, args.config)
    catalog = _load_json(Path(args.catalog))

    queue_dir = shared_root / "tasks" / "queue"
    orphan_locks_removed = 0
    if args.clear_orphan_queue_locks:
        orphan_locks_removed = clear_orphan_queue_locks(queue_dir)

    stale_runtime_meta = collect_stale_runtime_meta_tasks(
        shared_root,
        stale_seconds=max(30, int(args.runtime_meta_stale_seconds)),
    )
    cleaned_stale_runtime_meta = 0
    if stale_runtime_meta and args.cleanup_stale_runtime_meta:
        cleaned_stale_runtime_meta = cleanup_stale_runtime_meta_tasks(
            shared_root,
            stale_seconds=max(30, int(args.runtime_meta_stale_seconds)),
        )
        stale_runtime_meta = collect_stale_runtime_meta_tasks(
            shared_root,
            stale_seconds=max(30, int(args.runtime_meta_stale_seconds)),
        )

    test_containers = list_test_runtime_containers()
    removed_test_containers = 0
    if test_containers and args.cleanup_test_runtime_containers:
        removed_test_containers = cleanup_test_runtime_containers()
        test_containers = list_test_runtime_containers()

    runtime_meta_processing, non_runtime_processing = _list_processing_tasks(shared_root)
    worker_scan = scan_worker_ports(config)
    heartbeats = _load_worker_heartbeats(shared_root)
    split_ports = _collect_split_ports(catalog)
    all_ports = sorted({int(port) for _, port, _ in worker_scan if port} | set(split_ports))
    listeners = _active_listeners_for_ports(all_ports)
    containers = _list_llama_containers()
    conflicts = _listener_conflicts(worker_scan, split_ports, listeners, containers, heartbeats)
    split_reservation_files = _active_split_reservation_files(shared_root)

    result = {
        "ok": True,
        "shared_root": str(shared_root),
        "config": args.config,
        "orphan_queue_locks_removed": orphan_locks_removed,
        "stale_runtime_meta_count": len(stale_runtime_meta),
        "cleaned_stale_runtime_meta": cleaned_stale_runtime_meta,
        "test_runtime_container_count": len(test_containers),
        "removed_test_runtime_containers": removed_test_containers,
        "runtime_meta_processing_count": len(runtime_meta_processing),
        "non_runtime_processing_count": len(non_runtime_processing),
        "worker_scan": [
            {"name": name, "port": port, "loaded_model": loaded or ""}
            for name, port, loaded in worker_scan
        ],
        "worker_runtime_states": {
            name: {
                "runtime_state": str((heartbeats.get(name) or {}).get("runtime_state") or ""),
                "meta_task_active": bool((heartbeats.get(name) or {}).get("meta_task_active")),
                "runtime_transition_task_id": str((heartbeats.get(name) or {}).get("runtime_transition_task_id") or ""),
            }
            for name, _port, _loaded in worker_scan
        },
        "active_listener_ports": sorted(listeners.keys(), key=int),
        "llama_containers": containers,
        "listener_conflict_count": len(conflicts),
        "split_reservation_files": split_reservation_files,
    }
    if stale_runtime_meta:
        result["stale_runtime_meta_tasks"] = [
            {
                "task_id": str(entry["task"].get("task_id") or entry["task_path"].stem),
                "command": str(entry["task"].get("command") or ""),
                "age_s": int(entry["heartbeat_age_seconds"]),
            }
            for entry in stale_runtime_meta[:10]
        ]
    if test_containers:
        result["test_runtime_containers"] = test_containers
    if non_runtime_processing:
        result["non_runtime_processing"] = non_runtime_processing[:10]
    if listeners:
        result["listener_details"] = listeners
    if conflicts:
        result["listener_conflicts"] = conflicts

    if args.fail_on_active_processing and non_runtime_processing:
        result["ok"] = False
        result["error"] = "active_non_runtime_processing_tasks"
    if conflicts:
        result["ok"] = False
        result["error"] = "listener_conflicts_detected"

    if args.json:
        print(json.dumps(result))
    else:
        print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
