#!/usr/bin/env python3
"""
Check status of the agent system and tasks.

Usage:
  python status.py           # Overview
  python status.py --tasks   # List all tasks
  python status.py --task ID # Show specific task
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def resolve_shared_path(config_path: str, config: dict) -> Path:
    """Resolve shared path relative to config location when needed."""
    shared = Path(config.get("shared_path", "../"))
    if shared.is_absolute():
        return shared
    return (Path(config_path).resolve().parent / shared).resolve()


def iter_task_files(folder_path: Path):
    """Yield real task JSON files (skip lock/heartbeat sidecars)."""
    for task_file in folder_path.glob("*.json"):
        if task_file.name.endswith(".heartbeat.json"):
            continue
        yield task_file


def get_tasks(shared_path: Path) -> dict:
    """Get all tasks organized by status."""
    tasks = {"queued": [], "processing": [], "complete": [], "failed": []}

    for status, folder in [
        ("queued", "queue"),
        ("processing", "processing"),
        ("complete", "complete"),
        ("failed", "failed")
    ]:
        folder_path = shared_path / "tasks" / folder
        for task_file in iter_task_files(folder_path):
            try:
                with open(task_file) as f:
                    task = json.load(f)
                    tasks[status].append(task)
            except Exception:
                pass  # Skip malformed task file

    return tasks


def get_workers(config: dict) -> List[dict]:
    """Support both legacy workers[] and current gpus[] schema."""
    if "workers" in config and isinstance(config["workers"], list):
        return config["workers"]
    workers = []
    for gpu in config.get("gpus", []):
        workers.append({
            "name": gpu.get("name", f"gpu-{gpu.get('id', '?')}"),
            "model": gpu.get("model", "?"),
            "gpu": gpu.get("id", "?"),
            "port": gpu.get("port", "?"),
        })
    return workers


def _heartbeat_age_seconds(last_updated: str) -> int | None:
    if not last_updated:
        return None
    try:
        return int((datetime.now() - datetime.fromisoformat(last_updated)).total_seconds())
    except Exception:
        return None


def get_worker_heartbeats(shared_path: Path) -> Dict[str, dict]:
    """Load GPU + CPU worker heartbeats."""
    heartbeats: Dict[str, dict] = {}

    # GPU heartbeats
    gpus_dir = shared_path / "gpus"
    if gpus_dir.exists():
        for hb_file in sorted(gpus_dir.glob("gpu_*/heartbeat.json")):
            try:
                hb = json.loads(hb_file.read_text())
                name = hb.get("name") or hb_file.parent.name
                hb["worker_type"] = "gpu"
                hb["age_s"] = _heartbeat_age_seconds(hb.get("last_updated", ""))
                heartbeats[name] = hb
            except Exception:
                continue

    # CPU heartbeats
    cpus_dir = shared_path / "cpus"
    if cpus_dir.exists():
        for hb_file in sorted(cpus_dir.glob("*/heartbeat.json")):
            try:
                hb = json.loads(hb_file.read_text())
                name = hb.get("name") or hb_file.parent.name
                hb["worker_type"] = "cpu"
                hb["age_s"] = _heartbeat_age_seconds(hb.get("last_updated", ""))
                heartbeats[name] = hb
            except Exception:
                continue

    return heartbeats


def get_active_batches(shared_path: Path) -> Dict[str, dict]:
    """Load active batch metadata from brain state."""
    state_path = shared_path / "brain" / "state.json"
    if not state_path.exists():
        return {}
    try:
        with open(state_path) as f:
            state = json.load(f)
        active = state.get("active_batches", {})
        if isinstance(active, dict):
            return active
    except Exception:
        pass
    return {}


def batch_counts(tasks: dict, batch_id: str) -> Dict[str, int]:
    counts = {"queued": 0, "processing": 0, "complete": 0, "failed": 0}
    for status in counts:
        counts[status] = sum(1 for t in tasks[status] if t.get("batch_id") == batch_id)
    return counts


def show_overview(config_path: str):
    """Show system overview."""
    config = load_config(config_path)
    shared_path = resolve_shared_path(config_path, config)
    tasks = get_tasks(shared_path)
    active_batches = get_active_batches(shared_path)
    heartbeats = get_worker_heartbeats(shared_path)

    print("="*60)
    print("AGENT SYSTEM STATUS")
    print("="*60)

    print(f"\nConfiguration: {config_path}")
    print(f"Shared path: {shared_path}")

    print(f"\nBrain: {config['brain']['model']} (GPUs {config['brain']['gpus']})")
    print("Workers:")
    for w in get_workers(config):
        print(f"  {w['name']}: {w['model']} (GPU {w['gpu']}, port {w['port']})")

    cpu_workers = [hb for hb in heartbeats.values() if hb.get("worker_type") == "cpu"]
    if cpu_workers:
        print("CPU Workers:")
        for hb in sorted(cpu_workers, key=lambda x: x.get("name", "")):
            state = hb.get("state", "?")
            age = hb.get("age_s")
            age_s = f"{age}s" if isinstance(age, int) else "?"
            host = hb.get("hostname", "?")
            print(f"  {hb.get('name', '?')}: state={state} host={host} hb_age={age_s}")

    print(f"\nGlobal Tasks (all batches):")
    print(f"  Queued:     {len(tasks['queued'])}")
    print(f"  Processing: {len(tasks['processing'])}")
    print(f"  Complete:   {len(tasks['complete'])}")
    print(f"  Failed:     {len(tasks['failed'])}")

    if heartbeats:
        total = len(heartbeats)
        stale = sum(1 for hb in heartbeats.values() if isinstance(hb.get("age_s"), int) and hb["age_s"] > 120)
        hot = sum(1 for hb in heartbeats.values() if hb.get("state") == "hot")
        running = sum(1 for hb in heartbeats.values() if hb.get("state") in ("hot", "cold", "running", "idle"))
        print("\nWorker Heartbeats:")
        print(f"  Total:      {total}")
        print(f"  Active-ish: {running}")
        print(f"  Hot:        {hot}")
        print(f"  Stale>120s: {stale}")

    if active_batches:
        print("\nActive Batch Progress:")
        for batch_id, meta in sorted(active_batches.items()):
            counts = batch_counts(tasks, batch_id)
            observed_total = sum(counts.values())
            total_hint = meta.get("total_tasks")
            if isinstance(total_hint, int) and total_hint > 0:
                total_tasks = max(total_hint, observed_total)
            else:
                total_tasks = observed_total
            complete = counts["complete"]
            print(
                f"  {batch_id}: {complete}/{total_tasks} complete "
                f"(queued={counts['queued']}, processing={counts['processing']}, failed={counts['failed']})"
            )

    if tasks["processing"]:
        print("\nActive tasks:")
        for t in tasks["processing"]:
            batch = t.get("batch_id", "-")
            print(
                f"  {t['task_id'][:8]}... ({t.get('type', '?')}) "
                f"[batch={batch}] -> {t.get('assigned_to', 'unassigned')}"
            )


def show_tasks(config_path: str):
    """List all tasks."""
    config = load_config(config_path)
    shared_path = resolve_shared_path(config_path, config)
    tasks = get_tasks(shared_path)

    for status in ["queued", "processing", "complete", "failed"]:
        if tasks[status]:
            print(f"\n{'='*60}")
            print(f"{status.upper()} ({len(tasks[status])})")
            print("="*60)
            for t in tasks[status]:
                created = t.get("created_at", "")[:19]
                assigned = t.get("assigned_to", "-")
                prompt = t.get("prompt", "")[:50]
                print(f"{t['task_id'][:8]} | {t.get('type', '?'):10} | {assigned:10} | {created}")
                print(f"         | {prompt}...")


def show_task(config_path: str, task_id: str):
    """Show details of a specific task."""
    config = load_config(config_path)
    shared_path = resolve_shared_path(config_path, config)

    for folder in ["queue", "processing", "complete", "failed"]:
        for task_file in iter_task_files(shared_path / "tasks" / folder):
            if task_id in task_file.name:
                with open(task_file) as f:
                    task = json.load(f)
                print(json.dumps(task, indent=2))
                return

    print(f"Task {task_id} not found")


def main():
    parser = argparse.ArgumentParser(description="Agent system status")
    default_config = str(Path(__file__).resolve().parent.parent / "shared" / "agents" / "config.json")
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--tasks", action="store_true", help="List all tasks")
    parser.add_argument("--task", metavar="ID", help="Show specific task")
    args = parser.parse_args()

    if args.task:
        show_task(args.config, args.task)
    elif args.tasks:
        show_tasks(args.config)
    else:
        show_overview(args.config)


if __name__ == "__main__":
    main()
