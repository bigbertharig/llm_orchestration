#!/usr/bin/env python3
"""Kill a running plan and clean up system state.

Usage:
    python scripts/kill_plan.py [batch_id]

If no batch_id provided, kills all active batches.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
STATE_FILE = BASE_DIR / "shared/brain/state.json"
TASKS_DIR = BASE_DIR / "shared/tasks"
BRAIN_PRIVATE_DIR = BASE_DIR / "shared/brain/private_tasks"
CONFIG_FILE = BASE_DIR / "shared/agents/config.json"


def kill_transcription_processes():
    """Kill any running transcription processes."""
    print("Killing transcription processes...")
    subprocess.run(["pkill", "-9", "-f", "video_transcribe.py"],
                   capture_output=True)
    time.sleep(1)


def kill_workers():
    """Kill worker processes."""
    print("Killing workers...")
    subprocess.run(["pkill", "-f", "worker.py"], capture_output=True)
    time.sleep(2)


def _safe_unlink(path: Path) -> bool:
    """Remove a file if present; ignore missing files from concurrent cleanup."""
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


def _task_matches_batch(task_path: Path, batch_id: str | None) -> bool:
    """Return True if task should be removed for this batch scope."""
    if batch_id is None:
        return True
    try:
        with open(task_path) as f:
            task = json.load(f)
        return task.get("batch_id") == batch_id
    except Exception:
        return False


def _remove_related_artifacts(task_file: Path):
    """Remove heartbeat/lock files corresponding to a task file."""
    base = task_file.with_suffix("")
    heartbeat = base.with_name(f"{base.name}.heartbeat").with_suffix(".json")
    lock_file = task_file.with_name(f"{task_file.name}.lock")
    if _safe_unlink(heartbeat):
        print(f"  Removed heartbeat: {heartbeat.name}")
    if _safe_unlink(lock_file):
        print(f"  Removed lock: {lock_file.name}")


def clear_task_queue(batch_id: str = None):
    """Clear queued/processing/failed tasks for a specific batch or all batches."""
    scope = f"batch {batch_id}" if batch_id else "all batches"
    print(f"Clearing task queues ({scope})...")
    for subdir in ["queue", "processing", "failed"]:
        queue_dir = TASKS_DIR / subdir
        if not queue_dir.exists():
            continue
        for f in queue_dir.glob("*.json"):
            if not _task_matches_batch(f, batch_id):
                continue
            if _safe_unlink(f):
                print(f"  Removed: {f.name}")
            _remove_related_artifacts(f)

        # Remove orphan lock files (lock without matching task json)
        for lock in queue_dir.glob("*.lock"):
            task_name = lock.name[:-5]  # strip ".lock"
            task_path = queue_dir / task_name
            if not task_path.exists() and _safe_unlink(lock):
                print(f"  Removed orphan lock: {lock.name}")


def clear_private_tasks(batch_id: str = None):
    """Clear brain private tasks.

    If batch_id is provided, only remove private tasks for that batch.
    Otherwise remove all private task json/lock files for a clean slate.
    """
    print("Clearing brain private tasks...")
    if not BRAIN_PRIVATE_DIR.exists():
        print("  No private_tasks directory found")
        return

    removed_json = 0
    removed_locks = 0

    for f in BRAIN_PRIVATE_DIR.glob("*.json"):
        try:
            remove_file = False
            if batch_id:
                with open(f) as fh:
                    task = json.load(fh)
                remove_file = task.get("batch_id") == batch_id
            else:
                remove_file = True

            if remove_file:
                f.unlink()
                removed_json += 1
                print(f"  Removed private task: {f.name}")
        except Exception as e:
            print(f"  Warning: Failed to inspect/remove {f.name}: {e}")

    for f in BRAIN_PRIVATE_DIR.glob("*.lock"):
        try:
            f.unlink()
            removed_locks += 1
            print(f"  Removed private lock: {f.name}")
        except Exception as e:
            print(f"  Warning: Failed to remove lock {f.name}: {e}")

    print(f"  Private tasks removed: {removed_json}, locks removed: {removed_locks}")


def clear_active_batch(batch_id: str = None):
    """Remove batch from brain state."""
    if not STATE_FILE.exists():
        print("No state file found")
        return

    with open(STATE_FILE) as f:
        state = json.load(f)

    if batch_id:
        if batch_id in state.get("active_batches", {}):
            del state["active_batches"][batch_id]
            print(f"Removed batch: {batch_id}")
        else:
            print(f"Batch {batch_id} not found in active batches")
    else:
        count = len(state.get("active_batches", {}))
        state["active_batches"] = {}
        print(f"Cleared {count} active batch(es)")

    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def unload_worker_models(ports: list = None):
    """Unload models from worker Ollama instances (not the brain)."""
    print("Unloading worker models...")

    if ports is None:
        # Fallback: try to read from config
        config_path = BASE_DIR / "shared" / "agents" / "config.json"
        try:
            with open(config_path) as f:
                config = json.load(f)
            ports = [gpu["port"] for gpu in config["gpus"] if gpu.get("port")]
        except Exception as e:
            print(f"  Warning: Could not read config ({e}), using no ports")
            return

    for port in ports:
        try:
            result = subprocess.run(
                ["curl", "-s", f"http://localhost:{port}/api/ps"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                for model in data.get("models", []):
                    model_name = model.get("name", "")
                    print(f"  Unloading {model_name} from port {port}...")
                    subprocess.run(
                        ["curl", "-s", f"http://localhost:{port}/api/generate",
                         "-d", json.dumps({"model": model_name, "keep_alive": 0})],
                        capture_output=True, timeout=10
                    )
        except Exception:
            pass  # Port not responding, skip


def load_config() -> dict:
    """Load orchestration config, returning empty dict on failure."""
    try:
        with open(CONFIG_FILE) as f:
            return json.load(f)
    except Exception as e:
        print(f"  Warning: Could not read config file {CONFIG_FILE}: {e}")
        return {}


def queue_default_warm_workers(count: int):
    """Queue startup-equivalent load_llm tasks for default resting state."""
    if count <= 0:
        return

    queue_dir = TASKS_DIR / "queue"
    queue_dir.mkdir(parents=True, exist_ok=True)

    print(f"Queueing default warm worker tasks: {count}")
    for idx in range(count):
        task = {
            "task_id": str(uuid.uuid4()),
            "type": "meta",
            "command": "load_llm",
            "batch_id": "system",
            "name": f"load_llm_default_{idx + 1}",
            "priority": 10,
            "task_class": "meta",
            "depends_on": [],
            "executor": "worker",
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "created_by": "kill_plan",
            "retry_count": 0
        }
        task_file = queue_dir / f"{task['task_id']}.json"
        with open(task_file, "w") as f:
            json.dump(task, f, indent=2)
        print(f"  Queued: {task['name']} ({task['task_id'][:8]})")


def main():
    parser = argparse.ArgumentParser(description="Kill a running plan")
    parser.add_argument("batch_id", nargs="?", help="Specific batch ID to kill (optional)")
    parser.add_argument("--keep-workers", action="store_true", help="Don't kill workers")
    parser.add_argument("--keep-models", action="store_true", help="Don't unload models")
    parser.add_argument("--no-default-warm", action="store_true",
                        help="Don't queue default load_llm warmup tasks after cleanup")
    args = parser.parse_args()
    config = load_config()

    print("=" * 50)
    print("Killing plan...")
    print("=" * 50)

    # Step 1: Kill any running script processes
    kill_transcription_processes()

    # Step 2: Clear task queues (before killing workers so they don't pick up more)
    clear_task_queue(args.batch_id)

    # Step 2.5: Clear private tasks so brain has a clean slate too
    clear_private_tasks(args.batch_id)

    # Step 3: Kill workers (optional)
    if not args.keep_workers:
        kill_workers()

    # Step 4: Clear batch from state
    clear_active_batch(args.batch_id)

    # Step 5: Unload worker models (brain stays loaded)
    if not args.keep_models:
        unload_worker_models()

    # Step 6: Restore default resting state (brain + configured warm workers) unless disabled
    if not args.no_default_warm:
        warm_count = int(config.get("initial_hot_workers", 3))
        queue_default_warm_workers(max(0, warm_count))

    print("=" * 50)
    print("Done. System should return to idle state.")
    print("Restart workers with: python shared/agents/launch.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
