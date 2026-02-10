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
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
STATE_FILE = BASE_DIR / "shared/brain/state.json"
TASKS_DIR = BASE_DIR / "shared/tasks"


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


def clear_task_queue():
    """Clear pending and processing tasks, and stale lock files."""
    print("Clearing task queues...")
    for subdir in ["queue", "processing", "failed"]:
        queue_dir = TASKS_DIR / subdir
        if queue_dir.exists():
            # Remove task files
            for f in queue_dir.glob("*.json"):
                f.unlink()
                print(f"  Removed: {f.name}")
            # Remove stale lock files
            for f in queue_dir.glob("*.lock"):
                f.unlink()
                print(f"  Removed lock: {f.name}")


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


def main():
    parser = argparse.ArgumentParser(description="Kill a running plan")
    parser.add_argument("batch_id", nargs="?", help="Specific batch ID to kill (optional)")
    parser.add_argument("--keep-workers", action="store_true", help="Don't kill workers")
    parser.add_argument("--keep-models", action="store_true", help="Don't unload models")
    args = parser.parse_args()

    print("=" * 50)
    print("Killing plan...")
    print("=" * 50)

    # Step 1: Kill any running script processes
    kill_transcription_processes()

    # Step 2: Clear task queues (before killing workers so they don't pick up more)
    clear_task_queue()

    # Step 3: Kill workers (optional)
    if not args.keep_workers:
        kill_workers()

    # Step 4: Clear batch from state
    clear_active_batch(args.batch_id)

    # Step 5: Unload worker models (brain stays loaded)
    if not args.keep_models:
        unload_worker_models()

    print("=" * 50)
    print("Done. System should return to idle state.")
    print("Restart workers with: python shared/agents/launch.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
