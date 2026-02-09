#!/usr/bin/env python3
"""
Launch the agent system (brain + workers).

Usage:
  python launch.py              # Start brain + all workers
  python launch.py --workers-only   # Start only workers (brain running elsewhere)
  python launch.py --brain-only     # Start only brain
"""

import argparse
import json
import subprocess
import sys
import time
import signal
import os
import urllib.request
import urllib.error
from pathlib import Path

# =============================================================================
# TUNABLE WAIT TIMES (seconds)
# Adjust these based on your PCIe bandwidth and model sizes
# =============================================================================
BRAIN_MAX_WAIT = 300      # Max seconds to wait for brain model to load
WORKER_MAX_WAIT = 300     # Max seconds to wait for each worker model to load
FLAG_CHECK_INTERVAL = 0.5 # How often to check for ready flags (seconds)
# =============================================================================

READY_FLAG_DIR = Path("/tmp/llm-orchestration-flags")
LAUNCH_LOCK = READY_FLAG_DIR / "launch.lock"
processes = []


def check_ollama_models(ollama_host: str = "http://localhost:11434") -> dict:
    """
    Check what models are currently loaded in Ollama.
    Returns dict with model names as keys, or empty dict if Ollama not running.
    """
    try:
        req = urllib.request.Request(f"{ollama_host}/api/ps")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            loaded = {}
            for model in data.get("models", []):
                name = model.get("name", "")
                loaded[name] = {
                    "size_vram": model.get("size_vram", 0),
                    "expires_at": model.get("expires_at", ""),
                }
            return loaded
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return {}


def check_existing_launch():
    """Check if another launch.py is already running. Exit if so."""
    if LAUNCH_LOCK.exists():
        try:
            pid = int(LAUNCH_LOCK.read_text().strip())
            # Check if process is alive
            os.kill(pid, 0)
            print(f"ERROR: Another launch.py is already running (PID {pid})")
            print(f"Kill it first: kill {pid}")
            print(f"Or remove stale lock: rm {LAUNCH_LOCK}")
            sys.exit(1)
        except (ProcessLookupError, ValueError):
            # Process dead or invalid PID, remove stale lock
            print(f"Removing stale lock file (old PID: {LAUNCH_LOCK.read_text().strip()})")
            LAUNCH_LOCK.unlink()

    # Create lock with our PID
    READY_FLAG_DIR.mkdir(parents=True, exist_ok=True)
    LAUNCH_LOCK.write_text(str(os.getpid()))


def remove_launch_lock():
    """Remove the launch lock file."""
    try:
        if LAUNCH_LOCK.exists():
            LAUNCH_LOCK.unlink()
    except Exception:
        pass


def clear_ready_flags():
    """Remove all ready flag files."""
    if READY_FLAG_DIR.exists():
        for f in READY_FLAG_DIR.glob("*.ready"):
            f.unlink()
    else:
        READY_FLAG_DIR.mkdir(parents=True, exist_ok=True)


def wait_for_ready_flag(name: str, max_wait: int) -> bool:
    """
    Wait for an agent to signal it's ready via flag file.
    Returns True if ready, False if timeout.
    """
    flag_file = READY_FLAG_DIR / f"{name}.ready"
    start_time = time.time()

    print(f"  Waiting for {name} to signal ready (max {max_wait}s)...")

    last_report = 0
    while time.time() - start_time < max_wait:
        if flag_file.exists():
            elapsed = int(time.time() - start_time)
            print(f"  {name} ready! (took {elapsed}s)")
            return True

        elapsed = int(time.time() - start_time)
        # Only print progress every 10 seconds to reduce spam
        if elapsed - last_report >= 10:
            print(f"  {name} loading... ({elapsed}s / {max_wait}s)")
            last_report = elapsed

        time.sleep(FLAG_CHECK_INTERVAL)

    print(f"  WARNING: {name} did not signal ready within {max_wait}s")
    return False


def cleanup(signum=None, frame=None):
    """Clean up all processes."""
    print("\nShutting down agents...")
    remove_launch_lock()
    for name, proc in processes:
        print(f"  Stopping {name}...")
        proc.terminate()

    for name, proc in processes:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    # Clean up ready flags
    clear_ready_flags()

    # Also kill any lingering ollama processes
    subprocess.run(["pkill", "-f", "ollama serve"], capture_output=True)
    print("All agents stopped.")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Launch agent system")
    # Default to config.json in same directory as this script
    default_config = Path(__file__).parent / "config.json"
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--workers-only", action="store_true", help="Start only workers")
    parser.add_argument("--brain-only", action="store_true", help="Start only brain (no workers)")
    parser.add_argument("--workers", type=int, default=None, metavar="N",
                        help="Number of workers to start (default: all). Use 0 for brain-only.")
    parser.add_argument("--hot", type=int, default=0, metavar="N",
                        help="Number of workers to start HOT with LLM loaded (default: 0, brain manages loading)")
    args = parser.parse_args()

    # Prevent concurrent launches
    check_existing_launch()

    # Convenience flag aliases
    if args.brain_only:
        args.workers = 0
    elif args.workers is None:
        args.workers = None  # None = all workers (default)

    if args.workers == 0:
        print("Note: Running in brain-only mode. Use --workers=N to start workers.")
    else:
        hot_count = args.hot
        print(f"Starting all workers: {hot_count} hot (LLM loaded), rest cold (script-only)")
    print()

    with open(args.config) as f:
        config = json.load(f)

    # Resolve relative paths in config
    agents_dir = Path(__file__).parent
    if not Path(config["shared_path"]).is_absolute():
        config["shared_path"] = str((agents_dir / config["shared_path"]).resolve())
    if not Path(config["permissions_path"]).is_absolute():
        config["permissions_path"] = str((agents_dir / config["permissions_path"]).resolve())

    scripts_dir = agents_dir

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print(f"Launch settings: BRAIN_MAX_WAIT={BRAIN_MAX_WAIT}s, WORKER_MAX_WAIT={WORKER_MAX_WAIT}s")
    print()

    # Clear any stale ready flags
    clear_ready_flags()

    # Check if models are already pre-loaded
    ollama_host = config.get("ollama_host", "http://localhost:11434")
    loaded_models = check_ollama_models(ollama_host)

    brain_model = config["brain"]["model"]
    worker_model = config["workers"][0]["model"] if config["workers"] else None

    brain_preloaded = brain_model in loaded_models
    worker_preloaded = worker_model in loaded_models if worker_model else False

    if loaded_models:
        print(f"Found pre-loaded models: {', '.join(loaded_models.keys())}")
        if brain_preloaded:
            print(f"  Brain model ({brain_model}) is pre-loaded - will reuse")
        if worker_preloaded:
            print(f"  Worker model ({worker_model}) is pre-loaded - will reuse")
        print()
        # Don't kill Ollama if we have pre-loaded models we want to use
        skip_ollama_restart = brain_preloaded or worker_preloaded
    else:
        skip_ollama_restart = False

    if skip_ollama_restart:
        print("Reusing existing Ollama instance with pre-loaded models")
    else:
        # Kill any existing ollama
        print("Stopping any existing Ollama instances...")
        subprocess.run(["pkill", "-f", "ollama serve"], capture_output=True)
        time.sleep(2)

    if not args.workers_only:
        # Start brain
        print(f"Starting brain on GPUs {config['brain']['gpus']}...")
        brain_cmd = [sys.executable, str(scripts_dir / "brain.py"), "--config", args.config]
        if brain_preloaded:
            brain_cmd.append("--model-preloaded")
        brain_proc = subprocess.Popen(
            brain_cmd,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        processes.append(("brain", brain_proc))

        # Wait for brain to signal ready (shorter wait if model already loaded)
        wait_time = 60 if brain_preloaded else BRAIN_MAX_WAIT
        if not wait_for_ready_flag("brain", wait_time):
            print("Brain failed to start, aborting...")
            cleanup()

    # Determine which workers to start
    if args.workers == 0:
        workers_to_start = []
    elif args.workers is None:
        workers_to_start = config["workers"]  # All workers
    else:
        workers_to_start = config["workers"][:args.workers]  # First N workers

    if workers_to_start:
        # Determine which workers start hot (with LLM loaded)
        hot_count = getattr(args, 'hot', 1)

        # Start workers sequentially, waiting for each to signal ready
        for i, worker in enumerate(workers_to_start):
            is_hot = i < hot_count
            mode = "HOT" if is_hot else "COLD"
            print(f"\nStarting {worker['name']} on GPU {worker['gpu']} ({mode})...")

            # Pass --hot flag to worker if it should preload model
            worker_cmd = [sys.executable, str(scripts_dir / "worker.py"), "--config", args.config, worker["name"]]
            if is_hot:
                worker_cmd.append("--hot")
            if worker_preloaded and is_hot:
                worker_cmd.append("--model-preloaded")

            worker_proc = subprocess.Popen(
                worker_cmd,
                stdout=sys.stdout,
                stderr=sys.stderr
            )
            processes.append((worker["name"], worker_proc))

            # Wait for this worker to signal ready before starting next
            # Shorter wait if model already loaded
            wait_time = 30 if (worker_preloaded and is_hot) else WORKER_MAX_WAIT
            if not wait_for_ready_flag(worker["name"], wait_time):
                print(f"WARNING: {worker['name']} may not be ready, continuing anyway...")

    # Get hot_count for display
    hot_count = getattr(args, 'hot', 1)

    print("\n" + "="*60)
    print("Agent system running!")
    print("="*60)
    print(f"Brain: {config['brain']['model']} on GPUs {config['brain']['gpus']}")

    if workers_to_start:
        print("\nActive workers:")
        for i, w in enumerate(workers_to_start):
            mode = "HOT" if i < hot_count else "COLD"
            print(f"  {w['name']}: GPU {w['gpu']} - {mode}")

        # Show which workers are available but not started
        idle_workers = config["workers"][len(workers_to_start):]
        if idle_workers:
            print("\nIdle workers (brain can start these when needed):")
            for w in idle_workers:
                print(f"  {w['name']}: GPU {w['gpu']} - NOT STARTED")
    else:
        print("\nWorkers: None started")
        print("Free GPUs for script tasks:")
        for w in config["workers"]:
            print(f"  GPU {w['gpu']} - FREE (can start {w['name']} when needed)")

    print("\nSubmit plans with: python scripts/submit.py <plan_folder> --config '{...}'")
    print("Press Ctrl+C to stop all agents")
    print("="*60 + "\n")

    # Monitor processes - restart workers that fail, but brain failure is fatal
    while True:
        for i, (name, proc) in enumerate(processes):
            if proc.poll() is not None:
                exit_code = proc.returncode
                print(f"\n{name} exited with code {exit_code}")

                if name == "brain":
                    print("Brain died - shutting down system")
                    cleanup()
                else:
                    # Worker died - just log it, don't restart automatically
                    # Worker can be restarted manually or by a supervisor
                    print(f"Worker {name} died. Tasks assigned to it will remain in queue.")
                    print(f"Restart with: python worker.py {name}")
                    # Remove from process list so we don't keep checking it
                    processes[i] = (name, None)

        # Check if all workers are dead
        alive_workers = [p for name, p in processes if p is not None and name != "brain"]
        if not alive_workers and not args.brain_only:
            print("\nAll workers have died. Brain still running.")
            print("Restart workers manually or press Ctrl+C to stop.")

        time.sleep(5)  # Check less frequently


if __name__ == "__main__":
    main()
