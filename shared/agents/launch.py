#!/usr/bin/env python3
"""
Launch the agent system (brain + GPU agents).

DEPRECATED: Use startup.py instead, which adds hardware verification and
degraded-mode support. Generate config with 'python setup.py' first.

Usage:
  python launch.py              # Start brain + all GPU agents
  python launch.py --brain-only # Start only brain
  python launch.py --gpus 2     # Start brain + first N GPUs
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
# =============================================================================
BRAIN_MAX_WAIT = 300      # Max seconds to wait for brain model to load
GPU_MAX_WAIT = 60         # Max seconds to wait for each GPU agent to start
FLAG_CHECK_INTERVAL = 0.5 # How often to check for ready flags (seconds)
# =============================================================================

READY_FLAG_DIR = Path("/tmp/llm-orchestration-flags")
LAUNCH_LOCK = READY_FLAG_DIR / "launch.lock"
processes = []


def check_ollama_models(ollama_host: str = "http://localhost:11434") -> dict:
    """Check what models are currently loaded in Ollama."""
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
            os.kill(pid, 0)
            print(f"ERROR: Another launch.py is already running (PID {pid})")
            print(f"Kill it first: kill {pid}")
            print(f"Or remove stale lock: rm {LAUNCH_LOCK}")
            sys.exit(1)
        except (ProcessLookupError, ValueError):
            print(f"Removing stale lock file (old PID: {LAUNCH_LOCK.read_text().strip()})")
            LAUNCH_LOCK.unlink()

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
    """Wait for an agent to signal ready via flag file."""
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

    clear_ready_flags()

    # Kill any lingering ollama processes
    subprocess.run(["pkill", "-f", "ollama serve"], capture_output=True)
    print("All agents stopped.")
    sys.exit(0)


def main():
    import warnings
    warnings.warn(
        "launch.py is deprecated. Use 'python startup.py' instead "
        "(with 'python setup.py' to generate config).",
        DeprecationWarning, stacklevel=2
    )
    print("WARNING: launch.py is deprecated. Use 'python startup.py' instead.")
    print("         Run 'python setup.py' first to generate config from hardware.\n")

    parser = argparse.ArgumentParser(description="Launch agent system")
    default_config = Path(__file__).parent / "config.json"
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--brain-only", action="store_true", help="Start only brain")
    parser.add_argument("--gpus", type=int, default=None, metavar="N",
                        help="Number of GPU agents to start (default: all)")
    args = parser.parse_args()

    check_existing_launch()

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

    print(f"Launch settings: BRAIN_MAX_WAIT={BRAIN_MAX_WAIT}s, GPU_MAX_WAIT={GPU_MAX_WAIT}s")
    print()

    clear_ready_flags()

    # Check if brain model is already pre-loaded
    ollama_host = config.get("ollama_host", "http://localhost:11434")
    loaded_models = check_ollama_models(ollama_host)
    brain_model = config["brain"]["model"]
    brain_preloaded = brain_model in loaded_models

    if loaded_models:
        print(f"Found pre-loaded models: {', '.join(loaded_models.keys())}")
        if brain_preloaded:
            print(f"  Brain model ({brain_model}) is pre-loaded - will reuse")
        print()
        skip_ollama_restart = brain_preloaded
    else:
        skip_ollama_restart = False

    if skip_ollama_restart:
        print("Reusing existing Ollama instance with pre-loaded models")
    else:
        print("Stopping any existing Ollama instances...")
        subprocess.run(["pkill", "-f", "ollama serve"], capture_output=True)
        time.sleep(2)

    # --- Start Brain ---
    print(f"Starting brain on GPUs {config['brain']['gpus']}...")
    brain_cmd = [sys.executable, str(scripts_dir / "brain.py"), "--config", args.config]
    if brain_preloaded:
        brain_cmd.append("--model-preloaded")
    brain_proc = subprocess.Popen(
        brain_cmd, stdout=sys.stdout, stderr=sys.stderr
    )
    processes.append(("brain", brain_proc))

    wait_time = 60 if brain_preloaded else BRAIN_MAX_WAIT
    if not wait_for_ready_flag("brain", wait_time):
        print("Brain failed to start, aborting...")
        cleanup()

    # --- Start GPU Agents ---
    if args.brain_only:
        gpus_to_start = []
    elif args.gpus is not None:
        gpus_to_start = config["gpus"][:args.gpus]
    else:
        gpus_to_start = config["gpus"]

    if gpus_to_start:
        print(f"\nStarting {len(gpus_to_start)} GPU agents...")

        for gpu_config in gpus_to_start:
            gpu_name = gpu_config["name"]
            print(f"\n  Starting {gpu_name} (GPU {gpu_config['id']})...")

            gpu_cmd = [
                sys.executable, str(scripts_dir / "gpu.py"),
                gpu_name,
                "--config", args.config
            ]

            gpu_proc = subprocess.Popen(
                gpu_cmd, stdout=sys.stdout, stderr=sys.stderr
            )
            processes.append((gpu_name, gpu_proc))

            if not wait_for_ready_flag(gpu_name, GPU_MAX_WAIT):
                print(f"WARNING: {gpu_name} may not be ready, continuing anyway...")

    # --- Running ---
    print("\n" + "=" * 60)
    print("Agent system running!")
    print("=" * 60)
    print(f"Brain: {config['brain']['model']} on GPUs {config['brain']['gpus']}")

    if gpus_to_start:
        print(f"\nGPU agents ({len(gpus_to_start)}):")
        for g in gpus_to_start:
            print(f"  {g['name']}: GPU {g['id']} - {g['model']} (port {g['port']})")
    else:
        print("\nGPU agents: None started")

    print("\nSubmit plans with: python scripts/submit.py <plan_folder> --config '{...}'")
    print("Press Ctrl+C to stop all agents")
    print("=" * 60 + "\n")

    # Monitor processes
    while True:
        for i, (name, proc) in enumerate(processes):
            if proc.poll() is not None:
                exit_code = proc.returncode
                print(f"\n{name} exited with code {exit_code}")

                if name == "brain":
                    print("Brain died - shutting down system")
                    cleanup()
                else:
                    print(f"GPU agent {name} died. Tasks assigned to it may need re-queuing.")
                    print(f"Restart with: python gpu.py {name}")
                    processes[i] = (name, None)

        alive_gpus = [p for name, p in processes if p is not None and name != "brain"]
        if not alive_gpus and not args.brain_only:
            print("\nAll GPU agents have died. Brain still running.")
            print("Restart GPU agents manually or press Ctrl+C to stop.")

        time.sleep(5)


if __name__ == "__main__":
    main()
