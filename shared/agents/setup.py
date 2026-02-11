#!/usr/bin/env python3
"""
Interactive hardware setup for LLM orchestration.

Scans hardware, suggests GPU assignment, lets user confirm or edit,
writes config.json. Run once per hardware change.

Usage:
  python setup.py                          # Interactive
  python setup.py --yes                    # Accept defaults
  python setup.py --brain-model qwen2.5:14b --worker-model qwen2.5:7b --yes
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from hardware import scan_gpus, scan_ollama, scan_system, suggest_assignment


def print_hardware_scan(gpus, ollama, system):
    """Display discovered hardware."""
    print()
    print("=" * 50)
    print("  Hardware Scan")
    print("=" * 50)

    # System
    print(f"\n  Host: {system['hostname']}")
    print(f"  CPU: {system['cpu_cores']} cores")
    print(
        f"  RAM: {system['ram_available_mb']} MB available "
        f"/ {system['ram_total_mb']} MB total"
    )

    # GPUs
    print(f"\n  GPUs found: {len(gpus)}")
    if gpus:
        for gpu in gpus:
            print(
                f"    GPU {gpu['index']}: {gpu['name']} "
                f"[{gpu['vram_mb']} MB VRAM] "
                f"{gpu['temp_c']}C"
            )
    else:
        print("    (none — CPU-only mode)")

    # Ollama
    if ollama["running"]:
        print(f"\n  Ollama: running ({ollama['host']})")
        if ollama["available_models"]:
            models_str = ", ".join(
                f"{m['name']} ({m['size_mb']} MB)"
                for m in ollama["available_models"]
            )
            print(f"    Available models: {models_str}")
        else:
            print("    Available models: (none — pull some with 'ollama pull')")
        if ollama["loaded_models"]:
            loaded_str = ", ".join(
                f"{m['name']} ({m['vram_mb']} MB VRAM)"
                for m in ollama["loaded_models"]
            )
            print(f"    Loaded models: {loaded_str}")
    else:
        print(f"\n  Ollama: NOT running ({ollama['host']})")
        print("    Start with: ollama serve")


def print_suggestion(assignment):
    """Display suggested configuration."""
    print()
    print("=" * 50)
    print("  Suggested Configuration")
    print("=" * 50)
    print(f"\n  Mode: {assignment['mode']}")

    brain = assignment["brain"]
    if brain["gpus"]:
        gpu_str = ", ".join(str(g) for g in brain["gpus"])
        print(f"  Brain: GPU {gpu_str} -> {brain['model'] or '(no model fits)'}")
    else:
        print(f"  Brain: CPU -> {brain['model'] or '(no model available)'}")

    workers = assignment["workers"]
    if workers:
        for w in workers:
            print(
                f"  Worker: GPU {w['index']} ({w['name']}, {w['vram_mb']} MB) "
                f"-> {w['model'] or '(no model fits)'} "
                f"(port {w['port']})"
            )
    else:
        print("  Workers: (none)")

    print(f"  Worker mode: {assignment['worker_mode']}")


def prompt_confirm(assignment, ollama):
    """Ask user to accept, reject, or edit the configuration."""
    print()
    choice = input("  Accept this configuration? [Y/n/edit]: ").strip().lower()

    if choice in ("", "y", "yes"):
        return assignment
    elif choice == "edit":
        return prompt_edit(assignment, ollama)
    else:
        print("  Setup cancelled.")
        sys.exit(0)


def prompt_edit(assignment, ollama):
    """Let user edit the suggested configuration."""
    available_models = ollama.get("available_models", [])
    model_names = [m["name"] for m in available_models]

    while True:
        print()
        print("  What would you like to change?")
        print("    1. Brain model")
        print("    2. Brain GPU(s)")
        if assignment["workers"]:
            print("    3. Worker model (applies to all)")
            print("    4. Worker mode (hot/cold)")
        print("    d. Done editing")
        print()

        choice = input("  > ").strip().lower()

        if choice == "1":
            if model_names:
                print(f"  Available: {', '.join(model_names)}")
            model = input("  Brain model: ").strip()
            if model:
                assignment["brain"]["model"] = model

        elif choice == "2":
            gpu_str = input("  Brain GPU indices (comma-separated): ").strip()
            try:
                indices = [int(x.strip()) for x in gpu_str.split(",")]
                assignment["brain"]["gpus"] = indices
            except ValueError:
                print("  Invalid GPU indices")

        elif choice == "3" and assignment["workers"]:
            if model_names:
                print(f"  Available: {', '.join(model_names)}")
            model = input("  Worker model: ").strip()
            if model:
                for w in assignment["workers"]:
                    w["model"] = model

        elif choice == "4" and assignment["workers"]:
            mode = input("  Worker mode (hot/cold): ").strip().lower()
            if mode in ("hot", "cold"):
                assignment["worker_mode"] = mode
            else:
                print("  Invalid mode (use 'hot' or 'cold')")

        elif choice == "d":
            break

    return assignment


def prompt_worker_mode(assignment):
    """Ask user for hot/cold worker startup preference."""
    if not assignment["workers"]:
        return assignment

    print()
    print("  Worker startup mode:")
    print("    1. Hot  - preload LLM into worker GPUs on startup")
    print("    2. Cold - leave GPUs empty on startup (for script/compute)")
    print()
    choice = input("  > ").strip()

    if choice == "2":
        assignment["worker_mode"] = "cold"
    else:
        assignment["worker_mode"] = "hot"

    return assignment


def build_config(assignment, ollama, system):
    """Build the full config.json structure from the assignment."""
    config = {
        "_generated_by": "setup.py",
        "_generated_at": datetime.now().isoformat(),
        "_discovery_mode": assignment["mode"],
        "_hardware": {
            "hostname": system["hostname"],
            "gpu_count": len(assignment["brain"]["gpus"])
            + len(assignment["workers"]),
        },
        "shared_path": "../",
        "permissions_path": "permissions/",
        "ollama_host": ollama["host"],
        "brain": {
            "name": "brain",
            "model": assignment["brain"]["model"],
            "gpus": assignment["brain"]["gpus"],
        },
        "gpus": [],
        "worker_mode": assignment["worker_mode"],
        "timeouts": {
            "poll_interval_seconds": 5,
            "brain_think_seconds": 120,
            "worker_task_seconds": 120,
        },
        "resource_limits": {
            "max_temp_c": 80,
            "max_vram_percent": 95,
            "max_power_w": 140,
        },
        "retry_policy": {"max_attempts": 3},
    }

    for w in assignment["workers"]:
        config["gpus"].append(
            {
                "name": w["name"],
                "id": w["index"],
                "vram_mb": w["vram_mb"],
                "model": w["model"],
                "port": w["port"],
                "permissions": "worker.json",
            }
        )

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Interactive hardware setup for LLM orchestration"
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config.json"),
        help="Output config file path (default: config.json)",
    )
    parser.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="Ollama API host",
    )
    parser.add_argument("--brain-model", help="Override brain model choice")
    parser.add_argument("--worker-model", help="Override worker model choice")
    parser.add_argument(
        "--worker-mode",
        choices=["hot", "cold"],
        help="Worker startup mode",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Accept defaults without prompting",
    )
    args = parser.parse_args()

    # Scan hardware
    gpus = scan_gpus()
    ollama = scan_ollama(args.ollama_host)
    system = scan_system()

    # Build preferences from CLI args
    preferences = {}
    if args.worker_mode:
        preferences["worker_mode"] = args.worker_mode

    # Get suggestion
    assignment = suggest_assignment(gpus, ollama, preferences)

    # Apply CLI overrides
    if args.brain_model:
        assignment["brain"]["model"] = args.brain_model
    if args.worker_model:
        for w in assignment["workers"]:
            w["model"] = args.worker_model
    if args.worker_mode:
        assignment["worker_mode"] = args.worker_mode

    # Display scan results and suggestion
    print_hardware_scan(gpus, ollama, system)
    print_suggestion(assignment)

    if not args.yes:
        # Interactive confirmation
        assignment = prompt_confirm(assignment, ollama)

        # Ask about worker mode if not already set via CLI
        if not args.worker_mode and assignment["workers"]:
            assignment = prompt_worker_mode(assignment)

    # Build and write config
    config = build_config(assignment, ollama, system)

    config_path = Path(args.config)
    if config_path.exists() and not args.yes:
        overwrite = (
            input(f"\n  {config_path} exists. Overwrite? [y/N]: ").strip().lower()
        )
        if overwrite not in ("y", "yes"):
            print("  Cancelled.")
            sys.exit(0)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    print(f"\n  Config written to {config_path}")
    print(f"  Run 'python startup.py' to launch the system.")
    print()


if __name__ == "__main__":
    main()
