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

from hardware import scan_gpus, scan_runtime, scan_system, suggest_assignment


def print_hardware_scan(gpus, runtime, system):
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

    backend = str(runtime.get("backend", "llama")).strip()
    runtime_label = backend.capitalize()
    if runtime["running"]:
        print(f"\n  Runtime: {runtime_label} running ({runtime['host']})")
        if runtime["available_models"]:
            models_str = ", ".join(
                f"{m['name']} ({m['size_mb']} MB)"
                for m in runtime["available_models"]
            )
            print(f"    Available models: {models_str}")
        else:
            if backend == "llama":
                print("    Available models: reported only when the runtime is already serving")
            else:
                print("    Available models: (none — no models available)")
        if runtime["loaded_models"]:
            loaded_str = ", ".join(
                f"{m['name']} ({m['vram_mb']} MB VRAM)"
                for m in runtime["loaded_models"]
            )
            print(f"    Loaded models: {loaded_str}")
    else:
        print(f"\n  Runtime: {runtime_label} NOT responding ({runtime['host']})")


def print_suggestion(assignment):
    """Display suggested configuration."""
    print()
    print("=" * 50)
    print("  Suggested Configuration")
    print("=" * 50)
    print(f"\n  Mode: {assignment['_discovery_mode']}")

    brain = assignment["brain"]
    if brain["gpus"]:
        gpu_str = ", ".join(str(g) for g in brain["gpus"])
        print(f"  Brain: GPU {gpu_str} -> {brain['model'] or '(no model fits)'}")
    else:
        print(f"  Brain: CPU -> {brain['model'] or '(no model available)'}")

    workers = assignment["gpus"]
    if workers:
        for w in workers:
            print(
                f"  Worker: GPU {w['id']} ({w['name']}, {w['vram_mb']} MB) "
                f"-> {w['model'] or '(no model fits)'} "
                f"(port {w['port']})"
            )
    else:
        print("  Workers: (none)")

    print(f"  Worker mode: {assignment['worker_mode']}")
    print(f"  Initial hot workers: {assignment.get('initial_hot_workers', 0)}")


def prompt_confirm(assignment, runtime):
    """Ask user to accept, reject, or edit the configuration."""
    print()
    choice = input("  Accept this configuration? [Y/n/edit]: ").strip().lower()

    if choice in ("", "y", "yes"):
        return assignment
    elif choice == "edit":
        return prompt_edit(assignment, runtime)
    else:
        print("  Setup cancelled.")
        sys.exit(0)


def prompt_edit(assignment, runtime):
    """Let user edit the suggested configuration."""
    available_models = runtime.get("available_models", [])
    model_names = [m["name"] for m in available_models]

    while True:
        print()
        print("  What would you like to change?")
        print("    1. Brain model")
        print("    2. Brain GPU(s)")
        if assignment["gpus"]:
            print("    3. Worker model (applies to all)")
            print("    4. Worker mode (hot/cold)")
            print("    5. Initial hot workers on startup")
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

        elif choice == "3" and assignment["gpus"]:
            if model_names:
                print(f"  Available: {', '.join(model_names)}")
            model = input("  Worker model: ").strip()
            if model:
                for w in assignment["gpus"]:
                    w["model"] = model

        elif choice == "4" and assignment["gpus"]:
            mode = input("  Worker mode (hot/cold): ").strip().lower()
            if mode in ("hot", "cold"):
                assignment["worker_mode"] = mode
            else:
                print("  Invalid mode (use 'hot' or 'cold')")

        elif choice == "5" and assignment["gpus"]:
            raw = input("  Initial hot workers (0..N): ").strip()
            try:
                value = max(0, int(raw))
                assignment["initial_hot_workers"] = min(
                    value, len(assignment["gpus"])
                )
            except ValueError:
                print("  Invalid number")

        elif choice == "d":
            break

    return assignment


def prompt_worker_mode(assignment):
    """Ask user for hot/cold worker startup preference."""
    if not assignment["gpus"]:
        return assignment

    print()
    print("  Worker startup mode:")
    print("    1. Hot  - workers should prefer LLM-ready behavior")
    print("    2. Cold - leave workers empty by default (recommended)")
    print()
    choice = input("  > ").strip()

    if choice == "1":
        assignment["worker_mode"] = "hot"
    else:
        assignment["worker_mode"] = "cold"

    return assignment


def build_config(assignment, runtime, system):
    """Build the full config.json structure from the assignment."""
    worker_ids = [int(w["id"]) for w in assignment["gpus"] if w.get("id") is not None]
    warm_gpu_id = 2 if 2 in worker_ids else (worker_ids[0] if worker_ids else None)
    warm_gpu_name = f"gpu-{warm_gpu_id}" if warm_gpu_id is not None else ""
    config = {
        "_generated_by": "setup.py",
        "_generated_at": datetime.now().isoformat(),
        "_discovery_mode": assignment["_discovery_mode"],
        "_hardware": {
            "hostname": system["hostname"],
            "gpu_count": len(assignment["brain"]["gpus"])
            + len(assignment["gpus"]),
        },
        "shared_path": "../",
        "permissions_path": "permissions/",
        "runtime_backend": "llama",
        "model_search_roots": [
            "/mnt/shared/models",
        ],
        "runtime_host": runtime["host"],
        "brain_keep_alive": "30m",
        "worker_keep_alive": "30m",
        "brain_context_tokens": 8192,
        "worker_context_tokens": 8192,
        "auto_default_gpu": warm_gpu_name or "gpu-2",
        "auto_default_model": "qwen2.5:7b",
        "startup_warm_workers": [warm_gpu_name] if warm_gpu_name else [],
        "startup_meta_tasks": [
            {
                "name": "startup_single_default",
                "command": "load_llm",
                "target_model": "qwen2.5:7b",
                "load_mode": "single",
                "candidate_workers": [warm_gpu_name or "gpu-2"],
            },
            {
                "name": "startup_split_pair_1_3",
                "command": "load_split_llm",
                "target_model": "qwen2.5-coder:14b",
                "load_mode": "split",
                "candidate_groups": [
                    {
                        "id": "pair_1_3",
                        "members": ["gpu-1", "gpu-3"],
                        "port": 11440,
                    }
                ],
            },
        ],
        "llama_single_defaults": {
            "ctx_size": 2048,
            "batch_size": 64,
            "parallel": 1,
            "n_gpu_layers": 999,
        },
        "llama_split_defaults": {
            "ctx_size": 4096,
            "batch_size": 128,
            "parallel": 1,
            "n_gpu_layers": 999,
        },
        "llama_single_profiles": {
            "qwen2.5:7b": {
                "ctx_size": 2048,
                "batch_size": 64,
                "parallel": 1,
                "n_gpu_layers": 999,
                "extra_args": ["--no-warmup"],
            },
        },
        "llama_split_profiles": {
            "qwen2.5-coder:14b": {
                "ctx_size": 4096,
                "batch_size": 128,
                "parallel": 1,
                "n_gpu_layers": 999,
                "tensor_split": "1,1",
                "extra_args": ["--no-warmup"],
                "meta_timeout_seconds": 600,
            },
        },
        "brain": {
            "name": "brain",
            "model": assignment["brain"]["model"],
            "gpus": assignment["brain"]["gpus"],
        },
        "gpus": [],
        "worker_mode": assignment["worker_mode"],
        "initial_hot_workers": assignment.get("initial_hot_workers", 0),
        "timeouts": {
            "poll_interval_seconds": 5,
            "brain_think_seconds": 120,
            "worker_task_seconds": 0,
        },
        "resource_limits": {
            "max_temp_c": 80,
            "gpu_temp_warning_c": 75,
            "gpu_temp_critical_c": 90,
            "cpu_temp_warning_c": 80,
            "cpu_temp_critical_c": 95,
            "max_vram_percent": 95,
            "max_power_w": 140,
        },
        "retry_policy": {"max_attempts": 3},
    }

    for w in assignment["gpus"]:
        config["gpus"].append(
            {
                "name": w["name"],
                "id": w["id"],
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
        "--runtime-host",
        default="http://localhost:11434",
        help="Runtime API host",
    )
    parser.add_argument("--brain-model", help="Override brain model choice")
    parser.add_argument("--worker-model", help="Override worker model choice")
    parser.add_argument(
        "--worker-mode",
        choices=["hot", "cold"],
        help="Worker startup mode",
    )
    parser.add_argument(
        "--initial-hot-workers",
        type=int,
        default=None,
        help="Number of workers to warm via startup load_llm tasks (default: 0)",
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
    runtime_host = str(args.runtime_host).strip()
    runtime = scan_runtime(runtime_host, runtime_backend="llama")
    system = scan_system()

    # Build preferences from CLI args
    preferences = {"worker_mode": "cold"}
    if args.worker_mode:
        preferences["worker_mode"] = args.worker_mode

    # Get suggestion
    assignment = suggest_assignment(gpus, runtime, preferences)

    # Apply CLI overrides
    if args.brain_model:
        assignment["brain"]["model"] = args.brain_model
    if args.worker_model:
        for w in assignment["gpus"]:
            w["model"] = args.worker_model
    if args.worker_mode:
        assignment["worker_mode"] = args.worker_mode
    if args.initial_hot_workers is not None:
        assignment["initial_hot_workers"] = max(0, args.initial_hot_workers)
    else:
        assignment["initial_hot_workers"] = assignment.get("initial_hot_workers", 0)

    # Display scan results and suggestion
    print_hardware_scan(gpus, runtime, system)
    print_suggestion(assignment)

    if not args.yes:
        # Interactive confirmation
        assignment = prompt_confirm(assignment, runtime)

        # Ask about worker mode if not already set via CLI
        if not args.worker_mode and assignment["gpus"]:
            assignment = prompt_worker_mode(assignment)
        if args.initial_hot_workers is None and assignment["gpus"]:
            default_hot = assignment.get("initial_hot_workers", 0)
            raw = input(
                f"  Initial hot workers on startup (0..{len(assignment['gpus'])}) [{default_hot}]: "
            ).strip()
            if raw:
                try:
                    assignment["initial_hot_workers"] = max(0, int(raw))
                except ValueError:
                    print("  Invalid number, keeping default")

    if assignment.get("initial_hot_workers", 0) > len(assignment.get("gpus", [])):
        assignment["initial_hot_workers"] = len(assignment.get("gpus", []))

    # Build and write config
    config = build_config(assignment, runtime, system)

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
