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


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


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
        for task_file in folder_path.glob("*.json"):
            try:
                with open(task_file) as f:
                    task = json.load(f)
                    tasks[status].append(task)
            except Exception:
                pass  # Skip malformed task file

    return tasks


def show_overview(config_path: str):
    """Show system overview."""
    config = load_config(config_path)
    shared_path = Path(config["shared_path"])
    tasks = get_tasks(shared_path)

    print("="*60)
    print("AGENT SYSTEM STATUS")
    print("="*60)

    print(f"\nConfiguration: {config_path}")
    print(f"Shared path: {shared_path}")

    print(f"\nBrain: {config['brain']['model']} (GPUs {config['brain']['gpus']})")
    print("Workers:")
    for w in config["workers"]:
        print(f"  {w['name']}: {w['model']} (GPU {w['gpu']}, port {w['port']})")

    print(f"\nTasks:")
    print(f"  Queued:     {len(tasks['queued'])}")
    print(f"  Processing: {len(tasks['processing'])}")
    print(f"  Complete:   {len(tasks['complete'])}")
    print(f"  Failed:     {len(tasks['failed'])}")

    if tasks["processing"]:
        print("\nActive tasks:")
        for t in tasks["processing"]:
            print(f"  {t['task_id'][:8]}... ({t.get('type', '?')}) -> {t.get('assigned_to', 'unassigned')}")


def show_tasks(config_path: str):
    """List all tasks."""
    config = load_config(config_path)
    shared_path = Path(config["shared_path"])
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
    shared_path = Path(config["shared_path"])

    for folder in ["queue", "processing", "complete", "failed"]:
        for task_file in (shared_path / "tasks" / folder).glob("*.json"):
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
