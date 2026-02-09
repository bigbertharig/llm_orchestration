#!/usr/bin/env python3
"""
Submit a plan for brain to execute.

This script creates a single execute_plan task. The brain:
1. Parses plan.md
2. Creates tasks with dependencies
3. Releases tasks when ready
4. Workers execute

Usage:
  python submit.py /path/to/plan_folder
  python submit.py /path/to/plan_folder --config '{"INPUT_FOLDER": "/data"}'

Monitor:
  tail -f ~/Documents/llm_orchestration/shared/logs/brain_decisions.log
"""
import json
import uuid
import argparse
from pathlib import Path
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Submit plan to brain")
    parser.add_argument("plan", help="Path to plan folder or plan.md")
    parser.add_argument("--config", type=str, default="{}",
                        help="JSON config for variable substitution")
    args = parser.parse_args()

    plan_path = Path(args.plan).absolute()
    if plan_path.is_file():
        plan_path = plan_path.parent

    if not (plan_path / "plan.md").exists():
        print(f"Error: No plan.md in {plan_path}")
        return 1

    config = json.loads(args.config)

    # Create execute_plan task for brain
    task = {
        "task_id": str(uuid.uuid4()),
        "type": "execute_plan",
        "executor": "brain",  # Brain handles plan parsing, not workers
        "plan_path": str(plan_path),
        "config": config,
        "priority": 10,  # High priority
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "created_by": "submit"
    }

    # Write to queue
    queue_path = Path(__file__).parent.parent / "shared" / "tasks" / "queue"
    queue_path.mkdir(parents=True, exist_ok=True)

    task_file = queue_path / f"{task['task_id']}.json"
    with open(task_file, 'w') as f:
        json.dump(task, f, indent=2)

    print(f"Submitted: {plan_path.name}")
    print(f"Task ID: {task['task_id'][:8]}")
    print(f"Config: {config}")
    print(f"\nMonitor: tail -f ~/Documents/llm_orchestration/shared/logs/brain_decisions.log")


if __name__ == "__main__":
    main()
