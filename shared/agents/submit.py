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
  tail -f ~/llm_orchestration/shared/logs/brain_decisions.log
"""
import json
import uuid
import argparse
import re
from pathlib import Path
from datetime import datetime


def _parse_plan_tasks(plan_content: str):
    tasks = []
    in_tasks = False
    current = None
    for raw in plan_content.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if stripped == "## Tasks":
            in_tasks = True
            continue
        if not in_tasks:
            continue
        if stripped.startswith("## ") and stripped != "## Tasks":
            break
        if stripped.startswith("### "):
            if current and current.get("command"):
                tasks.append(current)
            current = {
                "id": stripped[4:].strip(),
                "executor": "worker",
                "task_class": None,
                "command": "",
                "depends_on": [],
                "foreach": None,
                "batch_size": 1,
                "requires": [],
                "produces": [],
            }
            continue
        if not current:
            continue
        if stripped.startswith("- **executor**:"):
            current["executor"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("- **task_class**:"):
            current["task_class"] = stripped.split(":", 1)[1].strip().lower()
        elif stripped.startswith("- **command**:"):
            match = re.search(r"`([^`]+)`", stripped)
            if match:
                current["command"] = match.group(1)
        elif stripped.startswith("- **depends_on**:"):
            deps = stripped.split(":", 1)[1].strip()
            if deps.lower() != "none":
                current["depends_on"] = [d.strip() for d in deps.split(",") if d.strip()]
        elif stripped.startswith("- **foreach**:"):
            current["foreach"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("- **batch_size**:"):
            raw_size = stripped.split(":", 1)[1].strip()
            try:
                current["batch_size"] = max(1, int(raw_size))
            except ValueError:
                current["batch_size"] = 1
        elif stripped.startswith("- **requires**:"):
            req = stripped.split(":", 1)[1].strip()
            current["requires"] = [r.strip() for r in req.split(",") if r.strip()]
        elif stripped.startswith("- **produces**:"):
            out = stripped.split(":", 1)[1].strip()
            current["produces"] = [o.strip() for o in out.split(",") if o.strip()]
    if current and current.get("command"):
        tasks.append(current)
    return tasks


def _substitute(text: str, variables: dict) -> str:
    out = text
    for key, value in variables.items():
        out = out.replace(key, value)
    return out


def _preview_foreach(task: dict, variables: dict):
    spec = task.get("foreach")
    if not spec:
        return {"ok": True, "item_count": None, "expanded_count": 1}
    if ":" not in spec:
        return {"ok": False, "error": f"invalid foreach format: {spec}"}
    file_path, json_path = spec.rsplit(":", 1)
    file_path = Path(_substitute(file_path, variables))
    if not file_path.exists():
        return {"ok": False, "error": f"foreach source missing: {file_path}"}
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "error": f"foreach source unreadable: {file_path} ({exc})"}
    items = data
    for key in json_path.split("."):
        if isinstance(items, dict) and key in items:
            items = items[key]
        else:
            return {"ok": False, "error": f"json path not found: {json_path} in {file_path}"}
    if not isinstance(items, list):
        return {"ok": False, "error": f"foreach target is not a list: {json_path}"}
    batch_size = max(1, int(task.get("batch_size", 1) or 1))
    expanded = (len(items) + batch_size - 1) // batch_size
    return {"ok": True, "item_count": len(items), "expanded_count": expanded}


def main():
    parser = argparse.ArgumentParser(description="Submit plan to brain")
    parser.add_argument("plan", help="Path to plan folder or plan.md")
    parser.add_argument("--config", type=str, default="{}",
                        help="JSON config for variable substitution")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview parsed tasks and foreach expansion without queueing")
    parser.add_argument("--preview-batch-id", type=str, default=None,
                        help="Use this batch ID for dry-run variable substitution (lets foreach read existing manifest)")
    args = parser.parse_args()

    plan_path = Path(args.plan).absolute()
    if plan_path.is_file():
        plan_path = plan_path.parent

    if not (plan_path / "plan.md").exists():
        print(f"Error: No plan.md in {plan_path}")
        return 1

    try:
        config = json.loads(args.config)
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON for --config: {exc}")
        return 1
    if not isinstance(config, dict):
        print("Error: --config must decode to a JSON object")
        return 1

    if args.dry_run:
        plan_file = plan_path / "plan.md"
        plan_content = plan_file.read_text(encoding="utf-8")
        tasks = _parse_plan_tasks(plan_content)
        preview_batch_id = args.preview_batch_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        variables = {
            "{BATCH_ID}": preview_batch_id,
            "{PLAN_PATH}": str(plan_path),
            "{BATCH_PATH}": str((plan_path / "history" / preview_batch_id)),
        }
        for key, value in config.items():
            variables[f"{{{key}}}"] = str(value)

        total_expanded = 0
        print(f"Dry run: {plan_path.name}")
        print(f"Plan path: {plan_path}")
        print(f"Preview batch_id: {preview_batch_id}")
        print(f"Template tasks: {len(tasks)}")
        print("")
        for task in tasks:
            fx = _preview_foreach(task, variables)
            expanded = fx.get("expanded_count", 0) if fx.get("ok") else 0
            total_expanded += expanded
            print(f"- {task['id']} [{task.get('task_class') or 'unknown'}] executor={task.get('executor','worker')}")
            print(f"  depends_on: {', '.join(task.get('depends_on') or ['none'])}")
            if task.get("requires"):
                print(f"  requires: {', '.join(_substitute(x, variables) for x in task['requires'])}")
            if task.get("produces"):
                print(f"  produces: {', '.join(_substitute(x, variables) for x in task['produces'])}")
            if task.get("foreach"):
                print(f"  foreach: {_substitute(task['foreach'], variables)}")
                if fx.get("ok"):
                    print(f"  expansion: items={fx['item_count']} batch_size={task.get('batch_size',1)} tasks={fx['expanded_count']}")
                else:
                    print(f"  expansion: ERROR {fx.get('error')}")
            else:
                print("  expansion: tasks=1")
        print("")
        print(f"Estimated runtime task count (before brain auto-summary): {total_expanded}")
        print(f"Estimated runtime task count (including auto-summary): {total_expanded + 1}")
        print("Dry run only: nothing queued.")
        return 0

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
    queue_path = Path("/mnt/shared/tasks/queue")
    queue_path.mkdir(parents=True, exist_ok=True)

    task_file = queue_path / f"{task['task_id']}.json"
    with open(task_file, 'w') as f:
        json.dump(task, f, indent=2)

    print(f"Submitted: {plan_path.name}")
    print(f"Task ID: {task['task_id'][:8]}")
    print(f"Config: {config}")
    print(f"\nMonitor: tail -f ~/llm_orchestration/shared/logs/brain_decisions.log")


if __name__ == "__main__":
    main()
