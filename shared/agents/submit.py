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
import shutil
from pathlib import Path
from datetime import datetime

PRIORITY_TIER_TO_VALUE = {
    "low": 3,
    "normal": 5,
    "high": 8,
    "urgent": 10,
}
SHARED_ALIASES = (
    "/mnt/shared",
    "/home/bryan/llm_orchestration/shared",
    "/media/bryan/shared",
)


def _to_runtime_shared_path(path_text: str) -> str:
    value = str(path_text)
    for prefix in SHARED_ALIASES:
        if value == prefix or value.startswith(prefix + "/"):
            return "/mnt/shared" + value[len(prefix):]
    return value


def _normalize_config_paths(value):
    if isinstance(value, str):
        return _to_runtime_shared_path(value)
    if isinstance(value, list):
        return [_normalize_config_paths(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize_config_paths(v) for k, v in value.items()}
    return value


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


def _find_plan_root(start: Path) -> Path:
    candidate = start.resolve()
    for folder in [candidate, *candidate.parents]:
        if (folder / "scripts").exists():
            return folder
    return start.resolve()


def _resolve_starter_file(plan_root: Path, plan_file_arg: str | None) -> Path:
    if not plan_file_arg:
        return plan_root / "plan.md"

    requested = Path(plan_file_arg)
    if requested.is_absolute():
        starter = requested
    else:
        direct = plan_root / requested
        in_input = plan_root / "input" / requested
        starter = direct if direct.exists() else in_input
    return starter.resolve()


def _prepare_runtime_plan_dir(plan_root: Path, starter_file: Path) -> Path:
    default_plan = (plan_root / "plan.md").resolve()
    if starter_file.resolve() == default_plan:
        return plan_root

    runtime_base = plan_root / "input" / ".submit_runtime"
    runtime_base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runtime_dir = runtime_base / f"{starter_file.stem}_{stamp}_{uuid.uuid4().hex[:8]}"
    runtime_dir.mkdir(parents=True, exist_ok=False)

    shutil.copy2(starter_file, runtime_dir / "plan.md")
    for name in ("scripts", "input", "history"):
        src = plan_root / name
        dst = runtime_dir / name
        if src.exists() and not dst.exists():
            dst.symlink_to(src, target_is_directory=True)
    return runtime_dir


def main():
    parser = argparse.ArgumentParser(description="Submit plan to brain")
    parser.add_argument("plan", help="Path to plan folder or plan.md")
    parser.add_argument(
        "--plan-file",
        type=str,
        default=None,
        help="Starter markdown file. Relative paths resolve under plan root, then plan_root/input/.",
    )
    parser.add_argument("--config", type=str, default="{}",
                        help="JSON config for variable substitution")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview parsed tasks and foreach expansion without queueing")
    parser.add_argument("--preview-batch-id", type=str, default=None,
                        help="Use this batch ID for dry-run variable substitution (lets foreach read existing manifest)")
    args = parser.parse_args()

    plan_arg = Path(args.plan).absolute()
    if plan_arg.is_file():
        starter_file = plan_arg.resolve()
        plan_path = _find_plan_root(starter_file.parent)
    else:
        plan_path = plan_arg.resolve()
        starter_file = _resolve_starter_file(plan_path, args.plan_file)

    if not plan_path.exists() or not plan_path.is_dir():
        print(f"Error: plan path is not a directory: {plan_path}")
        return 1
    if not starter_file.exists() or not starter_file.is_file():
        print(f"Error: starter plan file not found: {starter_file}")
        return 1
    try:
        starter_file.relative_to(plan_path)
    except ValueError:
        print(f"Error: starter plan file must be inside plan directory: {starter_file}")
        return 1

    try:
        config = json.loads(args.config)
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON for --config: {exc}")
        return 1
    if not isinstance(config, dict):
        print("Error: --config must decode to a JSON object")
        return 1
    config = _normalize_config_paths(config)

    if args.dry_run:
        plan_content = starter_file.read_text(encoding="utf-8")
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
        print(f"Starter file: {starter_file}")
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

    execute_plan_path = _prepare_runtime_plan_dir(plan_path, starter_file)
    priority_label = str(config.get("PRIORITY", "normal")).strip().lower()
    priority_value = PRIORITY_TIER_TO_VALUE.get(priority_label, PRIORITY_TIER_TO_VALUE["normal"])

    # Create execute_plan task for brain
    task = {
        "task_id": str(uuid.uuid4()),
        "type": "execute_plan",
        "executor": "brain",  # Brain handles plan parsing, not workers
        "plan_path": str(execute_plan_path),
        "config": config,
        "priority": priority_value,
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
    print(f"Starter file: {starter_file}")
    if execute_plan_path != plan_path:
        print(f"Runtime plan path: {execute_plan_path}")
    print(f"Config: {config}")
    print(f"\nMonitor: tail -f ~/llm_orchestration/shared/logs/brain_decisions.log")


if __name__ == "__main__":
    main()
