#!/usr/bin/env python3
"""
Live monitor for the agent system.
Shows task queues, active work, and recent completions.

Usage:
  python watch.py              # Full dashboard, refreshes every 2s
  python watch.py --once       # Single snapshot
  python watch.py --tail       # Follow completed task outputs
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# ANSI colors
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"


def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def resolve_shared_path(config_path: str, config: dict) -> Path:
    shared = Path(config.get("shared_path", "../"))
    if shared.is_absolute():
        return shared
    return (Path(config_path).resolve().parent / shared).resolve()


def iter_task_files(folder_path: Path):
    for task_file in folder_path.glob("*.json"):
        if task_file.name.endswith(".heartbeat.json"):
            continue
        yield task_file


def get_tasks(shared_path: Path) -> dict:
    tasks = {"queue": [], "processing": [], "complete": [], "failed": []}

    for status, folder in tasks.items():
        folder_path = shared_path / "tasks" / status
        files = sorted(iter_task_files(folder_path), key=lambda p: p.stat().st_mtime, reverse=True)
        for task_file in files:
            try:
                with open(task_file) as f:
                    task = json.load(f)
                    tasks[status].append(task)
            except Exception:
                pass  # Skip malformed task file

    return tasks


def get_workers(config: dict):
    if "workers" in config and isinstance(config["workers"], list):
        return config["workers"]
    workers = []
    for gpu in config.get("gpus", []):
        workers.append({
            "name": gpu.get("name", f"gpu-{gpu.get('id', '?')}"),
            "gpu": gpu.get("id", "?"),
            "model": gpu.get("model", "?"),
            "port": gpu.get("port", "?"),
        })
    return workers


def _heartbeat_age_seconds(last_updated: str):
    if not last_updated:
        return None
    try:
        return int((datetime.now() - datetime.fromisoformat(last_updated)).total_seconds())
    except Exception:
        return None


def get_worker_heartbeats(shared_path: Path):
    rows = []

    gpus_dir = shared_path / "gpus"
    if gpus_dir.exists():
        for hb_file in sorted(gpus_dir.glob("gpu_*/heartbeat.json")):
            try:
                hb = json.loads(hb_file.read_text())
                rows.append({
                    "name": hb.get("name", hb_file.parent.name),
                    "worker_type": "gpu",
                    "state": hb.get("state", "?"),
                    "host": hb.get("hostname", "-"),
                    "last_updated": hb.get("last_updated", ""),
                    "age_s": _heartbeat_age_seconds(hb.get("last_updated", "")),
                    "temp_c": hb.get("temperature_c"),
                    "cpu_temp_c": hb.get("cpu_temp_c"),
                    "active_workers": hb.get("active_workers", 0),
                })
            except Exception:
                continue

    cpus_dir = shared_path / "cpus"
    if cpus_dir.exists():
        for hb_file in sorted(cpus_dir.glob("*/heartbeat.json")):
            try:
                hb = json.loads(hb_file.read_text())
                rows.append({
                    "name": hb.get("name", hb_file.parent.name),
                    "worker_type": "cpu",
                    "state": hb.get("state", "?"),
                    "host": hb.get("hostname", "-"),
                    "last_updated": hb.get("last_updated", ""),
                    "age_s": _heartbeat_age_seconds(hb.get("last_updated", "")),
                    "temp_c": None,
                    "cpu_temp_c": None,
                    "active_workers": 1 if hb.get("active_task_id") else 0,
                })
            except Exception:
                continue

    return rows


def get_active_batches(shared_path: Path):
    state_path = shared_path / "brain" / "state.json"
    if not state_path.exists():
        return {}
    try:
        with open(state_path) as f:
            state = json.load(f)
        active = state.get("active_batches", {})
        if isinstance(active, dict):
            return active
    except Exception:
        pass
    return {}


def batch_counts(tasks: dict, batch_id: str):
    return {
        "queue": sum(1 for t in tasks["queue"] if t.get("batch_id") == batch_id),
        "processing": sum(1 for t in tasks["processing"] if t.get("batch_id") == batch_id),
        "complete": sum(1 for t in tasks["complete"] if t.get("batch_id") == batch_id),
        "failed": sum(1 for t in tasks["failed"] if t.get("batch_id") == batch_id),
    }


def format_time(iso_str: str) -> str:
    """Format ISO timestamp to relative or short time."""
    if not iso_str:
        return "-"
    try:
        dt = datetime.fromisoformat(iso_str)
        delta = datetime.now() - dt
        if delta.total_seconds() < 60:
            return f"{int(delta.total_seconds())}s ago"
        elif delta.total_seconds() < 3600:
            return f"{int(delta.total_seconds() / 60)}m ago"
        else:
            return dt.strftime("%H:%M:%S")
    except Exception:
        return iso_str[:8]


def truncate(s: str, length: int) -> str:
    if len(s) <= length:
        return s
    return s[:length-3] + "..."


def print_header(config: dict):
    print(f"{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}║           LLM AGENT SYSTEM MONITOR                               ║{RESET}")
    print(f"{BOLD}{CYAN}╚══════════════════════════════════════════════════════════════════╝{RESET}")
    print()

    # System info
    brain = config["brain"]
    print(f"{BOLD}Brain:{RESET} {brain['model']} on GPUs {brain['gpus']}")

    workers = get_workers(config)
    worker_str = ", ".join(f"{w['name']}(GPU {w['gpu']})" for w in workers)
    print(f"{BOLD}Workers:{RESET} {worker_str}")
    print()


def print_progress_summary(tasks: dict, shared_path: Path):
    print(f"{BOLD}Global Tasks:{RESET} "
          f"queued={len(tasks['queue'])} processing={len(tasks['processing'])} "
          f"complete={len(tasks['complete'])} failed={len(tasks['failed'])}")

    active_batches = get_active_batches(shared_path)
    if not active_batches:
        print(f"{BOLD}Active Batch:{RESET} (none)")
        print()
        return

    print(f"{BOLD}Active Batch Progress:{RESET}")
    for batch_id, meta in sorted(active_batches.items()):
        counts = batch_counts(tasks, batch_id)
        observed_total = sum(counts.values())
        total_hint = meta.get("total_tasks")
        if isinstance(total_hint, int) and total_hint > 0:
            total_tasks = max(total_hint, observed_total)
        else:
            total_tasks = observed_total
        print(
            f"  {batch_id}: {counts['complete']}/{total_tasks} complete "
            f"(queued={counts['queue']} processing={counts['processing']} failed={counts['failed']})"
        )
    print()


def print_worker_health(shared_path: Path):
    rows = get_worker_heartbeats(shared_path)
    print(f"{BOLD}Worker Heartbeats:{RESET}")
    if not rows:
        print(f"{DIM}  (none){RESET}\n")
        return

    stale = 0
    for row in rows:
        age = row.get("age_s")
        if isinstance(age, int) and age > 120:
            stale += 1
        age_s = f"{age}s" if isinstance(age, int) else "?"
        state = row.get("state", "?")
        typ = row.get("worker_type", "?")
        host = row.get("host", "-")
        temp = row.get("temp_c")
        cpu_temp = row.get("cpu_temp_c")
        temp_str = f"gpu={temp}C cpu={cpu_temp}C" if temp is not None else "-"
        print(
            f"  {row.get('name','?'):14} type={typ:3} state={state:7} "
            f"hb_age={age_s:>6} host={host:14} {temp_str}"
        )

    print(f"  summary: total={len(rows)} stale>120s={stale}\n")


def print_queue_section(title: str, tasks: list, color: str, show_assigned: bool = True, show_result: bool = False):
    print(f"{color}{BOLD}┌─ {title} ({len(tasks)}) {'─' * (50 - len(title))}┐{RESET}")

    if not tasks:
        print(f"{DIM}  (empty){RESET}")
    else:
        for task in tasks[:10]:  # Show max 10
            task_id = task.get("task_id", "?")[:8]
            task_type = task.get("type", "?")[:10]
            prompt = truncate(task.get("prompt", ""), 35)
            assigned = task.get("assigned_to", "-")

            if show_assigned and assigned and assigned != "-":
                print(f"  {BOLD}{task_id}{RESET} {task_type:10} → {YELLOW}{assigned}{RESET}")
            else:
                print(f"  {BOLD}{task_id}{RESET} {task_type:10}")

            print(f"  {DIM}{prompt}{RESET}")

            if show_result and task.get("result"):
                result = task["result"]
                if result.get("success"):
                    output = truncate(result.get("output", "")[:100].replace("\n", " "), 60)
                    print(f"  {GREEN}✓ {output}{RESET}")
                else:
                    error = truncate(result.get("error", "unknown"), 60)
                    print(f"  {RED}✗ {error}{RESET}")

            print()

        if len(tasks) > 10:
            print(f"  {DIM}... and {len(tasks) - 10} more{RESET}")

    print()


def print_dashboard(config: dict, tasks: dict):
    clear_screen()
    print_header(config)
    shared_path = Path(config.get("__resolved_shared_path", "."))
    print_progress_summary(tasks, shared_path)
    print_worker_health(shared_path)

    # Processing (active work)
    print_queue_section("PROCESSING", tasks["processing"], YELLOW, show_assigned=True)

    # Queue (waiting)
    print_queue_section("QUEUED", tasks["queue"], BLUE, show_assigned=True)

    # Recent completions
    print_queue_section("RECENT COMPLETIONS", tasks["complete"][:5], GREEN, show_assigned=True, show_result=True)

    # Recent failures
    if tasks["failed"]:
        print_queue_section("RECENT FAILURES", tasks["failed"][:3], RED, show_assigned=True, show_result=True)

    # Footer
    print(f"{DIM}Last updated: {datetime.now().strftime('%H:%M:%S')} | Ctrl+C to exit{RESET}")


def tail_completions(config: dict, shared_path: Path):
    """Follow completed tasks and print their outputs."""
    seen = set()
    complete_path = shared_path / "tasks" / "complete"

    # Get initial set
    for f in complete_path.glob("*.json"):
        seen.add(f.name)

    print(f"{BOLD}Watching for completed tasks... (Ctrl+C to stop){RESET}\n")

    while True:
        for task_file in complete_path.glob("*.json"):
            if task_file.name not in seen:
                seen.add(task_file.name)
                try:
                    with open(task_file) as f:
                        task = json.load(f)

                    task_id = task.get("task_id", "?")[:8]
                    worker = task.get("assigned_to", "?")
                    task_type = task.get("type", "?")
                    result = task.get("result", {})

                    print(f"{CYAN}{'═' * 70}{RESET}")
                    print(f"{BOLD}Task:{RESET} {task_id} | {BOLD}Type:{RESET} {task_type} | {BOLD}Worker:{RESET} {worker}")
                    print(f"{BOLD}Prompt:{RESET} {task.get('prompt', '')[:100]}")
                    print()

                    if result.get("success"):
                        print(f"{GREEN}{BOLD}Output:{RESET}")
                        print(result.get("output", "(no output)"))
                    else:
                        print(f"{RED}{BOLD}Error:{RESET} {result.get('error', 'unknown')}")

                    print()

                except Exception as e:
                    print(f"{RED}Error reading {task_file}: {e}{RESET}")

        time.sleep(1)


def tail_brain_decisions(shared_path: Path):
    """Follow brain decision log."""
    log_file = shared_path / "logs" / "brain_decisions.log"

    if not log_file.exists():
        print(f"{YELLOW}No brain decisions yet. Waiting...{RESET}\n")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.touch()

    print(f"{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}║                    BRAIN DECISION LOG                            ║{RESET}")
    print(f"{BOLD}{CYAN}╚══════════════════════════════════════════════════════════════════╝{RESET}")
    print(f"{DIM}Watching {log_file}... (Ctrl+C to stop){RESET}\n")

    # Start from end of file
    with open(log_file) as f:
        f.seek(0, 2)  # Go to end

        while True:
            line = f.readline()
            if line:
                try:
                    entry = json.loads(line.strip())
                    ts = entry.get("timestamp", "")[-8:]  # Just time portion
                    dtype = entry.get("type", "?")
                    msg = entry.get("message", "")
                    details = entry.get("details", {})

                    # Color by type
                    if "ASSIGN" in dtype:
                        color = YELLOW
                    elif "COMPLETE" in dtype or "SUCCESS" in dtype:
                        color = GREEN
                    elif "FAIL" in dtype or "ERROR" in dtype:
                        color = RED
                    elif "DECOMPOSE" in dtype:
                        color = CYAN
                    else:
                        color = RESET

                    print(f"{DIM}{ts}{RESET} {color}{BOLD}[{dtype}]{RESET} {msg}")

                    if details:
                        detail_str = " | ".join(f"{k}={v}" for k, v in details.items() if v)
                        print(f"         {DIM}{detail_str}{RESET}")

                except json.JSONDecodeError:
                    print(line.strip())
            else:
                time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(description="Agent system monitor")
    default_config = str(Path(__file__).resolve().parent.parent / "shared" / "agents" / "config.json")
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--once", action="store_true", help="Show once and exit")
    parser.add_argument("--tail", action="store_true", help="Follow completed task outputs")
    parser.add_argument("--brain", action="store_true", help="Follow brain decision log")
    parser.add_argument("--interval", type=float, default=2.0, help="Refresh interval in seconds")
    args = parser.parse_args()

    config = load_config(args.config)
    shared_path = resolve_shared_path(args.config, config)
    config["__resolved_shared_path"] = str(shared_path)

    if args.brain:
        try:
            tail_brain_decisions(shared_path)
        except KeyboardInterrupt:
            print("\nStopped.")
        return

    if args.tail:
        try:
            tail_completions(config, shared_path)
        except KeyboardInterrupt:
            print("\nStopped.")
        return

    try:
        while True:
            tasks = get_tasks(shared_path)
            print_dashboard(config, tasks)

            if args.once:
                break

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
