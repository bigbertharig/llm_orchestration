#!/usr/bin/env python3
"""Prepare llama-server runtimes deterministically for any workflow.

Loads models onto GPU workers by inserting load_llm meta tasks into the
orchestrator queue. Workers claim and execute them one at a time.

Reusable across benchmarks, plans, and ad-hoc ops.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
import uuid
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_MODELS_CATALOG = "/mnt/shared/plans/shoulders/benchmarking/models.catalog.json"
DEFAULT_LOAD_TIMEOUT_SECONDS = 300
RUNTIME_META_COMMANDS = {
    "load_llm",
    "unload_llm",
    "load_split_llm",
    "unload_split_llm",
    "reset_split_runtime",
}


def run_cmd(cmd: List[str], env: Dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, check=False, env=env)


def get_json(url: str, timeout: int = 10) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode("utf-8", errors="replace")
        return json.loads(text) if text.strip() else {}


def probe_llama_endpoint(port: int) -> Optional[str]:
    """Probe a llama-server endpoint. Returns loaded model ID or None."""
    try:
        data = get_json(f"http://127.0.0.1:{port}/v1/models", timeout=5)
        models = data.get("data", [])
        if models:
            return str(models[0].get("id", "")).strip() or None
    except Exception:
        pass
    return None


def scan_worker_ports(config: dict) -> List[Tuple[str, int, Optional[str]]]:
    """Scan all configured worker ports, return (name, port, loaded_model)."""
    results = []
    for gpu in config.get("gpus", []):
        name = gpu.get("name", "")
        port = gpu.get("port")
        if not port:
            continue
        loaded = probe_llama_endpoint(port)
        results.append((name, port, loaded))
    return results


def print_scan(label: str, scan: List[Tuple[str, int, Optional[str]]]) -> None:
    print(f"== {label} ==")
    for name, port, loaded in scan:
        status = loaded if loaded else "cold"
        print(f"  {name} port={port} model={status}")


def restart_orchestrator_agents() -> None:
    """Restart brain + gpu agents using canonical startup script."""
    clear_stale_launch_lock()
    _ = run_cmd(["pkill", "-f", "brain.py|gpu.py"])
    time.sleep(2)
    proc = run_cmd(["python3", "/mnt/shared/agents/startup.py"])
    if proc.returncode != 0 and "launch.lock" in (proc.stderr + proc.stdout):
        clear_stale_launch_lock(force=True)
        proc = run_cmd(["python3", "/mnt/shared/agents/startup.py"])
    if proc.returncode != 0:
        raise RuntimeError(f"startup.py failed: {proc.stderr.strip() or proc.stdout.strip()}")


def clear_orphan_queue_locks(queue_dir: Path) -> int:
    removed = 0
    for lock in sorted(queue_dir.glob("*.json.lock")):
        base = Path(str(lock)[:-5])
        if not base.exists():
            lock.unlink(missing_ok=True)
            removed += 1
    return removed


def _read_json_file(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_iso_epoch(raw: str) -> Optional[float]:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return dt.datetime.fromisoformat(text).timestamp()
    except Exception:
        return None


def _task_heartbeat_age_seconds(task_path: Path) -> Optional[int]:
    hb_path = task_path.with_name(task_path.stem + ".heartbeat.json")
    if not hb_path.exists():
        return None
    try:
        hb = _read_json_file(hb_path)
    except Exception:
        return None
    updated_at = _parse_iso_epoch(str(hb.get("updated_at") or ""))
    if updated_at is None:
        try:
            updated_at = hb_path.stat().st_mtime
        except Exception:
            return None
    return max(0, int(time.time() - updated_at))


def collect_stale_runtime_meta_tasks(
    shared_root: Path,
    *,
    stale_seconds: int,
) -> list[dict]:
    processing_dir = shared_root / "tasks" / "processing"
    stale: list[dict] = []
    for task_path in sorted(processing_dir.glob("*.json")):
        if task_path.name.endswith(".heartbeat.json"):
            continue
        try:
            task = _read_json_file(task_path)
        except Exception:
            continue
        command = str(task.get("command") or "").strip()
        created_by = str(task.get("created_by") or "").strip()
        if command not in RUNTIME_META_COMMANDS:
            continue
        if created_by not in {"runtime-prep", "brain", "system", ""}:
            continue
        age = _task_heartbeat_age_seconds(task_path)
        if age is None:
            age = max(0, int(time.time() - task_path.stat().st_mtime))
        if age < stale_seconds:
            continue
        stale.append(
            {
                "task_path": task_path,
                "task": task,
                "heartbeat_age_seconds": age,
            }
        )
    return stale


def cleanup_stale_runtime_meta_tasks(
    shared_root: Path,
    *,
    stale_seconds: int,
) -> int:
    failed_dir = shared_root / "tasks" / "failed"
    failed_dir.mkdir(parents=True, exist_ok=True)
    cleaned = 0
    for entry in collect_stale_runtime_meta_tasks(shared_root, stale_seconds=stale_seconds):
        task_path: Path = entry["task_path"]
        task: dict = entry["task"]
        hb_age = int(entry["heartbeat_age_seconds"])
        task["status"] = "failed"
        task["failed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        task["failure_reason"] = f"stale_runtime_meta_preflight_cleanup:{hb_age}s"
        dest = failed_dir / task_path.name
        dest.write_text(json.dumps(task, indent=2), encoding="utf-8")
        hb_path = task_path.with_name(task_path.stem + ".heartbeat.json")
        task_path.unlink(missing_ok=True)
        hb_path.unlink(missing_ok=True)
        cleaned += 1
    return cleaned


def list_test_runtime_containers() -> list[str]:
    proc = run_cmd(["docker", "ps", "-a", "--format", "{{.Names}}"])
    if proc.returncode != 0:
        return []
    names = []
    for raw in (proc.stdout or "").splitlines():
        name = raw.strip()
        if not name:
            continue
        if name.startswith("test-"):
            names.append(name)
    return names


def cleanup_test_runtime_containers() -> int:
    removed = 0
    for name in list_test_runtime_containers():
        proc = run_cmd(["docker", "rm", "-f", name])
        if proc.returncode == 0:
            removed += 1
    return removed


def clear_stale_launch_lock(lock_path: str = "/tmp/llm-orchestration-flags/launch.lock", force: bool = False) -> bool:
    p = Path(lock_path)
    if not p.exists():
        return False
    proc = run_cmd(["pgrep", "-af", "startup.py"])
    startup_running = proc.returncode == 0 and bool(proc.stdout.strip())
    if startup_running and not force:
        return False
    p.unlink(missing_ok=True)
    return True


def assert_scheduler_clean(shared_root: Path, strict_processing_empty: bool) -> None:
    tasks = shared_root / "tasks"
    queue_dir = tasks / "queue"
    processing_dir = tasks / "processing"
    queue_json = sorted(f for f in queue_dir.glob("*.json") if not f.name.endswith(".heartbeat.json"))
    processing_json = sorted(f for f in processing_dir.glob("*.json") if not f.name.endswith(".heartbeat.json"))
    if queue_json:
        raise RuntimeError(
            f"queue not empty ({len(queue_json)} pending). "
            "Clear queue before isolated runtime prep."
        )
    if strict_processing_empty and processing_json:
        raise RuntimeError(
            f"processing not empty ({len(processing_json)} active). "
            "Wait for/stop active tasks before isolated runtime prep."
        )


def write_meta_task(shared_root: Path, task: dict) -> Path:
    queue_dir = shared_root / "tasks" / "queue"
    out = queue_dir / f"{task['task_id']}.json"
    out.write_text(json.dumps(task, indent=2), encoding="utf-8")
    return out


def build_meta_task(command: str, target_model: str, candidate_groups: List[dict] | None = None) -> dict:
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    task = {
        "task_id": str(uuid.uuid4()),
        "type": "meta",
        "command": command,
        "batch_id": "system",
        "name": command,
        "priority": 10,
        "task_class": "meta",
        "depends_on": [],
        "executor": "worker",
        "status": "pending",
        "created_at": now,
        "created_by": "runtime-prep",
        "retry_count": 0,
        "target_model": target_model,
    }
    if candidate_groups:
        task["candidate_groups"] = candidate_groups
    return task


def build_meta_task_with_batch(
    command: str,
    target_model: str,
    batch_id: str,
    candidate_groups: List[dict] | None = None,
    target_worker: str | None = None,
    group_id: str | None = None,
) -> dict:
    task = build_meta_task(command=command, target_model=target_model, candidate_groups=candidate_groups)
    task["batch_id"] = batch_id
    if target_worker:
        task["target_worker"] = target_worker
    if group_id:
        task["group_id"] = group_id
    return task


def load_split_group(model: str, group_id: str, catalog_path: str = DEFAULT_MODELS_CATALOG) -> dict:
    catalog = Path(catalog_path)
    if not catalog.exists():
        raise RuntimeError(f"models catalog missing: {catalog}")
    data = json.loads(catalog.read_text(encoding="utf-8"))
    for item in data.get("models", []):
        if str(item.get("id", "")).strip() != model:
            continue
        for group in item.get("split_groups", []):
            if str(group.get("id", "")).strip() != group_id:
                continue
            return {
                "id": str(group.get("id", "")).strip(),
                "members": [str(member).strip() for member in group.get("members", []) if str(member).strip()],
                "port": int(group.get("port")),
            }
        raise RuntimeError(f"split group {group_id} not defined for model {model}")
    raise RuntimeError(f"model {model} not found in models catalog")


def wait_task_terminal(shared_root: Path, task_id: str, timeout_s: int) -> str:
    tasks = shared_root / "tasks"
    deadline = time.time() + max(1, timeout_s)
    while time.time() < deadline:
        if (tasks / "complete" / f"{task_id}.json").exists():
            return "complete"
        if (tasks / "failed" / f"{task_id}.json").exists():
            return "failed"
        time.sleep(1.0)
    return "timeout"


def run_split_recovery(
    shared_root: Path,
    split_model: str,
    candidate_group: str,
    task_timeout_s: int,
    batch_id: str,
) -> None:
    group = load_split_group(split_model, candidate_group)
    sequence = []
    for member in group["members"]:
        sequence.append({
            "command": "reset_split_runtime",
            "target_worker": member,
            "group_id": group["id"],
        })
    sequence.append({
        "command": "load_split_llm",
        "candidate_groups": [group],
    })
    for item in sequence:
        cmd = item["command"]
        task = build_meta_task_with_batch(
            command=cmd,
            target_model=split_model,
            batch_id=batch_id,
            candidate_groups=item.get("candidate_groups"),
            target_worker=item.get("target_worker"),
            group_id=item.get("group_id"),
        )
        path = write_meta_task(shared_root, task)
        print(f"[split-recovery] queued {cmd} task={task['task_id']} file={path}")
        state = wait_task_terminal(shared_root, task["task_id"], timeout_s=task_timeout_s)
        if state != "complete":
            raise RuntimeError(
                f"split recovery step '{cmd}' did not complete (state={state}). "
                f"task_id={task['task_id']}"
            )


def load_model_via_meta_task(
    shared_root: Path,
    model: str,
    batch_id: str,
    timeout_s: int,
) -> str:
    """Queue a load_llm meta task and wait for completion. Returns terminal state."""
    task = build_meta_task_with_batch(
        command="load_llm",
        target_model=model,
        batch_id=batch_id,
    )
    task["load_mode"] = "single"
    path = write_meta_task(shared_root, task)
    print(f"  queued load_llm task={task['task_id'][:8]} -> {path.name}")
    state = wait_task_terminal(shared_root, task["task_id"], timeout_s=timeout_s)
    return state


def unload_all_workers(
    shared_root: Path,
    config: dict,
    batch_id: str,
    timeout_s: int,
) -> None:
    """Issue unload_llm meta tasks for all currently loaded workers."""
    scan = scan_worker_ports(config)
    loaded = [(name, port, model) for name, port, model in scan if model]
    if not loaded:
        print("  no workers currently loaded, nothing to unload")
        return
    for name, port, model in loaded:
        task = build_meta_task_with_batch(
            command="unload_llm",
            target_model=model,
            batch_id=batch_id,
        )
        task["target_worker"] = name
        path = write_meta_task(shared_root, task)
        print(f"  queued unload_llm {name} ({model}) task={task['task_id'][:8]}")
        state = wait_task_terminal(shared_root, task["task_id"], timeout_s=timeout_s)
        if state != "complete":
            print(f"  WARNING: unload {name} ended with state={state}")


def load_config(shared_root: Path, config_name: str) -> dict:
    """Load an orchestrator config file."""
    config_path = shared_root / "agents" / config_name
    if not config_path.exists():
        raise RuntimeError(f"config not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Prepare llama-server runtimes by issuing load_llm meta tasks.",
    )
    ap.add_argument("--shared-root", default="/mnt/shared")
    ap.add_argument("--config", default="config.benchmark.json",
                     help="Orchestrator config file name (relative to shared/agents/)")
    ap.add_argument("--clear-orphan-queue-locks", action="store_true")
    ap.add_argument("--strict-processing-empty", action="store_true")
    ap.add_argument(
        "--runtime-meta-stale-seconds",
        type=int,
        default=180,
        help="Age threshold for treating processing runtime meta tasks as stale.",
    )
    ap.add_argument(
        "--cleanup-stale-runtime-meta",
        action="store_true",
        help="Move stale runtime meta tasks from processing/ to failed/ before load prep.",
    )
    ap.add_argument(
        "--cleanup-test-runtime-containers",
        action="store_true",
        help="Remove unmanaged ad-hoc test containers with names matching test-* before load prep.",
    )
    ap.add_argument("--load-timeout-seconds", type=int, default=DEFAULT_LOAD_TIMEOUT_SECONDS)
    ap.add_argument(
        "--meta-batch-id",
        default="",
        help=(
            "Batch id for meta tasks. Default is an isolated runtime-prep id; "
            "avoid using 'system' to prevent cross-interference with auto-default flows."
        ),
    )
    ap.add_argument(
        "--force-unload-first",
        action="store_true",
        help="Unload all currently loaded workers before loading new models.",
    )
    ap.add_argument(
        "--restart-agents-first",
        action="store_true",
        help="Restart brain+gpu agents via startup.py before loading.",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model IDs to load (from models.catalog.json). One per available cold worker.",
    )
    ap.add_argument(
        "--split-model",
        default="",
        help="Split model to load after singles (optional).",
    )
    ap.add_argument("--split-candidate-group", default="pair_4_5")
    ap.add_argument("--split-recovery-timeout-seconds", type=int, default=180)
    ap.add_argument(
        "--disable-split-recovery",
        action="store_true",
        help="Disable auto split runtime recovery sequence on split load failure.",
    )
    return ap


def main() -> int:
    args = build_parser().parse_args()
    shared_root = Path(args.shared_root).expanduser().resolve()
    queue_dir = shared_root / "tasks" / "queue"
    config = load_config(shared_root, args.config)

    meta_batch_id = (
        args.meta_batch_id.strip()
        or f"runtime_prep_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )

    # Housekeeping
    if args.clear_orphan_queue_locks:
        removed = clear_orphan_queue_locks(queue_dir)
        print(f"removed_orphan_queue_locks={removed}")
    stale_runtime_meta = collect_stale_runtime_meta_tasks(
        shared_root,
        stale_seconds=max(30, int(args.runtime_meta_stale_seconds)),
    )
    print(f"stale_runtime_meta={len(stale_runtime_meta)}")
    if stale_runtime_meta:
        for entry in stale_runtime_meta[:5]:
            task = entry["task"]
            print(
                "  stale_runtime_meta_task="
                f"{str(task.get('task_id', ''))[:8]} "
                f"command={task.get('command', '')} "
                f"age_s={entry['heartbeat_age_seconds']}"
            )
        if args.cleanup_stale_runtime_meta:
            cleaned = cleanup_stale_runtime_meta_tasks(
                shared_root,
                stale_seconds=max(30, int(args.runtime_meta_stale_seconds)),
            )
            print(f"cleaned_stale_runtime_meta={cleaned}")
    test_containers = list_test_runtime_containers()
    print(f"unmanaged_test_runtime_containers={len(test_containers)}")
    if test_containers:
        for name in test_containers[:10]:
            print(f"  unmanaged_test_container={name}")
        if args.cleanup_test_runtime_containers:
            removed = cleanup_test_runtime_containers()
            print(f"removed_test_runtime_containers={removed}")
    launch_lock_removed = clear_stale_launch_lock()
    if launch_lock_removed:
        print("removed_stale_launch_lock=1")

    assert_scheduler_clean(shared_root, strict_processing_empty=args.strict_processing_empty)

    # Pre-scan
    scan = scan_worker_ports(config)
    print_scan("pre-scan", scan)

    # Restart agents if requested
    if args.restart_agents_first:
        print("restarting brain+gpu agents via startup.py")
        restart_orchestrator_agents()
        time.sleep(3)
        scan = scan_worker_ports(config)
        print_scan("post-restart", scan)

    # Unload if requested
    if args.force_unload_first:
        print("[unload] unloading all workers")
        unload_all_workers(shared_root, config, meta_batch_id, args.load_timeout_seconds)
        scan = scan_worker_ports(config)
        print_scan("post-unload", scan)

    # Load models sequentially via meta tasks
    results = {}
    for i, model in enumerate(args.models):
        print(f"[{i+1}/{len(args.models)}] loading {model}")
        state = load_model_via_meta_task(
            shared_root=shared_root,
            model=model,
            batch_id=meta_batch_id,
            timeout_s=args.load_timeout_seconds,
        )
        results[model] = state
        if state == "complete":
            print(f"  -> {model} OK")
        else:
            print(f"  -> {model} FAILED (state={state})")

    # Load split model if requested
    if args.split_model:
        split_model = args.split_model
        print(f"[split] loading {split_model} on {args.split_candidate_group}")
        group = load_split_group(split_model, args.split_candidate_group)
        task = build_meta_task_with_batch(
            command="load_split_llm",
            target_model=split_model,
            batch_id=meta_batch_id,
            candidate_groups=[group],
        )
        write_meta_task(shared_root, task)
        state = wait_task_terminal(shared_root, task["task_id"], timeout_s=args.split_recovery_timeout_seconds)
        if state == "complete":
            print(f"  -> {split_model} split OK")
            results[f"{split_model}(split)"] = state
        elif not args.disable_split_recovery:
            print(f"  -> split load failed (state={state}), running recovery")
            run_split_recovery(
                shared_root=shared_root,
                split_model=split_model,
                candidate_group=args.split_candidate_group,
                task_timeout_s=args.split_recovery_timeout_seconds,
                batch_id=meta_batch_id,
            )
            results[f"{split_model}(split)"] = "complete(recovered)"
        else:
            print(f"  -> {split_model} split FAILED (state={state})")
            results[f"{split_model}(split)"] = state

    # Final scan
    scan = scan_worker_ports(config)
    print_scan("final-scan", scan)

    # Summary
    print("\n== summary ==")
    all_ok = True
    for model, state in results.items():
        ok = "complete" in state
        print(f"  {model}: {state}")
        if not ok:
            all_ok = False

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
