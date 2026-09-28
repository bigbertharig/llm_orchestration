#!/usr/bin/env python3
"""Targeted restart for selected GPU agents (without restarting brain)."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_shared_path(config_path: Path, config: dict) -> Path:
    shared = Path(config.get("shared_path", "../"))
    if shared.is_absolute():
        return shared
    return (config_path.resolve().parent / shared).resolve()


def parse_gpu_list(gpus_arg: str | None, config: dict) -> list[str]:
    configured = [g.get("name") for g in config.get("gpus", []) if g.get("name")]
    if not gpus_arg:
        return configured
    requested = [x.strip() for x in gpus_arg.split(",") if x.strip()]
    unknown = [x for x in requested if x not in configured]
    if unknown:
        raise ValueError(f"Unknown gpu names: {unknown}. Configured: {configured}")
    return requested


def find_agent_pids(gpu_name: str) -> list[int]:
    """Find PIDs for `gpu.py <gpu_name>` processes."""
    result = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        capture_output=True,
        text=True,
        check=False,
    )
    pids: list[int] = []
    needle = f"agents/gpu.py {gpu_name} "
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid_str, args = line.split(" ", 1)
            pid = int(pid_str)
        except Exception:
            continue
        if needle in args or args.endswith(f"agents/gpu.py {gpu_name}"):
            pids.append(pid)
    return pids


def stop_pids(pids: list[int], timeout: int = 8, dry_run: bool = False) -> None:
    if not pids:
        return
    if dry_run:
        print(f"  DRY RUN: would SIGTERM pids {pids}")
        return

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.time() + timeout
    remaining = set(pids)
    while remaining and time.time() < deadline:
        done = set()
        for pid in remaining:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                done.add(pid)
        remaining -= done
        if remaining:
            time.sleep(0.2)

    for pid in list(remaining):
        try:
            os.kill(pid, signal.SIGKILL)
            print(f"  force-killed pid {pid}")
        except ProcessLookupError:
            pass


def remove_stale_heartbeat(shared_path: Path, gpu_name: str, dry_run: bool = False) -> None:
    try:
        gpu_id = int(gpu_name.split("-")[-1])
    except ValueError:
        return
    hb = shared_path / "gpus" / f"gpu_{gpu_id}" / "heartbeat.json"
    if hb.exists():
        if dry_run:
            print(f"  DRY RUN: would remove {hb}")
            return
        try:
            hb.unlink()
            print(f"  removed stale heartbeat {hb}")
        except OSError as e:
            print(f"  warning: could not remove {hb}: {e}")


def start_gpu_agent(python_bin: str, config_path: Path, shared_path: Path, gpu_name: str, dry_run: bool = False) -> None:
    logs_dir = shared_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{gpu_name}-manual.log"

    cmd = [python_bin, "/mnt/shared/agents/gpu.py", gpu_name, "--config", str(config_path)]
    if dry_run:
        print(f"  DRY RUN: would start {' '.join(cmd)} > {log_path}")
        return

    with open(log_path, "ab") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(f"  started {gpu_name} pid={proc.pid} log={log_path}")


def print_status(gpu_names: list[str]) -> None:
    print("GPU agent process status:")
    for gpu_name in gpu_names:
        pids = find_agent_pids(gpu_name)
        if pids:
            print(f"  {gpu_name}: running pids={pids}")
        else:
            print(f"  {gpu_name}: not running")


def main() -> int:
    parser = argparse.ArgumentParser(description="Targeted GPU agent restart")
    parser.add_argument("--config", default="/mnt/shared/agents/config.json")
    parser.add_argument("--gpus", help="Comma-separated gpu names (e.g., gpu-1,gpu-3)")
    parser.add_argument("--status", action="store_true", help="Only print status")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    shared_path = resolve_shared_path(config_path, config)
    gpu_names = parse_gpu_list(args.gpus, config)

    if args.status:
        print_status(gpu_names)
        return 0

    candidates = [
        "/home/bryan/llm-orchestration-venv/bin/python",
    ]
    python_bin = next((candidate for candidate in candidates if Path(candidate).exists()), sys.executable)

    print(f"Restarting GPU agents: {gpu_names}")
    for gpu_name in gpu_names:
        print(f"- {gpu_name}")
        pids = find_agent_pids(gpu_name)
        if pids:
            print(f"  stopping existing pids {pids}")
        else:
            print("  no existing process")
        stop_pids(pids, dry_run=args.dry_run)
        remove_stale_heartbeat(shared_path, gpu_name, dry_run=args.dry_run)
        start_gpu_agent(python_bin, config_path, shared_path, gpu_name, dry_run=args.dry_run)

    if not args.dry_run:
        time.sleep(1.0)
        print_status(gpu_names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
