"""Shared hard-reset helper for a single GPU worker runtime.

Used by both the dashboard control plane and brain-side retry recovery.

This helper cleans the runtime state for one worker. It should not take over
worker supervision when the orchestrator launcher is active; `startup.py`
remains the single owner of worker process lifecycle.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _default_agent_python() -> str:
    candidates = [
        "/home/bryan/llm-orchestration-venv/bin/python",
        "python3",
    ]
    for candidate in candidates:
        if candidate == "python3" or Path(candidate).exists():
            return candidate
    return "python3"


def _iter_pids_for_port(port: int) -> List[int]:
    try:
        proc = subprocess.run(
            ["ss", "-ltnp"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return []

    pids: set[int] = set()
    token = f":{port} "
    for line in proc.stdout.splitlines():
        if token not in line:
            continue
        marker = "pid="
        idx = line.find(marker)
        if idx == -1:
            continue
        idx += len(marker)
        digits = []
        while idx < len(line) and line[idx].isdigit():
            digits.append(line[idx])
            idx += 1
        if digits:
            try:
                pids.add(int("".join(digits)))
            except ValueError:
                pass
    return sorted(pids)


def _kill_hard(pid: int) -> bool:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    except Exception:
        pass

    time.sleep(0.2)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except Exception:
        pass

    try:
        os.kill(pid, signal.SIGKILL)
        return True
    except ProcessLookupError:
        return True
    except Exception:
        return False


def _remove_if_exists(path: Path) -> bool:
    try:
        if path.exists():
            path.unlink()
            return True
    except Exception:
        return False
    return False


def _write_reset_marker(
    signals_dir: Path,
    gpu_name: str,
    worker_port: int,
    split_ports: list[int],
    split_ids: list[str],
) -> None:
    marker_dir = signals_dir / "runtime_reset"
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker_path = marker_dir / f"{gpu_name}.json"
    payload = {
        "gpu": gpu_name,
        "worker_port": worker_port,
        "split_ports": split_ports,
        "split_ids": split_ids,
        "requested_at": datetime.now().isoformat(),
        "requested_by": "runtime_reset",
    }
    tmp_path = marker_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, marker_path)


def hard_reset_gpu_runtime(
    gpu_name: str,
    worker_port: int,
    split_ports: Iterable[int] | None = None,
    split_ids: Iterable[str] | None = None,
    config_path: str = "/mnt/shared/agents/config.json",
    signals_path: str = "/mnt/shared/signals",
    agent_python: str = _default_agent_python(),
    gpu_agent_path: str = "/mnt/shared/agents/gpu.py",
    clear_global_load_owner: bool = True,
    spawn_replacement: bool = False,
) -> Dict[str, Any]:
    split_ports = sorted({int(p) for p in (split_ports or []) if str(p).strip()})
    split_ids = sorted({str(s).strip() for s in (split_ids or []) if str(s).strip()})
    signals_dir = Path(signals_path)

    _write_reset_marker(signals_dir, gpu_name, worker_port, split_ports, split_ids)

    killed_gpu_proc = 0
    killed_port_listeners = 0
    removed_split_files = 0
    removed_split_locks = 0
    cleared_load_owner = False

    try:
        pkill = subprocess.run(
            ["pkill", "-9", "-f", f"{gpu_agent_path} {gpu_name}"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if pkill.returncode == 0:
            killed_gpu_proc = 1
    except Exception:
        pass

    for port in [worker_port, *split_ports]:
        for pid in _iter_pids_for_port(int(port)):
            if _kill_hard(pid):
                killed_port_listeners += 1

    split_dir = signals_dir / "split_llm"
    for group_id in split_ids:
        if _remove_if_exists(split_dir / f"{group_id}.json"):
            removed_split_files += 1
        if _remove_if_exists(split_dir / f"{group_id}.runtime_owner.json"):
            removed_split_files += 1
        if _remove_if_exists(split_dir / f"{group_id}.json.lock"):
            removed_split_locks += 1

    owner_path = signals_dir / "model_load.global.json"
    if clear_global_load_owner:
        try:
            if owner_path.exists():
                with open(owner_path, "r", encoding="utf-8") as f:
                    owner = json.load(f)
                if str(owner.get("worker", "")).strip() == gpu_name:
                    owner_path.unlink(missing_ok=True)
                    cleared_load_owner = True
        except Exception:
            pass

    proc = None
    stdout_path = Path(f"/tmp/{gpu_name}.log")
    gpu_agent_running = False
    if spawn_replacement:
        with open(stdout_path, "ab") as out:
            proc = subprocess.Popen(
                [agent_python, gpu_agent_path, gpu_name, "--config", config_path],
                stdout=out,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )

        time.sleep(2)
        try:
            os.kill(proc.pid, 0)
            gpu_agent_running = True
        except Exception:
            gpu_agent_running = False
    else:
        gpu_agent_running = True

    return {
        "ok": gpu_agent_running,
        "gpu": gpu_name,
        "worker_port": worker_port,
        "split_ports": split_ports,
        "split_ids": split_ids,
        "killed_gpu_proc": killed_gpu_proc,
        "killed_port_listeners": killed_port_listeners,
        "removed_split_files": removed_split_files,
        "removed_split_locks": removed_split_locks,
        "cleared_load_owner": cleared_load_owner,
        "agent_pid": proc.pid if proc is not None else None,
        "spawn_replacement": spawn_replacement,
        "stdout_log": str(stdout_path),
        "config_path": config_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hard reset a single GPU runtime")
    parser.add_argument("--gpu-name", required=True)
    parser.add_argument("--worker-port", required=True, type=int)
    parser.add_argument("--split-port", action="append", default=[], type=int)
    parser.add_argument("--split-id", action="append", default=[])
    parser.add_argument("--config-path", default="/mnt/shared/agents/config.json")
    parser.add_argument("--signals-path", default="/mnt/shared/signals")
    parser.add_argument("--agent-python", default=_default_agent_python())
    parser.add_argument("--gpu-agent-path", default="/mnt/shared/agents/gpu.py")
    parser.add_argument("--spawn", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = hard_reset_gpu_runtime(
        gpu_name=args.gpu_name,
        worker_port=args.worker_port,
        split_ports=args.split_port,
        split_ids=args.split_id,
        config_path=args.config_path,
        signals_path=args.signals_path,
        agent_python=args.agent_python,
        gpu_agent_path=args.gpu_agent_path,
        spawn_replacement=args.spawn,
    )
    print(json.dumps(result))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
