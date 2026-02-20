#!/usr/bin/env python3
"""
CPU agent for Pi workers.

Claims only worker-owned CPU tasks from queue, executes shell commands, and
writes results to complete/failed task directories.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTS_DIR_CANDIDATES = [
    REPO_ROOT / "shared" / "agents",
    Path("/media/bryan/shared/agents"),
]
for candidate in AGENTS_DIR_CANDIDATES:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from executor import PermissionExecutor  # noqa: E402
from hardware import scan_cpu_temps  # noqa: E402


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


class CpuAgent:
    def __init__(self, config_path: Path, agent_name: Optional[str], once: bool = False):
        self.config_path = config_path
        self.config = _load_json(config_path)
        self.once = once
        self.agent_name = agent_name or f"cpu-{socket.gethostname()}"

        config_dir = config_path.parent
        shared_path = Path(self.config.get("shared_path", "../"))
        if not shared_path.is_absolute():
            shared_path = (config_dir / shared_path).resolve()
        self.shared_path = shared_path

        self.queue_path = self.shared_path / "tasks" / "queue"
        self.processing_path = self.shared_path / "tasks" / "processing"
        self.complete_path = self.shared_path / "tasks" / "complete"
        self.failed_path = self.shared_path / "tasks" / "failed"
        self.cpu_heartbeat_path = self.shared_path / "cpus" / self.agent_name
        self.cpu_heartbeat_file = self.cpu_heartbeat_path / "heartbeat.json"
        self.unified_heartbeat_path = self.shared_path / "heartbeats"
        self.unified_heartbeat_file = self.unified_heartbeat_path / f"{self.agent_name}.json"

        perm_rel = self.config.get("permissions_path", "permissions/")
        permissions_path = Path(perm_rel)
        if not permissions_path.is_absolute():
            permissions_path = (config_dir / permissions_path).resolve()
        self.permissions_file = permissions_path / "worker.json"

        self.poll_interval = int(self.config.get("timeouts", {}).get("poll_interval_seconds", 5))
        if self.poll_interval <= 0:
            self.poll_interval = 5
        self.worker_timeout = int(self.config.get("timeouts", {}).get("worker_task_seconds", 0))
        self.worker_timeout = None if self.worker_timeout <= 0 else self.worker_timeout
        self.task_heartbeat_interval = int(
            self.config.get("timeouts", {}).get("task_heartbeat_interval_seconds", 15)
        )
        if self.task_heartbeat_interval <= 0:
            self.task_heartbeat_interval = 15
        self.resource_limits = self.config.get("resource_limits", {})
        cpu_agent_limits = self.config.get("cpu_agent", {}).get("resource_limits", {})
        effective_cpu_limits = dict(self.resource_limits)
        effective_cpu_limits.update(cpu_agent_limits)
        self.cpu_temp_warning_c = int(effective_cpu_limits.get("cpu_temp_warning_c", 80))
        self.cpu_temp_critical_c = int(effective_cpu_limits.get("cpu_temp_critical_c", 95))
        pause_cfg = self.config.get("thermal_pause", {})
        self.thermal_resume_margin_c = int(pause_cfg.get("resume_margin_c", 3))
        self.thermal_pause_active = False
        self.thermal_pause_reason: str | None = None

        # Runtime baseline for command execution.
        self.venv_activate = (
            os.environ.get("CPU_AGENT_VENV_ACTIVATE")
            or self.config.get("cpu_agent", {}).get("venv_activate")
            or self._pick_default_venv_activate()
        )

        self._active_task_id: Optional[str] = None
        self._active_task_heartbeat: Optional[Path] = None
        self._active_task_pid: Optional[int] = None
        self._active_task_started_at: Optional[str] = None

        os.environ.setdefault("LLM_ORCH_LOG_PATH", str(self.shared_path / "logs"))
        os.environ["WORKER_NAME"] = self.agent_name

    def _log(self, message: str) -> None:
        ts = datetime.now().isoformat(timespec="seconds")
        print(f"{ts} [{self.agent_name}] {message}", flush=True)

    def _pick_default_venv_activate(self) -> str | None:
        candidates = [
            "/home/bryan/ml-env/bin/activate",
            "/opt/worker-env/bin/activate",   # fallback when user env is absent
        ]
        for p in candidates:
            if Path(p).exists():
                return f"source {p}"
        return None

    def _detect_ip_address(self) -> str | None:
        # Prefer the route-address used on the worker LAN over localhost aliases.
        targets = [("10.0.0.2", 2049), ("10.0.0.3", 22), ("10.0.0.1", 53)]
        for host, port in targets:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect((host, port))
                ip = s.getsockname()[0]
                s.close()
                if ip and not ip.startswith("127."):
                    return ip
            except Exception:
                continue
        return None

    def _normalize_command(self, command: str) -> str:
        fixed = command

        # Remove task-authored venv activation segments.
        # CPU agent applies one explicit baseline env for consistency.
        source_pattern = re.compile(r"^source\s+\S+/bin/activate\s*&&\s*")
        while True:
            m = source_pattern.match(fixed)
            if not m:
                break
            fixed = fixed[m.end():]

        # Prefer python3 for portability.
        if "python " in fixed and "python3 " not in fixed:
            fixed = fixed.replace("python ", "python3 ", 1)

        # Give CPU tasks a consistent runtime baseline via configured venv.
        if self.venv_activate and self.venv_activate not in fixed:
            fixed = f"{self.venv_activate} && {fixed}"

        # Add venv activation for known orchestration utility invocations.
        if "generate_batch_summary.py" in fixed and self.venv_activate and self.venv_activate not in fixed:
            fixed = f"{self.venv_activate} && {fixed}"

        # Path translation for tasks authored on GPU rig.
        remaps = [
            ("/mnt/shared", "/media/bryan/shared"),
            ("/mnt/shared", str(self.shared_path)),
        ]
        for src, dst in remaps:
            if src in fixed and not Path(src).exists() and Path(dst).exists():
                fixed = fixed.replace(src, dst)

        return fixed

    def _write_task_heartbeat(self) -> None:
        if not self._active_task_id or not self._active_task_heartbeat:
            return
        hb = {
            "task_id": self._active_task_id,
            "worker_id": self.agent_name,
            "worker": self.agent_name,
            "pid": self._active_task_pid,
            "updated_at": datetime.now().isoformat(),
        }
        _write_json(self._active_task_heartbeat, hb)

    def _write_cpu_heartbeat(self, state: str = "idle") -> None:
        self.cpu_heartbeat_path.mkdir(parents=True, exist_ok=True)
        self.unified_heartbeat_path.mkdir(parents=True, exist_ok=True)
        ip_addr = self._detect_ip_address()
        cpu_temp = self._get_cpu_temp()
        pi_throttled, pi_reason = self._get_pi_throttled_status()
        hb = {
            "worker_type": "cpu",
            "name": self.agent_name,
            "hostname": socket.gethostname(),
            "ip_address": ip_addr,
            "state": state,
            "active_task_id": self._active_task_id,
            "active_task_started_at": self._active_task_started_at,
            "active_task_pid": self._active_task_pid,
            "last_updated": datetime.now().isoformat(),
            "cpu_temp_c": cpu_temp,
            "pi_throttled_now": pi_throttled,
            "pi_throttle_reason": pi_reason,
            "thermal_pause_active": self.thermal_pause_active,
            "thermal_pause_reason": self.thermal_pause_reason,
            "queue_path": str(self.queue_path),
            "processing_path": str(self.processing_path),
            "stats": {
                "queue_count": len(list(self.queue_path.glob("*.json"))),
                "processing_count": len(list(self.processing_path.glob("*.json"))),
            },
        }
        _write_json(self.cpu_heartbeat_file, hb)
        _write_json(self.unified_heartbeat_file, hb)

    def _get_cpu_temp(self) -> int | None:
        try:
            temps = scan_cpu_temps()
        except Exception:
            return None
        values = [int(t.get("temp_c")) for t in temps if isinstance(t.get("temp_c"), (int, float))]
        return max(values) if values else None

    def _get_pi_throttled_status(self) -> tuple[bool, str | None]:
        try:
            out = subprocess.run(
                ["vcgencmd", "get_throttled"],
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (FileNotFoundError, PermissionError, subprocess.SubprocessError):
            return False, None

        if out.returncode != 0:
            return False, None

        m = re.search(r"throttled=(0x[0-9a-fA-F]+)", out.stdout.strip())
        if not m:
            return False, None

        raw_hex = m.group(1)
        try:
            flags = int(raw_hex, 16)
        except ValueError:
            return False, None

        if flags == 0:
            return False, None

        now_flags = {
            0: "under_voltage_now",
            1: "freq_capped_now",
            2: "throttled_now",
            3: "soft_temp_limit_now",
        }
        active_now = [name for bit, name in now_flags.items() if flags & (1 << bit)]
        if active_now:
            return True, f"Pi throttling active: {','.join(active_now)} ({raw_hex})"
        return False, None

    def _thermal_pause_check(self) -> tuple[bool, int | None, str | None]:
        temp = self._get_cpu_temp()
        pi_throttled, pi_reason = self._get_pi_throttled_status()
        if pi_throttled:
            return True, temp, pi_reason
        if temp is None:
            return False, None, None

        if temp >= self.cpu_temp_critical_c:
            return True, temp, f"CPU {temp}C >= critical {self.cpu_temp_critical_c}C"
        if temp >= self.cpu_temp_warning_c:
            return True, temp, f"CPU {temp}C >= warning {self.cpu_temp_warning_c}C"

        resume_limit = max(0, self.cpu_temp_warning_c - self.thermal_resume_margin_c)
        if self.thermal_pause_active and temp >= resume_limit:
            return True, temp, f"CPU {temp}C >= resume_limit {resume_limit}C"
        return False, temp, None

    def _preflight(self) -> None:
        required = [
            self.queue_path,
            self.processing_path,
            self.complete_path,
            self.failed_path,
            self.permissions_file,
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise RuntimeError(f"preflight failed; missing paths: {missing}")

        check = subprocess.run(["python3", "--version"], capture_output=True, text=True)
        if check.returncode != 0:
            raise RuntimeError("preflight failed; python3 not available")

    def _claim_one_task(self) -> Optional[Dict[str, Any]]:
        def _task_priority_key(task: Dict[str, Any]) -> tuple:
            try:
                priority = int(task.get("priority", 5))
            except Exception:
                priority = 5
            created_at = str(task.get("created_at", "") or "")
            task_id = str(task.get("task_id", "") or "")
            return (-priority, created_at, task_id)

        ranked_files: list[Path] = []
        staged: list[tuple[Path, Dict[str, Any]]] = []
        for task_file in sorted(self.queue_path.glob("*.json")):
            if task_file.name.endswith(".lock") or task_file.name.endswith(".heartbeat.json"):
                continue
            try:
                task = _load_json(task_file)
            except Exception:
                continue
            staged.append((task_file, task))
        staged.sort(key=lambda x: _task_priority_key(x[1]))
        ranked_files = [task_file for task_file, _ in staged]

        for task_file in ranked_files:
            lock_path = Path(str(task_file) + ".lock")
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(lock_path, "w") as lock_f:
                    try:
                        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue

                    if not task_file.exists():
                        continue
                    task = _load_json(task_file)

                    if task.get("executor") == "brain":
                        continue
                    if task.get("task_class", "cpu") != "cpu":
                        continue

                    task["attempts"] = task.get("attempts", 0) + 1
                    task["workers_attempted"] = task.get("workers_attempted", [])
                    task["workers_attempted"].append(self.agent_name)
                    now = datetime.now().isoformat()
                    task["last_attempt_at"] = now
                    if not task.get("first_attempted_at"):
                        task["first_attempted_at"] = now
                    task["status"] = "processing"
                    task["assigned_to"] = self.agent_name
                    task["started_at"] = now

                    dest = self.processing_path / task_file.name
                    # Atomic claim: move queue file to processing while holding
                    # queue lock so only one worker can own this task.
                    os.replace(task_file, dest)
                    _write_json(dest, task)
                    return task
            except Exception:
                continue
        return None

    def _finalize_task(self, task: Dict[str, Any], result: Dict[str, Any]) -> None:
        task_id = task["task_id"]
        task["completed_at"] = datetime.now().isoformat()
        task["result"] = result
        task["status"] = "complete" if result.get("success") else "failed"

        src = self.processing_path / f"{task_id}.json"
        if src.exists():
            try:
                src.unlink()
            except Exception:
                pass

        if self._active_task_heartbeat and self._active_task_heartbeat.exists():
            try:
                self._active_task_heartbeat.unlink()
            except Exception:
                pass

        out_dir = self.complete_path if task["status"] == "complete" else self.failed_path
        _write_json(out_dir / f"{task_id}.json", task)

    def _execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        task_id = task["task_id"]
        cmd = task.get("command") or task.get("prompt")
        if not cmd:
            return {
                "success": False,
                "output": "",
                "action": "blocked",
                "reason": "No command/prompt on task",
                "worker": self.agent_name,
            }

        cmd = self._normalize_command(cmd)
        task["command"] = cmd

        self._active_task_id = task_id
        self._active_task_heartbeat = self.processing_path / f"{task_id}.heartbeat.json"
        self._active_task_started_at = datetime.now().isoformat()
        self._write_cpu_heartbeat(state="running")

        executor = PermissionExecutor(
            str(self.permissions_file),
            agent_name=self.agent_name,
            heartbeat_callback=self._write_task_heartbeat,
        )
        stop_hb = threading.Event()

        def _heartbeat_loop() -> None:
            # Keep heartbeats fresh while long-running commands are active.
            while not stop_hb.is_set():
                if executor.active_process:
                    self._active_task_pid = executor.active_process.pid
                self._write_task_heartbeat()
                self._write_cpu_heartbeat(state="running")
                if stop_hb.wait(self.task_heartbeat_interval):
                    break

        hb_thread = threading.Thread(target=_heartbeat_loop, name=f"{self.agent_name}-task-hb", daemon=True)
        hb_thread.start()
        try:
            res = executor.run_bash(cmd, timeout=task.get("timeout_seconds") or self.worker_timeout)
        finally:
            stop_hb.set()
            hb_thread.join(timeout=2)
        self._active_task_pid = executor.active_process.pid if executor.active_process else self._active_task_pid
        self._write_task_heartbeat()
        self._write_cpu_heartbeat(state="running")
        return {
            "success": res.success,
            "output": res.output,
            "action": res.action.value,
            "reason": res.reason,
            "worker": self.agent_name,
        }

    def run(self) -> int:
        self._preflight()
        self._log(
            f"starting: queue={self.queue_path} permissions={self.permissions_file} "
            f"once={self.once} poll={self.poll_interval}s venv={self.venv_activate or '-'}"
        )
        self._write_cpu_heartbeat(state="idle")
        while True:
            blocked, temp, reason = self._thermal_pause_check()
            if blocked:
                if not self.thermal_pause_active or reason != self.thermal_pause_reason:
                    self._log(f"thermal throttle active: {reason}")
                self.thermal_pause_active = True
                self.thermal_pause_reason = reason
                self._write_cpu_heartbeat(state="thermal_pause")
                if self.once:
                    self._log("stopping in once mode due to thermal throttle")
                    return 2
                time.sleep(self.poll_interval)
                continue
            if self.thermal_pause_active:
                self._log(
                    f"thermal throttle cleared: cpu_temp={temp}C "
                    f"(resume margin {self.thermal_resume_margin_c}C)"
                )
            self.thermal_pause_active = False
            self.thermal_pause_reason = None

            task = self._claim_one_task()
            if not task:
                self._write_cpu_heartbeat(state="idle")
                if self.once:
                    self._log("no eligible cpu task in queue")
                    return 0
                time.sleep(self.poll_interval)
                continue

            tid = task["task_id"][:8]
            self._log(f"claimed cpu task {tid} ({task.get('name', '')})")
            result = self._execute_task(task)
            self._finalize_task(task, result)
            self._log(f"finished cpu task {tid}: {'OK' if result.get('success') else 'FAIL'}")

            self._active_task_id = None
            self._active_task_heartbeat = None
            self._active_task_pid = None
            self._active_task_started_at = None
            self._write_cpu_heartbeat(state="idle")

            if self.once:
                return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU task worker agent")
    default_configs = [
        "/media/bryan/shared/agents/config.json",  # NFS-first worker image path
        "/home/bryan/llm_orchestration/shared/agents/config.json",
    ]
    default_config = next((p for p in default_configs if Path(p).exists()), default_configs[-1])
    parser.add_argument(
        "--config",
        default=default_config,
        help="Path to orchestration config.json",
    )
    parser.add_argument("--name", default=None, help="CPU agent identity")
    parser.add_argument("--once", action="store_true", help="Process at most one eligible task")
    args = parser.parse_args()

    agent = CpuAgent(config_path=Path(args.config), agent_name=args.name, once=args.once)
    return agent.run()


if __name__ == "__main__":
    raise SystemExit(main())
