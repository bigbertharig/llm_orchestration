#!/usr/bin/env python3
"""
CPU agent for Pi workers.

Claims only worker-owned CPU tasks from queue, executes shell commands, and
writes results to complete/failed task directories.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
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

        self.poll_interval = int(
            self.config.get("cpu_agent", {}).get("poll_interval_seconds",
            self.config.get("timeouts", {}).get("poll_interval_seconds", 5))
        )
        if self.poll_interval <= 0:
            self.poll_interval = 5
        # Jitter range (seconds) added to each poll cycle to stagger NFS access
        self.poll_jitter = float(
            self.config.get("cpu_agent", {}).get("poll_jitter_seconds", 3.0)
        )
        cpu_agent_cfg = self.config.get("cpu_agent", {})
        self.startup_retry_interval_seconds = int(
            cpu_agent_cfg.get("startup_retry_interval_seconds", 30)
        )
        if self.startup_retry_interval_seconds <= 0:
            self.startup_retry_interval_seconds = 30
        # 0 means retry forever (default).
        self.startup_retry_max_seconds = int(cpu_agent_cfg.get("startup_retry_max_seconds", 0))
        if self.startup_retry_max_seconds < 0:
            self.startup_retry_max_seconds = 0
        self.worker_timeout = int(self.config.get("timeouts", {}).get("worker_task_seconds", 0))
        self.worker_timeout = None if self.worker_timeout <= 0 else self.worker_timeout
        self.task_heartbeat_interval = int(
            self.config.get("timeouts", {}).get("task_heartbeat_interval_seconds", 15)
        )
        if self.task_heartbeat_interval <= 0:
            self.task_heartbeat_interval = 15
        self.heartbeat_cleanup_interval_seconds = int(
            self.config.get("cpu_agent", {}).get("heartbeat_cleanup_interval_seconds", 60)
        )
        if self.heartbeat_cleanup_interval_seconds <= 0:
            self.heartbeat_cleanup_interval_seconds = 60
        self.heartbeat_cleanup_min_age_seconds = int(
            self.config.get("cpu_agent", {}).get("heartbeat_cleanup_min_age_seconds", 120)
        )
        if self.heartbeat_cleanup_min_age_seconds < 0:
            self.heartbeat_cleanup_min_age_seconds = 120
        self._last_heartbeat_cleanup_ts = 0.0
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
        self.env_check_cache: Dict[str, Dict[str, Any]] = {}
        self.env_block_reason: Optional[str] = None

        os.environ.setdefault("LLM_ORCH_LOG_PATH", str(self.shared_path / "logs"))
        os.environ["WORKER_NAME"] = self.agent_name

    def _log(self, message: str) -> None:
        ts = datetime.now().isoformat(timespec="seconds")
        print(f"{ts} [{self.agent_name}] {message}", flush=True)

    def _pick_default_venv_activate(self) -> str | None:
        candidates = [
            "/home/bryan/llm-orchestration-venv/bin/activate",
            "/opt/worker-env/bin/activate",   # fallback when user env is absent
        ]
        for p in candidates:
            if Path(p).exists():
                return f"source {p}"
        return None

    def _env_python_executable(self) -> str:
        if self.venv_activate:
            m = re.search(r"source\s+(\S+)/bin/activate", self.venv_activate)
            if m:
                candidate = Path(m.group(1)) / "bin" / "python"
                if candidate.exists():
                    return str(candidate)
        return sys.executable or "python3"

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
            "env_block_reason": self.env_block_reason,
            "queue_path": str(self.queue_path),
            "processing_path": str(self.processing_path),
            "stats": {
                "queue_count": len(list(self.queue_path.glob("*.json"))),
                "processing_count": len(list(self.processing_path.glob("*.json"))),
            },
        }
        _write_json(self.cpu_heartbeat_file, hb)
        _write_json(self.unified_heartbeat_file, hb)

    def _cleanup_orphan_task_heartbeats(self) -> None:
        """Remove stale processing heartbeats that have no live task file."""
        now = time.time()
        if now - self._last_heartbeat_cleanup_ts < self.heartbeat_cleanup_interval_seconds:
            return
        self._last_heartbeat_cleanup_ts = now

        removed = 0
        for hb_file in self.processing_path.glob("*.heartbeat.json"):
            task_id = hb_file.name.replace(".heartbeat.json", "")
            # Keep the currently active task heartbeat.
            if task_id == self._active_task_id:
                continue

            processing_file = self.processing_path / f"{task_id}.json"
            queue_file = self.queue_path / f"{task_id}.json"
            if processing_file.exists() or queue_file.exists():
                continue

            age_seconds = now - hb_file.stat().st_mtime
            if age_seconds < self.heartbeat_cleanup_min_age_seconds:
                continue

            try:
                hb_file.unlink()
                removed += 1
            except OSError:
                continue

        if removed:
            self._log(f"removed {removed} stale task heartbeat file(s)")

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

    def _preflight_with_retry(self) -> None:
        started_at = time.time()
        attempts = 0
        last_error: Exception | None = None
        while True:
            try:
                self._preflight()
                return
            except Exception as exc:
                last_error = exc
                attempts += 1
                elapsed = int(time.time() - started_at)
                if self.startup_retry_max_seconds > 0 and elapsed >= self.startup_retry_max_seconds:
                    break
                self._log(
                    f"startup preflight not ready ({exc}); retrying in "
                    f"{self.startup_retry_interval_seconds}s (attempt {attempts})"
                )
                time.sleep(self.startup_retry_interval_seconds)

        max_window = f"{self.startup_retry_max_seconds}s"
        if self.startup_retry_max_seconds == 0:
            max_window = "forever"
        raise RuntimeError(
            f"preflight failed after retry window {max_window}: {last_error}"
        )

    def _resolve_env_manifest_path(self, task: Dict[str, Any]) -> Optional[Path]:
        direct = task.get("env_manifest_path")
        if isinstance(direct, str) and direct.strip():
            p = Path(direct.strip())
            if p.exists():
                return p
        batch_path = task.get("batch_path")
        if isinstance(batch_path, str) and batch_path.strip():
            p = Path(batch_path.strip()) / "env_manifest.json"
            if p.exists():
                return p
        batch_id = str(task.get("batch_id", "")).strip()
        if not batch_id:
            return None
        plans_dir = self.shared_path / "plans"
        # Support nested plan layouts (e.g. plans/shoulders/<name>/history/<batch_id>/...).
        for candidate in plans_dir.glob(f"**/history/{batch_id}/env_manifest.json"):
            if candidate.exists():
                return candidate
        return None

    def _normalize_shared_file_path(self, raw_path: str) -> Path:
        raw = raw_path.strip()
        candidates = [raw]
        candidates.append(raw.replace("/mnt/shared", str(self.shared_path)))
        candidates.append(
            raw.replace("/home/bryan/llm_orchestration/shared", str(self.shared_path))
        )
        for candidate in candidates:
            p = Path(candidate)
            if p.exists():
                return p
        return Path(candidates[-1])

    def _manifest_probe_paths(self, manifest: Dict[str, Any]) -> list[str]:
        paths: list[str] = []
        seen: set[str] = set()
        scanned = manifest.get("scripts_scanned", [])
        if not isinstance(scanned, list):
            return paths
        for item in scanned:
            if not isinstance(item, str) or not item.strip():
                continue
            normalized = self._normalize_shared_file_path(item)
            base_dir = normalized.parent if normalized.suffix == ".py" else normalized
            if not base_dir.exists() or not base_dir.is_dir():
                continue
            key = str(base_dir.resolve())
            if key in seen:
                continue
            seen.add(key)
            paths.append(key)
        return paths

    def _module_available(
        self,
        py_exec: str,
        module_name: str,
        probe_paths: list[str],
    ) -> bool:
        env = os.environ.copy()
        if probe_paths:
            existing = env.get("PYTHONPATH", "")
            joined = os.pathsep.join(probe_paths + ([existing] if existing else []))
            env["PYTHONPATH"] = joined
        try:
            probe = subprocess.run(
                [
                    py_exec,
                    "-c",
                    "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)",
                    module_name,
                ],
                capture_output=True,
                text=True,
                timeout=5,
                env=env,
            )
        except Exception:
            return False
        return probe.returncode == 0

    def _check_task_env_requirements(self, task: Dict[str, Any]) -> tuple[bool, str]:
        batch_id = str(task.get("batch_id", "")).strip()
        if not batch_id:
            return True, ""

        manifest_path = self._resolve_env_manifest_path(task)
        if not manifest_path:
            return False, f"env manifest missing for batch {batch_id}"

        cache_key = f"{batch_id}:{manifest_path}"
        mtime = manifest_path.stat().st_mtime
        cached = self.env_check_cache.get(cache_key)
        if cached and cached.get("mtime") == mtime:
            return bool(cached.get("ok")), str(cached.get("reason", ""))

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as exc:
            reason = f"env manifest unreadable: {exc}"
            self.env_check_cache[cache_key] = {"mtime": mtime, "ok": False, "reason": reason}
            return False, reason

        required = manifest.get("required_modules", [])
        if not isinstance(required, list):
            required = []
        py_exec = self._env_python_executable()
        probe_paths = self._manifest_probe_paths(manifest)
        missing: list[str] = []
        for module_name in required:
            if not isinstance(module_name, str) or not module_name:
                continue
            if not self._module_available(py_exec, module_name, probe_paths):
                missing.append(module_name)
        if missing:
            reason = f"missing modules: {', '.join(sorted(missing))}"
            self.env_check_cache[cache_key] = {"mtime": mtime, "ok": False, "reason": reason}
            return False, reason

        self.env_check_cache[cache_key] = {"mtime": mtime, "ok": True, "reason": ""}
        return True, ""

    @staticmethod
    def _remove_stale_lock(lock_path: Path, max_age_seconds: int = 60) -> None:
        """Remove a lock file if it's older than max_age_seconds."""
        try:
            age = time.time() - lock_path.stat().st_mtime
            if age > max_age_seconds:
                lock_path.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass

    @staticmethod
    def _try_acquire_lock(lock_path: Path) -> bool:
        """Atomically create a lock file. Returns True if we got the lock.

        Uses O_CREAT | O_EXCL which is atomic even on NFS — if the file
        already exists the call fails immediately, no race window.
        """
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
            os.close(fd)
            return True
        except FileExistsError:
            return False
        except OSError:
            return False

    @staticmethod
    def _release_lock(lock_path: Path) -> None:
        """Remove a lock file (best-effort)."""
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass

    def _claim_one_task(self) -> Optional[Dict[str, Any]]:
        for task_file in sorted(self.queue_path.glob("*.json")):
            if task_file.name.endswith(".lock") or task_file.name.endswith(".heartbeat.json"):
                continue

            lock_path = Path(str(task_file) + ".lock")

            # Clean up stale locks (older than 60s) left by crashed workers
            if lock_path.exists():
                self._remove_stale_lock(lock_path, max_age_seconds=60)
                if lock_path.exists():
                    continue

            # Atomic lock acquisition — O_CREAT|O_EXCL is NFS-safe
            if not self._try_acquire_lock(lock_path):
                continue

            try:
                if not task_file.exists():
                    self._release_lock(lock_path)
                    continue
                task = _load_json(task_file)

                if task.get("executor") == "brain":
                    self._release_lock(lock_path)
                    continue
                if task.get("task_class", "cpu") != "cpu":
                    self._release_lock(lock_path)
                    continue

                env_ok, env_reason = self._check_task_env_requirements(task)
                if not env_ok:
                    self.env_block_reason = f"{task.get('batch_id', '-')}: {env_reason}"
                    prior = task.get("env_blocked_reason")
                    if prior != env_reason:
                        task["env_blocked_reason"] = env_reason
                        task["env_blocked_at"] = datetime.now().isoformat()
                        _write_json(task_file, task)
                    self._log(
                        f"skip task {task.get('task_id', '')[:8]} ({task.get('name', '')}) "
                        f"- env check failed: {env_reason}"
                    )
                    self._release_lock(lock_path)
                    continue
                self.env_block_reason = None

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
                os.replace(task_file, dest)
                _write_json(dest, task)
                self._release_lock(lock_path)
                return task
            except Exception:
                self._release_lock(lock_path)
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

        # Clean up any leftover lock file from the queue
        queue_lock = self.queue_path / f"{task_id}.json.lock"
        self._release_lock(queue_lock)

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
        self._preflight_with_retry()
        self._log(
            f"starting: queue={self.queue_path} permissions={self.permissions_file} "
            f"once={self.once} poll={self.poll_interval}s venv={self.venv_activate or '-'} "
            f"startup_retry=every {self.startup_retry_interval_seconds}s "
            f"(max {self.startup_retry_max_seconds or 'forever'})"
        )
        self._write_cpu_heartbeat(state="idle")
        while True:
            self._cleanup_orphan_task_heartbeats()
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
                time.sleep(self.poll_interval + random.uniform(0, self.poll_jitter))
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
                time.sleep(self.poll_interval + random.uniform(0, self.poll_jitter))
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
