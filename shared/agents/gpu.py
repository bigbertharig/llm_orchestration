#!/usr/bin/env python3
"""
GPU Agent - Owns a single physical GPU and manages worker subprocesses.

The GPU is the agent, not the worker. It visits the task queue on a 30-second
heartbeat cycle, claims tasks within its VRAM budget, spawns worker subprocesses,
collects results, and reports back. Workers are short-lived children that execute
a single task and exit.

Hot/Cold state:
  - Cold: No LLM loaded. Dynamic VRAM budget. Grabs multiple script/cpu tasks.
  - Hot:  LLM loaded, VRAM full. Grabs one LLM task at a time.

Usage:
  python gpu.py gpu-1 --config config.json
"""

import argparse
import importlib.util
import json
import os
import sys
import time
import signal
import logging
import subprocess
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from multiprocessing import Process, Queue
from filelock import FileLock, Timeout
from hardware import scan_cpu_temps

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)

# =============================================================================
# Constants
# =============================================================================
VALID_TASK_CLASSES = ['cpu', 'script', 'llm', 'meta']

# Timing
EXTERNAL_HEARTBEAT_INTERVAL = 30  # seconds - filesystem heartbeat to brain
INTERNAL_POLL_INTERVAL = 5        # seconds - check worker status internally
SIGNAL_CHECK_INTERVAL = 5         # seconds - check stop/abort signals

# VRAM budget
VRAM_BUDGET_RATIO = 0.8           # Use at most 80% of total VRAM
DEFAULT_CPU_VRAM_COST = 1024      # MB - virtual cost for CPU tasks to limit concurrency


class WorkerResult:
    """Result from a completed worker subprocess."""
    def __init__(self, task_id: str, task: Dict, result: Dict, peak_vram_mb: int = 0):
        self.task_id = task_id
        self.task = task
        self.result = result
        self.peak_vram_mb = peak_vram_mb


class GPUAgent:
    def __init__(self, config_path: str, gpu_name: str):
        self.config_path = Path(config_path)
        self.config = self._load_config(config_path)
        self.gpu_config = self._get_gpu_config(gpu_name)
        self.name = gpu_name
        self.gpu_id = self.gpu_config["id"]
        self.gpu_total_vram = self.gpu_config.get("vram_mb") or self._query_gpu_vram()
        self.logger = logging.getLogger(self.name)

        # Set logging level from env
        log_level = os.environ.get("GPU_LOG_LEVEL", "INFO").upper()
        try:
            self.logger.setLevel(getattr(logging, log_level))
        except AttributeError:
            self.logger.setLevel(logging.INFO)

        # Resolve paths
        config_dir = self.config_path.parent
        shared_path = Path(self.config["shared_path"])
        if not shared_path.is_absolute():
            shared_path = (config_dir / shared_path).resolve()
        self.shared_path = shared_path

        # Task queue paths
        self.queue_path = self.shared_path / "tasks" / "queue"
        self.processing_path = self.shared_path / "tasks" / "processing"
        self.complete_path = self.shared_path / "tasks" / "complete"
        self.failed_path = self.shared_path / "tasks" / "failed"

        # GPU state path (we are sole owner, no FileLock needed)
        self.gpu_state_dir = self.shared_path / "gpus" / f"gpu_{self.gpu_id}"
        self.gpu_state_dir.mkdir(parents=True, exist_ok=True)
        self.heartbeat_file = self.gpu_state_dir / "heartbeat.json"

        # Signals
        self.signals_path = self.shared_path / "signals"
        self.signals_path.mkdir(parents=True, exist_ok=True)

        # Logs
        self.log_path = self.shared_path / "logs"
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.training_log = self.log_path / "training_samples.jsonl"

        # Ollama config
        self.model = self.gpu_config.get("model")
        self.port = self.gpu_config.get("port")
        self.api_url = f"http://localhost:{self.port}/api/generate" if self.port else None
        self.worker_keep_alive = str(self.config.get("worker_keep_alive", "-1"))
        self.worker_num_ctx = int(self.config.get("worker_context_tokens", 8192))

        # Permissions
        permissions_path = Path(self.config["permissions_path"])
        if not permissions_path.is_absolute():
            permissions_path = (config_dir / permissions_path).resolve()
        self.permissions_file = str(
            permissions_path / self.gpu_config.get("permissions", "worker.json")
        )

        # Resource limits
        self.resource_limits = self.config["resource_limits"]
        self.task_timeout = self.config["timeouts"]["worker_task_seconds"]

        # GPU state
        self.running = True
        self.state = "cold"       # "cold" or "hot"
        self.model_loaded = False
        self.ollama_process: Optional[subprocess.Popen] = None

        # Worker tracking
        # active_workers: worker_id -> {process, task, vram_estimate, pid, started_at}
        self.active_workers: Dict[str, Dict] = {}
        # Outbox: completed results waiting to be written to filesystem on next cycle
        self.outbox: List[WorkerResult] = []
        # Internal status queue: workers post updates here every INTERNAL_POLL_INTERVAL
        self.status_queue: Queue = Queue()

        # VRAM budget tracking
        self.claimed_vram = 0  # Sum of VRAM estimates for currently running workers

        # Stats
        self.stats = {"tasks_completed": 0, "tasks_failed": 0}
        self.last_external_heartbeat = 0
        self.env_check_cache: Dict[str, Dict[str, Any]] = {}
        self.env_block_reason: str | None = None

        # Ollama health tracking
        self.ollama_healthy = True
        self.ollama_consecutive_failures = 0
        self.ollama_health_threshold = 3  # Restart after this many consecutive failures
        self.ollama_circuit_breaker = 5   # Stop claiming LLM tasks after this many

        # Thermal/constrained-state tracking for explicit stall visibility
        self.thermal_pause_active = False
        self.thermal_pause_reasons: List[str] = []
        self.last_thermal_event: Optional[Dict[str, Any]] = None
        self.thermal_pause_until: float = 0.0
        self.thermal_pause_current_seconds: int = 0
        self.thermal_pause_attempts: int = 0

        # Thermal pause policy (warning-level cooldown with exponential backoff)
        pause_cfg = self.config.get("thermal_pause", {})
        self.thermal_pause_initial_seconds = int(pause_cfg.get("initial_seconds", 60))
        self.thermal_pause_max_seconds = int(pause_cfg.get("max_seconds", 600))
        self.thermal_pause_backoff_factor = float(pause_cfg.get("backoff_factor", 2.0))
        self.thermal_resume_margin_c = int(pause_cfg.get("resume_margin_c", 3))

        # Set CUDA device for this process and all children
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        # Verify core/ security before starting
        self._verify_core_security()

        self.logger.info(
            f"GPU agent initialized: {self.name} (GPU {self.gpu_id}), "
            f"port {self.port}, model {self.model}"
        )

    # =========================================================================
    # Config
    # =========================================================================

    def _load_config(self, config_path: str) -> dict:
        if not Path(config_path).exists():
            template = Path(config_path).parent / "config.template.json"
            print(f"ERROR: Config file not found: {config_path}")
            if template.exists():
                print(f"  Copy the template and fill in your values:")
                print(f"  cp {template} {config_path}")
            else:
                print(f"  See config.template.json for the required schema.")
            sys.exit(1)
        with open(config_path) as f:
            return json.load(f)

    def _get_gpu_config(self, gpu_name: str) -> dict:
        for gpu in self.config["gpus"]:
            if gpu["name"] == gpu_name:
                return gpu
        raise ValueError(f"GPU '{gpu_name}' not found in config")

    def _query_gpu_vram(self) -> int:
        """Query total VRAM for this GPU via nvidia-smi. Fallback for configs without vram_mb."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.total',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 2 and int(parts[0]) == self.gpu_id:
                    return int(float(parts[1]))
        except Exception:
            pass
        return 6144  # Last-resort default

    def _verify_core_security(self):
        """Verify core/ directory is properly secured before starting.

        Checks: root ownership, no group/world-writable files, not running as root.
        Exits with error if any check fails — these are hard security requirements.
        """
        import stat
        core_path = self.shared_path / "core"
        if not core_path.exists():
            self.logger.warning(f"core/ directory not found at {core_path}")
            return

        # Agents must never run as root
        if os.getuid() == 0:
            self.logger.error("SECURITY: Agents must NOT run as root.")
            sys.exit(1)

        # core/ must be root-owned
        st = core_path.stat()
        if st.st_uid != 0:
            self.logger.error(
                f"SECURITY: core/ is not root-owned (uid={st.st_uid}). "
                f"Run: sudo chown -R root:root {core_path}")
            sys.exit(1)

        # Files in core/ must not be group/world-writable
        for f in core_path.iterdir():
            if f.is_file():
                mode = f.stat().st_mode
                if mode & stat.S_IWOTH or mode & stat.S_IWGRP:
                    self.logger.error(
                        f"SECURITY: {f} is group/world-writable. "
                        f"Run: sudo chmod 644 {f}")
                    sys.exit(1)

        self.logger.info("core/ security check passed")

    # =========================================================================
    # Ollama Management
    # =========================================================================

    def start_ollama(self):
        """Start Ollama instance on this GPU's dedicated port."""
        if not self.port:
            self.logger.info("No port configured - script-only GPU, skipping Ollama")
            return

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        env["OLLAMA_HOST"] = f"0.0.0.0:{self.port}"

        self.logger.info(f"Starting Ollama on GPU {self.gpu_id}, port {self.port}")

        self.ollama_process = subprocess.Popen(
            ["ollama", "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        for i in range(30):
            try:
                requests.get(f"http://localhost:{self.port}/api/tags", timeout=1)
                self.logger.info("Ollama server ready")
                return
            except Exception:
                time.sleep(1)

        raise RuntimeError("Ollama failed to start")

    def stop_ollama(self):
        """Stop the Ollama instance."""
        if self.ollama_process:
            self.ollama_process.terminate()
            try:
                self.ollama_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.ollama_process.kill()
            self.logger.info("Ollama stopped")

    def check_ollama_health(self) -> Dict[str, Any]:
        """Check Ollama instance health via /api/tags endpoint.

        Returns dict with health status and loaded models. Tracks consecutive
        failures and triggers restart after threshold. Implements circuit breaker
        to stop claiming LLM tasks after sustained failures.
        """
        health = {
            "healthy": False,
            "loaded_models": [],
            "consecutive_failures": self.ollama_consecutive_failures,
            "response_ms": None,
        }

        if not self.port:
            health["healthy"] = True  # Script-only GPU, no Ollama expected
            health["note"] = "no_ollama_configured"
            return health

        try:
            start = time.time()
            resp = requests.get(
                f"http://localhost:{self.port}/api/tags",
                timeout=5
            )
            elapsed_ms = int((time.time() - start) * 1000)
            health["response_ms"] = elapsed_ms

            if resp.status_code == 200:
                data = resp.json()
                models = data.get("models", [])
                health["loaded_models"] = [m.get("name", "") for m in models]
                health["healthy"] = True
                self.ollama_consecutive_failures = 0
                self.ollama_healthy = True
            else:
                self.ollama_consecutive_failures += 1
                self.logger.warning(
                    f"Ollama health check returned {resp.status_code} "
                    f"(failure {self.ollama_consecutive_failures})")

        except Exception as e:
            self.ollama_consecutive_failures += 1
            self.logger.warning(
                f"Ollama health check failed: {e} "
                f"(failure {self.ollama_consecutive_failures})")

        health["consecutive_failures"] = self.ollama_consecutive_failures

        # Auto-restart after threshold
        if self.ollama_consecutive_failures >= self.ollama_health_threshold:
            self.logger.error(
                f"Ollama failed {self.ollama_consecutive_failures} consecutive health checks, "
                f"attempting restart")
            try:
                self.stop_ollama()
                time.sleep(2)
                self.start_ollama()
                self.ollama_consecutive_failures = 0
                self.ollama_healthy = True
                # Model was lost in restart
                self.model_loaded = False
                self.state = "cold"
                self.logger.info("Ollama restarted successfully after health check failures")
            except Exception as e:
                self.logger.error(f"Ollama restart failed: {e}")

        # Circuit breaker
        if self.ollama_consecutive_failures >= self.ollama_circuit_breaker:
            self.ollama_healthy = False
            self.logger.error(
                f"Circuit breaker: Ollama unhealthy after {self.ollama_consecutive_failures} "
                f"failures, will not claim LLM tasks")

        return health

    def load_model(self):
        """Load LLM model into VRAM. Transitions GPU to Hot state."""
        if self.model_loaded:
            return

        if not self.api_url:
            self.logger.warning("Cannot load model - no Ollama port configured")
            return

        self.logger.info(f"Loading model {self.model} into VRAM...")
        start_time = time.time()

        try:
            # Initial pull/load request (can block while weights are loaded)
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": "Hello",
                    "stream": False,
                    "keep_alive": self.worker_keep_alive,
                    "options": {
                        "num_gpu": 1,
                        "num_ctx": self.worker_num_ctx,
                    }
                },
                timeout=600
            )
            response.raise_for_status()

            # Readiness gate: only mark hot once generation is reliably responsive.
            if not self._wait_for_model_ready(max_wait_seconds=90):
                raise RuntimeError("model readiness probe timed out after load")

            elapsed = int(time.time() - start_time)
            self.model_loaded = True
            self.state = "hot"
            self.logger.info(f"Model loaded in {elapsed}s - GPU is now HOT")

        except Exception as e:
            self.model_loaded = False
            self.state = "cold"
            self.logger.error(f"Failed to load model: {e}")

    def _wait_for_model_ready(self, max_wait_seconds: int = 90) -> bool:
        """
        Probe Ollama until model is actually ready to serve.
        Prevents immediate LLM task claims while model is still warming.
        """
        if not self.api_url:
            return False

        deadline = time.time() + max_wait_seconds
        last_error = ""

        while time.time() < deadline:
            try:
                r = requests.post(
                    self.api_url,
                    json={
                        "model": self.model,
                        "prompt": "READY?",
                        "stream": False,
                        "keep_alive": self.worker_keep_alive,
                        "options": {
                            "num_predict": 4,
                            "num_gpu": 1,
                            "num_ctx": self.worker_num_ctx,
                        }
                    },
                    timeout=30
                )
                if r.status_code == 200:
                    _ = r.json().get("response", "")
                    return True
                last_error = f"status={r.status_code}"
            except Exception as e:
                last_error = str(e)

            time.sleep(2)

        self.logger.warning(f"Model readiness probe failed/timed out: {last_error}")
        return False

    def _wait_for_model_unloaded(self, max_wait_seconds: int = 30) -> bool:
        """Poll Ollama until this model no longer appears as loaded."""
        if not self.port:
            return True

        ps_url = f"http://localhost:{self.port}/api/ps"
        deadline = time.time() + max_wait_seconds
        last_error = ""

        while time.time() < deadline:
            try:
                r = requests.get(ps_url, timeout=5)
                if r.status_code == 200:
                    models = r.json().get("models", [])
                    loaded_names = [m.get("name", "") for m in models]
                    if self.model not in loaded_names:
                        return True
                else:
                    last_error = f"status={r.status_code}"
            except Exception as e:
                last_error = str(e)
            time.sleep(1)

        self.logger.warning(f"Model unload verify timed out: {last_error}")
        return False

    def unload_model(self):
        """Unload LLM model from VRAM. Transitions GPU to Cold state.

        Retries up to 3 times with 5s backoff on failure, since a failed unload
        leaves VRAM in an inconsistent state (agent thinks model is loaded but
        VRAM may be partially freed).
        """
        if not self.model_loaded:
            return

        if not self.api_url:
            return

        self.logger.info(f"Unloading model {self.model} to free VRAM...")

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(
                    self.api_url,
                    json={"model": self.model, "prompt": "", "keep_alive": 0},
                    timeout=30
                )

                if response.status_code == 200:
                    if self._wait_for_model_unloaded(max_wait_seconds=30):
                        self.model_loaded = False
                        self.state = "cold"
                        self.logger.info("Model unloaded - GPU is now COLD")
                        return
                    self.logger.warning(
                        f"Unload attempt {attempt}/{max_retries} did not verify model removal"
                    )
                else:
                    self.logger.warning(
                        f"Unload attempt {attempt}/{max_retries} returned "
                        f"status {response.status_code}")

            except Exception as e:
                self.logger.warning(
                    f"Unload attempt {attempt}/{max_retries} failed: {e}")

            if attempt < max_retries:
                time.sleep(5)

    # =========================================================================
    # GPU Stats
    # =========================================================================

    def _get_gpu_stats(self) -> Dict[str, Any]:
        """Get current GPU resource stats via nvidia-smi."""
        try:
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=index,temperature.gpu,power.draw,memory.used,memory.total,'
                'utilization.gpu,clocks.sm,clocks_throttle_reasons.active',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)

            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 8:
                    gpu_index = int(parts[0])
                    if gpu_index == self.gpu_id:
                        vram_used = int(parts[3])
                        vram_total = int(parts[4])
                        vram_percent = int(100 * vram_used / vram_total) if vram_total > 0 else 0
                        return {
                            "temperature_c": int(parts[1]),
                            "power_draw_w": float(parts[2]),
                            "vram_used_mb": vram_used,
                            "vram_total_mb": vram_total,
                            "vram_percent": vram_percent,
                            "gpu_util_percent": int(parts[5]),
                            "clock_mhz": int(parts[6]),
                            "throttle_status": parts[7] if parts[7] != "0x0000000000000000" else "None"
                        }

        except Exception as e:
            self.logger.debug(f"Failed to get GPU stats: {e}")

        return {
            "temperature_c": 0, "power_draw_w": 0.0,
            "vram_used_mb": 0, "vram_total_mb": self.gpu_total_vram,
            "vram_percent": 0, "gpu_util_percent": 0,
            "clock_mhz": 0, "throttle_status": "Unknown"
        }

    def _get_worker_vram(self, pid: int) -> int:
        """Get VRAM usage for a specific worker PID via nvidia-smi."""
        try:
            result = subprocess.run([
                'nvidia-smi',
                '--query-compute-apps=pid,used_memory',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)

            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 2 and int(parts[0]) == pid:
                    return int(parts[1])
        except Exception:
            pass
        return 0

    def _get_cpu_temp(self) -> Optional[int]:
        """Get the highest CPU temperature reading. Returns temp in C or None."""
        try:
            temps = scan_cpu_temps()
            if temps:
                return max(t["temp_c"] for t in temps)
        except Exception:
            pass
        return None

    def _is_resource_constrained(self, gpu_stats: Dict) -> tuple:
        """Check if GPU resources are constrained. Returns (bool, reasons)."""
        constrained = False
        reasons = []

        gpu_temp = gpu_stats["temperature_c"]
        gpu_warn = self.resource_limits.get("gpu_temp_warning_c", self.resource_limits["max_temp_c"])
        if gpu_temp >= gpu_warn:
            constrained = True
            reasons.append(f"GPU temp {gpu_temp}C >= {gpu_warn}C")

        cpu_temp = self._get_cpu_temp()
        cpu_warn = self.resource_limits.get("cpu_temp_warning_c", 80)
        if cpu_temp is not None and cpu_temp >= cpu_warn:
            constrained = True
            reasons.append(f"CPU temp {cpu_temp}C >= {cpu_warn}C")

        if gpu_stats["vram_percent"] >= self.resource_limits["max_vram_percent"]:
            constrained = True
            reasons.append(f"VRAM {gpu_stats['vram_percent']}% >= {self.resource_limits['max_vram_percent']}%")

        if gpu_stats["power_draw_w"] >= self.resource_limits["max_power_w"]:
            constrained = True
            reasons.append(f"power {gpu_stats['power_draw_w']}W >= {self.resource_limits['max_power_w']}W")

        return constrained, reasons

    def _check_thermal_safety(self, gpu_stats: Dict) -> bool:
        """Check for critical temperatures. Returns True if safe, False if shutdown needed."""
        gpu_temp = gpu_stats["temperature_c"]
        gpu_crit = self.resource_limits.get("gpu_temp_critical_c", 90)
        cpu_temp = self._get_cpu_temp()
        cpu_crit = self.resource_limits.get("cpu_temp_critical_c", 95)

        critical_reasons = []
        if gpu_temp >= gpu_crit:
            critical_reasons.append(f"GPU {gpu_temp}C >= {gpu_crit}C CRITICAL")
        if cpu_temp is not None and cpu_temp >= cpu_crit:
            critical_reasons.append(f"CPU {cpu_temp}C >= {cpu_crit}C CRITICAL")

        if not critical_reasons:
            return True

        active_task_ids = [
            info["task"]["task_id"][:8] for info in self.active_workers.values()
        ]
        self.last_thermal_event = {
            "timestamp": datetime.now().isoformat(),
            "event": "critical_shutdown",
            "reasons": critical_reasons,
            "active_task_ids": active_task_ids,
        }
        self.logger.error(
            f"THERMAL SAFETY SHUTDOWN: {', '.join(critical_reasons)} - "
            f"killing all workers and stopping agent. "
            f"TASKS_STALLED_TEMP: {active_task_ids or ['none']}"
        )
        self.running = False
        return False

    def _pause_worker_processes(self):
        """Pause all active worker subprocesses (best-effort)."""
        paused = 0
        for worker_id, info in self.active_workers.items():
            proc = info["process"]
            if proc.poll() is None and not info.get("paused", False):
                try:
                    os.kill(proc.pid, signal.SIGSTOP)
                    info["paused"] = True
                    paused += 1
                    self.logger.warning(
                        f"THERMAL_PAUSE_WORKER: paused {worker_id} pid={proc.pid} "
                        f"task={info['task']['task_id'][:8]}"
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to pause worker {worker_id}: {e}")
        return paused

    def _resume_worker_processes(self):
        """Resume all paused worker subprocesses (best-effort)."""
        resumed = 0
        for worker_id, info in self.active_workers.items():
            proc = info["process"]
            if proc.poll() is None and info.get("paused", False):
                try:
                    os.kill(proc.pid, signal.SIGCONT)
                    info["paused"] = False
                    resumed += 1
                    self.logger.info(
                        f"THERMAL_RESUME_WORKER: resumed {worker_id} pid={proc.pid} "
                        f"task={info['task']['task_id'][:8]}"
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to resume worker {worker_id}: {e}")
        return resumed

    def _temps_below_resume_threshold(self, gpu_stats: Dict) -> bool:
        """Require temps below warning-minus-margin to avoid pause flapping."""
        gpu_warn = self.resource_limits.get("gpu_temp_warning_c", self.resource_limits["max_temp_c"])
        cpu_warn = self.resource_limits.get("cpu_temp_warning_c", 80)
        gpu_limit = max(0, gpu_warn - self.thermal_resume_margin_c)
        cpu_limit = max(0, cpu_warn - self.thermal_resume_margin_c)

        gpu_ok = gpu_stats["temperature_c"] < gpu_limit
        cpu_temp = self._get_cpu_temp()
        cpu_ok = True if cpu_temp is None else (cpu_temp < cpu_limit)
        return gpu_ok and cpu_ok

    def _update_thermal_pause_state(self, gpu_stats: Dict):
        """Manage warning-level thermal pause with exponential backoff and logs."""
        constrained, reasons = self._is_resource_constrained(gpu_stats)
        now = time.time()
        active_task_ids = [info["task"]["task_id"][:8] for info in self.active_workers.values()]

        if constrained:
            self.thermal_pause_reasons = reasons.copy()
            if not self.thermal_pause_active:
                self.thermal_pause_active = True
                self.thermal_pause_attempts = 1
                self.thermal_pause_current_seconds = self.thermal_pause_initial_seconds
                self.thermal_pause_until = now + self.thermal_pause_current_seconds
                paused = self._pause_worker_processes()
                self.last_thermal_event = {
                    "timestamp": datetime.now().isoformat(),
                    "event": "pause_entered",
                    "reasons": reasons.copy(),
                    "pause_seconds": self.thermal_pause_current_seconds,
                    "pause_attempt": self.thermal_pause_attempts,
                    "active_task_ids": active_task_ids,
                }
                self.logger.warning(
                    f"THERMAL_PAUSE_ENTER: reasons={reasons} pause={self.thermal_pause_current_seconds}s "
                    f"paused_workers={paused} active_task_ids={active_task_ids or ['none']}"
                )
            elif now >= self.thermal_pause_until:
                self.thermal_pause_attempts += 1
                next_pause = int(self.thermal_pause_current_seconds * self.thermal_pause_backoff_factor)
                self.thermal_pause_current_seconds = min(self.thermal_pause_max_seconds, max(1, next_pause))
                self.thermal_pause_until = now + self.thermal_pause_current_seconds
                paused = self._pause_worker_processes()
                self.last_thermal_event = {
                    "timestamp": datetime.now().isoformat(),
                    "event": "pause_extended",
                    "reasons": reasons.copy(),
                    "pause_seconds": self.thermal_pause_current_seconds,
                    "pause_attempt": self.thermal_pause_attempts,
                    "active_task_ids": active_task_ids,
                }
                self.logger.warning(
                    f"THERMAL_PAUSE_EXTEND: reasons={reasons} pause={self.thermal_pause_current_seconds}s "
                    f"attempt={self.thermal_pause_attempts} paused_workers={paused} "
                    f"active_task_ids={active_task_ids or ['none']}"
                )
            return

        if self.thermal_pause_active and now >= self.thermal_pause_until:
            if self._temps_below_resume_threshold(gpu_stats):
                resumed = self._resume_worker_processes()
                self.thermal_pause_active = False
                self.thermal_pause_reasons = []
                self.thermal_pause_until = 0.0
                self.thermal_pause_current_seconds = 0
                self.thermal_pause_attempts = 0
                self.last_thermal_event = {
                    "timestamp": datetime.now().isoformat(),
                    "event": "pause_cleared",
                    "reasons": [],
                    "active_task_ids": active_task_ids,
                }
                self.logger.info(
                    f"THERMAL_PAUSE_EXIT: resumed_workers={resumed} active_task_ids={active_task_ids or ['none']}"
                )
            else:
                self.thermal_pause_attempts += 1
                next_pause = int(self.thermal_pause_current_seconds * self.thermal_pause_backoff_factor)
                self.thermal_pause_current_seconds = min(self.thermal_pause_max_seconds, max(1, next_pause))
                self.thermal_pause_until = now + self.thermal_pause_current_seconds
                self.last_thermal_event = {
                    "timestamp": datetime.now().isoformat(),
                    "event": "pause_hold_hot",
                    "reasons": self.thermal_pause_reasons.copy(),
                    "pause_seconds": self.thermal_pause_current_seconds,
                    "pause_attempt": self.thermal_pause_attempts,
                    "active_task_ids": active_task_ids,
                }
                self.logger.warning(
                    f"THERMAL_PAUSE_HOLD: temp recovery threshold not met, "
                    f"extending pause to {self.thermal_pause_current_seconds}s "
                    f"attempt={self.thermal_pause_attempts}"
                )

    # =========================================================================
    # VRAM Budget
    # =========================================================================

    def _get_vram_budget(self) -> int:
        """Calculate available VRAM budget for new tasks."""
        total_budget = int(self.gpu_total_vram * VRAM_BUDGET_RATIO)
        available = total_budget - self.claimed_vram
        return max(0, available)

    def _get_task_vram_cost(self, task: Dict) -> int:
        """Get the VRAM cost for a task."""
        task_class = task.get("task_class", "cpu")

        if task_class == "llm":
            return int(self.gpu_total_vram * VRAM_BUDGET_RATIO)  # LLM uses full budget
        elif task_class == "cpu":
            return DEFAULT_CPU_VRAM_COST
        elif task_class == "script":
            return task.get("vram_estimate_mb", DEFAULT_CPU_VRAM_COST)
        elif task_class == "meta":
            return 0  # Meta tasks handled by GPU directly
        else:
            return DEFAULT_CPU_VRAM_COST

    # =========================================================================
    # Task Claiming
    # =========================================================================

    def _get_preferred_classes(self) -> List[str]:
        """Get task classes this GPU should claim based on current state.

        Circuit breaker: if Ollama is unhealthy, exclude LLM tasks to prevent
        burning through timeouts on a broken Ollama instance.
        """
        if not self.ollama_healthy:
            self.logger.debug("Ollama unhealthy - excluding LLM tasks from claim")
            return ['meta', 'script']
        if self.state == "hot":
            return ['meta', 'llm']
        # Cold GPUs should focus on script/cpu until a load_llm task
        # explicitly transitions them to hot.
        return ['meta', 'script']

    def _can_claim_meta_task(self, task: Dict) -> bool:
        """Enforce hot/cold ownership rules for meta tasks."""
        command = task.get("command", "")
        if command == "load_llm":
            # Only cold workers should claim load commands.
            return not self.model_loaded
        if command == "unload_llm":
            # Only hot workers should claim unload commands.
            return self.model_loaded
        # Unknown meta command: allow claim so it can fail explicitly.
        return True

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
        for candidate in plans_dir.glob(f"*/history/{batch_id}/env_manifest.json"):
            if candidate.exists():
                return candidate
        return None

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
        missing = [m for m in required if isinstance(m, str) and m and importlib.util.find_spec(m) is None]
        if missing:
            reason = f"missing modules: {', '.join(sorted(missing))}"
            self.env_check_cache[cache_key] = {"mtime": mtime, "ok": False, "reason": reason}
            return False, reason

        self.env_check_cache[cache_key] = {"mtime": mtime, "ok": True, "reason": ""}
        return True, ""

    def claim_tasks(self) -> List[Dict]:
        """
        Visit the board: claim as many tasks as fit within VRAM budget.
        Returns list of claimed tasks.
        """
        if self.thermal_pause_active:
            remaining = max(0, int(self.thermal_pause_until - time.time()))
            self.logger.warning(
                f"TASKS_THERMAL_PAUSED: skip claiming new work for {remaining}s "
                f"reasons={self.thermal_pause_reasons}"
            )
            return []

        preferred = self._get_preferred_classes()

        # Categorize available tasks
        tasks_by_class = {tc: [] for tc in VALID_TASK_CLASSES}

        for task_file in sorted(self.queue_path.glob("*.json")):
            if str(task_file).endswith('.lock'):
                continue
            try:
                with open(task_file) as f:
                    task = json.load(f)

                if task.get("executor") == "brain":
                    continue

                task_class = task.get("task_class", "cpu")
                if task_class in tasks_by_class:
                    tasks_by_class[task_class].append(task_file)
            except Exception:
                continue

        # Build ordered candidate list
        candidates = []
        for tc in preferred:
            candidates.extend(tasks_by_class.get(tc, []))

        # Claim tasks until budget is full
        claimed = []
        budget = self._get_vram_budget()

        for task_file in candidates:
            lock_file = str(task_file) + ".lock"
            lock = FileLock(lock_file, timeout=1)

            try:
                with lock:
                    if not task_file.exists():
                        continue

                    with open(task_file) as f:
                        task = json.load(f)

                    task_class = task.get("task_class", "cpu")
                    vram_cost = self._get_task_vram_cost(task)

                    # Meta tasks: handle directly, don't spawn worker
                    if task_class == "meta":
                        if not self._can_claim_meta_task(task):
                            self.logger.debug(
                                f"Skipping meta task {task['task_id'][:8]} ({task.get('command')}): "
                                f"state={self.state}, model_loaded={self.model_loaded}"
                            )
                            continue
                        # Keep attempt bookkeeping consistent with worker-claimed tasks.
                        task["attempts"] = task.get("attempts", 0) + 1
                        task["workers_attempted"] = task.get("workers_attempted", [])
                        task["workers_attempted"].append(self.name)
                        task["last_attempt_at"] = datetime.now().isoformat()
                        if not task.get("first_attempted_at"):
                            task["first_attempted_at"] = task["last_attempt_at"]
                        task["status"] = "processing"
                        task["assigned_to"] = self.name
                        task["started_at"] = task["last_attempt_at"]
                        # Keep meta tasks visible in processing/ for observability.
                        new_path = self.processing_path / task_file.name
                        with open(new_path, 'w') as f:
                            json.dump(task, f, indent=2)
                        task_file.unlink()
                        claimed.append(task)
                        continue

                    # llm tasks require a hot/ready model on this GPU.
                    if task_class == "llm" and not self.model_loaded:
                        self.logger.debug(
                            f"Skipping llm task {task['task_id'][:8]}: model not loaded yet"
                        )
                        continue

                    env_ok, env_reason = self._check_task_env_requirements(task)
                    if not env_ok:
                        self.env_block_reason = f"{task.get('batch_id', '-')}: {env_reason}"
                        prior = task.get("env_blocked_reason")
                        if prior != env_reason:
                            task["env_blocked_reason"] = env_reason
                            task["env_blocked_at"] = datetime.now().isoformat()
                            with open(task_file, 'w') as f:
                                json.dump(task, f, indent=2)
                        self.logger.warning(
                            f"Skipping task {task.get('task_id', '')[:8]} ({task.get('name', '')}) "
                            f"- environment check failed: {env_reason}"
                        )
                        continue
                    self.env_block_reason = None

                    # Budget check
                    if vram_cost > budget:
                        self.logger.debug(
                            f"Skipping {task['task_id'][:8]}: needs {vram_cost}MB, "
                            f"only {budget}MB available"
                        )
                        continue

                    # Claim it
                    task["attempts"] = task.get("attempts", 0) + 1
                    task["workers_attempted"] = task.get("workers_attempted", [])
                    task["workers_attempted"].append(self.name)
                    task["last_attempt_at"] = datetime.now().isoformat()
                    if not task.get("first_attempted_at"):
                        task["first_attempted_at"] = task["last_attempt_at"]

                    task["status"] = "processing"
                    task["assigned_to"] = self.name
                    task["started_at"] = task["last_attempt_at"]

                    # Move to processing
                    new_path = self.processing_path / task_file.name
                    with open(new_path, 'w') as f:
                        json.dump(task, f, indent=2)
                    task_file.unlink()

                    # Update budget
                    budget -= vram_cost
                    self.claimed_vram += vram_cost

                    claimed.append(task)
                    self.logger.info(
                        f"Claimed {task_class} task: {task['task_id'][:8]}... "
                        f"({task.get('name', '')}) [{vram_cost}MB, {budget}MB remaining]"
                    )

            except Timeout:
                continue
            except Exception:
                continue

        if not claimed:
            task_summary = ", ".join(
                f"{k}:{len(v)}" for k, v in tasks_by_class.items() if len(v) > 0
            )
            if task_summary:
                self.logger.debug(f"No tasks claimed - Queue: [{task_summary}], budget: {budget}MB")
            else:
                self.logger.debug("Queue empty")

        return claimed

    # =========================================================================
    # Worker Management
    # =========================================================================

    def _spawn_worker(self, task: Dict):
        """Spawn a worker subprocess to execute a task."""
        worker_id = f"{self.name}-w{len(self.active_workers)}-{task['task_id'][:8]}"
        vram_cost = self._get_task_vram_cost(task)

        # Build worker command
        worker_cmd = [
            sys.executable,
            str(Path(__file__).parent / "worker.py"),
            "--execute",
            "--config", str(self.config_path),
            "--gpu-name", self.name,
            "--permissions", self.permissions_file,
            "--task", json.dumps(task),
        ]

        # Pass Ollama URL if available (for LLM tasks)
        if self.port:
            worker_cmd.extend(["--ollama-url", f"http://localhost:{self.port}"])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        # Ensure script-style worker tasks can self-identify and self-route.
        env["WORKER_NAME"] = self.name
        if self.port:
            env["WORKER_OLLAMA_URL"] = f"http://localhost:{self.port}"

        proc = subprocess.Popen(
            worker_cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        self.active_workers[worker_id] = {
            "process": proc,
            "task": task,
            "vram_estimate": vram_cost,
            "pid": proc.pid,
            "started_at": time.time(),
            "peak_vram_mb": 0,
            "paused": False,
        }

        self.logger.info(
            f"Spawned worker {worker_id} (PID {proc.pid}) for "
            f"{task.get('task_class', 'cpu')} task {task['task_id'][:8]}"
        )
        self._write_task_heartbeat(task["task_id"], worker_id, proc.pid, peak_vram_mb=0)

    def _collect_finished_workers(self):
        """Check for completed worker subprocesses, collect results into outbox."""
        finished = []

        for worker_id, info in self.active_workers.items():
            proc = info["process"]
            returncode = proc.poll()

            if returncode is not None:
                # Worker finished - read its output
                stdout, stderr = proc.communicate()

                try:
                    result = json.loads(stdout)
                except (json.JSONDecodeError, ValueError):
                    result = {
                        "success": False,
                        "error": f"Worker output parse error. stderr: {stderr[:500]}",
                        "worker": self.name,
                    }

                # Get final VRAM reading for this worker
                peak_vram = info.get("peak_vram_mb", 0)

                # Release VRAM budget
                self.claimed_vram -= info["vram_estimate"]
                self.claimed_vram = max(0, self.claimed_vram)

                self.outbox.append(WorkerResult(
                    task_id=info["task"]["task_id"],
                    task=info["task"],
                    result=result,
                    peak_vram_mb=peak_vram,
                ))

                if result.get("success"):
                    self.stats["tasks_completed"] += 1
                else:
                    self.stats["tasks_failed"] += 1

                finished.append(worker_id)
                self._remove_task_heartbeat(info["task"]["task_id"])

                self.logger.info(
                    f"Worker {worker_id} finished: "
                    f"{'OK' if result.get('success') else 'FAIL'} "
                    f"(VRAM peak: {peak_vram}MB, freed {info['vram_estimate']}MB budget)"
                )

        for worker_id in finished:
            del self.active_workers[worker_id]

    def _update_worker_vram(self):
        """Poll per-PID VRAM usage for active workers (internal heartbeat)."""
        for worker_id, info in self.active_workers.items():
            pid = info["pid"]
            vram = self._get_worker_vram(pid)
            if vram > info["peak_vram_mb"]:
                info["peak_vram_mb"] = vram
            self._write_task_heartbeat(
                info["task"]["task_id"],
                worker_id,
                pid,
                peak_vram_mb=info["peak_vram_mb"]
            )

    def _task_heartbeat_file(self, task_id: str) -> Path:
        return self.processing_path / f"{task_id}.heartbeat.json"

    def _write_task_heartbeat(self, task_id: str, worker_id: str, pid: int, peak_vram_mb: int = 0):
        """Write per-task progress heartbeat for stuck-task detection."""
        hb = {
            "task_id": task_id,
            "worker_id": worker_id,
            "gpu": self.name,
            "pid": pid,
            "peak_vram_mb": int(peak_vram_mb),
            "updated_at": datetime.now().isoformat(),
        }
        try:
            with open(self._task_heartbeat_file(task_id), "w") as f:
                json.dump(hb, f, indent=2)
        except Exception:
            pass

    def _remove_task_heartbeat(self, task_id: str):
        hb_file = self._task_heartbeat_file(task_id)
        try:
            if hb_file.exists():
                hb_file.unlink()
        except Exception:
            pass

    # =========================================================================
    # Filesystem I/O (batched per cycle)
    # =========================================================================

    def _flush_outbox(self):
        """Write all completed task results to filesystem in one batch."""
        for wr in self.outbox:
            task = wr.task
            result = wr.result

            task["status"] = "complete" if result.get("success") else "failed"
            task["completed_at"] = datetime.now().isoformat()
            task["result"] = result
            task["result"]["max_vram_used_mb"] = wr.peak_vram_mb

            dest_path = self.complete_path if result.get("success") else self.failed_path
            dest_file = dest_path / f"{task['task_id']}.json"

            with open(dest_file, 'w') as f:
                json.dump(task, f, indent=2)

            # Remove from processing
            proc_file = self.processing_path / f"{task['task_id']}.json"
            if proc_file.exists():
                proc_file.unlink()
            self._remove_task_heartbeat(task["task_id"])

            status = "completed" if result.get("success") else "failed"
            self.logger.info(f"Task {task['task_id'][:8]}... {status}")

        count = len(self.outbox)
        self.outbox.clear()
        return count

    def _write_heartbeat(self):
        """Write GPU-level heartbeat to filesystem. We are sole owner, no lock needed.

        Includes Ollama health check results so the brain can detect unhealthy GPUs
        and avoid assigning LLM work to them.
        """
        gpu_stats = self._get_gpu_stats()
        constrained, constrained_reasons = self._is_resource_constrained(gpu_stats)

        # Run Ollama health check as part of heartbeat cycle
        ollama_health = self.check_ollama_health()

        active_task_info = []
        for worker_id, info in self.active_workers.items():
            active_task_info.append({
                "worker_id": worker_id,
                "task_id": info["task"]["task_id"][:8],
                "task_class": info["task"].get("task_class", "cpu"),
                "task_name": info["task"].get("name", ""),
                "vram_estimate_mb": info["vram_estimate"],
                "peak_vram_mb": info["peak_vram_mb"],
                "pid": info["pid"],
                "started_at": datetime.fromtimestamp(info["started_at"]).isoformat(),
            })

        cpu_temp = self._get_cpu_temp()

        heartbeat = {
            "gpu_id": self.gpu_id,
            "name": self.name,
            "state": self.state,
            "model_loaded": self.model_loaded,
            "ollama_healthy": self.ollama_healthy,
            "ollama": ollama_health,
            "last_updated": datetime.now().isoformat(),
            "temperature_c": gpu_stats["temperature_c"],
            "cpu_temp_c": cpu_temp,
            "power_draw_w": gpu_stats["power_draw_w"],
            "vram_used_mb": gpu_stats["vram_used_mb"],
            "vram_total_mb": gpu_stats["vram_total_mb"],
            "vram_percent": gpu_stats["vram_percent"],
            "gpu_util_percent": gpu_stats["gpu_util_percent"],
            "clock_mhz": gpu_stats["clock_mhz"],
            "throttle_status": gpu_stats["throttle_status"],
            "claimed_vram_mb": self.claimed_vram,
            "budget_available_mb": self._get_vram_budget(),
            "active_workers": len(self.active_workers),
            "active_tasks": active_task_info,
            "thermal_constrained": constrained,
            "thermal_reasons": constrained_reasons,
            "thermal_pause_active": self.thermal_pause_active,
            "thermal_pause_reasons": self.thermal_pause_reasons,
            "thermal_pause_until": datetime.fromtimestamp(self.thermal_pause_until).isoformat() if self.thermal_pause_until else None,
            "thermal_pause_remaining_seconds": max(0, int(self.thermal_pause_until - time.time())) if self.thermal_pause_active else 0,
            "thermal_pause_current_seconds": self.thermal_pause_current_seconds,
            "thermal_pause_attempts": self.thermal_pause_attempts,
            "last_thermal_event": self.last_thermal_event,
            "env_block_reason": self.env_block_reason,
            "stats": self.stats,
        }

        with open(self.heartbeat_file, 'w') as f:
            json.dump(heartbeat, f, indent=2)

    # =========================================================================
    # Signal Handling
    # =========================================================================

    def _check_stop_signal(self):
        """Check if brain has signaled this GPU to stop."""
        signal_file = self.signals_path / f"{self.name}.stop"
        if signal_file.exists():
            self.logger.info("Received stop signal from brain")
            signal_file.unlink()
            self.running = False

    def _check_abort_signal(self) -> bool:
        """Check if brain has signaled to abort a task."""
        signal_file = self.signals_path / f"{self.name}.abort"
        if signal_file.exists():
            try:
                with open(signal_file) as f:
                    abort_data = json.load(f)
                signal_file.unlink()

                target_task_id = abort_data.get("task_id", "")
                reason = abort_data.get("reason", "unknown")
                self.logger.warning(f"Abort signal: {reason} (task: {target_task_id[:8]})")

                # Kill the worker running this task
                for worker_id, info in self.active_workers.items():
                    if info["task"]["task_id"] == target_task_id:
                        self._kill_worker(worker_id, reason="abort_signal")
                        break

                return True
            except Exception as e:
                self.logger.error(f"Error processing abort signal: {e}")
                try:
                    signal_file.unlink()
                except Exception:
                    pass

        return False

    def _check_kill_signal(self) -> bool:
        """Check if brain has sent force kill signal."""
        signal_file = self.signals_path / f"{self.name}.kill"
        if signal_file.exists():
            try:
                with open(signal_file) as f:
                    kill_data = json.load(f)
                signal_file.unlink()

                target_task_id = kill_data.get("task_id", "")
                self.logger.error(f"FORCE KILL signal for task {target_task_id[:8]}")

                for worker_id, info in self.active_workers.items():
                    if info["task"]["task_id"] == target_task_id:
                        self._kill_worker(worker_id, reason="force_kill", force=True)
                        break

                return True
            except Exception as e:
                self.logger.error(f"Error processing kill signal: {e}")
                try:
                    signal_file.unlink()
                except Exception:
                    pass

        return False

    def _kill_worker(self, worker_id: str, reason: str = "", force: bool = False):
        """Kill a specific worker subprocess."""
        info = self.active_workers.get(worker_id)
        if not info:
            return

        proc = info["process"]
        try:
            if force:
                proc.kill()
            else:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

            self.logger.info(f"Killed worker {worker_id} ({reason})")
            self._remove_task_heartbeat(info["task"]["task_id"])
        except Exception as e:
            self.logger.debug(f"Worker already dead: {e}")

    def _kill_all_workers(self):
        """Kill all active worker subprocesses."""
        for worker_id in list(self.active_workers.keys()):
            self._kill_worker(worker_id, reason="shutdown", force=True)

    # =========================================================================
    # Meta Task Handling
    # =========================================================================

    def _handle_meta_task(self, task: Dict):
        """Handle a meta task directly (no worker needed)."""
        command = task.get("command", "")
        self.logger.info(f"Handling meta task: {command}")

        if command == "load_llm":
            self.load_model()
            result = {
                "success": self.model_loaded,
                "output": f"Model {'loaded' if self.model_loaded else 'failed to load'}",
                "model_loaded": self.model_loaded,
                "worker": self.name,
                "max_vram_used_mb": 0,
            }
        elif command == "unload_llm":
            self.unload_model()
            result = {
                "success": not self.model_loaded,
                "output": f"Model {'unloaded' if not self.model_loaded else 'failed to unload'}",
                "model_loaded": self.model_loaded,
                "worker": self.name,
                "max_vram_used_mb": 0,
            }
        else:
            result = {
                "success": False,
                "error": f"Unknown meta command: {command}",
                "worker": self.name,
                "max_vram_used_mb": 0,
            }

        # Meta tasks go straight to outbox (no worker subprocess)
        self.outbox.append(WorkerResult(
            task_id=task["task_id"],
            task=task,
            result=result,
            peak_vram_mb=0,
        ))

    # =========================================================================
    # Main Loop
    # =========================================================================

    def run(self):
        """Main GPU agent loop.

        External cycle (30s): filesystem I/O - heartbeat, flush outbox, claim tasks
        Internal cycle (5s): check worker status, update VRAM, check signals
        """
        self.logger.info(f"GPU agent {self.name} starting")

        try:
            # Start Ollama if this GPU has a port
            self.start_ollama()

            # Signal ready to launcher
            flag_dir = Path("/tmp/llm-orchestration-flags")
            flag_dir.mkdir(parents=True, exist_ok=True)
            (flag_dir / f"{self.name}.ready").touch()
            self.logger.info("GPU agent ready")

            last_external = 0

            while self.running:
                now = time.time()

                # --- Internal tick (every 5s) ---
                self._check_stop_signal()
                if not self.running:
                    break

                self._check_abort_signal()
                self._check_kill_signal()

                # Thermal safety check every internal tick
                gpu_stats_check = self._get_gpu_stats()
                self._update_thermal_pause_state(gpu_stats_check)
                if not self._check_thermal_safety(gpu_stats_check):
                    break

                # Collect any finished workers
                self._collect_finished_workers()

                # Update VRAM tracking for running workers
                self._update_worker_vram()

                # --- External tick (every 30s OR all workers done with pending outbox) ---
                all_done = len(self.active_workers) == 0 and len(self.outbox) > 0
                external_due = (now - last_external) >= EXTERNAL_HEARTBEAT_INTERVAL

                if external_due or all_done:
                    # 1. Flush completed results to filesystem
                    flushed = self._flush_outbox()

                    # 2. Write heartbeat
                    self._write_heartbeat()

                    # 3. Claim new tasks
                    claimed = self.claim_tasks()

                    # 4. Handle meta tasks directly, spawn workers for the rest
                    # Dedup: only handle one meta task per command per cycle
                    handled_meta_commands = set()
                    for task in claimed:
                        if task.get("task_class") == "meta":
                            cmd = task.get("command", "")
                            if cmd in handled_meta_commands:
                                self.logger.info(
                                    f"Dedup: skipping duplicate {cmd} meta task "
                                    f"{task['task_id'][:8]}")
                                # Complete it as no-op success
                                self.outbox.append(WorkerResult(
                                    task_id=task["task_id"], task=task,
                                    result={"success": True,
                                            "output": f"Dedup: {cmd} already handled this cycle",
                                            "worker": self.name, "max_vram_used_mb": 0},
                                    peak_vram_mb=0))
                            else:
                                handled_meta_commands.add(cmd)
                                self._handle_meta_task(task)
                        else:
                            self._spawn_worker(task)

                    last_external = time.time()

                    if flushed > 0 or claimed:
                        self.logger.info(
                            f"Cycle: flushed {flushed}, claimed {len(claimed)}, "
                            f"active {len(self.active_workers)}, "
                            f"budget {self._get_vram_budget()}MB"
                        )

                time.sleep(INTERNAL_POLL_INTERVAL)

        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
        except Exception as e:
            self.logger.error(f"GPU agent crashed: {e}", exc_info=True)
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up all resources on shutdown."""
        self.logger.info(
            f"GPU {self.name} shutting down - "
            f"completed {self.stats['tasks_completed']}, failed {self.stats['tasks_failed']}"
        )

        # Kill all active workers
        self._kill_all_workers()

        # Flush any remaining results
        self._flush_outbox()

        # Final heartbeat
        self._write_heartbeat()

        # Stop Ollama
        self.stop_ollama()

        self.logger.info("Cleanup complete")

    def stop(self):
        """Signal the GPU agent to stop."""
        self.running = False


# =============================================================================
# Entry Point
# =============================================================================

def main():
    gpu_agent = None

    def signal_handler(signum, frame):
        if gpu_agent:
            gpu_agent.stop()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description="GPU Agent")
    parser.add_argument("gpu_name", help="Name of the GPU (e.g., gpu-1)")
    default_config = Path(__file__).parent / "config.json"
    parser.add_argument("--config", default=str(default_config), help="Path to config file")
    args = parser.parse_args()

    gpu_agent = GPUAgent(args.config, args.gpu_name)
    try:
        gpu_agent.run()
    except KeyboardInterrupt:
        pass
    finally:
        if gpu_agent:
            gpu_agent.cleanup()


if __name__ == "__main__":
    main()
