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
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from multiprocessing import Queue
from pathlib import Path
from typing import Dict, List, Optional

from gpu_constants import (
    ACTIVE_WORK_HEARTBEAT_INTERVAL,
    DEFAULT_LLM_MIN_TIER,
    EXTERNAL_HEARTBEAT_INTERVAL,
    INTERNAL_POLL_INTERVAL,
    RUNTIME_STATE_COLD,
)
from gpu_core import GPUCoreMixin
from gpu_ollama import GPUOllamaMixin
from gpu_split import GPUSplitMixin
from gpu_state import GPUStateMixin
from gpu_tasks import GPUTaskMixin
from gpu_thermal import GPUThermalMixin
from gpu_workers import GPUWorkerMixin, WorkerResult

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)


class GPUAgent(
    GPUCoreMixin,
    GPUStateMixin,
    GPUOllamaMixin,
    GPUSplitMixin,
    GPUThermalMixin,
    GPUTaskMixin,
    GPUWorkerMixin,
):
    """GPU agent that owns a single GPU and coordinates task execution.

    Inherits functionality from specialized mixins:
    - GPUCoreMixin: Config loading, utility methods
    - GPUStateMixin: Runtime state machine
    - GPUOllamaMixin: Ollama server and model management
    - GPUSplitMixin: Split GPU runtime coordination
    - GPUThermalMixin: Thermal safety and resource management
    - GPUTaskMixin: Task claiming and meta task handling
    - GPUWorkerMixin: Worker subprocess management
    """

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
        self.model_catalog = self._load_model_catalog(config_dir)
        self.model_tier_by_id = self._build_model_tier_map(self.model_catalog)
        self.model_meta_by_id = self._build_model_meta_map(self.model_catalog)

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
        self.split_state_dir = self.signals_path / "split_llm"
        self.split_state_dir.mkdir(parents=True, exist_ok=True)
        self.model_load_lock_path = self.signals_path / "model_load.global.lock"
        self.model_load_owner_path = self.signals_path / "model_load.global.json"

        # Logs
        self.log_path = self.shared_path / "logs"
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.training_log = self.log_path / "training_samples.jsonl"

        # Ollama config
        self.model = self.gpu_config.get("model")
        self.model_tier = int(self.model_tier_by_id.get(self.model, DEFAULT_LLM_MIN_TIER))
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
        self.state = "cold"       # "cold" or "hot" (legacy, kept for compatibility)
        self.model_loaded = False
        self.loaded_model: Optional[str] = None
        self.loaded_tier: int = 0
        self.runtime_placement: str = "single_gpu"
        self.runtime_group_id: Optional[str] = None
        self.runtime_port: Optional[int] = self.port
        self.runtime_ollama_url: Optional[str] = f"http://localhost:{self.port}" if self.port else None
        self.split_runtime_owner: bool = False
        self.split_runtime_process: Optional[subprocess.Popen] = None
        self.last_split_runtime_error: str = ""
        self.split_runtime_owner_meta_path: Optional[Path] = None
        self.split_runtime_invariant_failures: int = 0
        self.ollama_process: Optional[subprocess.Popen] = None

        # Runtime state machine (authoritative)
        self.runtime_state: str = RUNTIME_STATE_COLD
        self.runtime_state_updated_at: str = datetime.now().isoformat()
        self.runtime_transition_task_id: Optional[str] = None
        self.runtime_transition_phase: Optional[str] = None
        self.runtime_error_code: Optional[str] = None
        self.runtime_error_detail: Optional[str] = None

        # Worker tracking
        # active_workers: worker_id -> {process, task, vram_estimate, pid, started_at}
        self.active_workers: Dict[str, Dict] = {}
        # Outbox: completed results waiting to be written to filesystem on next cycle
        self.outbox: List[WorkerResult] = []
        # Internal status queue: workers post updates here every INTERNAL_POLL_INTERVAL
        self.status_queue: Queue = Queue()
        self.active_meta_task: Optional[Dict] = None
        self.last_meta_heartbeat: float = 0.0

        # VRAM budget tracking
        self.claimed_vram = 0  # Sum of VRAM estimates for currently running workers

        # Stats
        self.stats = {"tasks_completed": 0, "tasks_failed": 0}
        self.last_external_heartbeat = 0
        self.env_check_cache: Dict[str, Dict] = {}
        self.env_block_reason: Optional[str] = None

        # Ollama health tracking
        self.ollama_healthy = True
        self.ollama_consecutive_failures = 0
        self.ollama_health_threshold = 6  # Restart after ~30s of consecutive failures
        self.ollama_circuit_breaker = 8   # Stop claiming LLM tasks before forced restart

        # Thermal/constrained-state tracking for explicit stall visibility
        self.thermal_pause_active = False
        self.thermal_pause_reasons: List[str] = []
        self.last_thermal_event: Optional[Dict] = None
        self.thermal_pause_until: float = 0.0
        self.thermal_pause_current_seconds: int = 0
        self.thermal_pause_attempts: int = 0

        # Thermal incident tracking (for brain-level recovery coordination)
        # An "incident" is a sustained overheat condition requiring escalation
        self.thermal_overheat_started_at: Optional[float] = None  # Unix timestamp when overheat began
        self.thermal_overheat_sustained_seconds: int = 0  # How long continuously overheated
        self.thermal_overheat_incident_id: Optional[str] = None  # Unique ID for this incident
        self.thermal_recovery_reset_count: int = 0  # Resets issued by brain during this incident
        self.thermal_recovery_last_reset_at: Optional[float] = None  # Last reset timestamp

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
                # Publish full task_id so brain orphan recovery can match exactly.
                "task_id": info["task"]["task_id"],
                "task_class": info["task"].get("task_class", "cpu"),
                "task_name": info["task"].get("name", ""),
                "vram_estimate_mb": info["vram_estimate"],
                "peak_vram_mb": info["peak_vram_mb"],
                "pid": info["pid"],
                "started_at": datetime.fromtimestamp(info["started_at"]).isoformat(),
            })
        if self.active_meta_task:
            active_task_info.append({
                "worker_id": f"{self.name}-meta",
                "task_id": str(self.active_meta_task.get("task_id", "")),
                "task_class": "meta",
                "task_name": str(self.active_meta_task.get("task_name", "meta")),
                "vram_estimate_mb": 0,
                "peak_vram_mb": 0,
                "pid": os.getpid(),
                "started_at": datetime.fromtimestamp(
                    float(self.active_meta_task.get("started_at", time.time()))
                ).isoformat(),
                "phase": str(self.active_meta_task.get("phase", "")),
            })

        cpu_temp = self._get_cpu_temp()

        # Capability-ready: runtime is ready to accept work and model is loaded
        capability_ready = (
            self._is_runtime_ready()
            and self.model_loaded
            and self.ollama_healthy
        )

        heartbeat = {
            "gpu_id": self.gpu_id,
            "name": self.name,
            "state": self.state,
            # Authoritative runtime state machine fields
            "runtime_state": self.runtime_state,
            "runtime_state_updated_at": self.runtime_state_updated_at,
            "runtime_transition_task_id": self.runtime_transition_task_id,
            "runtime_transition_phase": self.runtime_transition_phase,
            "runtime_error_code": self.runtime_error_code,
            "runtime_error_detail": self.runtime_error_detail,
            # Effective capability fields (for capability-based task routing)
            # These describe what this runtime CAN do, independent of placement
            "capability_ready": capability_ready,
            "effective_model_id": self.loaded_model if self.model_loaded else None,
            "effective_tier": self.loaded_tier if self.model_loaded else 0,
            "effective_context_tokens": self.worker_num_ctx if self.model_loaded else 0,
            # Legacy and derived fields
            "model_loaded": self.model_loaded,
            "loaded_model": self.loaded_model,
            "loaded_tier": self.loaded_tier,
            "configured_model": self.model,
            "configured_model_tier": self.model_tier,
            "runtime_placement": self.runtime_placement,
            "runtime_group_id": self.runtime_group_id,
            "split_runtime_owner": self.split_runtime_owner,
            "runtime_port": self.runtime_port,
            "runtime_ollama_url": self.runtime_ollama_url,
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
            "active_workers": len(self.active_workers) + (1 if self.active_meta_task else 0),
            "active_tasks": active_task_info,
            "meta_task_active": bool(self.active_meta_task),
            "meta_task_id": str(self.active_meta_task.get("task_id", "")) if self.active_meta_task else None,
            "meta_task_phase": str(self.active_meta_task.get("phase", "")) if self.active_meta_task else None,
            "thermal_constrained": constrained,
            "thermal_reasons": constrained_reasons,
            "thermal_pause_active": self.thermal_pause_active,
            "thermal_pause_reasons": self.thermal_pause_reasons,
            "thermal_pause_until": datetime.fromtimestamp(self.thermal_pause_until).isoformat() if self.thermal_pause_until else None,
            "thermal_pause_remaining_seconds": max(0, int(self.thermal_pause_until - time.time())) if self.thermal_pause_active else 0,
            "thermal_pause_current_seconds": self.thermal_pause_current_seconds,
            "thermal_pause_attempts": self.thermal_pause_attempts,
            "last_thermal_event": self.last_thermal_event,
            # Thermal incident tracking (for brain-level recovery coordination)
            "thermal_overheat_started_at": datetime.fromtimestamp(self.thermal_overheat_started_at).isoformat() if self.thermal_overheat_started_at else None,
            "thermal_overheat_sustained_seconds": self.thermal_overheat_sustained_seconds,
            "thermal_overheat_incident_id": self.thermal_overheat_incident_id,
            "thermal_recovery_reset_count": self.thermal_recovery_reset_count,
            "thermal_recovery_last_reset_at": datetime.fromtimestamp(self.thermal_recovery_last_reset_at).isoformat() if self.thermal_recovery_last_reset_at else None,
            "env_block_reason": self.env_block_reason,
            "stats": self.stats,
        }

        with open(self.heartbeat_file, 'w') as f:
            json.dump(heartbeat, f, indent=2)

    def run(self):
        """Main GPU agent loop.

        External cycle (30s): filesystem I/O - heartbeat, flush outbox, claim tasks
        Internal cycle (5s): check worker status, update VRAM, check signals
        """
        self.logger.info(f"GPU agent {self.name} starting")
        crash_exception = None

        try:
            # Start Ollama if this GPU has a port
            self.start_ollama()
            self._reclaim_orphan_split_ports_on_startup()

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
                self._service_split_reservations()
                self._check_split_runtime_invariants()

                # Auto-recovery check: if wedged with no active work, trigger recovery
                should_recover, recover_reason = self._should_trigger_auto_recovery()
                if should_recover:
                    placement = str(getattr(self, 'runtime_placement', '')).strip()
                    if placement == "split_gpu" or recover_reason == "wedged_split_idle":
                        # Split runtime recovery
                        group_id = getattr(self, 'runtime_group_id', None) or ""
                        split_port = getattr(self, 'runtime_port', None)
                        self._run_auto_recovery_workflow(group_id, split_port, recover_reason)
                    else:
                        # Single-runtime recovery - simpler path
                        self._run_single_runtime_recovery(recover_reason)

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
                heartbeat_interval = (
                    ACTIVE_WORK_HEARTBEAT_INTERVAL
                    if self._has_active_work()
                    else EXTERNAL_HEARTBEAT_INTERVAL
                )
                external_due = (now - last_external) >= heartbeat_interval

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
                            cmd = str(task.get("command", ""))
                            dedup_key = json.dumps({
                                "command": cmd,
                                "target_model": task.get("target_model"),
                                "group_id": task.get("group_id"),
                            }, sort_keys=True)
                            if dedup_key in handled_meta_commands:
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
                                handled_meta_commands.add(dedup_key)
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
            crash_exception = None
        except Exception as e:
            self.logger.error(f"GPU agent crashed: {e}", exc_info=True)
            crash_exception = e
        finally:
            self.cleanup()
            if crash_exception:
                raise crash_exception  # Re-raise so supervisor sees non-zero exit

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

        # Stop runtimes BEFORE final heartbeat so heartbeat reflects actual state
        self.stop_ollama()
        self._stop_split_runtime()

        # Final heartbeat (now reflects stopped runtime)
        self._write_heartbeat()

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
        gpu_agent.run()  # run() owns cleanup in its finally block
    except KeyboardInterrupt:
        pass  # Graceful shutdown handled by run()
    except Exception:
        sys.exit(1)  # Crash already logged in run(), exit non-zero for supervisor


if __name__ == "__main__":
    main()
