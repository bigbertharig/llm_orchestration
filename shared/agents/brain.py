#!/usr/bin/env python3
"""
Brain Agent - Coordinates tasks and directs workers.
Runs on multiple GPUs with the larger model for reasoning.

Architecture:
- Private task list: Tasks waiting for dependencies (shared/brain/private_tasks/)
- Public queue: Tasks ready for workers (shared/tasks/queue/)
- Dependency-based release: Tasks move to public when depends_on tasks complete
- Resource monitoring: Smart worker start/stop based on queue composition

Usage:
  python brain.py
  python brain.py --config /path/to/config.json
"""

import argparse
import ast
import json
import os
import re
import shlex
import socket
import sys
import time
import traceback
import uuid
import logging
import requests
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from filelock import FileLock, Timeout

from brain_goal import BrainGoalMixin
from brain_constants import DEFAULT_LLM_MIN_TIER
from brain_core import BrainCoreMixin
from brain_dispatch import BrainDispatchMixin
from brain_plan import BrainPlanMixin
from brain_failures import BrainFailureMixin
from brain_monitor import BrainMonitorMixin
from brain_tasks import BrainTaskQueueMixin
from brain_resources import BrainResourceMixin

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)

# =============================================================================
# Task Classification
# Task classes determine which workers can claim them:
#   - cpu: Any worker (including RPi)
#   - script: GPU workers only (no LLM needed, uses GPU for compute)
#   - llm: GPU workers only (needs LLM model loaded)
#   - brain: Brain-only tasks (always executed by brain, never by workers)
#   - meta: Model load/unload tasks (inserted by brain, claimed by GPU workers)
# =============================================================================
class Brain(BrainGoalMixin, BrainCoreMixin, BrainPlanMixin, BrainTaskQueueMixin, BrainMonitorMixin, BrainResourceMixin, BrainFailureMixin, BrainDispatchMixin):
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config(config_path)
        self.brain_config = self.config["brain"]
        self.name = self.brain_config["name"]
        self.logger = logging.getLogger(self.name)

        # Set independent logging level for brain
        # Use BRAIN_LOG_LEVEL env var, default to INFO
        brain_log_level = os.environ.get("BRAIN_LOG_LEVEL", "INFO").upper()
        try:
            level = getattr(logging, brain_log_level)
            self.logger.setLevel(level)
        except AttributeError:
            self.logger.setLevel(logging.INFO)
            self.logger.warning(f"Invalid BRAIN_LOG_LEVEL '{brain_log_level}', defaulting to INFO")

        # Resolve paths relative to config file location
        config_dir = self.config_path.parent
        shared_path = Path(self.config["shared_path"])
        if not shared_path.is_absolute():
            shared_path = (config_dir / shared_path).resolve()
        self.shared_path = shared_path
        self.model_catalog = self._load_model_catalog(config_dir)
        self.model_tier_by_id = self._build_model_tier_map(self.model_catalog)
        self.model_meta_by_id = self._build_model_meta_map(self.model_catalog)
        self.default_llm_min_tier = int(
            self.model_catalog.get("default_llm_min_tier", DEFAULT_LLM_MIN_TIER)
        )

        # Public task queue (workers see this)
        self.queue_path = self.shared_path / "tasks" / "queue"
        self.processing_path = self.shared_path / "tasks" / "processing"
        self.complete_path = self.shared_path / "tasks" / "complete"
        self.failed_path = self.shared_path / "tasks" / "failed"

        # Brain state and private task list
        self.brain_path = self.shared_path / "brain"
        self.heartbeats_path = self.shared_path / "heartbeats"
        self.private_tasks_path = self.brain_path / "private_tasks"
        self.escalations_path = self.brain_path / "escalations"
        self.state_file = self.brain_path / "state.json"
        self.brain_heartbeat_file = self.brain_path / "heartbeat.json"
        self.brain_heartbeat_file_unified = self.heartbeats_path / "brain.json"

        # Ensure directories exist
        for path in [self.queue_path, self.processing_path, self.complete_path,
                     self.failed_path, self.brain_path, self.heartbeats_path, self.private_tasks_path,
                     self.escalations_path]:
            path.mkdir(parents=True, exist_ok=True)
        self.brain_lock_file = self.brain_path / "brain_agent.lock"
        self.brain_lock = FileLock(str(self.brain_lock_file), timeout=0)
        try:
            self.brain_lock.acquire()
        except Timeout:
            print(
                f"ERROR: Another brain instance already holds lock: {self.brain_lock_file}",
                file=sys.stderr,
            )
            sys.exit(1)

        self.model = self.brain_config["model"]
        self.gpus = self.brain_config["gpus"]
        self.api_url = f"{self.config['ollama_host']}/api/generate"
        self.brain_keep_alive = str(self.config.get("brain_keep_alive", "-1"))
        self.brain_num_ctx = int(self.config.get("brain_context_tokens", 8192))

        self.poll_interval = self.config["timeouts"]["poll_interval_seconds"]
        self.think_timeout = self.config["timeouts"]["brain_think_seconds"]

        self.gpu_agents = {g["name"]: g for g in self.config.get("gpus", [])}
        self.brain_only = len(self.gpu_agents) == 0
        self.ollama_process: Optional[subprocess.Popen] = None
        self.running = True

        # Track active batches: batch_id -> {plan, started_at, config}
        self.active_batches: Dict[str, Dict] = {}

        # Decision log file
        self.log_path = self.shared_path / "logs"
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.decision_log = self.log_path / "brain_decisions.log"

        # Resource monitoring
        self.last_monitor_check = 0
        self.monitor_interval = 30  # seconds
        # Rail safety: cap number of hot worker GPUs at once.
        self.max_hot_workers = int(self.config.get("max_hot_workers", 1))
        self.gpu_pids: Dict[str, int] = {}
        self.gpu_miss_count: Dict[str, int] = {}  # Track consecutive misses for 3-miss tolerance
        self.signals_path = self.shared_path / "signals"
        self.signals_path.mkdir(parents=True, exist_ok=True)

        # P1: Track load_llm requests to detect when workers don't pick them up
        self.load_llm_requests: Dict[str, Dict] = {}
        # Throttle repeated resource commands so balancing changes happen gradually.
        self.resource_task_cooldown_seconds = int(
            self.config.get("timeouts", {}).get("resource_task_cooldown_seconds", 45)
        )
        self.last_resource_task_at: Dict[str, datetime] = {}
        # Infrastructure escalation tracking for missing GPU agents
        self.gpu_missing_escalations: Dict[str, Dict[str, Any]] = {}
        # Heartbeat/missing thresholds (4 missed 30s heartbeats = 120s)
        self.heartbeat_stale_seconds = int(self.monitor_interval * 4)
        self.missing_gpu_miss_threshold = 4
        self.task_no_progress_thresholds = {
            "script": 600,
            "llm": 420,
            "cpu": 900,
            "meta": 180,
        }
        self.force_kill_requeue_seconds = 60
        self.thermal_wait_logged: set[str] = set()
        self.last_thermal_wait_count = 0
        # Track dependency auto-fix attempts (missing Python modules, etc.)
        self.dependency_fix_attempts: Dict[str, Dict[str, Any]] = {}
        # Incident state machine for worker->brain->cloud escalation
        self.incidents: Dict[str, Dict[str, Any]] = {}
        self.max_brain_fix_attempts = int(
            self.config.get("retry_policy", {}).get("max_brain_fix_attempts", 3)
        )

        # Verify core/ security before starting
        self._verify_core_security()

        # Load existing brain state
        self._load_brain_state()

    def clear_resolved_failures(self):
        """
        Auto-clear stale failed entries when the same task_id later succeeds.

        This supports escalation/re-submit flows where a task can be retried
        and completed after being marked failed/blocked earlier.
        """
        successful_task_ids = set()

        for task_file in self.complete_path.glob("*.json"):
            try:
                with open(task_file) as f:
                    task = json.load(f)
                task_id = task.get("task_id")
                if not task_id:
                    continue

                # Treat explicit success as authoritative.
                # If result is absent, keep backward-compat behavior and consider complete as success.
                result = task.get("result")
                is_success = (
                    task.get("status") == "complete"
                    and (
                        not isinstance(result, dict)
                        or result.get("success", True) is True
                    )
                )
                if is_success:
                    successful_task_ids.add(task_id)
            except Exception:
                continue

        if not successful_task_ids:
            return

        cleared = 0
        for task_file in self.failed_path.glob("*.json"):
            try:
                with open(task_file) as f:
                    failed_task = json.load(f)
            except Exception:
                continue

            task_id = failed_task.get("task_id")
            if not task_id or task_id not in successful_task_ids:
                continue

            # Mark linked incident resolved (if present) before clearing the failed file.
            incident_id = failed_task.get("incident_id")
            resolved_incident = False
            if incident_id:
                for incident in self.incidents.values():
                    if incident.get("incident_id") == incident_id:
                        incident["resolved"] = True
                        incident["resolved_at"] = datetime.now().isoformat()
                        incident["resolution"] = "task_completed_successfully"
                        incident["updated_at"] = datetime.now().isoformat()
                        incident.setdefault("history", []).append({
                            "at": datetime.now().isoformat(),
                            "event": "auto_cleared_after_success",
                            "task_id": task_id
                        })
                        resolved_incident = True
                        break

            # Fallback: resolve by deterministic incident key for this task payload.
            if not resolved_incident:
                key = self._incident_key(failed_task)
                incident = self.incidents.get(key)
                if incident:
                    incident["resolved"] = True
                    incident["resolved_at"] = datetime.now().isoformat()
                    incident["resolution"] = "task_completed_successfully"
                    incident["updated_at"] = datetime.now().isoformat()
                    incident.setdefault("history", []).append({
                        "at": datetime.now().isoformat(),
                        "event": "auto_cleared_after_success",
                        "task_id": task_id
                    })

            try:
                task_file.unlink()
                cleared += 1
                self.log_decision(
                    "FAILED_CLEARED",
                    f"Cleared failed task after successful completion: {failed_task.get('name', task_id)}",
                    {"task_id": task_id, "failed_file": str(task_file)}
                )
            except Exception:
                continue

        if cleared:
            self._save_brain_state()

    # =========================================================================
    # Task Creation
    # =========================================================================

    def run(self, model_preloaded: bool = False):
        """Main brain loop."""
        import time as _time
        start_time = _time.time()

        self.logger.info(f"Starting brain: {self.name}")
        self.logger.info(f"Model: {self.model}, GPUs: {self.gpus}")
        self.logger.info(f"GPU agents: {list(self.gpu_agents.keys())}")

        try:
            self.logger.info("[STARTUP] Phase 1: Connecting to Ollama...")
            self.start_ollama()
            self.logger.info(f"[STARTUP] Ollama ready ({_time.time() - start_time:.1f}s)")

            # Preload model with progress logging (skip if already loaded)
            if model_preloaded:
                self.logger.info(f"[STARTUP] Phase 2: Model {self.model} already preloaded, skipping warmup")
            else:
                self.logger.info(f"[STARTUP] Phase 2: Preloading model {self.model}...")
                preload_start = _time.time()
                self.think("Hello, are you ready?")
                preload_time = _time.time() - preload_start
                self.logger.info(f"[STARTUP] Model loaded ({preload_time:.1f}s)")

            total_startup = _time.time() - start_time
            self.logger.info(f"[STARTUP] Brain ready (total: {total_startup:.1f}s)")
            self._write_brain_heartbeat()

            # Signal ready to launcher
            flag_dir = Path("/tmp/llm-orchestration-flags")
            flag_dir.mkdir(parents=True, exist_ok=True)
            (flag_dir / f"{self.name}.ready").touch()

            while self.running:
                # 1. Check for brain tasks (execute_plan, decide, brain-targeted shell)
                self.claim_brain_tasks()

                # 2. Check and release tasks whose dependencies are met
                self.check_and_release_tasks()

                # 2b. Process goal-driven plan validations
                self._process_goal_validations()

                # 2c. Reconcile batch completion for all active batches.
                # Prevent stale active_batches entries when no new task releases occur.
                for _batch_id in list(self.active_batches.keys()):
                    self._check_batch_completion(_batch_id)

                # 3. Handle failed tasks (retry logic)
                self.handle_failed_tasks()

                # 3b. Auto-clear stale failures when same task_id later succeeds
                self.clear_resolved_failures()

                # 4. Monitor GPU and worker health
                self.monitor_system()

                # Save state periodically
                self._save_brain_state()
                self._write_brain_heartbeat()

                time.sleep(self.poll_interval)

        except KeyboardInterrupt:
            self.logger.info("Shutting down...")
        finally:
            self._save_brain_state()
            self._write_brain_heartbeat()
            self.stop_ollama()

    def stop(self):
        self.running = False


def main():
    parser = argparse.ArgumentParser(description="Brain Agent")
    default_config = Path(__file__).parent / "config.json"
    parser.add_argument("--config", default=str(default_config),
                        help="Path to config file")
    parser.add_argument("--model-preloaded", action="store_true",
                        help="Skip model warmup (model already loaded in Ollama)")
    args = parser.parse_args()

    brain = Brain(args.config)
    brain.run(model_preloaded=args.model_preloaded)


if __name__ == "__main__":
    main()
