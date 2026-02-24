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
VALID_TASK_CLASSES = ['cpu', 'script', 'llm', 'brain', 'meta']
VALID_VRAM_POLICIES = ['default', 'infer', 'fixed']
DEFAULT_LLM_MIN_TIER = 1
PRIORITY_TIER_TO_VALUE = {
    "low": 3,
    "normal": 5,
    "high": 8,
    "urgent": 10,
}


class Brain(BrainGoalMixin, BrainPlanMixin, BrainTaskQueueMixin, BrainMonitorMixin, BrainResourceMixin, BrainFailureMixin):
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

    def _load_config(self, config_path: str) -> dict:
        if not Path(config_path).exists():
            print(f"ERROR: Config file not found: {config_path}")
            print(f"  Run 'python setup.py' to scan hardware and generate config.json.")
            sys.exit(1)
        with open(config_path) as f:
            return json.load(f)

    def _load_model_catalog(self, config_dir: Path) -> Dict[str, Any]:
        """Load model capability catalog used for llm tier scheduling."""
        raw_path = str(self.config.get("model_catalog_path", "models.catalog.json")).strip()
        catalog_path = Path(raw_path)
        if not catalog_path.is_absolute():
            catalog_path = (config_dir / catalog_path).resolve()

        if not catalog_path.exists():
            print(f"ERROR: Model catalog not found: {catalog_path}")
            print("  Set model_catalog_path in config.json or create models.catalog.json.")
            sys.exit(1)

        try:
            with open(catalog_path, "r", encoding="utf-8") as f:
                catalog = json.load(f)
        except Exception as exc:
            print(f"ERROR: Failed to load model catalog {catalog_path}: {exc}")
            sys.exit(1)

        models = catalog.get("models", [])
        if not isinstance(models, list) or not models:
            print(f"ERROR: Model catalog {catalog_path} must contain a non-empty 'models' list.")
            sys.exit(1)

        for item in models:
            model_id = str(item.get("id", "")).strip()
            tier = item.get("tier")
            if not model_id:
                print(f"ERROR: Invalid model catalog entry missing 'id': {item}")
                sys.exit(1)
            if not isinstance(tier, int) or tier < 1:
                print(f"ERROR: Invalid tier for model '{model_id}': {tier}")
                sys.exit(1)

        self.logger.info(f"Loaded model catalog from {catalog_path}")
        return catalog

    def _build_model_tier_map(self, catalog: Dict[str, Any]) -> Dict[str, int]:
        tier_map: Dict[str, int] = {}
        for item in catalog.get("models", []):
            model_id = str(item.get("id", "")).strip()
            tier = int(item.get("tier", DEFAULT_LLM_MIN_TIER))
            if model_id:
                tier_map[model_id] = max(1, tier)
        return tier_map

    def _build_model_meta_map(self, catalog: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        model_meta: Dict[str, Dict[str, Any]] = {}
        for item in catalog.get("models", []):
            model_id = str(item.get("id", "")).strip()
            if not model_id:
                continue
            placement = str(item.get("placement", "single_gpu") or "single_gpu").strip()
            split_groups: List[Dict[str, Any]] = []

            if isinstance(item.get("split_groups"), list):
                for g in item.get("split_groups", []):
                    if not isinstance(g, dict):
                        continue
                    members = [str(m).strip() for m in g.get("members", []) if str(m).strip()]
                    if len(members) < 2:
                        continue
                    group_id = str(g.get("id") or f"group_{'_'.join(sorted(members))}").strip()
                    try:
                        port = int(g.get("port"))
                    except Exception:
                        port = None
                    split_groups.append({"id": group_id, "members": members, "port": port})

            # Backward-compatible pair schema.
            if not split_groups and isinstance(item.get("allowed_pairs"), list):
                next_port = 11440
                for pair in item.get("allowed_pairs", []):
                    if not isinstance(pair, list):
                        continue
                    members = [str(m).strip() for m in pair if str(m).strip()]
                    if len(members) < 2:
                        continue
                    group_id = f"group_{'_'.join(sorted(members))}"
                    split_groups.append({"id": group_id, "members": members, "port": next_port})
                    next_port += 1

            model_meta[model_id] = {
                "tier": int(self.model_tier_by_id.get(model_id, DEFAULT_LLM_MIN_TIER)),
                "placement": placement,
                "split_groups": split_groups,
            }
        return model_meta

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

    def _load_brain_state(self):
        """Load brain state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    state = json.load(f)
                    self.active_batches = state.get("active_batches", {})
                    # Restore load_llm_requests (convert timestamps back to datetime)
                    saved_requests = state.get("load_llm_requests", {})
                    for task_id, req in saved_requests.items():
                        req['created_at'] = datetime.fromisoformat(req['created_at'])
                        self.load_llm_requests[task_id] = req
                    saved_resource_times = state.get("last_resource_task_at", {})
                    for command, ts in saved_resource_times.items():
                        self.last_resource_task_at[command] = datetime.fromisoformat(ts)
                    self.incidents = state.get("incidents", {})
                    self.gpu_missing_escalations = state.get("gpu_missing_escalations", {})
                    self.logger.info(
                        f"Loaded state: {len(self.active_batches)} active batches, "
                        f"{len(self.load_llm_requests)} pending load_llm requests")
            except Exception as e:
                self.logger.warning(f"Failed to load state: {e}")
                self.active_batches = {}
                self.incidents = {}
                self.gpu_missing_escalations = {}

    def _save_brain_state(self):
        """Persist brain state to disk."""
        # Serialize load_llm_requests (convert datetime to ISO string)
        serializable_requests = {}
        for task_id, req in self.load_llm_requests.items():
            serializable_requests[task_id] = {
                'created_at': req['created_at'].isoformat(),
                'gpus_needed': req['gpus_needed']
            }
        state = {
            "pid": os.getpid(),
            "started_at": datetime.now().isoformat(),
            "status": "active",
            "active_batches": self.active_batches,
            "load_llm_requests": serializable_requests,
            "last_resource_task_at": {
                command: ts.isoformat()
                for command, ts in self.last_resource_task_at.items()
            },
            "incidents": self.incidents,
            "gpu_missing_escalations": self.gpu_missing_escalations
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def _write_brain_heartbeat(self):
        """Publish brain heartbeat for dashboard/status consumers."""
        try:
            hb = {
                "worker_type": "brain",
                "name": "brain",
                "hostname": socket.gethostname(),
                "state": "online" if self.running else "stopping",
                "last_updated": datetime.now().isoformat(timespec="seconds"),
                "brain_gpus": self.gpus,
                "active_batches": len(self.active_batches),
                "brain_pids": {"pid": os.getpid()},
            }
            with open(self.brain_heartbeat_file, "w", encoding="utf-8") as f:
                json.dump(hb, f, indent=2)
            with open(self.brain_heartbeat_file_unified, "w", encoding="utf-8") as f:
                json.dump(hb, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to write brain heartbeat: {e}")

    # =========================================================================
    # Logging
    # =========================================================================

    def log_decision(self, decision_type: str, message: str, details: dict = None):
        """Log a brain decision for monitoring."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": decision_type,
            "message": message,
            "details": details or {}
        }

        with open(self.decision_log, 'a') as f:
            f.write(json.dumps(entry) + "\n")

        self.logger.info(f"[{decision_type}] {message}")

    def emit_cloud_escalation(self, escalation_type: str, title: str, details: dict,
                              source_task: Dict[str, Any] = None) -> str:
        """
        Write a cloud escalation request for external brain/gateway pickup.
        Returns escalation_id.
        """
        now = datetime.now()
        escalation_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

        payload = {
            "escalation_id": escalation_id,
            "created_at": now.isoformat(),
            "status": "pending",
            "target": "cloud_brain",
            "source": "local_brain",
            "brain_name": self.name,
            "hostname": socket.gethostname(),
            "type": escalation_type,
            "title": title,
            "details": details or {},
            "recommended_action": "diagnose_and_resubmit_plan"
        }

        if source_task:
            payload["source_task"] = {
                "task_id": source_task.get("task_id"),
                "type": source_task.get("type"),
                "name": source_task.get("name"),
                "batch_id": source_task.get("batch_id"),
                "plan_path": source_task.get("plan_path"),
                "config": source_task.get("config", {})
            }

        out_file = self.escalations_path / f"{escalation_id}.json"
        with open(out_file, 'w') as f:
            json.dump(payload, f, indent=2)

        self.log_decision(
            "ESCALATION",
            f"Escalated to cloud brain: {title}",
            {"escalation_id": escalation_id, "type": escalation_type, "file": str(out_file)}
        )
        return escalation_id

    def _build_execute_plan_escalation_context(self, plan_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Build context package to guide cloud-brain investigation."""
        plan_name = Path(str(plan_path)).name if plan_path else ""
        candidate_plan_dirs = []

        if plan_path:
            candidate_plan_dirs.append(str(Path(plan_path)))
        if plan_name:
            candidate_plan_dirs.append(str(self.shared_path / "plans" / plan_name))

        # De-duplicate while preserving order
        unique_plan_dirs = []
        seen = set()
        for p in candidate_plan_dirs:
            if p not in seen:
                seen.add(p)
                unique_plan_dirs.append(p)

        existing_plan_md = []
        for p in unique_plan_dirs:
            plan_md = Path(p) / "plan.md"
            if plan_md.exists():
                existing_plan_md.append(str(plan_md))

        resume_batch_id = str(config.get("RESUME_BATCH_ID", "")).strip() if isinstance(config, dict) else ""
        resume_manifest_candidates = []
        if resume_batch_id:
            for p in unique_plan_dirs:
                resume_manifest_candidates.append(
                    str(Path(p) / "history" / resume_batch_id / "manifest.json")
                )

        return {
            "dig_order": [
                "1) Verify a candidate plan path contains plan.md",
                "2) If RUN_MODE=resume, verify resume manifest exists",
                "3) Check brain_decisions.log and task result payload",
                "4) Resubmit execute_plan with corrected path/config"
            ],
            "requested_plan_path": plan_path,
            "candidate_plan_dirs": unique_plan_dirs,
            "existing_plan_md": existing_plan_md,
            "resume_batch_id": resume_batch_id or None,
            "resume_manifest_candidates": resume_manifest_candidates,
            "key_files": {
                "brain_decisions_log": str(self.decision_log),
                "brain_state": str(self.state_file),
                "task_complete_dir": str(self.complete_path),
                "task_failed_dir": str(self.failed_path),
                "queue_dir": str(self.queue_path)
            }
        }

    def _build_resubmit_payload_for_abandoned_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Create a safe task skeleton cloud brain can resubmit after verification."""
        payload = {
            "type": task.get("type"),
            "executor": task.get("executor", "worker"),
            "batch_id": task.get("batch_id"),
            "name": task.get("name"),
            "task_class": task.get("task_class"),
            "command": task.get("command"),
            "depends_on": task.get("depends_on", []),
            "priority": task.get("priority", 5),
            "retry_count": 0,
            "attempts": 0,
            "workers_attempted": []
        }

        # Include plan execution fields when relevant.
        if task.get("type") == "execute_plan":
            payload["plan_path"] = task.get("plan_path")
            payload["config"] = task.get("config", {})

        return payload

    def _cleanup_stale_plan_batches(self, plan_dir: Path) -> Dict[str, int]:
        """
        For RUN_MODE=fresh, clean stale orchestration artifacts for this plan only.

        Multiple plans may run in parallel. This cleanup is intentionally scoped to
        older active batches with the same plan_dir, so unrelated plans are untouched.
        """
        plan_dir_resolved = str(plan_dir.resolve())
        stale_batch_ids = [
            bid for bid, meta in self.active_batches.items()
            if str(meta.get("plan_dir", "")) == plan_dir_resolved
        ]

        if not stale_batch_ids:
            return {"stale_batches": 0, "private_removed": 0, "queue_removed": 0, "processing_removed": 0, "orph_lock_removed": 0}

        removed_private = 0
        removed_queue = 0
        removed_processing = 0

        def _remove_batch_tasks(path: Path) -> int:
            removed = 0
            for task_file in path.glob("*.json"):
                try:
                    with open(task_file) as f:
                        task = json.load(f)
                    if task.get("batch_id") in stale_batch_ids:
                        task_file.unlink(missing_ok=True)
                        removed += 1
                except Exception:
                    continue
            return removed

        removed_private = _remove_batch_tasks(self.private_tasks_path)
        removed_queue = _remove_batch_tasks(self.queue_path)
        removed_processing = _remove_batch_tasks(self.processing_path)

        # Remove orphan lock files in queue/processing (safe cleanup).
        orphan_lock_removed = 0
        for path in [self.queue_path, self.processing_path]:
            for lock_file in path.glob("*.lock"):
                try:
                    json_file = Path(str(lock_file)[:-5])  # strip '.lock'
                    if not json_file.exists():
                        lock_file.unlink(missing_ok=True)
                        orphan_lock_removed += 1
                except Exception:
                    continue

        for bid in stale_batch_ids:
            self.active_batches.pop(bid, None)

        self._save_brain_state()
        return {
            "stale_batches": len(stale_batch_ids),
            "private_removed": removed_private,
            "queue_removed": removed_queue,
            "processing_removed": removed_processing,
            "orph_lock_removed": orphan_lock_removed,
        }

    def _incident_key(self, task: Dict[str, Any]) -> str:
        """Stable key for retries/fixes around the same failing work item."""
        return f"{task.get('batch_id','')}/{task.get('name','')}/{task.get('type','')}"

    def _get_or_create_incident(self, task: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch or initialize incident state for a failing task."""
        key = self._incident_key(task)
        incident = self.incidents.get(key)
        if incident:
            return incident

        incident = {
            "incident_id": f"inc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "task_name": task.get("name"),
            "task_type": task.get("type"),
            "task_class": task.get("task_class"),
            "batch_id": task.get("batch_id"),
            "worker_cycles": 0,
            "brain_fix_attempts": 0,
            "cloud_escalated": False,
            "cloud_escalation_id": None,
            "last_result": result,
            "history": []
        }
        self.incidents[key] = incident
        return incident

    def log_training_sample(self, sample_type: str, prompt: str, response: str,
                            outcome: str, context: str = "", metadata: dict = None):
        """Log a training data sample for future fine-tuning."""
        training_log = self.log_path / "training_samples.jsonl"

        sample = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "sample_type": sample_type,
            "model": self.model,
            "prompt": prompt,
            "context": context,
            "response": response,
            "outcome": outcome,
            "metadata": metadata or {},
            "human_rating": None,
            "human_feedback": None,
            "preferred_response": None,
        }

        with open(training_log, 'a') as f:
            f.write(json.dumps(sample) + "\n")

        self.logger.debug(f"Logged training sample: {sample_type} -> {outcome}")

    # =========================================================================
    # Ollama Management
    # =========================================================================

    def start_ollama(self):
        """Ensure Ollama is running - use existing if available."""
        self.logger.info(f"Checking Ollama on GPUs {self.gpus}")

        # Check if ollama is already running
        try:
            response = requests.get(f"{self.config['ollama_host']}/api/tags", timeout=2)
            if response.status_code == 200:
                self.logger.info("Using existing Ollama server")
                return
        except:
            pass

        # Start our own if nothing is running
        self.logger.info("Starting Ollama...")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpus))

        self.ollama_process = subprocess.Popen(
            ["ollama", "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        for i in range(30):
            try:
                requests.get(f"{self.config['ollama_host']}/api/tags", timeout=1)
                self.logger.info("Ollama server ready")
                return
            except:
                time.sleep(1)

        raise RuntimeError("Ollama failed to start")

    def stop_ollama(self):
        """Stop Ollama if we started it."""
        if self.ollama_process:
            self.ollama_process.terminate()
            self.ollama_process.wait(timeout=10)
            self.logger.info("Ollama stopped")

    def think(self, prompt: str, context: str = "", log_as: str = None) -> str:
        """Use the brain model to reason about something."""
        full_prompt = ""
        if context:
            full_prompt = f"Context:\n{context}\n\n"
        full_prompt += prompt

        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "keep_alive": self.brain_keep_alive,
                    "options": {
                        "num_ctx": self.brain_num_ctx
                    }
                },
                timeout=self.think_timeout
            )
            response.raise_for_status()
            result = response.json().get("response", "")

            if log_as:
                self.log_training_sample(
                    sample_type=log_as,
                    prompt=prompt,
                    response=result,
                    outcome="pending",
                    context=context
                )

            return result
        except Exception as e:
            self.logger.error(f"Think error: {e}")
            if log_as:
                self.log_training_sample(
                    sample_type=log_as,
                    prompt=prompt,
                    response=f"ERROR: {e}",
                    outcome="failure",
                    context=context
                )
            return ""

    # =========================================================================
    # Private/Public Task List Management
    # =========================================================================

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

    def handle_execute_plan_task(self, task: Dict[str, Any]):
        """Handle an execute_plan task."""
        plan_path = task.get("plan_path", task.get("prompt", ""))
        config_overrides = task.get("config", {})

        try:
            batch_id = self.execute_plan(plan_path, config_overrides)
            task["status"] = "complete"
            task["result"] = {"success": True, "batch_id": batch_id, "handler": "brain"}
        except Exception as e:
            context = self._build_execute_plan_escalation_context(plan_path, config_overrides)
            escalation_id = self.emit_cloud_escalation(
                escalation_type="execute_plan_failure",
                title="Local brain failed to start plan execution",
                details={
                    "error": str(e),
                    "plan_path": plan_path,
                    "config_keys": sorted(list(config_overrides.keys())) if isinstance(config_overrides, dict) else [],
                    "traceback_tail": traceback.format_exc().strip().splitlines()[-8:],
                    "context": context
                },
                source_task=task
            )
            task["status"] = "failed"
            task["result"] = {
                "success": False,
                "error": str(e),
                "handler": "brain",
                "escalated": True,
                "escalation_id": escalation_id
            }

        task["completed_at"] = datetime.now().isoformat()
        dest_file = self.complete_path / f"{task['task_id']}.json"
        with open(dest_file, 'w') as f:
            json.dump(task, f, indent=2)

    def handle_shell_task(self, task: Dict[str, Any]):
        """Handle a shell task (brain executes it directly)."""
        command = task.get("command", "")
        task_name = task.get("name", task["task_id"][:8])
        task_class = str(task.get("task_class", "")).lower()
        task_id = str(task.get("task_id", ""))
        processing_file = self.processing_path / f"{task_id}.json"
        task_hb_file = self.processing_path / f"{task_id}.heartbeat.json"

        self.log_decision("SHELL_EXECUTE", f"Executing: {task_name}", {
            "task_id": task["task_id"][:8],
            "command": command[:80] + "..." if len(command) > 80 else command
        })

        start_time = time.time()
        now_iso = datetime.now().isoformat()
        task["status"] = "processing"
        task["assigned_to"] = self.name
        task["started_at"] = task.get("started_at") or now_iso
        task["last_attempt_at"] = now_iso
        with open(processing_file, "w") as f:
            json.dump(task, f, indent=2)

        try:
            env = os.environ.copy()
            # Strict brain-task ownership: brain-class work always runs with
            # brain model/runtime defaults (GPU 0 / brain ollama).
            if task_class == "brain" or str(task.get("executor", "")).lower() == "brain":
                env["BRAIN_MODEL"] = str(self.model)
                env["BRAIN_OLLAMA_URL"] = str(self.config.get("ollama_host", "http://localhost:11434"))
                # Compatibility bridge for scripts that still read WORKER_* vars.
                env["WORKER_MODEL"] = env["BRAIN_MODEL"]
                env["WORKER_OLLAMA_URL"] = env["BRAIN_OLLAMA_URL"]

            proc = subprocess.Popen(
                command,
                shell=True,
                executable='/bin/bash',
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            timeout_s = 1800  # 30 min timeout
            heartbeat_interval_s = 5
            next_hb = time.time()
            timed_out = False

            while True:
                ret = proc.poll()
                now = time.time()
                if now >= next_hb:
                    self._write_brain_heartbeat()
                    hb = {
                        "task_id": task_id,
                        "worker": self.name,
                        "updated_at": datetime.now().isoformat(),
                        "state": "running",
                    }
                    with open(task_hb_file, "w") as f:
                        json.dump(hb, f, indent=2)
                    task["last_progress_at"] = hb["updated_at"]
                    with open(processing_file, "w") as f:
                        json.dump(task, f, indent=2)
                    next_hb = now + heartbeat_interval_s
                if ret is not None:
                    break
                if now - start_time > timeout_s:
                    timed_out = True
                    proc.kill()
                    break
                time.sleep(1)

            try:
                stdout_text, stderr_text = proc.communicate(timeout=15)
            except Exception:
                stdout_text = ""
                stderr_text = "failed to collect process output after termination"

            elapsed = time.time() - start_time
            success = (not timed_out) and (proc.returncode == 0)
            output = stdout_text
            if stderr_text:
                output += f"\n[stderr: {stderr_text}]"

            error_text = ""
            if timed_out:
                error_text = "Command timed out"
            elif not success:
                stderr_lines = [ln.strip() for ln in stderr_text.splitlines() if ln.strip()]
                if stderr_lines:
                    error_text = stderr_lines[-1][:400]
                else:
                    error_text = f"command exited with return code {proc.returncode}"

            command_lc = str(command).lower()
            task_name_lc = str(task_name).lower()
            fatal_marker = "fatal:" in (stdout_text + "\n" + stderr_text).lower()
            critical_task_names = {
                str(name).strip().lower()
                for name in self.config.get("critical_task_names", [])
                if str(name).strip()
            }
            critical_command_markers = [
                str(marker).strip().lower()
                for marker in self.config.get("critical_command_markers", [])
                if str(marker).strip()
            ]
            critical_stage = task_name_lc in critical_task_names or any(
                marker in command_lc for marker in critical_command_markers
            )
            error_type = "fatal" if (not success and (fatal_marker or critical_stage)) else "worker"
            if not success and task_class == "brain":
                error_type = "brain_task_failure"

            task["status"] = "complete" if success else "failed"
            task["result"] = {
                "success": success,
                "output": output,
                "return_code": proc.returncode,
                "handler": "brain",
                "elapsed_seconds": round(elapsed, 1),
                "error": error_text,
                "error_type": error_type,
            }
        except Exception as e:
            elapsed = time.time() - start_time
            task["status"] = "failed"
            task["result"] = {
                "success": False,
                "error": str(e),
                "error_type": "worker",
                "handler": "brain",
                "elapsed_seconds": round(elapsed, 1),
            }
        finally:
            if processing_file.exists():
                try:
                    processing_file.unlink()
                except Exception:
                    pass
            if task_hb_file.exists():
                try:
                    task_hb_file.unlink()
                except Exception:
                    pass

        task["completed_at"] = datetime.now().isoformat()
        dest_base = self.complete_path if task["result"].get("success") else self.failed_path
        dest_file = dest_base / f"{task['task_id']}.json"
        with open(dest_file, 'w') as f:
            json.dump(task, f, indent=2)

        # Log completion
        elapsed = task["result"].get("elapsed_seconds", 0)
        if task["result"]["success"]:
            self.log_decision("SHELL_COMPLETE", f"SUCCESS: {task_name} ({elapsed}s)", {
                "task_id": task["task_id"][:8]
            })
        else:
            self.log_decision("SHELL_FAILED", f"FAILED: {task_name} ({elapsed}s)", {
                "task_id": task["task_id"][:8],
                "error": task["result"].get("error", "")[:200],
                "return_code": task["result"].get("return_code"),
                "error_type": task["result"].get("error_type", "worker"),
            })

    def handle_decide_task(self, task: Dict[str, Any]):
        """Handle a decide task (brain-only reasoning)."""
        self.log_decision("DECIDE", f"Handling decision task: {task.get('prompt', '')[:50]}...")
        result = self.think(task.get("prompt", ""), task.get("context", ""))

        task["status"] = "complete"
        task["result"] = {"success": True, "output": result, "handler": "brain"}
        task["completed_at"] = datetime.now().isoformat()

        dest_file = self.complete_path / f"{task['task_id']}.json"
        with open(dest_file, 'w') as f:
            json.dump(task, f, indent=2)

    def claim_brain_tasks(self):
        """Look for tasks that need brain processing."""
        def _task_priority_key(task: Dict[str, Any]) -> tuple:
            try:
                priority = int(task.get("priority", 5))
            except Exception:
                priority = 5
            created_at = str(task.get("created_at", "") or "")
            task_id = str(task.get("task_id", "") or "")
            return (-priority, created_at, task_id)

        ranked_task_files: List[Path] = []
        staged: List[tuple[Path, Dict[str, Any]]] = []
        for task_file in sorted(self.queue_path.glob("*.json")):
            if str(task_file).endswith('.lock'):
                continue
            try:
                with open(task_file) as f:
                    task = json.load(f)
            except Exception:
                continue
            task_type = task.get("type", "")
            executor = str(task.get("executor", "worker")).lower()
            task_class = str(task.get("task_class", "")).lower()
            is_brain_task = (
                task_type in {"execute_plan", "decide"}
                or executor == "brain"
                or task_class == "brain"
            )
            if is_brain_task:
                staged.append((task_file, task))

        staged.sort(key=lambda x: _task_priority_key(x[1]))
        ranked_task_files = [task_file for task_file, _ in staged]

        for task_file in ranked_task_files:
            if str(task_file).endswith('.lock'):
                continue

            lock_file = str(task_file) + ".lock"
            lock = FileLock(lock_file, timeout=1)

            try:
                with lock:
                    if not task_file.exists():
                        continue

                    with open(task_file) as f:
                        task = json.load(f)

                    task_type = task.get("type", "")
                    executor = str(task.get("executor", "worker")).lower()
                    task_class = str(task.get("task_class", "")).lower()

                    # Brain handles: execute_plan, decide, and tasks marked executor=brain
                    if task_type == "execute_plan":
                        task_file.unlink()
                        self.handle_execute_plan_task(task)
                    elif task_type == "decide":
                        task_file.unlink()
                        self.handle_decide_task(task)
                    elif (executor == "brain" or task_class == "brain") and task_type == "shell":
                        task["executor"] = "brain"
                        task["task_class"] = "brain"
                        task_file.unlink()
                        self.handle_shell_task(task)
                    elif executor == "brain" or task_class == "brain":
                        task_file.unlink()
                        task["status"] = "failed"
                        task["completed_at"] = datetime.now().isoformat()
                        task["result"] = {
                            "success": False,
                            "error": (
                                f"Unsupported brain task type '{task_type}'. "
                                "Brain tasks must use type='shell' (or built-in 'execute_plan'/'decide')."
                            ),
                            "handler": "brain",
                        }
                        dest_file = self.failed_path / f"{task['task_id']}.json"
                        with open(dest_file, "w") as f:
                            json.dump(task, f, indent=2)
                        self.log_decision(
                            "BRAIN_TASK_INVALID",
                            f"Rejected invalid brain task type: {task_type}",
                            {"task_id": task.get("task_id", "")[:8], "name": task.get("name", "")},
                        )
                    # Let workers handle other tasks

            except Exception as e:
                self.logger.error(f"Error processing task: {e}")

    # =========================================================================
    # Output Evaluation
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
