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
import json
import os
import re
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
from filelock import FileLock

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
#   - meta: Model load/unload tasks (inserted by brain, claimed by GPU workers)
# =============================================================================
VALID_TASK_CLASSES = ['cpu', 'script', 'llm', 'meta']


class Brain:
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

        # Public task queue (workers see this)
        self.queue_path = self.shared_path / "tasks" / "queue"
        self.processing_path = self.shared_path / "tasks" / "processing"
        self.complete_path = self.shared_path / "tasks" / "complete"
        self.failed_path = self.shared_path / "tasks" / "failed"

        # Brain state and private task list
        self.brain_path = self.shared_path / "brain"
        self.private_tasks_path = self.brain_path / "private_tasks"
        self.escalations_path = self.brain_path / "escalations"
        self.state_file = self.brain_path / "state.json"

        # Ensure directories exist
        for path in [self.queue_path, self.processing_path, self.complete_path,
                     self.failed_path, self.brain_path, self.private_tasks_path,
                     self.escalations_path]:
            path.mkdir(parents=True, exist_ok=True)

        self.model = self.brain_config["model"]
        self.gpus = self.brain_config["gpus"]
        self.api_url = f"{self.config['ollama_host']}/api/generate"

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
        self.gpu_pids: Dict[str, int] = {}
        self.gpu_miss_count: Dict[str, int] = {}  # Track consecutive misses for 3-miss tolerance
        self.signals_path = self.shared_path / "signals"
        self.signals_path.mkdir(parents=True, exist_ok=True)

        # P1: Track load_llm requests to detect when workers don't pick them up
        self.load_llm_requests: Dict[str, Dict] = {}
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
                    self.incidents = state.get("incidents", {})
                    self.logger.info(
                        f"Loaded state: {len(self.active_batches)} active batches, "
                        f"{len(self.load_llm_requests)} pending load_llm requests")
            except Exception as e:
                self.logger.warning(f"Failed to load state: {e}")
                self.active_batches = {}
                self.incidents = {}

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
            "incidents": self.incidents
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

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
                    "stream": False
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

    def save_to_private(self, task: Dict[str, Any]):
        """Save a task to the private list (not visible to workers)."""
        task_file = self.private_tasks_path / f"{task['task_id']}.json"
        with open(task_file, 'w') as f:
            json.dump(task, f, indent=2)

    def save_to_public(self, task: Dict[str, Any]):
        """Save a task to the public queue (workers can claim it)."""
        task_file = self.queue_path / f"{task['task_id']}.json"
        with open(task_file, 'w') as f:
            json.dump(task, f, indent=2)

        self.log_decision("TASK_RELEASED", f"Released task to queue: {task.get('name', task['task_id'][:8])}",
                          {"task_id": task['task_id'][:8], "depends_on": task.get('depends_on', [])})

    def save_to_failed(self, task: Dict[str, Any]):
        """Save a task directly to failed (for definition errors)."""
        task_file = self.failed_path / f"{task['task_id']}.json"
        with open(task_file, 'w') as f:
            json.dump(task, f, indent=2)

    def get_private_tasks(self, batch_id: str = None) -> List[Dict]:
        """Get private tasks, optionally filtered by batch."""
        tasks = []
        for task_file in self.private_tasks_path.glob("*.json"):
            try:
                with open(task_file) as f:
                    task = json.load(f)
                    if batch_id is None or task.get("batch_id") == batch_id:
                        tasks.append(task)
            except:
                pass
        return tasks

    def get_completed_task_ids(self, batch_id: str) -> set:
        """Get set of completed task names for a batch."""
        completed = set()
        for task_file in self.complete_path.glob("*.json"):
            try:
                with open(task_file) as f:
                    task = json.load(f)
                    if task.get("batch_id") == batch_id:
                        # Check if task succeeded
                        result = task.get("result", {})
                        if result.get("success", False):
                            completed.add(task.get("name", ""))
            except:
                pass
        return completed

    def check_and_release_tasks(self):
        """Check private tasks and release any whose dependencies are met."""
        for batch_id in list(self.active_batches.keys()):
            satisfied = self.get_satisfied_task_ids(batch_id)
            private_tasks = self.get_private_tasks(batch_id)

            all_released = True
            for task in private_tasks:
                depends_on = task.get("depends_on", [])
                # Foreach templates may include per-item dependency placeholders.
                # Ignore those at template-gating time; they are applied on expansion.
                template_depends = depends_on
                if task.get("foreach"):
                    template_depends = [d for d in depends_on if "{ITEM" not in d]

                # Check if all template dependencies are satisfied
                deps_met = all(dep in satisfied for dep in template_depends)

                if deps_met:
                    task_file = self.private_tasks_path / f"{task['task_id']}.json"

                    # Check if this is a foreach task that needs expansion
                    foreach_spec = task.get("foreach")
                    if foreach_spec:
                        # Expand the foreach task into N tasks
                        expanded_names = self._expand_foreach_task(task, batch_id)
                        if expanded_names:
                            # Remove the template task
                            if task_file.exists():
                                task_file.unlink()
                            # Update any tasks that depend on this one to depend on ALL expanded tasks
                            self._update_foreach_dependents(batch_id, task.get("name"), expanded_names)
                        else:
                            all_released = False
                    else:
                        # Regular task - release to public queue
                        if task_file.exists():
                            task_file.unlink()
                            self.save_to_public(task)
                else:
                    all_released = False
                    pending_deps = [d for d in template_depends if d not in satisfied]
                    self.logger.debug(f"Task {task.get('name')} waiting on: {pending_deps}")

            # Check if batch is complete (no private tasks, no public/processing tasks)
            if all_released and len(private_tasks) > 0:
                self._check_batch_completion(batch_id)

    def _expand_foreach_task(self, template_task: Dict, batch_id: str) -> List[str]:
        """
        Expand a foreach task into N individual tasks.

        foreach format: {BATCH_PATH}/manifest.json:videos
        - Path to JSON file (with variable substitution)
        - Colon separator
        - JSON path to array (e.g., "videos" or "data.items")

        Returns list of expanded task names, or empty list on failure.
        """
        foreach_spec = template_task.get("foreach", "")
        if ":" not in foreach_spec:
            self.logger.error(f"Invalid foreach format: {foreach_spec} (expected path:jsonpath)")
            return []

        file_path, json_path = foreach_spec.rsplit(":", 1)

        # Substitute variables in file path
        batch_info = self.active_batches.get(batch_id, {})
        batch_dir = Path(batch_info.get("batch_dir", ""))
        plan_dir = Path(batch_info.get("plan_dir", ""))

        file_path = file_path.replace("{BATCH_PATH}", str(batch_dir))
        file_path = file_path.replace("{PLAN_PATH}", str(plan_dir))

        # Read the JSON file
        try:
            with open(file_path) as f:
                data = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to read foreach source {file_path}: {e}")
            return []

        # Navigate to the array using json_path (supports simple dotted paths)
        items = data
        for key in json_path.split("."):
            if isinstance(items, dict) and key in items:
                items = items[key]
            else:
                self.logger.error(f"JSON path '{json_path}' not found in {file_path}")
                return []

        if not isinstance(items, list):
            self.logger.error(f"foreach target is not a list: {json_path}")
            return []

        batch_size = max(1, int(template_task.get("batch_size", 1)))
        expanded_target = (len(items) + batch_size - 1) // batch_size
        self.log_decision(
            "FOREACH_EXPAND",
            f"Expanding '{template_task.get('name')}' into {expanded_target} task(s) from {len(items)} item(s)",
            {
                "template": template_task.get("name"),
                "item_count": len(items),
                "batch_size": batch_size,
                "expanded_count": expanded_target
            }
        )

        # Create one task per item (batch_size=1) or micro-batch task (batch_size>1)
        expanded_names = []
        template_depends = template_task.get("depends_on", [])
        for start in range(0, len(items), batch_size):
            group = items[start:start + batch_size]
            group_commands = []
            group_depends = []
            group_item_ids = []

            for i, item in enumerate(group, start=start):
                # Build item-specific command
                command = template_task.get("command", "")

                # Substitute {ITEM.field} patterns
                if isinstance(item, dict):
                    for key, value in item.items():
                        command = command.replace(f"{{ITEM.{key}}}", str(value))
                else:
                    command = command.replace("{ITEM}", str(item))
                group_commands.append(command)

                item_id = item.get("id", str(i)) if isinstance(item, dict) else str(i)
                group_item_ids.append(str(item_id))

                # Build per-item dependencies from template depends_on.
                item_depends = []
                for dep in template_depends:
                    dep_name = dep
                    if isinstance(item, dict):
                        for key, value in item.items():
                            dep_name = dep_name.replace(f"{{ITEM.{key}}}", str(value))
                    else:
                        dep_name = dep_name.replace("{ITEM}", str(item))
                    if dep_name and dep_name.lower() != "none":
                        item_depends.append(dep_name)

                for dep in item_depends:
                    if dep not in group_depends:
                        group_depends.append(dep)

            if batch_size == 1:
                task_name = f"{template_task.get('name')}_{group_item_ids[0]}"
                task_command = group_commands[0]
            else:
                task_name = f"{template_task.get('name')}_batch_{start + 1:04d}_{start + len(group):04d}"
                # Sequential micro-batch execution in one worker claim.
                task_command = "set -e\n" + "\n".join(group_commands)

            task = self.create_task(
                task_type="shell",
                command=task_command,
                batch_id=batch_id,
                task_name=task_name,
                priority=template_task.get("priority", 5),
                depends_on=group_depends,
                executor=template_task.get("executor", "worker"),
                task_class=template_task.get("task_class")
            )
            task["batch_size"] = len(group)
            task["item_ids"] = group_item_ids

            # Release if ready, otherwise keep private until dependencies are met.
            if group_depends:
                self.save_to_private(task)
            else:
                self.save_to_public(task)
            expanded_names.append(task_name)

            self.log_decision(
                "TASK_CREATED",
                f"Created expanded task: {task_name}",
                {
                    "task_id": task["task_id"][:8],
                    "batch_size": len(group),
                    "item_ids": group_item_ids,
                    "depends_on": group_depends
                }
            )

        return expanded_names

    def _update_foreach_dependents(self, batch_id: str, original_name: str, expanded_names: List[str]):
        """
        Update tasks that depend on the foreach template to depend on ALL expanded tasks.

        e.g., if 'aggregate' depends_on ['transcribe'], and transcribe expanded to
        ['transcribe_v1', 'transcribe_v2'], update aggregate to depend on both.
        """
        for task_file in self.private_tasks_path.glob("*.json"):
            try:
                with open(task_file) as f:
                    task = json.load(f)

                if task.get("batch_id") != batch_id:
                    continue

                depends_on = task.get("depends_on", [])
                if original_name in depends_on:
                    # Replace the template name with all expanded names
                    new_deps = [d for d in depends_on if d != original_name]
                    new_deps.extend(expanded_names)
                    task["depends_on"] = new_deps

                    # Save updated task
                    with open(task_file, 'w') as f:
                        json.dump(task, f, indent=2)

                    self.log_decision("DEPS_UPDATED",
                        f"Updated '{task.get('name')}' to depend on {len(expanded_names)} expanded tasks",
                        {"task": task.get("name"), "new_deps_count": len(new_deps)})

            except Exception as e:
                self.logger.error(f"Error updating dependents: {e}")

    def _check_batch_completion(self, batch_id: str):
        """Check if a batch is fully complete."""
        # Any tasks still in queue or processing?
        for path in [self.queue_path, self.processing_path]:
            for task_file in path.glob("*.json"):
                try:
                    with open(task_file) as f:
                        task = json.load(f)
                        if task.get("batch_id") == batch_id:
                            return  # Still has pending tasks
                except:
                    pass

        # Any private tasks left?
        if self.get_private_tasks(batch_id):
            return

        # Batch is complete
        self.log_decision("BATCH_COMPLETE", f"Batch {batch_id} finished successfully",
                          {"batch_id": batch_id, "plan": self.active_batches.get(batch_id, {}).get("plan")})
        del self.active_batches[batch_id]
        self._save_brain_state()

    def get_satisfied_task_ids(self, batch_id: str) -> set:
        """
        Get task names considered dependency-satisfied:
        - successful completed tasks
        - terminal failed tasks (max retries exhausted / cloud-blocked)
        """
        satisfied = self.get_completed_task_ids(batch_id)
        max_attempts = self.config.get("retry_policy", {}).get("max_attempts", 3)

        for task_file in self.failed_path.glob("*.json"):
            try:
                with open(task_file) as f:
                    task = json.load(f)
                if task.get("batch_id") != batch_id:
                    continue

                is_terminal = (
                    task.get("status") == "blocked_cloud"
                    or bool(task.get("cloud_escalated", False))
                    or int(task.get("attempts", 0)) >= int(max_attempts)
                )
                if is_terminal:
                    name = task.get("name", "")
                    if name:
                        satisfied.add(name)
            except Exception:
                continue

        return satisfied

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

    def create_task(self, task_type: str, command: str, batch_id: str,
                    task_name: str = "", priority: int = 5,
                    depends_on: List[str] = None, executor: str = "worker",
                    task_class: str = None) -> Dict[str, Any]:
        """Create a new task.

        task_class must be specified in plan.md. If missing or invalid,
        the task is created but marked for immediate failure.
        """
        definition_error = None
        if task_class is None:
            definition_error = f"missing task_class (must be one of: cpu, script, llm)"
            task_class = "cpu"  # Placeholder so task structure is valid
        elif task_class not in VALID_TASK_CLASSES:
            definition_error = f"invalid task_class '{task_class}' (must be one of: cpu, script, llm)"
            task_class = "cpu"  # Placeholder

        task = {
            "task_id": str(uuid.uuid4()),
            "type": task_type,
            "command": command,
            "batch_id": batch_id,
            "name": task_name,
            "priority": priority,
            "task_class": task_class,  # cpu, script, or llm
            "depends_on": depends_on or [],
            "executor": executor,  # "brain" or "worker"
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "created_by": self.name,
            "retry_count": 0
        }

        # Mark task with definition error so it goes to failed/ immediately
        if definition_error:
            task["definition_error"] = definition_error
            task["error_type"] = "definition"

        return task

    # =========================================================================
    # Plan Parsing and Execution
    # =========================================================================

    def parse_plan_md(self, plan_content: str) -> List[Dict[str, Any]]:
        """
        Parse a structured plan.md file into task definitions.

        Expected format for each task:
        ### task_id
        - **executor**: brain|worker
        - **task_class**: cpu|script|llm
        - **command**: `shell command here`
        - **depends_on**: task1, task2
        - **foreach**: manifest.videos  (optional - expands to N tasks)
        - **batch_size**: 4  (optional - groups foreach expansion into micro-batches)
        """
        tasks = []

        # Split by ### to get task sections
        sections = re.split(r'\n### ', plan_content)

        for section in sections[1:]:  # Skip header before first ###
            lines = section.strip().split('\n')
            if not lines:
                continue

            task_id = lines[0].strip()
            task = {
                "id": task_id,
                "executor": "worker",
                "task_class": None,
                "command": "",
                "depends_on": [],
                "foreach": None,
                "batch_size": 1,
            }

            for line in lines[1:]:
                line = line.strip()
                if line.startswith('- **executor**:'):
                    task["executor"] = line.split(':', 1)[1].strip()
                elif line.startswith('- **task_class**:'):
                    task_class = line.split(':', 1)[1].strip().lower()
                    if task_class in VALID_TASK_CLASSES:
                        task["task_class"] = task_class
                    else:
                        self.logger.warning(f"Invalid task_class '{task_class}' for {task_id}, will use fallback")
                elif line.startswith('- **command**:'):
                    # Extract command from backticks
                    match = re.search(r'`([^`]+)`', line)
                    if match:
                        task["command"] = match.group(1)
                elif line.startswith('- **depends_on**:'):
                    deps = line.split(':', 1)[1].strip()
                    if deps.lower() != 'none':
                        task["depends_on"] = [d.strip() for d in deps.split(',') if d.strip()]
                elif line.startswith('- **foreach**:'):
                    # e.g., "manifest.videos" means expand based on videos array in manifest
                    task["foreach"] = line.split(':', 1)[1].strip()
                elif line.startswith('- **batch_size**:'):
                    raw = line.split(':', 1)[1].strip()
                    try:
                        task["batch_size"] = max(1, int(raw))
                    except ValueError:
                        self.logger.warning(f"Invalid batch_size '{raw}' for {task_id}, defaulting to 1")
                        task["batch_size"] = 1

            if task["command"]:  # Only add tasks with commands
                tasks.append(task)

        return tasks

    def execute_plan(self, plan_path: str, config_overrides: dict = None) -> str:
        """
        Execute a plan by reading plan.md and generating tasks.

        1. Read and parse plan.md directly (no LLM needed)
        2. Create all tasks with dependencies
        3. Store in private list
        4. Release tasks with no dependencies immediately
        5. Return batch_id
        """
        plan_dir = Path(plan_path)
        if plan_dir.is_file():
            plan_dir = plan_dir.parent

        # Read plan.md
        plan_file = plan_dir / "plan.md"
        if not plan_file.exists():
            raise FileNotFoundError(f"No plan.md found in {plan_dir}")

        with open(plan_file) as f:
            plan_content = f.read()

        # Generate timestamp-based execution batch ID for orchestration tracking
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = plan_dir / "history" / batch_id

        self.log_decision("PLAN_READ", f"Read plan from {plan_dir.name}", {
            "batch_id": batch_id,
            "plan_size": len(plan_content)
        })

        # Build variable substitution map
        config = dict(config_overrides or {})

        # Enforce explicit run mode contract for plans that use RUN_MODE.
        run_mode = str(config.get("RUN_MODE", "fresh")).strip().lower()
        if run_mode not in ["fresh", "resume"]:
            raise ValueError("RUN_MODE must be 'fresh' or 'resume'")
        config["RUN_MODE"] = run_mode

        if run_mode == "resume":
            resume_batch_id = str(config.get("RESUME_BATCH_ID", "")).strip()
            if not resume_batch_id:
                raise ValueError("RUN_MODE=resume requires RESUME_BATCH_ID")
            if "/" in resume_batch_id or ".." in resume_batch_id:
                raise ValueError("RESUME_BATCH_ID contains invalid path characters")

            resume_manifest = plan_dir / "history" / resume_batch_id / "manifest.json"
            if not resume_manifest.exists():
                raise FileNotFoundError(
                    f"Resume batch manifest not found: {resume_manifest}"
                )

            # Use the existing data batch for plan variables while keeping a
            # new orchestration batch_id for dependency tracking.
            config["BATCH_ID"] = resume_batch_id
            self.log_decision("PLAN_MODE",
                              f"Resume mode for {plan_dir.name}: {resume_batch_id}",
                              {"run_mode": "resume", "resume_batch_id": resume_batch_id})

            # If this exact plan/data-batch is already active, reuse it.
            # Avoid creating a duplicate orchestration batch that re-releases
            # the whole plan on each resume submission.
            plan_dir_resolved = str(plan_dir.resolve())
            for existing_batch_id, batch_meta in self.active_batches.items():
                existing_plan_dir = str(batch_meta.get("plan_dir", ""))
                existing_cfg = batch_meta.get("config", {}) or {}
                existing_data_batch = str(existing_cfg.get("BATCH_ID", "")).strip()

                if existing_plan_dir == plan_dir_resolved and existing_data_batch == resume_batch_id:
                    self.log_decision(
                        "PLAN_RESUME_REUSE",
                        f"Resume requested for already-active batch {existing_batch_id}; reusing existing orchestration state",
                        {
                            "plan_dir": plan_dir_resolved,
                            "resume_batch_id": resume_batch_id,
                            "orchestration_batch_id": existing_batch_id
                        }
                    )
                    return existing_batch_id
        else:
            # Fresh mode defaults plan variables to the new execution batch.
            cleanup_stats = self._cleanup_stale_plan_batches(plan_dir)
            if cleanup_stats.get("stale_batches", 0) > 0:
                self.log_decision(
                    "PLAN_CLEANUP",
                    f"Fresh run cleanup for {plan_dir.name}",
                    cleanup_stats
                )

            config["BATCH_ID"] = str(config.get("BATCH_ID", batch_id)).strip() or batch_id
            if "/" in config["BATCH_ID"] or ".." in config["BATCH_ID"]:
                raise ValueError("BATCH_ID contains invalid path characters")
            self.log_decision("PLAN_MODE",
                              f"Fresh mode for {plan_dir.name}: {config['BATCH_ID']}",
                              {"run_mode": "fresh", "batch_id": config["BATCH_ID"]})

        effective_batch_id = config["BATCH_ID"]
        effective_batch_dir = plan_dir / "history" / effective_batch_id
        effective_batch_dir.mkdir(parents=True, exist_ok=True)
        (effective_batch_dir / "results").mkdir(exist_ok=True)
        (effective_batch_dir / "output").mkdir(exist_ok=True)
        (effective_batch_dir / "logs").mkdir(exist_ok=True)

        variables = {
            "{BATCH_ID}": effective_batch_id,
            "{PLAN_PATH}": str(plan_dir.resolve()),
            "{BATCH_PATH}": str(effective_batch_dir.resolve()),
        }
        # Add any config overrides as variables
        for key, value in config.items():
            variables[f"{{{key}}}"] = str(value)

        # Parse plan.md directly
        task_defs = self.parse_plan_md(plan_content)

        self.log_decision("PLAN_PARSED", f"Parsed {len(task_defs)} tasks from plan.md", {
            "task_ids": [t["id"] for t in task_defs],
            "batch_id": batch_id
        })

        # Analyze task types for resource planning
        class_counts = {"cpu": 0, "script": 0, "llm": 0}
        missing_class = []
        for t in task_defs:
            tc = t.get("task_class")
            if tc and tc in VALID_TASK_CLASSES:
                class_counts[tc] += 1
            else:
                missing_class.append(t["id"])
                class_counts["cpu"] += 1  # Default to cpu

        if missing_class:
            self.logger.warning(f"Tasks missing task_class (defaulting to cpu): {missing_class}")

        self.log_decision("PLAN_ANALYSIS",
            f"Task breakdown: {class_counts['cpu']} cpu, {class_counts['script']} script, {class_counts['llm']} llm",
            {"task_classes": class_counts})

        # Create tasks with variable substitution
        tasks_with_no_deps = []
        for task_def in task_defs:
            # Substitute all variables in command
            command = task_def["command"]
            for var, value in variables.items():
                command = command.replace(var, value)

            task = self.create_task(
                task_type="shell",
                command=command,
                batch_id=batch_id,
                task_name=task_def["id"],
                priority=5,
                depends_on=task_def.get("depends_on", []),
                executor=task_def.get("executor", "worker"),
                task_class=task_def.get("task_class")
            )

            # Preserve foreach spec for later expansion
            if task_def.get("foreach"):
                # Substitute variables in foreach path
                foreach_spec = task_def["foreach"]
                for var, value in variables.items():
                    foreach_spec = foreach_spec.replace(var, value)
                task["foreach"] = foreach_spec
                task["batch_size"] = max(1, int(task_def.get("batch_size", 1)))

            # Check for definition errors - send to failed/ immediately
            if task.get("definition_error"):
                task["status"] = "failed"
                task["result"] = {
                    "success": False,
                    "error": task["definition_error"],
                    "error_type": "definition"
                }
                self.save_to_failed(task)
                self.log_decision("TASK_DEFINITION_ERROR",
                    f"Task '{task_def['id']}' has definition error: {task['definition_error']}",
                    {"task_id": task["task_id"][:8], "error": task["definition_error"]})
            else:
                # Save valid tasks to private list
                self.save_to_private(task)

                self.log_decision("TASK_CREATED", f"Created task: {task_def['id']} ({task['task_class']})", {
                    "task_id": task["task_id"][:8],
                    "task_class": task["task_class"],
                    "depends_on": task_def.get("depends_on", []),
                    "executor": task_def.get("executor", "worker")
                })

                # Track tasks with no dependencies for immediate release
                if not task_def.get("depends_on"):
                    tasks_with_no_deps.append(task)

        # Auto-insert execution summary task (depends on all other tasks)
        all_task_ids = [t["id"] for t in task_defs]
        summary_command = f"python {self.shared_path}/scripts/generate_batch_summary.py --batch-id {batch_id} --plan-name {plan_dir.name} --plan-dir {plan_dir.resolve()}"

        summary_task = self.create_task(
            task_type="shell",
            command=summary_command,
            batch_id=batch_id,
            task_name="batch_summary",
            priority=1,  # Low priority - runs at very end
            depends_on=all_task_ids,  # Depends on everything
            executor="worker",
            task_class="cpu"
        )

        self.save_to_private(summary_task)
        self.log_decision("TASK_CREATED", "Created automatic summary task (depends on all tasks)", {
            "task_id": summary_task["task_id"][:8],
            "depends_on_count": len(all_task_ids)
        })

        # Track this batch (include paths for foreach expansion)
        self.active_batches[batch_id] = {
            "plan": plan_dir.name,
            "plan_dir": str(plan_dir.resolve()),
            "batch_dir": str(batch_dir.resolve()),
            "started_at": datetime.now().isoformat(),
            "config": config,
            "total_tasks": len(task_defs) + 1  # +1 for automatic summary task
        }
        self._save_brain_state()

        # Release tasks with no dependencies immediately
        for task in tasks_with_no_deps:
            task_file = self.private_tasks_path / f"{task['task_id']}.json"
            if task_file.exists():
                task_file.unlink()
                self.save_to_public(task)

        return batch_id

    # =========================================================================
    # Task Handling
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

        self.log_decision("SHELL_EXECUTE", f"Executing: {task_name}", {
            "task_id": task["task_id"][:8],
            "command": command[:80] + "..." if len(command) > 80 else command
        })

        start_time = time.time()
        try:
            result = subprocess.run(
                command,
                shell=True,
                executable='/bin/bash',
                capture_output=True,
                text=True,
                timeout=1800  # 30 min timeout
            )
            elapsed = time.time() - start_time
            success = result.returncode == 0
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr: {result.stderr}]"

            task["status"] = "complete" if success else "failed"
            task["result"] = {
                "success": success,
                "output": output,
                "return_code": result.returncode,
                "handler": "brain",
                "elapsed_seconds": round(elapsed, 1)
            }
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            task["status"] = "failed"
            task["result"] = {"success": False, "error": "Command timed out", "handler": "brain", "elapsed_seconds": round(elapsed, 1)}
        except Exception as e:
            elapsed = time.time() - start_time
            task["status"] = "failed"
            task["result"] = {"success": False, "error": str(e), "handler": "brain", "elapsed_seconds": round(elapsed, 1)}

        task["completed_at"] = datetime.now().isoformat()
        dest_file = self.complete_path / f"{task['task_id']}.json"
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
                "error": task["result"].get("error", "")[:200]
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
        for task_file in self.queue_path.glob("*.json"):
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
                    executor = task.get("executor", "worker")

                    # Brain handles: execute_plan, decide, and tasks marked executor=brain
                    if task_type == "execute_plan":
                        task_file.unlink()
                        self.handle_execute_plan_task(task)
                    elif task_type == "decide":
                        task_file.unlink()
                        self.handle_decide_task(task)
                    elif executor == "brain" and task_type == "shell":
                        task_file.unlink()
                        self.handle_shell_task(task)
                    # Let workers handle other tasks

            except Exception as e:
                self.logger.error(f"Error processing task: {e}")

    # =========================================================================
    # Output Evaluation
    # =========================================================================

    def evaluate_worker_output(self, task: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Brain evaluates worker output for quality."""
        if not result.get("success"):
            return {
                "acceptable": False,
                "rating": 1,
                "issues": [f"Worker error: {result.get('error', 'unknown')}"],
                "retry": True,
                "feedback": "Worker failed to complete task"
            }

        output = result.get("output", "")
        command = task.get("command", "")

        eval_prompt = f"""Evaluate this task output for correctness.

Task command: {command[:200]}

Output:
{output[:1000]}

Rate 1-5:
- 5: Perfect
- 4: Good
- 3: Acceptable
- 2: Poor (should retry)
- 1: Failed

Return JSON: {{"acceptable": true/false, "rating": 1-5, "issues": [], "feedback": "brief explanation"}}

JSON only:"""

        eval_response = self.think(eval_prompt, log_as="brain_evaluation")

        try:
            eval_response = eval_response.strip()
            if eval_response.startswith("```"):
                eval_response = eval_response.split("```")[1]
                if eval_response.startswith("json"):
                    eval_response = eval_response[4:]
            evaluation = json.loads(eval_response)

            evaluation.setdefault("acceptable", True)
            evaluation.setdefault("rating", 3)
            evaluation.setdefault("issues", [])
            evaluation.setdefault("feedback", "")
            evaluation["retry"] = not evaluation["acceptable"] and evaluation["rating"] <= 2

            return evaluation

        except json.JSONDecodeError:
            self.logger.warning("Could not parse brain evaluation, assuming acceptable")
            return {
                "acceptable": True,
                "rating": 3,
                "issues": [],
                "feedback": "Evaluation parse failed",
                "retry": False
            }

    # =========================================================================
    # Failed Task Handling
    # =========================================================================

    def handle_failed_tasks(self):
        """Check for failed tasks and handle based on error type.

        - Worker failures: Retry up to max_attempts times (uses task memory)
        - Definition errors: Attempt to fix (e.g., infer missing task_class)
        """
        max_attempts = self.config.get("retry_policy", {}).get("max_attempts", 3)

        for task_file in self.failed_path.glob("*.json"):
            try:
                with open(task_file) as f:
                    task = json.load(f)

                result = task.get("result", {})
                error_type = result.get("error_type", "worker")  # Default to worker failure

                # Use task memory fields (attempts, workers_attempted)
                attempts = task.get("attempts", 0)
                workers = task.get("workers_attempted", [])

                # Already escalated/blocked tasks are terminal for local brain.
                # Do not re-process every loop.
                if task.get("status") == "blocked_cloud" or task.get("cloud_escalated", False):
                    continue

                if error_type == "definition":
                    # Definition error - try to fix the task
                    fixed = self._try_fix_definition_error(task)
                    if fixed:
                        task_file.unlink()
                        self.save_to_public(task)
                        self.log_decision("TASK_FIXED",
                            f"Fixed definition error for '{task.get('name', '')}', re-queued",
                            {"task_id": task["task_id"][:8], "fix_applied": task.get("fix_applied", "")})
                    else:
                        self.log_decision("TASK_UNFIXABLE",
                            f"Could not fix definition error for '{task.get('name', '')}': {result.get('error', '')}",
                            {"task_id": task["task_id"][:8]})
                        # Leave in failed/ for manual intervention

                elif self._try_fix_missing_module(task, result):
                    # Brain fixed a missing dependency. Requeue with clean retry state.
                    incident = self._get_or_create_incident(task, result)
                    incident["brain_fix_attempts"] = int(incident.get("brain_fix_attempts", 0)) + 1
                    incident["updated_at"] = datetime.now().isoformat()
                    incident["last_result"] = result
                    incident["history"].append({
                        "at": datetime.now().isoformat(),
                        "event": "brain_fix_applied",
                        "fix_applied": task.get("fix_applied", "")
                    })

                    task["status"] = "pending"
                    task["attempts"] = 0
                    task["workers_attempted"] = []

                    task_file.unlink()
                    self.save_to_public(task)

                    self.log_decision(
                        "TASK_FIXED",
                        f"Installed missing dependency for '{task.get('name', '')}', re-queued",
                        {
                            "task_id": task["task_id"][:8],
                            "fix_applied": task.get("fix_applied", ""),
                            "name": task.get("name", ""),
                            "incident_id": incident.get("incident_id")
                        }
                    )
                    self._save_brain_state()

                elif attempts < max_attempts:
                    # Worker failure - retry
                    task["status"] = "pending"

                    task_file.unlink()
                    self.save_to_public(task)

                    workers_str = ", ".join(workers) if workers else "unknown"
                    self.log_decision("RETRY",
                        f"Retrying task (attempt {attempts}/{max_attempts}) - previous workers: {workers_str}",
                        {"task_id": task["task_id"][:8], "name": task.get("name", ""), "workers_attempted": workers})
                else:
                    incident = self._get_or_create_incident(task, result)
                    incident["worker_cycles"] = int(incident.get("worker_cycles", 0)) + 1
                    incident["updated_at"] = datetime.now().isoformat()
                    incident["last_result"] = result
                    incident["history"].append({
                        "at": datetime.now().isoformat(),
                        "event": "worker_cycle_exhausted",
                        "attempts": attempts,
                        "workers_attempted": workers
                    })

                    workers_str = ", ".join(workers) if workers else "unknown"
                    self.log_decision("ABANDON",
                        f"Task abandoned after {attempts} attempts by workers [{workers_str}]",
                        {"task_id": task["task_id"][:8], "name": task.get("name", ""),
                         "workers_attempted": workers, "attempts": attempts,
                         "incident_id": incident.get("incident_id")})

                    # Brain gets up to N fix revisions before cloud escalation.
                    if int(incident.get("brain_fix_attempts", 0)) < self.max_brain_fix_attempts:
                        fixed = self._try_fix_missing_module(task, result)
                        incident["brain_fix_attempts"] = int(incident.get("brain_fix_attempts", 0)) + 1
                        incident["updated_at"] = datetime.now().isoformat()
                        incident["history"].append({
                            "at": datetime.now().isoformat(),
                            "event": "brain_fix_attempt",
                            "attempt_index": incident["brain_fix_attempts"],
                            "fix_succeeded": bool(fixed),
                            "fix_applied": task.get("fix_applied", "") if fixed else ""
                        })

                        if fixed:
                            task["status"] = "pending"
                            task["attempts"] = 0
                            task["workers_attempted"] = []
                            task_file.unlink()
                            self.save_to_public(task)
                            self.log_decision(
                                "TASK_FIXED",
                                f"Brain fix attempt {incident['brain_fix_attempts']}/{self.max_brain_fix_attempts} succeeded; re-queued",
                                {"task_id": task["task_id"][:8], "incident_id": incident.get("incident_id")}
                            )
                            self._save_brain_state()
                            continue

                    # Cloud escalation gate: escalate once after brain budget exhausted.
                    if not incident.get("cloud_escalated", False):
                        escalation_id = self.emit_cloud_escalation(
                            escalation_type="verification_failure",
                            title="Task exhausted worker+brain retries; needs cloud verification",
                            details={
                                "reason": "max_retries_exhausted",
                                "policy": {
                                    "worker_max_attempts_per_revision": max_attempts,
                                    "brain_max_fix_revisions": self.max_brain_fix_attempts
                                },
                                "incident_id": incident.get("incident_id"),
                                "worker_cycles": incident.get("worker_cycles"),
                                "brain_fix_attempts": incident.get("brain_fix_attempts"),
                                "task_name": task.get("name"),
                                "task_type": task.get("type"),
                                "task_class": task.get("task_class"),
                                "batch_id": task.get("batch_id"),
                                "workers_attempted": workers,
                                "last_result": result,
                                "verification_required": True,
                                "proposed_resubmit_task": self._build_resubmit_payload_for_abandoned_task(task)
                            },
                            source_task=task
                        )
                        incident["cloud_escalated"] = True
                        incident["cloud_escalation_id"] = escalation_id
                        incident["updated_at"] = datetime.now().isoformat()

                    # Freeze this task for cloud review; do not loop local retries.
                    task["status"] = "blocked_cloud"
                    task["cloud_escalated"] = True
                    task["incident_id"] = incident.get("incident_id")
                    task["blocked_reason"] = "awaiting_cloud_review"
                    with open(task_file, 'w') as f:
                        json.dump(task, f, indent=2)
                    self._save_brain_state()

            except Exception as e:
                self.logger.error(f"Error handling failed task: {e}")

    def _try_fix_missing_module(self, task: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """
        Attempt to auto-fix common Python module import failures from worker output.
        Returns True if a fix was applied and verified.
        """
        if task.get("dependency_fix_applied", False):
            return False  # Avoid endless fix/retry loops per task

        output = f"{result.get('output', '')}\n{result.get('error', '')}"
        match = re.search(r"No module named ['\"]([^'\"]+)['\"]", output)
        if not match:
            return False

        module = match.group(1).strip()
        if not module:
            return False

        # Try generic package name variants (no module-specific hardcoding).
        candidates = [module]
        if "_" in module:
            candidates.append(module.replace("_", "-"))
        if "-" in module:
            candidates.append(module.replace("-", "_"))

        # Ask brain model for likely pip package names from context.
        infer_prompt = (
            "A Python task failed with missing module import.\n"
            f"Missing module: {module}\n"
            "Suggest up to 3 likely pip package names as a comma-separated list.\n"
            "Return only package names, no explanation."
        )
        inferred = self.think(infer_prompt).strip()
        if inferred:
            inferred = inferred.replace("\n", ",")
            for part in inferred.split(","):
                name = part.strip().strip("`'\"")
                if name and " " not in name and len(name) < 80:
                    candidates.append(name)
        # Preserve order, remove duplicates.
        seen = set()
        packages = []
        for p in candidates:
            if p not in seen:
                seen.add(p)
                packages.append(p)

        install_errors = []
        for pkg in packages:
            install_cmd = (
                "source ~/ml-env/bin/activate && "
                "python -m pip install --disable-pip-version-check "
                f"{pkg}"
            )
            self.log_decision(
                "DEPENDENCY_FIX",
                f"Attempting dependency install for missing module '{module}' via package '{pkg}'",
                {"module": module, "package": pkg, "task": task.get("name", "")}
            )
            try:
                res = subprocess.run(
                    install_cmd,
                    shell=True,
                    executable="/bin/bash",
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                if res.returncode != 0:
                    install_errors.append(f"{pkg}: {res.stderr.strip()[:220]}")
                    continue

                verify_cmd = (
                    "source ~/ml-env/bin/activate && "
                    f"python -c \"import importlib; importlib.import_module('{module}')\""
                )
                ver = subprocess.run(
                    verify_cmd,
                    shell=True,
                    executable="/bin/bash",
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if ver.returncode == 0:
                    self.dependency_fix_attempts[module] = {
                        "success": True,
                        "package": pkg,
                        "at": datetime.now().isoformat()
                    }
                    task["dependency_fix_applied"] = True
                    task["fix_applied"] = f"installed_python_dependency:{pkg}"
                    return True

                install_errors.append(f"{pkg}: import verify failed: {ver.stderr.strip()[:220]}")
            except Exception as e:
                install_errors.append(f"{pkg}: {str(e)[:220]}")

        self.dependency_fix_attempts[module] = {
            "success": False,
            "attempted_packages": packages,
            "errors": install_errors[-5:],
            "at": datetime.now().isoformat()
        }
        self.log_decision(
            "DEPENDENCY_UNFIXABLE",
            f"Could not auto-fix missing module '{module}' for task '{task.get('name', '')}'",
            {"module": module, "attempted_packages": packages, "errors": install_errors[-3:]}
        )
        return False

    def _try_fix_definition_error(self, task: Dict[str, Any]) -> bool:
        """Attempt to fix a task with a definition error.

        Currently handles:
        - Missing task_class: Infer from command content

        Returns True if fixed, False if unfixable.
        """
        error = task.get("definition_error", "")

        if "missing task_class" in error or "invalid task_class" in error:
            # Try to infer task_class from command
            command = task.get("command", "").lower()

            inferred_class = None
            if any(kw in command for kw in ["whisper", "transcrib", "embed", "cuda", "gpu"]):
                inferred_class = "script"  # GPU compute task
            elif any(kw in command for kw in ["ollama", "generate", "prompt", "llm"]):
                inferred_class = "llm"  # Needs LLM model
            else:
                inferred_class = "cpu"  # Default to CPU

            # Apply the fix
            task["task_class"] = inferred_class
            task["status"] = "pending"
            task["fix_applied"] = f"inferred task_class='{inferred_class}' from command"
            del task["definition_error"]
            if "error_type" in task:
                del task["error_type"]
            task["result"] = {}  # Clear the error result

            self.logger.info(f"Fixed task '{task.get('name', '')}': {task['fix_applied']}")
            return True

        return False

    # =========================================================================
    # Resource Monitoring
    # =========================================================================

    def _detect_stuck_tasks(self) -> List[Dict]:
        """P0: Detect tasks that have been in processing state for too long.

        Returns list of stuck task info dicts with full task data.
        """
        stuck_tasks = []
        abort_threshold = 1200  # 20 minutes in seconds

        for task_file in self.processing_path.glob("*.json"):
            try:
                with open(task_file) as f:
                    task = json.load(f)

                started_at_str = task.get('started_at')
                if not started_at_str:
                    continue

                started = datetime.fromisoformat(started_at_str)
                elapsed = (datetime.now() - started).total_seconds()

                if elapsed > abort_threshold:
                    stuck_tasks.append({
                        'task': task,  # Full task object for abort handling
                        'task_file': task_file,
                        'task_id': task['task_id'][:8],
                        'assigned_to': task.get('assigned_to', 'unknown'),
                        'elapsed_min': int(elapsed / 60),
                        'elapsed_sec': int(elapsed),
                        'name': task.get('name', '')
                    })

            except Exception as e:
                self.logger.debug(f"Error checking task {task_file}: {e}")

        if stuck_tasks:
            summary = [{'task_id': t['task_id'], 'worker': t['assigned_to'], 'elapsed_min': t['elapsed_min']}
                      for t in stuck_tasks]
            self.logger.warning(f"Stuck tasks detected (>20 min): {summary}")

        return stuck_tasks

    def _send_abort_signal(self, worker_name: str, task_id: str, reason: str = "stuck_task"):
        """Send graceful abort signal to worker - asks it to stop current task."""
        signal_file = self.signals_path / f"{worker_name}.abort"
        signal_data = {
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "reason": reason
        }

        with open(signal_file, 'w') as f:
            json.dump(signal_data, f, indent=2)

        self.log_decision("ABORT_SIGNAL",
            f"Sent graceful abort to {worker_name} for task {task_id[:8]} (reason: {reason})")

    def _force_kill_worker_task(self, worker_name: str, task_id: str):
        """Force kill worker's subprocess if graceful abort failed."""
        signal_file = self.signals_path / f"{worker_name}.kill"
        signal_data = {
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "reason": "force_kill_stuck_task"
        }

        with open(signal_file, 'w') as f:
            json.dump(signal_data, f, indent=2)

        self.log_decision("FORCE_KILL",
            f"Sent force kill to {worker_name} for task {task_id[:8]} (graceful abort failed)")

    def _handle_stuck_tasks(self, stuck_tasks: List[Dict]):
        """Handle stuck tasks with graceful abort, escalating to force kill if needed."""
        for stuck_info in stuck_tasks:
            task = stuck_info['task']
            task_id = task['task_id']
            worker_name = stuck_info['assigned_to']
            elapsed_min = stuck_info['elapsed_min']

            # Check if we already sent an abort signal
            abort_signal = self.signals_path / f"{worker_name}.abort"
            kill_signal = self.signals_path / f"{worker_name}.kill"

            if kill_signal.exists():
                # Already sent kill signal, task is truly stuck
                # Last resort: could kill worker process entirely, but for now just log
                self.logger.error(
                    f"Task {task_id[:8]} still stuck after force kill signal ({elapsed_min} min). "
                    f"Worker {worker_name} may need manual intervention."
                )

            elif abort_signal.exists():
                # Abort signal sent, but task still stuck - check how long ago
                try:
                    with open(abort_signal) as f:
                        abort_data = json.load(f)
                    abort_time = datetime.fromisoformat(abort_data['timestamp'])
                    abort_age = (datetime.now() - abort_time).total_seconds()

                    if abort_age > 120:  # 2 minutes since abort signal
                        # Graceful abort failed, escalate to force kill
                        self._force_kill_worker_task(worker_name, task_id)
                except Exception:
                    pass

            else:
                # First detection - send graceful abort signal
                self._send_abort_signal(worker_name, task_id, f"task_timeout_{elapsed_min}min")

    def monitor_system(self):
        """Monitor GPU agent status, make resource allocation decisions.

        In brain-only mode (no worker GPUs configured), skips GPU monitoring
        but still checks for stuck tasks in the processing queue.
        """
        now = time.time()
        if now - self.last_monitor_check < self.monitor_interval:
            return
        self.last_monitor_check = now

        if self.brain_only:
            # Still check for stuck tasks even without GPU agents
            stuck_tasks = self._detect_stuck_tasks()
            if stuck_tasks:
                self._handle_stuck_tasks(stuck_tasks)
            return

        try:
            gpu_status = self._get_gpu_status()
            running_gpus = self._get_running_gpus()
            queue_stats = self._analyze_task_queue()

            # P0: Check for stuck tasks (processing > 20 minutes)
            stuck_tasks = self._detect_stuck_tasks()
            queue_stats['stuck_tasks'] = len(stuck_tasks)
            queue_stats['processing_count'] = len(list(self.processing_path.glob("*.json")))

            # Handle stuck tasks (send abort signals, escalate to force kill if needed)
            if stuck_tasks:
                self._handle_stuck_tasks(stuck_tasks)

            self._make_resource_decisions(gpu_status, running_gpus, queue_stats)

        except Exception as e:
            self.logger.warning(f"Monitor check failed: {e}")

    def _get_gpu_status(self) -> List[Dict]:
        """Get current GPU status from nvidia-smi."""
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )

        gpu_status = []
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    gpu_status.append({
                        "gpu": int(parts[0]),
                        "mem_used_mb": int(float(parts[1])),
                        "mem_total_mb": int(float(parts[2])),
                        "util_pct": int(float(parts[3])),
                        "power_w": float(parts[4])
                    })
        return gpu_status

    def _get_running_gpus(self) -> Dict[str, Dict]:
        """Check which GPU agent processes are actually running.

        Uses both pgrep (process detection) and heartbeat freshness.
        A GPU agent is considered running if:
        1. Its process exists (pgrep), OR
        2. Its heartbeat was updated within the last 60 seconds
        """
        running = {}

        for gpu_name, gpu_config in self.gpu_agents.items():
            gpu_id = gpu_config["id"]
            port = gpu_config.get("port")

            # Check process with regex pattern
            result = subprocess.run(
                ["pgrep", "-f", f"gpu.py.*{gpu_name}"],
                capture_output=True, text=True, timeout=5
            )

            if result.stdout.strip():
                pid = int(result.stdout.strip().split('\n')[0])
                running[gpu_name] = {"pid": pid, "gpu": gpu_id, "port": port}
                self.gpu_pids[gpu_name] = pid
            else:
                # Fallback: check GPU heartbeat freshness
                heartbeat_file = self.shared_path / "gpus" / f"gpu_{gpu_id}" / "heartbeat.json"
                if heartbeat_file.exists():
                    try:
                        with open(heartbeat_file) as f:
                            gpu_state = json.load(f)
                        last_updated = gpu_state.get("last_updated")
                        if last_updated:
                            timestamp = datetime.fromisoformat(last_updated)
                            age = (datetime.now() - timestamp).total_seconds()
                            if age < 60:
                                running[gpu_name] = {"pid": self.gpu_pids.get(gpu_name, 0),
                                                     "gpu": gpu_id, "port": port, "via_heartbeat": True}
                    except Exception:
                        pass

                if gpu_name not in running and gpu_name in self.gpu_pids:
                    del self.gpu_pids[gpu_name]

        return running

    def _analyze_task_queue(self) -> Dict:
        """Analyze pending tasks to determine resource needs."""
        stats = {
            "total_pending": 0,
            "cpu": 0,
            "script": 0,
            "llm": 0,
            "meta": 0,
            "brain_tasks": 0,
            "worker_tasks": 0
        }

        for task_file in self.queue_path.glob("*.json"):
            if task_file.suffix != ".json" or ".lock" in str(task_file):
                continue
            try:
                with open(task_file) as f:
                    task = json.load(f)

                stats["total_pending"] += 1
                executor = task.get("executor", "worker")

                if executor == "brain":
                    stats["brain_tasks"] += 1
                else:
                    stats["worker_tasks"] += 1

                task_class = task.get("task_class", "cpu")
                if task_class in stats:
                    stats[task_class] += 1

            except Exception as e:
                self.logger.debug(f"Failed to analyze task {task_file}: {e}")

        return stats

    def _get_gpu_states(self) -> Dict[str, Dict]:
        """Read GPU heartbeats to get current GPU agent states including model_loaded."""
        gpus_path = self.shared_path / "gpus"
        states = {}

        for gpu_name, gpu_config in self.gpu_agents.items():
            gpu_id = gpu_config["id"]
            heartbeat_file = gpus_path / f"gpu_{gpu_id}" / "heartbeat.json"
            if heartbeat_file.exists():
                try:
                    with open(heartbeat_file) as f:
                        gpu_state = json.load(f)
                    states[gpu_name] = gpu_state
                except Exception:
                    pass

        return states

    def _has_existing_meta_task(self, command: str) -> bool:
        """Check if a meta task with the given command already exists in queue or processing.

        Prevents duplicate meta tasks when the brain restarts or timing is tight
        between queue checks and task insertion.
        """
        # Check queue
        for task_file in self.queue_path.glob("*.json"):
            if str(task_file).endswith('.lock'):
                continue
            try:
                with open(task_file) as f:
                    task = json.load(f)
                if task.get("task_class") == "meta" and task.get("command") == command:
                    self.logger.debug(f"Dedup: {command} already in queue ({task['task_id'][:8]})")
                    return True
            except Exception:
                continue

        # Check processing
        for task_file in self.processing_path.glob("*.json"):
            if str(task_file).endswith('.lock'):
                continue
            try:
                with open(task_file) as f:
                    task = json.load(f)
                if task.get("task_class") == "meta" and task.get("command") == command:
                    self.logger.debug(f"Dedup: {command} already in processing ({task['task_id'][:8]})")
                    return True
            except Exception:
                continue

        return False

    def _insert_resource_task(self, command: str):
        """Insert a resource task (load_llm or unload_llm) for workers to claim.

        Performs a dedup scan first — skips insertion if the same command is already
        queued or in processing, preventing duplicate meta tasks after brain restart
        or under race conditions.
        """
        # Dedup: check if this exact command already exists
        if self._has_existing_meta_task(command):
            self.log_decision("RESOURCE_DEDUP",
                f"Skipping {command} — already exists in queue/processing", {})
            return

        task = {
            "task_id": str(uuid.uuid4()),
            "type": "meta",
            "command": command,
            "batch_id": "system",
            "name": command,
            "priority": 10,  # High priority
            "task_class": "meta",
            "depends_on": [],
            "executor": "worker",
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "created_by": self.name,
            "retry_count": 0
        }
        self.save_to_public(task)
        self.log_decision("RESOURCE_TASK", f"Inserted {command} task", {"task_id": task["task_id"][:8]})

        # Track load_llm requests to detect when GPUs don't pick them up
        if command == "load_llm":
            gpu_states = self._get_gpu_states()
            cold_gpus = [g for g, s in gpu_states.items() if not s.get("model_loaded", False)]
            self.load_llm_requests[task["task_id"]] = {
                'created_at': datetime.now(),
                'gpus_needed': cold_gpus.copy()
            }

    def _make_resource_decisions(self, gpu_status: List[Dict], running_gpus: Dict, queue_stats: Dict):
        """Make decisions about GPU resource allocation based on current state."""
        active_gpus = [g for g in gpu_status if g["util_pct"] > 10 or g["mem_used_mb"] > 1000]
        total_power = sum(g["power_w"] for g in gpu_status)

        # Get GPU agent states from heartbeats
        gpu_states = self._get_gpu_states()
        gpus_with_model = [g for g, s in gpu_states.items() if s.get("model_loaded", False)]
        gpus_without_model = [g for g, s in gpu_states.items() if not s.get("model_loaded", False)]

        # Track unhealthy GPUs (Ollama circuit breaker tripped)
        unhealthy_gpus = [g for g, s in gpu_states.items()
                          if not s.get("ollama_healthy", True)]
        if unhealthy_gpus:
            self.log_decision("GPU_UNHEALTHY",
                f"GPUs with unhealthy Ollama: {unhealthy_gpus}",
                {"unhealthy": unhealthy_gpus})

        # Only count healthy cold GPUs as candidates for load_llm
        healthy_cold_gpus = [g for g in gpus_without_model if g not in unhealthy_gpus]

        self.log_decision("MONITOR",
            f"GPUs active: {len(active_gpus)}/{len(gpu_status)}, "
            f"Agents: {len(running_gpus)}/{len(self.gpu_agents)}, "
            f"Queue: {queue_stats['total_pending']} (cpu:{queue_stats['cpu']}, script:{queue_stats['script']}, llm:{queue_stats['llm']}), "
            f"Processing: {queue_stats['processing_count']}, Stuck: {queue_stats['stuck_tasks']}, "
            f"Hot GPUs: {len(gpus_with_model)}",
            {
                "total_power_w": round(total_power, 1),
                "running_gpus": list(running_gpus.keys()),
                "hot_gpus": gpus_with_model,
                "queue_stats": queue_stats
            })

        # Track missing GPU agents with 3-miss tolerance
        missing_gpus = set(self.gpu_agents.keys()) - set(running_gpus.keys())

        for gpu_name in self.gpu_agents.keys():
            if gpu_name in missing_gpus:
                self.gpu_miss_count[gpu_name] = self.gpu_miss_count.get(gpu_name, 0) + 1
            else:
                self.gpu_miss_count[gpu_name] = 0

        truly_missing = [g for g in missing_gpus if self.gpu_miss_count.get(g, 0) >= 3]
        if truly_missing and queue_stats["worker_tasks"] > 0:
            self.log_decision("GPU_MISSING",
                f"GPU agents not running: {truly_missing} ({queue_stats['worker_tasks']} tasks pending)",
                {"missing": truly_missing, "miss_counts": {g: self.gpu_miss_count.get(g, 0) for g in missing_gpus}})

        # Check for stale load_llm requests
        for task_id, request in list(self.load_llm_requests.items()):
            age = (datetime.now() - request['created_at']).total_seconds()
            if age > 60:
                gpus_still_waiting = [g for g in request['gpus_needed'] if g in gpus_without_model]
                if gpus_still_waiting:
                    self.logger.warning(
                        f"load_llm task available for {age:.0f}s but GPUs {gpus_still_waiting} still cold"
                    )
                else:
                    del self.load_llm_requests[task_id]

        # Clean up completed load_llm requests
        for task_id in list(self.load_llm_requests.keys()):
            task_in_queue = (self.queue_path / f"{task_id}.json").exists()
            task_in_processing = (self.processing_path / f"{task_id}.json").exists()
            if not task_in_queue and not task_in_processing:
                del self.load_llm_requests[task_id]

        # Check if there's already a meta task in queue OR processing
        has_pending_resource = queue_stats.get("meta", 0) > 0

        if not has_pending_resource:
            for task_file in self.processing_path.glob("*.json"):
                try:
                    with open(task_file) as f:
                        task = json.load(f)
                    if task.get("task_class") == "meta":
                        has_pending_resource = True
                        self.logger.debug(f"Meta task already in processing: {task.get('command')}")
                        break
                except Exception:
                    pass

        if not has_pending_resource:
            # Need to load LLM? LLM tasks waiting but no GPUs are hot
            # Only consider healthy cold GPUs as candidates
            if queue_stats["llm"] > 0 and len(gpus_with_model) == 0 and len(healthy_cold_gpus) > 0:
                self.log_decision("RESOURCE_DECISION",
                    f"LLM tasks waiting ({queue_stats['llm']}) but no GPUs hot - inserting load_llm task",
                    {"llm_tasks": queue_stats["llm"], "cold_gpus": healthy_cold_gpus})
                self._insert_resource_task("load_llm")

            # Mixed workload: keep at least one hot GPU for llm while balancing script pressure.
            elif queue_stats["llm"] > 0 and queue_stats["script"] > 0:
                llm_tasks = queue_stats["llm"]
                script_tasks = queue_stats["script"]
                hot = len(gpus_with_model)
                cold = len(healthy_cold_gpus)
                script_to_llm = script_tasks / max(llm_tasks, 1)
                llm_to_script = llm_tasks / max(script_tasks, 1)

                # If llm backlog is dominant and we have healthy cold GPUs, add hot capacity.
                if llm_to_script >= 1.5 and cold > 0:
                    self.log_decision("RESOURCE_DECISION",
                        f"Mixed queue favors LLM ({llm_tasks} llm vs {script_tasks} script) - inserting load_llm task",
                        {"llm_tasks": llm_tasks, "script_tasks": script_tasks, "hot_gpus": gpus_with_model, "cold_gpus": healthy_cold_gpus})
                    self._insert_resource_task("load_llm")

                # If script backlog is dominant and >1 GPUs are hot, free one GPU.
                elif script_to_llm >= 3.0 and hot > 1:
                    self.log_decision("RESOURCE_DECISION",
                        f"Mixed queue favors script ({script_tasks} script vs {llm_tasks} llm) - inserting unload_llm task",
                        {"script_tasks": script_tasks, "llm_tasks": llm_tasks, "hot_gpus": gpus_with_model})
                    self._insert_resource_task("unload_llm")

            # Need to unload LLM? Only script tasks but GPUs are hot
            elif queue_stats["llm"] == 0 and queue_stats["script"] > 0 and len(gpus_with_model) > 0:
                self.log_decision("RESOURCE_DECISION",
                    f"Only script tasks waiting ({queue_stats['script']}) but GPUs hot - inserting unload_llm task",
                    {"script_tasks": queue_stats["script"], "hot_gpus": gpus_with_model})
                self._insert_resource_task("unload_llm")

    def _stop_gpu_agent(self, gpu_name: str):
        """Stop a GPU agent process gracefully via signal file."""
        signal_file = self.signals_path / f"{gpu_name}.stop"
        signal_file.write_text(datetime.now().isoformat())
        self.log_decision("GPU_STOP", f"Signaling {gpu_name} to stop", {"gpu": gpu_name})

    # =========================================================================
    # Main Loop
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

            # Signal ready to launcher
            flag_dir = Path("/tmp/llm-orchestration-flags")
            flag_dir.mkdir(parents=True, exist_ok=True)
            (flag_dir / f"{self.name}.ready").touch()

            while self.running:
                # 1. Check for brain tasks (execute_plan, decide, brain-targeted shell)
                self.claim_brain_tasks()

                # 2. Check and release tasks whose dependencies are met
                self.check_and_release_tasks()

                # 3. Handle failed tasks (retry logic)
                self.handle_failed_tasks()

                # 3b. Auto-clear stale failures when same task_id later succeeds
                self.clear_resolved_failures()

                # 4. Monitor GPU and worker health
                self.monitor_system()

                # Save state periodically
                self._save_brain_state()

                time.sleep(self.poll_interval)

        except KeyboardInterrupt:
            self.logger.info("Shutting down...")
        finally:
            self._save_brain_state()
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
