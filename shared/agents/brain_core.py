"""Brain core services mixin.

Extracted from brain.py to isolate config/model catalog loading, state
persistence, decision logging/escalation, incident bookkeeping, and Ollama
client lifecycle helpers.
"""

import json
import os
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import requests

DEFAULT_LLM_MIN_TIER = 1


class BrainCoreMixin:
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

