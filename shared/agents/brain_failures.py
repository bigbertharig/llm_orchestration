"""Brain failure handling and retry/escalation mixin.

Extracted from brain.py to isolate worker-result evaluation, retry policy,
auto-fixes, and cloud escalation gating.
"""

import json
import os
import re
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class BrainFailureMixin:
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

    def _result_text(self, task: Dict[str, Any], result: Dict[str, Any]) -> str:
        pieces = [
            str(result.get("output", "")),
            str(result.get("error", "")),
            str(task.get("blocked_reason", "")),
        ]
        return "\n".join(pieces).lower()

    def _is_recoverable_llm_timeout(self, task: Dict[str, Any], result: Dict[str, Any]) -> bool:
        if task.get("task_class") != "llm":
            return False
        text = self._result_text(task, result)
        return (
            "read timed out" in text
            or "no response from llm" in text
            or "httpconnectionpool" in text
            or "task timed out" in text
        )

    def _is_missing_scraped_file(self, result: Dict[str, Any]) -> bool:
        text = self._result_text({}, result)
        return "scraped file not found" in text

    def _extract_person_id(self, task: Dict[str, Any]) -> str:
        item_ids = task.get("item_ids", [])
        if item_ids and isinstance(item_ids, list):
            return str(item_ids[0])
        name = str(task.get("name", ""))
        if "_contact_" in name:
            return "contact_" + name.split("_contact_", 1)[1]
        if "contact_" in name:
            return name[name.find("contact_"):]
        return ""

    def _llm_fallback_models(self) -> List[str]:
        models = []
        # Configured fallback list has highest priority.
        cfg_models = self.config.get("retry_policy", {}).get("llm_fallback_models", [])
        if isinstance(cfg_models, list):
            for m in cfg_models:
                if isinstance(m, str) and m.strip():
                    models.append(m.strip())
        # Add worker models as baseline candidates.
        for g in self.config.get("gpus", []):
            m = str(g.get("model", "")).strip()
            if m:
                models.append(m)

        dedup = []
        seen = set()
        for m in models:
            if m not in seen:
                seen.add(m)
                dedup.append(m)
        return dedup

    def _set_next_llm_model(self, task: Dict[str, Any]) -> str:
        candidates = self._llm_fallback_models()
        if not candidates:
            return ""

        # Try to infer current model from command.
        command = str(task.get("command", ""))
        current_model = ""
        model_match = re.search(r"--model\s+([^\s]+)", command)
        if model_match:
            current_model = model_match.group(1).strip().strip("'\"")

        if current_model and current_model in candidates:
            start_idx = (candidates.index(current_model) + 1) % len(candidates)
        else:
            start_idx = int(task.get("llm_model_retry_index", 0)) % len(candidates)

        next_model = candidates[start_idx]
        task["llm_model_retry_index"] = (start_idx + 1) % len(candidates)

        if "--model" in command:
            new_cmd = re.sub(
                r"--model\s+[^\s]+",
                f"--model {next_model}",
                command,
                count=1
            )
        else:
            new_cmd = f"{command} --model {next_model}"

        task["command"] = new_cmd
        task["llm_retry_model"] = next_model
        return next_model

    def _queue_task_retry(self, task_file: Path, task: Dict[str, Any], reason: str):
        task["status"] = "pending"
        task["attempts"] = 0
        task["workers_attempted"] = []
        task["retry_recovered_by_brain"] = True
        task["retry_recovery_reason"] = reason
        task.pop("cloud_escalated", None)
        task.pop("cloud_escalation_id", None)
        task.pop("incident_id", None)
        task.pop("blocked_reason", None)
        task_file.unlink(missing_ok=True)
        self.save_to_public(task)

    def _abort_batch(self, batch_id: str, reason: str, source_task: Dict[str, Any]):
        """Terminate a batch and move pending tasks to failed/abandoned."""
        if not batch_id:
            return

        now = datetime.now().isoformat()
        batch_meta = self.active_batches.get(batch_id, {})
        aborted_count = 0
        source_task_id = source_task.get("task_id")

        for lane in (self.queue_path, self.private_tasks_path):
            for task_file in lane.glob("*.json"):
                try:
                    with open(task_file) as f:
                        task = json.load(f)
                except Exception:
                    continue

                if task.get("batch_id") != batch_id:
                    continue
                if source_task_id and task.get("task_id") == source_task_id:
                    continue

                task["status"] = "abandoned"
                task["completed_at"] = now
                task["abandoned_reason"] = "batch_aborted_after_fatal_task"
                task["result"] = {
                    "success": False,
                    "error": f"Batch aborted: {reason}",
                    "error_type": "batch_aborted",
                    "handler": "brain",
                }
                try:
                    task_file.unlink(missing_ok=True)
                except Exception:
                    pass

                dest_file = self.failed_path / f"{task.get('task_id')}.json"
                with open(dest_file, "w") as f:
                    json.dump(task, f, indent=2)
                aborted_count += 1

        for task_file in self.processing_path.glob("*.json"):
            try:
                with open(task_file) as f:
                    task = json.load(f)
            except Exception:
                continue
            if task.get("batch_id") != batch_id:
                continue
            worker_name = task.get("assigned_to") or task.get("worker")
            task_id = task.get("task_id", "")
            if worker_name and task_id:
                self._send_abort_signal(worker_name, task_id, reason="batch_aborted")

        if batch_meta:
            goal = batch_meta.get("goal")
            if isinstance(goal, dict):
                goal["status"] = "failed"
                goal["failed_at"] = now
                goal["failure_reason"] = reason
            batch_meta["status"] = "failed"
            batch_meta["failed_at"] = now
            batch_meta["failure_reason"] = reason

        batch_dir = batch_meta.get("batch_dir")
        if batch_dir:
            try:
                out = Path(batch_dir) / "results" / "batch_failure.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                with open(out, "w") as f:
                    json.dump(
                        {
                            "batch_id": batch_id,
                            "status": "failed",
                            "failed_at": now,
                            "reason": reason,
                            "source_task": source_task.get("name", ""),
                            "source_task_id": source_task.get("task_id", ""),
                            "abandoned_tasks": aborted_count,
                        },
                        f,
                        indent=2,
                    )
            except Exception as e:
                self.logger.warning(f"Failed writing batch failure artifact for {batch_id}: {e}")

        self.log_decision(
            "BATCH_ABORTED",
            f"Aborted batch {batch_id} due to fatal task failure",
            {
                "batch_id": batch_id,
                "source_task": source_task.get("name", ""),
                "source_task_id": str(source_task.get("task_id", ""))[:8],
                "reason": reason[:300],
                "abandoned_tasks": aborted_count,
            },
        )
        if batch_id in self.active_batches:
            del self.active_batches[batch_id]
        self._save_brain_state()

    def _requeue_upstream_scrape_for_person(self, batch_id: str, person_id: str) -> bool:
        if not batch_id or not person_id:
            return False
        task_name = f"execute_and_scrape_{person_id}"
        for task_file in self.failed_path.glob("*.json"):
            try:
                with open(task_file) as f:
                    t = json.load(f)
                if t.get("batch_id") != batch_id or t.get("name") != task_name:
                    continue
                self._queue_task_retry(task_file, t, "auto_recover_upstream_scrape")
                self.log_decision(
                    "RECOVERABLE_REQUEUE",
                    f"Re-queued upstream scrape for {person_id}",
                    {"batch_id": batch_id, "task_name": task_name}
                )
                return True
            except Exception:
                continue
        return False

    def _extract_permission_denied_path(self, result: Dict[str, Any]) -> str:
        """Extract denied filesystem path from Python PermissionError output."""
        text = (
            str(result.get("output", "") or "")
            + "\n"
            + str(result.get("error", "") or "")
        )
        m = re.search(r"PermissionError:\s*\[Errno\s*13\]\s*Permission denied:\s*'([^']+)'", text)
        return m.group(1).strip() if m else ""

    def _try_fix_permission_denied(self, task: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """
        Attempt to recover from file/dir PermissionError in plan artifacts.

        Strategy:
        1) Parse denied path from traceback.
        2) Ensure parent dirs exist.
        3) Relax perms for plan batch tree to group writable.
        """
        denied_path = self._extract_permission_denied_path(result)
        if not denied_path:
            return False

        candidates: List[Path] = [Path(denied_path)]
        if denied_path.startswith("/mnt/shared"):
            candidates.append(Path(denied_path.replace("/mnt/shared", "/media/bryan/shared", 1)))
        elif denied_path.startswith("/media/bryan/shared"):
            candidates.append(Path(denied_path.replace("/media/bryan/shared", "/mnt/shared", 1)))

        fixed = False
        for denied in candidates:
            try:
                parent = denied.parent
                parent.mkdir(parents=True, exist_ok=True)

                # If we have batch_path, normalize from there down so future writes work.
                batch_path = str(task.get("batch_path", "") or "")
                batch_root = Path(batch_path) if batch_path else None
                if batch_root and str(batch_root).startswith("/mnt/shared"):
                    batch_root = Path(str(batch_root).replace("/mnt/shared", "/media/bryan/shared", 1))
                if batch_root and batch_root.exists():
                    for d in [batch_root, *batch_root.glob("**/*")]:
                        if d.is_dir():
                            try:
                                os.chmod(d, 0o775)
                            except Exception:
                                pass
                        elif d.is_file():
                            try:
                                os.chmod(d, 0o664)
                            except Exception:
                                pass

                try:
                    os.chmod(parent, 0o775)
                except Exception:
                    pass
                if denied.exists():
                    try:
                        os.chmod(denied, 0o664)
                    except Exception:
                        pass

                # Verify write access in target directory.
                probe = parent / f".perm_probe_{uuid.uuid4().hex}.tmp"
                with open(probe, "w", encoding="utf-8") as f:
                    f.write("ok")
                probe.unlink(missing_ok=True)
                fixed = True
                break
            except Exception:
                continue

        if fixed:
            task["fix_applied"] = "permission_recovered"
        return fixed

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

                if task.get("status") == "abandoned":
                    continue

                # Use task memory fields (attempts, workers_attempted)
                attempts = task.get("attempts", 0)
                workers = task.get("workers_attempted", [])

                if str(task.get("task_class", "")).lower() == "brain":
                    reason = (
                        str(result.get("error", "")).strip()
                        or str(result.get("output", "")).strip()[:400]
                        or "brain task failed"
                    )
                    self.log_decision(
                        "BRAIN_TASK_ABORT",
                        f"Brain task failure triggers batch abort: {task.get('name', '')}",
                        {
                            "task_id": str(task.get("task_id", ""))[:8],
                            "batch_id": task.get("batch_id", ""),
                            "reason": reason[:300],
                        },
                    )
                    self._abort_batch(task.get("batch_id", ""), reason, task)
                    task["status"] = "abandoned"
                    task["abandoned_reason"] = "brain_task_failed_abort_batch"
                    task["result"]["error_type"] = "brain_task_failure_handled"
                    with open(task_file, "w") as f:
                        json.dump(task, f, indent=2)
                    continue

                # Recoverable blocked tasks are re-queued automatically.
                if task.get("status") == "blocked_cloud" or task.get("cloud_escalated", False):
                    recoverable_timeout = self._is_recoverable_llm_timeout(task, result)
                    recoverable_missing_scrape = self._is_missing_scraped_file(result)
                    recoverable_permission = self._try_fix_permission_denied(task, result)
                    if recoverable_timeout or recoverable_missing_scrape or recoverable_permission:
                        if recoverable_timeout:
                            model_used = self._set_next_llm_model(task)
                            self._queue_task_retry(task_file, task, "recoverable_llm_timeout")
                            self.log_decision(
                                "RECOVERABLE_REQUEUE",
                                f"Recovered blocked LLM timeout task '{task.get('name', '')}'",
                                {
                                    "task_id": task.get("task_id", "")[:8],
                                    "next_model": model_used or "unchanged"
                                }
                            )
                        elif recoverable_missing_scrape:
                            person_id = self._extract_person_id(task)
                            self._requeue_upstream_scrape_for_person(task.get("batch_id", ""), person_id)
                            self._queue_task_retry(task_file, task, "recoverable_missing_scrape")
                            self.log_decision(
                                "RECOVERABLE_REQUEUE",
                                f"Recovered blocked missing-scrape task '{task.get('name', '')}'",
                                {
                                    "task_id": task.get("task_id", "")[:8],
                                    "person_id": person_id
                                }
                            )
                        else:
                            self._queue_task_retry(task_file, task, "recoverable_permission_denied")
                            self.log_decision(
                                "RECOVERABLE_REQUEUE",
                                f"Recovered blocked permission error task '{task.get('name', '')}'",
                                {
                                    "task_id": task.get("task_id", "")[:8],
                                    "denied_path": self._extract_permission_denied_path(result),
                                }
                            )
                        continue
                    continue

                if error_type == "fatal":
                    reason = (
                        str(result.get("error", "")).strip()
                        or str(result.get("output", "")).strip()[:400]
                        or "fatal task failure"
                    )
                    self._abort_batch(task.get("batch_id", ""), reason, task)
                    task["status"] = "abandoned"
                    task["abandoned_reason"] = "batch_aborted_after_fatal_task"
                    task["result"]["error_type"] = "fatal_handled"
                    with open(task_file, "w") as f:
                        json.dump(task, f, indent=2)
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
                    if self._is_recoverable_llm_timeout(task, result):
                        self._set_next_llm_model(task)

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
                        if not fixed and self._is_recoverable_llm_timeout(task, result):
                            self._set_next_llm_model(task)
                            fixed = True
                            task["fix_applied"] = f"rotated_llm_model:{task.get('llm_retry_model', '')}"
                        if (not fixed) and self._is_missing_scraped_file(result):
                            person_id = self._extract_person_id(task)
                            self._requeue_upstream_scrape_for_person(task.get("batch_id", ""), person_id)
                            fixed = True
                            task["fix_applied"] = "requeued_upstream_scrape"
                        if (not fixed) and self._try_fix_permission_denied(task, result):
                            fixed = True
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

