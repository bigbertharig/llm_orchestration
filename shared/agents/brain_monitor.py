"""Brain monitoring and worker orchestration mixin.

Extracted from brain.py to isolate runtime monitoring, stuck-task handling,
and resource-state scans.
"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class BrainMonitorMixin:
    def _detect_stuck_tasks(self, running_workers: Dict[str, Dict]) -> List[Dict]:
        """
        Detect tasks with no progress beyond class-specific thresholds.

        Progress is inferred from per-task heartbeat sidecars written by GPU agents:
          tasks/processing/<task_id>.heartbeat.json
        """
        stuck_tasks = []
        gpu_states = self._get_gpu_states()
        prev_thermal_wait = set(self.thermal_wait_logged)
        current_thermal_wait = set()
        now = datetime.now()

        for task_file in self.processing_path.glob("*.json"):
            try:
                with open(task_file) as f:
                    task = json.load(f)

                task_id = task.get("task_id", "")
                if not task_id:
                    continue

                assigned_to = task.get("assigned_to", "")
                if not assigned_to or assigned_to not in running_workers:
                    # Orphan handling is done separately.
                    continue

                task_class = task.get("task_class", "cpu")
                threshold = int(self.task_no_progress_thresholds.get(task_class, 900))

                started_at_str = task.get('started_at') or task.get('last_attempt_at')
                if not started_at_str:
                    continue
                started = datetime.fromisoformat(started_at_str)
                elapsed = (now - started).total_seconds()

                # Default progress_age: total elapsed since start.
                progress_age = elapsed
                progress_source = "started_at"

                hb_file = self.processing_path / f"{task_id}.heartbeat.json"
                if hb_file.exists():
                    try:
                        with open(hb_file) as f:
                            hb = json.load(f)
                        hb_time = hb.get("updated_at")
                        if hb_time:
                            hb_dt = datetime.fromisoformat(hb_time)
                            progress_age = (now - hb_dt).total_seconds()
                            progress_source = "task_heartbeat"
                    except Exception:
                        pass

                if progress_age > threshold:
                    # Distinguish thermal waiting from true stuck:
                    # if GPU heartbeat says thermal-constrained/paused, do not
                    # trigger stuck kill/requeue flow yet.
                    gpu_state = gpu_states.get(assigned_to, {})
                    thermal_wait = bool(
                        gpu_state.get("thermal_pause_active")
                        or gpu_state.get("thermal_constrained")
                    )
                    if thermal_wait:
                        current_thermal_wait.add(task_id)
                        if task_id not in prev_thermal_wait:
                            self.log_decision(
                                "TASK_WAITING_THERMAL",
                                (
                                    f"Task {task_id[:8]} waiting due to thermal constraint "
                                    f"on {assigned_to}"
                                ),
                                {
                                    "task_id": task_id[:8],
                                    "worker": assigned_to,
                                    "progress_age_sec": int(progress_age),
                                    "threshold_sec": threshold,
                                    "cpu_temp_c": gpu_state.get("cpu_temp_c"),
                                    "gpu_temp_c": gpu_state.get("temperature_c"),
                                    "thermal_reasons": gpu_state.get("thermal_reasons", []),
                                },
                            )
                        continue

                    stuck_tasks.append({
                        'task': task,
                        'task_file': task_file,
                        'task_id': task_id[:8],
                        'assigned_to': assigned_to,
                        'elapsed_min': int(elapsed / 60),
                        'elapsed_sec': int(elapsed),
                        'progress_age_sec': int(progress_age),
                        'progress_source': progress_source,
                        'threshold_sec': threshold,
                        'name': task.get('name', '')
                    })

            except Exception as e:
                self.logger.debug(f"Error checking task {task_file}: {e}")

        # Log thermal wait clear transitions (either temps recovered, task moved,
        # or task completed).
        for task_id in sorted(prev_thermal_wait - current_thermal_wait):
            self.log_decision(
                "TASK_WAITING_THERMAL_CLEARED",
                f"Task {task_id[:8]} no longer in thermal-wait state",
                {"task_id": task_id[:8]}
            )
        self.thermal_wait_logged = current_thermal_wait
        self.last_thermal_wait_count = len(current_thermal_wait)

        if stuck_tasks:
            summary = [
                {
                    'task_id': t['task_id'],
                    'worker': t['assigned_to'],
                    'progress_age_sec': t['progress_age_sec'],
                    'threshold_sec': t['threshold_sec'],
                }
                for t in stuck_tasks
            ]
            self.logger.warning(f"No-progress stuck tasks detected: {summary}")

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

        self.log_decision(
            "TASK_ABORT_SIGNAL",
            f"Sent graceful abort to {worker_name} for task {task_id[:8]} (reason: {reason})",
            {"worker": worker_name, "task_id": task_id[:8], "reason": reason}
        )

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

        self.log_decision(
            "TASK_FORCE_KILL_SIGNAL",
            f"Sent force kill to {worker_name} for task {task_id[:8]} (graceful abort failed)",
            {"worker": worker_name, "task_id": task_id[:8]}
        )

    def _handle_stuck_tasks(self, stuck_tasks: List[Dict]):
        """Handle stuck tasks with abort -> force-kill -> requeue recovery."""
        for stuck_info in stuck_tasks:
            task = stuck_info['task']
            task_id = task['task_id']
            worker_name = stuck_info['assigned_to']
            elapsed_min = stuck_info['elapsed_min']
            task_file = stuck_info['task_file']
            now = datetime.now()
            incident = self._get_or_create_incident(task, {
                "success": False,
                "error": "stuck_no_progress",
                "reason": (
                    f"no_progress_{stuck_info.get('progress_age_sec', 0)}s"
                    f"_threshold_{stuck_info.get('threshold_sec', 0)}s"
                ),
            })
            incident["updated_at"] = now.isoformat()

            abort_sent_at = incident.get("stuck_abort_sent_at")
            kill_sent_at = incident.get("stuck_force_kill_sent_at")
            abort_count = int(incident.get("stuck_abort_count", 0))

            if not abort_sent_at:
                self._send_abort_signal(
                    worker_name,
                    task_id,
                    (
                        f"no_progress_{stuck_info.get('progress_age_sec', 0)}s"
                        f"_threshold_{stuck_info.get('threshold_sec', 0)}s"
                    )
                )
                incident["stuck_abort_sent_at"] = now.isoformat()
                incident["stuck_abort_count"] = abort_count + 1
                incident["history"].append({
                    "at": now.isoformat(),
                    "event": "stuck_abort_sent",
                    "task_id": task_id,
                    "worker": worker_name,
                })
                continue

            if not kill_sent_at:
                # User policy: if abort has been sent twice for the same stuck task,
                # escalate immediately to force-kill.
                if abort_count < 2:
                    self._send_abort_signal(
                        worker_name,
                        task_id,
                        (
                            f"no_progress_{stuck_info.get('progress_age_sec', 0)}s"
                            f"_threshold_{stuck_info.get('threshold_sec', 0)}s"
                        )
                    )
                    incident["stuck_abort_sent_at"] = now.isoformat()
                    incident["stuck_abort_count"] = abort_count + 1
                    incident["history"].append({
                        "at": now.isoformat(),
                        "event": "stuck_abort_sent_repeat",
                        "task_id": task_id,
                        "worker": worker_name,
                        "abort_count": incident["stuck_abort_count"],
                    })
                    continue

                self._force_kill_worker_task(worker_name, task_id)
                incident["stuck_force_kill_sent_at"] = now.isoformat()
                incident["history"].append({
                    "at": now.isoformat(),
                    "event": "stuck_force_kill_sent",
                    "task_id": task_id,
                    "worker": worker_name,
                })
                continue

            # Force kill was already sent. If task still hasn't moved after grace,
            # requeue from processing regardless of signal-file lifecycle.
            try:
                kill_age = (now - datetime.fromisoformat(kill_sent_at)).total_seconds()
            except Exception:
                kill_age = 0
            if kill_age < self.force_kill_requeue_seconds:
                continue

            if not task_file.exists():
                continue

            try:
                with open(task_file) as f:
                    current = json.load(f)
            except Exception:
                continue

            if current.get("task_id") != task_id:
                continue

            self._prepare_task_for_requeue(current, "force_kill_timeout")
            current["stuck_requeue_count"] = int(current.get("stuck_requeue_count", 0)) + 1
            self._assert_queue_requeue_invariants(current)

            queue_file = self.queue_path / task_file.name
            with open(queue_file, 'w') as f:
                json.dump(current, f, indent=2)
            task_file.unlink()

            hb_file = self.processing_path / f"{task_id}.heartbeat.json"
            if hb_file.exists():
                try:
                    hb_file.unlink()
                except Exception:
                    pass

            # Best-effort cleanup of stale control signals for this worker.
            for suffix in ("abort", "kill"):
                sig = self.signals_path / f"{worker_name}.{suffix}"
                if sig.exists():
                    try:
                        with open(sig) as f:
                            sig_data = json.load(f)
                        if sig_data.get("task_id") == task_id:
                            sig.unlink()
                    except Exception:
                        pass

            incident["history"].append({
                "at": now.isoformat(),
                "event": "stuck_requeued_after_force_kill_timeout",
                "task_id": task_id,
                "worker": worker_name,
            })

            self.log_decision(
                "TASK_REQUEUED_TIMEOUT",
                f"Requeued stuck task {task_id[:8]} after kill timeout",
                {
                    "task_id": task_id[:8],
                    "worker": worker_name,
                    "elapsed_min": elapsed_min,
                    "progress_age_sec": stuck_info.get("progress_age_sec", 0),
                    "threshold_sec": stuck_info.get("threshold_sec", 0),
                    "requeue_reason": "force_kill_timeout",
                }
            )

    def _recover_orphaned_processing_tasks(self, running_workers: Dict[str, Dict]):
        """
        Detect and recover orphaned processing tasks.

        Orphaned = task in processing assigned to a worker that is not running.
        These should be escalated/requeued quickly instead of waiting for the
        20-minute stuck-task timeout.
        """
        orphan_threshold_seconds = self.heartbeat_stale_seconds
        recovered = 0

        for task_file in self.processing_path.glob("*.json"):
            try:
                with open(task_file) as f:
                    task = json.load(f)

                assigned_to = task.get("assigned_to", "")
                task_id = str(task.get("task_id", "")).strip()
                if not assigned_to:
                    continue
                worker_state = running_workers.get(assigned_to)
                task_hb_file = self.processing_path / f"{task_id}.heartbeat.json"

                # Fast-path: worker no longer running at all.
                if worker_state is None:
                    is_orphan = True
                else:
                    # Worker is alive, but verify it still claims this task.
                    active_task_ids = [
                        str(x).strip() for x in (worker_state.get("active_task_ids", []) or []) if str(x).strip()
                    ]
                    short_task_id = task_id[:8]
                    # Backward compatibility with older GPU heartbeats that reported
                    # truncated 8-char task IDs.
                    is_claimed = bool(
                        task_id and (
                            task_id in active_task_ids
                            or short_task_id in active_task_ids
                            or any(task_id.startswith(active_id) for active_id in active_task_ids if len(active_id) == 8)
                        )
                    )
                    if is_claimed:
                        continue

                    # Worker heartbeat does not claim this task; only recover
                    # once task heartbeat is stale to avoid false positives.
                    hb_age = None
                    if task_hb_file.exists():
                        try:
                            with open(task_hb_file) as f:
                                hb = json.load(f)
                            hb_updated = hb.get("updated_at")
                            if hb_updated:
                                hb_age = (datetime.now() - datetime.fromisoformat(str(hb_updated))).total_seconds()
                        except Exception:
                            hb_age = None
                    if hb_age is not None and hb_age < orphan_threshold_seconds:
                        continue
                    is_orphan = True

                if not is_orphan:
                    continue

                started_at_str = task.get("started_at") or task.get("last_attempt_at")
                if not started_at_str:
                    continue

                started = datetime.fromisoformat(started_at_str)
                elapsed = (datetime.now() - started).total_seconds()
                if elapsed < orphan_threshold_seconds:
                    continue

                self._prepare_task_for_requeue(task, "orphan_recovered")
                task["orphan_recovered_at"] = datetime.now().isoformat()
                task["orphan_recovered_from"] = assigned_to
                self._assert_queue_requeue_invariants(task)

                # Move back to queue for reclaim
                new_path = self.queue_path / task_file.name
                with open(new_path, 'w') as f:
                    json.dump(task, f, indent=2)
                task_file.unlink()
                recovered += 1

                self.log_decision(
                    "ORPHAN_REQUEUE",
                    f"Recovered orphaned processing task {task.get('name', task.get('task_id','')[:8])}",
                    {
                        "task_id": task.get("task_id", "")[:8],
                        "batch_id": task.get("batch_id"),
                        "previous_worker": assigned_to,
                        "elapsed_sec": int(elapsed),
                        "worker_running": worker_state is not None,
                        "worker_claimed_task": bool(worker_state is not None and is_claimed),
                        "action": "requeued"
                    }
                )

            except Exception as e:
                self.logger.debug(f"Error recovering orphaned task {task_file}: {e}")

        return recovered

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
            stuck_tasks = self._detect_stuck_tasks(running_workers={})
            if stuck_tasks:
                self._handle_stuck_tasks(stuck_tasks)
            return

        try:
            gpu_status = self._get_gpu_status()
            running_gpus = self._get_running_gpus()
            running_workers = self._get_running_workers(running_gpus)
            queue_stats = self._analyze_task_queue()

            # Recover orphaned processing tasks promptly when a worker/gpu agent
            # disappears. This prevents long silent stalls.
            orphan_recovered = self._recover_orphaned_processing_tasks(running_workers)
            if orphan_recovered:
                queue_stats = self._analyze_task_queue()
                queue_stats["orphan_recovered"] = orphan_recovered

            # P0: Check for stuck tasks (processing > 20 minutes)
            stuck_tasks = self._detect_stuck_tasks(running_workers)
            queue_stats['stuck_tasks'] = len(stuck_tasks)
            queue_stats['thermal_wait_tasks'] = self.last_thermal_wait_count
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
            active_task_ids: list[str] = []
            gpu_state: Dict[str, Any] = {}
            heartbeat_file = self.shared_path / "gpus" / f"gpu_{gpu_id}" / "heartbeat.json"
            if heartbeat_file.exists():
                try:
                    with open(heartbeat_file) as f:
                        gpu_state = json.load(f)
                    for at in gpu_state.get("active_tasks", []) or []:
                        if not isinstance(at, dict):
                            continue
                        at_id = str(at.get("task_id") or "").strip()
                        if at_id:
                            active_task_ids.append(at_id)
                except Exception:
                    gpu_state = {}

            # Check process with regex pattern
            result = subprocess.run(
                ["pgrep", "-f", f"gpu.py.*{gpu_name}"],
                capture_output=True, text=True, timeout=5
            )

            if result.stdout.strip():
                pid = int(result.stdout.strip().split('\n')[0])
                running[gpu_name] = {
                    "pid": pid,
                    "gpu": gpu_id,
                    "port": port,
                    "active_task_ids": active_task_ids,
                }
                self.gpu_pids[gpu_name] = pid
            else:
                # Fallback: check GPU heartbeat freshness
                if heartbeat_file.exists():
                    try:
                        last_updated = gpu_state.get("last_updated")
                        if last_updated:
                            timestamp = datetime.fromisoformat(last_updated)
                            age = (datetime.now() - timestamp).total_seconds()
                            if age < self.heartbeat_stale_seconds:
                                running[gpu_name] = {"pid": self.gpu_pids.get(gpu_name, 0),
                                                     "gpu": gpu_id, "port": port,
                                                     "via_heartbeat": True,
                                                     "active_task_ids": active_task_ids}
                    except Exception:
                        pass

                if gpu_name not in running and gpu_name in self.gpu_pids:
                    del self.gpu_pids[gpu_name]

        return running

    def _get_running_cpu_workers(self) -> Dict[str, Dict]:
        """Return CPU workers with fresh heartbeats."""
        running: Dict[str, Dict] = {}
        cpus_path = self.shared_path / "cpus"
        if not cpus_path.exists():
            return running

        now = datetime.now()
        for hb_file in cpus_path.glob("*/heartbeat.json"):
            try:
                with open(hb_file) as f:
                    hb = json.load(f)
                worker_name = str(hb.get("name") or "").strip()
                if not worker_name:
                    continue
                updated_at = hb.get("last_updated")
                if not updated_at:
                    continue
                age = (now - datetime.fromisoformat(str(updated_at))).total_seconds()
                if age >= self.heartbeat_stale_seconds:
                    continue
                running[worker_name] = {
                    "type": "cpu",
                    "host": hb.get("hostname"),
                    "via_heartbeat": True,
                    "state": hb.get("state"),
                    "active_task_ids": (
                        [str(hb.get("active_task_id")).strip()]
                        if str(hb.get("active_task_id") or "").strip()
                        else []
                    ),
                }
            except Exception:
                continue

        return running

    def _get_running_workers(self, running_gpus: Dict[str, Dict] | None = None) -> Dict[str, Dict]:
        """Return all running workers (GPU + CPU) using consistent freshness rules."""
        workers: Dict[str, Dict] = {}
        if running_gpus is None:
            running_gpus = self._get_running_gpus()
        workers.update(running_gpus)
        workers.update(self._get_running_cpu_workers())
        return workers

    def _analyze_task_queue(self) -> Dict:
        """Analyze pending tasks to determine resource needs."""
        stats = {
            "total_pending": 0,
            "cpu": 0,
            "script": 0,
            "llm": 0,
            "llm_split_required": 0,
            "llm_max_tier": 0,
            "llm_model_demand": {},
            "llm_split_model_demand": {},
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
                if task_class == "llm":
                    placement = str(task.get("llm_placement", "")).strip()
                    llm_model = str(task.get("llm_model", "")).strip()
                    if llm_model:
                        stats["llm_model_demand"][llm_model] = (
                            int(stats["llm_model_demand"].get(llm_model, 0)) + 1
                        )
                        model_meta = self.model_meta_by_id.get(llm_model, {})
                        if str(model_meta.get("placement", "")) == "split_gpu":
                            stats["llm_split_model_demand"][llm_model] = (
                                int(stats["llm_split_model_demand"].get(llm_model, 0)) + 1
                            )
                    if placement == "split_gpu":
                        stats["llm_split_required"] += 1
                    try:
                        llm_tier = int(task.get("llm_min_tier", self.default_llm_min_tier) or self.default_llm_min_tier)
                    except Exception:
                        llm_tier = self.default_llm_min_tier
                    stats["llm_max_tier"] = max(stats["llm_max_tier"], llm_tier)

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

    def _stop_gpu_agent(self, gpu_name: str):
        """Stop a GPU agent process gracefully via signal file."""
        signal_file = self.signals_path / f"{gpu_name}.stop"
        signal_file.write_text(datetime.now().isoformat())
        self.log_decision("GPU_STOP", f"Signaling {gpu_name} to stop", {"gpu": gpu_name})

    # =========================================================================
    # Main Loop
    # =========================================================================
