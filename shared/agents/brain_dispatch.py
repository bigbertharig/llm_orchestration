"""Brain local task dispatch and execution mixin.

Extracted from brain.py to isolate brain-only task claiming and execution
(execute_plan/decide/brain shell tasks).
"""

import json
import os
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from filelock import FileLock, Timeout


class BrainDispatchMixin:
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
        batch_id = str(task.get("batch_id", "")).strip()
        if batch_id:
            event = "task_succeeded" if task.get("result", {}).get("success") else "task_failed"
            self._append_batch_event(batch_id, event, self._task_payload(task))
            self._refresh_batch_summary(batch_id)

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
        batch_id = str(task.get("batch_id", "")).strip()
        if batch_id:
            self._append_batch_event(batch_id, "task_started", self._task_payload(task))

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
        batch_id = str(task.get("batch_id", "")).strip()
        if batch_id:
            event = "task_succeeded" if task["result"].get("success") else "task_failed"
            self._append_batch_event(batch_id, event, self._task_payload(task))
            self._refresh_batch_summary(batch_id)

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
        batch_id = str(task.get("batch_id", "")).strip()
        if batch_id:
            self._append_batch_event(batch_id, "task_succeeded", self._task_payload(task))
            self._refresh_batch_summary(batch_id)

    def handle_system_task(self, task: Dict[str, Any]):
        """Handle brain-level system commands (orchestrator_full_reset, etc.)."""
        command = task.get("command", "")
        task_id = task.get("task_id", "")[:8]
        reason = task.get("reason", "manual")
        incident_id = task.get("incident_id", "")

        self.log_decision(
            "SYSTEM_TASK_START",
            f"Executing system command: {command}",
            {"task_id": task_id, "reason": reason, "incident_id": incident_id},
        )

        task["status"] = "processing"
        task["started_at"] = datetime.now().isoformat()
        processing_file = self.processing_path / f"{task['task_id']}.json"
        with open(processing_file, "w") as f:
            json.dump(task, f, indent=2)
        batch_id = str(task.get("batch_id", "")).strip()
        if batch_id:
            self._append_batch_event(batch_id, "task_started", self._task_payload(task))

        try:
            if command == "orchestrator_full_reset":
                result = self._execute_orchestrator_full_reset(task)
            else:
                result = {
                    "success": False,
                    "error": f"Unknown system command: {command}",
                    "handler": "brain",
                }
        except Exception as e:
            result = {
                "success": False,
                "error": str(e),
                "handler": "brain",
                "traceback": traceback.format_exc(),
            }

        # Complete the task
        task["status"] = "complete" if result.get("success") else "failed"
        task["result"] = result
        task["completed_at"] = datetime.now().isoformat()

        # Clean up processing file
        if processing_file.exists():
            processing_file.unlink()

        dest_base = self.complete_path if result.get("success") else self.failed_path
        dest_file = dest_base / f"{task['task_id']}.json"
        with open(dest_file, "w") as f:
            json.dump(task, f, indent=2)
        if batch_id:
            event = "task_succeeded" if result.get("success") else "task_failed"
            self._append_batch_event(batch_id, event, self._task_payload(task))
            self._refresh_batch_summary(batch_id)

        self.log_decision(
            "SYSTEM_TASK_COMPLETE" if result.get("success") else "SYSTEM_TASK_FAILED",
            f"System command {command}: {'SUCCESS' if result.get('success') else 'FAILED'}",
            {"task_id": task_id, "result": result},
        )

    def _execute_orchestrator_full_reset(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute full orchestrator reset for thermal recovery.

        This is a drastic recovery action that:
        1. Signals all GPU agents to shut down gracefully
        2. Waits for clean shutdown (up to 60s)
        3. Kills any stale Ollama worker processes
        4. Clears thermal incident state
        5. Returns control to the launcher (which will restart agents)

        Note: The brain itself does NOT restart - it signals completion and
        the launcher script is responsible for restarting GPU agents.
        """
        incident_id = task.get("incident_id", "unknown")
        reason = task.get("reason", "thermal_recovery")
        resets_attempted = task.get("resets_attempted", 0)

        self.log_decision(
            "ORCHESTRATOR_FULL_RESET_START",
            f"Starting full orchestrator reset",
            {
                "incident_id": incident_id,
                "reason": reason,
                "resets_attempted": resets_attempted,
                "gpu_agents": list(self.gpu_agents.keys()),
            },
        )

        shutdown_results = {}
        all_stopped = True

        # Step 1: Signal all GPU agents to stop via .stop signal files
        # The GPU agent's _check_stop_signal() will detect these and set running=False
        for gpu_name in self.gpu_agents.keys():
            stop_signal = self.signals_path / f"{gpu_name}.stop"
            try:
                # Touch the file - GPU agent just checks for existence
                stop_signal.touch()
                shutdown_results[gpu_name] = {"signal_sent": True}
                self.logger.info(f"FULL_RESET: Sent stop signal to {gpu_name}")
            except Exception as e:
                shutdown_results[gpu_name] = {"signal_sent": False, "error": str(e)}
                self.logger.warning(f"FULL_RESET: Failed to signal {gpu_name}: {e}")

        # Step 2: Wait for GPU agents to stop (poll heartbeats)
        wait_start = time.time()
        max_wait_seconds = 60
        check_interval = 5

        while time.time() - wait_start < max_wait_seconds:
            time.sleep(check_interval)
            self._write_brain_heartbeat()  # Keep brain heartbeat alive

            still_running = []
            for gpu_name in self.gpu_agents.keys():
                hb_file = self.heartbeats_path / f"{gpu_name}.json"
                if hb_file.exists():
                    try:
                        with open(hb_file) as f:
                            hb = json.load(f)
                        # Check if heartbeat is recent (within 2x check interval)
                        # GPU heartbeats use 'last_updated'; fallback to 'timestamp' for compatibility
                        hb_time_str = hb.get("last_updated") or hb.get("timestamp")
                        if not hb_time_str:
                            self.logger.debug(f"FULL_RESET: {gpu_name} heartbeat missing freshness field")
                            continue
                        hb_time = datetime.fromisoformat(hb_time_str)
                        age = (datetime.now() - hb_time).total_seconds()
                        if age < check_interval * 2:
                            still_running.append(gpu_name)
                    except Exception:
                        pass

            if not still_running:
                self.logger.info("FULL_RESET: All GPU agents stopped cleanly")
                break
            else:
                elapsed = int(time.time() - wait_start)
                self.logger.info(f"FULL_RESET: Waiting for {still_running} ({elapsed}s elapsed)")
        else:
            all_stopped = False
            self.logger.warning(f"FULL_RESET: Some agents did not stop within {max_wait_seconds}s")

        # Step 3: Force kill any stale Ollama worker/split processes
        # Worker ports: 11435-11439, Split ports: 11440-11441
        killed_processes = []
        all_ports = list(range(11435, 11442))  # 11435-11441 inclusive
        for port in all_ports:
            try:
                # Use fuser to find and kill processes on worker/split ports
                result = subprocess.run(
                    ["fuser", "-k", f"{port}/tcp"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    killed_processes.append(port)
                    port_type = "split" if port >= 11440 else "worker"
                    self.logger.info(f"FULL_RESET: Killed {port_type} process on port {port}")
            except Exception as e:
                self.logger.debug(f"FULL_RESET: fuser port {port}: {e}")

        # Step 4: Clean up stop signals (so agents can restart cleanly)
        # Note: The GPU agent deletes the .stop file when it processes it,
        # but we clean up any remaining ones just in case
        for gpu_name in self.gpu_agents.keys():
            stop_signal = self.signals_path / f"{gpu_name}.stop"
            try:
                if stop_signal.exists():
                    stop_signal.unlink()
            except Exception:
                pass

        # Step 5: Clear brain-level thermal incident tracking
        self.thermal_recovery_active_incident_id = None
        self.thermal_recovery_incident_started_at = None
        self.thermal_recovery_resets_issued = 0
        self.thermal_recovery_last_reset_at = None
        self.thermal_recovery_last_reset_gpu = None
        # Don't clear full_reset_at - keep cooldown active

        # Step 6: Write a restart signal for the launcher
        restart_signal = self.brain_path / "restart_workers.signal"
        try:
            restart_data = {
                "timestamp": datetime.now().isoformat(),
                "reason": f"orchestrator_full_reset:{incident_id}",
                "initiated_by": "brain",
                "all_stopped": all_stopped,
            }
            with open(restart_signal, "w") as f:
                json.dump(restart_data, f, indent=2)
            self.logger.info("FULL_RESET: Wrote restart signal for launcher")
        except Exception as e:
            self.logger.warning(f"FULL_RESET: Failed to write restart signal: {e}")

        self.log_decision(
            "ORCHESTRATOR_FULL_RESET_COMPLETE",
            f"Full reset complete - workers should restart via launcher",
            {
                "incident_id": incident_id,
                "all_stopped": all_stopped,
                "shutdown_results": shutdown_results,
                "killed_ports": killed_processes,
            },
        )

        return {
            "success": True,
            "handler": "brain",
            "all_stopped": all_stopped,
            "shutdown_results": shutdown_results,
            "killed_ports": killed_processes,
            "restart_signal_written": True,
        }

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

                    # Brain handles: execute_plan, decide, system, and tasks marked executor=brain
                    if task_type == "execute_plan":
                        task_file.unlink()
                        self.handle_execute_plan_task(task)
                    elif task_type == "decide":
                        task_file.unlink()
                        self.handle_decide_task(task)
                    elif task_type == "system":
                        # Brain-level system commands (orchestrator_full_reset, etc.)
                        task_file.unlink()
                        self.handle_system_task(task)
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
