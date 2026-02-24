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

