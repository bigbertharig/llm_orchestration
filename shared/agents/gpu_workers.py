"""GPU agent worker subprocess management mixin.

Extracted from gpu.py to isolate worker spawning, collection, heartbeating,
and signal handling.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class WorkerResult:
    """Result from a completed worker subprocess."""
    def __init__(self, task_id: str, task: Dict, result: Dict, peak_vram_mb: int = 0):
        self.task_id = task_id
        self.task = task
        self.result = result
        self.peak_vram_mb = peak_vram_mb


class GPUWorkerMixin:
    """Mixin providing worker subprocess management methods."""

    def _spawn_worker(self, task: Dict):
        """Spawn a worker subprocess to execute a task."""
        if str(task.get("task_class", "")).strip() == "llm":
            llm_ok, llm_reason = self._llm_task_runtime_compatible(task)
            if not llm_ok:
                self.logger.error(
                    f"Refusing to spawn llm worker for task {task.get('task_id', '')[:8]}: "
                    f"incompatible runtime ({llm_reason})"
                )
                self.outbox.append(WorkerResult(
                    task_id=task["task_id"],
                    task=task,
                    result={
                        "success": False,
                        "error": f"runtime_incompatible: {llm_reason}",
                        "worker": self.name,
                        "max_vram_used_mb": 0,
                    },
                    peak_vram_mb=0,
                ))
                self._remove_task_heartbeat(task["task_id"])
                return

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

        # Pass active runtime endpoint if available (for LLM tasks).
        if self.runtime_api_base:
            worker_cmd.extend(["--api-base", self.runtime_api_base])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        # Pass runtime backend so worker uses correct inference API
        env["WORKER_RUNTIME_BACKEND"] = str(getattr(self, 'runtime_backend', 'llama'))
        # Ensure script-style worker tasks can self-identify and self-route.
        env["WORKER_NAME"] = self.name
        if self.loaded_model:
            env["WORKER_MODEL"] = str(self.loaded_model)
        elif self.model:
            env["WORKER_MODEL"] = str(self.model)
        if self.runtime_api_base:
            env["WORKER_API_BASE"] = self.runtime_api_base
        if self.runtime_placement:
            env["WORKER_RUNTIME_PLACEMENT"] = str(self.runtime_placement)
        if self.runtime_group_id:
            env["WORKER_RUNTIME_GROUP_ID"] = str(self.runtime_group_id)
        if self.loaded_model:
            env["WORKER_RUNTIME_MODEL"] = str(self.loaded_model)

        # Pass task JSON file path for worker scripts to read repair metadata
        # Task is in processing/ after being claimed
        task_file_path = self.processing_path / f"{task['task_id']}.json"
        if task_file_path.exists():
            env["WORKER_TASK_JSON_PATH"] = str(task_file_path)

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

    def _write_task_heartbeat(
        self,
        task_id: str,
        worker_id: str,
        pid: int,
        peak_vram_mb: int = 0,
        is_meta: bool = False,
    ):
        """Write per-task progress heartbeat for stuck-task detection."""
        hb = {
            "task_id": task_id,
            "worker_id": worker_id,
            "is_meta": bool(is_meta),
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

    def _has_active_work(self) -> bool:
        """Return True when the rig appears to be in an active batch."""
        if self.active_workers or self.outbox or self.active_meta_task:
            return True

        if next(self.queue_path.glob("*.json"), None):
            return True
        # Exclude heartbeat sidecars - only count real task files
        for f in self.processing_path.glob("*.json"):
            if not f.name.endswith(".heartbeat.json"):
                return True

        return False

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
