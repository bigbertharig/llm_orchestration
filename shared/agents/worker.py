#!/usr/bin/env python3
"""
Worker Agent - Executes tasks assigned by the brain.
Each worker runs on a single GPU with its own Ollama instance.

Usage:
  python worker.py worker-1
  python worker.py --config /path/to/config.json worker-2
"""

import argparse
import json
import os
import sys
import time
import uuid
import logging
import requests
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

from executor import PermissionExecutor, ActionResult

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)

# Valid task classes - workers use task_class field, not pattern matching
# resource: Special tasks for model loading/unloading (inserted by brain)
VALID_TASK_CLASSES = ['cpu', 'script', 'llm', 'resource']

class Worker:
    def __init__(self, config_path: str, worker_name: str, start_hot: bool = False,
                 model_preloaded: bool = False):
        self.config_path = Path(config_path)
        self.config = self._load_config(config_path)
        self.worker_config = self._get_worker_config(worker_name)
        self.name = worker_name
        self.start_hot = start_hot
        self.model_preloaded = model_preloaded
        self.logger = logging.getLogger(self.name)

        # Resolve paths relative to config file location
        config_dir = self.config_path.parent
        shared_path = Path(self.config["shared_path"])
        if not shared_path.is_absolute():
            shared_path = (config_dir / shared_path).resolve()
        self.shared_path = shared_path

        self.queue_path = self.shared_path / "tasks" / "queue"
        self.processing_path = self.shared_path / "tasks" / "processing"
        self.complete_path = self.shared_path / "tasks" / "complete"
        self.failed_path = self.shared_path / "tasks" / "failed"
        self.log_path = self.shared_path / "logs"

        self.model = self.worker_config["model"]
        self.gpu = self.worker_config["gpu"]
        self.port = self.worker_config["port"]
        self.api_url = f"http://localhost:{self.port}/api/generate"

        self.poll_interval = self.config["timeouts"]["poll_interval_seconds"]
        self.task_timeout = self.config["timeouts"]["worker_task_seconds"]

        # Initialize permission executor - resolve relative path
        permissions_path = Path(self.config["permissions_path"])
        if not permissions_path.is_absolute():
            permissions_path = (config_dir / permissions_path).resolve()
        permissions_file = permissions_path / self.worker_config.get("permissions", "worker.json")
        self.executor = PermissionExecutor(
            str(permissions_file), self.name,
            heartbeat_callback=self._write_heartbeat
        )

        # Training data log
        self.training_log = self.shared_path / "logs" / "training_samples.jsonl"
        self.training_log.parent.mkdir(parents=True, exist_ok=True)

        self.ollama_process: Optional[subprocess.Popen] = None
        self.running = True
        self.model_loaded = False  # Track if LLM model is loaded in VRAM

        # Track stats for lifecycle logging
        self.stats = {"tasks_completed": 0}
        self.last_task_time = time.time()

        # Set environment for this process so all child processes inherit it
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        os.environ["WORKER_OLLAMA_URL"] = f"http://localhost:{self.port}"

        # Primary worker keeps model loaded as "hot standby" for immediate LLM response
        # Other workers stay lean for script tasks
        self.is_primary = self.worker_config.get("primary", False)
        if self.is_primary:
            self.logger.info("This is the PRIMARY worker - will keep model loaded when idle")

        # P0: Worker lifecycle logging
        self.logger.info(f"Worker initialized: {self.name} on GPU {self.gpu}, Ollama port {self.port}")

    def log_training_sample(self, task: Dict, prompt: str, response: str, outcome: str):
        """Log a training sample for future fine-tuning."""
        sample = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "sample_type": "worker_execution",
            "model": self.model,
            "worker": self.name,
            "task_type": task.get("type", "unknown"),
            "task_id": task.get("task_id", ""),
            "prompt": prompt,
            "context": task.get("context", ""),
            "response": response,
            "outcome": outcome,
            "human_rating": None,
            "human_feedback": None,
            "preferred_response": None,
        }

        with open(self.training_log, 'a') as f:
            f.write(json.dumps(sample) + "\n")

    def _load_config(self, config_path: str) -> dict:
        with open(config_path) as f:
            return json.load(f)

    def _get_worker_config(self, worker_name: str) -> dict:
        for worker in self.config["workers"]:
            if worker["name"] == worker_name:
                return worker
        raise ValueError(f"Worker '{worker_name}' not found in config")

    def start_ollama(self):
        """Start Ollama instance on dedicated GPU and port."""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        env["OLLAMA_HOST"] = f"0.0.0.0:{self.port}"

        self.logger.info(f"Starting Ollama on GPU {self.gpu}, port {self.port}")

        self.ollama_process = subprocess.Popen(
            ["ollama", "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # Wait for server to be ready
        for i in range(30):
            try:
                requests.get(f"http://localhost:{self.port}/api/tags", timeout=1)
                self.logger.info("Ollama server ready")
                return
            except:
                time.sleep(1)

        raise RuntimeError("Ollama failed to start")

    def stop_ollama(self):
        """Stop the Ollama instance."""
        if self.ollama_process:
            self.ollama_process.terminate()
            self.ollama_process.wait(timeout=10)
            self.logger.info("Ollama stopped")

    def load_model(self):
        """Load the LLM model into VRAM (lazy loading)."""
        if self.model_loaded:
            return

        self.logger.info(f"Loading model {self.model} into VRAM...")
        start_time = time.time()

        try:
            # Preload with a simple prompt
            response = requests.post(
                self.api_url,
                json={"model": self.model, "prompt": "Hello", "stream": False},
                timeout=600  # Long timeout for slow PCIe
            )
            response.raise_for_status()

            elapsed = int(time.time() - start_time)
            self.model_loaded = True
            self.logger.info(f"Model loaded in {elapsed}s - ready for LLM tasks")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")

    def unload_model(self):
        """Unload the LLM model from VRAM (frees memory for script tasks)."""
        if not self.model_loaded:
            return

        self.logger.info(f"Unloading model {self.model} to free VRAM...")

        try:
            # Send keep_alive: 0 to unload immediately
            response = requests.post(
                self.api_url,
                json={"model": self.model, "prompt": "", "keep_alive": 0},
                timeout=30
            )

            if response.status_code == 200:
                self.model_loaded = False
                self.logger.info("Model unloaded - VRAM freed for script tasks")
            else:
                self.logger.warning(f"Unload returned status {response.status_code}")

        except Exception as e:
            self.logger.warning(f"Failed to unload model: {e}")

    def analyze_queue(self) -> Dict[str, int]:
        """
        Analyze the queue to see what types of tasks are available.
        Returns counts by task_class.
        """
        stats = {"cpu": 0, "script": 0, "llm": 0, "total": 0}

        try:
            for task_file in self.queue_path.glob("*.json"):
                if str(task_file).endswith('.lock'):
                    continue

                try:
                    with open(task_file) as f:
                        task = json.load(f)

                    executor = task.get("executor", "worker")

                    # Skip brain-only tasks
                    if executor == "brain":
                        continue

                    stats["total"] += 1

                    # Use task_class field
                    task_class = task.get("task_class", "cpu")
                    if task_class in stats:
                        stats[task_class] += 1
                    else:
                        stats["cpu"] += 1  # Unknown class defaults to cpu

                except Exception:
                    pass

        except Exception as e:
            self.logger.debug(f"Queue analysis error: {e}")

        return stats

    def claim_task(self, preferred_classes: List[str] = None) -> Optional[Dict[str, Any]]:
        """
        Claim an available task that this worker can handle.

        preferred_classes: List of task classes in priority order.
        Default for GPU workers: ['llm', 'script', 'cpu'] (if model loaded) or ['script', 'cpu', 'llm']
        """
        from filelock import FileLock

        if preferred_classes is None:
            preferred_classes = ['cpu', 'script', 'llm']

        # First pass: categorize available tasks by class
        tasks_by_class = {tc: [] for tc in VALID_TASK_CLASSES}

        for task_file in sorted(self.queue_path.glob("*.json")):
            if str(task_file).endswith('.lock'):
                continue

            try:
                with open(task_file) as f:
                    task = json.load(f)

                executor = task.get("executor", "worker")

                # Skip brain-only tasks
                if executor == "brain":
                    continue

                task_class = task.get("task_class", "cpu")
                if task_class in tasks_by_class:
                    tasks_by_class[task_class].append(task_file)

            except Exception:
                continue

        # Second pass: try to claim from preferred classes in order
        candidates = []
        for tc in preferred_classes:
            candidates.extend(tasks_by_class.get(tc, []))

        for task_file in candidates:
            lock_file = str(task_file) + ".lock"
            lock = FileLock(lock_file, timeout=1)

            # P1: Log claim attempts (shortened task_id for readability)
            task_id = task_file.stem[:8]

            try:
                with lock:
                    self.logger.debug(f"Attempting claim: {task_id}")

                    if not task_file.exists():
                        self.logger.debug(f"Lost race: {task_id} (claimed by another worker)")
                        continue  # Another worker claimed it

                    with open(task_file) as f:
                        task = json.load(f)

                    task_class = task.get("task_class", "cpu")

                    # Claim it
                    task["status"] = "processing"
                    task["assigned_to"] = self.name
                    task["started_at"] = datetime.now().isoformat()

                    new_path = self.processing_path / task_file.name
                    with open(new_path, 'w') as f:
                        json.dump(task, f, indent=2)
                    task_file.unlink()

                    self.logger.info(f"Claimed {task_class} task: {task['task_id'][:8]}... ({task.get('name', '')})")
                    # Send heartbeat immediately on task claim
                    self._write_heartbeat()
                    return task

            except Exception as e:
                # P1: Log lock timeouts
                from filelock import Timeout
                if isinstance(e, Timeout):
                    self.logger.debug(f"Lock timeout: {task_id} (another worker has it)")
                continue

        # P0: Claim loop visibility - log what we saw if we didn't claim anything
        if len(candidates) > 0:
            # Saw tasks but couldn't claim any (lost races)
            self.logger.debug(f"Claim attempt: {len(candidates)} tasks available in {preferred_classes}, all claimed by others")
        else:
            # No tasks matching our preferences
            task_summary = ", ".join([f"{k}:{len(v)}" for k, v in tasks_by_class.items() if len(v) > 0])
            if task_summary:
                self.logger.debug(f"No suitable tasks - Queue: [{task_summary}], Preferred: {preferred_classes}, Model loaded: {self.model_loaded}")
            else:
                self.logger.debug(f"Queue empty")

        return None

    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task using the LLM or shell."""
        # Check for abort/kill signals before starting execution
        if self._check_kill_signal() or self._check_abort_signal():
            return {
                "success": False,
                "error": "Task aborted by brain before execution",
                "worker": self.name
            }

        task_type = task.get("type", "generate")
        prompt = task.get("prompt", "")
        context = task.get("context", "")

        # Handle resource management tasks from brain
        if task_type == "resource":
            command = task.get("command", "")
            if command == "load_llm":
                self.logger.info("Brain requested: load LLM model")
                self.load_model()
                return {
                    "success": True,
                    "output": f"Model {self.model} loaded",
                    "model_loaded": self.model_loaded,
                    "worker": self.name
                }
            elif command == "unload_llm":
                self.logger.info("Brain requested: unload LLM model")
                self.unload_model()
                return {
                    "success": True,
                    "output": f"Model {self.model} unloaded",
                    "model_loaded": self.model_loaded,
                    "worker": self.name
                }
            else:
                return {
                    "success": False,
                    "error": f"Unknown resource command: {command}",
                    "worker": self.name
                }

        # Handle shell tasks - execute commands directly
        if task_type == "shell":
            command = task.get("command", prompt)
            self.logger.info(f"Executing shell command: {command[:80]}...")
            result = self.executor.run_bash(command)

            # Log outcome
            outcome = "success" if result.success else "failure"
            self.log_training_sample(task, command, result.output, outcome)

            return {
                "success": result.success,
                "output": result.output,
                "action": result.action.value,
                "reason": result.reason,
                "worker": self.name
            }

        # Build the full prompt
        system_prompt = self._get_system_prompt(task_type)
        full_prompt = f"{system_prompt}\n\n"
        if context:
            full_prompt += f"Context:\n{context}\n\n"
        full_prompt += f"Task:\n{prompt}"

        self.logger.info(f"Executing task type: {task_type}")

        try:
            start_time = time.time()

            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False
                },
                timeout=self.task_timeout
            )
            response.raise_for_status()

            result = response.json()
            elapsed = time.time() - start_time
            output = result.get("response", "")

            # Log training sample
            self.log_training_sample(task, prompt, output, "success")

            return {
                "success": True,
                "output": output,
                "tokens": result.get("eval_count", 0),
                "duration_seconds": elapsed,
                "model": self.model,
                "worker": self.name
            }

        except requests.Timeout:
            self.log_training_sample(task, prompt, "", "timeout")
            return {"success": False, "error": "Task timed out", "worker": self.name}
        except Exception as e:
            self.log_training_sample(task, prompt, str(e), "failure")
            return {"success": False, "error": str(e), "worker": self.name}

    def _get_system_prompt(self, task_type: str) -> str:
        """Get system prompt based on task type."""
        prompts = {
            "parse": "You are a data parsing assistant. Extract structured information from the input and return it in the requested format. Be precise and follow the schema exactly.",
            "transform": "You are a data transformation assistant. Convert the input data to the requested format. Preserve all information accurately.",
            "generate": "You are a helpful assistant. Complete the task as requested. Be concise and accurate.",
            "execute": "You are a command execution assistant. Analyze the request and provide the exact commands or code needed. Only output what was requested.",
        }
        return prompts.get(task_type, prompts["generate"])

    def complete_task(self, task: Dict[str, Any], result: Dict[str, Any]):
        """Mark task as complete and save result."""
        task["status"] = "complete" if result["success"] else "failed"
        task["completed_at"] = datetime.now().isoformat()
        task["result"] = result

        task_file = self.processing_path / f"{task['task_id']}.json"
        dest_path = (self.complete_path if result["success"] else self.failed_path)
        dest_file = dest_path / f"{task['task_id']}.json"

        with open(dest_file, 'w') as f:
            json.dump(task, f, indent=2)

        if task_file.exists():
            task_file.unlink()

        status = "completed" if result["success"] else "failed"
        self.logger.info(f"Task {task['task_id'][:8]}... {status}")

        # Update stats for lifecycle logging
        if result["success"]:
            self.stats["tasks_completed"] += 1
        self.last_task_time = time.time()

    def run(self):
        """
        Main worker loop.

        Workers claim and execute tasks. The brain controls model loading/unloading
        by inserting special 'load_llm' and 'unload_llm' tasks.

        Task priority (GPU workers):
        - If model loaded: prefer llm > script > cpu
        - If model not loaded: prefer script > cpu > llm
        """
        # P0: Worker lifecycle logging
        self.logger.info(f"Worker {self.name} starting main loop")

        try:
            # Start Ollama server
            self.start_ollama()

            # If starting hot, preload the LLM model
            if self.start_hot:
                if self.model_preloaded:
                    self.logger.info("Starting HOT - model already preloaded, skipping warmup")
                    self.model_loaded = True
                else:
                    self.logger.info("Starting HOT - preloading LLM model...")
                    self.load_model()
            else:
                self.logger.info("Starting COLD - no LLM loaded, ready for script tasks")

            # Signal ready
            flag_dir = Path("/tmp/llm-orchestration-flags")
            flag_dir.mkdir(parents=True, exist_ok=True)
            (flag_dir / f"{self.name}.ready").touch()
            self.logger.info("Worker ready - polling for tasks")

            while self.running:
                # Check for signals from brain
                self._check_stop_signal()  # Shutdown entire worker

                # Check for abort/kill signals (for stuck tasks)
                if self._check_kill_signal() or self._check_abort_signal():
                    # Task was aborted/killed, continue to next iteration
                    continue

                # Write heartbeat with current state
                self._write_heartbeat()

                # Determine task class preference based on model state
                # Resource tasks always first (brain-controlled model management)
                # Hot workers skip 'script' tasks - whisper needs GPU VRAM that LLM is using
                if self.model_loaded:
                    preferred = ['resource', 'llm', 'cpu']
                else:
                    preferred = ['resource', 'script', 'cpu', 'llm']

                task = self.claim_task(preferred_classes=preferred)

                if task:
                    # Store current task for abort/kill handling
                    self.current_task = task
                    result = self.execute_task(task)
                    self.complete_task(task, result)
                    self.current_task = None
                else:
                    time.sleep(self.poll_interval)

        except KeyboardInterrupt:
            # P0: Worker lifecycle logging
            self.logger.info(f"Worker {self.name} received shutdown signal")
        except Exception as e:
            # P0: Worker lifecycle logging
            self.logger.error(f"Worker {self.name} crashed in main loop: {e}", exc_info=True)
        finally:
            self.cleanup()

    def _write_heartbeat(self):
        """Write heartbeat file so brain knows worker state."""
        workers_path = self.shared_path / "workers" / self.name
        workers_path.mkdir(parents=True, exist_ok=True)

        heartbeat = {
            "worker_id": self.name,
            "gpu_id": self.gpu,
            "timestamp": datetime.now().isoformat(),
            "model_loaded": self.model_loaded,
            "model": self.model,
            "port": self.port
        }

        heartbeat_file = workers_path / "heartbeat.json"
        with open(heartbeat_file, 'w') as f:
            json.dump(heartbeat, f, indent=2)

        # P1: Heartbeat logging (at debug level to avoid noise)
        idle_time = time.time() - self.last_task_time
        status = "processing" if hasattr(self, 'current_task') and self.current_task else "idle"
        self.logger.debug(f"Heartbeat sent - status: {status}, idle: {idle_time:.0f}s")

    def _check_stop_signal(self):
        """Check if brain has signaled this worker to stop."""
        signal_file = self.shared_path / "signals" / f"{self.name}.stop"
        if signal_file.exists():
            self.logger.info(f"Received stop signal from brain")
            signal_file.unlink()  # Remove signal file
            self.running = False

    def _check_abort_signal(self) -> bool:
        """Check if brain has signaled to abort current task (graceful).

        Returns True if abort signal found and handled.
        """
        signal_file = self.shared_path / "signals" / f"{self.name}.abort"
        if signal_file.exists():
            try:
                with open(signal_file) as f:
                    abort_data = json.load(f)
                self.logger.warning(
                    f"Received abort signal from brain: {abort_data.get('reason')} "
                    f"(task: {abort_data.get('task_id', 'unknown')[:8]})"
                )
                signal_file.unlink()  # Remove signal file

                # Kill active subprocess gracefully
                if hasattr(self, 'executor') and self.executor:
                    self.executor.kill_active_process()

                return True

            except Exception as e:
                self.logger.error(f"Error processing abort signal: {e}")
                try:
                    signal_file.unlink()
                except:
                    pass

        return False

    def _check_kill_signal(self) -> bool:
        """Check if brain has sent force kill signal.

        Returns True if kill signal found and handled.
        """
        signal_file = self.shared_path / "signals" / f"{self.name}.kill"
        if signal_file.exists():
            try:
                with open(signal_file) as f:
                    kill_data = json.load(f)
                self.logger.error(
                    f"Received FORCE KILL signal from brain: {kill_data.get('reason')} "
                    f"(task: {kill_data.get('task_id', 'unknown')[:8]})"
                )
                signal_file.unlink()  # Remove signal file

                # Force kill subprocess immediately
                if hasattr(self, 'executor') and self.executor:
                    self.executor.kill_active_process()

                return True

            except Exception as e:
                self.logger.error(f"Error processing kill signal: {e}")
                try:
                    signal_file.unlink()
                except:
                    pass

        return False

    def stop(self):
        """Signal the worker to stop."""
        self.running = False

    def cleanup(self):
        """Clean up resources on shutdown."""
        # P0: Worker lifecycle logging
        self.logger.info(f"Worker {self.name} shutting down - completed {self.stats.get('tasks_completed', 0)} tasks")

        # Kill any active subprocess
        if hasattr(self, 'executor') and self.executor:
            self.executor.kill_active_process()
        # Stop ollama
        self.stop_ollama()

        self.logger.info("Cleanup complete")


def main():
    import signal

    worker = None

    def signal_handler(signum, frame):
        if worker:
            worker.stop()
            worker.cleanup()
        sys.exit(0)

    # Register signal handlers for clean shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description="Worker Agent")
    parser.add_argument("worker_name", help="Name of the worker (e.g., worker-1)")
    default_config = Path(__file__).parent / "config.json"
    parser.add_argument("--config", default=str(default_config),
                        help="Path to config file")
    parser.add_argument("--hot", action="store_true",
                        help="Start HOT with LLM preloaded (default: start COLD)")
    parser.add_argument("--model-preloaded", action="store_true",
                        help="Skip model warmup (model already loaded in Ollama)")
    args = parser.parse_args()

    worker = Worker(args.config, args.worker_name, start_hot=args.hot,
                    model_preloaded=args.model_preloaded)
    try:
        worker.run()
    except KeyboardInterrupt:
        pass
    finally:
        worker.cleanup()


if __name__ == "__main__":
    main()
