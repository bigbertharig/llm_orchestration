#!/usr/bin/env python3
"""
Worker - Executes a single task and exits.

Spawned by gpu.py as a subprocess. Takes a task JSON, runs it, prints the
result as JSON to stdout, and exits. No loop, no heartbeat, no state.

Usage (called by gpu.py, not directly):
  python worker.py --execute --config config.json --gpu-name gpu-1 \
    --permissions /path/to/worker.json --task '{"task_id": "...", ...}'
"""

import argparse
import json
import os
import sys
import time
import uuid
import logging
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from executor import PermissionExecutor

# Setup logging to stderr (stdout is reserved for result JSON)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [worker] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stderr
)
logger = logging.getLogger("worker")

# Set logging level from env
log_level = os.environ.get("WORKER_LOG_LEVEL", "INFO").upper()
try:
    logger.setLevel(getattr(logging, log_level))
except AttributeError:
    logger.setLevel(logging.INFO)


def execute_task(task: Dict[str, Any], permissions_file: str,
                 gpu_name: str, ollama_url: str = None,
                 model: str = None, task_timeout: int | None = None) -> Dict[str, Any]:
    """
    Execute a single task and return the result.

    Handles two task types:
      - shell: Run a command via PermissionExecutor
      - LLM (generate/parse/transform/execute): Send prompt to Ollama

    Returns a result dict with at minimum: success, worker, output/error.
    """
    task_type = task.get("type", "generate")
    prompt = task.get("prompt", "")
    context = task.get("context", "")

    # Initialize permission executor
    executor = PermissionExecutor(permissions_file, gpu_name)

    # Handle shell tasks
    if task_type == "shell":
        command = task.get("command", prompt)
        logger.info(f"Executing shell: {command[:80]}...")

        shell_timeout = task.get("timeout_seconds")
        if shell_timeout is None:
            shell_timeout = task_timeout
        result = executor.run_bash(command, timeout=shell_timeout)

        _log_training_sample(task, command, result.output,
                             "success" if result.success else "failure",
                             gpu_name, model)

        return {
            "success": result.success,
            "output": result.output,
            "action": result.action.value,
            "reason": result.reason,
            "worker": gpu_name,
        }

    # Handle LLM tasks
    if not ollama_url:
        return {
            "success": False,
            "error": "LLM task but no Ollama URL provided",
            "worker": gpu_name,
        }

    api_url = f"{ollama_url}/api/generate"
    system_prompt = _get_system_prompt(task_type)
    full_prompt = f"{system_prompt}\n\n"
    if context:
        full_prompt += f"Context:\n{context}\n\n"
    full_prompt += f"Task:\n{prompt}"

    logger.info(f"Executing LLM task type: {task_type}")

    try:
        start_time = time.time()

        request_timeout = task.get("timeout_seconds")
        if request_timeout is None:
            request_timeout = task_timeout

        response = requests.post(
            api_url,
            json={"model": model, "prompt": full_prompt, "stream": False},
            timeout=request_timeout
        )
        response.raise_for_status()

        result = response.json()
        elapsed = time.time() - start_time
        output = result.get("response", "")

        _log_training_sample(task, prompt, output, "success", gpu_name, model)

        return {
            "success": True,
            "output": output,
            "tokens": result.get("eval_count", 0),
            "duration_seconds": elapsed,
            "model": model,
            "worker": gpu_name,
        }

    except requests.Timeout:
        _log_training_sample(task, prompt, "", "timeout", gpu_name, model)
        return {
            "success": False,
            "error": "Task timed out",
            "worker": gpu_name,
        }
    except Exception as e:
        _log_training_sample(task, prompt, str(e), "failure", gpu_name, model)
        return {
            "success": False,
            "error": str(e),
            "worker": gpu_name,
        }


def _get_system_prompt(task_type: str) -> str:
    """Get system prompt based on task type."""
    prompts = {
        "parse": "You are a data parsing assistant. Extract structured information from the input and return it in the requested format. Be precise and follow the schema exactly.",
        "transform": "You are a data transformation assistant. Convert the input data to the requested format. Preserve all information accurately.",
        "generate": "You are a helpful assistant. Complete the task as requested. Be concise and accurate.",
        "execute": "You are a command execution assistant. Analyze the request and provide the exact commands or code needed. Only output what was requested.",
    }
    return prompts.get(task_type, prompts["generate"])


def _log_training_sample(task: Dict, prompt: str, response: str, outcome: str,
                         gpu_name: str, model: str = None):
    """Log a training sample for future fine-tuning."""
    # Resolve log path from environment or default
    log_dir = os.environ.get("LLM_ORCH_LOG_PATH")
    if not log_dir:
        # Fall back to finding shared/logs relative to this script
        log_dir = str(Path(__file__).parent.parent / "logs")

    training_log = Path(log_dir) / "training_samples.jsonl"
    training_log.parent.mkdir(parents=True, exist_ok=True)

    sample = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "sample_type": "worker_execution",
        "model": model,
        "worker": gpu_name,
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

    try:
        with open(training_log, 'a') as f:
            f.write(json.dumps(sample) + "\n")
    except Exception as e:
        logger.debug(f"Failed to write training sample: {e}")


def main():
    parser = argparse.ArgumentParser(description="Worker - execute a single task")
    parser.add_argument("--execute", action="store_true", required=True,
                        help="Execute mode (required)")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--gpu-name", required=True, help="Name of parent GPU agent")
    parser.add_argument("--permissions", required=True, help="Path to permissions file")
    parser.add_argument("--task", required=True, help="Task JSON string")
    parser.add_argument("--ollama-url", default=None,
                        help="Ollama API URL (e.g., http://localhost:11435)")
    args = parser.parse_args()

    # Parse task
    try:
        task = json.loads(args.task)
    except json.JSONDecodeError as e:
        result = {"success": False, "error": f"Invalid task JSON: {e}", "worker": args.gpu_name}
        print(json.dumps(result))
        sys.exit(1)

    # Load config for timeout and model info
    try:
        with open(args.config) as f:
            config = json.load(f)
    except Exception as e:
        result = {"success": False, "error": f"Config load error: {e}", "worker": args.gpu_name}
        print(json.dumps(result))
        sys.exit(1)

    configured_timeout = config.get("timeouts", {}).get("worker_task_seconds", 0)
    task_timeout = None if not configured_timeout or configured_timeout <= 0 else configured_timeout

    # Get model name from GPU config
    model = None
    for gpu in config.get("gpus", []):
        if gpu["name"] == args.gpu_name:
            model = gpu.get("model")
            break

    # Set log path env for training samples
    config_dir = Path(args.config).parent
    shared_path = Path(config["shared_path"])
    if not shared_path.is_absolute():
        shared_path = (config_dir / shared_path).resolve()
    os.environ["LLM_ORCH_LOG_PATH"] = str(shared_path / "logs")

    logger.info(f"Worker starting: task {task.get('task_id', '')[:8]} on {args.gpu_name}")
    # Export worker identity/runtime endpoint for script commands executed via shell.
    os.environ["WORKER_NAME"] = args.gpu_name
    if args.ollama_url:
        os.environ["WORKER_OLLAMA_URL"] = args.ollama_url

    # Execute
    result = execute_task(
        task=task,
        permissions_file=args.permissions,
        gpu_name=args.gpu_name,
        ollama_url=args.ollama_url,
        model=model,
        task_timeout=task_timeout,
    )

    logger.info(f"Worker done: {'OK' if result.get('success') else 'FAIL'}")

    # Print result as JSON to stdout (gpu.py reads this)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
