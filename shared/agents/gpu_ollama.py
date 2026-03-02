"""GPU agent Ollama management mixin.

Extracted from gpu.py to isolate Ollama server lifecycle, model loading/unloading,
and global model load lock coordination.
"""

import json
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from filelock import FileLock, Timeout

from gpu_constants import (
    GLOBAL_MODEL_LOAD_OWNER_HEARTBEAT_INTERVAL,
    GLOBAL_MODEL_LOAD_OWNER_STALE_SECONDS,
    RUNTIME_STATE_COLD,
    RUNTIME_STATE_LOADING_SINGLE,
    RUNTIME_STATE_READY_SINGLE,
    RUNTIME_STATE_UNLOADING,
)


class GPUOllamaMixin:
    """Mixin providing Ollama server and model management methods."""

    def start_ollama(self):
        """Start Ollama instance on this GPU's dedicated port."""
        if not self.port:
            self.logger.info("No port configured - script-only GPU, skipping Ollama")
            return

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        env["OLLAMA_HOST"] = f"0.0.0.0:{self.port}"

        self.logger.info(f"Starting Ollama on GPU {self.gpu_id}, port {self.port}")

        self.ollama_process = subprocess.Popen(
            ["ollama", "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        for i in range(30):
            try:
                requests.get(f"http://localhost:{self.port}/api/tags", timeout=1)
                self.logger.info("Ollama server ready")
                return
            except Exception:
                time.sleep(1)

        raise RuntimeError("Ollama failed to start")

    def stop_ollama(self):
        """Stop the Ollama instance."""
        if self.ollama_process:
            self.ollama_process.terminate()
            try:
                self.ollama_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.ollama_process.kill()
            self.ollama_process = None
            self.logger.info("Ollama stopped")

    def check_ollama_health(self) -> Dict[str, Any]:
        """Check Ollama instance health via /api/tags endpoint.

        Returns dict with health status and loaded models. Tracks consecutive
        failures and triggers restart after threshold. Implements circuit breaker
        to stop claiming LLM tasks after sustained failures.
        """
        health = {
            "healthy": False,
            "loaded_models": [],
            "consecutive_failures": self.ollama_consecutive_failures,
            "response_ms": None,
        }

        if not self.port:
            health["healthy"] = True  # Script-only GPU, no Ollama expected
            health["note"] = "no_ollama_configured"
            return health

        try:
            start = time.time()
            resp = requests.get(
                f"http://localhost:{self.port}/api/tags",
                timeout=5
            )
            elapsed_ms = int((time.time() - start) * 1000)
            health["response_ms"] = elapsed_ms

            if resp.status_code == 200:
                data = resp.json()
                models = data.get("models", [])
                health["loaded_models"] = [m.get("name", "") for m in models]
                health["healthy"] = True
                self.ollama_consecutive_failures = 0
                self.ollama_healthy = True
            else:
                self.ollama_consecutive_failures += 1
                self.logger.warning(
                    f"Ollama health check returned {resp.status_code} "
                    f"(failure {self.ollama_consecutive_failures})")

        except Exception as e:
            self.ollama_consecutive_failures += 1
            self.logger.warning(
                f"Ollama health check failed: {e} "
                f"(failure {self.ollama_consecutive_failures})")

        health["consecutive_failures"] = self.ollama_consecutive_failures

        # Auto-restart after threshold
        if self.ollama_consecutive_failures >= self.ollama_health_threshold:
            self.logger.error(
                f"Ollama failed {self.ollama_consecutive_failures} consecutive health checks, "
                f"attempting restart")
            try:
                self.stop_ollama()
                time.sleep(2)
                self.start_ollama()
                self.ollama_consecutive_failures = 0
                self.ollama_healthy = True
                # Model was lost in restart - reset to cold state
                self.model_loaded = False
                self.loaded_model = None
                self.loaded_tier = 0
                self.runtime_placement = "single_gpu"
                self.runtime_group_id = None
                self.runtime_port = self.port
                self.runtime_ollama_url = f"http://localhost:{self.port}" if self.port else None
                self._set_runtime_state(
                    RUNTIME_STATE_COLD,
                    phase="ollama_restart_recovery",
                )
                self.logger.info("Ollama restarted successfully after health check failures")
            except Exception as e:
                self.logger.error(f"Ollama restart failed: {e}")

        # Circuit breaker
        if self.ollama_consecutive_failures >= self.ollama_circuit_breaker:
            self.ollama_healthy = False
            self.logger.error(
                f"Circuit breaker: Ollama unhealthy after {self.ollama_consecutive_failures} "
                f"failures, will not claim LLM tasks")

        return health

    def load_model(self, model_id: Optional[str] = None, task_id: Optional[str] = None):
        """Load an LLM model into VRAM on this worker's dedicated Ollama runtime."""
        # Preflight: check runtime state allows load
        can_load, preflight_reason = self._can_accept_load_task()
        if not can_load:
            self.logger.warning(
                f"LOAD_PREFLIGHT_REJECT worker={self.name} reason={preflight_reason}"
            )
            return

        if self.model_loaded:
            return

        if not self.api_url:
            self.logger.warning("Cannot load model - no Ollama port configured")
            return

        target_model = str(model_id or self.model or "").strip()
        if not target_model:
            self.logger.warning("Cannot load model - target model missing")
            return

        # Transition to loading state
        self._set_runtime_state(
            RUNTIME_STATE_LOADING_SINGLE,
            task_id=task_id,
            phase="preflight_passed",
        )

        self.logger.info(f"Loading model {target_model} into VRAM...")
        start_time = time.time()

        def _do_load():
            # Initial pull/load request (can block while weights are loaded)
            req_result: Dict[str, Any] = {"done": False, "response": None, "error": None}

            def _load_request():
                try:
                    req_result["response"] = requests.post(
                        self.api_url,
                        json={
                            "model": target_model,
                            "prompt": "Hello",
                            "stream": False,
                            "keep_alive": self._effective_keep_alive(),
                            "options": {
                                "num_gpu": 999,
                                "num_ctx": self.worker_num_ctx,
                            }
                        },
                        timeout=600
                    )
                except Exception as exc:
                    req_result["error"] = exc
                finally:
                    req_result["done"] = True

            req_thread = threading.Thread(target=_load_request, daemon=True)
            req_thread.start()
            req_started = time.time()
            last_wait_log_at = req_started
            while not req_result["done"]:
                self._touch_meta_task(phase="load_llm")
                now_wait = time.time()
                if (now_wait - last_wait_log_at) >= 20:
                    self.logger.info(
                        f"SINGLE_RUNTIME_LOAD_WAIT worker={self.name} model={target_model} "
                        f"elapsed_s={int(now_wait - req_started)}"
                    )
                    last_wait_log_at = now_wait
                time.sleep(1)

            if req_result["error"] is not None:
                raise req_result["error"]
            response = req_result["response"]
            if response is None:
                raise RuntimeError("single model load request finished without response")
            response.raise_for_status()

            # Readiness gate: only mark hot once generation is reliably responsive.
            if not self._wait_for_model_ready(model_id=target_model, max_wait_seconds=90):
                raise RuntimeError("model readiness probe timed out after load")

        try:
            self._set_runtime_state(
                RUNTIME_STATE_LOADING_SINGLE,
                task_id=task_id,
                phase="acquiring_global_lock",
            )
            self._run_with_global_model_load_lock(
                phase="single_model_load",
                fn=_do_load,
                max_wait_seconds=900,
            )
            elapsed = int(time.time() - start_time)
            self.model_loaded = True
            self.loaded_model = target_model
            self.loaded_tier = int(self.model_tier_by_id.get(target_model, self.model_tier))
            self.runtime_placement = str(
                self.model_meta_by_id.get(target_model, {}).get("placement", "single_gpu")
            )
            if self.runtime_placement != "split_gpu":
                self.runtime_group_id = None
            self.runtime_port = self.port
            self.runtime_ollama_url = f"http://localhost:{self.port}" if self.port else None
            # Transition to ready state
            self._set_runtime_state(
                RUNTIME_STATE_READY_SINGLE,
                task_id=task_id,
                phase="load_complete",
            )
            self.logger.info(f"Model loaded in {elapsed}s - GPU is now HOT")

        except Exception as e:
            self.model_loaded = False
            self.loaded_model = None
            self.loaded_tier = 0
            # Transition back to cold on failure (recoverable)
            self._set_runtime_state(
                RUNTIME_STATE_COLD,
                task_id=task_id,
                phase="load_failed",
                error_code="load_exception",
                error_detail=str(e)[:200],
            )
            self.logger.error(f"Failed to load model: {e}")

    def _run_with_global_model_load_lock(self, phase: str, fn, max_wait_seconds: int = 900):
        lock = FileLock(str(self.model_load_lock_path), timeout=1)
        deadline = time.time() + max_wait_seconds
        lease_acquired = False
        next_wait_log = 0.0
        while time.time() < deadline:
            try:
                lock.acquire(timeout=1)
                if self._try_acquire_global_model_load_owner(phase=phase):
                    lease_acquired = True
                    self.logger.info(
                        f"GLOBAL_LOAD_LOCK_ACQUIRED phase={phase} worker={self.name} pid={os.getpid()}"
                    )
                    break
                self._touch_meta_task(phase=f"{phase}_waiting_global_lock")
            except Timeout:
                self._touch_meta_task(phase=f"{phase}_waiting_global_lock")
            finally:
                try:
                    lock.release()
                except Exception:
                    pass
            if time.time() >= next_wait_log:
                self.logger.info(f"GLOBAL_LOAD_LOCK_WAIT phase={phase} worker={self.name}")
                next_wait_log = time.time() + 10
            time.sleep(1)
        else:
            raise RuntimeError(f"global model load lock timeout after {max_wait_seconds}s")

        stop_owner_heartbeat = threading.Event()
        owner_heartbeat_thread = threading.Thread(
            target=self._global_model_load_owner_heartbeat_loop,
            args=(stop_owner_heartbeat, phase),
            daemon=True,
        )
        owner_heartbeat_thread.start()

        try:
            self._touch_meta_task(phase=f"{phase}_global_lock_acquired", force=True)
            return fn()
        finally:
            stop_owner_heartbeat.set()
            try:
                owner_heartbeat_thread.join(timeout=2)
            except Exception:
                pass
            try:
                if lease_acquired:
                    self.logger.info(
                        f"GLOBAL_LOAD_LOCK_RELEASE phase={phase} worker={self.name} pid={os.getpid()}"
                    )
                    self._release_global_model_load_owner()
            except Exception:
                pass

    def _global_model_load_owner_payload(self, phase: str, acquired_at: Optional[str] = None) -> Dict[str, Any]:
        now_iso = datetime.now().isoformat()
        return {
            "worker": self.name,
            "pid": os.getpid(),
            "phase": phase,
            "acquired_at": acquired_at or now_iso,
            "heartbeat_at": now_iso,
        }

    def _write_global_model_load_owner(self, payload: Dict[str, Any]):
        tmp_path = self.model_load_owner_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, self.model_load_owner_path)

    def _read_global_model_load_owner(self) -> Optional[Dict[str, Any]]:
        try:
            if not self.model_load_owner_path.exists():
                return None
            with open(self.model_load_owner_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return raw if isinstance(raw, dict) else None
        except Exception:
            return None

    def _global_owner_is_stale(self, owner: Optional[Dict[str, Any]]) -> bool:
        if not owner:
            return True
        pid = owner.get("pid")
        try:
            pid_int = int(pid)
        except Exception:
            pid_int = None
        if pid_int and pid_int != os.getpid():
            try:
                os.kill(pid_int, 0)
                pid_alive = True
            except Exception:
                pid_alive = False
            if not pid_alive:
                return True
        heartbeat_raw = owner.get("heartbeat_at") or owner.get("acquired_at")
        try:
            heartbeat_dt = datetime.fromisoformat(str(heartbeat_raw))
            age = (datetime.now() - heartbeat_dt).total_seconds()
            return age > GLOBAL_MODEL_LOAD_OWNER_STALE_SECONDS
        except Exception:
            return True

    def _try_acquire_global_model_load_owner(self, phase: str) -> bool:
        owner_payload = self._global_model_load_owner_payload(phase=phase)
        try:
            fd = os.open(
                str(self.model_load_owner_path),
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o644,
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(owner_payload, f, indent=2)
            except Exception:
                try:
                    os.close(fd)
                except Exception:
                    pass
                raise
            return True
        except FileExistsError:
            owner = self._read_global_model_load_owner()
            if self._global_owner_is_stale(owner):
                self.logger.warning(
                    "GLOBAL_LOAD_LOCK_STALE_OWNER_TAKEOVER "
                    f"worker={self.name} stale_owner={owner or {}}"
                )
                try:
                    self.model_load_owner_path.unlink()
                except Exception:
                    return False
                return False
            return False

    def _global_model_load_owner_heartbeat_loop(self, stop_event: threading.Event, phase: str):
        while not stop_event.wait(GLOBAL_MODEL_LOAD_OWNER_HEARTBEAT_INTERVAL):
            try:
                owner = self._read_global_model_load_owner()
                if not owner:
                    continue
                if str(owner.get("worker", "")).strip() != self.name:
                    continue
                if int(owner.get("pid", -1)) != os.getpid():
                    continue
                owner["phase"] = phase
                owner["heartbeat_at"] = datetime.now().isoformat()
                self._write_global_model_load_owner(owner)
            except Exception:
                continue

    def _release_global_model_load_owner(self):
        try:
            owner = self._read_global_model_load_owner()
            if owner:
                if (
                    str(owner.get("worker", "")).strip() == self.name
                    and int(owner.get("pid", -1)) == os.getpid()
                ):
                    self.model_load_owner_path.unlink(missing_ok=True)
                    return
            # If owner file is already gone or rotated, do nothing.
        except Exception:
            pass

    def _wait_for_model_ready(self, model_id: str, max_wait_seconds: int = 90) -> bool:
        """
        Probe Ollama until model is actually ready to serve.
        Prevents immediate LLM task claims while model is still warming.
        """
        if not self.api_url:
            return False

        deadline = time.time() + max_wait_seconds
        last_error = ""

        while time.time() < deadline:
            try:
                r = requests.post(
                    self.api_url,
                    json={
                        "model": model_id,
                        "prompt": "READY?",
                        "stream": False,
                        "keep_alive": self.worker_keep_alive,
                        "options": {
                            "num_predict": 4,
                            "num_gpu": 999,
                            "num_ctx": self.worker_num_ctx,
                        }
                    },
                    timeout=30
                )
                if r.status_code == 200:
                    _ = r.json().get("response", "")
                    return True
                last_error = f"status={r.status_code}"
            except Exception as e:
                last_error = str(e)

            time.sleep(2)

        self.logger.warning(f"Model readiness probe failed/timed out: {last_error}")
        return False

    def _wait_for_model_unloaded(self, model_id: str, max_wait_seconds: int = 30) -> bool:
        """Poll Ollama until this model no longer appears as loaded."""
        if not self.port:
            return True

        ps_url = f"http://localhost:{self.port}/api/ps"
        deadline = time.time() + max_wait_seconds
        last_error = ""

        while time.time() < deadline:
            try:
                r = requests.get(ps_url, timeout=5)
                if r.status_code == 200:
                    models = r.json().get("models", [])
                    loaded_names = [m.get("name", "") for m in models]
                    if model_id not in loaded_names:
                        return True
                else:
                    last_error = f"status={r.status_code}"
            except Exception as e:
                last_error = str(e)
            time.sleep(1)

        self.logger.warning(f"Model unload verify timed out: {last_error}")
        return False

    def unload_model(self, model_id: Optional[str] = None, task_id: Optional[str] = None):
        """Unload LLM model from VRAM. Transitions GPU to Cold state.

        Retries up to 3 times with 5s backoff on failure, since a failed unload
        leaves VRAM in an inconsistent state (agent thinks model is loaded but
        VRAM may be partially freed).

        Postcondition verification: marks GPU as wedged if unload fails to
        fully clear runtime state.
        """
        if not self.model_loaded:
            return

        if not self.api_url:
            return

        target_model = str(model_id or self.loaded_model or self.model or "").strip()
        if not target_model:
            self.logger.warning("Cannot unload model - target model missing")
            return

        # Transition to unloading state
        self._set_runtime_state(
            RUNTIME_STATE_UNLOADING,
            task_id=task_id,
            phase="starting_unload",
        )

        self.logger.info(f"Unloading model {target_model} to free VRAM...")

        max_retries = 3
        unload_succeeded = False
        last_error = ""

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(
                    self.api_url,
                    json={"model": target_model, "prompt": "", "keep_alive": 0},
                    timeout=30
                )

                if response.status_code == 200:
                    if self._wait_for_model_unloaded(model_id=target_model, max_wait_seconds=30):
                        unload_succeeded = True
                        break
                    last_error = "model_still_present_after_unload"
                    self.logger.warning(
                        f"Unload attempt {attempt}/{max_retries} did not verify model removal"
                    )
                else:
                    last_error = f"http_status_{response.status_code}"
                    self.logger.warning(
                        f"Unload attempt {attempt}/{max_retries} returned "
                        f"status {response.status_code}")

            except Exception as e:
                last_error = str(e)[:100]
                self.logger.warning(
                    f"Unload attempt {attempt}/{max_retries} failed: {e}")

            if attempt < max_retries:
                time.sleep(5)

        # Postcondition verification
        if unload_succeeded:
            self.model_loaded = False
            self.loaded_model = None
            self.loaded_tier = 0
            self.runtime_placement = "single_gpu"
            self.runtime_group_id = None
            self.runtime_port = self.port
            self.runtime_ollama_url = f"http://localhost:{self.port}" if self.port else None
            self._set_runtime_state(
                RUNTIME_STATE_COLD,
                task_id=task_id,
                phase="unload_complete",
            )
            self.logger.info("Model unloaded - GPU is now COLD")
        else:
            # Postconditions failed - mark as wedged
            self._mark_wedged(
                error_code="unload_postcondition_failed",
                error_detail=f"Failed to unload {target_model}: {last_error}",
                task_id=task_id,
            )
