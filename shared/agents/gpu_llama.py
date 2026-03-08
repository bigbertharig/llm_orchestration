"""GPU agent llama-server runtime management mixin.

Manages containerized llama-server lifecycle. One container = one loaded model.

External contract:
- start_llama / stop_llama
- check_llama_health
- load_model / unload_model (same signatures)

Runtime helpers used:
- scripts/llama_runtime/run_runtime.sh
- scripts/llama_runtime/stop_runtime.sh
- scripts/llama_runtime/probe_runtime.sh
"""

import subprocess
import time
import re
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from gpu_constants import (
    RUNTIME_STATE_COLD,
    RUNTIME_STATE_LOADING_SINGLE,
    RUNTIME_STATE_READY_SINGLE,
    RUNTIME_STATE_UNLOADING,
    SINGLE_META_TIMEOUT_SECONDS,
)
from brain_core import resolve_llama_runtime_profile, resolve_model_search_roots

# Path to runtime helper scripts.
# Primary: shared drive (accessible from rig via NFS).
# Fallback: repo-local scripts directory.
_SCRIPTS_DIR_SHARED = Path("/mnt/shared/scripts/llama_runtime")
_SCRIPTS_DIR_REPO = Path(__file__).resolve().parent.parent.parent / "scripts" / "llama_runtime"
_SCRIPTS_DIR = _SCRIPTS_DIR_SHARED if _SCRIPTS_DIR_SHARED.exists() else _SCRIPTS_DIR_REPO

# Default image tag for the dedicated llama-server runtime image
LLAMA_IMAGE_TAG = "llama-runtime:sm61-sm86"

# Readiness probe settings
LLAMA_READINESS_TIMEOUT_SECONDS = SINGLE_META_TIMEOUT_SECONDS
LLAMA_READINESS_POLL_INTERVAL = 2

# Health check settings
LLAMA_HEALTH_RESTART_THRESHOLD = 6
LLAMA_HEALTH_CIRCUIT_BREAKER = 8
_LLAMA_OFFLOAD_RE = re.compile(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU")


class GPULlamaMixin:
    """Mixin providing containerized llama-server runtime management."""

    def _llama_container_name(self) -> str:
        """Canonical container name for this GPU's single-worker runtime."""
        return f"llama-worker-{self.name}"

    def _llama_probe_port(self) -> Optional[int]:
        """Return the active llama API port for this runtime."""
        if self.runtime_placement == "split_gpu" and self.runtime_port:
            return self.runtime_port
        return self.port

    def _llama_api_base(self) -> Optional[str]:
        """Base URL for the llama-server HTTP API on this GPU's port."""
        port = self._llama_probe_port()
        if not port:
            return None
        return f"http://127.0.0.1:{port}"

    def start_llama(self):
        """Prepare the llama runtime on agent startup.

        llama-server containers are started per-model via load_model().
        At startup we just clean up any orphaned containers from a previous run.
        """
        if not self.port:
            self.logger.info("No port configured - script-only GPU, skipping llama runtime")
            return

        # Clean up orphaned container from a previous crash/restart
        container_name = self._llama_container_name()
        self._stop_llama_container(container_name)
        self.logger.info(
            f"llama runtime initialized (cold) on GPU {self.gpu_id}, port {self.port}"
        )

    def stop_llama(self):
        """Stop the containerized llama-server runtime. Removes the container forcefully."""
        container_name = self._llama_container_name()
        self._stop_llama_container(container_name)
        self._llama_container_id = None
        self.logger.info("llama-server stopped")

    def _stop_llama_container(self, container_name: str):
        """Stop and remove a Docker container by name. Idempotent."""
        try:
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except Exception as e:
            self.logger.debug(f"Container stop/remove for {container_name}: {e}")

    def _get_container_logs(self, container_name: str, tail: int = 30) -> str:
        """Get recent container logs for diagnostics."""
        try:
            result = subprocess.run(
                ["docker", "logs", "--tail", str(tail), container_name],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return (result.stdout or "") + (result.stderr or "")
        except Exception:
            return "(failed to retrieve container logs)"

    def _parse_offload_status(self, logs: str) -> Dict[str, Optional[int]]:
        """Extract offload layer counts from llama logs."""
        match = None
        for candidate in _LLAMA_OFFLOAD_RE.finditer(logs or ""):
            match = candidate
        if not match:
            return {
                "offloaded_layers": None,
                "total_layers": None,
            }
        return {
            "offloaded_layers": int(match.group(1)),
            "total_layers": int(match.group(2)),
        }

    def _probe_existing_loaded_model(
        self,
        *,
        container_name: str,
        gguf_path: str,
        model_id: str,
    ) -> bool:
        """Return True when the existing local runtime already serves the target model."""
        base_url = self._llama_api_base()
        if not base_url:
            return False
        try:
            response = requests.get(f"{base_url}/v1/models", timeout=3)
            if response.status_code != 200:
                return False
            payload = response.json()
            models = payload.get("data", [])
            if not models:
                return False
            existing_id = str(models[0].get("id", "")).strip()
            if not existing_id:
                return False
            if Path(existing_id).name != Path(gguf_path).name:
                return False
            self._assert_full_gpu_offload(container_name, model_id)
            return True
        except Exception:
            return False

    def _assert_full_gpu_offload(self, container_name: str, model_id: str) -> None:
        """Fail when llama reports a partial or missing GPU offload."""
        logs = self._get_container_logs(container_name, tail=2000)
        status = self._parse_offload_status(logs)
        offloaded = status["offloaded_layers"]
        total = status["total_layers"]
        if offloaded is None or total is None:
            raise RuntimeError(
                f"Unable to verify GPU offload for {model_id}; "
                "llama logs did not contain an offload summary"
            )
        if offloaded != total:
            raise RuntimeError(
                f"Model {model_id} not fully GPU resident: offloaded {offloaded}/{total} layers"
            )

    def _wait_for_llama_ready(self, timeout: int = LLAMA_READINESS_TIMEOUT_SECONDS) -> bool:
        """Poll /v1/models until the llama-server is ready to serve."""
        base_url = self._llama_api_base()
        if not base_url:
            return False

        url = f"{base_url}/v1/models"
        deadline = time.time() + timeout

        while time.time() < deadline:
            try:
                r = requests.get(url, timeout=3)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(LLAMA_READINESS_POLL_INTERVAL)

        return False

    def check_llama_health(self) -> Dict[str, Any]:
        """Check llama-server health via /v1/models endpoint.

        Returns dict with health status.
        Tracks consecutive failures and triggers restart after threshold.
        Implements circuit breaker to stop claiming LLM tasks after sustained failures.
        """
        health = {
            "healthy": False,
            "loaded_models": [],
            "consecutive_failures": self.runtime_consecutive_failures,
            "response_ms": None,
        }

        base_url = self._llama_api_base()
        if not base_url:
            health["healthy"] = True
            health["note"] = "no_runtime_configured"
            return health

        # When cold (no model loaded), there is no container to probe.
        # Report healthy — the agent is simply idle.
        if not self.model_loaded:
            health["healthy"] = True
            health["note"] = "cold_no_container"
            self.runtime_consecutive_failures = 0
            self.runtime_healthy = True
            return health

        try:
            start = time.time()
            resp = requests.get(
                f"{base_url}/v1/models",
                timeout=5,
            )
            elapsed_ms = int((time.time() - start) * 1000)
            health["response_ms"] = elapsed_ms

            if resp.status_code == 200:
                data = resp.json()
                models = data.get("data", [])
                health["loaded_models"] = [
                    m.get("id", "") for m in models if isinstance(m, dict)
                ]
                health["healthy"] = True
                self.runtime_consecutive_failures = 0
                self.runtime_healthy = True
            else:
                self.runtime_consecutive_failures += 1
                self.logger.warning(
                    f"llama health check returned {resp.status_code} "
                    f"(failure {self.runtime_consecutive_failures})"
                )

        except Exception as e:
            self.runtime_consecutive_failures += 1
            self.logger.warning(
                f"llama health check failed: {e} "
                f"(failure {self.runtime_consecutive_failures})"
            )

        health["consecutive_failures"] = self.runtime_consecutive_failures

        # Auto-recovery after threshold: kill the container and go cold.
        # The brain will re-issue a load_llm meta task if needed.
        if self.runtime_consecutive_failures >= self.runtime_health_threshold:
            self.logger.error(
                f"llama-server failed {self.runtime_consecutive_failures} consecutive "
                f"health checks, stopping container and going cold"
            )
            try:
                self.stop_llama()
                self.runtime_consecutive_failures = 0
                self.runtime_healthy = True
                self.model_loaded = False
                self.loaded_model = None
                self.loaded_tier = 0
                self.runtime_placement = "single_gpu"
                self.runtime_group_id = None
                self.runtime_port = self.port
                self.runtime_api_base = self._llama_api_base()
                self._set_runtime_state(
                    RUNTIME_STATE_COLD,
                    phase="llama_health_recovery",
                )
                self.logger.info("llama-server stopped after health failures, GPU is cold")
            except Exception as e:
                self.logger.error(f"llama-server health recovery failed: {e}")

        # Circuit breaker
        if self.runtime_consecutive_failures >= self.runtime_circuit_breaker:
            self.runtime_healthy = False
            self.logger.error(
                f"Circuit breaker: llama-server unhealthy after "
                f"{self.runtime_consecutive_failures} failures, will not claim LLM tasks"
            )

        return health

    def _resolve_gguf_path(self, model_id: str) -> Optional[str]:
        """Resolve a model ID to a GGUF file path.

        Looks up the model in the catalog for a gguf_path field first,
        then falls back to convention-based path resolution under /mnt/shared/models/.
        """
        # Check catalog for explicit gguf_path
        meta = self.model_meta_by_id.get(model_id, {})
        explicit = meta.get("gguf_path")
        if explicit and Path(explicit).exists():
            return str(explicit)

        # Convention-based resolution: model_id like "qwen2.5:7b" or "qwen2.5-coder:14b"
        # Maps to one of the configured model search roots.
        model_roots = resolve_model_search_roots(self.config)

        # Try direct lookup with common naming patterns
        # "qwen2.5:7b" -> try "qwen2.5-7b", "qwen2.5-coder-7b" etc.
        normalized = model_id.replace(":", "-").replace("/", "-")

        # Search for matching directory
        for models_root in model_roots:
            if not models_root.exists():
                continue
            for candidate_dir in models_root.iterdir():
                if not candidate_dir.is_dir():
                    continue
                dir_name = candidate_dir.name.lower()
                if normalized.lower() in dir_name or dir_name in normalized.lower():
                    # Found a matching directory - look for the GGUF file
                    ggufs = list(candidate_dir.glob("*.gguf"))
                    if len(ggufs) == 1:
                        return str(ggufs[0])
                    # Multiple GGUFs - prefer Q4_K_M
                    for g in ggufs:
                        if "Q4_K_M" in g.name:
                            return str(g)
                    if ggufs:
                        return str(ggufs[0])

        self.logger.warning(f"Could not resolve GGUF path for model {model_id}")
        return None

    def _llama_load_model(self, model_id: Optional[str] = None, task_id: Optional[str] = None):
        """Load an LLM model by starting a containerized llama-server runtime.

        In the llama-server model, "loading" means starting a container with the
        target GGUF. The server process IS the loaded model.
        """
        # Preflight: check runtime state allows load
        can_load, preflight_reason = self._can_accept_load_task()
        if not can_load:
            self.logger.warning(
                f"LOAD_PREFLIGHT_REJECT worker={self.name} reason={preflight_reason}"
            )
            return

        if self.model_loaded:
            return

        if not self.port:
            self.logger.warning("Cannot load model - no port configured")
            return

        target_model = str(model_id or self.model or "").strip()
        if not target_model:
            self.logger.warning("Cannot load model - target model missing")
            return

        gguf_path = self._resolve_gguf_path(target_model)
        if not gguf_path:
            self._set_runtime_state(
                RUNTIME_STATE_COLD,
                task_id=task_id,
                phase="load_failed",
                error_code="gguf_not_found",
                error_detail=f"No GGUF file found for model {target_model}",
            )
            self.logger.error(f"Cannot load model {target_model} - GGUF path not found")
            return

        # Transition to loading state
        self._set_runtime_state(
            RUNTIME_STATE_LOADING_SINGLE,
            task_id=task_id,
            phase="preflight_passed",
        )

        self.logger.info(f"Loading model {target_model} (gguf={gguf_path})...")
        start_time = time.time()
        deadline = start_time + SINGLE_META_TIMEOUT_SECONDS
        container_name = self._llama_container_name()
        profile = resolve_llama_runtime_profile(
            self.config,
            model_id=target_model,
            gpu_config=getattr(self, "gpu_config", None),
            split=False,
        )

        def _do_load():
            remaining_budget = deadline - time.time()
            if remaining_budget <= 0:
                raise RuntimeError(
                    f"single model load timeout after {SINGLE_META_TIMEOUT_SECONDS}s "
                    "(no time left after global lock wait)"
                )

            if self._probe_existing_loaded_model(
                container_name=container_name,
                gguf_path=gguf_path,
                model_id=target_model,
            ):
                self.logger.info(
                    f"LLAMA_LOAD_REUSE worker={self.name} model={target_model} port={self.port}"
                )
                return

            # Stop any existing container first
            self._stop_llama_container(container_name)

            # Start new container
            run_script = str(_SCRIPTS_DIR / "run_runtime.sh")
            cmd = [
                run_script,
                "--name", container_name,
                "--model", gguf_path,
                "--port", str(self.port),
                "--gpus", f"device={self.gpu_id}",
                "--ctx-size", str(profile.get("ctx_size", self.worker_num_ctx)),
                "--n-gpu-layers", str(profile.get("n_gpu_layers", -1)),
                "--batch-size", str(profile.get("batch_size", 512)),
                "--parallel", str(profile.get("parallel", 1)),
            ]
            if profile.get("threads") is not None:
                cmd.extend(["--threads", str(profile["threads"])])
            if profile.get("tensor_split"):
                cmd.extend(["--tensor-split", str(profile["tensor_split"])])
            for extra_arg in profile.get("extra_args", []):
                cmd.extend(["--extra-arg", str(extra_arg)])

            self.logger.info(f"LLAMA_LOAD_CMD: {' '.join(cmd)}")
            self._llama_start_cmd = " ".join(cmd)
            self._llama_start_time = time.time()

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                stderr_tail = (result.stderr or "")[-500:]
                raise RuntimeError(
                    f"run_runtime.sh failed (rc={result.returncode}): {stderr_tail}"
                )
            self._llama_container_id = (result.stdout or "").strip()

            # Wait for readiness with remaining budget
            readiness_budget = max(10, int(deadline - time.time()))

            # Poll with meta-task heartbeat
            url = f"http://127.0.0.1:{self.port}/v1/models"
            readiness_deadline = time.time() + readiness_budget
            last_wait_log_at = time.time()

            while time.time() < readiness_deadline:
                self._touch_meta_task(phase="load_llm_waiting_ready")
                try:
                    r = requests.get(url, timeout=3)
                    if r.status_code == 200:
                        self._assert_full_gpu_offload(container_name, target_model)
                        return  # Ready!
                except Exception:
                    pass

                now_wait = time.time()
                if (now_wait - last_wait_log_at) >= 20:
                    self.logger.info(
                        f"LLAMA_LOAD_WAIT worker={self.name} model={target_model} "
                        f"elapsed_s={int(now_wait - self._llama_start_time)}"
                    )
                    last_wait_log_at = now_wait
                time.sleep(LLAMA_READINESS_POLL_INTERVAL)

            # Timeout - grab logs for diagnostics
            logs = self._get_container_logs(container_name, tail=30)
            raise RuntimeError(
                f"llama-server readiness probe timed out within "
                f"{SINGLE_META_TIMEOUT_SECONDS}s load budget. "
                f"Container logs:\n{logs}"
            )

        try:
            self._set_runtime_state(
                RUNTIME_STATE_LOADING_SINGLE,
                task_id=task_id,
                phase="acquiring_global_lock",
            )
            self._run_with_global_model_load_lock(
                phase="single_model_load",
                fn=_do_load,
                max_wait_seconds=SINGLE_META_TIMEOUT_SECONDS,
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
            self.runtime_api_base = self._llama_api_base()
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
            # Capture container logs on failure
            logs = self._get_container_logs(container_name, tail=20)
            error_detail = f"{str(e)[:200]}"
            if logs and "(failed to retrieve" not in logs:
                error_detail += f" | container_logs_tail: {logs[-200:]}"
            # Transition back to cold on failure
            self._set_runtime_state(
                RUNTIME_STATE_COLD,
                task_id=task_id,
                phase="load_failed",
                error_code="load_exception",
                error_detail=error_detail[:400],
            )
            # Clean up the failed container
            self._stop_llama_container(container_name)
            self.logger.error(f"Failed to load model: {e}")

    def _llama_wait_for_model_ready(self, model_id: str, max_wait_seconds: int = 90) -> bool:
        """Probe llama-server until it is ready to serve.

        For llama-server, readiness is determined by /v1/models returning 200.
        We also do a quick inference probe to confirm the model is warm.
        """
        if not self.port:
            return False

        base_url = f"http://127.0.0.1:{self.port}"
        deadline = time.time() + max_wait_seconds
        last_error = ""

        while time.time() < deadline:
            try:
                # First check /v1/models
                r = requests.get(f"{base_url}/v1/models", timeout=3)
                if r.status_code != 200:
                    last_error = f"models_status={r.status_code}"
                    time.sleep(LLAMA_READINESS_POLL_INTERVAL)
                    continue

                # Then do a minimal inference probe
                r = requests.post(
                    f"{base_url}/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": "READY?"}],
                        "max_tokens": 4,
                    },
                    timeout=30,
                )
                if r.status_code == 200:
                    return True
                last_error = f"inference_status={r.status_code}"
            except Exception as e:
                last_error = str(e)

            time.sleep(LLAMA_READINESS_POLL_INTERVAL)

        self.logger.warning(f"Model readiness probe failed/timed out: {last_error}")
        return False

    def _llama_wait_for_model_unloaded(self, model_id: str, max_wait_seconds: int = 30) -> bool:
        """Verify the llama-server container is stopped.

        For llama-server, "unloaded" means the container is gone and the port
        is no longer responding.
        """
        if not self.port:
            return True

        url = f"http://127.0.0.1:{self.port}/v1/models"
        deadline = time.time() + max_wait_seconds

        while time.time() < deadline:
            try:
                r = requests.get(url, timeout=2)
                # Still responding - not unloaded yet
            except requests.exceptions.ConnectionError:
                # Port refused connection - container is down
                return True
            except Exception:
                return True
            time.sleep(1)

        self.logger.warning("Model unload verify timed out - port still responding")
        return False

    def _llama_unload_model(self, model_id: Optional[str] = None, task_id: Optional[str] = None):
        """Unload LLM model by stopping the containerized runtime.

        In the llama-server model, unloading means stopping and removing the
        container.
        """
        if not self.model_loaded:
            return

        if not self.port:
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

        self.logger.info(f"Unloading model {target_model} (stopping container)...")

        container_name = self._llama_container_name()
        max_retries = 3
        unload_succeeded = False
        last_error = ""

        for attempt in range(1, max_retries + 1):
            try:
                self._stop_llama_container(container_name)

                if self._llama_wait_for_model_unloaded(model_id=target_model, max_wait_seconds=15):
                    unload_succeeded = True
                    break
                last_error = "port_still_responding_after_container_stop"
                self.logger.warning(
                    f"Unload attempt {attempt}/{max_retries} - port still responding"
                )
            except Exception as e:
                last_error = str(e)[:100]
                self.logger.warning(
                    f"Unload attempt {attempt}/{max_retries} failed: {e}"
                )

            if attempt < max_retries:
                time.sleep(3)

        # Postcondition verification
        if unload_succeeded:
            self.model_loaded = False
            self.loaded_model = None
            self.loaded_tier = 0
            self.runtime_placement = "single_gpu"
            self.runtime_group_id = None
            self.runtime_port = self.port
            self.runtime_api_base = self._llama_api_base()
            self._llama_container_id = None
            self._set_runtime_state(
                RUNTIME_STATE_COLD,
                task_id=task_id,
                phase="unload_complete",
            )
            self.logger.info("Model unloaded - GPU is now COLD")
        else:
            self._mark_wedged(
                error_code="unload_postcondition_failed",
                error_detail=f"Failed to unload {target_model}: {last_error}",
                task_id=task_id,
            )

    # Global model load lock methods are inherited from GPURuntimeMixin via MRO.
    # They use filesystem-based coordination and are not backend-specific.
