"""GPU agent task claiming and handling mixin.

Extracted from gpu.py to isolate task queue interaction, claiming logic,
and meta task (load/unload) handling.
"""

import importlib.util
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from filelock import FileLock, Timeout

import requests

from brain_core import resolve_llama_runtime_profile
from gpu_constants import (
    ATTESTATION_MISS_HARD_FAIL_THRESHOLD,
    ATTESTATION_MISS_SOFT_FAIL_THRESHOLD,
    DEFAULT_LLM_MIN_TIER,
    META_TASK_HEARTBEAT_INTERVAL,
    MISMATCH_CIRCUIT_BREAKER_COUNT,
    MISMATCH_CIRCUIT_BREAKER_WINDOW_SECONDS,
    MISMATCH_WEDGE_COOLDOWN_SECONDS,
    READY_MIN_AGE_SECONDS,
    RUNTIME_STATE_COLD,
    RUNTIME_STATE_RESETTING_THERMAL,
    RUNTIME_STATE_UNLOADING,
    RUNTIME_STATE_WEDGED,
    SPLIT_META_TIMEOUT_SECONDS,
    VALID_TASK_CLASSES,
    RUNTIME_STATES_READY,
)
from gpu_workers import WorkerResult


class GPUTaskMixin:
    """Mixin providing task claiming and meta task handling methods."""

    def _clear_runtime_mismatch_tracking(self) -> None:
        """Clear mismatch circuit-breaker tracking after explicit recovery."""
        self.mismatch_timestamps = []
        self.runtime_mismatch_count = 0
        self.runtime_recovery_cooldown_until = None

    def _safe_meta_detail(self, value: Any, limit: int = 400) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        return text[:limit]

    def _meta_runtime_diagnostics(self, command: str, task: Dict[str, Any]) -> Dict[str, Any]:
        target_model = str(task.get("target_model", "") or "").strip()
        candidate_groups = task.get("candidate_groups", [])
        details: Dict[str, Any] = {
            "command": command,
            "worker": self.name,
            "runtime_state": self.runtime_state,
            "runtime_error_code": self.runtime_error_code,
            "runtime_error_detail": self._safe_meta_detail(self.runtime_error_detail),
            "model_loaded": self.model_loaded,
            "loaded_model": self.loaded_model,
            "runtime_placement": getattr(self, "runtime_placement", None),
            "runtime_group_id": getattr(self, "runtime_group_id", None),
            "runtime_port": getattr(self, "runtime_port", None),
            "last_split_runtime_error": self._safe_meta_detail(
                getattr(self, "last_split_runtime_error", "")
            ),
        }
        if target_model:
            details["target_model"] = target_model
        if command == "load_split_llm" and isinstance(candidate_groups, list):
            details["candidate_group_ids"] = [
                str(group.get("id", "")).strip()
                for group in candidate_groups
                if isinstance(group, dict) and str(group.get("id", "")).strip()
            ]
        return details

    def _build_meta_result(
        self,
        command: str,
        task: Dict[str, Any],
        *,
        success: bool,
        output: str = "",
        error: str = "",
        error_type: str = "",
        error_code: str = "",
        diagnostic: str = "",
        **extra: Any,
    ) -> Dict[str, Any]:
        details = self._meta_runtime_diagnostics(command, task)
        result: Dict[str, Any] = {
            "success": success,
            "output": output,
            "worker": self.name,
            "max_vram_used_mb": 0,
            "model_loaded": self.model_loaded,
            "loaded_model": self.loaded_model,
            "runtime_state": self.runtime_state,
            "runtime_placement": details.get("runtime_placement"),
            "runtime_group_id": details.get("runtime_group_id"),
            "runtime_port": details.get("runtime_port"),
            "runtime_error_code": details.get("runtime_error_code"),
            "runtime_error_detail": details.get("runtime_error_detail"),
            "details": details,
        }
        if error_code:
            result["error_code"] = error_code
        if diagnostic:
            result["diagnostic"] = self._safe_meta_detail(diagnostic)
        if success:
            result["error"] = ""
            result["error_type"] = ""
        else:
            final_error = self._safe_meta_detail(error or output or diagnostic or "meta task failed")
            result["error"] = final_error
            result["error_type"] = error_type or "meta_runtime_failure"
            if not result.get("diagnostic"):
                result["diagnostic"] = final_error
        result.update(extra)
        return result

    def _task_required_llm_model(self, task: Dict[str, Any]) -> str:
        for key in ("llm_model", "worker_model", "target_model"):
            value = str(task.get(key, "") or "").strip()
            if value:
                return value
        return ""

    def _llm_task_runtime_compatible(self, task: Dict[str, Any]) -> tuple[bool, str]:
        """Check if current runtime can handle this LLM task.

        Capability-based routing: tasks are matched by model/tier requirements,
        not by placement. A 14B split runtime can serve 7B tasks if tier is sufficient.

        Placement (single_gpu, split_gpu) is advisory/telemetry only for normal LLM work.
        """
        if str(task.get("task_class", "")).strip() != "llm":
            return True, ""
        if self.runtime_state not in RUNTIME_STATES_READY:
            return False, f"runtime_not_ready:{self.runtime_state}"
        if not self.model_loaded:
            return False, "model_not_loaded"

        # 1. Exact model match (if task specifies a specific model)
        required_model = self._task_required_llm_model(task)
        if required_model:
            loaded = str(self.loaded_model or "").strip()
            if loaded != required_model:
                # Check if loaded model satisfies tier requirement instead of exact match
                # This allows 14B to serve tasks that specify "qwen2.5:7b" as long as
                # the loaded model has equal or higher tier
                required_tier = int(self.model_tier_by_id.get(required_model, DEFAULT_LLM_MIN_TIER))
                if self.loaded_tier < required_tier:
                    return False, f"tier_insufficient:{self.loaded_tier}<{required_tier}"
                # Tier is sufficient - model compatibility assumed for same family
                # (e.g. qwen2.5:14b can handle qwen2.5:7b tasks)

        # 2. Minimum tier check (if task specifies llm_min_tier)
        task_min_tier = task.get("llm_min_tier")
        if task_min_tier is not None:
            required_tier = int(task_min_tier)
            if self.loaded_tier < required_tier:
                return False, f"tier_insufficient:{self.loaded_tier}<{required_tier}"

        # 3. Placement is advisory only - no hard gating for normal LLM work
        # Placement may still be used by brain for scheduling preferences,
        # but workers should not reject based on placement mismatch alone.
        # (Meta tasks like load_split_llm/unload_split_llm handle placement explicitly)

        return True, ""

    def _attest_runtime_reality(self) -> Dict[str, Any]:
        """Probe actual runtime state (not cached heartbeat fields).

        Used for claim-time attestation to detect state drift between
        heartbeat/runtime fields and actual runtime /v1/models.

        Returns:
            {
                "ok": bool - True if reality matches expected state
                "mismatch_reason": str - Empty if ok, else reason code
                "details": {
                    "expected_model_loaded": bool,
                    "expected_loaded_model": str,
                    "actual_port_models": list[str],
                    "expected_runtime_state": str,
                    "probed_at": str,
                }
            }
        """
        result: Dict[str, Any] = {
            "ok": True,
            "mismatch_reason": "",
            "details": {
                "expected_model_loaded": self.model_loaded,
                "expected_loaded_model": self.loaded_model,
                "expected_runtime_state": self.runtime_state,
                "actual_port_models": [],
                "probed_at": datetime.now().isoformat(),
            },
        }

        # Determine which port to probe
        if self.runtime_placement == "split_gpu" and self.runtime_port:
            probe_port = self.runtime_port
        else:
            probe_port = self.port

        if not probe_port:
            # No port to probe - can't attest
            return result

        # Probe actual runtime state (backend-aware)
        actual_models = []
        backend = getattr(self, 'runtime_backend', 'llama')
        try:
            # llama-server: /v1/models returns model list; port alive = model loaded.
            # The model ID from /v1/models is the GGUF file path, not the
            # config-style model name. Since llama-server serves exactly one
            # model, a 200 response means our expected model is loaded.
            r = requests.get(f"http://127.0.0.1:{probe_port}/v1/models", timeout=2)
            if r.status_code == 200:
                # Use our internal model name as the actual model so attestation
                # comparison works correctly.
                actual_models = [str(self.loaded_model or "unknown")]
        except requests.exceptions.ConnectionError:
            # Port not responding - interpret as no models loaded
            actual_models = []
        except Exception as e:
            self.logger.debug(f"Runtime attestation probe error: {e}")
            # On probe error, we can't attest - treat as ok to avoid false positives
            return result

        result["details"]["actual_port_models"] = actual_models

        # Check for mismatches
        # Case 1: We think model is loaded but port shows no models
        if self.model_loaded and self.runtime_state in RUNTIME_STATES_READY:
            expected_model = str(self.loaded_model or "").strip()
            if expected_model and expected_model not in actual_models:
                result["ok"] = False
                result["mismatch_reason"] = (
                    f"expected_model_missing:expected={expected_model},"
                    f"actual={actual_models}"
                )
                return result

        # Case 2: We think model is not loaded but port shows models
        if not self.model_loaded and self.runtime_state == RUNTIME_STATE_COLD:
            if actual_models:
                result["ok"] = False
                result["mismatch_reason"] = (
                    f"unexpected_models_loaded:expected_cold,actual={actual_models}"
                )
                return result

        return result

    def _validate_split_ready_token(self) -> Dict[str, Any]:
        """Validate split ready token before claiming split-tier LLM tasks.

        Returns:
            {
                "ok": bool - True if token is valid
                "reason": str - Empty if ok, else reason code
                "details": dict with token info
            }
        """
        result = {
            "ok": True,
            "reason": "",
            "details": {
                "runtime_placement": self.runtime_placement,
                "runtime_group_id": self.runtime_group_id,
                "checked_at": datetime.now().isoformat(),
            },
        }

        # Only validate for split_gpu placement
        if self.runtime_placement != "split_gpu":
            return result

        if not self.runtime_group_id:
            result["ok"] = False
            result["reason"] = "no_runtime_group_id"
            return result

        # Read the reservation and check for ready_token
        try:
            res_path = self._split_reservation_path(self.runtime_group_id)
            if not res_path.exists():
                result["ok"] = False
                result["reason"] = "reservation_missing"
                return result

            with open(res_path, "r") as f:
                reservation = json.load(f)

            status = str(reservation.get("status", "")).strip()
            if status != "ready":
                result["ok"] = False
                result["reason"] = f"reservation_status_{status}"
                return result

            ready_token = reservation.get("ready_token")
            if not ready_token:
                result["ok"] = False
                result["reason"] = "no_ready_token"
                return result

            issued_at_str = reservation.get("ready_token_issued_at")
            if not issued_at_str:
                result["ok"] = False
                result["reason"] = "no_ready_token_issued_at"
                return result

            # Check token age
            try:
                issued_at = datetime.fromisoformat(str(issued_at_str))
                age_seconds = (datetime.now() - issued_at).total_seconds()
                result["details"]["token_age_seconds"] = age_seconds
                result["details"]["ready_token"] = ready_token[:8] + "..."

                if age_seconds < READY_MIN_AGE_SECONDS:
                    result["ok"] = False
                    result["reason"] = f"token_too_young:{age_seconds:.1f}s<{READY_MIN_AGE_SECONDS}s"
                    return result

            except Exception as e:
                result["ok"] = False
                result["reason"] = f"token_parse_error:{e}"
                return result

            # Also verify model is actually loaded on split port
            if self.runtime_port:
                target_model = str(reservation.get("target_model", "")).strip()
                if target_model:
                    try:
                        # llama-server: port responding = model loaded
                        r = requests.get(f"http://127.0.0.1:{self.runtime_port}/v1/models", timeout=2)
                        if r.status_code == 200:
                            actual_models = [target_model]  # llama-server serves exactly one model
                            result["details"]["actual_models"] = actual_models
                        else:
                            result["ok"] = False
                            result["reason"] = f"probe_failed:status={r.status_code}"
                            return result
                    except requests.exceptions.ConnectionError:
                        result["ok"] = False
                        result["reason"] = "split_port_unreachable"
                        return result
                    except Exception as e:
                        result["ok"] = False
                        result["reason"] = f"probe_error:{e}"
                        return result

        except Exception as e:
            result["ok"] = False
            result["reason"] = f"validation_error:{e}"

        return result

    def _handle_attestation_miss(self, reason: str) -> Dict[str, Any]:
        """Handle an attestation miss with phased recovery.

        First miss: soft-fail + requeue, keep runtime (no reset)
        Second consecutive miss: mark failed + coordinated cleanup

        Returns:
            {
                "action": "soft_fail" | "hard_fail"
                "miss_count": int
                "reason": str
            }
        """
        result = {
            "action": "soft_fail",
            "miss_count": 1,
            "reason": reason,
        }

        # Track miss count in local state
        miss_count = getattr(self, 'pending_attestation_miss_count', 0) + 1
        self.pending_attestation_miss_count = miss_count
        result["miss_count"] = miss_count

        if miss_count >= ATTESTATION_MISS_HARD_FAIL_THRESHOLD:
            result["action"] = "hard_fail"
            self.logger.warning(
                f"ATTESTATION_MISS_HARD_FAIL worker={self.name} "
                f"miss_count={miss_count} reason={reason}"
            )
            # Reset miss count after hard fail
            self.pending_attestation_miss_count = 0
        else:
            self.logger.warning(
                f"ATTESTATION_MISS_SOFT_FAIL worker={self.name} "
                f"miss_count={miss_count}/{ATTESTATION_MISS_HARD_FAIL_THRESHOLD} reason={reason}"
            )

        return result

    def _clear_attestation_miss_count(self):
        """Clear attestation miss count after successful claim."""
        self.pending_attestation_miss_count = 0

    def _attest_meta_task_precondition(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Command-specific attestation for meta tasks that mutate runtime.

        Different commands have different precondition requirements:
        - load_llm: local runtime should be cold/no models (unless valid reuse)
        - load_split_llm: member clean preconditions + no conflicting local models
        - unload_llm: model may or may not be present; mismatch is informational only
        - unload_split_llm: sanity-check ownership/runtime reality

        Returns:
            {
                "ok": bool,
                "mismatch_reason": str,
                "allow_continue": bool - True if mismatch is non-fatal for this command
            }
        """
        meta_cmd = str(task.get("command", "")).strip()
        result = {"ok": True, "mismatch_reason": "", "allow_continue": False}

        # Probe actual runtime
        attestation = self._attest_runtime_reality()
        actual_models = attestation["details"].get("actual_port_models", [])

        if meta_cmd == "load_llm":
            # load_llm: local runtime should be cold/no models loaded
            if self.model_loaded:
                # Already have model - may be trying to reload same model (ok)
                # or different model (attestation needed)
                pass
            if not self.model_loaded and actual_models:
                # State says cold but runtime has models - mismatch
                result["ok"] = False
                result["mismatch_reason"] = f"load_llm:expected_cold_but_models_present:{actual_models}"
                result["allow_continue"] = True
                return result

        elif meta_cmd == "load_split_llm":
            # load_split_llm: need clean member precondition (delegated to split mixin)
            # Also check no conflicting models on local port
            if not self.model_loaded and actual_models:
                result["ok"] = False
                result["mismatch_reason"] = f"load_split_llm:expected_cold_but_models_present:{actual_models}"
                result["allow_continue"] = True
                return result
            # Split-specific checks are done in _verify_local_split_member_clean_precondition

        elif meta_cmd == "unload_llm":
            # unload_llm: model may or may not be present
            # If we think model is loaded but it's not, that's weird but unload can still proceed
            # Don't fail here - just log if there's a mismatch
            if self.model_loaded and not actual_models:
                self.logger.warning(
                    f"META_ATTEST_INFO unload_llm: expected_model_loaded but port empty "
                    f"(proceeding anyway)"
                )
            result["allow_continue"] = True  # Unload can proceed even on mismatch

        elif meta_cmd == "unload_split_llm":
            # unload_split_llm: similar to unload_llm - informational only
            result["allow_continue"] = True

        # For strict commands, propagate any base attestation failure
        if meta_cmd in ("load_llm", "load_split_llm") and not attestation["ok"]:
            result["ok"] = False
            result["mismatch_reason"] = attestation["mismatch_reason"]
            result["allow_continue"] = True

        return result

    def _release_task_for_mismatch(
        self,
        task: Dict[str, Any],
        task_path: Path,
        mismatch_reason: str,
    ) -> None:
        """Release a tentatively claimed task back to queue without retry penalty.

        This is NOT a task failure - it's a claim-time runtime mismatch.
        The task goes back to queue for another worker to claim.

        Key semantics (from spec):
        - Do NOT increment attempts/retry count
        - Do NOT add to workers_attempted
        - Do NOT trigger retry/escalation logic
        - Record mismatch for observability
        """
        task_id = task.get("task_id", "")[:8]
        self.logger.warning(
            f"CLAIM_RELEASED_RUNTIME_MISMATCH task={task_id} "
            f"worker={self.name} reason={mismatch_reason}"
        )

        # Add mismatch record for debugging but don't touch retry fields
        task["last_mismatch_release"] = {
            "worker": self.name,
            "reason": mismatch_reason,
            "released_at": datetime.now().isoformat(),
        }

        # Write back to original location (queue) without incrementing attempts
        # Note: task has NOT been moved to processing yet, so just update in place
        try:
            with open(task_path, 'w') as f:
                json.dump(task, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to release task {task_id}: {e}")

    def _full_local_reset(
        self,
        reason: str,
        *,
        count_toward_circuit_breaker: bool = True,
    ) -> None:
        """Perform full local reset when runtime state drift is detected.

        Called after releasing a task due to claim-time runtime mismatch.
        The worker should treat itself as untrusted and reset before claiming again.

        Uses shared _runtime_reset_port_and_state() core plus circuit breaker logic.

        NOTE: Reset ends in verified cold state. Runtime is NOT auto-restarted.
        Re-warm happens through normal task flow (load_llm) to prevent thrashing.

        Circuit breaker: If too many mismatches in short window, worker enters wedged state.
        """
        self.logger.warning(
            f"FULL_LOCAL_RESET worker={self.name} reason={reason} "
            f"count_toward_circuit_breaker={count_toward_circuit_breaker}"
        )

        now = datetime.now()
        if count_toward_circuit_breaker:
            mismatch_timestamps = getattr(self, 'mismatch_timestamps', [])
            mismatch_timestamps.append(now)
            cutoff = now.timestamp() - MISMATCH_CIRCUIT_BREAKER_WINDOW_SECONDS
            mismatch_timestamps = [t for t in mismatch_timestamps if t.timestamp() > cutoff]
            self.mismatch_timestamps = mismatch_timestamps
            self.runtime_mismatch_count = len(mismatch_timestamps)

            if self.runtime_mismatch_count >= MISMATCH_CIRCUIT_BREAKER_COUNT:
                self.logger.error(
                    f"MISMATCH_CIRCUIT_BREAKER_TRIGGERED worker={self.name} "
                    f"count={self.runtime_mismatch_count} within {MISMATCH_CIRCUIT_BREAKER_WINDOW_SECONDS}s"
                )
                self.runtime_recovery_cooldown_until = (
                    now.timestamp() + MISMATCH_WEDGE_COOLDOWN_SECONDS
                )
                self._set_runtime_state(
                    RUNTIME_STATE_WEDGED,
                    phase=f"circuit_breaker:{reason}",
                )
                try:
                    self._write_heartbeat()
                except Exception as e:
                    self.logger.warning(f"Heartbeat write after wedge failed: {e}")
                return
        else:
            self._clear_runtime_mismatch_tracking()

        # Build list of ports to clean (local worker port + split port if different)
        ports_to_clean = []
        if self.port:
            ports_to_clean.append(self.port)
        split_port = getattr(self, 'runtime_port', None)
        if split_port and split_port != self.port:
            ports_to_clean.append(split_port)

        # Use shared reset core for port cleanup and state reset
        reset_result = self._runtime_reset_port_and_state(
            ports_to_clean=ports_to_clean,
            reason=f"mismatch:{reason}",
            task_id=None,
            stop_split_runtime=True,
            stop_local_runtime=True,
        )

        self.logger.info(
            f"FULL_LOCAL_RESET_COMPLETE worker={self.name} "
            f"mismatch_count={self.runtime_mismatch_count} state=cold (no_auto_restart) "
            f"ports_cleaned={list(reset_result.get('ports_cleaned', {}).keys())}"
        )

    def _get_preferred_classes(self) -> List[str]:
        """Get task classes this GPU should claim based on current state.

        Circuit breaker: if runtime is unhealthy, exclude LLM tasks to prevent
        burning through timeouts on a broken runtime instance.
        """
        if not self.runtime_healthy:
            self.logger.debug("Runtime unhealthy - excluding LLM tasks from claim")
            return ['meta', 'script']
        if self.state == "hot":
            return ['meta', 'llm']
        # Cold GPUs should focus on script/cpu until a load_llm task
        # explicitly transitions them to hot.
        return ['meta', 'script']

    def _scan_emergency_meta_tasks(self, allowed_commands: set) -> List[Dict]:
        """Scan queue for emergency meta-tasks that bypass thermal pause.

        These are high-priority tasks issued by the brain's thermal recovery controller
        that should run even when the GPU is thermally paused.

        Args:
            allowed_commands: Set of command strings that are allowed through

        Returns:
            List of claimed emergency tasks (at most one per command type)
        """
        claimed = []
        seen_commands = set()

        for task_file in sorted(self.queue_path.glob("*.json")):
            if str(task_file).endswith('.lock'):
                continue
            try:
                with open(task_file) as f:
                    task = json.load(f)

                # Only process meta tasks
                if task.get("task_class") != "meta":
                    continue

                command = str(task.get("command", ""))
                if command not in allowed_commands:
                    continue

                # Dedup: only one per command type per cycle
                if command in seen_commands:
                    continue

                # Check target worker matches this GPU
                # Brain uses target_worker; support target_gpu as fallback for compatibility
                target_worker = task.get("target_worker") or task.get("target_gpu")
                if target_worker and target_worker != self.name:
                    self.logger.debug(
                        f"THERMAL_EMERGENCY_SKIP: task {task.get('task_id', '')[:8]} "
                        f"targeted for {target_worker}, not {self.name}"
                    )
                    continue

                # Try to claim it
                lock_file = str(task_file) + ".lock"
                lock = FileLock(lock_file, timeout=1)

                try:
                    with lock:
                        if not task_file.exists():
                            continue

                        # Re-read inside lock
                        with open(task_file) as f:
                            task = json.load(f)

                        # Move to processing
                        dest = self.processing_path / task_file.name
                        task_file.rename(dest)

                        self.logger.warning(
                            f"THERMAL_EMERGENCY_CLAIMED: command={command} "
                            f"task_id={task.get('task_id', '')[:8]} "
                            f"incident_id={getattr(self, 'thermal_overheat_incident_id', 'none')}"
                        )

                        claimed.append(task)
                        seen_commands.add(command)

                except Exception as e:
                    self.logger.debug(f"Could not claim emergency task: {e}")
                    continue

            except Exception:
                continue

        return claimed

    def _can_claim_meta_task(self, task: Dict) -> tuple[bool, str]:
        """Enforce strict runtime state machine rules for meta tasks.

        Returns (can_claim, reject_reason). Reject reason is empty string if claimable.

        Key invariants enforced:
        1. No load while loaded (ready_single or ready_split)
        2. No work during transitions (loading_*, unloading)
        3. Wedged GPUs prefer reclaim/unload tasks
        """
        command = task.get("command", "")

        def _is_targeted_to_self() -> bool:
            target_worker = task.get("target_worker") or task.get("target_gpu")
            if target_worker and str(target_worker).strip() == self.name:
                return True
            candidates = task.get("candidate_workers", [])
            if isinstance(candidates, list) and candidates:
                allowed = [str(name).strip() for name in candidates if str(name).strip()]
                return len(allowed) == 1 and allowed[0] == self.name
            return False

        if command == "load_llm":
            target_model = str(task.get("target_model", "")).strip()
            if not target_model:
                return False, "missing_target_model"
            load_mode = str(task.get("load_mode", "")).strip()
            if load_mode and load_mode not in {"single", "either"}:
                return False, f"invalid_load_mode:{load_mode}"
            # Strict: Only cold workers should claim load commands.
            # Reject if in any ready, transitioning, or wedged state.
            can_load, reason = self._can_accept_load_task()
            if not can_load:
                # Targeted load_llm can be consumed as a no-op if this worker is
                # already hot and stable; this prevents stale targeted startup/default
                # load tasks from wedging queue progress forever.
                if (
                    reason in ("model_already_loaded", "ready_state")
                    and self.model_loaded
                    and self.runtime_placement != "split_gpu"
                    and self.runtime_state in RUNTIME_STATES_READY
                    and _is_targeted_to_self()
                ):
                    return True, ""
                if reason == "wedged_requires_reclaim":
                    self.logger.warning(
                        f"LOAD_META_AUTO_RESET command={command} worker={self.name} "
                        f"reason={reason}"
                    )
                    self._full_local_reset(
                        f"auto_reclaim_before_{command}",
                        count_toward_circuit_breaker=False,
                    )
                    can_load, reason = self._can_accept_load_task()
                    if can_load:
                        return True, ""
                return False, f"preflight_{reason}"
            if self.model_loaded:
                return False, "model_already_loaded"
            # Block load_llm while in split pair loading lock
            in_pair_lock, lock_info = self._is_in_split_pair_loading_lock()
            if in_pair_lock:
                return False, self._format_split_pair_lock_reason(lock_info)
            candidates = task.get("candidate_workers", [])
            if isinstance(candidates, list) and candidates:
                allowed = {str(name).strip() for name in candidates if str(name).strip()}
                if allowed and self.name not in allowed:
                    return False, "not_in_candidate_workers"
            return True, ""

        if command == "unload_llm":
            # Only single-GPU hot workers should claim generic unload_llm commands.
            # Split runtimes use unload_split_llm so the shared split listener can be
            # reclaimed correctly across both members.
            if not self.model_loaded or self.runtime_placement == "split_gpu":
                return False, "not_single_gpu_hot"
            if self._is_runtime_transitioning():
                return False, f"transitioning:{self.runtime_state}"
            # Block unload_llm while in split pair loading lock - a follower
            # that is still ready_single could disrupt an in-progress split load
            in_pair_lock, lock_info = self._is_in_split_pair_loading_lock()
            if in_pair_lock:
                return False, self._format_split_pair_lock_reason(lock_info)
            candidates = task.get("candidate_workers", [])
            if isinstance(candidates, list) and candidates:
                allowed = {str(name).strip() for name in candidates if str(name).strip()}
                if allowed and self.name not in allowed:
                    return False, "not_in_candidate_workers"
            return True, ""

        if command == "load_split_llm":
            load_mode = str(task.get("load_mode", "")).strip()
            if load_mode and load_mode not in {"split", "either"}:
                return False, f"invalid_load_mode:{load_mode}"
            # Strict: No load while in ready state
            can_load, reason = self._can_accept_load_task()
            if not can_load:
                if reason == "wedged_requires_reclaim":
                    self.logger.warning(
                        f"LOAD_META_AUTO_RESET command={command} worker={self.name} "
                        f"reason={reason}"
                    )
                    self._full_local_reset(
                        f"auto_reclaim_before_{command}",
                        count_toward_circuit_breaker=False,
                    )
                    can_load, reason = self._can_accept_load_task()
                    if can_load:
                        return True, ""
                return False, f"preflight_{reason}"
            target_model = str(task.get("target_model", "")).strip()
            target_tier = int(self.model_tier_by_id.get(target_model, DEFAULT_LLM_MIN_TIER))
            if self.model_loaded and self.loaded_tier >= target_tier:
                return False, f"loaded_tier_{self.loaded_tier}_ge_{target_tier}"
            groups = task.get("candidate_groups", [])
            if not isinstance(groups, list):
                return False, "no_candidate_groups"
            if not any(self._is_group_member(g) for g in groups if isinstance(g, dict)):
                return False, "not_group_member"

            # Block unrelated load_split_llm while in pair lock for different group
            in_pair_lock, lock_info = self._is_in_split_pair_loading_lock()
            if in_pair_lock and lock_info:
                active_group = lock_info.get("group_id")
                # Check if this task is for the same group
                task_groups = [
                    str(g.get("id", "")).strip()
                    for g in groups
                    if isinstance(g, dict) and self._is_group_member(g)
                ]
                if active_group and active_group not in task_groups:
                    return False, self._format_split_pair_lock_reason(lock_info)

            return True, ""

        if command == "unload_split_llm":
            # Unload tasks can be claimed even when wedged (for recovery)
            target_group = str(task.get("group_id", "")).strip()
            if not target_group:
                if self.runtime_placement != "split_gpu" and not self._is_runtime_wedged():
                    return False, "not_split_runtime"
                return True, ""
            if self.runtime_group_id != target_group and not self._is_runtime_wedged():
                return False, f"group_mismatch:{self.runtime_group_id}"
            return True, ""

        if command == "cleanup_split_runtime":
            target_workers = [str(w).strip() for w in task.get("target_workers", []) if str(w).strip()]
            if target_workers and self.name not in target_workers:
                return False, "not_target_worker"
            target_group = str(task.get("group_id", "")).strip()
            if target_group and self.runtime_group_id and target_group != self.runtime_group_id:
                return False, f"group_mismatch:{self.runtime_group_id}"
            return True, ""

        if command in ("reset_gpu_runtime", "reset_split_runtime"):
            # Thermal recovery reset tasks are targeted to specific workers
            # Brain uses target_worker; support target_gpu as fallback for compatibility
            target_worker = task.get("target_worker") or task.get("target_gpu")
            if target_worker and target_worker != self.name:
                return False, f"target_mismatch:{target_worker}"
            return True, ""

        # Unknown meta command: allow claim so it can fail explicitly.
        return True, ""

    def _resolve_env_manifest_path(self, task: Dict[str, Any]) -> Optional[Path]:
        direct = task.get("env_manifest_path")
        if isinstance(direct, str) and direct.strip():
            p = Path(direct.strip())
            if p.exists():
                return p

        batch_path = task.get("batch_path")
        if isinstance(batch_path, str) and batch_path.strip():
            p = Path(batch_path.strip()) / "env_manifest.json"
            if p.exists():
                return p

        batch_id = str(task.get("batch_id", "")).strip()
        if not batch_id:
            return None
        plans_dir = self.shared_path / "plans"
        for candidate in plans_dir.glob(f"*/history/{batch_id}/env_manifest.json"):
            if candidate.exists():
                return candidate
        return None

    def _check_task_env_requirements(self, task: Dict[str, Any]) -> tuple[bool, str]:
        batch_id = str(task.get("batch_id", "")).strip()
        if not batch_id:
            return True, ""

        manifest_path = self._resolve_env_manifest_path(task)
        if not manifest_path:
            return False, f"env manifest missing for batch {batch_id}"

        cache_key = f"{batch_id}:{manifest_path}"
        mtime = manifest_path.stat().st_mtime
        cached = self.env_check_cache.get(cache_key)
        if cached and cached.get("mtime") == mtime:
            return bool(cached.get("ok")), str(cached.get("reason", ""))

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as exc:
            reason = f"env manifest unreadable: {exc}"
            self.env_check_cache[cache_key] = {"mtime": mtime, "ok": False, "reason": reason}
            return False, reason

        required = manifest.get("required_modules", [])
        if not isinstance(required, list):
            required = []
        missing = [m for m in required if isinstance(m, str) and m and importlib.util.find_spec(m) is None]
        if missing:
            reason = f"missing modules: {', '.join(sorted(missing))}"
            self.env_check_cache[cache_key] = {"mtime": mtime, "ok": False, "reason": reason}
            return False, reason

        self.env_check_cache[cache_key] = {"mtime": mtime, "ok": True, "reason": ""}
        return True, ""

    def _begin_meta_task(self, task: Dict[str, Any]):
        self.active_meta_task = {
            "task_id": task.get("task_id", ""),
            "task_name": task.get("name", task.get("command", "meta")),
            "started_at": time.time(),
            "phase": "start",
        }
        self.last_meta_heartbeat = 0.0
        self._touch_meta_task(force=True)

    def _touch_meta_task(self, phase: Optional[str] = None, force: bool = False):
        if not self.active_meta_task:
            return
        if phase:
            self.active_meta_task["phase"] = phase
        now = time.time()
        if not force and (now - self.last_meta_heartbeat) < META_TASK_HEARTBEAT_INTERVAL:
            return
        task_id = str(self.active_meta_task.get("task_id", "")).strip()
        if task_id:
            self._write_task_heartbeat(
                task_id,
                f"{self.name}-meta",
                os.getpid(),
                peak_vram_mb=0,
                is_meta=True,
            )
        self._write_heartbeat()
        self.last_meta_heartbeat = now

    def _end_meta_task(self):
        if self.active_meta_task:
            task_id = str(self.active_meta_task.get("task_id", "")).strip()
            if task_id:
                self._remove_task_heartbeat(task_id)
        self.active_meta_task = None
        self.last_meta_heartbeat = 0.0

    def claim_tasks(self) -> List[Dict]:
        """
        Visit the board: claim as many tasks as fit within VRAM budget.
        Returns list of claimed tasks.

        During thermal pause, only emergency meta-tasks are allowed through:
        - reset_gpu_runtime: Full local reset for this GPU
        - reset_split_runtime: Reset split runtime (if this GPU is a member)
        These are issued by the brain's thermal recovery controller.
        """
        # Emergency meta-tasks allowed during thermal pause
        THERMAL_EMERGENCY_META_COMMANDS = {"reset_gpu_runtime", "reset_split_runtime"}

        if self.thermal_pause_active:
            remaining = max(0, int(self.thermal_pause_until - time.time()))

            # Check for emergency meta-tasks before blocking
            emergency_tasks = self._scan_emergency_meta_tasks(THERMAL_EMERGENCY_META_COMMANDS)
            if emergency_tasks:
                self.logger.warning(
                    f"THERMAL_EMERGENCY_CLAIM: found {len(emergency_tasks)} emergency meta-task(s) "
                    f"during thermal pause, allowing through"
                )
                return emergency_tasks

            self.logger.warning(
                f"TASKS_THERMAL_PAUSED: skip claiming new work for {remaining}s "
                f"reasons={self.thermal_pause_reasons}"
            )
            return []

        preferred = self._get_preferred_classes()

        def _task_priority_key(task: Dict[str, Any]) -> tuple:
            try:
                priority = int(task.get("priority", 5))
            except Exception:
                priority = 5
            created_at = str(task.get("created_at", "") or "")
            task_id = str(task.get("task_id", "") or "")
            # Higher priority first; older created_at first within same priority.
            return (-priority, created_at, task_id)

        # Categorize available tasks
        tasks_by_class = {tc: [] for tc in VALID_TASK_CLASSES}

        for task_file in sorted(self.queue_path.glob("*.json")):
            if str(task_file).endswith('.lock'):
                continue
            try:
                with open(task_file) as f:
                    task = json.load(f)

                if task.get("executor") == "brain":
                    continue
                if task.get("task_class") == "brain":
                    continue

                task_class = task.get("task_class", "cpu")
                if task_class in tasks_by_class:
                    tasks_by_class[task_class].append((task_file, task))
            except Exception:
                continue

        # Build ordered candidate list
        candidates = []
        for tc in preferred:
            ranked = sorted(tasks_by_class.get(tc, []), key=lambda x: _task_priority_key(x[1]))
            candidates.extend(task_file for task_file, _ in ranked)

        # Claim tasks until budget is full
        claimed = []
        budget = self._get_vram_budget()

        for task_file in candidates:
            lock_file = str(task_file) + ".lock"
            lock_path = Path(lock_file)
            if lock_path.exists() and not os.access(lock_path, os.W_OK):
                # Root-owned stale lock files can block non-root GPU workers.
                # Remove and recreate under current worker user.
                try:
                    lock_path.unlink()
                except Exception:
                    continue
            lock = FileLock(lock_file, timeout=1)

            try:
                with lock:
                    try:
                        os.chmod(lock_file, 0o666)
                    except Exception:
                        pass
                    if not task_file.exists():
                        continue

                    with open(task_file) as f:
                        task = json.load(f)

                    # Brain tasks are never claimable by GPU workers.
                    if task.get("executor") == "brain" or task.get("task_class") == "brain":
                        continue

                    task_class = task.get("task_class", "cpu")
                    vram_cost = self._get_task_vram_cost(task)

                    # Meta tasks: handle directly, don't spawn worker
                    if task_class == "meta":
                        can_claim, reject_reason = self._can_claim_meta_task(task)
                        if not can_claim:
                            self.logger.debug(
                                f"Skipping meta task {task['task_id'][:8]} ({task.get('command')}): "
                                f"runtime_state={self.runtime_state}, reason={reject_reason}"
                            )
                            continue
                        # Claim-time command-specific attestation for state-changing meta tasks
                        meta_cmd = str(task.get("command", ""))
                        if meta_cmd in ("load_llm", "unload_llm", "load_split_llm", "unload_split_llm"):
                            attestation = self._attest_meta_task_precondition(task)
                            if not attestation["ok"]:
                                if (
                                    attestation.get("allow_continue", False)
                                    and meta_cmd in ("load_llm", "load_split_llm")
                                ):
                                    self.logger.warning(
                                        f"RUNTIME_ATTESTATION_AUTO_RESET meta={meta_cmd} "
                                        f"task={task['task_id'][:8]} worker={self.name} "
                                        f"reason={attestation['mismatch_reason']}"
                                    )
                                    self._full_local_reset(attestation["mismatch_reason"])
                                    attestation = self._attest_meta_task_precondition(task)
                                if not attestation["ok"]:
                                    self.logger.warning(
                                        f"RUNTIME_ATTESTATION_FAIL meta={meta_cmd} task={task['task_id'][:8]} "
                                        f"worker={self.name} reason={attestation['mismatch_reason']}"
                                    )
                                    self._release_task_for_mismatch(
                                        task, task_file, attestation["mismatch_reason"]
                                    )
                                    self._full_local_reset(attestation["mismatch_reason"])
                                    return claimed
                        # Keep attempt bookkeeping consistent with worker-claimed tasks.
                        task["attempts"] = task.get("attempts", 0) + 1
                        task["workers_attempted"] = task.get("workers_attempted", [])
                        task["workers_attempted"].append(self.name)
                        task["last_attempt_at"] = datetime.now().isoformat()
                        if not task.get("first_attempted_at"):
                            task["first_attempted_at"] = task["last_attempt_at"]
                        task["status"] = "processing"
                        task["assigned_to"] = self.name
                        task["started_at"] = task["last_attempt_at"]
                        # Keep meta tasks visible in processing/ for observability.
                        new_path = self.processing_path / task_file.name
                        with open(new_path, 'w') as f:
                            json.dump(task, f, indent=2)
                        task_file.unlink()
                        claimed.append(task)
                        continue

                    # State machine guard: no work tasks during transitions or when wedged
                    can_work, work_reject = self._can_accept_work_task()
                    if not can_work:
                        self.logger.debug(
                            f"Skipping {task_class} task {task['task_id'][:8]}: "
                            f"runtime_state={self.runtime_state}, reason={work_reject}"
                        )
                        continue

                    # llm tasks require a hot/ready model on this GPU.
                    if task_class == "llm" and not self.model_loaded:
                        self.logger.debug(
                            f"Skipping llm task {task['task_id'][:8]}: model not loaded yet"
                        )
                        continue
                    if task_class == "llm":
                        try:
                            required_tier = int(
                                task.get("llm_min_tier", DEFAULT_LLM_MIN_TIER) or DEFAULT_LLM_MIN_TIER
                            )
                        except Exception:
                            required_tier = DEFAULT_LLM_MIN_TIER
                        if self.loaded_tier < required_tier:
                            self.logger.debug(
                                f"Skipping llm task {task['task_id'][:8]}: "
                                f"needs tier {required_tier}, this GPU has tier {self.loaded_tier}"
                            )
                            continue
                        required_placement = str(task.get("llm_placement", "")).strip()
                        if required_placement == "split_gpu" and self.runtime_placement != "split_gpu":
                            self.logger.debug(
                                f"Skipping llm task {task['task_id'][:8]}: requires split runtime"
                            )
                            continue
                        if (
                            required_placement == "split_gpu"
                            and self.runtime_placement == "split_gpu"
                            and not self.split_runtime_owner
                        ):
                            # Treat split pair as one logical worker: only launcher/owner claims.
                            self.logger.debug(
                                f"Skipping llm task {task['task_id'][:8]}: split runtime follower"
                            )
                            continue
                        llm_ok, llm_reason = self._llm_task_runtime_compatible(task)
                        if not llm_ok:
                            self.logger.debug(
                                f"Skipping llm task {task['task_id'][:8]}: runtime incompatible ({llm_reason})"
                            )
                            continue

                        # Claim-time runtime attestation: verify actual runtime state
                        # matches our cached state before committing to claim
                        attestation = self._attest_runtime_reality()
                        if not attestation["ok"]:
                            self.logger.warning(
                                f"RUNTIME_ATTESTATION_FAIL task={task['task_id'][:8]} "
                                f"worker={self.name} reason={attestation['mismatch_reason']}"
                            )

                            # Use phased recovery for attestation misses
                            miss_result = self._handle_attestation_miss(attestation["mismatch_reason"])

                            if miss_result["action"] == "hard_fail":
                                # Release task without retry penalty
                                self._release_task_for_mismatch(
                                    task, task_file, attestation["mismatch_reason"]
                                )
                                # Full local reset - worker is untrusted
                                self._full_local_reset(attestation["mismatch_reason"])
                                # Stop claiming entirely this cycle - we just reset
                                return claimed
                            else:
                                # Soft fail - just skip this task, don't reset
                                continue

                        # For split runtime: validate ready token before claiming
                        if self.runtime_placement == "split_gpu":
                            token_validation = self._validate_split_ready_token()
                            if not token_validation["ok"]:
                                self.logger.warning(
                                    f"SPLIT_TOKEN_VALIDATION_FAIL task={task['task_id'][:8]} "
                                    f"worker={self.name} reason={token_validation['reason']}"
                                )

                                # Use phased recovery
                                miss_result = self._handle_attestation_miss(
                                    f"split_token:{token_validation['reason']}"
                                )

                                if miss_result["action"] == "hard_fail":
                                    self._release_task_for_mismatch(
                                        task, task_file, f"split_token:{token_validation['reason']}"
                                    )
                                    self._full_local_reset(f"split_token:{token_validation['reason']}")
                                    return claimed
                                else:
                                    continue

                        # Clear miss count on successful validation
                        self._clear_attestation_miss_count()

                    env_ok, env_reason = self._check_task_env_requirements(task)
                    if not env_ok:
                        self.env_block_reason = f"{task.get('batch_id', '-')}: {env_reason}"
                        prior = task.get("env_blocked_reason")
                        if prior != env_reason:
                            task["env_blocked_reason"] = env_reason
                            task["env_blocked_at"] = datetime.now().isoformat()
                            with open(task_file, 'w') as f:
                                json.dump(task, f, indent=2)
                        self.logger.warning(
                            f"Skipping task {task.get('task_id', '')[:8]} ({task.get('name', '')}) "
                            f"- environment check failed: {env_reason}"
                        )
                        continue
                    self.env_block_reason = None

                    # Budget check
                    if vram_cost > budget:
                        self.logger.debug(
                            f"Skipping {task['task_id'][:8]}: needs {vram_cost}MB, "
                            f"only {budget}MB available"
                        )
                        continue

                    # Claim it
                    task["attempts"] = task.get("attempts", 0) + 1
                    task["workers_attempted"] = task.get("workers_attempted", [])
                    task["workers_attempted"].append(self.name)
                    task["last_attempt_at"] = datetime.now().isoformat()
                    if not task.get("first_attempted_at"):
                        task["first_attempted_at"] = task["last_attempt_at"]

                    task["status"] = "processing"
                    task["assigned_to"] = self.name
                    task["started_at"] = task["last_attempt_at"]

                    # Move to processing
                    new_path = self.processing_path / task_file.name
                    with open(new_path, 'w') as f:
                        json.dump(task, f, indent=2)
                    task_file.unlink()

                    # Update budget
                    budget -= vram_cost
                    self.claimed_vram += vram_cost

                    claimed.append(task)
                    self.logger.info(
                        f"Claimed {task_class} task: {task['task_id'][:8]}... "
                        f"({task.get('name', '')}) [{vram_cost}MB, {budget}MB remaining]"
                    )

            except Timeout:
                continue
            except Exception:
                continue

        if not claimed:
            task_summary = ", ".join(
                f"{k}:{len(v)}" for k, v in tasks_by_class.items() if len(v) > 0
            )
            if task_summary:
                self.logger.debug(f"No tasks claimed - Queue: [{task_summary}], budget: {budget}MB")
            else:
                self.logger.debug("Queue empty")

        return claimed

    def _handle_meta_task(self, task: Dict):
        """Handle a meta task directly (no worker needed)."""
        command = task.get("command", "")
        self.logger.info(f"Handling meta task: {command}")
        self._begin_meta_task(task)

        result = None
        try:
            result = self._execute_meta_command(task, command)
        except Exception as e:
            self.logger.error(f"Meta task {command} failed with exception: {e}", exc_info=True)
            result = self._build_meta_result(
                command,
                task,
                success=False,
                error=f"Meta task exception: {e}",
                error_type="meta_task_exception",
                error_code=type(e).__name__,
                diagnostic=f"exception={type(e).__name__}: {e}",
            )
        finally:
            self._end_meta_task()
            if result is not None:
                self.outbox.append(WorkerResult(
                    task_id=task["task_id"],
                    task=task,
                    result=result,
                    peak_vram_mb=0,
                ))

    def _execute_meta_command(self, task: Dict, command: str) -> Dict:
        """Execute the actual meta command logic. Returns result dict."""
        if command == "load_llm":
            task_id = task.get("task_id")
            target_model = str(task.get("target_model", "")).strip() or None
            if not target_model:
                return self._build_meta_result(
                    command,
                    task,
                    success=False,
                    output="Model failed to load",
                    error="load_llm missing target_model",
                    error_code="missing_target_model",
                    diagnostic="target_model= runtime_error_code=missing_target_model",
                )
            self._touch_meta_task(phase="load_llm")
            self._full_local_reset(
                f"pre_load_llm:{target_model or 'unknown'}",
                count_toward_circuit_breaker=False,
            )
            # llama backend: load_model handles container start internally
            self.load_model(model_id=target_model, task_id=task_id)
            load_success = bool(self.model_loaded)
            result = self._build_meta_result(
                command,
                task,
                success=load_success,
                output=f"Model {'loaded' if load_success else 'failed to load'}",
                error=(
                    f"Model failed to load: "
                    f"{self._safe_meta_detail(self.runtime_error_detail or self.runtime_error_code or 'unknown')}"
                ) if not load_success else "",
                error_code="" if load_success else str(self.runtime_error_code or "load_failed"),
                diagnostic=(
                    f"target_model={target_model or ''} "
                    f"runtime_error_code={self.runtime_error_code or ''} "
                    f"runtime_error_detail={self._safe_meta_detail(self.runtime_error_detail)}"
                ),
            )
        elif command == "unload_llm":
            task_id = task.get("task_id")
            self._touch_meta_task(phase="unload_llm")
            self.unload_model(task_id=task_id)
            unload_success = not self.model_loaded and self.runtime_state == RUNTIME_STATE_COLD
            result = self._build_meta_result(
                command,
                task,
                success=unload_success,
                output=f"Model {'unloaded' if unload_success else 'failed to unload'}",
                error=(
                    f"Model failed to unload: "
                    f"{self._safe_meta_detail(self.runtime_error_detail or self.runtime_error_code or self.loaded_model or 'unknown')}"
                ) if not unload_success else "",
                error_code=str(self.runtime_error_code or "unload_failed"),
                diagnostic=(
                    f"runtime_state={self.runtime_state} "
                    f"model_loaded={self.model_loaded} loaded_model={self.loaded_model or ''}"
                ),
            )
        elif command == "load_split_llm":
            target_model = str(task.get("target_model", "")).strip()
            groups = task.get("candidate_groups", [])
            if not target_model or not isinstance(groups, list):
                result = self._build_meta_result(
                    command,
                    task,
                    success=False,
                    error="load_split_llm missing target_model/candidate_groups",
                    error_type="meta_task_contract_error",
                    error_code="missing_split_inputs",
                )
            else:
                chosen = None
                for g in groups:
                    if isinstance(g, dict) and self._is_group_member(g):
                        chosen = g
                        break
                if not chosen:
                    result = self._build_meta_result(
                        command,
                        task,
                        success=False,
                        error="worker not eligible for any candidate split group",
                        error_code="split_group_not_eligible",
                    )
                else:
                    group_id = str(chosen.get("id", "")).strip()
                    members = [str(m).strip() for m in chosen.get("members", []) if str(m).strip()]
                    launcher = self.name
                    reservation_path = self._split_reservation_path(group_id)
                    lock = FileLock(str(reservation_path) + ".lock", timeout=2)
                    lock_wait_deadline = time.time() + 30
                    reservation_written = False
                    join_blocked_reason = ""
                    while time.time() < lock_wait_deadline:
                        self._touch_meta_task(phase="acquiring_split_reservation")
                        try:
                            with lock:
                                now_iso = datetime.now().isoformat()
                                # Create base reservation WITHOUT joined - authoritative join adds it
                                reservation = {
                                    "group_id": group_id,
                                    "target_model": target_model,
                                    "status": "waiting_partner",
                                    "members": members,
                                    "port": chosen.get("port"),
                                    "launcher": launcher,
                                    "joined": {},
                                    "member_clean": {},
                                    "created_at": now_iso,
                                    "updated_at": now_iso,
                                }
                                if reservation_path.exists():
                                    with open(reservation_path, "r", encoding="utf-8") as f:
                                        existing = json.load(f)
                                    existing_status = str(existing.get("status", "")).strip()
                                    existing_model = str(existing.get("target_model", "")).strip()
                                    terminal_statuses = {"failed", "expired", "unloaded"}
                                    if (
                                        existing_status == "ready"
                                        and existing_model == target_model
                                    ):
                                        reservation = existing
                                        reservation_written = True
                                        break
                                    elif (
                                        existing_model == target_model
                                        and existing_status not in terminal_statuses
                                    ):
                                        # Use existing reservation but DON'T directly write joined
                                        reservation = existing
                                    else:
                                        # Existing reservation is stale (terminal) or for a different model.
                                        # Replace it with a fresh reservation for this target model.
                                        pass  # Keep new reservation

                                # AUTHORITATIVE JOIN: Use atomic join that writes both
                                # joined AND member_clean together after verification
                                reservation, join_ok, join_reason = self._atomic_join_split_reservation(
                                    reservation, group_id
                                )
                                if not join_ok:
                                    join_blocked_reason = join_reason
                                    # Write reservation state even if join blocked (for observability)
                                    with open(reservation_path, "w", encoding="utf-8") as f:
                                        json.dump(reservation, f, indent=2)
                                    # Don't set reservation_written - we didn't successfully join
                                    break

                                with open(reservation_path, "w", encoding="utf-8") as f:
                                    json.dump(reservation, f, indent=2)
                                self._write_split_partner_nudge(group_id, members, launcher)
                            reservation_written = True
                            break
                        except Timeout:
                            self.logger.info(
                                f"Split reservation lock busy for {group_id}; retrying join for {target_model}"
                            )
                            self._service_split_reservations()
                            time.sleep(0.5)
                    if join_blocked_reason:
                        return self._build_meta_result(
                            command,
                            task,
                            success=False,
                            output=(
                                f"Split load failed: launcher join blocked for {group_id}: "
                                f"{join_blocked_reason}"
                            ),
                            error=(
                                f"Split load failed: launcher join blocked for {group_id}: "
                                f"{join_blocked_reason}"
                            ),
                            error_code="split_join_blocked",
                            diagnostic=f"group_id={group_id} join_blocked_reason={join_blocked_reason}",
                        )
                    if not reservation_written:
                        return self._build_meta_result(
                            command,
                            task,
                            success=False,
                            output=(
                                f"Split load failed: unable to acquire reservation lock for {group_id} "
                                "after 30s"
                            ),
                            error=(
                                f"Split load failed: unable to acquire reservation lock for {group_id} "
                                "after 30s"
                            ),
                            error_code="split_reservation_lock_timeout",
                            diagnostic=f"group_id={group_id} reservation_lock_timeout=30s",
                        )

                    split_profile = resolve_llama_runtime_profile(
                        getattr(self, "config", {}) or {},
                        model_id=target_model,
                        split=True,
                        override=chosen.get("llama_runtime") if isinstance(chosen, dict) else None,
                    )
                    try:
                        split_meta_timeout_seconds = int(
                            split_profile.get("meta_timeout_seconds", SPLIT_META_TIMEOUT_SECONDS)
                        )
                    except Exception:
                        split_meta_timeout_seconds = SPLIT_META_TIMEOUT_SECONDS
                    split_meta_timeout_seconds = max(30, split_meta_timeout_seconds)

                    deadline = time.time() + split_meta_timeout_seconds
                    success = False
                    failure_reason = ""
                    while time.time() < deadline:
                        self._touch_meta_task(phase="waiting_split_ready")
                        self._service_split_reservations()
                        try:
                            with open(reservation_path, "r", encoding="utf-8") as f:
                                reservation = json.load(f)
                        except Exception:
                            failure_reason = "reservation disappeared"
                            break
                        status = str(reservation.get("status", "")).strip()
                        if status == "ready":
                            success = True
                            break
                        if status in {"failed", "expired", "unloaded"}:
                            reservation_error = str(reservation.get("error", "")).strip()
                            if reservation_error:
                                failure_reason = f"reservation status={status}: {reservation_error}"
                            else:
                                failure_reason = f"reservation status={status}"
                            break
                        time.sleep(2)

                    if not success and not failure_reason:
                        failure_reason = "split load timed out waiting for partner readiness"
                        try:
                            lock = FileLock(str(reservation_path) + ".lock", timeout=2)
                            with lock:
                                if reservation_path.exists():
                                    with open(reservation_path, "r", encoding="utf-8") as f:
                                        reservation = json.load(f)
                                    reservation = self._set_reservation_status(
                                        reservation,
                                        "expired",
                                        reason="meta_wait_timeout",
                                    )
                                    with open(reservation_path, "w", encoding="utf-8") as f:
                                        json.dump(reservation, f, indent=2)
                        except Exception:
                            pass

                    cleanup_result = None
                    if not success and failure_reason == "reservation disappeared":
                        try:
                            split_port = int(chosen.get("port")) if chosen.get("port") else None
                        except Exception:
                            split_port = None
                        cleanup_result = self._reset_dead_split_runtime(
                            "reservation_disappeared",
                            group_id=group_id,
                            split_port=split_port,
                            task_id=task.get("task_id"),
                        )

                    result = self._build_meta_result(
                        command,
                        task,
                        success=success,
                        output=(
                            f"Split model loaded: {target_model} group={group_id}"
                            if success else f"Split load failed: {failure_reason}"
                        ),
                        error=f"Split load failed: {failure_reason}" if not success else "",
                        error_code="split_load_failed" if not success else "",
                        diagnostic=(
                            f"group_id={group_id} target_model={target_model} "
                            f"failure_reason={failure_reason or ''} "
                            "last_split_runtime_error="
                            f"{self._safe_meta_detail(getattr(self, 'last_split_runtime_error', ''))}"
                        ),
                    )
                    if cleanup_result:
                        result["cleanup"] = cleanup_result
        elif command == "unload_split_llm":
            task_id = task.get("task_id")
            group_id = str(task.get("group_id", "")).strip() or str(self.runtime_group_id or "")
            reservation_path = self._split_reservation_path(group_id) if group_id else None
            split_port = None
            target_model = ""

            # Transition to unloading state
            self._set_runtime_state(
                RUNTIME_STATE_UNLOADING,
                task_id=task_id,
                phase="starting_split_unload",
            )

            # Read reservation info and set transitional "unloading" status
            # (Don't set "unloaded" until postconditions pass)
            if reservation_path and reservation_path.exists():
                try:
                    lock = FileLock(str(reservation_path) + ".lock", timeout=5)
                    with lock:
                        with open(reservation_path, "r", encoding="utf-8") as f:
                            reservation = json.load(f)
                        try:
                            split_port = int(reservation.get("port"))
                        except Exception:
                            split_port = None
                        target_model = str(reservation.get("target_model", "")).strip()
                        # Set transitional "unloading" status
                        reservation = self._set_reservation_status(
                            reservation,
                            "unloading",
                            reason="meta_unload_split_llm_starting",
                        )
                        with open(reservation_path, "w", encoding="utf-8") as f:
                            json.dump(reservation, f, indent=2)
                except Exception:
                    pass
            elif self.runtime_placement == "split_gpu" and self.runtime_port:
                try:
                    split_port = int(self.runtime_port)
                except Exception:
                    split_port = None
                target_model = str(self.loaded_model or "").strip()

            if self.split_runtime_owner:
                self._stop_split_runtime()
            # Fail closed: any member processing unload should reclaim the shared split
            # listener if it is still up. This prevents "unloaded" reservations with
            # orphan split runtimes still holding VRAM.
            reclaimed_listeners = 0
            orphan_runners_killed = 0
            if split_port:
                if self._split_runtime_has_any_listener(split_port):
                    reclaimed_listeners += self._kill_local_listener_on_port(int(split_port))
                # Give the listener a brief moment to exit and drop children.
                wait_deadline = time.time() + 3
                while time.time() < wait_deadline and self._split_runtime_has_any_listener(split_port):
                    time.sleep(0.2)
                # If a listener or runner residue remains, do one orphan-runner sweep (scoped to this port).
                orphan_runners_killed = self._kill_orphan_runtime_processes(target_port=split_port)

            # Postcondition verification
            split_listener_still_up = bool(split_port and self._split_runtime_has_any_listener(split_port))
            split_model_still_loaded = bool(
                split_port and target_model and self._split_runtime_has_model_loaded(split_port, target_model)
            )
            unload_ok = not split_listener_still_up and not split_model_still_loaded

            # Update reservation status based on postcondition result
            if reservation_path and reservation_path.exists():
                try:
                    lock = FileLock(str(reservation_path) + ".lock", timeout=5)
                    with lock:
                        with open(reservation_path, "r", encoding="utf-8") as f:
                            reservation = json.load(f)
                        if unload_ok:
                            reservation = self._set_reservation_status(
                                reservation,
                                "unloaded",
                                reason="meta_unload_split_llm_complete",
                            )
                        else:
                            reservation = self._set_reservation_status(
                                reservation,
                                "failed",
                                reason="meta_unload_split_llm_postcondition_failed",
                            )
                            reservation["error"] = (
                                f"postcondition_failed: listener_up={split_listener_still_up} "
                                f"model_loaded={split_model_still_loaded}"
                            )
                        with open(reservation_path, "w", encoding="utf-8") as f:
                            json.dump(reservation, f, indent=2)
                except Exception:
                    pass

            if unload_ok:
                if self.runtime_placement == "split_gpu":
                    self._clear_split_runtime_loaded(task_id=task_id, reason="unload_complete")
            else:
                # Postconditions failed - mark as wedged
                self._mark_wedged(
                    error_code="split_unload_postcondition_failed",
                    error_detail=f"listener_up={split_listener_still_up} model_loaded={split_model_still_loaded}",
                    task_id=task_id,
                )
                # Still clear local state so we don't keep trying to use a broken runtime
                if self.runtime_placement == "split_gpu":
                    self.model_loaded = False
                    self.loaded_model = None
                    self.loaded_tier = 0
                    self.runtime_placement = "single_gpu"
                    self.runtime_group_id = None
                    self.runtime_port = self.port
                    self.runtime_api_base = f"http://localhost:{self.port}" if self.port else None
                    self.split_runtime_owner = False

            result = self._build_meta_result(
                command,
                task,
                success=unload_ok,
                output=(
                    f"Split runtime unloaded for group {group_id or '-'}"
                    if unload_ok else
                    f"Split unload incomplete for group {group_id or '-'}: "
                    f"listener_up={split_listener_still_up} model_still_loaded={split_model_still_loaded}"
                ),
                error=(
                    f"Split unload incomplete for group {group_id or '-'}: "
                    f"listener_up={split_listener_still_up} model_still_loaded={split_model_still_loaded}"
                ) if not unload_ok else "",
                error_code="split_unload_failed" if not unload_ok else "",
                diagnostic=(
                    f"group_id={group_id or ''} split_port={split_port} "
                    f"listener_up={split_listener_still_up} model_still_loaded={split_model_still_loaded}"
                ),
                split_port=split_port,
                reclaimed_listeners=reclaimed_listeners,
                orphan_runners_killed=orphan_runners_killed,
            )
        elif command == "cleanup_split_runtime":
            task_id = task.get("task_id")
            group_id = str(task.get("group_id", "")).strip() or str(self.runtime_group_id or "")
            cleanup_reason = str(task.get("cleanup_reason", "")).strip() or "brain_cleanup_command"
            expected_reservation_epoch = str(task.get("reservation_epoch", "")).strip() or None
            expected_generation = str(task.get("runtime_generation", "")).strip() or None
            actual_reservation_epoch = self._read_split_reservation_epoch(group_id)
            actual_generation = str(getattr(self, "split_runtime_generation", "") or "").strip() or None
            if expected_reservation_epoch and actual_reservation_epoch and expected_reservation_epoch != actual_reservation_epoch:
                result = self._build_meta_result(
                    command,
                    task,
                    success=True,
                    output=(
                        f"Skipped stale split cleanup for {group_id or '-'}: "
                        f"expected_reservation_epoch={expected_reservation_epoch} "
                        f"actual_reservation_epoch={actual_reservation_epoch}"
                    ),
                    stale_command=True,
                )
            elif expected_generation and actual_generation and expected_generation != actual_generation:
                result = self._build_meta_result(
                    command,
                    task,
                    success=True,
                    output=(
                        f"Skipped stale split cleanup for {group_id or '-'}: "
                        f"expected_generation={expected_generation} actual_generation={actual_generation}"
                    ),
                    stale_command=True,
                )
            else:
                self._touch_meta_task(phase="cleanup_split_runtime")
                split_port = task.get("split_port", self.runtime_port)
                try:
                    split_port_int = int(split_port) if split_port is not None else None
                except Exception:
                    split_port_int = None
                cleanup = self._coordinated_split_failure_cleanup(
                    group_id=group_id,
                    split_port=split_port_int,
                    reason=f"brain_command:{cleanup_reason}",
                    task_id=task_id,
                )
                result = self._build_meta_result(
                    command,
                    task,
                    success=True,
                    output=f"Split runtime cleanup executed for {group_id or '-'}",
                    cleanup=cleanup,
                )
        elif command == "reset_gpu_runtime":
            # Emergency thermal recovery: full local reset
            # Called by brain's thermal recovery controller during sustained overheat
            task_id = task.get("task_id")
            incident_id = task.get("incident_id", "unknown")
            reason = f"thermal_recovery:incident={incident_id}"

            self.logger.warning(
                f"THERMAL_RESET_GPU_RUNTIME: executing full local reset "
                f"incident_id={incident_id} task_id={task_id[:8] if task_id else 'none'}"
            )

            # Enter thermal reset state (blocks regular work claims)
            prev_state = self.runtime_state
            self.runtime_state = RUNTIME_STATE_RESETTING_THERMAL
            self._write_heartbeat_now()

            self._touch_meta_task(phase="reset_gpu_runtime")
            self._full_local_reset(reason, count_toward_circuit_breaker=False)

            # Verify we ended up in cold state
            if self.runtime_state != RUNTIME_STATE_COLD:
                self.logger.error(
                    f"THERMAL_RESET_GPU_RUNTIME: expected cold state after reset, "
                    f"got {self.runtime_state}"
                )
                self.runtime_state = RUNTIME_STATE_COLD  # Force cold state
            self._write_heartbeat_now()

            # Update incident tracking
            self.thermal_recovery_reset_count = getattr(self, 'thermal_recovery_reset_count', 0) + 1
            self.thermal_recovery_last_reset_at = time.time()

            result = self._build_meta_result(
                command,
                task,
                success=True,
                output=f"GPU runtime reset complete for thermal recovery incident={incident_id}",
                previous_state=prev_state,
                thermal_recovery_reset_count=self.thermal_recovery_reset_count,
            )

            # Log telemetry event for thermal reset result
            self.logger.info(
                f"THERMAL_TARGETED_RESET_RESULT: success={result['success']} "
                f"incident_id={incident_id} worker={self.name} "
                f"prev_state={prev_state} final_state={self.runtime_state} "
                f"reset_count={self.thermal_recovery_reset_count}"
            )
        elif command == "reset_split_runtime":
            # Emergency thermal recovery: reset split runtime for this GPU
            task_id = task.get("task_id")
            incident_id = task.get("incident_id", "unknown")
            group_id = task.get("group_id") or getattr(self, 'runtime_group_id', None)
            reason = f"thermal_recovery_split:incident={incident_id}"

            self.logger.warning(
                f"THERMAL_RESET_SPLIT_RUNTIME: resetting split runtime "
                f"incident_id={incident_id} group_id={group_id} task_id={task_id[:8] if task_id else 'none'}"
            )

            # Enter thermal reset state (blocks regular work claims)
            prev_state = self.runtime_state
            self.runtime_state = RUNTIME_STATE_RESETTING_THERMAL
            self._write_heartbeat_now()

            self._touch_meta_task(phase="reset_split_runtime")

            # Stop split runtime and reset state
            self._stop_split_runtime()
            self._full_local_reset(reason, count_toward_circuit_breaker=False)

            # Verify we ended up in cold state
            if self.runtime_state != RUNTIME_STATE_COLD:
                self.logger.error(
                    f"THERMAL_RESET_SPLIT_RUNTIME: expected cold state after reset, "
                    f"got {self.runtime_state}"
                )
                self.runtime_state = RUNTIME_STATE_COLD  # Force cold state
            self._write_heartbeat_now()

            # Update incident tracking
            self.thermal_recovery_reset_count = getattr(self, 'thermal_recovery_reset_count', 0) + 1
            self.thermal_recovery_last_reset_at = time.time()

            result = self._build_meta_result(
                command,
                task,
                success=True,
                output=f"Split runtime reset complete for thermal recovery incident={incident_id} group={group_id}",
                previous_state=prev_state,
                thermal_recovery_reset_count=self.thermal_recovery_reset_count,
            )

            # Log telemetry event for thermal reset result
            self.logger.info(
                f"THERMAL_TARGETED_RESET_RESULT: success={result['success']} "
                f"incident_id={incident_id} worker={self.name} group_id={group_id} "
                f"prev_state={prev_state} final_state={self.runtime_state} "
                f"reset_count={self.thermal_recovery_reset_count}"
            )
        else:
            return self._build_meta_result(
                command,
                task,
                success=False,
                error=f"Unknown meta command: {command}",
                error_type="meta_task_contract_error",
                error_code="unknown_meta_command",
            )

        return result
