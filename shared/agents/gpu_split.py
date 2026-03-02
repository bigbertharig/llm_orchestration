"""GPU agent split runtime handling mixin.

Extracted from gpu.py to isolate split GPU pair coordination, reservation
management, and shared runtime lifecycle.
"""

import json
import os
import re
import signal
import socket
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from filelock import FileLock, Timeout

from gpu_constants import (
    ATTESTATION_MISS_HARD_FAIL_THRESHOLD,
    ATTESTATION_MISS_SOFT_FAIL_THRESHOLD,
    AUTO_RECOVERY_CLEANUP_RETRY_DELAY_SECONDS,
    AUTO_RECOVERY_COLD_VRAM_THRESHOLD_MB,
    AUTO_RECOVERY_TIMEOUT_SECONDS,
    DEFAULT_LLM_MIN_TIER,
    READY_MIN_AGE_SECONDS,
    READY_STABLE_PROBE_COUNT,
    READY_STABLE_PROBE_INTERVAL_SECONDS,
    RUNTIME_STATE_COLD,
    RUNTIME_STATE_LOADING_SPLIT,
    RUNTIME_STATE_READY_SPLIT,
    RUNTIME_STATE_RECOVERING_SINGLE,
    RUNTIME_STATE_RECOVERING_SPLIT,
    RUNTIME_STATES_READY,
    SPLIT_LAUNCHER_HEARTBEAT_MAX_AGE_SECONDS,
    SPLIT_MEMBER_CLEAN_STALL_TIMEOUT_SECONDS,
    SPLIT_PORT_MODEL_MISS_CONSECUTIVE_COUNT,
    SPLIT_PORT_MODEL_MISS_WINDOW_SECONDS,
    SPLIT_QUARANTINE_COOLDOWN_SECONDS,
    SPLIT_QUARANTINE_FAILURE_COUNT,
    SPLIT_QUARANTINE_FAILURE_WINDOW_SECONDS,
    SPLIT_READY_GRACE_WINDOW_SECONDS,
    SPLIT_RESERVATION_LOADING_STATES,
)


class GPUSplitMixin:
    """Mixin providing split GPU runtime coordination methods."""

    def _split_reservation_path(self, group_id: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(group_id))
        return self.split_state_dir / f"{safe}.json"

    def _is_in_split_pair_loading_lock(self) -> tuple[bool, Optional[Dict[str, str]]]:
        """Check if this GPU is a member of any split reservation currently loading.

        Returns (is_locked, lock_info). When locked, both launcher and follower
        should reject work tasks to prevent interference with split load.

        lock_info contains: {"group_id": str, "status": str, "target_model": str}

        Locked statuses: waiting_partner, joining, loading, warming, ready_stabilizing
        """
        try:
            for res_path in self.split_state_dir.glob("*.json"):
                if res_path.name.endswith(".lock"):
                    continue
                if ".runtime_owner" in res_path.name:
                    continue
                try:
                    with open(res_path) as f:
                        reservation = json.load(f)
                except Exception:
                    continue

                status = str(reservation.get("status", "")).strip()
                if status not in SPLIT_RESERVATION_LOADING_STATES:
                    continue

                # Check if this worker is a member of this reservation
                members = reservation.get("members", [])
                if not isinstance(members, list):
                    continue
                member_names = {str(m).strip() for m in members if str(m).strip()}
                if self.name not in member_names:
                    continue

                # This worker is in a loading split reservation
                group_id = str(reservation.get("group_id", "")).strip()
                target_model = str(reservation.get("target_model", "")).strip()
                return True, {
                    "group_id": group_id,
                    "status": status,
                    "target_model": target_model,
                }

        except Exception as e:
            self.logger.debug(f"Error checking split pair lock: {e}")

        return False, None

    def _format_split_pair_lock_reason(self, lock_info: Optional[Dict[str, str]]) -> str:
        """Format lock_info dict into a human-readable reason string."""
        if not lock_info:
            return "split_pair_loading_lock"
        return (
            f"split_pair_loading_lock:group={lock_info.get('group_id', '-')},"
            f"status={lock_info.get('status', '-')},model={lock_info.get('target_model', '-')}"
        )

    def _verify_local_split_member_clean_precondition(
        self,
        group_id: str,
        allow_rejoin_group: bool = False,
    ) -> Dict[str, Any]:
        """Verify this worker is in a clean state to participate in split load.

        Hard fail if any of these are false:
        - runtime_state == cold (not ready_single/ready_split/loading/etc)
        - model_loaded == false
        - runtime_placement != split_gpu (unless allow_rejoin_group and same group)
        - local worker Ollama port has no loaded models (probe /api/ps)

        Returns structured result:
        - ok: bool
        - reason_code: str (empty if ok)
        - details: dict with runtime_state, model_loaded, local_port_models, etc
        """
        result: Dict[str, Any] = {
            "ok": True,
            "reason_code": "",
            "details": {
                "worker": self.name,
                "runtime_state": self.runtime_state,
                "model_loaded": self.model_loaded,
                "loaded_model": self.loaded_model,
                "runtime_placement": self.runtime_placement,
                "runtime_group_id": self.runtime_group_id,
                "local_port": self.port,
                "local_port_models": [],
                "checked_at": datetime.now().isoformat(),
            },
        }

        # Check 1: runtime_state must be cold
        if self.runtime_state != RUNTIME_STATE_COLD:
            # Allow ready states only if rejoin is permitted for the same group
            if allow_rejoin_group and self.runtime_state in RUNTIME_STATES_READY:
                if self.runtime_group_id == group_id:
                    # Same split group rejoin - ok
                    pass
                else:
                    result["ok"] = False
                    result["reason_code"] = f"runtime_not_cold:{self.runtime_state}"
                    return result
            else:
                result["ok"] = False
                result["reason_code"] = f"runtime_not_cold:{self.runtime_state}"
                return result

        # Check 2: model_loaded must be false
        if self.model_loaded:
            result["ok"] = False
            result["reason_code"] = f"model_loaded:{self.loaded_model}"
            return result

        # Check 3: runtime_placement must not be split_gpu (unless same group rejoin)
        if self.runtime_placement == "split_gpu":
            if allow_rejoin_group and self.runtime_group_id == group_id:
                pass  # Same group rejoin allowed
            else:
                result["ok"] = False
                result["reason_code"] = f"stale_split_placement:{self.runtime_group_id}"
                return result

        # Check 4: local worker Ollama port has no loaded models
        if self.port:
            try:
                r = requests.get(f"http://127.0.0.1:{self.port}/api/ps", timeout=2)
                if r.status_code == 200:
                    models = r.json().get("models", [])
                    loaded_names = [
                        str(m.get("name", "")).strip()
                        for m in models
                        if isinstance(m, dict) and str(m.get("name", "")).strip()
                    ]
                    result["details"]["local_port_models"] = loaded_names
                    if loaded_names:
                        result["ok"] = False
                        result["reason_code"] = f"local_port_has_models:{','.join(loaded_names)}"
                        return result
            except requests.exceptions.ConnectionError:
                # Port not listening - that's fine, no models loaded
                result["details"]["local_port_models"] = []
            except Exception as e:
                # Fail closed - if we can't probe, we can't guarantee member is clean
                result["ok"] = False
                result["reason_code"] = f"probe_error:{type(e).__name__}"
                self.logger.warning(f"Split precondition probe error (fail-closed): {e}")
                return result

        return result

    def _get_active_split_reservation_group_id(self) -> Optional[str]:
        """Get the group_id of any active split reservation this worker is part of.

        Active means status in {waiting_partner, joining, loading}.
        Returns None if not in any active reservation.
        """
        locked, lock_info = self._is_in_split_pair_loading_lock()
        if locked and lock_info:
            return lock_info.get("group_id")
        return None

    def _split_runtime_owner_meta_path_for_group(self, group_id: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(group_id))
        return self.split_state_dir / f"{safe}.runtime_owner.json"

    def _set_split_runtime_loaded(self, model_id: str, group: Dict[str, Any], as_owner: bool, task_id: Optional[str] = None):
        self.model_loaded = True
        self.loaded_model = model_id
        self.loaded_tier = int(self.model_tier_by_id.get(model_id, DEFAULT_LLM_MIN_TIER))
        self.runtime_placement = "split_gpu"
        self.runtime_group_id = str(group.get("id", "")).strip() or None
        try:
            self.runtime_port = int(group.get("port"))
        except Exception:
            self.runtime_port = None
        self.runtime_ollama_url = (
            f"http://localhost:{self.runtime_port}" if self.runtime_port else None
        )
        self.split_runtime_owner = bool(as_owner)
        self.split_runtime_invariant_failures = 0
        # Track when split became ready for grace window
        self.split_runtime_ready_at = time.time()
        # Reset port_model_missing tracking
        self.split_runtime_port_model_miss_timestamps = []
        # Transition to ready_split state
        self._set_runtime_state(
            RUNTIME_STATE_READY_SPLIT,
            task_id=task_id,
            phase="split_load_complete",
        )

    def _runtime_reset_port_and_state(
        self,
        ports_to_clean: list,
        reason: str,
        task_id: Optional[str] = None,
        *,
        stop_split_runtime: bool = False,
        stop_local_ollama: bool = False,
    ) -> Dict[str, Any]:
        """Shared core for runtime reset - used by both _full_local_reset and _coordinated_split_failure_cleanup.

        This is the authoritative path for cleaning up ports and resetting state.
        Callers handle their own specific logic (circuit breaker, member_reset ack, etc).

        Args:
            ports_to_clean: List of ports to reclaim (kill listeners, orphan runners)
            reason: Reason string for logging
            task_id: Optional task ID for state transition
            stop_split_runtime: If True, stop split runtime process if owner
            stop_local_ollama: If True, stop tracked local ollama process

        Returns:
            Dict with cleanup results for observability
        """
        result: Dict[str, Any] = {
            "worker": self.name,
            "reason": reason,
            "reset_started_at": datetime.now().isoformat(),
            "ports_cleaned": {},
        }

        # 1. Stop split runtime if requested and owner
        if stop_split_runtime and getattr(self, 'split_runtime_owner', False):
            try:
                self._stop_split_runtime()
                result["split_runtime_stopped"] = True
            except Exception as e:
                self.logger.warning(f"Split runtime stop during reset failed: {e}")
                result["split_runtime_stopped"] = False

        # 2. Stop local worker Ollama if requested
        if stop_local_ollama and getattr(self, 'ollama_process', None):
            try:
                self.stop_ollama()
                result["local_ollama_stopped"] = True
            except Exception as e:
                self.logger.warning(f"Ollama stop during reset failed: {e}")
                result["local_ollama_stopped"] = False

        # 3. Clean each port (kill listeners, orphan runners)
        local_port_reclaimed = False
        for port in ports_to_clean:
            if not port:
                continue
            port_result = {"reclaimed": False, "orphan_runners_killed": 0}
            try:
                # Check if anything is listening on port
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.5)
                try:
                    port_in_use = sock.connect_ex(("127.0.0.1", port)) == 0
                finally:
                    sock.close()

                if port_in_use:
                    self.logger.warning(
                        f"RUNTIME_RESET_RECLAIM_PORT port={port} reason={reason}"
                    )
                    self._kill_local_listener_on_port(port)
                    port_result["reclaimed"] = True
                    time.sleep(0.5)  # Brief wait for socket to close

                    # Track if we reclaimed the local worker port
                    if port == self.port:
                        local_port_reclaimed = True

                # Kill orphan runners scoped to this port
                killed = self._kill_orphan_ollama_runners(target_port=port)
                port_result["orphan_runners_killed"] = killed
            except Exception as e:
                self.logger.warning(f"Port cleanup during reset failed port={port}: {e}")
            result["ports_cleaned"][port] = port_result

        # 3b. Clear stale ollama_process handle if process is dead
        # This prevents lifecycle noise from a tracked Popen handle pointing to dead process
        # Check unconditionally - process may have died on its own without port reclaim
        if getattr(self, 'ollama_process', None):
            try:
                if self.ollama_process.poll() is not None:
                    # Process is dead - clear the handle
                    self.logger.debug(
                        f"RUNTIME_RESET_CLEAR_STALE_OLLAMA_HANDLE port={self.port} "
                        f"rc={self.ollama_process.returncode} local_port_reclaimed={local_port_reclaimed}"
                    )
                    self.ollama_process = None
                    result["stale_ollama_handle_cleared"] = True
            except Exception:
                # If we can't check, clear it anyway to be safe
                self.ollama_process = None
                result["stale_ollama_handle_cleared"] = True

        # 4. Clear runtime state fields
        self.model_loaded = False
        self.loaded_model = None
        self.loaded_tier = 0
        self.runtime_placement = "single_gpu"
        self.runtime_group_id = None
        self.runtime_port = self.port
        self.runtime_ollama_url = f"http://localhost:{self.port}" if self.port else None
        self.split_runtime_owner = False
        self.split_runtime_invariant_failures = 0
        self.split_runtime_ready_at = None
        self.split_runtime_port_model_miss_timestamps = []

        # 5. Transition to cold state
        self._set_runtime_state(
            RUNTIME_STATE_COLD,
            task_id=task_id,
            phase=f"runtime_reset:{reason}",
        )

        # 6. Write fresh heartbeat
        try:
            self._write_heartbeat()
        except Exception as e:
            self.logger.warning(f"Heartbeat write after reset failed: {e}")

        result["reset_completed_at"] = datetime.now().isoformat()
        return result

    def _clear_split_runtime_loaded(self, task_id: Optional[str] = None, reason: str = ""):
        self.model_loaded = False
        self.loaded_model = None
        self.loaded_tier = 0
        self.runtime_placement = "single_gpu"
        self.runtime_group_id = None
        self.runtime_port = self.port
        self.runtime_ollama_url = f"http://localhost:{self.port}" if self.port else None
        self.split_runtime_owner = False
        self.split_runtime_invariant_failures = 0
        # Transition to cold state
        self._set_runtime_state(
            RUNTIME_STATE_COLD,
            task_id=task_id,
            phase=f"split_cleared:{reason}" if reason else "split_cleared",
        )

    def _split_runtime_log_path(self, group_id: str) -> Path:
        """Get path for split runtime stdout/stderr capture."""
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(group_id))
        return self.split_state_dir / f"{safe}.runtime.log"

    def _coordinated_split_failure_cleanup(
        self,
        group_id: str,
        split_port: Optional[int],
        reason: str,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform full local reset on split failure, then clear split state.

        This is the authoritative cleanup path for split failures (warmup crash,
        invariant failure, etc). Uses shared _runtime_reset_port_and_state() core
        plus split-specific member_reset coordination.

        Returns cleanup result dict for logging/observability.
        """
        self.logger.warning(
            f"COORDINATED_SPLIT_FAILURE_CLEANUP group={group_id} "
            f"split_port={split_port} reason={reason}"
        )

        # Kill by owner metadata FIRST (before port-based cleanup)
        # This ensures we kill the correct process group, not just port listeners
        owner_kill = None
        if group_id:
            owner_kill = self._force_kill_split_runtime_owner(group_id)

        # Build list of ports to clean (split port + local port if different)
        ports_to_clean = []
        if split_port:
            ports_to_clean.append(split_port)
        if self.port and self.port != split_port:
            ports_to_clean.append(self.port)

        # Use shared reset core for port cleanup and state reset
        reset_result = self._runtime_reset_port_and_state(
            ports_to_clean=ports_to_clean,
            reason=f"split_failure:{reason}",
            task_id=task_id,
            stop_split_runtime=True,
            stop_local_ollama=False,  # Split failure doesn't need to stop local single-GPU ollama
        )

        # Build result dict for compatibility
        result: Dict[str, Any] = {
            "worker": self.name,
            "group_id": group_id,
            "reason": reason,
            "cleanup_started_at": reset_result.get("reset_started_at", ""),
            "cleanup_completed_at": reset_result.get("reset_completed_at", ""),
            "ports_cleaned": reset_result.get("ports_cleaned", {}),
            "owner_kill": owner_kill,  # Process group cleanup result
        }

        # Write member_reset ack to reservation for pair coordination
        if group_id:
            try:
                res_path = self._split_reservation_path(group_id)
                lock = FileLock(str(res_path) + ".lock", timeout=2)
                with lock:
                    if res_path.exists():
                        with open(res_path, "r", encoding="utf-8") as f:
                            reservation = json.load(f)
                        member_reset = reservation.get("member_reset", {})
                        if not isinstance(member_reset, dict):
                            member_reset = {}
                        member_reset[self.name] = {
                            "reset_at": datetime.now().isoformat(),
                            "reason": reason,
                            "result": result,
                        }
                        reservation["member_reset"] = member_reset
                        reservation["updated_at"] = datetime.now().isoformat()
                        with open(res_path, "w", encoding="utf-8") as f:
                            json.dump(reservation, f, indent=2)
            except Exception as e:
                self.logger.debug(f"Failed to write member_reset ack: {e}")

        # Post-cleanup verification gate: verify no orphan ollama runners remain
        if split_port:
            leaked_runners = self._verify_no_ollama_runners_on_port(split_port)
            if leaked_runners:
                self.logger.warning(
                    f"SPLIT_CLEANUP_LEAK group={group_id} port={split_port} "
                    f"leaked_runners={leaked_runners} (retrying forced kill)"
                )
                # Retry one forced kill cycle
                self._kill_local_listener_on_port(split_port)
                self._kill_orphan_ollama_runners(target_port=split_port)
                result["cleanup_leak_retry"] = True

        self.logger.info(
            f"COORDINATED_SPLIT_FAILURE_CLEANUP_DONE group={group_id} "
            f"ports_cleaned={list(result['ports_cleaned'].keys())} "
            f"owner_kill_success={result.get('owner_kill', {}).get('success', 'n/a')}"
        )

        return result

    def _verify_no_ollama_runners_on_port(self, port: int) -> list:
        """Verify no ollama runner processes are still using the given port.

        Returns list of (pid, cmd) tuples for any leaked runners found.
        """
        leaked = []
        if not port:
            return leaked

        try:
            # Find PIDs listening on the port
            port_pids = set()
            ss_result = subprocess.run(
                ["ss", "-ltnp"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            for line in ss_result.stdout.splitlines():
                if f":{port} " not in line:
                    continue
                for match in re.findall(r"pid=(\d+)", line):
                    port_pids.add(int(match))

            if not port_pids:
                return leaked

            # Check if any of these are ollama runners
            ps_result = subprocess.run(
                ["ps", "-eo", "pid,args"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            for line in ps_result.stdout.splitlines():
                parts = line.strip().split(None, 1)
                if len(parts) < 2:
                    continue
                try:
                    pid = int(parts[0])
                except Exception:
                    continue
                cmd = parts[1]
                if pid in port_pids and "ollama" in cmd.lower():
                    leaked.append((pid, cmd[:80]))

        except Exception as e:
            self.logger.debug(f"Error verifying ollama runners on port {port}: {e}")

        return leaked

    # =========================================================================
    # Auto-Recovery Workflow
    # =========================================================================

    def _record_split_failure_for_quarantine(self, group_id: str, reason: str):
        """Record a split failure for quarantine tracking.

        If failures exceed threshold within window, pair enters quarantine.
        """
        if not group_id:
            return

        now = time.time()
        failures = getattr(self, 'split_pair_failures', {})
        if group_id not in failures:
            failures[group_id] = []

        failures[group_id].append({
            "timestamp": now,
            "reason": reason,
        })

        # Prune old failures outside window
        cutoff = now - SPLIT_QUARANTINE_FAILURE_WINDOW_SECONDS
        failures[group_id] = [f for f in failures[group_id] if f["timestamp"] > cutoff]
        self.split_pair_failures = failures

        failure_count = len(failures[group_id])
        self.logger.debug(
            f"SPLIT_FAILURE_RECORDED group={group_id} reason={reason} "
            f"count={failure_count}/{SPLIT_QUARANTINE_FAILURE_COUNT}"
        )

        # Check if quarantine threshold reached
        if failure_count >= SPLIT_QUARANTINE_FAILURE_COUNT:
            self._enter_split_pair_quarantine(group_id, failures[group_id])

    def _enter_split_pair_quarantine(self, group_id: str, failures: list):
        """Enter quarantine for a split pair after repeated failures."""
        quarantine_until = time.time() + SPLIT_QUARANTINE_COOLDOWN_SECONDS
        quarantined = getattr(self, 'quarantined_split_pairs', {})
        quarantined[group_id] = {
            "entered_at": time.time(),
            "until": quarantine_until,
            "failure_count": len(failures),
            "recent_failures": failures[-3:],  # Keep last 3 for diagnostics
        }
        self.quarantined_split_pairs = quarantined

        self.logger.warning(
            f"AUTO_RECOVERY_QUARANTINE_ENTER group={group_id} "
            f"failures={len(failures)} cooldown_seconds={SPLIT_QUARANTINE_COOLDOWN_SECONDS}"
        )

    def _is_split_pair_quarantined(self, group_id: str) -> tuple[bool, Optional[Dict]]:
        """Check if a split pair is in quarantine.

        Returns (is_quarantined, quarantine_info). Also handles quarantine expiry.
        """
        if not group_id:
            return False, None

        quarantined = getattr(self, 'quarantined_split_pairs', {})
        if group_id not in quarantined:
            return False, None

        info = quarantined[group_id]
        now = time.time()

        # Check if quarantine has expired
        if now >= info.get("until", 0):
            # Quarantine expired - remove and allow
            del quarantined[group_id]
            self.quarantined_split_pairs = quarantined
            self.logger.info(
                f"AUTO_RECOVERY_QUARANTINE_EXIT group={group_id} "
                f"expired after {SPLIT_QUARANTINE_COOLDOWN_SECONDS}s"
            )
            return False, None

        remaining = int(info.get("until", 0) - now)
        return True, {"remaining_seconds": remaining, **info}

    def _trigger_auto_recovery_split(self, group_id: str, reason: str) -> Dict[str, Any]:
        """Stage A: Initiate auto-recovery for split runtime failure.

        Acquires reservation lock, marks reservation as failed_recoverable,
        and transitions local state to recovering_split.
        """
        result = {
            "stage": "A",
            "group_id": group_id,
            "reason": reason,
            "success": False,
            "error": None,
        }

        start_time = time.time()
        self.logger.info(
            f"AUTO_RECOVERY_TRIGGER worker={self.name} group={group_id} reason={reason}"
        )

        # Record failure for quarantine tracking
        self._record_split_failure_for_quarantine(group_id, reason)

        # Mark local state as recovering
        self._set_runtime_state(
            RUNTIME_STATE_RECOVERING_SPLIT,
            phase=f"auto_recovery_stage_a:{reason}",
        )
        self.recovery_started_at = start_time
        self.recovery_group_id = group_id
        self.recovery_reason = reason

        # Try to acquire reservation lock and mark as failed_recoverable
        if group_id:
            try:
                res_path = self._split_reservation_path(group_id)
                lock = FileLock(str(res_path) + ".lock", timeout=2)
                with lock:
                    if res_path.exists():
                        with open(res_path, "r", encoding="utf-8") as f:
                            reservation = json.load(f)

                        # Mark reservation as failed_recoverable
                        reservation["status"] = "failed_recoverable"
                        reservation["recovery_reason"] = reason
                        reservation["recovery_started_at"] = datetime.now().isoformat()
                        reservation["recovery_initiator"] = self.name
                        reservation["updated_at"] = datetime.now().isoformat()

                        with open(res_path, "w", encoding="utf-8") as f:
                            json.dump(reservation, f, indent=2)

                        result["reservation_updated"] = True
            except Exception as e:
                self.logger.warning(f"Failed to update reservation for recovery: {e}")
                result["reservation_error"] = str(e)

        self.logger.info(
            f"AUTO_RECOVERY_STAGE_A_LOCKED group={group_id} worker={self.name} "
            f"duration_ms={(time.time() - start_time) * 1000:.0f}"
        )

        result["success"] = True
        return result

    def _auto_recovery_stage_b_cleanup(self, group_id: str, split_port: Optional[int]) -> Dict[str, Any]:
        """Stage B: Coordinated hard cleanup for both members.

        Performs full port and state cleanup using shared reset core.
        """
        result = {
            "stage": "B",
            "group_id": group_id,
            "success": False,
        }

        start_time = time.time()
        self.logger.info(
            f"AUTO_RECOVERY_STAGE_B_MEMBER_RESET group={group_id} worker={self.name}"
        )

        # Use coordinated cleanup (which uses shared reset core)
        cleanup_result = self._coordinated_split_failure_cleanup(
            group_id=group_id,
            split_port=split_port,
            reason="auto_recovery",
            task_id=None,
        )

        result["cleanup"] = cleanup_result
        result["duration_ms"] = (time.time() - start_time) * 1000
        result["success"] = True

        self.logger.info(
            f"AUTO_RECOVERY_STAGE_B_MEMBER_RESET group={group_id} worker={self.name} "
            f"duration_ms={result['duration_ms']:.0f} result=success"
        )

        return result

    def _auto_recovery_stage_c_verify_cold(self, group_id: str, split_port: Optional[int]) -> Dict[str, Any]:
        """Stage C: Verified cold gate.

        Verify all conditions before returning to scheduler eligibility:
        - runtime_state == cold
        - model_loaded == false
        - split port has no model process/listener
        - VRAM below cold threshold
        """
        result = {
            "stage": "C",
            "group_id": group_id,
            "verified": False,
            "checks": {},
        }

        start_time = time.time()

        # Check 1: runtime_state == cold
        result["checks"]["runtime_state"] = {
            "expected": "cold",
            "actual": self.runtime_state,
            "passed": self.runtime_state == RUNTIME_STATE_COLD,
        }

        # Check 2: model_loaded == false
        result["checks"]["model_loaded"] = {
            "expected": False,
            "actual": self.model_loaded,
            "passed": not self.model_loaded,
        }

        # Check 3: split port has no listener
        if split_port:
            has_listener = self._split_runtime_has_any_listener(split_port)
            result["checks"]["split_port_clear"] = {
                "port": split_port,
                "has_listener": has_listener,
                "passed": not has_listener,
            }
        else:
            result["checks"]["split_port_clear"] = {"passed": True, "skipped": True}

        # Check 4: VRAM below cold threshold
        try:
            gpu_stats = self._read_gpu_stats()
            vram_used = gpu_stats.get("vram_used_mb", 0)
            result["checks"]["vram_cold"] = {
                "used_mb": vram_used,
                "threshold_mb": AUTO_RECOVERY_COLD_VRAM_THRESHOLD_MB,
                "passed": vram_used < AUTO_RECOVERY_COLD_VRAM_THRESHOLD_MB,
            }
        except Exception as e:
            result["checks"]["vram_cold"] = {"error": str(e), "passed": False}

        # Aggregate result
        all_passed = all(c.get("passed", False) for c in result["checks"].values())
        result["verified"] = all_passed
        result["duration_ms"] = (time.time() - start_time) * 1000

        self.logger.info(
            f"AUTO_RECOVERY_STAGE_C_VERIFIED_COLD group={group_id} worker={self.name} "
            f"verified={all_passed} checks={result['checks']}"
        )

        return result

    def _auto_recovery_stage_d_fallback_tasks(self, group_id: str) -> Dict[str, Any]:
        """Stage D: Recovery task fallback.

        Emit targeted meta tasks if direct cleanup failed to achieve verified cold.
        These are recovery-only tasks.
        """
        result = {
            "stage": "D",
            "group_id": group_id,
            "tasks_emitted": [],
        }

        self.logger.warning(
            f"AUTO_RECOVERY_STAGE_D_TARGETED_FALLBACK group={group_id} worker={self.name}"
        )

        # Get both pair members from reservation
        members = [self.name]  # Default to just self
        if group_id:
            try:
                res_path = self._split_reservation_path(group_id)
                if res_path.exists():
                    with open(res_path, "r") as f:
                        reservation = json.load(f)
                    members = [str(m).strip() for m in reservation.get("members", []) if str(m).strip()]
                    if not members:
                        members = [self.name]
            except Exception:
                pass

        # Build tasks for BOTH pair members (policy requirement)
        tasks_needed = [
            {"command": "unload_split_llm", "group_id": group_id},
        ]
        for member in members:
            tasks_needed.append({"command": "unload_llm", "candidate_workers": [member]})

        # Write recovery signal files for brain to pick up
        # (Brain will emit the actual meta tasks)
        recovery_signal = {
            "type": "split_recovery_fallback",
            "group_id": group_id,
            "worker": self.name,
            "members": members,
            "requested_at": datetime.now().isoformat(),
            "tasks_needed": tasks_needed,
        }

        try:
            signal_file = self.signals_path / f"{self.name}.recovery_fallback.json"
            with open(signal_file, "w") as f:
                json.dump(recovery_signal, f, indent=2)
            result["signal_written"] = True
            result["signal_file"] = str(signal_file)
            result["members"] = members
        except Exception as e:
            result["signal_error"] = str(e)

        return result

    def _run_auto_recovery_workflow(self, group_id: str, split_port: Optional[int], reason: str) -> Dict[str, Any]:
        """Execute full auto-recovery workflow (Stages A-D).

        Returns comprehensive result dict with all stage outcomes.
        """
        workflow_start = time.time()
        result = {
            "group_id": group_id,
            "worker": self.name,
            "reason": reason,
            "started_at": datetime.now().isoformat(),
            "stages": {},
            "final_state": None,
            "success": False,
        }

        def _ensure_exit_recovery():
            """Ensure we exit recovery state even on failure - prevents deadlock."""
            if self.runtime_state in {RUNTIME_STATE_RECOVERING_SPLIT, RUNTIME_STATE_RECOVERING_SINGLE}:
                self._set_runtime_state(
                    RUNTIME_STATE_COLD,
                    phase="auto_recovery_exit_cleanup",
                )
                self.recovery_started_at = None
                self.recovery_group_id = None
                self.recovery_reason = None
                try:
                    self._write_heartbeat()
                except Exception:
                    pass

        # Stage A: Lock + mark recovering
        result["stages"]["A"] = self._trigger_auto_recovery_split(group_id, reason)
        if not result["stages"]["A"].get("success"):
            result["error"] = "stage_a_failed"
            _ensure_exit_recovery()
            return result

        # Check timeout
        if (time.time() - workflow_start) > AUTO_RECOVERY_TIMEOUT_SECONDS:
            result["error"] = "timeout_before_stage_b"
            _ensure_exit_recovery()
            return result

        # Stage B: Hard cleanup
        result["stages"]["B"] = self._auto_recovery_stage_b_cleanup(group_id, split_port)
        if not result["stages"]["B"].get("success"):
            result["error"] = "stage_b_failed"
            _ensure_exit_recovery()
            return result

        # Check timeout
        if (time.time() - workflow_start) > AUTO_RECOVERY_TIMEOUT_SECONDS:
            result["error"] = "timeout_before_stage_c"
            _ensure_exit_recovery()
            return result

        # Stage C: Verify cold
        result["stages"]["C"] = self._auto_recovery_stage_c_verify_cold(group_id, split_port)

        if not result["stages"]["C"].get("verified"):
            # Retry cleanup once with backoff
            self.logger.info(
                f"AUTO_RECOVERY_RETRY_CLEANUP group={group_id} after {AUTO_RECOVERY_CLEANUP_RETRY_DELAY_SECONDS}s"
            )
            time.sleep(AUTO_RECOVERY_CLEANUP_RETRY_DELAY_SECONDS)

            # Re-run Stage B
            result["stages"]["B_retry"] = self._auto_recovery_stage_b_cleanup(group_id, split_port)

            # Re-run Stage C
            result["stages"]["C_retry"] = self._auto_recovery_stage_c_verify_cold(group_id, split_port)

            if not result["stages"]["C_retry"].get("verified"):
                # Stage D: Fallback to targeted meta tasks
                result["stages"]["D"] = self._auto_recovery_stage_d_fallback_tasks(group_id)
                result["final_state"] = "fallback_requested"

                # CRITICAL: Must exit recovery state to avoid deadlock
                # Transition to cold anyway - cleanup was attempted, brain fallback tasks will finish
                self._set_runtime_state(
                    RUNTIME_STATE_COLD,
                    phase="auto_recovery_fallback_cold",
                )
                self.recovery_started_at = None
                self.recovery_group_id = None
                self.recovery_reason = None
                self.wedged_since = None
                try:
                    self._write_heartbeat()
                except Exception:
                    pass

                self.logger.warning(
                    f"AUTO_RECOVERY_FALLBACK_COMPLETE group={group_id} worker={self.name} "
                    f"verified_cold=False - transitioned to cold, brain fallback tasks emitted"
                )
                result["success"] = False
                return result

        # Success - ensure we're in cold state
        if self.runtime_state != RUNTIME_STATE_COLD:
            self._set_runtime_state(
                RUNTIME_STATE_COLD,
                phase="auto_recovery_verified_cold",
            )

        result["final_state"] = "verified_cold"
        result["success"] = True
        result["duration_ms"] = (time.time() - workflow_start) * 1000

        # Clear recovery tracking
        self.recovery_started_at = None
        self.recovery_group_id = None
        self.recovery_reason = None
        self.wedged_since = None

        # Write fresh heartbeat
        try:
            self._write_heartbeat()
        except Exception:
            pass

        self.logger.info(
            f"AUTO_RECOVERY_COMPLETE group={group_id} worker={self.name} "
            f"duration_ms={result['duration_ms']:.0f} final_state=verified_cold"
        )

        return result

    def _run_single_runtime_recovery(self, reason: str) -> Dict[str, Any]:
        """Execute simplified recovery for single-GPU runtime wedge.

        Unlike split recovery, this doesn't need pair coordination.
        """
        workflow_start = time.time()
        result = {
            "worker": self.name,
            "reason": reason,
            "type": "single",
            "started_at": datetime.now().isoformat(),
            "success": False,
        }

        self.logger.info(
            f"AUTO_RECOVERY_TRIGGER worker={self.name} type=single reason={reason}"
        )

        # Mark as recovering single
        self._set_runtime_state(
            RUNTIME_STATE_RECOVERING_SINGLE,
            phase=f"auto_recovery_single:{reason}",
        )
        self.recovery_started_at = workflow_start

        try:
            # Simple cleanup: reset local port and state
            ports_to_clean = [self.port] if self.port else []
            reset_result = self._runtime_reset_port_and_state(
                ports_to_clean=ports_to_clean,
                reason=f"single_recovery:{reason}",
                task_id=None,
                stop_split_runtime=False,
                stop_local_ollama=True,  # For single-runtime, stop local ollama
            )
            result["reset"] = reset_result

            # Verify cold
            vram_ok = True
            try:
                gpu_stats = self._read_gpu_stats()
                vram_used = gpu_stats.get("vram_used_mb", 0)
                vram_ok = vram_used < AUTO_RECOVERY_COLD_VRAM_THRESHOLD_MB
                result["vram_used_mb"] = vram_used
            except Exception:
                pass

            state_ok = (self.runtime_state == RUNTIME_STATE_COLD and not self.model_loaded)
            result["verified_cold"] = state_ok and vram_ok

            if not result["verified_cold"]:
                # Force to cold anyway to prevent deadlock
                self._set_runtime_state(
                    RUNTIME_STATE_COLD,
                    phase="single_recovery_forced_cold",
                )

            result["success"] = True

        except Exception as e:
            result["error"] = str(e)
            # Force to cold to prevent deadlock
            self._set_runtime_state(
                RUNTIME_STATE_COLD,
                phase="single_recovery_error_cold",
            )

        # Clear recovery tracking
        self.recovery_started_at = None
        self.wedged_since = None

        # Write fresh heartbeat
        try:
            self._write_heartbeat()
        except Exception:
            pass

        result["duration_ms"] = (time.time() - workflow_start) * 1000

        self.logger.info(
            f"AUTO_RECOVERY_SINGLE_COMPLETE worker={self.name} "
            f"duration_ms={result['duration_ms']:.0f} verified_cold={result.get('verified_cold', False)}"
        )

        return result

    def _gather_split_crash_context(
        self,
        proc_returncode: Optional[int],
        port: int,
        group_id: str,
    ) -> Dict[str, Any]:
        """Gather diagnostic context when split runtime crashes.

        Best-effort attribution for rc=-9 (SIGKILL) and other crashes.
        """
        context: Dict[str, Any] = {
            "returncode": proc_returncode,
            "port": port,
            "group_id": group_id,
            "gathered_at": datetime.now().isoformat(),
            "signal_name": "",
            "possible_causes": [],
        }

        # Decode signal from negative return code
        if proc_returncode is not None and proc_returncode < 0:
            sig_num = -proc_returncode
            context["signal_name"] = {
                9: "SIGKILL",
                15: "SIGTERM",
                6: "SIGABRT",
                11: "SIGSEGV",
            }.get(sig_num, f"signal_{sig_num}")

            if sig_num == 9:  # SIGKILL
                context["possible_causes"].append("oom_killer")
                context["possible_causes"].append("thermal_critical")
                context["possible_causes"].append("external_reclaim")

        # Check for thermal pause (if we have access to thermal state)
        if getattr(self, 'thermal_pause_active', False):
            context["thermal_pause_active"] = True
            context["possible_causes"].insert(0, "thermal_critical_active")

        # Check if another reclaim happened on this port
        # (owner meta changed or disappeared)
        try:
            owner_path = self._split_runtime_owner_meta_path_for_group(group_id)
            if owner_path.exists():
                owner = self._read_json_file(owner_path)
                if owner:
                    owner_pid = owner.get("pid")
                    if owner_pid and not self._is_pid_alive(owner_pid):
                        context["owner_pid_dead"] = True
            else:
                context["owner_meta_missing"] = True
        except Exception:
            pass

        # Read last lines from runtime log if available
        try:
            log_path = self._split_runtime_log_path(group_id)
            if log_path.exists():
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                    context["last_log_lines"] = [l.rstrip() for l in lines[-20:]]
        except Exception:
            pass

        return context

    def _mark_split_reservation_runtime_invalid(self, group_id: str, reason: str):
        group_id = str(group_id or "").strip()
        if not group_id:
            return
        res_path = self._split_reservation_path(group_id)
        lock = FileLock(str(res_path) + ".lock", timeout=1)
        try:
            with lock:
                reservation = self._read_json_file(res_path)
                if not reservation:
                    return
                status = str(reservation.get("status", "")).strip()
                if status in {"unloaded", "failed", "expired"}:
                    return
                reservation = self._set_reservation_status(
                    reservation,
                    "failed",
                    reason=f"runtime_invalid_{reason}",
                )
                reservation["error"] = f"split runtime invalid: {reason}"
                with open(res_path, "w", encoding="utf-8") as f:
                    json.dump(reservation, f, indent=2)
        except Exception:
            return

    def _write_split_runtime_owner_meta(
        self,
        group: Dict[str, Any],
        model_id: str,
        port: int,
        proc_pid: int,
        *,
        pgid: Optional[int] = None,
        ready_token: Optional[str] = None,
    ):
        group_id = str(group.get("id", "")).strip()
        if not group_id:
            return
        payload = {
            "group_id": group_id,
            "launcher": self.name,
            "members": [str(m).strip() for m in group.get("members", []) if str(m).strip()],
            "model_id": str(model_id).strip(),
            "port": int(port),
            "pid": int(proc_pid),
            "pgid": pgid,  # Process group ID for cleanup
            "ready_token": ready_token,  # Token for readiness verification
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        path = self._split_runtime_owner_meta_path_for_group(group_id)
        self._write_json_atomic(path, payload)
        self.split_runtime_owner_meta_path = path

    def _touch_split_runtime_owner_meta(self):
        if not self.split_runtime_owner_meta_path:
            return
        meta = self._read_json_file(self.split_runtime_owner_meta_path)
        if not meta:
            return
        meta["updated_at"] = datetime.now().isoformat()
        self._write_json_atomic(self.split_runtime_owner_meta_path, meta)

    def _update_split_runtime_owner_meta_token(self, ready_token: str):
        """Update the owner meta file with the ready_token after stability gate passes."""
        if not self.split_runtime_owner_meta_path:
            return
        meta = self._read_json_file(self.split_runtime_owner_meta_path)
        if not meta:
            return
        meta["ready_token"] = ready_token
        meta["ready_token_issued_at"] = datetime.now().isoformat()
        meta["updated_at"] = datetime.now().isoformat()
        self._write_json_atomic(self.split_runtime_owner_meta_path, meta)

    def _verify_owner_meta_matches_reservation(self, group_id: str) -> tuple[bool, str]:
        """Compare launcher, model_id, ready_token between owner meta and reservation.

        Returns (matches, mismatch_reason). Mismatch = failed_precondition.
        """
        owner_path = self._split_runtime_owner_meta_path_for_group(group_id)
        owner_meta = self._read_json_file(owner_path)
        if not owner_meta:
            return False, "owner_meta_missing"

        res_path = self._split_reservation_path(group_id)
        reservation = self._read_json_file(res_path)
        if not reservation:
            return False, "reservation_missing"

        # Compare launcher
        owner_launcher = str(owner_meta.get("launcher", "")).strip()
        res_launcher = str(reservation.get("launcher", "")).strip()
        if owner_launcher != res_launcher:
            return False, f"launcher_mismatch:owner={owner_launcher},res={res_launcher}"

        # Compare model_id
        owner_model = str(owner_meta.get("model_id", "")).strip()
        res_model = str(reservation.get("target_model", "")).strip()
        if owner_model != res_model:
            return False, f"model_mismatch:owner={owner_model},res={res_model}"

        # Compare ready_token - fail closed on any mismatch or missing token
        res_token = reservation.get("ready_token")
        owner_token = owner_meta.get("ready_token")

        if res_token:
            # Reservation has token - owner MUST have matching token
            if not owner_token:
                return False, "owner_ready_token_missing"
            if owner_token != res_token:
                return False, f"token_mismatch:owner={owner_token[:8]},res={res_token[:8]}"

        # Optional stricter rule: stale owner token without reservation token = bad state
        if owner_token and not res_token:
            return False, "reservation_ready_token_missing"

        return True, ""

    def _clear_split_runtime_owner_meta(self):
        if self.split_runtime_owner_meta_path:
            try:
                self.split_runtime_owner_meta_path.unlink(missing_ok=True)
            except Exception:
                pass
        self.split_runtime_owner_meta_path = None

    def _split_runtime_owner_matches(
        self,
        port: Any,
        model_id: str,
        group_id: Optional[str] = None,
        require_alive_pid: bool = False,
    ) -> bool:
        for meta_path in self.split_state_dir.glob("*.runtime_owner.json"):
            meta = self._read_json_file(meta_path)
            if not meta:
                continue
            try:
                if int(meta.get("port")) != int(port):
                    continue
            except Exception:
                continue
            if str(meta.get("model_id", "")).strip() != str(model_id or "").strip():
                continue
            if group_id and str(meta.get("group_id", "")).strip() != str(group_id).strip():
                continue
            if require_alive_pid and not self._is_pid_alive(meta.get("pid")):
                continue
            return True
        return False

    def _set_reservation_status(
        self,
        reservation: Dict[str, Any],
        status: str,
        *,
        reason: str = "",
    ) -> Dict[str, Any]:
        old_status = str(reservation.get("status", "")).strip()
        reservation["status"] = status
        reservation["updated_at"] = datetime.now().isoformat()
        if status == "ready":
            reservation["ready_at"] = reservation.get("ready_at") or datetime.now().isoformat()
        if old_status != status or reason:
            self.logger.info(
                "SPLIT_RESERVATION_TRANSITION "
                f"group={reservation.get('group_id')} model={reservation.get('target_model')} "
                f"worker={self.name} launcher={reservation.get('launcher')} "
                f"{old_status or '-'}->{status} reason={reason or 'n/a'}"
            )
        return reservation

    def _issue_ready_token(self, reservation: Dict[str, Any]) -> Dict[str, Any]:
        """Issue a ready token for the reservation. Only callable by launcher.

        The ready token is a UUID that proves readiness was verified through the
        stability gate. Workers must check this token before claiming split tasks.
        Also updates owner meta file for consistency.
        """
        if str(reservation.get("launcher", "")).strip() != self.name:
            self.logger.warning(
                f"SPLIT_READY_TOKEN_ISSUE_BLOCKED: {self.name} is not launcher "
                f"(launcher={reservation.get('launcher')})"
            )
            return reservation

        token = str(uuid.uuid4())
        now_iso = datetime.now().isoformat()
        reservation["ready_token"] = token
        reservation["ready_token_issued_at"] = now_iso
        reservation["updated_at"] = now_iso

        # Also update owner meta file for consistency
        self._update_split_runtime_owner_meta_token(token)

        self.logger.info(
            f"SPLIT_READY_TOKEN_ISSUED group={reservation.get('group_id')} "
            f"token={token[:8]}... launcher={self.name}"
        )
        return reservation

    def _verify_ready_token_age(self, reservation: Dict[str, Any]) -> tuple[bool, str]:
        """Verify that ready_token exists and is old enough.

        Returns (valid, reason). Valid if token exists and age >= READY_MIN_AGE_SECONDS.
        """
        ready_token = reservation.get("ready_token")
        if not ready_token:
            return False, "no_ready_token"

        issued_at_str = reservation.get("ready_token_issued_at")
        if not issued_at_str:
            return False, "no_ready_token_issued_at"

        try:
            issued_at = datetime.fromisoformat(str(issued_at_str))
            age_seconds = (datetime.now() - issued_at).total_seconds()
            if age_seconds < READY_MIN_AGE_SECONDS:
                return False, f"token_too_young:{age_seconds:.1f}s<{READY_MIN_AGE_SECONDS}s"
            return True, ""
        except Exception as e:
            return False, f"token_parse_error:{e}"

    def _run_stability_gate(
        self,
        reservation: Dict[str, Any],
        reservation_path: Path,
        port: int,
        model_id: str,
    ) -> tuple[bool, Dict[str, Any]]:
        """Run stability gate: poll /api/ps N consecutive times to verify readiness.

        Only the launcher runs this gate. On success, issues ready_token and
        transitions to "ready" status.

        Returns (success, updated_reservation).
        """
        group_id = str(reservation.get("group_id", "")).strip()
        launcher = str(reservation.get("launcher", "")).strip()

        if launcher != self.name:
            self.logger.debug(
                f"SPLIT_STABILITY_GATE_SKIP: {self.name} is not launcher "
                f"(launcher={launcher})"
            )
            return False, reservation

        # Initialize or get probe count
        stable_probe_count = int(reservation.get("stable_probe_count", 0) or 0)

        self.logger.info(
            f"SPLIT_STABILITY_GATE_START group={group_id} port={port} "
            f"model={model_id} current_probes={stable_probe_count}/{READY_STABLE_PROBE_COUNT}"
        )

        # Probe the port for target model
        model_present = self._split_runtime_has_model_loaded(port=port, model_id=model_id)

        if model_present:
            stable_probe_count += 1
            reservation["stable_probe_count"] = stable_probe_count
            reservation["updated_at"] = datetime.now().isoformat()

            self.logger.info(
                f"SPLIT_STABILITY_GATE_PROBE_OK group={group_id} "
                f"probe={stable_probe_count}/{READY_STABLE_PROBE_COUNT}"
            )

            if stable_probe_count >= READY_STABLE_PROBE_COUNT:
                # Gate passed - issue token and transition to ready
                reservation = self._issue_ready_token(reservation)
                reservation = self._set_reservation_status(
                    reservation,
                    "ready",
                    reason="stability_gate_passed",
                )
                reservation["stable_probe_count"] = 0  # Reset for next time

                self.logger.info(
                    f"SPLIT_STABILITY_GATE_PASSED group={group_id} "
                    f"token={reservation.get('ready_token', '')[:8]}..."
                )
                return True, reservation
        else:
            # Model not present - reset probe count
            reservation["stable_probe_count"] = 0
            reservation["updated_at"] = datetime.now().isoformat()

            self.logger.warning(
                f"SPLIT_STABILITY_GATE_PROBE_FAIL group={group_id} "
                f"port={port} model={model_id} (resetting probe count)"
            )

        return False, reservation

    def _is_launcher_for_reservation(self, reservation: Dict[str, Any]) -> bool:
        """Check if this worker is the launcher for the given reservation."""
        launcher = str(reservation.get("launcher", "")).strip()
        return launcher == self.name

    def _close_split_runtime_log_file(self):
        """Close split runtime log file handle if open."""
        if getattr(self, '_split_runtime_log_file', None):
            try:
                self._split_runtime_log_file.close()
            except Exception:
                pass
            self._split_runtime_log_file = None

    def _start_split_runtime(self, group: Dict[str, Any], model_id: str, task_id: Optional[str] = None) -> tuple[bool, bool]:
        """Start split runtime and warm the model.

        Returns:
            (success, cleanup_done) - success indicates load worked,
            cleanup_done indicates whether coordinated cleanup was already performed
            (to avoid double cleanup in caller)
        """
        self.last_split_runtime_error = ""
        # Initialize variables at top to avoid UnboundLocalError in exception handlers
        group_id = str(group.get("id", "")).strip()
        port: Optional[int] = None
        log_file = None
        cleanup_done = False

        # Transition to loading_split state
        self._set_runtime_state(
            RUNTIME_STATE_LOADING_SPLIT,
            task_id=task_id,
            phase="starting_split_runtime",
        )
        try:
            members = [str(m).strip() for m in group.get("members", []) if str(m).strip()]
            if len(members) < 2:
                self.last_split_runtime_error = f"invalid split members: {members}"
                self.logger.error(f"split runtime start failed: {self.last_split_runtime_error}")
                return False, cleanup_done
            try:
                port = int(group.get("port"))
            except Exception:
                self.last_split_runtime_error = f"missing/invalid split port in group {group}"
                self.logger.error(f"split runtime start failed: {self.last_split_runtime_error}")
                return False, cleanup_done

            member_gpu_ids = []
            for member in members:
                cfg = self._get_gpu_config(member)
                member_gpu_ids.append(str(cfg["id"]))

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(member_gpu_ids)
            env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
            startup_started = time.time()

            # Avoid hijacking a stale runtime on the split port; that can make
            # readiness checks pass against the wrong process and wedge warmup.
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            try:
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    self.last_split_runtime_error = f"split port {port} already in use before launch"
                    self.logger.error(self.last_split_runtime_error)
                    return False, cleanup_done
            finally:
                try:
                    sock.close()
                except Exception:
                    pass

            # Capture stdout/stderr for crash diagnostics instead of discarding
            log_path = self._split_runtime_log_path(group_id) if group_id else None
            if log_path:
                try:
                    # Write launch context header
                    with open(log_path, "w", encoding="utf-8") as f:
                        f.write(f"# Split runtime log for group {group_id}\n")
                        f.write(f"# Launched at: {datetime.now().isoformat()}\n")
                        f.write(f"# CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES', '')}\n")
                        f.write(f"# OLLAMA_HOST: {env.get('OLLAMA_HOST', '')}\n")
                        f.write(f"# Model: {model_id}\n")
                        f.write(f"# Members: {members}\n")
                        f.write("#" + "=" * 60 + "\n")
                    log_file = open(log_path, "a", encoding="utf-8")
                except Exception as e:
                    self.logger.warning(f"Failed to open split runtime log: {e}")

            proc = subprocess.Popen(
                ["ollama", "serve"],
                env=env,
                stdout=log_file if log_file else subprocess.DEVNULL,
                stderr=subprocess.STDOUT if log_file else subprocess.DEVNULL,
                start_new_session=True,  # CRITICAL: Create new process group for cleanup
            )
            self.split_runtime_process = proc
            self._split_runtime_log_file = log_file  # Track for cleanup

            # Get process group ID for cleanup
            try:
                pgid = os.getpgid(proc.pid)
            except Exception:
                pgid = None

            self._write_split_runtime_owner_meta(
                group=group,
                model_id=model_id,
                port=port,
                proc_pid=proc.pid,
                pgid=pgid,
            )

            deadline = time.time() + 20
            ready = False
            while time.time() < deadline:
                self._touch_meta_task(phase="starting_split_runtime")
                self._touch_split_runtime_owner_meta()
                if proc.poll() is not None:
                    # Gather crash context before cleanup
                    crash_ctx = self._gather_split_crash_context(proc.returncode, port, group_id)
                    self.last_split_runtime_error = (
                        f"split runtime exited during startup on port {port} rc={proc.returncode} "
                        f"signal={crash_ctx.get('signal_name', '')} "
                        f"causes={crash_ctx.get('possible_causes', [])}"
                    )
                    self.logger.error(self.last_split_runtime_error)
                    # Close log file
                    self._close_split_runtime_log_file()
                    self.split_runtime_process = None
                    self._clear_split_runtime_owner_meta()
                    # Use coordinated cleanup instead of just state clear
                    self._coordinated_split_failure_cleanup(
                        group_id=group_id,
                        split_port=port,
                        reason=f"startup_crash_rc{proc.returncode}",
                        task_id=task_id,
                    )
                    return False, True  # cleanup_done=True
                try:
                    r = requests.get(f"http://127.0.0.1:{port}/api/tags", timeout=1.5)
                    if r.status_code == 200:
                        ready = True
                        break
                except Exception:
                    pass
                time.sleep(0.4)
            if not ready:
                self.last_split_runtime_error = f"split runtime failed to start on port {port}"
                self.logger.error(self.last_split_runtime_error)
                try:
                    proc.terminate()
                except Exception:
                    pass
                # Close log file
                self._close_split_runtime_log_file()
                self.split_runtime_process = None
                self._clear_split_runtime_owner_meta()
                # Use coordinated cleanup
                self._coordinated_split_failure_cleanup(
                    group_id=group_id,
                    split_port=port,
                    reason="startup_timeout",
                    task_id=task_id,
                )
                return False, True  # cleanup_done=True
            self.logger.info(
                f"SPLIT_RUNTIME_START_READY group={group.get('id')} port={port} "
                f"pid={proc.pid} startup_ms={int((time.time() - startup_started) * 1000)}"
            )

            # Warm/load the target model on shared runtime.
            def _warm_split_runtime():
                warmup_deadline = time.time() + 600
                last_warmup_error = ""
                warmup_started = time.time()
                attempts = 0
                while time.time() < warmup_deadline:
                    self._touch_meta_task(phase="warming_split_runtime")
                    self._touch_split_runtime_owner_meta()
                    if proc.poll() is not None:
                        raise RuntimeError(
                            f"split runtime exited during warmup on port {port} rc={proc.returncode}"
                        )
                    try:
                        attempts += 1
                        req_result: Dict[str, Any] = {"done": False, "response": None, "error": None}

                        def _warmup_request():
                            try:
                                req_result["response"] = requests.post(
                                    f"http://127.0.0.1:{port}/api/generate",
                                    json={
                                        "model": model_id,
                                        "prompt": "Hello",
                                        "stream": False,
                                        "keep_alive": self._effective_keep_alive(),
                                        "options": {
                                            "num_gpu": 999,
                                            "num_ctx": self.worker_num_ctx,
                                        }
                                    },
                                    # Split first-load warmup on GTX 1060 pairs can exceed 25s
                                    # even when healthy; premature client timeouts can retrigger
                                    # generate calls and wedge the runtime in practice.
                                    timeout=180
                                )
                            except Exception as exc:
                                req_result["error"] = exc
                            finally:
                                req_result["done"] = True

                        req_thread = threading.Thread(target=_warmup_request, daemon=True)
                        req_thread.start()
                        req_started = time.time()
                        last_wait_log_at = req_started
                        while not req_result["done"]:
                            self._touch_meta_task(phase="warming_split_runtime")
                            self._touch_split_runtime_owner_meta()
                            if proc.poll() is not None:
                                raise RuntimeError(
                                    f"split runtime exited during warmup on port {port} rc={proc.returncode}"
                                )
                            now_wait = time.time()
                            if (now_wait - last_wait_log_at) >= 20:
                                # Periodic progress log so long first loads do not look dead.
                                self.logger.info(
                                    f"SPLIT_RUNTIME_WARMUP_WAIT group={group.get('id')} port={port} "
                                    f"model={model_id} attempt={attempts} "
                                    f"elapsed_s={int(now_wait - req_started)}"
                                )
                                last_wait_log_at = now_wait
                            time.sleep(1)

                        if req_result["error"] is not None:
                            raise req_result["error"]
                        r = req_result["response"]
                        if r is None:
                            raise RuntimeError("split warmup request finished without response")
                        if r.status_code < 400:
                            self.logger.info(
                                f"SPLIT_RUNTIME_WARMUP_OK group={group.get('id')} port={port} "
                                f"model={model_id} attempts={attempts} "
                                f"warmup_ms={int((time.time() - warmup_started) * 1000)}"
                            )
                            return True
                        response_excerpt = (r.text or "").strip().replace("\n", " ")
                        if len(response_excerpt) > 300:
                            response_excerpt = response_excerpt[:300] + "..."
                        last_warmup_error = (
                            f"warmup HTTP {r.status_code} for model {model_id} on port {port}: {response_excerpt}"
                        )
                    except Exception as exc:
                        last_warmup_error = str(exc)
                    time.sleep(2)
                raise RuntimeError(
                    f"warmup timeout for model {model_id} on port {port}: {last_warmup_error or 'no response'}"
                )

            try:
                self._run_with_global_model_load_lock(
                    phase="split_model_load",
                    fn=_warm_split_runtime,
                    max_wait_seconds=900,
                )
                # Close log file on success (keep log file around for debugging)
                if self._split_runtime_log_file:
                    try:
                        self._split_runtime_log_file.write(
                            f"\n# Warmup completed successfully at {datetime.now().isoformat()}\n"
                        )
                    except Exception:
                        pass
                self._close_split_runtime_log_file()
                return True, False  # success, cleanup_done=False
            except Exception as exc:
                # Gather crash context if process died
                crash_ctx = {}
                if proc.poll() is not None:
                    crash_ctx = self._gather_split_crash_context(proc.returncode, port, group_id)
                    self.last_split_runtime_error = (
                        f"{exc} signal={crash_ctx.get('signal_name', '')} "
                        f"causes={crash_ctx.get('possible_causes', [])}"
                    )
                else:
                    self.last_split_runtime_error = str(exc)
                self.logger.error(f"split runtime start failed: {self.last_split_runtime_error}")
                try:
                    proc.terminate()
                except Exception:
                    pass
                # Close log file with failure note
                if self._split_runtime_log_file:
                    try:
                        self._split_runtime_log_file.write(
                            f"\n# Warmup FAILED at {datetime.now().isoformat()}: {exc}\n"
                        )
                    except Exception:
                        pass
                self._close_split_runtime_log_file()
                self.split_runtime_process = None
                self._clear_split_runtime_owner_meta()
                # Use coordinated cleanup instead of just returning
                self._coordinated_split_failure_cleanup(
                    group_id=group_id,
                    split_port=port,
                    reason=f"warmup_failed:{type(exc).__name__}",
                    task_id=task_id,
                )
                return False, True  # cleanup_done=True
        except Exception as exc:
            self.last_split_runtime_error = str(exc)
            self.logger.error(f"split runtime start failed: {self.last_split_runtime_error}")
            if self.split_runtime_process:
                try:
                    self.split_runtime_process.terminate()
                except Exception:
                    pass
                self.split_runtime_process = None
                self._clear_split_runtime_owner_meta()
            # Close log file if open
            if getattr(self, '_split_runtime_log_file', None):
                try:
                    self._split_runtime_log_file.write(
                        f"\n# FAILED at {datetime.now().isoformat()}: {exc}\n"
                    )
                except Exception:
                    pass
            self._close_split_runtime_log_file()
            # Use coordinated cleanup
            self._coordinated_split_failure_cleanup(
                group_id=group_id,
                split_port=port,
                reason=f"split_start_exception:{type(exc).__name__}",
                task_id=task_id,
            )
            return False, True  # cleanup_done=True

    def _stop_split_runtime(self):
        """Stop split runtime using owner metadata first for reliable cleanup.

        Owner metadata is the authoritative shutdown path because it persists
        even if the parent ollama process has already died. The in-memory Popen
        handle is only used as a secondary cleanup convenience.
        """
        # Step 1: Use owner metadata first (survives parent-process death)
        if self.split_runtime_owner_meta_path:
            try:
                owner_meta = self._read_json_file(self.split_runtime_owner_meta_path)
                group_id = str(owner_meta.get("group_id", "")).strip() if owner_meta else ""
                if group_id:
                    owner_kill_result = self._force_kill_split_runtime_owner(group_id)
                    self.logger.debug(
                        f"SPLIT_STOP_OWNER_META_KILL group={group_id} result={owner_kill_result}"
                    )
            except Exception as exc:
                self.logger.warning(f"SPLIT_STOP_OWNER_META_KILL_ERROR: {exc}")

        # Step 2: Fallback - use Popen handle as secondary cleanup path
        if self.split_runtime_process:
            # Try to get PGID for thorough cleanup
            pgid = None
            try:
                pgid = os.getpgid(self.split_runtime_process.pid)
            except Exception:
                pass

            if pgid:
                # Kill entire process group
                kill_result = self._kill_process_group(pgid)
                if not kill_result.get("success"):
                    # Fallback to direct process kill
                    try:
                        self.split_runtime_process.terminate()
                        self.split_runtime_process.wait(timeout=5)
                    except Exception:
                        try:
                            self.split_runtime_process.kill()
                        except Exception:
                            pass
            else:
                # No PGID available - use traditional method
                try:
                    self.split_runtime_process.terminate()
                    self.split_runtime_process.wait(timeout=5)
                except Exception:
                    try:
                        self.split_runtime_process.kill()
                    except Exception:
                        pass

            self.split_runtime_process = None

        # Step 3: Always clear state
        self._close_split_runtime_log_file()
        self._clear_split_runtime_owner_meta()
        self.split_runtime_owner = False

    def _atomic_join_split_reservation(
        self,
        reservation: Dict[str, Any],
        group_id: str,
        *,
        skip_precondition: bool = False,
    ) -> tuple[Dict[str, Any], bool, str]:
        """Authoritative single path for joining a split reservation.

        INVARIANT: A member is NOT "joined" unless member_clean[member] is also present.
        Both launcher and follower MUST use this function - no direct writes to joined.

        Args:
            reservation: The reservation dict to update (will be modified in place)
            group_id: The split group ID
            skip_precondition: If True, skip verification (for backfill of already-verified)

        Returns:
            (updated_reservation, success, reason_code)
            - success: True if joined + member_clean were both written
            - reason_code: Empty if success, else the precondition failure reason
        """
        members = [str(m).strip() for m in reservation.get("members", []) if str(m).strip()]
        if self.name not in members:
            return reservation, False, "not_member"

        joined = reservation.get("joined", {})
        if not isinstance(joined, dict):
            joined = {}
        member_clean = reservation.get("member_clean", {})
        if not isinstance(member_clean, dict):
            member_clean = {}

        # Already fully joined (both joined and member_clean present)
        if joined.get(self.name) and member_clean.get(self.name):
            return reservation, True, ""

        # Backfill case: joined exists but member_clean missing - re-verify
        if joined.get(self.name) and not member_clean.get(self.name):
            self.logger.info(
                f"SPLIT_MEMBER_CLEAN_BACKFILL_START group={group_id} member={self.name}"
            )
            precond = self._verify_local_split_member_clean_precondition(
                group_id=group_id,
                allow_rejoin_group=False,
            )
            if not precond["ok"]:
                # Backfill failed - remove stale join entry
                self.logger.warning(
                    f"SPLIT_MEMBER_CLEAN_BACKFILL_FAIL group={group_id} member={self.name} "
                    f"reason={precond['reason_code']} (removing stale join entry)"
                )
                del joined[self.name]
                reservation["joined"] = joined
                reservation["updated_at"] = datetime.now().isoformat()
                return reservation, False, f"backfill_failed:{precond['reason_code']}"
            # Backfill succeeded
            member_clean[self.name] = {
                "verified_at": datetime.now().isoformat(),
                "details": precond["details"],
            }
            reservation["member_clean"] = member_clean
            reservation["updated_at"] = datetime.now().isoformat()
            self.logger.info(
                f"SPLIT_MEMBER_CLEAN_BACKFILL_OK group={group_id} member={self.name}"
            )
            return reservation, True, ""

        # Fresh join: verify precondition first (unless skip_precondition)
        if not skip_precondition:
            precond = self._verify_local_split_member_clean_precondition(
                group_id=group_id,
                allow_rejoin_group=False,
            )
            if not precond["ok"]:
                self.logger.warning(
                    f"SPLIT_JOIN_BLOCKED group={group_id} member={self.name} "
                    f"reason={precond['reason_code']} "
                    f"runtime_state={precond['details'].get('runtime_state')} "
                    f"model_loaded={precond['details'].get('model_loaded')} "
                    f"local_port_models={precond['details'].get('local_port_models')}"
                )
                return reservation, False, precond["reason_code"]
            precond_details = precond["details"]
        else:
            precond_details = {"skipped": True}

        # ATOMIC: Write both joined AND member_clean together
        now_iso = datetime.now().isoformat()
        joined[self.name] = {
            "joined_at": now_iso,
        }
        member_clean[self.name] = {
            "verified_at": now_iso,
            "details": precond_details,
        }
        reservation["joined"] = joined
        reservation["member_clean"] = member_clean
        reservation["updated_at"] = now_iso

        self.logger.info(
            f"SPLIT_MEMBER_JOINED_ATOMIC group={group_id} member={self.name} "
            f"target_model={reservation.get('target_model')}"
        )
        return reservation, True, ""

    def _join_split_reservation(self, reservation: Dict[str, Any], reservation_path: Path) -> Dict[str, Any]:
        """Join a split reservation after verifying local clean-state precondition.

        Before joining, verifies:
        - runtime_state == cold
        - model_loaded == false
        - local Ollama port has no loaded models

        If precondition fails, does NOT join and logs explicit reason.
        The reservation will remain waiting for all members to join cleanly.

        NOTE: This is a wrapper for _atomic_join_split_reservation that handles
        the reservation file I/O. All join logic goes through the authoritative path.
        """
        members = [str(m).strip() for m in reservation.get("members", []) if str(m).strip()]
        if self.name not in members:
            return reservation

        group_id = str(reservation.get("group_id", "")).strip()

        # Use authoritative join path
        reservation, success, reason = self._atomic_join_split_reservation(
            reservation, group_id
        )

        # Write back to file if anything changed
        if success or reason:  # Either joined or failed with state change
            with open(reservation_path, "w", encoding="utf-8") as f:
                json.dump(reservation, f, indent=2)

        return reservation

    def _is_gpu_heartbeat_fresh(self, gpu_name: str, max_age_seconds: int = SPLIT_LAUNCHER_HEARTBEAT_MAX_AGE_SECONDS) -> bool:
        try:
            suffix = str(gpu_name).split("-")[-1]
            heartbeat_path = self.shared_path / "gpus" / f"gpu_{suffix}" / "heartbeat.json"
            if not heartbeat_path.exists():
                return False
            with open(heartbeat_path, "r", encoding="utf-8") as f:
                heartbeat = json.load(f)
            raw_last_updated = str(heartbeat.get("last_updated", "")).strip()
            if not raw_last_updated:
                return False
            last_updated = datetime.fromisoformat(raw_last_updated)
            return (datetime.now() - last_updated).total_seconds() <= float(max_age_seconds)
        except Exception:
            return False

    def _split_runtime_has_model_loaded(self, port: Any, model_id: str) -> bool:
        try:
            port_int = int(port)
        except Exception:
            return False
        if not model_id:
            return False
        try:
            r = requests.get(f"http://127.0.0.1:{port_int}/api/ps", timeout=2)
            if r.status_code != 200:
                return False
            models = r.json().get("models", [])
            loaded_names = [str(m.get("name", "")).strip() for m in models if isinstance(m, dict)]
            return model_id in loaded_names
        except Exception:
            return False

    def _split_runtime_has_expected_owner(self, port: Any, model_id: str, group_id: Optional[str] = None) -> bool:
        if not next(self.split_state_dir.glob("*.runtime_owner.json"), None):
            return True
        return self._split_runtime_owner_matches(
            port=port,
            model_id=model_id,
            group_id=group_id,
            require_alive_pid=False,
        )

    def _split_runtime_has_any_listener(self, port: Any) -> bool:
        try:
            port_int = int(port)
        except Exception:
            return False
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        try:
            return sock.connect_ex(("127.0.0.1", port_int)) == 0
        except Exception:
            return False
        finally:
            try:
                sock.close()
            except Exception:
                pass

    def _kill_local_listener_on_port(self, port: int) -> int:
        killed = 0
        try:
            result = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True, timeout=3)
            pids = set()
            for line in result.stdout.splitlines():
                if f":{port} " not in line:
                    continue
                for match in re.findall(r"pid=(\d+)", line):
                    pids.add(int(match))
            for pid in sorted(pids):
                try:
                    os.kill(pid, signal.SIGKILL)
                    killed += 1
                    self.logger.warning(f"SPLIT_ORPHAN_RECLAIM_KILL port={port} pid={pid}")
                except Exception as exc:
                    self.logger.warning(f"SPLIT_ORPHAN_RECLAIM_KILL_FAIL port={port} pid={pid}: {exc}")
        except Exception as exc:
            self.logger.warning(f"SPLIT_ORPHAN_RECLAIM_SCAN_FAIL port={port}: {exc}")
        return killed

    def _kill_orphan_ollama_runners(self, target_port: Optional[int] = None) -> int:
        """Kill orphan ollama runner processes (PPID 1) for a specific port only.

        Args:
            target_port: Only kill runners associated with this port. If None,
                         no runners are killed (safe default).

        Scoped cleanup prevents accidentally killing healthy runners for other
        split groups or single-GPU runtimes.
        """
        if target_port is None:
            return 0

        killed = 0
        try:
            # First, find PIDs listening on the target port
            port_pids = set()
            ss_result = subprocess.run(
                ["ss", "-ltnp"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            for line in ss_result.stdout.splitlines():
                if f":{target_port} " not in line:
                    continue
                for match in re.findall(r"pid=(\d+)", line):
                    port_pids.add(int(match))

            if not port_pids:
                return 0

            # Find orphan runners (PPID 1) that have the target port open
            # (inherited socket from their now-dead parent process)
            result = subprocess.run(
                ["ps", "-eo", "pid,ppid,args"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            for line in result.stdout.splitlines():
                raw = line.strip()
                if not raw:
                    continue
                parts = raw.split(None, 2)
                if len(parts) < 3:
                    continue
                try:
                    pid = int(parts[0])
                    ppid = int(parts[1])
                except Exception:
                    continue
                cmd = parts[2]
                if ppid != 1:
                    continue
                if "ollama runner" not in cmd:
                    continue
                # Only kill if this orphan runner has the target port open
                # (inherited socket from its now-dead parent)
                if pid in port_pids:
                    try:
                        os.kill(pid, signal.SIGKILL)
                        killed += 1
                        self.logger.warning(
                            f"SPLIT_PAIR_PREP_KILL_ORPHAN_RUNNER port={target_port} pid={pid}"
                        )
                    except Exception as exc:
                        self.logger.warning(
                            f"SPLIT_PAIR_PREP_KILL_ORPHAN_RUNNER_FAIL port={target_port} pid={pid}: {exc}"
                        )
        except Exception as exc:
            self.logger.warning(f"SPLIT_PAIR_PREP_SCAN_FAIL port={target_port}: {exc}")
        return killed

    def _kill_process_group(self, pgid: int) -> Dict[str, Any]:
        """Kill entire process group. SIGTERM, wait, then SIGKILL if needed.

        Returns result dict with outcome details.
        """
        result = {
            "pgid": pgid,
            "sigterm_sent": False,
            "sigkill_sent": False,
            "success": False,
        }

        if not pgid:
            result["error"] = "no_pgid"
            return result

        try:
            # First try SIGTERM to allow graceful shutdown
            os.killpg(pgid, signal.SIGTERM)
            result["sigterm_sent"] = True
            self.logger.info(f"SPLIT_KILL_PROCESS_GROUP_SIGTERM pgid={pgid}")

            # Wait up to 3 seconds for graceful termination
            deadline = time.time() + 3
            group_alive = True
            while time.time() < deadline:
                try:
                    # Check if process group leader is still alive
                    os.kill(pgid, 0)
                    time.sleep(0.3)
                except OSError:
                    # Process doesn't exist anymore
                    group_alive = False
                    break

            if group_alive:
                # Still alive - send SIGKILL
                try:
                    os.killpg(pgid, signal.SIGKILL)
                    result["sigkill_sent"] = True
                    self.logger.warning(f"SPLIT_KILL_PROCESS_GROUP_SIGKILL pgid={pgid}")
                except OSError as e:
                    # May have died between check and kill
                    if e.errno != 3:  # ESRCH (No such process)
                        result["sigkill_error"] = str(e)

            result["success"] = True

        except OSError as e:
            if e.errno == 3:  # ESRCH (No such process)
                result["success"] = True  # Already dead
                result["already_dead"] = True
            else:
                result["error"] = str(e)
                self.logger.warning(f"SPLIT_KILL_PROCESS_GROUP_FAIL pgid={pgid}: {e}")

        return result

    def _kill_pid(self, pid: int) -> bool:
        """Kill single process by PID. Returns True if killed or already dead."""
        if not pid:
            return False
        try:
            os.kill(pid, signal.SIGKILL)
            self.logger.info(f"SPLIT_KILL_PID pid={pid}")
            return True
        except OSError as e:
            if e.errno == 3:  # ESRCH (No such process)
                return True  # Already dead
            self.logger.warning(f"SPLIT_KILL_PID_FAIL pid={pid}: {e}")
            return False

    def _is_pid_alive(self, pid: Any) -> bool:
        """Check if a process is still alive."""
        if not pid:
            return False
        try:
            pid_int = int(pid)
            os.kill(pid_int, 0)
            return True
        except (OSError, ValueError, TypeError):
            return False

    def _force_kill_split_runtime_owner(self, group_id: str) -> Dict[str, Any]:
        """Force kill split runtime using owner metadata.

        Steps:
        1. Read pair_X_Y.runtime_owner.json
        2. If pgid exists and alive: os.killpg(pgid, SIGTERM), wait, SIGKILL if needed
        3. If pid still alive, kill directly
        4. Log results

        Returns result dict with all cleanup outcomes.
        """
        result = {
            "group_id": group_id,
            "owner_meta_found": False,
            "pgid_kill": None,
            "pid_kill": None,
            "success": False,
        }

        if not group_id:
            result["error"] = "no_group_id"
            return result

        # Read owner metadata
        owner_path = self._split_runtime_owner_meta_path_for_group(group_id)
        owner_meta = self._read_json_file(owner_path)

        if not owner_meta:
            result["error"] = "owner_meta_missing"
            return result

        result["owner_meta_found"] = True
        pid = owner_meta.get("pid")
        pgid = owner_meta.get("pgid")

        # Try process group kill first (more thorough)
        if pgid and self._is_pid_alive(pgid):
            result["pgid_kill"] = self._kill_process_group(int(pgid))
        elif pgid:
            result["pgid_kill"] = {"pgid": pgid, "already_dead": True, "success": True}

        # Fallback: kill PID directly if still alive
        if pid and self._is_pid_alive(pid):
            result["pid_kill"] = {"pid": pid, "killed": self._kill_pid(int(pid))}
        elif pid:
            result["pid_kill"] = {"pid": pid, "already_dead": True}

        # Determine overall success
        pgid_ok = result.get("pgid_kill", {}).get("success", True)
        pid_ok = result.get("pid_kill", {}).get("killed", True) or result.get("pid_kill", {}).get("already_dead", True)
        result["success"] = pgid_ok and pid_ok

        self.logger.info(
            f"SPLIT_FORCE_KILL_OWNER group={group_id} "
            f"pgid={pgid} pid={pid} success={result['success']}"
        )

        return result

    def _prepare_local_for_split_pairing(
        self,
        group_id: str,
        port: Any,
        is_launcher: bool = False,
        status: str = "",
    ) -> Dict[str, Any]:
        """Best-effort local cleanup before participating in a split load.

        Port reclamation is restricted to prevent race conditions:
        - Only reclaim port if we are the launcher, OR
        - Only reclaim port if status is not "loading" (runtime not started yet)

        This prevents a late-joining partner from killing a healthy split runtime
        that the launcher already started.
        """
        # Parse port first so we can scope cleanup
        try:
            port_int = int(port)
        except Exception:
            port_int = 0

        result: Dict[str, Any] = {
            "prepared_at": datetime.now().isoformat(),
            "orphan_runners_killed": 0,
            "split_port_reclaimed": False,
            "worker": self.name,
        }

        # Only reclaim the shared split port if:
        # 1. We are the launcher (we own the right to start the runtime), OR
        # 2. Status is not "loading" (runtime hasn't been started yet)
        # This prevents a partner from killing a launcher's live split runtime.
        can_reclaim_port = is_launcher or (status not in {"loading"})

        # Only kill orphan runners when we have permission to reclaim the port.
        # This prevents a non-launcher partner from killing a launcher's healthy runtime.
        if can_reclaim_port:
            result["orphan_runners_killed"] = self._kill_orphan_ollama_runners(
                target_port=port_int if port_int > 0 else None
            )

        if port_int > 0 and can_reclaim_port and self._split_runtime_has_any_listener(port_int):
            self.logger.warning(
                f"SPLIT_PAIR_PREP_RECLAIM_PORT group={group_id or '-'} port={port_int} "
                f"is_launcher={is_launcher} status={status}"
            )
            self._kill_local_listener_on_port(port_int)
            result["split_port_reclaimed"] = True
        elif port_int > 0 and not can_reclaim_port and self._split_runtime_has_any_listener(port_int):
            self.logger.info(
                f"SPLIT_PAIR_PREP_SKIP_RECLAIM group={group_id or '-'} port={port_int} "
                f"is_launcher={is_launcher} status={status} (launcher owns runtime)"
            )

        self.logger.info(
            "SPLIT_PAIR_PREP_DONE "
            f"group={group_id or '-'} port={port_int or '-'} "
            f"orphan_runners_killed={result['orphan_runners_killed']} "
            f"split_port_reclaimed={result['split_port_reclaimed']}"
        )
        return result

    def _reclaim_orphan_split_ports_on_startup(self):
        seen_ports = set()
        for meta in self.model_meta_by_id.values():
            for group in meta.get("split_groups", []) or []:
                members = [str(m).strip() for m in group.get("members", []) if str(m).strip()]
                if self.name not in members:
                    continue
                try:
                    port = int(group.get("port"))
                except Exception:
                    continue
                if port in seen_ports:
                    continue
                seen_ports.add(port)
                group_id = str(group.get("id", "")).strip()
                reservation = self._read_json_file(self._split_reservation_path(group_id)) if group_id else None
                status = str((reservation or {}).get("status", "")).strip()
                # Don't reclaim if reservation is in any active state (including new stabilization states)
                active_states = SPLIT_RESERVATION_LOADING_STATES | {"ready"}
                if status in active_states:
                    continue
                if not self._split_runtime_has_any_listener(port):
                    continue
                self.logger.warning(
                    f"SPLIT_ORPHAN_RECLAIM_STARTUP port={port} group={group_id or '-'} "
                    f"reservation_status={status or 'missing'}"
                )
                self._kill_local_listener_on_port(port)

    def _check_split_runtime_invariants(self):
        """Check split runtime health with stabilization grace and transient tolerance.

        Invariant check policy:
        - reservation_* issues: require 3 consecutive failures (may be transient)
        - owner_meta_mismatch: immediate clear (meta mismatch is authoritative)
        - port_model_missing: grace window + consecutive miss threshold
          - Skip entirely during SPLIT_READY_GRACE_WINDOW_SECONDS after ready
          - If port listener is down: clear faster (port truly dead, 2 misses)
          - If port up but model missing: require N consecutive within M seconds
        """
        if self.runtime_placement != "split_gpu":
            self.split_runtime_invariant_failures = 0
            self.split_runtime_port_model_miss_timestamps = []
            return
        group_id = str(self.runtime_group_id or "").strip()
        model_id = str(self.loaded_model or "").strip()
        port = self.runtime_port

        # Never run split invariant clear while work is active.
        # A member can legitimately appear transiently inconsistent while generating.
        if self.active_meta_task or self.active_workers:
            self.split_runtime_invariant_failures = 0
            self.split_runtime_port_model_miss_timestamps = []
            return
        if group_id and self._split_group_has_any_active_work(group_id):
            self.split_runtime_invariant_failures = 0
            self.split_runtime_port_model_miss_timestamps = []
            return

        if not group_id or not model_id or not port:
            self.split_runtime_invariant_failures += 1
            return

        now = time.time()
        ready_at = getattr(self, 'split_runtime_ready_at', None)
        in_grace_window = ready_at and (now - ready_at) < SPLIT_READY_GRACE_WINDOW_SECONDS

        reason = ""
        reservation = self._read_json_file(self._split_reservation_path(group_id))
        if not reservation:
            reason = "reservation_missing"
        else:
            status = str(reservation.get("status", "")).strip()
            target_model = str(reservation.get("target_model", "")).strip()
            if status != "ready":
                reason = f"reservation_status_{status or 'none'}"
            elif target_model != model_id:
                reason = f"reservation_model_{target_model or 'none'}"

        # owner_meta_mismatch: always check, immediate clear
        if not reason and not self._split_runtime_has_expected_owner(port, model_id, group_id):
            reason = "owner_meta_mismatch"

        # port_model_missing: grace window + consecutive threshold
        port_has_listener = self._split_runtime_has_any_listener(port)
        model_loaded_on_port = self._split_runtime_has_model_loaded(port, model_id)

        if not reason and not model_loaded_on_port:
            if in_grace_window:
                # Within grace window - warn but don't act
                self.logger.debug(
                    f"SPLIT_INVARIANT_GRACE group={group_id} model={model_id} port={port} "
                    f"age={(now - ready_at):.1f}s grace={SPLIT_READY_GRACE_WINDOW_SECONDS}s"
                )
                return  # Skip this check entirely during grace window

            # Track this miss
            miss_timestamps = getattr(self, 'split_runtime_port_model_miss_timestamps', [])
            miss_timestamps.append(now)
            # Prune old timestamps outside window
            cutoff = now - SPLIT_PORT_MODEL_MISS_WINDOW_SECONDS
            miss_timestamps = [ts for ts in miss_timestamps if ts > cutoff]
            self.split_runtime_port_model_miss_timestamps = miss_timestamps

            consecutive_misses = len(miss_timestamps)
            # If port listener is down, the runtime is truly dead - lower threshold
            if not port_has_listener:
                required_misses = 2
            else:
                required_misses = SPLIT_PORT_MODEL_MISS_CONSECUTIVE_COUNT

            if consecutive_misses < required_misses:
                # Not enough consecutive misses yet - warn and wait
                self.logger.warning(
                    f"SPLIT_INVARIANT_PORT_MODEL_MISS group={group_id} model={model_id} port={port} "
                    f"miss={consecutive_misses}/{required_misses} listener={port_has_listener}"
                )
                return  # Don't escalate to failure tracking yet

            # Threshold reached - set reason to trigger clear
            reason = f"port_model_missing:misses={consecutive_misses},listener={port_has_listener}"

        if reason:
            self.split_runtime_invariant_failures += 1
            # owner_meta_mismatch: immediate clear (authoritative mismatch)
            # port_model_missing with threshold: immediate clear (already passed threshold)
            # reservation_* issues: require 3 failures (may be transient sync issues)
            immediate_clear = reason.startswith("owner_meta_mismatch") or reason.startswith("port_model_missing")

            if self.split_runtime_invariant_failures == 1:
                self.logger.warning(
                    f"SPLIT_INVARIANT_WARN group={group_id} model={model_id} port={port} reason={reason}"
                )
            if immediate_clear or self.split_runtime_invariant_failures >= 3:
                self.logger.error(
                    f"SPLIT_INVARIANT_CLEAR group={group_id} model={model_id} port={port} reason={reason}"
                )
                self._mark_split_reservation_runtime_invalid(group_id, reason)
                if self.split_runtime_owner:
                    self._stop_split_runtime()
                # Use coordinated cleanup to actually reset local state, not just clear fields
                try:
                    split_port = int(port) if port else None
                except Exception:
                    split_port = None
                self._coordinated_split_failure_cleanup(
                    group_id=group_id,
                    split_port=split_port,
                    reason=f"invariant_{reason}",
                )
            return

        # All checks passed - reset failure tracking
        self.split_runtime_invariant_failures = 0
        self.split_runtime_port_model_miss_timestamps = []

    def _split_group_has_any_active_work(self, group_id: str) -> bool:
        """Return True if any reservation member currently reports active work.

        This prevents false-positive invariant clears while either member is actively
        running a task and split runtime signals are transient.
        """
        reservation = self._read_json_file(self._split_reservation_path(group_id))
        members = []
        if isinstance(reservation, dict):
            members = [str(m).strip() for m in reservation.get("members", []) if str(m).strip()]
        if not members:
            return False

        for member in members:
            hb_path = self.shared_path / "gpus" / member.replace("-", "_") / "heartbeat.json"
            try:
                hb = self._read_json_file(hb_path)
            except Exception:
                hb = None
            if not isinstance(hb, dict):
                continue
            if bool(hb.get("meta_task_active")):
                return True
            active_tasks = hb.get("active_tasks")
            if isinstance(active_tasks, list) and active_tasks:
                return True
        return False

    def _service_split_reservations(self):
        """Background sync so non-claiming partner can join and mirror split state."""
        for res_file in self.split_state_dir.glob("*.json"):
            if str(res_file).endswith(".lock"):
                continue
            lock = FileLock(str(res_file) + ".lock", timeout=1)
            start_runtime = None
            local_pair_prep = None
            deferred_cleanup = None
            lock_error = False
            try:
                with lock:
                    if not res_file.exists():
                        continue
                    with open(res_file, "r", encoding="utf-8") as f:
                        reservation = json.load(f)

                    members = [str(m).strip() for m in reservation.get("members", []) if str(m).strip()]
                    if self.name not in members:
                        continue

                    reservation = self._join_split_reservation(reservation, res_file)

                    status = str(reservation.get("status", "")).strip()
                    model_id = str(reservation.get("target_model", "")).strip()
                    group = {
                        "id": reservation.get("group_id"),
                        "members": members,
                        "port": reservation.get("port"),
                    }
                    launcher = str(reservation.get("launcher", "")).strip()
                    joined = reservation.get("joined", {}) if isinstance(reservation.get("joined"), dict) else {}
                    prepared = reservation.get("prepared", {}) if isinstance(reservation.get("prepared"), dict) else {}
                    member_clean = reservation.get("member_clean", {}) if isinstance(reservation.get("member_clean"), dict) else {}
                    all_joined = all(bool(joined.get(m)) for m in members)
                    all_prepared = all(bool(prepared.get(m)) for m in members if m in joined)
                    all_members_clean = all(bool(member_clean.get(m)) for m in members if m in joined)
                    port = group.get("port")

                    if (
                        status in {"waiting_partner", "joining", "loading"}
                        and bool(joined.get(self.name))
                        and not bool(prepared.get(self.name))
                    ):
                        local_pair_prep = {
                            "group_id": str(group.get("id") or ""),
                            "port": port,
                            "status": status,
                            "launcher": launcher,
                            "is_launcher": (self.name == launcher),
                        }

                    # If the split runtime is already loaded on the shared port, finalize
                    # to ready instead of re-launching and risking a false "port in use" failure.
                    # FIX: Only launcher can promote to ready, and must use stability gate.
                    if (
                        status in {"loading", "warming", "ready_stabilizing"}
                        and self._split_runtime_has_model_loaded(port=port, model_id=model_id)
                        and self._split_runtime_has_expected_owner(
                            port=port,
                            model_id=model_id,
                            group_id=str(group.get("id") or ""),
                        )
                    ):
                        # Only launcher can promote through stability gate
                        if self._is_launcher_for_reservation(reservation):
                            # Check if token already exists and is valid
                            token_valid, token_reason = self._verify_ready_token_age(reservation)
                            if token_valid:
                                # Token exists and is old enough - promote to ready
                                reservation = self._set_reservation_status(
                                    reservation,
                                    "ready",
                                    reason="runtime_already_loaded_with_token",
                                )
                                reservation.pop("error", None)
                                with open(res_file, "w", encoding="utf-8") as f:
                                    json.dump(reservation, f, indent=2)
                                status = "ready"
                            else:
                                # No valid token yet - transition to ready_stabilizing
                                # and run stability gate
                                if status != "ready_stabilizing":
                                    reservation = self._set_reservation_status(
                                        reservation,
                                        "ready_stabilizing",
                                        reason="entering_stability_gate",
                                    )
                                    reservation["warmup_completed_at"] = datetime.now().isoformat()
                                    with open(res_file, "w", encoding="utf-8") as f:
                                        json.dump(reservation, f, indent=2)
                                    status = "ready_stabilizing"

                                # Run stability gate (will probe and potentially issue token)
                                gate_passed, reservation = self._run_stability_gate(
                                    reservation, res_file, port, model_id
                                )
                                if gate_passed:
                                    reservation.pop("error", None)
                                    with open(res_file, "w", encoding="utf-8") as f:
                                        json.dump(reservation, f, indent=2)
                                    status = "ready"
                                else:
                                    # Still stabilizing - write updated probe count
                                    with open(res_file, "w", encoding="utf-8") as f:
                                        json.dump(reservation, f, indent=2)
                        else:
                            # Follower: skip promotion entirely, let launcher handle it
                            self.logger.debug(
                                f"SPLIT_FOLLOWER_SKIP_PROMOTION group={group.get('id')} "
                                f"status={status} (waiting for launcher to run stability gate)"
                            )

                    # Gate loading transition on ALL preconditions:
                    # 1. all_joined - both members have written joined[member]
                    # 2. all_prepared - both members have done port cleanup prep
                    # 3. all_members_clean - both members verified cold/unloaded via _verify_local_split_member_clean_precondition
                    can_advance_to_loading = all_joined and all_prepared and all_members_clean

                    if status in {"waiting_partner", "joining", "loading"} and can_advance_to_loading:
                        launcher_needs_election = False
                        if launcher not in members:
                            launcher_needs_election = True
                        elif status in {"waiting_partner", "joining"} and not self._is_gpu_heartbeat_fresh(launcher):
                            launcher_needs_election = True
                        elif status == "loading":
                            # Be conservative once loading has started. Re-election while the split
                            # runtime is warming can race with success and corrupt reservation state.
                            if (
                                not self._is_gpu_heartbeat_fresh(launcher)
                                and not self._split_runtime_has_model_loaded(port=port, model_id=model_id)
                            ):
                                owner = self._read_global_model_load_owner() or {}
                                owner_worker = str(owner.get("worker", "")).strip()
                                owner_phase = str(owner.get("phase", "")).strip()
                                if owner_worker not in members or owner_phase != "split_model_load":
                                    launcher_needs_election = True
                        if launcher_needs_election:
                            reservation["launcher"] = self.name
                            launcher = self.name
                        if self.name == launcher:
                            if status != "loading":
                                reservation = self._set_reservation_status(
                                    reservation,
                                    "loading",
                                    reason="launcher_starting_runtime",
                                )
                            with open(res_file, "w", encoding="utf-8") as f:
                                json.dump(reservation, f, indent=2)
                            start_runtime = {
                                "group": group,
                                "model_id": model_id,
                                "launcher": launcher,
                            }
                        else:
                            continue
                    elif status in {"waiting_partner", "joining", "loading"} and all_joined and not can_advance_to_loading:
                        # Identify what's blocking progress
                        missing_prepared = [m for m in members if m in joined and not prepared.get(m)]
                        missing_clean = [m for m in members if m in joined and not member_clean.get(m)]

                        if self.name == launcher and self.active_meta_task:
                            if missing_clean:
                                self._touch_meta_task(phase=f"waiting_member_clean:{','.join(missing_clean)}")
                            elif missing_prepared:
                                self._touch_meta_task(phase=f"waiting_split_pair_prep:{','.join(missing_prepared)}")
                            else:
                                self._touch_meta_task(phase="waiting_split_pair_prep")

                        # Log stall reason periodically (only if we're the launcher to avoid spam)
                        if self.name == launcher and (missing_prepared or missing_clean):
                            self.logger.debug(
                                f"SPLIT_RESERVATION_BLOCKED group={group.get('id')} status={status} "
                                f"missing_prepared={missing_prepared} missing_clean={missing_clean}"
                            )

                        # FAIL-FAST: If all_joined && all_prepared but member_clean still missing,
                        # this is an impossible state (with authoritative join, member_clean should
                        # always be written with joined). Fail after timeout to prevent infinite stall.
                        if all_joined and all_prepared and missing_clean:
                            stall_since_raw = reservation.get("member_clean_stall_since", "")
                            if not stall_since_raw:
                                # First detection of stall - record timestamp
                                reservation["member_clean_stall_since"] = datetime.now().isoformat()
                                with open(res_file, "w", encoding="utf-8") as f:
                                    json.dump(reservation, f, indent=2)
                            else:
                                try:
                                    stall_since = datetime.fromisoformat(stall_since_raw)
                                    stall_seconds = (datetime.now() - stall_since).total_seconds()
                                    if stall_seconds >= SPLIT_MEMBER_CLEAN_STALL_TIMEOUT_SECONDS:
                                        # Fail-fast: impossible state persisted too long
                                        self.logger.error(
                                            f"SPLIT_MEMBER_CLEAN_STALL_TIMEOUT group={group.get('id')} "
                                            f"missing_clean={missing_clean} stall_seconds={stall_seconds:.0f}"
                                        )
                                        reservation = self._set_reservation_status(
                                            reservation,
                                            "failed",
                                            reason=f"member_clean_missing:{','.join(missing_clean)}",
                                        )
                                        reservation["error"] = (
                                            f"member_clean stall timeout: joined members {missing_clean} "
                                            f"never wrote member_clean after {stall_seconds:.0f}s"
                                        )
                                        with open(res_file, "w", encoding="utf-8") as f:
                                            json.dump(reservation, f, indent=2)
                                        continue
                                except Exception:
                                    pass

                        # Let members with pending local prep run it outside the reservation lock.
                        if not local_pair_prep:
                            continue

                    if status == "ready":
                        # Do not mirror ready state unless the shared split port actually
                        # reports the target model loaded. Otherwise we can oscillate between
                        # wedged<->ready_split on stale reservations.
                        port = group.get("port")
                        ready_has_model = self._split_runtime_has_model_loaded(port=port, model_id=model_id)
                        ready_has_owner = self._split_runtime_has_expected_owner(
                            port=port,
                            model_id=model_id,
                            group_id=str(group.get("id") or ""),
                        )

                        # FIX: Verify ready_token exists and is valid before mirroring
                        token_valid, token_reason = self._verify_ready_token_age(reservation)
                        if not token_valid:
                            self.logger.warning(
                                "SPLIT_READY_TOKEN_INVALID "
                                f"group={group.get('id')} model={model_id} port={port} "
                                f"reason={token_reason} (not mirroring ready state)"
                            )
                            # Don't mirror - token not valid yet
                            # If we're already in this group, stay but don't refresh
                            if self.runtime_group_id == reservation.get("group_id"):
                                # Continue waiting - don't reset state yet
                                self.logger.debug(
                                    f"SPLIT_READY_WAIT_TOKEN group={group.get('id')} "
                                    f"reason={token_reason}"
                                )
                            return

                        if not ready_has_model or not ready_has_owner:
                            self.logger.warning(
                                "SPLIT_READY_REJECTED "
                                f"group={group.get('id')} model={model_id} port={port} "
                                f"has_model={ready_has_model} has_owner={ready_has_owner}"
                            )
                            if self.runtime_group_id == reservation.get("group_id"):
                                # Keep local state conservative until reservation becomes valid again.
                                self._set_runtime_state(
                                    RUNTIME_STATE_COLD,
                                    phase="split_ready_rejected",
                                    error_code="split_ready_invalid",
                                    error_detail=f"has_model={ready_has_model},has_owner={ready_has_owner}",
                                )
                                self.runtime_placement = "single_gpu"
                                self.runtime_group_id = None
                                self.runtime_port = self.port
                                self.runtime_ollama_url = f"http://localhost:{self.port}" if self.port else None
                                self.split_runtime_owner = False
                                self.model_loaded = False
                                self.loaded_model = None
                                self.loaded_tier = 0
                            return

                        # Both members mirror ready state; launcher is authoritative owner.
                        # FIX: Follower NEVER directly transitions to ready - only mirrors
                        # after launcher has issued token and it's old enough
                        self._set_split_runtime_loaded(
                            model_id,
                            group,
                            as_owner=(self.name == launcher),
                            task_id=self.active_meta_task.get("task_id") if self.active_meta_task else None,
                        )
                        return

                    if status in {"unloaded", "failed", "expired"}:
                        if self.runtime_group_id == reservation.get("group_id"):
                            # Collect cleanup params - execute AFTER releasing lock
                            # (cleanup tries to re-lock for member_reset, and does heavy I/O)
                            try:
                                deferred_cleanup_port = int(reservation.get("port")) if reservation.get("port") else None
                            except Exception:
                                deferred_cleanup_port = None
                            deferred_cleanup = {
                                "group_id": str(reservation.get("group_id") or ""),
                                "split_port": deferred_cleanup_port,
                                "reason": f"reservation_{status}",
                            }
            except Timeout:
                lock_error = True
            except Exception:
                lock_error = True

            # Execute deferred cleanup OUTSIDE the reservation lock (always, even on lock error)
            # This ensures cleanup runs even if an exception occurred after deferred_cleanup was set
            if deferred_cleanup:
                self._coordinated_split_failure_cleanup(
                    group_id=deferred_cleanup["group_id"],
                    split_port=deferred_cleanup["split_port"],
                    reason=deferred_cleanup["reason"],
                )
                deferred_cleanup = None
                continue

            # If lock acquisition failed, skip to next reservation
            if lock_error:
                continue

            if local_pair_prep:
                prep_result = self._prepare_local_for_split_pairing(
                    group_id=str(local_pair_prep.get("group_id") or ""),
                    port=local_pair_prep.get("port"),
                    is_launcher=bool(local_pair_prep.get("is_launcher")),
                    status=str(local_pair_prep.get("status") or ""),
                )
                try:
                    with lock:
                        if not res_file.exists():
                            continue
                        with open(res_file, "r", encoding="utf-8") as f:
                            reservation = json.load(f)
                        members = [str(m).strip() for m in reservation.get("members", []) if str(m).strip()]
                        if self.name not in members:
                            continue
                        joined = reservation.get("joined", {}) if isinstance(reservation.get("joined"), dict) else {}
                        if not joined.get(self.name):
                            continue
                        prepared = reservation.get("prepared", {}) if isinstance(reservation.get("prepared"), dict) else {}
                        prepared[self.name] = prep_result
                        reservation["prepared"] = prepared
                        reservation["updated_at"] = datetime.now().isoformat()
                        with open(res_file, "w", encoding="utf-8") as f:
                            json.dump(reservation, f, indent=2)
                except Exception:
                    pass
                continue

            if not start_runtime:
                continue

            ok, cleanup_done = self._start_split_runtime(start_runtime["group"], start_runtime["model_id"])
            try:
                with lock:
                    if not res_file.exists():
                        continue
                    with open(res_file, "r", encoding="utf-8") as f:
                        reservation = json.load(f)
                    if str(reservation.get("launcher", "")).strip() != self.name:
                        continue
                    reservation = self._set_reservation_status(
                        reservation,
                        "ready_stabilizing" if ok else "failed",
                        reason="warmup_complete_entering_stability_gate" if ok else "runtime_start_failed",
                    )
                    if ok:
                        reservation.pop("error", None)
                        # Record warmup completion for stability gate tracking
                        reservation["warmup_completed_at"] = datetime.now().isoformat()
                    else:
                        reservation["error"] = str(self.last_split_runtime_error or "split runtime failed")
                    with open(res_file, "w", encoding="utf-8") as f:
                        json.dump(reservation, f, indent=2)
            except Exception:
                pass

            # NOTE: Do NOT call _set_split_runtime_loaded here. Status is now
            # "ready_stabilizing" and the stability gate must pass first. When the
            # gate passes and status becomes "ready", the mirroring code at the
            # status == "ready" block will call _set_split_runtime_loaded for both
            # launcher and follower.
            if not ok and not cleanup_done:
                # Only do fallback cleanup if _start_split_runtime() didn't already do it
                # (e.g., early parameter validation fails before coordinated cleanup is called)
                try:
                    split_port = int(start_runtime["group"].get("port")) if start_runtime["group"].get("port") else None
                except Exception:
                    split_port = None
                self._coordinated_split_failure_cleanup(
                    group_id=str(start_runtime["group"].get("id") or ""),
                    split_port=split_port,
                    reason="runtime_start_failed_fallback",
                    task_id=self.active_meta_task.get("task_id") if self.active_meta_task else None,
                )
