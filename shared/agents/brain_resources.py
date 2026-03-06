"""Brain resource orchestration mixin.

Extracted from brain.py to keep split/meta resource decisions modular.
"""

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from filelock import FileLock
from gpu_constants import (
    GLOBAL_MODEL_LOAD_OWNER_STALE_SECONDS,
    SPLIT_ISSUE_SEVERITY_CRITICAL,
    SPLIT_ISSUE_SEVERITY_ERROR,
)

# Split wedge reclaim policy constants
SPLIT_WEDGE_RECONCILE_THRESHOLD = 3  # Consecutive wedged detections before reclaim
SPLIT_WEDGE_RECLAIM_COOLDOWN_S = 120  # Seconds between reclaim attempts per group
SPLIT_WEDGE_MAX_RECLAIMS_PER_HOUR = 6  # Max reclaims per group per hour

# =============================================================================
# Split Pair Selection Tuning Constants
# =============================================================================
# These control the heuristics for choosing which split GPU pair to promote.
# Bucket order is: clean_open > partially_busy > risky
# Within buckets, numeric scores determine winner.

# Heartbeat freshness thresholds
PAIR_HEARTBEAT_STALE_SECONDS = 60  # Heartbeat older than this = stale

# Runtime states considered stable for pair selection
# Transitional states (loading_*, unloading) should not be treated as "clean"
PAIR_STABLE_RUNTIME_STATES = {"cold", "ready_single", "ready_split"}
PAIR_TRANSITIONAL_RUNTIME_STATES = {"loading_single", "loading_split", "unloading"}
PAIR_ERROR_RUNTIME_STATES = {"wedged", "error_recoverable"}

# Thermal thresholds for pair scoring
PAIR_THERMAL_GOOD_C = 65  # Below this = full bonus
PAIR_THERMAL_WARN_C = 75  # Above this = penalty starts
PAIR_THERMAL_DANGER_C = 82  # Above this = risky bucket

# Scoring weights (higher = more important)
PAIR_WEIGHT_UNLOAD_PENALTY = 100  # Per member needing unload
PAIR_WEIGHT_THERMAL_HEADROOM = 2  # Per degree below THERMAL_WARN_C
PAIR_WEIGHT_THERMAL_PENALTY = 5  # Per degree above THERMAL_WARN_C
PAIR_WEIGHT_HEALTH_BONUS = 50  # For fully healthy pair
PAIR_WEIGHT_STUCK_RISK_PENALTY = 75  # For wedged/retry/timeout indicators
PAIR_WEIGHT_ELAPSED_TIME_BONUS = 1  # Per minute elapsed on task (capped)
PAIR_ELAPSED_TIME_BONUS_CAP_MINUTES = 10  # Don't reward elapsed time beyond this

# Power tie-breaker (when temps are close)
PAIR_POWER_TIEBREAK_WEIGHT = 0.1  # Per watt difference

# Task health indicators (for elapsed-time heuristic gate)
PAIR_TASK_HEARTBEAT_STALE_SECONDS = 60  # Task heartbeat older than this = unhealthy
PAIR_MAX_RECENT_RETRIES = 2  # More failures than this in recent stats = stuck risk

# Import quarantine constants (keep in sync with gpu_constants.py)
BRAIN_SPLIT_QUARANTINE_COOLDOWN_SECONDS = 900  # 15 minutes - must match gpu_constants


class BrainResourceMixin:
    def _split_group_members_for_group_id(self, group_id: str) -> List[str]:
        members: List[str] = []
        if not group_id:
            return members
        for meta in self.model_meta_by_id.values():
            groups = meta.get("split_groups", [])
            if not isinstance(groups, list):
                continue
            for group in groups:
                if not isinstance(group, dict):
                    continue
                current_group_id = str(group.get("id") or "").strip()
                if current_group_id != group_id:
                    continue
                members = [str(m).strip() for m in group.get("members", []) if str(m).strip()]
                if members:
                    return members
        return members

    def _evaluate_split_cleanup_decision(
        self,
        group_id: str,
        reports: List[Dict[str, Any]],
        gpu_states: Dict[str, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Return cleanup command details when observed issues justify cleanup."""
        if not reports:
            return None
        members = self._split_group_members_for_group_id(group_id)
        target_workers = members or [str(r.get("gpu", "")).strip() for r in reports if str(r.get("gpu", "")).strip()]
        if not target_workers:
            return None

        critical = []
        error_like = []
        runtime_generation = None
        split_port = None
        reservation_epoch = None
        for report in reports:
            issue = report["issue"]
            severity = str(issue.get("severity") or "").strip()
            if severity == SPLIT_ISSUE_SEVERITY_CRITICAL:
                critical.append(report)
            if severity in {SPLIT_ISSUE_SEVERITY_CRITICAL, SPLIT_ISSUE_SEVERITY_ERROR}:
                error_like.append(report)
            runtime_generation = runtime_generation or issue.get("runtime_generation") or report.get("runtime_generation")
            split_port = split_port or issue.get("split_port")
            reservation_epoch = reservation_epoch or issue.get("reservation_epoch")

        if critical:
            return {
                "reason": f"critical:{critical[0]['issue'].get('issue_code') or 'split_issue'}",
                "target_workers": target_workers,
                "runtime_generation": runtime_generation,
                "split_port": split_port,
                "reservation_epoch": reservation_epoch,
            }

        expected_members = len(members) if members else len(target_workers)
        if expected_members > 0 and len(error_like) >= expected_members:
            return {
                "reason": f"all_members_error:{error_like[0]['issue'].get('issue_code') or 'split_issue'}",
                "target_workers": target_workers,
                "runtime_generation": runtime_generation,
                "split_port": split_port,
                "reservation_epoch": reservation_epoch,
            }

        return None

    def _monitor_split_health_issues(self, gpu_states: Dict[str, Dict[str, Any]]) -> None:
        """Observe worker-reported split health issues and issue cleanup when warranted."""
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for gpu_name, state in gpu_states.items():
            issue = state.get("split_health_issue")
            if not isinstance(issue, dict) or not issue.get("has_issue"):
                continue
            group_id = str(issue.get("group_id") or "").strip() or f"worker:{gpu_name}"
            grouped.setdefault(group_id, []).append({
                "gpu": gpu_name,
                "issue": issue,
                "runtime_generation": state.get("split_runtime_generation"),
                "runtime_state": state.get("runtime_state"),
            })

        for group_id, reports in grouped.items():
            summary = []
            for report in reports:
                issue = report["issue"]
                summary.append(
                    {
                        "gpu": report["gpu"],
                        "severity": issue.get("severity"),
                        "issue_code": issue.get("issue_code"),
                        "awaiting_brain_decision": issue.get("awaiting_brain_decision"),
                        "runtime_generation": issue.get("runtime_generation") or report.get("runtime_generation"),
                        "runtime_state": report.get("runtime_state"),
                    }
                )
            signature = json.dumps(summary, sort_keys=True)
            seen = getattr(self, "_last_split_health_issue_signatures", {})
            if seen.get(group_id) == signature:
                continue
            seen[group_id] = signature
            self._last_split_health_issue_signatures = seen
            self.log_decision(
                "SPLIT_HEALTH_ISSUES_OBSERVED",
                f"Observed split health issues for {group_id}",
                {"group_id": group_id, "reports": summary},
            )
            cleanup_decision = self._evaluate_split_cleanup_decision(group_id, reports, gpu_states)
            if cleanup_decision:
                self._issue_split_cleanup_command(
                    group_id,
                    cleanup_decision["target_workers"],
                    cleanup_decision["reason"],
                    split_port=cleanup_decision.get("split_port"),
                    reservation_epoch=cleanup_decision.get("reservation_epoch"),
                    runtime_generation=cleanup_decision.get("runtime_generation"),
                )

        seen = getattr(self, "_last_split_health_issue_signatures", {})
        stale_groups = [group_id for group_id in seen if group_id not in grouped]
        for group_id in stale_groups:
            del seen[group_id]
        self._last_split_health_issue_signatures = seen

    def _monitor_global_load_owner_issues(self, gpu_states: Dict[str, Dict[str, Any]]) -> None:
        """Observe worker-reported global load-owner issues for future centralization."""
        seen = getattr(self, "_last_global_load_owner_issue_signatures", {})
        active_workers = set()
        for gpu_name, state in gpu_states.items():
            issue = state.get("global_load_owner_issue")
            if not isinstance(issue, dict) or not issue.get("has_issue"):
                continue
            signature = json.dumps(issue, sort_keys=True)
            active_workers.add(gpu_name)
            if seen.get(gpu_name) == signature:
                continue
            seen[gpu_name] = signature
            self.log_decision(
                "GLOBAL_LOAD_OWNER_ISSUE_OBSERVED",
                f"Observed global load-owner issue from {gpu_name}",
                {"gpu": gpu_name, "issue": issue},
            )
            self._brain_reclaim_stale_global_load_owner(issue)
        stale_workers = [gpu_name for gpu_name in seen if gpu_name not in active_workers]
        for gpu_name in stale_workers:
            del seen[gpu_name]
        self._last_global_load_owner_issue_signatures = seen

    def _brain_global_load_owner_is_stale(self, owner: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(owner, dict) or not owner:
            return True
        pid = owner.get("pid")
        try:
            pid_int = int(pid)
        except Exception:
            pid_int = None
        if pid_int:
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

    def _brain_reclaim_stale_global_load_owner(self, issue: Dict[str, Any]) -> None:
        owner_path = self.signals_path / "model_load.global.json"
        if not owner_path.exists():
            return
        try:
            with open(owner_path, "r", encoding="utf-8") as f:
                current_owner = json.load(f)
        except Exception:
            return
        if not isinstance(current_owner, dict):
            return
        expected_lease_id = str(issue.get("owner_lease_id", "")).strip()
        expected_worker = str(issue.get("owner_worker", "")).strip()
        if expected_lease_id and str(current_owner.get("lease_id", "")).strip() != expected_lease_id:
            return
        if expected_worker and str(current_owner.get("worker", "")).strip() != expected_worker:
            return
        if not self._brain_global_load_owner_is_stale(current_owner):
            return
        try:
            owner_path.unlink(missing_ok=True)
        except Exception as exc:
            self.logger.warning(f"Failed to reclaim stale global load owner: {exc}")
            return
        self.log_decision(
            "GLOBAL_LOAD_OWNER_RECLAIMED",
            "Brain reclaimed stale global load-owner lease",
            {"owner": current_owner},
        )

    def _iter_task_files_json(self, *paths: Path):
        for path in paths:
            for task_file in path.glob("*.json"):
                if str(task_file).endswith(".lock"):
                    continue
                try:
                    with open(task_file) as f:
                        task = json.load(f)
                    if isinstance(task, dict):
                        yield task
                except Exception:
                    continue

    def _task_llm_tier(self, task: Dict[str, Any]) -> int:
        model_id = str(task.get("llm_model") or task.get("target_model") or "").strip()
        if model_id:
            try:
                return int(self.model_tier_by_id.get(model_id, self.default_llm_min_tier))
            except Exception:
                return int(self.default_llm_min_tier)
        try:
            return int(task.get("llm_tier", 0) or 0)
        except Exception:
            return 0

    def _collect_llm_demand_window_snapshot(self) -> Dict[str, Any]:
        """Aggregate LLM demand across queue, processing, and private tasks.

        Purposefully conservative: private tasks count as future demand so the brain
        does not unload models during brief release gaps.
        """
        counts = {
            "queue_llm": 0,
            "processing_llm": 0,
            "private_llm": 0,
            "total_llm": 0,
            "split_llm": 0,
            "min_tier": 0,
            "max_tier": 0,
            "tiers": {},
        }

        def _accumulate(task: Dict[str, Any], source: str):
            if str(task.get("task_class", "")).strip() != "llm":
                return
            if source == "queue":
                counts["queue_llm"] += 1
            elif source == "processing":
                counts["processing_llm"] += 1
            elif source == "private":
                counts["private_llm"] += 1
            counts["total_llm"] += 1
            if str(task.get("llm_placement", "")).strip() == "split_gpu":
                counts["split_llm"] += 1
            tier = self._task_llm_tier(task)
            if tier > 0:
                counts["min_tier"] = tier if counts["min_tier"] <= 0 else min(counts["min_tier"], tier)
                counts["max_tier"] = max(counts["max_tier"], tier)
                counts["tiers"][str(tier)] = int(counts["tiers"].get(str(tier), 0)) + 1

        for task in self._iter_task_files_json(self.queue_path):
            _accumulate(task, "queue")
        for task in self._iter_task_files_json(self.processing_path):
            _accumulate(task, "processing")
        for task in self._iter_task_files_json(self.private_tasks_path):
            _accumulate(task, "private")

        return counts

    def _update_llm_demand_timers(self, demand: Dict[str, Any]) -> Dict[str, float]:
        now = datetime.now()
        if int(demand.get("total_llm", 0) or 0) > 0:
            self.last_any_llm_demand_at = now
        if int(demand.get("split_llm", 0) or 0) > 0 or int(demand.get("max_tier", 0) or 0) >= 2:
            self.last_split_llm_demand_at = now
        any_idle_s = max(0.0, (now - self.last_any_llm_demand_at).total_seconds())
        split_idle_s = max(0.0, (now - self.last_split_llm_demand_at).total_seconds())
        return {"any_llm_idle_s": any_idle_s, "split_llm_idle_s": split_idle_s}

    def _meta_task_signature(self, task: Dict[str, Any]) -> str:
        key = {
            "command": task.get("command"),
            "target_model": task.get("target_model"),
            "group_id": task.get("group_id"),
            "candidate_groups": task.get("candidate_groups"),
            # Include worker targeting fields so targeted unload/load tasks
            # for different GPUs do not dedupe into each other.
            "candidate_workers": task.get("candidate_workers"),
            "target_worker": task.get("target_worker"),
            "target_gpu": task.get("target_gpu"),
            "target_workers": task.get("target_workers"),
            "cleanup_reason": task.get("cleanup_reason"),
            "reservation_epoch": task.get("reservation_epoch"),
            "runtime_generation": task.get("runtime_generation"),
        }
        return json.dumps(key, sort_keys=True)

    def _has_existing_meta_task(self, command: str, meta: Optional[Dict[str, Any]] = None) -> bool:
        """Check if a meta task with the given command already exists in queue or processing.

        Prevents duplicate meta tasks when the brain restarts or timing is tight
        between queue checks and task insertion.
        """
        expected = {"command": command}
        if isinstance(meta, dict):
            expected.update(meta)
        expected_sig = self._meta_task_signature(expected)

        # Check queue
        for task_file in self.queue_path.glob("*.json"):
            if str(task_file).endswith('.lock'):
                continue
            try:
                with open(task_file) as f:
                    task = json.load(f)
                if task.get("task_class") == "meta" and self._meta_task_signature(task) == expected_sig:
                    self.logger.debug(f"Dedup: {command} already in queue ({task['task_id'][:8]})")
                    return True
            except Exception:
                continue

        # Check processing
        for task_file in self.processing_path.glob("*.json"):
            if str(task_file).endswith('.lock'):
                continue
            try:
                with open(task_file) as f:
                    task = json.load(f)
                if task.get("task_class") == "meta" and self._meta_task_signature(task) == expected_sig:
                    self.logger.debug(f"Dedup: {command} already in processing ({task['task_id'][:8]})")
                    return True
            except Exception:
                continue

        return False

    def _insert_resource_task(self, command: str, meta: Optional[Dict[str, Any]] = None):
        """Insert a resource task (load_llm or unload_llm) for workers to claim.

        Performs a dedup scan first — skips insertion if the same command is already
        queued or in processing, preventing duplicate meta tasks after brain restart
        or under race conditions.
        """
        # Cooldown: avoid rapid repeated resource commands.
        dedup_key = command
        if isinstance(meta, dict) and meta:
            dedup_key = f"{command}:{json.dumps(meta, sort_keys=True)}"
        last_at = self.last_resource_task_at.get(dedup_key)
        if last_at:
            elapsed = (datetime.now() - last_at).total_seconds()
            if elapsed < self.resource_task_cooldown_seconds:
                self.log_decision(
                    "RESOURCE_COOLDOWN",
                    f"Skipping {command} - cooldown active ({elapsed:.0f}s/{self.resource_task_cooldown_seconds}s)",
                    {"command": command, "elapsed_seconds": elapsed}
                )
                return

        # Dedup: check if this exact command already exists
        if self._has_existing_meta_task(command, meta=meta):
            self.log_decision("RESOURCE_DEDUP",
                f"Skipping {command} — already exists in queue/processing", {})
            return

        # Brain-level system commands (executed by brain, not workers)
        BRAIN_SYSTEM_COMMANDS = {"orchestrator_full_reset"}

        is_brain_command = command in BRAIN_SYSTEM_COMMANDS

        task = {
            "task_id": str(uuid.uuid4()),
            "type": "system" if is_brain_command else "meta",
            "command": command,
            "batch_id": "system",
            "name": command,
            "priority": 10,  # High priority
            "task_class": "brain" if is_brain_command else "meta",
            "depends_on": [],
            "executor": "brain" if is_brain_command else "worker",
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "created_by": self.name,
            "retry_count": 0
        }
        if isinstance(meta, dict):
            for k, v in meta.items():
                task[k] = v
        self.save_to_public(task)
        self.log_decision("RESOURCE_TASK", f"Inserted {command} task", {"task_id": task["task_id"][:8]})
        self.last_resource_task_at[dedup_key] = datetime.now()

        # Track load_llm requests to detect when GPUs don't pick them up
        if command == "load_llm":
            gpu_states = self._get_gpu_states()
            cold_gpus = [g for g, s in gpu_states.items() if not s.get("model_loaded", False)]
            self.load_llm_requests[task["task_id"]] = {
                'created_at': datetime.now(),
                'gpus_needed': cold_gpus.copy()
            }

    def _issue_split_cleanup_command(
        self,
        group_id: str,
        target_workers: List[str],
        reason: str,
        *,
        split_port: Optional[int] = None,
        reservation_epoch: Optional[str] = None,
        runtime_generation: Optional[str] = None,
    ) -> None:
        """Queue a fenced split cleanup command for workers to execute."""
        if group_id:
            self._record_split_failure_brain(group_id, reason)
        meta = {
            "group_id": group_id,
            "target_workers": [w for w in target_workers if str(w).strip()],
            "cleanup_reason": reason,
        }
        if split_port is not None:
            meta["split_port"] = split_port
        if reservation_epoch is not None:
            meta["reservation_epoch"] = reservation_epoch
        if runtime_generation is not None:
            meta["runtime_generation"] = runtime_generation
        self._insert_resource_task("cleanup_split_runtime", meta=meta)
        self.log_decision(
            "SPLIT_CLEANUP_COMMAND_QUEUED",
            f"Queued split cleanup for {group_id}",
            {
                "group_id": group_id,
                "target_workers": meta["target_workers"],
                "reason": reason,
                "split_port": split_port,
                "reservation_epoch": reservation_epoch,
                "runtime_generation": runtime_generation,
            },
        )

    def _handle_missing_gpu_escalations(self, truly_missing: List[str], queue_stats: Dict[str, Any]):
        """
        Escalate persistent missing GPU agents once, and log recovery when they return.
        """
        missing_now = set(truly_missing)
        tracked = set(self.gpu_missing_escalations.keys())

        # Recoveries: GPUs previously escalated are now back.
        for gpu in sorted(tracked - missing_now):
            esc = self.gpu_missing_escalations.pop(gpu, {})
            self.log_decision(
                "GPU_RECOVERED",
                f"GPU agent recovered: {gpu}",
                {"gpu": gpu, "previous_escalation_id": esc.get("escalation_id")}
            )

        # New persistent missing GPUs: escalate once per outage.
        for gpu in sorted(missing_now):
            if gpu in self.gpu_missing_escalations:
                continue

            escalation_id = self.emit_cloud_escalation(
                escalation_type="infrastructure_failure",
                title="GPU agent missing while worker queue has pending tasks",
                details={
                    "gpu": gpu,
                    "reason": "persistent_missing_gpu_agent",
                    "miss_count": self.gpu_miss_count.get(gpu, 0),
                    "heartbeat_stale_seconds": self.heartbeat_stale_seconds,
                    "queue_worker_tasks": queue_stats.get("worker_tasks", 0),
                    "queue_total_pending": queue_stats.get("total_pending", 0),
                    "processing_count": queue_stats.get("processing_count", 0),
                }
            )

            self.gpu_missing_escalations[gpu] = {
                "escalation_id": escalation_id,
                "first_seen_at": datetime.now().isoformat(),
                "miss_count": self.gpu_miss_count.get(gpu, 0),
            }
            self.log_decision(
                "GPU_MISSING_ESCALATED",
                f"Escalated persistent missing GPU agent: {gpu}",
                {
                    "gpu": gpu,
                    "escalation_id": escalation_id,
                    "miss_count": self.gpu_miss_count.get(gpu, 0),
                    "queue_worker_tasks": queue_stats.get("worker_tasks", 0),
                }
            )

    def _choose_split_model_for_tier(
        self,
        min_tier: int,
        preferred_models: Optional[List[str]] = None,
    ) -> Optional[str]:
        if isinstance(preferred_models, list):
            for model_id in preferred_models:
                model_id = str(model_id or "").strip()
                if not model_id:
                    continue
                meta = self.model_meta_by_id.get(model_id, {})
                if str(meta.get("placement", "")) != "split_gpu":
                    continue
                try:
                    tier = int(meta.get("tier", self.default_llm_min_tier))
                except Exception:
                    tier = self.default_llm_min_tier
                if tier >= min_tier:
                    return model_id

        candidates = []
        for model_id, meta in self.model_meta_by_id.items():
            if str(meta.get("placement", "")) != "split_gpu":
                continue
            tier = int(meta.get("tier", self.default_llm_min_tier))
            if tier < min_tier:
                continue
            candidates.append((tier, model_id))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def _split_groups_for_model(self, model_id: str) -> List[Dict[str, Any]]:
        meta = self.model_meta_by_id.get(model_id, {})
        groups = meta.get("split_groups", [])
        if not isinstance(groups, list):
            return None
        normalized = []
        for g in groups:
            if not isinstance(g, dict):
                continue
            members = [str(m).strip() for m in g.get("members", []) if str(m).strip()]
            if len(members) < 2:
                continue
            group_id = str(g.get("id") or f"group_{'_'.join(sorted(members))}").strip()
            try:
                port = int(g.get("port"))
            except Exception:
                port = None
            normalized.append({"id": group_id, "members": members, "port": port})
        return normalized

    def _has_pending_meta_command(self, command: str) -> bool:
        for lane in (self.queue_path, self.processing_path):
            for task_file in lane.glob("*.json"):
                try:
                    with open(task_file) as f:
                        task = json.load(f)
                    if task.get("task_class") == "meta" and str(task.get("command", "")) == command:
                        return True
                except Exception:
                    continue
        return False

    def _single_hot_gpu_rows(self, gpu_states: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for gpu_name, state in gpu_states.items():
            if not state.get("model_loaded", False):
                continue
            if str(state.get("runtime_placement", "")) != "single_gpu":
                continue
            try:
                tier = int(state.get("loaded_tier", 0) or 0)
            except Exception:
                tier = 0
            rows.append({
                "gpu": gpu_name,
                "tier": tier,
                "model": str(state.get("loaded_model", "")).strip(),
            })
        rows.sort(key=lambda r: (r["tier"], r["gpu"]))
        return rows

    def _split_group_members_needing_unload(
        self,
        group: Dict[str, Any],
        gpu_states: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        """Check if any split group members need unload before split load.

        Returns list of member GPU names that are currently hot with single-GPU
        runtimes and need to be unloaded before the split load can proceed.

        This enforces the brain sequencing rule: issue unload_llm for hot members
        BEFORE issuing load_split_llm.
        """
        members = [str(m).strip() for m in group.get("members", []) if str(m).strip()]
        target_group_id = str(group.get("id", "")).strip()
        need_unload = []
        for member in members:
            state = gpu_states.get(member, {})
            # Check if member has a model loaded (any placement)
            if not state.get("model_loaded", False):
                continue
            # Skip if already in split_gpu placement for THIS SAME group (rejoin case)
            if str(state.get("runtime_placement", "")) == "split_gpu":
                member_group_id = str(state.get("runtime_group_id", "")).strip()
                if target_group_id and member_group_id == target_group_id:
                    # Rejoining same split group - don't require unload
                    continue
                # Different split group - still needs unload
            # Member is hot (single-GPU or different split group) - needs unload first
            need_unload.append(member)
        return need_unload

    def _has_pending_unload_for_members(self, members: List[str]) -> Dict[str, bool]:
        """Check if unload_llm tasks already exist for specified members.

        Returns dict mapping member name to whether an unload is pending.
        """
        pending = {m: False for m in members}
        for lane in (self.queue_path, self.processing_path):
            for task_file in lane.glob("*.json"):
                try:
                    with open(task_file) as f:
                        task = json.load(f)
                except Exception:
                    continue
                if task.get("task_class") != "meta":
                    continue
                command = str(task.get("command", ""))
                if command != "unload_llm":
                    continue
                # Check candidate_workers
                candidates = task.get("candidate_workers", [])
                if isinstance(candidates, list):
                    for member in members:
                        if member in candidates:
                            pending[member] = True
        return pending

    def _targeted_single_unload_candidate(
        self,
        gpu_states: Dict[str, Dict[str, Any]],
        unhealthy_gpus: List[str],
        demand: Dict[str, Any],
    ) -> Optional[str]:
        """Pick a safe single-GPU unload target based on demand hierarchy.

        If any LLM demand exists, only unload a single-GPU worker whose tier is below
        *all* current LLM demand tiers. This preserves higher-tier runtimes so they can
        continue serving lower-tier work.
        """
        candidates = [
            row for row in self._single_hot_gpu_rows(gpu_states)
            if row["gpu"] not in unhealthy_gpus
        ]
        if not candidates:
            return None
        total_llm = int(demand.get("total_llm", 0) or 0)
        if total_llm <= 0:
            return candidates[0]["gpu"]
        min_demand_tier = int(demand.get("min_tier", 0) or 0)
        if min_demand_tier <= 0:
            return None
        for row in candidates:
            if int(row["tier"]) < min_demand_tier:
                return row["gpu"]
        return None

    def _pending_split_group_ids(self, model_id: str, candidate_group_ids: List[str]) -> set[str]:
        pending: set[str] = set()
        wanted = {str(gid).strip() for gid in candidate_group_ids if str(gid).strip()}
        if not wanted:
            return pending

        for lane in (self.queue_path, self.processing_path):
            for task_file in lane.glob("*.json"):
                try:
                    with open(task_file) as f:
                        task = json.load(f)
                except Exception:
                    continue
                if task.get("task_class") != "meta" or str(task.get("command", "")) != "load_split_llm":
                    continue
                if str(task.get("target_model", "")) != str(model_id):
                    continue
                groups = task.get("candidate_groups", [])
                if not isinstance(groups, list) or not groups:
                    pending |= wanted
                    continue
                for group in groups:
                    if not isinstance(group, dict):
                        continue
                    gid = str(group.get("id", "")).strip()
                    if gid and gid in wanted:
                        pending.add(gid)
        return pending

    def _split_reservation_file(self, group_id: str) -> Path:
        return self.signals_path / "split_llm" / f"{group_id}.json"

    def _read_split_reservation(self, group_id: str) -> Optional[Dict[str, Any]]:
        path = self._split_reservation_file(group_id)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return None
            return data
        except Exception:
            return None

    def _write_split_reservation(self, group_id: str, reservation: Dict[str, Any]) -> bool:
        path = self._split_reservation_file(group_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(path) + ".lock", timeout=1)
        try:
            with lock:
                with open(path, "w") as f:
                    json.dump(reservation, f, indent=2)
            return True
        except Exception as e:
            self.logger.warning(f"Failed to write split reservation {group_id}: {e}")
            return False

    # =========================================================================
    # Split Wedge Reclaim Policy
    # =========================================================================

    def _increment_split_wedge_count(self, group_id: str) -> int:
        """Increment and return the consecutive wedge detection count for a group."""
        self.split_wedge_counts[group_id] = self.split_wedge_counts.get(group_id, 0) + 1
        return self.split_wedge_counts[group_id]

    def _clear_split_wedge_count(self, group_id: str):
        """Clear wedge count when a group is healthy or reclaimed."""
        self.split_wedge_counts.pop(group_id, None)

    def _can_reclaim_split_group(self, group_id: str) -> tuple[bool, str]:
        """Check if we can attempt a reclaim for this group.

        Returns (can_reclaim, reason). Respects cooldown and hourly limits.
        """
        now = datetime.now()

        # Check cooldown
        last_reclaim = self.split_wedge_last_reclaim_at.get(group_id)
        if last_reclaim:
            elapsed = (now - last_reclaim).total_seconds()
            if elapsed < SPLIT_WEDGE_RECLAIM_COOLDOWN_S:
                return False, f"cooldown:{int(SPLIT_WEDGE_RECLAIM_COOLDOWN_S - elapsed)}s_remaining"

        # Check hourly limit
        reclaim_times = self.split_wedge_reclaims_this_hour.get(group_id, [])
        # Prune reclaims older than 1 hour
        one_hour_ago = now.timestamp() - 3600
        reclaim_times = [t for t in reclaim_times if t.timestamp() > one_hour_ago]
        self.split_wedge_reclaims_this_hour[group_id] = reclaim_times

        if len(reclaim_times) >= SPLIT_WEDGE_MAX_RECLAIMS_PER_HOUR:
            return False, f"hourly_limit:{len(reclaim_times)}/{SPLIT_WEDGE_MAX_RECLAIMS_PER_HOUR}"

        return True, ""

    def _record_split_reclaim_attempt(self, group_id: str):
        """Record that we attempted a reclaim for this group."""
        now = datetime.now()
        self.split_wedge_last_reclaim_at[group_id] = now
        if group_id not in self.split_wedge_reclaims_this_hour:
            self.split_wedge_reclaims_this_hour[group_id] = []
        self.split_wedge_reclaims_this_hour[group_id].append(now)
        self._clear_split_wedge_count(group_id)

    def _should_trigger_wedge_reclaim(self, group_id: str) -> tuple[bool, str]:
        """Check if we should trigger a reclaim for a wedged group.

        Returns (should_reclaim, reason).
        """
        count = self.split_wedge_counts.get(group_id, 0)
        if count < SPLIT_WEDGE_RECONCILE_THRESHOLD:
            return False, f"below_threshold:{count}/{SPLIT_WEDGE_RECONCILE_THRESHOLD}"

        can_reclaim, reason = self._can_reclaim_split_group(group_id)
        if not can_reclaim:
            return False, reason

        return True, f"threshold_reached:{count}"

    def _probe_split_runtime_models(self, port: Optional[int], timeout_s: float = 1.5) -> Dict[str, Any]:
        if not port:
            return {"reachable": False, "models": [], "error": "missing_port"}
        url = f"http://127.0.0.1:{int(port)}/api/ps"
        try:
            response = requests.get(url, timeout=timeout_s)
            response.raise_for_status()
            payload = response.json()
            models = []
            for item in payload.get("models", []) if isinstance(payload, dict) else []:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or item.get("model") or "").strip()
                if name:
                    models.append(name)
            return {"reachable": True, "models": models, "error": None}
        except Exception as e:
            return {"reachable": False, "models": [], "error": str(e)}

    def _reconcile_split_group_state(
        self,
        group: Dict[str, Any],
        target_model: str,
    ) -> Dict[str, Any]:
        """Non-destructive split control-plane/runtime reconciliation.

        Brain may repair reservation status to `ready` when the runtime is already
        serving the target model. It does not stop processes or force-kill workers.

        FIX: Check for valid ready_token before classifying as ready_real.
        Reachable port without target model must NOT infer ready.
        """
        group_id = str(group.get("id", "")).strip()
        port = group.get("port")
        members = [str(m).strip() for m in group.get("members", []) if str(m).strip()]
        reservation = self._read_split_reservation(group_id)
        probe = self._probe_split_runtime_models(port)
        models = [str(m).strip() for m in probe.get("models", []) if str(m).strip()]
        target_loaded = target_model in models
        status = str((reservation or {}).get("status", "")).strip()

        # Check for ready_token
        ready_token = (reservation or {}).get("ready_token") if reservation else None
        has_valid_token = bool(ready_token)

        if target_loaded:
            # Group has target model loaded - check if we should classify as ready_real
            self._clear_split_wedge_count(group_id)

            # FIX: If port has model but NO ready_token, this is suspicious
            # The stability gate wasn't run properly
            if not has_valid_token:
                self.log_decision(
                    "SPLIT_RECONCILE_LOADING_SUSPICIOUS",
                    f"Split group {group_id} has model but no ready_token",
                    {
                        "group_id": group_id,
                        "port": port,
                        "target_model": target_model,
                        "probe_models": models,
                        "reservation_status": status or None,
                        "has_ready_token": False,
                    },
                )
                return {
                    "classification": "loading_suspicious",
                    "group_id": group_id,
                    "port": port,
                    "probe_models": models,
                    "reservation_status": status or None,
                    "has_ready_token": False,
                }

            # Has valid token - can classify as ready
            if (
                not reservation
                or status != "ready"
                or str(reservation.get("target_model", "")) != str(target_model)
            ):
                now = datetime.now().isoformat()
                repaired = dict(reservation or {})
                repaired.update(
                    {
                        "group_id": group_id,
                        "target_model": target_model,
                        "status": "ready",
                        "members": members,
                        "port": port,
                        "updated_at": now,
                        "ready_at": now,
                    }
                )
                if reservation and reservation.get("created_at"):
                    repaired["created_at"] = reservation.get("created_at")
                else:
                    repaired.setdefault("created_at", now)
                # Preserve ready_token
                if ready_token:
                    repaired["ready_token"] = ready_token
                    repaired["ready_token_issued_at"] = (reservation or {}).get("ready_token_issued_at")
                wrote = self._write_split_reservation(group_id, repaired)
                self.log_decision(
                    "SPLIT_RECONCILE_READY",
                    f"Reconciled split group {group_id} to ready from runtime probe",
                    {
                        "group_id": group_id,
                        "port": port,
                        "target_model": target_model,
                        "reservation_status_before": status or None,
                        "probe_models": models,
                        "reservation_repaired": wrote,
                        "has_ready_token": True,
                    },
                )
            return {
                "classification": "ready_real",
                "group_id": group_id,
                "port": port,
                "probe_models": models,
                "reservation_status": status or None,
                "has_ready_token": True,
            }

        if probe.get("reachable") and not target_loaded:
            # Listener exists but target model is not loaded; treat as wedged/orphan-ish.
            wedge_count = self._increment_split_wedge_count(group_id)
            should_reclaim, reclaim_reason = self._should_trigger_wedge_reclaim(group_id)

            # Record failure for quarantine tracking when reclaim is triggered
            if should_reclaim:
                self._record_split_failure_brain(group_id, f"wedged_port_reclaim:{reclaim_reason}")

            self.log_decision(
                "SPLIT_RECONCILE_WEDGED_PORT",
                f"Split group {group_id} port {port} reachable without target model",
                {
                    "group_id": group_id,
                    "port": port,
                    "target_model": target_model,
                    "probe_models": models,
                    "reservation_status": status or None,
                    "wedge_count": wedge_count,
                    "should_reclaim": should_reclaim,
                    "reclaim_reason": reclaim_reason,
                },
            )
            return {
                "classification": "wedged_port",
                "group_id": group_id,
                "port": port,
                "probe_models": models,
                "reservation_status": status or None,
                "wedge_count": wedge_count,
                "should_reclaim": should_reclaim,
                "reclaim_reason": reclaim_reason,
            }

        if status in {"waiting_partner", "loading"}:
            return {
                "classification": "loading_control",
                "group_id": group_id,
                "port": port,
                "probe_models": models,
                "reservation_status": status,
            }

        if status == "ready":
            self.log_decision(
                "SPLIT_RECONCILE_STALE_READY",
                f"Split group {group_id} reservation says ready but runtime probe missing target",
                {
                    "group_id": group_id,
                    "port": port,
                    "target_model": target_model,
                    "probe_models": models,
                    "reservation_status": status,
                },
            )

        return {
            "classification": "empty",
            "group_id": group_id,
            "port": port,
            "probe_models": models,
            "reservation_status": status or None,
        }

    # =========================================================================
    # Split Pair Selection Heuristics
    # =========================================================================

    def _get_member_heartbeat_age_seconds(self, state: Dict[str, Any]) -> float:
        """Get age of GPU heartbeat in seconds."""
        last_updated = state.get("last_updated")
        if not last_updated:
            return float("inf")
        try:
            updated_dt = datetime.fromisoformat(str(last_updated))
            return (datetime.now() - updated_dt).total_seconds()
        except Exception:
            return float("inf")

    def _is_member_healthy(self, state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Check if a GPU member is healthy for split pair selection.

        Returns (is_healthy, list_of_issues).
        A member is healthy if:
        - Heartbeat is fresh
        - Not in thermal pause
        - Runtime state is stable (not transitioning, wedged, or error)
        - Ollama is healthy
        """
        issues = []

        # Heartbeat freshness
        hb_age = self._get_member_heartbeat_age_seconds(state)
        if hb_age > PAIR_HEARTBEAT_STALE_SECONDS:
            issues.append(f"stale_heartbeat:{int(hb_age)}s")

        # Thermal pause
        if state.get("thermal_pause_active"):
            issues.append("thermal_pause")

        # Runtime state check - reject transitional and error states
        runtime_state = str(state.get("runtime_state", "")).strip()
        if runtime_state in PAIR_TRANSITIONAL_RUNTIME_STATES:
            issues.append(f"transitional:{runtime_state}")
        elif runtime_state in PAIR_ERROR_RUNTIME_STATES:
            issues.append(f"error_state:{runtime_state}")
        elif runtime_state and runtime_state not in PAIR_STABLE_RUNTIME_STATES:
            # Unknown state - treat as potentially problematic
            issues.append(f"unknown_state:{runtime_state}")

        # Ollama health
        if not state.get("ollama_healthy", True):
            issues.append("ollama_unhealthy")

        return len(issues) == 0, issues

    def _is_member_task_healthy(self, state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Check if a member's active task is healthy (for elapsed-time heuristic gate).

        Returns (is_healthy, list_of_issues).
        Only applies if member has an active task.
        Gates the elapsed-time bonus - unhealthy tasks should not get "closer to done" credit.
        """
        issues = []

        # Check if there's an active task
        active_workers = state.get("active_workers", 0)
        has_meta_task = state.get("meta_task_active", False)
        if active_workers == 0 and not has_meta_task:
            # No active task - trivially healthy
            return True, []

        # 1. Check thermal pause flapping (multiple recent pause attempts)
        pause_attempts = state.get("thermal_pause_attempts", 0)
        if pause_attempts >= 2:
            issues.append(f"thermal_flapping:{pause_attempts}")

        # 2. Check active_tasks for long-running tasks (potential stuck indicator)
        # Each task in active_tasks has a started_at timestamp
        active_tasks = state.get("active_tasks", [])
        now = time.time()
        if isinstance(active_tasks, list):
            for task_info in active_tasks:
                if not isinstance(task_info, dict):
                    continue
                task_id = str(task_info.get("task_id", ""))[:8]
                started_at_str = task_info.get("started_at")
                if started_at_str:
                    try:
                        started_dt = datetime.fromisoformat(str(started_at_str))
                        task_age_s = (datetime.now() - started_dt).total_seconds()
                        # For meta tasks, long runtime is expected (model loads can take minutes)
                        # For regular tasks, extended runtime without heartbeat progress is suspicious
                        # Use PAIR_TASK_HEARTBEAT_STALE_SECONDS as the staleness threshold
                        # But only flag if GPU heartbeat is fresh but task seems stuck
                        task_class = str(task_info.get("task_class", "")).strip()
                        if task_class != "meta" and task_age_s > PAIR_TASK_HEARTBEAT_STALE_SECONDS * 5:
                            # Task running 5x longer than heartbeat threshold - potentially stuck
                            issues.append(f"long_running_task:{task_id}:{int(task_age_s)}s")
                    except Exception:
                        pass

        # 3. Check for high fail rate indicating systemic issues
        stats = state.get("stats", {})
        if isinstance(stats, dict):
            failed = stats.get("tasks_failed", 0)
            completed = stats.get("tasks_completed", 0)
            total = failed + completed
            if total > 0:
                fail_rate = failed / total
                if fail_rate > 0.3:
                    issues.append(f"high_fail_rate:{fail_rate:.0%}")

                # 4. Check for recent retry burst (proxy for stuck/flaky behavior)
                # If many failures recently relative to completions, likely stuck
                if failed >= PAIR_MAX_RECENT_RETRIES and completed < failed:
                    issues.append(f"retry_burst:{failed}_failures")

        # 5. Check meta task phase for stuck indicators
        if has_meta_task:
            meta_phase = str(state.get("meta_task_phase", "")).strip()
            # Phases that indicate potential issues
            if "error" in meta_phase.lower() or "fail" in meta_phase.lower():
                issues.append(f"meta_error_phase:{meta_phase}")
            # Could also check meta task staleness via meta_task_id lookup in processing/

        return len(issues) == 0, issues

    def _classify_split_pair_bucket(
        self,
        group: Dict[str, Any],
        gpu_states: Dict[str, Dict[str, Any]],
    ) -> Tuple[str, Dict[str, Any]]:
        """Classify a split pair into a selection bucket.

        Buckets (in priority order):
        - "clean_open": Both members cold/clean, ready for immediate load
        - "partially_busy": One or both hot but healthy
        - "risky": Thermal pause, stale heartbeat, wedged, recent failures

        Returns (bucket, details_dict).
        """
        members = [str(m).strip() for m in group.get("members", []) if str(m).strip()]
        group_id = str(group.get("id", "")).strip()

        details = {
            "group_id": group_id,
            "members": members,
            "member_states": {},
            "risky_reasons": [],
        }

        any_risky = False
        any_hot = False
        all_cold = True

        for member in members:
            state = gpu_states.get(member, {})
            member_info = {
                "model_loaded": state.get("model_loaded", False),
                "runtime_state": state.get("runtime_state", "unknown"),
                "temperature_c": state.get("temperature_c", 0),
                "thermal_pause_active": state.get("thermal_pause_active", False),
                "heartbeat_age_s": self._get_member_heartbeat_age_seconds(state),
            }

            is_healthy, health_issues = self._is_member_healthy(state)
            member_info["healthy"] = is_healthy
            member_info["health_issues"] = health_issues

            if health_issues:
                any_risky = True
                for issue in health_issues:
                    details["risky_reasons"].append(f"{member}:{issue}")

            # Check thermal danger zone
            temp = state.get("temperature_c", 0)
            if temp and temp > PAIR_THERMAL_DANGER_C:
                any_risky = True
                details["risky_reasons"].append(f"{member}:temp_danger:{temp}C")

            if state.get("model_loaded", False):
                any_hot = True
                all_cold = False

            details["member_states"][member] = member_info

        # Classify into bucket
        if any_risky:
            bucket = "risky"
        elif all_cold:
            bucket = "clean_open"
        else:
            bucket = "partially_busy"

        details["bucket"] = bucket
        return bucket, details

    def _is_split_pair_quarantined_brain(
        self,
        group_id: str,
        gpu_states: Dict[str, Dict[str, Any]],
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check if a split pair is quarantined by reading GPU heartbeats.

        The workers track quarantine state locally. Brain checks by reading
        heartbeat data for quarantine indicators.
        """
        if not group_id:
            return False, None

        # Check brain's local quarantine tracking
        quarantined = getattr(self, 'brain_quarantined_pairs', {})
        now = time.time()

        if group_id in quarantined:
            info = quarantined[group_id]
            if now < info.get("until", 0):
                remaining = int(info["until"] - now)
                return True, {"remaining_seconds": remaining, **info}
            else:
                # Expired - remove
                del quarantined[group_id]
                self.brain_quarantined_pairs = quarantined

        return False, None

    def _record_split_failure_brain(self, group_id: str, reason: str):
        """Record a split failure for brain-side quarantine tracking."""
        if not group_id:
            return

        now = time.time()
        failures = getattr(self, 'brain_split_failures', {})
        if group_id not in failures:
            failures[group_id] = []

        failures[group_id].append({"timestamp": now, "reason": reason})

        # Prune old failures (use 10-minute window)
        cutoff = now - 600
        failures[group_id] = [f for f in failures[group_id] if f["timestamp"] > cutoff]
        self.brain_split_failures = failures

        # Check if quarantine threshold reached (3 failures)
        if len(failures[group_id]) >= 3:
            quarantined = getattr(self, 'brain_quarantined_pairs', {})
            quarantined[group_id] = {
                "entered_at": now,
                "until": now + BRAIN_SPLIT_QUARANTINE_COOLDOWN_SECONDS,
                "failure_count": len(failures[group_id]),
                "reason": reason,
            }
            self.brain_quarantined_pairs = quarantined
            self.log_decision(
                "SPLIT_PAIR_QUARANTINE_ENTER",
                f"Split pair {group_id} quarantined after {len(failures[group_id])} failures",
                {"group_id": group_id, "cooldown_seconds": BRAIN_SPLIT_QUARANTINE_COOLDOWN_SECONDS},
            )

    def _process_recovery_fallback_signals(self):
        """Process recovery fallback signals from GPU workers.

        Workers emit *.recovery_fallback.json signals when auto-recovery Stage D
        triggers. Signals are observation-only; brain derives the actual recovery tasks.
        """
        processed = []
        for signal_file in self.signals_path.glob("*.recovery_fallback.json"):
            try:
                with open(signal_file) as f:
                    signal = json.load(f)

                signal_type = signal.get("type", "")
                if signal_type not in {"split_recovery_fallback", "split_recovery_observation"}:
                    continue

                group_id = signal.get("group_id", "")
                worker = signal.get("worker", "")
                members = [str(m).strip() for m in signal.get("members", []) if str(m).strip()]
                if not members and group_id:
                    members = self._split_group_members_for_group_id(group_id)
                issue_code = str(signal.get("issue_code", "")).strip() or "recovery_fallback"

                self.log_decision(
                    "RECOVERY_FALLBACK_SIGNAL",
                    f"Processing recovery fallback from {worker} for group {group_id}",
                    {"signal": signal},
                )

                # Record failure for quarantine tracking
                if group_id:
                    self._record_split_failure_brain(group_id, issue_code)

                # Brain derives the remediation plan from the observation.
                if group_id:
                    self._insert_resource_task(
                        "unload_split_llm",
                        meta={"group_id": group_id},
                    )
                for member in members:
                    self._insert_resource_task(
                        "unload_llm",
                        meta={"candidate_workers": [member]},
                    )

                processed.append(str(signal_file))

                # Remove processed signal
                signal_file.unlink()

            except Exception as e:
                self.logger.warning(f"Error processing recovery signal {signal_file}: {e}")
                try:
                    signal_file.unlink()  # Remove bad signal
                except Exception:
                    pass

        if processed:
            self.log_decision(
                "RECOVERY_FALLBACK_PROCESSED",
                f"Processed {len(processed)} recovery fallback signals",
                {"files": processed},
            )

    def _check_thermal_recovery_escalation(self, gpu_states: Dict[str, Dict[str, Any]]):
        """Brain-level thermal recovery controller.

        Monitors GPU heartbeats for sustained CPU overheat conditions and issues
        targeted reset_gpu_runtime tasks to recover. Escalates to full orchestrator
        reset if targeted resets don't resolve the issue.

        Trigger: any GPU with cpu_temp_c >= warning AND sustained_seconds >= threshold
        Actions:
          1. Issue one targeted reset_gpu_runtime every reset_interval seconds
          2. Target selection: hottest eligible GPU first
          3. Max targeted resets per incident: configurable (default 5)
          4. If still overheated after max resets: trigger full orchestrator reset
        """
        now = time.time()

        # Find GPUs with active thermal incidents (sustained overheat)
        incident_gpus = []
        for gpu_name, state in gpu_states.items():
            incident_id = state.get("thermal_overheat_incident_id")
            sustained_seconds = int(state.get("thermal_overheat_sustained_seconds", 0) or 0)
            cpu_temp = state.get("cpu_temp_c")

            if incident_id and sustained_seconds >= self.thermal_recovery_trigger_seconds:
                incident_gpus.append({
                    "gpu": gpu_name,
                    "incident_id": incident_id,
                    "sustained_seconds": sustained_seconds,
                    "cpu_temp_c": cpu_temp,
                    "gpu_temp_c": state.get("temperature_c"),
                })

        # No active thermal incidents meeting trigger threshold
        if not incident_gpus:
            # Clear brain-level incident if it was active
            if self.thermal_recovery_active_incident_id:
                self.log_decision(
                    "THERMAL_INCIDENT_BRAIN_CLEARED",
                    f"Thermal incident cleared: {self.thermal_recovery_active_incident_id}",
                    {
                        "incident_id": self.thermal_recovery_active_incident_id,
                        "total_resets_issued": self.thermal_recovery_resets_issued,
                        "duration_seconds": int(now - (self.thermal_recovery_incident_started_at or now)),
                    },
                )
                self.thermal_recovery_active_incident_id = None
                self.thermal_recovery_incident_started_at = None
                self.thermal_recovery_resets_issued = 0
                self.thermal_recovery_last_reset_at = None
                self.thermal_recovery_last_reset_gpu = None
            return

        # Sort by hottest CPU temp first (descending)
        incident_gpus.sort(key=lambda x: x.get("cpu_temp_c") or 0, reverse=True)

        # Start or continue brain-level incident tracking
        # Use the first incident_id from hottest GPU as canonical
        canonical_incident_id = incident_gpus[0]["incident_id"]
        if self.thermal_recovery_active_incident_id != canonical_incident_id:
            # New incident started
            self.thermal_recovery_active_incident_id = canonical_incident_id
            self.thermal_recovery_incident_started_at = now
            self.thermal_recovery_resets_issued = 0
            self.thermal_recovery_last_reset_at = None
            self.thermal_recovery_last_reset_gpu = None
            self.log_decision(
                "THERMAL_INCIDENT_BRAIN_START",
                f"Brain tracking thermal incident: {canonical_incident_id}",
                {
                    "incident_id": canonical_incident_id,
                    "affected_gpus": [g["gpu"] for g in incident_gpus],
                    "hottest_cpu_c": incident_gpus[0].get("cpu_temp_c"),
                    "trigger_threshold_seconds": self.thermal_recovery_trigger_seconds,
                },
            )

        # Check if we've exceeded max targeted resets
        if self.thermal_recovery_resets_issued >= self.thermal_recovery_max_resets:
            # Escalate to full orchestrator reset if enabled
            if self.thermal_recovery_enable_full_reset:
                # Check cooldown
                if self.thermal_recovery_full_reset_at:
                    elapsed_since_full = now - self.thermal_recovery_full_reset_at
                    if elapsed_since_full < self.thermal_recovery_full_reset_cooldown:
                        remaining = int(self.thermal_recovery_full_reset_cooldown - elapsed_since_full)
                        self.log_decision(
                            "THERMAL_FULL_RESET_COOLDOWN",
                            f"Full reset cooldown active ({remaining}s remaining)",
                            {
                                "incident_id": canonical_incident_id,
                                "cooldown_seconds": remaining,
                                "resets_issued": self.thermal_recovery_resets_issued,
                            },
                        )
                        return

                # Issue full orchestrator reset
                self.log_decision(
                    "THERMAL_FULL_RESET_TRIGGERED",
                    f"Thermal recovery escalating to full orchestrator reset",
                    {
                        "incident_id": canonical_incident_id,
                        "resets_issued": self.thermal_recovery_resets_issued,
                        "max_resets": self.thermal_recovery_max_resets,
                        "affected_gpus": [g["gpu"] for g in incident_gpus],
                    },
                )
                self._insert_resource_task(
                    "orchestrator_full_reset",
                    meta={
                        "reason": "thermal_recovery_escalation",
                        "incident_id": canonical_incident_id,
                        "resets_attempted": self.thermal_recovery_resets_issued,
                    },
                )
                self.thermal_recovery_full_reset_at = now
            else:
                self.log_decision(
                    "THERMAL_ESCALATION_DISABLED",
                    f"Max targeted resets reached but full reset disabled",
                    {
                        "incident_id": canonical_incident_id,
                        "resets_issued": self.thermal_recovery_resets_issued,
                    },
                )
            return

        # Check if we're within reset interval
        if self.thermal_recovery_last_reset_at:
            elapsed_since_reset = now - self.thermal_recovery_last_reset_at
            if elapsed_since_reset < self.thermal_recovery_reset_interval_seconds:
                # Still within interval, no action
                return

        # Select target GPU for reset (hottest that's not in backoff)
        target_gpu = None
        for gpu_info in incident_gpus:
            gpu_name = gpu_info["gpu"]
            last_reset = self.thermal_recovery_gpu_last_reset.get(gpu_name)
            if last_reset:
                elapsed = now - last_reset
                if elapsed < self.thermal_recovery_same_gpu_backoff:
                    # This GPU is in backoff
                    continue
            target_gpu = gpu_name
            break

        if not target_gpu:
            # All GPUs in backoff
            self.log_decision(
                "THERMAL_RESET_ALL_BACKOFF",
                f"All overheated GPUs in reset backoff",
                {
                    "incident_id": canonical_incident_id,
                    "gpus": [g["gpu"] for g in incident_gpus],
                    "backoff_seconds": self.thermal_recovery_same_gpu_backoff,
                },
            )
            return

        # Issue targeted reset_gpu_runtime task
        self.thermal_recovery_resets_issued += 1
        self.thermal_recovery_last_reset_at = now
        self.thermal_recovery_last_reset_gpu = target_gpu
        self.thermal_recovery_gpu_last_reset[target_gpu] = now

        target_info = next((g for g in incident_gpus if g["gpu"] == target_gpu), {})

        self.log_decision(
            "THERMAL_TARGETED_RESET_ISSUED",
            f"Issuing targeted reset for {target_gpu} (reset {self.thermal_recovery_resets_issued}/{self.thermal_recovery_max_resets})",
            {
                "incident_id": canonical_incident_id,
                "target_gpu": target_gpu,
                "reset_number": self.thermal_recovery_resets_issued,
                "max_resets": self.thermal_recovery_max_resets,
                "cpu_temp_c": target_info.get("cpu_temp_c"),
                "gpu_temp_c": target_info.get("gpu_temp_c"),
                "sustained_seconds": target_info.get("sustained_seconds"),
            },
        )

        self._insert_resource_task(
            "reset_gpu_runtime",
            meta={
                "target_worker": target_gpu,
                "reason": "thermal_recovery",
                "incident_id": canonical_incident_id,
                "reset_number": self.thermal_recovery_resets_issued,
            },
        )

    def _score_split_pair_for_promotion(
        self,
        group: Dict[str, Any],
        gpu_states: Dict[str, Dict[str, Any]],
        wedged_groups: set,
    ) -> Dict[str, Any]:
        """Score a split pair for promotion selection.

        Returns structured score with breakdown:
        - score: numeric score (higher = better)
        - bucket: classification bucket
        - clean_members: list of cold/clean members
        - members_needing_unload: list of members that need unload first
        - thermal_headroom: aggregate thermal margin
        - busy_state: description of busy state
        - stuck_risk: indicators of stuck/wedged state
        - time_on_task_score: elapsed time bonus (only if healthy)
        - rejection_reason: if pair should be rejected entirely
        """
        members = [str(m).strip() for m in group.get("members", []) if str(m).strip()]
        group_id = str(group.get("id", "")).strip()

        result = {
            "group_id": group_id,
            "members": members,
            "score": 0,
            "bucket": "unknown",
            "clean_members": [],
            "members_needing_unload": [],
            "thermal_headroom": 0,
            "max_temp_c": 0,
            "max_power_w": 0,
            "busy_state": "unknown",
            "stuck_risk": [],
            "time_on_task_score": 0,
            "health_bonus": 0,
            "rejection_reason": None,
            "score_breakdown": {},
        }

        # Check if group is wedged
        if group_id in wedged_groups:
            result["rejection_reason"] = "wedged_port"
            result["score"] = -10000
            return result

        # Check if group is quarantined (Stage E: repeated failure protection)
        is_quarantined, quarantine_info = self._is_split_pair_quarantined_brain(group_id, gpu_states)
        if is_quarantined:
            result["rejection_reason"] = f"quarantined:{quarantine_info.get('remaining_seconds', 0)}s_remaining"
            result["quarantine_info"] = quarantine_info
            result["score"] = -10000
            return result

        # Classify bucket first
        bucket, bucket_details = self._classify_split_pair_bucket(group, gpu_states)
        result["bucket"] = bucket
        result["risky_reasons"] = bucket_details.get("risky_reasons", [])

        # Gather member data
        temps = []
        powers = []
        all_healthy = True
        all_tasks_healthy = True

        for member in members:
            state = gpu_states.get(member, {})

            # Temperature
            temp = state.get("temperature_c", 0)
            if temp:
                temps.append(temp)

            # Power
            power = state.get("power_draw_w", 0)
            if power:
                powers.append(power)

            # Check if member needs unload
            if state.get("model_loaded", False):
                # Check if it's already in this split group (rejoin case)
                if str(state.get("runtime_placement", "")) == "split_gpu":
                    if str(state.get("runtime_group_id", "")).strip() == group_id:
                        # Same group - no unload needed
                        pass
                    else:
                        result["members_needing_unload"].append(member)
                else:
                    result["members_needing_unload"].append(member)
            else:
                result["clean_members"].append(member)

            # Health check
            is_healthy, _ = self._is_member_healthy(state)
            if not is_healthy:
                all_healthy = False

            # Task health check
            is_task_healthy, task_issues = self._is_member_task_healthy(state)
            if not is_task_healthy:
                all_tasks_healthy = False
                result["stuck_risk"].extend([f"{member}:{i}" for i in task_issues])

        # Compute thermal metrics
        result["max_temp_c"] = max(temps) if temps else 0
        result["max_power_w"] = max(powers) if powers else 0
        result["thermal_headroom"] = PAIR_THERMAL_WARN_C - result["max_temp_c"] if temps else 0

        # Determine busy state
        if len(result["clean_members"]) == len(members):
            result["busy_state"] = "all_cold"
        elif len(result["members_needing_unload"]) == len(members):
            result["busy_state"] = "all_hot"
        elif result["members_needing_unload"]:
            result["busy_state"] = "mixed"
        else:
            result["busy_state"] = "all_cold"

        # =====================================================================
        # Compute Score
        # =====================================================================
        score = 0
        breakdown = {}

        # 1. Unload penalty (fewer unloads = better)
        unload_penalty = len(result["members_needing_unload"]) * PAIR_WEIGHT_UNLOAD_PENALTY
        score -= unload_penalty
        breakdown["unload_penalty"] = -unload_penalty

        # 2. Thermal headroom bonus/penalty
        if result["thermal_headroom"] > 0:
            thermal_score = result["thermal_headroom"] * PAIR_WEIGHT_THERMAL_HEADROOM
            score += thermal_score
            breakdown["thermal_headroom_bonus"] = thermal_score
        else:
            thermal_penalty = abs(result["thermal_headroom"]) * PAIR_WEIGHT_THERMAL_PENALTY
            score -= thermal_penalty
            breakdown["thermal_penalty"] = -thermal_penalty

        # 3. Health bonus
        if all_healthy:
            score += PAIR_WEIGHT_HEALTH_BONUS
            breakdown["health_bonus"] = PAIR_WEIGHT_HEALTH_BONUS
            result["health_bonus"] = PAIR_WEIGHT_HEALTH_BONUS

        # 4. Stuck risk penalty
        if result["stuck_risk"]:
            stuck_penalty = len(result["stuck_risk"]) * PAIR_WEIGHT_STUCK_RISK_PENALTY
            score -= stuck_penalty
            breakdown["stuck_risk_penalty"] = -stuck_penalty

        # 5. Elapsed time bonus (only if healthy and has active work)
        if all_healthy and all_tasks_healthy and result["busy_state"] != "all_cold":
            # Look at meta task phase or active task elapsed time
            # For now, use a simple heuristic based on active workers
            for member in members:
                state = gpu_states.get(member, {})
                if state.get("meta_task_active"):
                    # Meta task running - could be closer to done
                    # We don't have exact elapsed time, so give small bonus
                    time_bonus = min(5, PAIR_ELAPSED_TIME_BONUS_CAP_MINUTES) * PAIR_WEIGHT_ELAPSED_TIME_BONUS
                    score += time_bonus
                    result["time_on_task_score"] += time_bonus
                    breakdown["elapsed_time_bonus"] = breakdown.get("elapsed_time_bonus", 0) + time_bonus

        # 6. Power tie-breaker (small weight)
        if result["max_power_w"] > 0:
            # Lower power is slightly better (less thermal buildup)
            power_score = -result["max_power_w"] * PAIR_POWER_TIEBREAK_WEIGHT
            score += power_score
            breakdown["power_tiebreak"] = power_score

        # Bucket modifier (ensure bucket order is respected)
        if bucket == "clean_open":
            score += 1000  # Strong preference
            breakdown["bucket_bonus"] = 1000
        elif bucket == "partially_busy":
            score += 500
            breakdown["bucket_bonus"] = 500
        elif bucket == "risky":
            score -= 500
            breakdown["bucket_penalty"] = -500

        result["score"] = score
        result["score_breakdown"] = breakdown

        return result

    def _select_best_split_pair(
        self,
        candidate_groups: List[Dict[str, Any]],
        gpu_states: Dict[str, Dict[str, Any]],
        wedged_groups: set,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """Select the best split pair from candidates using bucket-first + score selection.

        Implements the selection heuristic:
        1. If exactly one clean_open pair, choose immediately (fast path)
        2. If multiple clean_open pairs, choose coolest
        3. If no clean_open, score partially_busy pairs
        4. Avoid risky pairs unless no alternatives

        Returns (best_group, selection_details).
        selection_details contains all scoring info for logging.
        """
        if not candidate_groups:
            return None, {"reason": "no_candidates", "candidates": []}

        selection_details = {
            "candidate_count": len(candidate_groups),
            "candidates": [],
            "chosen": None,
            "chosen_reason": None,
            "buckets": {"clean_open": [], "partially_busy": [], "risky": []},
        }

        # Score all candidates
        scored_pairs = []
        for group in candidate_groups:
            score_result = self._score_split_pair_for_promotion(group, gpu_states, wedged_groups)
            scored_pairs.append((group, score_result))
            selection_details["candidates"].append(score_result)

            # Categorize by bucket
            bucket = score_result.get("bucket", "risky")
            if bucket in selection_details["buckets"]:
                selection_details["buckets"][bucket].append(score_result["group_id"])

        # Filter out rejected pairs
        viable_pairs = [(g, s) for g, s in scored_pairs if s.get("rejection_reason") is None]

        if not viable_pairs:
            selection_details["chosen_reason"] = "all_rejected"
            return None, selection_details

        # =====================================================================
        # Fast Path: Exactly one clean_open pair
        # =====================================================================
        clean_open_pairs = [(g, s) for g, s in viable_pairs if s.get("bucket") == "clean_open"]

        if len(clean_open_pairs) == 1:
            chosen_group, chosen_score = clean_open_pairs[0]
            selection_details["chosen"] = chosen_score["group_id"]
            selection_details["chosen_reason"] = "single_clean_open"
            selection_details["chosen_score"] = chosen_score
            return chosen_group, selection_details

        # =====================================================================
        # Multiple clean_open pairs: Choose coolest
        # =====================================================================
        if len(clean_open_pairs) > 1:
            # Sort by max_temp_c (ascending), then by max_power_w (ascending)
            clean_open_pairs.sort(key=lambda x: (x[1].get("max_temp_c", 999), x[1].get("max_power_w", 999)))
            chosen_group, chosen_score = clean_open_pairs[0]
            selection_details["chosen"] = chosen_score["group_id"]
            selection_details["chosen_reason"] = "coolest_clean_open"
            selection_details["chosen_score"] = chosen_score
            return chosen_group, selection_details

        # =====================================================================
        # No clean_open pairs: Score partially_busy pairs
        # =====================================================================
        partially_busy_pairs = [(g, s) for g, s in viable_pairs if s.get("bucket") == "partially_busy"]

        if partially_busy_pairs:
            # Sort by score (descending)
            partially_busy_pairs.sort(key=lambda x: x[1].get("score", -9999), reverse=True)
            chosen_group, chosen_score = partially_busy_pairs[0]
            selection_details["chosen"] = chosen_score["group_id"]
            selection_details["chosen_reason"] = "best_partially_busy"
            selection_details["chosen_score"] = chosen_score
            return chosen_group, selection_details

        # =====================================================================
        # Only risky pairs remain: Choose least risky
        # =====================================================================
        risky_pairs = [(g, s) for g, s in viable_pairs if s.get("bucket") == "risky"]

        if risky_pairs:
            # Sort by score (descending) - least negative wins
            risky_pairs.sort(key=lambda x: x[1].get("score", -9999), reverse=True)
            chosen_group, chosen_score = risky_pairs[0]
            selection_details["chosen"] = chosen_score["group_id"]
            selection_details["chosen_reason"] = "least_risky"
            selection_details["chosen_score"] = chosen_score
            return chosen_group, selection_details

        # Fallback: shouldn't reach here, but return first viable
        if viable_pairs:
            chosen_group, chosen_score = viable_pairs[0]
            selection_details["chosen"] = chosen_score["group_id"]
            selection_details["chosen_reason"] = "fallback_first_viable"
            selection_details["chosen_score"] = chosen_score
            return chosen_group, selection_details

        selection_details["chosen_reason"] = "no_viable_pairs"
        return None, selection_details

    # =========================================================================
    # Auto-Return to Default Policy
    # =========================================================================

    def _is_fresh_gpu_state(self, state: Dict[str, Any], max_age_seconds: Optional[float] = None) -> bool:
        """Check if a GPU heartbeat is fresh enough to trust.

        Args:
            state: GPU heartbeat state dict
            max_age_seconds: Override threshold. If None, uses self.heartbeat_stale_seconds.

        Returns True if heartbeat is recent, False if stale or unparseable.
        """
        if max_age_seconds is None:
            max_age_seconds = float(getattr(self, "heartbeat_stale_seconds", 120))
        last_updated = state.get("last_updated") or state.get("updated_at")
        if not last_updated:
            return False
        try:
            updated_dt = datetime.fromisoformat(str(last_updated))
            age_seconds = (datetime.now() - updated_dt).total_seconds()
            return age_seconds <= max_age_seconds
        except Exception:
            return False

    def _has_real_work_in_flight(
        self,
        queue_stats: Dict[str, Any],
        demand_window: Dict[str, Any],
        gpu_states: Dict[str, Dict[str, Any]],
    ) -> Tuple[bool, List[str]]:
        """Check if real work (not auto-default meta tasks) is in flight.

        Returns (has_real_work, reasons).
        Used to decide whether to abort auto-default sequence.
        """
        reasons = []

        # Active batches = real work
        if len(getattr(self, 'active_batches', {})) > 0:
            reasons.append(f"active_batches:{len(self.active_batches)}")

        # Non-meta tasks in queue = real work
        non_meta_pending = (
            int(queue_stats.get("cpu", 0) or 0) +
            int(queue_stats.get("script", 0) or 0) +
            int(queue_stats.get("llm", 0) or 0)
        )
        if non_meta_pending > 0:
            reasons.append(f"non_meta_pending:{non_meta_pending}")

        # Private tasks = real work
        total_llm = int(demand_window.get("total_llm", 0) or 0)
        if total_llm > 0:
            reasons.append(f"private_llm:{total_llm}")

        private_tasks_path = getattr(self, 'private_tasks_path', None)
        if private_tasks_path:
            try:
                private_count = sum(1 for _ in private_tasks_path.glob("*.json"))
                if private_count > 0:
                    reasons.append(f"private_tasks:{private_count}")
            except Exception:
                pass

        # GPU active_tasks (actual LLM work, not meta) = real work
        # Only trust fresh heartbeats - stale active_tasks must not trigger abort
        for gpu_name, state in gpu_states.items():
            if not self._is_fresh_gpu_state(state):
                # Stale heartbeat - cannot be trusted for real-work detection
                continue
            active_tasks = state.get("active_tasks", [])
            if isinstance(active_tasks, list) and active_tasks:
                # Check if any are non-meta tasks
                for task in active_tasks:
                    task_class = str(task.get("task_class", "")).strip() if isinstance(task, dict) else ""
                    if task_class in ("llm", "cpu", "script"):
                        reasons.append(f"active_task:{gpu_name}")
                        break

        return len(reasons) > 0, reasons

    def _is_system_globally_idle(
        self,
        queue_stats: Dict[str, Any],
        demand_window: Dict[str, Any],
        gpu_states: Dict[str, Dict[str, Any]],
    ) -> Tuple[bool, List[str]]:
        """Check if system is globally idle for auto-default policy.

        Returns (is_idle, reasons_not_idle).
        """
        reasons = []

        # 1. No active batches
        if len(getattr(self, 'active_batches', {})) > 0:
            reasons.append(f"active_batches:{len(self.active_batches)}")

        # 2. Queue empty
        if int(queue_stats.get("total_pending", 0)) > 0:
            reasons.append(f"queue_pending:{queue_stats.get('total_pending')}")

        # 3. Processing empty
        if int(queue_stats.get("processing_count", 0)) > 0:
            reasons.append(f"processing:{queue_stats.get('processing_count')}")

        # 4. No private tasks
        total_llm = int(demand_window.get("total_llm", 0) or 0)
        if total_llm > 0:
            reasons.append(f"private_llm:{total_llm}")

        # Check for private cpu/script/meta tasks
        private_tasks_path = getattr(self, 'private_tasks_path', None)
        if private_tasks_path:
            try:
                private_count = sum(1 for _ in private_tasks_path.glob("*.json"))
                if private_count > 0:
                    reasons.append(f"private_tasks:{private_count}")
            except Exception:
                pass

        # 5. No meta in queue/processing
        for meta_cmd in ["load_llm", "unload_llm", "load_split_llm", "unload_split_llm"]:
            if self._has_pending_meta_command(meta_cmd):
                reasons.append(f"pending_meta:{meta_cmd}")

        # 6. No active GPU tasks from heartbeats (only trust fresh heartbeats)
        for gpu_name, state in gpu_states.items():
            if not self._is_fresh_gpu_state(state):
                # Stale heartbeat - cannot be trusted as authoritative busy signal
                # Skip entirely - don't block idle detection based on stale data
                continue
            active_tasks = state.get("active_tasks", [])
            if isinstance(active_tasks, list) and active_tasks:
                reasons.append(f"active_tasks:{gpu_name}")
            if state.get("meta_task_active"):
                reasons.append(f"meta_task:{gpu_name}")

        return len(reasons) == 0, reasons

    def _check_auto_default_policy(
        self,
        queue_stats: Dict[str, Any],
        demand_window: Dict[str, Any],
        gpu_states: Dict[str, Dict[str, Any]],
        gpus_with_model: List[str],
        split_loaded: List[str],
    ) -> Dict[str, Any]:
        """Check and execute auto-default policy when system is idle.

        Uses a phased state machine to ensure proper sequencing:
        - Phase None: Detecting idle state
        - Phase "normalizing": Unloads issued, waiting for completion
        - Phase "default_ready": System in default state

        Returns {"triggered": bool, "managed": bool, "actions": list, "details": dict}.
        managed=True means auto-default owns the rig and normal resource logic should not run.
        """
        result = {
            "triggered": False,
            "managed": False,
            "actions": [],
            "details": {},
        }

        auto_default_enabled = getattr(self, 'auto_default_enabled', False)
        if not auto_default_enabled:
            return result

        now = datetime.now()
        current_phase = getattr(self, 'auto_default_phase', None)
        auto_default_gpu = getattr(self, 'auto_default_gpu', 'gpu-2')
        auto_default_model = getattr(self, 'auto_default_model', 'qwen2.5:7b')

        # Check if system is globally idle
        is_idle, not_idle_reasons = self._is_system_globally_idle(
            queue_stats, demand_window, gpu_states
        )

        if not is_idle:
            # System is busy - but distinguish REAL work from auto-default's own meta tasks
            has_real_work, real_work_reasons = self._has_real_work_in_flight(
                queue_stats, demand_window, gpu_states
            )

            if has_real_work:
                # Real work resumed - abort auto-default sequence
                if current_phase is not None:
                    self.log_decision(
                        "AUTO_DEFAULT_REAL_WORK_ABORT",
                        f"Real work detected, aborting auto-default (was in phase: {current_phase})",
                        {"phase": current_phase, "real_work_reasons": real_work_reasons},
                    )
                self.auto_default_last_busy_at = now
                self.auto_default_phase = None
                self.auto_default_started_at = None
                result["details"]["not_idle_reasons"] = not_idle_reasons
                result["details"]["real_work_abort"] = True
                return result
            else:
                # Busy only because auto-default's own meta tasks are in flight
                # Keep ownership, don't reset phase or timer
                if current_phase in {"normalizing", "default_ready"}:
                    result["managed"] = True
                    result["details"]["not_idle_reasons"] = not_idle_reasons
                    result["details"]["auto_default_only"] = True
                    self.log_decision(
                        "AUTO_DEFAULT_MANAGED_WAIT",
                        f"Auto-default meta tasks in flight, maintaining phase: {current_phase}",
                        {"phase": current_phase, "not_idle_reasons": not_idle_reasons},
                    )
                    # Continue to phase logic below (don't return early)
                else:
                    # Not in a managed phase yet - treat as normal busy
                    self.auto_default_last_busy_at = now
                    result["details"]["not_idle_reasons"] = not_idle_reasons
                    return result

        # Check idle duration
        idle_seconds = (now - getattr(self, 'auto_default_last_busy_at', now)).total_seconds()
        auto_default_idle_seconds = getattr(self, 'auto_default_idle_seconds', 90)

        if idle_seconds < auto_default_idle_seconds:
            result["details"]["idle_seconds"] = idle_seconds
            result["details"]["required_idle_seconds"] = auto_default_idle_seconds
            return result

        # =================================================================
        # Build unique set of loaded split group_ids (deduplicate)
        # Only include GPUs with fresh heartbeats - stale state is not authoritative
        # =================================================================
        loaded_split_groups: Dict[str, List[str]] = {}  # group_id -> list of member GPUs
        for gpu in split_loaded:
            state = gpu_states.get(gpu, {})
            if not self._is_fresh_gpu_state(state):
                # Stale heartbeat - cannot be trusted for split state inference
                continue
            group_id = state.get("runtime_group_id")
            if group_id:
                loaded_split_groups.setdefault(group_id, []).append(gpu)

        # =================================================================
        # Identify single-GPU workers that need unloading
        # Only include GPUs with fresh heartbeats - stale state is not authoritative
        # =================================================================
        single_hot_needing_unload: List[str] = []
        for gpu in gpus_with_model:
            if gpu in split_loaded:
                continue  # Handled by split unload
            state = gpu_states.get(gpu, {})
            if not self._is_fresh_gpu_state(state):
                # Stale heartbeat - cannot be trusted for single hot state inference
                continue
            if gpu == auto_default_gpu:
                # Check if it already has the default model
                loaded_model = str(state.get("loaded_model", "")).strip()
                if loaded_model == auto_default_model:
                    continue  # Already has default model - keep it
            single_hot_needing_unload.append(gpu)

        # =================================================================
        # Check if default GPU is in target state (requires fresh heartbeat)
        # =================================================================
        default_state = gpu_states.get(auto_default_gpu, {})
        default_heartbeat_fresh = self._is_fresh_gpu_state(default_state)
        default_model_loaded = str(default_state.get("loaded_model", "")).strip()
        default_placement = str(default_state.get("runtime_placement", "")).strip()
        # Fail closed: stale heartbeat cannot establish default_in_target
        default_in_target = (
            default_heartbeat_fresh and
            default_placement == "single_gpu" and
            default_model_loaded == auto_default_model
        )
        if not default_heartbeat_fresh and current_phase is not None:
            self.log_decision(
                "AUTO_DEFAULT_STALE_HEARTBEAT",
                f"Default GPU {auto_default_gpu} heartbeat is stale, cannot verify target state",
                {"gpu": auto_default_gpu, "phase": current_phase},
            )

        # =================================================================
        # Phase state machine
        # =================================================================

        # Check if system is already normalized (no splits, no wrong single-GPU models)
        system_normalized = (
            len(loaded_split_groups) == 0 and
            len(single_hot_needing_unload) == 0
        )

        # Phase "default_ready": Self-validate and repair if drifted
        if current_phase == "default_ready":
            result["managed"] = True
            # Re-validate: is system still in default state?
            if system_normalized and default_in_target:
                # Still good - maintain managed state
                result["details"]["validated"] = True
                return result
            else:
                # Drift detected - demote and repair
                self.log_decision(
                    "AUTO_DEFAULT_DRIFT",
                    f"Default state drifted, repairing (normalized={system_normalized}, in_target={default_in_target})",
                    {
                        "system_normalized": system_normalized,
                        "default_in_target": default_in_target,
                        "remaining_split_groups": list(loaded_split_groups.keys()),
                        "remaining_single_hot": single_hot_needing_unload,
                    },
                )
                # Demote to normalizing and fall through to issue repair tasks
                self.auto_default_phase = "normalizing"
                current_phase = "normalizing"
                # Fall through to normalizing logic below

        # Phase "normalizing": Check if normalization is complete
        if current_phase == "normalizing":
            if system_normalized:
                # Normalization complete - proceed to Phase B
                if default_in_target:
                    # Already in default state - mark complete
                    self.auto_default_phase = "default_ready"
                    self.auto_default_started_at = None  # Sequence complete
                    self.log_decision(
                        "AUTO_DEFAULT_COMPLETE",
                        f"System in default state: {auto_default_gpu} with {auto_default_model}",
                        {"gpu": auto_default_gpu, "model": auto_default_model},
                    )
                    result["triggered"] = True
                    result["managed"] = True
                    result["actions"].append("default_ready")
                    return result
                else:
                    # Issue load_llm for default model
                    self._insert_resource_task(
                        "load_llm",
                        meta={
                            "target_model": auto_default_model,
                            "candidate_workers": [auto_default_gpu],
                        },
                    )
                    result["triggered"] = True
                    result["actions"].append("load_default")
                    self.log_decision(
                        "AUTO_DEFAULT_LOAD",
                        f"Normalization complete, loading default model on {auto_default_gpu}",
                        {"gpu": auto_default_gpu, "model": auto_default_model},
                    )
                    # Stay in normalizing phase until load completes
                    result["managed"] = True
                    return result
            else:
                # Still normalizing - wait for unloads to complete
                result["managed"] = True
                result["details"]["waiting_for_normalization"] = True
                result["details"]["remaining_split_groups"] = list(loaded_split_groups.keys())
                result["details"]["remaining_single_hot"] = single_hot_needing_unload
                return result

        # Phase None: First trigger - start normalization
        self.log_decision(
            "AUTO_DEFAULT_TRIGGER",
            f"System idle for {idle_seconds:.0f}s, triggering auto-default normalization",
            {
                "idle_seconds": idle_seconds,
                "split_groups": list(loaded_split_groups.keys()),
                "single_hot_needing_unload": single_hot_needing_unload,
                "default_in_target": default_in_target,
            },
        )

        result["triggered"] = True

        # If already normalized and in default state, skip to complete
        if system_normalized and default_in_target:
            self.auto_default_phase = "default_ready"
            self.auto_default_started_at = None  # No sequence needed
            result["managed"] = True
            result["actions"].append("already_default")
            self.log_decision(
                "AUTO_DEFAULT_ALREADY_READY",
                f"System already in default state",
                {"gpu": auto_default_gpu, "model": auto_default_model},
            )
            return result

        # If already normalized but need to load default, just issue load
        if system_normalized and not default_in_target:
            self._insert_resource_task(
                "load_llm",
                meta={
                    "target_model": auto_default_model,
                    "candidate_workers": [auto_default_gpu],
                },
            )
            self.auto_default_phase = "normalizing"
            self.auto_default_started_at = now  # Start tracking sequence
            result["managed"] = True
            result["actions"].append("load_default")
            self.log_decision(
                "AUTO_DEFAULT_LOAD_ONLY",
                f"System normalized, loading default model on {auto_default_gpu}",
                {"gpu": auto_default_gpu, "model": auto_default_model},
            )
            return result

        # =================================================================
        # Phase A: Issue normalization tasks (unloads only)
        # =================================================================

        # Unload split groups - deduplicated by group_id
        for group_id in loaded_split_groups.keys():
            self._insert_resource_task(
                "unload_split_llm",
                meta={"group_id": group_id},
            )
            result["actions"].append(f"unload_split:{group_id}")

        # Unload single-GPU workers that are not the default
        for gpu in single_hot_needing_unload:
            self._insert_resource_task(
                "unload_llm",
                meta={"candidate_workers": [gpu]},
            )
            result["actions"].append(f"unload_single:{gpu}")

        # Transition to normalizing phase
        self.auto_default_phase = "normalizing"
        self.auto_default_started_at = now  # Start tracking sequence
        result["managed"] = True

        self.log_decision(
            "AUTO_DEFAULT_NORMALIZING",
            f"Auto-default normalization started: {len(result['actions'])} tasks issued",
            {
                "actions": result["actions"],
                "split_groups": list(loaded_split_groups.keys()),
                "single_hot": single_hot_needing_unload,
            },
        )

        return result

    def _make_resource_decisions(self, gpu_status: List[Dict], running_gpus: Dict, queue_stats: Dict):
        """Make decisions about GPU resource allocation based on current state."""
        # Process any pending recovery fallback signals first
        self._process_recovery_fallback_signals()

        # Get GPU agent states from heartbeats (moved earlier for thermal recovery)
        gpu_states = self._get_gpu_states()

        # Check for thermal recovery escalation (brain-level)
        # This takes priority over other resource decisions
        self._check_thermal_recovery_escalation(gpu_states)
        self._monitor_split_health_issues(gpu_states)
        self._monitor_global_load_owner_issues(gpu_states)

        active_gpus = [g for g in gpu_status if g["util_pct"] > 10 or g["mem_used_mb"] > 1000]
        total_power = sum(g["power_w"] for g in gpu_status)

        # gpu_states already retrieved above for thermal recovery
        gpus_with_model = [g for g, s in gpu_states.items() if s.get("model_loaded", False)]
        gpus_without_model = [g for g, s in gpu_states.items() if not s.get("model_loaded", False)]
        split_loaded = [
            g for g, s in gpu_states.items()
            if s.get("model_loaded", False) and str(s.get("runtime_placement", "")) == "split_gpu"
        ]
        max_loaded_tier = 0
        for s in gpu_states.values():
            if not s.get("model_loaded", False):
                continue
            try:
                tier = int(s.get("loaded_tier", 0))
            except Exception:
                tier = 0
            max_loaded_tier = max(max_loaded_tier, tier)

        # Track unhealthy GPUs (Ollama circuit breaker tripped)
        unhealthy_gpus = [g for g, s in gpu_states.items()
                          if not s.get("ollama_healthy", True)]
        if unhealthy_gpus:
            self.log_decision("GPU_UNHEALTHY",
                f"GPUs with unhealthy Ollama: {unhealthy_gpus}",
                {"unhealthy": unhealthy_gpus})

        # Only count healthy cold GPUs as candidates for load_llm
        healthy_cold_gpus = [g for g in gpus_without_model if g not in unhealthy_gpus]

        self.log_decision("MONITOR",
            f"GPUs active: {len(active_gpus)}/{len(gpu_status)}, "
            f"Agents: {len(running_gpus)}/{len(self.gpu_agents)}, "
            f"Queue: {queue_stats['total_pending']} (cpu:{queue_stats['cpu']}, script:{queue_stats['script']}, llm:{queue_stats['llm']}), "
            f"Processing: {queue_stats['processing_count']}, Stuck: {queue_stats['stuck_tasks']}, "
            f"Hot GPUs: {len(gpus_with_model)}, split-ready: {len(split_loaded)}",
            {
                "total_power_w": round(total_power, 1),
                "running_gpus": list(running_gpus.keys()),
                "hot_gpus": gpus_with_model,
                "split_loaded": split_loaded,
                "max_loaded_tier": max_loaded_tier,
                "queue_stats": queue_stats
            })

        # Track missing GPU agents with 3-miss tolerance
        missing_gpus = set(self.gpu_agents.keys()) - set(running_gpus.keys())

        for gpu_name in self.gpu_agents.keys():
            if gpu_name in missing_gpus:
                self.gpu_miss_count[gpu_name] = self.gpu_miss_count.get(gpu_name, 0) + 1
            else:
                self.gpu_miss_count[gpu_name] = 0

        truly_missing = [g for g in missing_gpus if self.gpu_miss_count.get(g, 0) >= self.missing_gpu_miss_threshold]
        if truly_missing and queue_stats["worker_tasks"] > 0:
            self.log_decision("GPU_MISSING",
                f"GPU agents not running: {truly_missing} ({queue_stats['worker_tasks']} tasks pending)",
                {"missing": truly_missing, "miss_counts": {g: self.gpu_miss_count.get(g, 0) for g in missing_gpus}})
            self._handle_missing_gpu_escalations(truly_missing, queue_stats)
        else:
            # If no persistent missing GPUs remain, log recovery for any tracked ones.
            self._handle_missing_gpu_escalations([], queue_stats)

        # Check for stale load_llm requests
        for task_id, request in list(self.load_llm_requests.items()):
            age = (datetime.now() - request['created_at']).total_seconds()
            if age > 60:
                gpus_still_waiting = [g for g in request['gpus_needed'] if g in gpus_without_model]
                if gpus_still_waiting:
                    self.logger.warning(
                        f"load_llm task available for {age:.0f}s but GPUs {gpus_still_waiting} still cold"
                    )
                else:
                    del self.load_llm_requests[task_id]

        # Clean up completed load_llm requests
        for task_id in list(self.load_llm_requests.keys()):
            task_in_queue = (self.queue_path / f"{task_id}.json").exists()
            task_in_processing = (self.processing_path / f"{task_id}.json").exists()
            if not task_in_queue and not task_in_processing:
                del self.load_llm_requests[task_id]

        demand_window = self._collect_llm_demand_window_snapshot()
        demand_idle = self._update_llm_demand_timers(demand_window)
        any_llm_idle_s = float(demand_idle.get("any_llm_idle_s", 0.0))
        split_llm_idle_s = float(demand_idle.get("split_llm_idle_s", 0.0))

        inserted_resource_task = False

        # =============================================================================
        # Auto-Return to Default Policy
        # When system is idle, normalize to cold + default single-GPU model
        # =============================================================================
        if getattr(self, 'auto_default_enabled', False):
            auto_default_result = self._check_auto_default_policy(
                queue_stats=queue_stats,
                demand_window=demand_window,
                gpu_states=gpu_states,
                gpus_with_model=gpus_with_model,
                split_loaded=split_loaded,
            )
            if auto_default_result.get("managed") or auto_default_result.get("triggered"):
                # Auto-default owns the rig (actively managing or just issued tasks)
                return

        # Tier/placement-aware split loading: if queue needs split runtime and none is ready,
        # enqueue a targeted load_split_llm task before generic hot/cold balancing.
        required_max_tier = int(queue_stats.get("llm_max_tier", 0) or 0)
        split_needed = int(queue_stats.get("llm_split_required", 0) or 0) > 0

        # Split convergence tracking for suppressing generic load_llm
        # These are set inside the split logic block and used to prevent useless
        # generic load_llm attempts when split capacity is already being built.
        split_tier_requested = split_needed or required_max_tier >= 2
        split_capacity_short = False
        split_convergence_active = False

        # Use demand_window (queue + processing + private) for suppression decisions.
        # This catches cases where split-tier work is in processing or private tasks,
        # not just in the public queue.
        total_llm_demand = int(demand_window.get("total_llm", 0) or 0)
        split_llm_demand = int(demand_window.get("split_llm", 0) or 0)
        demand_is_split_only = (
            total_llm_demand > 0 and split_llm_demand >= total_llm_demand
        )

        if (split_needed or required_max_tier >= 2):
            split_model_demand = queue_stats.get("llm_split_model_demand", {})
            preferred_split_models: List[str] = []
            if isinstance(split_model_demand, dict):
                preferred_split_models = [
                    str(model_id)
                    for model_id, _ in sorted(
                        split_model_demand.items(),
                        key=lambda item: (-int(item[1]), str(item[0])),
                    )
                ]

            split_model = self._choose_split_model_for_tier(
                min_tier=max(2, required_max_tier),
                preferred_models=preferred_split_models,
            )
            if split_model:
                split_groups = self._split_groups_for_model(split_model)
                viable_groups = []
                for g in split_groups:
                    members = g.get("members", [])
                    if not all(member in running_gpus for member in members):
                        continue
                    if any(member in unhealthy_gpus for member in members):
                        continue
                    viable_groups.append(g)
                if viable_groups:
                    viable_group_ids = [str(g.get("id", "")).strip() for g in viable_groups]
                    ready_groups = {
                        str(s.get("runtime_group_id", "")).strip()
                        for s in gpu_states.values()
                        if s.get("model_loaded", False)
                        and str(s.get("runtime_placement", "")) == "split_gpu"
                        and str(s.get("loaded_model", "")) == split_model
                        and str(s.get("runtime_group_id", "")).strip() in viable_group_ids
                    }
                    pending_groups = self._pending_split_group_ids(split_model, viable_group_ids)
                    reconciled_states = [
                        self._reconcile_split_group_state(group, split_model)
                        for group in viable_groups
                    ]
                    reconciled_ready_groups = {
                        str(item.get("group_id", "")).strip()
                        for item in reconciled_states
                        if str(item.get("classification", "")) == "ready_real"
                    }
                    reconciled_loading_groups = {
                        str(item.get("group_id", "")).strip()
                        for item in reconciled_states
                        if str(item.get("classification", "")) == "loading_control"
                    }
                    wedged_groups = {
                        str(item.get("group_id", "")).strip()
                        for item in reconciled_states
                        if str(item.get("classification", "")) == "wedged_port"
                    }
                    ready_groups |= reconciled_ready_groups
                    pending_groups |= reconciled_loading_groups
                    llm_split_required_count = int(queue_stats.get("llm_split_required", 0) or 0)
                    desired_group_count = min(
                        len(viable_groups),
                        max(1, llm_split_required_count),
                    )

                    # Update suppression tracking for generic load_llm
                    # split_capacity_short: we need more groups than we have ready
                    # split_convergence_active: we're already building capacity (pending/wedged)
                    split_capacity_short = bool(viable_groups) and len(ready_groups) < desired_group_count
                    split_convergence_active = split_capacity_short and bool(pending_groups or wedged_groups)

                    unavailable_group_ids = ready_groups | pending_groups
                    missing_groups = [
                        g for g in viable_groups
                        if str(g.get("id", "")).strip() not in unavailable_group_ids
                    ]
                    non_wedged_missing_groups = [
                        g for g in missing_groups
                        if str(g.get("id", "")).strip() not in wedged_groups
                    ]

                    # Build map of wedged groups that should be reclaimed
                    wedged_groups_to_reclaim = []
                    for item in reconciled_states:
                        if str(item.get("classification", "")) != "wedged_port":
                            continue
                        if item.get("should_reclaim"):
                            wedged_groups_to_reclaim.append(str(item.get("group_id", "")).strip())

                    if len(unavailable_group_ids) < desired_group_count and missing_groups:
                        if not non_wedged_missing_groups and wedged_groups:
                            # Only wedged groups remain - check if any should be reclaimed
                            if wedged_groups_to_reclaim:
                                # Trigger reclaim for the first wedged group that's ready
                                target_group_id = wedged_groups_to_reclaim[0]
                                self._record_split_reclaim_attempt(target_group_id)
                                self.log_decision(
                                    "RESOURCE_DECISION",
                                    f"Split capacity short for {split_model}, triggering reclaim for wedged group {target_group_id}",
                                    {
                                        "split_needed": split_needed,
                                        "required_max_tier": required_max_tier,
                                        "desired_group_count": desired_group_count,
                                        "ready_groups": sorted(ready_groups),
                                        "pending_groups": sorted(pending_groups),
                                        "wedged_groups": sorted(wedged_groups),
                                        "reclaiming_group": target_group_id,
                                    },
                                )
                                self._insert_resource_task(
                                    "unload_split_llm",
                                    meta={"group_id": target_group_id},
                                )
                                inserted_resource_task = True
                            else:
                                self.log_decision(
                                    "RESOURCE_DECISION",
                                    f"Split capacity short for {split_model}, but only wedged split ports remain; suppressing load_split_llm (reclaim threshold not reached or in cooldown)",
                                    {
                                        "split_needed": split_needed,
                                        "required_max_tier": required_max_tier,
                                        "desired_group_count": desired_group_count,
                                        "ready_groups": sorted(ready_groups),
                                        "pending_groups": sorted(pending_groups),
                                        "wedged_groups": sorted(wedged_groups),
                                        "wedge_counts": {gid: self.split_wedge_counts.get(gid, 0) for gid in wedged_groups},
                                    },
                                )
                        else:
                            # =================================================
                            # Split Pair Selection with Scoring Heuristics
                            # =================================================
                            # Use bucket-first + score selection instead of first-available
                            try:
                                target_group, selection_details = self._select_best_split_pair(
                                    non_wedged_missing_groups,
                                    gpu_states,
                                    wedged_groups,
                                )
                            except Exception as e:
                                # Safe fallback: use deterministic first-available
                                self.logger.warning(f"Split pair scoring failed, using fallback: {e}")
                                target_group = non_wedged_missing_groups[0]
                                selection_details = {"fallback": True, "error": str(e)}

                            if target_group is None:
                                # No viable pair found
                                self.log_decision(
                                    "RESOURCE_DECISION",
                                    f"Split pair selection found no viable candidates for {split_model}",
                                    {
                                        "split_model": split_model,
                                        "candidate_count": len(non_wedged_missing_groups),
                                        "selection_details": selection_details,
                                    },
                                )
                            else:
                                # Log the selection decision with full scoring details
                                self.log_decision(
                                    "SPLIT_PAIR_SELECTION",
                                    f"Selected split pair {selection_details.get('chosen', '?')} via {selection_details.get('chosen_reason', 'unknown')}",
                                    {
                                        "split_model": split_model,
                                        "chosen_group": selection_details.get("chosen"),
                                        "chosen_reason": selection_details.get("chosen_reason"),
                                        "chosen_score": selection_details.get("chosen_score"),
                                        "buckets": selection_details.get("buckets"),
                                        "candidate_count": selection_details.get("candidate_count"),
                                        "all_candidates": selection_details.get("candidates"),
                                    },
                                )

                                # Get members needing unload from the scoring result
                                chosen_score = selection_details.get("chosen_score", {})
                                members_needing_unload = chosen_score.get("members_needing_unload", [])

                                # Fallback: compute if not in score (shouldn't happen)
                                if not members_needing_unload:
                                    members_needing_unload = self._split_group_members_needing_unload(
                                        target_group, gpu_states
                                    )

                                if members_needing_unload:
                                    # Check if unload tasks already pending for these members
                                    pending_unloads = self._has_pending_unload_for_members(members_needing_unload)
                                    members_without_pending_unload = [
                                        m for m in members_needing_unload if not pending_unloads.get(m)
                                    ]

                                    if members_without_pending_unload:
                                        # Issue targeted unload_llm for each hot member
                                        self.log_decision(
                                            "RESOURCE_DECISION",
                                            f"Split promotion blocked: members {members_needing_unload} hot - issuing targeted unload_llm first",
                                            {
                                                "split_model": split_model,
                                                "target_group": target_group,
                                                "members_needing_unload": members_needing_unload,
                                                "pending_unloads": pending_unloads,
                                                "issuing_unload_for": members_without_pending_unload,
                                                "selection_reason": selection_details.get("chosen_reason"),
                                            }
                                        )
                                        for member in members_without_pending_unload:
                                            self._insert_resource_task(
                                                "unload_llm",
                                                meta={"candidate_workers": [member]},
                                            )
                                        inserted_resource_task = True
                                    else:
                                        # Unloads already pending - wait for them to complete
                                        self.log_decision(
                                            "RESOURCE_DECISION",
                                            f"Split promotion deferred: waiting for member unloads to complete",
                                            {
                                                "split_model": split_model,
                                                "target_group": target_group,
                                                "members_needing_unload": members_needing_unload,
                                                "pending_unloads": pending_unloads,
                                                "selection_reason": selection_details.get("chosen_reason"),
                                            }
                                        )
                                else:
                                    # All members are cold - safe to issue load_split_llm
                                    self.log_decision(
                                        "RESOURCE_DECISION",
                                        f"Need more split tier capacity for model {split_model} - inserting load_split_llm",
                                        {
                                            "split_needed": split_needed,
                                            "required_max_tier": required_max_tier,
                                            "preferred_split_models": preferred_split_models,
                                            "desired_group_count": desired_group_count,
                                            "ready_groups": sorted(ready_groups),
                                            "pending_groups": sorted(pending_groups),
                                            "wedged_groups": sorted(wedged_groups),
                                            "target_group": target_group,
                                            "selection_reason": selection_details.get("chosen_reason"),
                                            "selection_score": chosen_score.get("score") if chosen_score else None,
                                        }
                                    )
                                    self._insert_resource_task(
                                        "load_split_llm",
                                        meta={
                                            "target_model": split_model,
                                            "candidate_groups": [target_group],
                                        },
                                    )
                                    inserted_resource_task = True
        elif required_max_tier < 2 and split_loaded and not self._has_pending_meta_command("unload_split_llm"):
            split_groups_loaded = sorted({
                str(s.get("runtime_group_id", "")).strip()
                for s in gpu_states.values()
                if s.get("model_loaded", False)
                and str(s.get("runtime_placement", "")) == "split_gpu"
                and str(s.get("runtime_group_id", "")).strip()
            })
            if split_groups_loaded:
                if int(demand_window.get("total_llm", 0) or 0) > 0:
                    self.log_decision(
                        "RESOURCE_DECISION",
                        "Keeping split runtime(s) hot: LLM demand still present (queue/processing/private)",
                        {
                            "loaded_split_groups": split_groups_loaded,
                            "demand_window": demand_window,
                            "split_idle_s": round(split_llm_idle_s, 1),
                        }
                    )
                elif split_llm_idle_s < float(self.split_unload_idle_seconds):
                    self.log_decision(
                        "RESOURCE_DECISION",
                        "Deferring unload_split_llm: no LLM demand window too short",
                        {
                            "loaded_split_groups": split_groups_loaded,
                            "split_idle_s": round(split_llm_idle_s, 1),
                            "required_idle_s": self.split_unload_idle_seconds,
                            "demand_window": demand_window,
                        }
                    )
                else:
                    target_group = split_groups_loaded[0]
                    self.log_decision(
                        "RESOURCE_DECISION",
                        f"No LLM demand for {int(split_llm_idle_s)}s - inserting unload_split_llm for {target_group}",
                        {
                            "loaded_split_groups": split_groups_loaded,
                            "split_idle_s": round(split_llm_idle_s, 1),
                            "required_idle_s": self.split_unload_idle_seconds,
                            "demand_window": demand_window,
                        }
                    )
                    self._insert_resource_task(
                        "unload_split_llm",
                        meta={"group_id": target_group},
                    )
                    inserted_resource_task = True

        # Check if there's already a meta task in queue OR processing
        has_pending_resource = queue_stats.get("meta", 0) > 0 or inserted_resource_task

        if not has_pending_resource:
            for task_file in self.processing_path.glob("*.json"):
                try:
                    with open(task_file) as f:
                        task = json.load(f)
                    if task.get("task_class") == "meta":
                        has_pending_resource = True
                        self.logger.debug(f"Meta task already in processing: {task.get('command')}")
                        break
                except Exception:
                    pass

        if not has_pending_resource:
            # Suppress generic load_llm when ALL LLM demand (queue+processing+private)
            # is split-tier and split capacity is being built. This uses demand_window
            # to catch split work in processing or private tasks, not just public queue.
            # Mixed demand (any single-tier LLM work) allows generic load_llm.
            suppress_generic_single_load = (
                demand_is_split_only and (split_capacity_short or split_convergence_active)
            )

            # Need to load LLM? LLM tasks waiting but no GPUs are hot
            # Only consider healthy cold GPUs as candidates
            if (
                queue_stats["llm"] > 0
                and len(gpus_with_model) < self.max_hot_workers
                and len(healthy_cold_gpus) > 0
            ):
                if not suppress_generic_single_load:
                    self.log_decision("RESOURCE_DECISION",
                        f"LLM tasks waiting ({queue_stats['llm']}) and hot GPUs below cap "
                        f"({len(gpus_with_model)}/{self.max_hot_workers}) - inserting load_llm task",
                        {"llm_tasks": queue_stats["llm"], "cold_gpus": healthy_cold_gpus, "hot_gpus": gpus_with_model})
                    self._insert_resource_task("load_llm")
                else:
                    # Log suppression inside this branch - does NOT block later branches
                    self.log_decision(
                        "RESOURCE_DECISION",
                        "Suppressing generic load_llm: split-only demand with capacity being built",
                        {
                            "split_capacity_short": split_capacity_short,
                            "split_convergence_active": split_convergence_active,
                            "total_llm_demand": total_llm_demand,
                            "split_llm_demand": split_llm_demand,
                            "demand_is_split_only": demand_is_split_only,
                            "llm_tasks_public_queue": queue_stats["llm"],
                            "cold_gpus": healthy_cold_gpus,
                            "hot_gpus": gpus_with_model,
                        },
                    )

            # Mixed workload: keep at least one hot GPU for llm while balancing script pressure.
            elif queue_stats["llm"] > 0 and queue_stats["script"] > 0:
                llm_tasks = queue_stats["llm"]
                script_tasks = queue_stats["script"]
                hot = len(gpus_with_model)
                cold = len(healthy_cold_gpus)
                script_to_llm = script_tasks / max(llm_tasks, 1)
                llm_to_script = llm_tasks / max(script_tasks, 1)

                # If llm backlog is dominant and we have healthy cold GPUs, add hot capacity.
                # Note: Mixed queues are NOT split-only, so suppression does not apply here.
                if llm_to_script >= 1.5 and cold > 0 and hot < self.max_hot_workers:
                    self.log_decision("RESOURCE_DECISION",
                        f"Mixed queue favors LLM ({llm_tasks} llm vs {script_tasks} script) - inserting load_llm task",
                        {"llm_tasks": llm_tasks, "script_tasks": script_tasks, "hot_gpus": gpus_with_model, "cold_gpus": healthy_cold_gpus})
                    self._insert_resource_task("load_llm")

                # If script backlog is dominant and >1 GPUs are hot, free one GPU.
                elif script_to_llm >= 3.0 and hot > 1:
                    target_gpu = self._targeted_single_unload_candidate(gpu_states, unhealthy_gpus, demand_window)
                    if target_gpu:
                        self.log_decision("RESOURCE_DECISION",
                            f"Mixed queue favors script ({script_tasks} script vs {llm_tasks} llm) - inserting targeted unload_llm",
                            {"script_tasks": script_tasks, "llm_tasks": llm_tasks, "hot_gpus": gpus_with_model, "target_gpu": target_gpu, "demand_window": demand_window})
                        self._insert_resource_task("unload_llm", meta={"candidate_workers": [target_gpu]})
                    else:
                        self.log_decision("RESOURCE_DECISION",
                            "Suppressing unload_llm: hot LLM runtimes still useful for current/future LLM demand",
                            {"script_tasks": script_tasks, "llm_tasks": llm_tasks, "hot_gpus": gpus_with_model, "demand_window": demand_window})

            # Safety clamp: if hot workers exceed configured cap, cool one down.
            elif len(gpus_with_model) > self.max_hot_workers:
                target_gpu = self._targeted_single_unload_candidate(gpu_states, unhealthy_gpus, demand_window)
                if target_gpu and (
                    int(demand_window.get("total_llm", 0) or 0) == 0
                    or int(demand_window.get("min_tier", 0) or 0) > int(gpu_states.get(target_gpu, {}).get("loaded_tier", 0) or 0)
                ):
                    self.log_decision("RESOURCE_DECISION",
                        f"Hot workers exceed cap ({len(gpus_with_model)}/{self.max_hot_workers}) - inserting targeted unload_llm",
                        {"hot_gpus": gpus_with_model, "max_hot_workers": self.max_hot_workers, "target_gpu": target_gpu, "demand_window": demand_window})
                    self._insert_resource_task("unload_llm", meta={"candidate_workers": [target_gpu]})
                else:
                    self.log_decision("RESOURCE_DECISION",
                        "Suppressing unload_llm cap clamp: loaded runtimes still match demand hierarchy",
                        {"hot_gpus": gpus_with_model, "max_hot_workers": self.max_hot_workers, "demand_window": demand_window})

            # Need to unload LLM? Only script tasks but GPUs are hot
            elif queue_stats["llm"] == 0 and queue_stats["script"] > 0 and len(gpus_with_model) > 0:
                if int(demand_window.get("total_llm", 0) or 0) > 0:
                    self.log_decision("RESOURCE_DECISION",
                        "Keeping GPUs hot despite script-only public queue: private/processing LLM demand still exists",
                        {"script_tasks": queue_stats["script"], "hot_gpus": gpus_with_model, "demand_window": demand_window})
                elif any_llm_idle_s < float(self.single_unload_idle_seconds):
                    self.log_decision("RESOURCE_DECISION",
                        "Deferring unload_llm: LLM demand gap too short",
                        {"script_tasks": queue_stats["script"], "hot_gpus": gpus_with_model, "llm_idle_s": round(any_llm_idle_s, 1), "required_idle_s": self.single_unload_idle_seconds})
                else:
                    target_gpu = self._targeted_single_unload_candidate(gpu_states, unhealthy_gpus, demand_window)
                    if target_gpu:
                        self.log_decision("RESOURCE_DECISION",
                            f"Only script tasks waiting ({queue_stats['script']}) and no LLM demand for {int(any_llm_idle_s)}s - inserting targeted unload_llm task",
                            {"script_tasks": queue_stats["script"], "hot_gpus": gpus_with_model, "target_gpu": target_gpu, "llm_idle_s": round(any_llm_idle_s, 1), "demand_window": demand_window})
                        self._insert_resource_task("unload_llm", meta={"candidate_workers": [target_gpu]})
