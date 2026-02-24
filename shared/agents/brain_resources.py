"""Brain resource orchestration mixin.

Extracted from brain.py to keep split/meta resource decisions modular.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from filelock import FileLock


class BrainResourceMixin:
    def _meta_task_signature(self, task: Dict[str, Any]) -> str:
        key = {
            "command": task.get("command"),
            "target_model": task.get("target_model"),
            "group_id": task.get("group_id"),
            "candidate_groups": task.get("candidate_groups"),
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

        task = {
            "task_id": str(uuid.uuid4()),
            "type": "meta",
            "command": command,
            "batch_id": "system",
            "name": command,
            "priority": 10,  # High priority
            "task_class": "meta",
            "depends_on": [],
            "executor": "worker",
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
        """
        group_id = str(group.get("id", "")).strip()
        port = group.get("port")
        members = [str(m).strip() for m in group.get("members", []) if str(m).strip()]
        reservation = self._read_split_reservation(group_id)
        probe = self._probe_split_runtime_models(port)
        models = [str(m).strip() for m in probe.get("models", []) if str(m).strip()]
        target_loaded = target_model in models
        status = str((reservation or {}).get("status", "")).strip()

        if target_loaded:
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
                    },
                )
            return {
                "classification": "ready_real",
                "group_id": group_id,
                "port": port,
                "probe_models": models,
                "reservation_status": status or None,
            }

        if probe.get("reachable") and not target_loaded:
            # Listener exists but target model is not loaded; treat as wedged/orphan-ish.
            self.log_decision(
                "SPLIT_RECONCILE_WEDGED_PORT",
                f"Split group {group_id} port {port} reachable without target model",
                {
                    "group_id": group_id,
                    "port": port,
                    "target_model": target_model,
                    "probe_models": models,
                    "reservation_status": status or None,
                },
            )
            return {
                "classification": "wedged_port",
                "group_id": group_id,
                "port": port,
                "probe_models": models,
                "reservation_status": status or None,
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

    def _make_resource_decisions(self, gpu_status: List[Dict], running_gpus: Dict, queue_stats: Dict):
        """Make decisions about GPU resource allocation based on current state."""
        active_gpus = [g for g in gpu_status if g["util_pct"] > 10 or g["mem_used_mb"] > 1000]
        total_power = sum(g["power_w"] for g in gpu_status)

        # Get GPU agent states from heartbeats
        gpu_states = self._get_gpu_states()
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

        inserted_resource_task = False

        # Tier/placement-aware split loading: if queue needs split runtime and none is ready,
        # enqueue a targeted load_split_llm task before generic hot/cold balancing.
        required_max_tier = int(queue_stats.get("llm_max_tier", 0) or 0)
        split_needed = int(queue_stats.get("llm_split_required", 0) or 0) > 0
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

                    unavailable_group_ids = ready_groups | pending_groups
                    missing_groups = [
                        g for g in viable_groups
                        if str(g.get("id", "")).strip() not in unavailable_group_ids
                    ]
                    non_wedged_missing_groups = [
                        g for g in missing_groups
                        if str(g.get("id", "")).strip() not in wedged_groups
                    ]
                    if len(unavailable_group_ids) < desired_group_count and missing_groups:
                        if not non_wedged_missing_groups and wedged_groups:
                            self.log_decision(
                                "RESOURCE_DECISION",
                                f"Split capacity short for {split_model}, but only wedged split ports remain; suppressing load_split_llm",
                                {
                                    "split_needed": split_needed,
                                    "required_max_tier": required_max_tier,
                                    "desired_group_count": desired_group_count,
                                    "ready_groups": sorted(ready_groups),
                                    "pending_groups": sorted(pending_groups),
                                    "wedged_groups": sorted(wedged_groups),
                                },
                            )
                        else:
                            target_group = non_wedged_missing_groups[0]
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
                target_group = split_groups_loaded[0]
                self.log_decision(
                    "RESOURCE_DECISION",
                    f"No tier-2 llm demand - inserting unload_split_llm for {target_group}",
                    {"loaded_split_groups": split_groups_loaded}
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
            # Need to load LLM? LLM tasks waiting but no GPUs are hot
            # Only consider healthy cold GPUs as candidates
            if (
                queue_stats["llm"] > 0
                and len(gpus_with_model) < self.max_hot_workers
                and len(healthy_cold_gpus) > 0
            ):
                self.log_decision("RESOURCE_DECISION",
                    f"LLM tasks waiting ({queue_stats['llm']}) and hot GPUs below cap "
                    f"({len(gpus_with_model)}/{self.max_hot_workers}) - inserting load_llm task",
                    {"llm_tasks": queue_stats["llm"], "cold_gpus": healthy_cold_gpus, "hot_gpus": gpus_with_model})
                self._insert_resource_task("load_llm")

            # Mixed workload: keep at least one hot GPU for llm while balancing script pressure.
            elif queue_stats["llm"] > 0 and queue_stats["script"] > 0:
                llm_tasks = queue_stats["llm"]
                script_tasks = queue_stats["script"]
                hot = len(gpus_with_model)
                cold = len(healthy_cold_gpus)
                script_to_llm = script_tasks / max(llm_tasks, 1)
                llm_to_script = llm_tasks / max(script_tasks, 1)

                # If llm backlog is dominant and we have healthy cold GPUs, add hot capacity.
                if llm_to_script >= 1.5 and cold > 0 and hot < self.max_hot_workers:
                    self.log_decision("RESOURCE_DECISION",
                        f"Mixed queue favors LLM ({llm_tasks} llm vs {script_tasks} script) - inserting load_llm task",
                        {"llm_tasks": llm_tasks, "script_tasks": script_tasks, "hot_gpus": gpus_with_model, "cold_gpus": healthy_cold_gpus})
                    self._insert_resource_task("load_llm")

                # If script backlog is dominant and >1 GPUs are hot, free one GPU.
                elif script_to_llm >= 3.0 and hot > 1:
                    self.log_decision("RESOURCE_DECISION",
                        f"Mixed queue favors script ({script_tasks} script vs {llm_tasks} llm) - inserting unload_llm task",
                        {"script_tasks": script_tasks, "llm_tasks": llm_tasks, "hot_gpus": gpus_with_model})
                    self._insert_resource_task("unload_llm")

            # Safety clamp: if hot workers exceed configured cap, cool one down.
            elif len(gpus_with_model) > self.max_hot_workers:
                self.log_decision("RESOURCE_DECISION",
                    f"Hot workers exceed cap ({len(gpus_with_model)}/{self.max_hot_workers}) - inserting unload_llm task",
                    {"hot_gpus": gpus_with_model, "max_hot_workers": self.max_hot_workers})
                self._insert_resource_task("unload_llm")

            # Need to unload LLM? Only script tasks but GPUs are hot
            elif queue_stats["llm"] == 0 and queue_stats["script"] > 0 and len(gpus_with_model) > 0:
                self.log_decision("RESOURCE_DECISION",
                    f"Only script tasks waiting ({queue_stats['script']}) but GPUs hot - inserting unload_llm task",
                    {"script_tasks": queue_stats["script"], "hot_gpus": gpus_with_model})
                self._insert_resource_task("unload_llm")

