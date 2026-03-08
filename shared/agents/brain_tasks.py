"""Brain task queue/private-task orchestration mixin.

Extracted from brain.py to keep foreach expansion and release logic isolated.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class BrainTaskQueueMixin:
    _SUPPORTED_META_COMMANDS = {
        "load_llm",
        "unload_llm",
        "load_split_llm",
        "unload_split_llm",
        "cleanup_split_runtime",
        "reset_gpu_runtime",
        "reset_split_runtime",
    }

    def _validate_meta_task_contract(self, task: Dict[str, Any]) -> str:
        command = str(task.get("command", "")).strip()
        if command not in self._SUPPORTED_META_COMMANDS:
            return f"unsupported_meta_command:{command or 'missing'}"

        if command == "load_split_llm":
            target_model = str(task.get("target_model", "")).strip()
            if not target_model:
                return "load_split_llm:missing_target_model"
            groups = task.get("candidate_groups")
            if not isinstance(groups, list) or not groups:
                return "load_split_llm:missing_candidate_groups"
            for group in groups:
                if not isinstance(group, dict):
                    return "load_split_llm:invalid_candidate_group"
                group_id = str(group.get("id", "")).strip()
                members = group.get("members")
                port = group.get("port")
                if not group_id:
                    return "load_split_llm:group_missing_id"
                if not isinstance(members, list) or not all(str(member).strip() for member in members):
                    return "load_split_llm:group_missing_members"
                try:
                    int(port)
                except Exception:
                    return "load_split_llm:group_missing_port"

        if command in {"reset_gpu_runtime", "reset_split_runtime"}:
            target_worker = str(task.get("target_worker") or task.get("target_gpu") or "").strip()
            if not target_worker:
                return f"{command}:missing_target_worker"

        if command == "cleanup_split_runtime":
            target_workers = task.get("target_workers", [])
            group_id = str(task.get("group_id", "")).strip()
            if not group_id and not (isinstance(target_workers, list) and any(str(w).strip() for w in target_workers)):
                return "cleanup_split_runtime:missing_group_or_targets"

        return ""

    def save_to_private(self, task: Dict[str, Any]):
        """Save a task to the private list (not visible to workers)."""
        task_file = self.private_tasks_path / f"{task['task_id']}.json"
        with open(task_file, 'w') as f:
            json.dump(task, f, indent=2)

    def save_to_public(self, task: Dict[str, Any]) -> bool:
        """Save a task to the public queue (workers can claim it)."""
        if task.get("task_class") == "meta":
            reason = self._validate_meta_task_contract(task)
            if reason:
                failed_task = dict(task)
                failed_task["status"] = "failed"
                failed_task["result"] = {
                    "success": False,
                    "error": reason,
                    "error_type": "meta_task_contract_error",
                }
                self.save_to_failed(failed_task)
                self.log_decision(
                    "META_TASK_INVALID",
                    f"Rejected meta task before queue release: {task.get('command', task.get('name', task['task_id'][:8]))}",
                    {"task_id": task["task_id"][:8], "reason": reason},
                )
                return False
        task_file = self.queue_path / f"{task['task_id']}.json"
        with open(task_file, 'w') as f:
            json.dump(task, f, indent=2)

        batch_id = str(task.get("batch_id", "")).strip()
        if batch_id:
            self._append_batch_event(batch_id, "task_released", self._task_payload(task))

        self.log_decision("TASK_RELEASED", f"Released task to queue: {task.get('name', task['task_id'][:8])}",
                          {"task_id": task['task_id'][:8], "depends_on": task.get('depends_on', [])})
        return True

    def save_to_failed(self, task: Dict[str, Any]):
        """Save a task directly to failed (for definition errors)."""
        task_file = self.failed_path / f"{task['task_id']}.json"
        with open(task_file, 'w') as f:
            json.dump(task, f, indent=2)
        batch_id = str(task.get("batch_id", "")).strip()
        if batch_id:
            self._append_batch_event(batch_id, "task_failed", self._task_payload(task))
            self._refresh_batch_summary(batch_id)

    def get_private_tasks(self, batch_id: str = None) -> List[Dict]:
        """Get private tasks, optionally filtered by batch."""
        tasks = []
        for task_file in self.private_tasks_path.glob("*.json"):
            try:
                with open(task_file) as f:
                    task = json.load(f)
                    if batch_id is None or task.get("batch_id") == batch_id:
                        tasks.append(task)
            except:
                pass
        return tasks

    def _batch_events_log_path(self, batch_id: str) -> Path | None:
        batch_meta = self.active_batches.get(batch_id, {}) if hasattr(self, "active_batches") else {}
        batch_dir = Path(str(batch_meta.get("batch_dir", "")).strip())
        if not batch_dir:
            return None
        return batch_dir / "logs" / "batch_events.jsonl"

    def get_completed_task_ids(self, batch_id: str) -> set:
        """Get set of completed task names for a batch."""
        completed = set()

        # Prefer the batch-local event log. It is scoped, append-only, and avoids
        # a full scan across every historical completion artifact on the shared drive.
        events_log = self._batch_events_log_path(batch_id)
        if events_log and events_log.exists():
            try:
                with open(events_log, encoding="utf-8") as f:
                    for line in f:
                        raw = str(line or "").strip()
                        if not raw:
                            continue
                        event = json.loads(raw)
                        if event.get("batch_id") != batch_id:
                            continue
                        if event.get("event") != "task_succeeded":
                            continue
                        task_name = str(event.get("task_name") or "").strip()
                        if task_name:
                            completed.add(task_name)
                return completed
            except Exception:
                # Fall back to the legacy global completion scan when the per-batch
                # history file is missing or malformed.
                completed.clear()

        for task_file in self.complete_path.glob("*.json"):
            try:
                with open(task_file) as f:
                    task = json.load(f)
                if task.get("batch_id") != batch_id:
                    continue
                result = task.get("result", {})
                if result.get("success", False):
                    name = str(task.get("name", "")).strip()
                    if name:
                        completed.add(name)
            except Exception:
                continue
        return completed

    def _complete_private_task_noop(self, task: Dict[str, Any], output: str):
        """Mark a private task complete with a synthetic no-op result."""
        task = dict(task)
        now = datetime.now().isoformat()
        task["status"] = "complete"
        task["assigned_to"] = "brain"
        task["started_at"] = task.get("started_at") or now
        task["completed_at"] = now
        task["result"] = {
            "success": True,
            "output": output,
            "handler": "brain",
            "return_code": 0,
            "elapsed_seconds": 0.0,
            "error": "",
            "error_type": "",
        }
        out = self.complete_path / f"{task['task_id']}.json"
        with open(out, "w") as f:
            json.dump(task, f, indent=2)
        batch_id = str(task.get("batch_id", "")).strip()
        if batch_id:
            self._append_batch_event(batch_id, "task_succeeded", self._task_payload(task))
            self._refresh_batch_summary(batch_id)

    def check_and_release_tasks(self):
        """Check private tasks and release any whose dependencies are met."""
        for batch_id in list(self.active_batches.keys()):
            satisfied = self.get_satisfied_task_ids(batch_id)
            private_tasks = self.get_private_tasks(batch_id)
            batch_meta = self.active_batches.get(batch_id, {})
            goal = batch_meta.get("goal")

            # For goal-driven batches, add virtual _goal_complete if goal is done
            if goal and goal.get("status") in ("complete", "exhausted"):
                satisfied.add("_goal_complete")

            all_released = True
            goal_templates_saved_this_cycle = False
            for task in private_tasks:
                depends_on = task.get("depends_on", [])
                # Foreach templates may include per-item dependency placeholders.
                # Ignore those at template-gating time; they are applied on expansion.
                template_depends = depends_on
                if task.get("foreach"):
                    template_depends = [d for d in depends_on if "{ITEM" not in d]

                # Check if all template dependencies are satisfied
                deps_met = all(dep in satisfied for dep in template_depends)

                if deps_met:
                    task_file = self.private_tasks_path / f"{task['task_id']}.json"

                    # Check if this is a foreach task that needs expansion
                    foreach_spec = task.get("foreach")
                    if foreach_spec:
                        if task.get("goal_driven") and goal:
                            # Goal-driven foreach: save template, don't expand all at once
                            template_name = task.get("name", "")
                            goal["templates"][template_name] = {
                                "command": task.get("command", ""),
                                "depends_on": task.get("depends_on", []),
                                "executor": task.get("executor", "worker"),
                                "task_class": task.get("task_class"),
                                "priority": task.get("priority", 5),
                                "batch_priority": task.get("batch_priority", "normal"),
                                "preemptible": task.get("preemptible", True),
                                "foreach": foreach_spec,
                                "batch_size": task.get("batch_size", 1),
                                "vram_estimate_mb": task.get("vram_estimate_mb"),
                                "vram_estimate_source": task.get("vram_estimate_source"),
                            }
                            # Remove the template task from private
                            if task_file.exists():
                                task_file.unlink()

                            # Rewrite compile_output deps: replace this template name
                            # with virtual _goal_complete dependency
                            self._replace_goal_dep(batch_id, template_name)

                            goal_templates_saved_this_cycle = True
                            self.log_decision("GOAL_TEMPLATE_SAVED",
                                f"Saved goal template '{template_name}' (foreach intercepted)",
                                {"batch_id": batch_id, "template": template_name,
                                 "templates_saved": len(goal["templates"])})
                        else:
                            # Standard foreach: expand all items at once
                            expanded_names = self._expand_foreach_task(task, batch_id)
                            if expanded_names is None:
                                all_released = False
                            else:
                                # Remove the template task
                                if task_file.exists():
                                    task_file.unlink()
                                # Mark zero-item foreach as a no-op success so it is visible in task history.
                                if len(expanded_names) == 0:
                                    self._complete_private_task_noop(
                                        task,
                                        f"No-op: foreach expansion produced 0 items from {foreach_spec}",
                                    )
                                    self.log_decision(
                                        "FOREACH_EMPTY",
                                        f"Completed '{task.get('name')}' as no-op (0 foreach items)",
                                        {"task_id": task.get("task_id", "")[:8], "template": task.get("name")},
                                    )
                                # Update any tasks that depend on this one to depend on ALL expanded tasks
                                self._update_foreach_dependents(batch_id, task.get("name"), expanded_names)
                    else:
                        # Regular task - release to public queue
                        if task_file.exists():
                            task_file.unlink()
                            self.save_to_public(task)
                else:
                    all_released = False
                    pending_deps = [d for d in template_depends if d not in satisfied]
                    self.logger.debug(f"Task {task.get('name')} waiting on: {pending_deps}")

            # After saving goal templates, check if we should spawn initial wave
            if goal_templates_saved_this_cycle and goal:
                self._maybe_spawn_initial_wave(batch_id)

            # Check if batch is complete (no private tasks, no public/processing tasks)
            if all_released and len(private_tasks) > 0:
                self._check_batch_completion(batch_id)

    def _expand_foreach_task(self, template_task: Dict, batch_id: str) -> List[str] | None:
        """
        Expand a foreach task into N individual tasks.

        foreach format: {BATCH_PATH}/manifest.json:videos
        - Path to JSON file (with variable substitution)
        - Colon separator
        - JSON path to array (e.g., "videos" or "data.items")

        Returns list of expanded task names, or empty list on failure.
        """
        foreach_spec = template_task.get("foreach", "")
        if ":" not in foreach_spec:
            self.logger.error(f"Invalid foreach format: {foreach_spec} (expected path:jsonpath)")
            return None

        file_path, json_path = foreach_spec.rsplit(":", 1)

        # Substitute variables in file path
        batch_info = self.active_batches.get(batch_id, {})
        batch_dir = Path(batch_info.get("batch_dir", ""))
        plan_dir = Path(batch_info.get("plan_dir", ""))

        file_path = file_path.replace("{BATCH_PATH}", str(batch_dir))
        file_path = file_path.replace("{PLAN_PATH}", str(plan_dir))

        # Read the JSON file
        try:
            with open(file_path) as f:
                data = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to read foreach source {file_path}: {e}")
            return None

        # Navigate to the array using json_path (supports simple dotted paths)
        items = data
        for key in json_path.split("."):
            if isinstance(items, dict) and key in items:
                items = items[key]
            else:
                self.logger.error(f"JSON path '{json_path}' not found in {file_path}")
                return []

        if not isinstance(items, list):
            self.logger.error(f"foreach target is not a list: {json_path}")
            return None

        batch_size = max(1, int(template_task.get("batch_size", 1)))
        expanded_target = (len(items) + batch_size - 1) // batch_size
        self.log_decision(
            "FOREACH_EXPAND",
            f"Expanding '{template_task.get('name')}' into {expanded_target} task(s) from {len(items)} item(s)",
            {
                "template": template_task.get("name"),
                "item_count": len(items),
                "batch_size": batch_size,
                "expanded_count": expanded_target
            }
        )

        # Create one task per item (batch_size=1) or micro-batch task (batch_size>1)
        expanded_names = []
        template_depends = template_task.get("depends_on", [])
        for start in range(0, len(items), batch_size):
            group = items[start:start + batch_size]
            group_commands = []
            group_depends = []
            group_item_ids = []

            for i, item in enumerate(group, start=start):
                # Build item-specific command
                command = template_task.get("command", "")

                # Substitute {ITEM.field} patterns
                if isinstance(item, dict):
                    for key, value in item.items():
                        command = command.replace(f"{{ITEM.{key}}}", str(value))
                else:
                    command = command.replace("{ITEM}", str(item))
                group_commands.append(command)

                item_id = item.get("id", str(i)) if isinstance(item, dict) else str(i)
                group_item_ids.append(str(item_id))

                # Build per-item dependencies from template depends_on.
                item_depends = []
                for dep in template_depends:
                    dep_name = dep
                    if isinstance(item, dict):
                        for key, value in item.items():
                            dep_name = dep_name.replace(f"{{ITEM.{key}}}", str(value))
                    else:
                        dep_name = dep_name.replace("{ITEM}", str(item))
                    if dep_name and dep_name.lower() != "none":
                        item_depends.append(dep_name)

                for dep in item_depends:
                    if dep not in group_depends:
                        group_depends.append(dep)

            if batch_size == 1:
                task_name = f"{template_task.get('name')}_{group_item_ids[0]}"
                task_command = group_commands[0]
            else:
                task_name = f"{template_task.get('name')}_batch_{start + 1:04d}_{start + len(group):04d}"
                # Sequential micro-batch execution in one worker claim.
                task_command = "set -e\n" + "\n".join(group_commands)

            task = self.create_task(
                task_type="shell",
                command=task_command,
                batch_id=batch_id,
                task_name=task_name,
                priority=template_task.get("priority", 5),
                depends_on=group_depends,
                executor=template_task.get("executor", "worker"),
                task_class=template_task.get("task_class"),
                vram_estimate_mb=template_task.get("vram_estimate_mb"),
                vram_estimate_source=template_task.get("vram_estimate_source"),
                llm_min_tier=template_task.get("llm_min_tier"),
                llm_model=template_task.get("llm_model"),
                llm_placement=template_task.get("llm_placement"),
                batch_priority=template_task.get("batch_priority", "normal"),
                preemptible=template_task.get("preemptible", True),
            )
            # Preserve execution context required by worker env checks and path-based scripts.
            if template_task.get("plan_path"):
                task["plan_path"] = template_task.get("plan_path")
            if template_task.get("batch_path"):
                task["batch_path"] = template_task.get("batch_path")
            if template_task.get("env_manifest_path"):
                task["env_manifest_path"] = template_task.get("env_manifest_path")
            task["batch_size"] = len(group)
            task["item_ids"] = group_item_ids

            # Release if ready, otherwise keep private until dependencies are met.
            if group_depends:
                self.save_to_private(task)
            else:
                self.save_to_public(task)
            expanded_names.append(task_name)

            self.log_decision(
                "TASK_CREATED",
                f"Created expanded task: {task_name}",
                {
                    "task_id": task["task_id"][:8],
                    "batch_size": len(group),
                    "item_ids": group_item_ids,
                    "depends_on": group_depends,
                    "vram_estimate_mb": task.get("vram_estimate_mb")
                }
            )

        return expanded_names

    def _update_foreach_dependents(self, batch_id: str, original_name: str, expanded_names: List[str]):
        """
        Update tasks that depend on the foreach template to depend on ALL expanded tasks.

        e.g., if 'aggregate' depends_on ['transcribe'], and transcribe expanded to
        ['transcribe_v1', 'transcribe_v2'], update aggregate to depend on both.
        """
        for task_file in self.private_tasks_path.glob("*.json"):
            try:
                with open(task_file) as f:
                    task = json.load(f)

                if task.get("batch_id") != batch_id:
                    continue

                depends_on = task.get("depends_on", [])
                if original_name in depends_on:
                    # Replace the template name with all expanded names
                    new_deps = [d for d in depends_on if d != original_name]
                    new_deps.extend(expanded_names)
                    task["depends_on"] = new_deps

                    # Save updated task
                    with open(task_file, 'w') as f:
                        json.dump(task, f, indent=2)

                    self.log_decision("DEPS_UPDATED",
                        f"Updated '{task.get('name')}' to depend on {len(expanded_names)} expanded tasks",
                        {"task": task.get("name"), "new_deps_count": len(new_deps)})

            except Exception as e:
                self.logger.error(f"Error updating dependents: {e}")

    # =========================================================================
    # Goal-Driven Plan Support
    # =========================================================================

    def get_satisfied_task_ids(self, batch_id: str) -> set:
        """
        Get task names considered dependency-satisfied:
        - successful completed tasks
        - optionally terminal failed tasks (disabled by default)
        """
        satisfied = self.get_completed_task_ids(batch_id)
        allow_terminal = bool(
            self.config.get("retry_policy", {}).get("allow_terminal_failed_dependencies", False)
        )
        if not allow_terminal:
            return satisfied

        max_attempts = self.config.get("retry_policy", {}).get("max_attempts", 3)

        for task_file in self.failed_path.glob("*.json"):
            try:
                with open(task_file) as f:
                    task = json.load(f)
                if task.get("batch_id") != batch_id:
                    continue

                is_terminal = (
                    task.get("status") == "blocked_cloud"
                    or bool(task.get("cloud_escalated", False))
                    or int(task.get("attempts", 0)) >= int(max_attempts)
                )
                if is_terminal:
                    name = task.get("name", "")
                    if name:
                        satisfied.add(name)
            except Exception:
                continue

        return satisfied
