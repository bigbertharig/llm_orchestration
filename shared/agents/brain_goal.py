#!/usr/bin/env python3
"""Goal-driven planning helpers for Brain."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


class BrainGoalMixin:
    def _replace_goal_dep(self, batch_id: str, template_name: str):
        """
        Replace dependency on a goal template name with the virtual
        _goal_complete dependency in downstream private tasks.
        """
        for task_file in self.private_tasks_path.glob("*.json"):
            try:
                with open(task_file) as f:
                    task = json.load(f)
                if task.get("batch_id") != batch_id:
                    continue
                depends_on = task.get("depends_on", [])
                if template_name in depends_on:
                    new_deps = [d for d in depends_on if d != template_name]
                    if "_goal_complete" not in new_deps:
                        new_deps.append("_goal_complete")
                    task["depends_on"] = new_deps
                    with open(task_file, 'w') as f:
                        json.dump(task, f, indent=2)
                    self.log_decision("GOAL_DEP_REPLACED",
                        f"Task '{task.get('name')}': replaced dep '{template_name}' with '_goal_complete'",
                        {"task": task.get("name"), "new_deps": new_deps})
            except Exception as e:
                self.logger.error(f"Error replacing goal dep: {e}")

    def _maybe_spawn_initial_wave(self, batch_id: str):
        """
        After all goal foreach templates are saved, read the candidate pool
        and spawn the initial wave of candidate pipelines.
        """
        batch_meta = self.active_batches.get(batch_id, {})
        goal = batch_meta.get("goal")
        if not goal:
            return

        # Only spawn once — check if pool path is already set
        if goal.get("candidate_pool_path"):
            return

        templates = goal.get("templates", {})
        if not templates:
            return

        # Find the foreach spec from any template to get the pool path
        foreach_spec = None
        for tmpl in templates.values():
            if tmpl.get("foreach"):
                foreach_spec = tmpl["foreach"]
                break

        if not foreach_spec:
            return

        # Store the pool path and read the pool size
        goal["candidate_pool_path"] = foreach_spec
        pool = self._read_candidate_pool(goal)
        goal["candidates_total"] = len(pool)

        if not pool:
            self.logger.error(f"Goal batch {batch_id}: candidate pool is empty")
            goal["status"] = "exhausted"
            self._save_brain_state()
            return

        self.log_decision("GOAL_POOL_LOADED",
            f"Loaded {len(pool)} candidates for goal batch {batch_id}",
            {"pool_size": len(pool), "target": goal["target"]})

        # Spawn initial wave = target count
        self._spawn_goal_candidates(batch_id, goal["target"])

    def _read_candidate_pool(self, goal: Dict) -> List:
        """Read the candidate pool from manifest.json for a goal-driven batch."""
        pool_path = goal.get("candidate_pool_path", "")
        if not pool_path or ":" not in pool_path:
            return []

        file_path, json_path = pool_path.rsplit(":", 1)
        try:
            with open(file_path) as f:
                data = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to read candidate pool {file_path}: {e}")
            return []

        items = data
        for key in json_path.split("."):
            if isinstance(items, dict) and key in items:
                items = items[key]
            else:
                self.logger.error(f"JSON path '{json_path}' not found in {file_path}")
                return []

        return items if isinstance(items, list) else []

    def _spawn_goal_candidates(self, batch_id: str, count: int):
        """
        Create per-candidate pipeline tasks for goal-driven plans.

        Spawns `count` candidates from the pool, each getting the full
        template pipeline (e.g., plan_person_queries → scrape_person → score_person).
        Idempotent: skips candidates already in spawned_ids.
        """
        batch_meta = self.active_batches.get(batch_id, {})
        goal = batch_meta.get("goal")
        if not goal or goal["status"] not in ("active",):
            return

        # Throttle: don't exceed target
        headroom = goal["target"] - goal["accepted"] - len(goal["in_flight_ids"])
        count = min(count, headroom)
        if count <= 0:
            return

        pool = self._read_candidate_pool(goal)
        if not pool:
            self.logger.warning(f"Goal batch {batch_id}: candidate pool is empty or unreadable")
            return

        variables = goal.get("variables", {})
        templates = goal.get("templates", {})
        if not templates:
            return

        # Sort templates by dependency depth for correct ordering
        template_order = self._resolve_template_order(templates)

        spawned_this_call = 0
        while spawned_this_call < count and goal["next_index"] < len(pool):
            idx = goal["next_index"]
            item = pool[idx]
            candidate_id = item.get("id", str(idx)) if isinstance(item, dict) else str(idx)

            goal["next_index"] += 1

            # Idempotency guard
            if candidate_id in goal["spawned_ids"]:
                continue

            # Create tasks for each template in pipeline order
            for template_name in template_order:
                template = templates[template_name]
                command = template.get("command", "")
                depends_on_raw = list(template.get("depends_on", []))

                # Substitute {ITEM.field} in command
                if isinstance(item, dict):
                    for key, value in item.items():
                        command = command.replace(f"{{ITEM.{key}}}", str(value))
                else:
                    command = command.replace("{ITEM}", str(item))

                # Substitute plan variables in command
                for var, value in variables.items():
                    command = command.replace(var, value)

                # Build per-item dependencies
                item_depends = []
                for dep in depends_on_raw:
                    dep_name = dep
                    if isinstance(item, dict):
                        for key, value in item.items():
                            dep_name = dep_name.replace(f"{{ITEM.{key}}}", str(value))
                    else:
                        dep_name = dep_name.replace("{ITEM}", str(item))
                    if dep_name and dep_name.lower() != "none":
                        item_depends.append(dep_name)

                task_name = f"{template_name}_{candidate_id}"

                task = self.create_task(
                    task_type="shell",
                    command=command,
                    batch_id=batch_id,
                    task_name=task_name,
                    priority=5,
                    depends_on=item_depends,
                    executor=template.get("executor", "worker"),
                    task_class=template.get("task_class"),
                    vram_estimate_mb=template.get("vram_estimate_mb"),
                    vram_estimate_source=template.get("vram_estimate_source")
                )
                task["goal_candidate_id"] = candidate_id

                # Add plan/batch path metadata if available
                if batch_meta.get("plan_dir"):
                    task["plan_path"] = batch_meta["plan_dir"]
                if batch_meta.get("batch_dir"):
                    task["batch_path"] = batch_meta["batch_dir"]
                if batch_meta.get("env_manifest_path"):
                    task["env_manifest_path"] = batch_meta["env_manifest_path"]

                # Release if deps are met, otherwise keep private
                satisfied = self.get_satisfied_task_ids(batch_id)
                if all(d in satisfied for d in item_depends):
                    self.save_to_public(task)
                else:
                    self.save_to_private(task)

                self.log_decision("GOAL_TASK_CREATED",
                    f"Goal candidate '{candidate_id}': created {task_name}",
                    {"task_id": task["task_id"][:8], "candidate_id": candidate_id,
                     "depends_on": item_depends})

            goal["spawned_ids"].append(candidate_id)
            goal["in_flight_ids"].append(candidate_id)
            spawned_this_call += 1

        if spawned_this_call > 0:
            self.log_decision("GOAL_SPAWN",
                f"Spawned {spawned_this_call} candidate pipeline(s) for batch {batch_id}",
                {"spawned": spawned_this_call, "total_spawned": len(goal["spawned_ids"]),
                 "in_flight": len(goal["in_flight_ids"]), "accepted": goal["accepted"],
                 "next_index": goal["next_index"]})
            self._save_brain_state()

    def _resolve_template_order(self, templates: Dict) -> List[str]:
        """
        Sort goal templates by dependency depth so tasks are created in the
        right order (upstream before downstream).
        """
        template_names = set(templates.keys())
        order = []
        remaining = dict(templates)

        # Simple topological sort: emit templates whose deps are all resolved
        for _ in range(len(templates) + 1):
            if not remaining:
                break
            for name, tmpl in list(remaining.items()):
                deps = tmpl.get("depends_on", [])
                # Strip {ITEM.*} from deps for ordering purposes
                base_deps = set()
                for d in deps:
                    base = re.sub(r'_\{ITEM\.[^}]+\}', '', d)
                    if base in template_names:
                        base_deps.add(base)
                if base_deps.issubset(set(order)):
                    order.append(name)
                    del remaining[name]

        # Any remaining (circular deps) just append
        order.extend(remaining.keys())
        return order

    def _prefilter_candidate(self, batch_id: str, person_id: str) -> str:
        """
        Fast schema/score check on a completed candidate's profile.

        Returns:
            "accept"     - high confidence, all fields present → skip LLM
            "reject"     - profile missing/empty/fatally malformed → skip LLM
            "borderline" - needs brain LLM validation
        """
        batch_meta = self.active_batches.get(batch_id, {})
        batch_dir = batch_meta.get("batch_dir", "")
        profile_path = Path(batch_dir) / "results" / person_id / "profile.json"

        if not profile_path.exists():
            self.log_decision("GOAL_PREFILTER", f"Candidate '{person_id}': profile.json missing → reject",
                              {"person_id": person_id, "result": "reject"})
            return "reject"

        try:
            with open(profile_path) as f:
                profile = json.load(f)
        except Exception as e:
            self.log_decision("GOAL_PREFILTER", f"Candidate '{person_id}': profile.json unreadable → reject",
                              {"person_id": person_id, "error": str(e), "result": "reject"})
            return "reject"

        if not profile or not isinstance(profile, dict):
            return "reject"

        # Check required fields
        name = profile.get("name", "").strip()
        confidence = profile.get("confidence", "").strip().lower()

        if not name:
            self.log_decision("GOAL_PREFILTER", f"Candidate '{person_id}': missing name → reject",
                              {"person_id": person_id, "result": "reject"})
            return "reject"

        # Auto-accept high confidence with good data
        if confidence == "high":
            has_company = bool(profile.get("company", "").strip() or profile.get("title", "").strip())
            if has_company:
                self.log_decision("GOAL_PREFILTER", f"Candidate '{person_id}': high confidence → auto-accept",
                                  {"person_id": person_id, "result": "accept"})
                return "accept"

        # Auto-reject low confidence
        if confidence == "low":
            self.log_decision("GOAL_PREFILTER", f"Candidate '{person_id}': low confidence → reject",
                              {"person_id": person_id, "result": "reject"})
            return "reject"

        # Medium confidence or missing confidence → borderline
        self.log_decision("GOAL_PREFILTER", f"Candidate '{person_id}': confidence='{confidence}' → borderline (LLM)",
                          {"person_id": person_id, "confidence": confidence, "result": "borderline"})
        return "borderline"

    def _validate_candidate_llm(self, batch_id: str, person_id: str) -> tuple:
        """
        Brain LLM validation for borderline candidates.

        Returns (accepted: bool, reason: str).
        """
        batch_meta = self.active_batches.get(batch_id, {})
        goal = batch_meta.get("goal", {})
        batch_dir = batch_meta.get("batch_dir", "")
        profile_path = Path(batch_dir) / "results" / person_id / "profile.json"

        try:
            with open(profile_path) as f:
                profile_content = f.read()
        except Exception as e:
            return False, f"Could not read profile: {e}"

        query = goal.get("query", "unknown query")

        prompt = (
            f"Given the search query: \"{query}\"\n\n"
            f"Evaluate this candidate:\n{profile_content}\n\n"
            f"Is this person a good match? Consider:\n"
            f"- Relevance to the query\n"
            f"- Quality of information found\n"
            f"- Confidence level\n\n"
            f"Respond with exactly ACCEPT or REJECT on the first line, followed by a one-line reason."
        )

        try:
            response = self.think(prompt)
        except Exception as e:
            self.logger.error(f"Goal LLM validation failed for {person_id}: {e}")
            return False, f"LLM error: {e}"

        if not response:
            return False, "Empty LLM response"

        first_line = response.strip().split('\n')[0].strip().upper()
        reason = response.strip().split('\n')[1].strip() if '\n' in response.strip() else "No reason given"

        accepted = first_line.startswith("ACCEPT")
        self.log_decision("GOAL_LLM_VALIDATION",
            f"Candidate '{person_id}': LLM says {'ACCEPT' if accepted else 'REJECT'} — {reason}",
            {"person_id": person_id, "accepted": accepted, "reason": reason})

        return accepted, reason

    def _process_goal_validations(self):
        """
        Main-loop hook: validate completed tracked tasks for goal-driven batches.

        For each active goal batch, scans for newly completed tracked tasks,
        runs pre-filter + optional LLM validation, updates counters, spawns
        replacements, and handles status transitions.
        """
        for batch_id in list(self.active_batches.keys()):
            batch_meta = self.active_batches.get(batch_id, {})
            goal = batch_meta.get("goal")
            if not goal or goal["status"] not in ("active", "draining"):
                continue

            tracked_task = goal["tracked_task"]
            max_per_cycle = goal.get("max_validations_per_cycle", 3)
            validated_task_ids = set(goal.get("validated_task_ids", []))

            # Find newly completed tracked tasks
            new_completions = []
            for task_file in self.complete_path.glob("*.json"):
                try:
                    with open(task_file) as f:
                        task = json.load(f)
                    if task.get("batch_id") != batch_id:
                        continue
                    task_name = task.get("name", "")
                    task_id = task.get("task_id", "")
                    if not task_name.startswith(f"{tracked_task}_"):
                        continue
                    if task_id in validated_task_ids:
                        continue
                    # Check if task succeeded
                    result = task.get("result", {})
                    if not result.get("success", False):
                        new_completions.append((task_id, task_name, False))
                    else:
                        new_completions.append((task_id, task_name, True))
                except Exception:
                    continue

            if not new_completions:
                continue

            # Cap validations per cycle
            new_completions = new_completions[:max_per_cycle]

            rejected_this_cycle = 0
            for task_id, task_name, task_succeeded in new_completions:
                # Extract person_id from task name (e.g., score_person_john_doe → john_doe)
                person_id = task_name[len(f"{tracked_task}_"):]

                if not task_succeeded:
                    accepted = False
                    reason = "tracked task failed"
                else:
                    # Run pre-filter
                    prefilter_result = self._prefilter_candidate(batch_id, person_id)

                    if prefilter_result == "accept":
                        accepted = True
                        reason = "pre-filter: high confidence"
                    elif prefilter_result == "reject":
                        accepted = False
                        reason = "pre-filter: rejected"
                    else:
                        # Borderline — brain LLM validates
                        accepted, reason = self._validate_candidate_llm(batch_id, person_id)

                # Update goal state
                goal["validated_task_ids"].append(task_id)
                if person_id in goal["in_flight_ids"]:
                    goal["in_flight_ids"].remove(person_id)

                if accepted:
                    goal["accepted"] += 1
                    goal["accepted_ids"].append(person_id)
                    self.log_decision("GOAL_ACCEPTED",
                        f"Candidate '{person_id}' ACCEPTED ({goal['accepted']}/{goal['target']}): {reason}",
                        {"person_id": person_id, "accepted": goal["accepted"],
                         "target": goal["target"], "reason": reason})
                else:
                    goal["rejected"] += 1
                    goal["rejected_ids"].append(person_id)
                    rejected_this_cycle += 1
                    self.log_decision("GOAL_REJECTED",
                        f"Candidate '{person_id}' REJECTED ({goal['rejected']} total): {reason}",
                        {"person_id": person_id, "rejected": goal["rejected"],
                         "reason": reason})

            # Spawn replacements for rejections (if still active)
            if rejected_this_cycle > 0 and goal["status"] == "active":
                self._spawn_goal_candidates(batch_id, rejected_this_cycle)

            # Status transitions
            target = goal["target"]
            tolerance = goal["tolerance"]
            in_flight = len(goal["in_flight_ids"])

            if goal["status"] == "active" and goal["accepted"] >= target - tolerance:
                goal["status"] = "draining"
                self.log_decision("GOAL_DRAINING",
                    f"Goal reached draining: {goal['accepted']}/{target} accepted, {in_flight} in-flight",
                    {"accepted": goal["accepted"], "target": target, "in_flight": in_flight})

            if goal["status"] == "draining" and in_flight == 0:
                goal["status"] = "complete"
                self.log_decision("GOAL_COMPLETE",
                    f"Goal complete: {goal['accepted']}/{target} accepted",
                    {"accepted": goal["accepted"], "target": target,
                     "rejected": goal["rejected"]})
                self._release_goal_final_task(batch_id)

            # Circuit breaker: pool exhausted
            total_attempted = goal["accepted"] + goal["rejected"]
            if (goal["status"] == "active" and
                    goal["next_index"] >= goal["candidates_total"] and
                    in_flight == 0 and
                    goal["accepted"] < target - tolerance):
                goal["status"] = "exhausted"
                self.log_decision("GOAL_EXHAUSTED",
                    f"Goal exhausted: pool empty, only {goal['accepted']}/{target} accepted",
                    {"accepted": goal["accepted"], "target": target,
                     "rejected": goal["rejected"], "pool_size": goal["candidates_total"]})
                self._release_goal_final_task(batch_id)

            # Circuit breaker: max attempts
            if goal["status"] == "active" and total_attempted >= goal["max_attempts"]:
                goal["status"] = "exhausted"
                self.log_decision("GOAL_CIRCUIT_BREAKER",
                    f"Goal circuit breaker: {total_attempted} attempts >= {goal['max_attempts']} max",
                    {"total_attempted": total_attempted, "max_attempts": goal["max_attempts"],
                     "accepted": goal["accepted"]})
                self._release_goal_final_task(batch_id)

            self._save_brain_state()

    def _release_goal_final_task(self, batch_id: str):
        """Release compile_output (or equivalent final task) when goal is met/exhausted."""
        batch_meta = self.active_batches.get(batch_id, {})
        goal = batch_meta.get("goal", {})
        compile_task_id = goal.get("compile_output_task_id", "")

        if not compile_task_id:
            self.logger.warning(f"Goal batch {batch_id}: no compile_output_task_id to release")
            return

        # Find and release the task from private
        task_file = self.private_tasks_path / f"{compile_task_id}.json"
        if task_file.exists():
            try:
                with open(task_file) as f:
                    task = json.load(f)
                # Clear the _goal_complete dep since we're releasing explicitly
                task["depends_on"] = [d for d in task.get("depends_on", []) if d != "_goal_complete"]
                task_file.unlink()
                self.save_to_public(task)
                self.log_decision("GOAL_FINAL_RELEASED",
                    f"Released final task '{task.get('name')}' for goal batch {batch_id}",
                    {"task_id": compile_task_id[:8], "task_name": task.get("name")})
            except Exception as e:
                self.logger.error(f"Failed to release goal final task: {e}")
        else:
            self.logger.debug(f"Goal final task {compile_task_id[:8]} not in private (may be already released)")

    def _check_batch_completion(self, batch_id: str):
        """Check if a batch is fully complete."""
        batch_meta = self.active_batches.get(batch_id, {})
        goal = batch_meta.get("goal")

        # Goal-driven batches: don't complete until goal is done
        if goal and goal.get("status") in ("active", "draining"):
            return  # Goal still in progress

        # Any tasks still in queue or processing?
        for path in [self.queue_path, self.processing_path]:
            for task_file in path.glob("*.json"):
                try:
                    with open(task_file) as f:
                        task = json.load(f)
                        if task.get("batch_id") == batch_id:
                            return  # Still has pending tasks
                except:
                    pass

        # Any private tasks left?
        if self.get_private_tasks(batch_id):
            return

        # Batch is complete
        completion_details = {
            "batch_id": batch_id,
            "plan": batch_meta.get("plan")
        }
        if goal:
            completion_details["goal_accepted"] = goal.get("accepted", 0)
            completion_details["goal_target"] = goal.get("target", 0)
            completion_details["goal_rejected"] = goal.get("rejected", 0)
            completion_details["goal_status"] = goal.get("status", "")

        self.log_decision("BATCH_COMPLETE", f"Batch {batch_id} finished successfully",
                          completion_details)
        del self.active_batches[batch_id]
        self._save_brain_state()

    def _parse_goal_section(self, plan_content: str) -> Optional[Dict[str, Any]]:
        """
        Parse the optional ## Goal section from a plan.md file.

        Returns a dict with goal metadata if found, None otherwise.
        Backward-compatible: plans without ## Goal are unaffected.
        """
        # Find the ## Goal section
        goal_match = re.search(r'\n## Goal\s*\n(.*?)(?=\n## |\Z)', plan_content, re.DOTALL)
        if not goal_match:
            return None

        goal_text = goal_match.group(1)
        goal = {}

        for line in goal_text.strip().split('\n'):
            line = line.strip()
            if line.startswith('- **type**:'):
                goal["type"] = line.split(':', 1)[1].strip()
            elif line.startswith('- **target**:'):
                raw = line.split(':', 1)[1].strip()
                goal["target"] = raw  # Keep as string; may contain {VAR}
            elif line.startswith('- **tolerance**:'):
                try:
                    goal["tolerance"] = int(line.split(':', 1)[1].strip())
                except ValueError:
                    goal["tolerance"] = 2
            elif line.startswith('- **max_attempts_multiplier**:'):
                try:
                    goal["max_attempts_multiplier"] = int(line.split(':', 1)[1].strip())
                except ValueError:
                    goal["max_attempts_multiplier"] = 5
            elif line.startswith('- **tracked_task**:'):
                goal["tracked_task"] = line.split(':', 1)[1].strip()

        # Validate required fields
        if not goal.get("type") or not goal.get("tracked_task"):
            self.logger.warning("Goal section found but missing required fields (type, tracked_task)")
            return None

        # Defaults
        goal.setdefault("tolerance", 2)
        goal.setdefault("max_attempts_multiplier", 5)

        self.log_decision("GOAL_PARSED", f"Parsed goal: type={goal['type']}, target={goal.get('target')}, tracked={goal['tracked_task']}", goal)
        return goal

