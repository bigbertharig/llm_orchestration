"""Brain plan parsing and task creation mixin.

Extracted from brain.py to isolate plan parsing, task creation, VRAM inference,
goal discovery ordering, and execute_plan orchestration setup.
"""

import ast
import json
import re
import shlex
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from brain_constants import PRIORITY_TIER_TO_VALUE, VALID_TASK_CLASSES, VALID_VRAM_POLICIES


class BrainPlanMixin:
    def _extract_explicit_model_flags_from_command(self, command: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        try:
            tokens = shlex.split(command or "")
        except Exception:
            tokens = str(command or "").split()
        i = 0
        while i < len(tokens):
            tok = str(tokens[i]).strip()
            if tok.startswith("--") and tok.endswith("-model") and i + 1 < len(tokens):
                key = tok[2:]
                val = str(tokens[i + 1]).strip()
                if val and not (val.startswith("{") and val.endswith("}")):
                    out[key] = val
                i += 2
                continue
            i += 1
        return out

    def _validate_split_llm_task_definition(self, task: Dict[str, Any]) -> Optional[str]:
        if task.get("task_class") != "llm":
            return None
        if str(task.get("llm_placement") or "").strip() != "split_gpu":
            return None
        primary = str(task.get("llm_model") or "").strip()
        if not primary:
            return None
        explicit_models = self._extract_explicit_model_flags_from_command(task.get("command", ""))
        if not explicit_models:
            return None
        mismatched = {k: v for k, v in explicit_models.items() if str(v).strip() and str(v).strip() != primary}
        if not mismatched:
            return None
        return (
            "split_gpu tasks must be single-model; mixed explicit model args found: "
            + ", ".join(f"{k}={v}" for k, v in sorted(mismatched.items()))
            + f" (llm_model={primary})"
        )

    def _normalize_batch_priority(self, raw_priority: Any) -> tuple[str, int]:
        """Resolve batch priority tier to normalized label + numeric value."""
        label = str(raw_priority or "normal").strip().lower()
        if label not in PRIORITY_TIER_TO_VALUE:
            label = "normal"
        return label, PRIORITY_TIER_TO_VALUE[label]

    def _normalize_preemptible(self, raw_value: Any) -> bool:
        """Resolve PREEMPTIBLE config to bool (default true)."""
        if isinstance(raw_value, bool):
            return raw_value
        if raw_value is None:
            return True
        text = str(raw_value).strip().lower()
        if text in {"false", "0", "no", "off"}:
            return False
        return True

    def create_task(self, task_type: str, command: str, batch_id: str,
                    task_name: str = "", priority: int = 5,
                    depends_on: List[str] = None, executor: str = "worker",
                    task_class: str = None,
                    vram_estimate_mb: Optional[int] = None,
                    vram_estimate_source: Optional[str] = None,
                    llm_min_tier: Optional[int] = None,
                    llm_model: Optional[str] = None,
                    llm_placement: Optional[str] = None,
                    batch_priority: str = "normal",
                    preemptible: bool = True) -> Dict[str, Any]:
        """Create a new task.

        task_class must be specified in plan.md. If missing or invalid,
        the task is created but marked for immediate failure.
        """
        definition_error = None
        if task_class is None:
            definition_error = f"missing task_class (must be one of: cpu, script, llm, brain)"
            task_class = "cpu"  # Placeholder so task structure is valid
        elif task_class not in VALID_TASK_CLASSES:
            definition_error = (
                f"invalid task_class '{task_class}' (must be one of: cpu, script, llm, brain)"
            )
            task_class = "cpu"  # Placeholder

        # Keep brain routing strict and unambiguous.
        if task_class == "brain" and executor != "brain":
            executor = "brain"
        if executor == "brain" and task_class != "brain":
            task_class = "brain"

        if task_class == "llm":
            if llm_model:
                llm_model = str(llm_model).strip()
                if llm_model and llm_model not in self.model_tier_by_id:
                    definition_error = f"unknown llm_model '{llm_model}'"
                if llm_model:
                    if not llm_placement:
                        llm_placement = str(
                            self.model_meta_by_id.get(llm_model, {}).get("placement", "")
                        ).strip() or None
            if llm_min_tier is None and llm_model:
                llm_min_tier = self.model_tier_by_id.get(llm_model, self.default_llm_min_tier)
            if llm_min_tier is None:
                llm_min_tier = self.default_llm_min_tier
            try:
                llm_min_tier = int(llm_min_tier)
            except Exception:
                definition_error = f"invalid llm_min_tier '{llm_min_tier}'"
                llm_min_tier = self.default_llm_min_tier
            if llm_min_tier < 1:
                definition_error = f"invalid llm_min_tier '{llm_min_tier}' (must be >= 1)"
                llm_min_tier = self.default_llm_min_tier

        task = {
            "task_id": str(uuid.uuid4()),
            "type": task_type,
            "command": command,
            "batch_id": batch_id,
            "name": task_name,
            "priority": priority,
            "batch_priority": batch_priority,
            "preemptible": bool(preemptible),
            "task_class": task_class,  # cpu, script, llm, brain, or meta
            "depends_on": depends_on or [],
            "executor": executor,  # "brain" or "worker"
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "created_by": self.name,
            "retry_count": 0
        }

        if isinstance(vram_estimate_mb, int) and vram_estimate_mb > 0:
            task["vram_estimate_mb"] = int(vram_estimate_mb)
            task["vram_estimate_source"] = vram_estimate_source or "plan"

        if task_class == "llm":
            task["llm_min_tier"] = int(llm_min_tier or self.default_llm_min_tier)
            if llm_model:
                task["llm_model"] = llm_model
            if llm_placement:
                task["llm_placement"] = llm_placement

        # Mark task with definition error so it goes to failed/ immediately
        if definition_error:
            task["definition_error"] = definition_error
            task["error_type"] = "definition"

        return task

    def _infer_script_vram_estimate(self, command: str, task_name: str = "") -> (Optional[int], Optional[str]):
        """Infer script VRAM estimate via brain LLM once per task definition."""
        prompt = f"""Estimate peak VRAM (MB) for ONE worker running this script task.

Task name: {task_name}
Command:
{command}

Rules:
- Return a conservative but realistic estimate for peak VRAM in MB.
- This is for a script task (not LLM generation task).
- We want one number reused for all expanded items of this step.
- Output JSON only.

Required JSON format:
{{"vram_estimate_mb": <integer>, "reason": "<short reason>"}}"""

        raw = self.think(prompt, log_as="vram_inference")
        if not raw:
            return None, None

        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                parts = cleaned.split("```")
                if len(parts) >= 2:
                    cleaned = parts[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
                cleaned = cleaned.strip()

            parsed = json.loads(cleaned)
            estimate = int(parsed.get("vram_estimate_mb", 0))
            if estimate <= 0:
                return None, None

            # Clamp to sane bounds to avoid bad model outputs from destabilizing scheduler.
            estimate = max(256, min(65536, estimate))
            return estimate, "infer:llm"
        except Exception:
            return None, None

    def _extract_python_scripts_from_command(self, command: str, plan_dir: Path) -> List[Path]:
        scripts: List[Path] = []
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
        for i, tok in enumerate(tokens):
            base = Path(tok).name.lower()
            if base in {"python", "python3"} and i + 1 < len(tokens):
                cand = tokens[i + 1].strip().strip("'\"")
                if cand.endswith(".py"):
                    p = Path(cand)
                    if not p.is_absolute():
                        p = (plan_dir / p).resolve()
                    scripts.append(p)
        return scripts

    def _build_plan_env_manifest(self, plan_dir: Path, commands: List[str], batch_dir: Path) -> Path:
        script_paths: List[Path] = []
        for cmd in commands:
            script_paths.extend(self._extract_python_scripts_from_command(cmd, plan_dir))

        # De-duplicate while preserving order.
        seen = set()
        dedup_scripts: List[Path] = []
        for p in script_paths:
            s = str(p)
            if s in seen:
                continue
            seen.add(s)
            dedup_scripts.append(p)

        stdlib = set(getattr(sys, "stdlib_module_names", set()))
        local_modules: set[str] = set()
        for py_file in plan_dir.rglob("*.py"):
            local_modules.add(py_file.stem)
        for init_file in plan_dir.rglob("__init__.py"):
            local_modules.add(init_file.parent.name)

        required_modules: set[str] = set()
        scanned_scripts: List[str] = []
        scan_errors: List[str] = []

        for script_path in dedup_scripts:
            scanned_scripts.append(str(script_path))
            if not script_path.exists():
                scan_errors.append(f"missing_script:{script_path}")
                continue
            try:
                source = script_path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(script_path))
            except Exception as exc:
                scan_errors.append(f"parse_error:{script_path}:{exc}")
                continue

            for node in ast.walk(tree):
                mod = None
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        mod = alias.name.split(".")[0].strip()
                        if mod and mod not in stdlib and mod not in local_modules and mod != "__future__":
                            required_modules.add(mod)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    mod = node.module.split(".")[0].strip()
                    if mod and mod not in stdlib and mod not in local_modules and mod != "__future__":
                        required_modules.add(mod)

        manifest = {
            "generated_at": datetime.now().isoformat(),
            "plan_dir": str(plan_dir.resolve()),
            "batch_dir": str(batch_dir.resolve()),
            "required_modules": sorted(required_modules),
            "scripts_scanned": scanned_scripts,
            "scan_errors": scan_errors,
        }
        manifest_path = batch_dir / "env_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return manifest_path

    def _resolve_task_vram_estimate(self, task_def: Dict[str, Any], resolved_command: str) -> (Optional[int], Optional[str]):
        """Resolve script VRAM estimate from plan fields or one-time inference policy."""
        task_class = task_def.get("task_class")
        if task_class != "script":
            return None, None

        explicit = task_def.get("vram_estimate_mb")
        if isinstance(explicit, int) and explicit > 0:
            return explicit, "plan:explicit"

        policy = task_def.get("vram_policy", "default")
        if policy == "infer":
            estimate, source = self._infer_script_vram_estimate(resolved_command, task_def.get("id", ""))
            if estimate is not None:
                return estimate, source
            self.log_decision(
                "VRAM_INFER_FALLBACK",
                f"VRAM infer failed for '{task_def.get('id', '')}', falling back to worker default",
                {"task_id": task_def.get("id", ""), "policy": "infer"}
            )
            return None, None

        return None, None

    # =========================================================================
    # Plan Parsing and Execution
    # =========================================================================

    def parse_plan_md(self, plan_content: str) -> List[Dict[str, Any]]:
        """
        Parse a structured plan.md file into task definitions.

        Expected format for each task:
        ### task_id
        - **executor**: brain|worker
        - **task_class**: cpu|script|llm|brain
        - **llm_min_tier**: 1  (optional, llm tasks only)
        - **llm_model**: qwen2.5:7b  (optional, llm tasks only)
        - **llm_placement**: single_gpu|split_gpu (optional, llm tasks only)
        - **command**: `shell command here`
        - **depends_on**: task1, task2
        - **foreach**: manifest.videos  (optional - expands to N tasks)
        - **batch_size**: 4  (optional - groups foreach expansion into micro-batches)
        - **vram_policy**: default|infer|fixed (optional, script tasks)
        - **vram_estimate_mb**: 2400 (optional, script tasks)
        """
        tasks = []

        # Split by ### to get task sections
        sections = re.split(r'\n### ', plan_content)

        for section in sections[1:]:  # Skip header before first ###
            lines = section.strip().split('\n')
            if not lines:
                continue

            task_id = lines[0].strip()
            task = {
                "id": task_id,
                "executor": "worker",
                "task_class": None,
                "command": "",
                "depends_on": [],
                "foreach": None,
                "batch_size": 1,
                "vram_policy": None,
                "vram_estimate_mb": None,
                "llm_min_tier": None,
                "llm_model": None,
                "llm_placement": None,
            }

            for line in lines[1:]:
                line = line.strip()
                if line.startswith('- **executor**:'):
                    task["executor"] = line.split(':', 1)[1].strip()
                elif line.startswith('- **task_class**:'):
                    task_class = line.split(':', 1)[1].strip().lower()
                    if task_class in VALID_TASK_CLASSES:
                        task["task_class"] = task_class
                    else:
                        self.logger.warning(f"Invalid task_class '{task_class}' for {task_id}, will use fallback")
                elif line.startswith('- **command**:'):
                    # Extract command from backticks
                    match = re.search(r'`([^`]+)`', line)
                    if match:
                        task["command"] = match.group(1)
                elif line.startswith('- **llm_min_tier**:'):
                    raw = line.split(':', 1)[1].strip()
                    try:
                        task["llm_min_tier"] = max(1, int(raw))
                    except ValueError:
                        self.logger.warning(
                            f"Invalid llm_min_tier '{raw}' for {task_id}, defaulting from catalog"
                        )
                        task["llm_min_tier"] = None
                elif line.startswith('- **llm_model**:'):
                    model_id = line.split(':', 1)[1].strip()
                    task["llm_model"] = model_id or None
                elif line.startswith('- **llm_placement**:'):
                    placement = line.split(':', 1)[1].strip().lower()
                    if placement in {"single_gpu", "split_gpu"}:
                        task["llm_placement"] = placement
                    elif placement:
                        self.logger.warning(
                            f"Invalid llm_placement '{placement}' for {task_id}, ignoring"
                        )
                elif line.startswith('- **depends_on**:'):
                    deps = line.split(':', 1)[1].strip()
                    if deps.lower() != 'none':
                        task["depends_on"] = [d.strip() for d in deps.split(',') if d.strip()]
                elif line.startswith('- **foreach**:'):
                    # e.g., "manifest.videos" means expand based on videos array in manifest
                    task["foreach"] = line.split(':', 1)[1].strip()
                elif line.startswith('- **batch_size**:'):
                    raw = line.split(':', 1)[1].strip()
                    try:
                        task["batch_size"] = max(1, int(raw))
                    except ValueError:
                        self.logger.warning(f"Invalid batch_size '{raw}' for {task_id}, defaulting to 1")
                        task["batch_size"] = 1
                elif line.startswith('- **vram_policy**:'):
                    policy = line.split(':', 1)[1].strip().lower()
                    if policy in VALID_VRAM_POLICIES:
                        task["vram_policy"] = policy
                    else:
                        self.logger.warning(
                            f"Invalid vram_policy '{policy}' for {task_id}, defaulting to 'default'"
                        )
                        task["vram_policy"] = "default"
                elif line.startswith('- **vram_estimate_mb**:'):
                    raw = line.split(':', 1)[1].strip()
                    try:
                        task["vram_estimate_mb"] = max(1, int(raw))
                    except ValueError:
                        self.logger.warning(
                            f"Invalid vram_estimate_mb '{raw}' for {task_id}, ignoring explicit estimate"
                        )
                        task["vram_estimate_mb"] = None

            if task["command"]:  # Only add tasks with commands
                if task.get("task_class") == "llm":
                    if task.get("llm_model"):
                        catalog_tier = self.model_tier_by_id.get(task["llm_model"])
                        if catalog_tier is None:
                            self.logger.warning(
                                f"Unknown llm_model '{task['llm_model']}' for {task_id}; will fail at task creation"
                            )
                        elif task.get("llm_min_tier") is None:
                            task["llm_min_tier"] = catalog_tier
                        if not task.get("llm_placement"):
                            task["llm_placement"] = str(
                                self.model_meta_by_id.get(task["llm_model"], {}).get("placement", "")
                            ) or None
                    if task.get("llm_min_tier") is None:
                        task["llm_min_tier"] = self.default_llm_min_tier
                else:
                    task["llm_min_tier"] = None
                    task["llm_model"] = None
                    task["llm_placement"] = None

                if task.get("task_class") == "script" and not task.get("vram_policy"):
                    task["vram_policy"] = "infer"
                elif not task.get("vram_policy"):
                    task["vram_policy"] = "default"

                if task.get("vram_estimate_mb") and task.get("vram_policy") in [None, "default"]:
                    task["vram_policy"] = "fixed"
                split_def_error = self._validate_split_llm_task_definition(task)
                if split_def_error:
                    task["definition_error"] = split_def_error
                    task["error_type"] = "definition"
                tasks.append(task)

        return tasks

    def _normalize_goal_task_list(self, raw: Any) -> List[str]:
        if isinstance(raw, list):
            items = raw
        else:
            items = str(raw or "").split(",")
        normalized: List[str] = []
        for item in items:
            task_id = str(item or "").strip()
            if task_id and task_id not in normalized:
                normalized.append(task_id)
        return normalized

    def _toposort_subset(self, task_ids: List[str], task_defs: List[Dict[str, Any]]) -> List[str]:
        tasks_by_id = {t["id"]: t for t in task_defs}
        order_rank = {t["id"]: idx for idx, t in enumerate(task_defs)}
        remaining = set(task_ids)
        ordered: List[str] = []

        while remaining:
            progressed = False
            for task_id in sorted(remaining, key=lambda k: order_rank.get(k, 10**9)):
                deps = [
                    d for d in tasks_by_id.get(task_id, {}).get("depends_on", [])
                    if d in remaining
                ]
                if all(dep in ordered for dep in deps):
                    ordered.append(task_id)
                    remaining.remove(task_id)
                    progressed = True
            if not progressed:
                ordered.extend(
                    sorted(remaining, key=lambda k: order_rank.get(k, 10**9))
                )
                break
        return ordered

    def _infer_goal_discovery_order(
        self, task_defs: List[Dict[str, Any]], explicit_tasks: List[str]
    ) -> List[str]:
        tasks_by_id = {t["id"]: t for t in task_defs}
        non_foreach = {t["id"] for t in task_defs if not t.get("foreach")}

        # Explicit list from goal/config wins, then gets topologically normalized.
        if explicit_tasks:
            selected = [task_id for task_id in explicit_tasks if task_id in non_foreach]
            return self._toposort_subset(selected, task_defs)

        foreach_ids = [t["id"] for t in task_defs if t.get("foreach")]
        if not foreach_ids:
                return None

        ancestor_ids: set[str] = set()
        stack = list(foreach_ids)
        seen = set()
        while stack:
            current = stack.pop()
            task = tasks_by_id.get(current)
            if not task:
                continue
            for dep in task.get("depends_on", []):
                if dep in seen:
                    continue
                seen.add(dep)
                dep_task = tasks_by_id.get(dep)
                if not dep_task:
                    continue
                if not dep_task.get("foreach"):
                    ancestor_ids.add(dep)
                stack.append(dep)

        return self._toposort_subset(list(ancestor_ids), task_defs)

    def execute_plan(self, plan_path: str, config_overrides: dict = None) -> str:
        """
        Execute a plan by reading plan.md and generating tasks.

        1. Read and parse plan.md directly (no LLM needed)
        2. Create all tasks with dependencies
        3. Store in private list
        4. Release tasks with no dependencies immediately
        5. Return batch_id
        """
        plan_dir = Path(plan_path)
        if plan_dir.is_file():
            plan_dir = plan_dir.parent

        # Read plan.md
        plan_file = plan_dir / "plan.md"
        if not plan_file.exists():
            raise FileNotFoundError(f"No plan.md found in {plan_dir}")

        with open(plan_file) as f:
            plan_content = f.read()

        # Generate timestamp-based execution batch ID for orchestration tracking
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = plan_dir / "history" / batch_id

        self.log_decision("PLAN_READ", f"Read plan from {plan_dir.name}", {
            "batch_id": batch_id,
            "plan_size": len(plan_content)
        })

        # Build variable substitution map
        config = dict(config_overrides or {})
        batch_priority_label, batch_priority_value = self._normalize_batch_priority(config.get("PRIORITY", "normal"))
        batch_preemptible = self._normalize_preemptible(config.get("PREEMPTIBLE", True))
        config["PRIORITY"] = batch_priority_label
        config["PREEMPTIBLE"] = batch_preemptible

        # Enforce explicit run mode contract for plans that use RUN_MODE.
        run_mode = str(config.get("RUN_MODE", "fresh")).strip().lower()
        if run_mode not in ["fresh", "resume"]:
            raise ValueError("RUN_MODE must be 'fresh' or 'resume'")
        config["RUN_MODE"] = run_mode

        if run_mode == "resume":
            resume_batch_id = str(config.get("RESUME_BATCH_ID", "")).strip()
            if not resume_batch_id:
                raise ValueError("RUN_MODE=resume requires RESUME_BATCH_ID")
            if "/" in resume_batch_id or ".." in resume_batch_id:
                raise ValueError("RESUME_BATCH_ID contains invalid path characters")

            resume_manifest = plan_dir / "history" / resume_batch_id / "manifest.json"
            if not resume_manifest.exists():
                raise FileNotFoundError(
                    f"Resume batch manifest not found: {resume_manifest}"
                )

            # Use the existing data batch for plan variables while keeping a
            # new orchestration batch_id for dependency tracking.
            config["BATCH_ID"] = resume_batch_id
            self.log_decision("PLAN_MODE",
                              f"Resume mode for {plan_dir.name}: {resume_batch_id}",
                              {"run_mode": "resume", "resume_batch_id": resume_batch_id})

            # If this exact plan/data-batch is already active, reuse it.
            # Avoid creating a duplicate orchestration batch that re-releases
            # the whole plan on each resume submission.
            plan_dir_resolved = str(plan_dir.resolve())
            for existing_batch_id, batch_meta in self.active_batches.items():
                existing_plan_dir = str(batch_meta.get("plan_dir", ""))
                existing_cfg = batch_meta.get("config", {}) or {}
                existing_data_batch = str(existing_cfg.get("BATCH_ID", "")).strip()

                if existing_plan_dir == plan_dir_resolved and existing_data_batch == resume_batch_id:
                    self.log_decision(
                        "PLAN_RESUME_REUSE",
                        f"Resume requested for already-active batch {existing_batch_id}; reusing existing orchestration state",
                        {
                            "plan_dir": plan_dir_resolved,
                            "resume_batch_id": resume_batch_id,
                            "orchestration_batch_id": existing_batch_id
                        }
                    )
                    return existing_batch_id
        else:
            # Fresh mode defaults plan variables to the new execution batch.
            cleanup_stats = self._cleanup_stale_plan_batches(plan_dir)
            if cleanup_stats.get("stale_batches", 0) > 0:
                self.log_decision(
                    "PLAN_CLEANUP",
                    f"Fresh run cleanup for {plan_dir.name}",
                    cleanup_stats
                )

            config["BATCH_ID"] = str(config.get("BATCH_ID", batch_id)).strip() or batch_id
            if "/" in config["BATCH_ID"] or ".." in config["BATCH_ID"]:
                raise ValueError("BATCH_ID contains invalid path characters")
            self.log_decision("PLAN_MODE",
                              f"Fresh mode for {plan_dir.name}: {config['BATCH_ID']}",
                              {"run_mode": "fresh", "batch_id": config["BATCH_ID"]})

        effective_batch_id = config["BATCH_ID"]
        effective_batch_dir = plan_dir / "history" / effective_batch_id
        effective_batch_dir.mkdir(parents=True, exist_ok=True)
        (effective_batch_dir / "results").mkdir(exist_ok=True)
        (effective_batch_dir / "output").mkdir(exist_ok=True)
        (effective_batch_dir / "logs").mkdir(exist_ok=True)

        variables = {
            "{BATCH_ID}": effective_batch_id,
            "{PLAN_PATH}": str(plan_dir.resolve()),
            "{BATCH_PATH}": str(effective_batch_dir.resolve()),
        }
        # Add any config overrides as variables
        for key, value in config.items():
            variables[f"{{{key}}}"] = str(value)

        # Parse plan.md directly
        task_defs = self.parse_plan_md(plan_content)

        # Parse optional Goal section (backward-compatible)
        goal_spec = self._parse_goal_section(plan_content)
        if goal_spec:
            # Resolve goal numeric fields from literals or {VARS}
            def _resolve_goal_int(field: str) -> int:
                raw_val = str(goal_spec.get(field, ""))
                for var, value in variables.items():
                    raw_val = raw_val.replace(var, value)
                try:
                    return int(raw_val)
                except ValueError:
                    self.logger.error(f"Goal {field} could not be resolved to int: {raw_val}")
                    raise

            try:
                goal_spec["target"] = _resolve_goal_int("target")
                goal_spec["tolerance"] = _resolve_goal_int("tolerance")
                goal_spec["max_attempts_multiplier"] = _resolve_goal_int("max_attempts_multiplier")
            except ValueError:
                goal_spec = None

        goal_discovery_order: List[str] = []
        if goal_spec:
            discovery_task_config = config.get("DISCOVERY_TASKS", goal_spec.get("discovery_tasks", []))
            discovery_task_ids = self._normalize_goal_task_list(discovery_task_config)
            goal_discovery_order = self._infer_goal_discovery_order(task_defs, discovery_task_ids)
            goal_spec["discovery_tasks"] = goal_discovery_order

        if goal_spec:
            # Mark foreach task_defs so check_and_release_tasks() intercepts them
            tracked_task = goal_spec["tracked_task"]
            for t in task_defs:
                if t.get("foreach"):
                    t["goal_driven"] = True

        self.log_decision("PLAN_PARSED", f"Parsed {len(task_defs)} tasks from plan.md", {
            "task_ids": [t["id"] for t in task_defs],
            "batch_id": batch_id,
            "goal_driven": goal_spec is not None,
            "batch_priority": batch_priority_label,
            "batch_preemptible": batch_preemptible,
        })

        # Analyze task types for resource planning
        class_counts = {"cpu": 0, "script": 0, "llm": 0, "brain": 0}
        missing_class = []
        for t in task_defs:
            tc = t.get("task_class")
            if tc and tc in VALID_TASK_CLASSES:
                class_counts[tc] += 1
            else:
                missing_class.append(t["id"])
                class_counts["cpu"] += 1  # Default to cpu

        if missing_class:
            self.logger.warning(f"Tasks missing task_class (defaulting to cpu): {missing_class}")

        self.log_decision("PLAN_ANALYSIS",
            (
                f"Task breakdown: {class_counts['cpu']} cpu, "
                f"{class_counts['script']} script, {class_counts['llm']} llm, "
                f"{class_counts['brain']} brain"
            ),
            {"task_classes": class_counts})

        substituted_commands: List[str] = []
        for task_def in task_defs:
            cmd = task_def["command"]
            for var, value in variables.items():
                cmd = cmd.replace(var, value)
            substituted_commands.append(cmd)

        env_manifest_path = self._build_plan_env_manifest(
            plan_dir=plan_dir,
            commands=substituted_commands,
            batch_dir=effective_batch_dir,
        )
        self.log_decision("PLAN_ENV_MANIFEST", "Generated plan environment manifest", {
            "batch_id": batch_id,
            "manifest_path": str(env_manifest_path),
        })

        goal_discovery_templates = {}
        goal_round_cap = 0
        # Create tasks with variable substitution
        tasks_with_no_deps = []
        for task_def in task_defs:
            # Substitute all variables in command
            command = task_def["command"]
            for var, value in variables.items():
                command = command.replace(var, value)

            vram_estimate_mb, vram_estimate_source = self._resolve_task_vram_estimate(task_def, command)

            task = self.create_task(
                task_type="shell",
                command=command,
                batch_id=batch_id,
                task_name=task_def["id"],
                priority=batch_priority_value,
                depends_on=task_def.get("depends_on", []),
                executor=task_def.get("executor", "worker"),
                task_class=task_def.get("task_class"),
                vram_estimate_mb=vram_estimate_mb,
                vram_estimate_source=vram_estimate_source,
                llm_min_tier=task_def.get("llm_min_tier"),
                llm_model=task_def.get("llm_model"),
                llm_placement=task_def.get("llm_placement"),
                batch_priority=batch_priority_label,
                preemptible=batch_preemptible,
            )
            task["plan_path"] = str(plan_dir.resolve())
            task["batch_path"] = str(effective_batch_dir.resolve())
            task["env_manifest_path"] = str(env_manifest_path.resolve())

            # Preserve foreach spec for later expansion
            if task_def.get("foreach"):
                # Substitute variables in foreach path
                foreach_spec = task_def["foreach"]
                for var, value in variables.items():
                    foreach_spec = foreach_spec.replace(var, value)
                task["foreach"] = foreach_spec
                task["batch_size"] = max(1, int(task_def.get("batch_size", 1)))
                # Mark goal-driven foreach tasks for incremental expansion
                if task_def.get("goal_driven"):
                    task["goal_driven"] = True
            elif goal_spec and task_def["id"] in goal_discovery_order:
                goal_discovery_templates[task_def["id"]] = {
                    "command": command,
                    "executor": task_def.get("executor", "worker"),
                    "task_class": task_def.get("task_class"),
                    "llm_min_tier": task_def.get("llm_min_tier"),
                    "llm_model": task_def.get("llm_model"),
                    "llm_placement": task_def.get("llm_placement"),
                    "priority": batch_priority_value,
                    "depends_on": list(task_def.get("depends_on", [])),
                }

            # Check for definition errors - send to failed/ immediately
            if task.get("definition_error"):
                task["status"] = "failed"
                task["result"] = {
                    "success": False,
                    "error": task["definition_error"],
                    "error_type": "definition"
                }
                self.save_to_failed(task)
                self.log_decision("TASK_DEFINITION_ERROR",
                    f"Task '{task_def['id']}' has definition error: {task['definition_error']}",
                    {"task_id": task["task_id"][:8], "error": task["definition_error"]})
            else:
                # Save valid tasks to private list
                self.save_to_private(task)

                self.log_decision("TASK_CREATED", f"Created task: {task_def['id']} ({task['task_class']})", {
                    "task_id": task["task_id"][:8],
                    "task_class": task["task_class"],
                    "depends_on": task_def.get("depends_on", []),
                    "executor": task_def.get("executor", "worker"),
                    "vram_estimate_mb": task.get("vram_estimate_mb"),
                    "vram_estimate_source": task.get("vram_estimate_source")
                })

                # Track tasks with no dependencies for immediate release
                if not task_def.get("depends_on"):
                    tasks_with_no_deps.append(task)

        # Track this batch (include paths for foreach expansion)
        batch_meta = {
            "plan": plan_dir.name,
            "plan_dir": str(plan_dir.resolve()),
            # Effective data batch path (respects RUN_MODE + BATCH_ID overrides).
            "batch_dir": str(effective_batch_dir.resolve()),
            # Orchestration batch path (timestamp id used for this execution run).
            "orchestration_batch_dir": str(batch_dir.resolve()),
            "env_manifest_path": str(env_manifest_path.resolve()),
            "started_at": datetime.now().isoformat(),
            "config": config,
            "total_tasks": len(task_defs),
            "priority": batch_priority_label,
            "preemptible": batch_preemptible,
        }

        # Initialize goal state for goal-driven plans
        if goal_spec:
            target = goal_spec["target"]
            multiplier = goal_spec["max_attempts_multiplier"]
            def _resolve_goal_setting_int(config_key: str, goal_key: str, default: int, minimum: int = 0) -> int:
                raw_val = config.get(config_key, goal_spec.get(goal_key, default))
                raw_text = str(raw_val)
                for var, value in variables.items():
                    raw_text = raw_text.replace(var, value)
                try:
                    parsed = int(raw_text)
                except ValueError:
                    parsed = int(default)
                return max(minimum, parsed)

            try:
                configured_rounds = int(str(config.get("DISCOVERY_ROUNDS", "0") or "0"))
            except ValueError:
                configured_rounds = 0
            round_multiplier = _resolve_goal_setting_int(
                "DISCOVERY_ROUND_MULTIPLIER", "discovery_round_multiplier", default=2, minimum=1
            )
            max_rounds = max(1, target * round_multiplier)
            if configured_rounds > 0:
                goal_round_cap = min(max_rounds, configured_rounds)
            else:
                goal_round_cap = max_rounds
            prefill_divisor = _resolve_goal_setting_int(
                "DISCOVERY_PREFILL_DIVISOR", "discovery_prefill_divisor", default=4, minimum=1
            )
            prefill_target_rounds = max(1, (target + prefill_divisor - 1) // prefill_divisor)
            prefill_target_rounds = min(goal_round_cap, prefill_target_rounds)
            pool_multiplier = _resolve_goal_setting_int(
                "DISCOVERY_POOL_MULTIPLIER", "discovery_pool_multiplier", default=1, minimum=1
            )
            pool_cap = _resolve_goal_setting_int(
                "DISCOVERY_POOL_CAP", "discovery_pool_cap", default=0, minimum=0
            )
            discovery_pool_target = max(1, target * pool_multiplier)
            if pool_cap > 0:
                discovery_pool_target = min(discovery_pool_target, pool_cap)
            refill_divisor = _resolve_goal_setting_int(
                "DISCOVERY_REFILL_DIVISOR", "discovery_refill_divisor", default=4, minimum=1
            )
            max_validations_per_cycle = _resolve_goal_setting_int(
                "MAX_VALIDATIONS_PER_CYCLE", "max_validations_per_cycle", default=3, minimum=1
            )
            batch_meta["goal"] = {
                "goal_version": 1,
                "type": goal_spec["type"],
                "target": target,
                "tolerance": goal_spec["tolerance"],
                "max_attempts": target * multiplier,
                "max_rejections": target * multiplier,
                "max_attempts_multiplier": multiplier,
                "tracked_task": goal_spec["tracked_task"],
                "query": config.get("QUERY", ""),
                "candidate_pool_path": "",   # populated when manifest is ready
                "candidates_total": 0,
                "templates": {},             # populated when foreach deps are met
                "variables": variables,
                "compile_output_task_id": compile_output_task_id or "",
                "accepted": 0,
                "rejected": 0,
                "in_flight_ids": [],
                "accepted_ids": [],
                "rejected_ids": [],
                "spawned_ids": [],
                "validated_task_ids": [],
                "next_index": 0,
                "status": "active",
                "phase": "fill_pool",
                "max_validations_per_cycle": max_validations_per_cycle,
                "discovery_round_cap": goal_round_cap,
                "discovery_rounds_generated": 1,
                # Round 1 tasks are created by plan parsing above; keep discovery
                # idle so prefill can enqueue rounds 2..N immediately on first loop.
                "discovery_in_progress": False,
                "discovery_active_round": 0,
                "discovery_round_multiplier": round_multiplier,
                "discovery_prefill_divisor": prefill_divisor,
                "discovery_prefill_target_rounds": prefill_target_rounds,
                "discovery_prefill_scheduled_through": 1,
                "discovery_pool_target": discovery_pool_target,
                "discovery_refill_divisor": refill_divisor,
                "discovery_refill_watermark": max(
                    1, (discovery_pool_target + refill_divisor - 1) // refill_divisor
                ),
                "discovery_templates": goal_discovery_templates,
                "discovery_task_order": goal_discovery_order,
                "discovery_terminal_task": (
                    goal_discovery_order[-1] if goal_discovery_order else ""
                ),
                "processed_discovery_task_ids": [],
                "processed_identify_task_ids": [],
            }
            self.log_decision("GOAL_INITIALIZED",
                f"Goal-driven plan: target={target}, tolerance={goal_spec['tolerance']}, "
                f"max_rejections={target * multiplier}, tracked={goal_spec['tracked_task']}",
                batch_meta["goal"])

        self.active_batches[batch_id] = batch_meta
        self._save_brain_state()

        # Release tasks with no dependencies immediately
        for task in tasks_with_no_deps:
            task_file = self.private_tasks_path / f"{task['task_id']}.json"
            if task_file.exists():
                task_file.unlink()
                self.save_to_public(task)

        return batch_id

    # =========================================================================
    # Task Handling
    # =========================================================================
