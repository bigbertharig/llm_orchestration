"""Orchestration controls for the dashboard (served by llm-gateway; docs/DASHBOARD.md there).

Plans and batches (start, resume, kill, outputs) and task-file cleanup. Everything that acts on GPUs
or runtimes (unload, kill a card, reservations) belongs to the rig manager in llm-gateway, not here.
The methods below the public ones are the old dashboard server's, unchanged; `Controls` only gives
them a home without an HTTP handler.
"""

import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from .plans import (
    collect_batch_outputs,
    default_plan_config,
    discover_active_arm_bindings,
    discover_plan_input_files,
    discover_plan_starters,
    discover_plans,
    discover_recent_batches,
    plan_input_help,
    run_shell,
    write_inline_input_file,
)
from .utils import (
    INLINE_FILE_KEYS,
    load_json,
    normalize_github_url,
    sanitize_config_object,
    sanitize_text,
)
from .workers import load_brain_state

BATCH_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")
GITHUB_URL_PARSE_RE = re.compile(r"^https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)/?$")
STDIO_SNIPPET_LIMIT = 16000
STDIO_TIMEOUT_LIMIT = 8000

def _clip_stdio(text: str, limit: int = STDIO_SNIPPET_LIMIT) -> str:
    """Preserve both head and tail of long output (useful for tracebacks)."""
    if not isinstance(text, str):
        return ""
    if limit <= 0 or len(text) <= limit:
        return text
    head = max(1, int(limit * 0.45))
    tail = max(1, int(limit * 0.45))
    omitted = len(text) - head - tail
    return (
        text[:head]
        + f"\n\n...[truncated {omitted} chars]...\n\n"
        + text[-tail:]
    )


class Controls:
    """Orchestration actions over the shared drive (`shared_path`) and the orchestration config."""

    def __init__(self, shared_path: Path, config: dict[str, Any]):
        self.shared_path = Path(shared_path)
        self.config = config

    # ------------------------------------------------------------------ public actions

    def options(self) -> dict[str, Any]:
        return self._control_options()

    def start_plan(self, plan_name: str, config_json: str = "", plan_scope: str = "shoulders",
                   starter_file: str = "", repo_path: str = "", repo_url: str = "", claimed_behavior: str = "",
                   inline_query_text: str = "") -> dict[str, Any]:
        return self._start_plan(plan_name, config_json, plan_scope, starter_file, repo_path, repo_url,
                                claimed_behavior, inline_query_text)

    def resume_plan(self, batch_id: str) -> dict[str, Any]:
        return self._resume_plan(batch_id)

    def kill_plan(self, batch_id: str) -> dict[str, Any]:
        return self._kill_plan(batch_id)

    def kill_all_active(self) -> dict[str, Any]:
        return self._kill_all_active()

    def batch_outputs(self, batch_id: str) -> dict[str, Any]:
        return self._batch_outputs(batch_id)

    def cleanup_stale(self) -> dict[str, Any]:
        """Task-file cleanup only: stale heartbeats, orphan locks, leftover system meta tasks."""
        heartbeats = self._cleanup_stale_processing_heartbeats()
        artifacts = self._cleanup_stale_task_artifacts()
        system_meta = self._cleanup_system_meta_artifacts()
        errors = [*heartbeats.get("errors", []), *artifacts.get("errors", []), *system_meta.get("errors", [])]
        message = (f"heartbeats removed {heartbeats.get('removed', 0)}, task locks removed "
                   f"{artifacts.get('task_locks_removed', 0)}, failed meta removed "
                   f"{artifacts.get('failed_meta_removed', 0)}, system meta removed "
                   f"{system_meta.get('tasks_removed', 0)}")
        return {"ok": not errors, "message": message + (" (with errors)" if errors else ""),
                "cleanup": {"heartbeats": heartbeats, "artifacts": artifacts, "system_meta": system_meta},
                "errors": errors}

    def requeue_worker(self, worker: str = "") -> dict[str, Any]:
        """Return a worker's (or every worker's) processing tasks to the queue, for example after its
        card was killed through the rig manager."""
        out = self._requeue_processing_tasks_for_reset(assigned_to=worker or None)
        out["ok"] = not out.get("errors")
        out["message"] = f"requeued {out.get('moved_to_queue', 0)} task(s)" + (f" of {worker}" if worker else "")
        return out

    # ------------------------------------------------------------------ the old server's methods

    def _gpu_ssh(self, remote_cmd: str, timeout_s: int = 180) -> dict[str, Any]:
        # Pass remote command as a single shell-quoted argument so ssh does not
        # split `bash -lc` arguments and silently execute a different command.
        cmd = ["ssh", "-o", "BatchMode=yes", "gpu", f"bash -lc {shlex.quote(remote_cmd)}"]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            return {
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout": _clip_stdio(proc.stdout),
                "stderr": _clip_stdio(proc.stderr),
                "cmd": " ".join(cmd),
            }
        except subprocess.TimeoutExpired as e:
            return {
                "ok": False,
                "returncode": -1,
                "stdout": _clip_stdio(e.stdout or "", STDIO_TIMEOUT_LIMIT) if isinstance(e.stdout, str) else "",
                "stderr": _clip_stdio(e.stderr or "", STDIO_TIMEOUT_LIMIT) if isinstance(e.stderr, str) else "",
                "cmd": " ".join(cmd),
                "error": f"timeout after {timeout_s}s",
            }

    def _gpu_has_git(self) -> tuple[bool, str]:
        """Return whether git is available on the GPU host plus diagnostics."""
        out = self._gpu_ssh("command -v git >/dev/null 2>&1 && echo __OK__ || echo __MISSING__", timeout_s=20)
        stdout = str(out.get("stdout", ""))
        if out.get("ok") and "__OK__" in stdout:
            return True, ""
        detail = str(out.get("stderr", "")).strip() or stdout.strip() or "git not found on gpu host"
        return False, detail

    def _gpu_find_existing_repo_checkout(self, repo_url: str) -> tuple[str, str]:
        """Return (repo_path, normalized_url) for an existing local checkout on GPU host if found."""
        normalized = normalize_github_url(repo_url)
        m = GITHUB_URL_PARSE_RE.match(normalized)
        if not m:
            return "", normalized
        owner, repo = m.group(1), m.group(2)
        repo_base = f"{owner}-{repo}"
        candidates = [
            f"/mnt/shared/plans/arms/{repo_base}",
            f"/mnt/shared/plans/arms/exploratory/{repo_base}",
            f"/media/bryan/shared/plans/arms/{repo_base}",
            f"/media/bryan/shared/plans/arms/exploratory/{repo_base}",
        ]
        script_lines = ["set -e"]
        for path in candidates:
            qpath = shlex.quote(path)
            script_lines.append(f"if [ -d {qpath}/.git ]; then printf '%s\\n' {qpath}; exit 0; fi")
        script_lines.append("exit 1")
        out = self._gpu_ssh("\n".join(script_lines), timeout_s=20)
        if out.get("ok"):
            line = str(out.get("stdout", "") or "").strip().splitlines()
            if line and line[0].strip():
                return line[0].strip(), normalized
        return "", normalized

    def _discover_local_repo_choices(self) -> list[dict[str, str]]:
        """Discover git checkouts under arms/shoulders for dropdown selection."""
        plans_root = self.shared_path / "plans"
        seen: set[str] = set()
        out: list[dict[str, str]] = []

        def _to_mnt_path(p: Path) -> str:
            text = str(p)
            if text.startswith("/media/bryan/shared/"):
                return "/mnt/shared/" + text[len("/media/bryan/shared/"):]
            return text

        candidate_dirs: list[tuple[str, Path]] = [
            ("shoulders", plans_root / "shoulders"),
            ("arms", plans_root / "arms"),
            ("arms/exploratory", plans_root / "arms" / "exploratory"),
        ]

        for scope_label, base in candidate_dirs:
            try:
                if not base.exists() or not base.is_dir():
                    continue
            except Exception:
                continue
            try:
                children = sorted(base.iterdir())
            except Exception:
                continue
            for child in children:
                try:
                    if not child.is_dir():
                        continue
                    if not (child / ".git").is_dir():
                        continue
                except Exception:
                    continue
                mnt_path = _to_mnt_path(child)
                if mnt_path in seen:
                    continue
                seen.add(mnt_path)
                out.append(
                    {
                        "path": mnt_path,
                        "label": f"{scope_label}: {child.name} ({mnt_path})",
                        "scope": scope_label,
                        "name": child.name,
                    }
                )

        out.sort(key=lambda r: (str(r.get("scope", "")), str(r.get("name", ""))))
        return out

    def _control_options(self) -> dict[str, Any]:
        brain = load_brain_state(self.shared_path)
        active = brain.get("active_batches", {}) if isinstance(brain.get("active_batches"), dict) else {}
        shoulder_plans = discover_plans(self.shared_path, plan_scope="shoulders")
        arm_plans = discover_plans(self.shared_path, plan_scope="arms")
        plans = shoulder_plans + [p for p in arm_plans if p not in set(shoulder_plans)]
        plan_scopes: dict[str, str] = {}
        for p in shoulder_plans:
            plan_scopes.setdefault(p, "shoulders")
        for p in arm_plans:
            plan_scopes[p] = "arms"
        shoulder_arm_bindings = discover_active_arm_bindings(self.shared_path)
        recent_batches = discover_recent_batches(self.shared_path, active)
        plan_defaults = {p: default_plan_config(self.shared_path, p) for p in plans}
        plan_starters = {p: discover_plan_starters(self.shared_path, p, plan_scope=plan_scopes.get(p, "shoulders")) for p in plans}
        plan_input_files = {p: discover_plan_input_files(self.shared_path, p, plan_scope=plan_scopes.get(p, "shoulders")) for p in plans}
        plan_default_starter = {p: ("plan.md" if "plan.md" in plan_starters[p] else (plan_starters[p][0] if plan_starters[p] else "")) for p in plans}
        plan_inputs = {
            p: {starter: plan_input_help(self.shared_path, p, starter, plan_scope=plan_scopes.get(p, "shoulders")) for starter in plan_starters[p]}
            for p in plans
        }
        plan_defaults = {p: self._dashboard_visible_plan_defaults(p, cfg) for p, cfg in plan_defaults.items()}
        plan_inputs = {
            p: {starter: self._dashboard_visible_plan_inputs(p, rows) for starter, rows in starters.items()}
            for p, starters in plan_inputs.items()
        }
        active_meta = []
        batch_labels: dict[str, str] = {}
        for batch_id, meta in active.items():
            if not isinstance(meta, dict):
                continue
            plan_name = str(meta.get("plan", "")).strip() or "unknown_plan"
            display_name = str(meta.get("display_name", "")).strip() or plan_name
            active_meta.append(
                {
                    "batch_id": batch_id,
                    "plan": plan_name,
                    "display_name": display_name,
                    "scope": "arms" if "/plans/arms/" in str(meta.get("plan_dir", "")) else "shoulders",
                    "started_at": meta.get("started_at", ""),
                }
            )
            batch_labels[batch_id] = f"{display_name} | {batch_id}"
        for row in recent_batches:
            batch_id = str(row.get("batch_id", "")).strip()
            if not batch_id:
                continue
            display_name = str(row.get("display_name", "")).strip() or str(row.get("plan", "")).strip() or "unknown_plan"
            batch_labels.setdefault(batch_id, f"{display_name} | {batch_id}")
        active_meta.sort(key=lambda x: str(x.get("started_at") or ""), reverse=True)
        return {
            "ok": True,
            "active_batches": sorted(active.keys()),
            "active_batches_meta": active_meta,
            "resumable_batches": sorted(active.keys()),
            "recent_batches": recent_batches,
            "plans": plans,
            "shoulder_plans": shoulder_plans,
            "arm_plans": arm_plans,
            "plan_scopes": plan_scopes,
            "shoulder_arm_bindings": shoulder_arm_bindings,
            "plan_defaults": plan_defaults,
            "plan_starters": plan_starters,
            "plan_input_files": plan_input_files,
            "plan_default_starter": plan_default_starter,
            "plan_inputs": plan_inputs,
            "batch_labels": batch_labels,
            "local_repo_choices": self._discover_local_repo_choices(),
        }

    def _dashboard_visible_plan_defaults(self, plan_name: str, cfg: Any) -> dict[str, Any]:
        if not isinstance(cfg, dict):
            return {}
        if plan_name != "github_analyzer":
            return dict(cfg)
        keep = {"REPO_PATH", "REPO_URL", "CLAIMED_BEHAVIOR", "ANALYSIS_DEPTH"}
        return {k: v for k, v in cfg.items() if k in keep}

    def _dashboard_visible_plan_inputs(self, plan_name: str, rows: Any) -> list[dict[str, Any]]:
        if not isinstance(rows, list):
            return []
        if plan_name != "github_analyzer":
            return [r for r in rows if isinstance(r, dict)]
        keep = {"REPO_PATH", "REPO_URL", "CLAIMED_BEHAVIOR", "ANALYSIS_DEPTH"}
        out: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            key = str(row.get("key", "")).strip()
            if key in keep:
                out.append(dict(row))
        return out

    def _kill_plan(self, batch_id: str) -> dict[str, Any]:
        if not batch_id:
            return {"ok": False, "message": "batch_id is required"}
        if not BATCH_ID_RE.fullmatch(batch_id):
            return {"ok": False, "message": f"invalid batch_id: {batch_id!r}"}
        scripts_dir = Path(__file__).resolve().parent.parent
        local_kill = scripts_dir / "kill_plan.py"
        cmd = f"python3 {shlex.quote(str(local_kill))} {shlex.quote(batch_id)} --keep-workers --keep-models --no-default-warm"
        out = run_shell(cmd, timeout_s=240)
        lock_cleanup = self._cleanup_orphan_task_locks()
        system_meta = self._cleanup_system_meta_artifacts()
        cleaned = self._cleanup_stale_processing_heartbeats()
        out["cleanup"] = {
            "heartbeats": cleaned,
            "orphan_task_locks": lock_cleanup,
            "system_meta": system_meta,
        }
        if cleaned.get("errors"):
            out["stderr"] = f"{out.get('stderr', '')}\nheartbeat_cleanup_errors={cleaned.get('errors')}".strip()
        if lock_cleanup.get("errors"):
            out["stderr"] = f"{out.get('stderr', '')}\nlock_cleanup_errors={lock_cleanup.get('errors')}".strip()
        if system_meta.get("errors"):
            out["stderr"] = f"{out.get('stderr', '')}\nsystem_meta_cleanup_errors={system_meta.get('errors')}".strip()
        out["message"] = f"Killed batch {batch_id}" if out.get("ok") else f"Failed to kill batch {batch_id}"
        return out

    def _kill_all_active(self) -> dict[str, Any]:
        scripts_dir = Path(__file__).resolve().parent.parent
        local_kill = scripts_dir / "kill_plan.py"
        cmd = f"python3 {local_kill} --keep-workers --keep-models --no-default-warm"
        out = run_shell(cmd, timeout_s=300)
        lock_cleanup = self._cleanup_orphan_task_locks()
        system_meta = self._cleanup_system_meta_artifacts()
        cleaned = self._cleanup_stale_processing_heartbeats()
        out["cleanup"] = {
            "heartbeats": cleaned,
            "orphan_task_locks": lock_cleanup,
            "system_meta": system_meta,
        }
        if cleaned.get("errors"):
            out["stderr"] = f"{out.get('stderr', '')}\nheartbeat_cleanup_errors={cleaned.get('errors')}".strip()
        if lock_cleanup.get("errors"):
            out["stderr"] = f"{out.get('stderr', '')}\nlock_cleanup_errors={lock_cleanup.get('errors')}".strip()
        if system_meta.get("errors"):
            out["stderr"] = f"{out.get('stderr', '')}\nsystem_meta_cleanup_errors={system_meta.get('errors')}".strip()
        out["message"] = "Killed all active batches" if out.get("ok") else "Failed to kill all active batches"
        return out

    def _cleanup_stale_processing_heartbeats(self) -> dict[str, Any]:
        processing_dir = self.shared_path / "tasks" / "processing"
        removed = 0
        kept = 0
        errors: list[str] = []
        brain = load_brain_state(self.shared_path)
        active_batches = set()
        if isinstance(brain.get("active_batches"), dict):
            active_batches = {str(k) for k in brain.get("active_batches", {}).keys()}

        for hb in sorted(processing_dir.glob("*.heartbeat.json")):
            task_json = hb.with_name(hb.name.replace(".heartbeat.json", ".json"))
            task = load_json(task_json) if task_json.exists() else None
            should_remove = False

            if not task_json.exists() or not isinstance(task, dict):
                should_remove = True
            else:
                batch_id = str(task.get("batch_id", "")).strip()
                # Keep heartbeat only if its task still belongs to an active batch.
                if not batch_id or batch_id not in active_batches:
                    should_remove = True

            if should_remove:
                try:
                    hb.unlink()
                    removed += 1
                except Exception as exc:
                    errors.append(f"{hb.name}: {exc}")
            else:
                kept += 1

        return {"removed": removed, "kept": kept, "errors": errors}

    def _cleanup_stale_task_artifacts(self) -> dict[str, Any]:
        tasks_dir = self.shared_path / "tasks"
        errors: list[str] = []
        task_locks_removed = 0
        failed_meta_removed = 0

        brain = load_brain_state(self.shared_path)
        active_batches = set()
        if isinstance(brain.get("active_batches"), dict):
            active_batches = {str(k) for k in brain.get("active_batches", {}).keys()}

        # Remove orphan/stale task lock files. These are filesystem lock artifacts only.
        for lane in ("queue", "processing", "failed"):
            lane_dir = tasks_dir / lane
            if not lane_dir.exists():
                continue
            for lock_path in sorted(lane_dir.glob("*.json.lock")):
                task_path = lock_path.with_name(lock_path.name[:-5])  # strip ".lock"
                # Keep a lock if a task file currently exists in processing (possible active writer).
                if lane == "processing" and task_path.exists():
                    continue
                try:
                    lock_path.unlink()
                    task_locks_removed += 1
                except Exception as exc:
                    errors.append(f"{lock_path.name}: {exc}")

        # Remove stale blocked-cloud meta load failures from inactive batches; these are terminal artifacts
        # that skew dashboard failed counts after cleanup/reset.
        failed_dir = tasks_dir / "failed"
        if failed_dir.exists():
            for task_path in sorted(failed_dir.glob("*.json")):
                try:
                    task = load_json(task_path)
                except Exception as exc:
                    errors.append(f"{task_path.name}: {exc}")
                    continue
                if not isinstance(task, dict):
                    continue
                if str(task.get("status", "")).strip() != "blocked_cloud":
                    continue
                if str(task.get("task_class", "")).strip() != "meta":
                    continue
                if str(task.get("command", "")).strip() not in {"load_llm", "load_split_llm"}:
                    continue
                batch_id = str(task.get("batch_id", "")).strip()
                if batch_id and batch_id in active_batches:
                    continue
                try:
                    task_path.unlink()
                    failed_meta_removed += 1
                except Exception as exc:
                    errors.append(f"{task_path.name}: {exc}")

        return {
            "task_locks_removed": task_locks_removed,
            "failed_meta_removed": failed_meta_removed,
            "errors": errors,
        }

    def _cleanup_system_meta_artifacts(self) -> dict[str, Any]:
        """Remove leftover system meta tasks and clear brain load_llm request trackers."""
        tasks_dir = self.shared_path / "tasks"
        errors: list[str] = []
        tasks_removed = 0
        heartbeats_removed = 0
        brain_load_llm_requests_cleared = 0
        meta_commands = {"load_llm", "unload_llm", "load_split_llm", "unload_split_llm"}

        for lane in ("queue", "processing", "failed"):
            lane_dir = tasks_dir / lane
            if not lane_dir.exists():
                continue
            for task_path in sorted(lane_dir.glob("*.json")):
                try:
                    task = load_json(task_path)
                except Exception as exc:
                    errors.append(f"{task_path.name}: {exc}")
                    continue
                if not isinstance(task, dict):
                    continue
                if str(task.get("task_class", "")).strip() != "meta":
                    continue
                if str(task.get("batch_id", "")).strip() != "system":
                    continue
                if str(task.get("command", "")).strip() not in meta_commands:
                    continue
                try:
                    task_path.unlink(missing_ok=True)
                    tasks_removed += 1
                    hb_path = tasks_dir / "processing" / f"{task_path.stem}.heartbeat.json"
                    if hb_path.exists():
                        hb_path.unlink(missing_ok=True)
                        heartbeats_removed += 1
                except Exception as exc:
                    errors.append(f"{task_path.name}: {exc}")

        state_path = self.shared_path / "brain" / "state.json"
        if state_path.exists():
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                if isinstance(state, dict):
                    reqs = state.get("load_llm_requests")
                    if isinstance(reqs, dict) and reqs:
                        brain_load_llm_requests_cleared = len(reqs)
                    state["load_llm_requests"] = {}
                    with open(state_path, "w", encoding="utf-8") as f:
                        json.dump(state, f, indent=2)
            except Exception as exc:
                errors.append(f"brain/state.json: {exc}")

        return {
            "tasks_removed": tasks_removed,
            "heartbeat_files_removed": heartbeats_removed,
            "brain_load_llm_requests_cleared": brain_load_llm_requests_cleared,
            "errors": errors,
        }

    def _cleanup_orphan_task_locks(self) -> dict[str, Any]:
        """Remove orphan *.json.lock files across queue/processing/failed lanes."""
        tasks_dir = self.shared_path / "tasks"
        removed = 0
        errors: list[str] = []
        for lane in ("queue", "processing", "failed"):
            lane_dir = tasks_dir / lane
            if not lane_dir.exists():
                continue
            for lock_path in sorted(lane_dir.glob("*.json.lock")):
                task_json = Path(str(lock_path)[:-5])  # strip ".lock"
                if task_json.exists():
                    continue
                try:
                    lock_path.unlink(missing_ok=True)
                    removed += 1
                except Exception as exc:
                    errors.append(f"{lock_path.name}: {exc}")
        return {"removed": removed, "errors": errors}

    def _requeue_processing_tasks_for_reset(self, assigned_to: str | None = None) -> dict[str, Any]:
        """Return non-brain processing tasks back to queue before reset.

        `Reset to default` kills agents, so worker-owned tasks left in processing/
        become orphans unless we explicitly requeue them. This makes reset semantics
        match operator expectations: tasks drop and can be reclaimed after restart.
        """
        processing_dir = self.shared_path / "tasks" / "processing"
        queue_dir = self.shared_path / "tasks" / "queue"
        queue_dir.mkdir(parents=True, exist_ok=True)

        moved = 0
        skipped = 0
        heartbeat_files_removed = 0
        errors: list[str] = []

        target_worker = str(assigned_to or "").strip()

        for task_path in sorted(processing_dir.glob("*.json")):
            if str(task_path).endswith(".lock") or str(task_path).endswith(".heartbeat.json"):
                continue
            try:
                task = load_json(task_path)
            except Exception as exc:
                errors.append(f"{task_path.name}: {exc}")
                continue
            if not isinstance(task, dict):
                skipped += 1
                continue

            # Do not move brain-owned tasks; brain manages its own lifecycle.
            if str(task.get("executor", "")).strip() == "brain" or str(task.get("task_class", "")).strip() == "brain":
                skipped += 1
                continue

            if target_worker:
                if str(task.get("assigned_to", "")).strip() != target_worker:
                    skipped += 1
                    continue

            # Clear processing-only ownership fields so any worker can reclaim.
            task["status"] = "pending"
            for key in (
                "assigned_to",
                "started_at",
                "worker",
                "worker_pid",
                "task_runtime_sec",
                "peak_vram_mb",
            ):
                task.pop(key, None)

            # Keep attempt history but clear stale execution result/output.
            task.pop("result", None)
            task.pop("error", None)

            try:
                out_path = queue_dir / task_path.name
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(task, f, indent=2)
                task_path.unlink(missing_ok=True)
                moved += 1
            except Exception as exc:
                errors.append(f"{task_path.name}: {exc}")
                continue

            hb_path = processing_dir / f"{task_path.stem}.heartbeat.json"
            if hb_path.exists():
                try:
                    hb_path.unlink(missing_ok=True)
                    heartbeat_files_removed += 1
                except Exception as exc:
                    errors.append(f"{hb_path.name}: {exc}")

        return {
            "moved_to_queue": moved,
            "skipped": skipped,
            "heartbeat_files_removed": heartbeat_files_removed,
            "errors": errors,
        }

    def _start_plan(
        self,
        plan_name: str,
        config_json: str,
        plan_scope: str = "shoulders",
        starter_file: str = "",
        repo_path: str = "",
        repo_url: str = "",
        claimed_behavior: str = "",
        inline_query_text: str = "",
    ) -> dict[str, Any]:
        if not plan_name:
            return {"ok": False, "message": "plan_name is required"}
        scope = str(plan_scope or "").strip().lower()
        if scope not in {"shoulders", "arms"}:
            return {"ok": False, "message": f"invalid plan_scope: {plan_scope}"}

        plans = set(discover_plans(self.shared_path, plan_scope=scope))
        if plan_name not in plans:
            return {"ok": False, "message": f"unknown plan in {scope}: {plan_name}"}

        starters = set(discover_plan_starters(self.shared_path, plan_name, plan_scope=scope))
        if starter_file and starter_file not in starters:
            return {"ok": False, "message": f"invalid starter_file for {plan_name}: {starter_file}"}

        try:
            cfg_obj = json.loads(config_json) if config_json.strip() else {}
            if not isinstance(cfg_obj, dict):
                raise ValueError("config JSON must be an object")
        except Exception as e:
            return {"ok": False, "message": f"invalid config JSON: {e}"}

        # Fill missing keys from plan defaults so required placeholders are always resolvable.
        default_cfg = default_plan_config(self.shared_path, plan_name)
        if isinstance(default_cfg, dict):
            merged_cfg = dict(default_cfg)
            merged_cfg.update(cfg_obj)
            cfg_obj = merged_cfg

        try:
            cfg_obj = sanitize_config_object(cfg_obj)
        except Exception as e:
            return {"ok": False, "message": f"invalid config values: {e}"}

        repo_path_clean = sanitize_text(repo_path, max_len=800, single_line=True) if str(repo_path).strip() else ""
        try:
            repo_url_clean = normalize_github_url(repo_url) if str(repo_url).strip() else ""
        except Exception as e:
            return {"ok": False, "message": f"invalid repo_url: {e}"}
        claimed_clean = sanitize_text(claimed_behavior, max_len=8000, single_line=True) if str(claimed_behavior).strip() else ""
        inline_clean = sanitize_text(inline_query_text, max_len=16000, single_line=False) if str(inline_query_text).strip() else ""

        auto_repo_path = ""
        if plan_name == "github_analyzer" and not repo_path_clean and repo_url_clean:
            try:
                auto_repo_path, repo_url_clean = self._gpu_find_existing_repo_checkout(repo_url_clean)
            except Exception:
                auto_repo_path = ""

        if repo_path_clean:
            cfg_obj["REPO_PATH"] = repo_path_clean
        if repo_url_clean:
            cfg_obj["REPO_URL"] = repo_url_clean
            if plan_name == "github_analyzer" and not repo_path_clean and not auto_repo_path:
                cfg_obj["REPO_PATH"] = ""
        if auto_repo_path:
            cfg_obj["REPO_PATH"] = auto_repo_path
        if claimed_clean:
            cfg_obj["CLAIMED_BEHAVIOR"] = claimed_clean
        if inline_clean:
            inline_path = write_inline_input_file(self.shared_path, plan_name, inline_clean, plan_scope=scope)
            chosen_key = next((k for k in INLINE_FILE_KEYS if k in cfg_obj), "QUERY_FILE")
            cfg_obj[chosen_key] = inline_path

        # github_analyzer resolves REPO_URL by cloning when REPO_PATH is empty.
        # Preflight this dependency so dashboard users get an immediate actionable error.
        if plan_name == "github_analyzer":
            repo_path_cfg = str(cfg_obj.get("REPO_PATH", "") or "").strip()
            repo_url_cfg = str(cfg_obj.get("REPO_URL", "") or "").strip()
            if repo_url_cfg and not repo_path_cfg:
                has_git, detail = self._gpu_has_git()
                if not has_git:
                    return {
                        "ok": False,
                        "message": (
                            "Cannot submit github_analyzer with REPO_URL because git is missing on GPU host. "
                            "Provide REPO_PATH to a local checkout, or install git on gpu host."
                        ),
                        "stderr": _clip_stdio(detail),
                    }

        cfg_payload = json.dumps(cfg_obj, separators=(",", ":"))
        plan_path = f"/mnt/shared/plans/{scope}/{plan_name}"
        cfg_tmp = f"/tmp/dashboard_submit_{scope}_{plan_name}.json"
        starter_arg = starter_file.strip()
        starter_opt = f" --plan-file {shlex.quote(starter_arg)}" if starter_arg else ""
        script = (
            "set -e\n"
            f"cat > {cfg_tmp} <<'__CFG__'\n"
            f"{cfg_payload}\n"
            "__CFG__\n"
            f"python3 /mnt/shared/agents/submit.py {plan_path}{starter_opt} --config \"$(cat {cfg_tmp})\"\n"
            f"rm -f {cfg_tmp}\n"
        )
        try:
            proc = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "gpu", "bash", "-s"],
                input=script,
                capture_output=True,
                text=True,
                timeout=120,
            )
            out = {
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout": _clip_stdio(proc.stdout),
                "stderr": _clip_stdio(proc.stderr),
                "cmd": "ssh gpu bash -s (stdin: write config JSON to temp file, run submit.py)",
            }
        except subprocess.TimeoutExpired as e:
            out = {
                "ok": False,
                "returncode": -1,
                "stdout": _clip_stdio(e.stdout or "", STDIO_TIMEOUT_LIMIT) if isinstance(e.stdout, str) else "",
                "stderr": _clip_stdio(e.stderr or "", STDIO_TIMEOUT_LIMIT) if isinstance(e.stderr, str) else "",
                "cmd": "ssh gpu bash -s",
                "error": "timeout after 120s",
            }
        display_starter = starter_arg or "plan.md"
        out["message"] = (
            f"Submitted {scope}/{plan_name} ({display_starter})"
            if out.get("ok")
            else f"Failed to submit {scope}/{plan_name} ({display_starter})"
        )
        if out.get("ok") and plan_name == "github_analyzer":
            resolved_repo = str(cfg_obj.get("REPO_PATH", "") or "").strip()
            if resolved_repo:
                out["message"] += f" [local repo: {resolved_repo}]"
        return out

    def _resume_plan(self, batch_id: str) -> dict[str, Any]:
        if not batch_id:
            return {"ok": False, "message": "batch_id is required"}
        if not BATCH_ID_RE.fullmatch(batch_id):
            return {"ok": False, "message": f"invalid batch_id: {batch_id!r}"}

        brain = load_brain_state(self.shared_path)
        active = brain.get("active_batches", {}) if isinstance(brain.get("active_batches"), dict) else {}
        meta = active.get(batch_id) if isinstance(active.get(batch_id), dict) else {}
        if not meta:
            return {"ok": False, "message": f"batch_id not found in active_batches: {batch_id}"}

        plan_name = str(meta.get("plan") or "").strip()
        if not plan_name:
            plan_dir = str(meta.get("plan_dir") or "").strip()
            if plan_dir:
                plan_name = Path(plan_dir).name
        if not plan_name:
            return {"ok": False, "message": f"could not resolve plan name for batch {batch_id}"}

        cfg = meta.get("config") if isinstance(meta.get("config"), dict) else {}
        cfg = dict(cfg)
        cfg["RUN_MODE"] = "resume"
        cfg["BATCH_ID"] = batch_id
        cfg_json = json.dumps(cfg)

        plan_scope = "shoulders"
        plan_dir = str(meta.get("plan_dir") or "").strip()
        if "/plans/arms/" in plan_dir:
            plan_scope = "arms"

        out = self._start_plan(plan_name, cfg_json, plan_scope=plan_scope)
        if out.get("ok"):
            out["message"] = f"Resumed batch {batch_id} ({plan_scope}/{plan_name})"
        else:
            out["message"] = f"Failed to resume batch {batch_id} ({plan_scope}/{plan_name})"
        return out

    def _batch_outputs(self, batch_id: str) -> dict[str, Any]:
        if not batch_id:
            return {"ok": False, "message": "batch_id is required"}
        if not BATCH_ID_RE.fullmatch(batch_id):
            return {"ok": False, "message": f"invalid batch_id: {batch_id!r}"}
        return collect_batch_outputs(self.shared_path, batch_id)
