"""Execution controls for prepared plans and active batches."""

import base64
import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from .plans import collect_batch_outputs, run_shell
from .utils import load_json


def load_brain_state(shared_path: Path) -> dict[str, Any]:
    return load_json(shared_path / "brain" / "state.json") or {}

BATCH_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")
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

    def submit_prepared(self, plan_name: str, plan_scope: str, starter_file: str,
                        config: dict[str, Any]) -> dict[str, Any]:
        """Consume a gateway-prepared request; never discover or build a plan."""
        scope = str(plan_scope).strip()
        if scope not in {"shoulders", "arms"} or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", plan_name):
            return {"ok": False, "message": "invalid plan reference"}
        plan_path = self.shared_path / "plans" / scope / plan_name
        if not plan_path.is_dir():
            return {"ok": False, "message": "plan directory not found"}
        if starter_file:
            starter = (plan_path / starter_file).resolve()
            if not starter.is_file() or not starter.is_relative_to(plan_path.resolve()):
                return {"ok": False, "message": "invalid starter file"}
        if not isinstance(config, dict) or len(config) > 120:
            return {"ok": False, "message": "invalid prepared config"}
        for key, value in config.items():
            if not isinstance(key, str) or not re.fullmatch(r"[A-Z0-9_]{1,64}", key):
                return {"ok": False, "message": "invalid prepared config key"}
            if value is not None and not isinstance(value, (str, int, float, bool)):
                return {"ok": False, "message": "invalid prepared config value"}
        payload = json.dumps(config, separators=(",", ":"))
        remote_path = f"/mnt/shared/plans/{scope}/{plan_name}"
        starter_opt = f" --plan-file {shlex.quote(starter_file)}" if starter_file else ""
        encoded = base64.b64encode(payload.encode("utf-8")).decode("ascii")
        script = ("set -e\n" +
                  f"python3 /mnt/shared/agents/submit.py {shlex.quote(remote_path)}{starter_opt} " +
                  f"--config \"$(printf %s {encoded} | base64 -d)\"\n")
        try:
            proc = subprocess.run(["ssh", "-o", "BatchMode=yes", "gpu", "bash", "-s"],
                                  input=script,
                                  capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            return {"ok": False, "message": "GPU submission timed out", "returncode": -1}
        return {"ok": proc.returncode == 0, "returncode": proc.returncode,
                "stdout": _clip_stdio(proc.stdout), "stderr": _clip_stdio(proc.stderr),
                "message": f"Submitted {scope}/{plan_name}" if proc.returncode == 0 else
                           f"Failed to submit {scope}/{plan_name}"}
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



    def _batch_outputs(self, batch_id: str) -> dict[str, Any]:
        if not batch_id:
            return {"ok": False, "message": "batch_id is required"}
        if not BATCH_ID_RE.fullmatch(batch_id):
            return {"ok": False, "message": f"invalid batch_id: {batch_id!r}"}
        return collect_batch_outputs(self.shared_path, batch_id)
