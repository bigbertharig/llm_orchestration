"""Claim tokens and first-result-wins finalization for task files.

A task moves queue/ -> processing/ -> complete/ or failed/. The brain can requeue a task whose worker
looks gone (orphan) or stuck, and another worker can claim it. If the first worker was only cut off
(a CPU worker that lost the shared drive for a while) it can still finish later. Without a check, that
late finish deleted the new owner's processing file and overwrote the result.

Rules:
- Every claim stamps a fresh `claim_id`; the brain scrubs it on requeue like `assigned_to`.
- A successful result is accepted if no result exists yet (first result wins). The complete/ file is
  created atomically with a hard link, which fails if it exists (NFS-safe).
- A worker only removes processing/ and the task heartbeat while it still owns the claim.
- A failure from a claim that has moved on is not written to failed/ (that would queue a duplicate
  retry); the current claim decides the outcome.
- Anything not accepted goes to tasks/late/<task_id>.<claim_id>.json for audit.
- Workers skip and remove queued copies of tasks that already have a result.

Design: llm-gateway docs/DESIGN_NOTES.md, "More than one rig" (the same rules apply across rigs).
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

LATE_DIR = "late"


def stamp_claim(task: Dict[str, Any]) -> str:
    """Give this claim a fresh id (call where `assigned_to` is set)."""
    task["claim_id"] = uuid.uuid4().hex
    return task["claim_id"]


def _read(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def owns_claim(current: Optional[Dict[str, Any]], task: Dict[str, Any]) -> bool:
    """True if the processing file `current` is still this worker's claim of `task`."""
    if current is None:
        return False
    claim_id = task.get("claim_id")
    if claim_id:
        return current.get("claim_id") == claim_id
    # Claimed before claim ids existed: same worker and same attempt.
    return (current.get("assigned_to") == task.get("assigned_to")
            and current.get("attempts") == task.get("attempts"))


def _create_exclusive(path: Path, data: Dict[str, Any]) -> bool:
    """Write `path` only if it does not exist yet. True if this call created it."""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    try:
        os.link(tmp, path)
        return True
    except FileExistsError:
        return False
    finally:
        tmp.unlink(missing_ok=True)


def _write_late(tasks_path: Path, task: Dict[str, Any], reason: str) -> Path:
    late_dir = tasks_path / LATE_DIR
    late_dir.mkdir(parents=True, exist_ok=True)
    task["late_reason"] = reason
    dest = late_dir / f"{task['task_id']}.{task.get('claim_id') or 'noclaim'}.json"
    with open(dest, "w") as f:
        json.dump(task, f, indent=2)
    return dest


def finalize_task(tasks_path: Path, task: Dict[str, Any], result: Dict[str, Any]) -> str:
    """Record a finished claim. Returns "complete", "failed" or "late".

    `tasks_path` is the tasks/ root (queue, processing, complete, failed, late below it).
    """
    task_id = task["task_id"]
    processing_file = tasks_path / "processing" / f"{task_id}.json"
    heartbeat_file = tasks_path / "processing" / f"{task_id}.heartbeat.json"
    owns = owns_claim(_read(processing_file), task)
    success = bool(result.get("success"))

    task["status"] = "complete" if success else "failed"
    task["completed_at"] = datetime.now().isoformat()
    task["result"] = result

    if success:
        if not _create_exclusive(tasks_path / "complete" / f"{task_id}.json", task):
            _write_late(tasks_path, task, "result_already_recorded")
            outcome = "late"
        else:
            outcome = "complete"
    elif owns:
        with open(tasks_path / "failed" / f"{task_id}.json", "w") as f:
            json.dump(task, f, indent=2)
        outcome = "failed"
    else:
        _write_late(tasks_path, task, "claim_moved_on")
        outcome = "late"

    if owns:
        processing_file.unlink(missing_ok=True)
        heartbeat_file.unlink(missing_ok=True)
    return outcome


def drop_if_already_recorded(tasks_path: Path, queue_file: Path) -> bool:
    """True (and the queued copy removed) if this task already has a result in complete/."""
    if (tasks_path / "complete" / queue_file.name).exists():
        queue_file.unlink(missing_ok=True)
        return True
    return False
