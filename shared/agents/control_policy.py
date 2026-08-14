"""Boot-profile and controller-admission policy.

This state describes operator intent. Live GPU activity remains derived from
leases, runtimes, tasks, and heartbeats.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from filelock import FileLock, Timeout

from gpu_leases import VALID_CONTROLLERS


SCHEMA_VERSION = 1
BOOT_PROFILE_NEUTRAL = "neutral"
BOOT_PROFILE_PLAN_READY = "plan_ready"
BOOT_PROFILE_BENCHMARK_READY = "benchmark_ready"
BOOT_PROFILE_SUPERVISION_READY = "supervision_ready"
BOOT_PROFILE_CUSTOM = "custom"
VALID_BOOT_PROFILES = {
    BOOT_PROFILE_NEUTRAL,
    BOOT_PROFILE_PLAN_READY,
    BOOT_PROFILE_BENCHMARK_READY,
    BOOT_PROFILE_SUPERVISION_READY,
    BOOT_PROFILE_CUSTOM,
}
PROFILE_DEFAULT_ADMISSION = {
    BOOT_PROFILE_NEUTRAL: set(),
    BOOT_PROFILE_PLAN_READY: {"plans"},
    BOOT_PROFILE_BENCHMARK_READY: {"benchmarks"},
    BOOT_PROFILE_SUPERVISION_READY: {"supervision"},
    BOOT_PROFILE_CUSTOM: {"interactive"},
}


class ControlPolicyError(RuntimeError):
    """Control policy cannot be read, validated, or persisted."""


def control_policy_file(shared_path: Path | str) -> Path:
    return Path(shared_path) / "brain" / "control_policy.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _default_payload(reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "boot_profile": BOOT_PROFILE_NEUTRAL,
        "admitted_controllers": [],
        "reason": reason,
        "updated_at": "",
        "updated_by": "",
        "transition": None,
    }


def _normalize_controllers(values: Iterable[str]) -> list[str]:
    normalized = sorted({str(value or "").strip().lower() for value in values})
    invalid = [value for value in normalized if value not in VALID_CONTROLLERS]
    if invalid:
        raise ValueError(f"unsupported admitted controller(s): {', '.join(invalid)}")
    return normalized


def read_control_policy(shared_path: Path | str) -> dict[str, Any]:
    path = control_policy_file(shared_path)
    if not path.exists():
        return _default_payload("missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _default_payload("unreadable")
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        return _default_payload("invalid_schema")
    profile = str(payload.get("boot_profile") or "").strip().lower()
    if profile not in VALID_BOOT_PROFILES:
        return _default_payload("invalid_boot_profile")
    admissions = payload.get("admitted_controllers")
    if not isinstance(admissions, list):
        return _default_payload("invalid_admission_policy")
    try:
        normalized_admissions = _normalize_controllers(admissions)
    except ValueError:
        return _default_payload("invalid_admission_policy")
    transition = payload.get("transition")
    if transition is not None and not isinstance(transition, dict):
        return _default_payload("invalid_transition")
    return {
        "schema_version": SCHEMA_VERSION,
        "boot_profile": profile,
        "admitted_controllers": normalized_admissions,
        "reason": str(payload.get("reason") or "").strip(),
        "updated_at": str(payload.get("updated_at") or "").strip(),
        "updated_by": str(payload.get("updated_by") or "").strip(),
        "transition": transition,
    }


def write_control_policy(
    shared_path: Path | str,
    boot_profile: str,
    *,
    admitted_controllers: Iterable[str] | None = None,
    reason: str = "",
    updated_by: str = "",
    transition: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = str(boot_profile or "").strip().lower()
    if profile not in VALID_BOOT_PROFILES:
        raise ValueError(f"invalid boot profile: {boot_profile}")
    admissions = _normalize_controllers(
        PROFILE_DEFAULT_ADMISSION[profile]
        if admitted_controllers is None
        else admitted_controllers
    )
    if transition is not None and not isinstance(transition, dict):
        raise ValueError("transition must be an object or null")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "boot_profile": profile,
        "admitted_controllers": admissions,
        "reason": str(reason or "").strip(),
        "updated_at": _now_iso(),
        "updated_by": str(updated_by or "").strip(),
        "transition": transition,
    }
    path = control_policy_file(shared_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(path) + ".lock", timeout=5)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with lock:
            try:
                temp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                temp_path.replace(path)
            finally:
                temp_path.unlink(missing_ok=True)
    except Timeout as exc:
        raise ControlPolicyError(f"timed out locking control policy: {path}") from exc
    return payload


def task_controller(task: dict[str, Any]) -> str:
    if str(task.get("batch_id") or "").strip() == "system":
        return "system"
    explicit = str(task.get("controller") or "").strip().lower()
    if explicit:
        return explicit
    created_by = str(task.get("created_by") or "").strip().lower()
    batch_id = str(task.get("batch_id") or "").strip().lower()
    if created_by in {"runtime-prep", "benchmark-prep"} or batch_id.startswith("runtime_prep_"):
        return "benchmarks"
    return "plans"


def control_policy_allows_task(
    shared_path: Path | str,
    task: dict[str, Any],
) -> tuple[bool, str]:
    controller = task_controller(task)
    if controller == "system":
        return True, ""
    if controller not in VALID_CONTROLLERS:
        return False, f"unsupported_controller:{controller or 'missing'}"
    policy = read_control_policy(shared_path)
    if controller in set(policy["admitted_controllers"]):
        return True, ""
    return False, f"controller_not_admitted:{controller}"
