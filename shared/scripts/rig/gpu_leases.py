"""Controller-neutral GPU lease storage.

Leases are allocation authority, not observed runtime state. Expiry does not
automatically make a GPU available: callers must reconcile the GPU and release
the expired lease explicitly before another controller can acquire it.
"""

from __future__ import annotations

import json
import os
import re
import socket
import uuid
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from filelock import FileLock, Timeout


SCHEMA_VERSION = 1
DEFAULT_TTL_SECONDS = 120
VALID_CONTROLLERS = {
    "plans",
    "benchmarks",
    "supervision",
    "interactive",
    "system",
}
_TOKEN_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


class GPULeaseError(RuntimeError):
    """Base class for lease failures."""


class GPULeaseConflict(GPULeaseError):
    """Requested GPUs overlap an existing lease."""


class GPULeaseNotFound(GPULeaseError):
    """Requested lease does not exist."""


class GPULeaseOwnershipError(GPULeaseError):
    """Caller does not own the requested lease."""


class GPULeaseStateError(GPULeaseError):
    """Persisted lease state is invalid or unavailable."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("timestamp is empty")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalize_gpu_ids(gpu_ids: Iterable[int | str]) -> list[int]:
    normalized: set[int] = set()
    for value in gpu_ids:
        try:
            gpu_id = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid GPU id: {value!r}") from exc
        if gpu_id < 0:
            raise ValueError(f"GPU id must be non-negative: {gpu_id}")
        normalized.add(gpu_id)
    if not normalized:
        raise ValueError("at least one GPU id is required")
    return sorted(normalized)


def _validate_token(value: str, label: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _TOKEN_RE.fullmatch(normalized):
        raise ValueError(f"invalid {label}: {value!r}")
    return normalized


class GPULeaseStore:
    """Locked repository for atomic single- and multi-GPU leases."""

    def __init__(self, shared_path: Path | str, *, lock_timeout: float = 5.0):
        self.shared_path = Path(shared_path)
        self.path = self.shared_path / "gpus" / "leases.json"
        self.lock_path = Path(str(self.path) + ".lock")
        self.lock_timeout = float(lock_timeout)

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "revision": 0,
            "updated_at": "",
            "leases": [],
        }

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty_state()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise GPULeaseStateError(f"cannot read GPU lease state: {self.path}") from exc
        if not isinstance(payload, dict):
            raise GPULeaseStateError(f"GPU lease state must be an object: {self.path}")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise GPULeaseStateError(
                f"unsupported GPU lease schema {payload.get('schema_version')!r}: {self.path}"
            )
        if not isinstance(payload.get("leases"), list):
            raise GPULeaseStateError(f"GPU lease state leases must be a list: {self.path}")
        try:
            payload["revision"] = int(payload.get("revision", 0))
        except (TypeError, ValueError) as exc:
            raise GPULeaseStateError(f"invalid GPU lease revision: {self.path}") from exc
        return payload

    def _write_unlocked(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            temp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            temp_path.replace(self.path)
        finally:
            temp_path.unlink(missing_ok=True)

    def _with_lock(self, operation):
        lock = FileLock(str(self.lock_path), timeout=self.lock_timeout)
        try:
            with lock:
                return operation()
        except Timeout as exc:
            raise GPULeaseStateError(
                f"timed out locking GPU lease state: {self.lock_path}"
            ) from exc

    @staticmethod
    def is_expired(lease: dict[str, Any], *, now: datetime | None = None) -> bool:
        current = now or _utc_now()
        try:
            return _parse_iso(lease.get("expires_at")) <= current
        except (TypeError, ValueError):
            return True

    def read_state(self) -> dict[str, Any]:
        return deepcopy(self._read_unlocked())

    def list_leases(self) -> list[dict[str, Any]]:
        return deepcopy(self._read_unlocked()["leases"])

    def get(self, lease_id: str) -> dict[str, Any] | None:
        wanted = str(lease_id or "").strip()
        for lease in self._read_unlocked()["leases"]:
            if isinstance(lease, dict) and str(lease.get("lease_id")) == wanted:
                return deepcopy(lease)
        return None

    def for_gpu(self, gpu_id: int | str) -> dict[str, Any] | None:
        wanted = int(gpu_id)
        matches = []
        for lease in self._read_unlocked()["leases"]:
            if not isinstance(lease, dict):
                continue
            try:
                members = {int(value) for value in lease.get("gpu_ids", [])}
            except (TypeError, ValueError):
                continue
            if wanted in members:
                matches.append(lease)
        if len(matches) > 1:
            raise GPULeaseStateError(f"GPU {wanted} appears in multiple leases")
        return deepcopy(matches[0]) if matches else None

    def acquire(
        self,
        *,
        controller: str,
        owner: str,
        gpu_ids: Iterable[int | str],
        run_id: str,
        lane: str | None = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        ports: Iterable[int | str] = (),
        runtime_ids: Iterable[str] = (),
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_controller = _validate_token(controller, "controller")
        if normalized_controller not in VALID_CONTROLLERS:
            raise ValueError(f"unsupported controller: {normalized_controller}")
        normalized_lane = _validate_token(lane or normalized_controller, "lane")
        normalized_owner = str(owner or "").strip()
        normalized_run_id = str(run_id or "").strip()
        if not normalized_owner:
            raise ValueError("owner is required")
        if not normalized_run_id:
            raise ValueError("run_id is required")
        normalized_gpu_ids = _normalize_gpu_ids(gpu_ids)
        normalized_ports = sorted({int(value) for value in ports})
        if len(normalized_ports) not in {0, len(normalized_gpu_ids)}:
            raise ValueError("ports must be empty or contain one unique port per GPU")
        ttl = int(ttl_seconds)
        if ttl <= 0:
            raise ValueError("ttl_seconds must be positive")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")

        def operation() -> dict[str, Any]:
            state = self._read_unlocked()
            requested = set(normalized_gpu_ids)
            now = _utc_now()
            for existing in state["leases"]:
                if not isinstance(existing, dict):
                    raise GPULeaseStateError("GPU lease state contains a non-object lease")
                try:
                    held = {int(value) for value in existing.get("gpu_ids", [])}
                except (TypeError, ValueError) as exc:
                    raise GPULeaseStateError("GPU lease contains invalid gpu_ids") from exc
                overlap = sorted(requested.intersection(held))
                if not overlap:
                    continue
                expiry_state = (
                    "expired_requires_reconciliation"
                    if self.is_expired(existing, now=now)
                    else "active"
                )
                raise GPULeaseConflict(
                    f"GPU(s) {overlap} held by lease {existing.get('lease_id')} "
                    f"({existing.get('controller')}/{existing.get('owner')}, {expiry_state})"
                )

            revision = int(state["revision"]) + 1
            lease = {
                "lease_id": f"{normalized_controller}-{uuid.uuid4().hex}",
                "generation": revision,
                "controller": normalized_controller,
                "lane": normalized_lane,
                "owner": normalized_owner,
                "owner_host": socket.gethostname(),
                "owner_pid": os.getpid(),
                "run_id": normalized_run_id,
                "gpu_ids": normalized_gpu_ids,
                "ports": normalized_ports,
                "runtime_ids": sorted(
                    {str(value).strip() for value in runtime_ids if str(value).strip()}
                ),
                "acquired_at": _iso(now),
                "renewed_at": _iso(now),
                "expires_at": _iso(now + timedelta(seconds=ttl)),
                "metadata": deepcopy(metadata or {}),
            }
            state["revision"] = revision
            state["updated_at"] = _iso(now)
            state["leases"].append(lease)
            self._write_unlocked(state)
            return deepcopy(lease)

        return self._with_lock(operation)

    def renew(
        self,
        lease_id: str,
        *,
        owner: str,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> dict[str, Any]:
        wanted = str(lease_id or "").strip()
        normalized_owner = str(owner or "").strip()
        ttl = int(ttl_seconds)
        if not wanted:
            raise ValueError("lease_id is required")
        if not normalized_owner:
            raise ValueError("owner is required")
        if ttl <= 0:
            raise ValueError("ttl_seconds must be positive")

        def operation() -> dict[str, Any]:
            state = self._read_unlocked()
            now = _utc_now()
            for lease in state["leases"]:
                if not isinstance(lease, dict) or str(lease.get("lease_id")) != wanted:
                    continue
                if str(lease.get("owner")) != normalized_owner:
                    raise GPULeaseOwnershipError(
                        f"lease {wanted} is owned by {lease.get('owner')}, not {normalized_owner}"
                    )
                if self.is_expired(lease, now=now):
                    raise GPULeaseConflict(
                        f"lease {wanted} expired and requires runtime reconciliation"
                    )
                lease["renewed_at"] = _iso(now)
                lease["expires_at"] = _iso(now + timedelta(seconds=ttl))
                state["revision"] = int(state["revision"]) + 1
                state["updated_at"] = _iso(now)
                self._write_unlocked(state)
                return deepcopy(lease)
            raise GPULeaseNotFound(f"lease not found: {wanted}")

        return self._with_lock(operation)

    def release(self, lease_id: str, *, owner: str, force: bool = False) -> None:
        wanted = str(lease_id or "").strip()
        normalized_owner = str(owner or "").strip()
        if not wanted:
            raise ValueError("lease_id is required")
        if not normalized_owner and not force:
            raise ValueError("owner is required unless force=True")

        def operation() -> None:
            state = self._read_unlocked()
            for index, lease in enumerate(state["leases"]):
                if not isinstance(lease, dict) or str(lease.get("lease_id")) != wanted:
                    continue
                if not force and str(lease.get("owner")) != normalized_owner:
                    raise GPULeaseOwnershipError(
                        f"lease {wanted} is owned by {lease.get('owner')}, not {normalized_owner}"
                    )
                state["leases"].pop(index)
                state["revision"] = int(state["revision"]) + 1
                state["updated_at"] = _iso(_utc_now())
                self._write_unlocked(state)
                return
            raise GPULeaseNotFound(f"lease not found: {wanted}")

        self._with_lock(operation)
