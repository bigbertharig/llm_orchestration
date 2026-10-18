"""Rig-side GPU allocation record: leases and claims, kept and written by the rig itself.

Controllers (the Pi manager today, possibly others later) decide placement and ask the rig to record
it. The rig is the single writer, on its own disk, under the lease store's lock (not the rig mode
lock, which a model load holds for minutes). So the record survives a lost network, does not depend
on which controller is active, and two controllers cannot double-book a card.

Every write is mirrored to the shared drive (`<shared>/gpus/`), where orchestration GPU workers read
leases, whenever the drive is mounted. The mirror is never read back, except to seed an empty local
record once (the first run after leases moved here from the drive).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from filelock import FileLock


class StateError(RuntimeError):
    """kind: invalid | conflict | not_found | ownership | state"""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


class HardwareState:
    def __init__(self, root: Path, agents_dir: Path, mirror_root: Path | None = None):
        sys.path.insert(0, str(agents_dir))
        import gpu_leases  # orchestration's lease store: same schema the GPU workers read

        self.leases_mod = gpu_leases
        self.root = root
        self.mirror_root = mirror_root
        self.store = gpu_leases.GPULeaseStore(root)
        self.claims_path = root / "gpus" / "claims.json"
        self.claims_lock = FileLock(str(self.claims_path) + ".lock", timeout=10)
        (root / "gpus").mkdir(parents=True, exist_ok=True)
        self._seed_from_mirror()

    # ---------------------------------------------------------------- mirror

    def _seed_from_mirror(self) -> None:
        if self.mirror_root is None:
            return
        for name in ("leases.json", "claims.json"):
            local, remote = self.root / "gpus" / name, self.mirror_root / "gpus" / name
            try:
                if not local.exists() and remote.is_file():
                    shutil.copyfile(remote, local)
            except OSError:
                pass  # drive missing or unreadable: start empty, the rig does not depend on it

    def mirror(self) -> str:
        if self.mirror_root is None or not (self.mirror_root / "gpus").is_dir():
            return "skipped"
        try:
            for name in ("leases.json", "claims.json"):
                local = self.root / "gpus" / name
                if local.exists():
                    _write_json(self.mirror_root / "gpus" / name, json.loads(local.read_text(encoding="utf-8")))
            return "ok"
        except (OSError, ValueError) as exc:
            return f"error: {exc}"

    # ---------------------------------------------------------------- claims

    def claims(self) -> list[dict[str, Any]]:
        if not self.claims_path.exists():
            return []
        return list(json.loads(self.claims_path.read_text(encoding="utf-8")).get("claims", []))

    def add_claim(self, name: str, gpu_ids: list[int], note: str = "", by: str = "") -> dict[str, Any]:
        name = str(name or "").strip()
        ids = sorted({int(g) for g in gpu_ids})
        if not name or not ids:
            raise StateError("invalid", "a claim needs a name and at least one GPU")
        with self.claims_lock:
            claims = self.claims()
            if any(c["name"] == name for c in claims):
                raise StateError("conflict", f"claim {name} already exists")
            for c in claims:
                overlap = sorted(set(ids) & set(c["gpu_ids"]))
                if overlap:
                    raise StateError("conflict", f"GPU {overlap} already claimed by {c['name']}")
            claim = {"name": name, "gpu_ids": ids, "note": note, "by": by, "at": _now()}
            _write_json(self.claims_path, {"schema_version": 1, "claims": claims + [claim]})
        self.mirror()
        return claim

    def remove_claim(self, name: str) -> dict[str, Any]:
        with self.claims_lock:
            claims = self.claims()
            match = [c for c in claims if c["name"] == name]
            if not match:
                raise StateError("not_found", f"no claim {name}")
            _write_json(self.claims_path, {"schema_version": 1, "claims": [c for c in claims if c["name"] != name]})
        self.mirror()
        return match[0]

    # ---------------------------------------------------------------- leases

    def leases(self) -> list[dict[str, Any]]:
        try:
            return self.store.list_leases()
        except self.leases_mod.GPULeaseStateError as exc:
            raise StateError("state", str(exc)) from exc

    def acquire(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Record a lease a controller placed. Refuses overlap with any lease, and claimed cards
        unless the spec names that claim (metadata.claim)."""
        gpu_ids = sorted({int(g) for g in spec.get("gpu_ids", [])})
        metadata = dict(spec.get("metadata") or {})
        for c in self.claims():
            taken = sorted(set(gpu_ids) & set(c["gpu_ids"]))
            if taken and metadata.get("claim") != c["name"]:
                raise StateError("conflict", f"GPU {taken} claimed by {c['name']}")
        try:
            lease = self.store.acquire(
                controller=spec.get("controller", ""), owner=spec.get("owner", ""), gpu_ids=gpu_ids,
                run_id=spec.get("run_id", ""), lane=spec.get("lane"), ttl_seconds=int(spec.get("ttl_s", 120)),
                ports=spec.get("ports", []), metadata=metadata)
        except self.leases_mod.GPULeaseConflict as exc:
            raise StateError("conflict", str(exc)) from exc
        except ValueError as exc:
            raise StateError("invalid", str(exc)) from exc
        self.mirror()
        return lease

    def renew(self, lease_id: str, owner: str, ttl_s: int) -> dict[str, Any]:
        try:
            lease = self.store.renew(lease_id, owner=owner, ttl_seconds=int(ttl_s))
        except self.leases_mod.GPULeaseNotFound as exc:
            raise StateError("not_found", str(exc)) from exc
        except self.leases_mod.GPULeaseOwnershipError as exc:
            raise StateError("ownership", str(exc)) from exc
        except self.leases_mod.GPULeaseConflict as exc:
            raise StateError("conflict", str(exc)) from exc
        self.mirror()
        return lease

    def release(self, lease_id: str, owner: str = "", force: bool = False) -> None:
        try:
            self.store.release(lease_id, owner=owner, force=force)
        except self.leases_mod.GPULeaseNotFound as exc:
            raise StateError("not_found", str(exc)) from exc
        except self.leases_mod.GPULeaseOwnershipError as exc:
            raise StateError("ownership", str(exc)) from exc
        except ValueError as exc:
            raise StateError("invalid", str(exc)) from exc
        self.mirror()
