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
from datetime import datetime, timedelta, timezone
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

    # A claim takes cards out of the shared pool: only leases naming it get them. It may be placed on
    # cards that are still leased (the holder keeps them until it releases), which is how a split
    # reserves its cards while it waits for them (docs: llm-gateway MANAGER.md, "Reservations").
    # With ttl_s the claim lapses unless its owner renews it, so a dead run frees its cards; without
    # one it is permanent until removed.

    @staticmethod
    def _expired(claim: dict[str, Any], now: datetime | None = None) -> bool:
        expires = claim.get("expires_at")
        if not expires:
            return False
        try:
            when = datetime.fromisoformat(str(expires).replace("Z", "+00:00"))
        except ValueError:
            return True
        return when <= (now or datetime.now(timezone.utc))

    @staticmethod
    def _expiry(ttl_s: int | None) -> str | None:
        if not ttl_s:
            return None
        if int(ttl_s) <= 0:
            raise StateError("invalid", "ttl_s must be positive")
        return (datetime.now(timezone.utc) + timedelta(seconds=int(ttl_s))).isoformat(
            timespec="seconds").replace("+00:00", "Z")

    def _read_claims(self) -> list[dict[str, Any]]:
        if not self.claims_path.exists():
            return []
        return list(json.loads(self.claims_path.read_text(encoding="utf-8")).get("claims", []))

    def _write_claims(self, claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Write the live claims (expired ones dropped and logged); returns them. Under claims_lock."""
        now = datetime.now(timezone.utc)
        live = [c for c in claims if not self._expired(c, now)]
        gone = [c for c in claims if self._expired(c, now)]
        if gone:
            with open(self.root / "gpus" / "claims_expired.log", "a", encoding="utf-8") as log:
                for c in gone:
                    log.write(f"{_now()} expired {c['name']} gpus={c['gpu_ids']} owner={c.get('owner', '')} "
                              f"expires_at={c.get('expires_at')} purpose={c.get('purpose', '')}\n")
        _write_json(self.claims_path, {"schema_version": 1, "claims": live})
        return live

    def claims(self) -> list[dict[str, Any]]:
        """Live claims: expired ones are ignored here and dropped from the record on the next write."""
        now = datetime.now(timezone.utc)
        return [c for c in self._read_claims() if not self._expired(c, now)]

    def add_claim(self, name: str, gpu_ids: list[int], note: str = "", by: str = "", *,
                  owner: str = "", ttl_s: int | None = None, purpose: str = "", model: str = "") -> dict[str, Any]:
        name = str(name or "").strip()
        ids = sorted({int(g) for g in gpu_ids})
        if not name or not ids:
            raise StateError("invalid", "a claim needs a name and at least one GPU")
        expires = self._expiry(ttl_s)
        with self.claims_lock:
            claims = self.claims()
            if any(c["name"] == name for c in claims):
                raise StateError("conflict", f"claim {name} already exists")
            for c in claims:
                overlap = sorted(set(ids) & set(c["gpu_ids"]))
                if overlap:
                    raise StateError("conflict", f"GPU {overlap} already claimed by {c['name']}")
            claim = {"name": name, "gpu_ids": ids, "note": note, "by": by, "at": _now(),
                     "owner": str(owner or by or "").strip()}
            if expires:
                claim.update(ttl_s=int(ttl_s), expires_at=expires)
            if purpose:
                claim["purpose"] = purpose
            if model:
                claim["model"] = model
            self._write_claims(self._read_claims() + [claim])
        self.mirror()
        return claim

    def extend_claim(self, name: str, gpu_ids: list[int], owner: str = "") -> dict[str, Any]:
        """Add cards to a live claim in one write (a wait that collects cards as they free). Refuses cards
        another live claim holds; the cards may still be leased, like any claimed card."""
        ids = sorted({int(g) for g in gpu_ids})
        if not ids:
            raise StateError("invalid", "extend needs at least one GPU")
        with self.claims_lock:
            claims = self._read_claims()
            now = datetime.now(timezone.utc)
            match = next((c for c in claims if c["name"] == name and not self._expired(c, now)), None)
            if match is None:
                raise StateError("not_found", f"no claim {name} (expired or removed)")
            if owner and match.get("owner") and match["owner"] != owner:
                raise StateError("ownership", f"claim {name} is owned by {match['owner']}, not {owner}")
            for c in claims:
                if c is match or self._expired(c, now):
                    continue
                overlap = sorted(set(ids) & set(c["gpu_ids"]))
                if overlap:
                    raise StateError("conflict", f"GPU {overlap} already claimed by {c['name']}")
            match["gpu_ids"] = sorted(set(match["gpu_ids"]) | set(ids))
            match["extended_at"] = _now()
            self._write_claims(claims)
        self.mirror()
        return match

    def renew_claim(self, name: str, owner: str = "", ttl_s: int | None = None) -> dict[str, Any]:
        """Push a claim's expiry out by ttl_s (default: its own ttl_s). An expired claim is gone."""
        with self.claims_lock:
            claims = self._read_claims()
            match = next((c for c in claims if c["name"] == name and not self._expired(c)), None)
            if match is None:
                raise StateError("not_found", f"no claim {name} (expired or removed)")
            if owner and match.get("owner") and match["owner"] != owner:
                raise StateError("ownership", f"claim {name} is owned by {match['owner']}, not {owner}")
            ttl = int(ttl_s or match.get("ttl_s") or 0)
            if not ttl:
                return match  # permanent claim: nothing to renew
            match.update(ttl_s=ttl, expires_at=self._expiry(ttl), renewed_at=_now())
            self._write_claims(claims)
        self.mirror()
        return match

    def remove_claim(self, name: str) -> dict[str, Any]:
        """Remove a claim by name. No owner check: this is also the operator's override."""
        with self.claims_lock:
            claims = self.claims()
            match = [c for c in claims if c["name"] == name]
            if not match:
                raise StateError("not_found", f"no claim {name}")
            self._write_claims([c for c in self._read_claims() if c["name"] != name])
        self.mirror()
        return match[0]

    def remove_claims_on(self, gpu_ids: list[int]) -> list[dict[str, Any]]:
        """Remove every claim that covers any of these cards (the operator's kill)."""
        ids = {int(g) for g in gpu_ids}
        with self.claims_lock:
            claims = self.claims()
            dropped = [c for c in claims if ids & set(c["gpu_ids"])]
            if dropped:
                names = {c["name"] for c in dropped}
                self._write_claims([c for c in self._read_claims() if c["name"] not in names])
        if dropped:
            self.mirror()
        return dropped

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
