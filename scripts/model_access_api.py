#!/usr/bin/env python3
"""Authenticated localhost API for interactive access to rig models.

This service delegates placement and runtime startup to load-gpu-runtime. It
only adds a request/session layer so a gateway can hold and renew the canonical
interactive GPU lease while inference is in flight.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlsplit


DEFAULT_SHARED_ROOTS = (
    Path("/media/bryan/shared"),
    Path("/mnt/shared"),
)


class ModelAccessError(RuntimeError):
    status_code = 400


class ModelNotFound(ModelAccessError):
    status_code = 404


class SessionNotFound(ModelAccessError):
    status_code = 404


class RuntimeUnavailable(ModelAccessError):
    status_code = 503


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def find_shared_root(explicit: str = "") -> Path:
    candidates = [Path(explicit)] if explicit else list(DEFAULT_SHARED_ROOTS)
    for candidate in candidates:
        if (candidate / "agents" / "models.catalog.json").is_file():
            return candidate.resolve()
    checked = ", ".join(str(path) for path in candidates)
    raise RuntimeError(f"cannot locate shared orchestration tree; checked: {checked}")


def load_access_model_ids(path: Path) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read interactive model config: {path}") from exc
    if payload.get("schema_version") != 1:
        raise RuntimeError(f"unsupported interactive model config schema: {path}")
    configured = payload.get("models")
    if not isinstance(configured, list) or not configured:
        raise RuntimeError(f"interactive model config has no models: {path}")
    model_ids = [str(model_id).strip() for model_id in configured]
    if any(not model_id for model_id in model_ids) or len(set(model_ids)) != len(model_ids):
        raise RuntimeError(f"interactive model config contains invalid model ids: {path}")
    return model_ids


def load_catalog(path: Path, access_config_path: Path) -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read model catalog: {path}") from exc
    catalog: dict[str, dict[str, Any]] = {}
    for row in payload.get("models", []):
        if not isinstance(row, dict):
            continue
        model_id = str(row.get("id") or "").strip()
        placement = str(row.get("placement") or "").strip()
        if model_id:
            catalog[model_id] = row

    models: dict[str, dict[str, Any]] = {}
    for model_id in load_access_model_ids(access_config_path):
        row = catalog.get(model_id)
        if row is None:
            raise RuntimeError(
                f"interactive model {model_id!r} is missing from catalog: {path}"
            )
        placement = str(row.get("placement") or "").strip()
        if placement not in {"single_gpu", "split_gpu"}:
            raise RuntimeError(
                f"interactive model {model_id!r} has unsupported placement: {placement!r}"
            )
        models[model_id] = row
    return models


def parse_last_json(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise RuntimeUnavailable("runtime loader returned no JSON result")


class RuntimeLoader:
    def __init__(
        self,
        script: Path,
        shared_root: Path,
        *,
        timeout_seconds: int = 420,
    ):
        self.script = script
        self.shared_root = shared_root
        self.timeout_seconds = int(timeout_seconds)

    def __call__(self, model_id: str, owner: str, ttl_seconds: int) -> dict[str, Any]:
        env = dict(os.environ)
        env["LLM_ORCHESTRATION_SHARED_ROOT"] = str(self.shared_root)
        command = [
            str(self.script),
            "--model", model_id,
            "--role", "worker",
            "--gpu", "auto",
            "--lane", "CHATTING",
            "--lease-owner", owner,
            "--lease-ttl-seconds", str(ttl_seconds),
            "--timeout", str(self.timeout_seconds),
            "--yes",
            "--json",
        ]
        try:
            result = subprocess.run(
                command,
                text=True,
                capture_output=True,
                check=False,
                timeout=self.timeout_seconds + 90,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeUnavailable(
                f"model load exceeded {self.timeout_seconds + 90} seconds"
            ) from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "runtime load failed").strip()
            raise RuntimeUnavailable(detail[-2000:])
        payload = parse_last_json(result.stdout)
        if not payload.get("ok") or not isinstance(payload.get("lease"), dict):
            raise RuntimeUnavailable("runtime loader returned an invalid result")
        return payload


@dataclass
class Allocation:
    model_id: str
    owner: str
    lease_id: str
    endpoint: str
    runtime_model: str
    placement: dict[str, Any]
    sessions: set[str] = field(default_factory=set)


@dataclass
class Session:
    session_id: str
    model_id: str
    expires_at: datetime


class ModelAccessService:
    def __init__(
        self,
        *,
        catalog_path: Path,
        access_config_path: Path,
        lease_store: Any,
        runtime_loader: Callable[[str, str, int], dict[str, Any]],
        rig_host: str,
        max_ttl_seconds: int = 3600,
    ):
        self.catalog_path = catalog_path
        self.access_config_path = access_config_path
        self.lease_store = lease_store
        self.runtime_loader = runtime_loader
        self.rig_host = rig_host
        self.max_ttl_seconds = int(max_ttl_seconds)
        self._guard = threading.RLock()
        self._allocations: dict[str, Allocation] = {}
        self._sessions: dict[str, Session] = {}

    def _catalog(self) -> dict[str, dict[str, Any]]:
        return load_catalog(self.catalog_path, self.access_config_path)

    def list_models(self) -> list[dict[str, Any]]:
        return [
            {
                "id": model_id,
                "placement": row.get("placement"),
                "tier": row.get("tier"),
                "tags": row.get("tags", []),
            }
            for model_id, row in self._catalog().items()
        ]

    def state(self, model_id: str) -> dict[str, Any]:
        if model_id not in self._catalog():
            raise ModelNotFound(f"unknown local model: {model_id}")
        with self._guard:
            self._reap_expired_locked()
            allocation = self._allocations.get(model_id)
            if not allocation:
                return {"model_id": model_id, "status": "available"}
            return {
                "model_id": model_id,
                "status": "ready",
                "endpoint": allocation.endpoint,
                "runtime_model": allocation.runtime_model,
                "placement": allocation.placement,
                "active_sessions": len(allocation.sessions),
            }

    def _normalize_ttl(self, ttl_seconds: Any) -> int:
        try:
            ttl = int(ttl_seconds)
        except (TypeError, ValueError) as exc:
            raise ModelAccessError("ttl_seconds must be an integer") from exc
        if ttl <= 0 or ttl > self.max_ttl_seconds:
            raise ModelAccessError(
                f"ttl_seconds must be between 1 and {self.max_ttl_seconds}"
            )
        return ttl

    def acquire(self, model_id: str, ttl_seconds: Any) -> dict[str, Any]:
        catalog = self._catalog()
        row = catalog.get(model_id)
        if row is None:
            raise ModelNotFound(f"unknown local model: {model_id}")
        ttl = self._normalize_ttl(ttl_seconds)
        expires_at = utc_now() + timedelta(seconds=ttl)

        with self._guard:
            self._reap_expired_locked()
            allocation = self._allocations.get(model_id)
            if allocation and self.lease_store.get(allocation.lease_id):
                self.lease_store.renew(
                    allocation.lease_id,
                    owner=allocation.owner,
                    ttl_seconds=self._lease_ttl(allocation, expires_at),
                )
            else:
                self._allocations.pop(model_id, None)
                owner = f"model-access:{uuid.uuid4().hex}"
                loaded = self.runtime_loader(model_id, owner, ttl)
                lease = loaded["lease"]
                port = int(loaded["port"])
                split_group = loaded.get("split_group")
                gpu_ids = lease.get("gpu_ids", [])
                allocation = Allocation(
                    model_id=model_id,
                    owner=owner,
                    lease_id=str(lease["lease_id"]),
                    endpoint=f"http://{self.rig_host}:{port}/v1",
                    runtime_model=str(loaded.get("runtime_model") or ""),
                    placement={
                        "type": row.get("placement"),
                        "gpus": gpu_ids,
                        "split_group": (
                            split_group.get("id")
                            if isinstance(split_group, dict)
                            else None
                        ),
                    },
                )
                if not allocation.runtime_model:
                    self.lease_store.release(
                        allocation.lease_id,
                        owner=allocation.owner,
                    )
                    raise RuntimeUnavailable("runtime loader omitted runtime_model")
                self._allocations[model_id] = allocation

            session_id = f"interactive-{uuid.uuid4().hex}"
            session = Session(session_id, model_id, expires_at)
            self._sessions[session_id] = session
            allocation.sessions.add(session_id)
            return self._session_payload(session, allocation, reused=len(allocation.sessions) > 1)

    def renew(self, session_id: str, ttl_seconds: Any) -> dict[str, Any]:
        ttl = self._normalize_ttl(ttl_seconds)
        with self._guard:
            self._reap_expired_locked()
            session = self._sessions.get(session_id)
            if not session:
                raise SessionNotFound(f"unknown or expired session: {session_id}")
            allocation = self._allocations.get(session.model_id)
            if not allocation:
                raise SessionNotFound(f"allocation missing for session: {session_id}")
            expires_at = utc_now() + timedelta(seconds=ttl)
            self.lease_store.renew(
                allocation.lease_id,
                owner=allocation.owner,
                ttl_seconds=self._lease_ttl(
                    allocation,
                    expires_at,
                    replacing_session=session_id,
                ),
            )
            session.expires_at = expires_at
            return self._session_payload(session, allocation, reused=True)

    def release(self, session_id: str) -> None:
        with self._guard:
            session = self._sessions.pop(session_id, None)
            if not session:
                raise SessionNotFound(f"unknown or expired session: {session_id}")
            allocation = self._allocations.get(session.model_id)
            if not allocation:
                return
            allocation.sessions.discard(session_id)
            if not allocation.sessions:
                self.lease_store.release(
                    allocation.lease_id,
                    owner=allocation.owner,
                )
                self._allocations.pop(session.model_id, None)

    def reap_expired(self) -> None:
        with self._guard:
            self._reap_expired_locked()

    def _reap_expired_locked(self) -> None:
        now = utc_now()
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if session.expires_at <= now
        ]
        for session_id in expired:
            session = self._sessions.pop(session_id)
            allocation = self._allocations.get(session.model_id)
            if allocation:
                allocation.sessions.discard(session_id)
        for model_id, allocation in list(self._allocations.items()):
            if allocation.sessions:
                continue
            try:
                self.lease_store.release(
                    allocation.lease_id,
                    owner=allocation.owner,
                )
            except Exception:
                continue
            else:
                self._allocations.pop(model_id, None)

    def _lease_ttl(
        self,
        allocation: Allocation,
        proposed_expiry: datetime,
        *,
        replacing_session: str = "",
    ) -> int:
        expiries = [proposed_expiry]
        for session_id in allocation.sessions:
            if session_id == replacing_session:
                continue
            session = self._sessions.get(session_id)
            if session:
                expiries.append(session.expires_at)
        return max(1, math.ceil((max(expiries) - utc_now()).total_seconds()))

    @staticmethod
    def _session_payload(
        session: Session,
        allocation: Allocation,
        *,
        reused: bool,
    ) -> dict[str, Any]:
        return {
            "status": "ready",
            "session_id": session.session_id,
            "model_id": session.model_id,
            "endpoint": allocation.endpoint,
            "runtime_model": allocation.runtime_model,
            "placement": allocation.placement,
            "expires_at": iso(session.expires_at),
            "reused": reused,
        }


def make_handler(service: ModelAccessService, api_key: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ModelAccess/0.1"

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _authorized(self) -> bool:
            expected = f"Bearer {api_key}"
            supplied = self.headers.get("Authorization", "")
            if secrets.compare_digest(supplied, expected):
                return True
            self._json(401, {"error": "invalid model-access credentials"})
            return False

        def _body(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ModelAccessError("invalid Content-Length") from exc
            if length < 0 or length > 65536:
                raise ModelAccessError("request body is too large")
            if not length:
                return {}
            try:
                payload = json.loads(self.rfile.read(length))
            except json.JSONDecodeError as exc:
                raise ModelAccessError("request body must be valid JSON") from exc
            if not isinstance(payload, dict):
                raise ModelAccessError("request body must be a JSON object")
            return payload

        def _path(self) -> str:
            return urlsplit(self.path).path

        def do_GET(self) -> None:
            if self._path() == "/health":
                self._json(200, {"status": "ok"})
                return
            if not self._authorized():
                return
            try:
                if self._path() == "/v1/local/models":
                    self._json(200, {"models": service.list_models()})
                    return
                prefix = "/v1/local/models/"
                if self._path().startswith(prefix):
                    model_id = unquote(self._path()[len(prefix):])
                    self._json(200, service.state(model_id))
                    return
                self._json(404, {"error": "not found"})
            except ModelAccessError as exc:
                self._json(exc.status_code, {"error": str(exc)})
            except Exception as exc:
                self._json(500, {"error": str(exc)})

        def do_POST(self) -> None:
            if not self._authorized():
                return
            try:
                path = self._path()
                acquire_prefix = "/v1/local/models/"
                acquire_suffix = "/acquire"
                renew_prefix = "/v1/local/sessions/"
                renew_suffix = "/renew"
                body = self._body()
                if path.startswith(acquire_prefix) and path.endswith(acquire_suffix):
                    model_id = unquote(path[len(acquire_prefix):-len(acquire_suffix)]).strip("/")
                    if not model_id or "/" in model_id:
                        raise ModelAccessError("invalid model id")
                    payload = service.acquire(model_id, body.get("ttl_seconds", 900))
                    self._json(200, payload)
                    return
                if path.startswith(renew_prefix) and path.endswith(renew_suffix):
                    session_id = unquote(path[len(renew_prefix):-len(renew_suffix)]).strip("/")
                    if not session_id or "/" in session_id:
                        raise ModelAccessError("invalid session id")
                    payload = service.renew(session_id, body.get("ttl_seconds", 900))
                    self._json(200, payload)
                    return
                self._json(404, {"error": "not found"})
            except ModelAccessError as exc:
                self._json(exc.status_code, {"error": str(exc)})
            except Exception as exc:
                self._json(500, {"error": str(exc)})

        def do_DELETE(self) -> None:
            if not self._authorized():
                return
            try:
                prefix = "/v1/local/sessions/"
                if not self._path().startswith(prefix):
                    self._json(404, {"error": "not found"})
                    return
                session_id = unquote(self._path()[len(prefix):]).strip("/")
                if not session_id or "/" in session_id:
                    raise ModelAccessError("invalid session id")
                service.release(session_id)
                self._json(200, {"released": session_id})
            except ModelAccessError as exc:
                self._json(exc.status_code, {"error": str(exc)})
            except Exception as exc:
                self._json(500, {"error": str(exc)})

        def log_message(self, fmt: str, *args: Any) -> None:
            print("[model-access] " + fmt % args)

    return Handler


def start_reaper(service: ModelAccessService, stop: threading.Event) -> threading.Thread:
    def run() -> None:
        while not stop.wait(15):
            try:
                service.reap_expired()
            except Exception as exc:
                print(f"[model-access] session reaper failed: {exc}", file=sys.stderr)

    thread = threading.Thread(target=run, name="model-access-reaper", daemon=True)
    thread.start()
    return thread


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve interactive rig model access")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--shared-root", default="")
    parser.add_argument("--access-config", default="")
    parser.add_argument("--rig-host", default="10.0.0.3")
    parser.add_argument("--api-key-env", default="ORCHESTRATOR_API_KEY")
    parser.add_argument("--load-timeout", type=int, default=420)
    parser.add_argument("--max-session-ttl", type=int, default=3600)
    args = parser.parse_args()

    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        parser.error(f"{args.api_key_env} is required")

    shared_root = find_shared_root(args.shared_root)
    sys.path.insert(0, str(shared_root / "agents"))
    from gpu_leases import GPULeaseStore

    loader = RuntimeLoader(
        Path(__file__).resolve().with_name("load-gpu-runtime"),
        shared_root,
        timeout_seconds=args.load_timeout,
    )
    service = ModelAccessService(
        catalog_path=shared_root / "agents" / "models.catalog.json",
        access_config_path=(
            Path(args.access_config).resolve()
            if args.access_config
            else shared_root / "agents" / "model_access.json"
        ),
        lease_store=GPULeaseStore(shared_root),
        runtime_loader=loader,
        rig_host=args.rig_host,
        max_ttl_seconds=args.max_session_ttl,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service, api_key))
    stop = threading.Event()
    reaper = start_reaper(service, stop)
    print(f"model-access API listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        reaper.join(timeout=2)
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
