"""Private, versioned HTTP API for orchestration status and controls."""

from __future__ import annotations

import argparse
import json
import os
import socket
import socketserver
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from . import Controls
from .status import activity, diagnostics, snapshot

DEFAULT_SOCKET = Path("/run/llm-orchestration/api.sock")
ACTIONS = frozenset({"kill_plan", "kill_all_active",
                     "batch_outputs", "cleanup_stale", "requeue_worker"})
MAX_BODY_BYTES = 1_048_576
_action_lock = threading.Lock()


def _context():
    shared = Path(os.getenv("SHARED_ROOT", "/media/bryan/shared"))
    path = Path(os.getenv("ORCHESTRATION_CONFIG", str(shared / "agents" / "config.json")))
    config = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return shared, config


def _run_action(action: str, body: dict):
    shared, config = _context()
    controls = Controls(shared, config)
    if action in ("kill_plan", "batch_outputs"):
        return getattr(controls, action)(str(body.get("batch_id") or ""))
    if action == "requeue_worker":
        return controls.requeue_worker(str(body.get("worker") or ""))
    return getattr(controls, action)()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, _format, *_args):
        pass

    def _reply(self, code: int, payload: dict):
        data = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlsplit(self.path)
        path = parsed.path
        try:
            if path == "/health":
                result = {"ok": True, "api_version": 1}
            elif path == "/v1/orchestration/status":
                result = snapshot(*_context())
            elif path == "/v1/orchestration/diagnostics":
                result = diagnostics(*_context())
            elif path == "/v1/orchestration/activity":
                query = parse_qs(parsed.query)
                batch_id = query.get("batch_id", [""])[0]
                if not batch_id or len(batch_id) > 160:
                    return self._reply(400, {"error": "batch_id required"})
                try:
                    after = int(query.get("after", ["0"])[0])
                    limit = int(query.get("limit", ["100"])[0])
                    result = activity(_context()[0], batch_id, after, limit)
                except ValueError as exc:
                    return self._reply(400, {"error": str(exc)})
            else:
                return self._reply(404, {"error": "unknown endpoint"})
            self._reply(200, result)
        except Exception as exc:
            self._reply(503, {"error": str(exc)})

    def do_POST(self):
        path = urlsplit(self.path).path
        prefix = "/v1/orchestration/actions/"
        action = path[len(prefix):] if path.startswith(prefix) else ""
        if action not in ACTIONS:
            return self._reply(404, {"error": "unknown action"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > MAX_BODY_BYTES:
                return self._reply(413, {"error": "request too large"})
            body = json.loads(self.rfile.read(length)) if length else {}
            if not isinstance(body, dict):
                return self._reply(400, {"error": "JSON object required"})
        except (ValueError, json.JSONDecodeError):
            return self._reply(400, {"error": "invalid JSON"})
        try:
            with _action_lock:
                result = _run_action(action, body)
            self._reply(200, result)
        except Exception as exc:
            self._reply(503, {"error": str(exc)})


class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def serve(path: Path = DEFAULT_SOCKET):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    if path.exists():
        probe = socket.socket(socket.AF_UNIX)
        try:
            probe.connect(str(path))
        except ConnectionRefusedError:
            path.unlink()
        else:
            raise RuntimeError(f"orchestration API already running at {path}")
        finally:
            probe.close()
    with Server(str(path), Handler) as server:
        os.chmod(path, 0o600)
        try:
            server.serve_forever()
        finally:
            path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Private orchestration API")
    parser.add_argument("--socket", type=Path, default=DEFAULT_SOCKET)
    serve(parser.parse_args().socket)
