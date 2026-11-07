"""Contract checks for the private orchestration API."""

import http.client
import json
import socket
import threading

from orchestration_api import api


class Connection(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__("localhost", timeout=5)
        self.path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX)
        self.sock.connect(self.path)


def test_versioned_status_options_and_action_contract(tmp_path, monkeypatch):
    class FakeControls:
        def __init__(self, shared, config):
            assert shared == tmp_path and config == {"marker": 1}

        def options(self):
            return {"plans": ["p"]}

        def kill_plan(self, batch_id):
            return {"message": f"killed {batch_id}"}

    monkeypatch.setattr(api, "_context", lambda: (tmp_path, {"marker": 1}))
    monkeypatch.setattr(api, "summarize", lambda shared, config: {"counts": {"queue": 2}})
    monkeypatch.setattr(api, "Controls", FakeControls)
    path = tmp_path / "api.sock"
    server = api.Server(str(path), api.Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()

    def request(method, route, body=None):
        connection = Connection(str(path))
        try:
            payload = json.dumps(body).encode() if body is not None else None
            connection.request(method, route, body=payload)
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    try:
        assert request("GET", "/health") == (200, {"ok": True, "api_version": 1})
        assert request("GET", "/v1/orchestration/status") == (200, {"counts": {"queue": 2}})
        assert request("GET", "/v1/orchestration/options") == (200, {"plans": ["p"]})
        assert request("POST", "/v1/orchestration/actions/kill_plan", {"batch_id": "b1"}) == (200, {"message": "killed b1"})
        assert request("POST", "/v1/orchestration/actions/rm_rf", {})[0] == 404
        assert request("POST", "/v1/orchestration/actions/kill_plan", [1])[0] == 400
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
