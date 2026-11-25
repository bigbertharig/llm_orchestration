from __future__ import annotations

import asyncio

import dataclasses

import json

import os

import sys

import textwrap

import time

from pathlib import Path

import httpx

import pytest

from fastapi.testclient import TestClient





from rig_control.manager.routes import _agent_profile_order

from rig_control.rig import Rig, RigUnavailable



def test_agent_type_resolves_disjoint_ready_route_without_loading(env):
    state = Path(os.environ["FAKE_RIG_STATE"])
    state.write_text(json.dumps({"mode": "interactive", "profiles": [
        {"id": "chat-27b", "model": "qwen3.6:27b", "ctx_size": 65536,
         "port": UPSTREAM_PORT, "state": "ready", "gpus": [0]},
        {"id": "coder-7b", "model": "qwen2.5-coder:7b", "ctx_size": 8192,
         "port": UPSTREAM_PORT, "state": "ready", "gpus": [2]},
    ]}))
    settings = dataclasses.replace(env["settings"], manager_enabled=True)
    app = create_app(settings, upstream_transport=httpx.MockTransport(fake_llama), run_reconciler=False)
    async def observed():
        return {"profiles": json.loads(state.read_text())["profiles"]}
    app.state.manager.status = observed
    with TestClient(app) as client:
        response = client.post("/v1/manager/agent-routes/resolve", json={
            "agent_type": "coding", "parent_route": "gpu-rig/chat-27b"})
    assert response.status_code == 200
    assert response.json()["route"] == "gpu-rig/coder-7b"
    assert response.json()["gpu_ids"] == [2]
    assert env["starts"]() == []

def test_cloud_codex_parent_resolves_ready_local_child_without_loading(env):
    state = Path(os.environ["FAKE_RIG_STATE"])
    state.write_text(json.dumps({"mode": "interactive", "profiles": [
        {"id": "coder-7b", "model": "qwen2.5-coder:7b", "ctx_size": 8192,
         "port": UPSTREAM_PORT, "state": "ready", "gpus": [2]},
    ]}))
    settings = dataclasses.replace(env["settings"], manager_enabled=True)
    app = create_app(settings, upstream_transport=httpx.MockTransport(fake_llama), run_reconciler=False)
    async def observed():
        return {"profiles": json.loads(state.read_text())["profiles"]}
    app.state.manager.status = observed
    with TestClient(app) as client:
        response = client.post("/v1/manager/agent-routes/resolve", json={
            "agent_type": "coding", "parent_route": "cloud/codex"})
    assert response.status_code == 200
    assert response.json()["route"] == "gpu-rig/coder-7b"
    assert response.json()["gpu_ids"] == [2]
    assert env["starts"]() == []

def test_cloud_codex_parent_prepares_cold_local_child(env):
    settings = dataclasses.replace(env["settings"], manager_enabled=True)
    app = create_app(settings, upstream_transport=httpx.MockTransport(fake_llama), run_reconciler=False)
    rows = []
    async def observed():
        return {"profiles": rows}
    async def load_profile(pid, spec):
        assert pid == "coder-7b"
        rows.append({"id": pid, "model": spec["model"], "ctx_size": 8192,
                     "port": UPSTREAM_PORT, "state": "ready", "gpus": [2]})
    async def ensure_ready(pid, start):
        await start()
        return UPSTREAM_PORT
    app.state.manager.status = observed
    app.state.manager.load_profile = load_profile
    app.state.manager.rig.ensure_ready = ensure_ready
    with TestClient(app) as client:
        response = client.post("/v1/manager/agent-routes/resolve", json={
            "agent_type": "coding", "parent_route": "cloud/codex"})
    assert response.status_code == 200
    assert response.json()["route"] == "gpu-rig/coder-7b"
    assert response.json()["gpu_ids"] == [2]

def test_agent_type_rejects_missing_separate_ready_route(env):
    state = Path(os.environ["FAKE_RIG_STATE"])
    state.write_text(json.dumps({"mode": "interactive", "profiles": [
        {"id": "chat-27b", "model": "qwen3.6:27b", "ctx_size": 65536,
         "port": UPSTREAM_PORT, "state": "ready", "gpus": [0]},
        {"id": "coder-7b", "model": "qwen2.5-coder:7b", "ctx_size": 8192,
         "port": UPSTREAM_PORT, "state": "ready", "gpus": [0]},
    ]}))
    settings = dataclasses.replace(env["settings"], manager_enabled=True)
    app = create_app(settings, upstream_transport=httpx.MockTransport(fake_llama), run_reconciler=False)
    async def observed():
        return {"profiles": json.loads(state.read_text())["profiles"]}
    app.state.manager.status = observed
    with TestClient(app) as client:
        response = client.post("/v1/manager/agent-routes/resolve", json={
            "agent_type": "coding", "parent_route": "gpu-rig/chat-27b"})
    assert response.status_code == 503
    assert env["starts"]() == []

def test_agent_type_prepares_cold_route_through_manager(env):
    settings = dataclasses.replace(env["settings"], manager_enabled=True)
    app = create_app(settings, upstream_transport=httpx.MockTransport(fake_llama), run_reconciler=False)
    rows = [{"id": "chat-27b", "model": "qwen3.6:27b", "ctx_size": 65536,
             "port": UPSTREAM_PORT, "state": "ready", "gpus": [0]}]
    async def observed():
        return {"profiles": rows}
    async def load_profile(pid, spec):
        assert pid == "coder-7b" and spec["model"] == "qwen2.5-coder:7b"
        rows.append({"id": pid, "model": spec["model"], "ctx_size": 8192,
                     "port": UPSTREAM_PORT, "state": "ready", "gpus": [2]})
    async def ensure_ready(pid, start):
        await start()
        return UPSTREAM_PORT
    app.state.manager.status = observed
    app.state.manager.load_profile = load_profile
    app.state.manager.rig.ensure_ready = ensure_ready
    with TestClient(app) as client:
        response = client.post("/v1/manager/agent-routes/resolve", json={
            "agent_type": "coding", "parent_route": "gpu-rig/chat-27b"})
    assert response.status_code == 200
    assert response.json()["route"] == "gpu-rig/coder-7b"
    assert response.json()["gpu_ids"] == [2]

def test_agent_type_rejects_unregistered_general_label(env):
    settings = dataclasses.replace(env["settings"], manager_enabled=True)
    with TestClient(create_app(settings, upstream_transport=httpx.MockTransport(fake_llama),
                               run_reconciler=False)) as client:
        response = client.post("/v1/manager/agent-routes/resolve", json={
            "agent_type": "general", "parent_route": "gpu-rig/chat-27b"})
    assert response.status_code == 400

def test_model_offers_exclude_in_testing_and_unapproved(monkeypatch, env):
    monkeypatch.setattr("rig_control.manager.models.catalog", lambda: {
        "models": {
            "approved": {"status": "active", "in_testing": False},
            "trial": {"status": "candidate", "in_testing": True},
            "paused": {"status": "active", "in_testing": True},
        },
        "roles": {
            "coding": {"model": "approved", "profile": "default",
                       "good_for": ["coding"], "why": "Reviewed code model"},
            "trial": {"model": "trial", "good_for": ["research"]},
            "paused": {"model": "paused", "good_for": ["planning"]},
        },
    })
    settings = dataclasses.replace(env["settings"], manager_enabled=True)
    with TestClient(create_app(settings, upstream_transport=httpx.MockTransport(fake_llama),
                               run_reconciler=False)) as client:
        response = client.get("/v1/manager/model-offers")
    assert response.status_code == 200
    assert response.json() == {"object": "model_offers", "version": 1, "data": [{
        "capability": "coding", "model": "approved", "profile": "default",
        "good_for": ["coding"], "description": "Reviewed code model"}]}

def test_child_route_order_uses_reviewed_roles(monkeypatch):
    monkeypatch.setattr("rig_control.manager.models.catalog", lambda: {
        "models": {"reviewed": {"status": "active", "in_testing": False},
                   "trial": {"status": "candidate", "in_testing": True}},
        "roles": {"brain_code": {"model": "trial"},
                  "single_1060_code": {"model": "reviewed"}},
    })
    published = {"trial-route": {"model": "trial"},
                 "reviewed-route": {"model": "reviewed"}}
    assert _agent_profile_order("coding", published) == ["reviewed-route"]

def test_research_type_selects_separate_ready_route(env):
    settings = dataclasses.replace(env["settings"], manager_enabled=True)
    app = create_app(settings, upstream_transport=httpx.MockTransport(fake_llama), run_reconciler=False)
    async def observed():
        return {"profiles": [
            {"id": "coder-7b", "model": "qwen2.5-coder:7b", "ctx_size": 8192,
             "port": UPSTREAM_PORT, "state": "ready", "gpus": [1]},
            {"id": "chat-27b", "model": "qwen3.6:27b", "ctx_size": 65536,
             "port": UPSTREAM_PORT, "state": "ready", "gpus": [0]},
        ]}
    app.state.manager.status = observed
    with TestClient(app) as client:
        response = client.post("/v1/manager/agent-routes/resolve", json={
            "agent_type": "research", "parent_route": "gpu-rig/coder-7b"})
    assert response.status_code == 200
    assert response.json()["route"] == "gpu-rig/chat-27b"
    assert response.json()["gpu_ids"] == [0]

def test_rig_arguments_survive_ssh():
    """ssh hands the remote shell one string; spaces and JSON must arrive as single arguments."""
    import shlex
    spec = '{"controller": "interactive", "gpu_ids": [0]}'
    over_ssh = Rig(("/usr/bin/ssh", "-o", "BatchMode=yes", "10.0.0.3", "/usr/local/bin/rig"), 30)
    remote_line = " ".join(over_ssh.argv("lease", "acquire", "--spec", spec)[5:])
    assert shlex.split(remote_line) == ["lease", "acquire", "--spec", spec]
    local = Rig(("rig",), 30)
    assert local.argv("claim", "add", "x", "--note", "two words") == ["rig", "claim", "add", "x", "--note", "two words"]
    windows = Rig((r"C:\WINDOWS\System32\OpenSSH\ssh.exe", "rig", "/usr/local/bin/rig"), 30)
    assert windows.argv("claim", "add", "x", "--note", "two words")[-1] == "'two words'"

from control_fixture import create_app, Settings
UPSTREAM_PORT = 11499

FAKE_RIG = textwrap.dedent("""\
    import json, os, sys, time
    if os.environ.get("FAKE_RIG_SSH") == "down":
        print("ssh: connect to host 10.0.0.3 port 22: No route to host"); sys.exit(255)
    state_path = os.environ["FAKE_RIG_STATE"]
    st = json.load(open(state_path))
    args = sys.argv[1:]
    if args[:2] == ["status", "--json"]:
        print(json.dumps(st)); sys.exit(0)
    if args[:2] == ["start", "remote"]:
        with open(state_path + ".starts", "a") as f: f.write(args[2] + "\\n")
        mode = os.environ.get("FAKE_RIG_START", "ok")
        if mode == "busy": print("busy: bench (ready, since t, from 10.0.0.2)"); sys.exit(2)
        if mode == "nocap": print("error: no capacity for x"); sys.exit(4)
        if mode == "fail": print("FAILED: container exited"); sys.exit(1)
        if mode == "locked-once" and not os.path.exists(state_path + ".locked"):
            open(state_path + ".locked", "w").close(); print("locked"); sys.exit(3)
        time.sleep(2.5 if mode == "slow" else 0.5)
        st["profiles"].append({"id": args[2], "port": %d, "state": "ready"})
        json.dump(st, open(state_path, "w")); sys.exit(0)
    sys.exit(64)
""" % UPSTREAM_PORT)

@pytest.fixture
def env(tmp_path, monkeypatch):
    (tmp_path / "rig.py").write_text(FAKE_RIG)
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"mode": "idle", "status": "idle", "profiles": []}))
    lib = tmp_path / "lib.json"
    lib.write_text(json.dumps({"remote_profiles": [
        {"id": "coder-7b", "model": "qwen2.5-coder:7b", "ctx_size": 16384, "description": "small"},
        {"id": "chat-27b", "model": "qwen3.6:27b", "ctx_size": 65536, "description": "big"},
    ]}))
    monkeypatch.setenv("FAKE_RIG_STATE", str(state))
    monkeypatch.setenv("FAKE_RIG_START", "ok")
    monkeypatch.setattr("rig_control.rig.LOCKED_RETRY_SECONDS", 0.1)
    settings = Settings(
        rig_command=(sys.executable, str(tmp_path / "rig.py")), profiles_path=lib,
        upstream_host="127.0.0.1", model_prefix="gpu-rig", bind="127.0.0.1", port=0,
        start_timeout_seconds=30, request_timeout_seconds=30,
    )

    def starts():
        p = Path(str(state) + ".starts")
        return p.read_text().split() if p.exists() else []

    return {"settings": settings, "lib": lib, "starts": starts}

def fake_llama(request: httpx.Request) -> httpx.Response:
    assert request.url.port == UPSTREAM_PORT
    body = json.loads(request.content)
    if request.url.path == "/tokenize":
        return httpx.Response(200, json={"tokens": list(range(len(body["content"].split())))})
    if "OVERFLOW" in json.dumps(body):
        return httpx.Response(400, json={"error": {
            "code": 400, "message": "request (20030 tokens) exceeds the available context size (16384 tokens)",
            "type": "exceed_context_size_error", "n_prompt_tokens": 20030, "n_ctx": 16384}})
    if body.get("stream"):
        chunks = [b'data: {"choices":[{"delta":{"content":"o"}}]}\n\n',
                  b'data: {"choices":[{"delta":{"content":"k"}}]}\n\n', b"data: [DONE]\n\n"]
        async def gen():
            for c in chunks:
                yield c
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=gen())
    return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "echo_model": body.get("model")})
