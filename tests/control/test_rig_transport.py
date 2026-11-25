import asyncio, json, os, sys, textwrap, time
from pathlib import Path
import pytest
import httpx
from rig_control.rig import Rig, RigUnavailable
from fixture_settings import Settings
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

def test_concurrent_cold_requests_share_one_start(env):
    rig = Rig(env["settings"].rig_command, 30)

    async def go():
        return await asyncio.gather(*(rig.ensure_ready("coder-7b") for _ in range(4)))

    assert asyncio.run(go()) == [UPSTREAM_PORT] * 4
    assert env["starts"]() == ["coder-7b"]

def test_start_failure_reaches_every_waiter(env, monkeypatch):
    monkeypatch.setenv("FAKE_RIG_START", "nocap")
    rig = Rig(env["settings"].rig_command, 30)

    async def go():
        return await asyncio.gather(*(rig.ensure_ready("coder-7b") for _ in range(3)), return_exceptions=True)

    results = asyncio.run(go())
    assert all(isinstance(r, RigUnavailable) and r.status == 503 for r in results)
    assert env["starts"]() == ["coder-7b"]

def test_status_shows_loads_in_progress(env, monkeypatch):
    monkeypatch.setenv("FAKE_RIG_START", "slow")
    rig = Rig(env["settings"].rig_command, 30)

    async def go():
        load = asyncio.create_task(rig.ensure_ready("coder-7b"))
        await asyncio.sleep(1.0)
        during = rig.loading()
        await load
        return during, rig.loading()

    during, after = asyncio.run(go())
    assert [x["id"] for x in during] == ["coder-7b"] and during[0]["elapsed_s"] >= 0.5
    assert after == []

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