#!/usr/bin/env python3
"""Control the experimental shared-weight, per-GPU KV chat service on the rig."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "brain/kv_chat_mode.json"
LOCK = ROOT / "brain/kv_chat_mode.lock"
MODEL = Path("/data/model-hotset/qwen3.6-35b-a3b-strata-xl/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")
BUILD = Path("/data/bench-staging/kvchat-b11669-build")
IMAGE = "4bfa1cb0a955"
CONTAINER = "qwen35-kv-chat-service"
PARENT_CLAIM = "qwen35-kv-main"
MODEL_ID = "qwen3.6:35b-a3b-xl"


def run(argv, *, check=True):
    p = subprocess.run(argv, text=True, capture_output=True)
    if check and p.returncode:
        raise RuntimeError(f"{' '.join(argv)} failed ({p.returncode}): {(p.stderr or p.stdout).strip()}")
    return p


@contextlib.contextmanager
def locked():
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def state():
    return json.loads(STATE.read_text()) if STATE.exists() else None


def save(value):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(STATE)


def controller():
    return json.loads(run(["rig", "controller", "show"]).stdout)["controller"]["id"]


def claim(name, gpu, note):
    ctl = controller()
    run(["rig", "claim", "add", name, "--gpus", str(gpu), "--by", "orchestration",
         "--owner", ctl, "--controller", ctl, "--purpose", "chat",
         "--model", MODEL_ID, "--note", note])


def release(name):
    ctl = controller()
    run(["rig", "claim", "remove", name, "--controller", ctl], check=False)


def request(port, path, method="GET", body=None, timeout=30):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 headers={"Content-Type": "application/json"}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        try: detail = json.load(exc)
        except ValueError: detail = {"error": exc.reason}
        return exc.code, detail


def healthy(port):
    try:
        code, value = request(port, "/health", timeout=3)
        return code == 200 and value.get("model_loaded") is True
    except (OSError, ValueError):
        return False


def container_running():
    return run(["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER], check=False).stdout.strip() == "true"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("start", "attach", "detach", "turn", "status", "stop"))
    p.add_argument("--port", type=int, default=11524)
    p.add_argument("--chat-id")
    p.add_argument("--gpu", type=int)
    p.add_argument("--message")
    p.add_argument("--max-tokens", type=int, default=128)
    args = p.parse_args()
    if args.action == "turn":
        st = state()
        if not st or not healthy(st["port"]): p.error("35B shared-weight service is not ready")
        if not args.chat_id or args.message is None: p.error("turn needs --chat-id and --message")
        code, value = request(st["port"], f"/chats/{args.chat_id}/turns", "POST",
                              {"message": args.message, "max_tokens": args.max_tokens}, timeout=600)
        print(json.dumps(value))
        raise SystemExit(0 if code == 200 else 1)
    with locked():
        st = state()
        if args.action == "status":
            if not st: print(json.dumps({"status": "absent"})); return
            ready = container_running() and healthy(st["port"])
            chats = request(st["port"], "/chats")[1].get("chats", []) if ready else []
            print(json.dumps({"status": "ready" if ready else "down", "port": st["port"],
                              "chats": chats, "claims": st["claims"]}, indent=2)); return
        if args.action == "start":
            if st and container_running() and healthy(st["port"]):
                print(json.dumps({"status": "ready", "port": st["port"], "already_running": True})); return
            if st or container_running(): p.error("stale service state or container; inspect and stop it first")
            if not (1024 <= args.port <= 65535): p.error("port must be 1024..65535")
            if not MODEL.is_file() or not (BUILD / "bin/llama-kv-chat-server").is_file():
                p.error("model hotset or KV server binary missing; build the runtime first")
            rig = json.loads(run(["rig", "status", "--json"]).stdout)
            if rig["mode"] != "idle" or rig.get("profiles"):
                p.error("rig has an active load; stop it intentionally before starting shared-weight mode")
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", args.port)) == 0: p.error("port already in use")
            existing = json.loads(run(["rig", "claim", "list"]).stdout)["claims"]
            if any(0 in item["gpu_ids"] for item in existing): p.error("GPU0 already claimed")
            memory = run(["nvidia-smi", "--id", "0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).stdout.strip()
            if not memory.isdigit() or int(memory) >= 1000: p.error("GPU0 is not free")
            claim(PARENT_CLAIM, 0, "35B shared weights and main Q4 KV")
            try:
                run(["docker", "run", "-d", "--rm", "--gpus", "all", "--network", "host",
                     "--name", CONTAINER, "--label", "org.llm-orchestration.kv-chat=true",
                     "-v", f"{BUILD}:/build:ro", "-v", f"{MODEL}:/model.gguf:ro", IMAGE,
                     "/build/bin/llama-kv-chat-server", "/model.gguf", str(args.port), "262144"])
                for _ in range(120):
                    if healthy(args.port): break
                    if not container_running(): raise RuntimeError("KV server exited during load")
                    time.sleep(1)
                else: raise RuntimeError("KV server did not become ready within 120 seconds")
                st = {"model": MODEL_ID, "mode": "gpu0-main-q4", "port": args.port,
                      "container": CONTAINER, "claims": {"main": PARENT_CLAIM}}
                save(st)
            except Exception:
                run(["docker", "stop", CONTAINER], check=False)
                release(PARENT_CLAIM)
                raise
            print(json.dumps({"status": "ready", "port": args.port, "main_chat": "main"})); return
        if args.action == "stop":
            if not st: p.error("no managed 35B shared-weight service")
            run(["docker", "stop", CONTAINER], check=False)
            for name in st["claims"].values(): release(name)
            STATE.unlink(missing_ok=True)
            print(json.dumps({"status": "stopped"})); return
        if not st or not container_running() or not healthy(st["port"]):
            p.error("35B shared-weight parent is not ready")
        if args.action == "attach":
            if not args.chat_id or not args.chat_id.replace("-", "").replace("_", "").isalnum():
                p.error("attach needs an alphanumeric --chat-id (hyphen and underscore allowed)")
            if args.chat_id == "main" or args.chat_id in st["claims"]: p.error("chat ID already in use")
            code, value = request(st["port"], "/chats")
            if code != 200: raise RuntimeError(value)
            used = {chat["gpu"] for chat in value["chats"]}
            candidates = [args.gpu] if args.gpu is not None else list(range(1, 6))
            selected = None
            for gpu in candidates:
                if gpu not in range(1, 6) or gpu in used: continue
                claims = json.loads(run(["rig", "claim", "list"]).stdout)["claims"]
                if any(gpu in item["gpu_ids"] for item in claims): continue
                memory = run(["nvidia-smi", "--id", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).stdout.strip()
                if memory.isdigit() and int(memory) < 1000: selected = gpu; break
            if selected is None: p.error("no free 1060 card for an attached chat")
            name = f"qwen35-kv-chat-{args.chat_id}"
            claim(name, selected, f"35B Q4 KV chat {args.chat_id}; parent {PARENT_CLAIM}")
            try:
                code, value = request(st["port"], "/chats", "POST", {"chat_id": args.chat_id, "gpu": selected}, timeout=120)
                if code != 201: raise RuntimeError(f"attach failed ({code}): {value}")
            except Exception:
                release(name)
                raise
            st["claims"][args.chat_id] = name; save(st)
            print(json.dumps({**value, "route": f"http://127.0.0.1:{st['port']}/chats/{args.chat_id}/turns"})); return
        if args.action == "detach":
            if not args.chat_id or args.chat_id == "main" or args.chat_id not in st["claims"]:
                p.error("detach needs an attached worker --chat-id")
            code, value = request(st["port"], f"/chats/{args.chat_id}", "DELETE")
            if code != 200: raise RuntimeError(f"detach failed ({code}): {value}")
            release(st["claims"].pop(args.chat_id)); save(st)
            print(json.dumps(value)); return


if __name__ == "__main__":
    main()
