#!/usr/bin/env python3
"""Control the experimental shared-weight, per-GPU KV chat service on the rig."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "brain/kv_chat_mode.json"
LOCK = ROOT / "brain/kv_chat_mode.lock"
MODEL = Path("/data/model-hotset/qwen3.6-35b-a3b-strata-xl/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")
BUILD = Path("/data/bench-staging/kvchat-b11669-build")
SERVER = "llama-kv-chat-server-v2"
IMAGE = "4bfa1cb0a955"
CONTAINER = "qwen35-kv-chat-service"
PARENT_CLAIM = "qwen35-kv-main"
MODEL_ID = "qwen3.6:35b-a3b-xl"
NEMO_ID = "nemotron-3.5-bolt:74b-a9b"
NEMO_MODEL = Path("/data/model-hotset/nemotron-3.5-bolt-74b-a9b/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-UD-Q3_K_XL.gguf")
NEMO_OVERRIDES = Path("/mnt/shared/plans/shoulders/benchmarking/results/runtime_experiments/kv_expert_chat_20261107/nemo_experts_1234.tsv")
MODELS = {
    MODEL_ID: {"path": MODEL, "mode": "gpu0-main-q4", "gpus": (0,), "kv": "q4_0", "port": 11524},
    NEMO_ID: {"path": NEMO_MODEL, "mode": "gpu0-main-f16-experts1234", "gpus": (0, 1, 2, 3, 4), "kv": "f16", "port": 11525},
}
CLAIM_TTL_S = 300
CLAIM_RENEW_S = 60


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


def claim(name, gpu, note, model_id=MODEL_ID):
    ctl = controller()
    gpu_spec = ",".join(map(str, gpu)) if isinstance(gpu, (tuple, list)) else str(gpu)
    run(["rig", "claim", "add", name, "--gpus", gpu_spec, "--by", "orchestration",
         "--owner", ctl, "--controller", ctl, "--purpose", "chat",
         "--model", model_id, "--ttl", str(CLAIM_TTL_S), "--note", note])


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


def clear_orphan_claims():
    """Clear this service's claims only when its container is not running."""
    if container_running():
        return
    claims = json.loads(run(["rig", "claim", "list"]).stdout)["claims"]
    for item in claims:
        if (item.get("model") in MODELS and item.get("purpose") == "chat"
                and (item["name"] in (PARENT_CLAIM, "nemo74-kv-main") or
                     item["name"].startswith(("qwen35-kv-chat-", "nemo74-kv-chat-")))):
            release(item["name"])


def renew_claims_once(instance_id: str) -> bool:
    """Renew only this live service instance; release its claims if the container died."""
    with locked():
        st = state()
        if not st or st.get("instance_id") != instance_id:
            return False
        if not container_running():
            for name in st["claims"].values():
                release(name)
            STATE.unlink(missing_ok=True)
            return False
        ctl = controller()
        for name in st["claims"].values():
            run(["rig", "claim", "renew", name, "--ttl", str(CLAIM_TTL_S), "--controller", ctl])
        return True


def stop_unrenewable_service(instance_id: str):
    """Do not leave a running container behind after its reservations lapse."""
    with locked():
        st = state()
        if not st or st.get("instance_id") != instance_id:
            return
        run(["docker", "stop", CONTAINER], check=False)
        for name in st["claims"].values():
            release(name)
        STATE.unlink(missing_ok=True)


def launch_renewer(instance_id: str):
    with (STATE.parent / "kv_chat_claim_renewer.log").open("a", encoding="utf-8") as log:
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "renew-loop",
                          "--instance-id", instance_id], stdin=subprocess.DEVNULL,
                         stdout=log, stderr=log, start_new_session=True)


def worker_cards(model_id=MODEL_ID) -> list[dict]:
    """Discover eligible chat cards; the Nemo expert placement leaves GPU5 only."""
    output = run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used",
                  "--format=csv,noheader,nounits"]).stdout
    cards = []
    for line in output.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 4:
            raise RuntimeError(f"invalid GPU inventory row: {line}")
        gpu, name, total, used = fields
        if not all(value.isdigit() for value in (gpu, total, used)):
            raise RuntimeError(f"invalid GPU inventory row: {line}")
        if ((int(gpu) == 5 and int(total) >= 6000) if model_id == NEMO_ID
                else (int(gpu) != 0 and int(total) >= 6000)):
            cards.append({"gpu": int(gpu), "name": name, "total_mib": int(total), "used_mib": int(used)})
    return sorted(cards, key=lambda card: (card["total_mib"], card["gpu"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("start", "attach", "detach", "turn", "status", "stop", "renew-loop"))
    p.add_argument("--port", type=int, default=11524)
    p.add_argument("--model", choices=tuple(MODELS))
    p.add_argument("--chat-id")
    p.add_argument("--gpu", type=int)
    p.add_argument("--message")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--instance-id")
    args = p.parse_args()
    selected_model = args.model or (state() or {}).get("model") or MODEL_ID
    spec = MODELS[selected_model]
    if args.action == "renew-loop":
        if not args.instance_id: p.error("renew-loop needs --instance-id")
        last_success = time.monotonic()
        while True:
            time.sleep(CLAIM_RENEW_S)
            try:
                if not renew_claims_once(args.instance_id): return
                last_success = time.monotonic()
            except (OSError, RuntimeError, ValueError) as exc:
                print(f"claim renewal failed: {exc}", file=sys.stderr, flush=True)
                if time.monotonic() - last_success >= CLAIM_TTL_S - CLAIM_RENEW_S:
                    stop_unrenewable_service(args.instance_id)
                    return
        return
    if args.action == "turn":
        st = state()
        if not st or st["model"] != selected_model or not healthy(st["port"]): p.error("shared-weight service is not ready for requested model")
        if not args.chat_id or args.message is None: p.error("turn needs --chat-id and --message")
        code, value = request(st["port"], f"/chats/{args.chat_id}/turns", "POST",
                              {"message": args.message, "max_tokens": args.max_tokens}, timeout=600)
        print(json.dumps(value))
        raise SystemExit(0 if code == 200 else 1)
    with locked():
        st = state()
        if args.action == "status":
            if not st: print(json.dumps({"status": "absent"})); return
            if args.model and st["model"] != args.model: p.error(f"active shared service uses {st['model']}")
            ready = container_running() and healthy(st["port"])
            chats = request(st["port"], "/chats")[1].get("chats", []) if ready else []
            print(json.dumps({"status": "ready" if ready else "down", "port": st["port"],
                              "model": st["model"], "mode": st["mode"], "instance_id": st.get("instance_id"),
                              "chats": chats, "claims": st["claims"]}, indent=2)); return
        if args.action == "start":
            if st and st["model"] != selected_model: p.error(f"shared service already uses {st['model']}; stop it explicitly first")
            if st and container_running() and healthy(st["port"]):
                if not st.get("instance_id"):
                    st["instance_id"] = uuid.uuid4().hex
                    save(st)
                    launch_renewer(st["instance_id"])
                print(json.dumps({"status": "ready", "port": st["port"], "already_running": True})); return
            if st and not container_running():
                for name in st["claims"].values(): release(name)
                STATE.unlink(missing_ok=True)
                st = None
            if st or container_running(): p.error("stale service state or container; inspect and stop it first")
            clear_orphan_claims()
            port = spec["port"] if args.port == 11524 and selected_model == NEMO_ID else args.port
            if not (1024 <= port <= 65535): p.error("port must be 1024..65535")
            if not spec["path"].is_file() or not (BUILD / "bin" / SERVER).is_file():
                p.error("model hotset or KV server binary missing; build the runtime first")
            if selected_model == NEMO_ID and not NEMO_OVERRIDES.is_file(): p.error("Nemo expert override file missing")
            rig = json.loads(run(["rig", "status", "--json"]).stdout)
            if rig["mode"] != "idle" or rig.get("profiles"):
                p.error("rig has an active load; stop it intentionally before starting shared-weight mode")
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", port)) == 0: p.error("port already in use")
            existing = json.loads(run(["rig", "claim", "list"]).stdout)["claims"]
            parent_gpus = spec["gpus"]
            if any(set(parent_gpus) & set(item["gpu_ids"]) for item in existing): p.error("required model GPU already claimed")
            inventory = run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"]).stdout
            usage = {int(a.strip()): int(b.strip()) for a, b in (line.split(",") for line in inventory.splitlines())}
            if any(usage.get(gpu, 100000) >= 1000 for gpu in parent_gpus): p.error("required model GPU is not free")
            parent_claim = "nemo74-kv-main" if selected_model == NEMO_ID else PARENT_CLAIM
            claim(parent_claim, parent_gpus, f"{selected_model} shared weights and main {spec['kv']} KV", selected_model)
            try:
                env = (["-e", "KV_TENSOR_OVERRIDES_FILE=/nemo_experts.tsv"]
                       if selected_model == NEMO_ID else [])
                mounts = (["-v", f"{NEMO_OVERRIDES}:/nemo_experts.tsv:ro"] if selected_model == NEMO_ID else [])
                run(["docker", "run", "-d", "--rm", "--gpus", "all", "--network", "host",
                     "--name", CONTAINER, "--label", "org.llm-orchestration.kv-chat=true",
                     *env, *mounts, "-v", f"{BUILD}:/build:ro", "-v", f"{spec['path']}:/model.gguf:ro", IMAGE,
                     f"/build/bin/{SERVER}", "/model.gguf", str(port), "262144", spec["kv"]])
                startup_limit = 600 if selected_model == NEMO_ID else 120
                for _ in range(startup_limit):
                    if healthy(port): break
                    if not container_running(): raise RuntimeError("KV server exited during load")
                    time.sleep(1)
                else: raise RuntimeError(f"KV server did not become ready within {startup_limit} seconds")
                st = {"model": selected_model, "mode": spec["mode"], "port": port,
                      "instance_id": uuid.uuid4().hex,
                      "container": CONTAINER, "claims": {"main": parent_claim}}
                save(st)
                launch_renewer(st["instance_id"])
            except Exception:
                run(["docker", "stop", CONTAINER], check=False)
                release(parent_claim)
                STATE.unlink(missing_ok=True)
                raise
            print(json.dumps({"status": "ready", "model": selected_model, "mode": spec["mode"], "port": port,
                              "instance_id": st["instance_id"], "main_chat": "main"})); return
        if args.action == "stop":
            if not st: p.error("no managed shared-weight service")
            if args.model and st["model"] != args.model: p.error(f"active shared service uses {st['model']}")
            run(["docker", "stop", CONTAINER], check=False)
            for name in st["claims"].values(): release(name)
            STATE.unlink(missing_ok=True)
            print(json.dumps({"status": "stopped"})); return
        if not st or not container_running() or not healthy(st["port"]):
            p.error("shared-weight parent is not ready")
        if args.model and st["model"] != args.model: p.error(f"active shared service uses {st['model']}")
        if args.action == "attach":
            if not args.chat_id or not args.chat_id.replace("-", "").replace("_", "").isalnum():
                p.error("attach needs an alphanumeric --chat-id (hyphen and underscore allowed)")
            if args.chat_id == "main" or args.chat_id in st["claims"]: p.error("chat ID already in use")
            code, value = request(st["port"], "/chats")
            if code != 200: raise RuntimeError(value)
            used = {chat["gpu"] for chat in value["chats"]}
            inventory = worker_cards(st["model"])
            eligible = {card["gpu"] for card in inventory}
            if args.gpu is not None and args.gpu not in eligible:
                p.error(f"GPU {args.gpu} is not an eligible worker card; detected {sorted(eligible)}")
            candidates = ([card for card in inventory if card["gpu"] == args.gpu] if args.gpu is not None
                          else inventory)
            claims = json.loads(run(["rig", "claim", "list"]).stdout)["claims"]
            selected = next((card["gpu"] for card in candidates
                             if card["gpu"] not in used and card["used_mib"] < 1000
                             and not any(card["gpu"] in item["gpu_ids"] for item in claims)), None)
            if selected is None: p.error("no eligible free GPU for an attached chat")
            name = f"{'nemo74' if st['model'] == NEMO_ID else 'qwen35'}-kv-chat-{args.chat_id}"
            claim(name, selected, f"{st['model']} {spec['kv']} KV chat {args.chat_id}; parent {st['claims']['main']}", st["model"])
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
