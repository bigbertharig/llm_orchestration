#!/usr/bin/env python3
"""Control the experimental shared-weight, per-GPU KV chat service on the rig."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
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
MODEL_LIBRARY = ROOT / "agents/model_library.json"
MODELS = {
    MODEL_ID: {"path": MODEL, "mode": "gpu0-main-q4", "gpus": (0,), "kv": "q4_0",
               "port": 11524, "context_allocation": "dynamic_at_attach"},
    NEMO_ID: {"path": NEMO_MODEL, "mode": "gpu0-main-f16-experts1234", "gpus": (0, 1, 2, 3, 4),
              "kv": "f16", "port": 11525, "context_allocation": "dynamic_at_attach"},
}
CLAIM_TTL_S = 300
CLAIM_RENEW_S = 60
MIN_CONTEXT_CAP = 16384
MAX_CONTEXT_CAP = 262144
WORKER_RESERVE_MIB = 256
DEFAULT_QWEN_CHILD_CONTEXT = 65536


def imported_candidate(candidate_id: str, fallback: dict) -> tuple[dict, str]:
    """Read measured admission data from Benchmarking's imported artifact."""
    try:
        library = json.loads(MODEL_LIBRARY.read_text(encoding="utf-8"))
        candidate = library["subagent_catalog"]["candidates"][candidate_id]
        return candidate, library["source_sha256"]
    except (OSError, ValueError, TypeError, KeyError):
        return fallback, "built-in-safety-fallback"


QWEN_CANDIDATE, MODEL_LIBRARY_SHA256 = imported_candidate(
    "qwen35_shared_1060", {
        "memory_estimate": {"fixed_compute_mib": 0, "kv_mib_at_262144": 1958,
                            "admission_reserve_mib": 272,
                            "shared_parent_mib_per_chat": 74,
                            "shared_parent_reserve_mib": 416},
        "max_attached_chats": 6,
    })
QWEN_HEADLESS_CANDIDATE, _ = imported_candidate(
    "qwen35_headless_worker_pool", {"max_attached_chats": 0})
NEMO_CANDIDATE, _ = imported_candidate(
    "nemotron74_shared_1060", {
        "memory_estimate": {"fixed_compute_mib": 6, "kv_mib_at_262144": 2048,
                            "admission_reserve_mib": 256,
                            "shared_parent_mib_per_chat": 182},
        "max_attached_chats": 4,
    })
QWEN_MEMORY = QWEN_CANDIDATE["memory_estimate"]
NEMO_MEMORY = NEMO_CANDIDATE["memory_estimate"]
QWEN_WORKER_MIB_AT_MAX = (int(QWEN_MEMORY["fixed_compute_mib"])
                          + int(QWEN_MEMORY["kv_mib_at_262144"]))
QWEN_SHARED_GPU0_MIB = int(QWEN_MEMORY["shared_parent_mib_per_chat"])
QWEN_PARENT_RESERVE_MIB = int(QWEN_MEMORY["shared_parent_reserve_mib"])
QWEN_WORKER_RESERVE_MIB = int(QWEN_MEMORY["admission_reserve_mib"])
NEMO_CONTEXT_OVERHEAD_MIB = int(NEMO_MEMORY["fixed_compute_mib"])
NEMO_KV_TOKENS_PER_MIB = MAX_CONTEXT_CAP // int(NEMO_MEMORY["kv_mib_at_262144"])
NEMO_SHARED_GPU0_MIB = int(NEMO_MEMORY["shared_parent_mib_per_chat"])
NEMO_MAX_CHILDREN = int(NEMO_CANDIDATE["max_attached_chats"])


def max_worker_chats(model_id: str, mode: str | None = None) -> int:
    """Enforce the imported measured chat count for this service layout."""
    if model_id == NEMO_ID:
        return NEMO_MAX_CHILDREN
    candidate = (QWEN_HEADLESS_CANDIDATE if mode == "gpu0-worker-pool-q4"
                 else QWEN_CANDIDATE)
    return int(candidate["max_attached_chats"])


def parent_reserve_mib(model_id: str) -> int:
    # Qwen's first active turns retain about 140 MiB of parent-side runtime
    # allocations with six chats.  Keep that above the ordinary 256 MiB floor.
    return QWEN_PARENT_RESERVE_MIB if model_id == MODEL_ID else WORKER_RESERVE_MIB


def worker_reserve_mib(model_id: str) -> int:
    # Qwen workers add about 8 MiB on their first active turn. Keep the active
    # floor above 256 MiB rather than admitting against an idle-only snapshot.
    return QWEN_WORKER_RESERVE_MIB if model_id == MODEL_ID else WORKER_RESERVE_MIB


def estimated_worker_mib(model_id: str, context_cap: int, total_mib: int | None = None) -> int:
    """Reserve the measured per-chat allocation for the requested context."""
    if model_id == NEMO_ID:
        return ((context_cap + NEMO_KV_TOKENS_PER_MIB - 1) // NEMO_KV_TOKENS_PER_MIB
                + NEMO_CONTEXT_OVERHEAD_MIB)
    return (QWEN_WORKER_MIB_AT_MAX * context_cap + MAX_CONTEXT_CAP - 1) // MAX_CONTEXT_CAP


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
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(STATE)
    directory = os.open(STATE.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


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


def recover_pending_operation(st: dict) -> dict:
    """Finish or roll back a durable attach/detach transaction.

    The controller lock serializes live operations.  The pending record closes
    the remaining process-crash window between changing the server and saving
    the authoritative reservation ledger.
    """
    operation = st.get("pending_operation")
    if not isinstance(operation, dict):
        return st
    action, chat_id = operation.get("action"), operation.get("chat_id")
    if action not in ("attach", "detach") or not isinstance(chat_id, str):
        raise RuntimeError("invalid pending shared-chat operation")
    code, value = request(st["port"], "/chats")
    if code != 200:
        raise RuntimeError(f"could not recover pending operation ({code}): {value}")
    live = {row.get("chat_id"): row for row in value.get("chats", [])
            if isinstance(row, dict)}
    claim_name = operation.get("claim")
    claim_key = operation.get("claim_key", chat_id)
    claim_created = operation.get("claim_created", True)
    if action == "attach":
        if chat_id in live:
            st.setdefault("reservations", {})[chat_id] = operation["reservation"]
            if claim_created and isinstance(claim_name, str):
                st.setdefault("claims", {})[claim_key] = claim_name
        elif claim_created and isinstance(claim_name, str):
            release(claim_name)
    else:
        if chat_id in live:
            delete_code, delete_value = request(st["port"], f"/chats/{chat_id}", "DELETE")
            if delete_code not in (200, 404):
                raise RuntimeError(
                    f"could not recover pending detach ({delete_code}): {delete_value}")
        st.setdefault("reservations", {}).pop(chat_id, None)
        still_used = any(receipt.get("claim") == claim_name for key, receipt in
                         st.get("reservations", {}).items() if key != "main")
        if claim_created and isinstance(claim_name, str) and not still_used:
            release(claim_name)
            st.setdefault("claims", {}).pop(claim_key, None)
    st.pop("pending_operation", None)
    save(st)
    return st


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
                     item["name"].startswith(("qwen35-kv-chat-", "qwen35-kv-gpu-",
                                              "nemo74-kv-chat-", "nemo74-kv-gpu-")))):
            release(item["name"])


def renew_claims_once(instance_id: str) -> bool:
    """Renew only this live service instance; release its claims if the container died."""
    with locked():
        st = state()
        if not st or st.get("instance_id") != instance_id:
            return False
        if not container_running():
            for name in claim_names(st):
                release(name)
            STATE.unlink(missing_ok=True)
            return False
        st = recover_pending_operation(st)
        ctl = controller()
        for name in claim_names(st):
            run(["rig", "claim", "renew", name, "--ttl", str(CLAIM_TTL_S), "--controller", ctl])
        return True


def stop_unrenewable_service(instance_id: str):
    """Do not leave a running container behind after its reservations lapse."""
    with locked():
        st = state()
        if not st or st.get("instance_id") != instance_id:
            return
        run(["docker", "stop", CONTAINER], check=False)
        for name in claim_names(st):
            release(name)
        STATE.unlink(missing_ok=True)


def launch_renewer(instance_id: str):
    with (STATE.parent / "kv_chat_claim_renewer.log").open("a", encoding="utf-8") as log:
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "renew-loop",
                          "--instance-id", instance_id], stdin=subprocess.DEVNULL,
                         stdout=log, stderr=log, start_new_session=True)


def claim_names(st: dict) -> set[str]:
    """Return unique live claims; pooled worker claims may have several chats."""
    return {name for name in st.get("claims", {}).values() if isinstance(name, str)}


def runtime_gpu_ordinals() -> dict[int, int]:
    """Map physical nvidia-smi IDs to deterministic CUDA PCI-order ordinals."""
    output = run(["nvidia-smi", "--query-gpu=index,pci.bus_id",
                  "--format=csv,noheader,nounits"]).stdout
    rows = []
    for line in output.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 2 or not fields[0].isdigit():
            raise RuntimeError(f"invalid GPU PCI inventory row: {line}")
        rows.append((fields[1].lower(), int(fields[0])))
    return {physical: ordinal for ordinal, (_, physical) in enumerate(sorted(rows))}


def worker_cards(model_id=MODEL_ID) -> list[dict]:
    """Discover physical cards and their deterministic runtime ordinals."""
    output = run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free",
                  "--format=csv,noheader,nounits"]).stdout
    ordinals = runtime_gpu_ordinals()
    cards = []
    for line in output.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 5:
            raise RuntimeError(f"invalid GPU inventory row: {line}")
        gpu, name, total, used, free = fields
        if not all(value.isdigit() for value in (gpu, total, used, free)):
            raise RuntimeError(f"invalid GPU inventory row: {line}")
        physical = int(gpu)
        if physical != 0 and int(total) >= 6000:
            cards.append({"gpu": physical, "runtime_gpu": ordinals.get(physical, physical),
                          "name": name, "total_mib": int(total), "used_mib": int(used),
                          "free_mib": int(free)})
    return sorted(cards, key=lambda card: (card["total_mib"], card["gpu"]))


def parent_gpu_free_mib() -> int:
    output = run(["nvidia-smi", "--query-gpu=memory.free",
                  "--format=csv,noheader,nounits", "-i", "0"]).stdout.strip()
    if not output.isdigit():
        raise RuntimeError(f"invalid parent GPU memory row: {output}")
    return int(output)


def observed_free_mib(card: dict) -> int:
    """Use the driver's free counter; total-used omits reserved memory."""
    if "free_mib" in card:
        return int(card["free_mib"])
    return int(card["total_mib"]) - int(card["used_mib"])


def available_worker_cards(model_id: str, context_cap: int, chats: list[dict],
                           inventory: list[dict], claims: list[dict],
                           reservations: dict | None = None,
                           owned_claims: set[str] | None = None,
                           parent_free_mib: int | None = None,
                           mode: str | None = None) -> list[dict]:
    """Use the same admission predicate for status and attach."""
    reservations = reservations or {}
    owned_claims = owned_claims or set()
    foreign_claimed = {gpu for item in claims if item.get("name") not in owned_claims
                       for gpu in item.get("gpu_ids", [])}
    workers = [receipt for chat_id, receipt in reservations.items() if chat_id != "main"]
    shared_parent_mib = (NEMO_SHARED_GPU0_MIB if model_id == NEMO_ID
                         else QWEN_SHARED_GPU0_MIB)
    if len(workers) >= max_worker_chats(model_id, mode) or (
            parent_free_mib is not None
            and parent_free_mib < shared_parent_mib + parent_reserve_mib(model_id)):
        return []
    attached_by_gpu = {}
    for receipt in workers:
        for gpu in receipt.get("placement", {}).get("gpus", []):
            attached_by_gpu[gpu] = attached_by_gpu.get(gpu, 0) + 1
    required = estimated_worker_mib(model_id, context_cap) + worker_reserve_mib(model_id)
    available = [dict(card, attached_chats=attached_by_gpu.get(card["gpu"], 0),
                      free_mib=observed_free_mib(card))
                 for card in inventory
                 if card["gpu"] not in foreign_claimed
                 and observed_free_mib(card) >= required]
    # Spread first for throughput, then use the tightest sufficient card so larger
    # cards remain available for larger future contexts.
    return sorted(available, key=lambda card: (
        card["attached_chats"] > 0, card["free_mib"], card["gpu"]))


def capacity_report(model_id: str, default_context: int, chats: list[dict],
                    inventory: list[dict], claims: list[dict], st: dict) -> dict:
    parent_free = parent_gpu_free_mib()
    available = available_worker_cards(
        model_id, default_context, chats, inventory, claims,
        st.get("reservations", {}), claim_names(st), parent_free,
        st.get("mode"))
    workers = [receipt for chat_id, receipt in st.get("reservations", {}).items()
               if chat_id != "main"]
    limit = max_worker_chats(model_id, st.get("mode"))
    cards = []
    worker_reserve = worker_reserve_mib(model_id)
    for card in inventory:
        free = observed_free_mib(card)
        if model_id == NEMO_ID:
            max_tokens = max(0, (free - worker_reserve - NEMO_CONTEXT_OVERHEAD_MIB)
                             * NEMO_KV_TOKENS_PER_MIB)
        else:
            max_tokens = max(0, (free - worker_reserve) * MAX_CONTEXT_CAP
                             // QWEN_WORKER_MIB_AT_MAX)
        cards.append({"gpu": card["gpu"], "runtime_gpu": card["runtime_gpu"],
                      "free_mib": free,
                      "max_context_tokens": min(MAX_CONTEXT_CAP, max_tokens)})
    shared_parent_mib = (NEMO_SHARED_GPU0_MIB if model_id == NEMO_ID
                         else QWEN_SHARED_GPU0_MIB)
    parent_slots = max(0, (parent_free - parent_reserve_mib(model_id))
                       // shared_parent_mib)
    target_slots = sum(max(0, observed_free_mib(card) - worker_reserve)
                       // estimated_worker_mib(model_id, default_context)
                       for card in inventory)
    available_slots = min(parent_slots, target_slots, max(0, limit - len(workers)))
    return {"status": "observed", "context_capacity": default_context,
            "free_gpu_ids": [card["gpu"] for card in available],
            "free_gpu_count": len(available),
            "available_chat_slots": available_slots,
            "max_attached_chats": limit,
            "shared_gpu0_free_mib": parent_free,
            "shared_gpu0_mib_per_chat": shared_parent_mib,
            "shared_gpu0_reserve_mib": parent_reserve_mib(model_id),
            "worker_gpu_reserve_mib": worker_reserve,
            "placement_scope": "multi_gpu_then_shared_gpu_contexts",
            "cards": cards}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("start", "attach", "detach", "turn", "status", "stop", "renew-loop"))
    p.add_argument("--port", type=int, default=11524)
    p.add_argument("--model", choices=tuple(MODELS))
    p.add_argument("--chat-id")
    p.add_argument("--gpu", type=int)
    p.add_argument("--message")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--context-cap", type=int,
                   help="context tokens reserved when attaching a new chat (16K..262K)")
    p.add_argument("--main-context-cap", type=int,
                   help="start with this GPU0 chat capacity; Qwen accepts 0 for a headless worker pool")
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
        if not st or st["model"] != selected_model or not healthy(st["port"]):
            p.error("shared-weight service is not ready for requested model")
        if not args.chat_id or args.message is None: p.error("turn needs --chat-id and --message")
        code, value = request(st["port"], f"/chats/{args.chat_id}/turns", "POST",
                              {"message": args.message, "max_tokens": args.max_tokens}, timeout=600)
        print(json.dumps(value))
        raise SystemExit(0 if code == 200 else 1)
    with locked():
        st = state()
        if st and container_running() and healthy(st["port"]):
            st = recover_pending_operation(st)
        if args.action == "status":
            if not st: print(json.dumps({"status": "absent"})); return
            if args.model and st["model"] != args.model: p.error(f"active shared service uses {st['model']}")
            ready = container_running() and healthy(st["port"])
            chats = request(st["port"], "/chats")[1].get("chats", []) if ready else []
            live_reservations = {chat_id: dict(receipt) for chat_id, receipt in
                                 st.get("reservations", {}).items()}
            for row in chats:
                if isinstance(row, dict) and row.get("chat_id") in live_reservations:
                    receipt = live_reservations[row["chat_id"]]
                    receipt["occupied_tokens"] = row.get("used_tokens", 0)
                    placement = receipt.get("placement", {})
                    if placement.get("gpus"):
                        row["runtime_gpu"] = row.get("gpu")
                        row["gpu"] = placement["gpus"][0]
                    row["reservation"] = receipt
            capacity = {"status": "unknown"}
            if ready:
                try:
                    inventory = worker_cards(st["model"])
                    claims = json.loads(run(["rig", "claim", "list"]).stdout)["claims"]
                    default_context = (MAX_CONTEXT_CAP if st["model"] == NEMO_ID
                                       else DEFAULT_QWEN_CHILD_CONTEXT)
                    capacity = capacity_report(
                        st["model"], default_context, chats, inventory, claims, st)
                except (OSError, RuntimeError, ValueError, KeyError) as exc:
                    capacity = {"status": "unknown", "reason": str(exc)}
            print(json.dumps({"status": "ready" if ready else "down", "port": st["port"],
                              "model": st["model"], "mode": st["mode"],
                              "admission_profile_source_sha256": MODEL_LIBRARY_SHA256,
                              "instance_id": st.get("instance_id"), "chats": chats,
                              "worker_capacity": capacity,
                              "reservations": live_reservations,
                              "claims": st["claims"]}, indent=2)); return
        if args.action == "start":
            main_context_cap = (MAX_CONTEXT_CAP if args.main_context_cap is None
                                else args.main_context_cap)
            if main_context_cap != 0 and not MIN_CONTEXT_CAP <= main_context_cap <= MAX_CONTEXT_CAP:
                p.error(f"--main-context-cap must be 0 or {MIN_CONTEXT_CAP}..{MAX_CONTEXT_CAP}")
            if selected_model == NEMO_ID and main_context_cap != MAX_CONTEXT_CAP:
                p.error("Nemotron currently requires the full main context")
            if st and st["model"] != selected_model:
                p.error(f"shared service already uses {st['model']}; stop it explicitly first")
            if st and container_running() and healthy(st["port"]):
                if not st.get("instance_id"):
                    st["instance_id"] = uuid.uuid4().hex
                    save(st)
                    launch_renewer(st["instance_id"])
                print(json.dumps({"status": "ready", "port": st["port"], "already_running": True})); return
            if st and not container_running():
                for name in claim_names(st): release(name)
                STATE.unlink(missing_ok=True)
                st = None
            if st or container_running(): p.error("stale service state or container; inspect and stop it first")
            clear_orphan_claims()
            port = spec["port"] if args.port == 11524 and selected_model == NEMO_ID else args.port
            if not (1024 <= port <= 65535): p.error("port must be 1024..65535")
            if not spec["path"].is_file() or not (BUILD / "bin" / SERVER).is_file():
                p.error("model hotset or KV server binary missing; build the runtime first")
            if selected_model == NEMO_ID and not NEMO_OVERRIDES.is_file():
                p.error("Nemo expert override file missing")
            rig = json.loads(run(["rig", "status", "--json"]).stdout)
            if rig["mode"] != "idle" or rig.get("profiles"):
                p.error("rig has an active load; stop it intentionally before starting shared-weight mode")
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", port)) == 0: p.error("port already in use")
            existing = json.loads(run(["rig", "claim", "list"]).stdout)["claims"]
            parent_gpus = spec["gpus"]
            if any(set(parent_gpus) & set(item["gpu_ids"]) for item in existing):
                p.error("required model GPU already claimed")
            inventory = run(["nvidia-smi", "--query-gpu=index,memory.used",
                             "--format=csv,noheader,nounits"]).stdout
            usage = {int(a.strip()): int(b.strip()) for a, b in
                     (line.split(",") for line in inventory.splitlines())}
            if any(usage.get(gpu, 100000) >= 1000 for gpu in parent_gpus):
                p.error("required model GPU is not free")
            parent_claim = "nemo74-kv-main" if selected_model == NEMO_ID else PARENT_CLAIM
            parent_note = (f"{selected_model} shared weights; headless worker pool"
                           if main_context_cap == 0 else
                           f"{selected_model} shared weights and main {spec['kv']} KV")
            claim(parent_claim, parent_gpus, parent_note, selected_model)
            try:
                env = (["-e", "KV_TENSOR_OVERRIDES_FILE=/nemo_experts.tsv"]
                       if selected_model == NEMO_ID else [])
                env += ["-e", "CUDA_DEVICE_ORDER=PCI_BUS_ID"]
                env += ["-e", "KV_ALLOW_SHARED_GPU=1"]
                env += ["-e", f"KV_MAIN_CONTEXT_CAP={main_context_cap}"]
                mounts = (["-v", f"{NEMO_OVERRIDES}:/nemo_experts.tsv:ro"]
                          if selected_model == NEMO_ID else [])
                run(["docker", "run", "-d", "--rm", "--gpus", "all", "--network", "host",
                     "--name", CONTAINER, "--label", "org.llm-orchestration.kv-chat=true",
                     *env, *mounts, "-v", f"{BUILD}:/build:ro",
                     "-v", f"{spec['path']}:/model.gguf:ro", IMAGE,
                     f"/build/bin/{SERVER}", "/model.gguf", str(port), "262144", spec["kv"]])
                startup_limit = 600 if selected_model == NEMO_ID else 120
                for _ in range(startup_limit):
                    if healthy(port): break
                    if not container_running(): raise RuntimeError("KV server exited during load")
                    time.sleep(1)
                else: raise RuntimeError(f"KV server did not become ready within {startup_limit} seconds")
                active_mode = ("gpu0-worker-pool-q4"
                               if selected_model == MODEL_ID and main_context_cap == 0
                               else spec["mode"])
                st = {"model": selected_model, "mode": active_mode, "port": port,
                      "instance_id": uuid.uuid4().hex,
                      "main_context_cap": main_context_cap,
                      "container": CONTAINER, "claims": {"main": parent_claim},
                      "reservations": {"main": {
                          "reservation_id": uuid.uuid4().hex,
                          "context_capacity": main_context_cap,
                          "occupied_tokens": 0,
                          "placement": {"gpus": [0]},
                          "reservation_scope": "service_parent"}}}
                st["reservations"]["main"]["service_instance_id"] = st["instance_id"]
                save(st)
                launch_renewer(st["instance_id"])
            except Exception:
                run(["docker", "stop", CONTAINER], check=False)
                release(parent_claim)
                STATE.unlink(missing_ok=True)
                raise
            print(json.dumps({"status": "ready", "model": selected_model, "mode": active_mode,
                              "port": port, "instance_id": st["instance_id"],
                              "main_chat": "main" if main_context_cap else None})); return
        if args.action == "stop":
            if not st: p.error("no managed shared-weight service")
            if args.model and st["model"] != args.model: p.error(f"active shared service uses {st['model']}")
            run(["docker", "stop", CONTAINER], check=False)
            for name in claim_names(st): release(name)
            STATE.unlink(missing_ok=True)
            print(json.dumps({"status": "stopped"})); return
        if not st or not container_running() or not healthy(st["port"]):
            p.error("shared-weight parent is not ready")
        if args.model and st["model"] != args.model: p.error(f"active shared service uses {st['model']}")
        if args.action == "attach":
            if not args.chat_id or not args.chat_id.replace("-", "").replace("_", "").isalnum():
                p.error("attach needs an alphanumeric --chat-id (hyphen and underscore allowed)")
            if args.chat_id == "main" or args.chat_id in st.get("reservations", {}):
                p.error("chat ID already in use")
            context_cap = args.context_cap or MAX_CONTEXT_CAP
            if not MIN_CONTEXT_CAP <= context_cap <= MAX_CONTEXT_CAP:
                p.error(f"--context-cap must be {MIN_CONTEXT_CAP}..{MAX_CONTEXT_CAP}")
            code, value = request(st["port"], "/chats")
            if code != 200: raise RuntimeError(value)
            inventory = worker_cards(st["model"])
            eligible = {card["gpu"] for card in inventory}
            if args.gpu is not None and args.gpu not in eligible:
                p.error(f"GPU {args.gpu} is not an eligible worker card; detected {sorted(eligible)}")
            candidates = ([card for card in inventory if card["gpu"] == args.gpu] if args.gpu is not None
                          else inventory)
            claims = json.loads(run(["rig", "claim", "list"]).stdout)["claims"]
            available = available_worker_cards(st["model"], context_cap,
                                               value["chats"], candidates, claims,
                                               st.get("reservations", {}), claim_names(st),
                                               parent_gpu_free_mib(), st.get("mode"))
            selected = available[0]["gpu"] if available else None
            if selected is None:
                p.error(f"no admissible placement for a {context_cap}-token attached chat")
            reservation_mib = estimated_worker_mib(
                st["model"], context_cap, available[0]["total_mib"])
            runtime_gpu = available[0].get("runtime_gpu", selected)
            parent_owned = selected in spec["gpus"]
            pooled = not parent_owned
            claim_key = f"worker-gpu-{selected}" if pooled else args.chat_id
            name = (st["claims"]["main"] if parent_owned else
                    st.get("claims", {}).get(claim_key) if pooled else None)
            if name is None:
                name = f"{'nemo74' if st['model'] == NEMO_ID else 'qwen35'}-kv-gpu-{selected}"
                if not pooled:
                    name += f"-{args.chat_id}"
            claim_created = not parent_owned and claim_key not in st.get("claims", {})
            reservation = {"reservation_id": uuid.uuid4().hex,
                           "service_instance_id": st["instance_id"],
                           "context_capacity": context_cap, "occupied_tokens": 0,
                           "placement": {"gpus": [selected], "runtime_gpu": runtime_gpu},
                           "reserved_mib": reservation_mib,
                           "shared_gpu0_mib": (NEMO_SHARED_GPU0_MIB if st["model"] == NEMO_ID
                                               else QWEN_SHARED_GPU0_MIB),
                           "reservation_scope": "gpu_shared_context",
                           "claim": name}
            st["pending_operation"] = {"action": "attach", "chat_id": args.chat_id,
                                       "claim": name, "claim_key": claim_key,
                                       "claim_created": claim_created,
                                       "reservation": reservation}
            save(st)
            try:
                if claim_created:
                    claim(name, selected, f"{st['model']} {spec['kv']} KV pool; "
                          f"context {context_cap}; reservation {reservation_mib} MiB; "
                          f"parent {st['claims']['main']}", st["model"])
                code, value = request(st["port"], "/chats", "POST", {
                    "chat_id": args.chat_id, "gpu": runtime_gpu,
                    "context_cap": context_cap}, timeout=120)
                if code != 201: raise RuntimeError(f"attach failed ({code}): {value}")
            except Exception:
                if claim_created:
                    release(name)
                st.pop("pending_operation", None)
                save(st)
                raise
            if claim_created:
                st["claims"][claim_key] = name
            st.setdefault("reservations", {})[args.chat_id] = reservation
            st.pop("pending_operation", None)
            save(st)
            value["gpu"] = selected
            value["runtime_gpu"] = runtime_gpu
            print(json.dumps({**value, "reservation": reservation,
                              "service_instance_id": st["instance_id"],
                              "estimated_reservation_mib": reservation_mib,
                              "route": f"http://127.0.0.1:{st['port']}/chats/{args.chat_id}/turns"})); return
        if args.action == "detach":
            if (not args.chat_id or args.chat_id == "main"
                    or args.chat_id not in st.get("reservations", {})):
                p.error("detach needs an attached worker --chat-id")
            reservation = st.setdefault("reservations", {}).get(args.chat_id)
            claim_name = reservation.get("claim") if reservation else None
            physical_gpu = reservation.get("placement", {}).get("gpus", [None])[0]
            claim_key = (f"worker-gpu-{physical_gpu}"
                         if claim_name != st["claims"].get("main") else args.chat_id)
            claim_created = claim_name != st["claims"].get("main")
            st["pending_operation"] = {"action": "detach", "chat_id": args.chat_id,
                                       "claim": claim_name, "claim_key": claim_key,
                                       "claim_created": claim_created,
                                       "reservation": reservation}
            save(st)
            code, value = request(st["port"], f"/chats/{args.chat_id}", "DELETE")
            if code not in (200, 404):
                raise RuntimeError(f"detach failed ({code}): {value}")
            st.setdefault("reservations", {}).pop(args.chat_id, None)
            still_used = any(receipt.get("claim") == claim_name for key, receipt in
                             st.get("reservations", {}).items() if key != "main")
            if claim_created and isinstance(claim_name, str) and not still_used:
                release(claim_name)
                st.setdefault("claims", {}).pop(claim_key, None)
            st.pop("pending_operation", None)
            save(st)
            print(json.dumps({**value, "released_reservation": reservation,
                              "service_instance_id": st["instance_id"]})); return


if __name__ == "__main__":
    main()
