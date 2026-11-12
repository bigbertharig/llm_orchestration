#!/usr/bin/env python3
"""Named, experimental model loads on the six-card rig.

The benchmark repository records evidence; this module owns the executable
placements. The rig controller still performs admission, loading and readiness.
"""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "agents"))
import orchestration_model_library as model_library  # noqa: E402


MODELS = ("qwen3.6:35b-a3b-xl", "nemotron-3.5-bolt:74b-a9b", "qwen3-coder-next:80b-a3b")


def expert_map(groups, kind):
    return ",".join(
        f"blk[.]({'|'.join(map(str, layers))})[.]ffn_{kind}_exps[.]weight=CUDA{gpu}"
        for gpu, layers in enumerate(groups, 1)
    )


CODER_GROUPS = [list(range(start, start + 6)) for start in range(0, 30, 6)]
NEMO_GROUPS = [
    [1, 3, 5, 8, 10, 12, 14],
    [17, 19, 21, 23, 26],
    [28, 30, 32, 34, 37],
    [39, 41, 43, 45, 48],
    [50, 52, 54, 56, 59, 61],
]


def mode_spec(model, mode):
    if model not in MODELS:
        raise ValueError(f"unknown model {model}; choose from {', '.join(MODELS)}")
    path = model_library.resolve(model)["gguf_path"]
    spec = {
        "model": model, "gguf_path": path, "use": "benchmark",
        "batch_size": 512, "parallel": 1, "image": "llama-runtime:b11669",
        "extra_args": ["--ubatch-size", "32", "-ctk", "f16", "-ctv", "f16"],
    }
    if model == "qwen3.6:35b-a3b-xl" and mode == "single-chat-fast":
        spec.update(ctx_size=16384, tensor_split="48")
        spec["extra_args"] = ["--reasoning", "off"]
        return spec, [0]
    if model == "nemotron-3.5-bolt:74b-a9b" and mode == "single-chat-fast":
        spec.update(ctx_size=262144, tensor_split="88,0,0,0,0")
        spec["extra_args"] += ["--reasoning", "off", "--override-tensor", expert_map(NEMO_GROUPS[:4], "(up|down)")]
        return spec, list(range(5))
    if model == "qwen3-coder-next:80b-a3b" and mode == "single-chat-fast":
        spec.update(ctx_size=262144, tensor_split="48,0,0,0,0,0")
        spec["extra_args"] += ["--override-tensor", expert_map(CODER_GROUPS, "(up|down|gate)")]
        return spec, list(range(6))
    raise ValueError(f"no executable mode {mode} for {model}")


def catalog():
    rows = [{"model": model, "mode": "single-chat-fast", "kind": "model_load",
             "executable": True} for model in MODELS]
    rows.append({"model": "qwen3.6:35b-a3b-xl", "mode": "gpu0-main-q4",
                 "kind": "model_load", "executable": True,
                 "chat_gpu": 0, "context_cap": 262144,
                 "runtime": "shared-kv-chat", "supports_mode": "gpu1060-chat-q4"})
    rows.append({"model": "qwen3.6:35b-a3b-xl", "mode": "gpu1060-chat-q4",
                 "kind": "attached_chat", "executable": True,
                 "requires": {"model": "qwen3.6:35b-a3b-xl", "mode": "gpu0-main-q4"},
                 "candidate_gpus": [1, 2, 3, 4, 5], "context_cap": 262144,
                 "runtime": "shared-kv-chat", "route_kind": "per-chat-turns"})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("list", "show", "start", "stop", "status", "turn"))
    p.add_argument("model", nargs="?")
    p.add_argument("mode", nargs="?")
    p.add_argument("--port", type=int)
    p.add_argument("--lease")
    p.add_argument("--detach", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--json", action="store_true", help="machine-readable list")
    p.add_argument("--chat-id")
    p.add_argument("--gpu", type=int, help="1060 GPU index for an attached chat; default selects a free one")
    p.add_argument("--message")
    p.add_argument("--max-tokens", type=int, default=128)
    a = p.parse_args()
    if a.action == "list":
        rows = catalog()
        if a.json:
            print(json.dumps(rows, indent=2))
        else:
            print("MODEL\tMODE")
            for row in rows:
                print(f"{row['model']}\t{row['mode']}")
        return
    if not a.model or not a.mode:
        p.error("show/start needs MODEL MODE")
    attached = next((row for row in catalog() if row["model"] == a.model and
                     row["mode"] == a.mode and row["kind"] == "attached_chat"), None)
    shared_main = a.model == "qwen3.6:35b-a3b-xl" and a.mode == "gpu0-main-q4"
    if attached or shared_main:
        if a.action == "show":
            if attached:
                print(json.dumps(attached, indent=2))
            else:
                print(json.dumps(next(row for row in catalog() if row["model"] == a.model and row["mode"] == a.mode), indent=2))
            return
        if a.lease:
            p.error("shared-KV chat mode uses rig claims; --lease is not supported yet")
        if attached and a.action in ("start", "stop") and not a.chat_id:
            p.error("worker start/stop needs --chat-id")
        if a.action == "turn" and (not a.chat_id or a.message is None):
            p.error("turn needs --chat-id and --message")
        action = ("attach" if a.action == "start" else "detach" if a.action == "stop" else a.action) if attached else a.action
        cmd = ["python3", str(Path(__file__).with_name("kv_chat_control.py")), action]
        if a.port: cmd += ["--port", str(a.port)]
        if a.chat_id: cmd += ["--chat-id", a.chat_id]
        if a.gpu is not None: cmd += ["--gpu", str(a.gpu)]
        if a.message is not None: cmd += ["--message", a.message]
        if a.action == "turn": cmd += ["--max-tokens", str(a.max_tokens)]
        if a.dry_run:
            print(shlex.join(cmd))
            return
        raise SystemExit(subprocess.run(cmd, check=False).returncode)
    if a.action not in ("show", "start"):
        p.error("this action is only available for the 35B shared-KV mode")
    try:
        spec, gpus = mode_spec(a.model, a.mode)
    except ValueError as exc:
        p.error(str(exc))
    if a.action == "show":
        print(json.dumps({"model": a.model, "mode": a.mode, "gpu_ids": gpus, "spec": spec}, indent=2))
        return
    if not a.port or not 1024 <= a.port <= 65535:
        p.error("start needs --port 1024..65535")
    name = f"mode-{a.model.replace(':', '-')}-{a.mode}"
    cmd = ["/usr/local/bin/rig", "start", "remote", name, "--spec", json.dumps(spec, separators=(",", ":")),
           "--gpus", ",".join(map(str, gpus)), "--port", str(a.port)]
    if a.lease:
        cmd += ["--lease", a.lease]
    if a.detach:
        cmd += ["--detach"]
    if a.dry_run:
        print(shlex.join(cmd))
        return
    raise SystemExit(subprocess.run(cmd, check=False).returncode)


if __name__ == "__main__":
    main()
