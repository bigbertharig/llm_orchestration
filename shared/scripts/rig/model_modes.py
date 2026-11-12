#!/usr/bin/env python3
"""Named, experimental model loads on the six-card rig.

The benchmark repository records evidence; this module owns the executable
placements. The rig controller still performs admission, loading and readiness.
"""

import argparse
import json
import shlex
import subprocess


MODELS = {
    "qwen3.6:35b-a3b-xl": {
        "path": "/mnt/shared/models/qwen3.6-35b-a3b-strata-xl/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
    },
    "nemotron-3.5-bolt:74b-a9b": {
        "path": "/mnt/shared/models/nemotron-3.5-bolt-74b-a9b/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-UD-Q3_K_XL.gguf",
    },
    "qwen3-coder-next:80b-a3b": {
        "path": "/mnt/shared/models/qwen3-coder-next/Qwen3-Coder-Next-Q4_K_S.gguf",
    },
}


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
    base = MODELS[model]
    spec = {
        "model": model, "gguf_path": base["path"], "use": "benchmark",
        "batch_size": 512, "parallel": 1, "image": "llama-runtime:b11669",
        "extra_args": ["--ubatch-size", "32", "-ctk", "f16", "-ctv", "f16"],
    }
    if model == "qwen3.6:35b-a3b-xl" and mode == "single-chat-fast":
        spec.update(ctx_size=16384, tensor_split="48")
        spec["extra_args"] = ["--reasoning", "off"]
        return spec, [0]
    if model == "qwen3.6:35b-a3b-xl" and mode == "gpu0-main-q4":
        spec.update(ctx_size=262144, tensor_split="48")
        spec["extra_args"] = ["--reasoning", "off", "-ctk", "q4_0", "-ctv", "q4_0"]
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
                 "chat_gpu": 0, "context_cap": 262144})
    rows.append({"model": "qwen3.6:35b-a3b-xl", "mode": "gpu1060-chat-q4",
                 "kind": "attached_chat", "executable": False,
                 "requires": {"model": "qwen3.6:35b-a3b-xl", "mode": "gpu0-main-q4"},
                 "candidate_gpus": [1, 2, 3, 4, 5], "context_cap": 262144,
                 "reason": "Current HTTP runtime cannot attach per-chat KV on another GPU to a loaded model."})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("list", "show", "start"))
    p.add_argument("model", nargs="?")
    p.add_argument("mode", nargs="?")
    p.add_argument("--port", type=int)
    p.add_argument("--lease")
    p.add_argument("--detach", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--json", action="store_true", help="machine-readable list")
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
    if attached:
        if a.action == "show":
            print(json.dumps(attached, indent=2))
            return
        p.error(attached["reason"] + " Benchmark replay is available; no worker HTTP route is ready.")
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
