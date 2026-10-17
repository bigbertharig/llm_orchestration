#!/usr/bin/env python3
"""Certify a llama-runtime image against every model the rig actually uses.

The model list is every model named by a role in the benchmarking model library or by a remote
profile llm-gateway offers (scripts/rig/remote_profiles.py). Each loads alone on its normal lane (3090, one 1060, or a 1060 pair) with its
orchestration profile (default profile if it has none), must give a clean answer to the litmus
question (no leaked reasoning) and answer a completion request, and is stopped before the next.
Results go to the share under logs/runtime_upgrades/.

    smoke_runtime.py llama-runtime:b11669-candidate
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREE = HERE.parents[1]  # <tree>/scripts/llama_runtime -> <tree>
sys.path.insert(0, str(TREE / "plans" / "shoulders" / "benchmarking" / "scripts" / "active"))
sys.path.insert(0, str(TREE / "agents"))
sys.path.insert(0, str(TREE / "scripts" / "rig"))
import model_library  # noqa: E402
import remote_profiles  # noqa: E402
from brain_core import resolve_hotset_gguf  # noqa: E402

PORT = 11450  # outside every mode's port range
NAME = "llama-runtime-smoke"
LOAD_TIMEOUT_S = 600
LITMUS = "What is 15 * 37? Reply with ONLY the number."
RESULTS_DIR = Path("/mnt/shared/logs/runtime_upgrades")


def lane(entry: dict) -> tuple[str, str | None]:
    """(docker gpus, tensor split) for a model's normal placement."""
    sched = entry.get("schedule", {})
    if sched.get("placement") == "split_gpu":
        return "device=1,3", "1,1"
    if int(sched.get("tier", 1)) >= 3:
        return "device=0", None
    return "device=1", None


def models_in_use() -> list[str]:
    roles = model_library.load_roles()
    used = [spec["model"] for spec in roles["roles"].values()]
    used += [rp["model"] for rp in remote_profiles.load().values()]
    return list(dict.fromkeys(used))


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())


def ready() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def stop() -> None:
    subprocess.run(["docker", "rm", "-f", NAME], capture_output=True)


def smoke(image: str, model_id: str, registry: dict, config: dict) -> dict:
    entry = registry["models"][model_id]
    profile = "orchestration" if "orchestration" in entry["profiles"] else "default"
    r = model_library.resolve(model_id, profile, registry)
    gpus, split = lane(entry)
    split = r.get("tensor_split") or split
    gguf = str(resolve_hotset_gguf(r["gguf_path"], config))
    cmd = [str(HERE / "run_runtime.sh"), "--name", NAME, "--model", gguf, "--port", str(PORT),
           "--gpus", gpus, "--image", image, "--ctx-size", str(r["runtime"]["ctx_size"]),
           "--batch-size", str(r["runtime"].get("batch_size", 64)), "--memory-limit", "20g"]
    if split and "," in gpus:
        cmd += ["--tensor-split", split]
    for arg in r.get("extra_args", []):
        cmd += ["--extra-arg", str(arg)]
    result = {"model": model_id, "profile": profile, "gpus": gpus, "gguf": gguf, "ok": False}
    stop()
    started = time.time()
    launch = subprocess.run(cmd, capture_output=True, text=True)
    if launch.returncode != 0:
        result["error"] = f"launch failed: {launch.stderr.strip()[-400:]}"
        return result
    while time.time() - started < LOAD_TIMEOUT_S and not ready():
        running = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", NAME],
                                 capture_output=True, text=True).stdout.strip()
        if running != "true":
            break
        time.sleep(3)
    result["load_s"] = round(time.time() - started, 1)
    try:
        if not ready():
            raise RuntimeError("not ready")
        # Litmus from NEW_MODEL_INTEGRATION: a clean short answer. Catches builds where a model's
        # thinking leaks into content (b11669 changed --reasoning-budget 0 for Gemma 4).
        chat = post("/v1/chat/completions", {"messages": [{"role": "user", "content": LITMUS}],
                                             "max_tokens": 64, "temperature": 0})
        post("/v1/completions", {"prompt": "Say OK", "max_tokens": 4})
        content = str(chat["choices"][0]["message"].get("content") or "").strip()
        result["reply"] = content[:40]
        if "555" not in content or len(content) > 12 or "<think" in content:
            raise RuntimeError(f"litmus answer not clean: {content[:80]!r}")
        result["ok"] = True
    except Exception as exc:
        logs = subprocess.run(["docker", "logs", "--tail", "12", NAME], capture_output=True, text=True)
        result["error"] = f"{exc}: {(logs.stdout + logs.stderr).strip()[-600:]}"
    finally:
        stop()
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--model", action="append", default=[], help="limit to these model ids")
    args = ap.parse_args()

    status = subprocess.run(["rig", "status", "--json"], capture_output=True, text=True)
    if status.returncode == 0 and json.loads(status.stdout).get("mode") != "idle":
        print("rig is not idle; run `rig stop` first", file=sys.stderr)
        return 2
    registry = model_library.load_registry()
    config = json.loads((TREE / "agents" / "config.json").read_text(encoding="utf-8"))
    results = []
    for model_id in args.model or models_in_use():
        res = smoke(args.image, model_id, registry, config)
        results.append(res)
        state = "ok  " if res["ok"] else "FAIL"
        print(f"{state} {model_id:<22} {res['gpus']:<11} load {res.get('load_s', '-')}s  "
              f"{res.get('reply', res.get('error', ''))[:90]}", flush=True)
    failed = [r for r in results if not r["ok"]]
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / f"{args.image.split(':')[-1]}_{datetime.now():%Y%m%d_%H%M%S}.json"
        out.write_text(json.dumps({"image": args.image, "results": results}, indent=2) + "\n")
        print(f"results: {out}")
    except OSError as exc:
        print(f"note: results not written ({exc})")
    print(f"{len(results) - len(failed)}/{len(results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
