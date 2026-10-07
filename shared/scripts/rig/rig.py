#!/home/bryan/llm-orchestration-venv/bin/python
"""Rig mode selector. Runs on the GPU rig; other machines call it over SSH.

    ssh rig rig status
    ssh rig rig start <orchestration|bench|remote> [profile ...] [--force]
    ssh rig rig stop

One mode at a time. Boot leaves the rig idle (nothing running, no model on any GPU).
`start` refuses while another mode is active unless --force is given.
Design: shared/workspace/implement/rig_boot_modes.md
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

SHARED = Path("/mnt/shared")
STATE_PATH = SHARED / "brain" / "rig_mode.json"
LOCK_PATH = SHARED / "brain" / "rig_mode.lock"
CONFIG_PATH = SHARED / "agents" / "config.json"
CATALOG_PATH = SHARED / "agents" / "models.catalog.json"
LIBRARY_PATH = SHARED / "plans" / "shoulders" / "benchmarking" / "model_task_library.json"
RUN_RUNTIME = SHARED / "scripts" / "llama_runtime" / "run_runtime.sh"

# Stack modes run startup.py under these systemd units (installed from ./systemd/).
STACK_UNITS = {
    "orchestration": "llm-orchestration.service",
    "bench": "llm-bench.service",
}
# control_policy profile written once a stack mode is ready.
STACK_POLICY = {
    "orchestration": "plan_ready",
    "bench": "benchmark_ready",
}
MODES = ("orchestration", "bench", "remote")
BRAIN_PORT = 11434
REMOTE_CONTAINER_PREFIX = "rig-remote-"
REMOTE_MEMORY_LIMIT = "20g"
STACK_READY_TIMEOUT_S = 600
REMOTE_READY_TIMEOUT_S = 600
LOCK_WAIT_S = 5

EXIT_FAIL = 1
EXIT_BUSY = 2
EXIT_LOCKED = 3


class RigError(RuntimeError):
    pass


def say(msg: str) -> None:
    print(msg, flush=True)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def caller() -> str:
    ssh = os.environ.get("SSH_CLIENT", "").split()
    return ssh[0] if ssh else "local"


# ---------------------------------------------------------------- state

def read_state() -> dict:
    if not STATE_PATH.exists():
        return {"mode": "idle", "status": "idle"}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def write_state(**fields) -> dict:
    state = {"mode": "idle", "status": "idle", "since": now_iso(), "from": caller()}
    state.update(fields)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    tmp.replace(STATE_PATH)
    return state


def acquire_lock():
    fh = open(LOCK_PATH, "w")
    deadline = time.time() + LOCK_WAIT_S
    while True:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fh
        except BlockingIOError:
            if time.time() >= deadline:
                st = read_state()
                say(
                    f"locked: another rig command is running "
                    f"({st.get('status')} {st.get('mode')}, from {st.get('from')})"
                )
                sys.exit(EXIT_LOCKED)
            time.sleep(0.5)


def write_policy(profile: str, reason: str) -> None:
    sys.path.insert(0, str(SHARED / "agents"))
    from control_policy import write_control_policy

    write_control_policy(SHARED, profile, reason=reason, updated_by="rig")


# ---------------------------------------------------------------- probes

def run(cmd: list[str], check: bool = True, timeout: int = 60) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and proc.returncode != 0:
        raise RigError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr.strip()}")
    return proc


def expected_gpu_ids() -> list[int]:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    ids = [int(g) for g in cfg["brain"]["gpus"]]
    ids += [int(g["id"]) for g in cfg.get("gpus", [])]
    return sorted(ids)


def gpu_check() -> dict:
    proc = run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used", "--format=csv,noheader"],
        check=False,
    )
    visible = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if parts and parts[0].isdigit():
            visible.append({"index": int(parts[0]), "name": parts[1], "mem_used": parts[2]})
    expected = expected_gpu_ids()
    missing = sorted(set(expected) - {g["index"] for g in visible})
    errors = [l.strip() for l in proc.stderr.splitlines() if l.strip()]
    return {"ok": not missing, "expected": len(expected), "visible": visible,
            "missing": missing, "errors": errors[:5]}


def http_get(url: str, timeout: float = 2.0) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception:
        return 0, ""


def served_model(port: int) -> str | None:
    code, body = http_get(f"http://127.0.0.1:{port}/v1/models")
    if code != 200:
        return None
    try:
        return Path(json.loads(body)["data"][0]["id"]).name
    except Exception:
        return "?"


def live_ports() -> dict[int, str]:
    out = {}
    for port in range(11434, 11446):
        m = served_model(port)
        if m:
            out[port] = m
    return out


def running_model_containers() -> list[str]:
    proc = run(["docker", "ps", "--format", "{{.Names}}"], check=False)
    return [n for n in proc.stdout.split()
            if n.startswith("llama-") or n.startswith(REMOTE_CONTAINER_PREFIX)]


def unit_active(unit: str) -> bool:
    return run(["systemctl", "is-active", "--quiet", unit], check=False).returncode == 0


# ---------------------------------------------------------------- stop

def do_stop() -> None:
    for unit in STACK_UNITS.values():
        if unit_active(unit):
            say(f"  stopping {unit}")
            run(["sudo", "-n", "systemctl", "stop", unit], timeout=90)
    # Orchestrator-owned llama containers outlive startup.py, so remove them explicitly.
    for name in running_model_containers():
        say(f"  removing container {name}")
        run(["docker", "rm", "-f", name], check=False, timeout=60)
    leftovers = live_ports()
    if leftovers:
        raise RigError(f"model ports still serving after stop: {leftovers}")
    write_policy("neutral", "rig_idle")
    write_state(mode="idle", status="idle")


# ---------------------------------------------------------------- start

def start_stack(mode: str) -> dict:
    unit = STACK_UNITS[mode]
    started = time.time()
    say(f"  starting {unit}")
    run(["sudo", "-n", "systemctl", "start", unit], timeout=60)
    worker_ids = [int(g["id"]) for g in json.loads(CONFIG_PATH.read_text())["gpus"]]
    deadline = started + STACK_READY_TIMEOUT_S
    last_note = 0.0
    while time.time() < deadline:
        if not unit_active(unit):
            tail = run(["journalctl", "-u", unit, "-n", "15", "--no-pager"], check=False).stdout
            raise RigError(f"{unit} exited during startup:\n{tail}")
        brain_ok = http_get(f"http://127.0.0.1:{BRAIN_PORT}/health")[0] == 200
        fresh = [
            w for w in worker_ids
            if (hb := SHARED / "gpus" / f"gpu_{w}" / "heartbeat.json").exists()
            and hb.stat().st_mtime > started
        ]
        if brain_ok and len(fresh) == len(worker_ids):
            break
        if time.time() - last_note > 30:
            say(f"  waiting: brain={'up' if brain_ok else 'loading'} workers={len(fresh)}/{len(worker_ids)}")
            last_note = time.time()
        time.sleep(3)
    else:
        raise RigError(f"{mode} not ready within {STACK_READY_TIMEOUT_S}s")
    write_policy(STACK_POLICY[mode], f"rig_start_{mode}")
    return {"unit": unit, "ports": {BRAIN_PORT: served_model(BRAIN_PORT)}}


def load_remote_profiles(requested: list[str]) -> list[dict]:
    lib = json.loads(LIBRARY_PATH.read_text(encoding="utf-8"))
    profiles = {p["id"]: p for p in lib.get("remote_profiles", [])}
    if not profiles:
        raise RigError(f"no remote_profiles in {LIBRARY_PATH}")
    if requested:
        unknown = [r for r in requested if r not in profiles]
        if unknown:
            raise RigError(f"unknown remote profile(s) {unknown}; published: {sorted(profiles)}")
        chosen = [profiles[r] for r in requested]
    else:
        chosen = [p for p in profiles.values() if p.get("default")]
    used_gpus: dict[int, str] = {}
    used_ports: dict[int, str] = {}
    for p in chosen:
        for g in p["gpus"]:
            if g in used_gpus:
                raise RigError(f"profiles {used_gpus[g]} and {p['id']} both need GPU {g}")
            used_gpus[g] = p["id"]
        if p["port"] in used_ports:
            raise RigError(f"profiles {used_ports[p['port']]} and {p['id']} both need port {p['port']}")
        used_ports[p["port"]] = p["id"]
    catalog = {m["id"]: m for m in json.loads(CATALOG_PATH.read_text())["models"]}
    for p in chosen:
        if p["model"] not in catalog:
            raise RigError(f"profile {p['id']}: model {p['model']} not in models.catalog.json")
        p["gguf_path"] = catalog[p["model"]]["gguf_path"]
    return chosen


def start_remote(requested: list[str]) -> dict:
    profiles = load_remote_profiles(requested)
    ports = {}
    for p in profiles:
        name = REMOTE_CONTAINER_PREFIX + p["id"]
        run(["docker", "rm", "-f", name], check=False)
        cmd = [
            str(RUN_RUNTIME), "--name", name, "--model", p["gguf_path"],
            "--port", str(p["port"]), "--gpus", "device=" + ",".join(map(str, p["gpus"])),
            "--ctx-size", str(p["ctx_size"]), "--batch-size", str(p["batch_size"]),
            "--parallel", str(p["parallel"]), "--memory-limit", REMOTE_MEMORY_LIMIT,
        ]
        if p.get("tensor_split"):
            cmd += ["--tensor-split", p["tensor_split"]]
        for a in p.get("extra_args", []):
            cmd += ["--extra-arg", a]
        say(f"  loading {p['id']} ({p['model']}) on GPU {p['gpus']} -> :{p['port']}")
        run(cmd, timeout=60)
    for p in profiles:
        name = REMOTE_CONTAINER_PREFIX + p["id"]
        deadline = time.time() + REMOTE_READY_TIMEOUT_S
        while True:
            if http_get(f"http://127.0.0.1:{p['port']}/health")[0] == 200:
                break
            running = run(["docker", "inspect", "-f", "{{.State.Running}}", name], check=False).stdout.strip()
            if running != "true":
                logs = run(["docker", "logs", "--tail", "20", name], check=False)
                raise RigError(f"{p['id']} container exited:\n{logs.stdout}{logs.stderr}")
            if time.time() > deadline:
                raise RigError(f"{p['id']} not ready within {REMOTE_READY_TIMEOUT_S}s")
            time.sleep(3)
        ports[p["port"]] = f"{p['id']} ({p['model']}, ctx {p['ctx_size']})"
        say(f"  ready {p['id']} on :{p['port']}")
    write_policy("custom", "rig_start_remote")
    return {"profiles": [p["id"] for p in profiles], "ports": ports}


# ---------------------------------------------------------------- commands

def cmd_status(_args) -> int:
    st = read_state()
    say(f"mode: {st['mode']}   status: {st['status']}   since: {st.get('since', '-')}   from: {st.get('from', '-')}")
    if st.get("detail"):
        say(f"detail: {json.dumps(st['detail'])}")
    if st.get("error"):
        say(f"last error: {st['error']}")
    g = gpu_check()
    say(f"gpus: {len(g['visible'])}/{g['expected']} visible" + (f"  MISSING {g['missing']}" if g["missing"] else ""))
    ports = live_ports()
    say("ports: " + (", ".join(f":{p} {m}" for p, m in ports.items()) if ports else "none serving"))
    if st["mode"] == "idle" and (ports or any(unit_active(u) for u in STACK_UNITS.values())):
        say("WARNING: state says idle but something is running; run `rig stop` to clean up")
    return 0


def cmd_start(args) -> int:
    if args.mode != "remote" and args.profiles:
        raise RigError("profiles are only valid for remote mode")
    lock = acquire_lock()
    st = read_state()
    if st["mode"] != "idle":
        if not args.force:
            say(f"busy: {st['mode']} ({st['status']}, since {st.get('since')}, from {st.get('from')})")
            say(f"re-run with --force to stop {st['mode']} and start {args.mode}")
            return EXIT_BUSY
        say(f"stopping {st['mode']} (forced by {caller()})...")
        do_stop()
    g = gpu_check()
    if not g["ok"]:
        raise RigError(f"GPU check failed: missing {g['missing']} {g['errors']}; power-cycle the rig")
    say(f"starting {args.mode}...")
    write_state(mode=args.mode, status="starting")
    try:
        detail = start_remote(args.profiles) if args.mode == "remote" else start_stack(args.mode)
    except Exception as exc:
        say(f"FAILED: {exc}")
        say("cleaning up -> idle")
        try:
            do_stop()
        finally:
            write_state(mode="idle", status="idle", error=f"{args.mode}: {str(exc)[:500]}")
        return EXIT_FAIL
    write_state(mode=args.mode, status="ready", detail=detail)
    say(f"done: {args.mode} ready")
    for port, what in detail.get("ports", {}).items():
        say(f"  :{port}  {what}")
    lock.close()
    return 0


def cmd_stop(_args) -> int:
    acquire_lock()
    st = read_state()
    say(f"stopping {st['mode']}...")
    write_state(mode=st["mode"], status="stopping")
    do_stop()
    say("idle")
    return 0


def cmd_boot(_args) -> int:
    """Called by rig-boot.service at power-on: record idle + GPU health."""
    acquire_lock()
    g = gpu_check()
    write_policy("neutral", "rig_idle")
    err = None if g["ok"] else f"boot GPU check: missing {g['missing']}"
    write_state(mode="idle", status="idle", **({"error": err} if err else {}), **{"from": "boot"})
    say(f"rig idle; gpus {len(g['visible'])}/{g['expected']}" + (f" MISSING {g['missing']}" if g["missing"] else ""))
    return 0 if g["ok"] else EXIT_FAIL


def main() -> int:
    if not SHARED.joinpath("agents").is_dir():
        say(f"error: {SHARED}/agents not found; rig.py must run on the GPU rig")
        return EXIT_FAIL
    ap = argparse.ArgumentParser(prog="rig", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    s = sub.add_parser("start")
    s.add_argument("mode", choices=MODES)
    s.add_argument("profiles", nargs="*", help="remote profile ids (default: profiles marked default)")
    s.add_argument("--force", action="store_true", help="stop the current mode first")
    s.set_defaults(fn=cmd_start)
    sub.add_parser("stop").set_defaults(fn=cmd_stop)
    sub.add_parser("boot").set_defaults(fn=cmd_boot)
    args = ap.parse_args()
    try:
        return args.fn(args)
    except RigError as exc:
        say(f"error: {exc}")
        return EXIT_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
