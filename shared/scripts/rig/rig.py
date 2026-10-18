#!/home/bryan/llm-orchestration-venv/bin/python
"""Rig mode selector. Runs on the GPU rig; other machines call it over SSH.

    ssh rig rig status [--json]
    ssh rig rig start <orchestration|bench> [--force]
    ssh rig rig start remote [profile ...] [--force]
    ssh rig rig stop [profile ...]

One mode at a time. Boot leaves the rig idle (nothing running, no model on any GPU).
`start` refuses while a different mode is active unless --force is given.
Remote mode is additive: `start remote <id>` loads that profile onto free GPUs that match
its requirements, alongside profiles already loaded. `stop <id>` unloads one profile.
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
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

# Rig-local tree: the checkout this file lives in (<LOCAL>/scripts/rig/rig.py). State, config,
# catalog, profiles and the runtime script come from here, so idle/remote never need the network.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hw_jobs  # noqa: E402  (rig-side detached jobs, beside this file)

LOCAL = Path(__file__).resolve().parents[2]
# Network storage (shared drive, later a NAS). Only the orchestration/bench stacks, the control
# policy they read, and the model archive live here; it may be missing at any time.
SHARED = Path("/mnt/shared")
STATE_PATH = LOCAL / "brain" / "rig_mode.json"
STATE_MIRROR_PATH = SHARED / "brain" / "rig_mode.json"  # read by watchers on the Pi
LOCK_PATH = LOCAL / "brain" / "rig_mode.lock"
CONFIG_PATH = LOCAL / "agents" / "config.json"
# Remote profiles: llm-gateway's generated remote_profiles.json (see remote_profiles.py beside this file).
RUN_RUNTIME = LOCAL / "scripts" / "llama_runtime" / "run_runtime.sh"

# Stack modes run startup.py under these systemd units (installed from ./systemd/).
STACK_UNITS = {
    "orchestration": "llm-orchestration.service",
    "bench": "llm-bench.service",
}
# control_policy profile written once a mode is ready.
MODE_POLICY = {
    "orchestration": "plan_ready",
    "bench": "benchmark_ready",
    "remote": "custom",
}
MODES = ("orchestration", "bench", "remote")
BRAIN_PORT = 11434
# Remote profiles get the lowest free port in this range (all inside the desktop rig-llm tunnel).
REMOTE_PORTS = range(11434, 11442)
REMOTE_CONTAINER_PREFIX = "rig-remote-"
REMOTE_MEMORY_LIMIT = "20g"
# A GPU counts as free for remote placement only if nearly empty (catches strays).
GPU_FREE_MAX_USED_MB = 1000
STACK_READY_TIMEOUT_S = 600
REMOTE_READY_TIMEOUT_S = 600
# A queued (detached) remote load waits this long for its turn while another load runs.
REMOTE_QUEUE_TIMEOUT_S = 1800
LOCK_WAIT_S = 5

EXIT_FAIL = 1
EXIT_BUSY = 2
EXIT_LOCKED = 3
EXIT_NO_CAPACITY = 4


class RigError(RuntimeError):
    def __init__(self, msg: str, code: int = EXIT_FAIL):
        super().__init__(msg)
        self.code = code


def say(msg: str) -> None:
    print(msg, flush=True)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def caller() -> str:
    # RIG_CALLER: a detached job (hw_jobs) keeps the address of whoever asked for it.
    if os.environ.get("RIG_CALLER"):
        return os.environ["RIG_CALLER"]
    ssh = os.environ.get("SSH_CLIENT", "").split()
    return ssh[0] if ssh else "local"


# ---------------------------------------------------------------- state

def read_state() -> dict:
    if not STATE_PATH.exists():
        return {"mode": "idle", "status": "idle"}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def write_atomic(path: Path, body: str) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(path)


def save_state(state: dict) -> dict:
    body = json.dumps(state, indent=2) + "\n"
    write_atomic(STATE_PATH, body)
    if STATE_MIRROR_PATH != STATE_PATH and shared_available():
        try:
            write_atomic(STATE_MIRROR_PATH, body)
        except OSError as exc:
            say(f"  note: state mirror not written ({exc})")
    return state


def shared_available() -> bool:
    return (SHARED / "agents").is_dir()


def write_state(**fields) -> dict:
    state = {"mode": "idle", "status": "idle", "since": now_iso(), "from": caller()}
    state.update(fields)
    return save_state(state)


class Lock:
    """Exclusive rig lock. Held only for quick state changes, never while a model loads."""

    def __enter__(self):
        self.fh = open(LOCK_PATH, "w")
        deadline = time.time() + LOCK_WAIT_S
        while True:
            try:
                fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except BlockingIOError:
                if time.time() >= deadline:
                    st = read_state()
                    self.fh.close()
                    raise RigError(
                        f"locked: another rig command is running "
                        f"({st.get('status')} {st.get('mode')}, from {st.get('from')})",
                        EXIT_LOCKED,
                    )
                time.sleep(0.5)

    def __exit__(self, *exc):
        self.fh.close()


def write_policy(profile: str, reason: str) -> None:
    # Only the orchestrator stack reads the policy, and it reads it from network storage.
    if not shared_available():
        say(f"  note: {SHARED} not available; control policy not written")
        return
    sys.path.insert(0, str(LOCAL / "agents"))
    from control_policy import write_control_policy

    write_control_policy(SHARED, profile, reason=reason, updated_by="rig")


# ---------------------------------------------------------------- probes

def run(cmd: list[str], check: bool = True, timeout: int | None = 60) -> subprocess.CompletedProcess:
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
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used",
         "--format=csv,noheader,nounits"],
        check=False,
    )
    visible = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4 and parts[0].isdigit():
            visible.append({"index": int(parts[0]), "name": parts[1],
                            "total_mb": int(parts[2]), "used_mb": int(parts[3])})
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


def gpu_telemetry() -> list[dict]:
    proc = run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,power.draw",
         "--format=csv,noheader,nounits"],
        check=False,
    )
    out = []
    for line in proc.stdout.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) != 6 or not p[0].isdigit():
            continue
        num = lambda v: float(v) if v.replace(".", "", 1).isdigit() else None  # noqa: E731
        out.append({"index": int(p[0]), "name": p[1], "total_mb": int(p[2]), "used_mb": int(p[3]),
                    "util_pct": num(p[4]), "power_w": num(p[5])})
    return out


def llama_containers() -> list[dict]:
    """Every running llama-server container, whoever started it: name, GPU ids, port."""
    ids = run(["docker", "ps", "-q"], check=False).stdout.split()
    if not ids:
        return []
    try:
        items = json.loads(run(["docker", "inspect", *ids], check=False, timeout=30).stdout or "[]")
    except ValueError:
        return []
    out = []
    for c in items:
        cfg = c.get("Config") or {}
        cmd = [str(x) for x in (cfg.get("Cmd") or [])]
        if not any("llama-server" in x for x in cmd + [str(x) for x in (cfg.get("Entrypoint") or [])]):
            continue
        gpus: set[int] = set()
        for req in (c.get("HostConfig") or {}).get("DeviceRequests") or []:
            for dev in req.get("DeviceIDs") or []:
                gpus.update(int(d) for d in str(dev).split(",") if d.strip().isdigit())
        for env in cfg.get("Env") or []:
            if env.startswith("NVIDIA_VISIBLE_DEVICES="):
                gpus.update(int(d) for d in env.split("=", 1)[1].split(",") if d.strip().isdigit())
        port = next((int(cmd[i + 1]) for i, a in enumerate(cmd[:-1]) if a == "--port" and cmd[i + 1].isdigit()), None)
        out.append({"name": str(c.get("Name", "")).lstrip("/"), "gpus": sorted(gpus), "port": port,
                    "started_at": (c.get("State") or {}).get("StartedAt")})
    return out


def unit_active(unit: str) -> bool:
    return run(["systemctl", "is-active", "--quiet", unit], check=False).returncode == 0


# ---------------------------------------------------------------- stop

def stop_everything() -> None:
    """Stop any mode and return to idle. Caller holds the lock."""
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


def drop_remote_profile(state: dict, pid: str, error: str | None = None) -> dict:
    """Unload one remote profile; go idle when none remain. Caller holds the lock."""
    entry = state.get("profiles", {}).pop(pid, None)
    if entry:
        run(["docker", "rm", "-f", entry["container"]], check=False, timeout=60)
    if error:
        state["error"] = error
    if not state.get("profiles"):
        write_policy("neutral", "rig_idle")
        return save_state({"mode": "idle", "status": "idle", "since": now_iso(),
                           "from": caller(), **({"error": error} if error else {})})
    return save_state(refresh_remote_status(state))


def refresh_remote_status(state: dict) -> dict:
    loading = any(p["state"] == "loading" for p in state["profiles"].values())
    state["status"] = "loading" if loading else "ready"
    return state


# ---------------------------------------------------------------- stack modes

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
    write_policy(MODE_POLICY[mode], f"rig_start_{mode}")
    return {"unit": unit, "ports": {BRAIN_PORT: served_model(BRAIN_PORT)}}


def cmd_start_stack(args) -> int:
    with Lock():
        st = read_state()
        if st["mode"] != "idle":
            if not args.force:
                return report_busy(st, args.mode)
            say(f"stopping {st['mode']} (forced by {caller()})...")
            stop_everything()
        if not shared_available():
            raise RigError(f"{args.mode} needs network storage at {SHARED}, which is not available")
        g = gpu_check()
        if not g["ok"]:
            raise RigError(f"GPU check failed: missing {g['missing']} {g['errors']}; power-cycle the rig")
        say(f"starting {args.mode}...")
        write_state(mode=args.mode, status="starting")
        try:
            detail = start_stack(args.mode)
        except Exception as exc:
            say(f"FAILED: {exc}")
            say("cleaning up -> idle")
            try:
                stop_everything()
            finally:
                write_state(mode="idle", status="idle", error=f"{args.mode}: {str(exc)[:500]}")
            return EXIT_FAIL
        write_state(mode=args.mode, status="ready", detail=detail)
    say(f"done: {args.mode} ready")
    for port, what in detail.get("ports", {}).items():
        say(f"  :{port}  {what}")
    return 0


def report_busy(st: dict, wanted: str) -> int:
    say(f"busy: {st['mode']} ({st['status']}, since {st.get('since')}, from {st.get('from')})")
    say(f"re-run with --force to stop {st['mode']} and start {wanted}")
    return EXIT_BUSY


# ---------------------------------------------------------------- remote mode

def published_profiles() -> dict[str, dict]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(LOCAL / "agents"))
    import remote_profiles
    from brain_core import resolve_hotset_gguf

    try:
        profiles = remote_profiles.load()
    except (OSError, KeyError, ValueError) as exc:
        raise RigError(f"cannot read remote profiles from {remote_profiles.PATH}: {exc}")
    if not profiles:
        raise RigError(f"no remote_profiles in {remote_profiles.PATH}")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    for p in profiles.values():
        if not p.get("gpus_needed"):
            raise RigError(f"profile {p['id']}: gpus_needed is required")
        p["gguf_path"] = str(resolve_hotset_gguf(p["gguf_path"], config))
    return profiles


def place_profile(profile: dict, free_gpus: list[dict]) -> list[int] | None:
    """Pick one free GPU per requirement, smallest card that fits first (keeps big cards free).

    A requirement is {"min_vram_mb": N} and optionally {"max_vram_mb": M}, which keeps a profile
    meant for small cards (the 1060s) off the 3090 even when the 3090 is free.

    Returns GPU indexes in requirement order (that order matches tensor_split), or None.
    """
    reqs = list(enumerate(profile["gpus_needed"]))
    reqs.sort(key=lambda r: -int(r[1]["min_vram_mb"]))
    pool = sorted(free_gpus, key=lambda g: (g["total_mb"], g["index"]))
    chosen: dict[int, int] = {}
    for pos, req in reqs:
        fit = next((g for g in pool if g["total_mb"] >= int(req["min_vram_mb"])
                    and g["total_mb"] <= int(req.get("max_vram_mb", g["total_mb"]))), None)
        if fit is None:
            return None
        chosen[pos] = fit["index"]
        pool.remove(fit)
    return [chosen[i] for i in range(len(profile["gpus_needed"]))]


def launch_profile(p: dict, gpus: list[int], port: int) -> str:
    name = REMOTE_CONTAINER_PREFIX + p["id"]
    run(["docker", "rm", "-f", name], check=False)
    cmd = [
        str(RUN_RUNTIME), "--name", name, "--model", p["gguf_path"],
        "--port", str(port), "--gpus", "device=" + ",".join(map(str, gpus)),
        "--ctx-size", str(p["ctx_size"]), "--batch-size", str(p["batch_size"]),
        "--parallel", str(p["parallel"]), "--memory-limit", REMOTE_MEMORY_LIMIT,
    ]
    if p.get("tensor_split"):
        cmd += ["--tensor-split", p["tensor_split"]]
    if p.get("image"):
        cmd += ["--image", p["image"]]
    for a in p.get("extra_args", []):
        cmd += ["--extra-arg", a]
    # No timeout: run_runtime.sh holds the rig-wide model-load lock until the model is
    # serving and bounds its own waits; killing it mid-load would free the lock.
    run(cmd, timeout=None)
    return name


def remote_loaded_summary(state: dict) -> str:
    profiles = state.get("profiles", {})
    if not profiles:
        return "none"
    return ", ".join(f"{pid} (GPU {e['gpus']}, :{e['port']}, {e['state']})" for pid, e in profiles.items())


def claim_remote(requested: list[str], force: bool,
                 placement: dict | None = None) -> tuple[list[str], int | None]:
    """Under the lock: enter remote mode if needed, place and launch new profiles.

    placement ({"gpus": [...], "port": N, "lease": id}, one profile only) comes from the rig manager
    (llm-gateway): launch exactly there instead of choosing, after checking the GPUs are free.
    Returns (profile ids to wait on, early exit code or None).
    """
    profiles = published_profiles()
    unknown = [r for r in requested if r not in profiles]
    if unknown:
        raise RigError(f"unknown remote profile(s) {unknown}; published: {sorted(profiles)}")
    wanted = requested or [pid for pid, p in profiles.items() if p.get("default")]

    st = read_state()
    if st["mode"] not in ("idle", "remote") or (force and st["mode"] != "idle"):
        if not force:
            return [], report_busy(st, "remote")
        say(f"stopping {st['mode']} (forced by {caller()})...")
        stop_everything()
        st = read_state()

    g = gpu_check()
    entering = st["mode"] == "idle"
    if entering:
        if not g["ok"]:
            raise RigError(f"GPU check failed: missing {g['missing']} {g['errors']}; power-cycle the rig")
        st = {"mode": "remote", "status": "loading", "since": now_iso(), "from": caller(), "profiles": {}}

    loaded = st.setdefault("profiles", {})
    # One load at a time: concurrent loads share disk and PCIe bandwidth and all finish slower.
    in_flight = [pid for pid, e in loaded.items() if e["state"] == "loading" and pid not in wanted]
    if in_flight:
        raise RigError(f"locked: {', '.join(in_flight)} still loading; retry when ready", EXIT_LOCKED)
    used_gpus = {i for e in loaded.values() for i in e["gpus"]}
    used_ports = {e["port"] for e in loaded.values()}
    free = [x for x in g["visible"] if x["index"] not in used_gpus and x["used_mb"] <= GPU_FREE_MAX_USED_MB]

    plan = []
    for pid in wanted:
        if pid in loaded:
            say(f"  {pid}: already {loaded[pid]['state']} on :{loaded[pid]['port']}")
            continue
        if placement is not None:
            gpus, port = list(placement["gpus"]), int(placement["port"])
            free_ids = {x["index"] for x in free}
            if len(gpus) != len(profiles[pid]["gpus_needed"]):
                raise RigError(f"{pid} needs {len(profiles[pid]['gpus_needed'])} GPU(s); manager gave {gpus}")
            if not set(gpus) <= free_ids:
                raise RigError(f"manager placed {pid} on GPU {gpus} but GPU {sorted(set(gpus) - free_ids)} "
                               f"is not free here", EXIT_NO_CAPACITY)
            if port in used_ports or served_model(port):
                raise RigError(f"manager placed {pid} on :{port} but that port is serving", EXIT_NO_CAPACITY)
        else:
            gpus = place_profile(profiles[pid], free)
            port = next((p for p in REMOTE_PORTS if p not in used_ports and not served_model(p)), None)
        if gpus is None or port is None:
            need = ", ".join(f"{r['min_vram_mb']}-{r['max_vram_mb']}MB" if "max_vram_mb" in r
                             else f">={r['min_vram_mb']}MB" for r in profiles[pid]["gpus_needed"])
            free_desc = ", ".join(f"GPU{x['index']} {x['total_mb']}MB" for x in free) or "none"
            raise RigError(
                f"no capacity for {pid} (needs GPUs [{need}]; free: {free_desc}). "
                f"loaded: {remote_loaded_summary(st)}. Unload with `rig stop <profile>`.",
                EXIT_NO_CAPACITY,
            )
        free = [x for x in free if x["index"] not in gpus]
        used_ports.add(port)
        plan.append((pid, gpus, port))

    if entering:
        write_policy(MODE_POLICY["remote"], "rig_start_remote")
    for pid, gpus, port in plan:
        p = profiles[pid]
        say(f"  loading {pid} ({p['model']}) on GPU {gpus} -> :{port}")
        # Recorded before the launch: run_runtime.sh waits for the rig load lock and returns only once
        # the model serves, and observers (the Pi manager) need to see the load while it waits.
        loaded[pid] = {"model": p["model"], "port": port, "gpus": gpus, "ctx_size": p["ctx_size"],
                       "state": "loading", "container": REMOTE_CONTAINER_PREFIX + pid, "since": now_iso(),
                       "from": caller(), "lease": (placement or {}).get("lease")}
        save_state(refresh_remote_status(st))
        try:
            launch_profile(p, gpus, port)
        except Exception:
            drop_remote_profile(st, pid, error=f"{pid}: launch failed")
            raise
    if loaded:
        save_state(refresh_remote_status(st))
    return wanted, None


def wait_remote(pid: str) -> str | None:
    """Wait (without the lock) until a profile is ready. Returns an error string or None."""
    deadline = time.time() + REMOTE_READY_TIMEOUT_S
    while True:
        entry = read_state().get("profiles", {}).get(pid)
        if entry is None:
            return f"{pid} was unloaded while waiting"
        if entry["state"] == "ready":
            return None
        if http_get(f"http://127.0.0.1:{entry['port']}/health")[0] == 200:
            with Lock():
                st = read_state()
                if pid in st.get("profiles", {}):
                    st["profiles"][pid]["state"] = "ready"
                    save_state(refresh_remote_status(st))
            return None
        running = run(["docker", "inspect", "-f", "{{.State.Running}}", entry["container"]],
                      check=False).stdout.strip()
        if running != "true" or time.time() > deadline:
            logs = run(["docker", "logs", "--tail", "15", entry["container"]], check=False)
            why = "container exited" if running != "true" else f"not ready within {REMOTE_READY_TIMEOUT_S}s"
            err = f"{pid}: {why}\n{logs.stdout}{logs.stderr}".strip()
            with Lock():
                drop_remote_profile(read_state(), pid, error=err[:800])
            return err
        time.sleep(3)


def detach_remote(pid: str, args) -> int:
    """Run `start remote <pid> --queue ...` as a rig job and return at once (see hw_jobs)."""
    profiles = published_profiles()
    if pid not in profiles:
        raise RigError(f"unknown remote profile {pid}; published: {sorted(profiles)}")
    entry = read_state().get("profiles", {}).get(pid)
    if entry:
        job = {"job": hw_jobs.unit_name(pid), "profile": pid, "active_state": "inactive",
               "profile_state": entry["state"], "port": entry["port"], "already_running": True}
    else:
        argv = [sys.executable, str(Path(__file__).resolve()), "start", "remote", pid, "--queue"]
        if args.gpus is not None:
            argv += ["--gpus", args.gpus, "--port", str(args.port)]
        if args.lease:
            argv += ["--lease", args.lease]
        job = {**hw_jobs.launch(pid, argv, caller(), f"rig remote load {pid}"), "profile": pid}
    print(json.dumps(job) if args.json else
          f"{pid}: job {job['job']} {'already running' if job['already_running'] else 'started'}", flush=True)
    return 0


def claim_remote_queued(pid: str, force: bool, placement: dict | None) -> tuple[list[str], int | None]:
    """claim_remote, waiting for our turn while another load runs (rig rule: one load at a time)."""
    deadline = time.time() + REMOTE_QUEUE_TIMEOUT_S
    while True:
        try:
            with Lock():
                return claim_remote([pid], force, placement)
        except RigError as exc:
            if exc.code != EXIT_LOCKED or time.time() > deadline:
                raise
            time.sleep(3)


def cmd_start_remote(args) -> int:
    requested = args.profiles or [pid for pid, p in published_profiles().items() if p.get("default")]
    placement = None
    if args.gpus is not None or args.port is not None or args.lease:
        if len(requested) != 1 or args.gpus is None or args.port is None:
            raise RigError("--gpus/--port/--lease place exactly one profile and need both --gpus and --port")
        placement = {"gpus": [int(g) for g in args.gpus.split(",")], "port": args.port, "lease": args.lease}
    if args.detach:
        if len(requested) != 1:
            raise RigError("--detach starts exactly one profile")
        return detach_remote(requested[0], args)
    wanted: list[str] = []
    errors: list[str] = []
    # Load sequentially: claim and launch one profile, wait until it serves, then the next.
    for i, pid in enumerate(requested):
        if args.queue:
            claimed, early = claim_remote_queued(pid, args.force and i == 0, placement)
        else:
            with Lock():
                claimed, early = claim_remote([pid], args.force and i == 0, placement)
        if early is not None:
            return early
        wanted += claimed
        if err := wait_remote(pid):
            errors.append(err)
            break
    st = read_state()
    for pid, e in st.get("profiles", {}).items():
        if pid in wanted:
            say(f"  ready {pid} on :{e['port']} (GPU {e['gpus']}, ctx {e['ctx_size']})")
    if errors:
        for e in errors:
            say(f"FAILED: {e}")
        return EXIT_FAIL
    say("done: remote ready")
    return 0


# ---------------------------------------------------------------- commands

def status_payload() -> dict:
    st = read_state()
    g = gpu_check()
    profiles = [
        {"id": pid, **{k: e.get(k) for k in ("model", "port", "gpus", "ctx_size", "state", "since", "from", "lease")}}
        for pid, e in st.get("profiles", {}).items()
    ]
    return {
        "mode": st["mode"], "status": st["status"], "since": st.get("since"), "from": st.get("from"),
        "last_error": st.get("error"), "profiles": profiles, "detail": st.get("detail"),
        "gpus": {"visible": len(g["visible"]), "expected": g["expected"], "missing": g["missing"]},
    }


def cmd_status(args) -> int:
    s = status_payload()
    if args.json:
        print(json.dumps(s, indent=2))
        return 0
    say(f"mode: {s['mode']}   status: {s['status']}   since: {s['since'] or '-'}   from: {s['from'] or '-'}")
    for p in s["profiles"]:
        say(f"  {p['id']:<10} {p['state']:<8} :{p['port']}  GPU {p['gpus']}  ctx {p['ctx_size']}  ({p['model']})")
    if s["detail"]:
        say(f"detail: {json.dumps(s['detail'])}")
    if s["last_error"]:
        say(f"last error: {s['last_error']}")
    g = s["gpus"]
    say(f"gpus: {g['visible']}/{g['expected']} visible" + (f"  MISSING {g['missing']}" if g["missing"] else ""))
    ports = live_ports()
    say("ports: " + (", ".join(f":{p} {m}" for p, m in ports.items()) if ports else "none serving"))
    if s["mode"] == "idle" and (ports or any(unit_active(u) for u in STACK_UNITS.values())):
        say("WARNING: state says idle but something is running; run `rig stop` to clean up")
    return 0


def cmd_observe(_args) -> int:
    """Read-only snapshot for the rig manager: GPUs, every llama container, served ports, remote state."""
    st = read_state()
    print(json.dumps({
        "observed_at": now_iso(),
        "expected_gpus": expected_gpu_ids(),
        "gpus": gpu_telemetry(),
        "containers": llama_containers(),
        "ports": {str(p): m for p, m in live_ports().items()},
        "mode": st["mode"],
        "last_error": st.get("error"),
        "profiles": {pid: {k: e.get(k) for k in ("port", "gpus", "state", "container", "lease")}
                     for pid, e in st.get("profiles", {}).items()},
        "jobs": hw_jobs.jobs(),
    }, indent=2))
    return 0


def cmd_start(args) -> int:
    if args.mode == "remote":
        return cmd_start_remote(args)
    if args.profiles:
        raise RigError("profiles are only valid for remote mode")
    return cmd_start_stack(args)


def cmd_stop(args) -> int:
    with Lock():
        st = read_state()
        if args.profiles:
            if st["mode"] != "remote":
                raise RigError(f"profile stop only applies in remote mode (mode is {st['mode']})")
            missing = [p for p in args.profiles if p not in st.get("profiles", {})]
            if missing:
                raise RigError(f"not loaded: {missing}; loaded: {remote_loaded_summary(st)}")
            for pid in args.profiles:
                say(f"unloading {pid}...")
                st = drop_remote_profile(st, pid)
            say(f"mode: {st['mode']}; loaded: {remote_loaded_summary(st)}")
            return 0
        say(f"stopping {st['mode']}...")
        save_state({**st, "status": "stopping"})
        stop_everything()
    say("idle")
    return 0


def cmd_boot(_args) -> int:
    """Called by rig-boot.service at power-on: record idle + GPU health."""
    with Lock():
        g = gpu_check()
        write_policy("neutral", "rig_idle")
        err = None if g["ok"] else f"boot GPU check: missing {g['missing']}"
        write_state(mode="idle", status="idle", **({"error": err} if err else {}), **{"from": "boot"})
    say(f"rig idle; gpus {len(g['visible'])}/{g['expected']}" + (f" MISSING {g['missing']}" if g["missing"] else ""))
    return 0 if g["ok"] else EXIT_FAIL


def main() -> int:
    if not LOCAL.joinpath("agents").is_dir():
        say(f"error: {LOCAL}/agents not found; rig.py must run from a rig checkout")
        return EXIT_FAIL
    STATE_PATH.parent.mkdir(exist_ok=True)
    ap = argparse.ArgumentParser(prog="rig", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_status)
    s = sub.add_parser("start")
    s.add_argument("mode", choices=MODES)
    s.add_argument("profiles", nargs="*", help="remote profile ids (default: profiles marked default)")
    s.add_argument("--force", action="store_true", help="stop the current mode first")
    s.add_argument("--gpus", help="remote, one profile: GPU ids chosen by the rig manager, e.g. 0 or 1,2")
    s.add_argument("--port", type=int, help="remote, one profile: port chosen by the rig manager")
    s.add_argument("--lease", help="remote, one profile: the manager lease it runs under")
    s.add_argument("--detach", action="store_true",
                   help="remote, one profile: run the load as a rig job and return at once (rig observe tracks it)")
    s.add_argument("--queue", action="store_true", help=argparse.SUPPRESS)  # detached job: wait for our turn
    s.add_argument("--json", action="store_true", help="with --detach: print the job as JSON")
    s.set_defaults(fn=cmd_start)
    sub.add_parser("observe", help="JSON snapshot for the rig manager").set_defaults(fn=cmd_observe)
    s = sub.add_parser("stop")
    s.add_argument("profiles", nargs="*", help="remote profile ids to unload (default: stop everything)")
    s.set_defaults(fn=cmd_stop)
    sub.add_parser("boot").set_defaults(fn=cmd_boot)
    args = ap.parse_args()
    try:
        return args.fn(args)
    except RigError as exc:
        say(f"error: {exc}")
        return exc.code


if __name__ == "__main__":
    raise SystemExit(main())
