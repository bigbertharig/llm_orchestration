#!/usr/bin/env python3
"""
Benchmark heartbeat keeper.

Keeps GPU heartbeats fresh and enriched while benchmarks run without the
orchestrator.  Mirrors the data the GPU agent would write so the dashboard
shows useful info (thermal, VRAM, active benchmark containers, progress).

Safety:
- Only writes heartbeats when no GPU agent process is detected for that GPU.
- Sets heartbeat_owner="benchmark" so the orchestrator can detect and
  skip or reclaim cleanly on restart.
- Exits automatically when all benchmark containers finish.

Usage:
    python3 benchmark_heartbeat_keeper.py [--interval 30] [--gpus-dir /mnt/shared/gpus]
"""
import argparse, json, os, re, subprocess, sys, time
from datetime import datetime
from pathlib import Path


def gpu_agent_running(gpu_id: int) -> bool:
    """Check if the orchestrator GPU agent is running for this GPU."""
    try:
        r = subprocess.run(
            ["pgrep", "-f", f"gpu.py.*gpu-{gpu_id}"],
            capture_output=True, text=True, timeout=5,
        )
        return bool(r.stdout.strip())
    except Exception:
        return False


def get_nvidia_smi() -> dict:
    """Return per-GPU stats from nvidia-smi keyed by GPU index."""
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu,clocks.current.sm",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        stats = {}
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7:
                idx = int(parts[0])
                stats[idx] = {
                    "temperature_c": _num(parts[1]),
                    "power_draw_w": _num(parts[2]),
                    "vram_used_mb": _num(parts[3]),
                    "vram_total_mb": _num(parts[4]),
                    "vram_percent": round(_num(parts[3]) / max(_num(parts[4]), 1) * 100, 1),
                    "gpu_util_percent": _num(parts[5]),
                    "clock_mhz": _num(parts[6]),
                }
        return stats
    except Exception:
        return {}


def _num(s: str) -> float:
    try:
        return float(s)
    except Exception:
        return 0.0


def get_cpu_temp() -> float | None:
    """Read CPU package temperature from thermal zones.

    Prefers x86_pkg_temp (actual CPU die temp) over acpitz (motherboard ambient).
    """
    best: float | None = None
    try:
        from pathlib import Path
        for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
            try:
                zone_type = (zone / "type").read_text().strip()
                temp_raw = int((zone / "temp").read_text().strip())
                temp_c = round(temp_raw / 1000, 1)
                if zone_type == "x86_pkg_temp":
                    return temp_c  # CPU package temp — best source
                if best is None:
                    best = temp_c  # fallback to first zone
            except Exception:
                continue
    except Exception:
        pass
    return best


def get_benchmark_containers() -> dict:
    """Return running benchmark containers grouped by GPU device(s).

    Strategy:
    1. Find llama-server containers (own GPUs directly via NVIDIA_VISIBLE_DEVICES)
    2. Find bench-* containers (connect to llama-servers via --runtime-base port)
    3. Map bench containers to GPUs via port -> llama-server -> GPU chain
    """
    try:
        r = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}\t{{.Status}}\t{{.Command}}"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        return {}

    # First pass: build port-to-GPU map from llama/runtime containers
    port_to_gpus: dict[int, list[int]] = {}
    all_lines = []
    for line in r.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        name, status = parts[0], parts[1]
        cmd = parts[2] if len(parts) > 2 else ""
        all_lines.append((name, status))

        if any(kw in name for kw in ("llama-", "tender_")):
            gpu_ids = _container_gpu_ids(name)
            # Find the host port this container exposes
            host_port = _container_host_port(name)
            if host_port and gpu_ids:
                port_to_gpus[host_port] = gpu_ids

    containers = {}
    for name, status in all_lines:
        if not any(kw in name for kw in ("bench-", "llama-", "tender_")):
            continue

        gpu_ids = _container_gpu_ids(name)

        # For bench-* containers, try to find GPU via runtime port
        if name.startswith("bench-") and not gpu_ids:
            full_cmd = _container_full_cmd(name)
            port_match = re.search(r"localhost:(\d+)", full_cmd)
            if port_match:
                port = int(port_match.group(1))
                gpu_ids = port_to_gpus.get(port, [])

        containers[name] = {
            "name": name,
            "status": status,
            "gpu_ids": gpu_ids,
        }
    return containers


def _container_full_cmd(container_name: str) -> str:
    """Get the full command/args for a container."""
    try:
        r = subprocess.run(
            ["docker", "inspect", container_name, "--format", "{{.Config.Cmd}}"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip()
    except Exception:
        return ""


def _container_host_port(container_name: str) -> int | None:
    """Get the host port a container is listening on."""
    try:
        r = subprocess.run(
            ["docker", "inspect", container_name, "--format",
             "{{range $p, $conf := .HostConfig.PortBindings}}"
             "{{range $conf}}{{.HostPort}}{{end}}{{end}}"],
            capture_output=True, text=True, timeout=5,
        )
        port_str = r.stdout.strip()
        if port_str and port_str.isdigit():
            return int(port_str)
    except Exception:
        pass
    # Fallback: check published ports
    try:
        r = subprocess.run(
            ["docker", "port", container_name],
            capture_output=True, text=True, timeout=5,
        )
        for line in r.stdout.splitlines():
            m = re.search(r"-> .*:(\d+)", line)
            if m:
                return int(m.group(1))
    except Exception:
        pass
    # Fallback: extract --port from container command args (for --network host containers)
    cmd = _container_full_cmd(container_name)
    m = re.search(r"--port\s+(\d+)", cmd)
    if m:
        return int(m.group(1))
    return None


def _container_gpu_ids(container_name: str) -> list:
    """Get GPU device IDs assigned to a container."""
    try:
        r = subprocess.run(
            ["docker", "inspect", container_name, "--format",
             "{{range .HostConfig.DeviceRequests}}{{range .DeviceIDs}}{{.}} {{end}}{{end}}"],
            capture_output=True, text=True, timeout=5,
        )
        ids = [int(x) for x in r.stdout.strip().split() if x.isdigit()]
        if ids:
            return ids
    except Exception:
        pass

    # Fallback: check NVIDIA_VISIBLE_DEVICES env var
    try:
        r = subprocess.run(
            ["docker", "inspect", container_name, "--format",
             "{{range .Config.Env}}{{println .}}{{end}}"],
            capture_output=True, text=True, timeout=5,
        )
        for line in r.stdout.splitlines():
            if line.startswith("NVIDIA_VISIBLE_DEVICES="):
                val = line.split("=", 1)[1]
                return [int(x) for x in val.split(",") if x.strip().isdigit()]
    except Exception:
        pass

    # Fallback: infer from container name
    m = re.search(r"gpu(\d+)", container_name)
    if m:
        return [int(m.group(1))]
    # Split containers: check name for split pattern
    m = re.search(r"split-(\d+)(\d+)", container_name)
    if m:
        return [int(m.group(1)), int(m.group(2))]
    return []


def _docker_logs_tail(container_name: str, lines: int = 20) -> str:
    """Get the last N lines of container logs."""
    try:
        r = subprocess.run(
            ["docker", "logs", "--tail", str(lines), container_name],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout + r.stderr
    except Exception:
        return ""


def _parse_task_position(logs: str, tasks_requested: list[str]) -> tuple[str, int | None, int | None]:
    """Parse current task and position from log markers.

    Returns (current_task, task_index, task_total).
    task_index is 1-based (completed + 1).
    """
    running_tasks: list[str] = []
    done_tasks: set[str] = set()
    for m in re.finditer(r"--- Running task:\s*(\S+)\s*---", logs):
        running_tasks.append(m.group(1))
    for m in re.finditer(r"---\s*(\S+)\s+done\s*---", logs):
        done_tasks.add(m.group(1))

    current_task = running_tasks[-1] if running_tasks else ""
    task_total = len(tasks_requested) if tasks_requested else (len(set(running_tasks)) or None)

    if current_task and task_total:
        # Index = number of completed tasks + 1
        if tasks_requested:
            try:
                task_index = tasks_requested.index(current_task) + 1
            except ValueError:
                task_index = len(done_tasks) + 1
        else:
            task_index = len(done_tasks) + 1
        return current_task, task_index, task_total

    return current_task, None, task_total


def _parse_sub_progress(logs: str, suite: str | None) -> tuple[int | None, int | None]:
    """Parse sub-task progress from container logs.

    Returns (sub_current, sub_total).
    """
    if not logs:
        return None, None

    # reasoning / knowledge: "Requesting API: X/Y"
    req_current, req_total = None, None
    for m in re.finditer(r"Requesting API:[^\n\r]*?(\d+)/(\d+)", logs):
        try:
            req_current = int(m.group(1))
            req_total = int(m.group(2))
        except Exception:
            continue
    if req_current is not None and req_total is not None:
        return min(req_current, req_total), req_total

    # code: "Codegen: HumanEval/76" or "Codegen: Mbpp/267"
    codegen_match = None
    for m in re.finditer(r"Codegen:\s*(\w+)/(\d+)", logs):
        codegen_match = m
    if codegen_match:
        task_name = codegen_match.group(1).lower()
        idx = int(codegen_match.group(2))
        # Known totals for common code benchmarks
        known_totals = {"humaneval": 164, "mbpp": 500}
        total = known_totals.get(task_name)
        if total:
            return min(idx, total), total
        return idx, None

    # knowledge: progress bar "X/Y" near "%|"
    for line in reversed(logs.splitlines()):
        if "%|" in line:
            frac = re.search(r"(\d+)/(\d+)", line)
            if frac:
                return int(frac.group(1)), int(frac.group(2))

    return None, None


# Cache for campaign status to avoid re-scanning every heartbeat cycle
_CAMPAIGN_CACHE: dict[str, tuple[float, dict]] = {}
_CAMPAIGN_CACHE_TTL = 30.0  # seconds


def _find_campaign_position(container_name: str) -> tuple[int | None, int | None]:
    """Find campaign step position for this container.

    Searches campaign status.json files for a step whose run_name or suite
    matches the container. Returns (step_index, total_steps) 1-based,
    or (None, None) for non-campaign runs.
    """
    import glob as glob_mod
    campaign_root = Path("/mnt/shared/logs/benchmarks/campaigns/history")
    if not campaign_root.exists():
        return None, None

    # Extract suite type from container name for matching
    container_suite = None
    for prefix in ("bench-reasoning", "bench-code", "bench-pipeline", "bench-knowledge"):
        if prefix in container_name:
            container_suite = prefix
            break

    if not container_suite:
        return None, None

    # Get container's full command to match run_name
    full_cmd = _container_full_cmd(container_name)

    now = time.time()
    for status_path in campaign_root.glob("*/*/status.json"):
        cache_key = str(status_path)
        cached = _CAMPAIGN_CACHE.get(cache_key)
        if cached and (now - cached[0]) < _CAMPAIGN_CACHE_TTL:
            status = cached[1]
        else:
            try:
                with open(status_path) as f:
                    status = json.load(f)
                _CAMPAIGN_CACHE[cache_key] = (now, status)
            except Exception:
                continue

        if not isinstance(status, dict):
            continue
        state = str(status.get("state") or "").lower()
        if state in ("completed", "failed", "cancelled"):
            continue

        steps = status.get("steps")
        if not isinstance(steps, list) or not steps:
            continue

        # Check worker port match
        worker_info = status.get("worker") if isinstance(status.get("worker"), dict) else {}
        worker_port = worker_info.get("port")
        # Try matching via port in container command
        port_in_cmd = re.search(r"localhost:(\d+)", full_cmd)
        cmd_port = int(port_in_cmd.group(1)) if port_in_cmd else None

        if worker_port and cmd_port and int(worker_port) != cmd_port:
            continue

        current_step = str(status.get("current_step") or "")
        enabled_steps = [s for s in steps if s.get("enabled", True)]
        total = len(enabled_steps)

        for i, step in enumerate(enabled_steps):
            step_suite = str(step.get("suite") or "")
            step_id = str(step.get("id") or "")
            step_state = str(step.get("state") or "").lower()
            if step_suite == container_suite or step_id == current_step:
                if step_state == "running" or step_id == current_step:
                    return i + 1, total

    return None, None


def get_benchmark_progress(container_name: str) -> dict | None:
    """Extract 3-level benchmark progress from a container.

    Returns dict with:
    - suite: benchmark suite type (reasoning, code, pipeline, knowledge)
    - campaign_step/campaign_total: position in campaign sequence (or None)
    - current_task/task_index/task_total: position within suite
    - sub_current/sub_total: item-level progress within current task
    - limit: --limit value from container args
    - status: "running" or "completed"
    """
    # 1. Get suite type from container name
    suite = None
    for prefix in ("bench-reasoning", "bench-code", "bench-pipeline", "bench-knowledge"):
        if prefix in container_name:
            suite = prefix.replace("bench-", "")
            break
    if not suite:
        return None

    # 2. Get tasks_requested and limit from container args
    full_cmd = _container_full_cmd(container_name)
    tasks_requested: list[str] = []
    tasks_match = re.search(r"--tasks\s+([^\s\]]+)", full_cmd)
    if tasks_match:
        tasks_requested = [t.strip() for t in tasks_match.group(1).split(",") if t.strip()]

    limit: int | None = None
    limit_match = re.search(r"--limit\s+(\d+)", full_cmd)
    if limit_match:
        limit = int(limit_match.group(1))

    # 3. Parse container logs for current task + progress
    logs = _docker_logs_tail(container_name, 20)

    current_task, task_index, task_total = _parse_task_position(logs, tasks_requested)
    sub_current, sub_total = _parse_sub_progress(logs, suite)

    # 4. Check for campaign membership
    campaign_step, campaign_total = _find_campaign_position(container_name)

    status = "running"
    if "COMPLETE" in logs:
        status = "completed"

    return {
        "suite": suite,
        "campaign_step": campaign_step,
        "campaign_total": campaign_total,
        "tasks_requested": tasks_requested,
        "current_task": current_task,
        "task_index": task_index,
        "task_total": task_total,
        "sub_current": sub_current,
        "sub_total": sub_total,
        "limit": limit,
        "status": status,
    }


def build_active_tasks(containers: dict, gpu_id: int) -> list:
    """Build active_tasks list for a GPU from benchmark containers."""
    tasks = []
    for name, info in containers.items():
        if gpu_id not in info["gpu_ids"]:
            continue
        task_entry = {
            "worker_id": f"benchmark-{name}",
            "task_id": name,
            "task_class": "benchmark",
            "task_name": name,
            "vram_estimate_mb": 0,
            "peak_vram_mb": 0,
            "pid": 0,
            "started_at": datetime.now().isoformat(),
            "phase": "running",
        }
        progress = get_benchmark_progress(name)
        if progress:
            task_entry["benchmark_progress"] = progress
        tasks.append(task_entry)
    return tasks


def update_heartbeat(gpu_id: int, gpus_dir: Path, gpu_stats: dict,
                     containers: dict, cpu_temp: float | None) -> bool:
    """Update heartbeat for one GPU. Returns True if written."""
    hb_path = gpus_dir / f"gpu_{gpu_id}" / "heartbeat.json"
    if not hb_path.parent.exists():
        return False

    # Don't touch if orchestrator agent is running for this GPU
    if gpu_agent_running(gpu_id):
        return False

    # Read existing heartbeat as base
    try:
        with open(hb_path) as f:
            hb = json.load(f)
    except Exception:
        hb = {"gpu_id": gpu_id, "name": f"gpu-{gpu_id}"}

    now = datetime.now().isoformat()
    stats = gpu_stats.get(gpu_id, {})
    active_tasks = build_active_tasks(containers, gpu_id)

    # Update live fields
    hb["last_updated"] = now
    hb["heartbeat_owner"] = "benchmark"
    hb["temperature_c"] = stats.get("temperature_c", hb.get("temperature_c"))
    hb["power_draw_w"] = stats.get("power_draw_w", hb.get("power_draw_w"))
    hb["vram_used_mb"] = stats.get("vram_used_mb", hb.get("vram_used_mb"))
    hb["vram_total_mb"] = stats.get("vram_total_mb", hb.get("vram_total_mb"))
    hb["vram_percent"] = stats.get("vram_percent", hb.get("vram_percent"))
    hb["gpu_util_percent"] = stats.get("gpu_util_percent", hb.get("gpu_util_percent"))
    hb["clock_mhz"] = stats.get("clock_mhz", hb.get("clock_mhz"))
    hb["cpu_temp_c"] = cpu_temp
    hb["active_tasks"] = active_tasks
    hb["active_workers"] = len(active_tasks)
    hb["state"] = "benchmark" if active_tasks else "idle"

    with open(hb_path, "w") as f:
        json.dump(hb, f, indent=2)
    return True


def any_benchmark_running() -> bool:
    """Check if any benchmark containers are still running."""
    try:
        r = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=10,
        )
        for name in r.stdout.strip().splitlines():
            if any(kw in name for kw in ("bench-", "llama-gpu", "llama-split")):
                return True
    except Exception:
        pass
    return False


def main():
    parser = argparse.ArgumentParser(description="Benchmark heartbeat keeper")
    parser.add_argument("--interval", type=int, default=30, help="Update interval in seconds")
    parser.add_argument("--gpus-dir", type=str, default="/mnt/shared/gpus", help="GPU heartbeat directory")
    parser.add_argument("--no-auto-exit", action="store_true", help="Don't exit when no benchmarks running")
    args = parser.parse_args()

    gpus_dir = Path(args.gpus_dir)
    idle_count = 0
    print(f"Benchmark heartbeat keeper starting (interval={args.interval}s, gpus_dir={gpus_dir})")

    while True:
        gpu_stats = get_nvidia_smi()
        containers = get_benchmark_containers()
        cpu_temp = get_cpu_temp()

        updated = []
        for gpu_id in sorted(gpu_stats.keys()):
            if update_heartbeat(gpu_id, gpus_dir, gpu_stats, containers, cpu_temp):
                updated.append(gpu_id)

        if updated:
            active = sum(1 for c in containers.values()
                        if any(kw in c["name"] for kw in ("bench-",)))
            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Updated GPUs {updated}, {active} benchmark containers active")

        # Auto-exit when no benchmarks running
        if not args.no_auto_exit:
            if not any_benchmark_running():
                idle_count += 1
                if idle_count >= 3:  # 3 consecutive idle checks
                    print("No benchmark containers running for 3 cycles, exiting.")
                    # Clear heartbeat_owner on exit
                    for gpu_id in range(6):
                        hb_path = gpus_dir / f"gpu_{gpu_id}" / "heartbeat.json"
                        try:
                            with open(hb_path) as f:
                                hb = json.load(f)
                            if hb.get("heartbeat_owner") == "benchmark":
                                hb["heartbeat_owner"] = None
                                with open(hb_path, "w") as f:
                                    json.dump(hb, f, indent=2)
                        except Exception:
                            pass
                    break
            else:
                idle_count = 0

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
