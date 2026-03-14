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
    """Read CPU temperature from thermal zone."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return round(int(f.read().strip()) / 1000, 1)
    except Exception:
        return None


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
    # Fallback: check -p flag from container name patterns
    # llama-gpu2 -> port 11436, llama-split-test -> port 11437, etc.
    # Better: check published ports
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


def get_benchmark_progress(container_name: str) -> dict | None:
    """Try to extract progress info from container logs."""
    try:
        r = subprocess.run(
            ["docker", "logs", "--tail", "5", container_name],
            capture_output=True, text=True, timeout=5,
        )
        output = r.stdout + r.stderr
        # Look for progress patterns like "43%|" or "459/2700"
        progress = {}
        for line in reversed(output.splitlines()):
            # Percentage pattern
            pct_match = re.search(r"(\d+)%\|", line)
            if pct_match and "percent" not in progress:
                progress["percent"] = int(pct_match.group(1))
            # Fraction pattern
            frac_match = re.search(r"(\d+)/(\d+)", line)
            if frac_match and "current" not in progress:
                progress["current"] = int(frac_match.group(1))
                progress["total"] = int(frac_match.group(2))
            # Running task pattern
            task_match = re.search(r"Running (?:task: )?(\w+)", line)
            if task_match and "current_task" not in progress:
                progress["current_task"] = task_match.group(1)
            if progress:
                break
        # Check for completion
        if "COMPLETE" in output:
            progress["status"] = "completed"
        return progress if progress else None
    except Exception:
        return None


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
