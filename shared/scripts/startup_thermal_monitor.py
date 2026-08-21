#!/usr/bin/env python3
"""Record CPU and GPU telemetry during rig startup and model loading."""

from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path
import signal
import subprocess
import time


def read_int(path: str) -> int | None:
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def cpu_temperatures() -> tuple[float | None, float | None]:
    package = None
    cores: list[float] = []
    for hwmon in glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            if Path(hwmon, "name").read_text().strip() != "coretemp":
                continue
        except OSError:
            continue
        for input_path in glob.glob(f"{hwmon}/temp*_input"):
            value = read_int(input_path)
            if value is None:
                continue
            temp_c = value / 1000.0
            label_path = input_path.replace("_input", "_label")
            try:
                label = Path(label_path).read_text().strip()
            except OSError:
                label = ""
            if label.startswith("Package"):
                package = temp_c
            elif label.startswith("Core"):
                cores.append(temp_c)
    return package, max(cores) if cores else None


def read_cpu_ticks() -> tuple[int, int]:
    fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
    ticks = [int(value) for value in fields]
    idle = ticks[3] + (ticks[4] if len(ticks) > 4 else 0)
    return sum(ticks), idle


def gpu_telemetry() -> tuple[list[float], list[float], list[int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=power.draw,temperature.gpu,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=3, check=True)
    except (OSError, subprocess.SubprocessError):
        return [], [], []

    powers: list[float] = []
    temperatures: list[float] = []
    utilizations: list[int] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            powers.append(float(parts[0]))
            temperatures.append(float(parts[1]))
            utilizations.append(int(parts[2]))
        except ValueError:
            continue
    return powers, temperatures, utilizations


def model_server_count() -> int:
    count = 0
    try:
        for cmdline_path in glob.glob("/proc/[0-9]*/cmdline"):
            try:
                arguments = [
                    value.decode(errors="replace")
                    for value in Path(cmdline_path).read_bytes().split(b"\0")
                    if value
                ]
            except OSError:
                continue
            if not arguments:
                continue
            executable = os.path.basename(arguments[0])
            if executable == "llama-server" or (executable == "ollama" and "runner" in arguments[1:]):
                count += 1
    except OSError:
        return -1
    return count


def process_snapshot() -> str:
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid,ppid,state,%cpu,%mem,comm,args", "--sort=-%cpu"],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
        return "\n".join(result.stdout.splitlines()[:16])
    except (OSError, subprocess.SubprocessError):
        return "process snapshot unavailable"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--duration", type=int, default=1800)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--abort-cpu-c", type=float, default=0.0)
    parser.add_argument("--abort-pid", type=int)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    events_path = args.output.with_suffix(".events.log")
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    started = time.time()
    last_total, last_idle = read_cpu_ticks()
    last_event = 0.0
    abort_sent = False

    fieldnames = [
        "timestamp",
        "elapsed_s",
        "boot_id",
        "cpu_package_c",
        "cpu_core_max_c",
        "cpu_util_pct",
        "load1",
        "total_gpu_power_w",
        "max_gpu_temp_c",
        "total_gpu_util_pct",
        "gpu_power_w",
        "gpu_temp_c",
        "gpu_util_pct",
        "model_server_count",
    ]

    with args.output.open("w", newline="", buffering=1) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        while time.time() - started <= args.duration:
            sample_time = time.time()
            package_c, core_max_c = cpu_temperatures()
            total_ticks, idle_ticks = read_cpu_ticks()
            tick_delta = total_ticks - last_total
            idle_delta = idle_ticks - last_idle
            cpu_util = 0.0 if tick_delta <= 0 else 100.0 * (tick_delta - idle_delta) / tick_delta
            last_total, last_idle = total_ticks, idle_ticks
            powers, gpu_temps, gpu_utils = gpu_telemetry()
            total_power = sum(powers) if powers else None

            writer.writerow(
                {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(sample_time)),
                    "elapsed_s": f"{sample_time - started:.1f}",
                    "boot_id": boot_id,
                    "cpu_package_c": "" if package_c is None else f"{package_c:.1f}",
                    "cpu_core_max_c": "" if core_max_c is None else f"{core_max_c:.1f}",
                    "cpu_util_pct": f"{cpu_util:.1f}",
                    "load1": f"{os.getloadavg()[0]:.2f}",
                    "total_gpu_power_w": "" if total_power is None else f"{total_power:.1f}",
                    "max_gpu_temp_c": "" if not gpu_temps else f"{max(gpu_temps):.1f}",
                    "total_gpu_util_pct": "" if not gpu_utils else str(sum(gpu_utils)),
                    "gpu_power_w": "|".join(f"{value:.1f}" for value in powers),
                    "gpu_temp_c": "|".join(f"{value:.1f}" for value in gpu_temps),
                    "gpu_util_pct": "|".join(str(value) for value in gpu_utils),
                    "model_server_count": model_server_count(),
                }
            )

            critical = package_c is not None and package_c >= 90
            high_power = total_power is not None and total_power >= 850
            if (critical or high_power) and sample_time - last_event >= 10:
                with events_path.open("a", buffering=1) as events:
                    events.write(
                        f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} "
                        f"cpu_package_c={package_c} total_gpu_power_w={total_power}\n"
                    )
                    events.write(process_snapshot() + "\n\n")
                last_event = sample_time

            if (
                not abort_sent
                and args.abort_pid
                and args.abort_cpu_c > 0
                and package_c is not None
                and package_c >= args.abort_cpu_c
            ):
                with events_path.open("a", buffering=1) as events:
                    events.write(
                        f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} "
                        f"THERMAL_ABORT cpu_package_c={package_c} "
                        f"threshold_c={args.abort_cpu_c} pid={args.abort_pid}\n"
                    )
                    events.write(process_snapshot() + "\n\n")
                try:
                    os.kill(args.abort_pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                abort_sent = True

            time.sleep(max(0.1, args.interval - (time.time() - sample_time)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
