#!/usr/bin/env python3
"""
Shared hardware discovery module.

Scans GPUs, Ollama, and system resources. Used by both setup.py (interactive)
and startup.py (non-interactive). Stdlib only — no pip dependencies.

Usage:
    from hardware import scan_gpus, scan_ollama, scan_system, suggest_assignment
"""

import json
import os
import platform
import subprocess
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional


def scan_gpus():
    # type: () -> List[Dict[str, Any]]
    """Scan all NVIDIA GPUs via nvidia-smi.

    Returns list of dicts sorted by GPU index:
        [{index, name, vram_mb, temp_c, power_w, power_limit_w, clock_mhz, throttle}]

    Returns empty list if nvidia-smi is not available (e.g., RPi, CPU-only).
    """
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=index,name,temperature.gpu,power.draw,power.limit,'
                'memory.total,clocks.sm,clocks_throttle_reasons.active',
                '--format=csv,noheader,nounits'
            ],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return []
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    gpus = []
    for line in result.stdout.strip().split('\n'):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 8:
            continue

        try:
            throttle_raw = parts[7]
            throttle = throttle_raw if throttle_raw != "0x0000000000000000" else "None"

            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "vram_mb": int(float(parts[5])),
                "temp_c": int(parts[2]),
                "power_w": float(parts[3]),
                "power_limit_w": float(parts[4]),
                "clock_mhz": int(parts[6]),
                "throttle": throttle,
            })
        except (ValueError, IndexError):
            continue

    gpus.sort(key=lambda g: g["index"])
    return gpus


def scan_ollama(host="http://localhost:11434"):
    # type: (str) -> Dict[str, Any]
    """Scan Ollama status: running, available models, loaded models.

    Returns:
        {running, host, available_models: [{name, size_mb}],
         loaded_models: [{name, vram_mb}]}
    """
    result = {
        "running": False,
        "host": host,
        "available_models": [],
        "loaded_models": [],
    }

    # Check if Ollama is running via /api/tags
    try:
        req = urllib.request.Request(f"{host}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            result["running"] = True
            for model in data.get("models", []):
                name = model.get("name", "")
                size_bytes = model.get("size", 0)
                size_mb = int(size_bytes / (1024 * 1024))
                result["available_models"].append({
                    "name": name,
                    "size_mb": size_mb,
                })
            # Sort largest first for suggestion logic
            result["available_models"].sort(
                key=lambda m: m["size_mb"], reverse=True
            )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return result

    # Get currently loaded models via /api/ps
    try:
        req = urllib.request.Request(f"{host}/api/ps")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            for model in data.get("models", []):
                name = model.get("name", "")
                vram_bytes = model.get("size_vram", 0)
                vram_mb = int(vram_bytes / (1024 * 1024))
                result["loaded_models"].append({
                    "name": name,
                    "vram_mb": vram_mb,
                })
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        pass

    return result


def scan_system():
    # type: () -> Dict[str, Any]
    """Scan basic system resources.

    Returns:
        {hostname, cpu_cores, ram_total_mb, ram_available_mb}
    """
    info = {
        "hostname": platform.node(),
        "cpu_cores": os.cpu_count() or 1,
        "ram_total_mb": 0,
        "ram_available_mb": 0,
    }

    # Parse /proc/meminfo (Linux only)
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    info["ram_total_mb"] = int(line.split()[1]) // 1024
                elif line.startswith("MemAvailable:"):
                    info["ram_available_mb"] = int(line.split()[1]) // 1024
    except (FileNotFoundError, ValueError):
        pass

    return info


def _find_best_model(models, max_vram_mb):
    # type: (List[Dict], int) -> Optional[str]
    """Pick the largest available Ollama model that fits in the given VRAM.

    Uses a 0.85 safety factor for runtime overhead.
    Returns model name or None if nothing fits.
    """
    if not models:
        return None

    usable_vram = int(max_vram_mb * 0.85)
    fitting = [m for m in models if m["size_mb"] <= usable_vram]
    if not fitting:
        return None

    fitting.sort(key=lambda m: m["size_mb"], reverse=True)
    return fitting[0]["name"]


def suggest_assignment(gpus, ollama, preferences=None):
    # type: (List[Dict], Dict, Optional[Dict]) -> Dict[str, Any]
    """Suggest brain/worker GPU assignment based on discovered hardware.

    Assignment modes:
        0 GPUs -> cpu_only:    brain on CPU, no workers
        1 GPU  -> single_gpu:  brain gets it, no workers
        2 GPUs -> minimal:     biggest -> brain, other -> worker
        3+ GPUs -> standard:   see below

    For 3+ GPUs:
        1. Sort by VRAM descending
        2. Brain: if 2+ identical GPUs share max VRAM, pair them. Otherwise single largest.
        3. Workers: all remaining GPUs
        4. Ports: auto-assigned from 11435

    Args:
        gpus: Output from scan_gpus()
        ollama: Output from scan_ollama()
        preferences: Optional overrides (brain_model, worker_model, worker_mode)

    Returns config-ready structure with _discovery_mode, brain, gpus, worker_mode.
    """
    prefs = preferences or {}
    available_models = ollama.get("available_models", [])

    # --- 0 GPUs: CPU-only mode ---
    if not gpus:
        brain_model = prefs.get("brain_model") or _find_best_model(available_models, 0) or ""
        return {
            "_discovery_mode": "cpu_only",
            "brain": {
                "name": "brain",
                "model": brain_model,
                "gpus": [],
            },
            "gpus": [],
            "worker_mode": prefs.get("worker_mode", "cold"),
        }

    # Sort by VRAM descending, then by index for stability
    sorted_gpus = sorted(gpus, key=lambda g: (-g["vram_mb"], g["index"]))

    # --- 1 GPU: single-GPU mode ---
    if len(gpus) == 1:
        gpu = sorted_gpus[0]
        brain_model = prefs.get("brain_model") or _find_best_model(available_models, gpu["vram_mb"]) or ""
        return {
            "_discovery_mode": "single_gpu",
            "brain": {
                "name": "brain",
                "model": brain_model,
                "gpus": [gpu["index"]],
            },
            "gpus": [],
            "worker_mode": prefs.get("worker_mode", "cold"),
        }

    # --- 2 GPUs: minimal mode ---
    if len(gpus) == 2:
        brain_gpu = sorted_gpus[0]
        worker_gpu = sorted_gpus[1]
        brain_model = prefs.get("brain_model") or _find_best_model(available_models, brain_gpu["vram_mb"]) or ""
        worker_model = prefs.get("worker_model") or _find_best_model(available_models, worker_gpu["vram_mb"]) or ""
        return {
            "_discovery_mode": "minimal",
            "brain": {
                "name": "brain",
                "model": brain_model,
                "gpus": [brain_gpu["index"]],
            },
            "gpus": [
                {
                    "name": "gpu-%d" % worker_gpu["index"],
                    "id": worker_gpu["index"],
                    "vram_mb": worker_gpu["vram_mb"],
                    "model": worker_model,
                    "port": 11435,
                    "permissions": "worker.json",
                },
            ],
            "worker_mode": prefs.get("worker_mode", "hot"),
        }

    # --- 3+ GPUs: standard mode ---
    # Brain: check if 2+ GPUs share the max VRAM (identical cards -> pair them)
    max_vram = sorted_gpus[0]["vram_mb"]
    top_gpus = [g for g in sorted_gpus if g["vram_mb"] == max_vram]

    if len(top_gpus) >= 2 and top_gpus[0]["name"] == top_gpus[1]["name"]:
        # Pair the first two identical top-VRAM GPUs for brain
        brain_gpu_indices = [top_gpus[0]["index"], top_gpus[1]["index"]]
        brain_vram = max_vram  # Each card's VRAM (Ollama handles multi-GPU)
        worker_gpus = [g for g in sorted_gpus if g["index"] not in brain_gpu_indices]
    else:
        # Single largest GPU for brain
        brain_gpu_indices = [sorted_gpus[0]["index"]]
        brain_vram = max_vram
        worker_gpus = sorted_gpus[1:]

    brain_model = prefs.get("brain_model") or _find_best_model(available_models, brain_vram) or ""
    worker_model_pref = prefs.get("worker_model")

    # Build worker entries with auto-assigned ports
    base_port = 11435
    worker_entries = []
    for i, wgpu in enumerate(worker_gpus):
        model = worker_model_pref or _find_best_model(available_models, wgpu["vram_mb"]) or ""
        worker_entries.append({
            "name": "gpu-%d" % wgpu["index"],
            "id": wgpu["index"],
            "vram_mb": wgpu["vram_mb"],
            "model": model,
            "port": base_port + i,
            "permissions": "worker.json",
        })

    return {
        "_discovery_mode": "standard",
        "brain": {
            "name": "brain",
            "model": brain_model,
            "gpus": brain_gpu_indices,
        },
        "gpus": worker_entries,
        "worker_mode": prefs.get("worker_mode", "hot"),
    }


def format_scan_report(gpus, ollama, system):
    # type: (List[Dict], Dict, Dict) -> str
    """Format a human-readable hardware scan report for display."""
    lines = []
    lines.append("===== Hardware Scan =====")

    # System
    lines.append("Host: %s (%d cores, %d MB RAM)" % (
        system['hostname'], system['cpu_cores'], system['ram_total_mb']))

    # GPUs
    if gpus:
        lines.append("\nGPUs found: %d" % len(gpus))
        for g in gpus:
            lines.append("  GPU %d: %s [%d MB VRAM] %dC %.0fW" % (
                g['index'], g['name'], g['vram_mb'], g['temp_c'], g['power_w']))
    else:
        lines.append("\nGPUs found: 0 (CPU-only mode)")

    # Ollama
    if ollama["running"]:
        lines.append("\nOllama: running (%s)" % ollama['host'])
        if ollama["available_models"]:
            models_str = ", ".join(
                "%s (%d MB)" % (m['name'], m['size_mb'])
                for m in ollama["available_models"]
            )
            lines.append("  Available models: %s" % models_str)
        else:
            lines.append("  Available models: none")

        if ollama["loaded_models"]:
            loaded_str = ", ".join(
                "%s (%d MB VRAM)" % (m['name'], m['vram_mb'])
                for m in ollama["loaded_models"]
            )
            lines.append("  Loaded models: %s" % loaded_str)
    else:
        lines.append("\nOllama: not running (%s)" % ollama['host'])

    return "\n".join(lines)


def format_suggestion_report(suggestion):
    # type: (Dict) -> str
    """Format a human-readable suggestion report for display."""
    lines = []
    lines.append("===== Suggested Configuration =====")
    lines.append("Mode: %s" % suggestion['_discovery_mode'])

    brain = suggestion["brain"]
    gpu_str = ", ".join(str(g) for g in brain["gpus"]) if brain["gpus"] else "CPU"
    lines.append("  Brain: GPU %s -> %s" % (gpu_str, brain['model'] or '(no model fits)'))

    workers = suggestion["gpus"]
    if workers:
        for w in workers:
            lines.append("  Worker: GPU %d (%d MB) -> %s" % (
                w['id'], w['vram_mb'], w['model'] or '(no model fits)'))
    else:
        lines.append("  Workers: none")

    lines.append("  Worker mode: %s" % suggestion['worker_mode'])

    return "\n".join(lines)
