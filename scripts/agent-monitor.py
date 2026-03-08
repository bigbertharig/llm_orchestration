#!/usr/bin/env python3
"""
Agent Monitor Dashboard
Shows GPU status, runtime models, and task queue.

Usage:
    python agent-monitor.py           # Live dashboard (updates every 2s)
    python agent-monitor.py --once    # Single snapshot
"""

import subprocess
import json
import time
import argparse
import os

def get_gpu_stats():
    """Get GPU utilization, memory, and temperature."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = [p.strip() for p in line.split(',')]
                gpus.append({
                    'id': int(parts[0]),
                    'name': parts[1],
                    'util': int(parts[2]) if parts[2] != '[N/A]' else 0,
                    'mem_used': int(float(parts[3])),
                    'mem_total': int(float(parts[4])),
                    'temp': int(parts[5]) if parts[5] != '[N/A]' else 0,
                    'power': float(parts[6]) if parts[6] != '[N/A]' else 0,
                })
        return gpus
    except Exception as e:
        return []

def get_runtime_status():
    """Get currently loaded llama runtime models."""
    try:
        result = subprocess.run(
            ["curl", "-sS", "http://localhost:11434/v1/models"],
            capture_output=True, text=True, timeout=5,
        )
        models = []
        payload = json.loads(result.stdout or "{}")
        for item in payload.get("data", []) if isinstance(payload, dict) else []:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("id") or "").strip()
            if model_id:
                models.append({
                    'name': model_id,
                    'id': model_id,
                    'size': 'loaded',
                })
        return models
    except Exception:
        return []

def get_runtime_models():
    """Get runtime models visible on the default port."""
    return [item['name'] for item in get_runtime_status()]

def get_task_queue():
    """Read task queue from file if exists."""
    queue_file = os.path.expanduser('~/tasks/queue.json')
    if os.path.exists(queue_file):
        try:
            with open(queue_file) as f:
                return json.load(f)
        except:
            pass
    return []

def bar(pct, width=20):
    """Create a progress bar."""
    filled = int(width * pct / 100)
    return '█' * filled + '░' * (width - filled)

def clear_screen():
    print('\033[2J\033[H', end='')

def print_dashboard(gpus, running_models, available_models, tasks):
    """Print the monitoring dashboard."""
    clear_screen()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║                    AGENT MONITOR DASHBOARD                       ║")
    print("╠══════════════════════════════════════════════════════════════════╣")

    # GPU Section
    print("║ GPUs                                                             ║")
    print("╟──────────────────────────────────────────────────────────────────╢")
    for gpu in gpus:
        mem_pct = int(100 * gpu['mem_used'] / gpu['mem_total'])
        status = "IDLE" if gpu['util'] < 5 else "BUSY"
        color = '\033[92m' if status == "IDLE" else '\033[93m'  # Green/Yellow
        reset = '\033[0m'
        print(f"║ GPU {gpu['id']}: {bar(gpu['util'])} {gpu['util']:3d}% │ "
              f"Mem: {gpu['mem_used']:4d}/{gpu['mem_total']}MB │ "
              f"{gpu['temp']:2d}°C │ {color}{status}{reset} ║")

    # Runtime Section
    print("╟──────────────────────────────────────────────────────────────────╢")
    print("║ Runtime Models                                                   ║")
    print("╟──────────────────────────────────────────────────────────────────╢")
    if running_models:
        for m in running_models:
            print(f"║  🟢 {m['name']:<20} ({m['size']:<12}) LOADED            ║")
    else:
        print("║  (no models currently loaded)                                   ║")
    print("║  Available:", ', '.join(available_models[:4]) + ("..." if len(available_models) > 4 else ""))

    # Task Queue Section
    print("╟──────────────────────────────────────────────────────────────────╢")
    print("║ Task Queue                                                       ║")
    print("╟──────────────────────────────────────────────────────────────────╢")
    if tasks:
        for i, task in enumerate(tasks[:5]):
            status_icon = {'pending': '⏳', 'running': '🔄', 'done': '✅', 'error': '❌'}.get(task.get('status', 'pending'), '?')
            print(f"║  {status_icon} {task.get('type', 'task'):<12} {task.get('description', '')[:40]:<40} ║")
        if len(tasks) > 5:
            print(f"║  ... and {len(tasks) - 5} more tasks                                      ║")
    else:
        print("║  (no tasks in queue - ~/tasks/queue.json)                       ║")

    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f" Last updated: {time.strftime('%H:%M:%S')} │ Press Ctrl+C to exit")

def main():
    parser = argparse.ArgumentParser(description='Agent Monitor Dashboard')
    parser.add_argument('--once', action='store_true', help='Single snapshot, no loop')
    parser.add_argument('--interval', type=float, default=2.0, help='Update interval in seconds')
    args = parser.parse_args()

    try:
        while True:
            gpus = get_gpu_stats()
            running = get_runtime_status()
            available = get_runtime_models()
            tasks = get_task_queue()

            print_dashboard(gpus, running, available, tasks)

            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()
