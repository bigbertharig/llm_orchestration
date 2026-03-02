#!/usr/bin/env python3
"""Clear worker/split Ollama runtimes on the GPU rig.

Leaves the brain Ollama service intact (e.g. port 11434) and targets only
worker single-GPU ports from config plus split runtime ports from models catalog.
Also clears split/model-load signal artifacts that commonly wedge future loads.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = BASE_DIR / "shared" / "agents" / "config.json"
DEFAULT_CATALOG = BASE_DIR / "shared" / "agents" / "models.catalog.json"


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return data


def _collect_worker_ports(config: dict) -> list[int]:
    ports: set[int] = set()
    for gpu in config.get("gpus", []) or []:
        if not isinstance(gpu, dict):
            continue
        port = gpu.get("port")
        try:
            if port is not None:
                ports.add(int(port))
        except Exception:
            continue
    return sorted(ports)


def _collect_split_ports(catalog: dict) -> list[int]:
    ports: set[int] = set()
    for model in catalog.get("models", []) or []:
        if not isinstance(model, dict):
            continue
        for group in model.get("split_groups", []) or []:
            if not isinstance(group, dict):
                continue
            port = group.get("port")
            try:
                if port is not None:
                    ports.add(int(port))
            except Exception:
                continue
    return sorted(ports)


def _run_remote(script: str, timeout_s: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "gpu", "bash", "-s"],
        input=script,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Clear worker/split Ollama runtimes on GPU rig")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to shared/agents/config.json")
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG), help="Path to shared/agents/models.catalog.json")
    parser.add_argument("--timeout", type=int, default=90, help="SSH timeout seconds")
    parser.add_argument("--no-clear-signals", action="store_true", help="Do not clear split/model-load signal artifacts")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary on final line")
    args = parser.parse_args()

    config = _load_json(Path(args.config))
    catalog = _load_json(Path(args.catalog))
    worker_ports = _collect_worker_ports(config)
    split_ports = _collect_split_ports(catalog)
    target_ports = sorted(set(worker_ports + split_ports))

    if not target_ports:
        raise RuntimeError("No target worker/split ports found in config/catalog")

    ports_str = " ".join(str(p) for p in target_ports)
    clear_signals = "1" if not args.no_clear_signals else "0"
    remote_script = f"""\
set +e
worker_port_killed=0
split_port_killed=0
orphan_ollama_killed=0
orphan_runner_killed=0
signals_removed=0
target_port_listeners_remaining=0

brain_pid="$(ss -ltnp 2>/dev/null | grep ':11434\\b' | sed -n 's/.*pid=\\([0-9]\\+\\).*/\\1/p' | head -n1 || true)"

is_split_port() {{
  case "$1" in
{"".join([f"    {p}) return 0 ;;\n" for p in split_ports])}    *) return 1 ;;
  esac
}}

kill_pid_hard() {{
  pid="$1"
  [ -n "$pid" ] || return 0
  kill "$pid" 2>/dev/null || true
  sleep 0.2
  kill -0 "$pid" 2>/dev/null || return 0
  kill -9 "$pid" 2>/dev/null || true
}}

for p in {ports_str}; do
  pids="$(ss -ltnp 2>/dev/null | grep ":${{p}}\\b" | sed -n 's/.*pid=\\([0-9]\\+\\).*/\\1/p' | sort -u || true)"
  [ -n "$pids" ] || continue
  for pid in $pids; do
    kill_pid_hard "$pid"
    if is_split_port "$p"; then
      split_port_killed=$((split_port_killed + 1))
    else
      worker_port_killed=$((worker_port_killed + 1))
    fi
  done
done

# Kill all non-brain ollama serve processes (workers/split runtimes), even if still parented.
for pid in $(ps -eo pid=,args= | awk '$0 ~ /ollama serve/ {{print $1}}'); do
  [ -n "$pid" ] || continue
  if [ -n "$brain_pid" ] && [ "$pid" = "$brain_pid" ]; then
    continue
  fi
  kill_pid_hard "$pid"
  orphan_ollama_killed=$((orphan_ollama_killed + 1))
done

# Kill all non-brain ollama runners (brain runner is direct child of brain serve).
for pid in $(ps -eo pid=,ppid=,args= | awk '$0 ~ /ollama runner/ {{print $1":"$2}}'); do
  [ -n "$pid" ] || continue
  runner_pid="${{pid%%:*}}"
  runner_ppid="${{pid##*:}}"
  if [ -n "$brain_pid" ] && [ "$runner_ppid" = "$brain_pid" ]; then
    continue
  fi
  kill_pid_hard "$runner_pid"
  orphan_runner_killed=$((orphan_runner_killed + 1))
done

# Reap any orphan runners left behind by dead GPU/split runtimes (second pass).
for pid in $(ps -eo pid=,ppid=,args= | awk '$2==1 && $0 ~ /ollama runner/ {{print $1}}'); do
  [ -n "$pid" ] || continue
  kill_pid_hard "$pid"
  orphan_runner_killed=$((orphan_runner_killed + 1))
done

if [ "{clear_signals}" = "1" ]; then
  for f in /mnt/shared/signals/model_load.global.json /mnt/shared/signals/model_load.global.json.lock; do
    if [ -e "$f" ]; then rm -f "$f" && signals_removed=$((signals_removed + 1)); fi
  done
  for f in /mnt/shared/signals/split_llm/pair_*.json /mnt/shared/signals/split_llm/pair_*.runtime_owner.json /mnt/shared/signals/split_llm/*.lock; do
    [ -e "$f" ] || continue
    rm -f "$f" && signals_removed=$((signals_removed + 1))
  done
fi

for p in {ports_str}; do
  if ss -ltnp 2>/dev/null | grep -q ":${{p}}\\b"; then
    target_port_listeners_remaining=$((target_port_listeners_remaining + 1))
  fi
done

printf '{{"worker_ports_killed":%s,"split_ports_killed":%s,"non_brain_ollama_serve_killed":%s,"non_brain_ollama_runner_killed":%s,"signals_removed":%s,"target_port_listeners_remaining":%s}}\\n' \\
  "$worker_port_killed" "$split_port_killed" "$orphan_ollama_killed" "$orphan_runner_killed" "$signals_removed" "$target_port_listeners_remaining"
"""

    print(f"Target worker ports: {worker_ports}")
    print(f"Target split ports: {split_ports}")
    print("Clearing worker/split Ollama runtimes on gpu host...")
    proc = _run_remote(remote_script, timeout_s=max(10, int(args.timeout)))

    stdout = (proc.stdout or "").rstrip()
    stderr = (proc.stderr or "").rstrip()
    summary: dict[str, object] = {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "worker_ports": worker_ports,
        "split_ports": split_ports,
        "clear_signals": not args.no_clear_signals,
    }
    if stdout:
        lines = stdout.splitlines()
        last = lines[-1].strip()
        try:
            parsed = json.loads(last)
            if isinstance(parsed, dict):
                summary.update(parsed)
        except Exception:
            pass
    if stderr:
        summary["stderr"] = stderr

    if not args.json:
        if stdout:
            print(stdout)
        if stderr:
            print(stderr, file=sys.stderr)
        print("Clear Ollama summary:", json.dumps(summary, indent=2))
    else:
        print(json.dumps(summary))

    return 0 if proc.returncode == 0 else proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
