#!/usr/bin/env python3
"""Clear stale manual llama runtime state that can interfere with defaults."""

from __future__ import annotations

import json
import os
import socket
import subprocess
from pathlib import Path


SHARED_ROOT = Path("/mnt/shared")
SIGNALS_DIR = SHARED_ROOT / "signals" / "split_llm"
ALLOWED_CONTAINERS = {
    "llama-brain",
}
ALLOWED_PREFIXES = (
    "llama-worker-gpu-",
    "llama-split-",
)


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _docker_rows() -> list[dict[str, str]]:
    result = _run(
        [
            "docker",
            "ps",
            "-a",
            "--no-trunc",
            "--format",
            "{{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Command}}",
        ]
    )
    rows: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        parts = line.split("\t", 3)
        if len(parts) != 4:
            continue
        rows.append(
            {
                "id": parts[0].strip(),
                "name": parts[1].strip(),
                "status": parts[2].strip(),
                "command": parts[3].strip(),
            }
        )
    return rows


def _is_allowed_container(name: str) -> bool:
    if name in ALLOWED_CONTAINERS:
        return True
    return any(name.startswith(prefix) for prefix in ALLOWED_PREFIXES)


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _remove_container(container_id: str, name: str) -> None:
    result = _run(["docker", "rm", "-f", container_id])
    if result.returncode == 0:
        print(f"REMOVED_CONTAINER {name} ({container_id[:12]})")
    else:
        detail = (result.stderr or result.stdout).strip()
        print(f"FAILED_CONTAINER_REMOVE {name}: {detail}")


def cleanup_manual_llama_containers() -> None:
    for row in _docker_rows():
        name = row["name"]
        command = row["command"]
        if "llama-server" not in command:
            continue
        if _is_allowed_container(name):
            continue
        _remove_container(row["id"], name)


def cleanup_stale_split_signals() -> None:
    if not SIGNALS_DIR.exists():
        return

    for signal_file in sorted(SIGNALS_DIR.glob("pair_*.json")):
        try:
            data = json.loads(signal_file.read_text())
        except Exception as exc:
            print(f"FAILED_SIGNAL_READ {signal_file}: {exc}")
            continue

        group_id = str(data.get("group_id") or signal_file.stem).strip()
        container_name = f"llama-split-{group_id}"
        port = int(data.get("port") or 0)
        status = str(data.get("status") or "").strip()
        if status not in {"loading", "ready"}:
            continue

        inspect = _run(["docker", "inspect", "-f", "{{.State.Running}}", container_name])
        container_running = inspect.returncode == 0 and "true" in inspect.stdout.lower()
        listener_up = port > 0 and _port_open(port)
        if container_running or listener_up:
            continue

        signal_file.unlink(missing_ok=True)
        print(
            f"REMOVED_STALE_SPLIT_SIGNAL {signal_file.name} "
            f"status={status} port={port} container={container_name}"
        )


def main() -> int:
    if os.geteuid() != 0 and not shutil_which("docker"):
        print("docker not available")
        return 1

    cleanup_manual_llama_containers()
    cleanup_stale_split_signals()
    return 0


def shutil_which(name: str) -> str | None:
    for path in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(path) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
