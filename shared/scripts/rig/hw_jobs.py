"""Detached rig jobs: long hardware work (model loads) runs as its own systemd unit on the rig.

The rig manager on the Pi sends one command and gets a job id back at once, so a load never rides
on an ssh session (a dropped connection used to be able to kill a load half-way and free the
model-load lock early). The job itself waits its turn on the rig: loads run one at a time.
`rig observe` reports running and failed jobs; a finished load shows up as a ready profile, and
a successful transient unit disappears on exit.
"""

from __future__ import annotations

import pwd
import os
import re
import subprocess

UNIT_PREFIX = "rig-load-"
RUN_PREFIX = "rig-run-"     # long runs (campaigns, lanes) started from `job` plan items
LOG_TAIL_LINES = 15


def _run(cmd: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def unit_name(job_key: str, prefix: str = UNIT_PREFIX) -> str:
    return prefix + re.sub(r"[^A-Za-z0-9_.-]", "-", job_key)


def state(unit: str) -> dict:
    """ActiveState/SubState/Result of a job unit ("inactive" + "success" once a good job is gone)."""
    out = _run(["systemctl", "show", unit, "-p", "ActiveState", "-p", "SubState", "-p", "Result",
                "-p", "ExecMainStatus"]).stdout
    fields = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    return {"job": unit, "active_state": fields.get("ActiveState", "unknown"),
            "sub_state": fields.get("SubState", ""), "result": fields.get("Result", ""),
            "exit_status": fields.get("ExecMainStatus", "")}


def log_tail(unit: str, lines: int = LOG_TAIL_LINES) -> str:
    return _run(["sudo", "-n", "journalctl", "-u", unit, "-n", str(lines), "--no-pager", "-o", "cat"]).stdout.strip()


def launch(job_key: str, argv: list[str], caller: str, description: str, *, prefix: str = UNIT_PREFIX,
           cwd: str | None = None, env: dict[str, str] | None = None) -> dict:
    """Start argv as a transient unit owned by the calling user. A running job is not started twice."""
    unit = unit_name(job_key, prefix)
    current = state(unit)
    if current["active_state"] in ("active", "activating"):
        return {**current, "already_running": True}
    _run(["sudo", "-n", "systemctl", "reset-failed", unit])  # clear a previous failed run of this job
    user = pwd.getpwuid(os.getuid()).pw_name
    proc = _run(["sudo", "-n", "systemd-run", "--quiet", "--unit", unit, "--description", description,
                 "--uid", user, "--gid", user, "--setenv", f"HOME={os.path.expanduser('~')}",
                 "--setenv", f"RIG_CALLER={caller}",
                 *(["--working-directory", cwd] if cwd else []),
                 *[a for k, v in (env or {}).items() for a in ("--setenv", f"{k}={v}")], *argv])
    if proc.returncode != 0:
        raise RuntimeError(f"systemd-run {unit} failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return {**state(unit), "already_running": False}


def jobs() -> list[dict]:
    """Running and failed job units, with the log tail of each failed one."""
    out = _run(["systemctl", "list-units", "--all", "--plain", "--no-legend", "--type=service",
                f"{UNIT_PREFIX}*"]).stdout
    found = []
    for line in out.splitlines():
        parts = line.split()
        if not parts or not parts[0].startswith(UNIT_PREFIX):
            continue
        unit = parts[0].removesuffix(".service")
        st = state(unit)
        if st["active_state"] == "failed":
            st["log_tail"] = log_tail(unit)
        found.append(st)
    return found
