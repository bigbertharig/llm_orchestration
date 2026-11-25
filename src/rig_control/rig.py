from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import time
from uuid import uuid4
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

# Exit codes of the rig mode selector (llm_orchestration shared/scripts/rig/rig.py).
RIG_OK = 0
RIG_FAILED = 1
RIG_BUSY = 2
RIG_LOCKED = 3
RIG_NO_CAPACITY = 4
# ssh's own exit code when it cannot reach the rig (RIG_COMMAND="ssh ... rig" on the Pi).
SSH_UNREACHABLE = 255

LOCKED_RETRY_SECONDS = 3.0


class RigUnavailable(RuntimeError):
    """The rig cannot serve the profile right now. Carries the HTTP status to return."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class Rig:
    """Async wrapper around `rig status --json` / `rig start remote <id>`.

    The command may run the rig selector locally or over ssh. Over ssh, killing the client can
    kill the remote `rig start` mid-load (which frees the rig's model-load lock early), so a
    start is never killed: past its deadline the gateway stops waiting and lets it finish.
    """

    def __init__(self, command: tuple[str, ...], start_timeout_seconds: float, *, wire_version: str | None = None):
        self.wire_version = wire_version
        if wire_version not in (None, "rig-cli/1"):
            raise ValueError("unsupported rig wire version")
        self.command = command
        self.start_timeout_seconds = start_timeout_seconds
        self._inflight: dict[str, asyncio.Task] = {}
        self._loading_since: dict[str, float] = {}
        self._orphans: set[asyncio.Task] = set()
        # ssh joins its arguments into one string for the remote shell, which splits it again.
        # Matches ssh, ssh.exe and a full path to either (Windows OpenSSH).
        # Split on both separators: a Windows path reaches this code on Linux in tests and on the Pi.
        self._over_ssh = bool(command) and os.path.splitext(re.split(r"[\\/]", command[0])[-1])[0].lower() == "ssh"

    def argv(self, *args: str) -> list[str]:
        """The full command line; over ssh each argument is shell-quoted so it arrives intact."""
        return [*self.command, *(shlex.quote(a) for a in args)] if self._over_ssh else [*self.command, *args]

    async def _run(self, *args: str, timeout: float, kill_on_timeout: bool = True) -> tuple[int, str]:
        wire_request = None
        dispatched = args
        if self.wire_version:
            wire_request = {'schema': self.wire_version, 'request_id': uuid4().hex, 'arguments': list(args)}
            dispatched = ('wire', '--request', json.dumps(wire_request))
            # Never kill a remotely dispatched write merely because the caller stopped waiting.
            if args and args[0] not in ('status','observe','model-fit','schedule-preview','inspect'):
                kill_on_timeout = False
        try:
            proc = await asyncio.create_subprocess_exec(
                *self.argv(*dispatched),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            )
        except OSError as exc:
            raise RigUnavailable(502, f"cannot run rig command {self.command}: {exc}") from exc
        done = asyncio.ensure_future(proc.communicate())
        try:
            out, _ = await asyncio.wait_for(asyncio.shield(done), timeout)
        except TimeoutError as exc:
            if kill_on_timeout:
                proc.kill()
                await done
            else:
                self._orphans.add(done)  # keep a reference until it exits; output is dropped
                done.add_done_callback(self._orphans.discard)
            raise RigUnavailable(504, f"rig {args[0] if args else 'operation'} timed out after {timeout:.0f}s" +
                                 (f"; reconcile request_id={wire_request['request_id']}" if wire_request else "")) from exc
        text = out.decode("utf-8", "replace").strip()
        if proc.returncode == SSH_UNREACHABLE:
            raise RigUnavailable(502, f"rig unreachable over ssh: {text[-500:] or 'no output'}")
        if wire_request:
            try:
                packet = json.loads(text)
            except ValueError as exc:
                raise RigUnavailable(502, 'Runtime has no compatible rig-cli/1 response; upgrade the configured runtime') from exc
            if (not isinstance(packet, dict) or packet.get('schema') != self.wire_version
                    or packet.get('request_id') != wire_request['request_id']
                    or packet.get('arguments') != list(args)):
                raise RigUnavailable(502, 'Runtime wire version or request identity mismatch')
            if type(packet.get('exit_code')) is not int or not isinstance(packet.get('output'), str):
                raise RigUnavailable(409, 'Runtime dispatch is unresolved: ' + str(packet.get('kind', 'invalid_receipt')))
            if packet.get('ok') is not (packet['exit_code'] == 0) or proc.returncode != packet['exit_code']:
                raise RigUnavailable(502, 'Runtime returned inconsistent exit evidence')
            return packet['exit_code'], packet['output'].strip()
        return proc.returncode, text

    async def shared_kv(self, action: str, *, model: str | None = None, chat_id: str | None = None,
                        message: str | None = None, max_tokens: int | None = None,
                        context_tokens: int | None = None,
                        main_context_cap: int | None = None) -> dict:
        if action not in ("status", "start", "attach", "detach", "turn", "stop"):
            raise ValueError("unsupported shared chat action")
        args = ["shared-kv", action]
        if model is not None: args += ["--model", model]
        if chat_id is not None: args += ["--chat-id", chat_id]
        if message is not None: args += ["--message", message]
        if max_tokens is not None: args += ["--max-tokens", str(max_tokens)]
        if context_tokens is not None: args += ["--context-cap", str(context_tokens)]
        if main_context_cap is not None: args += ["--main-context-cap", str(main_context_cap)]
        code, out = await self._run(*args, timeout=900 if action in ("start", "turn") else 180)
        if code != RIG_OK:
            raise RigUnavailable(503 if action == "attach" else 502,
                                 f"shared chat {action} failed: {out[-500:]}")
        try:
            return json.loads(out.splitlines()[-1])
        except (ValueError, IndexError) as exc:
            raise RigUnavailable(502, f"shared chat {action} returned invalid JSON") from exc

    async def status(self) -> dict:
        code, out = await self._run("status", "--json", timeout=30)
        if code != RIG_OK:
            raise RigUnavailable(502, f"rig status failed: {out}")
        try:
            return json.loads(out)
        except json.JSONDecodeError as exc:
            raise RigUnavailable(502, f"rig status returned invalid JSON: {out[:200]}") from exc

    async def observe(self) -> dict:
        """`rig observe`: GPUs, every llama container, served ports, remote profiles, rig jobs."""
        code, out = await self._run("observe", timeout=60)
        if code != RIG_OK:
            raise RigUnavailable(502, f"rig observe failed: {out[-500:]}")
        try:
            return json.loads(out)
        except json.JSONDecodeError as exc:
            raise RigUnavailable(502, f"rig observe returned invalid JSON: {out[:200]}") from exc

    async def start_detached(self, profile_id: str, gpus: list[int], port: int, lease_id: str,
                             cache_ram_mib: int | None = None) -> dict:
        """Hand one load to the rig as a job and return at once. The rig queues and runs it."""
        extra = ("--cache-ram", str(int(cache_ram_mib))) if cache_ram_mib is not None else ()
        code, out = await self._run(
            "start", "remote", profile_id, "--gpus", ",".join(map(str, gpus)), "--port", str(port),
            "--lease", lease_id, *extra, "--detach", "--json", timeout=60)
        if code != RIG_OK:
            status = 503 if code in (RIG_BUSY, RIG_NO_CAPACITY) else 502
            raise RigUnavailable(status, f"rig refused {profile_id}: {out[-800:]}")
        try:
            return json.loads(out.splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as exc:
            raise RigUnavailable(502, f"rig start --detach returned no job: {out[:200]}") from exc

    async def stop_profile(self, profile_id: str) -> str:
        code, out = await self._run("stop", profile_id, timeout=120)
        if code != RIG_OK:
            raise RigUnavailable(502, f"rig stop {profile_id} failed: {out[-500:]}")
        return out

    def loading(self) -> list[dict]:
        """Profiles this gateway is loading right now, with how long each has been loading."""
        now = time.monotonic()
        return [
            {"id": pid,
             "since": datetime.fromtimestamp(time.time() - (now - t0), timezone.utc).isoformat(timespec="seconds"),
             "elapsed_s": round(now - t0, 1)}
            for pid, t0 in self._loading_since.items()
        ]

    async def ready_port(self, profile_id: str) -> int | None:
        for p in (await self.status()).get("profiles", []):
            if p.get("id") == profile_id and p.get("state") == "ready":
                return int(p["port"])
        return None

    async def ensure_ready(self, profile_id: str, start: Callable[[], Awaitable[None]] | None = None) -> int:
        """Return the port serving profile_id, loading it on the rig first if needed.

        `start` replaces the plain `rig start remote <id>` (the rig manager passes its own load).
        Concurrent callers share one start. The start is shielded: a client that disconnects
        doesn't cancel a load other requests (or its own retry) are waiting on.
        """
        port = await self.ready_port(profile_id)
        if port is not None:
            return port
        task = self._inflight.get(profile_id)
        if task is None:
            task = asyncio.create_task(start() if start else self._start(profile_id))
            self._inflight[profile_id] = task
            self._loading_since[profile_id] = time.monotonic()

            def _done(_t: asyncio.Task) -> None:
                self._inflight.pop(profile_id, None)
                self._loading_since.pop(profile_id, None)

            task.add_done_callback(_done)
        await asyncio.shield(task)
        port = await self.ready_port(profile_id)
        if port is None:
            raise RigUnavailable(502, f"rig reported {profile_id} started but it is not ready")
        return port

    async def _start(self, profile_id: str) -> None:
        deadline = time.monotonic() + self.start_timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RigUnavailable(504, f"{profile_id} not ready within {self.start_timeout_seconds:.0f}s")
            try:
                code, out = await self._run("start", "remote", profile_id, timeout=remaining,
                                            kill_on_timeout=False)
            except RigUnavailable as exc:
                if exc.status == 504:
                    raise RigUnavailable(504, f"{profile_id} not ready within {self.start_timeout_seconds:.0f}s; "
                                              "the rig is still loading it, retry shortly") from exc
                raise
            if code == RIG_OK:
                return
            if code == RIG_LOCKED:
                await asyncio.sleep(LOCKED_RETRY_SECONDS)
                continue
            if code == RIG_BUSY:
                raise RigUnavailable(503, f"rig busy: {out.splitlines()[0] if out else 'another mode'}")
            if code == RIG_NO_CAPACITY:
                raise RigUnavailable(503, out)
            raise RigUnavailable(502, f"rig failed to start {profile_id}: {out[-1500:]}")
