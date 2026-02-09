#!/usr/bin/env python3
"""
Permission-aware executor for agent actions.
Validates file access and bash commands against permission rules.
"""

import json
import re
import os
import signal
import subprocess
import fnmatch
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class ActionResult(Enum):
    ALLOWED = "allowed"
    BLOCKED = "blocked"
    NEEDS_APPROVAL = "needs_approval"


@dataclass
class ExecutionResult:
    success: bool
    output: str
    action: ActionResult
    reason: str = ""


class PermissionExecutor:
    """Executes actions with permission checking."""

    def __init__(self, permissions_file: str, agent_name: str = "agent",
                 heartbeat_callback=None):
        self.agent_name = agent_name
        self.logger = logging.getLogger(f"{agent_name}.executor")
        self.permissions = self._load_permissions(permissions_file)
        self.pending_approvals: List[Dict] = []
        self.active_process: Optional[subprocess.Popen] = None  # Track running subprocess
        self.heartbeat_callback = heartbeat_callback  # Called during long-running commands

    def _load_permissions(self, path: str) -> dict:
        with open(path) as f:
            return json.load(f)

    def _match_path(self, path: str, patterns: List[str]) -> bool:
        """Check if path matches any of the glob patterns."""
        path = os.path.expanduser(path)
        path = os.path.abspath(path)

        for pattern in patterns:
            pattern = os.path.expanduser(pattern)
            # Handle ** glob
            if "**" in pattern:
                # Convert to regex
                regex = pattern.replace("**", ".*").replace("*", "[^/]*")
                if re.match(regex, path):
                    return True
            elif fnmatch.fnmatch(path, pattern):
                return True
            elif fnmatch.fnmatch(os.path.basename(path), pattern):
                return True
        return False

    def _check_file_read(self, path: str) -> Tuple[ActionResult, str]:
        """Check if file read is allowed."""
        rules = self.permissions.get("file_access", {}).get("read", {})

        # Check blocked first (deny takes priority)
        blocked = rules.get("blocked_paths", [])
        if self._match_path(path, blocked):
            return ActionResult.BLOCKED, f"Path matches blocked pattern"

        # Check allowed
        allowed = rules.get("allowed_paths", [])
        if not allowed or self._match_path(path, allowed):
            # Check file size
            max_size = rules.get("max_file_size_mb", 10) * 1024 * 1024
            try:
                if os.path.getsize(path) > max_size:
                    return ActionResult.BLOCKED, f"File exceeds max size ({max_size // 1024 // 1024}MB)"
            except OSError:
                pass  # File might not exist yet
            return ActionResult.ALLOWED, ""

        return ActionResult.BLOCKED, "Path not in allowed list"

    def _check_file_write(self, path: str) -> Tuple[ActionResult, str]:
        """Check if file write is allowed."""
        rules = self.permissions.get("file_access", {}).get("write", {})

        blocked = rules.get("blocked_paths", [])
        if self._match_path(path, blocked):
            return ActionResult.BLOCKED, f"Path matches blocked pattern"

        allowed = rules.get("allowed_paths", [])
        if not allowed or self._match_path(path, allowed):
            return ActionResult.ALLOWED, ""

        return ActionResult.BLOCKED, "Path not in allowed list"

    def _check_bash_command(self, command: str) -> Tuple[ActionResult, str]:
        """Check if bash command is allowed."""
        rules = self.permissions.get("bash", {})

        # Check blocked patterns first (these are dangerous)
        blocked = rules.get("blocked_patterns", [])
        for pattern in blocked:
            if re.search(pattern, command):
                return ActionResult.BLOCKED, f"Matches blocked pattern: {pattern}"

        # Check allowed commands
        allowed = rules.get("allowed_commands", [])
        for pattern in allowed:
            if re.match(pattern, command):
                return ActionResult.ALLOWED, ""

        # Not explicitly allowed
        caps = self.permissions.get("capabilities", {})
        if "any_bash_command" in caps.get("requires_human_approval", []):
            return ActionResult.NEEDS_APPROVAL, "Bash commands require approval"

        return ActionResult.BLOCKED, "Command not in allowed list"

    def read_file(self, path: str) -> ExecutionResult:
        """Read a file with permission checking."""
        action, reason = self._check_file_read(path)

        if action == ActionResult.BLOCKED:
            self.logger.warning(f"BLOCKED read: {path} - {reason}")
            return ExecutionResult(False, "", action, reason)

        if action == ActionResult.NEEDS_APPROVAL:
            self.logger.info(f"PENDING approval for read: {path}")
            self.pending_approvals.append({"type": "read", "path": path})
            return ExecutionResult(False, "", action, reason)

        try:
            with open(os.path.expanduser(path)) as f:
                content = f.read()
            self.logger.debug(f"ALLOWED read: {path}")
            return ExecutionResult(True, content, action)
        except Exception as e:
            return ExecutionResult(False, str(e), ActionResult.ALLOWED, f"Read error: {e}")

    def write_file(self, path: str, content: str) -> ExecutionResult:
        """Write a file with permission checking."""
        action, reason = self._check_file_write(path)

        if action == ActionResult.BLOCKED:
            self.logger.warning(f"BLOCKED write: {path} - {reason}")
            return ExecutionResult(False, "", action, reason)

        if action == ActionResult.NEEDS_APPROVAL:
            self.logger.info(f"PENDING approval for write: {path}")
            self.pending_approvals.append({"type": "write", "path": path, "content": content[:100]})
            return ExecutionResult(False, "", action, reason)

        try:
            path = os.path.expanduser(path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                f.write(content)
            self.logger.debug(f"ALLOWED write: {path}")
            return ExecutionResult(True, f"Wrote {len(content)} bytes", action)
        except Exception as e:
            return ExecutionResult(False, str(e), ActionResult.ALLOWED, f"Write error: {e}")

    def run_bash(self, command: str) -> ExecutionResult:
        """Run a bash command with permission checking.

        Uses process groups so child processes can be killed cleanly if needed.
        """
        action, reason = self._check_bash_command(command)

        if action == ActionResult.BLOCKED:
            self.logger.warning(f"BLOCKED bash: {command} - {reason}")
            return ExecutionResult(False, "", action, reason)

        if action == ActionResult.NEEDS_APPROVAL:
            self.logger.info(f"PENDING approval for bash: {command}")
            self.pending_approvals.append({"type": "bash", "command": command})
            return ExecutionResult(False, "", action, reason)

        timeout = self.permissions.get("bash", {}).get("timeout_seconds", 30)

        try:
            # Use Popen with new process group so we can kill all children
            self.active_process = subprocess.Popen(
                command,
                shell=True,
                executable='/bin/bash',
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True  # Creates new process group
            )

            try:
                # Poll loop instead of blocking communicate() - allows heartbeat updates
                import time
                start_time = time.time()
                heartbeat_interval = 10  # seconds
                last_heartbeat = start_time

                while True:
                    # Check if process finished
                    returncode = self.active_process.poll()
                    if returncode is not None:
                        break

                    # Check timeout
                    elapsed = time.time() - start_time
                    if elapsed > timeout:
                        self.kill_active_process()
                        return ExecutionResult(False, "", ActionResult.BLOCKED, f"Command timed out ({timeout}s)")

                    # Call heartbeat callback if provided
                    if time.time() - last_heartbeat >= heartbeat_interval:
                        if self.heartbeat_callback:
                            self.heartbeat_callback()
                        last_heartbeat = time.time()

                    time.sleep(0.5)  # Poll every 500ms

                # Process finished - read output
                stdout, stderr = self.active_process.communicate()
                returncode = self.active_process.returncode
            finally:
                self.active_process = None

            output = stdout
            if returncode != 0:
                output += f"\n[stderr: {stderr}]"
            self.logger.debug(f"ALLOWED bash: {command}")
            return ExecutionResult(returncode == 0, output, action)
        except Exception as e:
            self.active_process = None
            return ExecutionResult(False, str(e), ActionResult.ALLOWED, f"Execution error: {e}")

    def kill_active_process(self):
        """Kill any active subprocess and its entire process group."""
        if self.active_process is not None:
            try:
                # Kill the entire process group
                pgid = os.getpgid(self.active_process.pid)
                os.killpg(pgid, signal.SIGTERM)
                # Give it a moment to terminate gracefully
                try:
                    self.active_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if still alive
                    os.killpg(pgid, signal.SIGKILL)
                self.logger.info(f"Killed process group {pgid}")
            except (ProcessLookupError, OSError) as e:
                self.logger.debug(f"Process already dead: {e}")
            finally:
                self.active_process = None

    def check_capability(self, capability: str) -> bool:
        """Check if agent has a specific capability."""
        caps = self.permissions.get("capabilities", {})
        return caps.get(capability, False)

    def get_pending_approvals(self) -> List[Dict]:
        """Get list of actions waiting for human approval."""
        return self.pending_approvals.copy()

    def clear_approvals(self):
        """Clear pending approvals."""
        self.pending_approvals.clear()
