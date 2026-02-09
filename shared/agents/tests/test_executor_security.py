#!/usr/bin/env python3
"""
Security tests for PermissionExecutor.

Tests edge cases in path matching (regex metacharacters, symlinks)
and command injection via shell operator chaining / subshells.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add parent directory to path so we can import executor
sys.path.insert(0, str(Path(__file__).parent.parent))
from executor import PermissionExecutor


def create_test_permissions(tmp_dir, extra_blocked_patterns=None):
    """Create a minimal permissions file for testing."""
    perms = {
        "file_access": {
            "read": {
                "allowed_paths": [f"{tmp_dir}/**"],
                "blocked_paths": [f"{tmp_dir}/blocked/**"],
                "max_file_size_mb": 10
            },
            "write": {
                "allowed_paths": [f"{tmp_dir}/**"],
                "blocked_paths": [f"{tmp_dir}/core/**"],
                "max_file_size_mb": 10
            }
        },
        "bash": {
            "allowed_commands": [".*"],
            "blocked_patterns": [
                "rm\\s+-rf\\s+/$",
                "sudo\\s",
                ">.*core/",
                ">>.*core/",
                "2>.*core/",
                "tee.*core/",
                "cp.*core/",
                "mv.*core/",
                "RULES\\.md"
            ] + (extra_blocked_patterns or []),
            "timeout_seconds": 5
        },
        "capabilities": {}
    }
    perms_file = os.path.join(tmp_dir, "test_permissions.json")
    with open(perms_file, "w") as f:
        json.dump(perms, f)
    return perms_file


class TestPathMatchingMetachars(unittest.TestCase):
    """Test that regex metacharacters in paths don't cause unintended matches."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        perms_file = create_test_permissions(self.tmp_dir)
        self.executor = PermissionExecutor(perms_file, "test-agent")

    def test_brackets_in_path_no_regex_interpretation(self):
        """Path with brackets should not be interpreted as regex character class."""
        # Pattern: /tmp/xxx/blocked/**
        # Path with brackets should match literally, not as [a-z] regex
        patterns = [f"{self.tmp_dir}/data[1]/**"]
        # This path should NOT match — [1] is literal, not a regex char class
        self.assertFalse(self.executor._match_path(
            f"{self.tmp_dir}/data1/file.txt", patterns))
        # This path SHOULD match — literal [1] in path
        self.assertTrue(self.executor._match_path(
            f"{self.tmp_dir}/data[1]/file.txt", patterns))

    def test_parentheses_in_path(self):
        """Parentheses in patterns should not create regex groups."""
        patterns = [f"{self.tmp_dir}/project(old)/**"]
        self.assertTrue(self.executor._match_path(
            f"{self.tmp_dir}/project(old)/file.txt", patterns))
        self.assertFalse(self.executor._match_path(
            f"{self.tmp_dir}/projectold/file.txt", patterns))

    def test_plus_in_path(self):
        """Plus sign in patterns should not mean 'one or more' in regex."""
        patterns = [f"{self.tmp_dir}/c++/**"]
        self.assertTrue(self.executor._match_path(
            f"{self.tmp_dir}/c++/main.cpp", patterns))
        self.assertFalse(self.executor._match_path(
            f"{self.tmp_dir}/ccc/main.cpp", patterns))

    def test_question_mark_in_path(self):
        """Question mark in non-glob context should be literal, not regex '?'."""
        # fnmatch treats ? as single-char wildcard, but in ** patterns
        # with our regex conversion, ? should be escaped
        patterns = [f"{self.tmp_dir}/what?/**"]
        # With ** in pattern, this goes through regex path.
        # The ? should be escaped to literal
        self.assertTrue(self.executor._match_path(
            f"{self.tmp_dir}/what?/file.txt", patterns))

    def test_dot_in_path(self):
        """Dots should be literal, not regex 'any character'."""
        patterns = [f"{self.tmp_dir}/.hidden/**"]
        self.assertTrue(self.executor._match_path(
            f"{self.tmp_dir}/.hidden/file.txt", patterns))
        self.assertFalse(self.executor._match_path(
            f"{self.tmp_dir}/xhidden/file.txt", patterns))


class TestPathMatchingSymlinks(unittest.TestCase):
    """Test that symlinks are resolved before permission checks."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        perms_file = create_test_permissions(self.tmp_dir)
        self.executor = PermissionExecutor(perms_file, "test-agent")

    def test_symlink_to_blocked_path_is_blocked(self):
        """A symlink pointing into blocked/ should be detected and blocked."""
        # Create blocked directory and a file in it
        blocked_dir = os.path.join(self.tmp_dir, "blocked", "secret")
        os.makedirs(blocked_dir, exist_ok=True)
        secret_file = os.path.join(blocked_dir, "data.txt")
        with open(secret_file, "w") as f:
            f.write("secret")

        # Create symlink outside blocked/ that points into it
        link_path = os.path.join(self.tmp_dir, "innocent_link")
        os.symlink(secret_file, link_path)

        # The blocked pattern is /tmp/xxx/blocked/**
        # Without realpath, _match_path would check /tmp/xxx/innocent_link (not blocked)
        # With realpath, it resolves to /tmp/xxx/blocked/secret/data.txt (blocked)
        blocked_patterns = [f"{self.tmp_dir}/blocked/**"]
        self.assertTrue(self.executor._match_path(link_path, blocked_patterns))

    def test_symlink_to_allowed_path_is_allowed(self):
        """A symlink pointing to an allowed path should work fine."""
        allowed_dir = os.path.join(self.tmp_dir, "data")
        os.makedirs(allowed_dir, exist_ok=True)
        data_file = os.path.join(allowed_dir, "file.txt")
        with open(data_file, "w") as f:
            f.write("data")

        link_path = os.path.join(self.tmp_dir, "data_link")
        os.symlink(data_file, link_path)

        allowed_patterns = [f"{self.tmp_dir}/**"]
        self.assertTrue(self.executor._match_path(link_path, allowed_patterns))


class TestCommandInjection(unittest.TestCase):
    """Test that dangerous commands embedded via shell operators are caught."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        perms_file = create_test_permissions(self.tmp_dir)
        self.executor = PermissionExecutor(perms_file, "test-agent")

    def test_simple_blocked_command(self):
        """Direct blocked commands should be caught."""
        from executor import ActionResult
        result, _ = self.executor._check_bash_command("sudo rm -rf /")
        self.assertEqual(result, ActionResult.BLOCKED)

    def test_chained_with_ampersand(self):
        """Dangerous command after && should be caught."""
        from executor import ActionResult
        result, _ = self.executor._check_bash_command("echo hello && sudo rm -rf /")
        self.assertEqual(result, ActionResult.BLOCKED)

    def test_chained_with_semicolon(self):
        """Dangerous command after ; should be caught."""
        from executor import ActionResult
        result, _ = self.executor._check_bash_command("echo hello; sudo rm -rf /")
        self.assertEqual(result, ActionResult.BLOCKED)

    def test_chained_with_or(self):
        """Dangerous command after || should be caught."""
        from executor import ActionResult
        result, _ = self.executor._check_bash_command("false || sudo reboot")
        self.assertEqual(result, ActionResult.BLOCKED)

    def test_piped_to_dangerous(self):
        """Dangerous command after pipe should be caught."""
        from executor import ActionResult
        result, _ = self.executor._check_bash_command("echo x | tee core/RULES.md")
        self.assertEqual(result, ActionResult.BLOCKED)

    def test_subshell_dollar_paren(self):
        """Dangerous command inside $() should be caught."""
        from executor import ActionResult
        result, _ = self.executor._check_bash_command("echo $(sudo whoami)")
        self.assertEqual(result, ActionResult.BLOCKED)

    def test_subshell_backtick(self):
        """Dangerous command inside backticks should be caught."""
        from executor import ActionResult
        result, _ = self.executor._check_bash_command("echo `sudo whoami`")
        self.assertEqual(result, ActionResult.BLOCKED)

    def test_redirect_append_to_core(self):
        """Append redirect to core/ should be caught."""
        from executor import ActionResult
        result, _ = self.executor._check_bash_command("echo x >> core/RULES.md")
        self.assertEqual(result, ActionResult.BLOCKED)

    def test_stderr_redirect_to_core(self):
        """Stderr redirect to core/ should be caught."""
        from executor import ActionResult
        result, _ = self.executor._check_bash_command("cmd 2> core/errors.log")
        self.assertEqual(result, ActionResult.BLOCKED)

    def test_cp_to_core(self):
        """Copy to core/ should be caught."""
        from executor import ActionResult
        result, _ = self.executor._check_bash_command("cp evil.py core/SYSTEM.md")
        self.assertEqual(result, ActionResult.BLOCKED)

    def test_mv_to_core(self):
        """Move to core/ should be caught."""
        from executor import ActionResult
        result, _ = self.executor._check_bash_command("mv evil.py core/RULES.md")
        self.assertEqual(result, ActionResult.BLOCKED)

    def test_safe_command_allowed(self):
        """Normal safe commands should be allowed."""
        from executor import ActionResult
        result, _ = self.executor._check_bash_command("ls -la /tmp")
        self.assertEqual(result, ActionResult.ALLOWED)

    def test_safe_pipe_allowed(self):
        """Normal pipes should be allowed."""
        from executor import ActionResult
        result, _ = self.executor._check_bash_command("cat file.txt | grep pattern")
        self.assertEqual(result, ActionResult.ALLOWED)

    def test_safe_chained_allowed(self):
        """Normal chained commands should be allowed."""
        from executor import ActionResult
        result, _ = self.executor._check_bash_command("mkdir -p /tmp/test && echo done")
        self.assertEqual(result, ActionResult.ALLOWED)


class TestExtractCommandSegments(unittest.TestCase):
    """Test the command segment extraction helper."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        perms_file = create_test_permissions(self.tmp_dir)
        self.executor = PermissionExecutor(perms_file, "test-agent")

    def test_simple_command(self):
        segments = self.executor._extract_command_segments("echo hello")
        self.assertIn("echo hello", segments)

    def test_ampersand_chain(self):
        segments = self.executor._extract_command_segments("cmd1 && cmd2")
        self.assertIn("cmd1", segments)
        self.assertIn("cmd2", segments)

    def test_semicolon_chain(self):
        segments = self.executor._extract_command_segments("cmd1; cmd2; cmd3")
        self.assertIn("cmd1", segments)
        self.assertIn("cmd2", segments)
        self.assertIn("cmd3", segments)

    def test_pipe(self):
        segments = self.executor._extract_command_segments("cmd1 | cmd2")
        self.assertIn("cmd1", segments)
        self.assertIn("cmd2", segments)

    def test_subshell_dollar(self):
        segments = self.executor._extract_command_segments("echo $(dangerous_cmd)")
        self.assertIn("dangerous_cmd", segments)

    def test_subshell_backtick(self):
        segments = self.executor._extract_command_segments("echo `dangerous_cmd`")
        self.assertIn("dangerous_cmd", segments)

    def test_mixed_operators(self):
        segments = self.executor._extract_command_segments("a && b | c; d")
        self.assertIn("a", segments)
        self.assertIn("b", segments)
        self.assertIn("c", segments)
        self.assertIn("d", segments)


if __name__ == "__main__":
    unittest.main()
