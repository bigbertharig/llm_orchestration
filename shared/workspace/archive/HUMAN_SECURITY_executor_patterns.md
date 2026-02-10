# HUMAN: Regex Metacharacter Bypass in Executor Glob Patterns

**Created**: 2026-02-09
**Severity**: security
**Context**: Code review of executor.py _match_path() for permission pattern safety
**Problem**: The glob-to-regex conversion in executor.py replaces `**` with `.*` and `*` with `[^/]*` but does not escape other regex metacharacters in the original pattern. Patterns containing `[`, `]`, `(`, `)`, `+`, `?`, or `{` would be interpreted as regex syntax, potentially matching unintended files. Additionally, path resolution uses `os.path.expanduser()` and `os.path.abspath()` but does not call `os.path.realpath()` to resolve symlinks. A symlink pointing outside allowed paths could bypass blocked_paths checks since the permission check runs against the pre-symlink path.
**Attempts**: Code review only. This is core security code (executor.py) which is root-owned and requires human review before modification.
**Recommendation**:
1. Use `re.escape()` on pattern segments before replacing glob wildcards with regex equivalents
2. Add `os.path.realpath()` call to resolve symlinks before checking against allowed/blocked paths
3. Consider using `pathlib.PurePath.match()` instead of hand-rolled regex conversion
4. Add test cases for edge-case patterns (brackets, question marks, plus signs in paths)

Human comment: Yes, implement 1,2,4.  Is the extra library big, what overhead would that add?