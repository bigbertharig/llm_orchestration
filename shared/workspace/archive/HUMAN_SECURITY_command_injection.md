# HUMAN: Bash Command Injection via Operator Chaining

**Created**: 2026-02-09
**Severity**: security
**Context**: Code review of executor.py _check_bash_command() and permissions/*.json blocked_patterns
**Problem**: The blocked_patterns check uses `re.search(pattern, command)` against the full command string, but does not account for shell operator chaining. A command like `echo hello && rm -rf /` would only be caught if the blocked pattern regex happens to match the `rm -rf /` substring within the full string. However, if blocked patterns use anchors like `^rm` or `^sudo`, they would fail to match because the command string starts with `echo`. Dangerous commands can be embedded after `&&`, `||`, `;`, `|`, backticks, or `$()` subshells and evade detection. Current blocked patterns in worker.json and brain.json do not appear to use anchors, which helps, but patterns like `>.*core/` only catch `>` redirect, not `>>`, `2>`, or `| tee` to core/.
**Attempts**: Code review only. The executor is core security infrastructure (root-owned).
**Recommendation**:
1. Split commands on shell operators (`&&`, `||`, `;`, `|`) and check each segment independently
2. Also detect and check subshell constructs: backticks and `$()`
3. Add blocked patterns for all redirect forms targeting core/: `>>.*core/`, `2>.*core/`, `tee.*core/`
4. Consider using `shlex.split()` for proper shell parsing
5. Add integration tests with injection payloads to verify blocked pattern coverage

Human comments: Yes good, fix it
