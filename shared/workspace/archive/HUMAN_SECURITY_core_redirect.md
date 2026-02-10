# HUMAN: Workers Can Write to core/ via Output Redirection

**Created**: 2026-02-09
**Severity**: security
**Context**: Reviewing defense-in-depth for the protected core/ directory
**Problem**: The executor's `_check_file_write()` method blocks direct Python-level writes to `core/` paths. However, `run_bash()` executes arbitrary shell commands where output redirection is handled by the shell, not the executor. A command like `echo "x" > /media/bryan/shared/core/RULES.md` or `cmd | tee /media/bryan/shared/core/SYSTEM.md` would bypass the Python-level write check. The bash blocked_patterns include `>.*core/` but not `>>`, `2>`, or pipe-to-tee variants. The primary protection is that core/ is root-owned with chmod 644, so OS-level permissions should block writes if agents run as non-root. But this is a single layer of defense.
**Attempts**: Code review only. Verified core/ is root-owned (chmod 644) which should prevent writes at OS level.
**Recommendation**:
1. Verify agents NEVER run as root (document this as a hard requirement)
2. Add blocked patterns: `>>.*core/`, `2>.*core/`, `tee.*core/`, `cp.*core/`, `mv.*core/`
3. Consider a filesystem watcher (inotifywait) on core/ that alerts on any modification attempt
4. Add a startup check that verifies core/ ownership and permissions before agents begin

Human comments: I agree the workers shouldnt need any sudo permissions, we can remove that from the brain as well.  Add the blocked requirements and check. We'll think about the watcher, document for later 