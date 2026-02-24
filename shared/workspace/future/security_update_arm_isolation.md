# Security Update: Arm Folder Isolation

Date: 2026-02-24

## What Changed Now

- `github_analyzer` `prepare_repo.py` now performs a dependency risk preflight scan and writes:
  - `repo_dependency_risk.json`
- If suspicious dependency install inputs are detected, it hard-fails with a clear message:
  - `DONT DO THIS: suspicious dependency install inputs detected...`
- This stops deep-mode dependency installs before `pip/npm` execution.

## New Helper Added

- Added `shared/plans/shoulders/github_analyzer/scripts/sandbox_run.py`
- Current behavior:
  - repo-local isolated `HOME`, `TMPDIR`, XDG dirs, pip/npm/cargo caches
  - strips common secret-bearing env vars
  - deterministic local cache paths per repo
- Current limitation:
  - environment isolation only (not a container / namespace sandbox yet)

## Next Security Steps (Planned)

1. Route install/build commands in `prepare_repo.py` through `sandbox_run.py`
2. Add real process isolation (`bwrap` or rootless container)
3. Add per-arm policy file (`.llm_orch_policy.json`) for:
   - installs allowed/blocked
   - network mode
   - runtime/memory limits
   - risky dependency approval requirements
4. Expose a clear dashboard notice/approval flow for risky dependency scans

## Goal

Treat each `arms/<repo>` folder as an isolated execution cell with explicit policy and minimal host exposure.
