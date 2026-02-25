# GitHub Analyzer Runtime Reduction Plan (Batch 20260224_214410)

## Context
Batch `20260224_214410` took `7h 24m 50s` (started `2026-02-24T21:44:11`, completed `2026-02-25T05:09:02`).

Main contributors were LLM review/verify waves (especially verify), plus a split-runtime stall before `worker_gap_review_verify` started.

## Key Findings
1. `worker_review_verify_slice_*` dominated wall time (~3h26m).
2. `worker_gap_review_verify_slice_*` added another long wave (~1h41m).
3. A split runtime wedge caused a ~29-minute stall before gap verify started (queue filled with split-required tasks but low/zero processing).
4. Brain/setup tasks were not the bottleneck.
5. `.venv` scanning polluted quality/runtime signals but was not the primary wall-time driver.

## Goals
1. Reduce long-tail verify phase makespan.
2. Prevent split-runtime wedge stalls from idling the queue.
3. Improve runtime predictability via phase-specific controls.
4. Preserve acceptable review quality (current ~`40/51` usable slices baseline).

## Proposal

### 1. Phase-Specific Sharding (Highest Impact)
Add separate shard controls for heavy phases instead of relying only on global `WORKER_SHARDS`.

Proposed config keys:
- `DOC_SHARDS`
- `REVIEW_SHARDS`
- `VERIFY_SHARDS`
- `GAP_REVIEW_SHARDS`
- `GAP_VERIFY_SHARDS`

Behavior:
- If a phase-specific key is missing, fall back to `WORKER_SHARDS`.

Rationale:
- Verify slices are significantly slower than review slices and need finer partitioning to reduce stragglers.

### 2. Straggler Mitigation for Verify Phases
Add a straggler policy for long-running verify slices.

Options:
- Preferred: split more aggressively at manifest-build time using file count / bytes / complexity heuristics.
- Next-best: dynamic split/requeue when a verify slice exceeds a threshold (e.g. `15m`) and remaining work can be partitioned.

Rationale:
- A few 17-21 minute slices are setting phase makespan.

### 3. Pre-Verify Split Runtime Preflight
Add an explicit preflight task before:
- `worker_review_verify`
- `worker_gap_review_verify`

Preflight responsibilities:
- validate split pair health
- detect wedged split ports
- reclaim/reset stale split reservations/ports
- prewarm required split model(s)
- fail fast if split capacity cannot be established

Rationale:
- Avoid queue stall loops where split-required tasks are released but cannot start.

### 4. Faster Split Wedge Recovery Policy
Change split reconciliation from passive suppression to active reclaim after repeated failed checks.

Proposed controls:
- `SPLIT_WEDGE_RECONCILE_THRESHOLD` (monitor cycles)
- `SPLIT_WEDGE_RECLAIM_COOLDOWN_S`

Behavior:
- After threshold breaches, issue explicit reclaim/unload/reset actions for wedged split group state.

Rationale:
- Current suppression loops can burn tens of minutes.

### 5. CPU Offload for Preprocessing (Medium Impact, Low Risk)
Add/expand CPU-side preprocessing to reduce LLM token load and improve slice signal quality.

Candidates:
- per-slice file inventory + token-hit index
- snippet extraction for likely evidence regions
- low-signal file filtering / deduping
- JSON/data metadata normalization summaries
- report consistency checks / evidence indexing

Rationale:
- Doesn’t replace LLM work, but can shorten verify tasks by reducing noise and context size.

### 6. Runtime Validation Noise Controls (Partially Implemented)
Enforce ignore rules for vendor/generated directories in:
- repo shape counting
- test discovery
- pytest collect

Ignore dirs include:
- `.venv`, `venv`, `node_modules`, `.git`, `__pycache__`, `dist`, `build`, `.turbo`

Optional next step:
- prefer `tests/`-scoped pytest probing unless repo has no local tests.

Rationale:
- Improves quality signal and avoids misleading runtime pass/fail results from site-packages tests.

### 7. Phase Runtime Metrics / Stall Accounting
Add phase-level metrics to batch artifacts:
- first task start / last task end by family
- queue idle time while tasks pending
- split-capacity wait time
- count of slices exceeding thresholds (e.g. >10m, >15m, >20m)
- time spent in split wedge suppression/recovery

Rationale:
- Enables precise regression tracking and validates improvements.

## Expected Impact (Rough)
1. `.venv` exclusion alone: small wall-time savings (seconds to minutes), significant quality improvement.
2. Phase-specific sharding: likely large verify-phase improvement (often 20-40% on long-tail phases).
3. Split preflight + wedge reclaim: can eliminate intermittent 10-30+ minute stalls.
4. CPU preprocessing: repo-dependent moderate improvements by shortening LLM slices.

## Recommended Rollout Order
1. Phase-specific sharding config + manifest support
2. Pre-verify split runtime preflight task
3. Split wedge recovery policy tuning
4. CPU preprocessing helpers for verify slices
5. Phase metrics instrumentation

## Validation Plan
1. Re-run `github_analyzer` on `county-map` with same models and compare:
   - total wall time
   - verify phase wall times
   - split stall time
   - usable slice ratio
2. Re-run on a smaller known repo (~1h historical baseline) to check for regressions.
3. Inspect logs for:
   - fewer `SPLIT_RECONCILE_WEDGED_PORT` loops
   - more stable verify concurrency
   - fewer >15m outlier slices

## Implementation Work Breakdown
1. Manifest/sharding changes (`build_scope_manifest.py`, phase manifest builders, plan scripts)
2. Split preflight task + orchestration wiring
3. Split wedge recovery tuning (brain resource logic)
4. CPU preprocessing pipeline additions (new scripts + `worker_review` inputs)
5. Metrics export enhancements (`execution_stats.json`, batch summary)

## Suggested First Step
Start with phase-specific sharding for verify phases (`VERIFY_SHARDS`, `GAP_VERIFY_SHARDS`) because it is the highest-impact, lowest-risk runtime reduction lever.

---

## Progress Update (Implemented)

### Completed: Quality / Noise Controls
- Added vendor/generated path ignores for analyzer runtime/test discovery and repo-shape counting.
- `.venv`, `venv`, `node_modules`, `.git`, `__pycache__`, `dist`, `build`, `.turbo` are now excluded in key analyzer paths.
- This fixes polluted runtime validation results (site-packages pytest leakage) and improves repo-size/quality threshold inputs.

### Completed: Phase Sharding Controls (Initial)
- Added plan inputs to `github_analyzer`:
  - `VERIFY_SHARDS`
  - `GAP_VERIFY_SHARDS`
- Wired `VERIFY_SHARDS` into `build_scope_manifest.py` via `--target-shards`.
- Added fallback inheritance (`--fallback-target-shards`) so omitted `VERIFY_SHARDS` falls back to `WORKER_SHARDS`.
- Wired `GAP_VERIFY_SHARDS` into `build_gap_manifest.py` using `--target-shards` alias (placeholder-safe parsing added).

Important current limitation:
- `VERIFY_SHARDS` changes the shared `analysis_manifest` used by both `worker_review` and `worker_review_verify`.
- `GAP_VERIFY_SHARDS` changes the `gap_manifest` used by both `worker_gap_review` and `worker_gap_review_verify`.
- This is useful now, but not yet true verify-only re-sharding.

### Completed: Split Preflight Script (Plan-Local)
- Added `scripts/preflight_split_capacity.py` to `github_analyzer`.
- Capabilities:
  - clear stale split reservation artifacts (optional)
  - queue targeted `load_split_llm` tasks
  - wait for required number of ready split groups before verify waves
- Intended use: gate verify phases on split capacity readiness to avoid long queue stalls.

### Completed: Split Preflight Hardening + Main Plan Wiring
- Hardened `preflight_split_capacity.py` with fail-closed behavior for wedged idle split groups:
  - detects inconsistent split group state from GPU heartbeats (split owner/group present but target model not ready)
  - avoids touching groups that are busy (`active_tasks` / meta task in progress)
  - queues targeted `unload_split_llm` for wedged idle groups before reattempting split loads
  - still clears stale reservation artifacts and queues one targeted `load_split_llm` at a time
- Wired split preflight into main `github_analyzer` before all 14B verify waves:
  - `worker_review_verify`
  - `worker_review_phase_backfill_verify`
  - `worker_gap_review_verify`
- Added plan inputs for preflight tuning:
  - `SPLIT_PREFLIGHT_MIN_READY_GROUPS`
  - `SPLIT_PREFLIGHT_TIMEOUT_SEC`
  - `SPLIT_PREFLIGHT_POLL_SEC`
- Updated `github_analyzer_verify_benchmark` to use the same `--reclaim-wedged-groups` preflight mode.

### Completed: Calibration / Benchmark Plans for Faster Tuning
Two new shoulder plans were created to avoid repeated 7h full runs:

1. `github_analyzer_manifest_sweep`
- CPU-only shard calibration projections (no LLM phases)
- Outputs `manifest_sweep.json/md`
- Use this first to compare shard grids across repos quickly

2. `github_analyzer_verify_benchmark`
- Runs shortened pipeline: `prepare_repo`, `warm_workers`, `brain_strategy`, `build_scope_manifest`, benchmark subset selection, `worker_review`, split preflight, `worker_review_verify`, timing summary
- Skips gap/backfill/runtime_validation/final report
- Outputs `phase_benchmark_summary.json/md`

### Completed: Calibration Helper Scripts
- `github_analyzer_manifest_sweep/scripts/manifest_sweep.py`
- `github_analyzer_verify_benchmark/scripts/build_benchmark_manifest.py`
- `github_analyzer_verify_benchmark/scripts/summarize_phase_benchmark.py`

### Completed: Test Presets for Arm Repos
Input presets added to both new plans for:
- `county-map`
- `temple_stuart_accounting`
- `disaster_clippy_public`
- `research_prospector`
- `research_contact_enrichment`

### Completed: Documentation for New Sessions / Compacts
Added plan-local docs:
- `github_analyzer_manifest_sweep/notes/README.md`
- `github_analyzer_verify_benchmark/notes/README.md`

These describe the calibration workflow and expected outputs.

## Remaining Work (Next Steps)

### A. Validate New Plans End-to-End (Short Runs)
- Run `github_analyzer_manifest_sweep` on the five target repos and collect projected recommendations.
- Run `github_analyzer_verify_benchmark` on a small benchmark subset for several `VERIFY_SHARDS` values.
- Compare p90/p95/max verify slice durations and estimated concurrency.

### B. Wire Split Preflight into Main `github_analyzer` Plan
- Done (plan-local mitigation implemented).
- Remaining related work is core-agent state machine enforcement (load/unload exclusivity and atomic split transitions), which is outside plan-local scope.

### C. Implement True Verify-Only Re-sharding
- Create separate verify manifests (or verify sub-slices) while preserving extractor output compatibility.
- Goal: allow `worker_review` and `worker_review_verify` to use different slice granularities.

### D. Add Phase Metrics / Stall Accounting to Batch Artifacts
- Record split stall time and queue-idle-while-pending metrics.
- Export p50/p90/p95/max task durations by phase directly in execution stats.

### E. CPU Offload (Optional, After Tuning)
- Add CPU preprocessing for snippet extraction / evidence indexing to reduce LLM verify token load.
