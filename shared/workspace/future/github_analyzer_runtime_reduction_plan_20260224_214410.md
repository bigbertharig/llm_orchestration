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

### Completed: CPU Tooling Bundle (MVP Scripts)
Implemented CPU-side support scripts in `shared/plans/shoulders/github_analyzer/scripts/`:
- `cpu_tooling_common.py` (shared filtering/tokenization/path helpers)
- `evidence_indexer.py` (symbols/imports/routes/top tokens/snippets by file)
- `slice_complexity_scorer.py` (per-slice complexity scoring + shard recommendations)
- `repo_structure_extractor.py` (entrypoints/routes/extensions/import summaries)
- `json_data_summarizer.py` (JSON/data shape summaries)
- `claim_evidence_linker.py` (CPU pre-linking of doc claims to candidate files)
- `result_normalizer.py` (normalize + dedupe worker review outputs)
- `contradiction_prefilter.py` (CPU contradiction candidate prefilter)
- `verify_straggler_splitter.py` (manifest re-splitting for long verify slices)
- `runtime_probe_classifier.py` (runtime probe signal/noise classification)
- `benchmark_analyzer.py` (batch timeline/phase timing + idle gap analysis)
- `work_eligibility_debugger.py` (queue/heartbeat claimability diagnostics)

Status:
- Standalone CLI tools implemented and smoke-tested on existing `github_analyzer` / benchmark artifacts.
- Not yet wired into the default `github_analyzer` plan graph.

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
- `github_analyzer_manifest_sweep` completed on five target repos (Feb 25, 2026); projected shard recommendations collected.
- `github_analyzer_verify_benchmark` completed on `county-map` (batch `20260225_120644`) in `1h 51m 0s`.
- Next:
  - run `github_analyzer_verify_benchmark` on a smaller repo (`research_prospector` or `temple_stuart_accounting`)
  - compare p90/p95/max verify slice durations and estimated concurrency
  - test one alternate `VERIFY_SHARDS` value on `county-map` (e.g. `6` from manifest-sweep recommendation vs current `8`)

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
- Initial CPU tooling bundle (MVP CLI scripts) is now implemented and documented in `shared/plans/shoulders/github_analyzer/notes/CPU_TOOLING.md`.

### F. Core-Agent Changes (Blocked Here, Documented for Human)
These are required for the next major throughput/stability improvements but cannot be completed in-session due immutable protections on `shared/agents/*.py`.

Saved human action specs:
- `shared/workspace/human/HUMAN_split_runtime_state_machine_20260225_132656.md`
  - explicit runtime state machine
  - load/unload exclusivity
  - split wedge reclaim policy / telemetry hardening
- `shared/workspace/human/HUMAN_capability_based_llm_task_routing_20260225_152228.md`
  - normal LLM work routed by capability (model/tier/context), not placement
  - placement retained for telemetry/meta-tasks/resource strategy

These should be treated as the authoritative design notes for core changes that plan-local mitigations cannot fully solve.

---

## Validation Results (So Far)

### Manifest Sweep (Completed)
Completed `github_analyzer_manifest_sweep` on the requested arm repos and collected initial recommendations:

- `county-map`: `verify_shards=6`, `gap_verify_shards=12`
- `temple_stuart_accounting`: `verify_shards=10`, `gap_verify_shards=12`
- `disaster_clippy_public`: `verify_shards=12`, `gap_verify_shards=18`
- `research_prospector`: `verify_shards=4`, `gap_verify_shards=12`
- `research_contact_enrichment`: `verify_shards=4`, `gap_verify_shards=12`

These are heuristic projections only (no LLM execution), but they give a useful size-band starting point.

### Verify Benchmark: `county-map` (Batch `20260225_120644`)

Batch outcome:
- Completed successfully in `1h 51m 0s` (`6654s`) with `100%` task success (`32/32`) and `3` retries.
- This is a shortened benchmark plan (review + verify only), not a full analyzer run.

Configuration highlights:
- `VERIFY_SHARDS=8`
- `BENCHMARK_MAX_SLICES=12`
- `BENCHMARK_PER_PHASE=3`

Phase timing summary (`output/phase_benchmark_summary.json`):

`worker_review_slice_*` (7B extract/review wave)
- `count=12`
- `wall_seconds=1399.0` (~`23.3m`)
- `avg_task_seconds=200.8`
- `p90=392.6s`, `max=454.0s`
- `estimated_concurrency=1.72`
- `stragglers_gt_600s=0`

`worker_review_verify_slice_*` (14B split verify wave)
- `count=12`
- `wall_seconds=5140.2` (~`85.7m`)
- `avg_task_seconds=467.7`
- `p90=867.5s`, `max=1271.5s` (~`21.2m`)
- `estimated_concurrency=1.09`
- `stragglers_gt_600s=5`, `stragglers_gt_900s=1`

Interpretation:
- Verify remains the dominant bottleneck and still exhibits long-tail stragglers.
- Split preflight/wedge reclaim improvements appear to have helped avoid a catastrophic split queue stall in this benchmark, but verify concurrency is still effectively near-serial (`~1.09`) in practice for this run.
- This strongly supports continuing with:
  1. true verify-only re-sharding (separate verify manifest granularity)
  2. additional split capacity stability testing
  3. benchmark-based default shard tuning by repo size band

### Immediate Tuning Experiments (Recommended)
1. Run `github_analyzer_verify_benchmark` on `research_prospector` (smaller repo) with current defaults (`VERIFY_SHARDS=8`) to establish baseline verify concurrency/straggler behavior.
2. Re-run `county-map` benchmark with `VERIFY_SHARDS=6` (manifest-sweep recommendation) for direct comparison against `VERIFY_SHARDS=8`.
3. Optional: run `county-map` with `VERIFY_SHARDS=12` to test whether more slices reduce stragglers or just add overhead.

### Verify Benchmark: `research_prospector` (Batch `20260225_141644`)

Batch outcome:
- Completed successfully in `43m 16s` (`2590s`) with `100%` task success (`24/24`) and `1` retry.
- This validates the benchmark plan on a smaller repo after syncing the benchmark preflight script copy.

Configuration highlights:
- `VERIFY_SHARDS=8`
- `BENCHMARK_MAX_SLICES=12`
- `BENCHMARK_PER_PHASE=3`

Phase timing summary (`output/phase_benchmark_summary.json`):

`worker_review_slice_*`
- `count=8`
- `wall_seconds=838.2` (~`14.0m`)
- `avg_task_seconds=87.6`
- `p90=143.0s`, `max=146.7s`
- `estimated_concurrency=0.84`

`worker_review_verify_slice_*`
- `count=8`
- `wall_seconds=1669.5` (~`27.8m`)
- `avg_task_seconds=208.6`
- `p90=178.6s`, `p95/max=647.7s`
- `estimated_concurrency=1.0`
- `stragglers_gt_600s=1`

Key lessons from this batch:
1. Verify remains the dominant phase even on smaller repos.
2. Effective verify concurrency was still ~serial (`1.0`) despite user-observed periods with multiple split runtimes appearing ready (`JOINED`).
3. This reinforces the need for **capability-based LLM routing** (normal LLM tasks should not be placement-hard).
4. Split wedge behavior still occurs intermittently (`SPLIT_RECONCILE_WEDGED_PORT`) but recovered during the run.
5. Current phase metrics are sufficient to compare gross runtime, but still need explicit stall attribution (queue wait vs compute) for reliable shard tuning.

Immediate follow-up experiments from this batch:
1. Re-run `research_prospector` with `VERIFY_SHARDS=4` (manifest-sweep recommendation) for direct comparison against `VERIFY_SHARDS=8`.
2. Re-run `county-map` with `VERIFY_SHARDS=6` and `VERIFY_SHARDS=12` to compare against current `VERIFY_SHARDS=8` benchmark.
3. Add/verify benchmark metrics for queue-visible split verify task count and split-ready group count over time (if feasible in plan-local summarization).

### Verify Benchmark: `research_prospector` (Batch `20260225_165712`, Artifact-Fed + `VERIFY_SHARDS=4`)

Batch outcome:
- Completed successfully in `29m 26s` (`1760s`) with `100%` task success (`29/29`) and `0` retries.
- CPU artifact tasks + artifact-fed `worker_review` prompts were active, and all extract/review slices completed.

Phase timing summary (`output/phase_benchmark_summary.json`):

`worker_review_slice_*`
- `count=8`
- `wall_seconds=777.5` (~`13.0m`)
- `avg_task_seconds=273.8`
- `p90=384.1s`, `max=392.9s`
- `estimated_concurrency=2.82`

`worker_review_verify_slice_*`
- `count=8`
- `wall_seconds=836.0` (~`13.9m`)
- `avg_task_seconds=182.5`
- `p90=246.9s`, `max=268.2s`
- `estimated_concurrency=1.75`
- `stragglers_gt_600s=0`

Comparison vs `research_prospector` baseline (`20260225_141644`, `VERIFY_SHARDS=8`):
- Total runtime improved: `2590s -> 1760s` (about `32%` faster)
- Verify wall time improved: `1669.5s -> 836.0s` (about `50%` faster)
- Verify concurrency improved: `1.0 -> 1.75`
- Retries improved: `1 -> 0`

Quality/reliability notes:
- All `8` extractor outputs and all `8` review outputs were produced.
- No `worker_review_failures/` directory for this batch (no preserved JSON parse failures).
- Some slices remain thin (`finding_count=0` on a small number of slices), but overall output quality is at least comparable to the prior benchmark and reliability is better.

Practical tuning guidance (size-band note):
- For repos similar to `research_prospector` (small/medium shoulder-wrapper repos, ~8 benchmark slices in this configuration), prefer:
  - `VERIFY_SHARDS=4` default starting point
  - not `VERIFY_SHARDS=8` unless later data shows a concurrency increase on that hardware
- Over-slicing smaller repos appears to add coordination overhead and may worsen verify throughput when split concurrency is limited.

Recommended next experiment:
1. Run `county-map` benchmark with `VERIFY_SHARDS=6` and preflight target `BENCHMARK_MIN_SPLIT_GROUPS=2` to test larger-repo scaling when both split groups are prepared before verify.
