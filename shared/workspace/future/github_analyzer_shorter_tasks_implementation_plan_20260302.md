# GitHub Analyzer Shorter Tasks Implementation Plan

## Goal

Reduce GitHub Analyzer LLM task duration, especially 14B verifier work, by making slices smaller, more even, and more targeted.

Primary targets:
- Lower `worker_review_verify_slice_` average and p95 time
- Reduce long-tail dense slices on larger repos like `county-map`
- Preserve quality while increasing effective concurrency

## Definitive Status

### Code Implementation Status: Complete

The code-path implementation work for the shorter-tasks effort is now complete.

Verified in code:

1. Phase 1: Complexity-based slice planning
- `cpu_tooling_common.py`
  - content-based scoring is implemented
  - control-flow is not counted as functions
  - `if` and `while` are not double-counted
  - exception fallback uses a non-zero conservative score
  - Go struct detection is included
- `build_scope_manifest.py`
  - files are scored before slicing
  - slices are packed by complexity budget
  - contradiction slices are reserved and not skipped
  - contradiction reservation is clamped for small slice budgets
  - `analysis_manifest.json` includes complexity metadata and summary statistics

2. Phase 2: Heavy-file isolation and segmentation
- `build_scope_manifest.py`
  - heavy-file detection is implemented
  - extremely heavy file detection is implemented
  - extremely heavy files can be segmented into multiple slices
  - segmented slices include:
    - `file_segments`
    - `is_segmented_slice`
    - `segment_index`
    - `total_segments`
- `cpu_tooling_common.py`
  - symbol extraction is implemented
  - symbol-based segmentation is implemented
  - fixed-window fallback segmentation is implemented

3. Phase 3: Token-budget preflight splitting
- `build_benchmark_manifest.py`
  - whole-file slices can be estimated and split by preflight token budget
  - segmented slices use line-range-aware estimation via `_estimate_segment_bytes()`
  - segmented slices are intentionally not split by file-based preflight logic
  - segmented slices are preserved with:
    - `preflight_split_reason = segmented_slice_not_splittable`
- Status:
  - complete for whole-file slices
  - safe and non-destructive for segmented slices

4. Priority 1 wiring: metadata passthrough and ranking
- `build_benchmark_manifest.py`
  - `SLICE_PASSTHROUGH_KEYS` preserves segmented-slice and complexity metadata
  - `_slice_rank_key()` uses `slice_complexity_score` first and falls back to `complexity_hint`
  - passthrough is applied in both the no-split and preflight-split paths

5. Priority 2 wiring: worker segment consumption
- `worker_review.py`
  - `_read_file_line_range()` reads only requested line windows
  - `collect_file_snippets()` accepts `file_segments`
  - worker call sites pass `sl.get('file_segments')`
  - when segments are present, the worker reads only the requested ranges

6. Phase 4: Verifier micro-splitting
- In-task splitting exists in `worker_review.py`:
  - `_is_dense_extractor_output()`
  - `_split_findings_for_verification()`
  - `_merge_verified_chunks()`
- Task-level splitting exists in the plan path:
  - `build_verify_tasks_manifest.py` creates `verify_tasks_manifest.json`
  - dense extracts expand into multiple scheduler-visible verify tasks
  - `worker_review.py` accepts:
    - `--verify-chunk-index`
    - `--verify-total-chunks`
    - `--findings-subset-file`
- Robust output handling is now in place:
  - chunk tasks write `{slice_id}__chunk_{index}.json`
  - `merge_verify_chunks.py` merges chunk outputs into canonical `{slice_id}.json`
  - merge reads `analysis.claim_matrix` and writes merged `analysis.claim_matrix`
  - merge exits non-zero if chunks are missing or merge errors occur (fail-closed by default)

7. Phase 5: CPU preprocessing artifacts
- `build_scope_manifest.py` writes:
  - `analysis_manifest.json`
  - `file_complexity_manifest.json`
  - `symbol_map.json`
  - `candidate_evidence_map.json`

8. Phase 6: Dynamic shard defaults
- `slice_complexity_scorer.py` implements:
  - `recommend_extract_shards()`
  - `recommend_verify_shards()`
  - `recommend_contradiction_slices()`
- `build_scope_manifest.py` supports and executes auto-shard mode
- `plan.md` now hardcodes `--target-shards auto` in the executable command path
- auto-shard calculation is now the actual runtime default behavior for this benchmark plan

## Verified Files

Key implementation files:
- `/home/bryan/llm_orchestration/shared/plans/shoulders/github_analyzer_verify_benchmark/scripts/cpu_tooling_common.py`
- `/home/bryan/llm_orchestration/shared/plans/shoulders/github_analyzer_verify_benchmark/scripts/build_scope_manifest.py`
- `/home/bryan/llm_orchestration/shared/plans/shoulders/github_analyzer_verify_benchmark/scripts/build_benchmark_manifest.py`
- `/home/bryan/llm_orchestration/shared/plans/shoulders/github_analyzer_verify_benchmark/scripts/build_verify_tasks_manifest.py`
- `/home/bryan/llm_orchestration/shared/plans/shoulders/github_analyzer_verify_benchmark/scripts/merge_verify_chunks.py`
- `/home/bryan/llm_orchestration/shared/plans/shoulders/github_analyzer_verify_benchmark/scripts/worker_review.py`
- `/home/bryan/llm_orchestration/shared/plans/shoulders/github_analyzer_verify_benchmark/scripts/slice_complexity_scorer.py`
- `/home/bryan/llm_orchestration/shared/plans/shoulders/github_analyzer_verify_benchmark/plan.md`

## Test Status

Verified in this environment:
- `tests/run_tests.py` passes successfully
- current total: `36` passing checks across 3 modules

Test files:
- `/home/bryan/llm_orchestration/shared/plans/shoulders/github_analyzer_verify_benchmark/tests/test_scoring.py`
- `/home/bryan/llm_orchestration/shared/plans/shoulders/github_analyzer_verify_benchmark/tests/test_slice_builder.py`
- `/home/bryan/llm_orchestration/shared/plans/shoulders/github_analyzer_verify_benchmark/tests/test_language_fixtures.py`
- `/home/bryan/llm_orchestration/shared/plans/shoulders/github_analyzer_verify_benchmark/tests/run_tests.py`

Important scope note:
- The test suite strongly covers helper logic, slicing, segmentation, merge schema, and fail-closed merge behavior.
- It does not replace real batch validation on actual repos.

## Plan Format Status

The shared plan-format guidance is correct and aligned with this implementation.

Verified file:
- `/media/bryan/shared/workspace/PLAN_FORMAT.md`

It correctly states that:
- documented defaults in `## Inputs` are documentation only unless the command/script enforces them
- symbolic values like `auto` are valid only when the called script explicitly supports them
- manifest-producing scripts must preserve downstream-critical per-item metadata
- constrained items should fail closed or be marked non-transformable instead of silently dropping constraints

The plan-format doc is not a blocker.

## Remaining Work

### Runtime Validation Only

The remaining work is runtime proof, not additional implementation.

1. Run benchmark validation on real repos
- Repos:
  - `research_prospector`
  - `county-map`
  - `temple-stuart`
- Compare against prior baselines:
  - `worker_review_slice_` avg / p95 / max
  - `worker_review_verify_slice_` avg / p95 / max
  - total slice count
  - heavy-file slice count
  - segmented slice count
  - slice score spread

2. Validate failure behavior on real runs
- Track:
  - failed tasks
  - retried tasks
  - empty / near-empty outputs
  - whether segmented slices behave stably under real workloads

3. Validate the new task-level Phase 4 path in real batches
- Confirm:
  - dense extracts actually expand into multiple verify tasks
  - chunk outputs merge correctly in live runs
  - final review artifacts are complete and schema-consistent
  - verify tail latency improves rather than adding overhead

## Exit Criteria

### Code Implementation Exit Criteria

All met:
1. `analysis_manifest.json` preserves segmented slice metadata
2. `benchmark_manifest.json` preserves segmented slice metadata
3. `worker_review.py` consumes `file_segments`
4. benchmark slice selection uses `slice_complexity_score`
5. task-level Phase 4 output path is robust and schema-consistent
6. task-level Phase 4 merge fails closed on missing chunks
7. plan documentation matches the actual runtime interface
8. Phase 6 is a true executable default for this plan

### Runtime Validation Exit Criteria

Still pending:
1. benchmark repos show reduced verifier p95 and narrower spread versus baseline
2. no increase in failure rate or empty-output rate
3. task-level Phase 4 fan-out is confirmed to work and help in real batches

## Final Assessment

The code implementation is complete.

Authoritative status:
- code work: complete
- shared plan-format guidance: correct
- remaining work: runtime benchmark validation only

The next step is to run the benchmark batches and compare results against the previous baselines.
