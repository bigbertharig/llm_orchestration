# Benchmark Runtime Identity Preflight

Priority: high
Created: 2026-08-22

## Problem

Campaign `20260820_modern_validation_l50_retry3` reported successful brain
model loads while port 11434 was still served by the existing
`qwen3.6:27b`/`Qwen3.6-27B-Q4_K_M.gguf` container. Health-only readiness let the
Qwen3.5, Qwen Coder, and Devstral suites run against the wrong model and record
misattributed results.

## Required Implementation

1. Before starting a lane, inspect port ownership and stop/reclaim only the
   runtime explicitly owned by the benchmark controller.
2. After startup, query `/v1/models` and require its model/GGUF identity to
   match the manifest entry. Do not treat generic HTTP health as readiness.
3. On a mismatch, fail the lane loudly, stop its owned runtime, and prohibit
   benchmark result recording.
4. Persist requested identity, observed identity, container ID, image, port,
   and verification timestamp in the lane status and campaign log.
5. Add a regression test with a healthy stale server on the requested port.

## Evidence Cleanup And Rerun

- Invalid retry3 brain records were removed from
  `plans/shoulders/benchmarking/results/model_benchmark_records.jsonl`; raw
  campaign history remains preserved.
- After the controller fix, rerun only `qwen3.5:27b`,
  `qwen3-coder:30b-a3b`, and `devstral-small:24b` for l50 GSM8K, DROP, IFEval,
  and runtime.
- Use `--max-active-models 1` with the 95C thermal guard. Do not retest archived
  or already validated worker models.

## Acceptance Criteria

- A stale healthy model on port 11434 causes a deterministic preflight failure.
- No result can be appended unless observed identity equals requested identity.
- The three brain-only reruns complete with identity evidence and temperatures
  below the guard threshold.
