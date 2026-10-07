#!/usr/bin/env bash
#
# Resume Gemma 4 worker benchmarks after interrupted run.
# HumanEval is done for both; resumes from MBPP checkpoint, then reasoning.
# Runtimes are already loaded on GPU 1 (port 11435) and GPU 3 (port 11437).
#
# Usage:
#   ssh 10.0.0.3 'nohup bash /mnt/shared/scripts/benchmarks/resume_workers.sh \
#     > /mnt/shared/logs/benchmarks/campaigns/resume_workers.log 2>&1 &'

set -euo pipefail

SHARED="/mnt/shared"
LOG_DIR="$SHARED/logs/benchmarks"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_code() {
    local model_id=$1 port=$2 run_name=$3
    log "Running bench-code: $model_id (resume with same run-name)"
    docker run --rm --network host \
        -e BENCHMARK_DISABLE_AUTO_RESERVE=1 \
        -v "$SHARED:$SHARED" \
        -v "$LOG_DIR/bench-code/history:/results" \
        -v "$SHARED/plans/shoulders/benchmarking:/benchmark-scripts:ro" \
        bench-code \
        --model "$model_id" \
        --runtime-base "http://localhost:${port}" \
        --tasks humaneval,mbpp \
        --run-name "$run_name"
    log "bench-code done: $model_id (exit: $?)"
}

run_reasoning() {
    local model_id=$1 port=$2 run_name=$3
    log "Running bench-reasoning: $model_id (gsm8k + bbh + drop, limit 50)"
    docker run --rm --network host \
        -e BENCHMARK_DISABLE_AUTO_RESERVE=1 \
        -v "$SHARED:$SHARED" \
        -v "$LOG_DIR/bench-reasoning/history:/results" \
        -v "$SHARED/plans/shoulders/benchmarking:/benchmark-scripts:ro" \
        bench-reasoning \
        --model "$model_id" \
        --runtime-base "http://localhost:${port}" \
        --tasks gsm8k,bbh,drop \
        --limit 50 \
        --run-name "$run_name"
    log "bench-reasoning done: $model_id (exit: $?)"
}

log "=== Resuming worker benchmarks ==="

# Run both in parallel — E2B in background, E4B in foreground
(
    log ">>> E2B starting (GPU 3, port 11437)"
    run_code "gemma-4:e2b-q8" 11437 "gemma4_e2b_l50_code"
    run_reasoning "gemma-4:e2b-q8" 11437 "gemma4_e2b_l50_reasoning"
    log "<<< E2B complete"
) &
E2B_PID=$!

log ">>> E4B starting (GPU 1, port 11435)"
run_code "gemma-4:e4b" 11435 "gemma4_e4b_l50_code"
run_reasoning "gemma-4:e4b" 11435 "gemma4_e4b_l50_reasoning"
log "<<< E4B complete"

wait $E2B_PID || log "WARNING: E2B had failures"

# Stop worker runtimes
docker stop llama-gemma4-e4b llama-gemma4-e2b 2>/dev/null || true

log "=== Workers complete ==="
