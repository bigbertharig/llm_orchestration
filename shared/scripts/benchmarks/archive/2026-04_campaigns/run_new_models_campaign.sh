#!/usr/bin/env bash
#
# Full benchmark campaign for Gemma 4 + Qwen 3.6 new models.
# Runs pipeline + code + reasoning (limit 50) on each model sequentially.
#
# Usage:
#   ssh 10.0.0.3 'bash /mnt/shared/scripts/benchmarks/run_new_models_campaign.sh'
#
# Brain models (GPU 0, 3090): sequential
# Worker models (GPU 1 + GPU 3, 1060s): parallel pair, then sequential brains
#
# Prerequisites:
#   - llama-runtime:b8884-candidate image built
#   - All model GGUFs in /mnt/shared/models/
#   - bench-pipeline, bench-code, bench-reasoning docker images built
#   - No other runtimes on ports 11434, 11435, 11437

set -euo pipefail

RUNTIME_IMAGE="llama-runtime:b8884-candidate"
SHARED="/mnt/shared"
LOG_DIR="$SHARED/logs/benchmarks"
CAMPAIGN_LOG="$LOG_DIR/campaigns/new_models_campaign_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$CAMPAIGN_LOG")"

# ---- helpers ----

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$CAMPAIGN_LOG"; }

wait_for_model() {
    local port=$1 max_wait=${2:-300}
    local start=$SECONDS
    while (( SECONDS - start < max_wait )); do
        local code
        code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${port}/v1/models" 2>/dev/null || echo "000")
        if [ "$code" = "200" ]; then
            log "  Model ready on port $port ($(( SECONDS - start ))s)"
            return 0
        fi
        sleep 5
    done
    log "  ERROR: Model not ready on port $port after ${max_wait}s"
    return 1
}

stop_runtime() {
    local name=$1
    docker stop "$name" 2>/dev/null || true
    sleep 2
}

start_runtime() {
    local name=$1 model=$2 port=$3 gpu=$4 ctx=${5:-16384}
    local extra_args=""

    # Qwen 3.6 models need --reasoning-budget 0
    if [[ "$model" == *"Qwen3.6"* ]]; then
        extra_args="--extra-arg --reasoning-budget --extra-arg 0"
    fi

    log "  Loading $name on GPU $gpu port $port..."
    bash "$SHARED/scripts/llama_runtime/run_runtime.sh" \
        --image "$RUNTIME_IMAGE" \
        --name "$name" \
        --model "$model" \
        --port "$port" \
        --gpus "device=$gpu" \
        --ctx-size "$ctx" \
        $extra_args
}

run_pipeline() {
    local model_id=$1 port=$2 run_name=$3
    log "  Running bench-pipeline: $model_id"
    docker run --rm --network host \
        -e BENCHMARK_DISABLE_AUTO_RESERVE=1 \
        -v "$SHARED:$SHARED" \
        -v "$LOG_DIR/bench-pipeline/history:/results" \
        -v "$SHARED/plans/shoulders/benchmarking:/benchmark-scripts:ro" \
        bench-pipeline \
        --model "$model_id" \
        --runtime-base "http://localhost:${port}" \
        --run-name "$run_name" 2>&1 | tail -5 | tee -a "$CAMPAIGN_LOG"
    log "  bench-pipeline done: $model_id (exit: ${PIPESTATUS[0]})"
}

run_code() {
    local model_id=$1 port=$2 run_name=$3
    log "  Running bench-code: $model_id (humaneval + mbpp)"
    docker run --rm --network host \
        -e BENCHMARK_DISABLE_AUTO_RESERVE=1 \
        -v "$SHARED:$SHARED" \
        -v "$LOG_DIR/bench-code/history:/results" \
        -v "$SHARED/plans/shoulders/benchmarking:/benchmark-scripts:ro" \
        bench-code \
        --model "$model_id" \
        --runtime-base "http://localhost:${port}" \
        --tasks humaneval,mbpp \
        --run-name "$run_name" 2>&1 | tail -5 | tee -a "$CAMPAIGN_LOG"
    log "  bench-code done: $model_id (exit: ${PIPESTATUS[0]})"
}

run_reasoning() {
    local model_id=$1 port=$2 run_name=$3 limit=${4:-50}
    shift 4 2>/dev/null || true
    local extra_flags=("$@")
    log "  Running bench-reasoning: $model_id (gsm8k + bbh + drop, limit $limit) flags: ${extra_flags[*]:-none}"
    docker run --rm --network host \
        -e BENCHMARK_DISABLE_AUTO_RESERVE=1 \
        -v "$SHARED:$SHARED" \
        -v "$LOG_DIR/bench-reasoning/history:/results" \
        -v "$SHARED/plans/shoulders/benchmarking:/benchmark-scripts:ro" \
        bench-reasoning \
        --model "$model_id" \
        --runtime-base "http://localhost:${port}" \
        --tasks gsm8k,bbh,drop \
        --limit "$limit" \
        "${extra_flags[@]}" \
        --run-name "$run_name" 2>&1 | tail -5 | tee -a "$CAMPAIGN_LOG"
    log "  bench-reasoning done: $model_id (exit: ${PIPESTATUS[0]})"
}

run_full_suite() {
    local model_id=$1 port=$2 tag=$3 limit=${4:-50}
    shift 4 2>/dev/null || true
    local extra_reasoning_flags=("$@")
    log ">>> Starting full suite for $model_id (limit $limit) reasoning flags: ${extra_reasoning_flags[*]:-none}"
    # run_pipeline "$model_id" "$port" "${tag}_pipeline"
    run_code "$model_id" "$port" "${tag}_code"
    run_reasoning "$model_id" "$port" "${tag}_reasoning" "$limit" "${extra_reasoning_flags[@]}"
    log "<<< Completed full suite for $model_id"
}

# ---- Per-model bench-reasoning flags ----
# Thinking models need --patch-think-tag-strip or scores are near-zero.
# See MODEL_LIBRARY.md family sections and SEQUENCED_TEST_SUITES.md for details.
# Runtime flags (--reasoning-budget) go in start_runtime(); bench flags go here.

GEMMA4_REASONING_FLAGS=(--patch-think-tag-strip)
QWEN36_REASONING_FLAGS=(--patch-think-tag-strip)
# Non-thinking models: leave empty
# CODER7B_REASONING_FLAGS=()

# ---- main campaign ----

log "=========================================="
log "New Models Benchmark Campaign"
log "Models: Gemma4 E4B, E2B, 26B-A4B, 31B + Qwen3.6 27B, 35B-A3B"
log "Suites: pipeline + code + reasoning (limit 50)"
log "=========================================="

# ---- Phase 1: Worker models (parallel on GPU 1 + GPU 3) ----
log ""
log "=== PHASE 1: Worker models (parallel) ==="

# Start both worker runtimes
start_runtime "llama-gemma4-e4b" "$SHARED/models/gemma-4-e4b/gemma-4-e4b-it-Q4_K_M.gguf" 11435 1 4096
start_runtime "llama-gemma4-e2b" "$SHARED/models/gemma-4-e2b/gemma-4-e2b-it-Q8_0.gguf" 11437 3 4096

wait_for_model 11435 300
wait_for_model 11437 300

# Run suites in parallel (background one, foreground the other)
log "Starting parallel worker benchmarks..."
(
    run_full_suite "gemma-4:e2b" 11437 "gemma4_e2b_l50" 50 "${GEMMA4_REASONING_FLAGS[@]}"
) &
E2B_PID=$!

run_full_suite "gemma-4:e4b" 11435 "gemma4_e4b_l50" 50 "${GEMMA4_REASONING_FLAGS[@]}"

# Wait for background E2B
wait $E2B_PID || log "WARNING: E2B suite had failures"

# Stop worker runtimes
stop_runtime "llama-gemma4-e4b"
stop_runtime "llama-gemma4-e2b"

log "=== PHASE 1 COMPLETE ==="

# ---- Phase 2: Brain models (sequential on GPU 0) ----
log ""
log "=== PHASE 2: Brain models (sequential on GPU 0) ==="

# Gemma 4 brains
start_runtime "llama-brain" "$SHARED/models/gemma-4-26b-a4b/gemma-4-26B-A4B-it-Q4_K_M.gguf" 11434 0 16384
wait_for_model 11434 300
run_full_suite "gemma-4:26b-a4b" 11434 "gemma4_26b_l50" 50 "${GEMMA4_REASONING_FLAGS[@]}"
stop_runtime "llama-brain"

start_runtime "llama-brain" "$SHARED/models/gemma-4-31b/gemma-4-31B-it-Q4_K_M.gguf" 11434 0 16384
wait_for_model 11434 300
run_full_suite "gemma-4:31b" 11434 "gemma4_31b_l50" 50 "${GEMMA4_REASONING_FLAGS[@]}"
stop_runtime "llama-brain"

# Qwen 3.6 brains
start_runtime "llama-brain" "$SHARED/models/qwen3.6-27b/Qwen3.6-27B-Q4_K_M.gguf" 11434 0 16384
wait_for_model 11434 300
run_full_suite "qwen3.6:27b" 11434 "qwen36_27b_l50" 50 "${QWEN36_REASONING_FLAGS[@]}"
stop_runtime "llama-brain"

start_runtime "llama-brain" "$SHARED/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" 11434 0 16384
wait_for_model 11434 300
run_full_suite "qwen3.6:35b-a3b" 11434 "qwen36_35b_l50" 50 "${QWEN36_REASONING_FLAGS[@]}"
stop_runtime "llama-brain"

log "=== PHASE 2 COMPLETE ==="

log ""
log "=========================================="
log "CAMPAIGN COMPLETE"
log "Log: $CAMPAIGN_LOG"
log "=========================================="
