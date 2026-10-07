#!/usr/bin/env bash
#
# L100 Upgrade Campaign — bring all active models to limit 100 reasoning.
#
# Also reruns pipeline for Llama-3.2-3B and SmolLM3-3B (missing per-test scores).
#
# Phase 1: Gemma-4 E4B (GPU 1) + E2B (GPU 3) parallel — reasoning gsm8k,bbh l100
# Phase 2: Llama-3.2-3B (GPU 1) + SmolLM3-3B (GPU 3) parallel — pipeline rerun
# Phase 3: Phi-4-14B (GPU 4+5 split) parallel with brain models (GPU 0 sequential)
#   Brain sequence: Gemma-4 26B-A4B, 31B (gsm8k,bbh l100), Qwen3.6 27B, 35B-A3B (gsm8k,bbh,drop l100)
#
# Prerequisites:
#   - llama-runtime:b8884-candidate image built
#   - bench-pipeline, bench-reasoning docker images built
#   - All model GGUFs in /mnt/shared/models/
#   - No other runtimes on ports 11434, 11435, 11437, 11438
#
# Usage:
#   ssh 10.0.0.3 'nohup bash /mnt/shared/scripts/benchmarks/run_l100_upgrade_campaign.sh \
#     > /mnt/shared/logs/benchmarks/campaigns/l100_upgrade_live.log 2>&1 &'

set -euo pipefail

RUNTIME_IMAGE="llama-runtime:b8884-candidate"
SHARED="/mnt/shared"
LOG_DIR="$SHARED/logs/benchmarks"
CAMPAIGN_LOG="$LOG_DIR/campaigns/l100_upgrade_$(date +%Y%m%d_%H%M%S).log"
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
    log "  Stopping runtime: $name"
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

    log "  Loading $name on GPU $gpu port $port (ctx $ctx)..."
    bash "$SHARED/scripts/llama_runtime/run_runtime.sh" \
        --image "$RUNTIME_IMAGE" \
        --name "$name" \
        --model "$model" \
        --port "$port" \
        --gpus "device=$gpu" \
        --ctx-size "$ctx" \
        $extra_args
}

start_runtime_split() {
    local name=$1 model=$2 port=$3 gpu_a=$4 gpu_b=$5 ctx=${6:-8192}
    log "  Loading $name split on GPUs $gpu_a+$gpu_b port $port (ctx $ctx)..."
    bash "$SHARED/scripts/llama_runtime/run_runtime.sh" \
        --image "$RUNTIME_IMAGE" \
        --name "$name" \
        --model "$model" \
        --port "$port" \
        --gpus "device=$gpu_a,$gpu_b" \
        --ctx-size "$ctx" \
        --tensor-split "1,1"
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

run_reasoning() {
    local model_id=$1 port=$2 run_name=$3 limit=$4
    shift 4
    local extra_flags=("$@")
    local tasks="gsm8k,bbh,drop"

    log "  Running bench-reasoning: $model_id (tasks: $tasks, limit $limit) flags: ${extra_flags[*]:-none}"
    docker run --rm --network host \
        -e BENCHMARK_DISABLE_AUTO_RESERVE=1 \
        -v "$SHARED:$SHARED" \
        -v "$LOG_DIR/bench-reasoning/history:/results" \
        -v "$SHARED/plans/shoulders/benchmarking:/benchmark-scripts:ro" \
        bench-reasoning \
        --model "$model_id" \
        --runtime-base "http://localhost:${port}" \
        --tasks "$tasks" \
        --limit "$limit" \
        "${extra_flags[@]}" \
        --run-name "$run_name" 2>&1 | tail -5 | tee -a "$CAMPAIGN_LOG"
    log "  bench-reasoning done: $model_id (exit: ${PIPESTATUS[0]})"
}

run_reasoning_partial() {
    # Run only specific tasks (e.g. gsm8k,bbh without drop)
    local model_id=$1 port=$2 run_name=$3 limit=$4 tasks=$5
    shift 5
    local extra_flags=("$@")

    log "  Running bench-reasoning: $model_id (tasks: $tasks, limit $limit) flags: ${extra_flags[*]:-none}"
    docker run --rm --network host \
        -e BENCHMARK_DISABLE_AUTO_RESERVE=1 \
        -v "$SHARED:$SHARED" \
        -v "$LOG_DIR/bench-reasoning/history:/results" \
        -v "$SHARED/plans/shoulders/benchmarking:/benchmark-scripts:ro" \
        bench-reasoning \
        --model "$model_id" \
        --runtime-base "http://localhost:${port}" \
        --tasks "$tasks" \
        --limit "$limit" \
        "${extra_flags[@]}" \
        --run-name "$run_name" 2>&1 | tail -5 | tee -a "$CAMPAIGN_LOG"
    log "  bench-reasoning done: $model_id (exit: ${PIPESTATUS[0]})"
}

# ---- Per-model bench-reasoning flags ----
GEMMA4_REASONING_FLAGS=(--patch-think-tag-strip)
QWEN36_REASONING_FLAGS=(--patch-think-tag-strip)
# Phi-4 and non-thinking models: no extra flags
PHI4_REASONING_FLAGS=()

# ---- main campaign ----

log "=========================================="
log "L100 Upgrade Campaign"
log "Phase 1: Gemma-4 E4B + E2B reasoning (gsm8k,bbh l100)"
log "Phase 2: Llama-3.2-3B + SmolLM3-3B pipeline rerun"
log "Phase 3: Phi-4-14B reasoning (l100) parallel with brain models"
log "=========================================="

# ---- Phase 1: Gemma-4 workers reasoning upgrade (parallel GPU 1 + GPU 3) ----
log ""
log "=== PHASE 1: Gemma-4 E4B + E2B reasoning gsm8k,bbh l100 ==="

start_runtime "llama-e4b" "$SHARED/models/gemma-4-e4b/gemma-4-e4b-it-Q4_K_M.gguf" 11435 1 4096
start_runtime "llama-e2b" "$SHARED/models/gemma-4-e2b/gemma-4-e2b-it-Q8_0.gguf" 11437 3 4096

wait_for_model 11435 300
wait_for_model 11437 300

# gsm8k + bbh only (drop already at l100 for these models)
log "Starting parallel reasoning (gsm8k,bbh only)..."
(
    run_reasoning_partial "gemma-4:e2b" 11437 "gemma4_e2b_l100_gsmbbh" 100 "gsm8k,bbh" "${GEMMA4_REASONING_FLAGS[@]}"
) &
E2B_PID=$!

run_reasoning_partial "gemma-4:e4b" 11435 "gemma4_e4b_l100_gsmbbh" 100 "gsm8k,bbh" "${GEMMA4_REASONING_FLAGS[@]}"

wait $E2B_PID || log "WARNING: E2B reasoning had failures"

stop_runtime "llama-e4b"
stop_runtime "llama-e2b"

log "=== PHASE 1 COMPLETE ==="

# ---- Phase 2: Llama + SmolLM3 pipeline rerun (parallel GPU 1 + GPU 3) ----
log ""
log "=== PHASE 2: Llama-3.2-3B + SmolLM3-3B pipeline rerun ==="

start_runtime "llama-llama3" "$SHARED/models/llama-3.2-3b/Llama-3.2-3B-Instruct-Q4_K_M.gguf" 11435 1 16384
start_runtime "llama-smollm3" "$SHARED/models/smollm3-3b/SmolLM3-3B-Q4_K_M.gguf" 11437 3 16384

wait_for_model 11435 300
wait_for_model 11437 300

log "Starting parallel pipeline reruns..."
(
    run_pipeline "smollm3:3b" 11437 "smollm3_3b_pipeline_v2"
) &
SMOLLM_PID=$!

run_pipeline "llama3.2:3b" 11435 "llama32_3b_pipeline_v2"

wait $SMOLLM_PID || log "WARNING: SmolLM3 pipeline had failures"

stop_runtime "llama-llama3"
stop_runtime "llama-smollm3"

log "=== PHASE 2 COMPLETE ==="

# ---- Phase 3: Phi-4-14B (GPU 4+5) parallel with brain models (GPU 0) ----
log ""
log "=== PHASE 3: Phi-4-14B (GPU 4+5) + brain models (GPU 0) ==="

# Start Phi-4 split runtime on GPUs 4+5
start_runtime_split "llama-phi4-split" "$SHARED/models/phi-4-14b/phi-4-Q4_K_M.gguf" 11438 4 5 8192
wait_for_model 11438 300

# Run Phi-4 reasoning in background (all 3 tasks, l100)
log "Starting Phi-4-14B reasoning (gsm8k,bbh,drop l100) in background..."
(
    run_reasoning "phi-4:14b" 11438 "phi4_14b_l100" 100 "${PHI4_REASONING_FLAGS[@]}"
    log "Stopping Phi-4 runtime..."
    stop_runtime "llama-phi4-split"
) &
PHI4_PID=$!

# Brain models sequential on GPU 0 (runs in parallel with Phi-4)
log "Starting brain model sequence on GPU 0..."

# Gemma-4 26B-A4B: gsm8k + bbh only (drop already l100)
start_runtime "llama-brain" "$SHARED/models/gemma-4-26b-a4b/gemma-4-26B-A4B-it-Q4_K_M.gguf" 11434 0 16384
wait_for_model 11434 300
run_reasoning_partial "gemma-4:26b-a4b" 11434 "gemma4_26b_l100_gsmbbh" 100 "gsm8k,bbh" "${GEMMA4_REASONING_FLAGS[@]}"
stop_runtime "llama-brain"

# Gemma-4 31B: gsm8k + bbh only (drop already l100)
start_runtime "llama-brain" "$SHARED/models/gemma-4-31b/gemma-4-31B-it-Q4_K_M.gguf" 11434 0 16384
wait_for_model 11434 300
run_reasoning_partial "gemma-4:31b" 11434 "gemma4_31b_l100_gsmbbh" 100 "gsm8k,bbh" "${GEMMA4_REASONING_FLAGS[@]}"
stop_runtime "llama-brain"

# Qwen3.6 27B: all 3 tasks at l100
start_runtime "llama-brain" "$SHARED/models/qwen3.6-27b/Qwen3.6-27B-Q4_K_M.gguf" 11434 0 16384
wait_for_model 11434 300
run_reasoning "qwen3.6:27b" 11434 "qwen36_27b_l100" 100 "${QWEN36_REASONING_FLAGS[@]}"
stop_runtime "llama-brain"

# Qwen3.6 35B-A3B: all 3 tasks at l100
start_runtime "llama-brain" "$SHARED/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" 11434 0 16384
wait_for_model 11434 300
run_reasoning "qwen3.6:35b-a3b" 11434 "qwen36_35b_l100" 100 "${QWEN36_REASONING_FLAGS[@]}"
stop_runtime "llama-brain"

# Wait for Phi-4 background job
log "Waiting for Phi-4-14B to finish..."
wait $PHI4_PID || log "WARNING: Phi-4 reasoning had failures"

log "=== PHASE 3 COMPLETE ==="

log ""
log "=========================================="
log "L100 UPGRADE CAMPAIGN COMPLETE"
log "Log: $CAMPAIGN_LOG"
log ""
log "Results in:"
log "  Reasoning: $LOG_DIR/bench-reasoning/history/"
log "  Pipeline:  $LOG_DIR/bench-pipeline/history/"
log "=========================================="
