#!/usr/bin/env bash
#
# Run remaining brain models after 26B-A4B finishes.
# Expects 26B-A4B to be done and llama-brain-manual stopped.
#
# Usage:
#   ssh 10.0.0.3 'nohup bash /mnt/shared/scripts/benchmarks/run_remaining_brains.sh \
#     > /mnt/shared/logs/benchmarks/campaigns/remaining_brains.log 2>&1 &'

set -euo pipefail

RUNTIME_IMAGE="llama-runtime:b8884-candidate"
SHARED="/mnt/shared"
LOG_DIR="$SHARED/logs/benchmarks"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

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
        --run-name "$run_name"
    log "  bench-code done: $model_id (exit: $?)"
}

run_reasoning() {
    local model_id=$1 port=$2 run_name=$3 limit=${4:-50}
    log "  Running bench-reasoning: $model_id (gsm8k + bbh + drop, limit $limit)"
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
        --run-name "$run_name"
    log "  bench-reasoning done: $model_id (exit: $?)"
}

# ---- main ----

log "=== Remaining Brain Models (GPU 0) ==="

# Ensure nothing on port 11434
stop_runtime "llama-brain-manual"
stop_runtime "llama-brain"

# Model 1: Gemma 4 31B (dense)
log ""
log "--- Brain 1/3: Gemma 4 31B ---"
start_runtime "llama-brain" "$SHARED/models/gemma-4-31b/gemma-4-31B-it-Q4_K_M.gguf" 11434 0 16384
wait_for_model 11434 300
run_code "gemma-4:31b" 11434 "gemma4_31b_l50_code"
run_reasoning "gemma-4:31b" 11434 "gemma4_31b_l50_reasoning" 50
stop_runtime "llama-brain"

# Model 2: Qwen 3.6 27B (dense)
log ""
log "--- Brain 2/3: Qwen 3.6 27B ---"
start_runtime "llama-brain" "$SHARED/models/qwen3.6-27b/Qwen3.6-27B-Q4_K_M.gguf" 11434 0 16384
wait_for_model 11434 300
run_code "qwen3.6:27b" 11434 "qwen36_27b_l50_code"
run_reasoning "qwen3.6:27b" 11434 "qwen36_27b_l50_reasoning" 50
stop_runtime "llama-brain"

# Model 3: Qwen 3.6 35B-A3B (MoE)
log ""
log "--- Brain 3/3: Qwen 3.6 35B-A3B ---"
start_runtime "llama-brain" "$SHARED/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" 11434 0 16384
wait_for_model 11434 300
run_code "qwen3.6:35b-a3b" 11434 "qwen36_35b_l50_code"
run_reasoning "qwen3.6:35b-a3b" 11434 "qwen36_35b_l50_reasoning" 50
stop_runtime "llama-brain"

log ""
log "=== ALL BRAIN MODELS COMPLETE ==="
