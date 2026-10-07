#!/usr/bin/env bash
#
# Sequential brain benchmark campaign: 4 models on GPU 0 (3090).
# Each model: load → code (humaneval+mbpp) → reasoning (gsm8k+bbh+drop limit 50) → unload → next.
#
# Usage:
#   ssh 10.0.0.3 'nohup bash /mnt/shared/scripts/benchmarks/run_brain_campaign.sh \
#     > /mnt/shared/logs/benchmarks/campaigns/brain_campaign.log 2>&1 &'

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

run_model() {
    local name=$1 model_path=$2 model_id=$3 tag=$4 ctx=${5:-16384}
    local extra_args=""

    if [[ "$model_path" == *"Qwen3.6"* ]]; then
        extra_args="--extra-arg --reasoning-budget --extra-arg 0"
    fi

    log "--- Loading $model_id ---"
    bash "$SHARED/scripts/llama_runtime/run_runtime.sh" \
        --image "$RUNTIME_IMAGE" \
        --name llama-brain \
        --model "$model_path" \
        --port 11434 \
        --gpus "device=0" \
        --ctx-size "$ctx" \
        $extra_args
    wait_for_model 11434 300

    log "  Running bench-code: $model_id"
    docker run --rm --network host \
        -e BENCHMARK_DISABLE_AUTO_RESERVE=1 \
        -v "$SHARED:$SHARED" \
        -v "$LOG_DIR/bench-code/history:/results" \
        -v "$SHARED/plans/shoulders/benchmarking:/benchmark-scripts:ro" \
        bench-code \
        --model "$model_id" \
        --runtime-base http://localhost:11434 \
        --tasks humaneval,mbpp \
        --run-name "${tag}_code"
    log "  bench-code done: $model_id"

    log "  Running bench-reasoning: $model_id (limit 50)"
    docker run --rm --network host \
        -e BENCHMARK_DISABLE_AUTO_RESERVE=1 \
        -v "$SHARED:$SHARED" \
        -v "$LOG_DIR/bench-reasoning/history:/results" \
        -v "$SHARED/plans/shoulders/benchmarking:/benchmark-scripts:ro" \
        bench-reasoning \
        --model "$model_id" \
        --runtime-base http://localhost:11434 \
        --tasks gsm8k,bbh,drop \
        --limit 50 \
        --run-name "${tag}_reasoning"
    log "  bench-reasoning done: $model_id"

    docker stop llama-brain 2>/dev/null || true
    sleep 3
    log "--- $model_id complete ---"
    echo
}

log "=========================================="
log "Brain Campaign: 4 models on GPU 0 (3090)"
log "Suites: code + reasoning (limit 50)"
log "=========================================="
echo

run_model "brain1" "$SHARED/models/gemma-4-26b-a4b/gemma-4-26B-A4B-it-Q4_K_M.gguf" "gemma-4:26b-a4b" "gemma4_26b_l50" 16384
run_model "brain2" "$SHARED/models/gemma-4-31b/gemma-4-31B-it-Q4_K_M.gguf" "gemma-4:31b" "gemma4_31b_l50" 16384
run_model "brain3" "$SHARED/models/qwen3.6-27b/Qwen3.6-27B-Q4_K_M.gguf" "qwen3.6:27b" "qwen36_27b_l50" 16384
run_model "brain4" "$SHARED/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" "qwen3.6:35b-a3b" "qwen36_35b_l50" 16384

log "=========================================="
log "BRAIN CAMPAIGN COMPLETE"
log "=========================================="
