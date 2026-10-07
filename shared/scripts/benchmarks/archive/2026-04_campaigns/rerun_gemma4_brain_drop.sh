#!/usr/bin/env bash
#
# Rerun DROP on Gemma 4 brain models (26B-A4B, 31B) with --patch-think-tag-strip.
# Sequential: load model, run DROP l100, stop, next.
#
# Usage:
#   ssh 10.0.0.3 'nohup bash /mnt/shared/scripts/benchmarks/rerun_gemma4_brain_drop.sh \
#     > /mnt/shared/logs/benchmarks/campaigns/gemma4_brain_drop.log 2>&1 &'

set -uo pipefail

RUNTIME_IMAGE="llama-runtime:b8884-candidate"
SHARED="/mnt/shared"
LOG_DIR="$SHARED/logs/benchmarks"
PORT=11434
GPU=0
CONTAINER="llama-brain"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_for_model() {
    local max_wait=${1:-300}
    local start=$SECONDS
    while (( SECONDS - start < max_wait )); do
        local code
        code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/v1/models" 2>/dev/null || echo "000")
        if [ "$code" = "200" ]; then
            log "  Model ready on port $PORT ($(( SECONDS - start ))s)"
            return 0
        fi
        sleep 5
    done
    log "  ERROR: Model not ready on port $PORT after ${max_wait}s"
    return 1
}

run_drop() {
    local model_path=$1 model_id=$2 tag=$3 ctx=${4:-16384}

    log "--- Loading $model_id on GPU $GPU port $PORT ---"
    docker stop "$CONTAINER" 2>/dev/null || true
    sleep 2
    bash "$SHARED/scripts/llama_runtime/run_runtime.sh" \
        --image "$RUNTIME_IMAGE" \
        --name "$CONTAINER" \
        --model "$model_path" \
        --port "$PORT" \
        --gpus "device=$GPU" \
        --ctx-size "$ctx"
    wait_for_model 300

    log "  Running DROP: $model_id (limit 100, --patch-think-tag-strip)"
    if docker run --rm --network host \
        -e BENCHMARK_DISABLE_AUTO_RESERVE=1 \
        -v "$SHARED:$SHARED" \
        -v "$LOG_DIR/bench-reasoning/history:/results" \
        -v "$SHARED/plans/shoulders/benchmarking:/benchmark-scripts:ro" \
        bench-reasoning \
        --model "$model_id" \
        --runtime-base "http://localhost:${PORT}" \
        --tasks drop \
        --limit 100 \
        --patch-think-tag-strip \
        --run-name "${tag}_tts_drop_l100"; then
        log "  DROP done: $model_id"
    else
        log "  WARNING: DROP failed for $model_id (exit $?), continuing..."
    fi

    docker stop "$CONTAINER" 2>/dev/null || true
    sleep 3
    log "--- $model_id complete ---"
    echo
}

log "=========================================="
log "Gemma 4 Brain DROP rerun: --patch-think-tag-strip"
log "2 models, limit 100"
log "=========================================="
echo

run_drop "$SHARED/models/gemma-4-26b-a4b/gemma-4-26B-A4B-it-Q4_K_M.gguf" "gemma-4:26b-a4b" "gemma4_26b" 16384
run_drop "$SHARED/models/gemma-4-31b/gemma-4-31B-it-Q4_K_M.gguf" "gemma-4:31b" "gemma4_31b" 16384

log "=========================================="
log "Gemma 4 Brain DROP rerun COMPLETE"
log "=========================================="
