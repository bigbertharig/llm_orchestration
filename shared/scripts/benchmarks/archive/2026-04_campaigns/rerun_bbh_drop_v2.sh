#!/usr/bin/env bash
#
# Rerun BBH + DROP on all Gemma 4 / Qwen 3.6 models with correct flags.
#
# Gemma 4: patched image alone (no \n\n stop removed, \n DROP stop) is sufficient.
# Qwen 3.6: needs --patch-think-tag-strip to remove </think> tags from content.
#
# Usage:
#   ssh 10.0.0.3 'nohup bash /mnt/shared/scripts/benchmarks/rerun_bbh_drop_v2.sh \
#     > /mnt/shared/logs/benchmarks/campaigns/bbh_drop_rerun_v2.log 2>&1 &'

set -uo pipefail

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

run_bbh_drop() {
    local model_path=$1 model_id=$2 tag=$3 port=$4 gpu=$5 container=$6 ctx=${7:-16384}
    shift 7
    local extra_bench_flags=("$@")
    local runtime_extra_args=""

    if [[ "$model_path" == *"Qwen3.6"* ]]; then
        runtime_extra_args="--extra-arg --reasoning-budget --extra-arg 0"
    fi

    log "--- Loading $model_id on GPU $gpu port $port ---"
    docker stop "$container" 2>/dev/null || true
    sleep 2
    bash "$SHARED/scripts/llama_runtime/run_runtime.sh" \
        --image "$RUNTIME_IMAGE" \
        --name "$container" \
        --model "$model_path" \
        --port "$port" \
        --gpus "device=$gpu" \
        --ctx-size "$ctx" \
        $runtime_extra_args
    wait_for_model "$port" 300

    log "  Running BBH+DROP: $model_id (limit 50) flags: ${extra_bench_flags[*]:-none}"
    if docker run --rm --network host \
        -e BENCHMARK_DISABLE_AUTO_RESERVE=1 \
        -v "$SHARED:$SHARED" \
        -v "$LOG_DIR/bench-reasoning/history:/results" \
        -v "$SHARED/plans/shoulders/benchmarking:/benchmark-scripts:ro" \
        bench-reasoning \
        --model "$model_id" \
        --runtime-base "http://localhost:${port}" \
        --tasks bbh,drop \
        --limit 50 \
        "${extra_bench_flags[@]}" \
        --run-name "${tag}_bbhdrop_v2"; then
        log "  BBH+DROP done: $model_id"
    else
        log "  WARNING: BBH+DROP failed for $model_id (exit $?), continuing..."
    fi

    docker stop "$container" 2>/dev/null || true
    sleep 3
    log "--- $model_id complete ---"
    echo
}

log "=========================================="
log "BBH+DROP Rerun v2: correct thinking-mode flags"
log "6 models, limit 50"
log "=========================================="
echo

# --- Brain models on GPU 0 ---
# Gemma 4 brain: no extra flags needed (patched image handles it)
run_bbh_drop "$SHARED/models/gemma-4-26b-a4b/gemma-4-26B-A4B-it-Q4_K_M.gguf" "gemma-4:26b-a4b" "gemma4_26b_l50" 11434 0 llama-brain 16384
run_bbh_drop "$SHARED/models/gemma-4-31b/gemma-4-31B-it-Q4_K_M.gguf" "gemma-4:31b" "gemma4_31b_l50" 11434 0 llama-brain 16384

# Qwen 3.6 brain: needs --patch-think-tag-strip
run_bbh_drop "$SHARED/models/qwen3.6-27b/Qwen3.6-27B-Q4_K_M.gguf" "qwen3.6:27b" "qwen36_27b_l50" 11434 0 llama-brain 16384 --patch-think-tag-strip
run_bbh_drop "$SHARED/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" "qwen3.6:35b-a3b" "qwen36_35b_l50" 11434 0 llama-brain 16384 --patch-think-tag-strip

# --- Worker models on GPU 1 ---
# Gemma 4 workers: no extra flags needed
run_bbh_drop "$SHARED/models/gemma-4-e4b/gemma-4-e4b-it-Q4_K_M.gguf" "gemma-4:e4b" "gemma4_e4b_l50" 11435 1 llama-worker-gpu-1 8192
run_bbh_drop "$SHARED/models/gemma-4-e2b/gemma-4-e2b-it-Q8_0.gguf" "gemma-4:e2b" "gemma4_e2b_l50" 11435 1 llama-worker-gpu-1 8192

log "=========================================="
log "BBH+DROP RERUN v2 COMPLETE"
log "=========================================="
