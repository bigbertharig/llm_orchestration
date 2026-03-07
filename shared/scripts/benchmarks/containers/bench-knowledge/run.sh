#!/bin/bash
set -euo pipefail

GGUF_PATH="${1:?Usage: bench-knowledge <path-to-gguf> [--tasks TASKS] [--limit N]}"
shift

# Parse optional args
TASKS="mmlu,arc_challenge,hellaswag,truthfulqa_mc2,boolq"
LIMIT="150"
RESULTS_DIR="/results"
MODEL_NAME="unknown"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks) TASKS="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        --model-name) MODEL_NAME="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Derive model name from GGUF filename if not provided
if [ "$MODEL_NAME" = "unknown" ]; then
    MODEL_NAME=$(basename "$(dirname "$GGUF_PATH")")
fi

echo "=== bench-knowledge ==="
echo "GGUF: $GGUF_PATH"
echo "Tasks: $TASKS"
echo "Model: $MODEL_NAME"
echo "Results: $RESULTS_DIR"
[ -n "$LIMIT" ] && echo "Limit: $LIMIT samples per task"

# Verify GGUF exists
if [ ! -f "$GGUF_PATH" ]; then
    echo "ERROR: GGUF file not found: $GGUF_PATH"
    exit 1
fi

# Start llama.cpp server in background
echo ""
echo "Starting llama.cpp server..."
python3 -m llama_cpp.server \
    --model "$GGUF_PATH" \
    --n_gpu_layers -1 \
    --host 0.0.0.0 \
    --port 8000 &
SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server..."
for i in $(seq 1 60); do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server process died"
        exit 1
    fi
    sleep 1
done

# Verify server is actually responding
if ! curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "ERROR: Server failed to start within 60s"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# Build lm_eval command
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${RESULTS_DIR}/bench-knowledge_${MODEL_NAME}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

CMD="lm_eval --model gguf --model_args base_url=http://localhost:8000"
CMD="$CMD --tasks $TASKS"
CMD="$CMD --output_path $OUTPUT_DIR"
CMD="$CMD --batch_size 1"

if [ -n "$LIMIT" ]; then
    CMD="$CMD --limit $LIMIT"
fi

echo ""
echo "Running: $CMD"
echo ""
eval $CMD
EXIT_CODE=$?

# Cleanup
echo ""
echo "Stopping server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "=== bench-knowledge COMPLETE ==="
    echo "Results in: $OUTPUT_DIR"
    # Print summary if results exist
    if ls "$OUTPUT_DIR"/results*.json 1>/dev/null 2>&1; then
        echo ""
        echo "--- Results Summary ---"
        python3 -c "
import json, glob
for f in sorted(glob.glob('${OUTPUT_DIR}/results*.json')):
    data = json.load(open(f))
    results = data.get('results', {})
    for task, metrics in sorted(results.items()):
        acc = metrics.get('acc,none', metrics.get('acc_norm,none', 'N/A'))
        print(f'  {task}: {acc}')
" 2>/dev/null || echo "(could not parse results)"
    fi
else
    echo "=== bench-knowledge FAILED (exit code $EXIT_CODE) ==="
fi

exit $EXIT_CODE
