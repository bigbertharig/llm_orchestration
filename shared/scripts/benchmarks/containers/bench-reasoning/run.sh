#!/bin/bash
set -euo pipefail

# Defaults
MODEL=""
OLLAMA_BASE="http://localhost:11436"
TASKS="gsm8k,bbh,drop,mmlu_pro,ifeval"
LIMIT="150"
RESULTS_DIR="/results"
NUM_FEWSHOT=""
TOKENIZER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --ollama-base) OLLAMA_BASE="$2"; shift 2 ;;
        --tasks) TASKS="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        --num-fewshot) NUM_FEWSHOT="$2"; shift 2 ;;
        --tokenizer) TOKENIZER="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "ERROR: --model is required (e.g. --model qwen2.5-coder:7b)"
    exit 1
fi

# Derive model name for output directory
MODEL_SAFE=$(echo "$MODEL" | tr ':/' '_')

echo "=== bench-reasoning ==="
echo "Model: $MODEL"
echo "Ollama: $OLLAMA_BASE"
echo "Tasks: $TASKS"
echo "Results: $RESULTS_DIR"
[ -n "$LIMIT" ] && echo "Limit: $LIMIT samples per task"

# Verify Ollama is reachable
if ! curl -s "${OLLAMA_BASE}/api/tags" > /dev/null 2>&1; then
    echo "ERROR: Cannot reach Ollama at ${OLLAMA_BASE}"
    echo "Make sure Ollama is running and --network host is set"
    exit 1
fi

# Verify model is loaded
if ! curl -s "${OLLAMA_BASE}/api/tags" | python3 -c "
import sys, json
data = json.load(sys.stdin)
models = [m['name'] for m in data.get('models', [])]
if '${MODEL}' not in models:
    print(f'ERROR: Model ${MODEL} not found in Ollama. Available: {models}')
    sys.exit(1)
print(f'Model ${MODEL} found in Ollama')
"; then
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${RESULTS_DIR}/bench-reasoning_${MODEL_SAFE}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# Build lm_eval command
CMD="lm_eval"
CMD="$CMD --model local-chat-completions"
CMD="$CMD --model_args model=${MODEL},base_url=${OLLAMA_BASE}/v1/chat/completions,num_concurrent=1,max_retries=3,tokenized_requests=False"
CMD="$CMD --tasks $TASKS"
CMD="$CMD --output_path $OUTPUT_DIR"
CMD="$CMD --apply_chat_template"
CMD="$CMD --batch_size 1"

if [ -n "$LIMIT" ]; then
    CMD="$CMD --limit $LIMIT"
fi

if [ -n "$NUM_FEWSHOT" ]; then
    CMD="$CMD --num_fewshot $NUM_FEWSHOT"
fi

echo ""
echo "Running: $CMD"
echo ""
eval $CMD
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "=== bench-reasoning COMPLETE ==="
    echo "Results in: $OUTPUT_DIR"
    # Print summary
    python3 -c "
import json, glob
for f in sorted(glob.glob('${OUTPUT_DIR}/results*.json')):
    data = json.load(open(f))
    results = data.get('results', {})
    for task, metrics in sorted(results.items()):
        # Try common metric names
        for key in ['exact_match,strict-match', 'exact_match,flexible-extract', 'acc,none', 'acc_norm,none']:
            if key in metrics:
                print(f'  {task}: {key.split(\",\")[0]}={metrics[key]}')
                break
        else:
            # Print first metric
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and not k.endswith('stderr'):
                    print(f'  {task}: {k}={v}')
                    break
" 2>/dev/null || echo "(could not parse results)"
else
    echo "=== bench-reasoning FAILED (exit code $EXIT_CODE) ==="
fi

exit $EXIT_CODE
