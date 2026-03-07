#!/bin/bash
set -euo pipefail

MODEL=""
OLLAMA_BASE="http://localhost:11436"
TASKS="humaneval,mbpp"
RESULTS_DIR="/results"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --ollama-base) OLLAMA_BASE="$2"; shift 2 ;;
        --tasks) TASKS="$2"; shift 2 ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "ERROR: --model is required (e.g. --model qwen2.5-coder:7b)"
    exit 1
fi

MODEL_SAFE=$(echo "$MODEL" | tr ':/' '_')

echo "=== bench-code ==="
echo "Model: $MODEL"
echo "Ollama: $OLLAMA_BASE"
echo "Tasks: $TASKS"
echo "Results: $RESULTS_DIR"

# Verify Ollama is reachable
if ! curl -s "${OLLAMA_BASE}/api/tags" > /dev/null 2>&1; then
    echo "ERROR: Cannot reach Ollama at ${OLLAMA_BASE}"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${RESULTS_DIR}/bench-code_${MODEL_SAFE}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# evalplus uses OPENAI_BASE_URL to talk to Ollama
export OPENAI_BASE_URL="${OLLAMA_BASE}/v1"
export OPENAI_API_KEY="ollama"

IFS=',' read -ra TASK_ARRAY <<< "$TASKS"

for TASK in "${TASK_ARRAY[@]}"; do
    echo ""
    echo "--- Running ${TASK} ---"

    # evalplus codegen: positional args are MODEL DATASET
    # Note: evalplus evaluate requires ALL problems — no partial runs
    GEN_CMD="python3 -m evalplus.codegen ${MODEL} ${TASK} --backend openai --greedy --root ${OUTPUT_DIR}"

    echo "Generating: $GEN_CMD"
    eval $GEN_CMD

    # Find the generated samples file
    SAMPLE_DIR="${OUTPUT_DIR}/${TASK}"
    SAMPLE_FILE=$(find "$SAMPLE_DIR" -name "*.jsonl" ! -name "*.raw.jsonl" 2>/dev/null | head -1)

    if [ -z "$SAMPLE_FILE" ]; then
        echo "ERROR: No sample file found in $SAMPLE_DIR"
        continue
    fi

    # Evaluate generated samples
    echo "Evaluating: python3 -m evalplus.evaluate --dataset ${TASK} --samples ${SAMPLE_FILE}"
    python3 -m evalplus.evaluate --dataset "$TASK" --samples "$SAMPLE_FILE"

    echo "--- ${TASK} done ---"
done

echo ""
echo "=== bench-code COMPLETE ==="
echo "Results in: $OUTPUT_DIR"

# Summarize results
python3 -c "
import json, glob, os
for f in sorted(glob.glob('${OUTPUT_DIR}/**/*_eval_results.json', recursive=True)):
    data = json.load(open(f))
    dataset = os.path.basename(os.path.dirname(f))
    base = data.get('pass@1', {}).get('base', 'N/A')
    plus = data.get('pass@1', {}).get('plus', 'N/A')
    print(f'  {dataset}: pass@1 base={base}, plus={plus}')
" 2>/dev/null || echo "(could not parse results)"
