#!/bin/bash
set -euo pipefail

MODEL=""
OLLAMA_BASE="http://localhost:11436"
RESULTS_DIR="/results"
SCRIPTS_DIR="/benchmark-scripts"
TESTS="custom_json_schema_strict,custom_command_safety,custom_ambiguity_handling,custom_tool_plan_sequence,custom_orchestration_tradeoff,custom_long_context_extract"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --ollama-base) OLLAMA_BASE="$2"; shift 2 ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        --scripts-dir) SCRIPTS_DIR="$2"; shift 2 ;;
        --tests) TESTS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "ERROR: --model is required (e.g. --model qwen2.5-coder:7b)"
    exit 1
fi

MODEL_SAFE=$(echo "$MODEL" | tr ':/' '_')

echo "=== bench-pipeline ==="
echo "Model: $MODEL"
echo "Ollama: $OLLAMA_BASE"
echo "Tests: $TESTS"
echo "Results: $RESULTS_DIR"
echo "Scripts: $SCRIPTS_DIR"

# Verify Ollama is reachable
if ! curl -s "${OLLAMA_BASE}/api/tags" > /dev/null 2>&1; then
    echo "ERROR: Cannot reach Ollama at ${OLLAMA_BASE}"
    exit 1
fi

# Verify the custom runner and cases exist
RUNNER="${SCRIPTS_DIR}/run_local_custom_task.py"
CASES="${SCRIPTS_DIR}/custom_tasks/cases.json"
CATALOG="${SCRIPTS_DIR}/benchmark_catalog.json"

if [ ! -f "$RUNNER" ]; then
    echo "ERROR: Custom test runner not found: $RUNNER"
    echo "Mount the benchmarks scripts dir with -v /mnt/shared/scripts/benchmarks:/benchmark-scripts:ro"
    exit 1
fi

IFS=',' read -ra TEST_ARRAY <<< "$TESTS"
TOTAL_TESTS=${#TEST_ARRAY[@]}
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

for TEST_ID in "${TEST_ARRAY[@]}"; do
    echo ""
    echo "--- Running ${TEST_ID} ---"

    python3 "$RUNNER" \
        --id "$TEST_ID" \
        --model "$MODEL" \
        --base-url "$OLLAMA_BASE" \
        --catalog "$CATALOG" \
        --cases "$CASES" \
        --output-dir "$RESULTS_DIR" \
        --suite "bench-pipeline" \
        --no-record \
        && PASSED_TESTS=$((PASSED_TESTS + 1)) \
        || FAILED_TESTS=$((FAILED_TESTS + 1))

    echo "--- ${TEST_ID} done ---"
done

echo ""
echo "=== bench-pipeline COMPLETE ==="
echo "Passed: ${PASSED_TESTS}/${TOTAL_TESTS}"
if [ $FAILED_TESTS -gt 0 ]; then
    echo "Failed: ${FAILED_TESTS}/${TOTAL_TESTS}"
fi
