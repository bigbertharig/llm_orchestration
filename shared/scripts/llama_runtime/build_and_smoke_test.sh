#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  build_and_smoke_test.sh --name NAME --model GGUF_PATH --port PORT --gpus GPU_SPEC [run options]

This builds the dedicated image, starts one runtime, waits for readiness, runs
chat and completions smoke tests, then stops the runtime.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NAME=""
MODEL=""
PORT=""
GPUS=""
TIMEOUT="${TIMEOUT:-120}"
RUN_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --name) NAME="${2:?}"; RUN_ARGS+=("$1" "$2"); shift 2 ;;
    --model) MODEL="${2:?}"; RUN_ARGS+=("$1" "$2"); shift 2 ;;
    --port) PORT="${2:?}"; RUN_ARGS+=("$1" "$2"); shift 2 ;;
    --gpus) GPUS="${2:?}"; RUN_ARGS+=("$1" "$2"); shift 2 ;;
    --timeout) TIMEOUT="${2:?}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) RUN_ARGS+=("$1"); shift ;;
  esac
done

if [ -z "${NAME}" ] || [ -z "${MODEL}" ] || [ -z "${PORT}" ] || [ -z "${GPUS}" ]; then
  echo "Missing required args" >&2
  usage >&2
  exit 64
fi

cleanup() {
  "${SCRIPT_DIR}/stop_runtime.sh" "${NAME}" >/dev/null 2>&1 || true
}

trap cleanup EXIT

"${SCRIPT_DIR}/build_image.sh"
"${SCRIPT_DIR}/run_runtime.sh" "${RUN_ARGS[@]}" >/dev/null
"${SCRIPT_DIR}/smoke_test.sh" --port "${PORT}" --timeout "${TIMEOUT}"
