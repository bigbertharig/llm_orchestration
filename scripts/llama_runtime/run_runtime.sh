#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_runtime.sh --name NAME --model GGUF_PATH --port PORT --gpus GPU_SPEC [options]

Required:
  --name NAME                 Container name
  --model GGUF_PATH           GGUF path mounted from /mnt/shared/models
  --port PORT                 Host port on 127.0.0.1
  --gpus GPU_SPEC             docker --gpus value, for example:
                              device=1
                              all

Optional:
  --image IMAGE               Default: llama-runtime:sm61-sm86
  --host HOST                 Default: 127.0.0.1
  --ctx-size N                Default: 2048
  --n-gpu-layers N            Default: -1
  --batch-size N              Default: 512
  --threads N                 Default: host nproc
  --parallel N                Default: 1
  --tensor-split CSV          Comma-separated mask, for example 0,1,0,1,0,0
  --extra-arg ARG             Repeatable extra llama-server arg
  --dry-run                   Print command only

Example:
  run_runtime.sh \
    --name llama-worker-gpu1 \
    --model /mnt/shared/models/qwen2.5-coder-7b/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf \
    --port 11436 \
    --gpus device=1
EOF
}

IMAGE_TAG="${IMAGE_TAG:-llama-runtime:sm61-sm86}"
HOST="${HOST:-127.0.0.1}"
CTX_SIZE="${CTX_SIZE:-2048}"
N_GPU_LAYERS="${N_GPU_LAYERS:--1}"
BATCH_SIZE="${BATCH_SIZE:-512}"
THREADS="${THREADS:-$(nproc)}"
PARALLEL="${PARALLEL:-1}"
NAME=""
MODEL=""
PORT=""
GPUS=""
DOCKER_GPUS_ARG=""
TENSOR_SPLIT=""
DRY_RUN=0
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --name) NAME="${2:?}"; shift 2 ;;
    --model) MODEL="${2:?}"; shift 2 ;;
    --port) PORT="${2:?}"; shift 2 ;;
    --gpus) GPUS="${2:?}"; shift 2 ;;
    --image) IMAGE_TAG="${2:?}"; shift 2 ;;
    --host) HOST="${2:?}"; shift 2 ;;
    --ctx-size) CTX_SIZE="${2:?}"; shift 2 ;;
    --n-gpu-layers) N_GPU_LAYERS="${2:?}"; shift 2 ;;
    --batch-size) BATCH_SIZE="${2:?}"; shift 2 ;;
    --threads) THREADS="${2:?}"; shift 2 ;;
    --parallel) PARALLEL="${2:?}"; shift 2 ;;
    --tensor-split) TENSOR_SPLIT="${2:?}"; shift 2 ;;
    --extra-arg) EXTRA_ARGS+=("${2:?}"); shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage >&2; exit 64 ;;
  esac
done

if [ -z "${NAME}" ] || [ -z "${MODEL}" ] || [ -z "${PORT}" ] || [ -z "${GPUS}" ]; then
  echo "Missing required args" >&2
  usage >&2
  exit 64
fi

if [ ! -f "${MODEL}" ]; then
  echo "Model not found: ${MODEL}" >&2
  exit 66
fi

if ! [[ "${PORT}" =~ ^[0-9]+$ ]]; then
  echo "Port must be numeric: ${PORT}" >&2
  exit 64
fi

DOCKER_GPUS_ARG="${GPUS}"
if [[ "${GPUS}" == device=* ]] && [[ "${GPUS#device=}" == *,* ]]; then
  # Docker expects a quoted device request for multi-GPU selections.
  DOCKER_GPUS_ARG="\"${GPUS}\""
fi

CMD=(
  docker run --rm --detach
  --name "${NAME}"
  --gpus "${DOCKER_GPUS_ARG}"
  --network host
  -v "$(dirname "${MODEL}")":"$(dirname "${MODEL}")":ro
  "${IMAGE_TAG}"
  llama-server
  --model "${MODEL}"
  --host "${HOST}"
  --port "${PORT}"
  --ctx-size "${CTX_SIZE}"
  --n-gpu-layers "${N_GPU_LAYERS}"
  --batch-size "${BATCH_SIZE}"
  --threads "${THREADS}"
  --parallel "${PARALLEL}"
)

if [ -n "${TENSOR_SPLIT}" ]; then
  IFS=',' read -r -a SPLIT_PARTS <<< "${TENSOR_SPLIT}"
  CMD+=(--tensor-split "${SPLIT_PARTS[@]}")
fi

if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

if [ "${DRY_RUN}" -eq 1 ]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}"
