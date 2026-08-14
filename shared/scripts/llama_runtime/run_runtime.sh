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
  --image IMAGE               Default: llama-runtime:b10333
  --host HOST                 Default: 127.0.0.1
  --ctx-size N                Default: 2048
  --n-gpu-layers N            Default: 999
  --batch-size N              Default: 128
  --threads N                 Default: host nproc
  --parallel N                Default: 1
  --tensor-split CSV          Comma-separated mask, for example 0,1,0,1,0,0
  --memory-limit SIZE         Docker --memory limit (e.g. 10g). Default: none (unlimited)
  --memory-swap SIZE          Docker --memory-swap limit (e.g. 12g). Default: none
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

IMAGE_TAG="${IMAGE_TAG:-llama-runtime:b10333}"
HOST="${HOST:-127.0.0.1}"
CTX_SIZE="${CTX_SIZE:-2048}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
BATCH_SIZE="${BATCH_SIZE:-128}"
THREADS="${THREADS:-$(nproc)}"
PARALLEL="${PARALLEL:-1}"
NAME=""
MODEL=""
PORT=""
GPUS=""
DOCKER_GPUS_ARG=""
TENSOR_SPLIT=""
MEMORY_LIMIT=""
MEMORY_SWAP=""
DRY_RUN=0
EXTRA_ARGS=()
GPU_DOCKER_ARGS=()
LLAMA_VISIBLE_DEVICES=()

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
    --memory-limit) MEMORY_LIMIT="${2:?}"; shift 2 ;;
    --memory-swap) MEMORY_SWAP="${2:?}"; shift 2 ;;
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

GPU_DOCKER_ARGS=(--gpus "${DOCKER_GPUS_ARG}")
if [[ "${GPUS}" == device=* ]]; then
  if [[ "${GPUS#device=}" == *,* ]]; then
    IFS=',' read -r -a HOST_GPU_IDS <<< "${GPUS#device=}"
    for idx in "${!HOST_GPU_IDS[@]}"; do
      LLAMA_VISIBLE_DEVICES+=("CUDA${idx}")
    done
  else
    LLAMA_VISIBLE_DEVICES=("CUDA0")
  fi
fi

# --- OOM protection ---
# Docker memory limits prevent a single container from consuming all system RAM.
# oom-score-adj=500 ensures the kernel targets llama-server before system services
# even if memory limits are accidentally omitted.
# Tune limits in gpu_llama.py model profiles (memory_limit / memory_swap keys).
#
# --- VRAM-pure defaults ---
# mmap (default):  file-backed pages are reclaimable under Docker cgroup pressure.
#                  --no-mmap was previously used here but caused OOM kills: it converts
#                  the GGUF file into non-reclaimable anonymous memory (malloc), which
#                  Docker counts as hard usage. With mmap, the kernel pages in only
#                  what's needed and reclaims under pressure. Tested: E4B (5GB GGUF)
#                  peaks at 1.6GB cgroup vs 6GB with --no-mmap. (2026-06-10)
# --flash-attn on: enables flash attention, reduces VRAM for KV cache operations
# -ctk q8_0:       quantize KV cache keys from f16 to q8_0 (halves KV cache size)
# -ctv q8_0:       quantize KV cache values from f16 to q8_0
DOCKER_MEMORY_ARGS=(--oom-score-adj 500)
if [ -n "${MEMORY_LIMIT}" ]; then
  DOCKER_MEMORY_ARGS+=(--memory "${MEMORY_LIMIT}")
else
  echo "WARNING: No --memory-limit set. Container can consume unlimited RAM." >&2
fi
if [ -n "${MEMORY_SWAP}" ]; then
  DOCKER_MEMORY_ARGS+=(--memory-swap "${MEMORY_SWAP}")
fi

CMD=(
  docker run --detach
  --name "${NAME}"
  "${GPU_DOCKER_ARGS[@]}"
  "${DOCKER_MEMORY_ARGS[@]}"
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
  --flash-attn on
  -ctk q8_0
  -ctv q8_0
)

if [ "${#LLAMA_VISIBLE_DEVICES[@]}" -gt 0 ]; then
  CMD+=(--device "$(IFS=,; echo "${LLAMA_VISIBLE_DEVICES[*]}")")
fi

if [ -n "${TENSOR_SPLIT}" ]; then
  CMD+=(--tensor-split "${TENSOR_SPLIT}")
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
