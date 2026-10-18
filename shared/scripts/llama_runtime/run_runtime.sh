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
  --image IMAGE               Default: the model library runtime_image
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
  --cache-ram MIB             llama-server host prompt cache. Default: 0 (off; see below)
  --lease LEASE_ID            Rig lease this runtime runs under (audited, see below)
  --dry-run                   Print command only

Sequential loading (rig rule, not optional):
  Only one model loads at a time on this machine. The script takes a machine-wide
  lock, waits until every other runtime it started is serving, starts the container,
  and returns only when this one answers /health (or after removing it on failure).
  Callers must not put a short timeout on this script: killing it mid-load frees the
  lock while the model is still loading. Stdout is the container id, as before.

  LLAMA_LOAD_LOCK             Lock file. Default: /run/lock/llama-model-load.lock
  LLAMA_LOAD_LOCK_WAIT_S      Max wait for the lock. Default: 3600
  LLAMA_LOAD_READY_TIMEOUT_S  Max wait for this model, and for others to finish loading. Default: 900
  LLAMA_LOAD_SETTLE_S         Hold the lock this long after ready. Default: 5

Rig guard (scripts/rig/hw_guard.py, rig-guard.service):
  With the lock held, the load waits until the guard reads ok (power, temperature, memory), up to
  LLAMA_GUARD_WAIT_S (default 600), and is refused (exit 75) if it does not. A guard that is not
  running admits with a warning. LLAMA_GUARD=off skips the check (emergencies only).

Host prompt cache:
  llama-server keeps up to 8192 MiB of past prompt state in system RAM by default, per server.
  System RAM is the rig's scarce resource (2026-10-17: five default caches filled 32 GB and hung
  the rig), so the floor here is 0. The harness picks a cache per use and passes --cache-ram.
  A --cache-ram given through --extra-arg also wins.

Lease audit:
  Every load is logged to <RIG_HW_DIR>/lease_audit.log with its lease, or "no-lease", and
  whether the lease holds these GPUs and this port. Audit only for now; nothing is refused.
  RIG_HW_DIR (rig hardware state) defaults to the rig-local /home/bryan/llm_orchestration/shared/brain/hw.

Example:
  run_runtime.sh \
    --name llama-worker-gpu1 \
    --model /mnt/shared/models/qwen2.5-coder-7b/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf \
    --port 11436 \
    --gpus device=1
EOF
}

# Default image: the model library's runtime_image (one image for every model).
IMAGE_TAG="${IMAGE_TAG:-}"
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
CACHE_RAM_MIB="${LLAMA_CACHE_RAM_MIB:-0}"
LEASE=""
EXTRA_ARGS=()
TREE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GUARD_SCRIPT="${TREE}/scripts/rig/hw_guard.py"
GUARD_PYTHON="${LLAMA_GUARD_PYTHON:-/home/bryan/llm-orchestration-venv/bin/python}"
[ -x "${GUARD_PYTHON}" ] || GUARD_PYTHON="python3"
GUARD_WAIT_S="${LLAMA_GUARD_WAIT_S:-600}"
# The rig's hardware state (guard, lease record, audit) lives in one rig-local place, whichever copy
# of this script runs (rig checkout, or the shared drive's copy that lanes and the stack use).
if [ -z "${RIG_HW_DIR:-}" ]; then
  RIG_HW_DIR="/home/bryan/llm_orchestration/shared/brain/hw"
  [ -d "${RIG_HW_DIR}" ] || RIG_HW_DIR="${TREE}/brain/hw"
fi
export RIG_HW_DIR
LEASE_RECORD="${LLAMA_LEASE_RECORD:-${RIG_HW_DIR}/gpus/leases.json}"
LEASE_AUDIT_LOG="${LLAMA_LEASE_AUDIT_LOG:-${RIG_HW_DIR}/lease_audit.log}"
GPU_DOCKER_ARGS=()
LLAMA_VISIBLE_DEVICES=()
DOCKER_BIN="${DOCKER_BIN:-docker}"
LOAD_LOCK="${LLAMA_LOAD_LOCK:-/run/lock/llama-model-load.lock}"
LOAD_LOCK_WAIT_S="${LLAMA_LOAD_LOCK_WAIT_S:-3600}"
LOAD_READY_TIMEOUT_S="${LLAMA_LOAD_READY_TIMEOUT_S:-900}"
LOAD_SETTLE_S="${LLAMA_LOAD_SETTLE_S:-5}"
# Every container this script starts carries these labels, so a later load can find
# runtimes that are still loading (for example after a launcher was killed).
RUNTIME_LABEL="org.llm-orchestration.llama-runtime"

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
    --cache-ram) CACHE_RAM_MIB="${2:?}"; shift 2 ;;
    --lease) LEASE="${2:?}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage >&2; exit 64 ;;
  esac
done

if [ -z "${IMAGE_TAG}" ]; then
  LIBRARY="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/plans/shoulders/benchmarking/scripts/active/model_library.py"
  IMAGE_TAG="$(python3 "${LIBRARY}" runtime-image 2>/dev/null || true)"
  if [ -z "${IMAGE_TAG}" ]; then
    echo "No --image given and the model library runtime_image could not be read (${LIBRARY})" >&2
    exit 66
  fi
fi

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
# -lv 4:           llama.cpp b10333+ defaults to verbosity 3, which drops the
#                  "load_tensors: offloaded N/N layers to GPU" line. gpu_llama.py
#                  _assert_full_gpu_offload() requires it; without it every load
#                  times out at the readiness budget. (2026-10-06)
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
  "${DOCKER_BIN}" run --detach
  --name "${NAME}"
  --label "${RUNTIME_LABEL}=1"
  --label "${RUNTIME_LABEL}.health=http://${HOST}:${PORT}/health"
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
  -lv 4
)

if [ "${#LLAMA_VISIBLE_DEVICES[@]}" -gt 0 ]; then
  CMD+=(--device "$(IFS=,; echo "${LLAMA_VISIBLE_DEVICES[*]}")")
fi

if [ -n "${TENSOR_SPLIT}" ]; then
  CMD+=(--tensor-split "${TENSOR_SPLIT}")
fi

if ! [[ "${CACHE_RAM_MIB}" =~ ^-?[0-9]+$ ]]; then
  echo "--cache-ram must be an integer MiB: ${CACHE_RAM_MIB}" >&2
  exit 64
fi
cache_in_extra=0
for arg in "${EXTRA_ARGS[@]}"; do
  case "${arg}" in --cache-ram|-cram|--cache-ram=*) cache_in_extra=1 ;; esac
done
if [ "${cache_in_extra}" -eq 0 ]; then
  CMD+=(--cache-ram "${CACHE_RAM_MIB}")
fi

if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

if [ "${DRY_RUN}" -eq 1 ]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

log() { echo "run_runtime[${NAME}]: $*" >&2; }

http_ok() {
  curl -fsS -o /dev/null --max-time 3 "$1" 2>/dev/null
}

# File name of the GGUF the server on HOST:PORT reports, empty if nothing answers.
# Never fails: "nothing answers" is a normal result under set -euo pipefail.
served_model() {
  curl -fsS --max-time 3 "http://${HOST}:${PORT}/v1/models" 2>/dev/null | python3 -c '
import json, os, sys
try:
    print(os.path.basename(str(json.load(sys.stdin)["data"][0]["id"])))
except Exception:
    pass
' 2>/dev/null || true
}

# Wait until every other runtime this script started is serving. Covers loads whose
# launcher died before releasing the lock, so they can never overlap a new load.
wait_for_other_loads() {
  local deadline=$(( $(date +%s) + LOAD_READY_TIMEOUT_S ))
  local pending id url
  while :; do
    pending=()
    while read -r id url; do
      [ -n "${id}" ] || continue
      if ! http_ok "${url}"; then
        pending+=("${id}")
      fi
    done < <("${DOCKER_BIN}" ps --filter "label=${RUNTIME_LABEL}=1" \
               --format "{{.ID}} {{.Label \"${RUNTIME_LABEL}.health\"}}")
    [ "${#pending[@]}" -eq 0 ] && return 0
    if [ "$(date +%s)" -ge "${deadline}" ]; then
      log "other runtime(s) still loading after ${LOAD_READY_TIMEOUT_S}s: ${pending[*]}; refusing to load"
      return 1
    fi
    sleep 2
  done
}

exec 9>>"${LOAD_LOCK}"
chmod 666 "${LOAD_LOCK}" 2>/dev/null || true
log "waiting for the model-load lock (${LOAD_LOCK})"
if ! flock -w "${LOAD_LOCK_WAIT_S}" 9; then
  log "model-load lock not acquired within ${LOAD_LOCK_WAIT_S}s"
  exit 75
fi
wait_for_other_loads || exit 75

# Rig guard: a new load adds power, heat and memory. Checked with the lock held, so it sees the
# rig after the previous load settled.
if [ "${LLAMA_GUARD:-on}" != "off" ]; then
  if ! "${GUARD_PYTHON}" "${GUARD_SCRIPT}" admit --wait "${GUARD_WAIT_S}"; then
    log "rig guard did not admit this load; not starting it"
    exit 75
  fi
fi

# Lease audit (audit only): which lease this load runs under and whether it holds its GPUs and port.
audit_lease() {
  local verdict
  verdict="$(LEASE="${LEASE}" GPUS="${GPUS}" PORT="${PORT}" python3 - "${LEASE_RECORD}" <<'PY' 2>/dev/null || echo "audit-error"
import json, os, sys
lease_id = os.environ["LEASE"]
if not lease_id:
    print("no-lease"); sys.exit()
try:
    leases = json.load(open(sys.argv[1]))["leases"]
except Exception:
    print("record-unreadable"); sys.exit()
lease = next((l for l in leases if l.get("lease_id") == lease_id), None)
if lease is None:
    print("unknown-lease"); sys.exit()
spec = os.environ["GPUS"].removeprefix("device=")
gpus = {int(g) for g in spec.split(",") if g.strip().isdigit()}
problems = []
if not gpus <= set(lease.get("gpu_ids", [])):
    problems.append(f"gpus {sorted(gpus)} not in lease {lease.get('gpu_ids')}")
if int(os.environ["PORT"]) not in lease.get("ports", []):
    problems.append(f"port {os.environ['PORT']} not in lease {lease.get('ports')}")
print("ok" if not problems else "mismatch: " + "; ".join(problems))
PY
)"
  mkdir -p "$(dirname "${LEASE_AUDIT_LOG}")" 2>/dev/null || true
  printf '%s name=%s gpus=%s port=%s lease=%s %s\n' "$(date -Iseconds)" "${NAME}" "${GPUS}" "${PORT}" \
    "${LEASE:-none}" "${verdict}" >> "${LEASE_AUDIT_LOG}" 2>/dev/null || true
  log "lease audit: ${verdict}"
}
audit_lease

# Containers share the host network, so a server already on this port would answer this
# load's readiness probe and its requests. Refuse instead of serving the wrong model.
occupant="$(served_model)"
if [ -n "${occupant}" ] || http_ok "http://${HOST}:${PORT}/health"; then
  log "port ${PORT} is already serving ${occupant:-a runtime}; stop it first"
  exit 75
fi

log "lock held; starting container"
CONTAINER_ID="$("${CMD[@]}")"

deadline=$(( $(date +%s) + LOAD_READY_TIMEOUT_S ))
while :; do
  state="$("${DOCKER_BIN}" inspect -f '{{.State.Running}}' "${NAME}" 2>/dev/null || echo missing)"
  if [ "${state}" = "true" ] && http_ok "http://${HOST}:${PORT}/health"; then
    served="$(served_model)"
    if [ "${served}" != "$(basename "${MODEL}")" ]; then
      log "port ${PORT} serves '${served}', expected '$(basename "${MODEL}")'; removing this container"
      "${DOCKER_BIN}" rm -f "${NAME}" >/dev/null 2>&1 || true
      exit 70
    fi
    break
  fi
  if [ "${state}" != "true" ]; then
    log "container stopped during load (running=${state}); last log lines:"
    "${DOCKER_BIN}" logs --tail 20 "${NAME}" >&2 2>&1 || true
    "${DOCKER_BIN}" rm -f "${NAME}" >/dev/null 2>&1 || true
    exit 70
  fi
  if [ "$(date +%s)" -ge "${deadline}" ]; then
    log "not ready after ${LOAD_READY_TIMEOUT_S}s; removing it so no load overlaps it"
    "${DOCKER_BIN}" logs --tail 20 "${NAME}" >&2 2>&1 || true
    "${DOCKER_BIN}" rm -f "${NAME}" >/dev/null 2>&1 || true
    exit 70
  fi
  sleep 2
done

sleep "${LOAD_SETTLE_S}"
log "ready; releasing the model-load lock"
echo "${CONTAINER_ID}"
