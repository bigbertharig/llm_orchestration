#!/usr/bin/env bash
# Experimental Coder Next expert placement; run only after reserving the six GPUs.
set -euo pipefail
usage() {
  echo "Usage: $0 {32k|262k} PORT [--dry-run]" >&2
  exit 64
}
[[ $# -ge 2 && $# -le 3 ]] || usage
case "$1" in
  32k)
    ctx=32768
    override='blk[.][0-4][.]ffn_(up|down|gate)_exps[.]weight=CUDA1,blk[.][5-9][.]ffn_(up|down|gate)_exps[.]weight=CUDA2,blk[.]1[0-4][.]ffn_(up|down|gate)_exps[.]weight=CUDA3,blk[.]1[5-9][.]ffn_(up|down|gate)_exps[.]weight=CUDA4,blk[.]2[0-3][.]ffn_(up|down|gate)_exps[.]weight=CUDA5'
    ;;
  262k)
    ctx=262144
    override='blk[.][0-4][.]ffn_(up|down|gate)_exps[.]weight=CUDA1,blk[.]([6-9]|1[01])[.]ffn_(up|down|gate)_exps[.]weight=CUDA2,blk[.]1[2-7][.]ffn_(up|down|gate)_exps[.]weight=CUDA3,blk[.](1[89]|2[0-3])[.]ffn_(up|down|gate)_exps[.]weight=CUDA4,blk[.](5|2[4-7])[.]ffn_(up|down|gate)_exps[.]weight=CUDA5'
    ;;
  *) usage ;;
esac
port=$2
[[ "$port" =~ ^[0-9]+$ ]] || usage
name="coder-next-experts-$1-$port"
if [[ $# -eq 3 ]]; then
  [[ "$3" == "--dry-run" ]] || usage
  set -- --dry-run
else
  set --
fi
runtime=/home/bryan/llm_orchestration/shared/scripts/llama_runtime/run_runtime.sh
model=/data/model-hotset/qwen3-coder-next/Qwen3-Coder-Next-Q4_K_S.gguf
exec "$runtime" \
  --name "$name" \
  --model "$model" --port "$port" --gpus all \
  --ctx-size "$ctx" --batch-size 64 --parallel 1 --memory-limit 20g \
  --tensor-split 48,0,0,0,0,0 \
  --extra-arg --override-tensor --extra-arg "$override" \
  "$@"