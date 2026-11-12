#!/usr/bin/env bash
set -euo pipefail
src=/home/bryan/llm_orchestration/shared/scripts/rig/llama_kv_chat_server.cpp
llama_src=/data/bench-staging/kvchat-b11669-src
llama_build=/data/bench-staging/kvchat-b11669-build
image=4bfa1cb0a955
[[ -f "$src" && -d "$llama_src" && -d "$llama_build" ]] || { echo 'patched llama.cpp source/build missing' >&2; exit 2; }
docker run --rm --gpus all --network none \
  -v "$llama_src:/src:ro" -v "$llama_build:/build" -v "$src:/input.cpp:ro" \
  -w /build "$image" bash -lc \
  'g++ -std=c++17 -O2 -pthread -I/src/include -I/src/ggml/include -I/src/vendor/nlohmann -I/src/vendor/cpp-httplib /input.cpp /src/vendor/cpp-httplib/httplib.cpp -o /build/bin/llama-kv-chat-server -L/build/bin -Wl,-rpath,/build/bin -l:libllama.so.0.7.0 -l:libggml.so.0.27.0 -l:libggml-cpu.so.0.27.0 -l:libggml-cuda.so.0.27.0 -l:libggml-base.so.0.27.0'
echo "$llama_build/bin/llama-kv-chat-server"
