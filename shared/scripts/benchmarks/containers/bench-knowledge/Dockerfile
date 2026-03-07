# bench-knowledge: MC/loglikelihood benchmarks via llama.cpp + lm_eval gguf model type
# Targets: mmlu, arc_challenge, hellaswag, truthfulqa_mc2, boolq
#
# Build:
#   docker build -t bench-knowledge .
#
# Run:
#   docker run --gpus '"device=1"' \
#     -v /mnt/shared/models:/models:ro \
#     -v /mnt/shared/logs/benchmarks:/results \
#     bench-knowledge /models/qwen2.5-coder-7b/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf

FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git curl \
    && rm -rf /var/lib/apt/lists/*

# Create venv so pip doesn't complain about externally-managed
RUN python3 -m venv /opt/bench
ENV PATH="/opt/bench/bin:$PATH"

# Build llama-cpp-python with CUDA for both GPU architectures (sm_61=1060, sm_86=3090)
ENV CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=61;86"
RUN pip install --no-cache-dir llama-cpp-python[server]==0.3.16

# Install lm_eval and transformers (needed for tokenizer/chat templates)
RUN pip install --no-cache-dir lm_eval transformers

# Entrypoint script
COPY run.sh /opt/bench/run.sh
RUN chmod +x /opt/bench/run.sh

WORKDIR /opt/bench
ENTRYPOINT ["/opt/bench/run.sh"]
