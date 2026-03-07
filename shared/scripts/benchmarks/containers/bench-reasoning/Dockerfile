# bench-reasoning: Generation benchmarks via Ollama on host
# Targets: gsm8k, bbh, drop, mmlu_pro, ifeval
#
# Build:
#   docker build -t bench-reasoning .
#
# Run:
#   docker run --network host \
#     -v /mnt/shared/logs/benchmarks:/results \
#     bench-reasoning --model qwen2.5-coder:7b

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/bench
ENV PATH="/opt/bench/bin:$PATH"

# lm_eval[api] for chat completions + transformers for chat templates + langdetect for ifeval
RUN pip install --no-cache-dir "lm_eval[api]" transformers langdetect

COPY run.sh /opt/bench/run.sh
RUN chmod +x /opt/bench/run.sh

WORKDIR /opt/bench
ENTRYPOINT ["/opt/bench/run.sh"]
