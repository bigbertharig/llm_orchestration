# bench-code: Code generation benchmarks via evalplus
# Targets: humaneval_plus, mbpp_plus
#
# Build:
#   docker build -t bench-code .
#
# Run:
#   docker run --network host \
#     -v /mnt/shared/logs/benchmarks:/results \
#     bench-code --model qwen2.5-coder:7b

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/bench
ENV PATH="/opt/bench/bin:$PATH"

# evalplus for HumanEval+ and MBPP+, openai for Ollama API calls
RUN pip install --no-cache-dir evalplus openai

COPY run.sh /opt/bench/run.sh
RUN chmod +x /opt/bench/run.sh

WORKDIR /opt/bench
ENTRYPOINT ["/opt/bench/run.sh"]
