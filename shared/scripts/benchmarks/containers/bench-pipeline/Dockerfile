# bench-pipeline: Custom pipeline reliability tests via Ollama on host
# Targets: 6 custom tests (json_schema, tool_plan, command_safety, ambiguity, orchestration, long_context)
#
# Build:
#   docker build -t bench-pipeline .
#
# Run:
#   docker run --network host \
#     -v /mnt/shared/logs/benchmarks:/results \
#     -v /mnt/shared/scripts/benchmarks:/benchmark-scripts:ro \
#     bench-pipeline --model qwen2.5-coder:7b

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/bench
ENV PATH="/opt/bench/bin:$PATH"

# requests for Ollama API calls
RUN pip install --no-cache-dir requests

COPY run.sh /opt/bench/run.sh
RUN chmod +x /opt/bench/run.sh

WORKDIR /opt/bench
ENTRYPOINT ["/opt/bench/run.sh"]
