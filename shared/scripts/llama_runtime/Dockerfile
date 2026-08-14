ARG CUDA_VERSION=12.9.2
ARG LLAMA_CPP_REPO=https://github.com/ggml-org/llama.cpp.git
ARG LLAMA_CPP_REF=08659901c43b51de735740f1cf61bb82fbe0c4e4
ARG LLAMA_CPP_VERSION=b10333
ARG CUDA_ARCHITECTURES=61\;86

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS build

ARG LLAMA_CPP_REPO
ARG LLAMA_CPP_REF
ARG LLAMA_CPP_VERSION
ARG CUDA_ARCHITECTURES

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone "${LLAMA_CPP_REPO}" llama.cpp

WORKDIR /src/llama.cpp
RUN git checkout "${LLAMA_CPP_REF}"
RUN cmake -S . -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
    -DLLAMA_BUILD_SERVER=ON \
    -DBUILD_SHARED_LIBS=OFF
RUN cmake --build build --config Release --target llama-server -j6

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04

ARG CUDA_VERSION
ARG LLAMA_CPP_REF
ARG LLAMA_CPP_VERSION

LABEL org.opencontainers.image.source="https://github.com/ggml-org/llama.cpp" \
      org.opencontainers.image.version="${LLAMA_CPP_VERSION}" \
      org.opencontainers.image.revision="${LLAMA_CPP_REF}" \
      com.bigbertha.cuda.version="${CUDA_VERSION}" \
      com.bigbertha.cuda.architectures="61;86"

ENV DEBIAN_FRONTEND=noninteractive
ENV LLAMA_SERVER_BIN=/opt/llama/bin/llama-server

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /src/llama.cpp/build/bin/llama-server /opt/llama/bin/llama-server
COPY --from=build /src/llama.cpp/models/templates /opt/llama/chat-templates
COPY entrypoint.sh /opt/llama/entrypoint.sh

RUN chmod +x /opt/llama/entrypoint.sh

WORKDIR /opt/llama
ENTRYPOINT ["/opt/llama/entrypoint.sh"]
