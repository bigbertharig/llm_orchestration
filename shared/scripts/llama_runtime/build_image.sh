#!/bin/bash
# Build a llama-server runtime image for one llama.cpp release tag.
#   build_image.sh b11669            -> llama-runtime:b11669-candidate
#   IMAGE_TAG=... build_image.sh b11669
# CUDA version and GPU architectures are the Dockerfile's pinned defaults (CUDA 12 for the
# Pascal sm_61 1060s plus sm_86 for the 3090); change them there only when the GPUs change.
set -euo pipefail

VERSION="${1:-${LLAMA_CPP_VERSION:-}}"
if ! [[ "${VERSION}" =~ ^b[0-9]+$ ]]; then
  echo "Usage: build_image.sh <llama.cpp tag, e.g. b11669>" >&2
  exit 64
fi
REPO="${LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"
REF="$(git ls-remote --tags "${REPO}" "refs/tags/${VERSION}" | awk '{print $1}')"
if [ -z "${REF}" ]; then
  echo "llama.cpp tag ${VERSION} not found in ${REPO}" >&2
  exit 66
fi
IMAGE_TAG="${IMAGE_TAG:-llama-runtime:${VERSION}-candidate}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "building ${IMAGE_TAG} from llama.cpp ${VERSION} (${REF})"
docker build \
  -t "${IMAGE_TAG}" \
  --build-arg "LLAMA_CPP_VERSION=${VERSION}" \
  --build-arg "LLAMA_CPP_REF=${REF}" \
  -f "${SCRIPT_DIR}/Dockerfile" \
  "${SCRIPT_DIR}"
