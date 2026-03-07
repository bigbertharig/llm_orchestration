#!/bin/bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-llama-runtime:sm61-sm86}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker build \
  -t "${IMAGE_TAG}" \
  -f "${SCRIPT_DIR}/Dockerfile" \
  "${SCRIPT_DIR}"
