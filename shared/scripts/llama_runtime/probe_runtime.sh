#!/bin/bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: probe_runtime.sh PORT" >&2
  exit 64
fi

PORT="$1"

if ! [[ "${PORT}" =~ ^[0-9]+$ ]]; then
  echo "Port must be numeric: ${PORT}" >&2
  exit 64
fi

curl --fail --silent --show-error "http://127.0.0.1:${PORT}/v1/models"
