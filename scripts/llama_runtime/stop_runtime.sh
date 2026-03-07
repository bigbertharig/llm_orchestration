#!/bin/bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: stop_runtime.sh CONTAINER_NAME" >&2
  exit 64
fi

docker rm -f "$1"
