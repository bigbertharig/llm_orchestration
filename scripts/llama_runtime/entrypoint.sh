#!/bin/sh
set -eu

if [ $# -eq 0 ]; then
    echo "Usage: llama-server <args>" >&2
    exit 64
fi

if [ "$1" = "llama-server" ]; then
    shift
fi

exec "${LLAMA_SERVER_BIN}" "$@"
