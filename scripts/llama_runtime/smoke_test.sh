#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  smoke_test.sh --port PORT [--chat-prompt TEXT] [--completion-prompt TEXT]

Optional:
  --timeout SECONDS           Default: 120
  --chat-prompt TEXT          Default: Reply with exactly: OK
  --completion-prompt TEXT    Default: Say OK
EOF
}

PORT=""
TIMEOUT="${TIMEOUT:-120}"
CHAT_PROMPT="${CHAT_PROMPT:-Reply with exactly: OK}"
COMPLETION_PROMPT="${COMPLETION_PROMPT:-Say OK}"

while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="${2:?}"; shift 2 ;;
    --timeout) TIMEOUT="${2:?}"; shift 2 ;;
    --chat-prompt) CHAT_PROMPT="${2:?}"; shift 2 ;;
    --completion-prompt) COMPLETION_PROMPT="${2:?}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage >&2; exit 64 ;;
  esac
done

if [ -z "${PORT}" ]; then
  echo "Missing required --port" >&2
  usage >&2
  exit 64
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for _ in $(seq 1 "${TIMEOUT}"); do
  if "${SCRIPT_DIR}/probe_runtime.sh" "${PORT}" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

"${SCRIPT_DIR}/probe_runtime.sh" "${PORT}" >/dev/null

curl --fail --silent --show-error "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"${CHAT_PROMPT}\"}],\"max_tokens\":8}" >/dev/null

curl --fail --silent --show-error "http://127.0.0.1:${PORT}/v1/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"prompt\":\"${COMPLETION_PROMPT}\",\"max_tokens\":4,\"logprobs\":5}" >/dev/null

echo "smoke test passed on port ${PORT}"
