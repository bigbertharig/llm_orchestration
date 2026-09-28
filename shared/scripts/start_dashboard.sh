#!/usr/bin/env bash
set -euo pipefail

DASH_DIR="/home/bryan/llm_orchestration/scripts"
LOG_FILE="/tmp/llm_dashboard.log"
HOST="127.0.0.1"
PORT="8787"
URL="http://${HOST}:${PORT}/"

# Always restart so code/UI updates are picked up immediately.
pkill -f "python.*-m dashboard" >/dev/null 2>&1 || true
cd "${DASH_DIR}"
setsid /usr/bin/python3 -m dashboard --host 0.0.0.0 --port "${PORT}" >"${LOG_FILE}" 2>&1 < /dev/null &

for _ in 1 2 3 4 5 6 7 8 9 10; do
  if curl -fsS "http://${HOST}:${PORT}/api/status" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

# Open browser only in desktop sessions.
if [[ -n "${DISPLAY:-}" || -n "${WAYLAND_DISPLAY:-}" ]]; then
  if command -v xdg-open >/dev/null 2>&1; then
    nohup /usr/bin/xdg-open "${URL}" >/dev/null 2>&1 < /dev/null &
  fi
fi
