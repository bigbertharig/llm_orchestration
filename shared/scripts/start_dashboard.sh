#!/usr/bin/env bash
# The dashboard moved to llm-gateway (docs/DASHBOARD.md there).
echo "The dashboard is served by llm-gateway on the Pi: http://127.0.0.1:8080/dashboard"
if [[ -n "${DISPLAY:-}" || -n "${WAYLAND_DISPLAY:-}" ]] && command -v xdg-open >/dev/null; then
  nohup xdg-open http://127.0.0.1:8080/dashboard >/dev/null 2>&1 < /dev/null &
fi
