#!/usr/bin/env bash
set -euo pipefail

# Prefer deterministic CPU performance for orchestration workloads.
for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  [ -w "$gov" ] && echo performance > "$gov" || true
done

# Reduce swap aggressiveness for inference tasks.
sysctl -w vm.swappiness=10 >/dev/null || true

# Keep NVIDIA devices initialized to reduce model load jitter.
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -pm 1 >/dev/null 2>&1 || true
fi
