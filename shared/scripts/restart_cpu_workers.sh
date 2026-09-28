#!/usr/bin/env bash
set -euo pipefail

if [ -n "${CPU_RESTART_LOCAL_LOG_DIR:-}" ]; then
  LOCAL_LOG_DIR="${CPU_RESTART_LOCAL_LOG_DIR}"
elif [ -w "/mnt/shared" ]; then
  LOCAL_LOG_DIR="/mnt/shared/logs/cpu_workers"
elif [ -w "/media/bryan/shared" ]; then
  LOCAL_LOG_DIR="/media/bryan/shared/logs/cpu_workers"
else
  LOCAL_LOG_DIR="/tmp/cpu_worker_restart_logs"
fi
REMOTE_LOG_DIR="${CPU_RESTART_REMOTE_LOG_DIR:-/tmp/cpu_workers}"
# Optional hard override for worker shared root; if empty, script auto-detects.
REMOTE_SHARED_ROOT="${CPU_RESTART_REMOTE_SHARED_ROOT:-}"
# Candidate mount points on CPU workers (ordered).
REMOTE_SHARED_CANDIDATES="${CPU_RESTART_REMOTE_SHARED_CANDIDATES:-/media/bryan/shared /mnt/shared}"
PYTHON_BIN="${CPU_RESTART_PYTHON_BIN:-/usr/bin/python3}"
SSH_USER="${CPU_RESTART_SSH_USER:-bryan}"
REFRESH_HOSTKEYS="${CPU_RESTART_REFRESH_HOSTKEYS:-1}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=5)

mkdir -p "${LOCAL_LOG_DIR}"

if [ "$#" -gt 0 ]; then
  WORKERS=("$@")
else
  WORKERS=(10.0.0.10 10.0.0.11 10.0.0.12 10.0.0.13 10.0.0.14 10.0.0.15 10.0.0.16 10.0.0.17)
fi

if [ "${REFRESH_HOSTKEYS}" = "1" ]; then
  mkdir -p "${HOME}/.ssh"
  touch "${HOME}/.ssh/known_hosts"
fi

ok=0
fail=0
for ip in "${WORKERS[@]}"; do
  suffix="${ip##*.}"
  name="cpu-worker-${suffix}"
  log_file="${REMOTE_LOG_DIR}/${name}.log"
  echo "Restarting ${name} (${ip})"

  if [ "${REFRESH_HOSTKEYS}" = "1" ]; then
    ssh-keygen -R "${ip}" >/dev/null 2>&1 || true
    ssh-keygen -R "${SSH_USER}@${ip}" >/dev/null 2>&1 || true
    ssh-keyscan -H "${ip}" >> "${HOME}/.ssh/known_hosts" 2>/dev/null || true
  fi

  if ssh "${SSH_OPTS[@]}" "${SSH_USER}@${ip}" "set -e; mkdir -p '${REMOTE_LOG_DIR}';
    ROOT='';
    if [ -n '${REMOTE_SHARED_ROOT}' ]; then
      ROOT='${REMOTE_SHARED_ROOT}';
    else
      for cand in ${REMOTE_SHARED_CANDIDATES}; do
        if [ -f \"\$cand/scripts/cpu_agent.py\" ] && [ -f \"\$cand/agents/config.json\" ]; then
          ROOT=\"\$cand\";
          break;
        fi;
      done;
    fi;
    if [ -z \"\$ROOT\" ]; then
      echo 'NO_SHARED_ROOT scripts/config not found on worker';
      exit 42;
    fi;
    SCRIPT=\"\$ROOT/scripts/cpu_agent.py\";
    CONFIG=\"\$ROOT/agents/config.json\";
    (pkill -f \"cpu_agent.py --config \$CONFIG --name ${name}\" || true);
    nohup ${PYTHON_BIN} \"\$SCRIPT\" --config \"\$CONFIG\" --name ${name} >>'${log_file}' 2>&1 < /dev/null &"; then
    ok=$((ok + 1))
  else
    echo "FAILED ${name} (${ip})"
    fail=$((fail + 1))
  fi
done

echo "Restart submitted: ok=${ok} fail=${fail} total=${#WORKERS[@]}"
if [ "${fail}" -gt 0 ]; then
  exit 1
fi
