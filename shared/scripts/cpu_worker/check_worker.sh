#!/bin/bash
# Readiness check for a CPU worker against the fleet standard. Read-only except a scratch file it
# writes and removes in the shared task tree. Exit 1 if anything fails.
#   ssh 10.0.0.<octet> bash /media/bryan/shared/scripts/cpu_worker/check_worker.sh
# From the Pi for the fleet:
#   for i in 10 11 12 14 15 16 17; do ssh 10.0.0.$i bash /media/bryan/shared/scripts/cpu_worker/check_worker.sh; done
HERE="$(cd "$(dirname "$0")" && pwd)"
SHARED=/media/bryan/shared
VENV=/opt/worker-env
HOST="$(hostname)"
NAME="cpu-${HOST}"
fails=0
check() {  # name, then a command that succeeds on pass
  local name="$1"; shift
  if out="$("$@" 2>&1)"; then echo "PASS $HOST $name"; else echo "FAIL $HOST $name ${out:+($(echo "$out" | tail -1))}"; fails=$((fails + 1)); fi
}

check "hostname is worker-<octet>" bash -c '[ "$(hostname)" = "worker-$(hostname -I | awk "{print \$1}" | cut -d. -f4)" ]'
check "fstab mounts the Pi share" grep -q '^10\.0\.0\.2:/media/bryan/shared /media/bryan/shared ' /etc/fstab
check "share mounted from the Pi" bash -c "ls $SHARED/scripts/cpu_agent.py >/dev/null && [ \"\$(findmnt -no SOURCE $SHARED | tail -1)\" = 10.0.0.2:/media/bryan/shared ]"
check "can write the task tree" bash -c "f=$SHARED/tasks/.write-check-$HOST; touch \$f && rm \$f"
check "cpu-agent active" systemctl is-active --quiet cpu-agent.service
check "heartbeat fresh (<90 s)" python3 -c "
import json, datetime, glob, sys
hb = [json.load(open(f)) for f in glob.glob('$SHARED/cpus/*/heartbeat.json')]
mine = [h for h in hb if h.get('hostname') == '$HOST']
assert mine, 'no heartbeat with hostname $HOST'
age = (datetime.datetime.now() - datetime.datetime.fromisoformat(mine[0]['last_updated'])).total_seconds()
assert age < 90, f'{age:.0f} s old'"
check "passwordless sudo" sudo -n true
check "venv matches requirements.txt" bash -c "diff <(grep -vE '^\s*(#|$)' $HERE/requirements.txt | tr 'A-Z_' 'a-z-' | sort) <($VENV/bin/pip freeze --disable-pip-version-check | tr 'A-Z_' 'a-z-' | sort) >/dev/null"
check "venv imports requests, ddgs" "$VENV/bin/python" -c "import requests, ddgs"
check "internet (pypi)" curl -sf -o /dev/null -m 8 https://pypi.org/simple/requests/
check "clock NTP-synchronized" bash -c '[ "$(timedatectl show -p NTPSynchronized --value)" = yes ]'
check "disk free > 5 GB" bash -c '[ "$(df --output=avail -k / | tail -1)" -gt 5000000 ]'
check "CPU < 80 C" bash -c 't=$(cat /sys/class/thermal/thermal_zone0/temp); [ "$t" -lt 80000 ] || { echo "$((t/1000)) C"; false; }'
check "agent unit is the fleet standard" cmp -s "$HERE/cpu-agent.service" /etc/systemd/system/cpu-agent.service
check "first-boot files are the fleet standard" bash -c "cmp -s $HERE/first-boot.sh /opt/first-boot.sh && cmp -s $HERE/first-boot.service /etc/systemd/system/first-boot.service"
check "agent runs as cpu-<hostname>" bash -c "pgrep -af 'cpu_agent.py' | grep -q -- '--name cpu-$HOST\$'"
check "no dead ensure-shared-mount timer" bash -c '! systemctl list-unit-files ensure-shared-mount.timer --no-legend | grep -q .'

echo "$HOST: $fails failed"
[ "$fails" -eq 0 ]
