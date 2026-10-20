#!/bin/bash
# Bring a CPU worker to the fleet standard (shared/workspace/cpu-worker-setup.md). Idempotent.
# Run on the worker as bryan (uses sudo):
#   bash /media/bryan/shared/scripts/cpu_worker/prepare_worker.sh
# From the Pi for the fleet:
#   for i in 10 11 12 14 15 16 17; do ssh 10.0.0.$i bash /media/bryan/shared/scripts/cpu_worker/prepare_worker.sh; done
# Checked afterwards by check_worker.sh in this folder.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV=/opt/worker-env
SHARE_LINE='10.0.0.2:/media/bryan/shared /media/bryan/shared nfs nofail,_netdev,noauto,x-systemd.automount,x-systemd.mount-timeout=10s,nolock 0 0'
step() { echo "== $(hostname) $*"; }

step "sudo rule"
if [ ! -f /etc/sudoers.d/90-bryan-nopasswd ]; then
  echo 'bryan ALL=(ALL) NOPASSWD:ALL' | sudo tee /etc/sudoers.d/90-bryan-nopasswd >/dev/null
  sudo chmod 440 /etc/sudoers.d/90-bryan-nopasswd
fi
sudo visudo -c >/dev/null

step "shared mount from the Pi"
fstab_changed=0
if grep -q '^10\.0\.0\.3:/mnt/shared /media/bryan/shared ' /etc/fstab; then
  sudo cp -n /etc/fstab /etc/fstab.bak-rig-share
  sudo sed -i 's#^10\.0\.0\.3:/mnt/shared /media/bryan/shared .*#'"$SHARE_LINE"'#' /etc/fstab
  fstab_changed=1
fi
grep -qF "$SHARE_LINE" /etc/fstab || { echo "fstab lacks the Pi share line: $SHARE_LINE" >&2; exit 1; }
if [ "$fstab_changed" = 1 ]; then
  sudo systemctl daemon-reload
  sudo systemctl restart media-bryan-shared.automount
fi

step "worker venv matches requirements.txt"
[ -x "$VENV/bin/python" ] || sudo python3 -m venv "$VENV"
sudo "$VENV/bin/pip" install --quiet --no-cache-dir --disable-pip-version-check -r "$HERE/requirements.txt"
wanted="$(grep -vE '^\s*(#|$)' "$HERE/requirements.txt" | cut -d= -f1 | tr 'A-Z_' 'a-z-' | sort)"
extra="$("$VENV/bin/pip" freeze --disable-pip-version-check | cut -d= -f1 | tr 'A-Z_' 'a-z-' | sort | comm -23 - <(echo "$wanted"))"
if [ -n "$extra" ]; then
  echo "removing extras: $(echo $extra)"
  sudo "$VENV/bin/pip" uninstall --quiet --yes $extra
fi

step "clean up"
sudo apt-get clean
rm -rf "$HOME/.cache" "$HOME/ml-env"
sudo journalctl --vacuum-time=7d --quiet || true
sudo find /var/log /var/log.hdd -xdev -type f \( -name '*.gz' -o -name '*.[0-9]' -o -name '*.old' \) -delete 2>/dev/null || true

step "services a fleet worker does not need"
# ensure-shared-mount.timer points at a service that no longer exists; the fstab automount does the job.
sudo systemctl disable --now ensure-shared-mount.timer >/dev/null 2>&1 || true
sudo rm -f /etc/systemd/system/ensure-shared-mount.timer /etc/systemd/system/timers.target.wants/ensure-shared-mount.timer
# Wired only; no unattended updates (workers change with the image, not by themselves); no man-db rebuilds.
for unit in wpa_supplicant.service apt-daily.timer apt-daily-upgrade.timer man-db.timer; do
  sudo systemctl disable --now "$unit" >/dev/null 2>&1 || true
done
sudo systemctl daemon-reload

step "system files (agent unit, first boot)"
unit_changed=0
install_if_changed() {  # src dest mode
  if ! sudo cmp -s "$1" "$2"; then
    sudo install -m "$3" -o root -g root "$1" "$2"
    echo "installed $2"
    return 0
  fi
  return 1
}
install_if_changed "$HERE/cpu-agent.service" /etc/systemd/system/cpu-agent.service 644 && unit_changed=1 || true
install_if_changed "$HERE/first-boot.service" /etc/systemd/system/first-boot.service 644 || true
install_if_changed "$HERE/first-boot.sh" /opt/first-boot.sh 755 || true
sudo systemctl daemon-reload
sudo systemctl enable cpu-agent.service first-boot.service >/dev/null 2>&1

step "cpu agent"
sudo systemctl reset-failed cpu-agent.service >/dev/null 2>&1 || true
if [ "$unit_changed" = 1 ]; then
  sudo systemctl restart cpu-agent.service
else
  systemctl is-active --quiet cpu-agent.service || sudo systemctl start cpu-agent.service
fi
sleep 2
systemctl is-active cpu-agent.service
echo "== $(hostname) done; used $(df -h / | awk 'NR==2{print $3}') of $(df -h / | awk 'NR==2{print $2}')"
