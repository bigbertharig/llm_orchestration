#!/bin/bash
# Turn a prepared CPU worker into the source for a new base image, then power it off.
# Run on the worker as bryan, right before pulling its SD card:
#   bash /media/bryan/shared/scripts/cpu_worker/make_image_ready.sh
# Use the worker with the smallest card (an image only fits cards at least its size; first-boot grows
# the filesystem on larger ones). Capture steps: shared/workspace/cpu-worker-setup.md, "Base Image".
#
# What it does: brings the worker to the fleet standard (prepare_worker.sh) and checks it, then removes
# what must differ per worker so first-boot regenerates it on each clone: the first-boot flag (hostname,
# SSH host keys, root key), the machine id (systemd makes a new one at boot), logs and shell history.
# When this card goes back into its worker, first-boot runs again there: same IP, same name, new SSH
# host key (clear it on the Pi: ssh-keygen -f ~/.ssh/known_hosts_workers -R 10.0.0.<octet>).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

bash "$HERE/prepare_worker.sh"
bash "$HERE/check_worker.sh"

echo "== $(hostname) generalizing for the image"
sudo systemctl stop cpu-agent.service
sudo rm -f /opt/.first-boot-done /var/log/first-boot.log
sudo truncate -s 0 /etc/machine-id
sudo rm -f /var/lib/dbus/machine-id
sudo journalctl --rotate --quiet || true
sudo journalctl --vacuum-time=1s --quiet || true
sudo find /var/log /var/log.hdd -xdev -type f -exec truncate -s 0 {} + 2>/dev/null || true
rm -f "$HOME/.bash_history"
sudo rm -f /root/.bash_history
sudo apt-get clean
sync
echo "== $(hostname) ready for imaging; powering off in 5 s (Ctrl-C to stay up)"
sleep 5
sudo poweroff
