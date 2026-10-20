#!/usr/bin/env bash
# Bring the GPU rig host to the standard in shared/workspace/systems_prep.md. Idempotent; run as bryan.
# usage: prepare_rig.sh            show what would change (nothing is touched)
#        prepare_rig.sh --apply    make the changes
# It does not install the OS, the NVIDIA driver, Docker or the container toolkit, build images, or set
# the BIOS: those are the numbered rebuild steps in systems_prep.md. Run check_rig.sh afterwards.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TREE="$(cd "$HERE/../../.." && pwd)"             # ~/llm_orchestration
SYS="$HERE/systemd"
DRIVER_BRANCH=580
APPLY=0; [ "${1:-}" = "--apply" ] && APPLY=1
changes=0
step() {  # description, command...: run (or only report) a change
  local what=$1; shift
  changes=$((changes+1))
  if [ "$APPLY" = 1 ]; then echo "apply  $what"; "$@"; else echo "would  $what"; fi
}
install_file() {  # src dest mode
  cmp -s "$1" "$2" 2>/dev/null || step "install $2" sudo install -m "$3" "$1" "$2"
}

# sudo for bryan (the controller and the harness run system commands over ssh)
rule='bryan ALL=(ALL) NOPASSWD: ALL'
if ! sudo -n grep -qx "$rule" /etc/sudoers.d/010_pi-nopasswd 2>/dev/null; then
  step "passwordless sudo rule" bash -c "echo '$rule' | sudo tee /etc/sudoers.d/010_pi-nopasswd >/dev/null && sudo chmod 440 /etc/sudoers.d/010_pi-nopasswd && sudo visudo -c >/dev/null"
fi

# updates only by hand: no unattended upgrades, the driver held at its branch
for t in apt-daily.timer apt-daily-upgrade.timer; do
  [ "$(systemctl is-enabled "$t" 2>/dev/null)" = "disabled" ] || step "disable $t" sudo systemctl disable --now "$t"
done
unheld=$(comm -23 <(dpkg-query -W -f='${db:Status-Abbrev} ${Package}\n' '*nvidia*' 2>/dev/null |
  awk -v b="$DRIVER_BRANCH" '$1 ~ /^.i/ && $2 ~ b {sub(/:.*/, "", $2); print $2}' | sort -u) <(apt-mark showhold | sed 's/:.*//' | sort -u) | xargs)
[ -z "$unheld" ] || step "hold $unheld" sudo apt-mark hold $unheld

# rig units, tuning script and command
for u in rig-boot rig-guard rig-plan rig-power rig-performance-tune llm-bench llm-orchestration; do
  install_file "$SYS/$u.service" "/etc/systemd/system/$u.service" 644
done
install_file "$HERE/rig-performance-tune.sh" /usr/local/sbin/rig-performance-tune.sh 755
[ "$APPLY" = 1 ] && [ "$changes" -gt 0 ] && sudo systemctl daemon-reload
for u in rig-boot rig-guard rig-plan rig-power rig-performance-tune; do
  systemctl is-enabled --quiet "$u" 2>/dev/null || step "enable $u" sudo systemctl enable "$u"
done
[ "$(readlink -f /usr/local/bin/rig 2>/dev/null)" = "$HERE/rig.py" ] || step "link /usr/local/bin/rig" sudo ln -sfn "$HERE/rig.py" /usr/local/bin/rig

# memory: earlyoom RAM-only, low swappiness; never add swap
if ! cmp -s "$SYS/earlyoom.default" /etc/default/earlyoom 2>/dev/null; then
  step "earlyoom settings" bash -c "sudo install -m 644 '$SYS/earlyoom.default' /etc/default/earlyoom && sudo systemctl restart earlyoom"
fi
echo 'vm.swappiness = 10' | cmp -s - /etc/sysctl.d/99-llm-orchestration.conf 2>/dev/null ||
  step "vm.swappiness 10" bash -c "echo 'vm.swappiness = 10' | sudo tee /etc/sysctl.d/99-llm-orchestration.conf >/dev/null && sudo sysctl -q --system"

# time from the Pi only
printf '# Network time leader: the Pi (2026-10-15). All rig time comes from it.\nserver 10.0.0.2 iburst\n' |
  cmp -s - /etc/chrony/sources.d/pi.sources 2>/dev/null ||
  step "chrony source: the Pi" bash -c "printf '# Network time leader: the Pi (2026-10-15). All rig time comes from it.\nserver 10.0.0.2 iburst\n' | sudo tee /etc/chrony/sources.d/pi.sources >/dev/null && sudo chronyc reload sources >/dev/null"
if [ -f /etc/chrony/sources.d/ubuntu-ntp-pools.sources ]; then
  step "disable the Ubuntu NTP pool sources" sudo mv /etc/chrony/sources.d/ubuntu-ntp-pools.sources /etc/chrony/sources.d/ubuntu-ntp-pools.sources.disabled
fi

# shared drive from the Pi (automount, soft, so a lost Pi does not hang logins)
line='10.0.0.2:/media/bryan/shared /mnt/shared nfs nofail,_netdev,x-systemd.automount,x-systemd.idle-timeout=600,x-systemd.mount-timeout=15,soft,timeo=100,retrans=3 0 0'
if ! grep -qxF "$line" /etc/fstab; then
  step "fstab share line (replaces any other /mnt/shared line)" bash -c "sudo sed -i '\#[[:space:]]/mnt/shared[[:space:]]#d' /etc/fstab && echo '$line' | sudo tee -a /etc/fstab >/dev/null && sudo mkdir -p /mnt/shared && sudo systemctl daemon-reload && sudo systemctl restart mnt-shared.automount"
fi

# Docker access and the NVIDIA runtime
id -nG bryan | tr ' ' '\n' | grep -qx docker || step "add bryan to the docker group" sudo usermod -aG docker bryan
if command -v nvidia-ctk >/dev/null && ! grep -q nvidia-container-runtime /etc/docker/daemon.json 2>/dev/null; then
  step "docker nvidia runtime" bash -c "sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
fi

# orchestration venv and hot-set folders
VENV=/home/bryan/llm-orchestration-venv
if ! "$VENV/bin/python" -c 'import requests, filelock' 2>/dev/null; then
  step "orchestration venv" bash -c "python3 -m venv '$VENV' && '$VENV/bin/pip' install -q -r '$TREE/requirements.txt'"
fi
[ -d /home/bryan/model-hotset ] || step "SATA hot-set folder" mkdir -p /home/bryan/model-hotset
if mountpoint -q /data && [ ! -d /data/model-hotset ]; then
  step "NVMe hot-set folder" bash -c "sudo mkdir -p /data/model-hotset && sudo chown bryan:bryan /data/model-hotset"
fi

if [ "$changes" = 0 ]; then echo "no changes: host matches the standard"
elif [ "$APPLY" = 1 ]; then echo "$changes change(s) applied; run check_rig.sh"
else echo "$changes change(s) to make; run with --apply"; fi
