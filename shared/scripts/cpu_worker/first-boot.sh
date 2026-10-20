#!/bin/bash
# First-boot script for CPU worker clones
# Runs once: expand filesystem, set hostname, regenerate SSH keys, provision SSH access
# Source of truth: llm_orchestration shared/scripts/cpu_worker/first-boot.sh (installed to /opt by
# prepare_worker.sh; first-boot.service runs it while /opt/.first-boot-done is absent).

FLAG="/opt/.first-boot-done"
if [ -f "$FLAG" ]; then
    exit 0
fi

LOG="/var/log/first-boot.log"

# Expand filesystem to fill SD card
echo "Expanding filesystem..." >> $LOG
parted ---pretend-input-tty /dev/mmcblk0 resizepart 1 100% <<< yes >> $LOG 2>&1
resize2fs /dev/mmcblk0p1 >> $LOG 2>&1
echo "Filesystem: $(df -h / | tail -1)" >> $LOG

# Wait for network - DHCP may not be ready at early boot
echo "Waiting for network..." >> $LOG
for attempt in $(seq 1 30); do
    IP=$(hostname -I | awk '{print $1}' | tr -d ' ')
    if [ -n "$IP" ] && [ "$IP" != "127.0.0.1" ]; then
        break
    fi
    sleep 2
done

if [ -z "$IP" ] || [ "$IP" = "127.0.0.1" ]; then
    echo "FATAL: No network after 60s, aborting first-boot" >> $LOG
    exit 1
fi

OCTET=$(echo "$IP" | awk -F. '{print $4}')
HOSTNAME="worker-${OCTET}"
echo "Network ready: $IP -> $HOSTNAME (attempt $attempt)" >> $LOG

# Set hostname
hostnamectl set-hostname "$HOSTNAME"
echo "$HOSTNAME" > /etc/hostname
sed -i "s/worker-[0-9]*/$HOSTNAME/g" /etc/hosts
sed -i "s/orangepiprime/$HOSTNAME/g" /etc/hosts

# Regenerate SSH host keys
rm -f /etc/ssh/ssh_host_*
dpkg-reconfigure openssh-server
systemctl restart sshd

# Provision control plane SSH access
mkdir -p /root/.ssh
chmod 700 /root/.ssh
cat > /root/.ssh/authorized_keys << SSHEOF
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJhtVVmuqEoRmZkfEOtK9x4Am1zutnPZVXRvBaaz/Rop laptop@10.0.0.3 2026-03-07
SSHEOF
chmod 600 /root/.ssh/authorized_keys
echo "SSH key provisioned for control plane" >> $LOG

# Mark as done BEFORE any optional steps
touch "$FLAG"
echo "First boot complete: $HOSTNAME ($IP)" >> $LOG

# cpu-agent.service starts after this unit (After=first-boot.service), so the agent's name
# (cpu-%H) uses the hostname set above.
