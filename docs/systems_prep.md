# System Preparation for LLM Machine

## Overview
Optimizations performed 2026-02-05 to prepare the system as a dedicated LLM orchestration machine.

This document serves as:
1. **Recovery reference** - Undo changes if needed
2. **Redo script** - Apply same config to fresh install
3. **Change log** - Track what was modified

---

## Phase 1: Disabled Unnecessary Services

### DO (disable services)
```bash
sudo systemctl disable --now cups cups-browsed avahi-daemon ModemManager \
    switcheroo-control power-profiles-daemon packagekit colord thermald \
    unattended-upgrades anacron wpa_supplicant

sudo systemctl disable --now cups.socket cups.path avahi-daemon.socket anacron.timer
```

### UNDO (re-enable services)
```bash
sudo systemctl enable --now cups cups-browsed avahi-daemon ModemManager \
    switcheroo-control power-profiles-daemon packagekit colord thermald \
    unattended-upgrades anacron wpa_supplicant

sudo systemctl enable --now cups.socket cups.path avahi-daemon.socket anacron.timer
```

### What Each Service Does
| Service | Purpose | Why Disabled |
|---------|---------|--------------|
| cups, cups-browsed | Printing | No printer |
| avahi-daemon | mDNS/Bonjour discovery | Not needed |
| ModemManager | Cellular modems | No modem |
| switcheroo-control | Hybrid graphics switching | Single GPU type |
| power-profiles-daemon | Power profiles | Fixed performance mode |
| packagekit | GUI package management | Using apt directly |
| colord | Color calibration | Not doing color work |
| thermald | Intel thermal daemon | GPUs manage themselves |
| unattended-upgrades | Auto updates | Manual control preferred |
| anacron | Scheduled jobs | Not using anacron |
| wpa_supplicant | WiFi | Using ethernet |

---

## Phase 2: Removed Snap System

### DO (remove snaps)
```bash
# Remove snaps in order (dependencies matter)
sudo snap remove --purge firefox
sudo snap remove --purge snapd-desktop-integration
sudo snap remove --purge gnome-42-2204
sudo snap remove --purge gtk-common-themes
sudo snap remove --purge bare
sudo snap remove --purge core22

# Remove snapd
sudo systemctl disable --now snapd snapd.socket snapd.seeded.service
sudo apt purge -y snapd

# Install Firefox from Mozilla PPA
sudo add-apt-repository -y ppa:mozillateam/ppa
echo 'Package: firefox*
Pin: release o=LP-PPA-mozillateam
Pin-Priority: 1001' | sudo tee /etc/apt/preferences.d/mozilla-firefox
sudo apt update && sudo apt install -y firefox
```

### UNDO (restore snaps)
```bash
# Re-install snapd
sudo apt install -y snapd
sudo systemctl enable --now snapd snapd.socket

# Wait for snapd to initialize
sleep 10

# Re-install Firefox snap
sudo snap install firefox

# Remove PPA Firefox
sudo apt purge -y firefox
sudo rm /etc/apt/preferences.d/mozilla-firefox
sudo add-apt-repository --remove ppa:mozillateam/ppa
```

---

## Phase 3: Removed GNOME Bloatware

### DO (remove bloatware)
```bash
sudo apt purge -y \
    gnome-calculator gnome-characters gnome-clocks \
    gnome-font-viewer gnome-initial-setup gnome-bluetooth-sendto \
    gnome-disk-utility cheese shotwell simple-scan deja-dup \
    transmission-gtk yelp gnome-weather gnome-maps gnome-contacts \
    gnome-calendar totem rhythmbox gnome-tour gnome-user-docs

sudo apt autoremove -y
```

### UNDO (restore apps)
```bash
sudo apt install -y \
    gnome-calculator gnome-characters gnome-clocks \
    gnome-font-viewer gnome-initial-setup gnome-bluetooth-sendto \
    gnome-disk-utility cheese shotwell simple-scan deja-dup \
    transmission-gtk yelp gnome-weather gnome-maps gnome-contacts \
    gnome-calendar totem rhythmbox gnome-tour gnome-user-docs
```

---

## Phase 4: Clock/NTP Fix

### Problem
Ubuntu 25.10 defaults to NTP-NTS (NTP with TLS) which fails when clock is already wrong.

### DO (add regular NTP)
```bash
echo "pool pool.ntp.org iburst" | sudo tee /etc/chrony/sources.d/pool-ntp.sources
sudo systemctl restart chrony
```

### UNDO (remove regular NTP)
```bash
sudo rm /etc/chrony/sources.d/pool-ntp.sources
sudo systemctl restart chrony
```

### Emergency Clock Set
If clock is way off and NTP can't sync:
```bash
# Get time from HTTP header and set manually
sudo date -s "$(curl -sI google.com | grep -i '^date:' | cut -d' ' -f3-6)"
sudo chronyc -a makestep
```

---

## Phase 5: Sudo Configuration

### DO (enable passwordless sudo)
```bash
echo "$USER ALL=(ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/nopasswd-$USER
sudo chmod 440 /etc/sudoers.d/nopasswd-$USER
```

### UNDO (require password again)
```bash
sudo rm /etc/sudoers.d/nopasswd-$USER
```

---

## Phase 6: GPU Power Limits (Persistence)

### DO (create systemd service)
```bash
cat << 'EOF' | sudo tee /etc/systemd/system/nvidia-power.service
[Unit]
Description=Set NVIDIA GPU Power Limits
After=nvidia-persistenced.service

[Service]
Type=oneshot
ExecStart=/usr/bin/nvidia-smi -pm 1
ExecStart=/usr/bin/nvidia-smi -i 0,1,2,3,4 -pl 140
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now nvidia-power.service
```

### UNDO (remove service)
```bash
sudo systemctl disable --now nvidia-power.service
sudo rm /etc/systemd/system/nvidia-power.service
sudo systemctl daemon-reload
```

---

## Full Redo Script (Fresh Install)

Copy this entire block to apply all changes to a new machine:

```bash
#!/bin/bash
# LLM Machine Setup Script
# Run after fresh Ubuntu 25.10 install

set -e

echo "=== Phase 1: Disable services ==="
sudo systemctl disable --now cups cups-browsed avahi-daemon ModemManager \
    switcheroo-control power-profiles-daemon packagekit colord thermald \
    unattended-upgrades anacron wpa_supplicant
sudo systemctl disable --now cups.socket cups.path avahi-daemon.socket anacron.timer

echo "=== Phase 2: Remove snaps ==="
sudo snap remove --purge firefox || true
sudo snap remove --purge snapd-desktop-integration || true
sudo snap remove --purge gnome-42-2204 || true
sudo snap remove --purge gtk-common-themes || true
sudo snap remove --purge bare || true
sudo snap remove --purge core22 || true
sudo systemctl disable --now snapd snapd.socket snapd.seeded.service || true
sudo apt purge -y snapd

echo "=== Phase 2b: Install Firefox from PPA ==="
sudo add-apt-repository -y ppa:mozillateam/ppa
echo 'Package: firefox*
Pin: release o=LP-PPA-mozillateam
Pin-Priority: 1001' | sudo tee /etc/apt/preferences.d/mozilla-firefox
sudo apt update && sudo apt install -y firefox

echo "=== Phase 3: Remove GNOME bloatware ==="
sudo apt purge -y gnome-calculator gnome-characters gnome-clocks \
    gnome-font-viewer gnome-initial-setup gnome-bluetooth-sendto \
    gnome-disk-utility cheese shotwell simple-scan deja-dup \
    transmission-gtk yelp gnome-weather gnome-maps gnome-contacts \
    gnome-calendar totem rhythmbox gnome-tour gnome-user-docs || true
sudo apt autoremove -y

echo "=== Phase 4: Fix NTP ==="
echo "pool pool.ntp.org iburst" | sudo tee /etc/chrony/sources.d/pool-ntp.sources
sudo systemctl restart chrony

echo "=== Phase 5: Passwordless sudo ==="
echo "$USER ALL=(ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/nopasswd-$USER
sudo chmod 440 /etc/sudoers.d/nopasswd-$USER

echo "=== Phase 6: GPU power limits service ==="
cat << 'GPUEOF' | sudo tee /etc/systemd/system/nvidia-power.service
[Unit]
Description=Set NVIDIA GPU Power Limits
After=nvidia-persistenced.service

[Service]
Type=oneshot
ExecStart=/usr/bin/nvidia-smi -pm 1
ExecStart=/usr/bin/nvidia-smi -i 0,1,2,3,4 -pl 140
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
GPUEOF
sudo systemctl daemon-reload
sudo systemctl enable --now nvidia-power.service

echo "=== Done! Reboot recommended ==="
```

---

## Verification Commands

```bash
# Check running services (should be ~20)
systemctl list-units --type=service --state=running | wc -l

# Check boot time
systemd-analyze

# Check GPU status
nvidia-smi --query-gpu=index,temperature.gpu,power.draw,power.limit --format=csv

# Check NTP sync
chronyc tracking
timedatectl status

# Check snap is gone
snap list  # should error
```

---

## Results Summary

| Metric | Before | After |
|--------|--------|-------|
| Running services | 33 | ~20 |
| Snaps installed | 7 | 0 |
| GNOME apps | ~15 | 0 |
| Firefox startup | ~5s (snap) | <1s (native) |

---

## Future Optimizations (Not Yet Applied)

1. **SSD upgrade** - Replace HDD boot drive (removes 32s spinup wait)
2. **Disable Plymouth** - `sudo kernelstub -a "plymouth.enable=0"` (saves ~25s)
3. **Reduce initrd modules** - Remove unused drivers
4. **Switch to terminal login** - Replace GDM with getty for pure CLI
