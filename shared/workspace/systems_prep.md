# System Preparation For The GPU Rig

Reference for host-level operating-system preparation on the GPU rig.

This is a machine-prep and recovery document, not the normal orchestration
runbook. Do not confuse these host changes with the supported daily operator
workflow in [quickstart.md](quickstart.md).

---

## Purpose

This document exists for three reasons:

1. rebuild or recover a host after reinstall
2. understand which host-level optimizations were applied
3. distinguish machine setup from orchestration behavior

The orchestration system should normally be started through:

- `shared/agents/startup.py`
- `scripts/start_default_mode.py`
- `scripts/start_benchmark_mode.py`
- `scripts/start_custom_mode.py`

Not by redoing machine-prep steps during ordinary operation.

---

## Machine-Prep Themes

The GPU rig was prepared as a dedicated orchestration host with these goals:

- fewer unnecessary background services
- explicit control over updates and power behavior
- predictable boot behavior
- compatible NVIDIA driver support for mixed-generation GPUs
- explicit shared-drive / local-runtime split

---

## Applied Host Changes

### 1. Disabled Unnecessary Services

Intent:

- reduce background noise
- shorten boot path
- avoid desktop/server services that do not help the rig

Examples include printing, modem, WiFi, and desktop convenience services that
are irrelevant to a fixed-function local orchestration host.

### 2. Removed Snap And Desktop Bloat

Intent:

- avoid heavyweight default packages
- keep disk usage and background churn lower
- make the machine easier to reason about

This is a host policy choice, not an orchestration requirement.

### 3. Fixed Time Sync Behavior

Intent:

- make NTP recovery work even when the clock is already wrong

Accurate time matters for:

- task timestamps
- heartbeat staleness
- batch history ordering
- summary/event timelines

### 4. Enabled Passwordless Sudo For The Operator Account

Intent:

- make controlled local maintenance practical

This is a host-management convenience decision and should be treated as part of
the trusted operator environment, not something agents should modify.

### 5. Applied Persistent GPU Power Limits

Intent:

- keep the worker GPUs within predictable thermal/power behavior

### 6. Installed Driver Version Compatible With Both 3090 Ti And GTX 1060

Intent:

- keep mixed Ampere + Pascal support on one host

This matters because newer driver tracks can break Pascal support.

### 7. Reduced Boot Friction

Examples:

- auto-login
- reduced wait behavior during boot
- SSD boot migration

These are operator convenience and startup-latency choices.

---

## Important Boundary

Host prep is not the same thing as runtime orchestration.

Current orchestration behavior should be understood as:

- the brain and workers are started by orchestrator startup scripts
- worker runtime ownership is coordinated by the orchestrator
- worker model state is not supposed to be maintained primarily by ad hoc
  always-on host services

Historical host notes about fixed runtime instances are useful context, but they
do not define the normal operator path anymore.

---

## Recovery / Rebuild Checklist

After a reinstall or major repair, verify at minimum:

1. shared drive mounts correctly
2. NVIDIA driver supports the full GPU set
3. `nvidia-smi` sees all expected GPUs
4. the operator Python environment exists
5. orchestrator startup scripts run
6. shared-model archive is reachable
7. timestamps and time sync are sane

Useful checks:

```bash
nvidia-smi -L
mount | grep /mnt/shared
timedatectl status
chronyc tracking
pgrep -af "brain.py|gpu.py"
```

---

## Historical Command Record

The following sections keep the original command-oriented prep history because
they are useful during a rebuild. They should be read as host-recovery notes.

### Disable Unneeded Services

```bash
sudo systemctl disable --now cups cups-browsed avahi-daemon ModemManager \
    switcheroo-control power-profiles-daemon packagekit colord thermald \
    unattended-upgrades anacron wpa_supplicant

sudo systemctl disable --now cups.socket cups.path avahi-daemon.socket anacron.timer
```

### Remove Snap

```bash
sudo snap remove --purge firefox
sudo snap remove --purge snapd-desktop-integration
sudo snap remove --purge gnome-42-2204
sudo snap remove --purge gtk-common-themes
sudo snap remove --purge bare
sudo snap remove --purge core22

sudo systemctl disable --now snapd snapd.socket snapd.seeded.service
sudo apt purge -y snapd
```

### Install Native Firefox

```bash
sudo add-apt-repository -y ppa:mozillateam/ppa
echo 'Package: firefox*
Pin: release o=LP-PPA-mozillateam
Pin-Priority: 1001' | sudo tee /etc/apt/preferences.d/mozilla-firefox
sudo apt update && sudo apt install -y firefox
```

### Remove GNOME Bloat

```bash
sudo apt purge -y \
    gnome-calculator gnome-characters gnome-clocks \
    gnome-font-viewer gnome-initial-setup gnome-bluetooth-sendto \
    gnome-disk-utility cheese shotwell simple-scan deja-dup \
    transmission-gtk yelp gnome-weather gnome-maps gnome-contacts \
    gnome-calendar totem rhythmbox gnome-tour gnome-user-docs

sudo apt autoremove -y
```

### Fix NTP

```bash
echo "pool pool.ntp.org iburst" | sudo tee /etc/chrony/sources.d/pool-ntp.sources
sudo systemctl restart chrony
```

Emergency clock set:

```bash
sudo date -s "$(curl -sI google.com | grep -i '^date:' | cut -d' ' -f3-6)"
sudo chronyc -a makestep
```

### Enable Passwordless Sudo

```bash
echo "$USER ALL=(ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/nopasswd-$USER
sudo chmod 440 /etc/sudoers.d/nopasswd-$USER
```

### Persistent GPU Power Limits

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

### Install NVIDIA 580-Series Driver

```bash
sudo apt install -y nvidia-driver-580
sudo dkms install --force nvidia/580.126.09 -k $(uname -r)
sudo update-initramfs -u
sudo reboot
```

### Auto-Login / Boot Friction Reduction

```bash
sudo sed -i '/^\[daemon\]/,/^\[/{s/#  AutomaticLoginEnable = true/AutomaticLoginEnable = true/; s/#  AutomaticLogin = user1/AutomaticLogin = bryan/}' /etc/gdm3/custom.conf
sudo systemctl disable NetworkManager-wait-online.service
```

---

## Notes To Keep Honest

- This doc records host choices, not universal requirements for every future rig.
- If the rig architecture changes, update the high-level sections first and only
  then decide which old command blocks are still worth keeping.
- If a prep step is no longer desirable, archive it instead of silently leaving
  it here as if it were still the recommended baseline.
