# System Preparation For The GPU Rig

How the GPU rig host (<rig-hostname>, <rig-ip>) is built, and how to rebuild it. Machine prep only:
daily operation is the rig hardware layer (`shared/scripts/rig/README.md`) driven by the controller
on the Pi (llm-gateway `docs/MANAGER.md`). CPU workers have their own standard:
[cpu-worker-setup.md](cpu-worker-setup.md).

Last updated: 2026-10-20 (rewritten from the live rig; the earlier version described the host as of
early 2026).

Tools (in `shared/scripts/rig/`, same pattern as the CPU worker kit):

| File | Role |
|---|---|
| `check_rig.sh [--deep]` | PASS/FAIL readiness checks, exit 1 on any failure; read-only, safe while models run (`--deep` also starts a CUDA container on every GPU) |
| `prepare_rig.sh [--apply]` | brings the host settings to this standard; idempotent; shows what it would change unless `--apply` |
| `systemd/` | every rig unit, `earlyoom.default`; `rig-performance-tune.sh` next to it |

---

## The Machine

| Part | Value |
|---|---|
| Board | B250 MINING EXPERT, BIOS 1001 (2017-12-13); PCIe speed set to Gen2 board-wide (item 8) |
| CPU / RAM | i7-7700K (8 threads), 31.4 GB RAM; RAM is the scarce resource, not VRAM |
| Boot disk | SATA SSD 109 GB (`/`), ext4; holds `~/model-hotset` (SATA hot-set) |
| NVMe | 469 GB at `/data` (fstab by UUID, `noatime,nofail`); holds `/data/model-hotset` |
| Shared drive | the Pi's 4 TB drive over NFS at `/mnt/shared` (models archive, logs, results) |
| Network | onboard `<ethernet-nic>`, DHCP from the rig router (<rig-ip>); a saved Wi-Fi profile exists for `<wifi-nic>` in netplan (not used; the operator handles the airgap) |
| PSUs | 1250 W + 1000 W, rearranged and stress-tested 2026-10-19 |

GPUs (PCIe slot order is the CUDA index order):

| Index | Card | Bus | Factory limit | Planned limit (`hw_power.py`) |
|---|---|---|---|---|
| 0 | RTX 3090 Ti (EVGA) | 01:00.0 | 450 W | 450 W |
| 1 | GTX 1060 6GB, EVGA single fan (3842:6163), pair of GPU 3 | 02:00.0 | 120 W | 100 W |
| 2 | GTX 1060 6GB, ASUS 156 W OC (1043:85ac), pair of GPU 5 | 03:00.0 | 156 W | 120 W |
| 3 | GTX 1060 6GB, EVGA single fan (3842:6163), pair of GPU 1 | 07:00.0 | 120 W | 100 W (links at Gen1) |
| 4 | GTX 1060 6GB, ASUS DUAL-GTX1060-O6G white 2-fan (1043:85b6), the unique card | 0D:00.0 | 120 W | 100 W |
| 5 | GTX 1060 6GB, ASUS 156 W OC (1043:85ac), pair of GPU 2 | 10:00.0 | 156 W | 120 W |

Boards are identified by PCI subsystem ID (`lspci -s <bus> -vnn`), not by slot. All 1060s sit on x1
risers; the 3090 Ti is on x16. Hottest under sustained single-card load (2026-10-20): GPU 3, a single-fan
EVGA at 82-83 C with its fan at 90 percent; the 1060s report a 99 C slowdown and 102 C shutdown, and their
GPU Boost target is 83 C (clocks drop above it). After moving cards, rerun the stress compare
(llm-gateway `tools/rig_stress.py --compare`) and update this table.

---

## NVIDIA Driver (the part that matters most)

The rig mixes Pascal (GTX 1060, sm_61) and Ampere (RTX 3090 Ti, sm_86). One driver has to run both:

- **Branch 580, proprietary.** 580 is NVIDIA's last driver branch with Pascal support; later branches
  drop the 1060s. Use `nvidia-driver-580`, never an `-open` package: the open kernel modules only
  support Turing and newer, so they would not drive the 1060s. Never let `ubuntu-drivers install`
  pick for you.
- **Installed:** 580.159.03 from Ubuntu `questing-updates/restricted` (`nvidia-driver-580`
  580.159.03-0ubuntu0.25.10.1), built by DKMS for each kernel (Secure Boot is disabled, so the
  unsigned DKMS module loads). Container side: `nvidia-container-toolkit` 1.18.2 from NVIDIA's apt
  repo, `docker.io` 28.2.2 from Ubuntu, `/etc/docker/daemon.json` registers the `nvidia` runtime.
- **CUDA inside the images is pinned to 12.9.2** with architectures `61;86`
  (`shared/scripts/llama_runtime/Dockerfile`): CUDA 13 dropped sm_61. The host driver (580, CUDA 13
  capable) runs CUDA 12.9 containers fine.
- **History:** 580.126.09 installed 2026-02-11; 580.159.03 arrived on 2026-06-08 through
  `unattended-upgrade` (with a new kernel), not by hand. An unattended driver upgrade while models
  are loaded leaves the userspace libraries newer than the loaded kernel module, and NVML fails
  until a reboot.
- **Policy (from 2026-10-20):** the driver changes only by hand. `prepare_rig.sh --apply` disables
  the `apt-daily` timers and holds every installed 580 package. `check_rig.sh` fails if the loaded
  module and the userspace version differ, if the branch is not 580, if an open driver is installed,
  or if a 580 package is not held.

Deliberate driver or kernel update (rig idle, no leases):

```bash
sudo apt-mark unhold $(apt-mark showhold | grep nvidia)
sudo apt update && sudo apt install --only-upgrade 'nvidia-*-580' 'libnvidia-*-580' linux-generic-hwe-24.04
dkms status                       # nvidia/<new version> installed for the new kernel
sudo reboot
bash ~/llm_orchestration/shared/scripts/rig/check_rig.sh --deep
bash ~/llm_orchestration/shared/scripts/rig/prepare_rig.sh --apply    # re-holds the packages
```

Then run the runtime smoke (`shared/scripts/llama_runtime/README.md`) before benchmarks.

---

## Rebuild From Scratch

For a replacement rig at <rig-ip>. A second rig next to this one is not supported by the gateway yet
(llm-gateway `docs/DESIGN_NOTES.md`, "More than one rig").

1. **BIOS:** PCIe speed Gen2 (item 8), boot from the SATA SSD.
2. **OS:** Ubuntu desktop (today 25.10, kernel 6.17), user `bryan`, auto-login (item 7). 25.10 is an
   interim release with 9 months of standard support; a rebuild should use the current LTS once it is
   confirmed to ship the 580 driver for Pascal.
3. **Base packages and trim:** `sudo apt install -y git curl chrony nfs-common earlyoom zram-tools
   python3-venv openssh-server`, then the service and snap trims in the command record below.
4. **Driver:** `sudo apt install -y nvidia-driver-580`, reboot, `nvidia-smi` shows all six GPUs.
5. **Containers:** `sudo apt install -y docker.io`; add NVIDIA's container toolkit repo
   (`https://nvidia.github.io/libnvidia-container/stable/deb/`) and
   `sudo apt install -y nvidia-container-toolkit`.
6. **Pi access:** put the Pi's public key in `~/.ssh/authorized_keys`, create a rig key and add it to
   the Pi's `authorized_keys`, and add `Host pi / HostName <pi-ip>` to `~/.ssh/config`. The
   controller and the benchmark lanes both need ssh in each direction without a prompt.
7. **Repos:** clone from the Pi hubs (`RIG_PROJECT.md` section 3):
   `git clone pi:llm_orchestration ~/llm_orchestration`, then benchmarking into
   `~/llm_orchestration/shared/plans/shoulders/benchmarking` (branch `master`).
8. **Host settings:** `bash ~/llm_orchestration/shared/scripts/rig/prepare_rig.sh` (review), then
   `--apply`. This sets sudo, the update policy and driver hold, every rig unit, the tuning script,
   `/usr/local/bin/rig`, earlyoom, swappiness, chrony from the Pi, the share automount, Docker group
   and runtime, the venv and the hot-set folders.
9. **Machine config:** `shared/agents/config.json` is gitignored and per machine (machine facts only,
   never model names). Generate it with `shared/agents/setup.py`.
10. **Images:** `shared/scripts/llama_runtime/build_image.sh <runtime tag>` with the tag from
    `runtime_image` in benchmarking `model_registry.json`, then promote it
    (`shared/scripts/llama_runtime/README.md`); build each bench image with
    `docker build -t bench-<name> bench-<name>` in benchmarking `docker/`.
11. **Models:** the archive is on the shared drive. To fill the hot-sets use
    `shared/scripts/manage_model_hotset.py`. A model missing from the archive is re-downloaded from
    its `source_url` in `model_registry.json`.
12. **Gateway copy:** from the Pi, `~/llm-gateway/deploy/deploy_pi.sh` copies llm-gateway to the rig
    (`~/llm-gateway/remote_profiles.json`) and restarts the Pi services.
13. **Verify:** `check_rig.sh --deep` shows 0 failed; from the Pi, `rigctl.py status` shows six
    `free` GPUs and the guard ok; reboot once and check that the units come back.

---

## Applied Host Changes

### 1. Disabled Unnecessary Services

cups, cups-browsed, avahi-daemon, ModemManager, switcheroo-control, colord, thermald,
wpa_supplicant and anacron are disabled; NetworkManager-wait-online is disabled for boot speed.
Ollama (the old runtime) is masked; its units remain on disk unused.

power-profiles-daemon must be masked, not just disabled: GNOME's `gsd-power` starts it over D-Bus at
login, and it sets the CPU governor back to `powersave` after `rig-performance-tune` has set
`performance` (found 2026-10-20: all 8 threads on powersave with the daemon disabled).
`prepare_rig.sh` masks it.

### 2. Removed Snap And Desktop Bloat

GNOME extras were purged (command record below). Snap is back as of 2026-10-20: `snapd` is enabled
and Firefox runs as a snap with its bases and `mesa-2404`. Snaps refresh themselves; nothing on the
GPU path depends on them.

### 3. Time From The Pi Only

The Pi runs chrony and serves `<rig-subnet>` (since 2026-10-15). The rig syncs only from it:
`/etc/chrony/sources.d/pi.sources` = `server <pi-ip> iburst`; the Ubuntu pool file is renamed
`.disabled`. The pool lines in `/etc/chrony/chrony.conf` remain and fail quietly without internet.

### 4. Passwordless Sudo For bryan

`/etc/sudoers.d/010_pi-nopasswd` = `bryan ALL=(ALL) NOPASSWD: ALL`. The controller and the harness
run system commands over ssh. Same standard as the Pi and the CPU workers.

### 5. GPU Power Limits

`rig-power.service` runs `hw_power.py apply` at boot: GTX 1060s at 100 W, the factory-overclocked
156 W boards at 120 W, others at their default (planned limits in the GPU table). Rig-local
`brain/hw/power.json` overrides; `/run/rig/power_override.json` overrides until the next boot.
`run_runtime.sh` refuses a load whose cards are off plan. Persistence mode is set by
`rig-performance-tune.service`, which also sets the CPU governor to `performance` and swappiness 10.
The old `nvidia-power.service` (140 W on GPUs 0-4) is gone.

### 6. Driver For Both The 3090 Ti And The 1060s

See "NVIDIA Driver" above: branch 580, proprietary, held, updated only by hand.

### 7. Reduced Boot Friction

GDM auto-login for `bryan` (`/etc/gdm3/custom.conf`), graphical target, NetworkManager-wait-online
disabled. The rig boots from the SATA SSD. `rig-boot.service` brings the rig layer up at boot
(fixed 2026-10-18: it failed with ENODEV on the share automount).

### 8. PCIe Set To Gen2 In BIOS (2026-10-19)

The BIOS PCIe speed setting is board-wide (no per-slot control). Measured on 2026-10-19:

- Gen2: zero link replays on every GPU through a stress run, a 5-way split load, a 10-minute split
  traffic test and a split sweep. NVMe (x1) reads 434 MB/s direct.
- Gen3: NVMe reads 847 MB/s direct and a cold 5-way split load drops from 132 s to 90 s, but GPUs
  1, 2, 4 and 5 logged link replays from boot (tens of thousands after testing; GPUs 2 and 4 hit
  the 65535 counter cap). No AER or Xid, but the risers are marginal. Inference speed did not change.
- Decision: Gen2. GPU 3 links at Gen1 regardless (its slot or riser).
- Check after any riser or BIOS change: `nvidia-smi -q | grep "Replays Since Reset"` before and after
  a split load (`check_rig.sh` prints the total).

Details: benchmarking `BENCHMARK_LESSONS_LEARNED.md`, "PCIe generation and split scaling".

### 9. Windows Desktop As A WSL Worker (2026-10-19)

A Windows 10 desktop can join as a second rig through WSL2 Ubuntu. Scripts and steps:
`shared/scripts/desktop_worker/README.md`.

- Windows NVIDIA driver on the 580 branch (582.78 for Pascal cards); WSL uses the Windows driver.
- Native `docker-ce` and `nvidia-container-toolkit` inside Ubuntu (not Docker Desktop).
- Windows OpenSSH server with `bash.exe` as its default shell, key-only, Ethernet-only firewall rules.
- Manual worker mode switch; it does not auto-start at boot.
- 2026-10-20: paused (`teardown_desktop_rig_ssh.ps1 -Level pause`): sshd stopped and disabled, the
  firewall rules disabled; the key and config are kept, so setup can turn it back on.

### 10. Memory Policy (2026-10-17)

zram swap 15 GB (`zramswap`, zstd, 50 percent of RAM, priority 100) plus `/swap.img` 8 GB; earlyoom
RAM-only (`-s 100`, settings tracked as `systemd/earlyoom.default`); `rig-guard.service` frees RAM
itself at critical. Never add swap to relieve memory pressure: on 2026-10-17 a swapfile disarmed
earlyoom and hung the rig until a hard reset.

---

## Rig-Local State (Not In Git)

Lost with the rig and rebuilt by the steps above:

- `shared/agents/config.json` (machine facts; regenerate with `setup.py`)
- `shared/brain/hw/`: lease and claim record (mirrored to `/mnt/shared/gpus/`), guard state and
  events, lease audit log, plan queue; optional `power.json` and `guard.json` overrides. Current
  `guard.json` (2026-10-20): `{"gpu_temp_c": {"warn": 85, "critical": 90}}` (default warn 83 flapped on
  GPU 3); restart `rig-guard` after changing it
- the hot-sets (copies of archive models) and the Docker images (rebuilt from the repos)
- the ssh keys

---

## Historical Command Record

Commands from the original prep that still describe the host. Read them as rebuild notes.

### Disable Unneeded Services

```bash
sudo systemctl disable --now cups cups-browsed avahi-daemon ModemManager \
    switcheroo-control power-profiles-daemon colord thermald wpa_supplicant anacron
sudo systemctl disable --now cups.socket cups.path avahi-daemon.socket anacron.timer
sudo systemctl disable NetworkManager-wait-online.service
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

### Auto-Login

```bash
sudo sed -i '/^\[daemon\]/,/^\[/{s/#  AutomaticLoginEnable = true/AutomaticLoginEnable = true/; s/#  AutomaticLogin = user1/AutomaticLogin = bryan/}' /etc/gdm3/custom.conf
```

### Emergency Clock Set

```bash
sudo date -s "$(curl -sI google.com | grep -i '^date:' | cut -d' ' -f3-6)"
sudo chronyc -a makestep
```

Retired (kept for reference only): the snap removal and native Firefox PPA steps (snap is back,
item 2), `nvidia-power.service` (replaced by `rig-power.service`, item 5), the `pool.ntp.org` chrony
source (replaced by the Pi, item 3), and the 580.126.09 `dkms install --force` step (the package's
DKMS hook builds the module itself).

---

## Notes To Keep Honest

- Update this doc and `check_rig.sh` together; a host change that is not in both is drift.
- If a step is no longer wanted, move it to "Retired" instead of leaving it as if it were current.
