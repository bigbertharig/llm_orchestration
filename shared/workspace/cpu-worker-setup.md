# CPU Worker Setup — Orange Pi Prime Cluster

## Hardware
- 8x Orange Pi Prime (Allwinner H5, 4-core ARM64, 2GB RAM)
- MicroSD boot (7-8GB cards)
- Gigabit ethernet to dedicated router
- Heatsinks recommended (SoC runs 70-77C idle, will climb under load)

## Network
- Subnet: 10.0.0.0/24
- DHCP with static assignments: workers at 10.0.0.10 through 10.0.0.17
- RPi control plane: 10.0.0.2
- GPU rig: 10.0.0.3

## Base Image

Image file: `worker-image.img.xz` (~289MB compressed, 1.9GB raw)
Path: `/media/bryan/shared/plans/shoulders/research_assistant/docs/worker-image.img.xz`

### Burning
```bash
xzcat /media/bryan/shared/plans/shoulders/research_assistant/docs/worker-image.img.xz | sudo dd of=/dev/sdX bs=4M status=progress conv=fsync
```

### OS
- Armbian Community 26.2.0-trunk.385 (Debian Trixie / Debian 13)
- Kernel: 6.12.68-current-sunxi64
- No desktop — terminal only
- Source: https://dl.armbian.com/orangepiprime/Trixie_current_minimal

### System Config
- Timezone: America/Vancouver (matches control plane)
- Locale: en_US.UTF-8
- Users: `root` (pw: cpuworker), `bryan` (no pw)
- RPi SSH key (`pi-control-plane`) in authorized_keys for both root and bryan
- Apt sources: standard Debian mirrors (Chinese mirror replaced)

### NFS Mount
- Mounts `/media/bryan/shared` from 10.0.0.2 (RPi) on boot
- fstab entry: `10.0.0.2:/media/bryan/shared /media/bryan/shared nfs defaults,nolock,_netdev 0 0`
- Agent code, task queues, and output all live on NFS

### Python
- Python 3.13.5 (from Armbian repos)
- python3-venv, python3-pip installed
- Local venv at `/opt/worker-env` (empty — libs installed per-task as needed)

### Agent
- Agent code lives on NFS: `/media/bryan/shared/agents/`
- Workers run agent from NFS path using local venv
- Single source of truth — update agent once, all workers pick it up
- Agent runs in cpu_only mode (no GPU detection)
- cpu_agent.py in development (separate effort)

### CPU Agent (Current MVP Alignment)
- CPU agent script path (NFS shared):
  - `/media/bryan/shared/scripts/cpu_agent.py`
- Agent behavior:
  - Claims only `task_class: cpu` tasks where `executor: worker`
  - Writes normal task lifecycle files (`queue -> processing -> complete/failed`)
  - Publishes heartbeats to:
    - `/media/bryan/shared/cpus/<worker-name>/heartbeat.json`
    - `/media/bryan/shared/heartbeats/<worker-name>.json`
- Runtime baseline:
  - Prefers venv activation from `/opt/worker-env/bin/activate`
  - Fallback to `/home/bryan/ml-env/bin/activate` if present
  - Converts `python` to `python3` for shell commands
- Test command:
```bash
python3 /media/bryan/shared/scripts/cpu_agent.py --once --name cpu-worker-10
```

### Persistent Logs (No /tmp)
- CPU worker logs should be written to shared persistent storage:
  - `/media/bryan/shared/logs/cpu_workers/`
- Recommended restart helper (keeps logs persistent):
```bash
/media/bryan/shared/scripts/restart_cpu_workers.sh
```
- Target specific workers:
```bash
/media/bryan/shared/scripts/restart_cpu_workers.sh 10.0.0.10 10.0.0.11
```

### First-Boot Script (`/opt/first-boot.sh`)
Runs once on first boot of each cloned image via systemd (`first-boot.service`):
1. Reads IP, extracts last octet
2. Sets hostname to `worker-{octet}` (e.g., `worker-10` for 10.0.0.10)
3. Regenerates SSH host keys (`dpkg-reconfigure openssh-server`)
4. Creates flag file `/opt/.first-boot-done` to prevent re-running
5. Logs to `/var/log/first-boot.log`

Tested: worker-10 confirmed working — hostname set, keys regenerated, NFS mounted on boot.

### Pre-Image Cleanup (already done)
Before the image was captured:
- [x] SSH host keys deleted (first-boot regenerates)
- [x] machine-id cleared (regenerates on boot)
- [x] apt cache cleaned
- [x] Logs cleared
- [x] Bash history cleared
- [x] Filesystem shrunk (resize2fs -M + partition shrink via parted)
- [x] First-boot systemd service enabled

## Post-Clone Per-Worker
1. Burn `worker-image.img.xz` to SD card
2. Boot — first-boot script runs automatically
3. Armbian will auto-expand filesystem to fill the SD card
4. Verify: `ssh root@10.0.0.{octet}` (will need `ssh-keygen -R` for reused IPs)
5. Hostname should be `worker-{octet}`, NFS mounted, venv ready
6. Start CPU agent from control plane:
   ```bash
   /media/bryan/shared/scripts/restart_cpu_workers.sh 10.0.0.{octet}
   ```
7. Install task-specific Python libs into `/opt/worker-env` as needed

## Shopping List
- [ ] 8x MicroSD cards (8GB+)
- [ ] 8x heatsinks for H5 SoC
- [ ] Quality multi-port USB power (avoid cheap converters — coil whine)
