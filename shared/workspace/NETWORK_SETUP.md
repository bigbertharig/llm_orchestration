# Network Setup — LLM Orchestration Network

Last updated: 2026-02-11

## Network Topology

```
Internet
   │
Home Router
   │ LAN port (ethernet)
   │
TP-Link Archer C20 (10.0.0.1) ── WAN port
   │
   │ LAN ports ──→ Switch(es)
   │
   ├── Raspberry Pi       (10.0.0.2)   Control plane, NFS server
   ├── GPU Rig            (10.0.0.3)   Brain + LLM workers
   └── Orange Pi workers  (10.0.0.10+) CPU workers (future, ×8)
```

All devices on `10.0.0.0/24` subnet. Internet flows through the TP-Link router's WAN connection to the home router.

## IP Addressing

| Device | IP | Method | MAC |
|--------|----|--------|-----|
| TP-Link Router | 10.0.0.1 | Static (LAN IP) | — |
| Raspberry Pi | 10.0.0.2 | DHCP reservation | 88:A2:9E:8E:FE:88 |
| GPU Rig | 10.0.0.3 | DHCP reservation | 18:31:BF:27:65:7D |
| Orange Pi workers | 10.0.0.10+ | DHCP pool | — |

DHCP pool: `10.0.0.10` – `10.0.0.254`

## Router — TP-Link Archer C20 AC750

**Admin access**: `http://10.0.0.1` — locked to Pi's MAC address only.

### Security Hardening (applied 2026-02-11)

| Setting | Status |
|---------|--------|
| WiFi (both bands) | Disabled |
| WPS | Disabled |
| SPI Firewall | Enabled |
| DoS Protection | Enabled (ICMP/UDP/TCP-SYN flood) |
| Ping from WAN | Blocked |
| Ping from LAN | Allowed |
| UPnP | Disabled |
| Remote Management | Disabled (IP 0.0.0.0) |
| Admin access | MAC-locked to Pi |
| VPN Passthrough (PPTP/L2TP/IPSec) | Disabled |
| ALG (FTP/TFTP/H323/SIP/RTSP) | All disabled |
| Port forwarding | None |

### Security Notes

- **No wireless entry point** — WiFi disabled on router, WiFi and Bluetooth disabled on Pi
- **Physical access = full access** — anyone plugging into the switch is on the network with no authentication (NFS, SSH keys, Ollama all trust the LAN)
- **Router is the perimeter** — all external traffic flows through WAN NAT, no ports forwarded inbound
- **Admin panel** — change default password, keep firmware updated

## Hardware — GPU Rig

| GPU | Model | VRAM | Role |
|-----|-------|------|------|
| 0 | RTX 3090 Ti | 24GB | Primary — large models (32B) |
| 1 | GTX 1060 6GB | 6GB | Secondary — small models (7B) |
| 2 | GTX 1060 6GB | 6GB | Available |
| 3 | GTX 1060 6GB | 6GB | Available |
| 4 | GTX 1060 6GB | 6GB | Available |

**Driver:** NVIDIA 580.126.09 (proprietary) — required for both Ampere (3090 Ti) and Pascal (1060s). The 590 driver dropped Pascal support; the open kernel driver also doesn't support Pascal.

**Boot config:**
- GRUB: 0s timeout, hidden menu (no OS selection delay)
- GDM: auto-login as `bryan` (no login screen)
- NetworkManager-wait-online: disabled (faster boot)

## Ollama Instances

Two Ollama instances run as systemd services, pinned to specific GPUs:

| Service | Port | GPU | CUDA_VISIBLE_DEVICES | Models |
|---------|------|-----|---------------------|--------|
| `ollama.service` | 11434 | RTX 3090 Ti | 0 | qwen2.5-coder:32b |
| `ollama-1060.service` | 11435 | GTX 1060 #1 | 1 | qwen2.5-coder:7b |

Models are not stored on the SSD image. After imaging/restoring, reload from shared drive:
```bash
cd /mnt/shared/models/qwen2.5-coder-32b && ollama create qwen2.5-coder:32b -f Modelfile
cd /mnt/shared/models/qwen2.5-coder-7b && OLLAMA_HOST=http://localhost:11435 ollama create qwen2.5-coder:7b -f Modelfile
```

## Shared Drive

- Physical disk: 3.6TB WD HDD (WDC WD40EDAZ-11SLVB0) attached to the Pi via USB
- Filesystem: ext4
- UUID: `be91c4c6-bb29-467d-85c2-d39f2358b156`

### Pi mounts (server)

| Mount Point                          | Type | Source    |
|--------------------------------------|------|-----------|
| `/media/bryan/shared`                | ext4 | /dev/sda1 |
| `/home/bryan/llm_orchestration/shared` | bind | above   |

Pi fstab entries:
```
UUID=be91c4c6-bb29-467d-85c2-d39f2358b156  /media/bryan/shared  ext4  defaults,nofail  0  2
/media/bryan/shared  /home/bryan/llm_orchestration/shared  none  bind,nofail  0  0
```

### GPU Rig mount (client)

| Mount Point   | Type | Source                              |
|---------------|------|-------------------------------------|
| `/mnt/shared` | nfs  | 10.0.0.2:/media/bryan/shared        |

GPU Rig fstab entry:
```
10.0.0.2:/media/bryan/shared  /mnt/shared  nfs  defaults,noatime,nofail,mountport=4002,nfsvers=3  0  0
```

Systemd auto-mount service: `mount-shared.service`
- Waits for network, pings Pi, then mounts
- Desktop shortcut (`Mount-Shared-Drive.sh`) available if manual mount needed

## NFS Configuration

Server: Pi (10.0.0.2)

Export (`/etc/exports`):
```
/media/bryan/shared 10.0.0.0/24(rw,sync,no_subtree_check,no_root_squash)
```

Fixed ports (`/etc/nfs.conf`):

| Service     | Port |
|-------------|------|
| portmapper  | 111  |
| nfsd        | 2049 |
| lockd       | 4001 |
| mountd      | 4002 |
| statd       | 4003 |

## SSH

Bidirectional passwordless SSH between Pi and GPU Rig.

| From    | To       | Command    | Key             |
|---------|----------|------------|-----------------|
| Pi      | GPU Rig  | `ssh gpu`  | pi-control-plane (ed25519) |
| GPU Rig | Pi       | `ssh pi`   | gpu-rig (ed25519)          |

Keys stored on shared drive: `/shared/scripts/gpu_rig_ed25519[.pub]`

## Chat Scripts (on Pi)

Quick access to LLM chat sessions from the Pi:

| Command | What it does |
|---------|-------------|
| `chat brain` | Chat with 32B model (current terminal) |
| `chat worker` | Chat with 7B model (current terminal) |
| `chat b --new` | Chat with 32B model (new window) |
| `chat w --new` | Chat with 7B model (new window) |

Scripts in `~/llm_orchestration/scripts/` (chat, chat-brain, chat-worker).

## Setup Scripts (on shared drive)

| Script                                  | Purpose                              | Run on    |
|-----------------------------------------|--------------------------------------|-----------|
| `/shared/scripts/setup-sudo-permissions.sh` | Replicate Pi sudo/groups config  | GPU Rig   |
| `/shared/scripts/setup-ssh.sh`              | Install SSH keys + config        | GPU Rig   |

## GPU Rig Desktop Files

| File | Purpose |
|------|---------|
| `Mount-Shared-Drive.sh` | Manual mount of NFS shared drive |
| `NETWORK_SETUP.md` | This document (offline reference) |

## Troubleshooting

**Can't reach router admin panel:**
1. Ensure you're on the Pi (MAC-locked admin access)
2. Go to `http://10.0.0.1`

**Shared drive not mounted on GPU Rig:**
1. Double-click `Mount-Shared-Drive.sh` on the Desktop
2. Or run: `sudo mount /mnt/shared`

**NFS mount hangs:**
1. Check Pi is reachable: `ping 10.0.0.2`
2. Check NFS ports from GPU rig: `nc -zw2 10.0.0.2 2049 && nc -zw2 10.0.0.2 4002`
3. If ports unreachable, restart NFS on Pi: `ssh pi "sudo systemctl restart nfs-server"`

**Shared drive not mounted on Pi:**
1. Run: `sudo mount /media/bryan/shared && sudo mount /home/bryan/llm_orchestration/shared`

**SSH host key changed (after reimage):**
1. `ssh-keygen -R 10.0.0.3` (on Pi) or `ssh-keygen -R 10.0.0.2` (on GPU Rig)

**1060s not detected (only 3090 Ti visible):**
1. Check driver: must be `nvidia-driver-580` (proprietary), NOT `590` or `open`
2. Verify: `sudo dmesg | grep NVRM` — look for "580.xx Legacy" messages
3. Fix: `sudo apt install nvidia-driver-580`, then `sudo dkms install --force nvidia/580.126.09`, reboot

**Reload models after reimage:**
1. Mount shared drive first
2. `cd /mnt/shared/models/qwen2.5-coder-32b && ollama create qwen2.5-coder:32b -f Modelfile`
3. `cd /mnt/shared/models/qwen2.5-coder-7b && OLLAMA_HOST=http://localhost:11435 ollama create qwen2.5-coder:7b -f Modelfile`

**New device not getting IP:**
1. Check cable and switch LEDs
2. Check router DHCP client list at `http://10.0.0.1`
3. If device needs a fixed IP, add DHCP reservation by MAC address
