# GPU Rig: Shared Drive + Pi Networking Runbook

Date: 2026-02-12

## What stays local on the rig

These are intentionally local (not on shared drive):
- `/home/bryan/bin/llm-orchestrator-start.sh`
- `/etc/systemd/system/llm-orchestrator.service`

Purpose:
- Prevent boot failure if NFS/shared mount is slow.
- Wait for `/mnt/shared/agents/startup.py`, then launch orchestrator.

## First boot on a fresh imaged rig

1. Connect ethernet to your control LAN.
2. Boot rig and confirm SSH reachable.
3. Ensure Wi-Fi is removed/disabled.

## Connect shared drive (NFS)

Expected mount:
- Server: `10.0.0.1:/media/bryan/shared`
- Mountpoint: `/mnt/shared`

Check fstab line:
```bash
grep -n '/mnt/shared' /etc/fstab
```

Mount + verify:
```bash
sudo mkdir -p /mnt/shared
sudo mount -a
mount | grep '/mnt/shared'
ls -la /mnt/shared/agents
```

## Pi networking prep (control-plane)

Goal:
- Deterministic LAN path between Pi/control host and rig.
- No unexpected route changes via Wi-Fi.

On rig:
```bash
ip -br a
ip route
```

Expected:
- Ethernet interface up on control subnet (example `10.0.0.0/24`).
- Default/control route uses ethernet, not Wi-Fi.

If needed, disable Wi-Fi stack:
```bash
sudo nmcli radio wifi off
```

Optional static addressing (if you standardize by image):
- Reserve DHCP IP by MAC in router (preferred), or
- Set static netplan profile per rig.

## Move models from shared storage to local SSD

Example local model staging paths:
- `/home/bryan/local-models/`
- `/var/lib/ollama-import/`

Copy only selected models (keep image small):
```bash
mkdir -p /home/bryan/local-models
rsync -avh /mnt/shared/models/<model-folder>/ /home/bryan/local-models/<model-folder>/
```

## Import/register selected models locally

Brain (11434):
```bash
OLLAMA_HOST=127.0.0.1:11434 ollama create <brain-tag> -f /home/bryan/local-models/<brain-model>/Modelfile
```

Workers (11435-11438):
```bash
for p in 11435 11436 11437 11438; do
  OLLAMA_HOST=127.0.0.1:$p ollama create <worker-tag> -f /home/bryan/local-models/<worker-model>/Modelfile
done
```

## Start orchestration

Enable/start when models are ready:
```bash
sudo systemctl enable --now llm-orchestrator.service
```

3-minute readiness check:
```bash
sleep 180
systemctl is-active mnt-shared.mount
systemctl is-active llm-orchestrator.service
ss -ltnp | grep -E ':1143[4-8]\b'
pgrep -af 'agents/(startup|brain|gpu)\.py'
```

## Pre-image cleanup (keep image minimal)

Stop orchestrator and remove local model payloads:
```bash
sudo systemctl disable --now llm-orchestrator.service
sudo rm -rf /home/bryan/local-models/*
sudo rm -rf /var/lib/ollama-import/*
sudo rm -rf /home/bryan/.ollama/models/*
sudo rm -rf /usr/share/ollama/.ollama/models/*
```

Confirm free space:
```bash
df -h /
```

## Success criteria

- Local bootstrap files exist and are unchanged.
- Shared drive mounts cleanly.
- Pi/control LAN path is deterministic over ethernet.
- Models are copied/imported locally only when needed.
- Reboot + 3 minute wait yields brain + worker ports up.
