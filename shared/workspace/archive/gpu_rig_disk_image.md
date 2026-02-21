# GPU Rig Disk Image (Clonezilla)

**Created**: 2026-02-11
**Priority**: Medium — rig HDD failing, need backup before SSD swap

## Goal

Create a minimal Clonezilla image of the GPU rig's OS drive, saved to the shared 4TB USB drive.

## Pre-Image Cleanup Script

Run this on the GPU rig before booting into Clonezilla:

```bash
#!/bin/bash
# gpu_rig_prep_for_image.sh
# Run as root on GPU rig before Clonezilla imaging

set -e
echo "=== GPU Rig Image Prep ==="

# Stop services that might be writing
systemctl stop ollama 2>/dev/null || true

# Clean Ollama model cache (models live on shared drive as GGUFs)
echo "Cleaning Ollama model cache..."
rm -rf /usr/share/ollama/.ollama/models/blobs/*
rm -rf /usr/share/ollama/.ollama/models/manifests/*

# Clean package cache
echo "Cleaning apt cache..."
apt clean

# Clean temp files
echo "Cleaning temp files..."
rm -rf /tmp/* /var/tmp/*

# Trim journal logs
echo "Trimming journal logs..."
journalctl --vacuum-size=50M

# Clean user caches
echo "Cleaning user caches..."
rm -rf /home/bryan/.cache/pip/*
rm -rf /home/bryan/.cache/huggingface/* 2>/dev/null

# Clear bash history (optional)
# > /home/bryan/.bash_history

# Report disk usage
echo ""
echo "=== Disk Usage After Cleanup ==="
df -h /
echo ""
du -sh /usr /var /home /opt 2>/dev/null | sort -rh
echo ""
echo "Ready for Clonezilla imaging."
echo "Expected image size: ~15-20GB compressed"
```

Save as `/mnt/shared/scripts/gpu_rig_prep_for_image.sh` on the rig.

## What's Preserved in the Image

- Ubuntu OS + kernel
- NVIDIA drivers + CUDA toolkit
- Ollama binary (no models — reload from shared drive GGUFs)
- Python 3 + pip + venv setup
- NFS client + server packages
- SSH server + config
- Netplan config (`/etc/netplan/99-pi-link.yaml` — static IP 10.0.0.2/24)
- fstab entries (NFS mount to /mnt/shared)
- User account (bryan)

## What's NOT in the Image

- Ollama models (30GB+ — rebuild from `/mnt/shared/models/` GGUFs)
- Anything in `/mnt/shared` (NFS mount, not local)
- Pip/apt caches
- Temp files, old logs

## Clonezilla Procedure

1. Run the prep script on the GPU rig
2. Shut down the rig
3. Plug shared USB drive into a port Clonezilla can see (or use NFS)
4. Boot from Clonezilla USB
5. Choose: `device-image` → `savedisk`
6. Mount target: the 4TB shared drive (label: `shared`)
7. Save to: `shared/images/gpu-rig-YYYY-MM-DD/`
8. Source: the rig's OS disk (likely `/dev/sda`)
9. Use `-z9p` (parallel zstd) for best compression
10. After imaging, boot rig normally and re-import models

## Restoring to New SSD

1. Install new SSD in rig
2. Boot Clonezilla USB
3. `device-image` → `restoredisk`
4. Source: `shared/images/gpu-rig-YYYY-MM-DD/`
5. Target: the new SSD
6. After restore, expand partition if SSD is larger:
   ```bash
   sudo growpart /dev/sda 1
   sudo resize2fs /dev/sda1
   ```
7. Re-import Ollama models from shared drive

## Re-importing Models After Restore

```bash
# For each model in /mnt/shared/models/:
cd /mnt/shared/models/qwen2.5-coder-32b
sudo -u ollama ollama create qwen2.5-coder:32b -f Modelfile

cd /mnt/shared/models/qwen2.5-coder-7b
sudo -u ollama ollama create qwen2.5-coder:7b -f Modelfile
# ... etc
```
