#!/bin/bash
# gpu_rig_prep_for_image.sh
# Run as root on GPU rig before Clonezilla imaging
# Strips the OS drive down to essentials for a minimal image

set -e
echo "=== GPU Rig Image Prep ==="

# Stop services and containers that might be writing
echo "Stopping llama runtime containers..."
docker rm -f $(docker ps -aq --filter "name=llama-") 2>/dev/null || true

# Clean legacy Ollama cache leftovers if present
echo "Cleaning legacy runtime cache leftovers..."
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
rm -rf /home/bryan/.cache/pip/* 2>/dev/null
rm -rf /home/bryan/.cache/huggingface/* 2>/dev/null

# Unmount NFS
echo "Unmounting NFS..."
umount /mnt/shared 2>/dev/null || true

# Report disk usage
echo ""
echo "=== Disk Usage After Cleanup ==="
df -h /
echo ""
du -sh /usr /var /home /opt 2>/dev/null | sort -rh
echo ""
echo "Ready for Clonezilla imaging."
echo "Shut down and boot from Clonezilla USB."
