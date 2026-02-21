# GPU Rig Network Setup

**Created**: 2026-02-11
**Updated**: 2026-02-11
**Priority**: High — blocks all orchestration work

## Goal

Establish the direct ethernet link between RPi 5 (control plane) and GPU rig so the shared drive is accessible from both machines.

## Network Overview

```
┌─────────────────┐         ethernet         ┌─────────────────┐
│     GPU Rig     │◄───────────────────────►│       Pi        │
│   10.0.0.2      │        1 Gbps            │    10.0.0.1     │
└─────────────────┘                          └─────────────────┘
```

The shared drive can live in either machine. Whichever has it mounts locally and exports via NFS to the other.

---

## OPTION A: Drive on Pi (ACTIVE SETUP)

```
GPU Rig                              Pi
────────                             ──────
/mnt/shared  ◄── NFS client ───  /media/bryan/shared (local drive)
                                     │
                                 NFS server exports to 10.0.0.0/24
```

### Pi Side (has the drive)

1. Drive is at `/media/bryan/shared` (auto-mounted or via fstab)
2. NFS server exports it:
   ```bash
   sudo apt install nfs-kernel-server
   echo '/media/bryan/shared 10.0.0.0/24(rw,sync,no_subtree_check,no_root_squash)' | sudo tee -a /etc/exports
   sudo exportfs -ra
   sudo systemctl enable --now nfs-kernel-server
   ```

### GPU Rig Side (mounts via NFS)

1. Apply static IP (once ethernet connected):
   ```bash
   sudo netplan apply
   # Verify: ip addr show enp0s31f6 should show 10.0.0.2/24
   ```

2. fstab entry (already configured):
   ```
   10.0.0.1:/media/bryan/shared  /mnt/shared  nfs  defaults,noatime,nofail  0  0
   ```

3. Mount:
   ```bash
   sudo mount /mnt/shared
   ls /mnt/shared/agents/  # verify
   ```

---

## OPTION B: Drive on GPU Rig (ALTERNATE SETUP)

```
GPU Rig                              Pi
────────                             ──────
/mnt/shared (local drive)  ───►  /mnt/gpu-share (NFS client)
    │
NFS server exports to 10.0.0.0/24
```

### GPU Rig Side (has the drive)

Already configured:
- Drive: `/dev/sdb1` (UUID: `be91c4c6-bb29-467d-85c2-d39f2358b156`, label: `shared`)
- NFS server installed, export configured: `/mnt/shared 10.0.0.0/24(...)`
- Netplan config at `/etc/netplan/99-pi-link.yaml`

To activate:
```bash
# Mount the local drive
sudo mount /dev/sdb1 /mnt/shared

# Apply network config
sudo netplan apply

# Restart NFS server
sudo systemctl restart nfs-server
```

### Pi Side (mounts via NFS)

```bash
sudo apt install nfs-common
sudo mkdir -p /mnt/gpu-share
sudo mount -t nfs 10.0.0.2:/mnt/shared /mnt/gpu-share

# For persistence, add to fstab:
echo '10.0.0.2:/mnt/shared  /mnt/gpu-share  nfs  defaults,noatime,nofail  0  0' | sudo tee -a /etc/fstab
```

---

## Current Machine Configurations

### GPU Rig (10.0.0.2)

| Component | Status | Notes |
|-----------|--------|-------|
| Interface | `enp0s31f6` | Only ethernet port |
| Netplan config | `/etc/netplan/99-pi-link.yaml` | Static IP 10.0.0.2/24 |
| SSH server | Installed, running | `ssh bryan@10.0.0.2` |
| NFS server | Installed | Export: `/mnt/shared` |
| NFS client | Installed | For Option A |
| fstab (Option A) | `10.0.0.1:/media/bryan/shared → /mnt/shared` | NFS from Pi |

**To switch fstab between options:**
```bash
# For Option A (drive on Pi):
sudo sed -i 's|^/dev/disk/by-uuid/be91c4c6.*|10.0.0.1:/media/bryan/shared  /mnt/shared  nfs  defaults,noatime,nofail  0  0|' /etc/fstab

# For Option B (drive on Rig):
sudo sed -i 's|^10.0.0.1:/media/bryan/shared.*|/dev/disk/by-uuid/be91c4c6-bb29-467d-85c2-d39f2358b156 /mnt/shared ext4 defaults,noatime 0 2|' /etc/fstab
```

### Pi (10.0.0.1)

| Component | Status | Notes |
|-----------|--------|-------|
| Interface | `eth0` | Static IP via NetworkManager profile `gpu-rig` |
| Drive mount | `/media/bryan/shared` | When drive is connected |
| NFS server | Needs setup | See Option A instructions |
| NFS client | Needs setup | For Option B |

---

## Quick Reference: Switching Setups

### Moving drive FROM Rig TO Pi:

On GPU Rig:
```bash
sudo umount /mnt/shared
# Physically disconnect drive, connect to Pi
```

On Pi:
```bash
# Drive should auto-mount or:
sudo mount /dev/sdX1 /media/bryan/shared
sudo systemctl restart nfs-kernel-server
```

On GPU Rig:
```bash
sudo mount /mnt/shared  # mounts via NFS from Pi
```

### Moving drive FROM Pi TO Rig:

On Pi:
```bash
sudo umount /media/bryan/shared
# Physically disconnect drive, connect to Rig
```

On GPU Rig:
```bash
sudo mount /dev/sdb1 /mnt/shared
sudo systemctl restart nfs-server
```

On Pi:
```bash
sudo mount /mnt/gpu-share  # mounts via NFS from Rig
```

---

## SSH Key Setup (TODO — on GPU rig)

The Pi's public key is at `/mnt/shared/pi_ssh_key.pub` (or wherever the shared drive is mounted).

Run on the GPU rig:
```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
cat /mnt/shared/pi_ssh_key.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

Then delete the key file from the shared drive (no need to keep it there):
```bash
rm /mnt/shared/pi_ssh_key.pub
```

Test from Pi side: `ssh bryan@10.0.0.2 hostname`

---

## Verification Checklist

- [ ] Ethernet connected between Pi and GPU rig
- [ ] `ping 10.0.0.1` works from GPU rig
- [ ] `ping 10.0.0.2` works from Pi
- [ ] Drive mounted on whichever machine has it
- [ ] NFS export active on machine with drive
- [ ] NFS mount works on other machine
- [ ] Can access `/agents/brain.py` from both machines

---

## Future: Network Switch Setup

Planning to add a network switch to allow:
- GPU rig to have both internet AND Pi link simultaneously
- Some devices isolated (air-gapped)
- SSH security to be revisited at that time
