# CPU Worker Setup

Reference for the Orange Pi CPU-worker cluster and the current CPU-worker
operating model.

Use this doc for CPU-worker provisioning and deployment details. For normal rig
operation, use [quickstart.md](quickstart.md) and [CONTEXT.md](CONTEXT.md).

---

## Current Status

- CPU workers are an auxiliary execution pool, not the primary orchestration path.
- They are intended to claim `task_class: cpu` work only.
- They use the same shared task lanes and heartbeat patterns as the rest of the
  orchestration system.
- CPU agents auto-start on boot via systemd (`cpu-agent.service`).

Current shared scripts:

- CPU agent:
  - `/media/bryan/shared/scripts/cpu_agent.py`
- Restart helper (manual override):
  - `/media/bryan/shared/scripts/restart_cpu_workers.sh`

Normal behavior:

- claim only `executor: worker` + `task_class: cpu`
- write `queue -> processing -> complete/failed`
- publish heartbeats under:
  - `shared/cpus/cpu-worker-<id>/heartbeat.json` (worker folder path)

This is separate from the local repo copy at
`/home/bryan/llm_orchestration/scripts/cpu_agent.py`, which is useful for local
development and diagnostics. The shared-script path is the cluster runtime path.

---

## Hardware

- 8x Orange Pi Prime
- Allwinner H5, 4 cores, 2 GB RAM
- microSD boot
- gigabit ethernet
- heatsinks strongly recommended
- All 8 share a single power supply and network switch (power-cycled together)

These machines are for low-memory CPU work, not heavyweight local model
inference.

---

## Network

- subnet: `10.0.0.0/24`
- control plane: `10.0.0.2` (laptop/operator)
- GPU rig: `10.0.0.3`
- CPU workers: `10.0.0.10` through `10.0.0.17`

**DHCP reservations are set on the rig router (10.0.0.1).** Each Pi's MAC address
is mapped to a fixed IP. This is required — without reservations, IPs shuffle on
reboot and hostnames won't match.

| MAC | Reserved IP | Hostname |
|---|---|---|
| `02:01:70:ec:f9:ba` | 10.0.0.10 | worker-10 |
| `02:01:2f:19:e6:17` | 10.0.0.11 | worker-11 |
| `02:01:3d:03:b7:54` | 10.0.0.12 | worker-12 |
| `02:01:fd:c3:2a:35` | 10.0.0.13 | worker-13 |
| `02:01:26:29:35:6a` | 10.0.0.14 | worker-14 |
| `02:01:f9:fe:a5:b9` | 10.0.0.15 | worker-15 |
| `02:01:dc:02:f8:a7` | 10.0.0.16 | worker-16 |
| `02:01:58:5f:42:44` | 10.0.0.17 | worker-17 |

The workers mount shared storage from the GPU rig (`10.0.0.3`) and execute
directly against shared scripts, queues, and outputs.

---

## SSH Access

Two users have SSH key auth configured:

- `root` — used for system administration
- `bryan` — used by `restart_cpu_workers.sh` and the systemd service

Both have the control plane's public key in `~/.ssh/authorized_keys`.
The key is baked into the base image via the first-boot script.

SSH config on control plane (`~/.ssh/config`):

```
Host 10.0.0.10 10.0.0.11 10.0.0.12 10.0.0.13 10.0.0.14 10.0.0.15 10.0.0.16 10.0.0.17
    User root
    StrictHostKeyChecking accept-new
    UserKnownHostsFile ~/.ssh/known_hosts_workers
```

If host keys change (reimaging, first-boot re-run), clear the stale entries:

```bash
for i in 10 11 12 13 14 15 16 17; do
  ssh-keygen -f ~/.ssh/known_hosts_workers -R "10.0.0.$i"
done
```

---

## Base Image

Image path:

- `/media/bryan/shared/plans/shoulders/research_assistant/docs/worker-image.img.xz`

Last updated: 2026-03-23

Burning:

```bash
xzcat /media/bryan/shared/plans/shoulders/research_assistant/docs/worker-image.img.xz \
  | sudo dd of=/dev/sdX bs=4M status=progress conv=fsync
```

Image contents:

- Armbian / Debian minimal (terminal-only)
- NFS mount to GPU rig in fstab
- Python 3 available
- First-boot script at `/opt/first-boot.sh`
- systemd service `cpu-agent.service` enabled
- Control plane SSH public key for both `root` and `bryan`

---

## Shared Mount And Runtime Paths

Expected shared mount on workers:

- `/media/bryan/shared`

Current required fstab line (normalized across workers 10-17):

```fstab
10.0.0.3:/mnt/shared /media/bryan/shared nfs nofail,_netdev,noauto,x-systemd.automount,x-systemd.mount-timeout=10s,nolock 0 0
```

Expected runtime paths:

- CPU agent:
  - `/media/bryan/shared/scripts/cpu_agent.py`
- CPU worker logs (local to Pi):
  - `/var/log/cpu-agent/cpu-worker-<id>.log`
- config:
  - `/media/bryan/shared/agents/config.json`

The shared-script path is intentional. It keeps one runtime copy for all CPU
workers.

---

## First Boot

Script location: `/opt/first-boot.sh`

Runs once on first boot (flag: `/opt/.first-boot-done`). Steps:

1. Expands filesystem to fill SD card
2. Waits for network (up to 60s retry loop for DHCP)
3. Derives hostname from IP last octet (`worker-{octet}`)
4. Sets hostname
5. Regenerates SSH host keys
6. Provisions control plane SSH key for `root` and `bryan`
7. Marks first-boot complete

If network isn't available within 60s, the script aborts and will retry on
next boot (flag not set).

Expected hostname shape:

- `worker-10` through `worker-17`

---

## Auto-Start (systemd)

CPU agents start automatically on boot via:

```
/etc/systemd/system/cpu-agent.service
```

The service:
- Waits for NFS mount (`RequiresMountsFor=/media/bryan/shared`)
- Waits for the agent script to be accessible (ExecStartPre loop)
- Runs as user `bryan`
- Restarts on failure (10s delay)
- Logs to `/var/log/cpu-agent/cpu-worker-<id>.log`

Manual control:

```bash
# Check status
ssh root@10.0.0.10 'systemctl status cpu-agent.service'

# Restart
ssh root@10.0.0.10 'systemctl restart cpu-agent.service'

# Stop
ssh root@10.0.0.10 'systemctl stop cpu-agent.service'
```

---

## Starting CPU Workers (Manual)

The systemd service handles auto-start. These are fallback methods:

Restart all default workers via SSH helper:

```bash
/media/bryan/shared/scripts/restart_cpu_workers.sh
```

Restart specific workers:

```bash
/media/bryan/shared/scripts/restart_cpu_workers.sh 10.0.0.10 10.0.0.11
```

Single-run smoke test:

```bash
python3 /media/bryan/shared/scripts/cpu_agent.py --once --name cpu-worker-10
```

---

## Runtime Notes

- CPU workers log locally to `/var/log/cpu-agent/` (zram-backed on Armbian,
  survives reboot but not reimaging).
- CPU workers are expected to stay simple:
  - no GPU ownership
  - no LLM runtime ownership
  - no shared coordination authority
- They behave like lightweight task executors that report status upward.

---

## Provisioning Checklist

1. Ensure DHCP reservation exists on router for the Pi's MAC address.
2. Burn the base image to microSD.
3. Boot the worker and wait ~60s for first-boot to complete.
4. Verify: `ssh root@10.0.0.<octet> 'hostname'` → `worker-<octet>`
5. Verify: NFS mount active (`mount | grep shared`)
6. Verify: agent running (`systemctl is-active cpu-agent.service`)
7. Verify: heartbeat updating (`shared/cpus/cpu-worker-<id>/heartbeat.json`)

---

## Dashboard Visibility

The dashboard drops CPU workers whose heartbeats are older than 10 minutes
(`HEARTBEAT_MAX_S = 600`). If workers disappear from the dashboard:

1. Check if the Pi is reachable: `ping 10.0.0.<octet>`
2. Check agent status: `ssh root@10.0.0.<octet> 'systemctl status cpu-agent.service'`
3. Check NFS mount: `ssh root@10.0.0.<octet> 'mount | grep shared'`
4. Check heartbeat file timestamp on shared drive

---

## Troubleshooting

### Workers disappear from dashboard after power cycle

**Cause:** DHCP reservations not set, IPs shuffled, hostnames don't match.

**Fix:** Set DHCP reservations on router (see Network section above). Clear
first-boot flag on all Pis (`rm /opt/.first-boot-done`), reboot fleet.

### SSH "REMOTE HOST IDENTIFICATION HAS CHANGED"

**Cause:** Pi was reimaged or first-boot regenerated host keys.

**Fix:** Clear stale keys from `~/.ssh/known_hosts_workers`:
```bash
ssh-keygen -f ~/.ssh/known_hosts_workers -R 10.0.0.<octet>
```

### SSH "Permission denied (publickey,password)"

**Cause:** Control plane SSH key changed (regenerated on control plane) but
base image still has old public key.

**Fix:** Push current key to the Pi (requires password or physical access):
```bash
sshpass -p '<password>' ssh root@10.0.0.<octet> \
  "echo '$(cat ~/.ssh/id_ed25519.pub)' >> /root/.ssh/authorized_keys"
```
Then update the base image to include the new key.

### Agent fails to start (systemd)

Check journal:
```bash
ssh root@10.0.0.<octet> 'journalctl -u cpu-agent.service -n 30 --no-pager'
```

Common causes:
- NFS mount not available (check fstab, check GPU rig NFS server)
- Permission error writing heartbeat (check file ownership on shared drive)
- Python not available

### Heartbeat PermissionError

**Cause:** Stale root-owned heartbeat files from older runs.

**Fix:** On NFS server (`10.0.0.3`), reset owner/mode:
```bash
chown bryan:bryan /mnt/shared/cpus/cpu-worker-<id>/heartbeat.json
chmod 664 /mnt/shared/cpus/cpu-worker-<id>/heartbeat.json
```

---

## Open Questions

- [ ] Do CPU workers stay Orange Pi specific, or should this doc become a
  generic CPU-worker contract?
- [ ] Should the shared CPU agent and repo-local CPU agent be unified into one
  authoritative path?
- [ ] Do we want a dedicated `config.cpu_workers.json` instead of reusing the
  general config?
- [ ] Re-capture base image periodically or only on major changes?

---

## Shopping / Spare Parts

- [ ] microSD cards (spares)
- [ ] heatsinks
- [ ] reliable multi-port power supplies

---

*Last Updated: 2026-03-23*
