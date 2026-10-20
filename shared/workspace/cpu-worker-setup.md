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
- write `queue -> processing -> complete/failed`, finishing through
  `shared/agents/task_claims.py` (same as GPU workers): each claim gets a `claim_id`,
  the first result wins, and a worker that finishes after the brain requeued its task
  (for example after losing the shared drive) writes to `tasks/late/` instead of
  overwriting the result or deleting the new owner's processing file
- publish heartbeats under:
  - `shared/cpus/cpu-worker-<id>/heartbeat.json` (worker folder path)

Source of truth (2026-10-19): the repo file `shared/scripts/cpu_agent.py`, deployed to
`/media/bryan/shared/scripts/cpu_agent.py` with
`python3 scripts/sync_shared_source.py --live-root /media/bryan/shared` (compare) and
`--apply` (copy). `scripts/cpu_agent.py` in a repo checkout only forwards to it; it used
to be a separate, older copy.

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

Access standard (2026-10-20, same on every worker, like the Pi and the rig): log in as `bryan`,
use passwordless `sudo` for system changes. `root` keeps the control plane's key as a fallback only
(for example when sudo is broken); do not work as root day to day, it leaves root-owned files the
agent (`bryan`) cannot write.

- `bryan` - normal login, `restart_cpu_workers.sh`, the systemd service; passwordless sudo from
  `/etc/sudoers.d/90-bryan-nopasswd` (one line: `bryan ALL=(ALL) NOPASSWD:ALL`, mode 440)
- `root` - fallback only: `ssh root@10.0.0.<octet>`

Both have the control plane's public key in `~/.ssh/authorized_keys`.
The key is baked into the base image via the first-boot script.

Add the sudo rule to a worker that lacks it (from the control plane, once, needs a working sudo or
the root fallback):

```bash
ssh -t 10.0.0.<octet> "echo 'bryan ALL=(ALL) NOPASSWD:ALL' | sudo tee /etc/sudoers.d/90-bryan-nopasswd && sudo chmod 440 /etc/sudoers.d/90-bryan-nopasswd && sudo visudo -c"
```

SSH config on control plane (`~/.ssh/config`):

```
Host 10.0.0.10 10.0.0.11 10.0.0.12 10.0.0.13
    User bryan
    StrictHostKeyChecking accept-new
    UserKnownHostsFile ~/.ssh/known_hosts_workers
```

(14-17 have no entry and also log in as `bryan`, the default user.)

If host keys change (reimaging, first-boot re-run), clear the stale entries:

```bash
for i in 10 11 12 13 14 15 16 17; do
  ssh-keygen -f ~/.ssh/known_hosts_workers -R "10.0.0.$i"
done
```

---

## Fleet Standard (source of truth)

Since 2026-10-20 one folder defines a CPU worker: `shared/scripts/cpu_worker/` in this repo, live on the
share at `/media/bryan/shared/scripts/cpu_worker/` (the Pi checkout's `shared/` is the drive).

| File | Role |
|---|---|
| `prepare_worker.sh` | bring a worker to the standard; idempotent; run as bryan |
| `check_worker.sh` | 17 PASS/FAIL readiness checks; exit 1 on any failure |
| `requirements.txt` | the `/opt/worker-env` venv, pinned (`requests`, `ddgs` and dependencies: the third-party modules CPU batches have needed) |
| `cpu-agent.service` | the agent unit, the same file on every worker (`--name cpu-%H`) |
| `first-boot.sh`, `first-boot.service` | per-clone setup on a fresh card |
| `make_image_ready.sh` | prepare + check + generalize a worker, then power off for imaging |

`prepare_worker.sh` applies: the sudo rule, the Pi share fstab line, the venv exactly matching
`requirements.txt` (missing installed, extras removed), the unit and first-boot files, and cleanup
(apt cache, `~/.cache`, stray venvs, rotated logs). It turns off what a fleet worker does not use: the
dead `ensure-shared-mount.timer`, `wpa_supplicant` (wired only), `man-db.timer`, and the `apt-daily`
timers, so workers change only with the image or this script, not by themselves.

Fleet:

```bash
# from the Pi
for i in 10 11 12 14 15 16 17; do ssh 10.0.0.$i bash /media/bryan/shared/scripts/cpu_worker/prepare_worker.sh; done
for i in 10 11 12 14 15 16 17; do ssh 10.0.0.$i bash /media/bryan/shared/scripts/cpu_worker/check_worker.sh | grep -E "FAIL|failed"; done
```

State 2026-10-20: workers 10-12 and 14-17 pass all checks (1.2-1.3 GB used each); an end-to-end test
(seven CPU tasks queued with the brain stopped) completed on workers 11, 12, 14, 16 and the Pi's own
agent with claim ids and durations recorded. Worker 13 (its SD card had been loose) was brought to the
standard in place with provisioning steps 5-7 and passes all checks. Worker 10 stopped answering
(no ARP reply) later the same day and needs a physical check.

## Base Image

Image path:

- `/media/bryan/shared/plans/shoulders/research_assistant/docs/worker-image.img.xz`

Last updated: 2026-10-20 (from worker 17 after `make_image_ready.sh`, 0 failed checks). Raw size
7,910,457,344 bytes (fits any card of 8 GB or more), sha256 of the uncompressed image
`dbc34240e9228a587882cf945bd928de76beb7a926d4708d60bb519e92983229`. The previous image is kept as
`worker-image-20260323.img.xz`.

Making a new image (needs the SD card in a Linux card reader, e.g. on the Pi):

1. Pick the worker with the smallest card (worker 17, 8 GB; 14 and 16 have 16 GB, the rest 32 GB). An
   image only fits cards at least its size; first-boot grows the filesystem on larger ones.
2. `ssh 10.0.0.17 bash /media/bryan/shared/scripts/cpu_worker/make_image_ready.sh` - prepares, checks
   (stops on any FAIL), removes the first-boot flag, blanks the machine id, empties logs and history,
   then powers off.
3. Move the card to the reader and find it with `lsblk`. On the Pi the shared drive is `/dev/sda`, so
   the card is another device (for example `/dev/sdb`); check the size before reading.
4. Keep the old image and write the new one:
   ```bash
   cd /media/bryan/shared/plans/shoulders/research_assistant/docs
   mv worker-image.img.xz worker-image-20260323.img.xz
   sudo dd if=/dev/sdX bs=4M status=progress | xz -T0 -6 > worker-image.img.xz
   ```
5. Put the card back and boot the worker: first-boot runs again (same IP and name, new SSH host key).
   Clear the old key on the Pi (`ssh-keygen -f ~/.ssh/known_hosts_workers -R 10.0.0.17`), then run
   `check_worker.sh` on it.
6. Update "Last updated" above and the image contents below.

Capturing on Windows instead (done 2026-10-20; the Pi had no card reader attached): read the card as
an administrator from `\\.\PhysicalDriveN` (check the number and size with `Get-Disk`) into a raw
`.img`, then stream it to the Pi and compress there:
`cat worker.img | ssh pi 'xz -T0 -6 > .../worker-image-new.img.xz.partial'`. Notes from that run:
plain Python `read()` on the raw disk fails with EINVAL (use `readinto` into a page-aligned `mmap`
buffer at explicit offsets); one USB reader hung for minutes per read and had to be swapped; Windows
reports the card a little smaller than its real size, which is fine as long as the partition ends at
or before the reported size. Before swapping the new image in, check it on the Pi: `xzcat | sha256sum`
against the raw image's hash, then a read-only loop mount (`losetup -fP -r`, `e2fsck -fn`) to confirm
the first-boot flag and machine id are gone.

Burning:

```bash
xzcat /media/bryan/shared/plans/shoulders/research_assistant/docs/worker-image.img.xz \
  | sudo dd of=/dev/sdX bs=4M status=progress conv=fsync
```

Image contents:

- Armbian / Debian minimal (terminal-only), one ext4 partition (`armbi_root`), 1.2 GB used
- Fleet standard from `prepare_worker.sh`: the Pi share fstab line (`10.0.0.2:/media/bryan/shared`),
  `/etc/sudoers.d/90-bryan-nopasswd`, `/opt/worker-env` matching `requirements.txt`, unattended
  updates and the other unused timers off
- First-boot script at `/opt/first-boot.sh`, `first-boot.service` and `cpu-agent.service` enabled
- Generalized: no `/opt/.first-boot-done`, empty `/etc/machine-id`, empty logs and shell history
- Control plane SSH public key for both `root` and `bryan`

---

## Shared Mount And Runtime Paths

Expected shared mount on workers:

- `/media/bryan/shared`

Current required fstab line (normalized across workers 10-17). Since 2026-10-15 the share is served by
the Pi (`10.0.0.2`), not the rig; workers still on the old `10.0.0.3:/mnt/shared` line need this change:

```fstab
10.0.0.2:/media/bryan/shared /media/bryan/shared nfs nofail,_netdev,noauto,x-systemd.automount,x-systemd.mount-timeout=10s,nolock 0 0
```

Expected runtime paths:

- CPU agent:
  - `/media/bryan/shared/scripts/cpu_agent.py`
- CPU worker logs: the journal (`journalctl -u cpu-agent.service`); there is no
  `/var/log/cpu-agent/` file
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

The service (source: `shared/scripts/cpu_worker/cpu-agent.service`, the same file on every worker):
- Starts after first boot, the network and remote filesystems
- Waits for the agent script on the share (ExecStartPre loop, no start timeout). It no longer uses
  `RequiresMountsFor`: a mount that failed at boot used to fail the agent for good
- Runs as user `bryan`, named `cpu-<hostname>` (`--name cpu-%H`)
- Restarts always (10 s delay)
- Logs to the journal (`journalctl -u cpu-agent.service`)

Manual control:

```bash
# Check status
ssh 10.0.0.10 'systemctl status cpu-agent.service'

# Restart
ssh 10.0.0.10 'sudo systemctl restart cpu-agent.service'

# Stop
ssh 10.0.0.10 'sudo systemctl stop cpu-agent.service'
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

- CPU workers log to the journal (Armbian keeps logs in RAM via armbian-ramlog and syncs them to
  `/var/log.hdd`; they do not survive reimaging).
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
4. Verify: `ssh 10.0.0.<octet> 'hostname'` returns `worker-<octet>`
5. Old (2026-03-23) image only, not needed with the 2026-10-20 image: point fstab at the Pi first, because `prepare_worker.sh` lives on the
   share (use `ssh root@...` if bryan's sudo asks for a password):
   `ssh 10.0.0.<octet> "sudo sed -i 's#^10\.0\.0\.3:/mnt/shared /media/bryan/shared #10.0.0.2:/media/bryan/shared /media/bryan/shared #' /etc/fstab && sudo systemctl daemon-reload && sudo systemctl restart media-bryan-shared.automount"`
6. `ssh 10.0.0.<octet> bash /media/bryan/shared/scripts/cpu_worker/prepare_worker.sh`
7. `ssh 10.0.0.<octet> bash /media/bryan/shared/scripts/cpu_worker/check_worker.sh` shows 0 failed

Worker 13 was brought up this way on 2026-10-20 (steps 5-7 on its existing card).

---

## Dashboard Visibility

The dashboard drops CPU workers whose heartbeats are older than 10 minutes
(`HEARTBEAT_MAX_S = 600`). If workers disappear from the dashboard:

1. Check if the Pi is reachable: `ping 10.0.0.<octet>`
2. Check agent status: `ssh 10.0.0.<octet> 'systemctl status cpu-agent.service'`
3. Check NFS mount: `ssh 10.0.0.<octet> 'mount | grep shared'` (should be `10.0.0.2:/media/bryan/shared`)
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
ssh 10.0.0.<octet> 'journalctl -u cpu-agent.service -u media-bryan-shared.mount -n 30 --no-pager'
```

Common causes:
- NFS mount not available (check fstab points at the Pi `10.0.0.2:/media/bryan/shared`, not the old
  `10.0.0.3:/mnt/shared`; check the Pi NFS server). Seen 2026-10-20: every worker still on the rig
  line, the mount timed out at boot and `cpu-agent.service` failed with result 'dependency'.
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
