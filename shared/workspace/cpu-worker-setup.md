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

Current shared scripts:

- CPU agent:
  - `/media/bryan/shared/scripts/cpu_agent.py`
- Restart helper:
  - `/media/bryan/shared/scripts/restart_cpu_workers.sh`

Normal behavior:

- claim only `executor: worker` + `task_class: cpu`
- write `queue -> processing -> complete/failed`
- publish heartbeats under:
  - `shared/heartbeats/cpu-worker-<id>.json` (dashboard path)
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

These machines are for low-memory CPU work, not heavyweight local model
inference.

---

## Network Assumptions

- subnet: `10.0.0.0/24`
- control plane: `10.0.0.2` (laptop/operator)
- GPU rig: `10.0.0.3`
- CPU workers: `10.0.0.10` through `10.0.0.17`

The workers mount shared storage from the GPU rig (`10.0.0.3`) and execute
directly against shared scripts, queues, and outputs.

---

## Base Image

Image path:

- `/media/bryan/shared/plans/shoulders/research_assistant/docs/worker-image.img.xz`

Burning:

```bash
xzcat /media/bryan/shared/plans/shoulders/research_assistant/docs/worker-image.img.xz \
  | sudo dd of=/dev/sdX bs=4M status=progress conv=fsync
```

Image assumptions:

- Armbian / Debian minimal image
- terminal-only
- shared-drive mount on boot
- local Python available
- first-boot script regenerates identity-specific state

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
- CPU worker logs:
  - `/media/bryan/shared/logs/cpu_workers/`
- config:
  - `/media/bryan/shared/agents/config.json`

The shared-script path is intentional. It keeps one runtime copy for all CPU
workers.

---

## First Boot

The image is expected to run a one-time first-boot script that:

1. sets hostname from the assigned IP suffix
2. regenerates SSH host keys
3. marks first-boot completion
4. leaves the worker ready for shared-drive execution

Expected hostname shape:

- `worker-10`
- `worker-11`
- and so on

---

## Starting CPU Workers

Restart all default workers:

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

- CPU workers should write logs to shared persistent storage, not `/tmp`.
- The helper script currently launches the shared CPU agent over SSH and keeps
  logs in `/media/bryan/shared/logs/cpu_workers/`.
- The helper script starts workers with:
  - `python3 /media/bryan/shared/scripts/cpu_agent.py --config /media/bryan/shared/agents/config.json --name cpu-worker-<id>`
- CPU workers are expected to stay simple:
  - no GPU ownership
  - no LLM runtime ownership
  - no shared coordination authority

They should behave like lightweight task executors that report status upward.

---

## Provisioning Checklist

1. Burn the base image to microSD.
2. Boot the worker and let first-boot finish.
3. Confirm hostname and shared mount.
4. Confirm `python3` is available.
5. Start the CPU agent with the shared restart helper.
6. Confirm heartbeats update under both:
   - `shared/heartbeats/cpu-worker-<id>.json`
   - `shared/cpus/cpu-worker-<id>/heartbeat.json`

---

## Known Failure Mode (Fixed)

Symptom:
- worker process starts, then exits with `PermissionError` writing heartbeat.

Cause:
- stale root-owned heartbeat files from older runs.

Fix:
- on NFS server (`10.0.0.3`), reset owner/mode for worker heartbeat files:
  - `/mnt/shared/heartbeats/cpu-worker-<id>.json`
  - `/mnt/shared/cpus/cpu-worker-<id>/heartbeat.json`
  - owner/group `bryan:bryan`, file mode `664`, worker dir mode `775`

---

## Open Questions

- [ ] Do CPU workers stay Orange Pi specific, or should this doc become a
  generic CPU-worker contract?
- [ ] Should the shared CPU agent and repo-local CPU agent be unified into one
  authoritative path?
- [ ] Do we want a dedicated `config.cpu_workers.json` instead of reusing the
  general config?

---

## Shopping / Spare Parts

- [ ] microSD cards
- [ ] heatsinks
- [ ] reliable multi-port power supplies
