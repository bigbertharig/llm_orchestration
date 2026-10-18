# Rig hardware layer

Rig-side code for the GPU rig. It carries out commands from a controller (llm-gateway's rig
manager on the Pi, `docs/MANAGER.md` there) and enforces the physical rules itself, close to the
hardware. It keeps working when the controller or the network is gone.

| File | Owns |
|---|---|
| `rig.py` (`/usr/local/bin/rig`) | command layer: modes, remote profiles, `observe`, `lease`, `claim`, `start`, `stop`, `why` (why a runtime is down: exit, OOM, guard action, earlyoom) |
| `hw_jobs.py` | detached jobs: a load runs as its own systemd unit (`rig-load-<id>`), never on an ssh session |
| `hw_state.py` | the GPU lease and claim record (rig-local `shared/brain/hw/gpus/`), mirrored to the drive |
| `hw_guard.py` | always-on power, thermal and memory guard (`rig-guard.service`) and load admission |
| `hw_plan.py` | the active controller and the plan queue (rig-local `brain/hw/plan.json`); `rig-plan.service` works through it |
| `remote_profiles.py` | reads the profiles llm-gateway offers (`~/llm-gateway/remote_profiles.json`) |
| `systemd/` | units: `rig-boot`, `rig-guard`, `rig-plan`, the stack modes; `earlyoom.default` (tracked `/etc/default/earlyoom`) |

`../llama_runtime/run_runtime.sh` is the one path every model load takes (brain, workers, split
workers, `rig.py`, campaigns, benchmark lanes, runtime smoke). It holds the rig rules for a load:

1. one model load at a time, rig-wide (flock), and never on a port that is already serving;
2. the guard must read ok (`hw_guard.py admit`), checked with the lock held;
3. host prompt cache off unless the caller asks (`--cache-ram`, floor 0);
4. every load is logged with its lease, or `no-lease`, in `brain/hw/lease_audit.log` (audit only).

## Controller and plan

`rig observe` is the single report a controller reads: what the rig is doing now (GPUs, containers,
profiles, leases, guard) and next (`plan`: the active controller and the queue). One controller is
active at a time (`rig controller claim --id ID`); it heartbeats, and the rig refuses plan, lease
and claim writes from any other controller. Another controller takes over when the heartbeat is
older than 120 s (crashed) or by handoff (`--force`). `rig-plan.service` works through the queued
items one at a time, in order (`load`, `unload`), then waits; loaded models stay loaded unless an
item unloads them, and it keeps going with no controller connected. `rig plan show` lists the queue.

## Monitoring

`rig-guard.service` samples every 2 s: CPU package temperature, per-GPU temperature, instant power
and utilization, available RAM and swap. The worst reading over 10 s sets `ok` / `warn` / `critical`
(limits in `hw_guard.py` DEFAULT_LIMITS, overrides in rig-local `brain/hw/guard.json`).

| Level | Loads | Running work |
|---|---|---|
| ok | admitted | untouched |
| warn | wait (up to `LLAMA_GUARD_WAIT_S`, 600 s) | untouched |
| critical | refused | memory critical: the llama runtime holding the most host RAM is stopped (once per 60 s). Power/thermal critical: untouched (open decision) |

State: `brain/hw/guard_state.json`; level changes and actions: `brain/hw/guard_events.log` (with a
process snapshot on critical). `rig status`, `rig observe`, and the controller's status show it.
earlyoom stays as the backstop, RAM-only (`systemd/earlyoom.default`).

Rules from the 2026-10-17 hang (run notes `logs/benchmarks/worker_lanes_20261017_203028/RUN_NOTES.md`
sections 10-11):
- never add swap to relieve memory pressure: it frees no RAM (zram fills first and lives in RAM) and
  it disarmed earlyoom, turning one runtime kill into a whole-rig hang;
- prevent the fill instead: host prompt cache off by default, the controller sets it per use, and a
  rig-wide RAM budget (open);
- if RAM must be freed, free it early; a stalled box cannot be reached over ssh.

Open: what power/thermal critical should do; `fail_closed` when the guard is down; VRAM and RAM fit
before placement; one stop path for every rig job; local results fallback when the drive is gone;
serialized EvalPlus scoring (run notes section 3).
