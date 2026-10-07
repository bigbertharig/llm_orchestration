# Rig Boot Modes + Published Model Profiles

Status: v1 IMPLEMENTED 2026-10-06 (work items 1, 2, and a first cut of 5). Verified: reboot -> idle,
all three modes start/stop, busy refusal, --force switch. Items 3-4 still open.

v1 as built:
- `shared/scripts/rig/rig.py` (rig: `/usr/local/bin/rig`), units in `shared/scripts/rig/systemd/`
  installed to `/etc/systemd/system/`. `rig-boot.service` enabled. Old `llm-orchestrator.service` removed.
- Remote = plain llama containers `rig-remote-<profile>` via run_runtime.sh. No orchestrator.
  Profiles: `chat-27b` (qwen3.6:27b, GPU 0+4, ctx 65536, :11434, `--reasoning off`) and
  `coder-7b` (qwen2.5-coder:7b, GPU 1, ctx 16384, :11435). Both are defaults. Load times: 2m17s cold, 29s warm.
- Watchdog no-ops unless mode is orchestration or bench.
- Exit codes: 0 ok, 1 failed (rig left idle), 2 busy, 3 another rig command running.
Related: `/home/bryan/Desktop/CLAUDE_COORDINATION.md`.

## Principle: keep it basic

The system was designed for a rig that's always busy: many modes, lots of queued jobs,
smart swapping. That may come back later. The real usage now: **load one thing, use it for a
few days, then the rig goes quiet.** Build for that. One mode at a time, explicit start/stop,
no automatic swapping or cross-mode coordination. Upgrade later if usage changes.

## Decisions (user, 2026-10-06)

1. **Idle = nothing.** Boot leaves no orchestrator processes and no LLM on any GPU.
   "Brain up, workers cold" is a mode (`orchestration`), not idle.
2. Mode coexistence: deferred. Until decided, **one mode at a time** (start = stop the current one first).
3. **Remote mode may use all GPUs**, including the 3090.
4. **JSON is the single source** for published model profiles:
   `plans/shoulders/benchmarking/model_task_library.json`. Human docs (MODEL_BEST_OF.md)
   follow it, not the other way round.

## Target behavior

### Boot -> idle
- Rig boots with `llm-orchestrator.service` **not** starting the stack.
- Only a preflight runs: NFS mounted, all 6 GPUs visible to `nvidia-smi` (catches Xid 79
  dropouts like today's), temps sane. It writes `control_policy`: `boot_profile: idle`.
- `gpu-worker-watchdog` does nothing while idle (today it would restart "missing" workers).

### Modes (exactly one at a time)

| Mode | What runs | Models allowed |
|---|---|---|
| `orchestration` | brain (with its model on the 3090) + cold workers. Equals today's neutral/plan boot. | published profiles |
| `bench` | brain + workers in benchmark config, `benchmarks` admitted | full catalog |
| `remote` | worker runtimes on request, `interactive` admitted, no brain model; may use **all GPUs** | published profiles |

### Mode selector: `rig` command on the rig, called over SSH

The rig runs headless. Idle = it sits and waits, and **sshd is the listener** (no daemon).
Any machine (desktop, Pi) drives it with `ssh rig rig <cmd>`.

```
rig status                      # mode, since, started-from, models+ports, GPU health
rig start <mode> [profile ...]  # orchestration | bench | remote
rig start <mode> --force        # stop current mode first, then start
rig stop                        # unload everything, stop the stack -> idle
```

Rules (basic, no auto-swap):
- Idle: `start` boots the mode and blocks until ready, printing `starting <mode>... done` + ports.
- Busy: `start` prints `busy: <mode> (since <time>, from <ip>)` and exits non-zero.
  `--force` stops the current mode, then starts the new one.
- `stop` always ends at idle and is safe to repeat.
- State: one file on the rig (`/mnt/shared/brain/rig_mode.json`: mode, since, from = `$SSH_CLIENT` ip,
  ports) under a lock, so two callers can't start modes at the same time.
- The stack runs under systemd (`llm-mode@<mode>.service` or similar), never as stray processes.
- Fails loud: if a GPU is missing or a load fails, `start` reports it and leaves the rig idle.

Lives on the rig (e.g. `/mnt/shared/scripts/rig`, symlinked into `~/bin`). The Pi and desktop keep
no logic, just `ssh rig rig ...`.

### Published model profiles
- `model_task_library.json` = named profiles -> `{model, fallback, placement, ctx}`. Runtime detail
  stays in `models.catalog.json` / `model_tuning_profiles.json`.
- Outside `bench`, loading a model that isn't in the published list fails fast (`model_not_published`).
- `agents/model_access.json` is retired. Remote callers ask for a profile by name.
- Needs an interactive profile with ctx >= 32k for Roo (first request ~19.4k tokens). Candidate:
  Qwen3.6-27B on 3090 (+1x1060 if needed) at 65k, the measured config in
  BENCHMARK_LESSONS_LEARNED 2026-08-03. Allowed now that remote may use the 3090.

## Work items

1. Idle boot: change `llm-orchestrator.service` to preflight-only (or disable it and add a
   `rig-preflight.service`). Add `idle` to `control_policy.py`. Watchdog no-ops when idle.
2. `rig` mode selector (on the rig, via SSH) wrapping existing pieces: startup.py configs, control_policy,
   `prepare_llm_runtimes.py`, kill/unload. Replaces start_*_mode scripts once it works.
3. Refresh `model_task_library.json` (stale 2026-04-28) from current BEST_OF results; add
   brain-tier + interactive 32k+ profiles.
4. Enforce published-only loads outside bench.
5. Remote: serve profiles by name (tunnel ports first; gateway later).

Future direction (user, 2026-10-06): the Pi as the single coordinator. The desktop tells the Pi
what it wants, and the Pi drives the rig (`ssh rig rig ...`), the Pi CPU cluster, and workers.
The `rig` selector stays the rig-side primitive either way. Planned hardware: NVMe on a PCIe x1 slot
(500 GB, model hotset), better CPU cooler; later 8x GTX 1070 + 3090. GPU count comes from
`agents/config.json`, so rig.py's GPU check follows config changes.

Deferred (upgrade later): mode coexistence, lease-split between modes, gateway/model_access_api
session leases, auto-default/return-to-default logic, scheduler-only brain.

## Side issues found 2026-10-06

- DONE: `run_runtime.sh` passes `-lv 4` (b10333 dropped the offload log line, so every worker
  load timed out). Commit f9b3fae on main (local, not pushed).
- `gpu_llama.py` readiness loop (~line 588) swallows `_assert_full_gpu_offload` exceptions, which
  turns a real error into a silent 300s timeout. It should fail fast.
- The orchestrator currently runs outside systemd (started by start_custom_mode at 16:59).
- 3x GTX 1060 dropped off the bus together at ~16:40 (Xid 79). Suspect power/riser.
