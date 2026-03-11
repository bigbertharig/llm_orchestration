# GPU Rig Audit Report

Date: 2026-02-11 (PST) / 2026-02-12 (UTC)
Host: `bryan-GPU-Rig` (`10.0.0.2`)
Auditor: Codex via SSH

## Scope

- System and OS baseline
- GPU topology, PCIe links, thermals, power state
- Network and shared storage mount
- Ollama and orchestration readiness
- Risk and optimization recommendations

## Executive Summary

The rig is reachable and generally healthy, but not yet optimized for sustained orchestration workloads. The most important constraints are:

1. PCIe bandwidth is very limited on the worker GPUs (`GTX 1060` cards are `x1 Gen1`).
2. Runtime services need to align with task-driven worker model loading (brain fixed, workers optional).
3. CPU policy and kernel tunables are desktop defaults, not compute-node tuned.
4. Reliability and security posture can be improved (NFS mount race at boot; temporary Wi-Fi still installed).

## Current State

### Hardware / OS

- Board: ASUS `B250 MINING EXPERT`, BIOS `1001` (2017-12-13)
- CPU: `i7-7700K` (4C/8T), RAM: `30 GiB`, swap: `8 GiB`
- OS: Ubuntu `25.10`, kernel `6.17.0-14-generic`
- Root disk usage: ~19% used

### GPU Inventory

- GPU0: `RTX 3090 Ti` 24GB
- GPU1-4: `GTX 1060 6GB` (x4)
- Driver: `580.126.09`, CUDA: `13.0`
- Idle thermals are good (27C-45C observed)

### PCIe Topology and Link State

- GPU0 (`3090 Ti`): `x16 Gen1` (`2.5 GT/s`)
- GPU1-4 (`1060`): `x1 Gen1` downgraded
- This is expected for a mining-board style layout but constrains host<->GPU transfer throughput.

### Network / Shared Storage

- NFS mount active: `10.0.0.1:/media/bryan/shared -> /mnt/shared`
- Ethernet (`enp0s31f6`) at `1000Mb/s full duplex`
- Wi-Fi (`wlp2s0`) connected, and default route currently goes via Wi-Fi
- Boot logs show at least one transient mount failure for `/mnt/shared` during startup

### Ollama / Orchestration Runtime

- Active services:
  - `ollama.service` (GPU0, port `11434`)
  - `ollama-1060.service` (GPU1, port `11435`)
  - `nvidia-persistenced.service`
- No orchestrator processes running: `brain.py`, `gpu.py`, `worker.py` not active
- `config.json` exists at `/mnt/shared/agents/config.json`:
  - brain on GPU0 (`qwen2.5:14b`)
  - workers on GPU1-4 (`qwen2.5:7b`, ports `11435-11438`)
  - `worker_mode`: `hot`
- Models directory under `/usr/share/ollama/.ollama/models` appears effectively empty right now

## Confirmed Design Decisions

1. PCIe layout is intentional
- `3090 Ti` on x16 slot is the correct placement.
- `1060` cards on `x1 Gen1` are accepted as an immutable hardware constraint.

2. Unified worker LLM lifecycle via task queue (no hard-spawned worker models)
- Brain model remains the only fixed baseline model.
- Worker GPUs are optional and start cold.
- On launch, enqueue one immediate `meta` task (`load_llm`) so any available GPU can claim it.
- Future load/unload stays task-driven (`load_llm` / `unload_llm`) rather than hardcoding a specific worker/GPU.

3. Model storage tiering
- Shared drive is long-term model storage (can hold many models, e.g. 10+).
- Rig SSD holds only the active working set (e.g. 3-4 models) for faster load/use time.
- Promotion/demotion between shared and SSD should be an explicit operational step.

4. Clean-slate image philosophy
- Backward compatibility and transitional stability are not priorities for this initial image.
- Prefer minimal and deterministic runtime over preserving old service patterns.
- If something breaks during first-run setup, fix-forward is acceptable.

## Findings and Impact

### High Priority

1. Service model mismatch with orchestrator ownership
- Evidence: systemd has always-on Ollama services on `11434` and `11435`, while `gpu.py` is designed to start/stop per-GPU Ollama instances.
- Impact: port conflicts or hidden state coupling are likely once agents start, especially on GPU1 (`11435`).

2. Worker bandwidth bottleneck is hard-limited by PCIe x1 Gen1
- Evidence: `LnkSta` for GPUs 1-4 is `Speed 2.5GT/s, Width x1 (downgraded)`.
- Impact: slower model loads, slower host-device transfer for script workloads, and lower throughput ceiling.

3. Rig not in orchestrator-running state
- Evidence: no `brain.py`/`gpu.py` processes; queues/logs are empty.
- Impact: appears "ready" but does not execute plans until startup is explicitly run.

### Medium Priority

4. CPU governor is `powersave` on all cores
- Impact: extra latency and less consistent CPU-side preprocessing/queue management.

5. Persistence mode is disabled on all GPUs
- Evidence: `nvidia-smi --query-gpu=index,persistence_mode` reports `Disabled`.
- Impact: additional model init overhead and less stable long-run behavior under repeated load/unload cycles.

6. Boot-order reliability issue on shared mount
- Evidence: journal contains failed mount event for `/mnt/shared` during boot.
- Impact: startup race can break orchestration if services launch before NFS is available.

7. Air-gap posture is weakened by active Wi-Fi default route
- Impact: unnecessary exposure/routing ambiguity and non-deterministic network pathing.

### Low Priority

8. Time sync not yet marked synchronized
- `timedatectl`: NTP service active but `System clock synchronized: no`.
- Impact: mostly affects log correlation and forensic clarity.

## Optimization Plan (Prioritized)

1. Unify process ownership for Ollama vs agents
- Adopt a single owner model:
  - Brain service may remain always-on.
  - Worker LLM lifecycle is owned by orchestration tasks only (`load_llm` / `unload_llm`).
- At launch, enqueue one immediate `load_llm` meta task to warm exactly one worker, whichever claims it.
- Remove/disable worker-specific always-on Ollama units now (clean-slate image baseline).
- Ensure `load_llm` is idempotent (already-loaded worker returns success/no-op).

2. Make orchestrator startup deterministic at boot
- Add a dedicated systemd service for orchestration startup (brain + GPU agents) with:
  - `After=network-online.target mnt-shared.mount ollama.service`
  - `Requires=mnt-shared.mount`
  - health/restart policy and log routing.
- Add startup preflight for worker ports:
  - if a worker port is occupied by a non-agent process, terminate it and take ownership (no compatibility mode).

3. Tune compute host defaults
- Switch CPU governor to `performance` for this node class.
- Lower `vm.swappiness` (e.g. 10-20) for inference workloads.
- Keep THP at `madvise` unless measured otherwise (currently reasonable).

4. Enable GPU persistence mode at boot
- Ensure persistence is actually on (`nvidia-smi -pm 1` per GPU during boot).
- Verify again with `nvidia-smi --query-gpu=index,persistence_mode`.

5. Harden network intent
- Temporary state: Wi-Fi card is installed now but planned for removal.
- After removal, keep only direct ethernet path (`10.0.0.0/24`) for control-plane file exchange.

6. Fix NFS boot race
- Use explicit mount dependencies and network-wait strategy.
- Consider `x-systemd.automount`, `x-systemd.requires=network-online.target`, and mount retry semantics.

7. Validate/refresh model placement strategy
- Implement two-tier model policy:
  - Shared drive: full model archive.
  - SSD: limited hot set (target 3-4 active models).
- User manually chooses favorites and moves them to SSD; this set can evolve over time.
- Preload SSD-resident models and verify workers can serve them when loaded.

## Gold Image Deployment Goal

Target workflow for reproducible rig bring-up:

1. Build and capture a clean GPU rig image.
2. Boot a new rig from this image.
3. Connect the shared drive.
4. Move chosen favorite models from shared storage to local SSD model path.
5. Run setup/startup scripts from the shared drive.
6. Rig becomes ready for orchestration without manual per-rig service surgery.

Design implications:
- Keep the image minimal and deterministic.
- Keep model archive on shared drive, hot set on SSD.
- Keep worker runtime ownership inside orchestration task flow.

## Recommended Verification Checklist After Changes

1. `systemctl status mnt-shared.mount` shows stable active mount on boot.
2. `ss -ltnp | grep 1143` matches the intended service ownership model.
3. `pgrep -af "brain.py|gpu.py"` shows expected agent count.
4. `nvidia-smi --query-gpu=index,persistence_mode` reports `Enabled` for all GPUs.
5. `cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor` reports `performance`.
6. `python /mnt/shared/agents/startup.py` launches cleanly with no port/mount errors.
7. Submit a small plan and confirm queue -> processing -> complete flow.

## Notes

- PCIe x1 Gen1 on worker cards is a physical platform limitation; software tuning cannot remove it.
- For best LLM throughput, reserve GPU0 (`3090 Ti`) for brain and keep 1060 workers focused on smaller, parallelizable tasks.
