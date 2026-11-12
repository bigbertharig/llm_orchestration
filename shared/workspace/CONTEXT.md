# LLM Orchestration — Context (Short)

Addresses: these docs use role placeholders (`<pi-ip>`, `<rig-ip>`, `<router-ip>`, `<worker-ip>`, `<desktop-ip>`, `<rig-subnet>`) instead of our real ones. The real addresses, the MAC reservations and the ssh config are in the local site inventory (`shared/workspace/local/inventory.md`, not committed). Scripts carry the real values.

Purpose: Brain orchestrates tasks; GPU/CPU workers execute; plans define work.

## Read First
- `workspace/quickstart.md` (submit, monitor, kill plan)
- `workspace/PLAN_FORMAT.md` (task schema)
- `workspace/brain-behavior.md` (brain loop details)
- `core/RULES.md` (immutable safety constraints)
- `workspace/gpu_lane_reservations.md` (canonical GPU leases, admission policy, and lane display)

## Active Runtime Docs
- `workspace/implement/system_unification_audit_2026-08-13.md` (latest whole-system audit and prioritized integration gaps)
- `workspace/implement/runtime_unification_cleanup.md` (current cleanup track and next steps)
- `workspace/implement/modular_runtime_target_system.md` (target modular system and code areas to change)
- `shared/scripts/llama_runtime/README.md` (runtime image/build/run helpers)
- `shared/plans/shoulders/benchmarking/README.md` (benchmark runtime usage and current runtime compatibility notes)

## Benchmarking Docs
- `shared/plans/shoulders/benchmarking/MODEL_LIBRARY.md` (model selection hub — best choices, inventory, links)
- `shared/plans/shoulders/benchmarking/MODEL_SELECTION_FOR_PLANS.md` (plan-facing model choices by task type)
- `shared/plans/shoulders/benchmarking/BENCHMARK_SCORES.md` (pure score tables — pipeline, code, reasoning, knowledge)
- `shared/plans/shoulders/benchmarking/MODEL_RUNTIME_GUIDE.md` (per-model runtime requirements, flags, memory, compatibility)
- `shared/plans/shoulders/benchmarking/BENCHMARK_LESSONS_LEARNED.md` (tuning histories, debugging narratives, fix details)
- `shared/plans/shoulders/benchmarking/docker/README.md` (Docker suite operator guide)
- `shared/plans/shoulders/benchmarking/docker/SEQUENCED_TEST_SUITES.md` (campaign sequencer and thinking-model flags)

## Key Paths
- `shared/agents/` code + config
- `shared/plans/` shoulders/arms plan repos
- `shared/tasks/{queue,processing,complete,failed}/` runtime lanes
- `shared/brain/` brain state + private tasks
- `shared/gpus/`, `shared/cpus/` worker heartbeats
- `shared/logs/brain_decisions.log` primary decision log
- `shared/plans/<plan>/history/<batch_id>/` per-run artifacts
- `shared/plans/<plan>/history/_summary/` cross-run rollups
- `shared/scripts/summarize_history_run.py` summarize one historical batch
- `~/llm_orchestration/scripts/rollup_history.py` roll up an entire history tree
- `shared/scripts/batch_runtime_diag.py` inspect runtime state for a batch
- Meta tasks (`load_llm`, `unload_llm`): brain inserts via `brain_resources.py:_insert_resource_task()`, workers execute via `gpu_llama.py:load_model()`
- Model knowledge originates in the benchmarking repo. Run
  `python3 shared/scripts/sync_model_library.py what-new` to inspect changes and
  `python3 shared/scripts/sync_model_library.py sync` to import them. Orchestration
  runs from its own `shared/agents/model_library.json`,
  `shared/agents/models.catalog.json`, and `shared/agents/model_access.json`;
  benchmarking is not read during ordinary runtime. See
  `shared/agents/MODEL_LIBRARY_SYNC.md`.

## Shared Storage Topology (Current)

- The shared drive (4TB WD, later a NAS) is attached to the Pi at `<pi-ip>` and exported over NFS
  (`/media/bryan/shared`, `root_squash,anonuid=1000,anongid=1000` to `<rig-subnet>`). The Pi is the network
  leader: storage and time (chrony serves the LAN; the rig syncs only from the Pi).
- The rig mounts it at `/mnt/shared` (NFS automount, soft, 10 min idle timeout). The rig itself is
  self-contained for idle/remote mode: code, config, model library and the NVMe model cache are rig-local
  (`~/llm_orchestration`, `/data/model-hotset`); the share holds the canonical model archive
  and the orchestration/bench stack files.
- Rig-local path convention: `/mnt/shared/...`
- Pi/operator path convention: `/media/bryan/shared/...`
- CPU worker path convention: usually `/media/bryan/shared/...` (some hosts may expose `/mnt/shared/...`)
- The same files are reached through different mount prefixes depending on where the command runs.

Path rule:
- if a command executes on the rig itself, prefer `/mnt/shared/...`
- if a command executes on the Pi at `<pi-ip>`, prefer `/media/bryan/shared/...`
- if a wrapper proxies work to the rig, pass shared paths that can be normalized to rig-visible paths

Operational rule:
- Scripts that run on CPU workers must resolve shared root dynamically (`/media/bryan/shared` vs `/mnt/shared`) instead of hardcoding one path.
- If CPU workers disappear from dashboard, first verify worker shared mount points resolve to rig `<rig-ip>` and are actually mounted.

## Architecture — Who Does What
- **Brain**: Sequences work. Promotes private→public tasks when dependencies resolve. Handles retries (max 3 attempts). Manages model lifecycle via meta tasks.
- **Workers**: Dumb grabbers. Scan `tasks/queue/`, claim whatever matches. No dependency checking, no sequencing. If 5 tasks are in the queue, 5 workers grab them simultaneously.
- **Scripts**: `prepare_llm_runtimes.py` enforces sequential loading by writing one meta task, waiting for completion, then writing the next. This is the correct external entry point — not writing batches of tasks to the queue.

## Non-Negotiables
- Fail fast with clear errors.
- No backward-compatibility hacks.
- Don't run plan scripts directly; always submit through `submit.py` or dashboard.
- Model loading/unloading happens through brain-managed meta tasks only — never write directly to the task queue.

## Standard Procedures

These are the canonical operator actions to use in normal work. For deeper
detail and rationale, see `workspace/quickstart.md`.

### Rig Side First

Important execution boundary:
- Runtime control actions must be run from the GPU rig side, not from the laptop side.
- If you are on the laptop, SSH to the rig first:

```bash
ssh <rig-ip>
```

Use the rig side for:
- starting/stopping/restarting orchestrator services
- submitting batches
- resetting GPUs / return-to-default flows
- checking live local runtime ports and rig-local processes
- running benchmark harnesses that talk directly to worker runtime ports

The laptop side is still fine for:
- reading shared history folders
- reviewing logs and summaries on `/media/bryan/shared/`
- editing repo code/docs

Rule of thumb:
- If the command controls live runtime state, do it on `<rig-ip>`.
- If the command only reads shared artifacts or edits code, either side is fine.
- Worker runtime ports (`11435+`, split ports, and similar) are rig-local operator ports.
- If a benchmark/custom harness talks directly to a worker port, run that harness on the rig side or use an explicit proxy/tunnel.

### Start The System

The rig boots **idle**: nothing running, no model on any GPU (`rig-boot.service`).
Pick one mode at a time with the `rig` mode selector, from any machine:

```bash
ssh <rig-ip> rig status
ssh <rig-ip> rig start orchestration   # brain (27B on GPU0) + 5 cold workers, plans admitted
# bench mode: entered by a benchmark load through the rig manager (rigctl load-model --use benchmark, lanes, campaigns)
python3 ~/llm-gateway/tools/rigctl.py load-model MODEL --use chat  # manager sends a full rig load spec
ssh <rig-ip> rig start <mode> --force  # replace whatever mode is running
ssh <rig-ip> rig stop                  # back to idle
```

- `start` refuses (exit 2) if another mode is running and reports who started it and when.
- Remote loads are additive through the manager: it fits and leases cards, then sends a
  full spec to `rig start remote <id> --spec ... --gpus ... --port ...`.
  `rig stop <id>` unloads one. `rig status --json` reports the actual state.
- llm-gateway (Pi `~/llm-gateway`, `pi-gateway.service` on Pi 127.0.0.1:8080) serves `gpu-rig/<profile>` over
  the OpenAI API and loads profiles on first request (`rig start remote` over ssh; rig ports reach the Pi
  through `pi-rig-tunnel.service`). Deploy: `~/llm-gateway/deploy/deploy_pi.sh` on the Pi.
- Orchestration runs under systemd (`llm-orchestration.service`).
  Never start `startup.py` by hand.
- Gateway owns its published offers. The rig receives each accepted load as a complete
  manager spec and does not read gateway offer files.
- State: `brain/rig_mode.json`. Code: `shared/scripts/rig/rig.py`. Design: `workspace/implement/rig_boot_modes.md`.
- Rig hardware layer (2026-10-17): `shared/scripts/rig/README.md`. The controller is llm-gateway's rig manager on
  the Pi (`~/llm-gateway/docs/MANAGER.md`); the rig keeps the GPU lease/claim record (`rig lease`, `rig claim`,
  rig-local `brain/hw/gpus/`, mirrored to `/mnt/shared/gpus/`), runs loads as detached jobs, and enforces the
  hardware rules: `rig-guard.service` (power/thermal/RAM; gates every load in `run_runtime.sh`, frees RAM at
  critical), one load at a time, host prompt cache off unless the caller sets `--cache-ram`, lease audit.
- Benchmark models load through the rig manager (`rigctl load-model MODEL --use benchmark`, or a lane or
  campaign: benchmarking `NEW_MODEL_INTEGRATION.md`); that puts the rig in bench mode, and the last unload
  returns it to idle. One mode at a time: a load for another mode is refused as busy. `start_plan_mode.py`
  is superseded by `rig start orchestration`.

### Check The Rig

```bash
pgrep -af "brain.py|gpu.py"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
# Runtime-specific probe (depends on runtime_backend config):
#   active runtime: curl -s http://127.0.0.1:<port>/v1/models
#   health probe:    curl -s http://127.0.0.1:<port>/health
#   containers:      docker ps --format '{{.Names}}\t{{.Status}}\t{{.Ports}}'
```

Important:
- inspect controller-owned GPU allocations with `python3 /mnt/shared/scripts/manage_gpu_lease.py --shared-path /mnt/shared list` on the rig
- do not rely on a shared `status.py` path under `/mnt/shared`; there is no supported status helper there right now
- if docs mention `/mnt/shared/scripts/status.py`, treat that as stale and use direct probes instead
- `python3 ~/llm_orchestration/scripts/runtime_preflight.py --config config.benchmark.json --json` is the operator preflight entrypoint for current rig-state checks

Primary live log:

```bash
tail -f ~/llm_orchestration/shared/logs/brain_decisions.log
```

### Submit A Batch

Preferred CLI path:

```bash
python3 ~/llm_orchestration/scripts/submit.py \
  ~/llm_orchestration/shared/plans/<shoulder-or-arm>/<plan_name> \
  --config '{"KEY":"VALUE"}'
```

Submit path notes:
- local operator entrypoint: `~/llm_orchestration/scripts/submit.py`
- that wrapper proxies normal submissions to rig-side `python3 /mnt/shared/agents/submit.py`
- config values that refer to files the rig must open should resolve to rig-visible shared paths
- accepted shared aliases are normalized by the wrapper:
  - `~/llm_orchestration/shared/...`
  - `/media/bryan/shared/...`
  - `/mnt/shared/...`

Use the dashboard as the other normal front door:

```bash
/media/bryan/shared/scripts/start_dashboard.sh
```

Controls:
- Dashboard: `http://127.0.0.1:8787/`
- Controls: `http://127.0.0.1:8787/controls`

Important:
- wrapper `scripts/submit.py` or dashboard only for normal runs
- direct `/mnt/shared/agents/submit.py` is debug-only

For complex JSON quoting, use the temp-file method in `workspace/quickstart.md`.

### Check A Batch

Task lanes:

```bash
ls ~/llm_orchestration/shared/tasks/queue
ls ~/llm_orchestration/shared/tasks/processing
ls ~/llm_orchestration/shared/tasks/complete
ls ~/llm_orchestration/shared/tasks/failed
```

Read-only batch review can be done from either the rig or the laptop.

Summarize one batch with the shared review tool:

```bash
python3 /media/bryan/shared/scripts/summarize_history_run.py \
  --history-dir /media/bryan/shared/plans/<shoulder-or-arm>/<plan_name>/history/<batch_id>
```

Roll up many historical batches with the repo script:

```bash
python3 ~/llm_orchestration/scripts/rollup_history.py \
  /media/bryan/shared/plans/<shoulder-or-arm>/<plan_name>/history
```

Optional runtime-focused diagnostics for one batch:

Run this on the rig side when possible. Laptop-side runs can still read shared
heartbeats, but local runtime port probes and GPU telemetry will be degraded.

```bash
# preferred:
#   ssh <rig-ip>
#   python3 /mnt/shared/scripts/batch_runtime_diag.py --batch-id <batch_id> --shared-path /mnt/shared
#
# laptop-side fallback:
python3 /media/bryan/shared/scripts/batch_runtime_diag.py \
  --batch-id <batch_id> \
  --shared-path /media/bryan/shared
```

Where results land:
- one-batch summary: `.../history/<batch_id>/RUN_SUMMARY.md`
- rollup summary: `.../history/_summary/ROLLUP_SUMMARY.md`
- rollup failures list: `.../history/_summary/failures.jsonl`

### Kill A Batch

```bash
cd ~/llm_orchestration
source ~/llm-orchestration-venv/bin/activate
python3 scripts/kill_plan.py [batch_id]
```

Rig note:
- the shared drive does not currently expose `kill_plan.py` under `/mnt/shared/scripts/`
- use the repo copy from `~/llm_orchestration/scripts/kill_plan.py` on the rig

Useful options:
- `--keep-workers`
- `--keep-models`
- `--clean-reset`

### Load Models Onto GPUs

Outside orchestration, a model loads through the rig manager on the Pi: `rigctl load-model MODEL [--gpus
..] [--use benchmark|chat|worker]` (llm-gateway `docs/MANAGER.md`, "Model loads"). The manager checks the
fit, picks the cards and the context, and the rig loads it in the matching mode.

In orchestration mode the brain loads worker models with `load_llm` meta tasks
(`prepare_llm_runtimes.py` queues them one at a time). Model IDs come from the model registry
(`models.catalog.json` is generated from it).

Key scripts in the chain:
- `~/llm_orchestration/scripts/runtime_preflight.py` — operator preflight entry point
- `shared/scripts/prepare_llm_runtimes.py` — rig-side, sequential meta task submission
- `shared/scripts/runtime_preflight.py` — rig-side cleanup/preflight checks
- `shared/agents/gpu_tasks.py` — worker-side task claiming
- `shared/agents/gpu_llama.py` — worker-side container lifecycle

Debug-only direct smoke path:
- `/mnt/shared/scripts/llama_runtime/run_runtime.sh` on the rig
- use this only to isolate runtime/container failures or validate GGUF compatibility
- do not leave direct runtime containers running alongside orchestrator-owned worker ports

Known runtime compatibility issue:
- the current `llama-runtime:sm61-sm86` build does not load Gemma 4 GGUF files
- observed failure on April 2, 2026: `unknown model architecture: 'gemma4'`
- this blocks both worker-tier and brain-tier Gemma 4 smoke runs until the runtime image is updated

### Reset A GPU

Normal operator reset path:
- dashboard `Reset selected GPU`

Whole-rig reset path:
- dashboard `Return To Default`

Do not treat internal `reset_gpu_runtime` meta tasks as the normal operator tool.

### Stop The System

```bash
ssh <rig-ip> rig stop
```

## Dashboard Actions

Canonical operator-facing dashboard actions:
- `Start Plan`
- `Kill Plan`
- `Reset selected GPU`
- `Return To Default`

These are part of the same orchestration system, not separate execution paths.
