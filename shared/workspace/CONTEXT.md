# LLM Orchestration — Context (Short)

Purpose: Brain orchestrates tasks; GPU/CPU workers execute; plans define work.

## Read First
- `workspace/quickstart.md` (submit, monitor, kill plan)
- `workspace/PLAN_FORMAT.md` (task schema)
- `workspace/brain-behavior.md` (brain loop details)
- `core/RULES.md` (immutable safety constraints)

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
- Model catalog: `shared/agents/models.catalog.json` (model IDs, GGUF paths, placement, tiers)

## Shared Storage Topology (Current)

- Shared drive/NFS source is the GPU rig at `10.0.0.3` (not `10.0.0.2`).
- Rig-local path convention: `/mnt/shared/...`
- Laptop/operator path convention: `/media/bryan/shared/...`
- CPU worker path convention: usually `/media/bryan/shared/...` (some hosts may expose `/mnt/shared/...`)

Operational rule:
- Scripts that run on CPU workers must resolve shared root dynamically (`/media/bryan/shared` vs `/mnt/shared`) instead of hardcoding one path.
- If CPU workers disappear from dashboard, first verify worker shared mount points resolve to rig `10.0.0.3` and are actually mounted.

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
ssh 10.0.0.3
```

Use the rig side for:
- starting/stopping/restarting orchestrator services
- submitting batches
- resetting GPUs / return-to-default flows
- checking live local runtime ports and rig-local processes

The laptop side is still fine for:
- reading shared history folders
- reviewing logs and summaries on `/media/bryan/shared/`
- editing repo code/docs

Rule of thumb:
- If the command controls live runtime state, do it on `10.0.0.3`.
- If the command only reads shared artifacts or edits code, either side is fine.

### Start The System

Normal orchestrator start:

```bash
cd ~/llm_orchestration/shared/agents
python3 startup.py
```

Wrapper start modes when you intentionally want them:

```bash
python3 ~/llm_orchestration/scripts/start_default_mode.py
python3 ~/llm_orchestration/scripts/start_benchmark_mode.py
python3 ~/llm_orchestration/scripts/start_custom_mode.py --models qwen3.5:4b qwen2.5-coder:7b ...
```

### Check The Rig

```bash
pgrep -af "brain.py|gpu.py"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
python3 ~/llm_orchestration/scripts/status.py
# Runtime-specific probe (depends on runtime_backend config):
#   active runtime: curl -s http://127.0.0.1:<port>/v1/models
```

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
#   ssh 10.0.0.3
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

Useful options:
- `--keep-workers`
- `--keep-models`
- `--clean-reset`

### Load Models Onto GPUs

Use `start_custom_mode.py` to load specific models onto workers. It starts
benchmark mode (cold workers), then calls `prepare_llm_runtimes.py` on the rig
which queues one `load_llm` meta task at a time, waits for completion, then
queues the next.

```bash
python3 ~/llm_orchestration/scripts/start_custom_mode.py \
  --models qwen3.5:4b qwen2.5-coder:7b qwen3.5:9b-q3km mistral:7b-instruct deepseek-r1:7b \
  --force-unload-first
```

Model IDs come from `shared/agents/models.catalog.json`. One model per
available cold worker. The brain and workers handle sequencing, retry, and
placement — do not try to replicate this manually.

Key scripts in the chain:
- `~/llm_orchestration/scripts/benchmarks/start_custom_mode.py` — laptop-side entry point
- `shared/scripts/prepare_llm_runtimes.py` — rig-side, sequential meta task submission
- `shared/agents/gpu_tasks.py` — worker-side task claiming
- `shared/agents/gpu_llama.py` — worker-side container lifecycle

### Reset A GPU

Normal operator reset path:
- dashboard `Reset selected GPU`

Whole-rig reset path:
- dashboard `Return To Default`

Do not treat internal `reset_gpu_runtime` meta tasks as the normal operator tool.

### Stop The System

```bash
pkill -f "brain.py|gpu.py"
```

Or stop foreground `startup.py` with `Ctrl+C`.

## Dashboard Actions

Canonical operator-facing dashboard actions:
- `Start Plan`
- `Kill Plan`
- `Reset selected GPU`
- `Return To Default`

These are part of the same orchestration system, not separate execution paths.
