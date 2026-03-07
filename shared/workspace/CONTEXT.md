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

## Non-Negotiables
- Fail fast with clear errors.
- No backward-compatibility hacks.
- Don’t run plan scripts directly; always submit through `submit.py` or dashboard.

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
- checking live local Ollama ports and rig-local processes

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
python3 ~/llm_orchestration/scripts/start_custom_mode.py --brain-model ... --single-model ... --split-model ...
```

### Check The Rig

```bash
pgrep -af "brain.py|gpu.py"
curl -s http://localhost:11434/api/ps | jq -r '.models[].name'
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
python3 ~/llm_orchestration/scripts/status.py
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

```bash
python3 /media/bryan/shared/scripts/batch_runtime_diag.py \
  /media/bryan/shared/plans/<shoulder-or-arm>/<plan_name>/history/<batch_id>
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
