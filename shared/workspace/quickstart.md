# Quick Start Guide

This is the operator quickstart for the orchestration system. It documents the one supported workflow used by plans, the brain, and the dashboard.

For deeper detail, see [CONTEXT.md](CONTEXT.md), [architecture.md](architecture.md), [PLAN_FORMAT.md](PLAN_FORMAT.md), and [brain-behavior.md](brain-behavior.md).

---

## Core Rules

1. Start and run the system through the orchestrator, not by manually managing worker runtimes.
2. Submit plans through the wrapper submit script or the dashboard only.
3. Let the brain decide when worker models load and unload through `meta` tasks.
4. Treat manual runtime processes (for example `docker run llama-server`), manual task JSON insertion, and direct agent submit paths as debug-only recovery tools.

If a workflow bypasses queue-driven `meta` tasks, it is not the normal path.

---

## Mental Model

- `startup.py` starts the orchestrator stack and loads the brain model.
- GPU workers start cold by default.
- The brain inserts `load_llm` / `unload_llm` / split-runtime `meta` tasks when work requires a model change.
- GPU agents handle those `meta` tasks directly and update heartbeat/runtime state.
- Plans, the dashboard, and runtime management all use the same task lanes and state files.
- The runtime backend is `llama` and is set explicitly in `config.json` via `runtime_backend`.

That is the unified system. There should not be separate "plan loading", "dashboard loading", and "manual runtime loading" pathways in normal operation.

---

## Check System State

```bash
pgrep -af "brain.py|gpu.py"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
ls ~/llm_orchestration/shared/tasks/{queue,processing,complete}/ 2>/dev/null | head -20
# Runtime-specific probe:
#   active runtime: curl -s http://127.0.0.1:<port>/v1/models
#                   curl -s http://127.0.0.1:<port>/health
#                   docker ps --filter name=llama-worker
```

What to expect:
- `brain.py` and GPU agents are running
- brain model responds on its configured port
- worker VRAM use matches actual queued/running orchestrator state

---

## Start The System

Normal start:

```bash
cd ~/llm_orchestration/shared/agents
python3 startup.py
```

What `startup.py` is responsible for:
1. Read `config.json`
2. Verify hardware against config
3. Start the brain and load the brain model
4. Reclaim configured worker runtime ports
5. Start GPU agents cold by default
6. Optionally queue startup warm loads if explicitly configured

Notes:
- Brain model loading belongs to startup.
- Worker model loading belongs to orchestrator `meta` tasks.
- Keep always-on worker-specific runtime services disabled so the orchestrator has single ownership of worker runtime ports.

Use wrapper mode scripts only when you intentionally want that mode's startup behavior:

```bash
python3 ~/llm_orchestration/scripts/start_default_mode.py
python3 ~/llm_orchestration/scripts/start_benchmark_mode.py
python3 ~/llm_orchestration/scripts/start_custom_mode.py --brain-model ... --single-model ... --split-model ...
```

Those wrappers are still orchestration entry points, not separate runtime systems.

---

## Model Loading

### Canonical Path

Use orchestrator `meta` tasks for all worker model load and unload operations.

Canonical ownership:
- Brain model: loaded by `startup.py`
- Single-worker model changes: `load_llm` / `unload_llm`
- Split-pair model changes: `load_split_llm` / `unload_split_llm`

Authoritative runtime implementation:
- `/media/bryan/shared/agents/gpu_tasks.py`

Why this matters:
- heartbeats stay accurate
- dashboard state stays accurate
- runtime port ownership stays coherent
- brain resource logic does not fight unmanaged runtime instances

### What Not To Do

Do not start manual worker runtime processes (`docker run llama-server`) in normal operation.

Symptoms of bypassing the orchestrator:
- VRAM is used but dashboard/runtime linkage is missing
- workers show loaded models that the brain did not request
- ports are occupied by the wrong owner
- models load and then get immediately unloaded by brain policy

---

## Submit A Plan

Always submit plans through one of these two front doors:

1. Wrapper submit script
2. Dashboard Start Plan control

Both should feed the same orchestration path. The dashboard is an operator UI for the same system, not a separate execution mode.

### Preferred CLI Path

```bash
python3 ~/llm_orchestration/scripts/submit.py \
  ~/llm_orchestration/shared/plans/arms/<plan_name> \
  --config '{"VAR1":"value1","VAR2":"value2"}'
```

What the wrapper adds:
1. worker preflight scan/reset gate
2. placeholder validation
3. shared-path translation
4. SSH proxy to rig when needed

Path notes:
- local operator entrypoint: `~/llm_orchestration/scripts/submit.py`
- rig-side submit implementation: `/mnt/shared/agents/submit.py`
- for file-based config values, prefer shared paths that the rig can open directly
- the wrapper normalizes these shared aliases:
  - `~/llm_orchestration/shared/...`
  - `/media/bryan/shared/...`
  - `/mnt/shared/...`

### Dashboard Path

Start dashboard:

```bash
/media/bryan/shared/scripts/start_dashboard.sh
```

Use:
- Dashboard: `http://127.0.0.1:8787/`
- Controls: `http://127.0.0.1:8787/controls`

Controls page actions:
- `Start Plan`
- `Kill Plan`
- `Return To Default`
- `Reset selected GPU`

### Direct Agent Submit

```bash
python3 /mnt/shared/agents/submit.py ...
```

This is debug-only. Do not use it for normal runs because it bypasses wrapper safeguards.

---

## Plan Start Checklist

1. Confirm `brain.py` and GPU agents are running.
2. Confirm you are using the correct plan under `shared/plans/shoulders/` or `shared/plans/arms/`.
3. Build valid JSON config from the plan's `## Inputs`.
4. Submit through the wrapper or dashboard.
5. Capture the returned `task_id` / `batch_id`.
6. Monitor logs and task lanes.

For long JSON, prefer the temp-file method:

```bash
cat >/tmp/plan_config.json <<'JSON'
{
  "VAR1": "value1",
  "VAR2": "value2"
}
JSON

CONFIG=$(python3 -c 'import json;print(json.dumps(json.load(open("/tmp/plan_config.json")),separators=(",",":")))')
python3 ~/llm_orchestration/scripts/submit.py \
  ~/llm_orchestration/shared/plans/arms/<plan_name> \
  --config "$CONFIG"
```

---

## Monitor Execution

```bash
tail -f ~/llm_orchestration/shared/logs/brain_decisions.log
ls ~/llm_orchestration/shared/tasks/queue
ls ~/llm_orchestration/shared/tasks/processing
ls ~/llm_orchestration/shared/tasks/complete
ls ~/llm_orchestration/shared/tasks/processing/*.heartbeat.json
```

What to watch for:
- `meta` tasks appear when the brain needs runtime changes
- task files move from `queue/` to `processing/` to `complete/`
- worker heartbeats reflect the expected runtime state after load/unload

Archive old finished task files when lanes get noisy:

```bash
python3 ~/llm_orchestration/scripts/archive_tasks.py
```

---

## Review Batch History

Use these commands when you want to inspect completed or failed runs. These are read-only history tools, so they are safe to run from either the laptop or the rig.

### One Batch

Shared summarizer:

```bash
python3 /media/bryan/shared/scripts/summarize_history_run.py \
  --history-dir /media/bryan/shared/plans/<shoulder-or-arm>/<plan_name>/history/<batch_id>
```

What it does:
1. reads the batch artifact directory
2. refreshes `RUN_SUMMARY.json`
3. refreshes `RUN_SUMMARY.md`

Output location:

```text
/media/bryan/shared/plans/<shoulder-or-arm>/<plan_name>/history/<batch_id>/RUN_SUMMARY.md
```

### Many Batches

Repo rollup tool:

```bash
python3 ~/llm_orchestration/scripts/rollup_history.py \
  /media/bryan/shared/plans/<shoulder-or-arm>/<plan_name>/history
```

What it writes:
1. `_summary/ROLLUP_SUMMARY.md`
2. `_summary/ROLLUP_SUMMARY.json`
3. `_summary/runs.jsonl`
4. `_summary/failures.jsonl`

Typical follow-up commands:

```bash
cat /media/bryan/shared/plans/<shoulder-or-arm>/<plan_name>/history/_summary/ROLLUP_SUMMARY.md
tail -n 20 /media/bryan/shared/plans/<shoulder-or-arm>/<plan_name>/history/_summary/failures.jsonl
```

### Runtime-Focused Review

If the batch looks like a load/unload or GPU ownership problem, use:

```bash
python3 /media/bryan/shared/scripts/batch_runtime_diag.py \
  /media/bryan/shared/plans/<shoulder-or-arm>/<plan_name>/history/<batch_id>
```

Use this after:
- repeated `load_llm` failures
- split-runtime preflight timeouts
- dashboard state that does not match task history

### Script Location Rule

Use the exact script paths below:
- `/media/bryan/shared/scripts/summarize_history_run.py` for one batch
- `~/llm_orchestration/scripts/rollup_history.py` for many batches
- `/media/bryan/shared/scripts/batch_runtime_diag.py` for runtime diagnosis

Do not guess the path or omit required flags. The one-batch summarizer requires `--history-dir`.

---

## Reset And Recovery

Use recovery actions only when normal orchestration state is inconsistent.

### First-Line Recovery

If runtime state looks wrong in dashboard or heartbeats:
1. Stop creating manual runtime changes
2. Confirm `queue/` and `processing/` are not carrying stale runtime tasks
3. Use dashboard `Return To Default` or the equivalent orchestrator reset path
4. Restart in the intended mode
5. Let the brain re-establish runtime state through `meta` tasks

### Reset Options

Use the reset path that matches the problem:

- Dashboard `Reset selected GPU`:
  - operator-facing targeted hard reset
  - kills and restarts only the selected GPU worker
  - clears that GPU's worker-port listener and related split artifacts
  - this is the correct targeted reset path for normal runtime recovery
- Dashboard `Return To Default`:
  - full operator reset back to startup defaults
  - use when the whole rig is mixed up, not just one worker
- Internal `reset_gpu_runtime` meta task:
  - agent-side thermal recovery path
  - not the normal operator reset tool
  - do not use this as the default manual recovery path unless it is explicitly the recovery flow being tested

Practical rule:
- one bad worker: use dashboard `Reset selected GPU`
- whole rig drifted: use dashboard `Return To Default`
- agent thermal recovery logic: internal `reset_gpu_runtime`

### When To Escalate To Deeper Runtime Debugging

Only move into low-level runtime work when one of these is true:
- stale port ownership persists after orchestrator reset
- split runtime repeatedly fails through normal `meta` task recovery
- dashboard and heartbeat state remain inconsistent after restart

At that point, use the dedicated recovery scripts and inspect `gpu_tasks.py`. Do not treat those procedures as the normal operating path.

---

## Kill A Running Plan

```bash
cd ~/llm_orchestration
source ~/llm-orchestration-venv/bin/activate
python3 scripts/kill_plan.py [batch_id]
```

What it does:
1. kills running plan work
2. clears queued/in-flight batch artifacts
3. optionally keeps workers alive
4. unloads worker models unless told not to

Useful options:
- `--keep-workers`
- `--keep-models`
- `--clean-reset`

---

## Stop The System

```bash
pkill -f "brain.py|gpu.py"
```

Or stop the foreground `startup.py` process with `Ctrl+C`.

---

## Branch Safety

Keep `/home/bryan/llm_orchestration` on `main` for live rig operations.

Do not switch that repo to feature branches during runtime work. Use a separate git worktree for feature development.

Check before running orchestration commands:

```bash
git -C /home/bryan/llm_orchestration rev-parse --abbrev-ref HEAD
```

Expected output:

```text
main
```

---

## Key Paths

| Path | Purpose |
|------|---------|
| `shared/agents/` | Brain and GPU agent code |
| `shared/core/` | Protected core instructions |
| `shared/workspace/` | Docs, escalations, notes |
| `shared/plans/` | Plan definitions |
| `shared/tasks/` | Queue and task state |
| `shared/logs/` | Brain and runtime logs |

---

## Common Problems

| Symptom | Likely Cause | Action |
|---|---|---|
| VRAM used but dashboard not linked | Worker runtime started outside orchestrator path | Reset to default, restart cleanly, then load only via `meta` tasks |
| Models load and unload immediately | Brain policy says too many hot workers for current config | Fix mode/config, then retry through orchestrator |
| Split runtime behaves inconsistently | Stale split ownership or port state | Use orchestrator reset first, then split recovery path if still broken |
| Plan behaves differently between CLI and dashboard | One path bypassed the wrapper/orchestrator flow | Use wrapper submit or dashboard only; avoid direct agent submit |
| Tasks appear stuck | stale processing task, dead worker, or bad runtime state | inspect `brain_decisions.log`, task heartbeats, and active agents |

---

## Bottom Line

Keep the system simple:
- start with `startup.py` or an orchestration wrapper
- submit plans with the wrapper or dashboard
- let the brain drive worker model state through `meta` tasks
- treat direct runtime manipulation (manual containers, ad hoc local runtime commands) as debugging, not operations

If reality does not match that model, fix the system back toward this flow instead of documenting another side path.

---

*Last Updated: March 6, 2026*
