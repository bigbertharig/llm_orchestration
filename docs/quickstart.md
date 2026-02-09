# Quick Start Guide

This document tells you how to interact with the LLM orchestration system. Read this first when starting a session.

**For deeper understanding:** [CONTEXT.md](CONTEXT.md) (project overview) | [architecture.md](architecture.md) (system design)

---

## System Overview

A multi-GPU LLM orchestration system running on 5x GTX 1060 GPUs:
- **Brain** (Qwen 14B on GPUs 0+3) - coordinates tasks, monitors workers
- **Workers** (Qwen 7B on GPUs 1, 2, 4) - execute tasks in parallel
- **Plans** - markdown files defining goals, scripts, and task dependencies

---

## Check System State

```bash
# Are agents running?
pgrep -af "brain.py|worker.py"

# What's loaded in Ollama?
curl -s http://localhost:11434/api/ps | jq -r '.models[].name'

# GPU memory usage
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

# Task queue status
ls ~/Documents/llm_orchestration/shared/tasks/{queue,processing,complete}/ 2>/dev/null | head -20
```

---

## Starting the System

If agents aren't running:

```bash
cd ~/Documents/llm_orchestration/shared/agents
source ~/ml-env/bin/activate
python launch.py
```

**What launch.py does:**
1. Connects to Ollama (auto-starts on boot via systemd)
2. Starts brain, loads 14B model on GPUs 0+3
3. Starts 3 COLD workers (no models loaded - brain manages model loading)
4. System enters idle state, ready for plans

**Startup time:** ~2-3 minutes (only brain model needs loading)

---

## Submitting a Plan

```bash
cd ~/Documents/llm_orchestration
source ~/ml-env/bin/activate

python scripts/submit.py shared/plans/<plan_name> \
  --config '{"VAR1": "value1", "VAR2": "value2"}'
```

**Example - video_zim_batch:**
```bash
python scripts/submit.py shared/plans/video_zim_batch \
  --config '{
    "ZIM_PATH": "/path/to/videos.zim",
    "SOURCE_ID": "my-source",
    "OUTPUT_FOLDER": "/tmp/output"
  }'
```

Plans live in `shared/plans/`. Each has a `plan.md` defining the workflow. See [PLAN_FORMAT.md](../shared/plans/PLAN_FORMAT.md) for how to write plans.

---

## Monitoring

```bash
# Main log - brain decisions and task flow
tail -f ~/Documents/llm_orchestration/shared/logs/brain_decisions.log

# Task queues
ls shared/tasks/queue/       # waiting for workers
ls shared/tasks/processing/  # being worked on
ls shared/tasks/complete/    # finished
```

---

## Killing a Running Plan

To stop a plan mid-execution and clean up:

```bash
cd ~/Documents/llm_orchestration
source ~/ml-env/bin/activate
python scripts/kill_plan.py [batch_id]
```

**What kill_plan.py does:**
1. Kills running script processes (e.g., transcriptions)
2. Clears task queues (pending, processing, failed) and stale locks
3. Kills workers (unless `--keep-workers`)
4. Clears batch from brain state
5. Unloads worker models (brain model stays loaded)

**Options:**
- `batch_id` - Kill specific batch (optional, default: all)
- `--keep-workers` - Don't kill workers
- `--keep-models` - Don't unload worker models

---

## Stopping the System

Press `Ctrl+C` in the launch.py terminal, or:

```bash
pkill -f "brain.py|worker.py"
```

---

## Key Paths

| Path | Purpose |
|------|---------|
| `shared/agents/` | Brain and worker code, config.json |
| `shared/plans/` | Plan definitions |
| `shared/tasks/` | Task queue (queue/, processing/, complete/, failed/) |
| `shared/logs/` | Logs including brain_decisions.log |
| `docs/` | Documentation |

---

## Common Issues

| Symptom | Check |
|---------|-------|
| Agents not running | `python launch.py` to start |
| Tasks stuck | Check `brain_decisions.log`, worker heartbeats |
| GPU OOM | `nvidia-smi` - models may be on wrong GPUs |
| Stale state | Clear `shared/brain/state.json` active_batches |

---

## Next Steps

- **Understand the system:** [architecture.md](architecture.md)
- **Write a plan:** [PLAN_FORMAT.md](../shared/plans/PLAN_FORMAT.md)
- **Debug brain behavior:** [brain-behavior.md](brain-behavior.md)

---

*Last Updated: February 2026*
