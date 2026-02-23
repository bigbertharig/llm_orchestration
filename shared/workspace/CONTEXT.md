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

## Non-Negotiables
- Fail fast with clear errors.
- No backward-compatibility hacks.
- Don’t run plan scripts directly; always submit through `submit.py` or dashboard.

## Start Plan (GPU rig)
```bash
python3 /mnt/shared/agents/submit.py \
  /mnt/shared/plans/arms/<plan_name> \
  --config '{"KEY":"VALUE"}'
```

For complex JSON quoting, use the temp-file method in `workspace/quickstart.md`.
