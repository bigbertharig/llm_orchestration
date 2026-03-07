# Requeue Owner Scrub Core Fix (2026-03-05)

## Problem
Some tasks are retried/requeued with `status: pending` but still contain stale ownership/terminal fields:
- `assigned_to`
- `started_at`
- `completed_at`
- `result`

This causes confusion in dashboard/task lanes and can interfere with claim/recovery logic.

Observed example:
- `tasks/queue/78366171-19f4-4d75-bde6-0b2be7518d66.json`

## Immediate Mitigation Implemented
Operational scrubber added and run:
- Script: `/home/bryan/llm_orchestration/scripts/scrub_requeued_task_owners.py`
- Run command:
  - `python3 /home/bryan/llm_orchestration/scripts/scrub_requeued_task_owners.py --shared-path /media/bryan/shared`

## Required Permanent Core Fix
Add one shared helper in brain failure/monitor paths, then call it before writing to `tasks/queue`.

Suggested helper behavior:
1. Set `task["status"] = "pending"`
2. Remove: `assigned_to`, `started_at`, `completed_at`, `result`
3. Keep: `attempts`, `workers_attempted`, `last_attempt_at`, `first_attempted_at`
4. Optionally stamp: `requeued_at`, `requeue_reason`

### Patch sites
1. `shared/agents/brain_failures.py`
   - Worker retry branch (`attempts < max_attempts`)
   - Missing-module fix requeue branch
   - Brain-fix succeeded requeue branch
2. `shared/agents/brain_monitor.py`
   - `_recover_orphaned_processing_tasks()`
   - force-kill timeout requeue path in stuck-task recovery

## Acceptance Criteria
1. Any `tasks/queue/*.json` with `status: pending` has no `assigned_to` or `started_at`.
2. Any `tasks/queue/*.json` with `status: pending` has no `completed_at` or `result`.
3. No RETRY/ORPHAN_REQUEUE writes stale ownership fields back into queue files.

