# Cloud Escalation Artifact Schema

Reference document for brain agents writing cloud escalation requests.

## When to Write

- `execute_plan` fails to start (missing plan path, invalid resume batch, parse/start errors)
- Repeated unfixable definition/runtime failures needing external reasoning
- Local context insufficient to recover automatically

## File Location

`shared/brain/escalations/{timestamp}_{id}.json`

## Required Fields

```json
{
  "escalation_id": "string",
  "created_at": "ISO-8601",
  "status": "pending",
  "target": "cloud_brain",
  "source": "local_brain",
  "brain_name": "string",
  "hostname": "string",
  "type": "e.g. execute_plan_failure",
  "title": "string",
  "details": "string or object",
  "recommended_action": "string"
}
```

## Optional: Source Task Context

```json
{
  "source_task": {
    "task_id": "string",
    "type": "string",
    "name": "string",
    "batch_id": "string",
    "plan_path": "string",
    "config": {}
  }
}
```

## Required Context for Cloud (`details.context`)

- `dig_order` — ordered recovery steps
- `requested_plan_path`
- `candidate_plan_dirs`
- `existing_plan_md`
- `resume_batch_id` (if any)
- `resume_manifest_candidates`
- `key_files`: `brain_decisions_log`, `brain_state`, `task_complete_dir`, `task_failed_dir`, `queue_dir`

## Local Behavior After Escalation

- Mark originating task failed with `escalated: true` and `escalation_id`
- Do not silently retry the same invalid launch context
- Keep system available for other tasks

## HUMAN File Format

```markdown
# HUMAN: {Topic}
**Created**: {timestamp}
**Severity**: low | medium | high | security
**Context**: What was being attempted
**Problem**: What went wrong
**Attempts**: What was tried
**Recommendation**: What the brain thinks should happen
```
