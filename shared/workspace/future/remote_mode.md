# Remote Mode

## Concept

A toggle-able mode on the Pi that enables external access while away. Activate before leaving, deactivate when home. Default state is locked down.

## Activation

```bash
# Before leaving
python scripts/remote.py on

# When back home
python scripts/remote.py off
```

On activation:
- Starts a polling loop that checks external command queue
- Begins pushing status/heartbeat updates to external store
- Dashboard shows live GPU status, task progress, plan outputs

On deactivation:
- Stops polling
- Stops pushing status
- External dashboard goes read-only (shows last known state)
- No commands accepted

## Architecture

```
You (phone/laptop)
  |
  v
Vercel app (dashboard + API routes)
  |
  v
Vercel KV / Firestore (command queue + status store)
  ^
  |  (Pi polls outbound, never accepts inbound)
  |
Pi (remote_mode.py loop)
  |
  v
Local GPU rig (executes plans as normal)
  |
  v
Google Drive (output delivery to dedicated Gmail)
```

No ports opened. No tunnels. Pi initiates all connections.

## What You Can Do Remotely

- View GPU status, temperatures, task progress
- Trigger pre-approved plans with validated parameters
- View completed output (in dashboard or Google Drive)
- Kill a running plan

## What You Cannot Do Remotely

- Edit plans or scripts
- Create new plans
- Access the Pi shell
- Change system config
- Modify core/ files
- Run arbitrary commands

## Command Queue Format

Pi polls the external store for command files:

```json
{
  "id": "cmd_001",
  "action": "run_plan",
  "plan": "research_assistant",
  "arm": "research_prospector",
  "params": {
    "QUERY": "Find 20 AI real estate startup leads",
    "TARGET_COUNT": 20,
    "SEARCH_DEPTH": "basic"
  },
  "submitted_at": "2026-02-14T20:00:00Z",
  "status": "pending"
}
```

## Sanitization (Pi Side)

Pi does NOT trust anything from the command queue:

- `plan` must be in allowlist (e.g. `["research_assistant", "github_analyzer"]`)
- `arm` must be valid for that plan
- `action` must be in `["run_plan", "kill_plan", "status"]`
- String params: strip shell metacharacters, length limits, no path traversal
- Numeric params: type-check, range-check
- No raw file paths accepted — params map to predefined input slots
- All validation happens before anything touches the executor

## Allowlist Config

```json
{
  "remote_plans": {
    "research_assistant": {
      "arms": ["research_prospector", "research_contact_enrichment", "research_topic_research"],
      "params": {
        "QUERY": {"type": "string", "max_length": 500},
        "TARGET_COUNT": {"type": "int", "min": 1, "max": 100},
        "SEARCH_DEPTH": {"type": "enum", "values": ["basic", "deep"]},
        "INPUT_FILE": {"type": "preset", "values": ["default_contacts.csv"]}
      }
    }
  }
}
```

## Status Updates (Pi → External Store)

Pi pushes status on each poll cycle:

```json
{
  "timestamp": "2026-02-14T20:01:00Z",
  "mode": "remote",
  "gpu_rig": "online",
  "active_plan": "research_assistant",
  "batch_id": "20260214_200100",
  "tasks": {"total": 62, "complete": 14, "failed": 0, "processing": 3},
  "gpu_temps": [65, 71, 68, 70, 67],
  "pi_temp": 52
}
```

## Output Delivery

When a plan completes:
- Results uploaded to Google Drive (dedicated Gmail account)
- Dashboard shows completion notification with link to Drive folder
- Local results stay on Pi as normal

## Auth

- Vercel app: Google OAuth locked to the dedicated Gmail account
- Command queue: API key required on all writes (stored in Vercel env, not in code)
- Pi polling: authenticates to external store with its own service credentials

## Open Questions

- Vercel KV vs Firestore vs something simpler for the command/status store?
- Poll interval? 30s feels right — responsive enough without hammering
- Should kill_plan require a confirmation step? (submit kill → Pi acks → confirm kill)
- Google Drive API vs just uploading to a shared folder?
- Notification when plan completes? (email to the Gmail account, push notification)
