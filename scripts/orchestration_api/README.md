# Orchestration status API

The gateway owns the dashboard, plan catalog, input preparation, and start/resume requests. Orchestration consumes a prepared plan, executes it, and reports its state through this versioned local HTTP API.

The API runs as `bryan` on `/run/llm-orchestration/api.sock` (Unix socket, mode `0600`; parent directory `0700`). It has no TCP listener. The gateway uses the same socket; its existing `/dashboard/api/orchestration` routes remain the UI-facing facade. This follows the versioned HTTP contract pattern of the gateway's `/v1/manager` API.

| Request | Purpose |
|---|---|
| `GET /health` | API version and liveness |
| `GET /v1/orchestration/status` | Current hardware guard and GPU scan, plus task worker state and active batches |
| `GET /v1/orchestration/diagnostics` | Always-on hardware health; task worker heartbeats only while a prepared batch is active |
| `GET /v1/orchestration/activity?batch_id=...&after=0&limit=100` | Pages through one active or recently finished batch's event journal; cursor is a byte offset |
| `POST /v1/orchestration/actions/submit_prepared` | Execute a gateway-selected `plan_name`, `plan_scope`, optional `starter_file`, and resolved `config` |
| `POST /v1/orchestration/actions/{action}` | `kill_plan`, `kill_all_active`, `batch_outputs`, `cleanup_stale`, `requeue_worker` |

The API rejects unknown actions and bodies over 1 MiB; actions are serialized. Status, diagnostics, and activity are read only. The rig's boot-enabled `rig-guard.service` samples hardware continuously and handles load admission locally; `rig-plan.service` waits for queued work. Both run independently of task mode and do not load models while idle. The Pi API is also boot-enabled, but it does not poll the rig in the background. Each status or diagnostics request reads the guard's current sample and performs a bounded GPU scan over SSH; connection failure is reported as unhealthy. Brain and worker heartbeats appear only while a prepared batch is active; old idle heartbeat files do not produce false alerts. Status includes a compact index of batches completed in the last 24 hours. Activity reads a single batch journal with bounded pages. The brain writes the recent-completion index when a batch ends. Full historical views belong to the owning repositories. Authentication is the Unix socket's filesystem permission.

The rig guard owns hardware protection and load admission. The gateway manager owns leases and model placement and provides `/v1/manager/status`. Orchestration's status API observes the rig and consumes prepared plans; it does not duplicate those controls. The dashboard and rig information MCP tools use the gateway's read endpoints for the same orchestration snapshot and diagnostics.

Run with `python3 -m orchestration_api --socket /run/llm-orchestration/api.sock` from `scripts/`. Test with `PYTHONPATH=scripts python3 -m pytest -q scripts/orchestration_api/test_api.py scripts/orchestration_api/test_status.py scripts/orchestration_api/test_prepared.py`.
