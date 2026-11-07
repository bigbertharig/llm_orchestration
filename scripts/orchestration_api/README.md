# Orchestration status API

The gateway owns the dashboard UI. This package owns orchestration status and plan controls and exposes them through a versioned local HTTP API. The gateway never imports this package or reads its internal files.

The API runs as `bryan` on `/run/llm-orchestration/api.sock` (Unix socket, mode `0600`; parent directory `0700`). It has no TCP listener. The gateway uses the same socket; its existing `/dashboard/api/orchestration` routes remain the UI-facing facade. This follows the versioned HTTP contract pattern of the gateway's `/v1/manager` API.

| Request | Purpose |
|---|---|
| `GET /health` | API version and liveness |
| `GET /v1/orchestration/status` | Workers, batches, task lanes, alerts, hardware reports as seen by orchestration |
| `GET /v1/orchestration/options` | Plan inputs and available choices |
| `POST /v1/orchestration/actions/{action}` | `start_plan`, `resume_plan`, `kill_plan`, `kill_all_active`, `batch_outputs`, `cleanup_stale`, `requeue_worker` |

Action bodies are JSON objects matching the gateway dashboard route's existing fields. The API rejects unknown actions and bodies over 1 MiB; actions are serialized. Status and options are read only. Authentication is the Unix socket's filesystem permission, suitable for this same-user, local-only phase. Review identity and authorization before exposing any dashboard route to LAN or remote clients.

The code under `data.py`, `controls.py`, `workers.py`, `plans.py`, `alerts.py`, `chains.py`, and `utils.py` remains owned by orchestration. Rig leases, GPU actions, and hardware inventory remain in the gateway manager, which already provides `/v1/manager/status`. The dashboard combines those two status feeds. No additional GPU control lives here.

Run with `python3 -m orchestration_api --socket /run/llm-orchestration/api.sock` from `scripts/`. Test with `python3 -m pytest -q orchestration_api/test_data.py orchestration_api/test_workers.py`.
