# Local model control plane

Orchestration owns local-model decisions and runtime execution. Its control service owns
profile resolution, offers, model-fit policy, placement, scheduling, admission, leases,
shared-context assignment, runtime recovery and controller heartbeats. The hardware `rig`
CLI enforces live constraints and owns hardware state. Both layers belong to this repository.
Gateway is an optional dashboard and request client. Orchestration alone has a CLI/API.

## Install and run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e . -r requirements-dev.txt
.venv/bin/rig-control serve --profiles /path/to/remote_profiles.json --state-root /path/to/control-state
```

Configure `ORCHESTRATION_CONTROL_TOKEN` in the protected process environment. The default
bind is `127.0.0.1:8092`; use a protected private connection or TLS ingress for a remote client.
`ORCHESTRATION_RIG_COMMAND` selects this product's installed `rig` command, locally or over
an explicitly configured SSH command. `MODEL_CATALOG` names optional reviewed model evidence.
`ORCHESTRATION_MODEL_PREFIX` defaults to `gpu-rig`. Service runtime state defaults to
`~/.local/state/llm-orchestration/control`; observations default to the same product's state
root. One service process holds the state-directory lock. Run one active controller per rig.

The optional checked-in `model_catalog.json` and `remote_profiles.json` are imported snapshots.
To refresh them, run `BENCHMARKING_ROOT=/path/to/export python3 tools/generate_profiles.py`.
This is an explicit import, never a required sibling checkout or service-start mutation.
Copy/reference the resulting files in the service's explicit configuration. Missing optional
profiles/evidence leaves capabilities and hardware status accessible, with no fabricated offers.
Direct local profile registration/loading through `rig api` remains available without evidence.

## Request protocol

`POST /v1/control`, authenticated with the service bearer token, accepts:

```json
{"schema":"orchestration-control/1","request_id":"operator-001",
 "operation":"manager.load_model","arguments":{"args":["local-model"],"kwargs":{"use":"chat"}}}
```

Responses repeat the schema and request ID, with `ok` and either `result` or a typed `error`.
The CLI is the same public client:

```bash
rig-control request capabilities
rig-control request manager.status
rig-control request manager.load_model --request-id operator-001 --arguments '{"args":["local-model"],"kwargs":{"use":"chat"}}'
```

Supported operation groups: `snapshot`, `catalog.role`, `rig.status`, `route.prepare`,
`shared.turn`, `chat.prepare/release/switch/reset`, and the allowlisted `manager.*` methods.
`manager.http` routes the established `/v1/manager/...` request shapes inside the owner and
returns their HTTP status and body. It does not expose a host-shell execution operation.
Manager run commands retain the existing explicit operator run-command semantics.

Writes are durably marked before dispatch. Identical completed IDs return the same receipt;
changed arguments under the same ID are refused. Running/interrupted writes return
`reconciliation_required`. Gateway cannot replay or reconstruct them. Service workers continue
when a client disconnects. The operator inspects owner status and reconciles an uncertain
operation before creating a new request. Ready-only checks verify served identity, context
and port, and never trigger loading.

Shared chat assignments are persisted in the owner's SQLite store. Gateway records only the
returned conversation binding. The owner allocates the main/attached context, prevents stale
unloads, and checks explicit reset intent. Conversation activity remains a client-side guard;
hardware admission and controller fencing remain authoritative at the owner.

## Verification and migration

Run `.venv/bin/python -m pytest -q tests/control shared/scripts/rig/tests`. The control tests
cover the migrated fit/admission/recovery behavior, shared-model policy, request replay,
version/authentication refusals and independent service startup. The product's control wheel
can be installed without Gateway. Hardware execution additionally requires the documented
rig installation on its selected host.

For an existing deployment, stop the old Gateway Manager, reconcile its outstanding work
and `llm-gateway` leases, then start this service and explicitly hand off controller ownership.
New interactive Manager leases are owned by `orchestration`. Do not run the old and new
controllers concurrently. Connect Gateway only after the owner is healthy. Live NVIDIA
loading, power restoration and deployment remain separate checks; preserve host-local edits.

The pre-extraction detailed Manager and importer documents are retained in `docs/history/`
for historical receipts. Current setup and ownership are defined here.

November 25 source validation: 149 control-plane tests and 118 hardware tests passed
(267 total). The final eight HTTP/shared-context checks were rerun after boundary fixes.
Separate installed environments passed CLI independence and the real HTTP Gateway connection;
the combined installed shared-chat scenario passed. These tests use fake hardware.


November 25 live acceptance: controller ownership moved from the deployed Gateway
Manager to the independent Orchestration service. The GPU producer was upgraded to
`rig-cli/1` after preserving its local edits and checking the merge. A real Ministral
3B load acquired an `orchestration` lease, served inference through the request client,
survived a Gateway restart with the same controller process, then unloaded cleanly.
The rig returned to idle with no leases; observed GPU power limits matched preflight.
This verifies the controller handoff and ordinary load/unload path; shared-KV and
power-changing benchmark session recovery retain their separate acceptance gates.
Keep site units, connection secrets, rollback copies and live receipts outside Git.
