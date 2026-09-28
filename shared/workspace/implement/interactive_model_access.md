# Interactive Model Access

Status: initial Pi-side API implemented on the
`feature/interactive-model-access` branch.

## Purpose

Expose a narrow local contract for `llm-gateway` without moving placement,
runtime startup, health checks, or GPU ownership into the gateway.

The service runs on the operator Pi and binds to `127.0.0.1:8790`. The gateway
runs on the same Pi, so this control API does not need another LAN listener.

```text
VS Code -> llm-gateway :8080 -> model_access_api :8790
                                  |
                                  `-> load-gpu-runtime -> GPU rig
```

## Contract

```text
GET    /health
GET    /v1/local/models
GET    /v1/local/models/{model_id}
POST   /v1/local/models/{model_id}/acquire
POST   /v1/local/sessions/{session_id}/renew
DELETE /v1/local/sessions/{session_id}
```

All `/v1` routes require the bearer token stored in
`ORCHESTRATOR_API_KEY`. Health is unauthenticated for local service checks.

`shared/agents/model_access.json` is the explicit allowlist presented to the
gateway. A model being present in the general catalog does not automatically
make it available for interactive use. This prevents unqualified targets and
brain-scale profiles from being offered to worker GPUs.

An acquire request delegates to `scripts/load-gpu-runtime`, which continues to
own:

- model lookup;
- single or split placement;
- preflight and conflict detection;
- canonical interactive GPU lease acquisition;
- runtime startup and readiness verification.

The API can issue multiple request sessions against one model allocation. It
releases the underlying lease after the last session ends or expires. Releasing
the lease does not stop the healthy runtime; a future request can cheaply lease
and reuse it.

## Run

From the Pi checkout:

```bash
ORCHESTRATOR_API_KEY=replace-me \
python3 scripts/model_access_api.py \
  --host 127.0.0.1 \
  --port 8790 \
  --shared-root /media/bryan/shared \
  --rig-host 10.0.0.3
```

The first integration validation should use a worker model already known to
load successfully. Do not start with a currently unsupported Gemma 4 target.

## Remaining Work

- install both Pi services under systemd after live validation;
- add explicit metrics/request logging once the VS Code workflow is stable;
- decide whether local embeddings or the Responses API are actually needed;
- add persistent session recovery only if service restarts prove frequent.
