# Dashboard package (orchestration data and controls)

The dashboard is served by llm-gateway on the Pi: `http://127.0.0.1:8080/dashboard` (from another
machine: `ssh -N -L 8080:127.0.0.1:8080 pi`). See llm-gateway `docs/DASHBOARD.md`. Moved 2026-10-20;
the old standalone server (`server.py`, port 8787) and its pages are gone.

This package keeps the orchestration part, which the gateway imports:

| Module | Owns |
|---|---|
| `data.py` | `summarize(shared_path, config)`: workers, brain GPUs, active batches and chains, task lanes, alerts |
| `controls.py` | `Controls(shared_path, config)`: `options`, `start_plan`, `resume_plan`, `kill_plan`, `kill_all_active`, `batch_outputs`, `cleanup_stale` (task files only), `requeue_worker` |
| `workers.py` | worker and brain heartbeats, thermal cause, GPU telemetry for worker rows |
| `plans.py` | plan discovery, starters and inputs, batch history and outputs |
| `alerts.py`, `chains.py` | batch failure and completion alerts, batch chain views |
| `utils.py` | constants, file and time helpers, sanitization |

Anything that acts on GPUs or runtimes (unload, kill a card, reservations, force load) is the rig
manager's job in llm-gateway (`docs/MANAGER.md`, "Reservations"). The old server's `clear_runtime`,
`reset_gpu`, `return_default` and remote runtime cleanup were dropped for that reason: the dashboard's
Rig tab kills or unloads through the manager, and `requeue_worker` returns that worker's tasks to the
queue.

Tests: `python -m pytest -q dashboard/test_data.py dashboard/test_workers.py` from `scripts/`.
