# Public read-only inspection CLI

Implemented 2026-11-24. These operations use this Orchestration installation's code and
site configuration. They require no Gateway service or Benchmarking checkout/model export.
They do not acquire leases, launch models, repair power limits or create hardware state.
Standalone model registration/loading is separate pending work.

```bash
rig inspect power --gpus 0,2
rig inspect model-path --path /models/example.gguf
```

Use the installed `rig` entry point or this repository's Python environment with
`shared/scripts/rig/rig.py`. This adds no network listener. A remote deployment supplies
its own authenticated command transport. Requests and replies must preserve argument
boundaries; the default Benchmarking client executes a local argv array without a shell.

## Response contract

Every completed inspection emits one JSON object. Exit 0 means `ok: true`; exit 1 means
`ok: false`. Argparse usage errors use exit 2. Consumers reject nonzero exits, malformed
JSON, unknown schema versions and incomplete successful evidence. `observed_at` is UTC.
Unknown additive fields can be ignored; incompatible fields require a new major schema.

| Operation | Schema | Success fields |
|---|---|---|
| `power` | `rig-power-check/1` | `gpu_ids` sorted; one ordered `checks` entry per GPU with `gpu_id`, finite positive `actual_w`/`planned_w` and `matches: true`. Absolute difference must be below 0.5 W. |
| `model-path` | `rig-model-path/1` | Exact `requested_path`, absolute `resolved_path` naming an existing file, and `source: site_path_resolution`. |

Failure objects include `schema`, `observed_at`, `ok: false`, `kind` and `error`.
Kinds are `invalid` (arguments/configuration/data), `not_found` (requested GPU/file),
`unavailable` (driver/telemetry/OS failure), and `power_mismatch` (observed cap off plan).
A power mismatch also includes the observed checks. Missing or invalid telemetry fails;
this contract never turns an unchecked power condition into success or repairs it.
GPU IDs must be distinct nonnegative integers and every requested GPU must be observed.

Power configuration defaults to this checkout's `shared/brain/hw/power.json` or the
explicit `RIG_HW_DIR`; the existing `RIG_POWER_OVERRIDE` behavior remains. Path resolution
uses `RIG_SITE_CONFIG` (default `shared/rig_site_config.json`) and the runtime's existing
archive/hotset resolver. Relative requests and missing resolved files fail. Path resolution
proves file availability only, not model identity, GGUF validity, readiness or capacity.
All live model loads still require admission and lifecycle checks.

## Verification and rollout

The rig suite passed 113 tests on 2026-11-24, including malformed/missing power evidence,
identity and path failures, plus a CLI installation copied without sibling repositories,
model snapshots or hardware state. The real Benchmarking consumer and Orchestration CLI
completed an existing-path request and rejected a missing file using separate environments.
This was file/contract verification; no GPU load or live power change was performed.

Run `.venv/bin/python -m pytest -q shared/scripts/rig/tests` after changes to this contract.
Both producer and consumer must be updated before using the migrated Benchmarking helpers
on a runtime host. The separate GPU-host checkout has unrelated local edits and has not
been overwritten or redeployed by this change.
