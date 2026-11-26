# Standalone Orchestration

Run this repository on the runtime host with its declared Python dependencies and supported
container/GPU runtime. Configure model paths and hardware on that host. No Gateway or Benchmarking
service is required. Local executable profiles are separate from imported evidence artifacts.

The supported direct API is `python shared/scripts/rig/rig.py api --request JSON` (or installed
`rig api`). Every request uses `schema: "rig-lifecycle/1"` and an `operation`. Mutations also
require an immutable `request_id`. Query `capabilities` for the operation list.

1. `profile.register`: `profile_id` and `spec` containing `model`, an existing absolute
   `gguf_path`, positive `ctx_size`, and optional runtime settings such as `batch_size`,
   `tensor_split` and `extra_args`. The profile need not exist in any imported registry.
2. `controller.claim`: explicit `controller` identity. Competing active controllers are refused.
3. `load`: that `controller`, `profile_id`, and distinct nonnegative `gpu_ids`. The owner checks
   current admission, obtains a lease and launches through the existing runtime guard/readiness path.
4. `observe`: current owner status. `stop` unloads one `profile_id` and releases its owned lease.
5. `controller.heartbeat` and `lease.renew` maintain authority during long work. Local load leases
   use the existing `interactive` category with the actual controller in owner/provenance fields.

Local profiles and operation receipts default to `brain/hw/local-api`; `RIG_LOCAL_STATE` can
select their storage. Existing rig hardware/state ownership is unchanged. Registration is
immutable: use a new profile ID for changed settings. Interrupted requests return `uncertain`
under the same request ID; inspect actual owner state before issuing a new operation. A failed
load only releases its lease if no partially launched profile remains.

`lease.acquire/preview/renew/release` and `claim.add/renew/remove` support explicit clients.
`power.set` requires a GPU lease owned by the controller and watts within the observed hardware
range. `power.restore` uses its durable `power_session_id` to restore the previous per-card
setting without overwriting another card's override. The process needs permission to update
`RIG_POWER_OVERRIDE` and restart the configured rig power service; missing permission fails.
Retain restoration records after interrupted power operations and reconcile before further changes.

## Hardware telemetry

The rig-local guard runs independently of Gateway and owns power, thermal, memory and
load-admission policy. `rig observe` publishes `guard` with schema
`rig-hardware-telemetry/1`, `owner: orchestration`, `level` (`ok`, `warn`,
`critical`, or `unknown`), `loads_admitted`, `why`, `sampled_at`, `fresh`,
and current aggregate power, temperature, available memory, and per-GPU readings. `sampled_at` is the guard's own
sample time, not the time a controller forwarded the response. A stopped or stale
guard reports `unknown`, `fresh: false`, and no sample time. The standalone
`rig observe` CLI and read-only `observe` API let local clients ask directly;
Gateway may relay this same owner decision to remote consumers. Consumers must
treat cached responses and old samples as unavailable and must not recalculate
the owner's thresholds.

## Versioned CLI transport

`rig wire --request JSON` accepts `rig-cli/1`, a `request_id` and an `arguments` array for
supported rig subcommands. It does not run arbitrary shell commands. The envelope returns the
same arguments and ID, exact `exit_code`, stdout `output`, and bounded diagnostics. Mutating
requests are recorded before dispatch; identical retries return the stored result, and an
interrupted dispatch stays uncertain. Read-only calls do not create receipt state. Unknown
versions and unsupported commands are refused. Long-running `plan run` is not a remote operation.
Gateway consumes this optional transport; standalone callers may continue using `rig api`.

Validation: `.venv/bin/python -m pytest -q shared/scripts/rig/tests` covers local profile
registration/load/stop with fake hardware and real lease/controller stores, admission conflicts,
failure cleanup, uncertain replay, power restoration, wire versions and inspection isolation.
Real NVIDIA loading and power experiments are separate site acceptance, not implied by fake tests.
Measurement drivers moved to Benchmarking; runtime readiness/admission tests remain here.

## Local decision service

The [control service and CLI](CONTROL_PLANE.md) own the Manager extracted from Gateway,
including offers, fit, placement, admission, leases, shared-chat allocation and recovery.
Gateway is optional presentation and request transport. Its connection is never required
for Orchestration CLI use. See the guide for authentication and controller migration.
