# Portability and accelerator backends

Orchestration is the authority for physical resources regardless of accelerator vendor. Gateway may
request a capability, profile, context, or scheduling outcome, but Orchestration discovers the actual
machine, admits or refuses the request, starts the runtime, and reports observed state. Benchmarking
supplies optional measured evidence; Harness uses a configured inference provider. The active
Harness loop currently requires a Gateway-ready route; direct-provider execution is Planned.

## Current backend

Orchestration is NVIDIA/CUDA-focused today. `rig.py`, fit and observation logic, power/thermal controls,
load-log parsing, `llama_runtime`, container preparation, health checks, and several experiments use
`nvidia-smi`, CUDA device naming, CUDA memory overheads, or NVIDIA container behavior. GPU count is discovered, so each
NVIDIA layout is calibrated without hard-coding Big Bertha's card indices. AMD/ROCm and Apple
Silicon/Metal integrations are planned.

## Architecture

Keep one repository and split the code into a backend-neutral control plane plus versioned accelerator
and runtime adapters:

```text
Authorized controller or local client request
    -> Orchestration control plane
       inventory / capabilities / admission / reservations / lifecycle / receipts
           -> accelerator backend: nvidia | amd | apple
           -> runtime adapter: llama.cpp | strata | another reviewed runtime
```

The control plane owns locks, leases, claims, request IDs, controller fencing, recovery, state
transitions, and normalized receipts. An accelerator backend owns vendor discovery, device identity,
memory/topology observations, telemetry, power controls, and the vendor-specific part of fit. A runtime
adapter owns command construction, readiness, stop/reconcile behavior, and parsing of runtime-specific
observations. Neither adapter may grant itself resources or publish a Gateway model offer.

A backend capability response should be explicit rather than forcing every vendor into CUDA concepts.
At minimum it needs a schema/backend version, stable device IDs, vendor and API, memory domains, usable
capacity, topology/peer capabilities, telemetry availability, supported runtimes, placement
constraints, and feature flags. Discrete VRAM, ROCm-visible devices, and Apple unified memory are
different resource models; an absent capability is reported as unsupported, not guessed or filled with
a CUDA default.

Site configuration selects installed backends, runtime adapters and executable local profiles.
Benchmarking evidence is an optional input, kept separate from explicit local configuration.
A standalone user can register an unbenchmarked model through the Planned local workflow.
Model recipes declare required
capabilities and retain the Benchmarking evidence environment. The control plane asks an adapter for a
preview and then rechecks admission under its own lock before allocation. Receipts record adapter name
and version, inventory identity, recipe/evidence identity, requested resources, observed placement, and
the reason for any refusal.

## Implementation order

1. Freeze normalized inventory, capability, admission, lifecycle, and receipt fixtures around the
   current NVIDIA behavior.
2. Extract an NVIDIA backend with no behavior change and run the existing rig tests against both the
   adapter and end-to-end command surface.
3. Remove vendor assumptions from Gateway requests and Benchmarking's portable recipe fields. Preserve
   NVIDIA-specific evidence as typed backend data rather than deleting it.
4. Add fake AMD and Apple backends first. They must demonstrate unsupported capabilities, unified versus
   discrete memory, admission refusal, lifecycle recovery, and deterministic receipts without vendor
   hardware.
5. Add real AMD/ROCm and Apple Silicon/Metal implementations only with host-specific integration tests,
   fresh measurements, and clearly labelled experimental recipes.

Create separate vendor repositories only if an adapter later needs independent licensing, release
cadence, dependency isolation, or maintainership. The versioned backend contract must allow that package
boundary, but three forks of the control plane would duplicate admission and recovery logic and invite
different answers to the same hardware question.

## Porting another installation

Supply site-private hosts, service identities, model paths, mounts, credentials, and safety limits.
Run inventory and synthetic hardware surveys, then remeasure every offered model/runtime recipe. Do not
copy Big Bertha's CUDA margins, power rules, topology choices, or successful placements as facts about
another machine. The cross-repository ownership and artifact contract is maintained in Benchmarking's
[model-library contract](https://github.com/bigbertharig/benchmarking/blob/master/MODEL_LIBRARY_CONTRACT.md).

## Standalone status (2026-11-24)

Running a local model without Gateway is a required product workflow. The current
quickstart delegates model loading to Gateway, while `rig start remote` accepts a
prepared spec from a controller. A documented local profile-to-admission-to-launch
workflow is still needed; it must preserve controller fencing, leases, and guard
checks. Standalone acceptance has not passed. See the [implementation status](../README.md#implementation-status).
