# Historical document before the runtime-authority extraction

Frozen pre-extraction behavior. Current ownership and setup: [control plane](../CONTROL_PLANE.md).

# Benchmarking to Manager model library handoff

Benchmarking owns model experiments, receipts, `model_registry.json`,
`model_roles.json`, and `load_recipes.json`. The unified entry point is
[`MODEL_LOADING_AND_CHAT_PLACEMENT.md`](https://github.com/bigbertharig/benchmarking/blob/master/MODEL_LOADING_AND_CHAT_PLACEMENT.md).
It validates its model library with:

```bash
cd <pi-shared-root>/plans/shoulders/benchmarking
python3 scripts/active/model_library.py export
python3 scripts/active/model_library.py check
```

On the Pi, Manager asks what changed and imports it with:

```bash
cd <gateway-checkout>
.venv/bin/python tools/sync_model_library.py --dry-run  # report only
.venv/bin/python tools/sync_model_library.py            # validate and import
.venv/bin/python tools/sync_model_library.py --check    # 0 current, 1 pending, 2 invalid
```

`BENCHMARKING_ROOT` selects another checkout. The import rejects an invalid or
unexported benchmarking library before changing Manager files. It reports
added, removed, and changed models, roles, passive load recipes, and published
offers, plus a SHA256 fingerprint of the registry, roles, recipes, and benchmarking
converter. The recipe import rejects unknown models, unknown schema versions,
unsupported statuses, command-like procedure names, invalid placement tiers,
and missing evidence links before changing Manager files. It prepares both
outputs before replacing either and restores the previous pair if a write fails.

The importer updates two Manager-owned files:

| File | Use |
| --- | --- |
| `model_catalog.json` | Resolved settings for every model and named profile, role mappings, runtime image, passive load recipes, and dashboard summary. Manager plans and loads models from this copy. |
| `remote_profiles.json` | The subset Manager offers to remote clients. The offer list lives in `tools/generate_profiles.py`; a registry profile alone does not publish a new route. |

Manager reads its imported copies at runtime; the rig does not need to read the
benchmarking checkout to answer a load request. Ordinary profiles give Manager
requirements such as context, batch, engine pack, and llama arguments. Manager
still chooses cards and context fit for the load. Where a model has a measured
`loading_stage.context_limit`, Manager caps both llama and Strata fit at that
limit even if the GGUF advertises a larger trained window.

## Testing and runtime eligibility

Benchmarking marks candidate models with `in_testing: true`. They remain in the
complete catalog so orchestration and benchmark lanes can request them with
`use=benchmark`. They are excluded from ordinary remote profiles and from
`GET /v1/manager/model-offers`. A chat or worker load by model ID also rejects
models that are in testing or not active. This separates permission to run a
controlled test from permission to advertise a model for routine work.

Reviewed `model_roles.json` entries and specific registry profiles carry
`good_for` category IDs. The importer carries these into Manager's catalog.
`GET /v1/manager/model-offers` returns approved roles and their categories
without launch commands. A published profile tagged `chatbot` appears in the
web chat model list; removing that tag removes it after the next import.
The child-agent route resolver maps its `coding` and `research` requests to
reviewed role IDs in Manager policy; it finds matching published profiles from
the imported catalog. It no longer keeps a second list of model IDs. Moving
orchestration's direct model selection and launches behind the same runtime
contract requires a separate controlled migration of its clients and leases.

## Runtime coordination boundary

Benchmarking and orchestration may exercise `in_testing` models in an explicit
benchmark run. They must not become routine harness or web-chat offers through
that test path. For normal work, harness and orchestration should consume the
same approved capability list and request a prepared route from the shared
runtime scheduler. A harness may call orchestration directly if orchestration
uses that same lease and readiness authority; Gateway remains the inference
route and API surface for clients that need it.

The present `agent-routes/resolve` covers `coding` and `research` child requests
on GPUs separate from a parent. It is not yet a generic scheduler request for
root agents or orchestration workers. Orchestration still reads the benchmarking
roles directly and launches its own brain/worker runtimes. Its launcher audits
lease identity but does not yet refuse a load without a valid lease. Those
clients and the rig lease gate must migrate together before there is one
runtime loading authority. The offer list is discovery, not an admission or
ready receipt.

## Passive recipe contract, schema v1

`load_recipes.json` is a dictionary of stable recipe IDs. Each entry names a
registered model, `research` or `validated` status, a symbolic procedure ID,
placement components (`weights`, `experts`, `lookup`, `chat_state`) mapped to
tiers (`gpu0`, `gpu_workers`, `host_ram`, `nvme`), and existing Markdown evidence
under `results/runtime_experiments/`. The importer carries the complete
document into `model_catalog.json` and reports changed recipe IDs. These
entries are **not executable**: the Manager loader does not select them, the
procedure field is never interpreted as a command, and `manager_ready` is
rejected. This prevents an import from advertising an experimental private
runtime as a working route.

To activate a recipe later, add a named Manager/rig capability with a fixed
launcher and explicit weight, expert, KV, workspace, host-memory, and storage
fit checks. The capability must verify actual placement and served context
after load, reserve every required device, and enforce per-chat admission.
Only then extend the schema to permit a Manager-ready state and map a request
to that vetted procedure. Keep the offer limited to measured Pareto choices:
fastest single chat, balanced concurrent chats, largest validated per-chat
context, and largest validated model fit. Harnesses request model, context,
chat count, and preference, not hardware details. A non-fitting request returns
a typed reason rather than an unrequested fallback.

Experimental CUDA kernels, tensor placement, Strata packs, and per-chat KV
workers start as benchmarking evidence with a runtime artifact and receipt.
Promote a configuration only when the runtime can execute it, the registry
expresses every required setting, the importer preserves it, and Manager knows
how to reserve the needed cards and launch it. The present registry has one
global `runtime_image`; a custom llama build or per-chat device map needs an
explicit runtime capability and Manager launch contract before it can become a
published profile. Do not encode an experimental placement as an ordinary
`tensor_split`: that controls model layer placement, not chat-owned KV.

## Shared-weight chat promotion: Nemotron 74B

### Tiered pool admission

The Strata and shared-KV experiments treat VRAM, host RAM, and storage as
tiers of one serving configuration. Benchmarking publishes measured placements
and memory curves; Orchestration owns the live ledger and is the only component
that may say a new context fits. Gateway asks for a model and context capacity,
then displays the returned placement and full reservation. Occupied tokens are
telemetry, never free capacity. The dashboard's free worker count comes from
Orchestration's `worker_capacity` observation for the published default child
context, using the same eligibility check as attach. A different requested
context is decided at attach time.

The current production shared-KV service still reserves one whole worker GPU
per attached context. The benchmarked Nemotron three-chat layout shared two
worker cards under a private runtime with Q4 KV and a different weight split;
the current HTTP service uses an F16 chat layout and explicitly rejects a
second context on one GPU. Qwen's dynamic context evidence likewise does not
validate same-card stacking in this HTTP service. Those results establish the
direction, but do not authorize GPU suballocation here. Promotion needs a
matching runtime build, a measured per-context fixed and KV memory curve for
that build, cumulative reservation checks, and restart/recovery testing before
the dashboard can report pooled subslots on one card.

The existing 35B shared-KV path shows the intended shape: orchestration has a
parent `gpu0-main-q4` mode and a dependent `gpu1060-chat-q4` mode. Its rig
controller owns the one weight process, GPU claims, chat attach/detach, and
per-chat turn endpoint. Gateway already exposes ready 35B contexts as separate
routes and can attach a child for agent delegation. This path does **not** yet
make shared chats selectable in the dashboard Chat dropdown.

For Qwen, `subagent_catalog.json` publishes `dynamic_at_attach`, a minimum,
default, maximum, and the measured memory-estimate inputs. Gateway imports those
fields and the manager passes an exact `context_tokens` value through rig control
to context construction. The catalog describes a new context only; it does not
claim that a live cache can grow or migrate. Memory estimates remain scoped to
the model, KV type, and placement that produced their evidence.

Nemotron 3.5 Bolt 74B-A9B also has a separate `single-chat-fast` HTTP mode. Its
original shared-weight multi-chat evidence came from the private b11669 probe.
The tested four-expert-card layout put a 262,144-position F16
main context on GPU0, expert tensors on GPUs1-4, and an attached F16 context
on GPU5. Both chats reached 50K occupied history together; a single GPU5
chat reached 250K separately. These results do not certify two deep 250K
chats. This layout has one main context and one attachable worker context;
adding more workers requires a separately measured placement. Benchmarking's
`nemotron74_expert1060_chat1060_v1` records the original probe and remains
`research`. The gateway requires a distinct validated
`nemotron74_shared_kv_http_v1` serving recipe before offering Chat.

Promotion follows one explicit path:

1. **Benchmarking publishes evidence.** Keep the canonical model ID, exact
   quant and runtime revision, 262K configured cap, measured occupied depths,
   concurrency, placement, speed, and failures in the recipe. Mark the
   serving recipe validated only after the serving tests below; model
   status and recipe status are separate decisions.
2. **Orchestration owns the executable parent and attach modes.** They must
   use one weight load, claim every occupied GPU, select a worker by measured
   free VRAM, report the actual model, mode, context cap, GPU IDs, and ready
   route, and reject a missing parent or conflicting reservation. A 35B or
   Nemotron service may run, never both on the same cards. The gateway must
   call a stable rig/controller command, not a benchmark script or a private
   source path.
3. **Gateway owns admission and conversation mapping.** The first dashboard
   conversation binds to the unused main context; later conversations attach
   available measured worker contexts without reloading weights. The current
   Nemotron placement has one worker; the 35B placement has up to five. The dashboard
   conversation store records the service instance and chat ID, checks readiness
   before each turn, and detaches a worker when its conversation is closed.
   The main context remains until the parent service stops. A service
   restart loses live KV until checkpoint/restore exists; the dashboard must
   report that state loss and never silently continue a saved conversation as
   though its cache survived.
4. **Offer only after an end-to-end serving receipt.** The gateway adds the
   dashboard choice only when its imported catalog marks the canonical model
   active and the matching load recipe validated; an imported `research` recipe
   remains hidden. Cold-start the parent
   from the gateway, send a real turn, attach a second chat and verify that
   the weight process did not restart, interleave distinct turns, check the
   served context and GPU claims, and detach a worker. The 2026-11-16
   Nemotron receipt covers these checks and about 37K occupied tokens in
   each chat. Service restart state loss is rejected by instance ID; two
   near-250K chats and long-context answer quality remain future tests.
   The dropdown is
   discovery; the live admission result still decides whether another chat
   fits now.

Gateway offers name the canonical model and service mode while orchestration
retains all tensor maps, GPU placement, and runtime build details. The
validated Nemotron HTTP recipe is separate from its earlier research probe.
