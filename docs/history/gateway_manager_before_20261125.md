# Historical document before the runtime-authority extraction

Frozen pre-extraction behavior. Current ownership and setup: [control plane](../CONTROL_PLANE.md).

# Rig manager

The Manager coordinates model requests, leases, loads, and work plans across the Gateway controller and rig executor. Gateway sends fit requirements or an explicit operator card pin; the rig selects cards from its live state and returns the admitted lease or a specific refusal. Gateway checks the model offer and coordinates load policy; the rig computes detailed model/context fit. The rig owns durable state and enforces hardware rules. Start with the [dashboard guide](DASHBOARD.md) for operator controls and [design notes](DESIGN_NOTES.md) for the resource model. The operator CLI is `tools/rigctl.py`.

Implementation status and detailed flows follow.

Status 2026-10-24: Phase A built (leases, claims, placement, observed status, detached rig loads,
gateway cold loads through the manager); lease record moved onto the rig; rig guard built (gates
every load, frees RAM at critical); cache policy per use; Phase C built (lanes and campaigns load through
the manager); isolated rig modes (one mode at a time: a benchmark load runs in `bench`, chat and worker
loads in `remote`). Next: B, then D (see "Pick up here").

Design knowledge the manager has to balance (host RAM budget, cache policy per use, layouts, more
than one rig): [DESIGN_NOTES.md](DESIGN_NOTES.md).

## Two layers

| | Gateway host: controller management (this repo) | Rig: hardware management (`llm_orchestration/shared/scripts/rig/`) |
|---|---|---|
| Decides | model offer, work owner and waiting policy | model/context fit, live card placement, lease GPUs and ports |
| Keeps | a recovery mirror of the rig report, not the authoritative lease record | the lease and claim record, its own mode state, job state |
| Does | sends requirements or an operator pin, observes, reconciles, reports | selects cards, starts admitted runtimes, reports `rig observe` |
| Enforces | model and user policy | atomic lease/claim admission, GPU health, port use and runtime rules |
| Disconnected | reports the rig unreachable | keeps going: finishes running and queued work; new loads wait for new leases |

Gateway never micromanages: it does not sequence loads, poll temperatures or retry around the rig's
rules. Monitoring and safety live on the rig, next to the hardware. The rig is meant to run fully
offline: its state is rig-local, observations and heartbeats exist for the Pi's benefit, and (to
build) results fall back to local storage when the shared drive is gone and sync on reconnect.

## Rules

1. **The rig keeps the record; controllers keep no state.** Leases and claims live on the rig
   (`rig lease ...`, `rig claim ...`, `hw_state.py`, rig-local `shared/brain/hw/gpus/`), written only
   by the rig under the lease store's own lock, in the orchestration schema (`gpu_leases.py`). Every
   write is mirrored to the drive (`/mnt/shared/gpus/`) for orchestration workers when it is mounted.
   A controller reads the record from `rig observe` and asks the rig to change it, so the record
   survives a lost network, and another controller (desktop, laptop) can take over without a
   migration.
1a. **One report, one active controller.** The rig also keeps the active-controller record and the
   plan queue (`hw_plan.py`, rig-local `shared/brain/hw/plan.json`). `rig observe` is the single
   report: "now" (GPUs, profiles, leases, guard) and "next" (the plan). The active controller writes
   the plan and heartbeats; the rig refuses plan, lease and claim writes from any other controller.
   A controller takes control when there is none, when the active one's heartbeat is older than 120 s
   (crashed), or by explicit handoff (`rigctl takeover`); otherwise it is read-only (409). A crashed
   or second controller just reads the report and picks up.
1b. **The rig works through the plan by itself** (`rig-plan.service`): one item at a time, in order,
   then waits. Loaded models stay loaded unless an `unload` item says otherwise. With no controller
   connected it finishes the queue. Item kinds today: `load`, `unload` (jobs such as benchmark lanes
   are the next kind). This mirrors the orchestration task queue in a single file.
1c. **One report, two copies (2026-10-18).** The rig's record and plan are also kept by the active
   controller: every observation (the 30 s heartbeat) saves `rig observe` to the controller's own disk
   (`RIG_MIRROR_PATH`, default `~/.local/state/llm-gateway/rig_report.json`). The plan has a `version`
   (bumped on every rig write) and the rig's `boot_id`. Either side recovers from the other:
   - rig reboots: on the first start of a new boot it marks the item that was running `interrupted`
     (with what it was doing) and holds the plan; it boots neutral and runs nothing. The controller's
     reconciler sees the hold, writes its copy back if the rig's version is behind (lost or replaced
     file, `rig plan restore`), reports what was interrupted, cancels any pre-reboot queued model
     load/unload requests, and resumes the executor for fresh on-demand work (`rig plan resume`).
     Interrupted loads are not retried (their caller got the failure); their leases are released as
     unbacked. Thus reconnecting the controller cannot populate an intentionally empty rig boot from
     stale model lifecycle work. A lost network without a reboot is not a hold: the rig finishes what
     it was given.
   - rig unreachable: status answers from the copy (`from_copy`, "RIG UNREACHABLE since ...") and the
     controller logs when the rig went away, when it came back and whether it rebooted (`rig.events`).
   - controller crashes: it restarts with its copy and reads the rig's live report.
   Runs (2026-10-22): long work (a campaign, a lane) started with `rigctl run-start` is a `job` plan item
   and a record in the plan's `runs`, so it is in the report and both copies. After a reboot a run that was
   running is listed as interrupted in the hold, and the controller asks instead of acting: "resume from
   checkpoint?" (`rigctl recover RUN`, the dashboard's Resume) or dismiss. A run that opted in
   (`--auto-resume`) is resumed on its own. A lost network is not an interruption: the run's unit keeps
   going, and the manager's waits keep waiting (see "Runs" below). Work started outside `run-start` (a bare
   `systemd-run`) is still invisible to the report.
2. **No lease, no load.** Every runtime starts under a lease naming its GPUs and port. Every launcher
   already goes through `run_runtime.sh` (brain, workers, split workers, `rig.py`, `run_campaign.py`,
   integration lanes, runtime smoke), so that is where Phase D enforces it.
3. **The port comes from the lease:** 11434 + the lowest GPU index. A lease reserves 11434 + each of
   its GPUs, so two leases never share a port (the worker and benchmark-lane convention already).
4. **The rig admits physical placement.** Gateway sends `gpus_needed` (per card `min_vram_mb`, optional
   `max_vram_mb`) or an explicit operator `gpu_ids` pin. The rig selects the smallest complete fit
   under its admission lock. A card is available only if no lease holds it, nothing runs on it, and
   it is unclaimed (or claimed by the claim the request names). `rig model-fit` uses the same
   live card view for GGUF, context, RAM, and registered Strata placement previews.
5. **Everything is in the pool by default. A claim takes cards out** (`rigctl claim bench --gpus 0`):
   only requests naming the claim get them. The rig refuses a lease on a claimed card that does not
   name the claim.
6. **Status is observed.** Each GPU gets a verdict from leases plus `rig observe`: `free`, `leased`,
   `loading`, `lease_idle`, `stale` (expired lease), `unleased_busy` (something runs without a lease).
7. **Expiry never frees a GPU by itself.** The manager auto-releases only its own leases with no load
   or runtime behind them; other controllers' expired leases are reported, not touched.

## The client process

For a nonbinding current answer, `POST /v1/manager/leases/preview` accepts
`{gpus_needed | gpu_ids, claim?}` and returns the rig's proposed `gpu_ids` and ports or its
specific refusal. It creates no lease. A later acquisition rechecks live state atomically.

```text
1. ask      POST /v1/manager/leases  {controller, owner, run_id, gpus_needed | gpu_ids, claim?, ttl_s, model, purpose}
            -> 201 {lease, port} | 403 wrong owner | 404 unknown claim | 409 held (names holder) | 503 no capacity
2. launch   on the GPUs and port of the lease (Phase C/D: run_runtime.sh --lease <id>)
3. renew    POST /v1/manager/leases/<id>/renew    {owner, ttl_s}, every ttl/3
4. release  POST /v1/manager/leases/<id>/release  {owner}; 409 while a runtime still runs on its GPUs
```

`tools/rigctl.py` wraps all of it (stdlib only; exit codes like `rig`). From the rig:
`ssh pi python3 ~/llm-gateway/tools/rigctl.py ...`. The HTTP API stays on Pi loopback.

## Interactive profiles (Roo)

### Shared child serving and concurrent agents (design, 2026-11-06)

For a local parent, agent-routes/resolve offers a child route on GPUs disjoint from its parent, then may cold-load a published profile. A `cloud/codex` parent has no local GPU reservation, so the same reviewed child policy can use any eligible ready or free local cards. Registry runtimes currently specify parallel: 1, so this path does not establish safe same-server concurrent child execution. Treat same-model multi-agent work as a separate manager capacity policy, not as a harness choice of GPUs or a second model load.

The manager should evaluate three modes for a child request: admit another isolated session into an already-ready model server when it has measured free serving slots; start a separate instance of the same model on disjoint free GPUs when isolation or throughput requires it; or prepare a different eligible model on disjoint GPUs. A shared server keeps one weight load, but each active request needs its own context and KV capacity and may lower throughput. A duplicate instance needs its own lease, port, fit, and ready receipt. The manager chooses or queues/refuses according to current claims, active requests, context per slot, memory, and measured throughput; it never evicts the parent or unrelated work implicitly. Return route, instance/slot admission, served per-request context, readiness, expiry, and a typed busy/load-failure reason. A model preference may constrain the offer, but the harness cannot force a placement. Preserve the five-minute cold-load bound separately from queue and child-turn deadlines.

First measure one ready route with two bounded concurrent requests, including context isolation, per-slot context, cancellation, and latency versus serial requests. Then test parent plus two children sharing that route, saturation and queueing, duplicate same-model instances on disjoint cards, a different-model child, partial GPU occupancy, and a failed load. Assert no lease or request is over-admitted and no parent context leaks into a child. Record throughput and memory before setting a default policy. The harness companion design owns child session prompts, tool grants, result collection, and trace links.

### Capability requests for child agents (2026-11-04)

`POST /v1/manager/agent-routes/resolve` accepts `{"agent_type":"coding|research", "parent_route":"gpu-rig/<id>"}`. The parent route is supplied by its controller; the model-facing delegation tool takes only `agent_type` and `task`. The manager chooses a published profile by its ordered capability policy in `manager/routes.py`. It first offers a ready route on GPUs disjoint from the parent's. If none is ready, it asks the manager to fit, lease, and load a cold profile on available cards. It returns only after the route is observed ready, with route ID, model, context, and GPU IDs. A five-minute preparation deadline returns HTTP 504; no capacity or eligible route returns 503, and a load failure is reported through the manager error response. A timed-out rig plan may continue under manager ownership; operators should inspect status before retrying or unloading.

`POST /v1/manager/local-agent-model/swap` is the explicit operator path for replacing one shared
local-agent model with another. The request names the target candidate and the currently observed
candidate/service instance. It refuses stale identity, unreviewed modes, and active child
reservations; ordinary delegation never evicts a different loaded shared model implicitly.

For a Codex dashboard parent, pass `parent_route: "cloud/codex"`. The manager uses the same reviewed
coding/research role order and readiness checks without requiring a local parent profile or reserving
parent GPUs. The harness MCP adapter exposes this as `delegate_task` and records a child receipt.
This route does not make the child a file-editing agent; its current runtime has no tools.

The harness never chooses a GPU or calls a profile load endpoint. It uses the returned route with `X-Require-Ready: true` and checks the ready roster before and after the child run. The first live control used `chat-27b` on GPU 0 and a cold `coding` request, which the manager placed as `coder-7b` on GPU 1. The native child and parent MCP handoff both returned `42`; receipts are kept with the harness trial results. Gateway contract tests passed 33/33. This is one bounded Pi-harness integration, not yet a six-harness delegation comparison. Manager policy and child model quality still need broader evaluation.

A cold request for `gpu-rig/<id>`: the manager leases cards for the profile (`interactive`, owner
`llm-gateway`), then queues a `load` item in the rig's plan (`rig plan add`) with the GPUs, port,
lease and cache size it chose. The rig's executor runs it (`rig start remote ... --queue`), waiting its
turn for the rig-wide load lock, and records the profile as `loading` then `ready`; the request is
held until then. A failed item's result comes back as the 502 message and the lease is released. The
reconciler heartbeats, renews leases with a load, runtime or queued item behind them, and releases
orphans. `rigctl unload` queues an `unload` item and releases the lease when it is done.
`rigctl status` shows the controller (this one, the active one) and the last plan items.

For an already published route that must stay off a busy card, use `rigctl load PROFILE --gpus 1,2,3`.
The manager fits and leases exactly those cards under the published profile ID, so the gateway ready
roster remains usable by harnesses. The API equivalent is
`POST /v1/manager/profiles/PROFILE/load` with `{"gpu_ids":[1,2,3]}`. An empty, duplicate, or
invalid GPU list is rejected; requesting a different GPU set from a profile that is already ready
returns 409. Without `--gpus`, the existing automatic placement policy applies. This change was
motivated by a 2026-11-02 SearXNG trial: a fit preview on GPU 5 did not constrain a later published
`coder-7b` load, which chose GPU 0. The load was immediately unloaded. The new contract passed
58 manager/gateway regression tests in `docs/receipts/gpu_pinned_profile_2026-11-02.txt`.
The code was staged without restarting the live gateway while GPU 0 was kept available for benchmarks.

## Model loads: any registry model on any cards (2026-10-18)

"Load X on Y cards" has one hardware fit authority:

1. **Gateway selects the offered model and coordinates work.** Its imported Benchmarking catalog
   supplies a quick status/profile check. `active` is available for routine Chat and workers;
   `candidate` is available for explicit benchmarking. Gateway sends the model, use, profile,
   context preference, optional card count or operator pin, and placement policy to
   `rig model-fit --request JSON`. It compares the rig's returned source hash with its local
   Benchmarking projection. Gateway keeps cards for queued work when necessary and asks the rig
   for an alternative placement; it does not calculate a GGUF split or choose replacement cards.
2. **The rig previews and admits hardware.** Its `model-fit` reads the imported library, the GGUF
   header, measured load receipts, live GPU and host-memory state, claims, and leases. It chooses
   the card set, context, split, or registered Strata tier and returns the fit and reason. The
   preview does not reserve resources. A load obtains a lease and the rig rechecks admission under
   its lock. A ready runtime returns its measured context and per-card allocation. A refusal
   returns a typed reason to Gateway. `rig fit-observation` stores completed load evidence for
   the rig's next fit; Gateway retains a copy for Benchmarking.

With no cards named, `smallest_fit` is the default for benchmark and worker use, and
`fastest_fit` is the default for Chat. `context_mode=working` uses the profile context,
`min` uses the profile floor, and `max` searches up to the trained/tested ceiling.
An explicit `ctx` is exact. The rig returns 422 for a layout that does not fit, 503 for
temporarily occupied capacity, and 409 for a conflicting pinned card. Strata placement is
limited to the card IDs validated in Benchmarking's model registry. The model library is
the source of those tested details; the rig scan and observations supply live hardware facts.

**Model requirements versus placement (2026-10-25).** The catalog needs the
GGUF identity and runtime settings (for example micro-batch, KV type,
parallelism, reasoning flags, and any supported context limits). It does
not need a GPU list or a saved layer split for each model. Context is an
input to the fit: full-attention KV use grows roughly per token, sliding
windows stop growing past their window, recurrent state can be fixed, and
weights stay fixed for a chosen partition. Compute buffers, graph
allocation, per-card overhead, and margins are calibrated from load
observations because they are not one perfectly linear context formula.
The rig computes the card set and `tensor_split` for the requested context
and available memory, then validates the actual allocation at load. Compute
is calibrated per card count. Comparable speed observations can inform the
rig's automatic placement. Calls without `ctx` use the selected profile's
working context by default.
The library records `context_bounds.min` and `context_bounds.max` for current models; the maximum
is the GGUF trained limit, not a guarantee of quality or a tested fit. `min` is a deliberate
lower-context request, and `max` is an opt-in capacity search. The rig checks
measured OOM ceilings, card memory, and real allocation at load. An explicit `ctx` remains exact.
The imported `schedule.placement` labels remain in model metadata; rig fit does not use them. For example
`ministral-3:8b` is labeled `single_gpu` there. Its default 16K fit on one
1060 was refused, while `context_mode=min` now plans 8K on that card; `max` can project
the largest fit on a chosen two-card split.

### Context capacity and GPU pools (decision recorded 2026-10-26)

The library's `runtime.ctx_size` is a **working context** for a use or benchmark.
`context_bounds.min` is currently only a **planning floor** (4K or 8K),
not the model's compact baseline. `context_bounds.max` is the GGUF trained
ceiling, not a measured promise that the model fits or recalls correctly at
that depth. Keep these three facts separate in results and API responses.

Plan for the **desired model at the desired context**, using only currently
available cards. An explicit context remains exact; otherwise use the
selected profile's working context.

- **3090 available:** try it alone. If that fit fails, add the fewest
  available 1060s that make a full-GPU split fit. A small model stops at
  the 3090-only step.
- **3090 occupied:** try one available 1060, then add only as many 1060s
  as the requested context requires. A small model may stop at one 1060.
- **No full-GPU fit:** unpinned interactive chat requests with an exact or
  profile working context try a bounded one-GPU + host-RAM placement. The
  largest free GPU is tried first. The planner reserves 4 GiB of available
  host RAM, keeps 2 GiB of GPU headroom, and caps the estimated host
  footprint at 18 GiB under the rig's 20g llama container limit. It keeps the requested context and records
  the numeric GPU layer count. Explicit GPU requests and benchmark loads
  stay GPU-only for reproducibility. If no GPU or admitted host-RAM plan
  fits, a registered Strata pack may use the validated GPU 0 storage tier.
  A Strata-only pack goes directly to that tier. `engine=strata` explicitly
  selects it for paired tests; `engine=auto` preserves the normal order.

Do not silently shrink the requested context. Extra split devices can
slow decode after the target fits. The ordering is a placement preference,
not a universal speed guarantee across models, contexts, layer splits,
and runtime engines. In matched Nemotron 32K runs, 3090 + three, four,
and five 1060s decoded at 24.63, 22.52, and 20.43 tok/s respectively,
versus 6.26 tok/s for 3090 + host RAM. Qwen3.6 35B at 262K decoded
faster on 3090 + one 1060 than on all six. A
storage-backed Strata placement needs its own warm/cold speed and recall
evidence. At the trained ceiling, record the fewest cards or memory
tiers that load and pass depth/recall checks; if it remains unreachable,
report the highest validated context separately.

The intended **minimum deployment profile** is the fewest cards that load the model at a chosen minimum-useful context, not the manager's 4K/8K planning floor. Provisional context targets are 8K for small models, 16K for roughly 30B-class models, and 32K for roughly 70B+ models; task needs and measured attention/KV behavior can justify exceptions. Parameter count is a useful workload proxy but does not directly determine KV memory.

The intended **maximum deployment profile** is the fewest allowed cards that load and pass deep-context recall at the model's trained ceiling. Adding cards beyond that point is not a better maximum and has often slowed split decode. If the trained ceiling cannot be reached, leave the maximum profile unavailable and report the highest validated context separately. A Strata-supported placement may later reach the ceiling using host RAM or storage, but it needs its own context, recall, throughput, and memory-tier receipts before promotion. A giant model whose weights need every card at the minimum-useful context can therefore have only a validated minimum profile. The current registry's `context_bounds.min/max` are a planning floor and trained ceiling, not stored placement profiles; the manager computes splits dynamically, and these deployment-profile semantics are not yet encoded as a separate machine-readable contract.

A model may have two placements for the **same context, batch, GGUF
and runtime options**: on the 3090, or on 1060s when it is occupied.
A separately requested lower-context run is a different capacity choice.
No side-worker model class or saved layer split is needed.

**Implementation status:** `context_mode=min` currently means the 4K/8K
planning floor, and `context_mode=max` currently estimates the largest
fit on the selected cards (all free cards if none are named). For ordinary
interactive chat fits, the manager now enforces the available-3090
branch (3090 alone, then smallest fitting 3090 + 1060 split) or the
occupied-3090 branch (one 1060, then smallest fitting 1060 split). There
is no separate side-worker class. Explicit GPU pinning and current card
availability cover placements that need a particular card. A bounded
host-RAM placement is now selected for unpinned chat loads after all
available full-GPU options fail. The rig verifies the actual GPU layer
count, exact context, and at least 4 GiB available host RAM before ready;
it disables container swap and host prompt cache for this path. A 2026-11-03 Nemotron 32K
trial at the initial 512 MiB GPU margin failed during CUDA compute-buffer
allocation; the margin is now 2 GiB and that placement is not admitted
under the current 18 GiB host budget. The manager now plans registered
Strata packs on GPU 0 with their measured context and host-memory bounds;
the rig owns launch, health checks, unload, and the lease-backed profile.
The Qwen3.8 Q2_0 pack is Strata-only, while Qwen3.6 XL still uses llama.cpp
when it fits unless a caller explicitly requests Strata. The successful Qwen
262K smoke and the failed Nemotron calibration are in
`receipts/ram_spillover_2026-11-03.md`. The benchmarking
`CONTEXT_PROFILE_AUDIT_2026-10-26.md` is a read-only capacity projection;
it is not the measured compact-baseline/full-ceiling matrix. Keep working
contexts stable for benchmark comparability while the placement policy and
capacity tests are added. A known exception is unpinned loads with
`wait_s != 0`: `_collect_cards` first requires a GPU-only fit and can
reject a request before the RAM/Strata fit path runs. Direct no-wait
fits use the cascade above; the wait path needs the same tier-aware
collection logic before it can promise that behavior.

Layers are never left to llama.cpp's `--fit`: given the choice it moves layers to the CPU (on
Ornith-9B it balanced two 1060s by putting the output layer in host RAM, 33/34 layers on GPU). The
estimate was checked on Ornith-1.5-9B over two 1060s: weights 4812 MiB and KV 17.0 KiB per token,
the same as llama.cpp measured.

```text
rigctl fit ornith-1.5:9b [--ctx 100000]         # answer only; manager plans placement
rigctl fit ornith-1.5:9b --cards 2              # working context on a fitting two-card set
rigctl fit ministral-3:8b --gpus 1 --context-mode min   # 8K on one 1060
rigctl fit ministral-3:8b --gpus 1,2 --context-mode max # largest projected fit on that split
rigctl load-model ornith-1.5:9b --gpus 1,3 [--ctx N] [--use benchmark] [--args "--reasoning on"]
rigctl unload ornith-1.5-9b-g1-3                # profile id: model plus its cards
```

HTTP: `POST /v1/manager/models/fit`, `POST /v1/manager/models/load` with
{model, cards | gpu_ids, ctx?, context_mode? (working|min|max), ctx_min?, use?, profile?,
claim?, margin_mb?, extra_args?, engine? (auto|llama|strata)}. `ctx` and a non-working
`context_mode` are mutually exclusive. Strata loads use the rig's pinned control and prepared pack;
the two size fields in the catalog sum to source-pack disk bytes, not resident RAM/VRAM or the
separately prepared expert pack. A ready managed Strata load appears in `/v1/models` and
`/v1/rig/ready-roster`; its gateway model id is `gpu-rig/<profile id>`.
The lease is the manager's (owner llm-gateway; controller by use: benchmarks, interactive, plans), so
the reconciler renews it while the rig profile lives and `rigctl unload` releases it.

Card size has two meanings. Placement plans on empty cards (total memory: what a set of cards can hold
once its current work frees, as the collecting wait uses it); a fit for an immediate load uses free memory
(total minus what is loaded now). Later: use the active size to stack several small jobs on one card.

**Models come from the gateway's own catalog (2026-10-25).** The manager never reads benchmarking at
runtime. Run `tools/sync_model_library.py --dry-run` to see what benchmarking learned and
`tools/sync_model_library.py` to validate and import it (see [MODEL_LIBRARY_HANDOFF.md](MODEL_LIBRARY_HANDOFF.md)).
The importer uses `tools/generate_profiles.py` to transform the library into `model_catalog.json`
(every model, its resolved profiles, the roles, the runtime image, the dashboard's model summary) and
`remote_profiles.json` (the offer); `manager/models.py` resolves from the catalog. Re-run the importer to take
in what benchmarking has learned. A published profile names no cards: `load_profile` plans it like any model
load (the largest context that fits, at least the profile's `ctx_size`) under the profile's own id. When the
free cards cannot hold it but an idle rig could, the answer is 503 (busy, retry), not 422 (won't fit).

### Chat context and capacity offers (2026-11-06)

The official published profile offer is `remote_profiles.json`, generated from the model registry by `tools/generate_profiles.py`. Both the manager and basic Chat read that same offer; `dashboard_chat: true` marks the small operator-facing menu. Chat contains no model-ID allowlist. A new menu model is added in the gateway's offer definition and regenerated, without changing Chat code. The manager still validates the chosen model and its fit at request time.


For a cold published **chat** profile, the manager now plans the largest context that fits its chosen placement, with the published profile context as a minimum. The chat placement still tries the fastest available single card first and adds cards only when that card cannot meet the minimum. Thus "max" means the measured/estimated fit on the selected card set, capped by the model's trained ceiling and failed-load evidence; it does not mean consuming every free GPU to maximize the number. A ready profile keeps the context it was loaded with. The ready roster's `context_length` is the served value, not the trained ceiling. A trained ceiling is not a quality guarantee at long prompts.

Current basic Chat busy behavior is conservative: `GET /v1/manager/chat/availability` reports active jobs, plans, profiles, claims, leases, runtimes, GPU memory use, and a closed load guard. `prepare_chat_profile` checks the same fresh rig observation before any cold load and returns HTTP 409 `rig busy` without starting a model when other work is present. An already-ready selected chat profile may continue if it is the rig's only work. The basic Chat tab starts preparation when New conversation opens; a resumed cold conversation prepares on its next Send. The general gateway inference route and operator manager load APIs still use the existing placement policy and can automatically choose free 1060s when the 3090 is occupied; the basic Chat guard does not change those APIs. The guard is a first signal, not a reservation across concurrent clients: the manager's existing lease and fit checks still protect the final placement.

Future capacity-offer contract: before a cold Chat load, provide a read-only, versioned plan containing the requested profile and separate options: (1) load now on the preferred fast placement, (2) load now on an available slower split, (3) wait for a named busy placement, or (4) explicitly choose another model. For each option return the projected GPUs, context, context limit reason, load time estimate and measured prefill/decode speeds where evidence exists, the busy holder/claim without private details, and a short expiry. Do not describe a speed estimate as measured when it is not. A placement that does not reach the profile minimum is not an automatic fallback. A different model or reduced context requires explicit selection. The manager rechecks fit, leases, and claims when the operator accepts an offer; a stale offer returns a revised set rather than preempting another job. A wait is a bounded manager claim with cancellation and a clear timeout, not a held browser request. The manager owns the request negotiation: it returns the allowed options, labels, ranking, context and placement facts, and the next action for each option. Chat submits only a published model ID and renders those options generically; it makes no GPU, context, fallback, or wait decision. The rig remains the authority on readiness.

Measured local control on 2026-11-06: a cold `harness-gemma4-e4b-64k` Chat request was planned on GPU 0 at 131,072 context (the registered ceiling), reached ready, answered `OK`, and was unloaded. The rig plan item spent 252 seconds from start to done; the full turn spent 257.7 seconds. The ready roster then returned empty. This confirms the maximum-context path works, and shows why offer estimates should include load latency. It does not establish that every model's trained ceiling loads or answers well.

Test the offer contract with a free 3090, a busy 3090 and free 1060s, partially occupied/claimed 1060s, all cards busy, a ready profile, a load that fails after acceptance, and a placement that changes before acceptance. Assert that no preview loads or evicts a model, accepted loads return the actual ready context and GPUs, and declines leave the rig unchanged. Compare at least one measured 3090 versus 1060 split so the user sees a concrete speed tradeoff.

**Learning from loads (2026-10-25).** The manager keeps its own history in this repo,
`lessons/load_observations.jsonl` (`manager/observations.py`; `OBSERVATIONS_PATH` overrides), committed, so
it carries across sessions and machines; it is also the gateway's lessons file for benchmarking to read
(`lessons/README.md`). A `load` row has the model, GGUF, micro-batch, use, the cards by UUID with link and
root port (the rig's `gpu_scan`), the estimate per card, what llama.cpp allocated per card (the rig's `fit`),
memory in use per card once serving, and the load time; a `load_failed` row has the same facts, the error and
whether it ran out of memory. Started 2026-10-25 from the loads in the benchmarking capacity records (marked
`from_record`). The fit plans from it, each load counted once (`observations.history`):

- compute buffer per card (`fit.compute_mb`): measured loads for the same GGUF, micro-batch, and
  card count, padded by 10% and 64 MiB. A measured buffer is never discarded when it exceeds the
  formula. Unmeasured combinations use the formula until a matching load is recorded.
- overhead per card (`fit.overhead_mb`): the highest observed serving memory beyond llama.cpp's
  reported buffers for a card of that size, plus 64 MiB; 150 MiB until measured.
- growth margin (`fit.growth_margin_mb`): four times observed full-depth memory growth, with a
  128 MiB minimum, only for the same GGUF and card-size layout. Other layouts retain 512 MiB.
  The qwen3.6:35b-a3b full-depth probes grew 18 MiB on the 3090 and 8 MiB per 1060, giving
  both measured layouts a 128 MiB margin.
- an OOM ceiling (`fit.failed_ceiling`): a prior OOM blocks explicit and automatically chosen
  contexts at or above the failure for that GGUF and card-size layout until a higher load succeeds.
- measured placement (`fit._measured_placement`): compares observed prefill and decode for fitting,
  free card sets with the same GGUF, micro-batch, runtime image, and nearby context. For
  `ministral-3:8b` at 16K, GPU0 measured 2,616/121.8 tok/s prefill/decode versus 475/20.2 on
  two 1060s; the manager selects GPU0 when both choices are free. Explicit GPU requests prevail.

The plan reports `compute_basis`, `overhead_mb`, `margin_mb`, `failed_ceiling` where present, and
`performance_basis` when measured placement applies. `rigctl observations [--kind KIND]
[--model M]` lists the history. The rig still checks llama.cpp's own projection. The corrected
planner admitted qwen3.6:35b-a3b at 230,144 on GPU0, and a real managed load reached ready in
84.5 seconds and decoded at 148.34 tok/s. The known qwen3-coder:30b-a3b 120K OOM boundary
remains blocked.

**Prefill tuning (2026-10-25).** Under the standing power caps, matched 131K qwen3.6:35b-a3b runs on the 3090 measured 1.87-2.32x prefill from batch 128 to 512 with unchanged depth decode and 138 MiB more compute buffer. The current qwen3.6:27b brain measured 1.41-1.46x prefill from its orchestration batch 64 to 512 at 16K, with unchanged depth decode and 101 MiB more compute. All recall probes passed. Benchmarking also measured a smaller 1.2-1.3x gain for gemma-4:31b at 32K. The imported catalog now gives these three models batch 512 in the relevant interactive/orchestration profiles; the benchmark defaults remain unchanged. The manager resolves the catalog by use at load time. See benchmarking `RIG_PERFORMANCE.md`, "Micro-batch at working context"; load time is cold-cache dependent and is not improved by this setting.

Open: a VRAM check for published profiles; context per slot with `--parallel`.

## Runs, cancel by owner, and demand (2026-10-22)

### Read-only benchmark campaign planning

Gateway can arrange a Benchmarking-owned campaign over placement alternatives supplied by
Orchestration without reserving or loading hardware:

```bash
rigctl campaign-preview campaign-plan.json --planning-mode after_current_work
```

The input uses `benchmark-campaign-plan/1`: `name` and `blocks`, where every block has a unique
`id`, a registry `model`, an optional `suite`, `depends_on`, `runtime_group`, and
`estimated_duration_s`. Blocks in one runtime group must use the same model and fit settings; their
durations are run sequentially on one load. A missing duration is an explicit 300-second fallback in
the response. `planning_mode=live` uses only placements free in the Orchestration snapshot;
`after_current_work` evaluates every offered placement after existing work releases it.

`POST /v1/manager/benchmark-campaigns/preview` asks the rig's `schedule-preview` command for a
revisioned snapshot and candidate placements. Gateway then performs a deterministic discrete-event
simulation: model loads are serialized, while suite groups on disjoint GPU sets may overlap. The
response includes the Orchestration revision, relative load/run times, placement, estimate basis,
and `admission_required_at_execution: true`.

The preview is never allocation authority. Gateway does not reproduce VRAM, topology, host-memory,
thermal, or runtime-health rules. When execution is added, every proposed load must return through
Orchestration's ordinary fit, lease, and guarded load path; a changed revision causes unfinished work
to be replanned.

**Runs.** `rigctl run-start --run ID [--resume "CMD"] [--auto-resume] [--needs MIB[xN]] -- CMD...` hands the rig
a long run: the rig's plan executor starts it as its own unit (`rig-run-ID`, under the rig service identity) and records it in the
plan's `runs` (owner, command, resume command, needs, attempts). Its state follows the unit: running, then
finished (exit 0) or failed; `stopped` and `dismissed` are set by the controller. The reconciler reports each
run that ends, once. `rigctl runs` lists them; `rigctl recover RUN` starts an interrupted, failed or stopped run
again with its resume command (for a campaign, the same command with `--run-id`, which continues from
`campaign_state.json`); `rigctl run-stop RUN` stops it and cleans up; `rigctl dismiss RUN` marks an ended run as
not to be resumed and cleans up. The dashboard's rig tab lists runs with Resume, Stop and Dismiss.

**Cancel by owner.** `rigctl cancel-owner WHO` (`POST /v1/manager/owners/WHO/cancel`) cleans up everything one
owner has, with no time limits: stops its running runs, cancels its queued plan items, unloads its models,
releases its leases and drops its claims (which also cancels its waits), and logs what it did. It matches
leases by owner, run id or `requested_by` (a load's `--owner`), claims by owner or name (and `campaign-WHO`,
the campaign runner's reservations), runs by run id or owner, and waits by owner. Stopping or dismissing a run
calls it for the run's owner.

**Demand.** The manager keeps its waits in progress (collecting and named-card waits, with the card sizes
each needs) and reads the `needs` of running runs. A load that names no cards takes the smallest cards that
hold it; when those are busy and it would spill onto a bigger card (a 1060 model onto the 3090), it does so
only if no other work is waiting for that size. Otherwise it uses other free cards, or answers 409 naming what
it kept free and for whom. A requester's own waits and runs do not count against it. Demand across callers
stops there: the rig runs one mode at a time, so nothing in another mode competes for the cards.

## Requirements gathered (2026-10-17 benchmark lanes)

- **Job queue:** benchmark lanes become jobs the manager schedules: each job states its model and
  card needs, and the next one that fits starts when a card frees up. Adding or removing cards
  mid-run is a claim change; stopping a run cancels its jobs (one stop path for every job).
- **VRAM fit before placement:** a per-model VRAM estimate (weights + KV at the profile's ctx) from
  the registry, checked against the card (ministral-3:8b at 16k ran at 5.97 of 6.1 GB). Built
  2026-10-18 for model loads ("Model loads" above); published profiles still use fixed shapes.
- **One rig guard:** a single always-on rig-side power and thermal guard that every job runs under
  (today `startup_thermal_monitor.py` is started per run, and a lane started by hand sat outside it).
- **Runtime memory policy per use** (run notes `logs/benchmarks/worker_lanes_20261017_203028/RUN_NOTES.md`
  section 9): host prompt cache (`--cache-ram`, llama.cpp default 8192 MiB per server), ctx, slots,
  KV type and placement, mmap, container memory limits. Registry settings per model, merged at load
  time with a use profile (benchmark: cache-ram 0, parallel 1; agent: small cache; chat: bounded cache
  on the 3090 profile), plus a rig-wide RAM budget next to the VRAM fit check. Evidence: five lanes x
  8 GB default cache against 31 GB RAM, earlyoom killed a lane's runtime at 21:41.

## Reservations: split loads wait for their cards (2026-10-20)

One rule for every split load, whoever starts it (orchestration's brain, a benchmark lane, a campaign,
the gateway): a split that cannot start because some of its cards are busy **reserves all of them**
and waits. Single work can never take a reserved card, so the split gets a matching set as soon as the
current holders finish. Without this, single tasks take every card as it frees and splits starve.

The primitive is the rig's claim (`hw_state.py`), extended:

| Field | Meaning |
|---|---|
| `name`, `gpu_ids` | as before |
| `owner` | who renews it (a lane, a campaign run, a manager wait, orchestration) |
| `ttl_s`, `expires_at`, `boot_id` | Every claim expires (default 10 minutes, maximum 1 hour) and is valid only for the current rig boot. Long-running owners renew before expiry; a dead owner or reboot frees its cards. |
| `purpose`, `model` | what it waits for, for status and for an operator deciding to override |

Rules (automatic):

1. A claim can be placed on busy cards. The current holder keeps them until it releases; from then on
   the rig refuses them to any lease that does not name the claim (rule 5 above).
2. Expired claims are ignored for placement and dropped from the record on the next claim write (each
   drop is logged in `gpus/claims_expired.log`).
3. A split load with `wait_s` (`rigctl load-model --gpus 1,3 --wait 7200`): when its cards are not all
   free, the manager reserves them (or uses the claim the request names), renews the reservation while
   it waits, and loads under it once every card is free. Past `wait_s` it drops the reservation and
   answers 504. A reservation the manager made for the wait is dropped once the load's lease holds the
   cards.
4. A lane claims its card group with a TTL and renews it while it runs, and loads with `--wait`.
4a. A load that names no cards, with `wait_s` (`rigctl load-model MODEL --wait forever`), collects its cards
   (2026-10-22): the plan the harness would make on an idle rig says how many cards of which size the model
   needs (for mellum2.1, two 1060s), and the manager claims free cards of those sizes one at a time as they
   free, in one claim it extends (`rig claim extend`), so single work cannot keep taking each card as it
   frees. It loads once the claim holds the set, and never takes a card of another size (a 1060 model does
   not collect the 3090). `wait_s` -1 (`--wait forever`) waits with no time limit, for collecting and for
   named-card waits alike. Removing the wait's claim (`rigctl unclaim wait-...`) cancels it: the load
   answers 409 "cancelled" and nothing is left behind. A wait whose claim lapses (the manager stopped
   renewing it) cancels the same way instead of re-taking it. While the rig is unreachable every wait keeps
   waiting (the rig keeps working; a lost network is not a crash), and a claim that lapsed during the outage
   (nobody could renew it) is taken again when the rig is back, not read as a cancel.
5. A campaign's runtime group loads like any other model load (`rigctl load-model --use benchmark --owner
   campaign-<run> --wait <s>`): pinned cards wait as in rule 3, a group that names no cards collects them as
   in 4a. Past its wait limit (2 h by default) the block fails with the manager's answer.
6. Orchestration's split reservations (`agents/gpu_split.py`: `waiting_partner` -> `joining` ->
   `loading` -> `ready`) also hold a rig claim (`plans-split-<group>`, owner `plans`) for as long as the
   reservation is in a loading state, renewed by the launcher while it waits; and orchestration workers
   skip work on a card held by a claim that is not theirs, as they already do for leases.

Operator overrides (`rigctl`, act immediately, logged in the manager's events):

| Command | Does |
|---|---|
| `rigctl claim NAME --gpus 1,3 [--ttl S]` / `rigctl unclaim NAME` | reserve or free cards by hand (unclaim works on anyone's claim) |
| `rigctl claims` | list claims with owner, expiry and purpose |
| `rigctl kill --gpus 1,3` | stop every runtime on those cards, release their leases, drop claims on them |
| `rigctl load-model MODEL --gpus 1,3 --force` | kill whatever is on those cards, then load |
| `rigctl unload PROFILE` | unload a profile (releases its lease) |
| `scripts/assign_task.py TASK_ID gpu-N` (llm_orchestration) | pin a queued task to one worker (`target_worker`, honored for work tasks as well as meta tasks) |

## Migration

| Phase | Scope | Behavior change |
|---|---|---|
| A (built) | Manager: leases, claims, placement, status, reconcile, `rigctl`; rig: `observe`, detached placed loads | Roo path only; everything else is reported (`unleased_busy`), not blocked |
| Rig guards | Always-on power/thermal guard and VRAM fit on the rig; one stop path for rig jobs; local results fallback | Safety independent of the Pi |
| B | Orchestration stays a mode for now (likely folded into the manager later); its brain and workers take leases through `rigctl` | Plans visible and protected |
| C (built) | Benchmark lanes and campaigns load through the manager and run as manager runs | Benchmarks visible and protected |
| D | `run_runtime.sh` refuses a load without a valid lease | Nothing loads outside the process |

## Pick up here (entry points per phase)

**B, orchestration (left as a mode for now; likely folded into the manager later).**
- `rig start orchestration` (`rig.py` `start_stack`) becomes a manager request: the manager leases
  the brain's GPU and the worker GPUs (`plans` controller) before starting the stack, releases on stop.
- Legacy GPU workers refuse plan claims on leased GPUs by reading the authoritative rig-local
  record (`shared/brain/hw/gpus/`) through `agents/gpu.py` and `scripts/rig/hw_state.py`.
  The shared lease and claim mirrors and old watchdog have been removed.
- Brain meta tasks (`agents/brain_resources.py` `_insert_resource_task`, executed by
  `agents/gpu_llama.py` `load_model`) keep their shape; a load_llm targets only GPUs under the
  stack's lease.
- Open: does the brain's model-choice logic move into the manager (the user expects so, not decided).

**C, benchmarking (built).** Lanes (`benchmarking/scripts/integration/worker_lane.sh`) claim their card
group and load with `rigctl load-model --use benchmark`; campaigns (`llm_orchestration/shared/scripts/
benchmarks/run_campaign.py`) load each runtime group the same way with `--owner campaign-<run>` and cancel
that owner when they end. Both start as runs (`rigctl run-start`). Open: drop
`BENCHMARK_DISABLE_AUTO_RESERVE`; the run-note fixes in `RUN_NOTES.md` sections 1-5
(`logs/benchmarks/worker_lanes_20261017_203028/`).

**D, enforcement.**
- `run_runtime.sh` gains `--lease <id>` and checks it against the rig record (`rig lease list`): the
  GPUs and port it is asked to start must be held by that lease. Every launcher in rule 2 passes it.
- Audit first (log loads without a lease for a while), then refuse.

**Rig guards (in progress 2026-10-17):** always-on `rig-guard.service` on the rig, see `hw_guard.py`.

## Experimental shared 35B child route (2026-11-12)

As of 2026-11-14, a cloud Codex dashboard parent may request `candidate_id=qwen35_shared_1060` with `parent_route=cloud/codex`. The manager starts the shared service through rig orchestration if it is absent, attaches a fresh worker chat, and returns an ephemeral ready route. Release accepts the cloud parent and detaches its child. The rig controller chooses the worker GPU; the cloud chat and harness never name a card. `GET /v1/manager/local-agent-capabilities` supplies a small discovery catalog for the dashboard's `list_local_agents` MCP tool. Default delegation without a candidate uses an already-ready shared 35B service when available; otherwise it follows the existing role-based ready/cold profile policy. If automatic shared attachment has no worker capacity, the manager falls back to the ordinary profile policy.

The Qwen shared candidate accepts `context_tokens` from 16,384 through 262,144 when a child is attached; omitting it uses the model-library default of 65,536. Context capacity is immutable after creation. The Qwen controller estimates admission from the observed 262K worker allocation: 1,029 MiB fixed compute, KV scaling to 1,009 MiB at 262K, plus a 256 MiB reserve. The accepted cap and estimated reservation are returned in the attach receipt. This curve is Qwen-topology-specific. Nemotron remains fixed at 262K until its current expert placement has its own sized-context memory matrix.

The rig-side orchestration controller is authoritative for that reservation. It
returns a service instance, context capacity, placement, reservation scope, and
reserved memory in the attach receipt; Gateway persists and exposes those
fields. `used_tokens` is reported alongside the reservation for utilization and
speed decisions, but never reduces it. A 262K context remains a 262K reservation
until detach even when only a small fraction is occupied.

Agent route requests may set `worker_strategy` to `auto`, `shared_model`, or
`separate_model`. `auto` prefers an attachable context in the currently loaded
shared model and falls back to a reviewed small-model profile when that shared
capacity is unavailable. `shared_model` requires that path and returns a typed
capacity error instead of silently changing models. `separate_model` skips the
shared service and asks for an independently loaded reviewed worker. A
Benchmarking candidate whose provider is `gateway_profile` may be named together
with `separate_model`; Gateway resolves its model identity to a published profile
and the manager still chooses placement. None of these modes accepts a GPU
identifier.

The benchmark [subagent catalog](https://github.com/bigbertharig/benchmarking/blob/master/subagent_catalog.json) is imported into `model_catalog.json`. `GET /v1/manager/subagent-candidates` lists the stable candidate IDs. `POST /v1/manager/agent-routes/resolve` accepts an optional `candidate_id`; for `qwen35_shared_1060` with a shared 35B parent, the manager attaches a fresh 1060 chat to the resident GPU0 weights and returns a ready route and `ephemeral: true`. `POST /v1/manager/agent-routes/release` detaches that child after the harness finishes. The local harness checks the ready roster after attachment and releases in a cleanup path. The parent route and model remain loaded.

Ready shared chats are exposed as `gpu-rig/shared-35b-<chat_id>` through `/v1/models`, `/v1/rig/ready-roster`, `/v1/chat/completions`, and a bounded `/v1/responses` adapter. Requests and tool calls use OpenAI-shaped JSON so the existing six harnesses can use the route. The adapter buffers generation before emitting SSE, accepts text and function calls, and caps a runtime turn at 512 output tokens. It is experimental: each chat is stateful on the rig, so clients must keep a stable route and aligned conversation history; it is not a stateless general OpenAI proxy. Long-running and arbitrary multimodal Responses operations are outside this contract. Codex Terra/Luna are separate online candidates selected by the harness through the live Codex model list, not rig routes.

See the harness [shared 35B subagent matrix](https://github.com/bigbertharig/llm-harness/blob/main/docs/research/shared35_subagent_matrix_2026-11-12.md) for scored outcomes and limitations.

## Automatic Codex continuation after a usage reset

Dashboard Codex turns that fail with app-server's exact
`codexErrorInfo=usageLimitExceeded` signal are scheduled to continue once. The
Gateway prefers app-server's epoch-based exhausted-window reset. If that is
unavailable, it parses the explicit `try again at` value in
`CODEX_RESET_TIME_ZONE` (defaulting to the Gateway host zone), adds a five-minute
safety margin, and persists the resulting UTC timestamp in the chat database. A
Gateway background task dispatches `Continue` even when no dashboard browser is
connected. Desktop and mobile render the scheduled time in the browser's local
zone with its offset abbreviation.

The job is keyed to the failed turn. A user message before the deadline cancels
it, and the dispatcher refuses it if the conversation has changed. On Gateway
restart, a claimed job is retried only when its failed source is still the most
recent turn; if a continuation had already started, the newer recovered turn
makes the old job stale. This keeps delivery at most once at the conversation
boundary. `CODEX_AUTO_CONTINUE_GRACE_SECONDS` and
`CODEX_AUTO_CONTINUE_POLL_SECONDS` tune the default 300-second margin and
15-second poll interval.

## Plan model assignments

For a structured plan task that declares `executor: worker`, `task_class: llm`,
and `llm_role: <benchmarking role>`, gateway plan preparation reads the chosen
starter and asks the Manager to fit that role's active model. An optional
`llm_fallback_roles` list is tried in written order when a candidate does not
fit or is busy. If every fitting choice is busy, the first fitting choice is
assigned and the worker waits for capacity. Plan steps without `llm_role` retain their prior
model behavior. The gateway submits a compact `GATEWAY_MODEL_ASSIGNMENTS`
mapping with the prepared config; orchestration applies it to matching task
definitions before making tasks and persists the chosen model in each task.
The plan remains the workflow, and the Manager owns the model choice and fit.
