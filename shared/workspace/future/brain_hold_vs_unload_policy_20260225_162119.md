# Brain Hold-vs-Unload Policy (Future Scheduling Intelligence)

## Why This Matters

As runtime ownership and model loading become more stable, the next optimization ceiling is **scheduling intelligence**:
- should the brain keep a model loaded while idle (hold)
- or unload now and reload later when demand returns

This matters because model loads are no longer negligible and are becoming predictable.

## Current Observations (Feb 25, 2026)

Empirical load times (rough, but increasingly stable):
- **single/solo load** (`load_llm`): ~`2 minutes`
- **split load** (`load_split_llm`): ~`4 minutes`

This is enough to justify explicit cost-based decisions later.

## Core Idea

Brain should eventually choose between `hold` and `unload` using a cost comparison.

### Inputs to compare
1. **Reload cost** (if we unload now)
- expected load duration for required model/placement
- split wedge / load failure risk penalty (especially for split runtimes)
- warmup delay impact on downstream queue

2. **Hold cost** (if we keep runtime hot)
- idle time while holding VRAM/resources
- opportunity cost (blocking other tasks / preventing better model placement)
- thermal/power impact (optional later)

3. **Expected near-future demand**
- current public queue
- private task queue (what will release soon)
- plan phase context (verify wave likely next vs no LLM demand)
- recent demand trend by task family/tier

## Key Missing Input (Important)

We cannot build a good policy yet because the second major input is not stable enough:

### Average task times by family are still moving
We need reliable duration distributions for task families such as:
- `worker_review`
- `worker_review_verify`
- `worker_gap_review`
- `worker_gap_review_verify`
- (and potentially phase-backfill variants)

Right now task durations are still shifting due to:
- split wedge/recovery changes
- shard tuning experiments
- routing underutilization (placement/capability mismatch semantics)
- new CPU hints entering prompts (recently wired in)

If we build hold/unload intelligence before those stabilize, the policy will be noisy and likely wrong.

## Prerequisites Before Implementing Policy

1. **Stabilize task routing and concurrency**
- capability-based LLM routing (normal work not hard-gated by placement)
- consistent split capacity behavior

2. **Stabilize shard defaults / size-band heuristics**
- benchmark comparisons across repo sizes
- fewer severe verify stragglers

3. **Collect runtime metrics consistently**
For each task family, track at minimum:
- `p50` duration
- `p90` duration
- `p95/max` (straggler risk)
- retry rate / timeout rate
- queue wait time (separate from compute time)
- split vs single execution counts (where relevant)

4. **Collect load-time metrics directly**
Per model + placement:
- `load_llm` duration rolling average (`solo_load_avg_s`)
- `load_split_llm` duration rolling average (`split_load_avg_s`)
- load failure / wedge rates

## Proposed Future Policy (Incremental Rollout)

### Phase 1: Simple Hold Window Heuristic (Low Risk)
Use a deterministic hold window derived from measured load times.

Example concept:
- keep runtime hot if predicted next compatible task arrival is within `K * avg_load_time`
- otherwise unload after idle timeout

Inputs:
- rolling average load time by placement/model
- queue + private queue demand counts by tier/model
- current runtime state

No complex ML/prediction required yet.

### Phase 2: Phase-Aware / Plan-Aware Lookahead
Add stronger signals from private tasks / plan graph:
- upcoming `worker_review_verify` wave -> preserve split 14B capacity
- end of verify wave + no pending split demand -> unload earlier
- preserve hot single 7B workers during dense extract waves

### Phase 3: Expected Value Scoring
Score possible actions per worker/group:
- `hold current runtime`
- `unload`
- `switch to another model`
- `prewarm split`

Choose action with lowest expected delay/cost under current queue + near-future release state.

## Suggested Metrics Artifact Additions (Future)

Add/track in brain metrics or execution stats:
- `model_load_metrics` by `model_id + placement`
  - `count`, `avg_s`, `p50_s`, `p90_s`, `fail_rate`
- `task_family_metrics` by task prefix
  - `count`, `p50_s`, `p90_s`, `retry_rate`, `queue_wait_p50_s`
- `idle_hold_windows`
  - how long a runtime was kept hot while idle
  - whether future demand arrived before unload
- `avoidable_reload_events`
  - unload followed by reload of same model shortly after (bad signal)

## Risks / Failure Modes to Avoid

1. **Premature unloading during phase transition**
- causes avoidable 2-4 minute reload penalties

2. **Over-holding split runtimes**
- blocks other work and increases wedge exposure if churn logic is wrong

3. **Using queue-only view**
- misses near-future private tasks and unloads too aggressively

4. **Using unstable averages**
- policy thrashes while workloads are still changing

## What We Should Do Now (Before Policy Work)

1. Continue benchmark and full-run timing collection
2. Finish capability-based routing fix (core)
3. Tune verify shard defaults and reduce stragglers
4. Let CPU hint integration settle and measure impact on task durations
5. Improve phase metrics/stall attribution so compute time is separated from scheduling delay

## Trigger To Revisit This

Revisit implementation when these conditions are true:
- load times remain stable for at least several runs (solo/split averages converged)
- verify task family durations are reasonably stable by repo size band
- split wedge rate is low enough that load time estimates are trustworthy
- benchmark metrics clearly separate compute vs queue/stall time

## Related Notes

- `shared/workspace/future/github_analyzer_runtime_reduction_plan_20260224_214410.md`
- `shared/workspace/human/HUMAN_split_runtime_state_machine_20260225_132656.md`
- `shared/workspace/human/HUMAN_capability_based_llm_task_routing_20260225_152228.md`
