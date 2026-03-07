# Model Readiness Admission Plan

## Problem

Plans can currently request Ollama model tags that exist only on paper:

- tag exists in plan defaults or config
- tag may exist in a manifest but not be runnable
- tag may be missing from the active model store used by a worker or split runtime
- tag may exist on one endpoint but not another

That creates obvious failures that should be prevented before batch execution.

Recent concrete examples:

- `qwen2.5:32b` was still referenced as a brain model default even though the rig's working brain endpoint was serving `qwen2.5-coder:32b`
- split runtime warmup for `qwen2.5-coder:14b` failed because the split-local/user model store did not actually have that tag registered

## Target Behavior

Before plan execution, the system should determine which models are both:

1. present
2. runnable

Then the planner should only choose from that admitted model set.

This should support:

- primary model selection by task type
- fallback model selection when the preferred model is unavailable
- endpoint-specific readiness checks
- split/single/brain capability differences

## Required Checks

Readiness should mean more than `ollama list`.

Per endpoint or runtime class, the system should verify:

- tag is registered
- warm generation succeeds
- model can appear in `/api/ps` when warmed
- required runtime class supports the model
  - brain endpoint
  - single-worker endpoint
  - split runtime endpoint

## Suggested Flow

1. Scan Ollama endpoints and runtime classes for available working models.
2. Build a machine-readable admitted model inventory.
3. When planning, restrict model selection to the admitted set.
4. If no suitable primary model is ready, choose a valid fallback.
5. If no acceptable model is ready, fail fast before the batch starts.

## Implications

- plan defaults should not reference stale or hypothetical tags
- benchmark work should produce cleaner admitted-model inventories over time
- runtime prep and planning should be connected by a readiness contract, not assumptions
