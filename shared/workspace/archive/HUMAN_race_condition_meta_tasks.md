# HUMAN: Race Condition in Brain Resource Meta-Task Insertion

**Created**: 2026-02-09
**Severity**: medium
**Context**: Code review of brain.py _make_resource_decisions() for load_llm/unload_llm meta task management
**Problem**: The brain's resource monitoring loop checks GPU agent heartbeat state and inserts load_llm/unload_llm meta tasks into the queue. The brain tracks pending requests in an in-memory dict (`load_llm_requests`), but this tracking is lost on brain restart. If the brain restarts while a load_llm task is in the queue (unclaimed), the brain would not know about it and could insert a duplicate. Additionally, between checking the queue stats and inserting a new meta task, a GPU agent could claim the existing meta task, leading the brain to correctly insert a new one -- but if timing is tight, two meta tasks for the same GPU could exist briefly. GPU agents handle this gracefully (load_model is idempotent), but duplicate unload_llm tasks could cause unnecessary state churn.
**Attempts**: Code review only. This requires an architectural decision about persistence vs. simplicity.
**Recommendation**:
1. Before inserting a meta task, scan the queue directory for existing meta tasks targeting the same GPU
2. Persist `load_llm_requests` to brain/state.json so it survives restarts
3. Add a dedup check in GPU agent claim_tasks() to skip duplicate meta tasks for the same command

Human comments: Im unsure about whether the brain should ever do targeted task assignments, I like the concept of the brain spitting out tasks and whatever is capable picking them up.
However, I can imagine some situations where direct tasks are valuable, so leave it for now, and yes I agree there needs to be a dedup system, so implement it. 
