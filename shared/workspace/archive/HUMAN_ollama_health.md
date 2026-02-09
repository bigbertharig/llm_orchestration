# HUMAN: No Health Checks on Ollama Instances

**Created**: 2026-02-09
**Severity**: medium
**Context**: Code review of brain.py and gpu.py for Ollama instance reliability
**Problem**: Neither the brain nor GPU agents perform health checks on their Ollama instances after initial startup. If an Ollama instance hangs (stops responding but process is alive), the brain will continue assigning LLM tasks that all timeout. GPU agents would burn through the 20-minute stuck task threshold for each task before escalation. Model unload failures (gpu.py unload_model()) log a warning but have no retry -- if unload fails, the GPU agent believes it's still "hot" with a loaded model, but VRAM may be in an inconsistent state. There is no circuit-breaker pattern: if Ollama starts returning errors consistently, agents keep trying indefinitely rather than backing off and escalating.
**Attempts**: Code review only. This requires architectural discussion about health check frequency vs. overhead on 6GB GPUs where Ollama competes with workers for VRAM.
**Recommendation**:
1. Add periodic `/api/tags` health check in GPU agent heartbeat cycle (every 30s, lightweight)
2. If Ollama fails health check 3 times consecutively, restart the Ollama process
3. Add retry with backoff for model unload operations (try 3 times, 5s apart)
4. Implement circuit breaker: after N consecutive Ollama failures, mark GPU as unhealthy in heartbeat and stop claiming LLM tasks
5. Brain should detect "unhealthy" GPU state in heartbeats and avoid assigning LLM work to it

Human comments: Yes, we need to add some sort of ollama metrics to the heartbeat as well.  We focused mostly on the script side, now we need to update the ollama side as well.
