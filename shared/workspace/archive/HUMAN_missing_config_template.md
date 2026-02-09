No config.json Template Exists

**Created**: 2026-02-09
**Severity**: low
**Context**: Code review of project setup experience for new deployments and GPU rig rebuild
**Problem**: config.json is required by brain.py, gpu.py, launch.py, and all monitoring scripts, but is excluded from git (correctly, as it contains host-specific values). There is no config.template.json or config.example.json to guide setup. If config.json is missing, brain.py crashes with a bare FileNotFoundError with no helpful message. New deployments or GPU rig rebuilds require reverse-engineering the config schema from code. Required fields identified from code analysis: shared_path, ollama_host, permissions_path, brain.name, brain.model, brain.gpus, gpus[].id, gpus[].name, gpus[].model, gpus[].port, gpus[].permissions, timeouts.poll_interval_seconds, timeouts.brain_think_seconds, timeouts.worker_task_seconds, resource_limits.max_temp_c, resource_limits.max_vram_percent, resource_limits.max_power_w, retry_policy.max_attempts.
**Attempts**: None - this is a documentation/onboarding gap identified during code review.
**Recommendation**:
1. Create shared/agents/config.template.json with all required fields and example values
2. Add a startup check in brain.py and gpu.py that prints a helpful error if config.json is missing, pointing to the template
3. Add comments in the template explaining each field's purpose and valid ranges
4. Document config setup in quickstart.md
