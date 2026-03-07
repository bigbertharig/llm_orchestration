# Benchmark History And Lessons Learned

This document captures the older benchmark drift, what failed, and the changes made to simplify the system.

Use it as background, not as the current operating procedure.

For the current procedure, use [CURRENT_BENCHMARK_PROCEDURE.md](CURRENT_BENCHMARK_PROCEDURE.md).

## Why This Exists

Benchmarking got messy because several pathways were treated like they were equally valid:
- manual Ollama runtime setup
- benchmark scripts
- dashboard/runtime state
- normal orchestration startup

That produced conflicting ownership and hard-to-compare results.

The current rule is simpler:
- benchmarks still run inside the orchestrator
- benchmark mode is just a controlled orchestrator mode
- worker model changes still happen through `meta` tasks

## Main Lessons Learned

### 1. Manual worker runtime launches caused state drift

What happened:
- worker or split `ollama serve` was launched directly
- VRAM usage appeared real
- dashboard and heartbeat linkage did not match reality
- brain resource logic could fight the unmanaged runtime

What changed:
- benchmark docs now treat manual worker runtime launches as debug-only
- current procedure requires orchestrator-owned runtime prep

### 2. Multiple startup paths were being treated as different systems

What happened:
- `startup.py`
- benchmark mode
- custom mode
- dashboard behavior

were described like separate operating models instead of one orchestrated system with different modes.

What changed:
- current docs define one benchmark flow
- benchmark mode and custom mode are described as orchestrator entry points, not separate stacks

### 3. Mixed configs and stale task lanes polluted load tests

What happened:
- benchmark and default stacks overlapped
- stale runtime tasks stayed in `queue/` or `processing/`
- queue lock leftovers blocked clean runs

What changed:
- deterministic prep now explicitly checks queue safety
- restart-first and force-unload-first paths were added
- hard reset sequence was standardized for stale-state recovery

### 4. Benchmark config could auto-unload the very models being tested

What happened:
- `max_hot_workers=0` or too low
- brain intentionally inserted `unload_llm`
- models disappeared during benchmark prep

What changed:
- current docs call this out explicitly
- benchmark prep now requires verifying hot-worker policy against intended concurrent loaded workers

### 5. Benchmark result storage was too diffuse

What happened:
- results lived in per-run notes, one-off logs, and local observations
- hard to compare latest scores across models

What changed:
- scored runs now belong in one ledger:
  - `/media/bryan/shared/logs/benchmarks/model_benchmark_records.jsonl`
- human-readable comparison belongs in:
  - `/media/bryan/shared/logs/benchmarks/MODEL_BENCHMARK_REFERENCE.md`

### 6. The benchmark harness path itself had bugs

Observed on 2026-03-05:
- `run_lm_eval_task.py` used deprecated task-list CLI
- invocation assumptions were stale
- some harness/backend combinations did not work with the active Ollama compatibility path

What changed:
- task listing was updated to the current lm-eval interface
- run invocation was updated
- current docs distinguish between supported executable tasks and blocked backend/task combinations

### 7. Brain endpoint model state was not always clean

Observed on 2026-03-05:
- `11434` brain endpoint could report model availability inconsistently
- some attempts triggered pull behavior or model-not-found responses despite apparent local state

What changed:
- current benchmark procedure treats runtime prep and endpoint verification as mandatory
- the benchmark report records the need to repair endpoint state before full parity runs

### 8. Local Ollama storage got bloated and duplicated

What happened:
- duplicate blobs across multiple Ollama stores
- stale import artifacts consumed root disk

What changed:
- hot-set management is now part of the benchmark process
- `manage_model_hotset.py report` and `dedupe --prune-imports` are part of routine prep

### 9. We mixed up internal reset paths and operator reset paths

What happened:
- internal `reset_gpu_runtime` meta tasks were treated like the normal manual reset tool
- that path is actually for agent-side thermal recovery
- operator recovery became confusing because the dashboard uses a different, harder reset path

What changed:
- current docs now distinguish the two reset layers
- normal operator targeted recovery should use dashboard `Reset selected GPU`
- full operator recovery should use dashboard `Return To Default`
- internal `reset_gpu_runtime` should stay in the thermal-recovery bucket unless deliberately testing that mechanism

### 10. Current fast-smoke suite is not executable on the present Ollama compatibility path

Observed on 2026-03-05:
- `fast_smoke` includes `boolq`, `piqa`, and `hellaswag`
- those tasks require loglikelihood-style evaluation
- `local-chat-completions` fails because chat completions do not implement loglikelihood
- `local-completions` gets further, but current Ollama OpenAI compatibility rejects the prompt shape sent by `lm_eval`

Observed errors:
- `Loglikelihood is not supported for chat completions`
- `json: cannot unmarshal array into Go struct field CompletionRequest.prompt of type string`

What changed:
- this is now tracked as a benchmarkability issue, not just a one-off run failure
- the living reference should record `fast_smoke` as blocked on the current stack
- generation-only tasks remain the compatible fallback lane until backend compatibility is repaired

### 12. We needed backend certification, not repeated rediscovery

What happened:
- catalog presence was treated like executable support
- suite runs rediscovered the same Ollama compatibility failures repeatedly
- blocked tests and unknown tests were not machine-readable

What changed:
- backend/test certification is now a first-class benchmark artifact
- `benchmark_status.json` records supported, blocked, and still-unknown coverage
- `certify_benchmark_backend.py` is the canonical probe path for new Ollama test compatibility

### 13. Custom tests were defined before they were runnable

What happened:
- custom test ids existed in the benchmark catalog
- but there was no executable `local_custom` runner behind them
- this made the custom suite look broader than it really was

What changed:
- a runnable `run_local_custom_task.py` path now exists
- baseline coverage exists for:
  - `custom_json_schema_strict`
  - `custom_command_safety`
  - `custom_ambiguity_handling`
- more custom tests still need implementation before the suite is complete

### 14. Standardized Ollama compatibility turned out to be profile-specific

What happened:
- we initially treated `local-chat-completions` as one backend state
- in practice, raw chat-completions and templated chat-completions behave differently
- this made the certification matrix misleading until the profiles were split

What changed:
- backend status now distinguishes:
  - `ollama_chat_completions_raw`
  - `ollama_chat_completions_templated`
  - `ollama_completions`
- current observed state:
  - raw chat-completions is broken for generation tasks
  - templated chat-completions is the working standardized generation lane
  - completions is still blocked for MC/loglikelihood tasks because of Ollama prompt-shape incompatibility

### 11. Split baseline on pair_4_5 is currently unstable

Observed on 2026-03-05:
- `qwen2.5:14b` split load on `pair_4_5` failed during warmup
- workers were reset afterward
- no stable split runtime remained on `11441`

What changed:
- this is now treated as an operational status item in the living benchmark reference
- split model scores should not be compared as if the split lane were healthy until `pair_4_5` can complete a clean load and stay ready

### 15. Ollama cannot run MC/loglikelihood benchmarks — period

Observed on 2026-03-06:

The persistent failures on boolq, arc_challenge, hellaswag, piqa, winogrande, mmlu, and truthfulqa_mc2 are not configuration problems. They are fundamental Ollama API limitations:

1. **Array prompt rejection**: Ollama's `/v1/completions` rejects array-shaped prompts. A proxy (port 11435) was built to flatten these, which fixed the shape issue.
2. **Missing logprobs**: Even with the proxy, Ollama does not return `logprobs` in completions responses. lm_eval's loglikelihood evaluation requires logprobs to score MC tasks. This is not fixable externally.

Both issues were confirmed on Ollama v0.17.6. The Ollama team has added logprobs support to chat completions (since v0.12.11) but the completions endpoint still does not return them.

What changed:
- MC/loglikelihood tasks are now routed to vLLM (or equivalent full-API backend)
- Ollama remains the production runtime and handles generation-based benchmarks + custom tests
- The 5 test suites now map across multiple environments, not just Ollama
- Suite presets track which environment each test requires
- This is documented in README.md under "Runtime Environments" and "Suite-to-Environment Mapping"

The key insight: run each test in the environment it was designed for. The model weights are identical across backends (loaded from the same shared archive), so scores are comparable. The backend is the test harness, not the thing being tested.

## Important Changes We Have Already Made

### Runtime and procedure changes

- standardized benchmark-mode startup
- added deterministic benchmark runtime prep wrappers
- standardized hard-reset sequence for stale runtime state
- documented orchestrator-only model loading

### Benchmark runner changes

- fixed `run_lm_eval_task.py` task listing path
- updated lm-eval run invocation expectations
- split requirements into core vs optional heavier dependencies
- improved dataset setup with free-space guard and repo-key fix

### Result tracking changes

- centralized scored runs in `model_benchmark_records.jsonl`
- generated shared living reference in `MODEL_BENCHMARK_REFERENCE.md`
- added suite and test inventories in machine-readable files

## Historical Notes Worth Keeping

- 7B vs paired 14B tradeoffs were captured in `/media/bryan/shared/scripts/benchmarks/llm_benchmark_plan.md`
- the 2026-03-05 execution details were captured in `/media/bryan/shared/scripts/benchmarks/benchmark_run_report_20260305.md`

Keep those as run-specific records.
Do not use them as the current procedure doc.

## Practical Conclusion

The benchmark system is cleaner when it is boring:
- one orchestrator-owned runtime path
- one benchmark catalog
- one result ledger
- one generated living reference

Whenever a new workaround starts to feel like a second benchmark system, it is probably drift and should be folded back into the main path.
