# Current Benchmark Procedure

This is the current benchmark operating procedure.

Use this doc when you are actually running tests.
Do not improvise alternate runtime paths unless you are explicitly doing recovery work.

## Goal

Run repeatable tests across multiple models, record the results in one place, and keep benchmark setup aligned with the orchestrator.

## Canonical Inputs And Outputs

Inputs:
- model archive: `/media/bryan/shared/models/`
- benchmark catalog: `/media/bryan/shared/scripts/benchmarks/benchmark_catalog.json`
- suite presets: `/media/bryan/shared/scripts/benchmarks/suite_presets.json`
- task-profile recommendations: `/media/bryan/shared/scripts/benchmarks/model_task_library.json`

Outputs:
- raw scored runs: `/media/bryan/shared/logs/benchmarks/model_benchmark_records.jsonl`
- living reference: `/media/bryan/shared/logs/benchmarks/MODEL_BENCHMARK_REFERENCE.md`
- backend/test certification status: `/media/bryan/shared/scripts/benchmarks/benchmark_status.json`
- per-run logs and harness outputs: `/media/bryan/shared/logs/benchmarks/`

## Operating Rules

1. Start benchmark sessions through the orchestrator.
2. Load and unload worker models through orchestrator `meta` tasks only.
3. Keep benchmark mode isolated from normal default operations.
4. Record every scored run in the shared benchmark ledger.
5. Treat the generated reference as the living scoreboard.
6. Certify backend/test compatibility before assuming a suite is runnable.

## Step 1: Put The Rig In Benchmark Mode

Normal benchmark start:

```bash
python3 ~/llm_orchestration/scripts/start_benchmark_mode.py
```

Or start a specific triplet:

```bash
python3 ~/llm_orchestration/scripts/start_custom_mode.py \
  --brain-model qwen2.5-coder:32b \
  --single-model qwen2.5:7b \
  --split-model qwen2.5-coder:14b \
  --split-candidate-group pair_4_5
```

Verify active agents:

```bash
pgrep -af "brain.py|gpu.py"
pgrep -af "brain.py|gpu.py" | grep config.benchmark.json
ss -ltnp | grep -E ':1143[0-9]|:11440|:11441'
```

## Step 2: Prepare Model Storage

Inspect current hot-set:

```bash
python3 /media/bryan/shared/scripts/manage_model_hotset.py report
python3 /media/bryan/shared/scripts/manage_model_hotset.py dedupe --prune-imports
```

Promote archive GGUFs to the target endpoint as needed:

```bash
python3 /media/bryan/shared/scripts/manage_model_hotset.py promote-gguf \
  --gguf /media/bryan/shared/models/<family>/<file>.gguf \
  --name <model:tag> \
  --host 127.0.0.1:11436
```

Keep only the intended hot-set on each endpoint after a run batch:

```bash
python3 /media/bryan/shared/scripts/manage_model_hotset.py evict \
  --host 127.0.0.1:11436 \
  --keep <model:tag>
```

## Step 3: Prepare Runtimes Deterministically

Canonical prep path:

```bash
python3 /media/bryan/shared/scripts/benchmarks/prepare_benchmark_runtimes.py \
  --clear-orphan-queue-locks \
  --strict-processing-empty \
  --force-unload-first \
  --restart-agents-first \
  --brain-endpoint 127.0.0.1:11434 --brain-model qwen2.5-coder:32b \
  --single-endpoint 127.0.0.1:11436 --single-model qwen2.5-coder:7b \
  --split-endpoint 127.0.0.1:11441 --split-model qwen2.5-coder:14b \
  --split-candidate-group pair_4_5
```

What this step is for:
- clear stale queue locks
- verify queue and processing safety
- restart agents when needed
- load models sequentially
- verify each endpoint answers and shows the expected model
- recover split runtimes through orchestrator-owned recovery if needed

Do not replace this with manual `ollama serve` launches.

## Step 4: Choose The Tests

Before running a new test family on Ollama, certify that the backend supports it:

```bash
python3 /media/bryan/shared/scripts/benchmarks/certify_benchmark_backend.py \
  --id gsm8k \
  --model local-chat-completions \
  --model-args "model=qwen2.5:7b,base_url=http://localhost:11436/v1/chat/completions,api_key=ollama"
```

Rules:
- backend certification is environment-level first, not model-by-model by default
- uncertified tests stay unknown until probed
- blocked backend/test pairs should fail fast instead of being rediscovered in suite runs
- for local custom tests, use `run_local_custom_task.py` instead of `run_lm_eval_task.py`
- when certifying standardized tests on Ollama, treat these as separate backend profiles:
  - `ollama_chat_completions_raw`
  - `ollama_chat_completions_templated`
  - `ollama_completions`

Run one test:

```bash
python3 /media/bryan/shared/scripts/benchmarks/run_lm_eval_task.py \
  --id gsm8k \
  --model local-chat-completions \
  --model-args "model=qwen2.5-coder:7b,base_url=http://localhost:11436/v1,api_key=ollama"
```

For the currently working standardized lane, include chat templating explicitly:

```bash
python3 /media/bryan/shared/scripts/benchmarks/run_lm_eval_task.py \
  --id gsm8k \
  --model local-chat-completions \
  --model-args "model=qwen2.5-coder:7b,base_url=http://localhost:11435/v1/chat/completions,api_key=ollama,eos_string=<|im_end|>" \
  --apply-chat-template
```

Run one local custom test:

```bash
python3 /media/bryan/shared/scripts/benchmarks/run_local_custom_task.py \
  --id custom_command_safety \
  --model qwen2.5:7b \
  --base-url http://localhost:11436
```

Build a preset suite first when you want a repeatable batch:

```bash
python3 /media/bryan/shared/scripts/benchmarks/build_benchmark_suite.py \
  --preset baseline_core \
  --output /media/bryan/shared/scripts/benchmarks/suites/baseline_core.json
```

Use the catalog and suite presets as the benchmark inventory:
- `/media/bryan/shared/scripts/benchmarks/benchmark_catalog.json`
- `/media/bryan/shared/scripts/benchmarks/suite_presets.json`

## Step 5: Record Results

Scored runs should end up in:
- `/media/bryan/shared/logs/benchmarks/model_benchmark_records.jsonl`
- `/media/bryan/shared/logs/benchmarks/MODEL_BENCHMARK_REFERENCE.md`

Manual record example:

```bash
python3 /media/bryan/shared/scripts/benchmarks/record_benchmark_result.py \
  --model qwen2.5-coder:14b \
  --test-id gsm8k \
  --score 0.721 \
  --metric exact_match \
  --harness lm_eval \
  --suite baseline_core
```

Rules:
- record the actual model tag used at the endpoint
- use the benchmark catalog `test-id`
- keep suite names stable for repeated comparisons
- do not scatter scores across ad-hoc markdown notes

## Step 6: Review The Living Results

Use these as the main reporting surfaces:
- human-readable summary: `/media/bryan/shared/logs/benchmarks/MODEL_BENCHMARK_REFERENCE.md`
- raw ledger: `/media/bryan/shared/logs/benchmarks/model_benchmark_records.jsonl`
- backend compatibility status: `/media/bryan/shared/scripts/benchmarks/benchmark_status.json`

The reference doc answers:
- what was the latest score per model and test
- what recent runs happened
- which model is currently stronger on which measured task
- which models or suites are currently blocked by runtime or harness compatibility

## Step 7: Feed Results Back Into Plans

Use measured results to update:
- `/media/bryan/shared/scripts/benchmarks/model_task_library.json`

Then generate plan recommendations:

```bash
python3 /media/bryan/shared/scripts/benchmarks/recommend_plan_models.py \
  --plan /media/bryan/shared/plans/arms/<plan_name>/plan.md \
  --output /media/bryan/shared/logs/benchmarks/<plan_name>_model_recommendations.json
```

This keeps:
- model choice explicit
- plan model fields measurable
- benchmark work tied back to operational use

## Recovery Rules

If benchmark runtime state drifts:
1. stop manual changes
2. confirm `queue/` and `processing/` are clean
3. return to default cleanly
4. restart benchmark mode
5. rerun deterministic runtime prep

### Reset Types

Use the dashboard/operator reset path for benchmark recovery:

- targeted worker issue:
  - dashboard `Reset selected GPU`
  - this is the normal hard reset for one worker during benchmark prep or loading
- full rig normalization:
  - dashboard `Return To Default`
  - use when multiple workers or task lanes are mixed up

Do not treat internal `reset_gpu_runtime` as the default operator reset path.
That command is an agent-side thermal recovery mechanism, not the standard benchmark recovery tool.

Useful reset sequence:

```bash
python3 ~/llm_orchestration/scripts/start_default_mode.py --json
python3 ~/llm_orchestration/scripts/start_benchmark_mode.py --json
```

If workers auto-unload unexpectedly, verify benchmark config can actually hold the intended number of hot workers.

## Current Known Compatibility Limits

As of 2026-03-06:

### Model issues
- `qwen3.5:4b` and `qwen3.5:9b` are not currently benchmarkable on this stack
  - load attempts and direct generate probes failed despite the tags being present
- `qwen2.5-coder:14b` split baseline on `pair_4_5` is currently unstable
  - warmup failed and the pair had to be reset

### Ollama limitations (fundamental, not fixable by config)
- Ollama `/v1/completions` does not return logprobs — this blocks all MC/loglikelihood lm_eval tasks
- Ollama `/v1/completions` rejects array-shaped prompts (a proxy at port 11435 can fix the shape issue, but not the missing logprobs)
- These are Ollama API limitations confirmed on v0.17.6 as of 2026-03-06
- **Do not keep trying to make MC tasks work on Ollama** — use vLLM instead

### What works on Ollama
- `local-chat-completions` with `--apply_chat_template` works for generation tasks:
  - gsm8k, drop, bbh, math_500, aime_2024
- `local_custom` harness works for all custom pipeline tests
- `local-chat-completions` without `--apply_chat_template` is broken for generation tasks

### What needs vLLM (or equivalent full-API backend)
- All MC/loglikelihood lm_eval tasks: mmlu, arc_challenge, hellaswag, boolq, piqa, winogrande, truthfulqa_mc2, gpqa_diamond, mmmlu, musr
- These tasks require logprobs in the completions response, which Ollama does not provide

### Environment setup status
- Ollama: working for generation tasks and custom tests
- vLLM: not yet set up — blocks baseline_core, fast_smoke, reasoning_heavy
- evalplus: not yet set up — blocks coding_heavy, part of baseline_core
- lighteval: partially installed, dependency conflict with latex2sympy2 — blocks mmlu_pro
- swebench/livecodebench/bfcl/harbor: not yet set up — deferred

### Chat template rule
- Always use `--apply_chat_template` for Ollama lm_eval runs
- Always specify the correct HF tokenizer for the model family
- See README.md "Chat Template Standardization" table for tokenizer mappings

## What Not To Do

- do not run worker benchmarks by manually spawning unmanaged `ollama serve`
- do not use direct-agent submit as the normal benchmark path
- do not treat ad-hoc markdown notes as the canonical result store
- do not leave the result ledger unrecorded after a successful scored run

## End Of Run

When the batch is done:
1. verify scores were recorded
2. inspect the living reference
3. clean endpoint hot-set if needed
4. return the rig to default mode if benchmarking is finished

Return to standard operations:

```bash
python3 ~/llm_orchestration/scripts/start_default_mode.py
```
