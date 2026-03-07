# Benchmarks

This folder is the benchmark hub for the rig.

It has four jobs:
1. define which models we can test
2. define which tests and suites we can run
3. document the current benchmark procedure
4. point to the living results ledger

## Canonical Docs

- Current procedure: [CURRENT_BENCHMARK_PROCEDURE.md](CURRENT_BENCHMARK_PROCEDURE.md)
- History and lessons learned: [BENCHMARK_HISTORY.md](BENCHMARK_HISTORY.md)
- Plan integration: [PLAN_FORMAT_INTEGRATION.md](PLAN_FORMAT_INTEGRATION.md)
- Worker strategy notes: [llm_benchmark_plan.md](llm_benchmark_plan.md)
- Session report archive: [benchmark_run_report_20260305.md](benchmark_run_report_20260305.md)
- Local custom-task notes: [custom_tasks/README.md](custom_tasks/README.md)

## Canonical Result Stores

- Living results summary:
  - `/media/bryan/shared/logs/benchmarks/MODEL_BENCHMARK_REFERENCE.md`
- Raw append-only benchmark ledger:
  - `/media/bryan/shared/logs/benchmarks/model_benchmark_records.jsonl`
- Backend/runtime certification status:
  - `/media/bryan/shared/scripts/benchmarks/benchmark_status.json`
- Per-run harness outputs:
  - `/media/bryan/shared/logs/benchmarks/`

`MODEL_BENCHMARK_REFERENCE.md` is the human-readable living document.
`model_benchmark_records.jsonl` is the source of truth for recorded scored runs.
`benchmark_status.json` is the source of truth for what is certified, blocked, or still unknown on the current benchmark backend.

## What "Good" Looks Like

The benchmark system should stay simple:
- pick a model
- promote it to the needed endpoint if required
- prepare orchestrator-owned runtimes
- run a named test or suite
- record the result into the benchmark ledger
- compare models in the generated reference

No separate manual-Ollama benchmark path should exist for normal runs.

## Runtime Environments

Different tests require different inference backends. Ollama does not support all evaluation modes (notably: logprobs for MC/loglikelihood scoring). Run each test in the environment it was designed for.

### Environment Map

| Environment | What It Runs | Status |
|-------------|-------------|--------|
| **Ollama** (chat completions, templated) | Generation lm_eval tasks (gsm8k, bbh, drop, math_500, aime_2024) + all custom tests | Working |
| **vLLM** | MC/loglikelihood lm_eval tasks (mmlu, arc_challenge, hellaswag, boolq, piqa, winogrande, truthfulqa_mc2, gpqa_diamond, mmmlu, musr) | Needs setup |
| **evalplus** | HumanEval+, MBPP+ | Needs setup |
| **lighteval** | mmlu_pro, ifeval | Partially installed |
| **swebench** | swe_bench_verified (Docker) | Needs setup (defer) |
| **livecodebench** | livecodebench | Needs setup (defer) |
| **bfcl** | bfcl_v4 (function calling) | Needs setup |
| **harbor** | terminal_bench_2 (Docker) | Needs setup (defer) |

### Why Not Just Ollama?

Ollama is our production runtime and is good at serving models. But its OpenAI-compatible API is incomplete:
- `/v1/completions` does not return logprobs (required for MC/loglikelihood evaluation)
- `/v1/completions` rejects array-shaped prompts that lm_eval sends
- These are Ollama limitations, not model limitations

vLLM, llama.cpp server, and other inference backends implement the full OpenAI completions API including logprobs. Benchmark results from these backends are valid for ranking models that run on Ollama in production, because the model weights and quantization are identical.

### Chat Template Standardization

All benchmark runs must use the correct chat template for the model family. The template is determined by the HuggingFace tokenizer, not the backend.

| Model Family | HF Tokenizer | Template Format |
|-------------|-------------|-----------------|
| qwen2.5-coder | `Qwen/Qwen2.5-Coder-7B-Instruct` (or 14B/32B) | ChatML (`<\|im_start\|>...<\|im_end\|>`) |
| qwen2.5 | `Qwen/Qwen2.5-7B-Instruct` (or 32B) | ChatML |
| qwen3.5 | TBD — verify after load issues resolved | TBD |
| mistral | `mistralai/Mistral-7B-Instruct-v0.2` | Mistral `[INST]...[/INST]` |
| deepseek-r1 | TBD — verify on first benchmark run | TBD |
| phi4 | TBD — verify on first benchmark run | TBD |
| gemma3 | TBD — verify on first benchmark run | TBD |

Rules:
- Always pass `--apply_chat_template` for lm_eval runs on Ollama
- For vLLM, the tokenizer auto-applies the template — verify it matches
- Record the tokenizer used in every benchmark run
- If two backends produce different scores for the same model+test, check template application first

### Model Weight Consistency

All environments load from the same physical model files in `/media/bryan/shared/models/`. This standardizes quantization levels automatically. There is no quant mismatch risk as long as every backend loads from the shared archive.

### Suite-to-Environment Mapping

Each suite spans one or more environments:

**baseline_core** (3 environments):
- Ollama: gsm8k
- vLLM: mmlu, arc_challenge, hellaswag, winogrande, truthfulqa_mc2
- evalplus: humaneval_plus

**fast_smoke** (2 environments):
- Ollama: gsm8k
- vLLM: boolq, piqa, hellaswag

**reasoning_heavy** (3 environments):
- Ollama: gsm8k, bbh, drop
- vLLM: arc_challenge, truthfulqa_mc2
- lighteval: mmlu_pro

**coding_heavy** (3 environments):
- evalplus: humaneval_plus, mbpp_plus
- livecodebench: livecodebench
- swebench: swe_bench_verified

**agent_reliability** (1 environment):
- Ollama: all 6 custom tests

### Environment Setup Priority

1. **vLLM** — unblocks baseline_core, fast_smoke, reasoning_heavy
2. **evalplus** — unblocks coding_heavy partially, completes baseline_core
3. **lighteval** — completes reasoning_heavy
4. **Remaining 3 custom tests** — completes agent_reliability
5. **swebench, livecodebench, bfcl, harbor** — defer until core suites work

### Deferred Environment Details (for future setup)

**swe_bench_verified** (swebench, Docker):
Takes ~500 verified GitHub issues from popular Python repos (Django, Flask, scikit-learn, etc.). Gives the model the issue description + the repo code. The model must write a patch that passes the repo's test suite. Runs in Docker because each issue needs its own clean repo clone + test runner. Tests agentic coding ability — reading large codebases, understanding issues, writing correct patches. Metric: `resolved_rate`.

**livecodebench** (livecodebench harness):
Uses competition-style coding problems published after model training cutoffs. Guards against benchmark leakage (models memorizing HumanEval answers). Tests genuine code generation without training contamination. Metric: `pass@1`.

**terminal_bench_2** (harbor, Docker):
89 curated tasks: protein assembly, async debugging, security vulnerability analysis, system admin ops. Each task runs in its own Docker container. The model gets a terminal and has to complete the task. Tests agentic terminal competence — exactly what our orchestrator workers do. Metric: `task_success_rate`.

**bfcl_v4** (bfcl harness):
Tests function/tool calling accuracy: simple calls, parallel calls, multi-turn, multi-step. V4 adds web search, memory management, format sensitivity. Tests tool use reliability — directly relevant to the rig's task execution model. Metrics: `ast_accuracy`, `exec_accuracy`.

## Backend Certification Rule

Certify backend/test combinations before assuming a suite is runnable.

Certification command:

```bash
python3 /media/bryan/shared/scripts/benchmarks/certify_benchmark_backend.py \
  --id gsm8k \
  --model local-chat-completions \
  --model-args "model=qwen2.5:7b,base_url=http://localhost:11436/v1/chat/completions,api_key=ollama"
```

Catalog audit command:

```bash
python3 /media/bryan/shared/scripts/benchmarks/audit_benchmark_catalog.py
```

This writes a machine-readable coverage summary showing which catalog tests are:
- supported
- blocked
- env_blocked
- missing_task
- missing_runner
- not_audited

## Model Library

Shared archive location:
- `/media/bryan/shared/models/`

Rig-local hot-set is managed explicitly with:
- `/media/bryan/shared/scripts/manage_model_hotset.py`

### Current Model Inventory

Archive verified against `/media/bryan/shared/models/` on 2026-03-05.

#### 6GB worker tier

| Model | Archive Family | Typical Placement | Status |
|------|----------------|-------------------|--------|
| `qwen2.5-coder:7b` | `qwen2.5-coder-7b` | single GPU worker | tested |
| `deepseek-r1:7b` | `deepseek-r1-7b` | single GPU worker | downloaded |
| `mistral:7b-instruct` | `mistral-7b-instruct` | single GPU worker | available |
| `qwen3.5:4b` | `qwen3.5-4b` | single GPU worker | downloaded |
| `qwen3.5:9b-q3km` | `qwen3.5-9b` | single GPU worker | downloaded |

#### 12GB paired-worker tier

| Model | Archive Family | Typical Placement | Status |
|------|----------------|-------------------|--------|
| `qwen2.5-coder:14b` | `qwen2.5-coder-14b` | split pair | tested |
| `deepseek-r1:14b` | `deepseek-r1-14b` | split pair | downloaded |
| `phi4:14b` | `phi-4-14b` | split pair | downloaded |
| `gemma3:12b` | `gemma-3-12b` | split pair | downloaded |

#### 24GB brain tier

| Model | Archive Family / Source | Typical Placement | Status |
|------|--------------------------|-------------------|--------|
| `qwen2.5-coder:32b` | `qwen2.5-coder-32b` | brain | tested |
| `deepseek-r1:32b` | `deepseek-r1-32b` | brain | downloaded |
| `qwen3.5:27b` | Ollama pull | brain | testing |
| `qwen3.5:35b-a3b` | `qwen3.5-35b-a3b` | brain | downloaded |

#### embedding

| Model | Archive Family | Purpose |
|------|----------------|---------|
| `nomic-embed-text` | `nomic-embed-text` | embeddings |

### Model Storage Rules

- shared drive is archive storage
- local Ollama storage is hot-set storage
- promote and evict models explicitly
- do not leave benchmark leftovers sprawled across endpoints

Useful commands:

```bash
python3 /media/bryan/shared/scripts/manage_model_hotset.py report
python3 /media/bryan/shared/scripts/manage_model_hotset.py dedupe --prune-imports
python3 /media/bryan/shared/scripts/manage_model_hotset.py promote-gguf --gguf /media/bryan/shared/models/<family>/<file>.gguf --name <model:tag> --host 127.0.0.1:11436
python3 /media/bryan/shared/scripts/manage_model_hotset.py evict --host 127.0.0.1:11436 --keep <model:tag>
```

## Test Library

Canonical machine-readable definitions:
- `/media/bryan/shared/scripts/benchmarks/benchmark_catalog.json`
- `/media/bryan/shared/scripts/benchmarks/suite_presets.json`

### Current Test Categories

| Category | Examples | Why We Run It |
|---------|----------|---------------|
| baseline reasoning | `gsm8k`, `arc_challenge`, `bbh`, `gpqa_diamond` | compare reasoning quality across models |
| QA and knowledge | `mmlu`, `boolq`, `drop`, `truthfulqa_mc2`, `mmmlu` | measure general reliability and breadth |
| coding | `humaneval_plus`, `mbpp_plus`, `livecodebench`, `swe_bench_verified` | measure coding and repo-fix ability |
| tool and agent behavior | `bfcl_v4`, `terminal_bench_2` | measure tool use and terminal competence |
| long context | `longbench`, `ruler` | measure long-context limits |
| local pipeline reliability | `custom_json_schema_strict`, `custom_tool_plan_sequence`, `custom_command_safety`, `custom_long_context_extract` | measure actual rig-specific behavior |

### Current Custom-Test State

- runnable baseline custom tests now exist for:
  - `custom_json_schema_strict`
  - `custom_command_safety`
  - `custom_ambiguity_handling`
- these are now executable through `run_local_custom_task.py` and recorded in the shared benchmark ledger
- current recorded baseline results show:
  - `custom_command_safety` can already serve as a useful reliability signal
  - `custom_ambiguity_handling` can already serve as a useful reliability signal
  - `custom_json_schema_strict` is useful, but still exposes formatting failures like fenced JSON
- the custom suite still needs expansion before it can be treated like a broad reliability benchmark
- next missing custom implementations:
  - `custom_tool_plan_sequence`
  - `custom_orchestration_tradeoff`
  - `custom_long_context_extract`

### Current Suite Presets

| Preset | Purpose |
|--------|---------|
| `baseline_core` | balanced baseline comparison |
| `fast_smoke` | quick regression pass |
| `reasoning_heavy` | hard reasoning focus |
| `coding_heavy` | coding-focused evaluation |
| `agent_reliability` | local orchestration reliability checks |

Build a suite file:

```bash
python3 /media/bryan/shared/scripts/benchmarks/build_benchmark_suite.py \
  --preset baseline_core \
  --output /media/bryan/shared/scripts/benchmarks/suites/baseline_core.json
```

## Benchmark Program Workflow

The benchmark program is:
1. maintain model inventory
2. maintain test inventory
3. run controlled benchmark batches
4. record results centrally
5. feed those results back into plan model selection

That feed-back loop currently uses:
- `/media/bryan/shared/scripts/benchmarks/model_task_library.json`
- `/media/bryan/shared/scripts/benchmarks/recommend_plan_models.py`

## Current State

For the exact current operating steps, use [CURRENT_BENCHMARK_PROCEDURE.md](CURRENT_BENCHMARK_PROCEDURE.md).

For prior failures, drift, and why the procedure is strict now, use [BENCHMARK_HISTORY.md](BENCHMARK_HISTORY.md).
