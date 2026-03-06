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

## Backend Certification Rule

Do not assume every catalog test is runnable on the current Ollama stack.

Current policy:
- certify backend/test combinations first
- leave uncertified tests blank until probed
- only treat model behavior as model-specific when the failure is clearly not backend-wide

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

Current backend-profile lessons:
- `local-chat-completions` is not one thing
- raw chat-completions and templated chat-completions must be tracked as separate profiles
- current working standardized lane is:
  - `ollama_chat_completions_templated`
- current broken standardized lane is:
  - `ollama_completions` for MC/loglikelihood-style tasks

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
