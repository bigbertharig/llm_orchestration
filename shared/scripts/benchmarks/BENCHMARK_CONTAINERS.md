# Benchmark Containers

Each container is a self-contained test environment. Load a container, run all its tests, get results, clean up.

The containers replace the old suite presets (baseline_core, fast_smoke, etc.) which split tests across environments. Now each container IS the suite.

## What We're Trying To Learn

This rig runs an LLM orchestrator: a brain model (32B on RTX 3090) dispatches tasks to worker models (7B on GTX 1060 6GB cards). We need to know:
- Which worker model handles which task type best (so we assign work correctly)
- How reliably models follow structured formats (so pipelines don't break)
- Whether models make safe decisions (so workers don't run dangerous operations)
- Where the quality cutoff is between models (so we pick the right one for each job)

Every test below ties back to one of those questions. If a test doesn't help answer them, it doesn't belong here.

## GPU Compatibility

| GPU | Compute Cap | Count | Role |
|-----|------------|-------|------|
| RTX 3090 Ti | sm_86 | 1 | Brain (not the primary benchmark target) |
| GTX 1060 6GB | sm_61 | 5 | Workers (the models being benchmarked) |

Benchmark focus is the 1060 worker tier. The brain runs the 32B model and is essentially guaranteed to outperform workers — we don't need to prove that.

PyTorch (current version) requires sm_70+ and does not support the 1060s. llama.cpp supports both sm_61 and sm_86 when compiled with `-DCMAKE_CUDA_ARCHITECTURES="61;86"`. This is why containers that need GPU inference use llama.cpp, not PyTorch.

## Model Loading — Two Strategies

All models are GGUF format stored on the shared drive at `/mnt/shared/models/`.

| Strategy | Containers | How It Works |
|----------|-----------|-------------|
| **Ollama on host** | 1, 3, 4, 5 | Container calls Ollama via `--network host`. No GPU in container. Model must be loaded in host Ollama first. |
| **llama.cpp in container** | 2 | Container runs llama-cpp-python server, loads GGUF from mounted shared drive, serves `/v1/completions` with logprobs. Needs `--gpus` for GPU passthrough. |

### Confirmed: llama.cpp server returns logprobs

Tested 2026-03-06: llama-cpp-python 0.3.16 serving a GGUF via `/v1/completions` returns full `token_logprobs` and `top_logprobs` arrays with `echo: true`. This is exactly what lm_eval's built-in `gguf` model type expects. End-to-end test passed: `lm_eval --model gguf → llama.cpp server → boolq loglikelihood scoring → acc: 0.3333 on 3 samples`.

### Confirmed: lm_eval has a native `gguf` model type

No need for `local-completions` or `openai_completions` model types. The `gguf` model type sends string prompts (not arrays), requests `logprobs: 10`, and handles `echo: true` for loglikelihood. This bypasses all Ollama compatibility issues.

### Confirmed: mmlu_pro and ifeval are `generate_until` tasks

Both use generation (not loglikelihood), so they run on Ollama via the standard `local-chat-completions` path. No need for lighteval. Container 4 (bench-instruction) was eliminated — these tests merged into container 1.

### GPU build requirement for llama-cpp-python

The Docker image for container 2 must compile llama-cpp-python with CUDA support targeting both GPU architectures:
```
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=61;86" pip install llama-cpp-python
```
CUDA toolkit goes inside the container (not needed on the host). The host only needs the NVIDIA driver (installed) and nvidia-container-toolkit (needs install).

### Confirmed: GPU test passed on GTX 1060

Tested 2026-03-06: Docker container with CUDA-compiled llama-cpp-python (ARCHS=610,860) loaded qwen2.5-coder-7b Q4_K_M onto a single GTX 1060 6GB. All 29/29 layers offloaded to GPU. Prompt eval: ~160 tokens/sec. lm_eval `gguf` model type scored boolq loglikelihood end-to-end. Container 2 is fully validated.

---

## Container 1: `bench-reasoning`

**What it answers:** How well does each model think through problems and follow instructions?

**Backend:** Ollama on host (generation via chat completions with `--apply_chat_template`)
**GPU in container:** No — calls host Ollama
**lm_eval model type:** `local-chat-completions`
**Key packages:** lm_eval, transformers

| Test | Type | What We Learn | Why We Care |
|------|------|--------------|-------------|
| **gsm8k** | generation | Grade-school math — can the model chain 3-5 arithmetic steps correctly? | Workers get tasks that require sequential reasoning. If a model can't chain basic steps, it can't plan a multi-step task. Baseline signal. |
| **bbh** | generation | BIG-Bench Hard — 23 diverse hard reasoning tasks (logical deduction, causal reasoning, disambiguation). | Broad reasoning stress test. A model that scores well here can handle varied task types without specialization. |
| **drop** | generation | Reading comprehension with counting, sorting, arithmetic over paragraphs. | Workers parse logs, configs, and documents. DROP tests whether the model can extract and compute from natural language — exactly what happens when a worker reads task output. |
| **mmlu_pro** | generation | Harder MMLU variant with 10 answer choices instead of 4, plus chain-of-thought. | Separates models that genuinely reason from models that get lucky on easier tests. Useful tier separator between model families. |
| **ifeval** | generation | Can the model follow specific formatting instructions? ("Write exactly 3 paragraphs", "Include the word X", "Respond in JSON") | Workers receive structured task definitions. If a model can't follow explicit formatting constraints, it will break pipelines that parse its output. Directly relevant to orchestrator reliability. |

Note: ifeval requires `langdetect` package (missing from current ml-env, install in container).

### Tests removed
| Test | Why Removed |
|------|-------------|
| math_500 | Competition math — much harder than what workers encounter. gsm8k covers the reasoning chain signal we need. |
| aime_2024 | Math olympiad difficulty. Interesting but not actionable — no worker task requires this. |

---

## Container 2: `bench-knowledge`

**What it answers:** How much does each model know, and does it hallucinate?

**Backend:** llama.cpp server inside container (needs logprobs for MC scoring)
**GPU in container:** Yes — loads GGUF from shared drive mount, serves `/v1/completions` with logprobs
**lm_eval model type:** `gguf` (native, sends string prompts, requests logprobs)
**Key packages:** lm_eval, transformers, llama-cpp-python[server] (built with CUDA sm_61;86)
**Mounts:** `/mnt/shared/models/` read-only, `/mnt/shared/logs/benchmarks/` for results

| Test | Type | What We Learn | Why We Care |
|------|------|--------------|-------------|
| **mmlu** | MC/loglikelihood | Broad knowledge across 57 academic subjects. | General capability thermometer. A model's MMLU score predicts how well it handles diverse tasks without domain-specific prompting. |
| **arc_challenge** | MC/loglikelihood | Science questions requiring non-trivial reasoning. | Tests whether the model can reason about cause/effect — relevant for workers that debug failures or analyze system behavior. |
| **hellaswag** | MC/loglikelihood | Commonsense completion — which continuation is plausible? | Workers must make sensible next-step decisions. A model that fails hellaswag makes implausible choices. |
| **truthfulqa_mc2** | MC/loglikelihood | Does the model avoid common misconceptions and confidently-wrong answers? | Workers that hallucinate confidently are worse than workers that say "I don't know." This directly measures hallucination tendency. |
| **boolq** | MC/loglikelihood | Simple yes/no questions over short passages. | Fast signal for basic reading comprehension. If a model fails boolq, it fails everything harder. |

### Tests removed
| Test | Why Removed |
|------|-------------|
| winogrande | Coreference resolution. Niche linguistic skill, not a meaningful differentiator for task assignment. |
| piqa | Physical commonsense. Our workers don't reason about the physical world. |
| gpqa_diamond | PhD-level science. Gated dataset, and way beyond what 7B models need for orchestrator tasks. |
| mmmlu | Multilingual MMLU. Our rig operates in English only. |
| musr | Multi-step soft reasoning. Overlaps with bbh and arc_challenge without adding a distinct signal. |

---

## Container 3: `bench-code`

**What it answers:** Can this model write correct code?

**Backend:** Ollama on host (generates code), container runs Python test cases locally
**GPU in container:** No — calls host Ollama
**Key packages:** evalplus, openai

| Test | Type | What We Learn | Why We Care |
|------|------|--------------|-------------|
| **humaneval_plus** | code gen + exec | 164 Python problems with hardened test cases. Can the model write a function that actually works? | Workers write scripts, patches, and automation. pass@1 on HumanEval+ is the most direct measure of "will the code this model writes actually run?" |
| **mbpp_plus** | code gen + exec | 974 beginner-to-intermediate Python problems. Broader coverage than HumanEval. | More problems = more stable signal. HumanEval is small enough that a few lucky/unlucky problems swing the score. MBPP+ gives a more reliable ranking. |

### Tests removed
| Test | Why Removed |
|------|-------------|
| livecodebench | Anti-contamination coding bench. Valuable concept but heavy setup. Revisit later (container 7). |
| swe_bench_verified | Full repo bug-fixing. Most relevant test for our use case but requires Docker-in-Docker. Defer (container 6). |

---

## Container 4: `bench-pipeline`

**What it answers:** Does this model work reliably in OUR specific pipeline?

**Backend:** Ollama on host (matches production runtime exactly)
**GPU in container:** No — calls host Ollama
**Key packages:** openai, custom test harness

| Test | Type | What We Learn | Why We Care |
|------|------|--------------|-------------|
| **custom_json_schema_strict** | custom | Given an extraction prompt, does the model return valid JSON matching an exact schema — no retries, no fenced markdown? | Every structured task in the orchestrator depends on parseable JSON output. This is the single most important reliability signal for production. |
| **custom_tool_plan_sequence** | custom | Given a multi-step task with ordered dependencies, does the model produce a valid execution plan with correct step ordering? | The brain model builds task plans. If it misoriders steps or drops dependencies, the whole pipeline breaks. |
| **custom_command_safety** | custom | Given a risky shell command, does the model flag it and suggest a safe alternative? | Workers execute shell commands. A model that runs `rm -rf /` instead of flagging it is a production hazard. |
| **custom_ambiguity_handling** | custom | When a task is ambiguous, does the model ask for clarification instead of guessing? | Workers that guess wrong waste GPU time and produce bad output. We want models that escalate ambiguity, not hide it. |
| **custom_orchestration_tradeoff** | custom | Given resource constraints (GPU memory, model sizes, task priorities), does the model make good allocation decisions? | The brain model makes exactly these decisions. This test directly measures brain-model quality. |
| **custom_long_context_extract** | custom | Given a long log file with mixed signals, can the model extract the specific requested information accurately? | Workers process logs, build outputs, and pull data from large documents. Hallucinating details from long context is a common failure mode. |

### Implementation status
- custom_json_schema_strict: implemented (2 cases), baseline results exist
- custom_command_safety: implemented (2 cases), baseline results exist
- custom_ambiguity_handling: implemented (2 cases), baseline results exist
- custom_tool_plan_sequence: implemented (3 cases) — tests dependency ordering and parallel recognition
- custom_orchestration_tradeoff: implemented (3 cases) — tests GPU sizing, priority, and memory fit decisions
- custom_long_context_extract: implemented (3 cases) — tests error extraction, detail extraction, and counting from logs

All 6 tests (15 total cases) defined in `custom_tasks/cases.json`. All use keyword grading except `custom_json_schema_strict` which uses schema validation.

---

## Containers 5-8: Deferred

These test important things but require heavy infrastructure. Build them after containers 1-4 are validated and running.

### Container 5: `bench-swebench`
**What it answers:** Can this model fix real bugs in real codebases?
Takes ~500 verified GitHub issues from popular Python repos (Django, Flask, scikit-learn). The model gets the issue + repo code and must write a patch that passes the repo's test suite. Runs in Docker because each issue needs its own clean repo clone + test runner. Tests agentic coding ability — reading large codebases, understanding issues, writing correct patches. Metric: `resolved_rate`.

### Container 6: `bench-livecodebench`
**What it answers:** Can this model write code it hasn't memorized from training?
Uses competition-style coding problems published after model training cutoffs. Guards against benchmark leakage (models memorizing HumanEval answers). Tests genuine code generation without training contamination. Metric: `pass@1`.

### Container 7: `bench-terminal`
**What it answers:** Can this model operate a terminal to complete real tasks?
89 curated tasks: protein assembly, async debugging, security vulnerability analysis, system admin ops. Each task runs in its own Docker container. The model gets a terminal and must complete the task. Tests agentic terminal competence — exactly what our orchestrator workers do. Metric: `task_success_rate`.

### Container 8: `bench-toolcall`
**What it answers:** Can this model call functions/tools correctly?
Tests function/tool calling accuracy: simple calls, parallel calls, multi-turn, multi-step. V4 adds web search, memory management, format sensitivity. Tests tool use reliability — directly relevant to the rig's task execution model. Metrics: `ast_accuracy`, `exec_accuracy`.

---

## Summary

| # | Container | Tests | GPU in Container | What It Tells Us |
|---|-----------|-------|-----------------|-----------------|
| 1 | bench-reasoning | gsm8k, bbh, drop, mmlu_pro, ifeval | No (Ollama on host) | Can it think and follow instructions? |
| 2 | bench-knowledge | mmlu, arc, hellaswag, truthfulqa, boolq | Yes (llama.cpp) | What does it know, and does it hallucinate? |
| 3 | bench-code | humaneval_plus, mbpp_plus | No (Ollama on host) | Can it write working code? |
| 4 | bench-pipeline | 6 custom tests | No (Ollama on host) | Does it work in OUR pipeline? |
| 5 | bench-swebench | swe_bench_verified | Yes | Can it fix real bugs? (deferred) |
| 6 | bench-livecodebench | livecodebench | No | Can it code without memorization? (deferred) |
| 7 | bench-terminal | terminal_bench_2 | Yes | Can it operate a terminal? (deferred) |
| 8 | bench-toolcall | bfcl_v4 | No | Can it call tools correctly? (deferred) |

**Active containers: 1-4 (17 tests total) — ALL BUILT AND SMOKE-TESTED**
**Deferred: 5-8**

### Container Build Status (2026-03-07)

| # | Image | Size | Smoke Test | Result |
|---|-------|------|-----------|--------|
| 1 | bench-reasoning | 969MB | gsm8k (3 samples) | exact_match=0.667 |
| 2 | bench-knowledge | 8.6GB | boolq (10 samples, GPU on 1060) | acc=0.2, ~160 tok/s |
| 3 | bench-code | 875MB | humaneval codegen (3 problems) | codegen OK, eval needs full set |
| 4 | bench-pipeline | 151MB | custom_command_safety | 2/2 cases passed, score=1.0 |

Dockerfiles and run scripts: `/mnt/shared/scripts/benchmarks/containers/<name>/`

Notes:
- Container 1 needs `lm_eval[api]` (not just `lm_eval`) for the tenacity dependency
- Container 3 (evalplus) cannot do partial evaluation — must run all 164/974 problems
- Container 2 is large because it includes CUDA devel toolkit for llama-cpp-python compilation

### Saved Images

Pre-built images saved to `/mnt/shared/scripts/benchmarks/containers/images/`:

| Image | Compressed Size |
|-------|----------------|
| bench-reasoning.tar.gz | 295MB |
| bench-code.tar.gz | 252MB |
| bench-pipeline.tar.gz | 53MB |
| bench-knowledge.tar.gz | 4.2GB |
| **Total** | **~4.8GB** |

To load a saved image: `docker load < bench-reasoning.tar.gz`
To save after rebuild: `docker save bench-reasoning | gzip > bench-reasoning.tar.gz`

**Future improvement:** `docker save` exports the full image every time (not a diff). As we add containers 5-8 and images grow, switch to a local Docker registry which only transfers changed layers. For now, 4.8GB of tarballs is manageable.

## Host Prerequisites

Before building containers:
1. ~~Install Docker Engine~~ **Done** — Docker 28.2.2 installed 2026-03-06
2. ~~Install nvidia-container-toolkit~~ **Done** — v1.18.2 installed 2026-03-06
3. Neither requires CUDA toolkit on the host — just the NVIDIA driver (already installed: 580.126.09)

### Verified 2026-03-06
- `docker run --gpus all` sees all 6 GPUs (1x 3090 Ti + 5x 1060 6GB)
- `docker run --gpus "device=1"` correctly isolates a single GTX 1060 (sm_61, 6144 MiB)
- CUDA 13.0 reported inside container (from host driver 580.126.09)
- User `bryan` added to `docker` group

## Running A Full Benchmark

```bash
# For each model, run all 4 containers in sequence:

# 1. Load model into host Ollama for containers 1, 3, 4
OLLAMA_HOST=http://localhost:11436 ollama run qwen2.5-coder:7b ""

# 2. Run reasoning tests (Ollama generation)
docker run --rm --network host \
  -v /mnt/shared/logs/benchmarks:/results \
  bench-reasoning --model qwen2.5-coder:7b

# 3. Run knowledge tests (llama.cpp with logprobs, GPU)
docker run --rm --gpus '"device=1"' \
  -v /mnt/shared/models:/models:ro \
  -v /mnt/shared/logs/benchmarks:/results \
  bench-knowledge /models/qwen2.5-coder-7b/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf

# 4. Run code tests (Ollama generation + local exec)
docker run --rm --network host \
  -v /mnt/shared/logs/benchmarks:/results \
  bench-code --model qwen2.5-coder:7b

# 5. Run pipeline tests (Ollama custom)
docker run --rm --network host \
  -v /mnt/shared/logs/benchmarks:/results \
  -v /mnt/shared/scripts/benchmarks:/benchmark-scripts:ro \
  bench-pipeline --model qwen2.5-coder:7b
```

Notes:
- Container 2 uses `--gpus '"device=1"'` to target a specific 1060. Adjust device index as needed.
- bench-code (evalplus) must run the full problem set — partial evaluation is not supported.

### Sample Limits

The full academic datasets have thousands to tens of thousands of questions per task. Running them all on a single 1060 would take days per model. That's not practical.

**Default: 150 samples per task** for containers 1 and 2. This is enough to rank models reliably (±3-5% accuracy) without burning excessive GPU time. Override with `--limit N` if needed.

| Container | Default Limit | Why |
|-----------|--------------|-----|
| 1 bench-reasoning | 150/task | 5 tasks × 150 = 750 generations. Reasonable. |
| 2 bench-knowledge | 150/task | 5 tasks × 150 × 2 requests = 1500 loglikelihood calls. Manageable. |
| 3 bench-code | none (full set) | HumanEval is only 164 problems. MBPP is 974 but evalplus needs the full set for evaluation. |
| 4 bench-pipeline | none (full set) | Only ~8 custom cases total. Minutes. |

Full dataset sizes for reference (do NOT run these without limits):

| Task | Full Size | At ~7s/sample | At ~25s/request |
|------|----------|---------------|-----------------|
| mmlu | 14,042 | — | ~97 hours |
| hellaswag | 10,042 | — | ~70 hours |
| drop | 9,536 | ~18 hours | — |
| bbh | 6,511 | ~13 hours | — |
| boolq | 3,270 | — | ~23 hours |
| gsm8k | 1,319 | ~2.5 hours | — |
| mbpp | 974 | ~5 hours | — |

## Chat Template Standardization

All benchmark runs must use the correct chat template for the model family.

| Model Family | HF Tokenizer | Template |
|-------------|-------------|----------|
| qwen2.5-coder | `Qwen/Qwen2.5-Coder-7B-Instruct` | ChatML |
| qwen2.5 | `Qwen/Qwen2.5-7B-Instruct` | ChatML |
| mistral | `mistralai/Mistral-7B-Instruct-v0.2` | Mistral [INST] |
| deepseek-r1 | TBD — verify on first run | TBD |

For Ollama containers (1, 3, 4): always pass `--apply_chat_template`.
For llama.cpp container (2): the `gguf` model type sends raw prompts — template is applied by lm_eval's tokenizer, not the server.
