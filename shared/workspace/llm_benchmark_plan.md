# LLM Benchmark Plan — 7B vs 14B Worker Strategy

## Goal

Determine whether to run 5x 7B workers, or split into 2x paired 14B + 1x 7B (or other combos). Results feed into GPU allocation strategy and potential paired-model feature.

## Hardware (Worker GPUs)

- Brain: RTX 3090 Ti (24GB) — GPU 0, holds 32B (qwen2.5:32b)
- Workers: 5x GTX 1060 6GB (GPUs 1-5) — three VBIOS variants, pairs (1,3) and (4,5) confirmed
- Models: Qwen2.5-Coder 7B Q4_K_M (~4.7GB), Qwen2.5-Coder 14B Q4_K_M (~9GB)

## Existing Scripts

| Script | Purpose |
|--------|---------|
| `scripts/worker-benchmark.py` | Tests 7B model on individual GPUs for worker role |
| `scripts/gpu-pair-benchmark.py` | Tests GPU pairs for multi-GPU LLM inference (brain model splitting) |
| `scripts/gpu-stress.py` | Runs all GPUs at max load for thermal/stability testing |
| `scripts/gpu-monitor.py` | Logs temps, power, clocks, throttle status |
| `scripts/embedding-stress.py` | GPU stress simulating embedding workloads |
| `shared/scripts/context_window_benchmark.py` | Measures practical context capacity per worker GPU via Ollama |

Previous results archived at:
- `workspace/archive/worker_benchmark_results.json`
- `workspace/archive/gpu_pair_benchmark_results.json`
- `shared/logs/context_window_benchmark_*.json` (multiple runs from Feb 13)
- `shared/logs/brain_context_benchmark_*.json`

Also see: `workspace/llm_benchmark_testing_guide.md` for the existing testing guide.

## Benchmark Tests

### Test 1: Raw Speed (tokens/sec)

Load each model on a single GPU, run identical prompts, measure:
- Tokens/sec generation
- Time to first token
- Total completion time for fixed output length (500 tokens)

Run 3x each, take median. Test on each physical GPU since cards may differ.

### Test 2: JSON Parse Success Rate

Run real prompts from the prospector pipeline, 50 iterations each:
- `plan_searches` prompt (generate search queries JSON)
- `score_person` prompt (extract profile + score JSON)
- `triage_extract` prompt (structured profile extraction)

Measure: % of responses that parse as valid JSON without retry.

### Test 3: Output Quality Comparison

Pick 10 real candidates. Run `score_person` through both models on the same scraped input. Compare:
- Profile completeness (fields filled vs empty)
- Score accuracy (manual review: did it score correctly?)
- Hallucination rate (did it invent data not in the source?)

### Test 4: VRAM & Concurrency

- Can 2x 7B run on one card simultaneously? (2 Ollama instances, different ports)
- What's the VRAM headroom with 14B on each card size?
- Does layer splitting across 2 GPUs for 14B actually work with Ollama? Measure overhead.

### Test 5: Throughput (End-to-End)

Run a batch of 20 candidates through the full prospector pipeline:
- Config A: 5x 7B workers (max parallelism)
- Config B: 2x 14B paired + 1x 7B (less parallelism, higher quality)
- Config C: 2x 14B paired + 1x 7B, with task-class routing (7B does plan_searches, 14B does score_person)

Measure: total batch wall-clock time, output quality score.

## Decision Matrix

| If... | Then... |
|-------|---------|
| 7B JSON failure rate >15% and 14B <5% | 14B is worth the parallelism cost for judgment tasks |
| 7B quality is "good enough" across all tasks | Stay with 5x 7B, max throughput |
| 14B quality is notably better but only on scoring | Split strategy: 7B for plan_searches, 14B for score/triage |
| Layer splitting adds >20% overhead | Pairs aren't worth it, run 14B on single cards that can fit it |

---

## Benchmark Results (2026-02-16)

### Hardware Identified

**Brain:** RTX 3090 Ti (24GB) — GPU 0, runs 32B model (qwen2.5:32b, 19.9GB)

**Workers:** 5x GTX 1060 6GB (GPUs 1-5), three VBIOS variants:
| Variant | GPUs | VBIOS |
|---------|------|-------|
| Model A | 1, 3 | 86.06.13.00.28 |
| Model B | 2    | 86.06.27.00.9B |
| Model C | 4, 5 | 86.06.63.00.60 |

### Test 1: Raw Speed (Dedicated GPU Comparison)

Simultaneous test with each model on its dedicated GPU(s):

| Model | GPU(s) | Tokens | Speed |
|-------|--------|--------|-------|
| 7B    | 2 (single) | 553 | **22.5 tok/s** |
| 14B   | 4,5 (paired) | 560 | **12.5 tok/s** |

**Ratio: 7B is ~1.8x faster than paired 14B**

### Test 2: 14B Paired GPU Performance

Layer splitting across 2x 1060s for qwen2.5-coder:14b:

| GPU Pair | Variant Match | Avg tok/s | Notes |
|----------|---------------|-----------|-------|
| (1, 3)   | Same (A+A)    | ~12.5     | 26+23 layer split |
| (4, 5)   | Same (C+C)    | ~12.5     | Same performance |

**Finding:** Both same-variant pairs work. 14B is ~1.8x slower than 7B (expected given 2x parameter count + PCIe overhead).

### Test 3: Quality Comparison (7B vs 14B)

| Test | 14B Result | 7B Result | Winner |
|------|------------|-----------|--------|
| **Bug Finding** (factorial return 0 bug) | ✓ Correct fix (return 1) | ⚠️ Wrong fix (added elif, kept return 0) | 14B |
| **Multi-step Reasoning** (sheep problem) | ✓ Correct (9 sheep) | ✓ Correct (9 sheep) | Tie |
| **JSON Parsing** (server status) | ⚠️ Hallucinated extra server | ✓ Clean, correct JSON | 7B |

**Speed comparison:** 7B ~22.5 tok/s vs 14B ~12.5 tok/s = **~1.8x faster**

### Conclusions

1. **Layer splitting works** — 14B runs successfully on 2x 1060 6GB cards
2. **Quality is task-dependent** — 14B better for code analysis, 7B better for structured output/instruction following
3. **Speed tradeoff is moderate** — 7B is ~1.8x faster than paired 14B (22.5 vs 12.5 tok/s)
4. **Same-variant pairing** — GPUs (1,3) and (4,5) confirmed working pairs by VBIOS match

### Recommended Configuration

| Role | GPUs | Model | Throughput |
|------|------|-------|------------|
| Brain | 0 (3090 Ti) | qwen2.5:32b | ~20 tok/s (est) |
| 14B Pair | 4, 5 | qwen2.5-coder:14b | ~12.5 tok/s |
| 7B Workers | 1, 2, 3 | qwen2.5-coder:7b | ~67.5 tok/s (3 × 22.5) |

**Tradeoff:** 14B is only 1.8x slower but may offer better reasoning on complex tasks. Use 14B for code analysis, 7B for structured extraction. Brain (32B) handles orchestration.

---

## Future: Task-Level Model Tagging

Regardless of benchmark outcome, add model tier support to task_class:

```
task_class: llm_7b    # lightweight — query gen, simple extraction
task_class: llm_14b   # heavyweight — scoring, judgment, complex extraction
task_class: llm       # any available LLM worker (backwards compat)
```

Brain routes tasks to workers based on what model they have loaded. Plans declare what they need, brain maps to what's available. If a plan requests `llm_14b` but only 7B is loaded, brain can:
- Downgrade and note it
- Issue a `load_llm` meta-task to swap models first

This is useful beyond the 7B/14B split — future GPU additions with more VRAM could run larger models, and the routing just works.

## Future: Paired GPU Model Loading

### Concept

Two GPUs cooperate to run a model too large for either alone (e.g. 14B across 2x 8GB cards). Brain orchestrates the pairing via meta-tasks.

### Config

Benchmark results determine which GPUs can pair:

```json
{
  "gpu_pairs": [
    {"gpus": [1, 3], "shared_model": "qwen2.5-coder:14b", "ollama_port": 11440},
    {"gpus": [4, 5], "shared_model": "qwen2.5-coder:14b", "ollama_port": 11441}
  ],
  "solo_workers": [2]
}
```

### Load Sequence

1. Brain decides it needs 14B capacity (heavy tasks in queue)
2. Brain posts `load_14b` meta-task targeting pair [1, 3] — meta-tasks are highest priority
3. GPU 1 finishes its current task, returns to task board, sees high-priority `load_14b`, claims it
4. GPU 1 unloads its 7B model, writes ready signal, waits
5. GPU 3 finishes its current task, returns to task board, sees `load_14b`, claims it
6. GPU 3 unloads its 7B model, writes ready signal
7. Both ready — shared Ollama instance starts on configured port with `CUDA_VISIBLE_DEVICES=1,3`
8. Pair starts claiming `llm_14b` tasks from queue

### Rendezvous Handshake

Uses existing signal filesystem:

```json
// shared/signals/pair_1_3.json
{
  "state": "loading",
  "gpu_1": "ready",      // GPU 1 checked in
  "gpu_3": "pending",    // GPU 3 still finishing previous task
  "target_model": "qwen2.5-coder:14b",
  "ollama_port": 11440
}
```

When both are `"ready"`, whichever writes last triggers the Ollama launch.

### Unload Sequence

1. Brain decides 14B capacity no longer needed (heavy tasks done)
2. Brain posts `unload_14b` meta-task
3. Paired agent stops the shared Ollama instance
4. Both GPUs revert to independent 7B workers
5. Handshake file cleared

### Priority

Meta-tasks (`load_llm`, `unload_llm`, `load_14b`, `unload_14b`) are already highest priority in the brain's task ordering. A GPU finishing its current task will always grab the meta-task before picking up another regular LLM task. No special priority logic needed.

### Open Questions

- ~~Does Ollama layer splitting actually work well across 2 cards?~~ **YES** — confirmed working on (1,3) and (4,5) pairs
- ~~Latency overhead of cross-GPU inference vs single-GPU?~~ **~1.8x slower** — 7B: 22.5 tok/s, 14B paired: 12.5 tok/s
- Should the "pair agent" be a third process, or does one of the two GPU agents take the lead?
- What happens if one GPU in a pair crashes mid-task? Recovery strategy needed.
