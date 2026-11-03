# Distributed Work Guide

How work moves through the current orchestration system, from plan creation to
runtime review.

For operator commands, use [quickstart.md](quickstart.md). For structure and
contracts, use [architecture.md](architecture.md), [PLAN_FORMAT.md](PLAN_FORMAT.md),
and [brain-behavior.md](brain-behavior.md).

---

## Workflow

1. Planning:
   - a plan folder is created under `shared/plans/arms/` or `shared/plans/shoulders/`
   - `plan.md` defines tasks, dependencies, and required artifacts
2. Submission:
   - operator submits through `scripts/submit.py` or the dashboard `Start Plan` action
   - normal runs do not use direct `/mnt/shared/agents/submit.py`
3. Brain interpretation:
   - brain parses `plan.md`
   - all tasks start in `shared/brain/private_tasks/`
   - ready tasks are released to `shared/tasks/queue/`
4. Worker execution:
   - GPU or CPU workers claim eligible tasks
   - workers manage local execution and report results
   - brain-issued `meta` tasks handle runtime changes like model load/unload
5. Brain monitoring:
   - brain watches completion/failure lanes
   - retries, requeues, quarantines, and shared runtime cleanup stay brain-owned
   - brain refreshes per-run summary artifacts during execution
6. Review:
   - inspect `history/<batch_id>/RUN_SUMMARY.json` or `RUN_SUMMARY.md`
   - for older or partial runs, use the history summary tools to reduce context first

---

## Submission Path

Preferred CLI:

```bash
python3 ~/llm_orchestration/scripts/submit.py \
  ~/llm_orchestration/shared/plans/<shoulder-or-arm>/<plan_name> \
  --config '{"KEY":"VALUE"}'
```

Equivalent operator UI:

- dashboard `Start Plan`

This creates an `execute_plan` task that the brain claims and expands into the
real task graph.

---

## Runtime Ownership

- brain owns queue truth, dependency release, retries, requeues, quarantine,
  and shared runtime authority
- workers own local process execution, probes, and child cleanup
- workers report observations upward; the brain decides shared-state changes

That keeps the brain authoritative without turning it into a per-step worker
controller.

---

## Batch Artifacts

Each batch writes under:

```text
shared/plans/<plan_name>/history/<batch_id>/
  logs/
    batch_events.jsonl
  output/
  results/
  RUN_SUMMARY.json
  RUN_SUMMARY.md
```

Legacy artifacts like `EXECUTION_SUMMARY.md` or `execution_stats.json` may
still exist for older runs, but the runtime-owned summary path is now
`RUN_SUMMARY.*`.

---

## Review Path

For one batch:

```bash
python3 ~/llm_orchestration/scripts/summarize_history_run.py \
  ~/llm_orchestration/shared/plans/<plan>/history/<batch_id>
```

For many batches:

```bash
python3 ~/llm_orchestration/scripts/rollup_history.py \
  ~/llm_orchestration/shared/plans/<plan>/history
```

These reducers shrink the evidence surface before an LLM reads it. They do not
replace review or make fix decisions.

---

## Compute Tiers

The system currently uses a 3090 Ti for larger models, five GTX 1060s for
worker and split loads, eight Orange Pi Prime CPU workers, and a Pi 5 control
plane. Use the [hardware upgrades plan](future/hardware_upgrades.md) for the
detailed inventory, measured split results, and future hardware choices.
This guide covers task submission, runtime ownership, and routing.

## CPU Inference Setup

llama.cpp built from source on the Pi 5, CPU-only (no CUDA).

Start a server:
```bash
/home/bryan/llama.cpp/build/bin/llama-server \
  --model /media/bryan/shared/models/<model-dir>/<model>.gguf \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 4096 --threads 4
```

Run benchmarks from the rig against the Pi:
```bash
docker run --rm --network host ... bench-pipeline \
  --model <model-id> \
  --runtime-base http://<pi-ip>:8080 \
  --run-name <run_name>
```

Notes:
- Models load over NFS from shared drive (30–60s first load)
- Add `--reasoning-budget 0` for Qwen 3.6 family (doesn't work for all model families)
- See [MODEL_RUNTIME_GUIDE.md](/media/bryan/shared/plans/shoulders/benchmarking/MODEL_RUNTIME_GUIDE.md) for per-model runtime flags

---

## Task Routing Matrix

Which worker tier and model for each task type. Based on benchmark data from [MODEL_LIBRARY.md](/media/bryan/shared/plans/shoulders/benchmarking/MODEL_LIBRARY.md).

| Task type | Recommended tier | Model | Why | Fallback |
|-----------|-----------------|-------|-----|----------|
| Code generation | GPU worker | `qwen2.5-coder:7b` / `:14b` | Best code scores, needs fast iteration | Brain (Qwen3.6-27B) for hard problems |
| Code review | GPU worker (split) | `qwen2.5-coder:14b` | Stronger reasoning for bug-finding | `qwen2.5-coder:7b` |
| Text extraction / QA | GPU worker | `gemma-4:e2b` | Best DROP score in worker tier, 8× faster than Qwen on reading tasks | `gemma-4:e4b` |
| Complex reasoning | GPU brain | `qwen3.6:27b` | Best BBH, GSM8K, DROP across all models | `qwen3.6:35b-a3b` (faster) |
| Structured JSON output | GPU worker | `qwen2.5-coder:7b` | Best json_schema score in worker tier | `qwen2.5-coder:14b` |
| Safe tool execution | GPU worker (split) | `phi-4:14b` | 100% cmd_safety, tool_plan, long_context | `qwen3.6:27b` |
| Orchestration decisions | GPU brain | `gemma-4:26b-a4b` | Best orch_tradeoff + ambiguity handling | `qwen3.6:27b` |
| Batch classification | CPU script workers | Sub-1B model or rules | High parallelism, no quality need | GPU if latency matters |
| Embedding generation | CPU script workers | `nomic-embed-text` | Tiny model, 8× parallel throughput | Single GPU node |
| Data prep / file ops | CPU script workers | None (scripted) | No model needed, pure compute | Any available node |
| Edge / RPi deployment | CPU LLM (Pi 5) | `smollm3:3b` | Best reasoning in 3B tier, fits 3GB | `llama3.2:3b` |

### How to add a new task type

1. Identify the capability needed (code, reasoning, extraction, classification, scripted)
2. Check [BENCHMARK_SCORES.md](/media/bryan/shared/plans/shoulders/benchmarking/BENCHMARK_SCORES.md) for which models score highest on the relevant benchmark
3. Check [MODEL_LIBRARY.md](/media/bryan/shared/plans/shoulders/benchmarking/MODEL_LIBRARY.md) "What Scores Mean for Real Tasks" to map benchmark → capability
4. Pick the cheapest tier that can run the model (worker GPU > brain GPU > CPU LLM)
5. Update the job routing (`jobs`) or the role in [model_roles.json](/home/bryan/llm_orchestration/shared/plans/shoulders/benchmarking/model_roles.json), then run `scripts/active/model_library.py check`
6. Test with a representative workload before committing to production routing

---

## Related Docs

| Doc | Purpose |
|-----|---------|
| [quickstart.md](quickstart.md) | operator commands and recovery flow |
| [PLAN_FORMAT.md](PLAN_FORMAT.md) | authoritative plan format and task structure |
| [architecture.md](architecture.md) | system layout and authority boundary |
| [brain-behavior.md](brain-behavior.md) | runtime coordinator behavior |
| [history_summary_tools.md](history_summary_tools.md) | one-run and many-run reducers |
| [cpu-worker-setup.md](cpu-worker-setup.md) | Orange Pi CPU worker provisioning and ops |
| [hardware_upgrades.md](future/hardware_upgrades.md) | current hardware, benchmarks, upgrade scenarios, and unified-memory options |
| [MODEL_LIBRARY.md](/media/bryan/shared/plans/shoulders/benchmarking/MODEL_LIBRARY.md) | model selection, benchmark scores, best choices per task |
| [MODEL_RUNTIME_GUIDE.md](/media/bryan/shared/plans/shoulders/benchmarking/MODEL_RUNTIME_GUIDE.md) | per-model runtime flags, memory, compatibility |
| [BENCHMARK_SCORES.md](/media/bryan/shared/plans/shoulders/benchmarking/BENCHMARK_SCORES.md) | raw benchmark score tables with timing |
| [model_roles.json](/home/bryan/llm_orchestration/shared/plans/shoulders/benchmarking/model_roles.json) | model + settings per lane and job (routing), remote profiles |
