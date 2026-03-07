# Benchmark Run Report (2026-03-05)

## Scope
- Goal: run individual benchmark tests across loaded models and prep benchmark environment for repeated model swaps.
- Rig: `10.0.0.3`
- Endpoints targeted:
  - Brain: `11434`
  - Single worker: `11436`
  - Split worker: `11441`

## Issues Found and Fixes Applied

1. `lm-eval` runner used deprecated task-list CLI
- Symptom: all runs failed with `Tasks not found: list`.
- Root cause: `run_lm_eval_task.py` used `python -m lm_eval --tasks list` (old interface).
- Fix:
  - Switched to `python -m lm_eval ls tasks`.
  - Updated parser to read table-formatted task output.
- File:
  - `/media/bryan/shared/scripts/benchmarks/run_lm_eval_task.py`

2. `run_lm_eval_task.py` used legacy run invocation assumptions
- Symptom: harness invocation mismatch and repeated early failures.
- Fix:
  - Updated command to explicit `lm_eval run`.
  - Added `--apply-chat-template` pass-through option.
  - Default backend moved to `local-chat-completions`.
- File:
  - `/media/bryan/shared/scripts/benchmarks/run_lm_eval_task.py`

3. Ollama API compatibility gap for loglikelihood/multiple-choice tasks
- Symptom:
  - `local-completions` + `/v1/completions` failed for tasks needing logprobs.
  - API model path mismatch for prompt format (array vs string).
- Root cause: Ollama OpenAI compatibility currently supports generation paths better than loglikelihood/logprobs paths for this harness setup.
- Outcome:
  - Ran the executable subset (`generate_until` path): `gsm8k`, `drop`, `bbh`.
  - Marked non-generation baseline IDs as blocked for current endpoint mode unless backend path changes.

4. Brain model endpoint mismatch (`11434`)
- Symptom: `model 'qwen2.5:32b' not found` via OpenAI endpoints even though model listed.
- Root cause: manifest/store state mismatch on brain runtime; attempted run triggered pull for missing blob.
- Outcome:
  - Continued matrix on worker/split endpoints to keep progress.
  - Brain model path requires dedicated repair before full 3-model parity runs.

5. Duplicate model storage consumed root disk
- Symptom: root filesystem nearly full.
- Root cause: duplicate blobs in `/usr/share/ollama/.ollama` and `/home/bryan/.ollama`, plus stale import artifact.
- Fix:
  - Removed duplicate root-store blobs that matched user store.
  - Removed stale `/var/lib/ollama-import/qwen2.5-coder-32b`.
- Result: root usage improved from `96%` to `66%` (approx `+31GB` free).

6. Requirements needed split by weight
- Symptom: risk of uncontrolled install footprint for optional heavy suites.
- Fix:
  - Added core/optional requirements split.
  - Kept `requirements.txt` as core wrapper.
- Files:
  - `/media/bryan/shared/scripts/benchmarks/requirements.txt`
  - `/media/bryan/shared/scripts/benchmarks/requirements-core.txt`
  - `/media/bryan/shared/scripts/benchmarks/requirements-optional.txt`

7. Dataset setup needed disk preflight and minor bug fix
- Symptom: no free-space guard before large dataset pulls; priority repo key typo.
- Fix:
  - Added `--min-free-gb` preflight (default `20GB`).
  - Fixed priority repo key to `terminal_bench_2`.
- File:
  - `/media/bryan/shared/scripts/benchmarks/setup_benchmark_datasets.py`

## Benchmarks Recorded (This Session)

Recorded to:
- `/mnt/shared/logs/benchmarks/model_benchmark_records.jsonl`
- `/mnt/shared/logs/benchmarks/MODEL_BENCHMARK_REFERENCE.md`

Successful recorded rows:
- `qwen2.5:7b`
  - `gsm8k`: `0.75` (`exact_match,flexible-extract`)
  - `drop`: `0.14875` (`f1,none`)
  - `bbh`: `0.555555...` (`exact_match,get-answer`)
- `qwen2.5-coder:14b`
  - `gsm8k`: `0.875` (`exact_match,flexible-extract`)
  - `drop`: `0.09875` (`f1,none`)

`qwen2.5-coder:14b` `bbh` run ended without score (interrupted during long subtask expansion).

## Environment Prep Status

- Core requirements install: complete (`requirements.txt` -> core set).
- Optional heavy stack: not installed (by design).
- Dataset repos status:
  - `bfcl`: downloaded
  - `swebench`: downloaded
  - `terminal_bench_2`: downloaded

## Next Run Recommendations

1. Repair brain `11434` model store so `qwen2.5:32b` is executable without remote pull.
2. Keep using individual-target matrix mode.
3. Run full catalog in two lanes:
   - Lane A: `generate_until` tasks via current chat-completions path.
   - Lane B: multiple-choice/loglikelihood tasks only after backend path is confirmed compatible.
