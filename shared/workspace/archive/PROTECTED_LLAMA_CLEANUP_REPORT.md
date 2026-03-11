# Protected Llama Cleanup Report

Scope: files under `/media/bryan/shared/agents` and the mirrored workspace repo at `/home/bryan/llm_orchestration/shared/agents`.

Goal: remove Ollama as a runtime concept, not just hide it in the UI.

## Status: COMPLETED (2026-03-08)

All 13 files have been cleaned. The codebase now uses a unified runtime vocabulary with llama as the sole backend. Details below.

---

## Already clean (no changes needed):

- `/media/bryan/shared/agents/runtime_reset.py` — already backend-neutral.
- `/media/bryan/shared/agents/tests/` — all test files were already clean of ollama references.

---

## Completed cleanup (13 files):

### 1. `brain_dispatch.py`
- Removed `BRAIN_OLLAMA_URL` and `WORKER_OLLAMA_URL` env injection.
- Default `BRAIN_RUNTIME_BACKEND` changed from `ollama` to `llama`.
- Comments updated to reference runtime/llama instead of Ollama.

### 2. `gpu_workers.py`
- Stopped exporting `WORKER_OLLAMA_URL`.
- Default worker runtime backend changed to `llama`.
- `self.runtime_ollama_url` renamed to `self.runtime_api_base`.

### 3. `gpu.py`
- Import changed: `from gpu_ollama import GPURuntimeMixin` (class was renamed inside gpu_ollama.py).
- Class now inherits `GPURuntimeMixin` instead of `GPUOllamaMixin`.
- Default `runtime_backend` changed from `"ollama"` to `"llama"`.
- API URL simplified to always use `/v1/chat/completions`.
- All instance variable renames:
  - `self.runtime_ollama_url` -> `self.runtime_api_base`
  - `self.ollama_process` -> `self.runtime_process`
  - `self.ollama_healthy` -> `self.runtime_healthy`
  - `self.ollama_consecutive_failures` -> `self.runtime_consecutive_failures`
  - `self.ollama_health_threshold` -> `self.runtime_health_threshold`
  - `self.ollama_circuit_breaker` -> `self.runtime_circuit_breaker`
- Dispatcher methods simplified: always call llama path (no more backend branching).

### 4. `gpu_tasks.py`
- Attestation probe simplified to always use `/v1/models` (removed Ollama `/api/ps` branch).
- `self.ollama_healthy` -> `self.runtime_healthy`.
- `stop_local_ollama` -> `stop_local_runtime`.
- Removed Ollama restart-after-reset code block.
- `self._kill_orphan_ollama_runners` -> `self._kill_orphan_runtime_processes`.
- `self.runtime_ollama_url` -> `self.runtime_api_base`.
- All comments and log messages updated.

### 5. `gpu_split.py`
- All `getattr(self, "runtime_backend", "ollama")` changed to `"llama"`.
- `self.runtime_ollama_url` -> `self.runtime_api_base`.
- `stop_local_ollama` -> `stop_local_runtime`.
- `self.ollama_process` -> `self.runtime_process`.
- `_kill_orphan_ollama_runners` -> `_kill_orphan_runtime_processes`.
- `_verify_no_ollama_runners_on_port` -> `_verify_no_runtime_processes_on_port`.
- Result dict keys renamed from ollama to runtime terminology.
- Subprocess commands: `["ollama", "serve"]` -> `["llama-server", ...]`.
- `OLLAMA_HOST` env var -> `LLAMA_ARG_HOST`.
- Process scanning checks `"llama"` instead of `"ollama runner"`.
- All comments updated. 44 references total cleaned.

### 6. `gpu_llama.py`
- Docstrings cleaned of Ollama references.
- All `self.ollama_*` -> `self.runtime_*` renames.
- `self.runtime_ollama_url` -> `self.runtime_api_base`.
- Comments about "GPUOllamaMixin" -> "GPURuntimeMixin".

### 7. `startup.py`
- Import changed: `scan_runtime` instead of `scan_ollama`.
- `verify_hardware` parameter `ollama` -> `runtime`.
- Display strings updated from "Ollama" to "Runtime".
- Defaults changed to `"llama"`.

### 8. `brain_core.py`
- Default `runtime_backend` changed to `"llama"`.
- `_start_ollama_brain` -> `_start_legacy_runtime_brain`.
- `_stop_ollama_brain` -> `_stop_legacy_runtime_brain`.
- `_kill_existing_ollama` -> `_kill_existing_runtime`.
- `self.ollama_process` -> `self.runtime_process`.
- `start_ollama`/`stop_ollama` -> `start_runtime_legacy`/`stop_runtime_legacy` aliases.
- All log messages updated.

### 9. `hardware.py`
- New `scan_runtime()` function replaces `scan_ollama()` as primary API.
- `scan_ollama()` kept as backwards-compat wrapper (calls `scan_runtime()`).
- `suggest_assignment` and `format_scan_report` parameter `ollama` -> `runtime`.
- Display strings "Ollama:" -> "Runtime:".

### 10. `setup.py`
- Default `runtime_backend` changed to `"llama"`.
- `--ollama-host` kept as deprecated alias for `--runtime-host`.
- Config generation updated to use runtime terminology.

### 11. `worker.py`
- Default `runtime_backend` changed to `"llama"`.
- `--ollama-url` -> `--runtime-url` (kept as suppress arg).
- Removed `WORKER_OLLAMA_URL` env export.

### 12. `launch.py`
- `check_ollama_models` -> `check_runtime_models`.
- Display strings updated.
- Config fallback to `ollama_host` kept for backwards compatibility.

### 13. `gpu_core.py`
- Comments about "Ollama-compatible keep_alive" updated to generic "keep_alive".

### 14. `brain.py`
- `self.ollama_process` -> `self.runtime_process`.
- Default `runtime_backend` -> `"llama"`.

### 15. `brain_resources.py`
- Default `runtime_backend` -> `"llama"`.

### 16. `brain_failures.py`
- `"ollama"` keyword -> `"llama"` in command classification.

---

## Intentionally kept backwards-compat references:

These ollama references were deliberately preserved for backwards compatibility:

- **Config fallbacks**: `config.get("ollama_host")` — supports older config.json files.
- **Legacy cleanup commands**: `pkill -f "ollama serve"`, `systemctl stop ollama.service` — ensures old Ollama processes get cleaned up during transition.
- **`scan_ollama()` wrapper**: Backwards-compat function in hardware.py that calls `scan_runtime()`.
- **`gpu_ollama.py` filename**: File kept (class inside renamed to `GPURuntimeMixin`).
- **`--ollama-host` CLI arg**: Kept as deprecated alias in setup.py.

---

## Clean end state achieved:

- One runtime vocabulary: `runtime_backend=llama`, `runtime_api_base`, runtime server/container, runtime health.
- One probe vocabulary: `/health`, `/v1/models`, and process/container ownership.
- No `*_OLLAMA_URL` env vars.
- No `ollama serve`, `ollama runner`, `OLLAMA_HOST`, `/api/ps`, or `/api/tags` in the active control plane.
- No UI, logs, or reset summaries that mention Ollama.
- All Python syntax verified clean across all edited files.
