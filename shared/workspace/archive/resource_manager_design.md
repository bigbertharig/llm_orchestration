# GPU Resource Manager Design

**Status: FUTURE** - This document describes planned enhancements not yet implemented.

## Overview

Upgrade the task orchestration system to handle both LLM tasks (Qwen) and Script tasks (Whisper, embeddings, etc.) with intelligent GPU allocation.

---

## Priority Enhancement: Dynamic GPU Discovery

**Current limitation:** Workers have fixed GPU assignments in `config.json`. Plans must either:
- Specify GPU IDs in commands (`export CUDA_VISIBLE_DEVICES=1`), or
- Scripts accept `--gpu` argument

**Problem:** If a GPU fails mid-run, its worker keeps getting assigned tasks that will fail. Plans are tied to specific GPU topology.

**Future solution: Dynamic resource discovery**

1. **Startup scan**: Detect available GPUs, their capabilities (VRAM, compute capability)
2. **Runtime health tracking**: Monitor GPU availability, not just worker heartbeats
3. **Capability-based assignment**: Tasks request "GPU with 6GB VRAM" not "GPU 1"
4. **Graceful degradation**: If GPU goes offline, stop routing tasks to it
5. **Worker auto-configuration**: Workers discover their GPU rather than reading from config

**Benefits:**
- Plans become truly generic (no GPU IDs anywhere)
- System adapts to hardware changes without config updates
- Hot-add/remove GPUs without restart
- Better failure handling

**Implementation notes:**
- Use `nvidia-smi` for GPU discovery and health
- Store GPU state in `shared/gpu_state.json`
- Brain makes routing decisions based on live GPU state
- Workers register with brain on startup, report their GPU

---

## GPU States

Each GPU can be in one of these states:

| State | Description | Can Accept |
|-------|-------------|------------|
| `idle` | No process, GPU free | Any task type |
| `booting_llm` | Loading Qwen model (~120s) | Nothing (wait) |
| `booting_script` | Loading script model (~5-30s) | Nothing (wait) |
| `llm_ready` | Qwen loaded, waiting for tasks | LLM tasks |
| `llm_busy` | Qwen processing a task | Nothing (queued) |
| `script_busy` | Running script (Whisper, etc.) | Nothing (wait) |
| `cooling` | Just finished, brief pause | Any (after delay) |

### State Transitions

```
idle ──────────────> booting_llm ──────> llm_ready <───> llm_busy
  │                                          │
  │                                          │ (unload)
  │                                          ▼
  └──────────────> booting_script ─────> script_busy ───> idle
```

---

## Task Schema (v2)

```json
{
  "task_id": "uuid",
  "task_class": "llm | script",
  "task_type": "parse | transform | generate | execute | decide | transcribe | embed | process",

  "prompt": "For LLM tasks: the text prompt",
  "script": "For script tasks: script path or name",
  "script_args": {},

  "priority": 1-5,
  "gpu_requirement": "any | specific_id | none",
  "estimated_duration_sec": 60,

  "status": "queued | assigned | running | complete | failed",
  "assigned_gpu": null,
  "created_at": "ISO8601",
  "started_at": null,
  "completed_at": null
}
```

### Task Classes

| Class | Types | GPU Need | Model |
|-------|-------|----------|-------|
| `llm` | parse, transform, generate, execute, decide | Yes | Qwen 7B/14B |
| `script` | transcribe, embed, process | Yes/Optional | Whisper, sentence-transformers, custom |

---

## Resource Manager

### State Tracking

```python
# gpu_state.json - persisted state
{
  "gpus": {
    "0": {
      "state": "llm_ready",
      "model": "qwen2.5:14b",
      "pid": 12345,
      "port": 11434,
      "since": "2025-02-05T10:30:00Z",
      "tasks_completed": 47
    },
    "1": {
      "state": "script_busy",
      "script": "whisper_transcribe",
      "pid": 12400,
      "task_id": "abc-123",
      "since": "2025-02-05T10:35:00Z",
      "progress": 0.45
    },
    "2": {
      "state": "booting_llm",
      "model": "qwen2.5:7b",
      "started_boot": "2025-02-05T10:36:00Z",
      "expected_ready": "2025-02-05T10:38:00Z"
    }
  },
  "boot_times": {
    "qwen2.5:14b": 160,
    "qwen2.5:7b": 120,
    "whisper:small.en": 5,
    "sentence-transformers": 3
  }
}
```

### Decision Logic

```python
def assign_task(task):
    if task.task_class == "llm":
        # Find GPU with LLM ready
        gpu = find_gpu_in_state("llm_ready")
        if gpu:
            return assign_to_gpu(task, gpu)

        # Check if any idle GPUs - boot LLM
        idle = find_gpu_in_state("idle")
        if idle:
            boot_llm(idle)
            queue_task(task)  # Will be assigned when boot completes
            return

        # All GPUs busy - queue it
        queue_task(task)

    elif task.task_class == "script":
        # Prefer idle GPUs (no unload needed)
        idle = find_gpu_in_state("idle")
        if idle:
            return run_script(task, idle)

        # Check queue depths - should we evict an LLM?
        if should_evict_for_scripts():
            gpu = select_llm_to_evict()
            unload_llm(gpu)
            return run_script(task, gpu)

        # Queue it
        queue_task(task)
```

### Eviction Policy

When to unload an LLM for script tasks:

```python
def should_evict_for_scripts():
    script_queue_depth = count_queued("script")
    llm_queue_depth = count_queued("llm")
    idle_llm_gpus = count_state("llm_ready")  # Ready but not busy

    # If we have idle LLM GPUs and script backlog
    if idle_llm_gpus > 0 and script_queue_depth > 5:
        return True

    # If script queue is huge and LLM queue is empty
    if script_queue_depth > 20 and llm_queue_depth == 0:
        return True

    # Cost-benefit: script queue wait vs LLM reload time
    # If scripts would wait longer than LLM boot time, evict
    avg_script_time = 45  # seconds per transcription
    llm_boot_time = 120   # seconds to reload

    if (script_queue_depth * avg_script_time / idle_llm_gpus) > llm_boot_time:
        return True

    return False
```

---

## Queue Structure

```
shared/tasks/
├── llm_queue/           # LLM tasks waiting
│   ├── priority_5/      # Urgent
│   ├── priority_3/      # Normal
│   └── priority_1/      # Low
├── script_queue/        # Script tasks waiting
│   ├── priority_5/
│   ├── priority_3/
│   └── priority_1/
├── processing/          # Currently running
├── complete/            # Finished
└── failed/              # Errors
```

---

## Batch Mode

For large script batches (like 125 video transcriptions):

```bash
# Submit batch job
python submit_batch.py --type transcribe --input /path/to/zim --batch-size 125

# Creates a batch record
{
  "batch_id": "whisper-urban-prepper-2025-02-05",
  "task_class": "script",
  "script": "whisper_transcribe",
  "total_items": 125,
  "completed": 0,
  "gpu_mode": "exclusive",  # Request all GPUs for this batch
  "items": [
    {"video_path": "videos/abc.webm", "status": "queued"},
    ...
  ]
}
```

### Exclusive GPU Mode

When a batch requests exclusive mode:
1. Stop accepting new LLM tasks (queue them)
2. Wait for current LLM tasks to complete
3. Unload all LLM models
4. Assign all GPUs to batch processing
5. When batch complete, reload LLMs

---

## Script Workers

Script workers are simpler than LLM workers - they run a script and exit:

```python
# script_worker.py
class ScriptWorker:
    def __init__(self, gpu_id: int, script_name: str):
        self.gpu = gpu_id
        self.script = SCRIPTS[script_name]

    def run(self, task: dict):
        # Set GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)

        # Run script
        result = self.script.execute(task["script_args"])

        # Return result
        return {
            "task_id": task["task_id"],
            "output": result,
            "duration": elapsed
        }

# Available scripts
SCRIPTS = {
    "whisper_transcribe": WhisperTranscriber,
    "embed_768": SentenceTransformerEmbedder,
    "embed_1536": OpenAIEmbedder,  # API call, no GPU
    "extract_keyframes": KeyframeExtractor,
}
```

---

## Monitoring Dashboard

```
╔══════════════════════════════════════════════════════════════════╗
║  GPU RESOURCE MANAGER                              2025-02-05    ║
╠══════════════════════════════════════════════════════════════════╣
║  GPU 0: [LLM_READY ] qwen2.5:14b    idle 5m    tasks: 23         ║
║  GPU 1: [SCRIPT    ] whisper        45% done   ETA: 25s          ║
║  GPU 2: [SCRIPT    ] whisper        67% done   ETA: 15s          ║
║  GPU 3: [LLM_READY ] qwen2.5:14b    idle 5m    (paired w/0)      ║
║  GPU 4: [SCRIPT    ] whisper        12% done   ETA: 40s          ║
╠══════════════════════════════════════════════════════════════════╣
║  LLM Queue: 0 waiting    |    Script Queue: 122 waiting          ║
║  Mode: BATCH (exclusive) |    Batch: whisper-urban-prepper       ║
║                          |    Progress: 3/125 (2.4%)             ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## Configuration

```json
{
  "resource_manager": {
    "gpu_count": 5,
    "gpu_roles": {
      "0": "brain_primary",
      "1": "worker_or_script",
      "2": "worker_or_script",
      "3": "brain_secondary",
      "4": "worker_or_script"
    },
    "boot_times": {
      "qwen2.5:14b": 160,
      "qwen2.5:7b": 120,
      "whisper:small.en": 5
    },
    "eviction_policy": {
      "min_script_queue_for_evict": 5,
      "prefer_idle_llm_eviction": true,
      "never_evict_brain": true
    },
    "batch_settings": {
      "auto_exclusive_threshold": 50,
      "reload_llms_after_batch": true
    }
  }
}
```

---

## Implementation Phases

### Phase 1: State Tracking
- Add `gpu_state.json`
- Track GPU states in existing launcher
- Add state reporting to workers

### Phase 2: Dual Queues
- Split task queue into llm_queue and script_queue
- Update brain to check both queues
- Add script worker launcher

### Phase 3: Dynamic Allocation
- Implement eviction policy
- Add batch mode support
- Build monitoring dashboard

### Phase 4: Smart Scheduling
- Track actual boot times per model
- Predictive scheduling based on queue patterns
- Auto-scale based on workload type

---

## Example Workflow: Video Transcription Batch

```bash
# 1. Submit batch (125 videos)
python submit_batch.py transcribe /path/to/urban-prepper.zim

# 2. Resource manager sees batch, enters exclusive mode
#    - Queues incoming LLM tasks
#    - Waits for current LLM tasks to finish
#    - Unloads Qwen from GPUs 1, 2, 4

# 3. Assigns videos to GPUs
#    GPU 1: videos 1, 4, 7, 10, ...
#    GPU 2: videos 2, 5, 8, 11, ...
#    GPU 4: videos 3, 6, 9, 12, ...

# 4. Each GPU runs Whisper at ~14x realtime
#    125 videos * 10 min avg = 1250 min of video
#    3 GPUs * 14x = 42x effective speed
#    Total time: ~30 min

# 5. Batch completes
#    - Resource manager exits exclusive mode
#    - Reloads Qwen workers (~2 min each, sequential)
#    - Resumes LLM task processing
```

---

## Boot Time Budget

| Action | Time | Notes |
|--------|------|-------|
| Unload Qwen worker | ~5s | Kill process, free VRAM |
| Load Whisper small.en | ~5s | Small model |
| Transcribe 10min video | ~45s | 14x realtime |
| Unload Whisper | ~1s | Process exits |
| Load Qwen 7B | ~120s | Cold boot |

For 125 videos on 3 GPUs:
- Unload time: 5s * 3 = 15s
- Batch time: (125 / 3) * 45s = ~31 min
- Reload time: 120s * 3 (sequential) = 6 min
- **Total: ~37 min** (vs 2.5 hours on single GPU)

---

*Design created: 2025-02-05*
