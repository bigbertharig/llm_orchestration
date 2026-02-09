# Logging Improvements - Debug Task Claiming Issues

## Problem Summary

During the video_zim_batch run on 2026-02-07, GPU agents 1 and 4 stopped claiming tasks after transcription completed, but logs didn't show why. Only gpu-2 claimed load_llm tasks and processed all topic generation alone.

**What we couldn't see:**
- Whether GPU agents 1 and 4 were still running their claim loop
- Why they didn't claim load_llm or llm tasks
- If they crashed, exited normally, or got stuck
- Which tasks were in "processing" state (stuck tasks)

**Note:** This analysis was written before the GPU agent refactor. The architecture has since changed from multi-worker-per-GPU to one GPU agent per GPU that spawns worker subprocesses. Many of the proposed improvements below are now handled differently in the GPU agent model.

## Priority Levels

**P0 (Critical)** - Would have immediately identified the bug
**P1 (High)** - Helpful context for debugging
**P2 (Nice to have)** - Useful for optimization and monitoring

---

## GPU Agent Improvements

### P0: GPU Agent Lifecycle Events (IMPLEMENTED)

GPU agents now have clear lifecycle logging in `shared/agents/gpu.py`:

```python
# GPUAgent.__init__()
self.logger.info(f"GPU agent initialized: {self.name} (GPU {self.gpu_id}), port {self.port}, model {self.model}")

# GPUAgent.run()
self.logger.info(f"GPU agent {self.name} starting")
# ... on crash:
self.logger.error(f"GPU agent crashed: {e}", exc_info=True)

# GPUAgent.cleanup()
self.logger.info(f"GPU {self.name} shutting down - completed {stats['tasks_completed']}, failed {stats['tasks_failed']}")
```

**Example output:**
```
[21:05:50] [gpu-1] GPU agent initialized: gpu-1 (GPU 1), port 11435, model qwen2.5:7b
[21:05:51] [gpu-1] GPU agent gpu-1 starting
[21:45:26] [gpu-1] GPU agent crashed: KeyError('model_name')
[21:45:26] [gpu-1] GPU gpu-1 shutting down - completed 42, failed 1
```

---

### P0: Claim Loop Visibility

Log what the worker sees in the queue even when not claiming anything.

**Location:** `shared/agents/worker.py` - `claim_task()` end

```python
def claim_task(self, preferred_classes: List[str] = None) -> Optional[Dict[str, Any]]:
    # ... existing code that tries to claim ...

    # If we get here, no task was claimed
    if len(candidates) > 0:
        # Saw tasks but couldn't claim any (lost races)
        self.logger.debug(f"Claim attempt: {len(candidates)} tasks available in {preferred_classes}, all claimed by others")
    else:
        # No tasks matching our preferences
        task_summary = ", ".join([f"{k}:{len(v)}" for k, v in tasks_by_class.items() if len(v) > 0])
        if task_summary:
            self.logger.debug(f"No suitable tasks - Queue: [{task_summary}], Preferred: {preferred_classes}, Model loaded: {self.model_loaded}")
        else:
            self.logger.debug(f"Queue empty")

    return None
```

**Example output:**
```
[2026-02-07 21:45:30] [worker-gpu1] No suitable tasks - Queue: [llm:125], Preferred: ['resource', 'script', 'cpu', 'llm'], Model loaded: False
[2026-02-07 21:45:35] [worker-gpu1] No suitable tasks - Queue: [llm:123], Preferred: ['resource', 'script', 'cpu', 'llm'], Model loaded: False
```

This would have immediately shown: "worker-1 is alive, sees 125 llm tasks, but isn't claiming them"

---

### P1: Claim Attempt Logging

Show when workers try to claim tasks and lose races.

**Location:** `shared/agents/worker.py` - `claim_task()` in the lock attempt

```python
for task_file in candidates:
    lock_file = str(task_file) + ".lock"
    lock = FileLock(lock_file, timeout=1)

    task_id = task_file.stem[:8]  # First 8 chars of UUID

    try:
        with lock:
            self.logger.debug(f"Attempting claim: {task_id}")

            if not task_file.exists():
                self.logger.debug(f"Lost race: {task_id} (claimed by another worker)")
                continue

            # ... existing claim logic ...

    except Timeout:
        self.logger.debug(f"Lock timeout: {task_id} (another worker has it)")
        continue
```

**Example output:**
```
[2026-02-07 21:45:28] [worker-gpu1] Attempting claim: d5de16d7
[2026-02-07 21:45:28] [worker-gpu1] Lost race: d5de16d7 (claimed by another worker)
[2026-02-07 21:45:28] [worker-gpu2] Attempting claim: d5de16d7
[2026-02-07 21:45:28] [worker-gpu2] Claimed meta task: d5de16d7 (load_llm)
```

---

### P1: Heartbeat Logging

Show heartbeat activity to confirm worker is alive.

**Location:** `shared/agents/worker.py` - `_write_heartbeat()`

```python
def _write_heartbeat(self):
    """Send heartbeat to brain."""
    heartbeat_data = {
        "worker": self.name,
        "timestamp": datetime.now().isoformat(),
        "status": "processing" if self.current_task else "idle",
        "model_loaded": self.model_loaded,
        "tasks_completed": self.stats.get('tasks_completed', 0)
    }

    # Write heartbeat file...

    # Log at debug level (can be noisy)
    idle_time = time.time() - self.last_task_time if hasattr(self, 'last_task_time') else 0
    self.logger.debug(f"Heartbeat sent - status: {heartbeat_data['status']}, idle: {idle_time:.0f}s")
```

**Example output:**
```
[2026-02-07 21:45:30] [worker-gpu1] Heartbeat sent - status: idle, idle: 4s
[2026-02-07 21:46:00] [worker-gpu1] Heartbeat sent - status: idle, idle: 34s
```

---

## Brain Improvements

### P0: Stuck Task Detection

Track tasks in "processing" state for too long.

**Location:** `shared/agents/brain.py` - in monitoring loop

```python
def _monitor_system(self):
    # ... existing monitoring ...

    # Check for stuck tasks (processing > 5 minutes)
    stuck_tasks = []
    for task_file in self.processing_path.glob("*.json"):
        with open(task_file) as f:
            task = json.load(f)
        started = datetime.fromisoformat(task.get('started_at', datetime.now().isoformat()))
        elapsed = (datetime.now() - started).total_seconds()
        if elapsed > 300:  # 5 minutes
            stuck_tasks.append({
                'task_id': task['task_id'][:8],
                'assigned_to': task.get('assigned_to', 'unknown'),
                'elapsed_min': int(elapsed / 60),
                'name': task.get('name', '')
            })

    if stuck_tasks:
        self.logger.warning(f"Stuck tasks detected: {stuck_tasks}")

    # Add to monitoring details
    details['stuck_tasks'] = len(stuck_tasks)
    details['processing_count'] = len(list(self.processing_path.glob("*.json")))
```

**Example output:**
```
[2026-02-07 21:50:00] [brain] Stuck tasks detected: [{'task_id': 'a1b2c3d4', 'assigned_to': 'worker-1', 'elapsed_min': 6, 'name': 'add_topics_...'}]
[2026-02-07 21:50:00] [brain] MONITOR: GPUs: 2/5, Workers: 3/3, Queue: 123 (llm:123), Processing: 2, Stuck: 2
```

---

### P1: Model Loading Expectations

Track when load_llm tasks are created but workers don't pick them up.

**Location:** `shared/agents/brain.py` - after inserting load_llm

```python
def _check_resource_needs(self):
    # ... existing code that inserts load_llm ...

    if llm_tasks > 0 and not models_loaded:
        self.logger.info(f"LLM tasks waiting ({llm_tasks}) but no models loaded - inserting load_llm task")
        task_id = self._insert_load_llm()

        # Track when we inserted it
        self.load_llm_requests[task_id] = {
            'created_at': datetime.now(),
            'workers_needed': workers_without_model.copy()
        }

# Then in monitoring, check if it's been too long:
def _monitor_system(self):
    # Check for stale load_llm requests
    for task_id, request in list(self.load_llm_requests.items()):
        age = (datetime.now() - request['created_at']).total_seconds()
        if age > 60:
            workers_still_waiting = [w for w in request['workers_needed'] if not self._worker_has_model(w)]
            if workers_still_waiting:
                self.logger.warning(
                    f"load_llm task available for {age:.0f}s but workers {workers_still_waiting} still have no model"
                )
```

**Example output:**
```
[2026-02-07 21:45:28] [brain] LLM tasks waiting (125) but no models loaded - inserting load_llm task
[2026-02-07 21:46:30] [brain] load_llm task available for 62s but workers ['worker-1', 'worker-3'] still have no model
```

---

### P2: Detailed Queue Statistics

Add more detail to monitoring output about queue composition.

**Location:** `shared/agents/brain.py` - `_monitor_system()`

```python
# Existing queue_stats collection...
details['queue_stats'] = {
    'total_pending': total_pending,
    'by_class': {k: v for k, v in queue_stats.items()},
    'by_worker': {
        'pending': total_pending,
        'processing': len(list(self.processing_path.glob("*.json"))),
        'completed_last_5min': self._count_recent_completions(300)
    }
}
```

---

## Log Level Recommendations

### INFO (always on)
- Worker lifecycle (start, stop, crash)
- Task claimed/completed
- Model load/unload
- Brain decisions (task creation, monitoring summary)
- Warnings (stuck tasks, stale load_llm requests)

### DEBUG (enable during troubleshooting)
- Claim attempts and races
- Queue visibility ("no suitable tasks")
- Heartbeats
- Detailed queue statistics

---

## Testing the Improvements

### Test Case 1: Worker Dies
**Setup:** Kill worker-1 mid-task
**Expected logs:**
```
[worker-1] Worker crashed in main loop: ConnectionError(...)
[worker-1] Worker shutting down - completed 15 tasks
[brain] Stuck tasks detected: [worker-1 task after 5+ min]
[brain] Worker worker-1 last heartbeat 90s ago (threshold: 60s)
```

### Test Case 2: No Model Workers
**Setup:** Start workers without models, queue LLM tasks
**Expected logs:**
```
[worker-1] No suitable tasks - Queue: [llm:125], Preferred: ['resource', 'script', 'cpu', 'llm'], Model loaded: False
[brain] LLM tasks waiting (125) but no models loaded - inserting load_llm task
[worker-1] Attempting claim: d5de16d7
[worker-1] Lost race: d5de16d7 (claimed by another worker)
[worker-2] Claimed meta task: d5de16d7 (load_llm)
```

### Test Case 3: All Workers Busy
**Setup:** Queue 200 tasks with only 3 workers
**Expected logs:**
```
[brain] MONITOR: Queue: 197, Processing: 3, Stuck: 0
[worker-1] Claim attempt: 197 tasks available in ['resource', 'llm'], all claimed by others
```

---

## Implementation Priority

1. **Phase 1 (Critical):** Worker lifecycle + claim loop visibility
   - Would have caught this bug immediately
   - ~30 minutes to implement

2. **Phase 2 (High):** Stuck task detection + model loading expectations
   - Prevents similar issues in the future
   - ~1 hour to implement

3. **Phase 3 (Polish):** Claim attempt details + heartbeat logging
   - Nice for deep debugging
   - ~30 minutes to implement

**Total implementation time:** ~2-3 hours

---

## Implemented: Independent Component Logging

**Status:** COMPLETE (2026-02-08)

Each component now has independent logging level control via environment variables:

### Environment Variables

- **EXECUTOR_LOG_LEVEL** - Controls executor (permission checking) logs
- **WORKER_LOG_LEVEL** - Controls worker (task execution, claiming) logs
- **BRAIN_LOG_LEVEL** - Controls brain (coordination, decision-making) logs

### Usage Examples

```bash
# Debug only executor permission checks
EXECUTOR_LOG_LEVEL=DEBUG python shared/agents/worker.py worker-1

# Debug worker task claiming and queue visibility
WORKER_LOG_LEVEL=DEBUG python shared/agents/worker.py worker-1

# Debug brain decisions and task routing
BRAIN_LOG_LEVEL=DEBUG python shared/agents/brain.py

# Mix different levels for targeted debugging
WORKER_LOG_LEVEL=DEBUG EXECUTOR_LOG_LEVEL=WARNING python shared/agents/worker.py worker-1

# Everything at DEBUG (maximum verbosity)
BRAIN_LOG_LEVEL=DEBUG WORKER_LOG_LEVEL=DEBUG EXECUTOR_LOG_LEVEL=DEBUG python shared/agents/brain.py
```

### What Each Level Shows

**EXECUTOR_LOG_LEVEL=DEBUG:**
- All ALLOWED file reads/writes and bash commands (usually excessive)
- Useful for debugging permission system issues

**WORKER_LOG_LEVEL=DEBUG:**
- Task claiming attempts and lost races
- Queue visibility (what tasks the worker sees)
- Heartbeat details (idle time, GPU stats)
- GPU monitoring failures

**BRAIN_LOG_LEVEL=DEBUG:**
- Task dependency tracking
- Resource task deduplication
- Task analysis errors
- Training sample logging

### Benefits

1. **Focused debugging** - Only enable verbose logging for the component you're investigating
2. **Reduced noise** - Other components stay at INFO level while debugging one
3. **Performance** - Less I/O overhead when not logging everything at DEBUG
4. **Flexibility** - Different log levels per-component in the same run

---

## Notes

- Keep debug logs at debug level to avoid log spam
- All timestamps are ISO format with milliseconds for precise ordering
- Consider log rotation if files get too large (>100MB)
- Default level for all components is INFO if env var not set
