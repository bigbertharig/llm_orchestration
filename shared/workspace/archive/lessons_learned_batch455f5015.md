# Lessons Learned - Video ZIM Batch Processing

## Test Run: 2026-02-06 (Batch 455f5015)

### What Worked
- **Plan parsing** - submit_plan.py correctly parsed plan.md and substituted variables
- **Task creation** - Shell tasks created with proper dependencies in context
- **Dependency checking** - Brain correctly waited for transcribe tasks before running aggregate
- **Enhanced logging** - plan_task_id, elapsed time, full commands visible in logs
- **Worker file locking** - FileLock prevents race conditions when workers claim tasks
- **Parallel transcription** - 3 GPUs transcribing simultaneously

### Issues Found

| Issue | Cause | Fix |
|-------|-------|-----|
| Transcription timeout | 600s too short for 8-min videos | Increased to 1800s |
| Brain grabbed worker tasks | Workers loading, brain picked up "stale" tasks after 30s | Increased stale threshold to 120s |
| Brain lacks FileLock | Could race with workers when claiming | Added FileLock to brain |
| Dependency cascade failure | transcribe_1 timeout caused aggregate to fail | Expected behavior, but timeout fix prevents this |
| GPUs at 0% despite running | LD_LIBRARY_PATH set in Python after imports | Set env vars in shell BEFORE Python starts |
| CUDA_VISIBLE_DEVICES too late | Set after CUDA libs already loaded | Set at very top of script before any imports |
| export command blocked | Worker permissions didn't allow export | Add regex pattern for export + CUDA commands |
| Worker timeout too short | worker.json had 600s, brain had 1800s | Match timeouts: 1800s for both |

### CRITICAL: CUDA Environment Initialization

**This is the most important lesson for future GPU-based plans.**

When using CUDA libraries (CTranslate2, PyTorch, faster-whisper, etc.):

1. **LD_LIBRARY_PATH must be set BEFORE Python starts** - Setting it in Python code is too late because:
   - Libraries are loaded at import time
   - Python's `os.environ` changes don't affect already-loaded libraries

2. **CUDA_VISIBLE_DEVICES must be set BEFORE any CUDA imports** - Place at very top of script:
   ```python
   import argparse
   import os
   import sys

   # FIRST THING: Parse GPU arg and set CUDA visibility
   parser = argparse.ArgumentParser(add_help=False)
   parser.add_argument("--gpu", type=int)
   args, _ = parser.parse_known_args()
   if args.gpu is not None:
       os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

   # NOW safe to import CUDA-dependent libraries
   from faster_whisper import WhisperModel
   ```

3. **For shell commands in plans, export LD_LIBRARY_PATH first** (CUDA_VISIBLE_DEVICES is set by the worker automatically):
   ```bash
   export LD_LIBRARY_PATH="$HOME/ml-env/lib/python3.13/site-packages/nvidia/cublas/lib:$HOME/ml-env/lib/python3.13/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH" && \
   source ~/ml-env/bin/activate && \
   python script.py
   ```

   **Note:** Plans should NOT specify GPU IDs. Each worker sets `CUDA_VISIBLE_DEVICES` to its assigned GPU before executing tasks. This keeps plans generic and allows workers to be added/removed without changing plans.

4. **Test GPU visibility** before long runs:
   ```bash
   nvidia-smi --query-gpu=utilization.gpu --format=csv
   # Should show >0% when model is loaded
   ```

### Configuration Changes
```
brain.py:
  - Shell task timeout: 600s → 1800s
  - Stale task pickup: 30s → 120s
  - Added FileLock for task claiming
```

### Best Practices Identified
1. **Start workers before submitting plan** - Avoids brain picking up worker tasks
2. **Workers should be ready within 2 minutes** - Or brain will help (by design)
3. **Transcription tasks need longer timeouts** - 8-min video ≈ 10-15 min transcription on GTX 1060
4. **Check both brain_decisions.log and batch events.log** - Different perspectives on same run

### GPU Power Observations
- Model loaded but idle: ~4-5W per GPU
- Model under load: ~30-100W per GPU
- VRAM stays allocated when idle (expected)

### Next Steps
- Test with workers pre-started
- Verify FileLock prevents race conditions
- Consider per-task-type timeouts in future
