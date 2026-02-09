# Video ZIM Batch Processing Plan

This plan coordinates the GPU cluster to process video ZIM files and produce output folders compatible with disaster-clippy.

## Goal

Process a video ZIM file using parallel transcription across 3 worker GPUs, then output a source folder that can be dropped directly into disaster-clippy's `BACKUP_PATH/`.

## Output

A source folder containing:
```
{source_id}/
  _metadata.json      # Document metadata for quick scanning
  _index.json         # Full document content
  _vectors_768.json   # 768-dim embeddings for offline search
```

Drop this folder into disaster-clippy's backup folder and it will automatically appear in the source list.

## Folder Structure

```
shared/plans/video_zim_batch/
  README.md                 # This file
  plan.md                   # Task Plan Document for brain
  scripts/
    batch_init.py           # Initialize batch, read ZIM, create manifest
    video_transcribe.py     # Worker script for transcription (imports from disaster-clippy)
    batch_aggregate.py      # Combine results, generate embeddings, save outputs
  history/
    {batch_id}/
      manifest.json         # Batch config and video status tracking
      transcripts/          # Per-video transcript JSONs from workers
      output/               # Final output files
      logs/
        batch.jsonl         # Main batch decision log (JSONL format)
        events.log          # Human-readable event log
```

## Logging Layers

Each batch has multi-layer logging to verify the system at each level:

```
batches/{batch_id}/logs/
  plan_interpretation.jsonl   # Layer 1: Brain's understanding of plan
  task_assignments.jsonl      # Layer 2: Task -> worker assignments
  worker_execution.jsonl      # Layer 3: Per-worker execution
  batch.jsonl                 # Combined log (all layers)
  events.log                  # Human-readable combined log
```

### Layer 1: Plan Interpretation (Brain)
Tracks how the brain understands the plan document:
- `PLAN_RECEIVED` - Brain received the plan
- `PLAN_PARSED` - Brain extracted tasks from plan
- `PLAN_VALIDATED` - Brain confirmed plan is executable
- `PLAN_ERROR` - Brain couldn't understand something

### Layer 2: Task Assignment (Brain -> Workers)
Tracks work distribution decisions:
- `TASK_CREATED` - New task created from plan
- `TASK_ASSIGNED` - Task assigned to specific worker
- `TASK_REASSIGNED` - Task moved to different worker
- `TASK_QUEUED` - Task waiting for available worker

### Layer 3: Worker Execution
Tracks individual worker performance:
- `WORKER_START` - Worker began processing
- `WORKER_CLAIM` - Worker claimed a video
- `WORKER_TRANSCRIBE_COMPLETE` - Transcription finished
- `WORKER_ERROR` - Worker encountered error
- `WORKER_DONE` - Worker finished all work

### Layer 4: Batch Lifecycle
Tracks overall batch progress:
- `BATCH_INIT` - Batch created
- `BATCH_START` - Processing started
- `BATCH_AGGREGATE` - Aggregation phase
- `BATCH_COMPLETE` - All done

## Scripts

All scripts import from disaster-clippy to reuse tested code:

1. **batch_init.py** - Creates manifest from ZIM file
2. **video_transcribe.py** - Worker transcription (runs on GPU 1, 2, or 4)
3. **batch_aggregate.py** - Combines results and generates final output

## Usage

```bash
# 1. Initialize batch
python scripts/batch_init.py \
  --zim /path/to/videos.zim \
  --source-id my-source \
  --output /path/to/disaster-clippy/backups/my-source

# 2. Run workers (on each GPU)
python scripts/video_transcribe.py --batch-id {uuid} --gpu 1 &
python scripts/video_transcribe.py --batch-id {uuid} --gpu 2 &
python scripts/video_transcribe.py --batch-id {uuid} --gpu 4 &

# 3. Wait for completion, then aggregate
python scripts/batch_aggregate.py --batch-id {uuid}
```

## Dependencies

From disaster-clippy (imported directly):
- `offline_tools.video_zim_indexer.VideoZIMIndexer`
- `offline_tools.video_processor.transcribe_with_timestamps`
- `offline_tools.video_processor.group_into_chunks`
- `offline_tools.video_processor.chunks_to_documents`
- `offline_tools.indexer.save_metadata`
- `offline_tools.indexer.save_index`
- `offline_tools.indexer.generate_768_vectors`

System requirements:
- faster-whisper (CUDA)
- sentence-transformers (CPU mode for GTX 1060)
- zimply_core
- Ollama (optional, for topic identification)
