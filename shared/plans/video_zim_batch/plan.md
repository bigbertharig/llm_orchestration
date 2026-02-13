# Plan: Video ZIM Batch Processor

## Goal

Extract videos from a ZIM archive, transcribe them using Whisper, identify topics with LLM, and produce disaster-clippy compatible output files (metadata, index, and embeddings).

## Inputs

- **ZIM_PATH**: Path to the ZIM file containing videos
- **SOURCE_ID**: Identifier for this source (used in output file naming)
- **OUTPUT_FOLDER**: Where to write the final output files
- **RUN_MODE**: `fresh` or `resume` (default: `fresh`)
- **RESUME_BATCH_ID**: Required when `RUN_MODE=resume`; existing batch to continue

## Outputs

- `{OUTPUT_FOLDER}/_metadata.json` - Document metadata
- `{OUTPUT_FOLDER}/_index.json` - Searchable index
- `{OUTPUT_FOLDER}/_vectors_768.json` - 768-dim embeddings for semantic search
- Transcripts saved in `{BATCH_PATH}/transcripts/`

## Available Scripts

### scripts/batch_init.py

- **Purpose**: Create manifest for fresh runs, or validate/reuse existing manifest for resume runs
- **GPU**: no
- **Run command**: `source ~/ml-env/bin/activate && python {PLAN_PATH}/scripts/batch_init.py --zim {ZIM_PATH} --source-id {SOURCE_ID} --output {OUTPUT_FOLDER} --batch-id {BATCH_ID} --run-mode {RUN_MODE}`
- **Output**: `{BATCH_PATH}/manifest.json` listing all videos to process

### scripts/video_transcribe_whisper.py

- **Purpose**: Transcribe a single video using Whisper (no LLM)
- **GPU**: yes (uses faster-whisper with CTranslate2)
- **Run command**: `export LD_LIBRARY_PATH="$HOME/ml-env/lib/python3.13/site-packages/nvidia/cublas/lib:$HOME/ml-env/lib/python3.13/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH" && source ~/ml-env/bin/activate && python {PLAN_PATH}/scripts/video_transcribe_whisper.py --batch-id {BATCH_ID} --video-id {VIDEO_ID}`
- **Output**: Transcript JSON file in `{BATCH_PATH}/transcripts/` (without topics)

### scripts/video_add_topics.py

- **Purpose**: Add LLM-generated topics to a transcribed video
- **GPU**: yes (uses worker's Ollama LLM)
- **Run command**: `source ~/ml-env/bin/activate && python {PLAN_PATH}/scripts/video_add_topics.py --batch-id {BATCH_ID} --video-id {VIDEO_ID}`
- **Output**: Updates transcript JSON with topic and keywords fields

### scripts/batch_aggregate.py

- **Purpose**: Combine transcripts into disaster-clippy format with embeddings
- **GPU**: no (embeddings run on CPU)
- **Run command**: `source ~/ml-env/bin/activate && python {PLAN_PATH}/scripts/batch_aggregate.py --batch-id {BATCH_ID}`
- **Output**: Final `_metadata.json`, `_index.json`, `_vectors_768.json` in output folder

## Tasks

### init
- **executor**: brain
- **task_class**: cpu
- **command**: `source ~/ml-env/bin/activate && python {PLAN_PATH}/scripts/batch_init.py --zim {ZIM_PATH} --source-id {SOURCE_ID} --output {OUTPUT_FOLDER} --batch-id {BATCH_ID} --run-mode {RUN_MODE}`
- **depends_on**: none

### transcribe_whisper
- **executor**: worker
- **task_class**: script
- **vram_policy**: infer
- **foreach**: {PLAN_PATH}/history/{BATCH_ID}/manifest.json:videos
- **batch_size**: 1
- **command**: `export LD_LIBRARY_PATH="$HOME/ml-env/lib/python3.13/site-packages/nvidia/cublas/lib:$HOME/ml-env/lib/python3.13/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH" && source ~/ml-env/bin/activate && python {PLAN_PATH}/scripts/video_transcribe_whisper.py --batch-id {BATCH_ID} --video-id {ITEM.id}`
- **depends_on**: init

### add_topics
- **executor**: worker
- **task_class**: llm
- **foreach**: {PLAN_PATH}/history/{BATCH_ID}/manifest.json:videos
- **batch_size**: 1
- **command**: `source ~/ml-env/bin/activate && python {PLAN_PATH}/scripts/video_add_topics.py --batch-id {BATCH_ID} --video-id {ITEM.id}`
- **depends_on**: init, transcribe_whisper_{ITEM.id}

### aggregate
- **executor**: brain
- **task_class**: cpu
- **command**: `source ~/ml-env/bin/activate && python {PLAN_PATH}/scripts/batch_aggregate.py --batch-id {BATCH_ID}`
- **depends_on**: add_topics

## Notes

### Task Flow

1. **init**: Brain creates manifest for fresh runs, or validates existing manifest for resume runs
2. **transcribe_whisper**: Workers transcribe videos using Whisper (script tasks, cold workers)
3. **add_topics**: Workers add LLM topics (llm tasks, hot workers with model loaded)
4. **aggregate**: Brain combines transcripts into final output

### Run Modes

- `fresh`: starts a new batch and writes a new `manifest.json`
- `resume`: reuses existing `history/{BATCH_ID}/manifest.json` and does not reset progress

### Foreach Expansion

Both `transcribe_whisper` and `add_topics` use `foreach` with micro-batching:
- Brain runs `init`, which creates `manifest.json` with video list
- Brain expands `transcribe_whisper` into per-video tasks (`batch_size: 1`)
- Brain expands `add_topics` into per-video tasks (`batch_size: 1`) with per-item dependency checks
- After ALL expanded topic tasks complete, brain runs `aggregate`

### Resource Management

The split allows proper GPU resource management:
- **transcribe_whisper** (script): Runs on cold workers, uses Whisper model
- **add_topics** (llm): Runs on hot workers, brain loads LLM before this phase
- Brain can insert load_llm/unload_llm tasks between phases

### Worker Environment Variables

Scripts receive these environment variables from the worker:
- `CUDA_VISIBLE_DEVICES`: GPU to use
- `WORKER_OLLAMA_URL`: URL to worker's Ollama instance (e.g., http://localhost:11435)
