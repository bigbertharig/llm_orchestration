# AI Box Upgrade — Unified Memory Hardware Evolution

*Brainstormed: 2026-03-23*
*Status: Future planning — no hardware purchased yet*

---

## Summary

Design notes for evolving the LLM orchestration system to support unified-memory
hardware (NVIDIA DGX Spark, DGX Station, Mac Studio) alongside or replacing the
current multi-GPU rig. The core insight: the orchestration system is already a
distributed resource manager — it just needs to speak in "memory and threads"
instead of "GPU slots and VRAM."

---

## Hardware Options Evaluated

### NVIDIA DGX Spark (GB10) — ~$4,700
- 128GB unified memory (Grace Blackwell superchip)
- 20-core ARM CPU
- Can run models up to ~200B parameters
- Multiple models concurrently: ~60GB combined model weight works well in parallel
- Runs vLLM, TensorRT-LLM, SGLang natively

### NVIDIA DGX Station (GB300) — price TBD
- 748GB coherent memory
- GB300 Grace Blackwell Ultra superchip
- Up to 20 PFLOPS AI compute
- Can run models up to 1 trillion parameters

### Mac Studio M3 Ultra — ~$9,500 (512GB config)
- 512GB unified memory, 80-core GPU
- Bandwidth scales poorly — 512GB model needs 4x bandwidth of 128GB to maintain same tok/s
- RDMA over Thunderbolt 5 (macOS Tahoe 26.2) enables multi-Mac clusters

---

## Current Architecture (For Reference)

| Worker Class | VRAM | Max Model (Q4) | Current Hardware |
|---|---|---|---|
| CPU | 0 | n/a (script tasks) | 8x Orange Pi Prime |
| Single GPU | 6GB | ~7B | GTX 1060 |
| Split GPU (pair) | 12GB effective | ~14B | 2x GTX 1060 |
| Brain GPU | 24GB | ~30B | RTX 3090 Ti |

Key constraints:
- 1 gpu.py agent = 1 physical GPU = 0 or 1 models
- Split GPU requires hardcoded pairs in `models.catalog.json`
- PCIe x1 Gen1 bottleneck on worker GPUs (board limitation)
- Brain uses LLM for output evaluation, decisions, goal planning — not purely scripted

---

## Unified Memory Box — What Changes

### Memory Budget Example (128GB box)

| Resident | Est. Size | Purpose |
|----------|-----------|---------|
| Brain model (70B Q4) | ~42GB | Coordination, output eval, decisions |
| OS + runtime overhead | ~8GB | |
| Available for workers | ~78GB | Flexible pool |

The 78GB worker pool could hold:
- 11x 7B models (~7GB each) for max parallelism
- 3x 14B models (~10GB each) + a couple 7Bs for mixed workloads
- 1x 30B + several 7Bs for a middle ground
- Any mix, swapped on the fly based on queue demand

On 256GB or 512GB boxes, the pool gets absurd.

### Split GPU Problem Disappears

No more PCIe pairs, no split_gpu placement, no reservation/quarantine logic for
paired GPUs. A 14B model just loads into the memory pool like any other model.
The `placement` and `split_groups` fields in `models.catalog.json` become irrelevant.

### llama-server Advantage

The current containerized llama-server approach maps perfectly:
- Each model gets its own llama-server instance on a dedicated port
- Each is an independent process with an OpenAI-compatible endpoint
- Brain and workers target specific `host:port` endpoints
- Full control over model lifecycle — no Ollama fighting the orchestrator

Ollama would conflict here (owns its own model lifecycle, auto-unloads idle models).
Raw llama-server lets the orchestration system be the model manager.

---

## Abstraction Evolution

The current hardware-specific abstractions collapse into generic resource management:

| Current Abstraction | Unified Abstraction |
|---|---|
| VRAM per GPU card | Memory pool budget |
| GPU count | Thread/process slots |
| hot/cold per GPU | Model loaded yes/no in pool |
| task_class routing (cpu/script/llm) | Does this task need a model? Is one available? |
| split_gpu pair coordination | Just load the model — it fits or it doesn't |
| nvidia-smi health checks | System memory + CPU utilization |
| 1 gpu.py per physical device | 1 resource manager per box |

### Brain Becomes a Resource Scheduler

Instead of coordinating hardware, the brain manages a resource pool:

- Queue has LLM tasks needing model X → is it loaded? → route to its endpoint
- Model X not loaded → do I have N GB free? → spin up a llama-server instance
- Model Y idle for 10 minutes, queue empty → unload, reclaim memory
- Queue has CPU tasks → spawn worker processes up to thread limit

The brain loop, task queue, heartbeat system, dependency graph, retry logic —
all stay the same. The vocabulary changes from "GPUs and VRAM" to "memory and threads."

---

## Multi-Rig Scaling

The file-based queue already supports distributed workers (proven by 8 Orange Pis).
Adding more GPU rigs or unified-memory boxes means each new machine:

1. Mounts the shared drive (NFS, or future NAS)
2. Runs worker processes
3. Writes heartbeats to `shared/gpus/` or `shared/cpus/`
4. Claims tasks from `shared/tasks/queue/`

### What Already Works Across Rigs
- Task queue — file-based, location-agnostic
- Heartbeat system — each worker owns its own file
- Brain monitoring — reads heartbeats regardless of source
- Signal system — file-based, workers poll their own signal files

### What Needs Work for Multi-Rig
- **nvidia-smi is local** — brain would need to trust heartbeats as source of truth instead of cross-checking with local nvidia-smi
- **Model placement locality** — split groups must be constrained to same physical box. Catalog needs a "rig" or "locality" concept
- **Model file access** — NFS works (slow for multi-GB loads). Local model cache per rig is faster but needs sync
- **Port addressing** — endpoints become `host:port` instead of just `port`

### Envisioned Multi-Rig Topology

| Rig | Hardware | Role |
|-----|----------|------|
| Unified memory box | 128-748GB | Brain + big models + multi-model serving |
| Current GPU rig | 3090 Ti + 1060s | Medium models (7-14B), script tasks |
| Orange Pi fleet | CPU only | Script/cpu tasks, cheap parallelism |
| Future rig N | Whatever | More of the above |

---

## NAS Centralization

Current storage chain is fragile:
```
Pi (USB ext4) → NFS export → GPU rig → NFS re-export → Orange Pis
```

A dedicated NAS flattens this:
```
NAS (RAID, 2.5GbE+) → direct NFS mount → all rigs
```

Benefits:
- Single authority for shared state
- RAID redundancy (current USB drive has none)
- Pi stops being a storage server, just runs control plane
- Adding new rigs = just mount the NAS
- NAS-native snapshots/backups

Considerations:
- Model load time over network (~30-60s for 7B GGUF on gigabit) — acceptable since models load once and stay resident
- NFS file locking reliability — test FileLock behavior, or rely on atomic rename (which the task lane moves may already use)
- Current git-backed plans/history means drive failure isn't catastrophic — mostly lose runtime state and downloadable models

---

## CPU Workers on Unified Memory Box

On a box where vLLM serves models from shared memory, the CPU/GPU distinction blurs.
A "CPU worker" on that box can call the local inference endpoint — it doesn't need
a separate `task_class` to route work.

CPU scaling is linear and embarrassingly parallel:
- 20 cores = ~15-20 concurrent script workers
- Adding more cores or another box cuts wall time proportionally
- Single-threaded Python tasks don't benefit from bigger cores (GIL)
- Orange Pi fleet still makes sense for cheap, always-on batch parallelism

The `task_class` system was invented to route work to capable hardware. On a box
where everything is available to every process, routing simplifies to "is there a
model loaded that can handle this?"

---

## Brain on CPU

The brain doesn't do inference itself — it calls a model via `think()`. Options:

1. **Brain on CPU, calls inference endpoint** — brain process runs on CPU, sends
   think() calls to a loaded model on the unified-memory box via OpenAI-compatible API
2. **Brain on the unified-memory box** — everything on one machine, brain is just
   another process hitting the same vLLM/llama-server
3. **Brain stays on dedicated GPU** — current approach, wastes a 3090 Ti

Option 2 is the natural end state with a unified-memory box. The brain process is
lightweight — it's the model it calls that needs resources.

---

## Implementation Priority

None of this is urgent. The current rig works. When hardware budget allows:

1. **NAS first** — centralizes storage, removes Pi as bottleneck, enables multi-rig
2. **Unified memory box** — replaces or supplements GPU rig for inference
3. **Refactor gpu.py to resource manager** — abstract from GPU slots to memory pool
4. **Simplify task_class routing** — relax CPU/GPU distinction on capable hardware
5. **Multi-rig heartbeat trust** — brain uses heartbeats as truth, drops nvidia-smi cross-check

The bones of the orchestration system are solid. The brain loop, task queue,
dependency graph, retry logic, and file-based communication all carry forward
unchanged. The evolution is a vocabulary change from "GPUs and VRAM" to
"memory and threads."

---

*References:*
- [NVIDIA DGX Spark](https://www.nvidia.com/en-us/project-digits/)
- [DGX Station GB300](https://nvidianews.nvidia.com/news/nvidia-announces-dgx-spark-and-dgx-station-personal-ai-computers)
- [Mac Studio 512GB](https://www.servethehome.com/new-512gb-unified-memory-apple-mac-studio-is-the-local-ai-play/)
- [Multiple Models Memory Management](https://www.sitepoint.com/multiple-local-models-memory-management/)
- GPU rig audit: `workspace/archive/gpu_rig_audit_2026-02-11.md`
- Current architecture: `workspace/architecture.md`
