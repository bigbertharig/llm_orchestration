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

### Current hardware

**GPU rig (10.0.0.3) — ASUS B250 Mining Expert**

| Component | Spec |
|-----------|------|
| Motherboard | ASUS B250 Mining Expert (19× PCIe slots: 1× x16, 18× x1) |
| CPU | Intel i7-7700K (4 cores / 8 threads, 4.2 GHz, LGA 1151) |
| RAM | 2× 16GB DDR4-2133 UDIMM (32GB total, non-ECC) |
| PCIe generation | 3.0 |
| PSU | Dual/triple PSU capable, ~2,200W total capacity |

| GPU | Type | VRAM | Slot | Role | Models |
|-----|------|------|------|------|--------|
| 0 | RTX 3090 | 24GB | x16 | Brain | 26B–35B models (Qwen3.6, Gemma-4 26B/31B) |
| 1–5 | GTX 1060 | 6GB each | x1 risers | Workers | 3B–7B single, 14B split (pairs of 2) |

Strengths: 19 PCIe slots, dual/triple PSU, proven stable, can hold many worker GPUs.
Limits: PCIe 3.0 only, x1 risers for workers, 32GB RAM (2 slots maxed), 2017 platform.

**CPU workers (10.0.0.10–17) — Orange Pi Prime fleet**

| Spec | Value |
|------|-------|
| Boards | 8× Orange Pi Prime (Allwinner H5) |
| CPU | 4× Cortex-A53 per node |
| RAM | 2GB |
| Storage | 30GB SD |
| Shared drive | Mounted at /media/bryan/shared |

Too constrained for LLM inference. Current viable uses:
- **Embedding generation** — `nomic-embed-text` (~270MB) for RAG indexing, 8× parallel throughput
- **Non-LLM task processing** — data prep, validation, formatting, file operations
- **Agent tool execution** — sandboxed environments for code/command execution
- **Classification with sub-1B models** — TinyLlama-1.1B, Qwen2-0.5B for yes/no routing

**Pi 5 (10.0.0.2) — Operator / control plane**

| Spec | Value |
|------|-------|
| Board | Raspberry Pi 5 Model B |
| CPU | 4× Cortex-A76 |
| RAM | 8GB |
| llama.cpp | Built at `/home/bryan/llama.cpp/build/bin/llama-server` |

Can run 3B–7B models on CPU. Measured performance:

| Metric | Pi 5 CPU (3B) | 1060 GPU (3B) | Ratio |
|--------|--------------|---------------|-------|
| Generation tok/s | 3.3 | 38.4 | ~12× slower |
| Prompt tok/s | 18–21 | 176 | ~9× slower |

**Scores are identical to GPU** — same weights, same math, only speed differs.

Speed ratio is NOT fixed — gets worse with model size (~8× for 3B, ~12× for 7B, ~15-20× for 14B estimated).

### Planned architecture: three-tier system

When a Threadripper brain rig is built, the existing B250 mining board becomes a dedicated GPU worker farm. Three tiers, each doing what its hardware is best at.

```
Brain rig (Threadripper)     — 70B models, complex reasoning, code review, orchestration
GPU worker farm (B250)       — 14× 7B workers, bulk inference, parallel task execution
CPU swarm (Orange Pi × 8)    — scripts, data prep, embeddings, file ops
Control plane (Pi 5)         — operator console, single 3B overflow
```

| Tier | Hardware | Strength | Role | Power |
|------|----------|----------|------|-------|
| **Brain** | Threadripper + 2× 3090 NVLink, 128-256GB RAM | 70B at ~16 tok/s, MoE hybrid | Complex reasoning, orchestration decisions | ~700-800W |
| **Worker farm** | B250 Mining Expert + 14× GTX 1070 8GB | 14 parallel 7B streams, ~420-490 aggregate tok/s | Bulk worker tasks, code gen, text extraction | ~2,200W |
| **CPU swarm** | 8× Orange Pi Prime (2GB) | 8 parallel script workers, cheap, always-on | Data prep, embeddings, validation, file ops | ~24W |
| **Control plane** | Pi 5 (8GB) | Operator console, single 3B model for testing | Control, monitoring, overflow | ~12W |

**How the B250 becomes a worker farm:**

1. Move the RTX 3090 to the new Threadripper brain rig
2. Keep existing 5× GTX 1060 6GB (already owned)
3. Add 9× GTX 1070 8GB at ~$65 each (~$585) to fill slots
4. Total: 14 worker GPUs (5× 1060 + 9× 1070), 96GB distributed VRAM
5. Triple PSU at ~2,200W handles 14 cards comfortably (14 × 150W = 2,100W + system)

GPU shopping for the worker farm (April 2026 used prices):

| Card | VRAM | Used price | $/GB | 7B tok/s | Power | Notes |
|------|------|-----------|------|---------|-------|-------|
| GTX 1060 6GB | 6GB | ~$59 | $9.83 | ~25-30 | ~120W | Already own 5 |
| GTX 1070 8GB | 8GB | ~$65 | $8.13 | ~30-35 | ~150W | Best $/GB, 7B fits comfortably |
| RTX 2060 6GB | 6GB | ~$125 | $20.83 | ~35-40 | ~160W | Faster but 2× price, same VRAM as 1060 |
| RTX 3060 12GB | 12GB | ~$250 | $20.83 | ~35 | ~170W | Best per-card capacity but 4× price of 1070 |

GTX 1070 8GB at $65 is the sweet spot for filling slots. PCIe x1 risers are fine — model loads once at startup, then inference runs entirely within VRAM. The 7700K just manages HTTP endpoints.

**Total system cost (incremental from current):**

| Component | Cost | Notes |
|-----------|------|-------|
| Threadripper brain rig (TRX50 path) | ~$3,650 | CPU + mobo + RAM + PSU + case (3090 moves from B250) |
| 9× GTX 1070 8GB for worker farm | ~$585 | Fill B250 slots |
| **Total new spend** | **~$4,235** | Current rig + Orange Pis + Pi 5 stay as-is |

The system gains 70B brain capability + 14 parallel 7B workers while reusing all existing hardware. The orchestrator's task queue, heartbeat system, and model placement routing already support multi-rig — each rig mounts the NFS share and claims tasks independently.

---

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
  --runtime-base http://10.0.0.2:8080 \
  --run-name <run_name>
```

Notes:
- Models load over NFS from shared drive (30–60s first load)
- Add `--reasoning-budget 0` for Qwen 3.6 family (doesn't work for all model families)
- See [MODEL_RUNTIME_GUIDE.md](/media/bryan/shared/plans/shoulders/benchmarking/MODEL_RUNTIME_GUIDE.md) for per-model runtime flags

---

## Swarm Upgrade Scenarios

### 8× Pi 5 (8GB) — replacing current Orange Pi fleet

Each node runs 3B–7B models. Aggregate throughput approaches GPU-tier for parallel workloads.

| Config | Per-node tok/s | 8-node aggregate | vs single 1060 |
|--------|---------------|-----------------|----------------|
| 3B model (SmolLM3, Llama-3.2) | ~3.3 | ~26 tok/s | ~70% |
| 7B model (Qwen2.5-Coder) | ~1.5 (est) | ~12 tok/s | ~40% |

Models that fit: SmolLM3-3B, Llama-3.2-3B, Gemma-4-E4B, Gemma-4-E2B, Qwen2.5-Coder-7B.

Use cases:
- **Parallel batch inference** — 8 independent streams. Slower per-request but 8× concurrency. Good for overnight batch jobs.
- **Redundant worker pool** — each node is a drop-in orchestrator worker. Node failure = 1/8 capacity loss, not total outage.
- **Model diversity routing** — different models on different nodes. Route code tasks to Qwen-7B nodes, text extraction to Gemma-E2B nodes, reasoning to SmolLM3 nodes.
- **Ensemble voting** — 8 nodes same prompt, different temperatures. Majority-vote improves classification reliability.
- **Draft generation for speculative decoding** — small models generate candidates, brain model verifies/accepts.

### 8× Pi 5 (16GB) — split-tier capacity expansion

Each node runs up to 14B models — more split-tier capacity than the rig (only 2 split pairs).

| Config | Per-node tok/s | 8-node aggregate | vs single 1060 |
|--------|---------------|-----------------|----------------|
| 7B model | ~1.5 (est) | ~12 tok/s | ~40% |
| 14B model (Qwen-14B, Phi-4) | ~0.8 (est) | ~6.4 tok/s | ~25% |

Additional models that fit: Qwen2.5-Coder-14B, Phi-4-14B (no split needed — single node).

Additional use cases:
- **14B model pool** — 8 parallel Qwen2.5-Coder-14B workers vs rig's 2 split instances
- **Mixed-tier fleet** — 4 nodes on 14B (quality), 4 on 3B (speed). Route by task complexity.
- **Brain offload** — some brain-tier models (Gemma-4-26B-A4B at Q3 quant) could fit in 16GB. Frees the 3090.

### Swarm economics

The value of CPU swarm is **parallelism, not speed**. Single-request latency is 10-15× worse than GPU. But:
- 8 slow workers on a 1000-task queue finish in wall-clock time comparable to 1 fast GPU
- Node failure is graceful (7/8 capacity vs 0/1)
- No GPU driver issues, CUDA conflicts, or VRAM fragmentation
- Scales linearly: add a Pi, add a worker
- Pi 5 8GB costs ~$80. Eight = $640. One 1060 6GB used = ~$120. Different cost/perf curves.

Break-even: if your workload sustains >8 concurrent requests, a Pi swarm matches GPU throughput-per-dollar. Below that, GPUs win on latency.

### Hybrid architecture (future target)

Combine all three tiers under the same orchestrator:

```
Brain (3090, GPU 0)           — complex reasoning, code review, orchestration decisions
GPU workers (1060s, GPU 1-5)  — fast single-request inference, code generation
CPU swarm (Pi 5 × 8)          — batch processing, embeddings, parallel classification
```

The brain routes tasks to the appropriate tier based on:
- **Latency requirement** — needs answer in <5s? GPU. Can wait 30s? CPU.
- **Task type** — code gen? GPU (Qwen-Coder). Text extraction? CPU (Gemma-E2B). Embeddings? CPU swarm.
- **Queue depth** — GPU queue full? Overflow to CPU nodes. All clear? Use GPU for speed.

This is the orchestrator's existing routing model extended to CPU workers. The pieces exist: task queue, worker claiming, model catalog. What's needed is CPU worker agents and model placement rules for RAM-only nodes.

---

## Split GPU Inference Benchmarks (April 2026)

Measured tok/s for multi-GPU tensor-parallel splits on the current rig. All tests
used llama-server in Docker, `--split-mode layer`, `--ctx-size 4096`,
`--flash-attn on`, 512-token generation, best-of-two warm runs.

### Results table

| Config | Model | Type | Params (active) | Prompt tok/s | Gen tok/s |
|--------|-------|------|-----------------|-------------|-----------|
| 3090 solo | gemma-4:26b-a4b | MoE | 26B (4B) | 886 | **152.4** |
| 3090 solo | qwen2.5-coder:14b | Dense | 14B | 1,138 | **82.6** |
| 3090 solo | gemma-4:31b | Dense | 31B | 454 | **40.3** |
| 3090 + 1060 (x16+x1) | gemma-4:26b-a4b | MoE | 26B (4B) | 496 | **49.1** |
| 3090 + 1060 (x16+x1) | qwen2.5-coder:14b | Dense | 14B | 415 | **32.0** |
| 3090 + 1060 (x16+x1) | gemma-4:31b | Dense | 31B | 203 | **15.4** |
| 1060 single (x1) | qwen2.5-coder:7b | Dense | 7B | 284 | **22.7** |
| 1060 2-way (x1+x1) | qwen2.5-coder:14b | Dense | 14B | 130 | **11.9** |
| 1060 4-way (x1×4) | qwen2.5-coder:14b | Dense | 14B | 135 | **11.0** |
| 1060 4-way (x1×4) | gemma-4:26b-a4b | MoE | 26B (4B) | 144 | **24.2** |
| 1060 5-way (x1×5) | gemma-4:26b-a4b | MoE | 26B (4B) | 131 | **24.0** |
| 1060 4-way (x1×4) | gemma-4:31b | Dense | 31B | 57 | **5.3** |

Prior baseline (Feb 2026, Ollama, different runtime):
- 1060 single 7B: 22.5 tok/s — matches current 22.7 tok/s within measurement noise
- 1060 2-way 14B: 12.5 tok/s — matches current 11.9 tok/s

### Key findings

**PCIe x1 is the bottleneck, not GPU count.** 4-way and 5-way MoE give identical
speed (24.2 vs 24.0 tok/s) — the activation tensor transfer through PCIe x1
(~0.25 GB/s) caps throughput regardless of how many GPUs share the work. Adding
more GPUs gives more VRAM headroom (bigger context, bigger models) without losing
speed.

**MoE on 4× cheap 1060s matches single-1060 7B speed.** The 26B MoE model at
24.2 tok/s is faster than a single 1060 running 7B dense (22.7 tok/s), despite
being a brain-class model with dramatically better reasoning. This is the free
upgrade — same speed, much smarter model, using hardware already in the rig.

**Adding one x1 card to the 3090 costs 2.6-3.1× speed.** The 3090+1060 split
shows the penalty of mixing a fast x16 card with a slow x1 card — the slowest
link determines inter-GPU transfer speed. On a proper x16+x16 board the split
penalty would be minimal (~10-15% for PCIe 4.0, ~5% for PCIe 5.0).

**MoE is consistently ~2× faster than dense at similar total params.** This
holds across every configuration — 3090 solo (152 vs 40 tok/s for 26B MoE vs
31B dense), mixed split (49 vs 15), pure 1060 4-way (24 vs 5.3).

**Asymmetric GPU splits work fine.** llama.cpp distributes layers proportionally
to available VRAM. The 3090+1060 split put 12.7GB on the 3090 and 3.3GB on the
1060 automatically. No matched pairs needed.

### Implications for upgrade planning

- A Threadripper build with 2× 3090 on PCIe 5.0 x16 slots would split with
  minimal penalty (~5%), giving ~70-75 tok/s on 14B and ~140+ tok/s on MoE
- The existing B250 worker farm can run brain-class MoE models at worker-tier
  speed using 4-5 way splits — useful when the 3090 is busy
- Future MoE models will keep getting better; the split infrastructure is ready

---

## Worker Tier Analysis

Four distinct worker architectures, each with different cost/performance tradeoffs.

### The four tiers

| Tier | What it is | Example hardware | LLM capable? |
|------|-----------|-----------------|-------------|
| **CPU script workers** | Low-RAM nodes running non-LLM tasks | Orange Pi Prime (2GB) | No — scripts, data prep, file ops only |
| **CPU LLM workers** | ARM/x86 nodes running models in system RAM | Pi 5 8-16GB, any PC with RAM | Yes — identical quality, 10-15× slower |
| **GPU workers** | Discrete GPUs running models in VRAM | GTX 1060 6GB, RTX 3060 12GB | Yes — fast inference, VRAM-limited |
| **Unified memory boxes** | Large shared memory pool, GPU + CPU | DGX Spark (128GB), Mac Studio | Yes — flexible, concurrent models |

### Side-by-side comparison

| Factor | CPU script | CPU LLM (Pi 5) | GPU (1060/3060) | Memory box (DGX Spark) |
|--------|-----------|----------------|-----------------|----------------------|
| **Speed (7B)** | n/a | ~1.5 tok/s | ~25-40 tok/s | ~50-80 tok/s (est) |
| **Speed (3B)** | n/a | ~3.3 tok/s | ~38 tok/s | ~60+ tok/s (est) |
| **Max model (Q4)** | n/a | ~7B (8GB), ~14B (16GB) | ~7B (6GB), ~14B (12GB) | ~200B (128GB) |
| **Concurrent models** | n/a | 1 per node | 1 per GPU | Many (memory pool) |
| **Output quality** | n/a | Identical to GPU | Reference | Identical to GPU |
| **Failure blast radius** | 1/8 nodes | 1/8 nodes | 1 GPU (others keep running) | Entire system |
| **Power draw** | ~3W/node | ~8-12W/node | ~120W/GPU | ~200-300W total |
| **Noise/heat** | Passive | Fan, low heat | Fans, significant heat | Fan, moderate heat |

### Cost reality (April 2026 pricing)

DRAM prices have spiked due to AI infrastructure demand. This changes the math significantly.

| Config | Unit cost | Total cost | LLM memory | Aggregate 7B tok/s |
|--------|----------|-----------|------------|-------------------|
| 8× Orange Pi Prime (2GB) | ~$30 | **~$240** | 0 (no LLM) | 0 |
| 8× Pi 5 (8GB) | ~$180 | **~$1,440** | 8× 8GB = 64GB | ~12 tok/s |
| 8× Pi 5 (16GB) | ~$300 | **~$2,400** | 8× 16GB = 128GB | ~12 tok/s |
| 1× RTX 3060 12GB (used) | ~$250 | **~$250** | 12GB VRAM | ~35 tok/s |
| 5× RTX 3060 12GB (used) | ~$250 | **~$1,250** | 60GB VRAM | ~175 tok/s |
| 1× RTX 4060 Ti 16GB (used) | ~$280 | **~$280** | 16GB VRAM | ~40 tok/s |
| 1× DGX Spark (128GB) | ~$4,700 | **~$4,700** | 128GB unified | ~50-80 tok/s |

Notes: Pi costs don't include power supplies, SD cards, cases, or switch (~$50-100 for a cluster). GPU costs don't include a host system. DGX Spark is self-contained.

### The honest tradeoff

**CPU LLM swarm (Pi cluster) vs GPU workers:**

The Pi cluster's only advantage is parallelism — 8 independent streams vs 1 GPU. But the economics are brutal:

- 8× Pi 5 16GB ($2,400) gives 128GB distributed RAM at ~12 tok/s aggregate for 7B models
- 5× RTX 3060 12GB ($1,250) gives 60GB VRAM at ~175 tok/s aggregate — **14× faster at half the price**
- A single RTX 3060 ($250) is faster than all 8 Pis combined for any single request

The Pi swarm only wins when:
- You have >8 concurrent requests sustained (rare outside batch jobs)
- You need physical distribution (edge deployment, separate locations)
- You already own the hardware and GPUs aren't an option
- Power budget matters (8 Pis = ~96W vs 5 GPUs = ~600W + host)

For a home lab adding LLM capacity, **GPUs are strictly better per dollar**.

**CPU LLM swarm vs unified memory box:**

If the goal is "lots of RAM for models," a memory box is better in every dimension:

- DGX Spark ($4,700) gives 128GB unified at ~50-80 tok/s — **same memory as 8× Pi 5 16GB but 4-7× faster, runs 200B models, multiple models concurrently**
- Pi cluster can't do what a memory box can (large models, true concurrent serving)
- Pi cluster does what a memory box can't: survive partial failure, distribute geographically

The 8× Pi 5 16GB cluster at $2,400 is the worst middle ground — too expensive for script work, too slow for serious inference, not enough memory per node for large models. It's half the cost of a DGX Spark but delivers a fraction of the capability.

**Where each tier actually makes sense:**

| Tier | Sweet spot | Avoid when... |
|------|-----------|--------------|
| CPU script (Orange Pi) | Cheap parallel task execution, embeddings, data prep. Already deployed, costs nothing to keep. | You need LLM inference of any kind. |
| CPU LLM (Pi 5) | Edge deployment where GPU isn't an option. Dev/test. Single Pi as overflow, not a cluster. | Building a cluster for throughput — GPUs win. |
| GPU workers | Primary LLM inference. Best tok/s per dollar. Proven, well-understood. | Models exceed VRAM (>7B on 6GB, >14B on 12GB, >30B on 24GB). |
| Memory box (DGX Spark) | Replacing the entire rig. Running many models concurrently. Brain + workers on one box. Future consolidation. | Budget < $4,700. Need for distributed/edge deployment. |

### Hardware upgrade options compared (April 2026 pricing)

Five realistic paths for expanding inference capacity, each with different tradeoffs.

**1. AMD Ryzen AI Max+ 395 mini PC (128GB unified) — ~$2,800**

Best budget memory box. x86 architecture means existing scripts, Docker images, and Python packages work unmodified. No CUDA (uses ROCm/Vulkan), so bench containers would need adaptation.

| Model size | Generation tok/s | Notes |
|-----------|-----------------|-------|
| 7B Q4 | ~40 | Comparable to a 3090 |
| 14B Q4 | ~20-25 (est) | Slower than discrete GPU |
| 70B Q4 | ~4.5 | Slow but it runs at all |

Multiple vendors (Framework Desktop, GMKtec, Minisforum, Beelink). Not locked to one ecosystem. ~300 GB/s memory bandwidth.

**2. NVIDIA DGX Spark (128GB unified) — $4,700**

Full CUDA stack but ARM architecture — Docker images need ARM64 rebuilds. MoE models are the killer app (~89 tok/s for 30B MoE). Dense models are slow due to 273 GB/s bandwidth (lower than AMD). Thermal throttling on sustained loads. Locked to DGX OS. Even at scale, the bandwidth wall doesn't go away — a [16× DGX Spark cluster ($75k, 2TB unified)](https://www.reddit.com/r/LocalLLaMA/comments/1sz0lyk/16x_dgx_sparks_what_should_i_run/) still maxes out at ~20 tok/s generation because each token is bottlenecked by per-node memory bandwidth, not aggregate capacity.

| Model size | Generation tok/s | Notes |
|-----------|-----------------|-------|
| 8B FP8 | ~20.5 (batch 1) | Scales to ~368 with batching |
| 20B MoE (MXFP4) | ~49.7 | MoE is the sweet spot |
| 30B MoE | ~89 | Excellent |
| 70B FP8 dense | ~2.7 | Nearly unusable |

**3. Mac Studio M3 Ultra (192-256GB) — $4,000-$8,000+**

Highest memory bandwidth (819 GB/s) means fastest 70B inference of any memory box. But macOS only — no Linux, no Docker GPU passthrough, no CUDA. The entire orchestration stack would need rework. Apple pulled the 512GB config due to DRAM costs. ARM architecture.

| Model size | Generation tok/s | Notes |
|-----------|-----------------|-------|
| 7B Q4 | ~40-50 (est) | Good |
| 14B Q4 | ~25-30 | Good |
| 70B Q4 | ~14 | Best of any memory box |

**4. Threadripper build with dual GPUs (recommended upgrade path)**

A purpose-built inference rig: Threadripper CPU, large DDR5 RAM pool, 2× high-end GPUs, full CUDA/x86/Docker compatibility, zero porting from current stack.

#### Why build instead of buying a memory box

All memory boxes (DGX Spark, Mac Studio, AMD Max+ mini PCs) have soldered RAM — what you buy is what you get forever. A Threadripper build lets you start with 128GB and scale to 1TB. GPUs swap out as better cards arrive. At every price point, the build wins on speed and flexibility.

| Budget | Build option | 70B tok/s | Memory box alternative | 70B tok/s | Verdict |
|--------|-------------|----------|----------------------|----------|---------|
| ~$5k | TRX50 + 2× 3090 + 128GB | ~16 | DGX Spark 128GB | 2.7 | Build is **6× faster** |
| ~$5k | TRX50 + 2× 3090 + 128GB | ~16 | Mac Studio M4 Max 128GB | ~7 | Build is **2× faster** |
| ~$8k | WRX90 + 2× 3090 + 128GB | ~16 | Mac Studio M3 Ultra 256GB | ~14 | Build is faster + upgradeable |
| ~$10k | WRX90 + 2× 3090 + 512GB | ~16 + MoE | Mac Studio M3 Ultra 256GB | ~14 | Build has 2× the RAM |

#### PCIe bandwidth for split models

LLM inference is memory-bandwidth-bound, not interconnect-bound. During token generation, each GPU reads model weights from its own VRAM (~936 GB/s per 3090). The inter-GPU transfer is just a small activation tensor (kilobytes). PCIe 4.0 x16 adds ~1-2ms per token — barely measurable.

| Interconnect | Bandwidth | Role | Penalty vs NVLink |
|-------------|-----------|------|-------------------|
| PCIe 4.0 x16 | ~31.5 GB/s | Activation tensor transfer | ~10-15% slower |
| PCIe 5.0 x16 | ~63 GB/s | Activation tensor transfer | ~5% slower |
| NVLink (3090 only) | 112.5 GB/s | Activation tensor transfer | Reference (best) |
| GPU VRAM (per card) | ~936 GB/s (3090) | Model weight reads — **this is the bottleneck** | n/a |

Note: NVLink is available on RTX 3090 but NOT on 3090 Ti, 4090, or 5090. All 40/50 series consumer cards are PCIe only. Buy 3090s while used stock exists.

#### GPU options

| Config | VRAM | 70B Q4 tok/s | Cost (GPUs only) | Notes |
|--------|------|-------------|-----------------|-------|
| 2× RTX 3090 + NVLink | 48GB | ~16 | ~$1,400 used | Best value for 70B |
| 2× RTX 3090 PCIe only | 48GB | ~12-14 | ~$1,400 used | Still 3-5× faster than any memory box |
| 2× RTX 5090 PCIe 5.0 | 64GB | ~40-60 (est) | ~$7,200 (street, ~$4,000 MSRP) | Fastest consumer 70B when prices drop |
| 1× RTX 5090 | 32GB | n/a (70B doesn't fit) | ~$3,600 | Fastest for ≤30B (~180 tok/s on 7B) |

#### Platform choice: TRX50 vs WRX90

| | TRX50 (Threadripper) | WRX90 (Threadripper Pro) |
|--|----------------------|--------------------------|
| **RAM slots** | 4 | **8** |
| **Memory channels** | 4 (~150-200 GB/s) | **8 (~300-400 GB/s)** |
| **Max RAM** | 1TB (4× 256GB) | **2TB** (8× 256GB) |
| **PCIe 5.0 x16 slots** | 2-3 | **7** |
| **Mobo price** | ~$600-900 | ~$1,200-1,500 |
| **Cheapest CPU** | 7960X 24-core (~$1,200-1,500) | 7965WX 24-core (~$3,500-4,000) |
| **CPU+Mobo** | ~$1,800-2,400 | ~$4,700-5,500 |

For GPU inference only (models in VRAM), extra channels don't matter — the CPU is just a bus driver. For hybrid MoE offloading (expert weights in RAM), 8-channel DDR5 at ~300-400 GB/s matches or beats a DGX Spark's 273 GB/s. That's where WRX90 earns its premium.

TRX50 has 4 slots — all populated from day one. To upgrade RAM, you replace sticks. WRX90 has 8 slots — start with 4, add 4 more later.

Key CPU point: during LLM inference the CPU is mostly idle. A 24-core and 64-core Threadripper on the same board have identical memory bandwidth and identical inference performance. The CPU doesn't do the math — it manages the PCIe bus and HTTP server. Save the money for GPUs and RAM. CPU-heavy parallel work (scripts, data prep) goes to the Orange Pi swarm.

Boards confirmed: ASUS Pro WS TRX50-SAGE WiFi (~$850), Gigabyte TRX50 AERO D (~$750), ASRock TRX50 WS (~$650). WRX90: ASUS Pro WS WRX90E-SAGE SE (~$1,350), ASRock WRX90 WS EVO (~$1,200). All support DDR5 RDIMM and multiple PCIe 5.0 x16 slots.

#### RAM strategy (April 2026 DRAM crisis pricing)

DDR5 RDIMM prices have roughly tripled since 2024 due to AI-driven fab demand. Expect relief in 2027.

**TRX50 (4 slots, must choose tier at build time):**

| Config | Sticks | Total | Est. cost | Upgrade path |
|--------|--------|-------|-----------|-------------|
| 4× 32GB RDIMM | 4 (full) | 128GB | ~$800-1,200 | Replace all 4 to grow |
| 4× 64GB RDIMM | 4 (full) | 256GB | ~$2,800-3,600 | Replace all 4 for 512GB |
| 4× 128GB RDIMM | 4 (full) | 512GB | ~$6,000-8,000+ | Max practical |

**WRX90 (8 slots, can grow incrementally):**

| Config | Sticks | Total | Est. cost | Upgrade path |
|--------|--------|-------|-----------|-------------|
| 4× 32GB (half populated) | 4 of 8 | 128GB | ~$800-1,200 | Add 4 more sticks |
| 8× 32GB (all slots) | 8 of 8 | 256GB | ~$1,600-2,400 | Replace with 64GB sticks |
| 4× 64GB (half populated) | 4 of 8 | 256GB | ~$2,800-3,600 | Add 4 more sticks |
| 8× 64GB (all slots) | 8 of 8 | 512GB | ~$5,600-7,200 | Replace with 128GB sticks |
| 8× 128GB | 8 of 8 | 1TB | ~$12,000+ | Max practical |

Starting with 128GB (4× 32GB, ~$1,000) is the pragmatic choice — covers 70B on GPU with room for context. Add or replace sticks when DRAM prices normalize.

#### Full build cost (April 2026)

**TRX50 build (value path):**

| Component | Spec | Est. price |
|-----------|------|-----------|
| CPU | Threadripper 7960X (24-core) | ~$1,300 |
| Motherboard | TRX50 (ASUS/Gigabyte/ASRock) | ~$750 |
| RAM | 4× 32GB DDR5 RDIMM (128GB) | ~$1,000 |
| GPUs | 2× RTX 3090 (used) | ~$1,400 |
| PSU | 1500-1600W | ~$250 (or secondhand) |
| Case, NVMe, cooling | — | ~$350 |
| **Total** | | **~$5,050** |

**WRX90 build (growth path):**

| Component | Spec | Est. price |
|-----------|------|-----------|
| CPU | Threadripper Pro 7965WX (24-core) | ~$3,800 |
| Motherboard | WRX90 (ASUS/ASRock) | ~$1,350 |
| RAM | 4× 32GB DDR5 RDIMM (128GB start) | ~$1,000 |
| GPUs | 2× RTX 3090 (used) | ~$1,400 |
| PSU | 1500-1600W | ~$250 (or secondhand) |
| Case, NVMe, cooling | — | ~$350 |
| **Total** | | **~$8,150** |

Both builds accept the same GPU and RAM upgrades later. The WRX90 premium (~$3,100) buys 8 RAM slots, 8 memory channels, and 7 PCIe x16 slots.

#### Power budget

The CPU doesn't do inference work — it's infrastructure. During LLM inference the CPU draws ~50-80W, not its 350W TDP.

| Component | Idle | LLM inference | Full stress (rare) |
|-----------|------|--------------|-------------------|
| Threadripper 7960X / 7965WX | ~23W | ~50-80W | ~350-420W |
| Motherboard + chipset | ~20W | ~30W | ~40W |
| 4× DDR5 RDIMM (128GB) | ~10W | ~15W | ~25W |
| NVMe + fans + misc | ~15W | ~20W | ~30W |
| **Non-GPU subtotal** | **~68W** | **~115W** | **~515W** |
| 2× RTX 3090 (350W each) | ~60W | ~600W | ~700W |
| **System total** | **~128W** | **~715W** | **~1,215W** |

A **1500-1600W PSU** covers dual 3090 inference comfortably. Dual 5090s (450W each) would push system total to ~850-1,000W under inference — still within a 1600W PSU.

Residential power: 15A/120V circuit gives 1,440W continuous (80% rule). Dual-3090 inference at ~715W is well within limits. Dual-5090 inference at ~850-1,000W still fits on a 15A circuit. Full stress exceeding 1,200W needs a 20A circuit or a dedicated run. Keep the rig on its own circuit — don't share with other high-draw devices.

#### Running giant models (671B+ / 1T)

For reference — how models like DeepSeek R1 671B and Kimi K2 1T run at home.

These are MoE (Mixture of Experts) models. Only a fraction of the model activates per token (37B of 671B for DeepSeek, 32B of 1T for Kimi). You need enough memory to store all experts, but effective bandwidth demand is similar to a 32-37B dense model.

**Hybrid GPU+RAM offloading for MoE:**
- VRAM holds attention layers + active expert path (the "hot" weights, ~32-37B)
- System RAM holds the full expert pool (all 671B-1T of weights)
- Router selects experts per token; popular experts stay cached in VRAM
- GPU accelerates the hot path; RAM provides the capacity

| Setup | RAM/VRAM | Model | tok/s | Cost |
|-------|----------|-------|-------|------|
| Dual EPYC, 384GB DDR5, CPU only | 384GB RAM | DeepSeek R1 671B IQ4_XS | ~5-8 | ~$2,000-6,000 |
| 24GB GPU + 256GB RAM (hybrid) | 24GB VRAM + 256GB RAM | Kimi K2 1T 1.8-bit | ~10 | GPU + RAM cost |
| Threadripper + 2× 3090 + 256GB | 48GB VRAM + 256GB RAM | 671B MoE Q4 hybrid | ~8-12 (est) | ~$6,000-9,000 |
| Single H100 (80GB) | 80GB HBM3 | 671B — doesn't fit | — | $25,000-40,000 |
| Single B200 (192GB) | 192GB HBM3e | 671B MoE | ~300+ | $30,000-50,000 |

Datacenter GPUs (H100, H200, B200) have HBM memory at 3,350-8,000 GB/s bandwidth — 10-30× faster than DDR5. That's why cloud inference is fast and local is slow. Nobody buys these for home use.

The Threadripper build's 256-512GB RAM pool makes it the most practical home platform for MoE hybrid offloading. The WRX90's 8-channel DDR5 at ~300-400 GB/s matches the DGX Spark's 273 GB/s for the RAM-resident expert reads.

**5. Add RTX 3060 12GB cards to existing rig — $250 each**

Drop-in replacement for 1060s. No new system needed. Each card runs 14B without splitting. ~35 tok/s on 7B. The boring option that works immediately.

### Summary table

| Option | Cost | Max model | 7B tok/s | 70B tok/s | Stack compat | Upgradeable | Best for |
|--------|------|----------|---------|----------|-------------|-------------|---------|
| AMD Max+ 395 (128GB) | ~$2,800 | ~70B | ~40 | ~4.5 | x86 ✓, no CUDA | No (soldered) | Budget memory box |
| DGX Spark (128GB) | $4,700 | ~200B MoE | ~20 | ~2.7 | CUDA ✓, ARM ✗ | No (soldered) | MoE models only |
| Mac Studio M3 Ultra (256GB) | ~$8,000 | ~200B+ | ~45 | ~14 | macOS only ✗ | No (soldered) | Standalone, no Linux |
| TRX50 + 2× 3090 (128GB) | ~$5,050 | 70B (GPU), MoE hybrid | ~50 (1 GPU) | ~16 | Full compat ✓ | GPUs + RAM (replace) | Best value 70B |
| WRX90 + 2× 3090 (128GB) | ~$8,150 | 70B (GPU), MoE hybrid | ~50 (1 GPU) | ~16 | Full compat ✓ | GPUs + RAM (add+replace) | Future-proof platform |
| WRX90 + 2× 5090 (256GB) | ~$12,000+ | 70B+ (GPU), 671B MoE | ~180 (1 GPU) | ~40-60 | Full compat ✓ | GPUs + RAM (add+replace) | Ultimate home rig |
| RTX 3060 12GB × 5 | $1,250 | ~14B per card | ~35 | n/a | Full compat ✓ | n/a (existing rig) | Cheapest, immediate |

### Upgrade path recommendation

Given current pricing, stack compatibility, and the system's needs:

1. **Now: Keep current rig + Orange Pis.** The 3090 + 5× 1060 + 8× Orange Pi setup works. Don't buy Pi 5s for LLM inference.

2. **Quick win: Add 1-2 RTX 3060 12GB cards.** Swap out 1-2 of the weakest 1060s. $250-500, immediate 14B capacity per card, no other changes needed.

3. **Medium upgrade: TRX50 + 2× RTX 3090 + NVLink + 128GB.** ~$5,050 total. Runs 70B at ~16 tok/s — 6× faster than a DGX Spark. Full CUDA/x86/Docker compatibility. The 3090 is the last consumer card with NVLink — buy used while stock exists. Use this rig as the new brain, free the current 3090 for worker duty.

4. **Big upgrade: WRX90 + 2× RTX 3090 + NVLink + 128GB.** ~$8,150 total. Same 70B performance, but 8 RAM slots (grow to 1TB), 8 memory channels (MoE hybrid ready), and 7 PCIe x16 slots (add more GPUs later). The 5-year platform.

5. **GPU upgrade path: Swap 3090s for 5090s.** When RTX 5090 prices normalize from ~$3,600 to MSRP ~$2,000. 64GB VRAM, ~40-60 tok/s on 70B, ~180 tok/s on 7B. The GPUs swap into either platform.

6. **RAM upgrade path: Add sticks as DRAM drops.** Current crisis pricing (~3× normal) expected to ease in 2027. On WRX90, start with 4× 32GB, add 4 more later. On TRX50, replace 4× 32GB with 4× 64GB when affordable.

7. **Orange Pis stay.** CPU script workers (data prep, embeddings, file ops) remain on the swarm. The Threadripper CPU doesn't need to do this work — it's a bus driver for GPUs and RAM, not a compute node. The swarm handles parallel CPU tasks.

For the full unified-memory abstraction evolution plan (orchestrator changes, multi-rig scaling, NAS centralization), see [ai_box_upgrade.md](future/ai_box_upgrade.md).

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
5. Update [model_task_library.json](/home/bryan/llm_orchestration/shared/plans/shoulders/benchmarking/model_task_library.json) with the routing rule
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
| [ai_box_upgrade.md](future/ai_box_upgrade.md) | unified memory hardware evolution (DGX Spark, Mac Studio, multi-rig) |
| [MODEL_LIBRARY.md](/media/bryan/shared/plans/shoulders/benchmarking/MODEL_LIBRARY.md) | model selection, benchmark scores, best choices per task |
| [MODEL_RUNTIME_GUIDE.md](/media/bryan/shared/plans/shoulders/benchmarking/MODEL_RUNTIME_GUIDE.md) | per-model runtime flags, memory, compatibility |
| [BENCHMARK_SCORES.md](/media/bryan/shared/plans/shoulders/benchmarking/BENCHMARK_SCORES.md) | raw benchmark score tables with timing |
| [model_task_library.json](/home/bryan/llm_orchestration/shared/plans/shoulders/benchmarking/model_task_library.json) | machine-readable task routing rules |
