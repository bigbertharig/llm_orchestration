# Hardware Upgrades

*Updated: 2026-11-13*
*Status: Research and future planning; no purchase decision*

This is the planning document for hardware build options, worker upgrades,
compact CPU/LLM boxes, and the supporting performance evidence. Current model
loads and measured speeds remain owned by the benchmarking repository. The
distributed work guide owns workflow and task routing.

The current planning direction is to keep the mining board as a small-worker box
and build a separate DDR4 host for the owned 3090 Ti plus future 3090-class or
larger GPUs. The November price table below is the current buying snapshot.
Hardware assignments and purchase timing remain experimental.

---

## Current hardware baseline

**GPU rig (<rig-ip>) — ASUS B250 Mining Expert**

| Component | Spec |
|-----------|------|
| Motherboard | ASUS B250 Mining Expert (19× PCIe slots: 1× x16, 18× x1) |
| CPU | Intel i7-7700K (4 cores / 8 threads, 4.2 GHz, LGA 1151) |
| RAM | 2× 16GB DDR4-2133 UDIMM (32GB total, non-ECC) |
| PCIe generation | CPU x16 at Gen3; chipset x1 risers at Gen2; C17/C18 behind a Gen1 x1 switch |
| PSU | Dual/triple PSU capable, ~2,200W total capacity |

| GPU | Type | VRAM | Slot | Role | Models |
|-----|------|------|------|------|--------|
| 0 | RTX 3090 Ti | 24GB | x16 | Brain | 26B–35B models (Qwen3.6, Gemma-4 26B/31B) |
| 1–5 | GTX 1060 | 6GB each | x1 risers | Workers | 3B–7B single, 14B split (pairs of 2) |

Strengths: 19 PCIe slots, dual/triple PSU, proven stable, can hold many worker GPUs.
Limits: 32GB RAM (2 slots maxed), one x16 slot, 2017 platform.

**CPU workers (<worker-ip>–17) — Orange Pi Prime fleet**

| Spec | Value |
|------|-------|
| Boards | 8× Orange Pi Prime (Allwinner H5) |
| CPU | 4× Cortex-A53 per node |
| RAM | 2GB |
| Storage | 30GB SD |
| Shared drive | Mounted at <pi-shared-root> |

Too constrained for LLM inference. Current viable uses:
- **Embedding generation** — `nomic-embed-text` (~270MB) for RAG indexing, 8× parallel throughput
- **Non-LLM task processing** — data prep, validation, formatting, file operations
- **Agent tool execution** — sandboxed environments for code/command execution
- **Classification with sub-1B models** — TinyLlama-1.1B, Qwen2-0.5B for yes/no routing

**Pi 5 (<pi-ip>) — Operator / control plane**

| Spec | Value |
|------|-------|
| Board | Raspberry Pi 5 Model B |
| CPU | 4× Cortex-A76 |
| RAM | 8GB |
| llama.cpp | Built at `<llama.cpp-checkout>/build/bin/llama-server` |

Can run 3B–7B models on CPU. Measured performance:

| Metric | Pi 5 CPU (3B) | 1060 GPU (3B) | Ratio |
|--------|--------------|---------------|-------|
| Generation tok/s | 3.3 | 38.4 | ~12× slower |
| Prompt tok/s | 18–21 | 176 | ~9× slower |

**Scores are identical to GPU** — same weights, same math, only speed differs.

Speed ratio is NOT fixed — gets worse with model size (~8× for 3B, ~12× for 7B, ~15-20× for 14B estimated).

---

## Compact CPU workers with side-LLM capacity (checked 2026-10-29)

Candidates for a single x86 worker that can run scripts for the Orange Pi fleet
and serve a local model when useful. These are additions to the system, not
assumed replacements for the GPU rig. Memory is shared by the CPU, GPU, OS,
model weights, KV cache, and runtime; it is not dedicated VRAM. Prices and
availability are snapshots, not purchase decisions.

| Box | Shared memory | Price / availability checked | Why watch it |
|---|---:|---|---|
| [LattePanda Sigma 2](https://www.dfrobot.com/product-3164.html) (Core Ultra X7 358H, Arc B390) | 64GB LPDDR5X-8533; up to 59GB allocatable to GPU; 136.5GB/s theoretical | $2,599 presale through mid-November 2026, then $2,799; still marked preorder with **no published ship date** | Very small x86 board with dual 10GbE, multiple M.2 slots, and OCuLink; possible CPU worker plus side LLM |
| [Khadas Mind Pro](https://www.khadas.com/product-page/mind-pro) (same X7 358H class) | 64GB LPDDR5X-8533 | Store lists $2,599 for the 64GB model | Tiny finished PC with broadly similar compute/memory; fewer exposed server-style connections than Sigma 2 |
| [GMKtec EVO-X2](https://www.gmktec.com/en/products/amd-ryzen%E2%84%A2-ai-max-395-evo-x2-ai-mini-pc) (Ryzen AI Max+ 395, Radeon 8060S) | 64GB or 128GB LPDDR5X-8000 | 64GB/1TB selected listing $1,999.99; check configuration and stock before purchase | Stronger 16-core/32-thread CPU and larger GPU; 128GB option offers more room for models and context |
| [Framework Desktop](https://frame.work/products/desktop-diy-amd-aimax300/body/new) (Ryzen AI Max+ 395) | 64GB or 128GB LPDDR5X | DIY 64GB $1,959 shown out of stock; DIY 128GB $3,449 before storage/OS | More serviceable desktop form factor; 128GB option for larger side models |
| [Mac mini M5 Pro](https://support.apple.com/en-am/128108) | Up to 64GB unified; 307GB/s stated bandwidth | Check configured price and stock | Compact standalone inference option; macOS/ARM requires a separate integration plan for Linux/x86 workers |

Capacity example: the rig's Llama 3.3 70B Q4 file is 42.52GB, so a
64GB box could plausibly hold it at a modest context, but that does not
establish its decode speed or long-context fit. The rig measured 4.57 tok/s
for dense Llama 70B and 19.16 tok/s for 74B/9B-active Nemotron in a
matched six-card, 2K-context run; the MoE and dense results must not be
treated as interchangeable
([rig measurements](../../plans/shoulders/benchmarking/RIG_PERFORMANCE.md#dense-70b-versus-74b-moe-on-nvme-2026-10-25)).
The 1070 worker-farm upgrade is still a separate path.

---

## Worker tier analysis

Four distinct worker architectures, each with different cost/performance tradeoffs.

### The four tiers

| Tier | What it is | Example hardware | LLM capable? |
|------|-----------|-----------------|-------------|
| **CPU script workers** | Low-RAM nodes running non-LLM tasks | Orange Pi Prime (2GB) | No — scripts, data prep, file ops only |
| **CPU LLM workers** | ARM/x86 nodes running models in system RAM | Pi 5 8-16GB, any PC with RAM | Yes — identical quality, 10-15× slower |
| **GPU workers** | Discrete GPUs running models in VRAM | GTX 1060 6GB, RTX 3060 12GB | Yes — fast inference, VRAM-limited |
| **Unified memory boxes** | Large shared memory pool, GPU + CPU | DGX Spark (128GB), Mac Studio | Yes — flexible, concurrent models (placement design: llm-gateway `docs/DESIGN_NOTES.md`, "Sharing capacity") |

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

### Cost and workload note

The current price and build table is below the driver section. The worker tiers above describe roles and memory architecture; purchasing choices should use the November table.

---

## Split-card platform notes

The PCIe and hardware specifications below remain useful for design. Current buying prices and build totals are in the November 2026 section after the driver notes.

### PCIe bandwidth for split models

LLM inference is memory-bandwidth-bound, not interconnect-bound. During token generation, each GPU reads model weights from its own VRAM (~936 GB/s per 3090). The inter-GPU transfer is just a small activation tensor (kilobytes). PCIe 4.0 x16 adds ~1-2ms per token — barely measurable.

| Interconnect | Bandwidth | Role | Penalty vs NVLink |
|-------------|-----------|------|-------------------|
| PCIe 4.0 x16 | ~31.5 GB/s | Activation tensor transfer | ~10-15% slower |
| PCIe 5.0 x16 | ~63 GB/s | Activation tensor transfer | ~5% slower |
| NVLink (3090 only) | 112.5 GB/s | Activation tensor transfer | Reference (best) |
| GPU VRAM (per card) | ~936 GB/s (3090) | Model weight reads — **this is the bottleneck** | n/a |

Note: NVLink is available on RTX 3090 but NOT on 3090 Ti, 4090, or 5090. All 40/50 series consumer cards are PCIe only. Buy 3090s while used stock exists.

### GPU purchase focus

For the new 100–300B host, plan around the already-owned 3090 Ti and additional RTX 3090-class or larger cards. The lower-capacity cards in the specification table are reference hardware, not a new-rig shopping list.

### GPU VRAM capacity per rated board watt (checked 2026-11-09)

This is a capacity and power-screening table, not a model-speed ranking or a shopping list. **W/GB = manufacturer maximum board power divided by physical VRAM; lower is better for this one metric.** The power column is a rating, not measured inference draw. Peak memory bandwidth is included because it can matter for decode, but it does not predict tokens/s on its own. Host power, idle draw, power limits, cooling, price, model fit, and runtime support also matter. A multi-card total is not one seamless VRAM pool; split models have per-card overhead and placement constraints.

| Card | VRAM | Rated board power | W/GB | Peak memory bandwidth | Capacity/power note |
|------|-----:|------------------:|-----:|----------------------:|---------------------|
| GTX 1060 6GB | 6 GB | 120 W | 20.00 | 192 GB/s | Existing worker card; reference specification |
| GTX 1070 | 8 GB | 150 W | 18.75 | 256 GB/s | Existing worker-farm comparison |
| RTX 3090 | 24 GB | 350 W | 14.58 | 936 GB/s | Older 24GB build option; NVLink capable |
| RTX 3090 Ti | 24 GB | 450 W | 18.75 | 1,008 GB/s | Current rig card; its configured 350W cap is separate from this rating |
| Tesla M10 | 4 × 8 GB | 225 W for board | 7.03 across 32 GB | 83 GB/s per GPU | Four separate Maxwell GPUs, not one 32GB pool; 1× PCIe 8-pin, passive cooling, CUDA `sm_50` build needed |
| RTX 3060 12GB | 12 GB | 170 W | 14.17 | 360 GB/s | Existing 12GB worker option |
| RTX 4060 Ti 16GB | 16 GB | 165 W | 10.31 | 288 GB/s | Older 16GB baseline |
| RTX 5060 Ti 16GB | 16 GB | 180 W | 11.25 | 448 GB/s | Use the 16GB variant, not the 8GB one |
| RTX 5070 Ti | 16 GB | 300 W | 18.75 | 896 GB/s | More bandwidth than 5060 Ti, same VRAM |
| RTX 5080 | 16 GB | 360 W | 22.50 | 960 GB/s | Faster memory, limited capacity per watt |
| RTX 5090 | 32 GB | 575 W | 17.97 | 1,792 GB/s | Largest current GeForce VRAM in this comparison |
| Tesla V100S 32GB PCIe | 32 GB HBM2 | 250 W | 7.81 | 1,134 GB/s | Used passive card; passive cooling and 32GB BAR1 BIOS mapping required |
| A100 40GB PCIe | 40 GB HBM2 | 250 W | 6.25 | 1,555 GB/s | Older data-center accelerator; passive cooling requires server airflow |
| A100 80GB PCIe | 80 GB HBM2e | 300 W | 3.75 | 1,935 GB/s | Older data-center accelerator; passive cooling requires server airflow |
| A800 40GB Active | 40 GB HBM2 | 240 W | 6.00 | 1,555 GB/s | Active-cooled workstation accelerator; check used availability |
| RTX A6000 (Ampere) | 48 GB ECC | 300 W | 6.25 | 768 GB/s | Active-cooled 48GB workstation card; `sm_86` |
| A40 | 48 GB ECC | 300 W | 6.25 | 696 GB/s | Older data-center card; passive cooling requires directed airflow |
| L40 | 48 GB ECC | 300 W | 6.25 | 864 GB/s | Ada data-center card; passive cooling requires directed airflow |
| L40S | 48 GB ECC | 350 W | 7.29 | 864 GB/s | Ada data-center card; passive cooling requires directed airflow |
| RTX 6000 Ada | 48 GB ECC | 300 W | 6.25 | 960 GB/s | Active-cooled 48GB workstation card; `sm_89` |
| RTX PRO 2000 Blackwell | 16 GB ECC | 70 W | 4.38 | 288 GB/s | Low-power workstation option |
| RTX PRO 4000 Blackwell SFF | 24 GB ECC | 70 W | 2.92 | 432 GB/s | Very low board rating; half-height, dual-slot |
| RTX PRO 4000 Blackwell | 24 GB ECC | 145 W | 6.04 | 672 GB/s | Single-slot; NVIDIA's comparison page lists 145W (another product page lists 140W) |
| RTX PRO 4500 Blackwell | 32 GB ECC | 200 W | 6.25 | 896 GB/s | 32GB at a lower rating than RTX 5090 |
| RTX PRO 5000 Blackwell | 48 GB ECC | 300 W | 6.25 | 1,344 GB/s | High single-card capacity |
| RTX PRO 5000 Blackwell 72GB | 72 GB ECC | 300 W | 4.17 | 1,344 GB/s | Same rated power as 48GB variant |
| RTX PRO 5500 Blackwell | 84 GB ECC | up to 600 W | 7.14 | 1,398 GB/s | Preliminary published specification |
| RTX PRO 6000 Blackwell Max-Q | 96 GB ECC | 300 W | 3.13 | 1,792 GB/s | Highest single-card capacity here at 300W |
| RTX PRO 6000 Blackwell | 96 GB ECC | 600 W | 6.25 | 1,792 GB/s | Full-power workstation version |

The 24GB/70W PRO 4000 SFF and 96GB/300W PRO 6000 Max-Q look strongest on **rated watts per GB**, while the 5090 has much higher bandwidth than the 16GB GeForce cards. This is an inference from specifications, not a measured LLM result. The 70W SFF card is not an automatic swap into an x1-riser farm: check slot power, riser wiring, form factor, airflow, driver, and llama.cpp support first. Before buying, compare actual board prices and run the same model, context, batch size, and power cap to measure tokens/s and joules/token.

Specification sources: [GTX 1060 reference specs](https://www.nvidia.com/en-us/geforce/news/nvidia-geforce-gtx-1060/), [GTX 1070 reference specs](https://images.nvidia.com/aem-dam/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf), [RTX 3090/3090 Ti reference bandwidth](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf), [RTX 3060 bandwidth](https://www.nvidia.com/en-us/geforce/news/game-on-you-asked-we-answered-qa/), [RTX 4060 Ti bandwidth](https://www.nvidia.com/en-us/geforce/news/rtx-40-series-vram-video-memory-explained/), [NVIDIA GeForce comparison](https://www.nvidia.com/en-us/geforce/graphics-cards/compare/) (RTX 30/40/50 memory and power), [NVIDIA A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf) (PCIe 40/80GB), [NVIDIA RTX PRO desktop comparison](https://www.nvidia.com/en-in/products/workstations/professional-desktop-gpus/) (PRO memory and power), [RTX PRO 2000 specifications](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-2000/), [RTX PRO 4000 SFF datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/rtx-pro-4000-blackwell/workstation-datasheet-blackwell-rtx-pro-4000-gtc25s-nvidia-3662515-r6.pdf), [RTX PRO 4500 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/rtx-pro-4500-blackwell/workstation-datasheet-blackwell-rtx-pro-4500-we-nvidia-us-5108623-web.pdf), [RTX PRO 5000 specifications](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-5000/), [RTX PRO 5500 specifications](https://www.nvidia.com/en-in/products/workstations/professional-desktop-gpus/rtx-pro-5500/), and [RTX PRO 6000 Max-Q specifications](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-max-q/). The current price table is below the driver section; rated watts and peak bandwidth are specifications, not measured model speeds.

### GPU upgrade watchlist for the present B250 rig

The B250 Mining Expert has one CPU x16 slot (occupied by the 3090 Ti), eighteen x1 slots, and a 32GB RAM limit. A card's fit on this rig depends on PCIe enumeration, BIOS address resources, driver and CUDA support, power, and cooling. No candidate below has been installed or verified here.

Watch used **RTX A6000 48GB**, **RTX 6000 Ada 48GB**, **A40 48GB**, **L40/L40S 48GB**, **Tesla V100S 32GB PCIe**, and **Tesla M10 (4 × 8GB)** for current-rig possibilities. **A100 40/80GB PCIe** is better planned with a newer host; A100 SXM modules are not PCIe cards. **A800 40GB Active** requires at least 40GB host RAM under NVIDIA's installation guide, above this board's 32GB maximum. Passive cards need directed airflow even in an open frame.

**Current B250 / 32GB RAM compatibility screen (planning, no card installed):**

| Candidate | Present rig | Reason / condition |
|-----------|-------------|--------------------|
| RTX A6000 48GB, RTX 6000 Ada 48GB | Best current-rig candidates | Workstation cards default to a 256MB BAR1 address window and appear on the current 580.159.03 supported-products list. The 32GB system RAM is not an inherent 48GB GPU limit. Ada requires adding `sm_89` to the runtime image; RTX A6000 uses the image's existing `sm_86`. |
| Tesla M10 (4 × 8GB) | Conditional current-rig candidate | Four separate Maxwell GPUs, not a shared 32GB pool. Requires a CUDA `sm_50` runtime build, one PCIe 8-pin lead, directed airflow, and verification of four-GPU enumeration on this motherboard. |
| Tesla V100S 32GB PCIe | Conditional current-rig candidate; strong price watch | NVIDIA lists it on driver 580.159.03, and CUDA 12.9 supports Volta `sm_70`, but the runtime image needs that target added. The card requires a 32GB BAR1 address window that may challenge the older B250 BIOS. It is a 250W passively cooled PCIe card with a CPU/EPS 8-pin socket; check the specific card's cooling, power adapter, BIOS mapping, and PCIe fit. The 32GB system RAM is not an automatic exclusion. |
| A40 48GB, L40/L40S 48GB | Conditional; prefer a proven host before purchase | These server cards default to display-off mode with a 64GB BAR1 address window, which an older B250 BIOS may not map. NVIDIA's display-mode tool documents a 256MB BAR1 display-on mode, but mode change and boot on this exact board are untested. They also need directed cooling. |
| A100 40GB/80GB PCIe | Plan a newer server/workstation host | The PCIe cards request 64GB/128GB BAR1 windows respectively, and NVIDIA does not qualify A100 on ordinary consumer motherboards. The 32GB physical RAM is not itself a published A100 prohibition; BIOS resource mapping and server airflow are the stronger blockers. |
| A800 40GB Active | New host required under NVIDIA's published minimum | Its guide requires host RAM at least equal to 40GB GPU memory (80GB recommended) and support for a 64GB BAR1 mapping; the B250 is limited to 32GB RAM. |
| RTX PRO 5000/6000 Blackwell 48-96GB | Separate host if keeping the GTX 1060/1070 on this rig | Blackwell needs NVIDIA's open kernel module while Pascal requires the proprietary module. Removing every Pascal card could allow a Blackwell trial on this board, but BIOS, power, and x16-slot fit remain unverified; a newer host is the sensible plan for these cards. |

BAR1 is a PCIe *address window*, not an equal amount of physical system RAM. The compatibility labels are planning guidance, not a successful boot test on this specific motherboard. Sources: [ASUS B250 manual](https://dlcdnet.asus.com/pub/ASUS/mb/LGA1151/B250_MINING_EXPERT/E13594_B250_MINING_EXPERT_UM_V2_WEB.pdf) (32GB maximum), [NVIDIA display-mode guide](https://developer.download.nvidia.com/display-mode-selector-tool/NVIDIA-Display-Mode-Selector-Tool-User-Guide_v08_Feb24.pdf) (workstation versus server BAR1 defaults), [A100 40GB](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/A100-PCIE-Prduct-Brief.pdf) and [80GB](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/PB-10577-001_v02.pdf) product briefs (BAR1 sizes), [A800 installation guide](https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/quadro-product-literature/proviz-a800-40gb-active-online-qsg-210x148mm-20231026-r11.pdf) (host requirements), [NVIDIA A100 motherboard guidance](https://forums.developer.nvidia.com/t/a100-pcie-isnt-recognized-by-bios/198546), [current driver supported chips](https://download.nvidia.com/XFree86/Linux-x86_64/580.159.03/README/supportedchips.html), and [kernel-module compatibility](https://download.nvidia.com/XFree86/Linux-x86_64/580.159.03/README/kernel_open.html).

V100S evidence: [NVIDIA's V100S datasheet](https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf) (memory, bandwidth, power, cooling), [NVIDIA V100S product brief copy](https://www.trungtammaychu.vn/content_trungtammaychu/upload/files/NVIDIA%20V100s%20Product%20Brief.pdf) (32GB BAR1 and CPU 8-pin socket), [CUDA 12.9 Volta target](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-compiler-driver-nvcc/index.html). Distinguish V100S PCIe from older V100 PCIe and V100 SXM2.

**Auxiliary power connectors for the reference cards:**

| Card | Card socket | Fit with existing PCIe 6/8-pin PSU leads |
|------|-------------|-------------------------------------------|
| RTX A6000, A40, A100 40GB/80GB PCIe | One CPU/EPS 8-pin | Requires a PSU CPU/EPS 8-pin lead or the correct GPU PCIe 8-pin-to-CPU 8-pin adapter; PCIe 8-pin and CPU 8-pin have different pinouts |
| Tesla V100S 32GB PCIe | One CPU/EPS 8-pin | NVIDIA's product brief supports a CPU 8-pin cable or its specified adapter fed by GPU PCIe 6/8-pin leads; do not plug a PCIe lead directly into the CPU/EPS socket |
| A800 40GB Active | One PCIe CEM5 16-pin | NVIDIA's installation guide calls for its included adapter and two separate PCIe 8-pin PSU cables |
| L40 | One PCIe CEM5 16-pin | Native compatible 16-pin cable, or NVIDIA-documented CPU 8-pin-to-16-pin adapter with correct sense wiring |
| L40S | One 16-pin | Confirm the exact card's approved 16-pin cable/adapter and PSU capacity before purchase |
| RTX 6000 Ada | One PCIe CEM5 16-pin | NVIDIA's installation guide calls for its included adapter and two separate PCIe 8-pin PSU cables |

Do not put a GPU PCIe 8-pin plug directly into a CPU/EPS 8-pin socket. Check the exact card and adapter included with a used listing; verify total PSU capacity for the card plus all remaining GPUs and fans. NVIDIA's [A800 installation guide](https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/quadro-product-literature/proviz-a800-40gb-active-online-qsg-210x148mm-20231026-r11.pdf) has inconsistent wording in its equipment table (one 8-pin) and installation step (two separate 8-pin leads), so inspect the actual included adapter before planning cables.

Connector sources: [RTX A6000 power guidelines](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/workstation-power-guidelines.pdf), [A40 datasheet](https://images.nvidia.com/content/Solutions/data-center/a40/nvidia-a40-datasheet.pdf), [A100 40GB product brief](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/A100-PCIE-Prduct-Brief.pdf), [A100 80GB product brief](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/PB-10577-001_v02.pdf), [L40 product brief](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/datasheets/L-40/product-brief-L40.pdf), [L40S specifications](https://www.nvidia.com/en-us/data-center/l40s/), and [RTX 6000 Ada installation guide](https://images.nvidia.com/aem-dam/Solutions/design-visualization/quadro-product-literature/proviz-rtx-6000-ada-gen-online-qsg-210x148mm-20230118-r9.pdf).

Watchlist specifications: [A800 40GB Active](https://www.nvidia.com/en-us/products/workstations/a800/), [RTX A6000](https://www.nvidia.com/en-us/products/workstations/rtx-a6000/), [A40](https://www.nvidia.com/en-us/data-center/a40/), [L40](https://www.nvidia.com/en-au/data-center/l40/), [L40S](https://www.nvidia.com/en-us/data-center/l40s/), [RTX 6000 Ada](https://www.nvidia.com/en-us/products/workstations/rtx-6000/), [Tesla M10 datasheet](https://images.nvidia.com/content/pdf/grid/data-sheet/nvidia-m10-datasheet.pdf), and [ASUS B250 Mining Expert manual](https://dlcdnets.asus.com/pub/ASUS/mb/LGA1151/B250_MINING_EXPERT/E13594_B250_MINING_EXPERT_UM_V2_WEB.pdf).

### Driver and CUDA fit with the current rig (checked 2026-11-09)

The live rig reports driver **580.159.03**, five GTX 1060 6GB cards (`sm_61`), and one RTX 3090 Ti (`sm_86`). Its `modinfo -F license nvidia` returns `NVIDIA`, confirming the proprietary Linux kernel module. The CUDA **12.9.2** llama runtime image is built only for architectures `61;86` (see `shared/scripts/llama_runtime/Dockerfile`). Driver version, kernel-module flavor, and CUDA build targets are separate requirements.

| Card family | With GTX 1060/1070 on this Linux host? | With RTX 3090 Ti on a separate Linux host? | Runtime build target |
|-------------|--------------------------------------|-------------------------------------------|---------------------|
| A100 40/80GB, A800 40GB Active, A40, RTX A6000, L40/L40S, RTX 6000 Ada | Driver family permits the proprietary 580 module; hardware and product-specific setup still need checking | Yes, with a suitable 580 driver | `sm_80` for A100/A800; `sm_86` for A40/RTX A6000; `sm_89` for L40/L40S/RTX 6000 Ada |
| GeForce RTX 50 and listed RTX PRO Blackwell cards (including PRO 5000 48/72GB and PRO 6000 96GB) | **No.** Blackwell needs the open module, Pascal needs the proprietary module, and both cannot load in one Linux kernel | **Yes in principle.** Both Blackwell and the 3090 Ti can use the open 580 module | `sm_120` for these Blackwell cards; retain `sm_86` for the 3090 Ti |

NVIDIA's [580.159.03 supported-products list](https://download.nvidia.com/XFree86/Linux-x86_64/580.159.03/README/supportedchips.html) includes the 1060/1070, 3090 Ti, RTX 5090, PRO 5000 48/72GB, and PRO 6000 96GB. **That list alone does not imply that Pascal and Blackwell work together:** the [same driver's kernel-module chapter](https://download.nvidia.com/XFree86/Linux-x86_64/580.159.03/README/kernel_open.html) says proprietary supports through pre-Blackwell, open supports Turing and newer, and the two flavors are mutually exclusive. NVIDIA's [architecture matrix](https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html) marks R580 as the final Pascal driver branch and CUDA 12.x as its final toolkit family; [CUDA 12.9.2's compiler](https://docs.nvidia.com/cuda/archive/12.9.2/cuda-compiler-driver-nvcc/index.html) can target both `sm_61` and `sm_120`. The present `61;86` image already includes the Ampere `sm_86` target used by RTX A6000/A40; it needs a rebuild to add `sm_80` for A100/A800, `sm_89` for Ada, or `sm_120` for Blackwell. A new Blackwell/3090 host can use an open 580 driver and a `86;120` build while leaving the Pascal rig on proprietary 580 and `61;86`.

The newly listed RTX PRO 5500 Blackwell (84GB) is **not present** in the exact 580.159.03 supported-products list checked here. Verify its device ID and minimum driver from NVIDIA before planning it on a 580 host. No new card was installed or runtime-tested by this review.

## November 2026 build prices and capacity targets

These are **US dollar asking prices checked November 13, 2026**, before tax, shipping, import fees, and testing. They are purchase-planning ranges, not completed-sale prices or speed measurements. Keep one current price column here; the older April build budgets and card shopping estimates have been removed. The existing 3090 Ti, PSUs, drives, and 4 × 8GB DDR4 UDIMMs are already owned. The UDIMMs cannot be mixed with the ECC RDIMMs required by the EPYC board.

| Component | November 2026 price | Source and constraint |
|---|---:|---|
| Unlocked EPYC 7302P, used | $100–160 | [Used listings](https://www.ebay.com/sch/i.html?_nkw=amd+epyc+7302p&_sop=12); exclude Dell-locked CPUs unless board compatibility is established. |
| Supermicro H12SSL-i, used/open box | $1,000–1,200 | [Used listings](https://www.ebay.com/sch/i.html?_nkw=supermicro+h12ssl-i&_sop=12); a CPU/board bundle was listed around $1,170. |
| 8 × 32GB DDR4-3200 ECC RDIMM, 256GB total | $1,760–2,600 | [Module listings](https://www.ebay.com/sch/i.html?_nkw=samsung+32gb+ddr4+3200+rdimm&_sop=12) show roughly $220–325 each. Verify rank, part number, and board QVL. |
| RTX 3090 24GB, used | $1,450–1,800 each | [Current listings](https://www.ebay.com/sch/i.html?_nkw=rtx+3090&_sop=12); [$1,649 median used ask](https://cresvament.com/prices/graphics-cards/rtx-3090) on Nov 13. Test memory, thermals, and power connectors. |
| RTX A6000 48GB, used | $4,300–4,600 each | [Current listings](https://www.ebay.com/sch/i.html?_nkw=nvidia+rtx+a6000&_sop=12); capacity option, not the budget default. |
| WRX80 motherboard, used | $1,100–1,650 | [Current listings](https://www.ebay.com/sch/i.html?_nkw=wrx80+motherboard&_sop=12); DDR4 workstation alternative when a suitable CPU bundle is available. |
| Threadripper Pro 7965WX, used | $1,500–1,900 unlocked | [Current listings](https://www.ebay.com/sch/i.html?_nkw=threadripper+7965wx&_sop=12); Dell-locked parts can appear much cheaper. Requires DDR5 RDIMM/WRX90, not DDR4. |
| WRX90 motherboard, used | $950–1,200 | [Current listings](https://www.ebay.com/sch/i.html?_nkw=asus+pro+ws+wrx90e-sage&_sop=12). |
| DDR5-4800 ECC RDIMM, 32GB | roughly $800–1,050 each | [Current listings](https://www.ebay.com/sch/i.html?_nkw=ddr5+4800+ecc+rdimm&_sop=12); 256GB at eight channels is roughly $6,400–8,400 in RAM alone. |
| DGX Spark, 128GB | about $6,950 | [November price check](https://physicalailab.io/en/nvidia-dgx-price-spark-station-b300/); fixed unified memory, separate from the expandable GPU-host route. |

The [H12SSL-i specification](https://www.supermicro.com/en/products/motherboard/h12ssl-i) gives **8 DDR4 RDIMM slots, 2TB maximum, five PCIe 4.0 x16 slots and two x8 slots**. The EPYC 7302P is the cheapest practical entry here. WRX80 has more workstation-style slot layouts but current boards cost more. WRX90 adds DDR5 and PCIe 5.0, at a much higher memory cost today. These are platform choices, not claims that any listed CPU delivers a measured token rate.

### Model-size planning

Q4 storage is often around **0.55–0.7 GB per billion parameters**, but higher-precision tensors, n-gram tables, vision modules, and KV cache can increase the total. Actual files are the authority. Multiple GPUs do not form one automatic VRAM pool: the runtime must support the model architecture and split placement. RAM-resident experts also need CPU memory bandwidth and may be much slower than VRAM-resident ones.

| Target | Approximate Q4 weights | Practical starting placement | Price with already-owned 3090 Ti |
|---|---:|---|---:|
| Around 100B, mostly VRAM | 55–70GB | 3090 Ti + 2–3 additional 3090s (72–96GB total VRAM); choose 3 for useful margin | **$5,800–9,400** new spend: EPYC board/CPU, 256GB RAM, 2–3 used GPUs; excludes cooling, chassis, extra PSU, tax |
| Around 150B, mostly VRAM | 85–110GB | 3090 Ti + 4 additional 3090s (120GB total VRAM), with model-specific margin check | **$8,700–11,200** new spend on the same basis |
| Around 200B, mostly VRAM | 110–145GB | Usually at least six 24GB cards (144GB aggregate), sometimes seven; check the actual quant and runtime buffers | **$10,100–13,000+** new spend for five additional 3090s; PSU and cooling costs rise materially |
| Around 300B, VRAM + RAM | 165–210GB | Begin with 72–96GB VRAM and 256GB DDR4 RAM; put the hot path on GPU, then measure CPU expert traffic | Same **$5,800–9,400** entry build; capacity is plausible, speed is unproven and model dependent |

Those totals use the ranges above and the owned 3090 Ti. They are arithmetic planning bands, not complete system quotations. More than three 3090-class cards needs deliberate spacing, risers, cooling, power cabling, and usually more than one PSU. At 250W per card, six cards alone are 1,500W before host power; that is not a one-circuit continuous-load plan. The current 1000W and 1250W PSUs can be reused only after the actual cable allocation and simultaneous-load limits are checked. Power limiting is an available experiment, not a substitute for a safe PSU and circuit budget.

### Two independent DDR4 machine designs (planning, November 2026)

These are separate machines, not an upgrade path from one to the other. Machine B uses two 1600W DC PSU outputs as the requested design constraint. AC wall draw is higher and the actual circuits must support that draw; a standard 120V/15A circuit has a lower continuous-load planning budget. All power figures below are estimates, not measured inference results.

**Machine A — incremental four-GPU host on one 1600W PSU.** EPYC 7302P, H12SSL-i, owned RTX 3090 Ti, then up to three RTX 3090 24GB cards (96GB aggregate VRAM). Begin with one compatible 64GB ECC DDR4 RDIMM and add matching 64GB modules one at a time, one per memory channel, to 8 × 64GB = 512GB. Add GPUs one at a time as well; the intermediate VRAM capacities are 24/48/72/96GB. RAM-heavy inference has less memory bandwidth before all eight channels are populated. A starting power-cap experiment is 275W for the 3090 Ti and 225W per 3090: 950W for four GPUs plus a 250–300W host allowance = about 1,200–1,250W DC, or 1,330–1,390W at the wall assuming 90% PSU efficiency. Verify PSU input rating at 120V, individual card power-limit ranges, cabling, transients, and measured wall draw. The existing 8 × 32GB price row and model-target totals are comparison estimates, not prices for this 64GB-DIMM staged plan.

**Machine B — seven or eight GPUs, 2TB DDR4, two 1600W PSUs on two dedicated circuits.** The single-socket EPYC H12SSL-i has five PCIe 4.0 x16 slots and two PCIe 4.0 x8 slots, not six x16 and one x8. Its motherboard takes a standard 24-pin ATX input plus two 8-pin CPU/EPS inputs, so the host itself does not require a proprietary PSU. Its 2TB ceiling requires 8 × 256GB supported 3DS RDIMM/LRDIMM modules; verify exact DIMMs against the board QVL. Dual-slot GPUs require suitable risers or open-frame mounting.

Seven GPUs can use all seven slots directly: five x16 links and two x8 links. With seven 300W cards and an estimated 450W host, budget 2,550W DC. A four-GPU / three-GPU-plus-host split is 1,200W / 1,350W DC, within the two 1600W PSU outputs. Seven 24GB/48GB/96GB cards provide 168GB/336GB/672GB aggregate VRAM.

**Eight GPUs can use this same board through bifurcation.** Its BIOS manual offers x8x8 mode on each x16 slot, and Supermicro confirms that all seven slots support bifurcation. A compatible x16-to-two-x8 adapter plus powered risers can turn one x16 connection into two separate x8 GPU connections. The resulting layout is **four x16 GPUs and four x8 GPUs**; the extra connection comes from dividing one x16 link. This is documented BIOS capability, not proof that a particular eight-GPU/riser combination has been validated. Check GPU enumeration, PCIe 4.0 signal integrity, BIOS address resources, cooling, and physical wiring before counting on it.

At the full rated 300W of eight A6000, PRO 5000 72GB, or PRO 6000 Max-Q 96GB cards, GPU output is 2,400W. With the estimated 450W host, the system is **2,850W DC total**. Put five complete GPU/riser assemblies on one PSU (1,500W DC) and three assemblies plus host on the other (1,350W DC). Both fit below 1600W DC, with only 100W output margin on the five-card PSU; card transients and PSU cable capacity still need checking. At 90% efficiency the AC inputs are about **1,667W and 1,500W** respectively, so circuit capacity must be confirmed separately. Eight 24GB/48GB/72GB/96GB cards provide 192GB/384GB/576GB/768GB aggregate VRAM. Eight stock 350W RTX 3090s exceed this PSU plan and would need a power cap.

A dedicated eight-GPU server such as Supermicro's H12DSG-O-CPU/AS-4124GS-TNR+ is an alternative when direct GPU connections or 32 DIMM slots are worth the proprietary system cost. It is not required merely to reach eight GPU connections on the H12SSL-i. Its included nominal 2000W modules deliver only 1000W each on 100–127V, so the server's PSU arrangement cannot be assumed equivalent to the two planned 1600W supplies.

For Machine B, power each GPU and its powered riser from the same PSU, synchronize PSU startup, and never join PSU outputs. The 2TB memory capacity can hold very large quantized checkpoints, but acceptable token speed with RAM-resident experts remains an experiment.

Sources: [H12SSL-i specifications](https://www.supermicro.com/en/products/motherboard/H12SSL-i), [H12SSL-i memory manual and bifurcation settings](https://www.supermicro.com/manuals/motherboard/EPYC7000/MNL-2314.pdf), [Supermicro bifurcation FAQ](https://www.supermicro.com/en/support/faqs/faq.php?faq=35490), [AMD EPYC 7002 memory guidance](https://www.amd.com/content/dam/amd/en/documents/epyc-technical-docs/tuning-guides/amd-epyc-7002-tg-nginx.pdf), [Supermicro eight-GPU server specifications and AC-input derating](https://www.supermicro.com/en/aplus/system/4u/4124/as-4124gs-tnr_.php), [RTX A5000 specifications](https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/nvidia-rtx-a5000-datasheet.pdf), and [RTX PRO 6000 Max-Q specifications](https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/professional-desktop-gpus/rtx-pro-6000-max-q/workstation-datasheet-blackwell-rtx-pro-6000-max-q-nvidia-3519233.pdf).

### Direction

The **DDR4 EPYC board plus owned 3090 Ti** is the lowest-cost expandable foundation for the 100–300B experiments. For Machine A, begin with one 64GB ECC RDIMM and add matched 64GB sticks and 3090s in stages; 256GB RAM is reached with four sticks, while all eight channels and 512GB require eight. It gives proper PCIe links and a growing RAM pool for 300B hybrid placement. A 200B Q4 model fully in VRAM is a later, significantly more expensive stage. The existing B250 can remain a separate small-worker box; its 4 × 8GB DDR4 UDIMMs are useful elsewhere but not in the EPYC build. Keep the older performance measurements below as test evidence, not as price forecasts.

---

### Power budget

The older two-GPU Threadripper example below assumes GPU-resident inference and light CPU work. It does not bound host power for RAM-resident experts or the 2TB designs above; use their separate host allowances and measure actual draw.

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

### Larger-model outlook

The 100–300B target is the first stage. Bigger MoE checkpoints can use RAM or NVMe placement, but the existing benchmarks show sharply variable speed and cold-read penalties. Revisit 500B–1T designs after measuring 300B hybrid placement on the new host. The November table above is the current price source; no separate speculative cost totals are retained here.

---

## Planned architecture: separate GPU systems

The B250 mining board can remain a small-worker system with the existing 1060/1070 cards. The proposed EPYC host would take the 3090 Ti and any future 3090-class or larger GPUs. The Pi remains the control plane. This is a design direction, not a fixed assignment or a purchase list.

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

The CPU swarm offers parallel script work and fault isolation. It is not the capacity path for a single 100–300B model. The earlier per-node price comparison has been removed; use the current component table above for buying decisions.

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

This section is the April 2026 snapshot used for upgrade planning. Current measurements and findings (card vs bus, splits, context capacity, PCIe) are in benchmarking `RIG_PERFORMANCE.md` and the generated `results/RUNTIME_REFERENCE.md`.

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

October 2026 additions (`llama-runtime:b11669`, `--split-mode layer`, `-fit off`, ctx 8192, 768-token
decode, single run; the 3090 was serving another model during the 1060 runs):

| Config | Model | Type | Params (active) | Gen tok/s | Notes |
|--------|-------|------|-----------------|-----------|-------|
| 3090 solo | nex-n2.5:mini | MoE | 35B (3B) | **142.2** | 20.8 GB of model buffers |
| 1060 5-way (x1×5) | nex-n2.5:mini | MoE | 35B (3B) | **21.7** | tensor split 1,1,1,1,1; 3.8-5.2 GB per card; 30-41 W avg per card |
| 1060 4-way (x1×4) | nex-n2.5:mini | MoE | 35B (3B) | **22.0** | 1,1,1,1 does not fit; 0.75,1,1,1 fits with ~750 MB free per card |
| 1060 5-way row split | nex-n2.5:mini | MoE | 35B (3B) | n/a | `--split-mode row` fails: "device CUDA0 does not support split buffers" (hybrid recurrent architecture) |

Same 6-7x pattern as April: MoE about 6.5x slower on the 1060 farm than on the 3090 (Nex 6.5x,
gemma-4:26b-a4b 6.3x), dense about 7.5x (qwen2.5-coder:14b, gemma-4:31b), and 4-way equals 5-way.
The first card always holds about 1.3 GB more than its layer share (5153 MiB on CUDA0 in both the 4-
and 5-way Nex loads), so give it a smaller tensor-split weight when VRAM is tight. Quality: litmus 3/3
on both splits; greedy outputs diverge from the 3090 (2 of 10 GSM8K responses identical) but the
4-way and 5-way splits agree with each other. Records:
`benchmarking/results/runtime_experiments/nex_n25_mini_split{5_concurrent,4}_runtime_characterization_20261017.json`.

Prior baseline (Feb 2026, Ollama, different runtime):
- 1060 single 7B: 22.5 tok/s — matches current 22.7 tok/s within measurement noise
- 1060 2-way 14B: 12.5 tok/s — matches current 11.9 tok/s

### Key findings

**The cards' own VRAM speed sets split decode, not PCIe (measured 2026-10-19).** A layer split
runs the cards one after another: each token passes through every card's layers in turn, and each
card is limited by how fast it reads its share of the weights from its own VRAM. The same 7B model
decodes at 22-24 tok/s on 1 to 5 GTX 1060s, at Gen2 and at Gen3, and a split moves only 6-58 MB/s
per x1 link (Gen2 x1 carries about 500 MB/s). That is why 4-way and 5-way MoE give the same speed
(24.2 vs 24.0 tok/s): the same bytes are read per token by cards of the same speed. Adding cards adds
VRAM (bigger context, bigger models) without losing or gaining decode speed. PCIe speed shows up only
in load time (cold 5-way load 132 s at Gen2, 90 s at Gen3) and prefill is set by micro-batch size
(`-ub 512` gave 2.4x). Details: benchmarking `BENCHMARK_LESSONS_LEARNED.md`, "PCIe generation and
split scaling (2026-10-19)".

> Earlier versions of this section said PCIe x1 was the bottleneck. That was an inference from the
> 4-way vs 5-way result; the 2026-10-19 sweep measured the links directly and ruled it out.

**MoE on 4× cheap 1060s matches single-1060 7B speed.** The 26B MoE model at
24.2 tok/s is faster than a single 1060 running 7B dense (22.7 tok/s), despite
being a brain-class model with dramatically better reasoning. This is the free
upgrade — same speed, much smarter model, using hardware already in the rig.

**Adding one 1060 to the 3090 costs 2.6-3.1x speed.** The cause is the 1060, not its x1 link:
the layers placed on the 1060 are read at the 1060's much lower VRAM bandwidth, and the 3090 waits
for them on every token. A mixed split's speed depends on how many bytes sit on the slow card. Two
cards of the same speed split with little penalty, whatever the slot (see the 1060 sweep above), so
a 2x 3090 split would lose little. Benchmarks keep the 3090 and the 1060s separate for this reason
(2026-10-20); only models too big for either side (70B and up) use an all-card split.

**MoE is consistently ~2× faster than dense at similar total params.** This
holds across every configuration — 3090 solo (152 vs 40 tok/s for 26B MoE vs
31B dense), mixed split (49 vs 15), pure 1060 4-way (24 vs 5.3).

**Asymmetric GPU splits work fine.** llama.cpp distributes layers proportionally
to available VRAM. The 3090+1060 split put 12.7GB on the 3090 and 3.3GB on the
1060 automatically. No matched pairs needed.

### Implications for upgrade planning

- A split of matched fast cards (for example 2x 3090) loses little, because both cards read at
  the same speed; faster slots mainly shorten loads
- Replacing slow cards speeds up splits directly; better PCIe or risers do not change decode
- The existing B250 worker farm can run brain-class MoE models at worker-tier
  speed using 4-5 way splits — useful when the 3090 is busy
- Future MoE models will keep getting better; the split infrastructure is ready

---

## Unified-memory alternatives

The compact-box table above has the October/November product prices. A 128GB unified-memory box may be useful as a separate experiment, but its fixed memory does not provide the 256GB RAM growth or multiple full-speed PCIe GPU slots targeted by the EPYC build. DGX Station and high-memory Mac Studio configurations require a fresh configured quote before a cost comparison.

---

## Unified-memory integration sketch

The sections below retain the original architecture brainstorm. Ownership of
model placement and hardware control has since evolved; check the current
gateway and orchestration design notes before implementing these ideas.

---

### Unified Memory Box — What Changes

#### Memory Budget Example (128GB box)

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

#### Split GPU Problem Disappears

No more PCIe pairs, no split_gpu placement, no reservation/quarantine logic for
paired GPUs. A 14B model just loads into the memory pool like any other model.
The `placement` and `split_groups` fields in `models.catalog.json` become irrelevant.

#### llama-server Advantage

The current containerized llama-server approach maps perfectly:
- Each model gets its own llama-server instance on a dedicated port
- Each is an independent process with an OpenAI-compatible endpoint
- Brain and workers target specific `host:port` endpoints
- Full control over model lifecycle — no Ollama fighting the orchestrator

Ollama would conflict here (owns its own model lifecycle, auto-unloads idle models).
Raw llama-server lets the orchestration system be the model manager.

---

### Abstraction Evolution

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

#### Brain Becomes a Resource Scheduler

Instead of coordinating hardware, the brain manages a resource pool:

- Queue has LLM tasks needing model X → is it loaded? → route to its endpoint
- Model X not loaded → do I have N GB free? → spin up a llama-server instance
- Model Y idle for 10 minutes, queue empty → unload, reclaim memory
- Queue has CPU tasks → spawn worker processes up to thread limit

The brain loop, task queue, heartbeat system, dependency graph, retry logic —
all stay the same. The vocabulary changes from "GPUs and VRAM" to "memory and threads."

---

### Multi-Rig Scaling

The file-based queue already supports distributed workers (proven by 8 Orange Pis).
Adding more GPU rigs or unified-memory boxes means each new machine:

1. Mounts the shared drive (NFS, or future NAS)
2. Runs worker processes
3. Writes heartbeats to `shared/gpus/` or `shared/cpus/`
4. Claims tasks from `shared/tasks/queue/`

#### What Already Works Across Rigs
- Task queue — file-based, location-agnostic
- Heartbeat system — each worker owns its own file
- Brain monitoring — reads heartbeats regardless of source
- Signal system — file-based, workers poll their own signal files

#### What Needs Work for Multi-Rig
- **nvidia-smi is local** — brain would need to trust heartbeats as source of truth instead of cross-checking with local nvidia-smi
- **Model placement locality** — split groups must be constrained to same physical box. Catalog needs a "rig" or "locality" concept
- **Model file access** — NFS works (slow for multi-GB loads). Local model cache per rig is faster but needs sync
- **Port addressing** — endpoints become `host:port` instead of just `port`

#### Envisioned Multi-Rig Topology

| Rig | Hardware | Role |
|-----|----------|------|
| Unified memory box | 128-748GB | Brain + big models + multi-model serving |
| Current GPU rig | 3090 Ti + 1060s | Medium models (7-14B), script tasks |
| Orange Pi fleet | CPU only | Script/cpu tasks, cheap parallelism |
| Future rig N | Whatever | More of the above |

---

### NAS Centralization

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

### CPU Workers on Unified Memory Box

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

### Brain on CPU

The brain doesn't do inference itself — it calls a model via `think()`. Options:

1. **Brain on CPU, calls inference endpoint** — brain process runs on CPU, sends
   think() calls to a loaded model on the unified-memory box via OpenAI-compatible API
2. **Brain on the unified-memory box** — everything on one machine, brain is just
   another process hitting the same vLLM/llama-server
3. **Brain stays on dedicated GPU** — current approach, wastes a 3090 Ti

Option 2 is the natural end state with a unified-memory box. The brain process is
lightweight — it's the model it calls that needs resources.

---

### Implementation Priority

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

## References and source of measured values

- [Rig performance](../../plans/shoulders/benchmarking/RIG_PERFORMANCE.md) - current measured speed and context-capacity evidence.
- [Runtime reference](../../plans/shoulders/benchmarking/results/RUNTIME_REFERENCE.md) - generated model placement and runtime summary.
- Gateway docs/DESIGN_NOTES.md, Sharing capacity - current placement design.
- [NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
- [DGX Station GB300](https://nvidianews.nvidia.com/news/nvidia-announces-dgx-spark-and-dgx-station-personal-ai-computers)
