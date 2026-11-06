# Qwen3-Coder-Next expert placement experiment

Run this wrapper on the rig after the gateway has unloaded other models and the six GPUs have been reserved for the experiment. It uses the existing run_runtime.sh lifecycle and the installed Q4_K_S GGUF from the rig hotset.

    bash shared/scripts/rig/experiments/qwen3_coder_next_expert_split.sh 32k 11514
    bash shared/scripts/rig/experiments/qwen3_coder_next_expert_split.sh 262k 11516

Use --dry-run as a third argument to inspect the command. Stop the resulting container with shared/scripts/llama_runtime/stop_runtime.sh and the name coder-next-experts-32k-11514 (adjust the context and port).

The wrapper keeps non-routed-expert tensors, including attention, shared experts, and KV cache, on the RTX 3090 Ti. --override-tensor places selected whole routed-expert tensors by layer on the five GTX 1060s. It does not shard an individual expert across GPUs like m10dis. The 262K recipe moves more expert layers off the 3090 to make room for its full-context KV cache.

This is a research load path. It does not publish a gateway route or change the manager's placement rules. Benchmark receipts and the comparison table live in the benchmarking repository's results/runtime_experiments/coder_next_expert_placement_20261105/ directory. The benchmark experiment has validated a 235K-token recall request on the full-context recipe and one matched coding task at 32K. Run repeated quality and workload checks before creating a manager recipe. The GTX 1060s currently use PCIe Gen2 x1 links and peer access to the 3090 is unavailable, so future per-expert sharding needs a separate transfer design.
## Placement sweeps, 2026-11-06

The additional launchers are research variants of the same run_runtime.sh lifecycle:

    python3 shared/scripts/rig/experiments/qwen3_coder_next_expert_sweep.py 24x5 11520
    python3 shared/scripts/rig/experiments/qwen3_coder_next_expert_sweep.py 24x4 11522
    python3 shared/scripts/rig/experiments/nemotron_bolt_expert_sweep.py 4 11530

Each accepts --dry-run. Stop by its printed container name with stop_runtime.sh before another load. The Coder Next sweep also has 28x5 and 28round research layouts. The Nemotron launcher uses expert-layer groups sized from the installed GGUF header, with four- and five-1060 options. These are tested placements for the current model files and six-card rig only; they do not adjust the gateway registry or reserve manager leases. Benchmarking scorecards and raw receipts are in results/runtime_experiments/coder_next_expert_sweep_20261106/ and nemotron_expert_sweep_20261106/.

The 24x5 Coder Next layout offers the best measured balance of speed and headroom in the 32K sweep. The 24x4 layout leaves only about 482 MiB free on one 1060. The four-1060 Nemotron layout is best in its sweep and repeated successfully, with over 1.8 GiB free on each used 1060. Keep manager promotion separate until quality and concurrency checks are complete.
