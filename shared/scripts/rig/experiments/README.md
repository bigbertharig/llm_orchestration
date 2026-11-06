# Qwen3-Coder-Next expert placement experiment

Run this wrapper on the rig after the gateway has unloaded other models and the six GPUs have been reserved for the experiment. It uses the existing run_runtime.sh lifecycle and the installed Q4_K_S GGUF from the rig hotset.

    bash shared/scripts/rig/experiments/qwen3_coder_next_expert_split.sh 32k 11514
    bash shared/scripts/rig/experiments/qwen3_coder_next_expert_split.sh 262k 11516

Use --dry-run as a third argument to inspect the command. Stop the resulting container with shared/scripts/llama_runtime/stop_runtime.sh and the name coder-next-experts-32k-11514 (adjust the context and port).

The wrapper keeps non-routed-expert tensors, including attention, shared experts, and KV cache, on the RTX 3090 Ti. --override-tensor places selected whole routed-expert tensors by layer on the five GTX 1060s. It does not shard an individual expert across GPUs like m10dis. The 262K recipe moves more expert layers off the 3090 to make room for its full-context KV cache.

This is a research load path. It does not publish a gateway route or change the manager's placement rules. Benchmark receipts and the comparison table live in the benchmarking repository's results/runtime_experiments/coder_next_expert_placement_20261105/ directory. The benchmark experiment has validated a 235K-token recall request on the full-context recipe and one matched coding task at 32K. Run repeated quality and workload checks before creating a manager recipe. The GTX 1060s currently use PCIe Gen2 x1 links and peer access to the 3090 is unavailable, so future per-expert sharding needs a separate transfer design.