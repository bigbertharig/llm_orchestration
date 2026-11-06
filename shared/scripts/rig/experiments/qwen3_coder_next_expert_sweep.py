#!/usr/bin/env python3
"""Experimental Qwen3-Coder-Next expert placement sweep launcher."""
import argparse
import subprocess

LAYOUTS = {
    "24x5": [list(range(0, 5)), list(range(5, 10)), list(range(10, 15)), list(range(15, 20)), list(range(20, 24))],
    "28x5": [list(range(0, 5)), list(range(5, 11)), list(range(11, 17)), list(range(17, 23)), list(range(23, 28))],
    "24x4": [list(range(0, 6)), list(range(6, 12)), list(range(12, 18)), list(range(18, 24))],
    "28round": [list(range(card, 28, 5)) for card in range(5)],
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("layout", choices=LAYOUTS)
    p.add_argument("port", type=int)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    cards = LAYOUTS[a.layout]
    mapping = []
    for card, layers in enumerate(cards, 1):
        numbers = "|".join(map(str, layers))
        mapping.append(f"blk[.]({numbers})[.]ffn_(up|down|gate)_exps[.]weight=CUDA{card}")
    argv = [
        "/home/bryan/llm_orchestration/shared/scripts/llama_runtime/run_runtime.sh",
        "--name", f"coder-sweep-{a.layout}-{a.port}",
        "--model", "/data/model-hotset/qwen3-coder-next/Qwen3-Coder-Next-Q4_K_S.gguf",
        "--port", str(a.port),
        "--gpus", "device=" + ",".join(map(str, range(len(cards) + 1))),
        "--ctx-size", "32768", "--batch-size", "64", "--parallel", "1",
        "--memory-limit", "20g",
        "--tensor-split", ",".join(["48"] + ["0"] * len(cards)),
        "--extra-arg", "--override-tensor", "--extra-arg", ",".join(mapping),
    ]
    if a.dry_run:
        print(" ".join(argv))
        return
    raise SystemExit(subprocess.run(argv, check=False).returncode)

if __name__ == "__main__":
    main()