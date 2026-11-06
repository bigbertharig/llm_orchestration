#!/usr/bin/env python3
"""Experimental Nemotron 3.5 Bolt routed-expert tensor placement."""
import argparse
import subprocess

GROUPS = [
    [1, 3, 5, 8, 10, 12, 14],
    [17, 19, 21, 23, 26],
    [28, 30, 32, 34, 37],
    [39, 41, 43, 45, 48],
    [50, 52, 54, 56, 59, 61],
]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("cards", type=int, choices=(4, 5))
    p.add_argument("port", type=int)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    groups = GROUPS[:a.cards]
    override = ",".join(
        f"blk[.]({'|'.join(map(str, layers))})[.]ffn_(up|down)_exps[.]weight=CUDA{card}"
        for card, layers in enumerate(groups, 1)
    )
    argv = [
        "/home/bryan/llm_orchestration/shared/scripts/llama_runtime/run_runtime.sh",
        "--name", f"nemotron-experts-{a.cards}-{a.port}",
        "--model", "/data/model-hotset/nemotron-3.5-bolt-74b-a9b/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-UD-Q3_K_XL.gguf",
        "--port", str(a.port),
        "--gpus", "device=" + ",".join(map(str, range(a.cards + 1))),
        "--ctx-size", "32768", "--batch-size", "64", "--parallel", "1",
        "--memory-limit", "20g",
        "--tensor-split", ",".join(["88"] + ["0"] * a.cards),
        "--extra-arg", "--reasoning", "--extra-arg", "off",
        "--extra-arg", "--override-tensor", "--extra-arg", override,
    ]
    if a.dry_run:
        print(" ".join(argv))
        return
    raise SystemExit(subprocess.run(argv, check=False).returncode)

if __name__ == "__main__":
    main()