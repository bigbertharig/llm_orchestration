#!/usr/bin/env python3
"""
Download and cache benchmark datasets to shared storage.

Run this on the GPU rig (10.0.0.3) to pre-fetch datasets before benchmark runs.
Datasets are cached to /mnt/shared/benchmarks/datasets/ for offline use.

Usage:
    python3 /mnt/shared/scripts/benchmarks/setup_benchmark_datasets.py --all
    python3 /mnt/shared/scripts/benchmarks/setup_benchmark_datasets.py --dataset gsm8k gpqa ifeval
    python3 /mnt/shared/scripts/benchmarks/setup_benchmark_datasets.py --list
    python3 /mnt/shared/scripts/benchmarks/setup_benchmark_datasets.py --check
"""

import argparse
import os
import sys
import subprocess
import shutil
from pathlib import Path

# Shared storage paths
SHARED_CACHE = Path("/mnt/shared/benchmarks/datasets")
BFCL_DIR = Path("/mnt/shared/benchmarks/bfcl")

# Dataset definitions: name -> (hf_dataset_id, config/subset)
DATASETS = {
    # Core reasoning
    "gsm8k": ("gsm8k", "main"),
    "gpqa": ("Idavidrein/gpqa", "gpqa_diamond"),
    "mmlu": ("cais/mmlu", "all"),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge"),
    "hellaswag": ("Rowan/hellaswag", None),
    "winogrande": ("allenai/winogrande", "winogrande_xl"),
    "piqa": ("ybisk/piqa", None),
    "boolq": ("google/boolq", None),
    "truthfulqa": ("truthfulqa/truthful_qa", "multiple_choice"),

    # Instruction following
    "ifeval": ("google/IFEval", None),

    # Math
    "math": ("lighteval/MATH", "all"),

    # Coding
    "humaneval": ("openai/openai_humaneval", None),
    "mbpp": ("google-research-datasets/mbpp", "sanitized"),

    # Long context
    "longbench": ("THUDM/LongBench", None),

    # Multilingual
    "mmmlu": ("openai/mmmlu", "default"),

    # BBH
    "bbh": ("lukaemon/bbh", None),

    # DROP
    "drop": ("ucinlp/drop", None),
}

# External repos that need cloning
EXTERNAL_REPOS = {
    "bfcl": {
        "url": "https://github.com/ShishirPatil/gorilla.git",
        "subdir": "berkeley-function-call-leaderboard",
        "description": "Berkeley Function Calling Leaderboard V3"
    },
    "swebench": {
        "url": "https://github.com/princeton-nlp/SWE-bench.git",
        "subdir": None,
        "description": "SWE-bench repository-level coding benchmark"
    },
    "terminal_bench_2": {
        "url": "https://github.com/laude-institute/terminal-bench-2.git",
        "subdir": None,
        "description": "Terminal-Bench 2.0 - CLI task benchmark with 89 curated tasks"
    }
}


def check_dependencies():
    """Check if required packages are installed."""
    missing = []
    try:
        import datasets
    except ImportError:
        missing.append("datasets")
    try:
        import huggingface_hub
    except ImportError:
        missing.append("huggingface_hub")

    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Install with: pip install datasets huggingface_hub")
        return False
    return True


def free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def check_disk_preflight(path: Path, min_free_gb: float) -> bool:
    available = free_gb(path)
    if available < min_free_gb:
        print(
            f"Disk preflight failed: {available:.1f}GB free on {path}, "
            f"requires at least {min_free_gb:.1f}GB."
        )
        return False
    print(f"Disk preflight OK: {available:.1f}GB free on {path}")
    return True


def download_hf_dataset(name: str, dataset_id: str, config: str = None):
    """Download a HuggingFace dataset to shared cache."""
    from datasets import load_dataset

    cache_dir = SHARED_CACHE / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {name} ({dataset_id})...")
    try:
        if config:
            ds = load_dataset(dataset_id, config, cache_dir=str(cache_dir))
        else:
            ds = load_dataset(dataset_id, cache_dir=str(cache_dir))
        print(f"  OK: {name} - {ds}")
        return True
    except Exception as e:
        print(f"  FAILED: {name} - {e}")
        return False


def clone_external_repo(name: str, info: dict):
    """Clone an external benchmark repository."""
    repos_dir = SHARED_CACHE / "repos"
    repos_dir.mkdir(parents=True, exist_ok=True)

    target = repos_dir / name
    if target.exists():
        print(f"  {name}: already cloned at {target}")
        return True

    print(f"Cloning {name} ({info['description']})...")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", info["url"], str(target)],
            check=True,
            capture_output=True
        )
        print(f"  OK: {name} cloned to {target}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  FAILED: {name} - {e.stderr.decode()}")
        return False


def list_datasets():
    """List all available datasets."""
    print("\nHuggingFace Datasets:")
    print("-" * 60)
    for name, (dataset_id, config) in sorted(DATASETS.items()):
        cfg = f" ({config})" if config else ""
        print(f"  {name:20} -> {dataset_id}{cfg}")

    print("\nExternal Repositories:")
    print("-" * 60)
    for name, info in sorted(EXTERNAL_REPOS.items()):
        print(f"  {name:20} -> {info['description']}")


def check_status():
    """Check which datasets are already downloaded."""
    print("\nDataset Status:")
    print("-" * 60)

    hf_cache = SHARED_CACHE / "huggingface"
    repos_dir = SHARED_CACHE / "repos"

    print(f"\nCache directory: {SHARED_CACHE}")
    print(f"  HF cache exists: {hf_cache.exists()}")
    print(f"  Repos dir exists: {repos_dir.exists()}")

    if hf_cache.exists():
        # List cached datasets (simplified check)
        cache_size = sum(f.stat().st_size for f in hf_cache.rglob("*") if f.is_file())
        print(f"  HF cache size: {cache_size / 1e9:.2f} GB")

    print("\nExternal repos:")
    for name in EXTERNAL_REPOS:
        status = "downloaded" if (repos_dir / name).exists() else "not downloaded"
        print(f"  {name:20} -> {status}")


def main():
    parser = argparse.ArgumentParser(description="Download benchmark datasets")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--dataset", nargs="+", help="Download specific datasets")
    parser.add_argument("--repos", action="store_true", help="Clone external repos only")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--check", action="store_true", help="Check download status")
    parser.add_argument("--priority", action="store_true",
                        help="Download high-priority datasets for agent testing")
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=20.0,
        help="Fail if less than this many GB are free on shared dataset mount (default: 20)",
    )
    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    if args.check:
        check_status()
        return

    if not check_dependencies():
        sys.exit(1)

    if not check_disk_preflight(SHARED_CACHE.parent, args.min_free_gb):
        sys.exit(1)

    SHARED_CACHE.mkdir(parents=True, exist_ok=True)

    # Determine what to download
    to_download = []
    clone_repos = []

    if args.all:
        to_download = list(DATASETS.keys())
        clone_repos = list(EXTERNAL_REPOS.keys())
    elif args.priority:
        # High priority for agent rig
        to_download = ["gsm8k", "gpqa", "ifeval", "humaneval", "mmlu", "arc_challenge", "bbh"]
        clone_repos = ["bfcl", "terminal_bench_2"]
    elif args.repos:
        clone_repos = list(EXTERNAL_REPOS.keys())
    elif args.dataset:
        to_download = args.dataset
    else:
        parser.print_help()
        return

    # Download HF datasets
    success = 0
    failed = 0
    for name in to_download:
        if name in DATASETS:
            dataset_id, config = DATASETS[name]
            if download_hf_dataset(name, dataset_id, config):
                success += 1
            else:
                failed += 1
        else:
            print(f"Unknown dataset: {name}")
            failed += 1

    # Clone repos
    for name in clone_repos:
        if name in EXTERNAL_REPOS:
            clone_external_repo(name, EXTERNAL_REPOS[name])

    print(f"\nDone: {success} succeeded, {failed} failed")


if __name__ == "__main__":
    main()
