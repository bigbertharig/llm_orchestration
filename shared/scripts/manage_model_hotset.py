#!/usr/bin/env python3
"""
Manage local Ollama model hot-set for benchmark workflows.

Typical usage on rig host:
  python3 /mnt/shared/scripts/manage_model_hotset.py report
  python3 /mnt/shared/scripts/manage_model_hotset.py dedupe --prune-imports
  python3 /mnt/shared/scripts/manage_model_hotset.py promote-gguf \
      --gguf /mnt/shared/models/qwen2.5-coder-14b/Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf \
      --name qwen2.5-coder:14b --host 127.0.0.1:11436
  python3 /mnt/shared/scripts/manage_model_hotset.py evict \
      --host 127.0.0.1:11436 --keep qwen2.5:7b,qwen2.5-coder:14b
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable


ROOT_BLOBS = Path("/usr/share/ollama/.ollama/models/blobs")
USER_BLOBS = Path("/home/bryan/.ollama/models/blobs")
IMPORT_DIR = Path("/var/lib/ollama-import")
ARCHIVE_MODELS = Path("/mnt/shared/models")


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, text=True, capture_output=True)


def run_shell(command: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, shell=True, check=check, text=True, capture_output=True)


def require_cmd(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Missing required command: {name}")


def human_bytes(num: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    val = float(num)
    for unit in units:
        if val < 1024 or unit == units[-1]:
            return f"{val:.1f}{unit}"
        val /= 1024
    return f"{num}B"


def parse_ollama_table_names(output: str) -> list[str]:
    names: list[str] = []
    lines = [ln.rstrip() for ln in output.splitlines() if ln.strip()]
    for line in lines:
        if line.startswith("NAME "):
            continue
        # First table column is NAME. Split on 2+ spaces.
        parts = re.split(r"\s{2,}", line)
        if not parts:
            continue
        names.append(parts[0].strip())
    return names


def ollama_list(host: str) -> list[str]:
    env_prefix = f"OLLAMA_HOST={host}"
    proc = run_shell(f"{env_prefix} ollama list", check=False)
    if proc.returncode != 0:
        raise SystemExit(f"ollama list failed on {host}: {proc.stderr.strip()}")
    return parse_ollama_table_names(proc.stdout)


def ollama_ps(host: str) -> list[str]:
    env_prefix = f"OLLAMA_HOST={host}"
    proc = run_shell(f"{env_prefix} ollama ps", check=False)
    if proc.returncode != 0:
        return []
    return parse_ollama_table_names(proc.stdout)


def blob_sizes(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    if not path.exists():
        return out
    for item in path.iterdir():
        if item.is_file() and item.name.startswith("sha256-"):
            out[item.name] = item.stat().st_size
    return out


def active_blob_paths() -> set[Path]:
    proc = run_cmd(["ps", "-eo", "pid,args"], check=False)
    active: set[Path] = set()
    for line in proc.stdout.splitlines():
        if "ollama runner" not in line:
            continue
        m = re.search(r"--model\s+(\S+)", line)
        if not m:
            continue
        active.add(Path(m.group(1)))
    return active


def rm_path(path: Path, sudo: bool) -> None:
    cmd = ["rm", "-f", str(path)]
    if sudo:
        cmd.insert(0, "sudo")
    run_cmd(cmd)


def rmtree(path: Path, sudo: bool) -> None:
    cmd = ["rm", "-rf", str(path)]
    if sudo:
        cmd.insert(0, "sudo")
    run_cmd(cmd)


def cmd_report(_: argparse.Namespace) -> int:
    print("Disk:")
    print(run_cmd(["df", "-h", "/"]).stdout.strip())
    print()

    for label, p in [("root_store", ROOT_BLOBS), ("user_store", USER_BLOBS), ("import_dir", IMPORT_DIR)]:
        if p.exists():
            out = run_cmd(["du", "-sh", str(p)]).stdout.strip()
            print(f"{label}: {out}")
        else:
            print(f"{label}: missing ({p})")
    print()

    root_files = blob_sizes(ROOT_BLOBS)
    user_files = blob_sizes(USER_BLOBS)
    overlap = sorted(set(root_files).intersection(user_files))
    overlap_size = sum(root_files[k] for k in overlap)
    print(f"duplicate_blob_count: {len(overlap)}")
    print(f"duplicate_blob_size:  {human_bytes(overlap_size)}")
    if overlap:
        for name in overlap:
            print(f"  {name} ({human_bytes(root_files[name])})")
    print()

    active = sorted(active_blob_paths())
    print(f"active_runner_blob_paths: {len(active)}")
    for p in active:
        print(f"  {p}")
    return 0


def cmd_dedupe(args: argparse.Namespace) -> int:
    root_files = blob_sizes(ROOT_BLOBS)
    user_files = blob_sizes(USER_BLOBS)
    overlap = sorted(set(root_files).intersection(user_files))
    active = active_blob_paths()

    delete_from = args.delete_from
    removed_count = 0
    removed_bytes = 0

    for blob in overlap:
        root_path = ROOT_BLOBS / blob
        user_path = USER_BLOBS / blob
        if root_files[blob] != user_files[blob]:
            print(f"skip_size_mismatch {blob} root={root_files[blob]} user={user_files[blob]}")
            continue
        candidate = root_path if delete_from == "root" else user_path
        if candidate in active:
            print(f"skip_in_use {candidate}")
            continue
        sudo = str(candidate).startswith("/usr/")
        rm_path(candidate, sudo=sudo)
        removed_count += 1
        removed_bytes += root_files[blob]
        print(f"removed {candidate} ({human_bytes(root_files[blob])})")

    if args.prune_imports and IMPORT_DIR.exists():
        for child in sorted(IMPORT_DIR.iterdir()):
            if not child.is_dir():
                continue
            rmtree(child, sudo=True)
            print(f"pruned_import_dir {child}")

    print(f"dedupe_removed_count: {removed_count}")
    print(f"dedupe_removed_size:  {human_bytes(removed_bytes)}")
    print(run_cmd(["df", "-h", "/"]).stdout.strip())
    return 0


def cmd_promote_gguf(args: argparse.Namespace) -> int:
    gguf = Path(args.gguf).resolve()
    if not gguf.exists():
        raise SystemExit(f"GGUF not found: {gguf}")
    if not gguf.is_file():
        raise SystemExit(f"Not a file: {gguf}")
    if not str(gguf).startswith(str(ARCHIVE_MODELS)):
        print(f"warning: gguf path is not under {ARCHIVE_MODELS}: {gguf}", file=sys.stderr)

    require_cmd("ollama")
    with tempfile.NamedTemporaryFile("w", delete=False, prefix="modelfile_", suffix=".tmp") as tf:
        tf.write(f"FROM {gguf}\n")
        tmp_modelfile = tf.name

    try:
        command = f"OLLAMA_HOST={args.host} ollama create {args.name} -f {tmp_modelfile}"
        proc = run_shell(command, check=False)
        if proc.returncode != 0:
            raise SystemExit(f"promote failed:\n{proc.stderr.strip()}")
        print(proc.stdout.strip())
        print(f"promoted model={args.name} host={args.host} source={gguf}")
    finally:
        Path(tmp_modelfile).unlink(missing_ok=True)
    return 0


def comma_list(values: str) -> list[str]:
    return [v.strip() for v in values.split(",") if v.strip()]


def cmd_evict(args: argparse.Namespace) -> int:
    keep_set = set(comma_list(args.keep))
    listed = ollama_list(args.host)
    active = set(ollama_ps(args.host))
    to_remove = [m for m in listed if m not in keep_set]

    removed = 0
    skipped_active = 0
    for model in to_remove:
        if (model in active) and not args.force_active:
            print(f"skip_active {model}")
            skipped_active += 1
            continue
        proc = run_shell(f"OLLAMA_HOST={args.host} ollama rm {model}", check=False)
        if proc.returncode == 0:
            print(f"removed {model}")
            removed += 1
        else:
            print(f"remove_failed {model}: {proc.stderr.strip()}")

    print(f"evict_removed: {removed}")
    print(f"evict_skipped_active: {skipped_active}")
    remaining = ollama_list(args.host)
    print("remaining_models:")
    for model in remaining:
        print(f"  {model}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage Ollama model hot-set on rig")
    sub = parser.add_subparsers(dest="command", required=True)

    p_report = sub.add_parser("report", help="Show disk/model inventory and duplicate blob estimate")
    p_report.set_defaults(func=cmd_report)

    p_dedupe = sub.add_parser("dedupe", help="Remove duplicate blob files between root/user stores")
    p_dedupe.add_argument("--delete-from", choices=["root", "user"], default="root")
    p_dedupe.add_argument("--prune-imports", action="store_true", help="Also remove subdirs in /var/lib/ollama-import")
    p_dedupe.set_defaults(func=cmd_dedupe)

    p_promote = sub.add_parser("promote-gguf", help="Create a model on target Ollama host from archive GGUF")
    p_promote.add_argument("--gguf", required=True, help="Path to GGUF, typically under /mnt/shared/models")
    p_promote.add_argument("--name", required=True, help="Destination model name (example: qwen2.5-coder:14b)")
    p_promote.add_argument("--host", default="127.0.0.1:11436", help="Target Ollama host:port")
    p_promote.set_defaults(func=cmd_promote_gguf)

    p_evict = sub.add_parser("evict", help="Remove models from host except keep-list")
    p_evict.add_argument("--host", required=True, help="Target Ollama host:port")
    p_evict.add_argument("--keep", required=True, help="Comma list of model names to keep")
    p_evict.add_argument("--force-active", action="store_true", help="Allow removing active models in ollama ps")
    p_evict.set_defaults(func=cmd_evict)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
