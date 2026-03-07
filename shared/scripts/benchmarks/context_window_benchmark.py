#!/usr/bin/env python3
"""Measure practical context capacity per worker GPU using Ollama endpoints.

Run on the GPU rig:
  python3 /mnt/shared/scripts/benchmarks/context_window_benchmark.py

By default this reads /mnt/shared/agents/config.json, tests all configured
worker GPUs, and writes CSV/JSON reports under /mnt/shared/logs/.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_cmd(cmd: list[str], timeout: int = 10) -> tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    return p.returncode, p.stdout, p.stderr


def nvidia_row(gpu_id: int) -> dict[str, Any] | None:
    rc, out, _ = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,temperature.gpu,power.draw,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout=5,
    )
    if rc != 0:
        return None
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        if idx != gpu_id:
            continue
        try:
            return {
                "gpu_id": idx,
                "mem_used_mb": int(float(parts[1])),
                "mem_total_mb": int(float(parts[2])),
                "temp_c": int(float(parts[3])),
                "power_w": float(parts[4]),
                "util_pct": int(float(parts[5])),
            }
        except ValueError:
            return None
    return None


def build_prompt(word_count: int) -> str:
    # Keep the pattern deterministic so tokenization variance is low.
    chunk = "alpha beta gamma delta epsilon zeta eta theta iota kappa "
    reps = max(1, int(word_count / 10))
    body = chunk * reps
    return (
        "Context stress test. Summarize in one short line.\n\n"
        "BEGIN CONTEXT:\n"
        f"{body}\n"
        "END CONTEXT.\n"
    )


def ollama_generate(
    host: str,
    port: int,
    model: str,
    prompt: str,
    timeout_s: int,
    keep_alive: str,
) -> tuple[bool, dict[str, Any], str]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {
            "temperature": 0,
            "num_predict": 1,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    url = f"http://{host}:{port}/api/generate"
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
        parsed = json.loads(body)
        return True, parsed if isinstance(parsed, dict) else {}, ""
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="ignore")
        return False, {}, f"HTTP {e.code}: {err[:300]}"
    except Exception as e:
        return False, {}, str(e)


def ollama_ps(host: str, port: int, timeout_s: int) -> tuple[bool, list[str], str]:
    url = f"http://{host}:{port}/api/ps"
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
        parsed = json.loads(body)
        models: list[str] = []
        for m in parsed.get("models", []) if isinstance(parsed, dict) else []:
            if isinstance(m, dict):
                name = m.get("name")
                if isinstance(name, str) and name.strip():
                    models.append(name.strip())
        return True, models, ""
    except Exception as e:
        return False, [], str(e)


def model_loaded(loaded_models: list[str], target_model: str) -> bool:
    target = target_model.strip()
    base = target.split(":")[0]
    for name in loaded_models:
        n = name.strip()
        if n == target:
            return True
        if n.split(":")[0] == base:
            return True
    return False


def preload_workers(
    workers: list[dict[str, Any]],
    host: str,
    keep_alive: str,
    timeout_s: int,
) -> None:
    print("\nPreloading models on target workers...")
    for w in workers:
        gpu_id = int(w["id"])
        name = str(w.get("name", f"gpu-{gpu_id}"))
        model = str(w.get("model", "qwen2.5:7b"))
        port = int(w.get("port", 11434 + gpu_id))

        ok, models, err = ollama_ps(host, port, timeout_s=min(20, timeout_s))
        if ok and model_loaded(models, model):
            print(f"  {name}: already loaded ({model})")
            continue

        if not ok:
            print(f"  {name}: /api/ps unavailable ({err[:120]}) -> trying load")
        else:
            print(f"  {name}: model not loaded ({model}), loaded={models} -> loading")

        load_ok, _, load_err = ollama_generate(
            host=host,
            port=port,
            model=model,
            prompt="warmup",
            timeout_s=timeout_s,
            keep_alive=keep_alive,
        )
        if not load_ok:
            raise RuntimeError(f"Failed to preload {name} ({model}): {load_err}")
        print(f"  {name}: loaded ({model})")


@dataclass
class PeakTracker:
    gpu_id: int
    stop: threading.Event
    peak_mem_mb: int = 0
    peak_util_pct: int = 0
    peak_temp_c: int = 0
    last_power_w: float = 0.0

    def loop(self) -> None:
        while not self.stop.is_set():
            row = nvidia_row(self.gpu_id)
            if row:
                self.peak_mem_mb = max(self.peak_mem_mb, int(row["mem_used_mb"]))
                self.peak_util_pct = max(self.peak_util_pct, int(row["util_pct"]))
                self.peak_temp_c = max(self.peak_temp_c, int(row["temp_c"]))
                self.last_power_w = float(row["power_w"])
            time.sleep(0.2)


def select_workers(config: dict[str, Any], names_csv: str | None) -> list[dict[str, Any]]:
    workers = [w for w in config.get("gpus", []) if isinstance(w, dict)]
    if not names_csv:
        return workers
    names = {x.strip() for x in names_csv.split(",") if x.strip()}
    return [w for w in workers if str(w.get("name")) in names]


def test_worker(
    worker: dict[str, Any],
    host: str,
    keep_alive: str,
    start_words: int,
    step_words: int,
    max_words: int,
    timeout_s: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gpu_id = int(worker["id"])
    name = str(worker.get("name", f"gpu-{gpu_id}"))
    model = str(worker.get("model", "qwen2.5:7b"))
    port = int(worker.get("port", 11434 + gpu_id))

    rows: list[dict[str, Any]] = []
    last_success_words = 0
    last_success_prompt_tokens = 0
    first_failure: str | None = None

    # Warm endpoint once for consistent measurements.
    ollama_generate(host, port, model, "warmup", timeout_s, keep_alive)

    for words in range(start_words, max_words + 1, step_words):
        before = nvidia_row(gpu_id) or {}
        stop_evt = threading.Event()
        tracker = PeakTracker(gpu_id=gpu_id, stop=stop_evt)
        t = threading.Thread(target=tracker.loop, daemon=True)
        t.start()
        t0 = time.time()
        ok, out, err = ollama_generate(host, port, model, build_prompt(words), timeout_s, keep_alive)
        elapsed = time.time() - t0
        stop_evt.set()
        t.join(timeout=1.0)
        after = nvidia_row(gpu_id) or {}

        prompt_tokens = int(out.get("prompt_eval_count", 0) or 0)
        prompt_eval_s = float(out.get("prompt_eval_duration", 0) or 0) / 1e9
        total_eval_s = float(out.get("eval_duration", 0) or 0) / 1e9

        row = {
            "worker": name,
            "gpu_id": gpu_id,
            "port": port,
            "model": model,
            "input_words": words,
            "prompt_eval_count": prompt_tokens,
            "request_ok": ok,
            "error": err[:260] if err else "",
            "elapsed_s": round(elapsed, 3),
            "prompt_eval_s": round(prompt_eval_s, 3),
            "eval_s": round(total_eval_s, 3),
            "mem_before_mb": before.get("mem_used_mb"),
            "mem_after_mb": after.get("mem_used_mb"),
            "mem_peak_mb": tracker.peak_mem_mb or after.get("mem_used_mb"),
            "mem_total_mb": after.get("mem_total_mb") or before.get("mem_total_mb"),
            "gpu_util_peak_pct": tracker.peak_util_pct,
            "gpu_temp_peak_c": tracker.peak_temp_c or after.get("temp_c"),
            "power_last_w": round(tracker.last_power_w, 2),
        }
        rows.append(row)

        if ok:
            last_success_words = words
            if prompt_tokens > 0:
                last_success_prompt_tokens = prompt_tokens
        elif first_failure is None:
            first_failure = f"words={words}: {err[:180]}"
            break

        mem_total = row.get("mem_total_mb") or 0
        mem_peak = row.get("mem_peak_mb") or 0
        if mem_total and mem_peak and (mem_peak / mem_total) >= 0.985:
            first_failure = f"memory ceiling reached at words={words}"
            break

    summary = {
        "worker": name,
        "gpu_id": gpu_id,
        "port": port,
        "model": model,
        "last_success_words": last_success_words,
        "last_success_prompt_eval_count": last_success_prompt_tokens,
        "first_failure": first_failure or "",
        "attempts": len(rows),
    }
    return rows, summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark context growth and VRAM behavior per GPU worker")
    ap.add_argument("--config", default="/mnt/shared/agents/config.json")
    ap.add_argument("--host", default="localhost", help="Ollama host on rig")
    ap.add_argument("--workers", default="", help="Comma-separated worker names (default: all)")
    ap.add_argument("--start-words", type=int, default=800)
    ap.add_argument("--step-words", type=int, default=800)
    ap.add_argument("--max-words", type=int, default=20000)
    ap.add_argument("--timeout-seconds", type=int, default=300)
    ap.add_argument("--keep-alive", default="45m")
    ap.add_argument("--log-dir", default="/mnt/shared/logs/benchmarks")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    workers = select_workers(cfg, args.workers or None)
    if not workers:
        raise SystemExit("No matching workers found in config.")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = now_stamp()
    csv_path = log_dir / f"context_window_benchmark_{stamp}.csv"
    json_path = log_dir / f"context_window_benchmark_{stamp}.json"

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    print(f"Context benchmark on {len(workers)} worker(s)")
    print(f"Config: {cfg_path}")
    print(f"Output: {csv_path}")

    preload_workers(
        workers=workers,
        host=args.host,
        keep_alive=args.keep_alive,
        timeout_s=args.timeout_seconds,
    )

    for w in workers:
        name = w.get("name")
        gpu_id = w.get("id")
        port = w.get("port", 11434 + int(gpu_id))
        model = w.get("model")
        print(f"\n--- {name} (gpu={gpu_id}, port={port}, model={model}) ---")
        rows, summary = test_worker(
            worker=w,
            host=args.host,
            keep_alive=args.keep_alive,
            start_words=args.start_words,
            step_words=args.step_words,
            max_words=args.max_words,
            timeout_s=args.timeout_seconds,
        )
        all_rows.extend(rows)
        summaries.append(summary)
        print(
            f"last_success_words={summary['last_success_words']} "
            f"prompt_eval={summary['last_success_prompt_eval_count']} "
            f"failure='{summary['first_failure']}'"
        )

    if all_rows:
        fields = list(all_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in all_rows:
                writer.writerow(r)

    report = {
        "generated_at": datetime.now().isoformat(),
        "config": str(cfg_path),
        "host": args.host,
        "workers_tested": [s["worker"] for s in summaries],
        "params": {
            "start_words": args.start_words,
            "step_words": args.step_words,
            "max_words": args.max_words,
            "timeout_seconds": args.timeout_seconds,
            "keep_alive": args.keep_alive,
        },
        "summary": summaries,
        "rows": all_rows,
        "csv_path": str(csv_path),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nSummary:")
    for s in summaries:
        print(
            f"{s['worker']}: last_success_words={s['last_success_words']}, "
            f"prompt_eval={s['last_success_prompt_eval_count']}, failure={s['first_failure'] or '-'}"
        )
    print(f"\nWrote CSV:  {csv_path}")
    print(f"Wrote JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
