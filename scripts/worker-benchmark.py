#!/usr/bin/env python3
"""
Worker GPU Benchmark - Tests 7B model on individual GPUs for worker role.
"""

import subprocess
import json
import time
import os
import requests

MODEL = "qwen2.5:7b"
PROMPT = "Parse this data and return JSON with fields: name, value, status.\n\nInput: Server alpha is running at 85% capacity. Server beta is offline. Server gamma is running at 42% capacity."
NUM_RUNS = 3
OLLAMA_API = "http://localhost:11434/api/generate"

# Test each worker GPU individually
WORKER_GPUS = [1, 2, 4]

def stop_ollama():
    subprocess.run(["pkill", "-f", "ollama serve"], capture_output=True)
    time.sleep(2)

def get_gpu_temps():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,temperature.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    temps = {}
    for line in result.stdout.strip().split('\n'):
        if line:
            parts = line.split(',')
            if len(parts) == 2:
                temps[int(parts[0].strip())] = int(parts[1].strip())
    return temps

def run_benchmark(gpu_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"\n{'='*60}")
    print(f"Testing GPU {gpu_id} with {MODEL}")
    print('='*60)

    stop_ollama()

    # Start ollama with single GPU
    ollama_proc = subprocess.Popen(
        ["ollama", "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(5)

    results = []
    temps_before = get_gpu_temps()

    for run in range(NUM_RUNS):
        print(f"\n  Run {run + 1}/{NUM_RUNS}...")

        start_time = time.time()
        first_token_time = None
        token_count = 0

        try:
            response = requests.post(
                OLLAMA_API,
                json={"model": MODEL, "prompt": PROMPT, "stream": True},
                stream=True,
                timeout=120
            )

            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data and data["response"]:
                        if first_token_time is None:
                            first_token_time = time.time() - start_time
                        token_count += 1

                    if data.get("done", False):
                        total_time = time.time() - start_time
                        eval_count = data.get("eval_count", token_count)
                        eval_duration = data.get("eval_duration", 0) / 1e9

                        if eval_duration > 0:
                            tokens_per_sec = eval_count / eval_duration
                        else:
                            tokens_per_sec = eval_count / (total_time - (first_token_time or 0))

                        results.append({
                            "run": run + 1,
                            "tokens": eval_count,
                            "tokens_per_sec": tokens_per_sec,
                            "time_to_first_token": first_token_time,
                            "total_time": total_time,
                        })

                        print(f"    Tokens: {eval_count}, Speed: {tokens_per_sec:.2f} tok/s, TTFT: {first_token_time:.2f}s")
                        break

        except Exception as e:
            print(f"    Error: {e}")
            results.append({"run": run + 1, "error": str(e)})

        time.sleep(1)

    temps_after = get_gpu_temps()
    ollama_proc.terminate()
    time.sleep(2)

    return {
        "gpu": gpu_id,
        "temp_before": temps_before.get(gpu_id),
        "temp_after": temps_after.get(gpu_id),
        "runs": results
    }

def main():
    print("Worker GPU Benchmark")
    print(f"Model: {MODEL}")
    print(f"Runs per GPU: {NUM_RUNS}")

    print("\nInitial GPU Status:")
    temps = get_gpu_temps()
    for gpu, temp in sorted(temps.items()):
        print(f"  GPU {gpu}: {temp}C")

    all_results = []

    for gpu in WORKER_GPUS:
        try:
            result = run_benchmark(gpu)
            all_results.append(result)
        except KeyboardInterrupt:
            print("\nBenchmark interrupted")
            break
        except Exception as e:
            print(f"\nError testing GPU {gpu}: {e}")

    # Restart normal ollama
    print("\nRestarting normal Ollama server...")
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Summary
    print("\n" + "="*60)
    print("WORKER BENCHMARK RESULTS")
    print("="*60)
    print(f"{'GPU':<8} {'Avg tok/s':<12} {'Avg TTFT':<12} {'Temp Rise':<12}")
    print("-"*60)

    for result in all_results:
        runs = [r for r in result["runs"] if "error" not in r]
        if runs:
            avg_speed = sum(r["tokens_per_sec"] for r in runs) / len(runs)
            avg_ttft = sum(r["time_to_first_token"] for r in runs) / len(runs)
            temp_rise = (result["temp_after"] or 0) - (result["temp_before"] or 0)
            print(f"GPU {result['gpu']:<4} {avg_speed:<12.2f} {avg_ttft:<12.2f}s {temp_rise:<12}C")

    total_throughput = sum(
        sum(r["tokens_per_sec"] for r in result["runs"] if "error" not in r) / len([r for r in result["runs"] if "error" not in r])
        for result in all_results if any("error" not in r for r in result["runs"])
    )
    print("-"*60)
    print(f"Combined worker throughput (3 GPUs): ~{total_throughput:.1f} tok/s")

    # Save results
    output_file = "/home/bryan/Documents/llm_orchestration/docs/worker_benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
