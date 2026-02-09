#!/usr/bin/env python3
"""
GPU Pairing Benchmark for Multi-GPU LLM Inference
Tests different GPU pairs to find optimal split for the brain model.
"""

import subprocess
import json
import time
import sys
import os
import requests

# Test configuration
MODEL = "qwen2.5:14b"
PROMPT = "Write a detailed 500-word analysis of distributed computing architectures, covering key concepts like load balancing, fault tolerance, and consensus algorithms."
NUM_RUNS = 3
OLLAMA_API = "http://localhost:11434/api/generate"

# GPU pairs to test (for 2-GPU split)
GPU_PAIRS = [
    (0, 3),  # Both high-binned, theory says best
    (0, 1),  # High + low comparison
    (1, 2),  # Both lower-binned baseline
]

def stop_ollama():
    """Stop any running model to free VRAM."""
    try:
        subprocess.run(["ollama", "stop", MODEL], capture_output=True, timeout=10)
    except:
        pass
    time.sleep(2)

def get_gpu_temps():
    """Get current GPU temperatures."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,temperature.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    temps = {}
    for line in result.stdout.strip().split('\n'):
        if line:
            idx, temp = line.split(',')
            temps[int(idx.strip())] = int(temp.strip())
    return temps

def run_benchmark(gpu_pair):
    """Run benchmark on a specific GPU pair."""
    gpu_str = ",".join(map(str, gpu_pair))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_str

    print(f"\n{'='*60}")
    print(f"Testing GPU pair: {gpu_pair}")
    print(f"CUDA_VISIBLE_DEVICES={gpu_str}")
    print('='*60)

    # Stop any running model first
    stop_ollama()

    # Restart ollama with specific GPUs
    # Kill existing ollama and restart with GPU restriction
    subprocess.run(["pkill", "-f", "ollama serve"], capture_output=True)
    time.sleep(2)

    # Start ollama with restricted GPUs
    ollama_proc = subprocess.Popen(
        ["ollama", "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(5)  # Wait for server to start

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
                json={
                    "model": MODEL,
                    "prompt": PROMPT,
                    "stream": True
                },
                stream=True,
                timeout=300
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
                        eval_duration = data.get("eval_duration", 0) / 1e9  # ns to s
                        prompt_eval_duration = data.get("prompt_eval_duration", 0) / 1e9

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
                            "prompt_eval_time": prompt_eval_duration,
                        })

                        print(f"    Tokens: {eval_count}, Speed: {tokens_per_sec:.2f} tok/s, TTFT: {first_token_time:.2f}s")
                        break

        except Exception as e:
            print(f"    Error: {e}")
            results.append({"run": run + 1, "error": str(e)})

        time.sleep(2)  # Cool down between runs

    temps_after = get_gpu_temps()

    # Clean up
    ollama_proc.terminate()
    time.sleep(2)

    return {
        "gpu_pair": gpu_pair,
        "temps_before": {k: temps_before.get(k) for k in gpu_pair},
        "temps_after": {k: temps_after.get(k) for k in gpu_pair},
        "runs": results
    }

def summarize_results(all_results):
    """Print summary comparison of all GPU pairs."""
    print("\n" + "="*70)
    print("SUMMARY - GPU PAIR BENCHMARK RESULTS")
    print("="*70)
    print(f"{'GPU Pair':<12} {'Avg tok/s':<12} {'Avg TTFT':<12} {'Temp Rise':<12}")
    print("-"*70)

    best_pair = None
    best_speed = 0

    for result in all_results:
        pair = result["gpu_pair"]
        runs = [r for r in result["runs"] if "error" not in r]

        if runs:
            avg_speed = sum(r["tokens_per_sec"] for r in runs) / len(runs)
            avg_ttft = sum(r["time_to_first_token"] for r in runs) / len(runs)
            temp_rise = max(
                result["temps_after"].get(g, 0) - result["temps_before"].get(g, 0)
                for g in pair
            )

            print(f"{str(pair):<12} {avg_speed:<12.2f} {avg_ttft:<12.2f}s {temp_rise:<12}C")

            if avg_speed > best_speed:
                best_speed = avg_speed
                best_pair = pair
        else:
            print(f"{str(pair):<12} FAILED")

    print("-"*70)
    if best_pair:
        print(f"\nRECOMMENDED PAIR FOR BRAIN MODEL: GPU {best_pair}")
        print(f"Remaining GPUs for workers: {[i for i in range(4) if i not in best_pair]}")

    return best_pair

def main():
    print("GPU Pairing Benchmark for LLM Inference")
    print(f"Model: {MODEL}")
    print(f"Runs per pair: {NUM_RUNS}")

    # Check current GPU status
    print("\nInitial GPU Status:")
    temps = get_gpu_temps()
    for gpu, temp in sorted(temps.items()):
        print(f"  GPU {gpu}: {temp}C")

    all_results = []

    for pair in GPU_PAIRS:
        try:
            result = run_benchmark(pair)
            all_results.append(result)
        except KeyboardInterrupt:
            print("\nBenchmark interrupted")
            break
        except Exception as e:
            print(f"\nError testing pair {pair}: {e}")
            all_results.append({"gpu_pair": pair, "runs": [], "error": str(e)})

    # Restart normal ollama
    print("\nRestarting normal Ollama server...")
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    best_pair = summarize_results(all_results)

    # Save results
    output_file = "/home/bryan/Documents/llm_orchestration/docs/gpu_pair_benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")

    return best_pair

if __name__ == "__main__":
    main()
