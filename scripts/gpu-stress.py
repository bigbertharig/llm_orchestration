#!/usr/bin/env python3
"""GPU stress test - runs all GPUs at max load for specified duration
Usage: python gpu-stress.py [duration_seconds]
Default: 120 seconds (2 minutes)
"""
import torch
import torch.multiprocessing as mp
import time
import sys

def stress_gpu(gpu_id, duration, results):
    """Hammer a single GPU with continuous matrix multiplications"""
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    props = torch.cuda.get_device_properties(gpu_id)

    print(f"[GPU {gpu_id}] {props.name} - Starting stress test")

    # Large matrices to maximize GPU utilization
    size = 4096
    a = torch.randn(size, size, device=device, dtype=torch.float32)
    b = torch.randn(size, size, device=device, dtype=torch.float32)

    iterations = 0
    start = time.time()

    while time.time() - start < duration:
        c = torch.mm(a, b)
        torch.cuda.synchronize(device)
        iterations += 1

        # Progress every 10 seconds
        elapsed = time.time() - start
        if iterations % 50 == 0:
            print(f"[GPU {gpu_id}] {elapsed:.0f}s - {iterations} iterations")

    elapsed = time.time() - start

    # Calculate TFLOPS
    flops_per_iter = 2 * size**3
    total_flops = flops_per_iter * iterations
    tflops = total_flops / elapsed / 1e12

    results[gpu_id] = {
        'iterations': iterations,
        'elapsed': elapsed,
        'tflops': tflops
    }
    print(f"[GPU {gpu_id}] Done - {iterations} iterations, {tflops:.2f} TFLOPS avg")

def main():
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 120

    num_gpus = torch.cuda.device_count()
    print(f"\n{'='*60}")
    print(f"GPU STRESS TEST - {num_gpus} GPUs for {duration} seconds")
    print(f"{'='*60}\n")

    if num_gpus == 0:
        print("No GPUs found!")
        sys.exit(1)

    # Show initial state
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    print()

    # Use multiprocessing to stress all GPUs simultaneously
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    results = manager.dict()

    processes = []
    start_time = time.time()

    for gpu_id in range(num_gpus):
        p = mp.Process(target=stress_gpu, args=(gpu_id, duration, results))
        p.start()
        processes.append(p)

    # Wait for all to complete
    for p in processes:
        p.join()

    total_time = time.time() - start_time

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"{'GPU':<5} {'Iterations':>12} {'Time (s)':>10} {'TFLOPS':>10}")
    print("-" * 40)

    total_tflops = 0
    for gpu_id in sorted(results.keys()):
        r = results[gpu_id]
        print(f"{gpu_id:<5} {r['iterations']:>12} {r['elapsed']:>10.1f} {r['tflops']:>10.2f}")
        total_tflops += r['tflops']

    print("-" * 40)
    print(f"{'TOTAL':<28} {total_tflops:>10.2f} TFLOPS")
    print(f"\nTest completed in {total_time:.1f} seconds")
    print(f"Expected per GTX 1060: ~3.5-4.0 TFLOPS (FP32)")

if __name__ == "__main__":
    main()
