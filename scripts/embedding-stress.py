#!/usr/bin/env python3
"""GPU stress test simulating embedding workloads on all GPUs simultaneously
Usage: python embedding-stress.py [duration_seconds]
Default: 120 seconds (2 minutes)
"""
import torch
import torch.multiprocessing as mp
import time
import subprocess
import threading
import os
import sys

def get_gpu_stats():
    """Get current GPU stats via nvidia-smi"""
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,temperature.gpu,power.draw,utilization.gpu',
         '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    stats = {}
    for line in result.stdout.strip().split('\n'):
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 4:
            idx = int(parts[0])
            stats[idx] = {
                'temp': int(parts[1]),
                'power': float(parts[2]),
                'util': int(parts[3])
            }
    return stats

def monitor_gpus(stop_event, num_gpus, interval=10):
    """Background thread to monitor GPU stats"""
    header = f"{'Time':<8}"
    for i in range(num_gpus):
        header += f" {'GPU'+str(i):>14}"
    print(f"\n{header}")

    subheader = f"{'':8}"
    for i in range(num_gpus):
        subheader += f" {'Temp/Pwr/Util':>14}"
    print(subheader)
    print("-" * (8 + 15 * num_gpus))

    start = time.time()
    while not stop_event.is_set():
        stats = get_gpu_stats()
        elapsed = time.time() - start

        line = f"{elapsed:>6.0f}s "
        for i in range(num_gpus):
            if i in stats:
                s = stats[i]
                line += f" {s['temp']:>2}C/{s['power']:>5.1f}W/{s['util']:>3}%"
            else:
                line += f" {'N/A':>14}"
        print(line)

        time.sleep(interval)

def embedding_stress_worker(gpu_id, duration, result_queue):
    """
    Simulate embedding workload on a single GPU.
    Embeddings use transformer-like patterns: attention + FFN
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    torch.cuda.set_device(0)  # Now device 0 is our assigned GPU
    device = torch.device('cuda:0')

    # Embedding model dimensions (similar to all-MiniLM-L6-v2)
    batch_size = 64
    seq_len = 128
    hidden_dim = 384
    num_heads = 6
    head_dim = hidden_dim // num_heads
    ffn_dim = hidden_dim * 4
    num_layers = 6

    # Pre-allocate tensors for transformer-like operations
    # Input embeddings
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.float32)

    # Attention weights (Q, K, V projections)
    Wq = torch.randn(hidden_dim, hidden_dim, device=device, dtype=torch.float32)
    Wk = torch.randn(hidden_dim, hidden_dim, device=device, dtype=torch.float32)
    Wv = torch.randn(hidden_dim, hidden_dim, device=device, dtype=torch.float32)
    Wo = torch.randn(hidden_dim, hidden_dim, device=device, dtype=torch.float32)

    # FFN weights
    W1 = torch.randn(hidden_dim, ffn_dim, device=device, dtype=torch.float32)
    W2 = torch.randn(ffn_dim, hidden_dim, device=device, dtype=torch.float32)

    torch.cuda.synchronize()

    iterations = 0
    embeddings_processed = 0
    start = time.time()

    while time.time() - start < duration:
        # Simulate transformer layers
        for layer in range(num_layers):
            # Multi-head attention
            Q = torch.matmul(x, Wq)
            K = torch.matmul(x, Wk)
            V = torch.matmul(x, Wv)

            # Reshape for multi-head
            Q = Q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            K = K.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            V = V.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

            # Attention scores
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
            attn = torch.softmax(scores, dim=-1)
            context = torch.matmul(attn, V)

            # Reshape back
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
            attn_out = torch.matmul(context, Wo)

            # Residual + FFN
            x = x + attn_out
            ffn_hidden = torch.relu(torch.matmul(x, W1))
            ffn_out = torch.matmul(ffn_hidden, W2)
            x = x + ffn_out

        # Mean pooling to get embeddings
        embeddings = x.mean(dim=1)

        torch.cuda.synchronize()
        iterations += 1
        embeddings_processed += batch_size

    elapsed = time.time() - start
    embeddings_per_sec = embeddings_processed / elapsed

    result_queue.put({
        'gpu_id': gpu_id,
        'iterations': iterations,
        'embeddings': embeddings_processed,
        'elapsed': elapsed,
        'emb_per_sec': embeddings_per_sec
    })

def main():
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    num_gpus = torch.cuda.device_count()

    print(f"\n{'='*60}")
    print(f"EMBEDDING STRESS TEST - {num_gpus} GPUs for {duration} seconds")
    print(f"{'='*60}")
    print(f"Simulating transformer embedding workload (6-layer, 384-dim)")
    print(f"Batch size: 64, Sequence length: 128\n")

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")

    # Start GPU monitor thread
    stop_monitor = threading.Event()
    monitor_thread = threading.Thread(target=monitor_gpus, args=(stop_monitor, num_gpus, 10))
    monitor_thread.start()

    # Start all GPU workers using spawn
    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()

    processes = []
    start_time = time.time()

    for gpu_id in range(num_gpus):
        p = mp.Process(target=embedding_stress_worker, args=(gpu_id, duration, result_queue))
        p.start()
        processes.append(p)

    # Wait for all workers
    for p in processes:
        p.join()

    # Stop monitor
    stop_monitor.set()
    monitor_thread.join()

    total_time = time.time() - start_time

    # Collect results
    results = {}
    while not result_queue.empty():
        r = result_queue.get()
        results[r['gpu_id']] = r

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"{'GPU':<5} {'Iterations':>12} {'Embeddings':>12} {'Emb/sec':>12}")
    print("-" * 45)

    total_emb_per_sec = 0
    for gpu_id in sorted(results.keys()):
        r = results[gpu_id]
        print(f"{gpu_id:<5} {r['iterations']:>12} {r['embeddings']:>12} {r['emb_per_sec']:>12.1f}")
        total_emb_per_sec += r['emb_per_sec']

    print("-" * 45)
    print(f"{'TOTAL':<30} {total_emb_per_sec:>12.1f} emb/sec")
    print(f"\nTest completed in {total_time:.1f} seconds")

    # Final GPU stats
    print(f"\nFinal GPU temperatures:")
    stats = get_gpu_stats()
    for i in sorted(stats.keys()):
        s = stats[i]
        print(f"  GPU {i}: {s['temp']}°C, {s['power']:.1f}W")

if __name__ == "__main__":
    main()
