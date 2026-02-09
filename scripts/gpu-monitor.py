#!/usr/bin/env python3
"""GPU monitoring script - logs temps, power, clocks, and throttle status
Usage: python gpu-monitor.py [interval_seconds] [--log filename]
Default: 5 second interval, output to console
"""
import subprocess
import time
import sys
import argparse
from datetime import datetime

def get_gpu_data():
    """Get comprehensive GPU stats via nvidia-smi"""
    result = subprocess.run([
        'nvidia-smi',
        '--query-gpu=index,name,temperature.gpu,power.draw,power.limit,'
        'clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,'
        'utilization.gpu,utilization.memory,memory.used,memory.total,'
        'clocks_throttle_reasons.active',
        '--format=csv,noheader,nounits'
    ], capture_output=True, text=True)

    gpus = []
    for line in result.stdout.strip().split('\n'):
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 14:
            gpus.append({
                'index': int(parts[0]),
                'name': parts[1],
                'temp': int(parts[2]),
                'power': float(parts[3]),
                'power_limit': float(parts[4]),
                'clock_sm': int(parts[5]),
                'clock_mem': int(parts[6]),
                'clock_sm_max': int(parts[7]),
                'clock_mem_max': int(parts[8]),
                'util_gpu': int(parts[9]),
                'util_mem': int(parts[10]),
                'mem_used': int(parts[11]),
                'mem_total': int(parts[12]),
                'throttle': parts[13]
            })
    return gpus

def parse_throttle(throttle_str):
    """Parse throttle reasons into short codes"""
    if 'None' in throttle_str or not throttle_str.strip():
        return '-'

    codes = []
    if 'SwThermalSlowdown' in throttle_str or 'HwThermalSlowdown' in throttle_str:
        codes.append('THERM')
    if 'SwPowerCap' in throttle_str or 'HwPowerBrakeSlowdown' in throttle_str:
        codes.append('PWR')
    if 'SyncBoost' in throttle_str:
        codes.append('SYNC')
    if 'GpuIdle' in throttle_str:
        codes.append('IDLE')
    if 'ApplicationsClocksSetting' in throttle_str:
        codes.append('APP')

    return ','.join(codes) if codes else throttle_str[:10]

def print_header(gpus, file=None):
    """Print column headers"""
    header1 = f"{'Time':<12}"
    header2 = f"{'':12}"

    for gpu in gpus:
        header1 += f" | {'GPU ' + str(gpu['index']):^28}"
        header2 += f" | {'Temp':>4} {'Pwr':>5} {'SM':>5} {'Util':>4} {'Throt':>6}"

    line = "=" * len(header1)

    output = f"\n{line}\n{header1}\n{header2}\n{line}"
    print(output)
    if file:
        file.write(output + '\n')

def print_stats(gpus, file=None):
    """Print current GPU stats"""
    now = datetime.now().strftime("%H:%M:%S")
    line = f"{now:<12}"

    for gpu in gpus:
        throttle = parse_throttle(gpu['throttle'])
        clock_pct = int(100 * gpu['clock_sm'] / gpu['clock_sm_max']) if gpu['clock_sm_max'] > 0 else 0
        line += f" | {gpu['temp']:>3}C {gpu['power']:>5.1f}W {clock_pct:>4}% {gpu['util_gpu']:>3}% {throttle:>6}"

    print(line)
    if file:
        file.write(line + '\n')
        file.flush()

def print_summary(gpus):
    """Print detailed GPU info"""
    print(f"\n{'='*70}")
    print("GPU DETAILS")
    print(f"{'='*70}")

    for gpu in gpus:
        print(f"\nGPU {gpu['index']}: {gpu['name']}")
        print(f"  Temperature: {gpu['temp']}°C")
        print(f"  Power:       {gpu['power']:.1f}W / {gpu['power_limit']:.1f}W limit")
        print(f"  SM Clock:    {gpu['clock_sm']} MHz (max {gpu['clock_sm_max']} MHz)")
        print(f"  Mem Clock:   {gpu['clock_mem']} MHz (max {gpu['clock_mem_max']} MHz)")
        print(f"  GPU Util:    {gpu['util_gpu']}%")
        print(f"  Memory:      {gpu['mem_used']} / {gpu['mem_total']} MB")
        print(f"  Throttle:    {gpu['throttle']}")

def main():
    parser = argparse.ArgumentParser(description='Monitor GPU stats')
    parser.add_argument('interval', nargs='?', type=float, default=5,
                        help='Monitoring interval in seconds (default: 5)')
    parser.add_argument('--log', '-l', type=str, help='Log to file')
    parser.add_argument('--summary', '-s', action='store_true',
                        help='Print detailed summary and exit')
    args = parser.parse_args()

    gpus = get_gpu_data()

    if not gpus:
        print("No GPUs found!")
        sys.exit(1)

    if args.summary:
        print_summary(gpus)
        sys.exit(0)

    log_file = open(args.log, 'a') if args.log else None

    print(f"\nMonitoring {len(gpus)} GPUs every {args.interval}s (Ctrl+C to stop)")
    print("Columns: Temp | Power | SM Clock % | GPU Util | Throttle Reason")
    print("Throttle codes: THERM=thermal, PWR=power cap, IDLE=idle, -=none")

    if log_file:
        log_file.write(f"\n--- Log started {datetime.now()} ---\n")
        print(f"Logging to: {args.log}")

    try:
        print_header(gpus, log_file)

        while True:
            gpus = get_gpu_data()
            print_stats(gpus, log_file)
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\nStopped.")
        if log_file:
            log_file.write(f"--- Log ended {datetime.now()} ---\n")
            log_file.close()

        # Print final summary
        gpus = get_gpu_data()
        print_summary(gpus)

if __name__ == "__main__":
    main()
