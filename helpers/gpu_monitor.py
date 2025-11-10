#!/usr/bin/env python3
"""
Simple GPU VRAM monitoring script.
Logs nvidia-smi memory usage every 30 seconds with timestamps.
Run this in parallel while you launch training in another terminal.
"""

import subprocess
import time
import signal
import sys
from datetime import datetime


def get_gpu_info():
    """Query nvidia-smi for VRAM usage"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu', 
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return None, f"Error querying nvidia-smi: {e}"
    except FileNotFoundError:
        return None, "nvidia-smi not found. Is NVIDIA driver installed?"


def log_gpu_memory(log_file):
    """Log GPU memory usage with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gpu_info = get_gpu_info()
    
    log_entry = f"\n{'='*80}\n"
    log_entry += f"[{timestamp}]\n"
    log_entry += f"{'='*80}\n"
    
    if gpu_info:
        lines = gpu_info.split('\n')
        for line in lines:
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 6:
                    gpu_idx, gpu_name, mem_used, mem_total, gpu_util, temp = parts
                    mem_pct = float(mem_used) / float(mem_total) * 100
                    
                    log_entry += f"GPU {gpu_idx} ({gpu_name}):\n"
                    log_entry += f"  Memory: {mem_used} MB / {mem_total} MB ({mem_pct:.1f}%)\n"
                    log_entry += f"  GPU Util: {gpu_util}%\n"
                    log_entry += f"  Temp: {temp}°C\n"
    else:
        log_entry += "No GPU information available\n"
    
    # Write to file
    with open(log_file, 'a') as f:
        f.write(log_entry)
    
    # Print to console
    print(log_entry, flush=True)


def main():
    # Configuration
    log_file = "gpu_vram_log.txt"
    interval = 30  # seconds
    duration_hours = None  # None = run forever
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        try:
            interval = int(sys.argv[1])
        except ValueError:
            print(f"Usage: {sys.argv[0]} [interval_seconds] [log_file] [duration_hours]")
            print(f"Example: {sys.argv[0]} 30 gpu_log.txt 10")
            print(f"Using default interval: {interval} seconds")
    
    if len(sys.argv) > 2:
        log_file = sys.argv[2]
    
    if len(sys.argv) > 3:
        try:
            duration_hours = float(sys.argv[3])
        except ValueError:
            print(f"Invalid duration. Running indefinitely.")
            duration_hours = None
    
    print(f"GPU VRAM Monitor")
    print(f"{'='*80}")
    print(f"Log file: {log_file}")
    print(f"Interval: {interval} seconds")
    if duration_hours:
        print(f"Duration: {duration_hours} hours ({duration_hours * 60:.0f} minutes)")
    else:
        print(f"Duration: Infinite (press Ctrl+C to stop)")
    print(f"{'='*80}\n")
    
    # Write header to log file
    with open(log_file, 'a') as f:
        f.write(f"\n{'#'*80}\n")
        f.write(f"# GPU Monitoring Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Interval: {interval} seconds\n")
        if duration_hours:
            f.write(f"# Duration: {duration_hours} hours\n")
        f.write(f"{'#'*80}\n")
    
    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\n\nStopping GPU monitoring...")
        with open(log_file, 'a') as f:
            f.write(f"\n{'#'*80}\n")
            f.write(f"# GPU Monitoring Stopped: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'#'*80}\n\n")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Calculate end time if duration is specified
    start_time = time.time()
    end_time = start_time + (duration_hours * 3600) if duration_hours else None
    
    # Monitoring loop
    try:
        iteration = 0
        while True:
            log_gpu_memory(log_file)
            iteration += 1
            
            # Check if duration has elapsed
            if end_time and time.time() >= end_time:
                elapsed = time.time() - start_time
                print(f"\n{'='*80}")
                print(f"Duration limit reached: {elapsed/3600:.2f} hours")
                print(f"Total logs: {iteration}")
                print(f"{'='*80}")
                signal_handler(None, None)
                break
            
            # Show progress if duration is set
            if end_time and iteration % 10 == 0:  # Every 10 logs
                elapsed = time.time() - start_time
                remaining = end_time - time.time()
                print(f"[Progress] Elapsed: {elapsed/3600:.2f}h | Remaining: {remaining/3600:.2f}h")
            
            time.sleep(interval)
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    main()