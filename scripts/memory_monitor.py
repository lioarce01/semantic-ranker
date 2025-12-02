#!/usr/bin/env python3
"""
Memory monitor for QG-Rerank training with 32GB RAM optimization.
Run this in a separate terminal to monitor memory usage during training.
"""

import psutil
import time
import os
from datetime import datetime

def get_memory_info():
    """Get detailed memory information"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    # System memory
    system = psutil.virtual_memory()

    return {
        'process_rss': mem_info.rss / 1024 / 1024,  # MB
        'process_vms': mem_info.vms / 1024 / 1024,  # MB
        'system_used': system.used / 1024 / 1024 / 1024,  # GB
        'system_total': system.total / 1024 / 1024 / 1024,  # GB
        'system_percent': system.percent
    }

def monitor_memory(interval=10, duration=None):
    """Monitor memory usage over time"""
    print("Memory Monitor for QG-Rerank Training (32GB RAM)")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
    print("Monitoring every 10 seconds...")
    print("Press Ctrl+C to stop\n")

    start_time = time.time()
    peak_memory = 0

    try:
        while True:
            mem = get_memory_info()
            current_time = time.time() - start_time

            # Track peak memory
            peak_memory = max(peak_memory, mem['process_rss'])

            print(f"[{current_time:.0f}s] Process: {mem['process_rss']:.1f}MB | "
                  f"System: {mem['system_used']:.1f}/{mem['system_total']:.1f}GB "
                  f"({mem['system_percent']:.1f}%) | Peak: {peak_memory:.1f}MB")

            # Warnings for high memory usage
            if mem['system_percent'] > 85:
                print("  ⚠️  WARNING: System memory > 85%")
            elif mem['process_rss'] > 8000:  # 8GB for process
                print("  ⚠️  WARNING: Process memory > 8GB")

            time.sleep(interval)

            if duration and current_time >= duration:
                break

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")

    print(f"\nFinal stats:")
    print(f"- Peak process memory: {peak_memory:.1f}MB")
    print(f"- System memory: {mem['system_used']:.1f}/{mem['system_total']:.1f}GB")
    print(f"- Monitoring duration: {current_time:.0f} seconds")

if __name__ == "__main__":
    print("Usage: python memory_monitor.py")
    print("This will monitor memory usage every 10 seconds")
    print("Run this while training QG-Rerank in another terminal")
    print()

    monitor_memory()