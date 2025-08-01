#!/usr/bin/env python3
"""
Simple benchmark script for Lance PyTorch DataLoader.
Automatically creates datasets based on test dimensions.
"""

import lance
import torch
from torch.utils.data import DataLoader
import numpy as np
import pyarrow as pa
import time
import os
import shutil
import csv
from datetime import datetime

from lance_map_dataset import LanceMapDataset


def create_dataset_if_needed(num_rows, num_cols, col_size_bytes, dataset_path):
    """Create dataset if it doesn't exist."""
    try:
        dataset = lance.dataset(dataset_path)
        if dataset.count_rows() == num_rows:
            print(f"Dataset already exists: {dataset_path} with {num_rows} rows")
            return
    except Exception:
        pass  # Dataset doesn't exist, will create
    
    print(f"Creating dataset: {num_rows} rows, {num_cols} cols, {col_size_bytes}B per col")
    
    # Generate data in batches
    batch_size = min(10000, num_rows)
    rows_written = 0
    
    while rows_written < num_rows:
        batch_rows = min(batch_size, num_rows - rows_written)
        
        # Create batch data
        data = {'id': np.arange(rows_written, rows_written + batch_rows, dtype=np.int64)}
        
        for i in range(num_cols):
            if col_size_bytes <= 8:
                data[f'col_{i}'] = np.random.randint(0, 1000, size=batch_rows, dtype=np.int32)
            else:
                data[f'col_{i}'] = [np.random.bytes(col_size_bytes) for _ in range(batch_rows)]
        
        table = pa.Table.from_pydict(data)
        
        if rows_written == 0:
            lance.write_dataset(table, dataset_path, mode="create")
        else:
            lance.write_dataset(table, dataset_path, mode="append")
        
        rows_written += batch_rows
        print(f"  Written {rows_written}/{num_rows} rows")
    
    print(f"Dataset created: {dataset_path}")


def run_benchmark(dataset_path, batch_size, num_workers, num_batches=100, prefetch_factor=2):
    """Run the benchmark and return metrics."""
    dataset = LanceMapDataset(dataset_path)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # Always shuffle for realistic data loading
        num_workers=num_workers,
        prefetch_factor=prefetch_factor
    )
    
    # Warmup
    for i, batch in enumerate(dataloader):
        if i >= 5:
            break
    
    # Benchmark
    batch_times = []
    start_time = time.time()
    rows_processed = 0
    
    for i, batch in enumerate(dataloader):
        batch_start = time.time()
        rows_processed += len(batch['id'])
        batch_time = time.time() - batch_start
        batch_times.append(batch_time * 1000)  # Convert to ms
        
        # Progress output every 10% or every 100 batches (whichever is smaller)
        progress_interval = min(100, max(1, num_batches // 10))
        if (i + 1) % progress_interval == 0:
            elapsed = time.time() - start_time
            current_qps = (i + 1) / elapsed
            print(f"    Progress: {i+1}/{num_batches} batches ({(i+1)/num_batches*100:.1f}%) | "
                  f"QPS: {current_qps:.1f} | Elapsed: {elapsed:.1f}s")
        
        if i >= num_batches:
            break
    
    total_time = time.time() - start_time
    
    # Calculate metrics
    batch_times = np.array(batch_times)
    qps = (i + 1) / total_time  # Queries (batches) per second
    rows_per_second = rows_processed / total_time
    
    return {
        'rows_processed': rows_processed,
        'total_time': total_time,
        'rows_per_second': rows_per_second,
        'qps': qps,
        'batches': i + 1,
        'p50_latency_ms': np.percentile(batch_times, 50),
        'p90_latency_ms': np.percentile(batch_times, 90),
        'p95_latency_ms': np.percentile(batch_times, 95),
        'p99_latency_ms': np.percentile(batch_times, 99),
    }

TEST_CONFIGS = [
    # Data Size: 10B, Row Size: 10B, Batch Size: 32K, Total Batches: 10K, Total Rows: 320M, ~3 GB
    {
        "columns": 1,
        "data_size": 10,
        "batch_size": 32000,
        "total_batches": 10000,
        "prefetch_factor": 1,
    },
    # Data Size: 100B, Row Size: 100B, Batch Size: 16K, Total Batches: 10K, Total Rows: 160M, ~15 GB
    {
        "columns": 1,
        "data_size": 100,
        "batch_size": 16000,
        "total_batches": 10000,
        "prefetch_factor": 1,
    },
    # Data Size: 1KB, Row Size: 1KB, Batch Size: 8K, Total Batches: 10K, Total Rows: 80M, ~80 GB
    {
        "columns": 1,
        "data_size": 1024,
        "batch_size": 8000,
        "total_batches": 10000,
        "prefetch_factor": 1,
    },
    # Data Size: 1KB, Row Size: 1KB, Batch Size: 8K, Total Batches: 1K, Total Rows: 8M, ~8 GB
    {
        "columns": 1,
        "data_size": 1024,
        "batch_size": 8000,
        "total_batches": 1000,
        "prefetch_factor": 1,
    },
    # Data Size: 10KB, Row Size: 10KB, Batch Size: 4K, Total Batches: 1K, Total Rows: 4M, ~40 GB
    {
        "columns": 1,
        "data_size": 10240,
        "batch_size": 4000,
        "total_batches": 1000,
        "prefetch_factor": 1,
    },
    # Data Size: 100KB, Row Size: 100KB, Batch Size: 1K, Total Batches: 1K, Total Rows: 1M, ~100 GB
    {
        "columns": 1,
        "data_size": 102400,
        "batch_size": 1000,
        "total_batches": 1000,
        "prefetch_factor": 1,
    },
    # Data Size: 1MB, Row Size: 1MB, Batch Size: 128, Total Batches: 1K, Total Rows: 128K, ~128 GB
    {
        "columns": 1,
        "data_size": 1048576,
        "batch_size": 128,
        "total_batches": 1000,
        "prefetch_factor": 1,
    },
    # Data Size: 100B, Row Size: 100B, Batch Size: 16K, Total Batches: 1K, Total Rows: 16M, ~1.5 GB
    {
        "columns": 1,
        "data_size": 100,
        "batch_size": 16000,
        "total_batches": 1000,
        "prefetch_factor": 1,
    },
    # Data Size: 10KB, Row Size: 10KB, Batch Size: 1K, Total Batches: 1K, Total Rows: 1M, ~10 GB
    {
        "columns": 1,
        "data_size": 10240,
        "batch_size": 1000,
        "total_batches": 1000,
        "prefetch_factor": 1,
    },
    # Data Size: 100KB, Row Size: 100KB, Batch Size: 512, Total Batches: 1K, Total Rows: 512K, ~50 GB
    {
        "columns": 1,
        "data_size": 102400,
        "batch_size": 512,
        "total_batches": 1000,
        "prefetch_factor": 1,
    },
]


def write_results_to_csv(config, metrics, csv_file="benchmark_results.csv"):
    """Write benchmark results to CSV file."""
    file_exists = os.path.exists(csv_file)
    
    # Calculate derived values
    total_rows = config['batch_size'] * config['total_batches']
    row_size = config['columns'] * config['data_size']
    total_dataset_size_mb = (total_rows * row_size) / (1024**2)
    throughput_mb_s = (metrics['rows_per_second'] * row_size) / (1024**2)
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file is new
        if not file_exists:
            header = [
                "timestamp",
                "Test Approach",
                "Columns",
                "Data Size",
                "Row Size", 
                "Batch Size",
                "Total Batches",
                "Total Rows",
                "Total Dataset Size (MB)",
                "data_loader_num_workers",
                "data_loader_prefetch_factor",
                "QPS",
                "Throughput (MB/s)",
                "P50 Latency (ms)",
                "P90 Latency (ms)",
                "P95 Latency (ms)",
                "P99 Latency (ms)"
            ]
            writer.writerow(header)
        
        # Write data row
        row = [
            datetime.now().isoformat(),
            "Storage",
            config['columns'],
            config['data_size'],
            row_size,
            config['batch_size'],
            config['total_batches'],
            total_rows,
            f"{total_dataset_size_mb:.2f}",
            config['num_workers'],
            config.get('prefetch_factor', 2),
            f"{metrics['qps']:.2f}",
            f"{throughput_mb_s:.2f}",
            f"{metrics['p50_latency_ms']:.2f}",
            f"{metrics['p90_latency_ms']:.2f}",
            f"{metrics['p95_latency_ms']:.2f}",
            f"{metrics['p99_latency_ms']:.2f}"
        ]
        writer.writerow(row)


def run_single_test(config, data_dir="./lance_data"):
    """Run a single test configuration."""
    # Generate table name: col{columns}_data{data_size}_batch{total_batches}_perbatch{batch_size}
    table_name = f"col{config['columns']}_data{config['data_size']}_batch{config['total_batches']}_perbatch{config['batch_size']}"
    dataset_path = f"{data_dir}/{table_name}.lance"
    
    # Calculate derived values
    total_rows = config['batch_size'] * config['total_batches']
    row_size = config['columns'] * config['data_size']
    total_dataset_size_mb = (total_rows * row_size) / (1024**2)
    
    print(f"\nRunning: {table_name}")
    print(f"  Columns: {config['columns']}")
    print(f"  Data size: {config['data_size']} bytes")
    print(f"  Row size: {row_size} bytes")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Total batches: {config['total_batches']}")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Total dataset size: {total_dataset_size_mb:.2f} MB")
    print(f"  Workers: {config['num_workers']}")
    
    # Create dataset if needed
    create_dataset_if_needed(
        total_rows, 
        config['columns'], 
        config['data_size'], 
        dataset_path
    )
    
    # Run benchmark
    metrics = run_benchmark(
        dataset_path, 
        config['batch_size'], 
        config['num_workers'], 
        config['total_batches'],
        config.get('prefetch_factor', 2)  # Default to 2 if not specified
    )
    
    # Calculate throughput
    throughput_mb_s = (metrics['rows_per_second'] * row_size) / (1024**2)
    
    print(f"\nResults:")
    print(f"  QPS: {metrics['qps']:.2f}")
    print(f"  Throughput: {throughput_mb_s:.2f} MB/s")
    print(f"  P50 latency: {metrics['p50_latency_ms']:.2f} ms")
    print(f"  P90 latency: {metrics['p90_latency_ms']:.2f} ms")
    print(f"  P99 latency: {metrics['p99_latency_ms']:.2f} ms")
    
    # Write to CSV
    write_results_to_csv(config, metrics)
    
    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./lance_data")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers")
    args = parser.parse_args()
    
    # Always run all tests
    print(f"Running all {len(TEST_CONFIGS)} tests...")
    print(f"Data directory: {args.data_dir}")
    print(f"Global num_workers: {args.num_workers}")
    print(f"Results will be saved to: benchmark_results.csv")
    
    for i, config in enumerate(TEST_CONFIGS):
        print(f"\n{'='*60}")
        print(f"Test {i+1}/{len(TEST_CONFIGS)}")
        print(f"{'='*60}")
        
        # Override num_workers with global setting, keep prefetch_factor from config
        config['num_workers'] = args.num_workers
        
        run_single_test(config, args.data_dir)
    
    print(f"\nAll results saved to: benchmark_results.csv")


if __name__ == "__main__":
    main()