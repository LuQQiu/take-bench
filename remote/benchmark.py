#!/usr/bin/env python3
"""
Benchmark script for LanceDB remote PyTorch DataLoader.
Tests the same configurations as the storage version but using remote take API.
"""

import lancedb
import torch
from torch.utils.data import DataLoader
import numpy as np
import pyarrow as pa
import time
import os
import csv
from datetime import datetime

from lancedb_map_dataset import LanceDBMapDataset


def create_remote_dataset_if_needed(num_rows, num_cols, col_size_bytes, table_name, db, force_recreate=False):
    """Create dataset in remote LanceDB if it doesn't exist."""
    
    try:
        # Check if table exists
        existing_tables = db.table_names()
        if table_name in existing_tables and not force_recreate:
            print(f"Table already exists: {table_name}")
            # Open and return row count
            table = db.open_table(table_name)
            return num_rows  # Assume it has the expected rows
    except:
        pass
    
    print(f"Creating remote table: {table_name}")
    print(f"  Rows: {num_rows}, Cols: {num_cols}, Col size: {col_size_bytes}B")
    
    # Drop table if exists
    try:
        db.drop_table(table_name)
    except:
        pass
    
    # Generate data in batches to avoid memory issues
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
        
        table_data = pa.Table.from_pydict(data)
        
        if rows_written == 0:
            # Create table with first batch
            db.create_table(table_name, data=table_data, mode="overwrite")
        else:
            # Append subsequent batches
            table = db.open_table(table_name)
            table.add(table_data)
        
        rows_written += batch_rows
        print(f"  Written {rows_written}/{num_rows} rows")
    
    print(f"Remote table created: {table_name}")
    return num_rows


def run_benchmark(table_name, dataset_length, batch_size, num_workers, num_batches=100, 
                  db_uri="db://my-db", api_key="sk_localtest", host_override="http://localhost:10024",
                  prefetch_factor=2):
    """Run the benchmark and return metrics."""
    
    # Create dataset
    dataset = LanceDBMapDataset(
        table_name=table_name,
        db_uri=db_uri,
        api_key=api_key,
        host_override=host_override
    )
    dataset.set_length(dataset_length)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # Always shuffle for realistic data loading
        num_workers=num_workers,
        prefetch_factor=prefetch_factor
    )
    
    # Warmup
    print("  Warming up...")
    for i, batch in enumerate(dataloader):
        if i >= 5:
            break
    
    # Benchmark
    print("  Running benchmark...")
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
    # small local tests
    {
        "columns": 1,
        "data_size": 10,
        "batch_size": 32,
        "total_batches": 100,
        "prefetch_factor": 1,
    },
    
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
            "Remote",  # Changed from "Storage" to "Remote"
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


def run_single_test(config, db_uri="db://my-db", api_key="sk_localtest", 
                   host_override="http://localhost:10024"):
    """Run a single test configuration."""
    # Generate table name matching storage version format
    table_name = f"col{config['columns']}_data{config['data_size']}_batch{config['total_batches']}_perbatch{config['batch_size']}"
    
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
    
    # Connect to remote LanceDB
    db = lancedb.connect(db_uri, api_key=api_key, host_override=host_override)
    
    # Create dataset if needed
    dataset_length = create_remote_dataset_if_needed(
        total_rows, 
        config['columns'], 
        config['data_size'], 
        table_name,
        db
    )
    
    # Run benchmark
    metrics = run_benchmark(
        table_name,
        dataset_length,
        config['batch_size'], 
        config['num_workers'], 
        config['total_batches'],
        db_uri=db_uri,
        api_key=api_key,
        host_override=host_override,
        prefetch_factor=config.get('prefetch_factor', 2)
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
    parser.add_argument("--db-uri", type=str, default="db://my-db", 
                       help="LanceDB database URI")
    parser.add_argument("--api-key", type=str, default="sk_localtest",
                       help="API key for LanceDB")
    parser.add_argument("--host-override", type=str, default="http://localhost:10024",
                       help="Host override for local testing")
    parser.add_argument("--num-workers", type=int, default=4, 
                       help="Number of dataloader workers")
    args = parser.parse_args()
    
    # Always run all tests
    print(f"Running all {len(TEST_CONFIGS)} tests with remote LanceDB...")
    print(f"Database URI: {args.db_uri}")
    print(f"Host override: {args.host_override}")
    print(f"Global num_workers: {args.num_workers}")
    print(f"Results will be saved to: benchmark_results.csv")
    
    for i, config in enumerate(TEST_CONFIGS):
        print(f"\n{'='*60}")
        print(f"Test {i+1}/{len(TEST_CONFIGS)}")
        print(f"{'='*60}")
        
        # Override num_workers with global setting, keep prefetch_factor from config
        config['num_workers'] = args.num_workers
        
        try:
            run_single_test(config, args.db_uri, args.api_key, args.host_override)
        except Exception as e:
            print(f"ERROR in test {i+1}: {e}")
            # Continue with next test
    
    print(f"\nAll results saved to: benchmark_results.csv")


if __name__ == "__main__":
    main()