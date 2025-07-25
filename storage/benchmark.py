#!/usr/bin/env python3
"""
Simple benchmark script for Lance PyTorch DataLoader.
Automatically creates datasets based on test dimensions.
"""

import multiprocessing
# Force spawn method for multiprocessing (required for lance)
try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass  # Already set

# Set environment variables to avoid multiprocessing issues
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

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
import gc

from lance_map_dataset import LanceMapDataset


def create_dataset_if_needed(num_rows, num_cols, col_size_bytes, dataset_path):
    """Create dataset if it doesn't exist."""
    try:
        if dataset_path.startswith('s3://'):
            dataset = lance.dataset(dataset_path, storage_options={
                "client_max_retries": "30", 
                "client_retry_timeout": "1800",
                "timeout": "60s",
                "connect_timeout": "30s",
                "aws_region": "us-west-2"
            })
        else:
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
            if dataset_path.startswith('s3://'):
                lance.write_dataset(table, dataset_path, mode="create", storage_options={
                    "timeout": "60s",
                    "connect_timeout": "30s",
                    "aws_region": "us-west-2"
                })
            else:
                lance.write_dataset(table, dataset_path, mode="create")
        else:
            if dataset_path.startswith('s3://'):
                lance.write_dataset(table, dataset_path, mode="append", storage_options={
                    "timeout": "60s",
                    "connect_timeout": "30s",
                    "aws_region": "us-west-2"
                })
            else:
                lance.write_dataset(table, dataset_path, mode="append")
        
        rows_written += batch_rows
        print(f"  Written {rows_written}/{num_rows} rows")
    
    print(f"Dataset created: {dataset_path}")


def run_benchmark(dataset_path, batch_size, num_workers, num_batches=100, prefetch_factor=2, epoch=1, row_size=100, config=None):
    """Run the benchmark and return metrics."""
    try:
        dataset = LanceMapDataset(dataset_path)
    except Exception as e:
        print(f"Error creating LanceMapDataset: {e}")
        raise
    
    # Validate prefetch_factor - if 0 or negative, don't set it (use None)
    if prefetch_factor <= 0:
        prefetch_factor = None
    
    # Initialize dataloader as None
    dataloader = None
    
    try:
        # Set multiprocessing context for DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,  # Always shuffle for realistic data loading
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            multiprocessing_context='spawn' if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False
        )
        
        # Force initialization of the iterator to ensure _shutdown is set
        if num_workers > 0:
            print(f"    Initializing {num_workers} workers...", flush=True)
            # Access the iterator to trigger initialization
            iter(dataloader)
            print(f"    Workers initialized", flush=True)
        
        # Warmup with only 1 batch
        print(f"    Warming up (1 batch)...", flush=True)
        warmup_start = time.time()
        for i, batch in enumerate(dataloader):
            if i >= 1:
                break
            print(f"      Warmup batch completed", flush=True)
        warmup_time = time.time() - warmup_start
        print(f"    Warmup completed in {warmup_time:.1f}s", flush=True)
        
        # Benchmark
        print(f"    Running benchmark (epoch {epoch})...", flush=True)
        batch_times = []
        start_time = time.time()
        rows_processed = 0
        
        last_interim_report = start_time
        interim_report_interval = 300  # Report every 5 minutes
        max_test_duration = 1800  # Maximum 30 minutes per test
        
        for i, batch in enumerate(dataloader):
            batch_start = time.time()
            rows_processed += len(batch['id'])
            batch_time = time.time() - batch_start
            batch_times.append(batch_time * 1000)  # Convert to ms
            
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Check for max test duration
            if elapsed > max_test_duration:
                print(f"\n    ⏱️  Test time limit reached: {i+1}/{num_batches} batches completed after {elapsed/60:.1f} minutes", flush=True)
                print(f"    Writing results and continuing to next test...", flush=True)
                break
            
            # Progress output every 10% or every 100 batches (whichever is smaller)
            progress_interval = min(100, max(1, num_batches // 10))
            should_report_progress = (i + 1) % progress_interval == 0
            
            # Also report interim results every N minutes
            should_report_interim = (current_time - last_interim_report) >= interim_report_interval
            
            if should_report_progress or should_report_interim:
                current_qps = (i + 1) / elapsed
                current_throughput_mb_s = (rows_processed * row_size) / (1024**2) / elapsed if elapsed > 0 else 0
                
                # Calculate current percentiles
                if batch_times:
                    batch_times_array = np.array(batch_times)
                    p50 = np.percentile(batch_times_array, 50)
                    p90 = np.percentile(batch_times_array, 90)
                    p95 = np.percentile(batch_times_array, 95)
                    p99 = np.percentile(batch_times_array, 99)
                else:
                    p50 = p90 = p95 = p99 = 0
                
                print(f"    Progress: {i+1}/{num_batches} batches ({(i+1)/num_batches*100:.1f}%) | "
                      f"QPS: {current_qps:.1f} | Elapsed: {elapsed:.1f}s", flush=True)
                
                if should_report_interim:
                    print(f"\n    📊 Interim Results (after {elapsed/60:.1f} minutes):", flush=True)
                    print(f"       QPS: {current_qps:.2f}", flush=True)
                    print(f"       Throughput: {current_throughput_mb_s:.2f} MB/s", flush=True)
                    print(f"       P50 Latency: {p50:.2f} ms", flush=True)
                    print(f"       P90 Latency: {p90:.2f} ms", flush=True)
                    print(f"       P95 Latency: {p95:.2f} ms", flush=True)
                    print(f"       P99 Latency: {p99:.2f} ms\n", flush=True)
                    
                    # Write interim results to CSV
                    interim_metrics = {
                        'rows_processed': rows_processed,
                        'total_time': elapsed,
                        'rows_per_second': rows_processed / elapsed if elapsed > 0 else 0,
                        'qps': current_qps,
                        'batches': i + 1,
                        'p50_latency_ms': p50,
                        'p90_latency_ms': p90,
                        'p95_latency_ms': p95,
                        'p99_latency_ms': p99,
                    }
                    
                    # Write interim results if config available
                    if config:
                        write_results_to_csv(config, interim_metrics, epoch, 
                                           csv_file="benchmark_results_interim.csv",
                                           error=f"Interim after {elapsed/60:.1f}min")
                    
                    last_interim_report = current_time
            
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
    finally:
        # Ensure DataLoader is properly cleaned up
        if hasattr(dataloader, '_iterator') and dataloader._iterator is not None:
            try:
                # First check if _shutdown attribute exists and set it
                if hasattr(dataloader._iterator, '_shutdown'):
                    dataloader._iterator._shutdown = True
                # Then try to call _shutdown_workers if it exists
                if hasattr(dataloader._iterator, '_shutdown_workers'):
                    dataloader._iterator._shutdown_workers()
            except Exception as e:
                print(f"Error during DataLoader cleanup: {e}")
                pass
        # Force garbage collection to clean up resources
        gc.collect()

TEST_CONFIGS = [
    # Test 0: 1 col, 10B, 32K batch, 1K batches, 32M rows, ~305 MB
    {"columns": 1, "data_size": 10, "batch_size": 32000, "total_batches": 1000},
    # Test 1: 1 col, 10B, 32K batch, 10K batches, 320M rows, ~3 GB
    {"columns": 1, "data_size": 10, "batch_size": 32000, "total_batches": 10000},
    # Test 2: 1 col, 100B, 16K batch, 1K batches, 16M rows, ~1.5 GB
    {"columns": 1, "data_size": 100, "batch_size": 16000, "total_batches": 1000},
    # Test 3: 1 col, 100B, 16K batch, 10K batches, 160M rows, ~15 GB
    {"columns": 1, "data_size": 100, "batch_size": 16000, "total_batches": 10000,
     "dataset_override": "col1_data100_batch10000_perbatch16000.lance",  # Use specific dataset
     "override_batch_size": 8000,  # Override batch size to 8K
     "override_total_batches": 20000},  # Override to 20K batches
    # Test 4: 1 col, 1KB, 8K batch, 1K batches, 8M rows, ~8 GB
    {"columns": 1, "data_size": 1024, "batch_size": 8000, "total_batches": 1000},
    # Test 5: 1 col, 1KB, 8K batch, 10K batches, 80M rows, ~80 GB
    {"columns": 1, "data_size": 1024, "batch_size": 8000, "total_batches": 10000},
    # Test 6: 1 col, 10KB, 4K batch, 1K batches, 4M rows, ~40 GB
    {"columns": 1, "data_size": 10240, "batch_size": 4000, "total_batches": 1000},
    # Test 7: 1 col, 100KB, 1K batch, 1K batches, 1M rows, ~100 GB
    {"columns": 1, "data_size": 102400, "batch_size": 1000, "total_batches": 1000},
    # Test 8: 1 col, 1MB, 128 batch, 1K batches, 128K rows, ~128 GB
    {"columns": 1, "data_size": 1048576, "batch_size": 128, "total_batches": 1000},
    # Test 9: 1 col, 10MB, 16 batch, 1K batches, 16K rows, ~160 GB
    {"columns": 1, "data_size": 10485760, "batch_size": 16, "total_batches": 1000},
    # Test 10: 1 col, 100MB, 2 batch, 1K batches, 2K rows, ~200 GB
    {"columns": 1, "data_size": 104857600, "batch_size": 2, "total_batches": 1000},
    # Test 11: 10 cols, 10B, 16K batch, 1K batches, 16M rows, ~1.5 GB
    {"columns": 10, "data_size": 10, "batch_size": 16000, "total_batches": 1000},
    # Test 12: 10 cols, 10B, 16K batch, 10K batches, 160M rows, ~15 GB
    {"columns": 10, "data_size": 10, "batch_size": 16000, "total_batches": 10000},
    # Test 13: 10 cols, 100B, 8K batch, 1K batches, 8M rows, ~8 GB
    {"columns": 10, "data_size": 100, "batch_size": 8000, "total_batches": 1000},
    # Test 14: 10 cols, 100B, 8K batch, 10K batches, 80M rows, ~80 GB
    {"columns": 10, "data_size": 100, "batch_size": 8000, "total_batches": 10000},
    # Test 15: 10 cols, 10KB, 1K batch, 1K batches, 1M rows, ~100 GB
    {"columns": 10, "data_size": 10240, "batch_size": 1000, "total_batches": 1000},
    # Test 16: 100 cols, 10B, 8K batch, 1K batches, 8M rows, ~8 GB
    {"columns": 100, "data_size": 10, "batch_size": 8000, "total_batches": 1000},
    # Test 17: 100 cols, 10B, 8K batch, 10K batches, 80M rows, ~80 GB
    {"columns": 100, "data_size": 10, "batch_size": 8000, "total_batches": 10000},
    # Test 18: 100 cols, 100B, 4K batch, 1K batches, 4M rows, ~40 GB
    {"columns": 100, "data_size": 100, "batch_size": 4000, "total_batches": 1000},
    # Test 19: 100 cols, 1KB, 1K batch, 1K batches, 1M rows, ~100 GB
    {"columns": 100, "data_size": 1024, "batch_size": 1000, "total_batches": 1000},
    # Test 20: 10 cols, 100KB, 1K batch, 128 batches, 128K rows, ~128 GB
    {"columns": 10, "data_size": 102400, "batch_size": 1000, "total_batches": 128},
]


def write_results_to_csv(config, metrics, epoch, csv_file="benchmark_results.csv", error=None):
    """Write benchmark results to CSV file."""
    file_exists = os.path.exists(csv_file)
    
    # Calculate derived values
    total_rows = config['batch_size'] * config['total_batches']
    row_size = config['columns'] * config['data_size']
    total_dataset_size_mb = (total_rows * row_size) / (1024**2)
    
    # Handle error case
    if error:
        throughput_mb_s = 0
        qps = 0
        p50 = p90 = p95 = p99 = 0
    else:
        throughput_mb_s = (metrics['rows_per_second'] * row_size) / (1024**2)
        qps = metrics['qps']
        p50 = metrics['p50_latency_ms']
        p90 = metrics['p90_latency_ms']
        p95 = metrics['p95_latency_ms']
        p99 = metrics['p99_latency_ms']
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file is new
        if not file_exists:
            header = [
                "timestamp",
                "Test Approach",
                "Epoch",
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
                "P99 Latency (ms)",
                "Error"
            ]
            writer.writerow(header)
        
        # Write data row
        row = [
            datetime.now().isoformat(),
            "Storage",
            epoch,
            config['columns'],
            config['data_size'],
            row_size,
            config['batch_size'],
            config['total_batches'],
            total_rows,
            f"{total_dataset_size_mb:.2f}",
            config['num_workers'],
            config['prefetch_factor'],
            f"{qps:.2f}",
            f"{throughput_mb_s:.2f}",
            f"{p50:.2f}",
            f"{p90:.2f}",
            f"{p95:.2f}",
            f"{p99:.2f}",
            error if error else "N/A"
        ]
        writer.writerow(row)


def run_single_test(config, data_dir="./lance_data", num_epochs=1):
    """Run a single test configuration."""
    # Check for dataset override in config
    if 'dataset_override' in config:
        # Use the override dataset path
        table_name = config['dataset_override'].replace('.lance', '')
        dataset_path = f"{data_dir}/{config['dataset_override']}"
        actual_batch_size = config.get('override_batch_size', config['batch_size'])
        actual_total_batches = config.get('override_total_batches', config['total_batches'])
        print(f"  NOTE: Using override dataset {table_name}")
        print(f"        with batch_size={actual_batch_size}, batches={actual_total_batches}")
    else:
        # Generate table name normally
        table_name = f"col{config['columns']}_data{config['data_size']}_batch{config['total_batches']}_perbatch{config['batch_size']}"
        dataset_path = f"{data_dir}/{table_name}.lance"
        actual_batch_size = config['batch_size']
        actual_total_batches = config['total_batches']
    
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
    
    # Run benchmark for each epoch
    all_metrics = []
    for epoch in range(1, num_epochs + 1):
        if num_epochs > 1:
            print(f"\n  === Epoch {epoch}/{num_epochs} ===")
        
        metrics = run_benchmark(
            dataset_path, 
            actual_batch_size,  # Use actual_batch_size instead of config['batch_size']
            config['num_workers'], 
            actual_total_batches,  # Use actual_total_batches
            config['prefetch_factor'],
            epoch=epoch,
            row_size=row_size,
            config=config
        )
        
        # Calculate throughput
        throughput_mb_s = (metrics['rows_per_second'] * row_size) / (1024**2)
        
        print(f"\nResults (Epoch {epoch}):")
        print(f"  QPS: {metrics['qps']:.2f}")
        print(f"  Throughput: {throughput_mb_s:.2f} MB/s")
        print(f"  P50 latency: {metrics['p50_latency_ms']:.2f} ms")
        print(f"  P90 latency: {metrics['p90_latency_ms']:.2f} ms")
        print(f"  P99 latency: {metrics['p99_latency_ms']:.2f} ms")
        
        # Write to CSV
        write_results_to_csv(config, metrics, epoch)
        
        all_metrics.append(metrics)
    
    return all_metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./lance_data")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of epochs to run for each test")
    parser.add_argument("--prefetch-factor", type=int, default=2,
                       help="Prefetch factor for dataloader (default: 2)")
    parser.add_argument("--test-indices", type=int, nargs='+',
                       help="Specific test indices to run (0-based). If not specified, runs all tests sequentially.")
    parser.add_argument("--skip-indices", type=int, nargs='+',
                       help="Test indices to skip (0-based).")
    parser.add_argument("--max-test-minutes", type=int, default=0,
                       help="Maximum minutes per test before moving to next (default: 0 = no limit)")
    args = parser.parse_args()
    
    # Check file descriptor limit when using many workers
    if args.num_workers > 16:
        try:
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            if soft < 65536:
                print(f"\nWARNING: File descriptor limit is {soft}, which may be too low for {args.num_workers} workers.")
                print(f"You may encounter 'Too many open files' errors.")
                print(f"To fix, run: ulimit -n 65536")
                print(f"Or reduce --num-workers to 16 or less.\n")
        except:
            pass
    
    # Extra warning for very high worker counts
    if args.num_workers > 64:
        print(f"\nWARNING: Using {args.num_workers} workers is extremely high!")
        print(f"This may cause:")
        print(f"  - System resource exhaustion")
        print(f"  - Slower performance due to overhead")
        print(f"  - S3 request throttling")
        print(f"Consider using 8-32 workers for optimal performance.\n")
    
    # Determine which tests to run
    if args.test_indices:
        tests_to_run = [(i, TEST_CONFIGS[i]) for i in args.test_indices if 0 <= i < len(TEST_CONFIGS)]
        print(f"Running specific tests: {args.test_indices}")
    else:
        tests_to_run = list(enumerate(TEST_CONFIGS))
        print(f"Running all {len(TEST_CONFIGS)} tests sequentially...")
    
    # Apply skip filter if specified
    if args.skip_indices:
        skip_set = set(args.skip_indices)
        tests_to_run = [(i, cfg) for i, cfg in tests_to_run if i not in skip_set]
        print(f"Skipping test indices: {sorted(args.skip_indices)}")
    
    print(f"Data directory: {args.data_dir}")
    print(f"Global num_workers: {args.num_workers}")
    print(f"Global prefetch_factor: {args.prefetch_factor}")
    print(f"Epochs per test: {args.epochs}")
    print(f"Results will be saved to: benchmark_results.csv")
    
    for test_num, (idx, config) in enumerate(tests_to_run):
        print(f"\n{'='*60}")
        print(f"Test {idx} ({test_num+1}/{len(tests_to_run)})")
        print(f"{'='*60}")
        
        # Override num_workers with global setting
        config['num_workers'] = args.num_workers
        # Set prefetch_factor from command line
        config['prefetch_factor'] = args.prefetch_factor
        # Add test index to config
        config['test_idx'] = idx
        
        try:
            run_single_test(config, args.data_dir, args.epochs)
            # Force garbage collection after each test
            gc.collect()
        except Exception as e:
            import traceback
            error_msg = str(e)
            print(f"\nERROR in test {idx}: {error_msg}")  # Print full error
            print(f"Full traceback:")
            traceback.print_exc()
            
            # Write detailed error to log file
            with open("error.log", "a") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Test Index: {idx}\n")
                f.write(f"Test Config:\n")
                f.write(f"  Columns: {config['columns']}\n")
                f.write(f"  Data Size: {config['data_size']} bytes\n")
                f.write(f"  Batch Size: {config['batch_size']}\n")
                f.write(f"  Total Batches: {config['total_batches']}\n")
                f.write(f"  Total Rows: {config['batch_size'] * config['total_batches']:,}\n")
                f.write(f"  Workers: {config['num_workers']}\n")
                f.write(f"  Prefetch Factor: {config['prefetch_factor']}\n")
                f.write(f"  Data Directory: {args.data_dir}\n")
                f.write(f"\nError:\n{error_msg}\n")
                f.write(f"{'='*80}\n")
            
            # Write to CSV with simple error indicator
            write_results_to_csv(config, None, 1, error=f"Error - see error.log (Test {idx})")
            
            print(f"Continuing with remaining tests...")
            continue  # Explicitly continue to next test
    
    print(f"\nAll results saved to: benchmark_results.csv")


if __name__ == "__main__":
    main()