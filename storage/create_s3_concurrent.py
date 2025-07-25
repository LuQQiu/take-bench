#!/usr/bin/env python3
"""
Concurrent dataset creation for S3 - launches all datasets at once.
Each dataset gets its proportional share of workers.
"""

import lance
import numpy as np
import pyarrow as pa
import os
import time
from multiprocessing import Process, cpu_count, Queue, set_start_method
import argparse
import threading

# Force spawn method for multiprocessing (required for lance + S3)
try:
    set_start_method('spawn')
except RuntimeError:
    pass  # Already set

TEST_CONFIGS = [
    # Test 0: 1 col, 10B, 32K batch, 1K batches, 32M rows, ~305 MB
    {"columns": 1, "data_size": 10, "batch_size": 32000, "total_batches": 1000},
    # Test 1: 1 col, 10B, 32K batch, 10K batches, 320M rows, ~3 GB
    {"columns": 1, "data_size": 10, "batch_size": 32000, "total_batches": 10000},
    # Test 2: 1 col, 100B, 16K batch, 1K batches, 16M rows, ~1.5 GB
    {"columns": 1, "data_size": 100, "batch_size": 16000, "total_batches": 1000},
    # Test 3: 1 col, 100B, 16K batch, 10K batches, 160M rows, ~15 GB
    {"columns": 1, "data_size": 100, "batch_size": 16000, "total_batches": 10000},
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


def format_bytes(bytes_value):
    """Format bytes into human readable format"""
    if bytes_value >= 1024**4:
        return f"{bytes_value / (1024**4):.1f}TB"
    elif bytes_value >= 1024**3:
        return f"{bytes_value / (1024**3):.1f}GB"
    elif bytes_value >= 1024**2:
        return f"{bytes_value / (1024**2):.1f}MB"
    elif bytes_value >= 1024:
        return f"{bytes_value / 1024:.1f}KB"
    else:
        return f"{bytes_value}B"


def calculate_optimal_batch_size(row_size, max_batch_memory=1024**3):
    """Calculate optimal batch size to stay under memory limit (default 1GB)"""
    if row_size <= 0:
        return 10000
    
    max_rows_per_batch = max_batch_memory // row_size
    
    # Use nice round numbers for batch sizes
    if max_rows_per_batch >= 1000000:
        return 1000000
    elif max_rows_per_batch >= 100000:
        return (max_rows_per_batch // 100000) * 100000
    elif max_rows_per_batch >= 10000:
        return (max_rows_per_batch // 10000) * 10000
    elif max_rows_per_batch >= 1000:
        return (max_rows_per_batch // 1000) * 1000
    elif max_rows_per_batch >= 100:
        return (max_rows_per_batch // 100) * 100
    else:
        return max(1, max_rows_per_batch)


def check_dataset_status(dataset_path, target_rows):
    """Check current dataset status and return existing row count."""
    try:
        # Add storage options for S3 paths
        if dataset_path.startswith('s3://'):
            dataset = lance.dataset(dataset_path, storage_options={
                "timeout": "60s",
                "connect_timeout": "30s",
                "aws_region": "us-west-2"
            })
        else:
            dataset = lance.dataset(dataset_path)
        current_rows = dataset.count_rows()
        return current_rows
    except Exception:
        return 0


def create_rows_worker(worker_id, dataset_idx, dataset_path, start_row, end_row, num_cols, col_size_bytes, 
                      write_batch_size, result_queue):
    """Worker process to create/append rows to dataset."""
    try:
        # Re-import in worker process (required for spawn)
        import lance
        import numpy as np
        import pyarrow as pa
        import time
        
        rows_to_write = end_row - start_row
        row_size = num_cols * col_size_bytes
        
        print(f"[Dataset {dataset_idx} Worker {worker_id}] Starting: rows {start_row:,} to {end_row:,} ({rows_to_write:,} rows)")
        
        start_time = time.time()
        rows_written = 0
        
        # For S3, smaller batches often work better
        if dataset_path.startswith('s3://'):
            write_batch_size = min(write_batch_size, 50000)  # Cap at 50k for S3
        
        while rows_written < rows_to_write:
            batch_rows = min(write_batch_size, rows_to_write - rows_written)
            current_start = start_row + rows_written
            
            # Create batch data
            data = {'id': np.arange(current_start, current_start + batch_rows, dtype=np.int64)}
            
            for i in range(num_cols):
                if col_size_bytes <= 8:
                    data[f'col_{i}'] = np.random.randint(0, 1000, size=batch_rows, dtype=np.int32)
                else:
                    data[f'col_{i}'] = [np.random.bytes(col_size_bytes) for _ in range(batch_rows)]
            
            table = pa.Table.from_pydict(data)
            
            # Write to dataset with retries for S3
            max_retries = 3
            for retry in range(max_retries):
                try:
                    # Always append since dataset is pre-created
                    if dataset_path.startswith('s3://'):
                        lance.write_dataset(table, dataset_path, mode="append", storage_options={
                            "timeout": "60s",
                            "connect_timeout": "30s",
                            "aws_region": "us-west-2"
                        })
                    else:
                        lance.write_dataset(table, dataset_path, mode="append")
                    break  # Success, exit retry loop
                except Exception as e:
                    if retry < max_retries - 1:
                        print(f"[Dataset {dataset_idx} Worker {worker_id}] Write failed, retrying... ({retry + 1}/{max_retries})")
                        time.sleep(2 ** retry)  # Exponential backoff
                    else:
                        raise e
            
            rows_written += batch_rows
            
            # Progress update - less frequent for concurrent mode
            if rows_to_write < 10000:
                progress_interval = rows_to_write
            elif rows_to_write < 100000:
                progress_interval = rows_to_write // 5
            else:
                progress_interval = rows_to_write // 10
            
            if rows_written % progress_interval == 0 or rows_written == rows_to_write:
                elapsed = time.time() - start_time
                if elapsed > 0:
                    rate_mb_s = (rows_written * row_size) / (1024**2) / elapsed
                    print(f"[Dataset {dataset_idx} Worker {worker_id}] Progress: {rows_written:,}/{rows_to_write:,} "
                          f"({rows_written/rows_to_write*100:.0f}%) | {rate_mb_s:.1f} MB/s")
        
        total_time = time.time() - start_time
        avg_rate_mb_s = (rows_written * row_size) / (1024**2) / total_time if total_time > 0 else 0
        
        print(f"[Dataset {dataset_idx} Worker {worker_id}] Completed: {rows_written:,} rows in {total_time:.1f}s | {avg_rate_mb_s:.1f} MB/s")
        result_queue.put((dataset_idx, worker_id, rows_written, total_time))
        
    except Exception as e:
        print(f"[Dataset {dataset_idx} Worker {worker_id}] Error: {e}")
        result_queue.put((dataset_idx, worker_id, 0, 0))


def create_dataset_initial(dataset_path, config):
    """Create dataset with initial row if it doesn't exist."""
    try:
        # Create initial dataset with 1 row
        data = {'id': np.array([0], dtype=np.int64)}
        for i in range(config['columns']):
            if config['data_size'] <= 8:
                data[f'col_{i}'] = np.array([np.random.randint(0, 1000)], dtype=np.int32)
            else:
                data[f'col_{i}'] = [np.random.bytes(config['data_size'])]
        
        table = pa.Table.from_pydict(data)
        if dataset_path.startswith('s3://'):
            lance.write_dataset(table, dataset_path, mode="create", storage_options={
                "timeout": "60s",
                "connect_timeout": "30s",
                "aws_region": "us-west-2"
            })
        else:
            lance.write_dataset(table, dataset_path, mode="create")
        return True
    except Exception as e:
        print(f"Failed to create initial dataset at {dataset_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Concurrent dataset creation for S3")
    parser.add_argument("--test-indices", type=int, nargs='+',
                        help="Specific test indices to process (0-based). If not specified, processes all.")
    parser.add_argument("--skip-indices", type=int, nargs='+',
                        help="Test indices to skip (0-based).")
    parser.add_argument("--data-dir", type=str, default="./lance_data",
                        help="Directory to store Lance datasets (supports s3:// paths)")
    # Removed --force-recreate option for safety
    parser.add_argument("--total-workers", type=int, default=None,
                        help="Total number of workers to distribute among datasets (default: CPU count)")
    args = parser.parse_args()
    
    # Create data directory if it doesn't exist (only for local paths)
    if not args.data_dir.startswith('s3://'):
        os.makedirs(args.data_dir, exist_ok=True)
    
    # Determine which configs to process
    if args.test_indices:
        configs_to_process = [(i, TEST_CONFIGS[i]) for i in args.test_indices if 0 <= i < len(TEST_CONFIGS)]
    else:
        configs_to_process = list(enumerate(TEST_CONFIGS))
    
    # Apply skip filter if specified
    if args.skip_indices:
        skip_set = set(args.skip_indices)
        configs_to_process = [(i, cfg) for i, cfg in configs_to_process if i not in skip_set]
        print(f"Skipping test indices: {sorted(args.skip_indices)}")
    
    print(f"Concurrent Dataset Creation for S3")
    print(f"Data directory: {args.data_dir}")
    print(f"Datasets to process: {len(configs_to_process)}")
    
    # Calculate remaining work for all datasets
    print("\nCalculating work distribution...")
    total_workers = args.total_workers if args.total_workers else cpu_count()
    dataset_info = []
    total_remaining_rows = 0
    
    for idx, config in configs_to_process:
        table_name = f"col{config['columns']}_data{config['data_size']}_batch{config['total_batches']}_perbatch{config['batch_size']}"
        dataset_path = f"{args.data_dir}/{table_name}.lance"
        target_rows = config['batch_size'] * config['total_batches']
        current_rows = check_dataset_status(dataset_path, target_rows)
        
        remaining_rows = target_rows - current_rows
        
        if remaining_rows > 0:
            dataset_info.append({
                'idx': idx,
                'config': config,
                'dataset_path': dataset_path,
                'target_rows': target_rows,
                'current_rows': current_rows,
                'remaining_rows': remaining_rows,
                'row_size': config['columns'] * config['data_size']
            })
            total_remaining_rows += remaining_rows
    
    if not dataset_info:
        print("No datasets need processing.")
        return
    
    print(f"Total workers available: {total_workers}")
    print(f"Datasets with work: {len(dataset_info)}")
    print(f"Total remaining rows: {total_remaining_rows:,}")
    
    # Distribute workers proportionally
    for info in dataset_info:
        proportion = info['remaining_rows'] / total_remaining_rows
        allocated_workers = max(1, int(total_workers * proportion))
        
        # Memory safety limits
        if info['row_size'] >= 10 * 1024 * 1024:  # 10MB+ rows
            allocated_workers = min(4, allocated_workers)
        elif info['row_size'] >= 1 * 1024 * 1024:  # 1MB+ rows
            allocated_workers = min(8, allocated_workers)
        
        # No S3 limit - use all allocated workers
        # if args.data_dir.startswith('s3://'):
        #     allocated_workers = min(allocated_workers, 128)  # Removed limit
        
        info['workers'] = allocated_workers
    
    # Print allocation
    print(f"\nWorker allocation (from {total_workers} total workers):")
    for info in dataset_info:
        proportion = info['remaining_rows'] / total_remaining_rows * 100
        print(f"  Test {info['idx']}: {info['workers']} workers for {info['remaining_rows']:,} rows ({proportion:.1f}% of work)")
    
    # Pre-create all datasets that don't exist
    print("\nPre-creating datasets...")
    for info in dataset_info:
        if info['current_rows'] == 0:
            print(f"Creating dataset for test {info['idx']}...")
            if create_dataset_initial(info['dataset_path'], info['config']):
                info['current_rows'] = 1
                info['remaining_rows'] = info['target_rows'] - 1
    
    # Launch all workers concurrently
    print("\nLaunching all workers concurrently...")
    all_processes = []
    result_queue = Queue()
    
    for info in dataset_info:
        if info['remaining_rows'] <= 0:
            continue
            
        write_batch_size = calculate_optimal_batch_size(info['row_size'])
        rows_per_worker = info['remaining_rows'] // info['workers']
        extra_rows = info['remaining_rows'] % info['workers']
        
        # Launch workers for this dataset
        for worker_id in range(info['workers']):
            start_row = info['current_rows'] + (worker_id * rows_per_worker)
            if worker_id == info['workers'] - 1:
                end_row = info['target_rows']  # Last worker gets extra rows
            else:
                end_row = start_row + rows_per_worker
            
            p = Process(target=create_rows_worker,
                       args=(worker_id, info['idx'], info['dataset_path'], start_row, end_row,
                             info['config']['columns'], info['config']['data_size'], 
                             write_batch_size, result_queue))
            p.start()
            all_processes.append((info['idx'], p))
    
    print(f"\nLaunched {len(all_processes)} worker processes across {len(dataset_info)} datasets")
    print("All datasets are being processed concurrently...")
    
    # Wait for all processes to complete
    start_time = time.time()
    for idx, p in all_processes:
        p.join()
    
    total_time = time.time() - start_time
    
    # Collect results
    results_by_dataset = {}
    while not result_queue.empty():
        dataset_idx, worker_id, rows_written, time_taken = result_queue.get()
        if dataset_idx not in results_by_dataset:
            results_by_dataset[dataset_idx] = {'rows': 0, 'max_time': 0}
        results_by_dataset[dataset_idx]['rows'] += rows_written
        results_by_dataset[dataset_idx]['max_time'] = max(results_by_dataset[dataset_idx]['max_time'], time_taken)
    
    # Print summary
    print(f"\n{'='*80}")
    print("Summary:")
    for info in dataset_info:
        idx = info['idx']
        if idx in results_by_dataset:
            final_rows = check_dataset_status(info['dataset_path'], info['target_rows'])
            print(f"  Test {idx}: {final_rows:,}/{info['target_rows']:,} rows "
                  f"({'✓' if final_rows >= info['target_rows'] else '✗'})")
    
    print(f"\nTotal processing time: {total_time:.1f}s")
    print("All datasets processed concurrently!")


if __name__ == "__main__":
    main()