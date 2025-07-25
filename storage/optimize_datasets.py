#!/usr/bin/env python3
"""
Optimize Lance datasets in parallel.
Each process handles one dataset's optimization (compact, cleanup, optimize indices).
"""

import lance
import datetime
import os
import time
from multiprocessing import Process, cpu_count, Queue, set_start_method
import argparse

# Force spawn method for multiprocessing (required for lance)
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


def cleanup_worker(idx, config, dataset_path, result_queue):
    """Worker process to cleanup old versions of a dataset."""
    try:
        # Re-import in worker process (required for spawn)
        import lance
        import datetime
        import time
        
        table_name = f"col{config['columns']}_data{config['data_size']}_batch{config['total_batches']}_perbatch{config['batch_size']}"
        start_time = time.time()
        
        print(f"[Test {idx}] Starting cleanup for {table_name}")
        
        # Open dataset
        try:
            if dataset_path.startswith('s3://'):
                dataset = lance.dataset(dataset_path, storage_options={
                    "timeout": "60s",
                    "connect_timeout": "30s",
                    "aws_region": "us-west-2"
                })
            else:
                dataset = lance.dataset(dataset_path)
            row_count = dataset.count_rows()
            print(f"[Test {idx}] Dataset has {row_count:,} rows")
        except Exception as e:
            print(f"[Test {idx}] Failed to open dataset: {e}")
            result_queue.put((idx, False, 0, str(e)))
            return
        
        # Cleanup old versions
        try:
            # Use 0 seconds to cleanup all old versions
            cleanup_time = datetime.timedelta(seconds=0)
            print(f"[Test {idx}] Cleaning up all old versions...")
            
            # Use cleanup_old_versions with timedelta
            dataset.cleanup_old_versions(older_than=cleanup_time)
            
            total_time = time.time() - start_time
            print(f"[Test {idx}] ✓ Cleanup completed in {total_time:.1f}s")
            
            result_queue.put((idx, True, total_time, "Cleanup completed"))
            
        except Exception as e:
            print(f"[Test {idx}] ✗ Cleanup failed: {e}")
            result_queue.put((idx, False, 0, str(e)))
            
    except Exception as e:
        print(f"[Test {idx}] ✗ Cleanup failed: {e}")
        result_queue.put((idx, False, 0, str(e)))


def optimize_dataset_worker(idx, config, dataset_path, result_queue):
    """Worker process to optimize a single dataset."""
    try:
        # Re-import in worker process (required for spawn)
        import lance
        import datetime
        import time
        
        table_name = f"col{config['columns']}_data{config['data_size']}_batch{config['total_batches']}_perbatch{config['batch_size']}"
        start_time = time.time()
        
        print(f"[Test {idx}] Starting optimization for {table_name}")
        
        # Open dataset
        try:
            if dataset_path.startswith('s3://'):
                dataset = lance.dataset(dataset_path, storage_options={
                    "timeout": "60s",
                    "connect_timeout": "30s",
                    "aws_region": "us-west-2"
                })
            else:
                dataset = lance.dataset(dataset_path)
            row_count = dataset.count_rows()
            print(f"[Test {idx}] Dataset has {row_count:,} rows")
        except Exception as e:
            print(f"[Test {idx}] Failed to open dataset: {e}")
            result_queue.put((idx, False, 0, str(e)))
            return
        
        # Step 1: Compact files with appropriate settings based on row size
        try:
            print(f"[Test {idx}] Compacting files...")
            compact_start = time.time()
            
            # Calculate row size
            row_size = config['columns'] * config['data_size']
            
            # Adjust compaction parameters based on row size
            if row_size >= 100 * 1024 * 1024:  # 100MB+ rows
                # For 100MB rows, use very small fragments
                print(f"[Test {idx}] Using small fragment size for 100MB rows ({format_bytes(row_size)}/row)")
                dataset.optimize.compact_files(
                    target_rows_per_fragment=1000,  # Only 1K rows per fragment
                    max_rows_per_group=100,  # Small groups
                    max_bytes_per_file=1024 * 1024 * 1024  # 1GB max file size
                )
            elif row_size >= 10 * 1024 * 1024:  # 10MB rows
                # For 10MB rows (Test 9), use 10K rows per fragment
                print(f"[Test {idx}] Using 10K fragment size for 10MB rows ({format_bytes(row_size)}/row)")
                dataset.optimize.compact_files(
                    target_rows_per_fragment=10000,  # 10K rows per fragment (100GB per fragment)
                    max_rows_per_group=100,  # Small groups
                    max_bytes_per_file=1024 * 1024 * 1024  # 1GB max file size
                )
            elif row_size >= 1 * 1024 * 1024:  # 1MB rows
                # For 1MB rows (Test 8), use 10K rows per fragment for 10GB fragments
                print(f"[Test {idx}] Using 10K fragment size for 1MB rows ({format_bytes(row_size)}/row)")
                dataset.optimize.compact_files(
                    target_rows_per_fragment=10000,  # 10K rows per fragment (10GB per fragment)
                    max_rows_per_group=100,  # Small groups like 10MB/100MB
                    max_bytes_per_file=1024 * 1024 * 1024  # 1GB max file size
                )
            elif row_size >= 100 * 1024:  # 100KB+ rows
                # For medium-large rows
                print(f"[Test {idx}] Using standard fragment size for medium rows ({format_bytes(row_size)}/row)")
                dataset.optimize.compact_files(
                    target_rows_per_fragment=100000,  # 100K rows per fragment
                    max_rows_per_group=512,
                    max_bytes_per_file=2 * 1024 * 1024 * 1024  # 2GB max file size
                )
            else:
                # For small rows, use default settings
                dataset.optimize.compact_files()
            
            compact_time = time.time() - compact_start
            print(f"[Test {idx}] Compaction completed in {compact_time:.1f}s")
        except Exception as e:
            print(f"[Test {idx}] Compaction failed: {e}")
        
        
        total_time = time.time() - start_time
        print(f"[Test {idx}] ✓ Optimization completed in {total_time:.1f}s")
        
        result_queue.put((idx, True, total_time, "Success"))
        
    except Exception as e:
        print(f"[Test {idx}] ✗ Optimization failed: {e}")
        result_queue.put((idx, False, 0, str(e)))


def main():
    parser = argparse.ArgumentParser(description="Optimize Lance datasets in parallel")
    parser.add_argument("--data-dir", type=str, default="./lance_data",
                        help="Directory containing Lance datasets (supports s3:// paths)")
    parser.add_argument("--test-indices", type=int, nargs='+',
                        help="Specific test indices to optimize (0-based). If not specified, optimizes all.")
    parser.add_argument("--skip-indices", type=int, nargs='+',
                        help="Test indices to skip (0-based).")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Maximum number of parallel workers (default: CPU count)")
    parser.add_argument("--completed-only", action="store_true",
                        help="Only optimize datasets that are fully completed")
    parser.add_argument("--reverse", action="store_true",
                        help="Process datasets in reverse order (from highest index to lowest)")
    parser.add_argument("--cleanup-old-versions", action="store_true",
                        help="Cleanup all old versions instead of optimizing")
    args = parser.parse_args()
    
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
    
    if args.cleanup_old_versions:
        print(f"Lance Dataset Cleanup (All Old Versions)")
    else:
        print(f"Lance Dataset Optimization (Compaction Only)")
    
    print(f"Data directory: {args.data_dir}")
    print(f"Datasets to process: {len(configs_to_process)}")
    print(f"Max workers: {args.max_workers or cpu_count()}")
    if args.completed_only:
        print(f"Mode: Completed datasets only")
    print()
    
    # Check which datasets exist
    datasets_to_optimize = []
    for idx, config in configs_to_process:
        table_name = f"col{config['columns']}_data{config['data_size']}_batch{config['total_batches']}_perbatch{config['batch_size']}"
        dataset_path = f"{args.data_dir}/{table_name}.lance"
        
        try:
            # Check if dataset exists
            if dataset_path.startswith('s3://'):
                dataset = lance.dataset(dataset_path, storage_options={
                    "timeout": "60s",
                    "connect_timeout": "30s",
                    "aws_region": "us-west-2"
                })
            else:
                dataset = lance.dataset(dataset_path)
            row_count = dataset.count_rows()
            target_rows = config['batch_size'] * config['total_batches']
            row_size = config['columns'] * config['data_size']
            total_size = target_rows * row_size
            
            # Check if we should include this dataset
            include_dataset = True
            
            # No longer skip large row datasets - we handle them with special parameters
            if args.completed_only:
                # Only include if dataset is complete
                if row_count < target_rows:
                    include_dataset = False
                    print(f"Test {idx}: {table_name} - {row_count:,}/{target_rows:,} rows (incomplete, skipping)")
                else:
                    print(f"Test {idx}: {table_name} - {row_count:,} rows, {format_bytes(total_size)} (complete)")
            else:
                print(f"Test {idx}: {table_name} - {row_count:,} rows, {format_bytes(total_size)}")
            
            if include_dataset:
                datasets_to_optimize.append({
                    'idx': idx,
                    'config': config,
                    'path': dataset_path,
                    'rows': row_count,
                    'size': total_size
                })
        except Exception as e:
            print(f"Test {idx}: {table_name} - Not found or inaccessible")
    
    if not datasets_to_optimize:
        print("\nNo datasets found to optimize.")
        return
    
    print(f"\nFound {len(datasets_to_optimize)} datasets to optimize")
    
    # Sort datasets based on reverse flag
    if args.reverse:
        datasets_to_optimize.sort(key=lambda x: x['idx'], reverse=True)
        print("Processing in reverse order (highest index first)")
    else:
        datasets_to_optimize.sort(key=lambda x: x['idx'])
        print("Processing in forward order (lowest index first)")
    
    # Process datasets sequentially
    result_queue = Queue()
    
    if args.cleanup_old_versions:
        print(f"\nCleaning up old versions sequentially...")
    else:
        print(f"\nOptimizing datasets sequentially...")
    start_time = time.time()
    
    # Process one at a time
    for dataset_info in datasets_to_optimize:
        if args.cleanup_old_versions:
            p = Process(target=cleanup_worker,
                       args=(dataset_info['idx'], dataset_info['config'], 
                             dataset_info['path'], result_queue))
        else:
            p = Process(target=optimize_dataset_worker,
                       args=(dataset_info['idx'], dataset_info['config'], 
                             dataset_info['path'], result_queue))
        p.start()
        p.join()  # Wait for completion before starting next
    
    total_time = time.time() - start_time
    
    # Collect results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    
    # Print summary
    print(f"\n{'='*60}")
    print("Optimization Summary:")
    print(f"{'='*60}")
    
    successful = 0
    failed = 0
    
    for idx, success, duration, message in sorted(results):
        status = "✓" if success else "✗"
        if success:
            print(f"{status} Test {idx}: Optimized in {duration:.1f}s")
            successful += 1
        else:
            print(f"{status} Test {idx}: Failed - {message}")
            failed += 1
    
    print(f"\nTotal: {successful} successful, {failed} failed")
    print(f"Total time: {total_time:.1f}s")


if __name__ == "__main__":
    main()