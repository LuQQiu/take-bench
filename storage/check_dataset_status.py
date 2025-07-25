#!/usr/bin/env python3
"""
Check the creation status of Lance datasets.
Shows completed/incomplete status, written rows, and remaining rows.
"""

import lance
import os
import argparse
from tabulate import tabulate

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


def format_number(num):
    """Format large numbers with commas and units"""
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return str(num)


def check_dataset_status(dataset_path, target_rows):
    """Check dataset status and return current row count"""
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
        return dataset.count_rows()
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser(description="Check Lance dataset creation status")
    parser.add_argument("--data-dir", type=str, default="./lance_data",
                        help="Directory containing Lance datasets")
    parser.add_argument("--test-indices", type=int, nargs='+',
                        help="Specific test indices to check (0-based). If not specified, checks all.")
    parser.add_argument("--incomplete-only", action="store_true",
                        help="Show only incomplete datasets")
    parser.add_argument("--summary", action="store_true",
                        help="Show summary statistics only")
    args = parser.parse_args()
    
    # Determine which configs to check
    if args.test_indices:
        configs_to_check = [(i, TEST_CONFIGS[i]) for i in args.test_indices if 0 <= i < len(TEST_CONFIGS)]
    else:
        configs_to_check = list(enumerate(TEST_CONFIGS))
    
    # Collect status information
    status_data = []
    total_expected = 0
    total_written = 0
    total_remaining = 0
    complete_count = 0
    incomplete_count = 0
    not_started_count = 0
    
    for idx, config in configs_to_check:
        # Calculate dataset properties
        table_name = f"col{config['columns']}_data{config['data_size']}_batch{config['total_batches']}_perbatch{config['batch_size']}"
        dataset_path = f"{args.data_dir}/{table_name}.lance"
        
        target_rows = config['batch_size'] * config['total_batches']
        row_size = config['columns'] * config['data_size']
        total_size = target_rows * row_size
        
        # Check current status
        current_rows = check_dataset_status(dataset_path, target_rows)
        remaining_rows = target_rows - current_rows
        progress_pct = (current_rows / target_rows * 100) if target_rows > 0 else 0
        
        # Determine status
        if current_rows == 0:
            status = "Not Started"
            not_started_count += 1
        elif current_rows >= target_rows:
            status = "Complete"
            complete_count += 1
        else:
            status = "In Progress"
            incomplete_count += 1
        
        # Update totals
        total_expected += target_rows
        total_written += current_rows
        total_remaining += remaining_rows
        
        # Skip if showing incomplete only and this is complete
        if args.incomplete_only and status == "Complete":
            continue
        
        status_data.append([
            idx,
            f"{config['columns']}",
            format_bytes(config['data_size']),
            format_bytes(row_size),
            format_number(target_rows),
            format_number(current_rows),
            format_number(remaining_rows),
            f"{progress_pct:.1f}%",
            format_bytes(total_size),
            status
        ])
    
    if not args.summary:
        # Print detailed table
        headers = ["Test", "Cols", "Col Size", "Row Size", "Target Rows", 
                   "Written Rows", "Remaining", "Progress", "Total Size", "Status"]
        print(f"\nDataset Status Report")
        print(f"Data directory: {args.data_dir}")
        print(f"{'='*120}")
        print(tabulate(status_data, headers=headers, tablefmt="simple"))
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total datasets: {len(configs_to_check)}")
    print(f"  Complete: {complete_count} ({complete_count/len(configs_to_check)*100:.1f}%)")
    print(f"  In Progress: {incomplete_count} ({incomplete_count/len(configs_to_check)*100:.1f}%)")
    print(f"  Not Started: {not_started_count} ({not_started_count/len(configs_to_check)*100:.1f}%)")
    print(f"\n  Total expected rows: {format_number(total_expected)}")
    print(f"  Total written rows: {format_number(total_written)}")
    print(f"  Total remaining rows: {format_number(total_remaining)}")
    print(f"  Overall progress: {(total_written/total_expected*100) if total_expected > 0 else 0:.1f}%")
    
    # List incomplete datasets for easy reference
    if incomplete_count > 0 and not args.summary:
        print(f"\nIncomplete datasets (test indices):")
        incomplete_indices = []
        for idx, config in configs_to_check:
            target_rows = config['batch_size'] * config['total_batches']
            table_name = f"col{config['columns']}_data{config['data_size']}_batch{config['total_batches']}_perbatch{config['batch_size']}"
            dataset_path = f"{args.data_dir}/{table_name}.lance"
            current_rows = check_dataset_status(dataset_path, target_rows)
            if 0 < current_rows < target_rows:
                incomplete_indices.append(str(idx))
        print(f"  {' '.join(incomplete_indices)}")
        
        print(f"\nTo resume incomplete datasets:")
        print(f"  python storage/smart_parallel_create.py --test-indices {' '.join(incomplete_indices)}")


if __name__ == "__main__":
    main()