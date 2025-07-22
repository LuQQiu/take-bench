# Take Operation Benchmark

This benchmark tests the performance of the plan executor's `take_row_indices` operation with parallel workers, similar to the Python PyTorch DataLoader benchmark.

## Prerequisites

1. Create datasets using the Python script first:
```bash
python benchmark_script.py --data-dir ./lance_data
```

## Usage

Run the benchmark:
```bash
cd src/rust
cargo run --release --bin take-benchmark -- --num-workers 4 --data-dir ./lance_data
```

Options:
- `--num-workers`: Number of parallel workers (default: 4)
- `--data-dir`: Directory containing Lance datasets (default: ./lance_data)
- `--output-csv`: Output CSV file for results (default: benchmark_results.csv)

## How it works

1. **Parallel execution**: The benchmark distributes row indices across multiple workers
2. **Index shuffling**: Indices are shuffled randomly then sorted within each worker's chunk
3. **Non-overlapping**: Each worker gets a unique set of indices
4. **Metrics collection**: Tracks latency percentiles, QPS, and throughput

## Test Configurations

The benchmark runs the same test configurations as the Python script:
- Columns: 1
- Data size: 10 bytes
- Batch size: 3200
- Total batches: 100

Add more configurations in `get_test_configs()` function in `src/main.rs`.

## Output

Results are written to a CSV file with the following metrics:
- QPS (Queries per second)
- Throughput (MB/s)
- Latency percentiles (P50, P90, P95, P99)
- Configuration details