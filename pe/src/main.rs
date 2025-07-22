use chrono::Utc;
use clap::Parser;
use csv::Writer;
use futures::stream::TryStreamExt;
use lance::dataset::{Dataset, ProjectionRequest};
use lance::datatypes::{Projection, Projectable};
use execution_plans::remote::{PlanExecutorClient, TakeRowIndices};
use plan_executor::server::build_in_process;
use rand::seq::SliceRandom;
use serde::Serialize;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::task::JoinSet;
use tracing::{info, warn};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "./lance_data")]
    data_dir: String,

    #[arg(long, default_value_t = 4)]
    num_workers: usize,

    #[arg(long, default_value = "benchmark_results.csv")]
    output_csv: String,
}

#[derive(Clone, Debug)]
struct TestConfig {
    columns: usize,
    data_size: usize,
    batch_size: usize,
    total_batches: usize,
    prefetch_factor: usize,
}

#[derive(Serialize)]
struct BenchmarkResult {
    timestamp: String,
    test_approach: String,
    columns: usize,
    data_size: usize,
    row_size: usize,
    batch_size: usize,
    total_batches: usize,
    total_rows: usize,
    total_dataset_size_mb: f64,
    data_loader_num_workers: usize,
    data_loader_prefetch_factor: usize,
    qps: f64,
    throughput_mb_s: f64,
    p50_latency_ms: f64,
    p90_latency_ms: f64,
    p95_latency_ms: f64,
    p99_latency_ms: f64,
}

fn get_test_configs() -> Vec<TestConfig> {
    vec![
        TestConfig {
            columns: 1,
            data_size: 10,
            batch_size: 3200,
            total_batches: 100,
            prefetch_factor: 2,
        },
        // Add more configurations here...
    ]
}

fn generate_table_name(config: &TestConfig) -> String {
    format!(
        "col{}_data{}_batch{}_perbatch{}",
        config.columns, config.data_size, config.total_batches, config.batch_size
    )
}

async fn shuffle_and_distribute_indices(
    total_rows: usize,
    num_workers: usize,
) -> Vec<Vec<u64>> {
    let mut indices: Vec<u64> = (0..total_rows as u64).collect();
    let mut rng = rand::thread_rng();
    indices.shuffle(&mut rng);

    let chunk_size = (total_rows + num_workers - 1) / num_workers;
    let mut worker_indices = Vec::new();

    for i in 0..num_workers {
        let start = i * chunk_size;
        let end = ((i + 1) * chunk_size).min(total_rows);
        if start < total_rows {
            let mut chunk: Vec<u64> = indices[start..end].to_vec();
            chunk.sort_unstable();
            worker_indices.push(chunk);
        }
    }

    worker_indices
}

async fn run_worker(
    worker_id: usize,
    indices: Vec<u64>,
    dataset_path: String,
    batch_size: usize,
    _columns: Option<Vec<String>>,
    storage_options: HashMap<String, String>,
    plan_executor: Arc<dyn PlanExecutorClient>,
    version: u64,
    projection: Arc<Projection>,
) -> Result<Vec<Duration>, Box<dyn std::error::Error + Send + Sync>> {
    let mut batch_times = Vec::new();

    for chunk in indices.chunks(batch_size) {
        let batch_start = Instant::now();
        
        let take_params = TakeRowIndices {
            table_uri: dataset_path.clone(),
            version,
            storage_options: Some(storage_options.clone()),
            row_indices: chunk.to_vec(),
            projection: projection.clone(),
        };

        let exec_stream = plan_executor.take_row_indices(&take_params).await?;
        let _batches: Vec<arrow_array::RecordBatch> = exec_stream.try_collect().await?;
        
        let batch_time = batch_start.elapsed();
        batch_times.push(batch_time);

        if (batch_times.len() % 10) == 0 {
            info!(
                "Worker {} progress: {}/{} batches",
                worker_id,
                batch_times.len(),
                (indices.len() + batch_size - 1) / batch_size
            );
        }
    }

    Ok(batch_times)
}

async fn run_benchmark(
    dataset_path: &str,
    config: &TestConfig,
    num_workers: usize,
) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
    let total_rows = config.batch_size * config.total_batches;
    
    info!(
        "Running benchmark: {} rows, {} workers, batch size {}",
        total_rows, num_workers, config.batch_size
    );

    // Create plan executor client
    let plan_executor = build_in_process().await?;

    // Open dataset to get version and schema
    let dataset = Dataset::open(dataset_path).await?;
    let version = dataset.version().version;
    let schema = dataset.schema();

    // Create projection for all columns
    let projection_request = ProjectionRequest::from_schema(schema.clone());
    let projection_plan = Arc::new(projection_request.into_projection_plan(&schema)?);
    let projection = Arc::new(
        Projection::empty(Arc::new(schema.clone()) as Arc<dyn Projectable>)
            .union_schema(&projection_plan.physical_schema)
    );

    // Warmup
    {
        let warmup_indices = vec![0, 1, 2, 3, 4];
        
        let take_params = TakeRowIndices {
            table_uri: dataset_path.to_string(),
            version,
            storage_options: None,
            row_indices: warmup_indices,
            projection: projection.clone(),
        };
        
        let _ = plan_executor.take_row_indices(&take_params).await?;
    }

    // Distribute indices among workers
    let worker_indices = shuffle_and_distribute_indices(total_rows, num_workers).await;
    
    let start_time = Instant::now();
    let mut join_set = JoinSet::new();

    // Spawn workers
    for (worker_id, indices) in worker_indices.into_iter().enumerate() {
        let dataset_path = dataset_path.to_string();
        let config = config.clone();
        let plan_executor = plan_executor.clone();
        let projection = projection.clone();
        
        join_set.spawn(async move {
            run_worker(
                worker_id,
                indices,
                dataset_path,
                config.batch_size,
                None, // No column projection for now
                HashMap::new(), // No storage options for local
                plan_executor,
                version,
                projection,
            )
            .await
        });
    }

    // Collect results
    let mut all_batch_times = Vec::new();
    while let Some(result) = join_set.join_next().await {
        match result {
            Ok(Ok(batch_times)) => all_batch_times.extend(batch_times),
            Ok(Err(e)) => warn!("Worker error: {}", e),
            Err(e) => warn!("Join error: {}", e),
        }
    }

    let total_time = start_time.elapsed();
    let total_batches_processed = all_batch_times.len();
    
    // Calculate metrics
    let mut latencies_ms: Vec<f64> = all_batch_times
        .iter()
        .map(|d| d.as_secs_f64() * 1000.0)
        .collect();
    latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let p50 = percentile(&latencies_ms, 50.0);
    let p90 = percentile(&latencies_ms, 90.0);
    let p95 = percentile(&latencies_ms, 95.0);
    let p99 = percentile(&latencies_ms, 99.0);

    let qps = total_batches_processed as f64 / total_time.as_secs_f64();
    let rows_per_second = total_rows as f64 / total_time.as_secs_f64();
    let row_size = config.columns * config.data_size;
    let throughput_mb_s = (rows_per_second * row_size as f64) / (1024.0 * 1024.0);
    let total_dataset_size_mb = (total_rows * row_size) as f64 / (1024.0 * 1024.0);

    Ok(BenchmarkResult {
        timestamp: Utc::now().to_rfc3339(),
        test_approach: "PE".to_string(),
        columns: config.columns,
        data_size: config.data_size,
        row_size,
        batch_size: config.batch_size,
        total_batches: config.total_batches,
        total_rows,
        total_dataset_size_mb,
        data_loader_num_workers: num_workers,
        data_loader_prefetch_factor: config.prefetch_factor,
        qps,
        throughput_mb_s,
        p50_latency_ms: p50,
        p90_latency_ms: p90,
        p95_latency_ms: p95,
        p99_latency_ms: p99,
    })
}

fn percentile(sorted_values: &[f64], p: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    let index = ((p / 100.0) * (sorted_values.len() - 1) as f64) as usize;
    sorted_values[index]
}

async fn write_results_to_csv(
    result: &BenchmarkResult,
    csv_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let file_exists = Path::new(csv_path).exists();
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(csv_path)?;

    let mut writer = Writer::from_writer(file);

    if !file_exists {
        writer.write_record(&[
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
            "P99 Latency (ms)",
        ])?;
    }

    writer.write_record(&[
        &result.timestamp,
        &result.test_approach,
        &result.columns.to_string(),
        &result.data_size.to_string(),
        &result.row_size.to_string(),
        &result.batch_size.to_string(),
        &result.total_batches.to_string(),
        &result.total_rows.to_string(),
        &format!("{:.2}", result.total_dataset_size_mb),
        &result.data_loader_num_workers.to_string(),
        &result.data_loader_prefetch_factor.to_string(),
        &format!("{:.2}", result.qps),
        &format!("{:.2}", result.throughput_mb_s),
        &format!("{:.2}", result.p50_latency_ms),
        &format!("{:.2}", result.p90_latency_ms),
        &format!("{:.2}", result.p95_latency_ms),
        &format!("{:.2}", result.p99_latency_ms),
    ])?;

    writer.flush()?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    let test_configs = get_test_configs();

    info!(
        "Running {} tests with {} workers",
        test_configs.len(),
        args.num_workers
    );
    info!("Data directory: {}", args.data_dir);
    info!("Results will be saved to: {}", args.output_csv);

    for (i, config) in test_configs.iter().enumerate() {
        info!("\n{}", "=".repeat(60));
        info!("Test {}/{}", i + 1, test_configs.len());
        info!("{}", "=".repeat(60));

        let table_name = generate_table_name(config);
        let dataset_path = format!("{}/{}.lance", args.data_dir, table_name);

        info!("Running: {}", table_name);
        info!("  Columns: {}", config.columns);
        info!("  Data size: {} bytes", config.data_size);
        info!("  Row size: {} bytes", config.columns * config.data_size);
        info!("  Batch size: {}", config.batch_size);
        info!("  Total batches: {}", config.total_batches);
        info!("  Total rows: {}", config.batch_size * config.total_batches);
        info!(
            "  Total dataset size: {:.2} MB",
            (config.batch_size * config.total_batches * config.columns * config.data_size) as f64
                / (1024.0 * 1024.0)
        );

        // Check if dataset exists
        if !Path::new(&dataset_path).exists() {
            warn!(
                "Dataset {} does not exist. Please create it using the Python script first.",
                dataset_path
            );
            continue;
        }

        match run_benchmark(&dataset_path, config, args.num_workers).await {
            Ok(result) => {
                info!("\nResults:");
                info!("  QPS: {:.2}", result.qps);
                info!("  Throughput: {:.2} MB/s", result.throughput_mb_s);
                info!("  P50 latency: {:.2} ms", result.p50_latency_ms);
                info!("  P90 latency: {:.2} ms", result.p90_latency_ms);
                info!("  P99 latency: {:.2} ms", result.p99_latency_ms);

                write_results_to_csv(&result, &args.output_csv).await?;
            }
            Err(e) => {
                warn!("Benchmark failed: {}", e);
            }
        }
    }

    info!("\nAll results saved to: {}", args.output_csv);
    Ok(())
}