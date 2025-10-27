// In src/api/cli.rs

use crate::engine::{builder::EngineBuilder, engine::DataEngine, config::EngineConfig};
use chrono::Utc;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::time::Instant;

// --- 1. Define the Command-Line Structure ---

#[derive(Parser, Debug)]
#[command(author, version, about = "A modular data engine using the HIBC architecture.")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    /// Increase message verbosity.
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    pub verbose: u8,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Build a complete database from a source file.
    Build(BuildArgs),
    /// Perform a vector similarity search.
    Search(SearchArgs),
    /// Serve an HTTP API exposing search and document endpoints.
    Serve(ServeArgs),
    Upsert(UpsertArgs),  // NEW
    Delete(DeleteArgs),  // NEW
    Flush(FlushArgs),    // NEW
}

#[derive(Parser, Debug)]
pub struct BuildArgs {
    /// Path to the input JSONL file.
    #[arg(short, long)]
    pub input_file: PathBuf,
    /// The base path for the output database files.
    #[arg(short, long)]
    pub db_path: PathBuf,
    /// Path to EngineConfig JSON
    #[arg(long)]
    pub config: PathBuf,
}

#[derive(Parser, Debug)]
pub struct SearchArgs {
    /// The base path of the database to query.
    #[arg(short, long)]
    pub db_path: PathBuf,
    /// The query vector, as a JSON array string (e.g., "[0.1, 0.2, ...]").
    #[arg(long)]
    pub vector: String,
    /// The number of nearest neighbors to return.
    #[arg(short, long, default_value_t = 10)]
    pub top_k: usize,
}

#[derive(Parser, Debug)]
pub struct ServeArgs {
    /// The base path of the database to serve.
    #[arg(short, long)]
    pub db_path: PathBuf,
    /// Bind address (ip:port)
    #[arg(long, default_value = "0.0.0.0:3000")]
    pub bind: String,
}

#[derive(Parser, Debug)]
pub struct UpsertArgs {
    #[arg(short, long)] pub db_path: PathBuf,
    #[arg(long)] pub id: String,
    #[arg(long)] pub vector: String, // JSON array
    #[arg(long)] pub metadata: String, // JSON obj
}

#[derive(Parser, Debug)]
pub struct DeleteArgs {
    #[arg(short, long)] pub db_path: PathBuf,
    #[arg(long)] pub id: String,
}

#[derive(Parser, Debug)]
pub struct FlushArgs {
    #[arg(short, long)] pub db_path: PathBuf,
}

// --- 2. Implement the Handler Functions ---

fn now_ts_u64() -> u64 {
    // Protect against negative timestamps (unlikely, but keeps types clean)
    Utc::now().timestamp().max(0) as u64
}

/// Handles the `build` command.
pub fn handle_build(args: BuildArgs) -> anyhow::Result<()> {
    log::info!("Starting database build...");
    let start_time = Instant::now();

    let cfg: EngineConfig = serde_json::from_slice(&std::fs::read(&args.config)?)?;
    cfg.validate()?;
    let mut builder = EngineBuilder::new(&args.db_path, cfg)?;
    builder.build_from_jsonl(&args.input_file)?;
    builder.finalize()?;

    println!(
        "✅ Build complete in {:.2} seconds.",
        start_time.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Handles the `search` command.
pub fn handle_search(args: SearchArgs) -> anyhow::Result<()> {
    log::info!("Loading database engine for search...");
    let engine = DataEngine::open(&args.db_path)?;

    let query_vector: Vec<f32> = serde_json::from_str(&args.vector)?;

    println!("\nPerforming search...");
    let start_time = Instant::now();
    let results = engine.search(&query_vector, args.top_k)?;
    let search_duration = start_time.elapsed();

    // --- Print results in a nice table format ---
    println!(
        "Found {} results in {:.4} ms:",
        results.len(),
        search_duration.as_micros() as f64 / 1000.0
    );
    println!("{:-<80}", "");
    println!("{:<38} {:<15} Metadata", "Document ID", "Distance");
    println!("{:-<80}", "");

    for result in results {
        let metadata_str = serde_json::to_string(&result.metadata)?;
        println!(
            "{:<38} {:.6}    {}",
            result.id, result.distance, metadata_str
        );
    }
    println!("{:-<80}", "");

    Ok(())
}

pub fn handle_serve(args: ServeArgs) -> anyhow::Result<()> {
    let addr: std::net::SocketAddr = args.bind.parse()?;
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("hibc-serve")
        .build()?;
    rt.block_on(crate::api::serve::serve(args.db_path, addr))
}

pub fn handle_upsert(a: UpsertArgs) -> anyhow::Result<()> {
    let engine = DataEngine::open(&a.db_path)?;
    let vector: Vec<f32> = serde_json::from_str(&a.vector)?;
    let metadata: serde_json::Value = serde_json::from_str(&a.metadata)?;
    engine.upsert(a.id, vector, metadata, now_ts_u64())?;
    println!("OK");
    Ok(())
}

pub fn handle_delete(a: DeleteArgs) -> anyhow::Result<()> {
    let engine = DataEngine::open(&a.db_path)?;
    engine.delete(a.id, now_ts_u64())?;
    println!("OK");
    Ok(())
}

pub fn handle_flush(a: FlushArgs) -> anyhow::Result<()> {
    let engine = DataEngine::open(&a.db_path)?;
    engine.flush_now()?;
    println!("Flushed");
    Ok(())
}
