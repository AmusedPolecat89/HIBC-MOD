// tests/integration_tests.rs

use anyhow::Result;
use hibc_mod::engine::{builder::EngineBuilder, engine::DataEngine};
use serde_json::json;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use tempfile::tempdir;

#[test]
fn test_full_build_and_search_roundtrip() -> Result<()> {
    // ---------- Setup temp workspace ----------
    let workdir = tempdir()?;
    let base: PathBuf = workdir.path().join("db");
    let data_path: PathBuf = workdir.path().join("data.jsonl");

    // ---------- Build: write a small, predictable dataset ----------
    // 512-dim exact vector of 0.1s (this will be our query)
    let exact_vec = vec![0.1_f32; 512];

    // Slightly different vector (close)
    let mut close_vec = exact_vec.clone();
    close_vec[0] = 0.2_f32;

    // Farther vector
    let far_vec = vec![0.9_f32; 512];

    // Define records: 3 documents with simple metadata
    let records = vec![
        json!({
            "id": "doc_exact_00000000000000000000000000000000",
            "vector": exact_vec,
            "metadata": { "title": "Exact", "tag": "golden" }
        }),
        json!({
            "id": "doc_close_0000000000000000000000000000000",
            "vector": close_vec,
            "metadata": { "title": "Close", "tag": "near" }
        }),
        json!({
            "id": "doc_far_000000000000000000000000000000000",
            "vector": far_vec,
            "metadata": { "title": "Far", "tag": "far" }
        }),
    ];

    // Write JSONL
    {
        let mut f = File::create(&data_path)?;
        for rec in &records {
            writeln!(f, "{}", serde_json::to_string(rec)?)?;
        }
    }

    // ---------- Build: run EngineBuilder over the JSONL ----------
    let vector_dim = 512usize;
    let capacity = records.len();
    let mut builder = EngineBuilder::new(&base, vector_dim, capacity)?;
    builder.build_from_jsonl(&data_path)?;
    builder.finalize()?;

    // ---------- Query: open engine and search ----------
    let engine = DataEngine::open(&base)?;
    let query = vec![0.1_f32; 512];
    let k = 3usize;
    let results = engine.search(&query, k)?;

    // ---------- Assert: correctness ----------
    // Has expected length
    assert_eq!(results.len(), 3, "expected exactly 3 results");

    // Sorted by ascending distance
    assert!(results[0].distance <= results[1].distance);
    assert!(results[1].distance <= results[2].distance);

    // First result is the exact vector -> distance 0.0, correct id & metadata
    assert_eq!(results[0].distance, 0.0, "exact match should have distance 0");
    assert_eq!(
        results[0].id,
        "doc_exact_00000000000000000000000000000000"
    );

    // Metadata equality (engine returns serde_json::Value)
    let expected_meta = json!({ "title": "Exact", "tag": "golden" });
    assert_eq!(results[0].metadata, expected_meta);

    // Optional: sanity check the on-disk files exist
    assert!(fs::metadata(base.join("ann").with_extension("ann_slab")).is_ok());
    assert!(fs::metadata(base.join("ann").with_extension("ann_blob")).is_ok());
    assert!(fs::metadata(base.join("docmap").with_extension("db")).is_ok());
    assert!(fs::metadata(base.join("docmap").with_extension("hibc")).is_ok());
    assert!(fs::metadata(base.join("idmap").with_extension("db")).is_ok());
    assert!(fs::metadata(base.join("idmap").with_extension("hibc")).is_ok());
    assert!(fs::metadata(base.join("metadata.blob")).is_ok());

    Ok(())
}
