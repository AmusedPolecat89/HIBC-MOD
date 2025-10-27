// tests/integration_tests.rs

mod lsm_updates;

use anyhow::Result;
use hibc_mod::engine::{
    config::{EngineConfig, LsmConfig},
    engine::DataEngine,
};
use serde_json::json;
use std::fs;
use std::path::PathBuf;
use tempfile::tempdir;

fn test_cfg(capacity: usize) -> EngineConfig {
    use hibc_mod::engine::config::{AlphabetSpec, HpinParams};
    
    EngineConfig {
        doc_id_key_len: 36,
        id_key_len: 8,
        builder_capacity_hint: capacity,
        docmap: HpinParams {
            n: 36,
            m: 30,
            alphabet: AlphabetSpec::ByteRange { start: 0, end: 255 },
        },
        idmap: HpinParams {
            n: 8,
            m: 4,
            alphabet: AlphabetSpec::ByteRange { start: 0, end: 255 },
        },
        lsm: Some(LsmConfig {
            flush_threshold_bytes: 256 * 1024,
            wal_fsync_each_write: false,
        }),
        ..Default::default()
    }
}

#[test]
fn test_full_build_and_search_roundtrip() -> Result<()> {
    // ---------- Setup temp workspace ----------
    let workdir = tempdir()?;
    let base: PathBuf = workdir.path().join("db");

    // ---------- Setup: create empty database ----------
    let cfg = test_cfg(3);
    fs::create_dir_all(&base)?;
    fs::write(
        base.join("config.json"),
        serde_json::to_string(&cfg)?,
    )?;
    // Create dummy index files so open doesn't fail
    fs::create_dir_all(base.join("ann"))?;
    fs::create_dir_all(base.join("docmap"))?;
    fs::create_dir_all(base.join("idmap"))?;
    fs::write(base.join("metadata.blob"), "")?;

    // ---------- Insert data using upsert API ----------
    // 512-dim exact vector of 0.1s (this will be our query)
    let exact_vec = vec![0.1_f32; 512];

    // Slightly different vector (close)
    let mut close_vec = exact_vec.clone();
    close_vec[0] = 0.2_f32;

    // Farther vector
    let far_vec = vec![0.9_f32; 512];

    // Open engine and insert records
    let engine = DataEngine::open(&base)?;
    
    engine.upsert(
        "doc_exact_00000000000000000000000000000000".to_string(),
        exact_vec.clone(),
        json!({ "title": "Exact", "tag": "golden" }),
        1
    )?;
    
    engine.upsert(
        "doc_close_0000000000000000000000000000000".to_string(),
        close_vec.clone(),
        json!({ "title": "Close", "tag": "near" }),
        2
    )?;
    
    engine.upsert(
        "doc_far_000000000000000000000000000000000".to_string(),
        far_vec.clone(),
        json!({ "title": "Far", "tag": "far" }),
        3
    )?;

    // ---------- Query: search ----------
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

    // Optional: sanity check the LSM on-disk files exist
    assert!(fs::metadata(base.join("wal")).is_ok());
    assert!(fs::metadata(base.join("wal").join("current.wal")).is_ok());

    Ok(())
}
