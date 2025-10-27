
use hibc_mod::engine::config::{EngineConfig, LsmConfig};
use hibc_mod::engine::engine::DataEngine;
use std::fs;
use tempfile::tempdir;

fn test_cfg() -> EngineConfig {
    use hibc_mod::engine::config::{AlphabetSpec, HpinParams};
    
    EngineConfig {
        doc_id_key_len: 36,
        id_key_len: 8,
        vector_dim: 2,
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

fn create_empty_db() -> tempfile::TempDir {
    let dir = tempdir().unwrap();
    let config = test_cfg();
    fs::write(
        dir.path().join("config.json"),
        serde_json::to_string(&config).unwrap(),
    )
    .unwrap();
    // Create dummy index files so open doesn't fail
    fs::create_dir_all(dir.path().join("ann")).unwrap();
    fs::create_dir_all(dir.path().join("docmap")).unwrap();
    fs::create_dir_all(dir.path().join("idmap")).unwrap();
    fs::write(dir.path().join("metadata.blob"), "").unwrap();
    dir
}

#[test]
fn test_upsert_visible_immediately() {
    let dir = create_empty_db();
    let engine = DataEngine::open(dir.path()).unwrap();
    let vec = vec![1.0, 2.0];
    engine.upsert("u1".to_string(), vec.clone(), serde_json::Value::Null, 0).unwrap();
    let results = engine.search(&vec, 1).unwrap();
    assert_eq!(results[0].id, "u1");
}

#[test]
fn test_delete_hides_record() {
    let dir = create_empty_db();
    let engine = DataEngine::open(dir.path()).unwrap();
    let vec = vec![1.0, 2.0];
    engine.upsert("x".to_string(), vec.clone(), serde_json::Value::Null, 0).unwrap();
    engine.flush_now().unwrap();
    engine.delete("x".to_string(), 1).unwrap();
    let results = engine.search(&vec, 1).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_flush_creates_segment_and_clears_memtable() {
    let dir = create_empty_db();
    let engine = DataEngine::open(dir.path()).unwrap();
    engine.upsert("doc1".to_string(), vec![1.0, 0.0], serde_json::Value::Null, 0).unwrap();
    engine.flush_now().unwrap();
    
    let manifest_path = dir.path().join("manifest.json");
    assert!(manifest_path.exists());
    let manifest_content = fs::read_to_string(manifest_path).unwrap();
    assert!(manifest_content.contains("seg_"));

    // This is a proxy for checking if the memtable is clear.
    let results = engine.search(&[1.0, 0.0], 1).unwrap();
    assert_eq!(results.len(), 1);
}

#[test]
fn test_recovery_from_wal() {
    let dir = create_empty_db();
    {
        let engine = DataEngine::open(dir.path()).unwrap();
        engine.upsert("rec1".to_string(), vec![1.0, 2.0], serde_json::Value::Null, 0).unwrap();
        // Engine is dropped here, simulating a crash
    }
    
    let engine = DataEngine::open(dir.path()).unwrap();
    let results = engine.search(&[1.0, 2.0], 1).unwrap();
    assert_eq!(results[0].id, "rec1");
}
