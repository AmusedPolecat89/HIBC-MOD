// tests/static_build_roundtrip.rs
//
// Tests the static build path (EngineBuilder) with linear validation.
// No HNSW search is used - we manually iterate vectors for verification.

use anyhow::Result;
use hibc_mod::engine::{builder::EngineBuilder, config::EngineConfig};
use hibc_mod::index::hibc::index::HibcIndex;
use hibc_mod::index::traits::Index;
use hibc_mod::storage::blob::BlobReader;
use hibc_mod::storage::slab::SlabReader;
use std::fs::File;
use std::io::Write;
use tempfile::tempdir;

// Helper to compute L2 distance
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

#[test]
fn test_static_build_legacy_format_linear_scan() -> Result<()> {
    // Test the static builder with legacy JSONL format (no "op" field)
    // Validates artifacts exist and performs linear scan verification
    
    let workdir = tempdir()?;
    let base = workdir.path().join("db");
    let data_path = workdir.path().join("data.jsonl");

    // Create small, distinct test vectors (dimension = 8 for simplicity)
    let dim = 8;
    let num_vecs = 5;
    let mut test_vecs = Vec::new();
    let mut test_ids = Vec::new();
    
    // Write legacy format JSONL (no "op" field, no "ts" field)
    {
        let mut f = File::create(&data_path)?;
        for i in 0..num_vecs {
            let mut vec = vec![0.0_f32; dim];
            vec[i % dim] = (i + 1) as f32; // Make each vector distinct
            test_vecs.push(vec.clone());
            let doc_id = format!("doc{}", i);
            test_ids.push(doc_id.clone());
            writeln!(
                f,
                r#"{{"id":"{}","vector":{},"metadata":{{"index":{}}}}}"#,
                doc_id,
                serde_json::to_string(&vec)?,
                i
            )?;
        }
    }

    // Build using EngineBuilder with safe config
    use hibc_mod::engine::config::{AlphabetSpec, AnnParams, HpinParams};
    
    let cfg = EngineConfig {
        vector_dim: dim,
        doc_id_key_len: 10,
        id_key_len: 8,
        builder_capacity_hint: num_vecs,
        docmap: HpinParams {
            n: 10,
            m: 5,
            alphabet: AlphabetSpec::ByteRange { start: 0, end: 255 },
        },
        idmap: HpinParams {
            n: 8,
            m: 4,
            alphabet: AlphabetSpec::ByteRange { start: 0, end: 255 },
        },
        ann: AnnParams {
            m: 4,
            ef_construction: 50,
            nb_layers: Some(1),
            ef_search: 32,
        },
        ..Default::default()
    };

    let mut builder = EngineBuilder::new(&base, cfg)?;
    builder.build_from_jsonl(&data_path)?;
    builder.finalize()?;

    // ===== VERIFY ARTIFACTS EXIST =====
    assert!(
        base.join("ann").with_extension("ann_slab").exists(),
        "ann_slab should exist"
    );
    assert!(
        base.join("ann").with_extension("ann_blob").exists(),
        "ann_blob should exist"
    );
    assert!(
        base.join("docmap").with_extension("hibc").exists(),
        "docmap.hibc should exist"
    );
    assert!(
        base.join("idmap").with_extension("hibc").exists(),
        "idmap.hibc should exist"
    );
    assert!(
        base.join("metadata.blob").exists(),
        "metadata.blob should exist"
    );
    assert!(
        base.join("config.json").exists(),
        "config.json should exist"
    );

    // ===== LINEAR SCAN VALIDATION (NO HNSW SEARCH) =====
    // Open the raw artifacts directly
    let slab_path = base.join("ann").with_extension("ann_slab");
    let blob_path = base.join("ann").with_extension("ann_blob");
    let slab = SlabReader::open(&slab_path)?;
    let blob = BlobReader::open(&blob_path)?;
    let idmap = HibcIndex::open(&base.join("idmap"))?;
    let docmap = HibcIndex::open(&base.join("docmap"))?;
    let meta_blob = BlobReader::open(&base.join("metadata.blob"))?;

    // Query vector: exact match with first test vector
    let query = &test_vecs[0];

    // Linear scan: for each record in slab, compute distance
    let mut scores: Vec<(f32, String, serde_json::Value)> = Vec::new();
    
    for rid in 0..slab.len() {
        // Read the ANN slab record
        let record_bytes = slab.read(rid)?;
        // The record is an AnnSlabRecord (vector_ptr, vector_len, neighbors_ptr)
        // We need to extract vector_ptr and read the vector from blob
        
        // Parse the slab record (it's a Pod struct)
        use hibc_mod::index::ann::builder::AnnSlabRecord;
        let record: &AnnSlabRecord = bytemuck::from_bytes(&record_bytes[..std::mem::size_of::<AnnSlabRecord>()]);
        
        // Read vector from blob
        let vector_bytes = blob.read(record.vector_ptr)?;
        let vector: &[f32] = bytemuck::cast_slice(vector_bytes);
        
        // Compute distance
        let dist = l2_distance(query, vector);
        
        // Lookup doc_id via idmap: record_id -> doc_id_ptr
        let rid_key = rid.to_be_bytes();
        if let Some(doc_id_ptr) = idmap.get(&rid_key)? {
            let doc_id_bytes = meta_blob.read(doc_id_ptr)?;
            let doc_id = String::from_utf8(doc_id_bytes.to_vec())?;
            
            // Lookup metadata via docmap: doc_id -> metadata_ptr
            let mut doc_key = doc_id.as_bytes().to_vec();
            doc_key.resize(10, b' '); // pad to doc_id_key_len
            if let Some(meta_ptr) = docmap.get(&doc_key)? {
                let meta_bytes = meta_blob.read(meta_ptr)?;
                let payload: serde_json::Value = serde_json::from_slice(meta_bytes)?;
                let metadata = payload.get("metadata").unwrap_or(&serde_json::Value::Null).clone();
                
                scores.push((dist, doc_id, metadata));
            }
        }
    }

    // Sort by distance
    scores.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // ===== ASSERTIONS =====
    assert_eq!(scores.len(), num_vecs, "Should have {} results", num_vecs);
    
    // First result should be exact match (distance ~0, doc0)
    assert!(scores[0].0 < 0.001, "Exact match should have distance ~0, got {}", scores[0].0);
    assert_eq!(scores[0].1, "doc0", "First result should be doc0");
    assert_eq!(scores[0].2["index"], 0, "Metadata should match");

    println!("✅ Static build + linear scan validation passed");
    println!("   Found {} results, best match: {} (dist={})", 
             scores.len(), scores[0].1, scores[0].0);

    Ok(())
}

#[test]
fn test_static_build_tagged_format() -> Result<()> {
    // Test the static builder with tagged JSONL format (with "op" field)
    // This test only validates artifacts, not search (avoiding HNSW)
    
    let workdir = tempdir()?;
    let base = workdir.path().join("db");
    let data_path = workdir.path().join("data.jsonl");

    let dim = 8;
    let vec1 = vec![1.0_f32; dim];
    let vec2 = vec![2.0_f32; dim];

    // Write tagged format JSONL (with "op" field)
    {
        let mut f = File::create(&data_path)?;
        writeln!(
            f,
            r#"{{"op":"Upsert","id":"tagged1","vector":{},"metadata":{{"type":"A"}},"ts":100}}"#,
            serde_json::to_string(&vec1)?
        )?;
        writeln!(
            f,
            r#"{{"op":"Upsert","id":"tagged2","vector":{},"metadata":{{"type":"B"}},"ts":200}}"#,
            serde_json::to_string(&vec2)?
        )?;
    }

    // Build using EngineBuilder
    use hibc_mod::engine::config::{AlphabetSpec, AnnParams, HpinParams};
    
    let cfg = EngineConfig {
        vector_dim: dim,
        doc_id_key_len: 10,
        id_key_len: 8,
        builder_capacity_hint: 2,
        docmap: HpinParams {
            n: 10,
            m: 5,
            alphabet: AlphabetSpec::ByteRange { start: 0, end: 255 },
        },
        idmap: HpinParams {
            n: 8,
            m: 4,
            alphabet: AlphabetSpec::ByteRange { start: 0, end: 255 },
        },
        ann: AnnParams {
            m: 4,
            ef_construction: 50,
            nb_layers: Some(1),
            ..Default::default()
        },
        ..Default::default()
    };

    let mut builder = EngineBuilder::new(&base, cfg)?;
    builder.build_from_jsonl(&data_path)?;
    builder.finalize()?;

    // Verify artifacts exist
    assert!(base.join("ann").with_extension("ann_slab").exists());
    assert!(base.join("ann").with_extension("ann_blob").exists());
    assert!(base.join("docmap").with_extension("hibc").exists());
    assert!(base.join("idmap").with_extension("hibc").exists());
    assert!(base.join("metadata.blob").exists());
    assert!(base.join("config.json").exists());

    println!("✅ Static build with tagged JSONL format passed");

    Ok(())
}
