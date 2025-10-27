// In src/engine/builder.rs

use crate::index::ann::builder::AnnIndexBuilder;
use crate::index::hibc::builder::HibcIndexBuilder;
use crate::index::hibc::hpin::Hpin;
use crate::storage::blob::BlobWriter;
use crate::engine::config::EngineConfig;
use anyhow::Context;
use serde::Deserialize;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// The top-level builder for creating a complete `DataEngine` instance.
///
/// It coordinates multiple underlying index builders.
pub struct EngineBuilder<'a> { // Add lifetime here
    base_path: PathBuf,
    ann_builder: AnnIndexBuilder<'a>, // And here
    docmap_builder: HibcIndexBuilder,
    idmap_builder: HibcIndexBuilder,
    metadata_store: BlobWriter,
    config: EngineConfig,
}

/// Represents the structure of a single line in the input JSONL file.
#[derive(Deserialize)]
struct InputRecord {
    id: String,
    vector: Vec<f32>,
    metadata: serde_json::Value,
}

impl<'a> EngineBuilder<'a> { // Add lifetime here
    pub fn new(base_path: &Path, config: EngineConfig) -> anyhow::Result<Self> {
        fs::create_dir_all(base_path)?;
        config.validate()?;

        // ANN builder from config
        let ann_builder = AnnIndexBuilder::new(config.clone());

        // Blob store for all text-based values (doc ids and metadata)
        let metadata_path = base_path.join("metadata.blob");
        let metadata_store = BlobWriter::new(&metadata_path)?;
        
        // --- HIBC Builders ---
        // For docmap: keys are user-provided strings
        let docmap_base_path = base_path.join("docmap");
        let docmap_alphabet = EngineConfig::alphabet_bytes(&config.docmap.alphabet);
        let docmap_hpin = Hpin::new(
            &docmap_alphabet,
            config.docmap.n,
            config.docmap.m,
        ).unwrap();
        let docmap_builder = HibcIndexBuilder::new(&docmap_base_path, docmap_hpin)?;
        
        // For idmap: keys are 8-byte u64 integers
        let idmap_base_path = base_path.join("idmap");
        let idmap_alphabet = EngineConfig::alphabet_bytes(&config.idmap.alphabet);
        let idmap_hpin = Hpin::new(&idmap_alphabet, config.idmap.n, config.idmap.m).unwrap();
        let idmap_builder = HibcIndexBuilder::new(&idmap_base_path, idmap_hpin)?;

        Ok(Self {
            base_path: base_path.to_path_buf(),
            ann_builder,
            docmap_builder,
            idmap_builder,
            metadata_store,
            config,
        })
    }

    /// Ingests data from a JSONL file, building all indexes in a single pass.
    pub fn build_from_jsonl(&mut self, input_path: &Path) -> anyhow::Result<()> {
        let file = fs::File::open(input_path)
            .with_context(|| format!("Failed to open input file: {}", input_path.display()))?;
        let reader = BufReader::new(file);

        log::info!("Starting single-pass build from {}", input_path.display());

        for (i, line) in reader.lines().enumerate() {
            let record: InputRecord = serde_json::from_str(&line?)?;
            let record_id = i as u64;

            // 1. Add vector to the in-memory ANN builder
            self.ann_builder.add(&record.vector);
            
            // 2. Write metadata to the blob store and get a pointer
            let metadata_bytes = serde_json::to_vec(&record.metadata)?;
            let metadata_ptr = self.metadata_store.append(&metadata_bytes)?;

            // 3. Add entry to docmap: doc_id -> metadata_ptr
            let mut doc_id_key = record.id.as_bytes().to_vec();
            doc_id_key.resize(self.config.doc_id_key_len, b' '); // pad to configured length
            self.docmap_builder.add(&doc_id_key, metadata_ptr)?;

            // 4. Write doc_id to the blob store (for the reverse index) and get a pointer
            let doc_id_ptr = self.metadata_store.append(record.id.as_bytes())?;

            // 5. Add entry to idmap: record_id -> doc_id_ptr
            let idmap_key = record_id.to_be_bytes();
            self.idmap_builder.add(&idmap_key, doc_id_ptr)?;
        }

        Ok(())
    }
    
    /// Finalizes the entire build process, consuming the builder.
    pub fn finalize(self) -> anyhow::Result<()> {
        log::info!("Finalizing all indexes...");
        
        let ann_base_path = self.base_path.join("ann");
        self.ann_builder.finalize(&ann_base_path)?;

        self.docmap_builder.finalize()?;
        self.idmap_builder.finalize()?;
        self.metadata_store.flush()?;
        // Write effective config.json
        let config_path = self.base_path.join("config.json");
        std::fs::write(&config_path, serde_json::to_vec_pretty(&self.config)?)?;

        log::info!("Engine build complete.");
        Ok(())
    }
}

