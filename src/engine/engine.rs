// In src/engine/engine.rs

use crate::index::ann::index::AnnIndex;
use crate::index::hibc::index::HibcIndex;
use crate::index::traits::{AnnIndex as AnnIndexTrait, Index as IndexTrait};
use crate::storage::blob::BlobReader;




fn path_exists(p: &std::path::Path) -> bool {
    std::fs::metadata(p).is_ok()
}

use crate::engine::config::EngineConfig;
use crate::engine::lsm::{
    memtable::MemTable,
    wal::{WalReader, WalWriter},
    manifest::{Manifest, ManifestStore},
    segment::{paths_for_segment, gen_segment_id, SegmentMeta, StoredPayload},
};
use crate::engine::lsm::memtable::MemEntry;
use crate::engine::lsm::wal::WalRecord as OpRecord;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::{RwLock, Mutex};

/// Represents a full document, combining its ID, vector, and metadata.
#[derive(Debug, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata: serde_json::Value,
}

/// Represents a single result from a vector search query.
#[derive(Debug, Serialize, Deserialize)]
pub struct QueryResult {
    pub id: String,
    pub distance: f32,
    pub metadata: serde_json::Value,
}

#[derive(Debug)]
struct Candidate {
    ts: u64,
    distance: f32,
    metadata: serde_json::Value,
    is_tombstone: bool,
}

/// The main coordinating engine for the HIBC-MOD platform.
///
/// This struct holds open handles to all the necessary indexes and storage files,
/// providing a high-level API to perform complex operations like vector search
/// and document retrieval.
pub struct DataEngine {
    base_path: PathBuf,
    // The ANN index for vector search
    #[allow(dead_code)]
    ann_index: Option<AnnIndex>,
    // The forward index: Document ID -> Pointers
    #[allow(dead_code)]
    docmap_index: Option<HibcIndex>,
    // The reverse index: RecordId -> Document ID
    #[allow(dead_code)]
    idmap_index: Option<HibcIndex>,
    // The raw storage for metadata blobs
    #[allow(dead_code)]
    metadata_store: Option<BlobReader>,
    // Engine configuration loaded at open time
    pub config: EngineConfig,
    // NEW LSM state:
    wal_writer: Mutex<WalWriter>,
    memtable: RwLock<MemTable>,
    manifest: RwLock<Manifest>,
}

impl DataEngine {
    /// Opens an existing database from a base path.
    ///
    pub fn open(base_path: &Path) -> anyhow::Result<Self> {
        log::info!("Opening data engine at base path: {}", base_path.display());
        let ann_base_path = base_path.join("ann");
        let ann_slab = ann_base_path.with_extension("ann_slab");
        let ann_index = if path_exists(&ann_slab) {
            Some(AnnIndex::open(&ann_base_path)?)
        } else {
            None
        };

        let docmap_base_path = base_path.join("docmap");
        let docmap_db = docmap_base_path.with_extension("db");
        let docmap_index = if path_exists(&docmap_db) {
            Some(HibcIndex::open(&docmap_base_path)?)
        } else {
            None
        };

        let idmap_base_path = base_path.join("idmap");
        let idmap_db = idmap_base_path.with_extension("db");
        let idmap_index = if path_exists(&idmap_db) {
            Some(HibcIndex::open(&idmap_base_path)?)
        } else {
            None
        };

        let metadata_path = base_path.join("metadata.blob");
        let metadata_store = if path_exists(&metadata_path) {
            Some(BlobReader::open(&metadata_path)?)
        } else {
            None
        };
        let config_path = base_path.join("config.json");
        let cfg: EngineConfig = serde_json::from_slice(&std::fs::read(&config_path)?)?;
        cfg.validate()?;

        // prepare wal dir & open writer
        let wal_path = base_path.join("wal").join("current.wal");
        std::fs::create_dir_all(wal_path.parent().unwrap())?;
        
        let engine = Self {
            base_path: base_path.to_path_buf(),
            ann_index,
            docmap_index,
            idmap_index,
            metadata_store,
            config: cfg,
            wal_writer: Mutex::new(WalWriter::open(&wal_path)?),
            memtable: RwLock::new(MemTable::new()),
            manifest: RwLock::new(Manifest::default()),
        };

                        // load or create manifest
        let man_path = base_path.join("manifest.json");
        let man_store = ManifestStore::new(base_path)?;
        let manifest = if man_path.exists() { man_store.load()? } else { Manifest::default() };
        *engine.manifest.write().unwrap() = manifest;

        // WAL recovery
        let wal_path = base_path.join("wal").join("current.wal");
        if wal_path.exists() {
            let mut wal_rdr = WalReader::open(&wal_path)?;
            let records = wal_rdr.read_all()?; // Vec<WalRecord>
            {
                let mut mt = engine.memtable.write().unwrap();
                mt.apply_wal_records(records); // your existing method
            }
        }

        Ok(engine)
    }



    pub fn search(&self, query_vector: &[f32], k: usize) -> anyhow::Result<Vec<QueryResult>> {
        let mut candidates = std::collections::HashMap::<String, Candidate>::new();

        // 1. Accumulate from memtable
        let mt = self.memtable.read().unwrap();
        for (id, doc) in mt.iter() {
            let dist = euclidean(query_vector, &doc.vector);
            let cand = Candidate {
                ts: doc.ts,
                distance: dist,
                metadata: doc.metadata.clone(),
                is_tombstone: doc.tombstone,
            };
            candidates.insert(id.clone(), cand);
        }

        // 2. Accumulate from segments, newest to oldest
        let manifest = self.manifest.read().unwrap().clone();
        for seg in &manifest.active {
            let seg_base = self.base_path.join("segments").join(&seg.id);
            let ann_base = seg_base.join("ann");
            let ann_idx = crate::index::ann::index::AnnIndex::open(&ann_base)?;
            let ann_results = ann_idx.search(query_vector, k)?;
            
            let idmap = crate::index::hibc::index::HibcIndex::open(&seg_base.join("idmap"))?;
            let docmap = crate::index::hibc::index::HibcIndex::open(&seg_base.join("docmap"))?;
            let meta_reader = crate::storage::blob::BlobReader::open(&seg_base.join("metadata.blob"))?;

            for r in ann_results {
                let record_id_key = r.id.to_be_bytes();
                if let Some(doc_id_ptr) = idmap.get(&record_id_key)? {
                    let doc_id_bytes = meta_reader.read(doc_id_ptr)?;
                    let doc_id = String::from_utf8(doc_id_bytes.to_vec())?;

                    let mut doc_key = doc_id.as_bytes().to_vec();
                    doc_key.resize(self.config.doc_id_key_len, b' ');
                    if let Some(meta_ptr) = docmap.get(&doc_key)? {
                        let meta_bytes = meta_reader.read(meta_ptr)?;
                        let payload: StoredPayload = serde_json::from_slice(meta_bytes)?;

                        if let Some(existing) = candidates.get_mut(&doc_id) {
                            if payload.ts > existing.ts {
                                existing.ts = payload.ts;
                                existing.distance = r.distance;
                                existing.metadata = payload.metadata;
                                existing.is_tombstone = payload.is_tombstone;
                            }
                        } else {
                            candidates.insert(doc_id, Candidate {
                                ts: payload.ts,
                                distance: r.distance,
                                metadata: payload.metadata,
                                is_tombstone: payload.is_tombstone,
                            });
                        }
                    }
                }
            }
        }

        // 3. Filter out tombstones
        let mut final_results: Vec<_> = candidates.into_iter()
            .filter(|(_, cand)| !cand.is_tombstone)
            .map(|(id, cand)| QueryResult {
                id,
                distance: cand.distance,
                metadata: cand.metadata,
            })
            .collect();

        // 4. Sort by distance and take top-k
        final_results.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        final_results.truncate(k);

        Ok(final_results)
    }

    /// Retrieves a single document by its ID.
    pub fn get_document_by_id(&self, doc_id: &str) -> anyhow::Result<Option<Document>> {
        // Check MemTable first
        if let Some(doc) = self.memtable.read().unwrap().get(doc_id) {
            if doc.tombstone {
                return Ok(None);
            }
            return Ok(Some(Document {
                id: doc_id.to_string(),
                vector: doc.vector.clone(),
                metadata: doc.metadata.clone(),
            }));
        }

        // Check baseline stores if they exist
        if let (Some(docmap), Some(meta_store)) = (&self.docmap_index, &self.metadata_store) {
            let mut doc_key = doc_id.as_bytes().to_vec();
            doc_key.resize(self.config.doc_id_key_len, b' ');
            if let Some(metadata_ptr) = docmap.get(&doc_key)? {
                let metadata_bytes = meta_store.read(metadata_ptr)?;
                let metadata = serde_json::from_slice(metadata_bytes)?;
                // Note: vector is not stored in the baseline docmap path, so it's empty.
                // The expectation is that for a full document retrieval, you might need
                // to fetch the vector from the ANN index separately if needed.
                return Ok(Some(Document {
                    id: doc_id.to_string(),
                    vector: Vec::new(), 
                    metadata,
                }));
            }
        }

        // Else iterate segments newest→oldest
        let manifest = self.manifest.read().unwrap().clone();
        for seg in &manifest.active {
            let seg_base = self.base_path.join("segments").join(&seg.id);
            let docmap = crate::index::hibc::index::HibcIndex::open(&seg_base.join("docmap"))?;
            let meta_reader = crate::storage::blob::BlobReader::open(&seg_base.join("metadata.blob"))?;
            
            let mut doc_key = doc_id.as_bytes().to_vec();
            doc_key.resize(self.config.doc_id_key_len, b' ');

            if let Some(metadata_ptr) = docmap.get(&doc_key)? {
                let metadata_bytes = meta_reader.read(metadata_ptr)?;
                let metadata = serde_json::from_slice(metadata_bytes)?;
                
                return Ok(Some(Document {
                    id: doc_id.to_string(),
                    vector: Vec::new(), // Placeholder
                    metadata,
                }));
            }
        }

        Ok(None)
    }

    pub fn upsert(&self, id: String, vector: Vec<f32>, metadata: serde_json::Value, ts: u64) -> anyhow::Result<()> {
        let sync_wal = self.config.lsm.as_ref().map(|c| c.wal_fsync_each_write).unwrap_or(true);
        // 1) append to WAL (durable)
        {
            let mut wal = self.wal_writer.lock().unwrap();
            wal.append(&OpRecord::Upsert { id: id.clone(), vector: vector.clone(), metadata: metadata.clone(), ts })?;
            if sync_wal {
                wal.sync()?;
            }
        }
        // 2) apply to memtable (in-memory)
        {
            self.memtable.write().unwrap().upsert(id, vector, metadata, ts);
        }
        // 3) check flush thresholds
        self.maybe_flush()?;
        Ok(())
    }

    pub fn delete(&self, id: String, ts: u64) -> anyhow::Result<()> {
        let sync_wal = self.config.lsm.as_ref().map(|c| c.wal_fsync_each_write).unwrap_or(true);
        {
            let mut wal = self.wal_writer.lock().unwrap();
            wal.append(&OpRecord::Delete { id: id.clone(), ts })?;
            if sync_wal {
                // wal.sync()?;
            }
        }
        {
            self.memtable.write().unwrap().delete(id, ts);
        }
        self.maybe_flush()?;
        Ok(())
    }

    fn maybe_flush(&self) -> anyhow::Result<()> {
        let threshold_bytes = self.config.lsm.as_ref().map(|c| c.flush_threshold_bytes).unwrap_or(64 * 1024 * 1024);
        if self.memtable.read().unwrap().approx_size_bytes() > threshold_bytes {
            self.flush_now()?;
        }
        Ok(())
    }

    pub fn flush_now(&self) -> anyhow::Result<()> {
        // 1) snapshot memtable
        let snapshot: Vec<(String, MemEntry)> = {
            let mt = self.memtable.read().unwrap();
            let mut entries: Vec<_> = mt.iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.ts.cmp(&b.1.ts)));
            entries
        };
        if snapshot.is_empty() {
            return Ok(());
        }

        // 2) create segment dir & build using existing EngineBuilder logic (INTO that dir)
        let seg_id = gen_segment_id();
        let seg_paths = paths_for_segment(&self.base_path, &seg_id);
        std::fs::create_dir_all(&seg_paths.base)?;

        // write temp data.jsonl inside seg dir, use EngineBuilder::new with config
        let data_jsonl = seg_paths.base.join("data.jsonl");
        {
            let mut f = std::fs::File::create(&data_jsonl)?;
            use std::io::Write;
            for (id, entry) in &snapshot {
                let op = if entry.tombstone {
                    OpRecord::Delete {
                        id: id.to_string(),
                        ts: entry.ts,
                    }
                } else {
                    OpRecord::Upsert {
                        id: id.to_string(),
                        vector: entry.vector.clone(),
                        metadata: entry.metadata.clone(),
                        ts: entry.ts,
                    }
                };
                let line = serde_json::to_string(&op)?;
                writeln!(f, "{}", line)?;
            }
        }
        // build the mini DB segment
        {
            let mut builder = crate::engine::builder::EngineBuilder::new(
                &seg_paths.base,
                self.config.clone(),
            )?;
            builder.build_from_jsonl(&data_jsonl)?;
            builder.finalize()?;
        }
        // 3) write segment.json
        let seg_meta = SegmentMeta {
            id: seg_id.clone(),
            created: chrono::Utc::now().to_rfc3339(),
            count: snapshot.len(),
            vector_dim: self.config.vector_dim,
        };
        std::fs::write(
            &seg_paths.segment_json,
            serde_json::to_vec_pretty(&seg_meta)?,
        )?;

        // 4) clear memtable & rotate WAL (truncate)
        {
            self.memtable.write().unwrap().clear();
            // rotate/truncate
            let wal_path = self.base_path.join("wal").join("current.wal");
            let mut wal = self.wal_writer.lock().unwrap();
            *wal = WalWriter::open(&wal_path)?;
        }

        // 5) atomically update manifest (prepend newest)
        {
            let man_store = ManifestStore::new(&self.base_path)?;
            let mut manifest = self.manifest.read().unwrap().clone();
            manifest.active.insert(
                0,
                crate::engine::lsm::manifest::SegmentRef {
                    id: seg_id.clone(),
                },
            );
            man_store.store(&manifest)?;
            *self.manifest.write().unwrap() = manifest;
        }

        Ok(())
    }
}

fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}
