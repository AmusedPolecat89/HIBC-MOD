// In src/engine/engine.rs

use crate::index::ann::index::{AnnIndex, AnnResult};
use crate::index::hibc::index::HibcIndex;
use crate::index::traits::{AnnIndex as AnnIndexTrait, Index as IndexTrait};
use crate::storage::blob::BlobReader;
use serde::{Deserialize, Serialize};
use std::path::Path;

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

/// The main coordinating engine for the HIBC-MOD platform.
///
/// This struct holds open handles to all the necessary indexes and storage files,
/// providing a high-level API to perform complex operations like vector search
/// and document retrieval.
pub struct DataEngine {
    // The ANN index for vector search
    ann_index: AnnIndex,
    // The forward index: Document ID -> Pointers
    docmap_index: HibcIndex,
    // The reverse index: RecordId -> Document ID
    idmap_index: HibcIndex,
    // The raw storage for metadata blobs
    metadata_store: BlobReader,
}

impl DataEngine {
    /// Opens an existing database from a base path.
    ///
    pub fn open(base_path: &Path) -> anyhow::Result<Self> {
        log::info!("Opening data engine at base path: {}", base_path.display());
        let ann_base_path = base_path.join("ann");
        let docmap_base_path = base_path.join("docmap");
        let idmap_base_path = base_path.join("idmap");
        let metadata_path = base_path.join("metadata.blob");

        Ok(Self {
            ann_index: AnnIndex::open(&ann_base_path)?,
            docmap_index: HibcIndex::open(&docmap_base_path)?,
            idmap_index: HibcIndex::open(&idmap_base_path)?,
            metadata_store: BlobReader::open(&metadata_path)?,
        })
    }

    /// Performs a vector similarity search.
    ///
    /// This is the primary method for the RAG/recommendation use case. It orchestrates
    /// a multi-step process across all underlying components.
    pub fn search(&self, query_vector: &[f32], k: usize) -> anyhow::Result<Vec<QueryResult>> {
        // 1. Perform the fast, approximate search on the ANN index.
        //    This returns a list of internal RecordIds.
        let ann_results: Vec<AnnResult> = self.ann_index.search(query_vector, k)?;

        let mut final_results = Vec::with_capacity(ann_results.len());

        for result in ann_results {
            // 2. For each internal RecordId, look up the human-readable Document ID.
            let record_id_key = (result.id as u64).to_be_bytes();
            if let Some(doc_id_ptr) = self.idmap_index.get(&record_id_key)? {
                let doc_id_bytes = self.metadata_store.read(doc_id_ptr)?;
                let doc_id = String::from_utf8(doc_id_bytes.to_vec())?;

                // 3. Look up the metadata for that Document ID.
                let mut metadata = serde_json::Value::Null;
                if let Some(metadata_ptr) = self.docmap_index.get(doc_id.as_bytes())? {
                    let metadata_bytes = self.metadata_store.read(metadata_ptr)?;
                    metadata = serde_json::from_slice(metadata_bytes)?;
                }

                final_results.push(QueryResult {
                    id: doc_id,
                    distance: result.distance,
                    metadata,
                });
            }
        }
        Ok(final_results)
    }

    /// Retrieves a single document by its ID.
    /// (Vector is not yet retrieved in this example for simplicity).
    pub fn get_document_by_id(&self, doc_id: &str) -> anyhow::Result<Option<Document>> {
        if let Some(metadata_ptr) = self.docmap_index.get(doc_id.as_bytes())? {
            let metadata_bytes = self.metadata_store.read(metadata_ptr)?;
            let metadata = serde_json::from_slice(metadata_bytes)?;
            
            // In a full implementation, the `docmap` would also contain the RecordId,
            // allowing us to fetch the vector from the SlabStore here.
            
            return Ok(Some(Document {
                id: doc_id.to_string(),
                vector: Vec::new(), // Placeholder
                metadata,
            }));
        }
        Ok(None)
    }
}
