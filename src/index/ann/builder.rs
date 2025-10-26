// In src/index/ann/builder.rs

use crate::storage::blob::{BlobPointer, BlobWriter};
use crate::storage::slab::{RecordId, SlabWriter};
use anyhow::Context;
use bytemuck::{Pod, Zeroable};
use hnsw::{Hnsw, Searcher};
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use space::Metric;
use std::path::Path;

// --- Define the Metric for HNSW ---
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
struct EuclideanF32;
impl Metric<Vec<f32>> for EuclideanF32 {
    type Unit = u32;
    fn distance(&self, a: &Vec<f32>, b: &Vec<f32>) -> Self::Unit {
        let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        sum.sqrt().to_bits()
    }
}

/// The fixed-size record that will be stored in the SlabStore.
/// It contains the vector data and a pointer to its neighbor list in the BlobStore.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AnnSlabRecord {
    /// The vector's raw data. We use a fixed-size array for a known layout.
    /// This example assumes a dimension of 512. This would need to be generic
    /// or configured in a real implementation.
    vector: [f32; 512],
    /// A pointer to the variable-length neighbor list in the companion BlobStore.
    neighbors_ptr: BlobPointer,
}

/// A builder for creating an `AnnIndex`.
///
/// It builds an HNSW graph in-memory and then serializes it to a highly-optimized
/// on-disk format using a SlabStore for vectors and a BlobStore for graph edges.
pub struct AnnIndexBuilder {
    vector_dim: usize,
    in_memory_hnsw: Hnsw<EuclideanF32, Vec<f32>, StdRng, 12, 24>,
}

impl AnnIndexBuilder {
    pub fn new(vector_dim: usize) -> Self {
        // This example uses a fixed dimension. A real implementation would need
        // to handle this more dynamically.
        assert_eq!(vector_dim, 512, "This example builder only supports 512 dimensions");

        Self {
            vector_dim,
            in_memory_hnsw: Hnsw::new(EuclideanF32, StdRng::from_seed([0; 32])),
        }
    }

    /// Adds a vector to the in-memory graph.
    pub fn add(&mut self, vector: Vec<f32>) {
        // A searcher is needed for the insert operation.
        let mut searcher = Searcher::default();
        self.in_memory_hnsw.insert(vector, &mut searcher);
    }

    /// Finalizes the build, writing the in-memory graph to disk.
    pub fn finalize(self, base_path: &Path) -> anyhow::Result<()> {
        let slab_path = base_path.with_extension("ann_slab");
        let blob_path = base_path.with_extension("ann_blob");

        // The record size must match our struct's size.
        let record_size = std::mem::size_of::<AnnSlabRecord>() as u64;
        let mut slab_writer = SlabWriter::new(&slab_path, record_size)?;
        let mut blob_writer = BlobWriter::new(&blob_path)?;

        log::info!("Serializing HNSW graph to on-disk format...");

        let num_nodes = self.in_memory_hnsw.num_nodes();
        for node_id in 0..num_nodes {
            // 1. Get the vector and neighbor list from the in-memory graph.
            let vector = self.in_memory_hnsw.get_vector(node_id);
            let neighbor_list = self.in_memory_hnsw.get_neighbors(node_id);

            // 2. Serialize the variable-length neighbor list and write it to the BlobStore.
            // Using bincode for simple serialization.
            let neighbors_bytes = bincode::serialize(&neighbor_list)
                .context("Failed to serialize neighbor list")?;
            let neighbors_ptr = blob_writer.append(&neighbors_bytes)?;

            // 3. Prepare the fixed-size SlabRecord.
            let mut vector_array = [0.0f32; 512];
            vector_array.copy_from_slice(vector);
            let record = AnnSlabRecord {
                vector: vector_array,
                neighbors_ptr,
            };

            // 4. Write the slab record to the SlabStore.
            // bytemuck::bytes_of safely converts our Pod struct to a byte slice.
            let record_bytes = bytemuck::bytes_of(&record);
            let written_record_id = slab_writer.append(record_bytes)?;

            // Sanity check: ensure node IDs are sequential.
            assert_eq!(written_record_id, node_id as u64, "Node IDs must be sequential");
        }
        
        slab_writer.flush()?;
        blob_writer.flush()?;
        
        log::info!("ANN index build complete. {} nodes written.", num_nodes);
        Ok(())
    }
}
