// In src/index/ann/builder.rs

use crate::storage::blob::{BlobPointer, BlobWriter};
use crate::storage::slab::SlabWriter;
use crate::engine::config::EngineConfig;
use bytemuck::{Pod, Zeroable};
use std::path::Path;

// --- CORRECTED IMPORTS for hnsw_rs v0.2.1 ---
use hnsw_rs::hnsw::Hnsw;
use hnsw_rs::dist::dist::DistL2; // The correct Euclidean distance type

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct AnnSlabRecord {
    pub vector_ptr: BlobPointer,
    pub vector_len: u32,       // number of f32 values
    pub _pad: u32,             // keep 16B alignment (optional)
    pub neighbors_ptr: BlobPointer,
}

// The lifetime parameter is required by Hnsw
pub struct AnnIndexBuilder<'a> {
    // Hnsw requires the lifetime, the data type, and a distance implementation
    in_memory_hnsw: Hnsw<'a, f32, DistL2>,
    // Store vectors ourselves since hnsw_rs doesn't expose them easily
    vectors: Vec<Vec<f32>>,
    cfg: EngineConfig,
}

impl<'a> AnnIndexBuilder<'a> {
    pub fn new(cfg: EngineConfig) -> Self {
        let vector_dim = cfg.vector_dim;
        let capacity = cfg.builder_capacity_hint;

        // Standard HNSW parameters (from config)
        let max_nb_conn = cfg.ann.m;
        let nb_layer = cfg.ann.nb_layers.unwrap_or_else(|| {
            16usize.min((capacity as f32).ln().trunc() as usize)
        });
        let ef_c = cfg.ann.ef_construction;

        log::debug!("Creating AnnIndexBuilder.");
        log::debug!("  - Vector Dim: {}", vector_dim);
        log::debug!("  - Capacity: {}", capacity);
        log::debug!("  - Max Connections (M): {}", max_nb_conn);
        log::debug!("  - Num Layers: {}", nb_layer);
        log::debug!("  - efConstruction: {}", ef_c);

        Self {
            // The constructor takes parameters and an instance of the distance metric.
            in_memory_hnsw: Hnsw::new(max_nb_conn, capacity, nb_layer, ef_c, DistL2 {}),
            vectors: Vec::with_capacity(capacity),
            cfg,
        }
    }

    pub fn add(&mut self, vector: &[f32]) {
        assert_eq!(vector.len(), self.cfg.vector_dim, "vector length must equal config.vector_dim");
        // Get the current number of points in the HNSW structure
        let current_len = self.in_memory_hnsw.get_nb_point();
        log::trace!("add(): About to insert vector at index {}. Vector (first 5 elements): {:?}", current_len, &vector[..5]);

        // The API uses `insert_slice` with a tuple of (vector_slice, external_id).
        self.in_memory_hnsw.insert_slice((vector, current_len));
        self.vectors.push(vector.to_vec());
        log::debug!("add(): Insertion complete. New index length: {}", self.in_memory_hnsw.get_nb_point());
    }

    pub fn finalize(self, base_path: &Path) -> anyhow::Result<()> {
        log::info!("Building HNSW graph in-memory from {} vectors...", self.vectors.len());
        log::info!("In-memory HNSW graph build complete.");
        log::info!("Serializing HNSW graph to on-disk format...");
        let slab_path = base_path.with_extension("ann_slab");
        let blob_path = base_path.with_extension("ann_blob");
        log::debug!("  - Slab Path: {:?}", slab_path);
        log::debug!("  - Blob Path: {:?}", blob_path);


        let record_size = std::mem::size_of::<AnnSlabRecord>() as u64;
        let mut slab_writer = SlabWriter::new(&slab_path, record_size)?;
        let mut blob_writer = BlobWriter::new(&blob_path)?;

        let num_nodes = self.in_memory_hnsw.get_nb_point();
        log::debug!("Total nodes to serialize: {}", num_nodes);

        for node_id in 0..num_nodes {
            if node_id % 1000 == 0 {
                log::trace!("  - Serializing node {} / {}", node_id, num_nodes);
            }
            let vector = &self.vectors[node_id];
            // Search for a small number of neighbors to build a basic graph
            let neighbors = self.in_memory_hnsw.search(
                vector,
                self.cfg.ann_build_neighbor_k,
                self.cfg.ann.ef_search.max(100),
            );
            let neighbor_ids: Vec<u64> = neighbors.iter().map(|n| n.d_id as u64).collect();

            let neighbors_bytes = bincode::serialize(&neighbor_ids)?;
            let neighbors_ptr = blob_writer.append(&neighbors_bytes)?;

            // Store vector bytes in blob
            let vector_bytes = bytemuck::cast_slice::<f32, u8>(vector);
            let vector_ptr = blob_writer.append(vector_bytes)?;
            let record = AnnSlabRecord {
                vector_ptr,
                vector_len: self.cfg.vector_dim as u32,
                _pad: 0,
                neighbors_ptr,
            };

            let record_bytes = bytemuck::bytes_of(&record);
            let written_record_id = slab_writer.append(&record_bytes)?;

            assert_eq!(written_record_id, node_id as u64, "Node IDs must be sequential");
        }

        log::debug!("Flushing slab and blob writers.");
        slab_writer.flush()?;
        blob_writer.flush()?;

        log::info!("ANN index build complete. {} nodes written.", num_nodes);
        Ok(())
    }
}