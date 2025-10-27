// In src/index/ann/builder.rs

use crate::storage::blob::{BlobPointer, BlobWriter};
use crate::storage::slab::SlabWriter;
use bytemuck::{Pod, Zeroable};
use std::path::Path;

// --- CORRECTED IMPORTS for hnsw_rs v0.2.1 ---
use hnsw_rs::hnsw::Hnsw;
use hnsw_rs::dist::dist::DistL2; // The correct Euclidean distance type

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct AnnSlabRecord {
    pub vector: [f32; 512],
    pub neighbors_ptr: BlobPointer,
}

// The lifetime parameter is required by Hnsw
pub struct AnnIndexBuilder<'a> {
    // Hnsw requires the lifetime, the data type, and a distance implementation
    in_memory_hnsw: Hnsw<'a, f32, DistL2>,
    // Store vectors ourselves since hnsw_rs doesn't expose them easily
    vectors: Vec<Vec<f32>>,
}

impl<'a> AnnIndexBuilder<'a> {
    pub fn new(vector_dim: usize, capacity: usize) -> Self {
        assert_eq!(vector_dim, 512, "This example builder only supports 512 dimensions");

        // Standard HNSW parameters
        let max_nb_conn = 24; // M parameter
        let nb_layer = 16usize.min((capacity as f32).ln().trunc() as usize);
        let ef_c = 400; // efConstruction parameter

        println!("[SUPER DEBUG] Creating AnnIndexBuilder.");
        println!("[SUPER DEBUG]   - Vector Dim: {}", vector_dim);
        println!("[SUPER DEBUG]   - Capacity: {}", capacity);
        println!("[SUPER DEBUG]   - Max Connections (M): {}", max_nb_conn);
        println!("[SUPER DEBUG]   - Num Layers: {}", nb_layer);
        println!("[SUPER DEBUG]   - efConstruction: {}", ef_c);

        Self {
            // The constructor takes parameters and an instance of the distance metric.
            in_memory_hnsw: Hnsw::new(max_nb_conn, capacity, nb_layer, ef_c, DistL2 {}),
            vectors: Vec::with_capacity(capacity),
        }
    }

    pub fn add(&mut self, vector: &[f32]) {
        // Get the current number of points in the HNSW structure
        let current_len = self.in_memory_hnsw.get_nb_point();
        println!("[SUPER DEBUG] add(): About to insert vector at index {}. Vector (first 5 elements): {:?}", current_len, &vector[..5]);

        // The API uses `insert_slice` with a tuple of (vector_slice, external_id).
        self.in_memory_hnsw.insert_slice((vector, current_len));
        self.vectors.push(vector.to_vec());
        println!("[SUPER DEBUG] add(): Insertion complete. New index length: {}", self.in_memory_hnsw.get_nb_point());
    }

    pub fn finalize(self, base_path: &Path) -> anyhow::Result<()> {
        log::info!("Building HNSW graph in-memory from {} vectors...", self.vectors.len());
        log::info!("In-memory HNSW graph build complete.");
        log::info!("Serializing HNSW graph to on-disk format...");
        let slab_path = base_path.with_extension("ann_slab");
        let blob_path = base_path.with_extension("ann_blob");
        println!("[SUPER DEBUG]   - Slab Path: {:?}", slab_path);
        println!("[SUPER DEBUG]   - Blob Path: {:?}", blob_path);


        let record_size = std::mem::size_of::<AnnSlabRecord>() as u64;
        let mut slab_writer = SlabWriter::new(&slab_path, record_size)?;
        let mut blob_writer = BlobWriter::new(&blob_path)?;

        let num_nodes = self.in_memory_hnsw.get_nb_point();
        println!("[SUPER DEBUG] Total nodes to serialize: {}", num_nodes);

        for node_id in 0..num_nodes {
            if node_id % 1000 == 0 {
                 println!("[SUPER DEBUG]   - Serializing node {} / {}", node_id, num_nodes);
            }
            let vector = &self.vectors[node_id];
            // Search for a small number of neighbors to build a basic graph
            let neighbors = self.in_memory_hnsw.search(vector, 10, 100);
            let neighbor_ids: Vec<u64> = neighbors.iter().map(|n| n.d_id as u64).collect();

            let neighbors_bytes = bincode::serialize(&neighbor_ids)?;
            let neighbors_ptr = blob_writer.append(&neighbors_bytes)?;

            let mut vector_array = [0.0f32; 512];
            vector_array.copy_from_slice(vector);
            let record = AnnSlabRecord {
                vector: vector_array,
                neighbors_ptr,
            };

            let record_bytes = bytemuck::bytes_of(&record);
            let written_record_id = slab_writer.append(&record_bytes)?;

            assert_eq!(written_record_id, node_id as u64, "Node IDs must be sequential");
        }

        println!("[SUPER DEBUG] Flushing slab and blob writers.");
        slab_writer.flush()?;
        blob_writer.flush()?;

        log::info!("ANN index build complete. {} nodes written.", num_nodes);
        Ok(())
    }
}