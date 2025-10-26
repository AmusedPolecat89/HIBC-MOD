// In src/index/ann/index.rs

use super::builder::EuclideanF32; // Reuse the metric
use crate::index::traits::AnnIndex;
use crate::storage::blob::BlobReader;
use crate::storage::slab::{RecordId, SlabReader};
use anyhow::Context;
use bytemuck::Pod;
use space::Neighbor;
use std::collections::{BinaryHeap, HashSet};
use std::path::Path;

/// A read-only, on-disk Approximate Nearest Neighbor index.
///
/// Implements the `AnnIndex` trait using an HNSW-like graph structure stored
/// across a SlabStore (for vectors) and a BlobStore (for graph connections).
pub struct AnnIndex {
    vector_slab: SlabReader,
    graph_store: BlobReader,
    // TODO: Store entry point and other metadata
}

impl AnnIndex {
    /// Opens an existing `AnnIndex` from its base path.
    pub fn open(base_path: &Path) -> anyhow::Result<Self> {
        let slab_path = base_path.with_extension("ann_slab");
        let blob_path = base_path.with_extension("ann_blob");

        let vector_slab = SlabReader::open(&slab_path)?;
        let graph_store = BlobReader::open(&blob_path)?;

        Ok(Self { vector_slab, graph_store })
    }

    /// Helper to safely read a record and cast it to our struct type.
    fn get_record<T: Pod>(&self, id: RecordId) -> anyhow::Result<&T> {
        let bytes = self.vector_slab.read(id)?;
        bytemuck::try_from_bytes(bytes)
            .map_err(|e| anyhow::anyhow!("Failed to cast slab record: {}", e))
    }
}

/// The result of an ANN search.
#[derive(Debug, Clone, Copy)]
pub struct AnnResult {
    pub id: RecordId,
    pub distance: f32,
}

impl AnnIndex {
    /// Performs a greedy search on the on-disk graph.
    ///
    /// This is a simplified version of an HNSW search, demonstrating the on-disk
    /// data access pattern. It starts at an entry point and greedily explores
    /// neighbors to find the nearest candidates.
    fn execute_greedy_search(&self, query: &[f32], k: usize) -> anyhow::Result<Vec<AnnResult>> {
        if self.vector_slab.is_empty() { return Ok(Vec::new()); }
        
        let metric = EuclideanF32;
        let entry_point_id: RecordId = 0; // Simplified: always start at node 0

        // Data structures for the search
        let mut visited: HashSet<RecordId> = HashSet::new();
        let mut candidates: BinaryHeap<Neighbor<u32>> = BinaryHeap::new();
        let mut results: BinaryHeap<Neighbor<u32>> = BinaryHeap::new();

        // Start the search at the entry point
        let entry_record: &super::builder::AnnSlabRecord = self.get_record(entry_point_id)?;
        let dist = metric.distance(&query.to_vec(), &entry_record.vector.to_vec());
        candidates.push(Neighbor { index: entry_point_id as usize, distance: dist });
        visited.insert(entry_point_id);

        while let Some(candidate) = candidates.pop() {
            // Add to results, keeping the heap bounded to size k
            if results.len() < k || candidate.distance < results.peek().unwrap().distance {
                results.push(candidate);
                if results.len() > k {
                    results.pop();
                }
            }

            // Get neighbors of the current best candidate
            let candidate_record: &super::builder::AnnSlabRecord = self.get_record(candidate.index as u64)?;
            let neighbors_bytes = self.graph_store.read(candidate_record.neighbors_ptr)?;
            let neighbor_list: Vec<usize> = bincode::deserialize(neighbors_bytes)?;

            // Evaluate all unvisited neighbors
            for &neighbor_id_usize in &neighbor_list {
                let neighbor_id = neighbor_id_usize as u64;
                if !visited.contains(&neighbor_id) {
                    visited.insert(neighbor_id);
                    let neighbor_record: &super::builder::AnnSlabRecord = self.get_record(neighbor_id)?;
                    let dist = metric.distance(&query.to_vec(), &neighbor_record.vector.to_vec());
                    candidates.push(Neighbor { index: neighbor_id_usize, distance: dist });
                }
            }
        }

        // Convert final results to our AnnResult type
        Ok(results.into_sorted_vec().iter().map(|n| AnnResult {
            id: n.index as u64,
            distance: f32::from_bits(n.distance),
        }).collect())
    }
}


impl AnnIndex<&[f32], AnnResult> for AnnIndex {
    fn search(&self, query: &[f32], k: usize) -> anyhow::Result<Vec<AnnResult>> {
        // In a full implementation, this would select the correct starting layer
        // and perform the full multi-layer HNSW search. For now, we use our
        // simplified greedy search on the base layer.
        self.execute_greedy_search(query, k)
    }
}
