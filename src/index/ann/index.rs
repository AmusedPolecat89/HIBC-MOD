// In src/index/ann/index.rs

use crate::index::traits::AnnIndex as AnnIndexTrait;
use crate::storage::blob::BlobReader;
use crate::storage::slab::{RecordId, SlabReader};
use bytemuck::Pod;
use std::collections::{BinaryHeap, HashSet};
use std::path::Path;

// Note: We no longer need space, hnsw, or custom metric imports here.

pub struct AnnIndex {
    vector_slab: SlabReader,
    graph_store: BlobReader,
}

#[derive(Debug, Clone, Copy)]
pub struct AnnResult {
    pub id: RecordId,
    pub distance: f32,
}

// A candidate for the min-priority queue used during search exploration.
// `Ord` is reversed to make `BinaryHeap` act as a min-heap.
#[derive(PartialEq, Clone, Copy, Debug)]
struct MinCandidate {
    id: RecordId,
    distance: f32,
}
impl Eq for MinCandidate {}
impl PartialOrd for MinCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        other.distance.partial_cmp(&self.distance)
    }
}
impl Ord for MinCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

// A candidate for the max-priority queue used to store final results.
// `Ord` is natural, so `BinaryHeap` acts as a max-heap.
#[derive(PartialEq, Clone, Copy, Debug)]
struct MaxCandidate {
    id: RecordId,
    distance: f32,
}
impl Eq for MaxCandidate {}
impl PartialOrd for MaxCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}
impl Ord for MaxCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}


impl AnnIndex {
    pub fn open(base_path: &Path) -> anyhow::Result<Self> {
        let slab_path = base_path.with_extension("ann_slab");
        let blob_path = base_path.with_extension("ann_blob");
        Ok(Self {
            vector_slab: SlabReader::open(&slab_path)?,
            graph_store: BlobReader::open(&blob_path)?,
        })
    }
    
    fn get_record<T: Pod>(&self, id: RecordId) -> anyhow::Result<&T> {
        bytemuck::try_from_bytes(self.vector_slab.read(id)?)
            .map_err(|e| anyhow::anyhow!("Failed to cast slab record: {}", e))
    }
    
    fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
    }

    fn execute_greedy_search(&self, query: &[f32], k: usize) -> anyhow::Result<Vec<AnnResult>> {
        if self.vector_slab.is_empty() { return Ok(Vec::new()); }
        
        let entry_point_id: RecordId = 0;
        let mut visited: HashSet<RecordId> = HashSet::new();
        let mut candidates: BinaryHeap<MinCandidate> = BinaryHeap::new();
        // Results is a max-heap, so peek() gives the worst result (largest distance).
        let mut results: BinaryHeap<MaxCandidate> = BinaryHeap::new();

        let entry_record: &super::builder::AnnSlabRecord = self.get_record(entry_point_id)?;
        let dist = Self::euclidean_distance(query, &entry_record.vector);
        candidates.push(MinCandidate { id: entry_point_id, distance: dist });
        visited.insert(entry_point_id);

        while let Some(candidate) = candidates.pop() {
            // If the best candidate is worse than our worst result, we can stop.
            if let Some(worst_result) = results.peek() {
                if results.len() >= k && candidate.distance > worst_result.distance {
                    break;
                }
            }

            // Add to results, maintaining heap size.
            results.push(MaxCandidate { id: candidate.id, distance: candidate.distance });
            if results.len() > k {
                results.pop();
            }

            let candidate_record: &super::builder::AnnSlabRecord = self.get_record(candidate.id)?;
            let neighbors_bytes = self.graph_store.read(candidate_record.neighbors_ptr)?;
            let neighbor_list: Vec<usize> = bincode::deserialize(neighbors_bytes)?;

            for &neighbor_id_usize in &neighbor_list {
                let neighbor_id = neighbor_id_usize as u64;
                if !visited.contains(&neighbor_id) {
                    visited.insert(neighbor_id);
                    let neighbor_record: &super::builder::AnnSlabRecord = self.get_record(neighbor_id)?;
                    let dist = Self::euclidean_distance(query, &neighbor_record.vector);
                    candidates.push(MinCandidate { id: neighbor_id, distance: dist });
                }
            }
        }
        
        // into_sorted_vec() drains the heap and sorts by the natural order of the items.
        // For MaxCandidate, this is ascending distance.
        Ok(results.into_sorted_vec().iter().map(|n| AnnResult {
            id: n.id,
            distance: n.distance,
        }).collect())
    }
}

impl AnnIndexTrait<&[f32], AnnResult> for AnnIndex {
    fn search(&self, query: &[f32], k: usize) -> anyhow::Result<Vec<AnnResult>> {
        self.execute_greedy_search(query, k)
    }
}
