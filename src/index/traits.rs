// In src/index/traits.rs

use std::any::Any;

/// A generic trait representing a key-value index.
///
/// This is the fundamental interface for all exact-match lookups in the system,
/// such as the HIBC-based indexes for ID mapping and metadata retrieval.
pub trait Index<K, V> {
    /// Retrieves a value for a given key.
    fn get(&self, key: K) -> anyhow::Result<Option<V>>;

    // We can extend this with more methods later, e.g.:
    // fn contains_key(&self, key: K) -> anyhow::Result<bool>;
}

/// A specialized trait for Approximate Nearest Neighbor (ANN) indexes.
///
/// This defines the contract for vector search components like HNSW.
pub trait AnnIndex<Q, R> {
    /// Performs a similarity search.
    ///
    /// # Arguments
    /// * `query`: The query item (e.g., a vector slice `&[f32]`).
    /// * `k`: The number of nearest neighbors to return.
    fn search(&self, query: Q, k: usize) -> anyhow::Result<Vec<R>>;
}

/// A helper trait to allow for downcasting concrete index types if needed.
pub trait AsAny {
    fn as_any(&self) -> &dyn Any;
}

impl<T: Any> AsAny for T {
    fn as_any(&self) -> &dyn Any {
        self
    }
}
