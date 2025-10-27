Architecture & Technical Deep Dive
hibc-mod is a high-performance vector database built on a modular, layered architecture in Rust. It combines a state-of-the-art Log-Structured Merge-tree (LSM) for dynamic updates with two specialized, high-performance indexing engines: an Approximate Nearest Neighbor (ANN) index for vector search and the novel Hybrid Indexed Block Compression (HIBC) index for exact-match key lookups.
This unique combination allows hibc-mod to achieve exceptional performance for both ingestion and complex queries.
The Four Layers
The platform is designed with a clean separation of concerns across four layers:
API Layer: Provides multiple interfaces for interaction: a CLI, an HTTP server (powered by Axum), and Python bindings (powered by PyO3).
Engine Layer: The central DataEngine that coordinates all underlying components and exposes a clean, high-level API (e.g., search, upsert).
Indexing Layer: Contains the pluggable indexing strategies. This is where the core novelty of the platform resides.
Storage Layer: The foundational layer that manages physical disk I/O via BlobStore (for variable-size data) and SlabStore (for fixed-size data) using memory-mapped files for fast reads.
The Indexing Layer: A Tale of Two Engines
The power of hibc-mod comes from its two specialized indexing engines that work in concert.
1. The ANN Index: For Fast Similarity Search
The ANN index is responsible for finding the "most similar" vectors in the database.
Algorithm: It uses a state-of-the-art implementation of the HNSW (Hierarchical Navigable Small World) algorithm.
On-Disk Format: The HNSW graph and its associated vectors are stored in a custom, locality-optimized format (.ann_slab file). A node's vector and its graph connections (neighbor lists) are stored contiguously. This design is critical for performance, as it minimizes the number of random disk I/O operations required to traverse the graph, ensuring that a single page fault often loads all the necessary data for one hop.
2. The HIBC Index: For Fast Exact-Match Lookups
The Hybrid Indexed Block Compression (HIBC) engine is a novel, read-optimized key-value store. It is not an ANN index, but rather the high-performance backbone for all metadata and ID lookups in the system. Its novelty comes from its synergistic, two-level architecture.
The first level partitions the entire keyspace into manageable chunks using Hierarchical Prefix-ID Notation (HPIN).
Concept: HPIN is a deterministic function that converts the fixed-length prefix of a key into a compact 64-bit integer (PrefixID). All keys that share the same prefix will map to the same PrefixID.
The Math: HPIN treats a key's prefix as a number in a different base (base-k), where k is the size of the defined alphabet. The formula to calculate the PrefixID for a prefix p of length L is:
code
Code
PrefixID(p) = Σ [ ordinal(p_i) * k^(L - 1 - i) ]  (for i from 0 to L-1)
Where ordinal(p_i) is the 0-based index of the character p_i in the alphabet. This calculation maps each unique prefix to a unique, non-negative integer.
Implementation: This top-level index, which maps each PrefixID to a pointer on disk, is stored in a robust SQLite database for transactional integrity.
Each PrefixID in the master index points to a specific block within a memory-mapped .hibc file. This block contains the remaining parts (the "suffixes") of all keys that share that prefix.
Concept: To achieve extreme data density, these blocks are compressed using a technique we call Block Indexed Compression (BIC).
The Math: BIC is a highly optimized implementation of prefix compression (also known as front coding), applied to the sorted key suffixes within a block.
The first key suffix in the block is stored in its entirety.
Every subsequent suffix, S_n, is stored by recording two pieces of information:
The length of the longest common prefix (LCP) it shares with the preceding suffix, S_(n-1).
The remaining, non-matching portion of S_n.
Because the suffixes within a block are sorted, the LCP is often very long, meaning only a few bytes are needed to store each subsequent key. This is particularly effective for structured keys like URLs, timestamps, or document IDs.
The Full Lookup Process (HIBC)
A fast, exact-match get(key) operation in a HIBC index involves a two-step process:
HPIN Calculation & Master Index Lookup: The system calculates the PrefixID from the key's prefix and performs a single O(log N) lookup in the SQLite master index to get the block pointer.
Block Read & Decompression: The system reads the compressed block from the memory-mapped .hibc file (a memory-speed operation) and performs a fast binary search on the decompressed key suffixes to find the exact match.
This two-level design allows hibc-mod to manage massive keyspaces with minimal I/O and exceptionally low read latency, making it the perfect engine to power the metadata, ID mapping, and future filtered search capabilities of the platform.