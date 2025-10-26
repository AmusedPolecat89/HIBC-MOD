// In src/index/hibc/index.rs

use super::bic;
use super::hpin::{Hpin, HpinError};
use crate::index::traits::Index;
use crate::storage::blob::BlobPointer;
use anyhow::Context;
use memmap2::Mmap;
use rusqlite::{Connection, OptionalExtension};
use std::collections::HashSet;
use std::path::Path;

/// A read-only, high-performance index using the HIBC architecture.
///
/// It implements the `Index` trait for exact-match key-value lookups,
/// where the value is a `BlobPointer` pointing to the actual data in a `BlobStore`.
pub struct HibcIndex {
    hpin: Hpin,
    index_conn: Connection,
    split_prefixes: HashSet<u64>,
    data_mmap: Mmap,
}

impl HibcIndex {
    /// Opens an existing HIBC index from a base path.
    ///
    /// It expects to find `{base_path}.db` and `{base_path}.hibc`.
    pub fn open(base_path: &Path) -> anyhow::Result<Self> {
        let db_path = base_path.with_extension("db");
        let data_path = base_path.with_extension("hibc");

        // Open master index and load metadata
        let index_conn = Connection::open_with_flags(db_path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        
        let alphabet_str: String = index_conn.query_row("SELECT value FROM metadata WHERE key = 'alphabet'", [], |r| r.get(0))?;
        let n: usize = index_conn.query_row("SELECT value FROM metadata WHERE key = 'n'", [], |r| r.get::<_, String>(0))?.parse()?;
        let m: usize = index_conn.query_row("SELECT value FROM metadata WHERE key = 'm'", [], |r| r.get::<_, String>(0))?.parse()?;
        
        let split_prefixes_str: String = index_conn.query_row("SELECT value FROM metadata WHERE key = 'split_prefixes'", [], |r| r.get(0))?;
        let split_prefixes = if split_prefixes_str.is_empty() {
            HashSet::new()
        } else {
            split_prefixes_str.split(',').map(|s| s.parse()).collect::<Result<HashSet<u64>, _>>()?
        };

        let hpin = Hpin::new(alphabet_str.as_bytes(), n, m).map_err(|e| anyhow::anyhow!(e))?;

        // Memory-map the data file
        let data_file = std::fs::File::open(&data_path)
            .with_context(|| format!("Failed to open data file: {}", data_path.display()))?;
        let data_mmap = unsafe { Mmap::map(&data_file)? };

        Ok(Self { hpin, index_conn, split_prefixes, data_mmap })
    }

    /// Private helper to get a decompressed block from the data file.
    fn get_decoded_block(&self, pid: u64) -> anyhow::Result<Option<Vec<(Vec<u8>, BlobPointer)>>> {
        let mut stmt = self.index_conn.prepare_cached("SELECT offset, size FROM master_index WHERE prefix_id = ?")?;
        let block_pointer: Option<(u64, u64)> = stmt.query_row([&pid.to_be_bytes()], |row| Ok((row.get(0)?, row.get(1)?))).optional()?;
        
        let Some((offset, size)) = block_pointer else { return Ok(None) };
        
        let block_data = self.data_mmap
            .get(offset as usize..(offset + size) as usize)
            .context("Block pointer out of bounds in data file")?;
        
        // TODO: Add CRC32 checksum validation here for production-readiness

        Ok(Some(bic::decode_block_kv(block_data)?))
    }
}

impl Index<&[u8], BlobPointer> for HibcIndex {
    /// Performs a key lookup.
    fn get(&self, key: &[u8]) -> anyhow::Result<Option<BlobPointer>> {
        let (pid, key_suffix) = match self.hpin.parse(key) {
            Ok(result) => result,
            Err(HpinError::InvalidLength { .. }) => return Ok(None), // Not a valid key for this index
            Err(e) => return Err(e.into()),
        };

        // Determine the effective prefix ID and key suffix, handling split prefixes.
        let (effective_pid, effective_key_suffix) = if self.split_prefixes.contains(&pid) {
            let first_byte = if key_suffix.is_empty() { 256 } else { key_suffix[0] as u64 };
            ((pid << 8) | first_byte, &key_suffix[1..])
        } else {
            (pid, key_suffix)
        };

        // Retrieve and decode the entire block associated with the prefix.
        if let Some(pairs) = self.get_decoded_block(effective_pid)? {
            // Perform a binary search on the decoded tails within the block.
            let search_result = pairs.binary_search_by(|(probe_key, _)| probe_key.as_slice().cmp(effective_key_suffix));
            if let Ok(index) = search_result {
                return Ok(Some(pairs[index].1)); // Found it!
            }
        }
        
        Ok(None) // Not found
    }
}
