// In src/index/hibc/tests.rs

use super::{builder::HibcIndexBuilder, hpin::Hpin, index::HibcIndex};
use crate::{index::traits::Index, storage::blob::BlobWriter};
use tempfile::tempdir;

const KEY_LENGTH: usize = 20;
const TAIL_LENGTH: usize = 16; // HPIN prefix length is 4

/// Helper to pad a key to the fixed length used in the test.
fn normalize(key: &[u8]) -> Vec<u8> {
    let mut padded = key.to_vec();
    padded.resize(KEY_LENGTH, 0); // Pad with null bytes
    padded
}

#[test]
fn test_hibc_index_roundtrip() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let base_path = dir.path().join("test_hibc");
    let blob_path = dir.path().join("test_values.blob");

    // --- SETUP ---
    let alphabet = (0..=127).collect::<Vec<u8>>(); // Use a simple ASCII alphabet
    let hpin = Hpin::new(&alphabet, KEY_LENGTH, TAIL_LENGTH)?;
    let mut hibc_builder = HibcIndexBuilder::new(&base_path, hpin.clone())?;
    let mut blob_writer = BlobWriter::new(&blob_path)?;

    let data = vec![
        (normalize(b"apple"), b"a fruit".to_vec()),
        (normalize(b"application"), b"a program".to_vec()),
        (normalize(b"apply"), b"an action".to_vec()),
        (normalize(b"banana"), b"another fruit".to_vec()),
        (normalize(b"bandana"), b"a piece of cloth".to_vec()),
    ];

    // --- WRITE PHASE ---
    for (key, value) in &data {
        let value_ptr = blob_writer.append(value)?;
        hibc_builder.add(key, value_ptr)?;
    }
    blob_writer.flush()?;
    hibc_builder.finalize()?;

    // --- READ PHASE ---
    let hibc_index = HibcIndex::open(&base_path)?;
    let blob_reader = crate::storage::blob::BlobReader::open(&blob_path)?;

    // Check hits
    for (key, value) in &data {
        let value_ptr = hibc_index.get(key)?.expect("key should be found");
        let read_value = blob_reader.read(value_ptr)?;
        assert_eq!(read_value, value.as_slice());
    }

    // Check miss
    let miss_key = normalize(b"carrot");
    let result = hibc_index.get(&miss_key)?;
    assert!(result.is_none());

    Ok(())
}
