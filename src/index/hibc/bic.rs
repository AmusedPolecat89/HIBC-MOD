// In src/index/hibc/bic.rs

use crate::storage::blob::BlobPointer;
use std::io::{Cursor, Read, Write};

// --- THIS IS THE NEW FUNCTION ---
/// Encodes a slice of sorted (key_suffix, BlobPointer) pairs.
pub fn encode_block_kv(sorted_pairs: &[(Vec<u8>, BlobPointer)]) -> anyhow::Result<Vec<u8>> {
    if sorted_pairs.is_empty() {
        return Ok(Vec::new());
    }

    let mut out = Vec::new();
    let (first_key, first_value) = &sorted_pairs[0];

    // Write the first pair completely.
    leb128::write::unsigned(&mut out, first_key.len() as u64)?;
    out.write_all(first_key)?;
    leb128::write::unsigned(&mut out, first_value.offset)?;
    leb128::write::unsigned(&mut out, first_value.size)?;

    let mut prev_key = first_key;
    for (current_key, current_value) in &sorted_pairs[1..] {
        let lcp_len = prev_key
            .iter()
            .zip(current_key.iter())
            .take_while(|(a, b)| a == b)
            .count();
        let suffix = &current_key[lcp_len..];

        leb128::write::unsigned(&mut out, lcp_len as u64)?;
        leb128::write::unsigned(&mut out, suffix.len() as u64)?;
        out.write_all(suffix)?;
        leb128::write::unsigned(&mut out, current_value.offset)?;
        leb128::write::unsigned(&mut out, current_value.size)?;

        prev_key = current_key;
    }
    Ok(out)
}


/// Decodes a key-value block, returning a Vec of (key_suffix, BlobPointer) pairs.
///
/// This function is central to the HIBC query path. It takes a compressed block
/// of data from the `.hibc` file and reconstructs the sorted list of key tails
/// and their associated pointers to the main value store.
pub fn decode_block_kv(buf: &[u8]) -> anyhow::Result<Vec<(Vec<u8>, BlobPointer)>> {
    if buf.is_empty() {
        return Ok(Vec::new());
    }
    let mut reader = Cursor::new(buf);
    let mut decoded_pairs = Vec::new();

    // Read the first pair, which is stored completely.
    let first_key_len = leb128::read::unsigned(&mut reader)? as usize;
    let mut first_key = vec![0u8; first_key_len];
    reader.read_exact(&mut first_key)?;
    let first_value_offset = leb128::read::unsigned(&mut reader)?;
    let first_value_size = leb128::read::unsigned(&mut reader)?;

    let mut prev_key = first_key.clone();
    decoded_pairs.push((
        first_key,
        BlobPointer {
            offset: first_value_offset,
            size: first_value_size,
        },
    ));

    // Read subsequent pairs, which are delta-encoded.
    while (reader.position() as usize) < buf.len() {
        let lcp_len = leb128::read::unsigned(&mut reader)? as usize;
        let suffix_len = leb128::read::unsigned(&mut reader)? as usize;

        let mut current_key = Vec::with_capacity(lcp_len + suffix_len);
        current_key.extend_from_slice(&prev_key[..lcp_len]); // Copy the prefix
        let mut suffix_buf = vec![0u8; suffix_len];
        reader.read_exact(&mut suffix_buf)?;
        current_key.extend_from_slice(&suffix_buf); // Append the new suffix

        let current_value_offset = leb128::read::unsigned(&mut reader)?;
        let current_value_size = leb128::read::unsigned(&mut reader)?;

        prev_key = current_key.clone();
        decoded_pairs.push((
            current_key,
            BlobPointer {
                offset: current_value_offset,
                size: current_value_size,
            },
        ));
    }

    Ok(decoded_pairs)
}
