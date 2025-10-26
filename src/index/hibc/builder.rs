// In src/index/hibc/builder.rs

use super::{bic, hpin::Hpin};
use crate::storage::blob::BlobPointer;
use anyhow::{anyhow, Context};
use rayon::prelude::*;
use rusqlite::{Connection, Transaction};
use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::io::{BufWriter, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread::{self, JoinHandle};

// Internal structs for managing the build process
#[derive(Debug, Clone, Copy)]
struct BlockPointerInternal {
    offset: u64,
    size: u64,
    tail_count: u32,
}

#[derive(Debug, Clone, Copy)]
struct IndexEntry {
    pid: u64,
    pointer: BlockPointerInternal,
}

enum WriterMessage {
    Block(u64, Vec<u8>),
    Shutdown,
}

/// A builder for creating a `HibcIndex`.
pub struct HibcIndexBuilder {
    hpin: Hpin,
    base_path: PathBuf,
    buffers: HashMap<u64, Vec<(Vec<u8>, BlobPointer)>>,
    // TODO: Add dynamic skew handling logic
    split_prefixes: HashSet<u64>,
    writer_tx: Sender<WriterMessage>,
    writer_handle: Option<JoinHandle<anyhow::Result<Vec<IndexEntry>>>>,
}

impl HibcIndexBuilder {
    /// Creates a new builder. This will delete any existing index files at the base path.
    pub fn new(base_path: &Path, hpin: Hpin) -> anyhow::Result<Self> {
        let db_path = base_path.with_extension("db");
        let data_path = base_path.with_extension("hibc");
        if db_path.exists() { std::fs::remove_file(db_path)?; }
        if data_path.exists() { std::fs::remove_file(data_path)?; }

        let (tx, rx) = channel();
        let data_path_clone = data_path.to_path_buf();
        let handle = thread::spawn(move || writer_thread_loop(rx, data_path_clone));

        Ok(Self {
            hpin,
            base_path: base_path.to_path_buf(),
            buffers: HashMap::new(),
            split_prefixes: HashSet::new(),
            writer_tx: tx,
            writer_handle: Some(handle),
        })
    }

    /// Adds a key-value pair to the builder.
    /// The value is a pointer to data already stored in a `BlobStore`.
    pub fn add(&mut self, key: &[u8], value_pointer: BlobPointer) -> anyhow::Result<()> {
        let (pid, key_suffix) = self.hpin.parse(key)?;
        // TODO: Implement split prefix logic for skew handling
        let buffer = self.buffers.entry(pid).or_default();
        buffer.push((key_suffix.to_vec(), value_pointer));
        Ok(())
    }

    /// Finalizes the build process.
    ///
    /// This method consumes the builder. It sorts and compresses all in-memory
    /// buffers in parallel, writes them to the data file, and builds the
    /// final SQLite master index.
    pub fn finalize(mut self) -> anyhow::Result<()> {
        log::info!("Finalizing HIBC index build for '{}'...", self.base_path.display());

        // Use Rayon to process all blocks in parallel
        self.buffers
            .into_par_iter()
            .for_each_with(self.writer_tx.clone(), |tx, (pid, mut key_value_pairs)| {
                if key_value_pairs.is_empty() { return; }
                
                // 1. Sort tails lexicographically
                key_value_pairs.sort_by(|a, b| a.0.cmp(&b.0));
                
                // 2. Compress the block using BIC
                let payload = bic::encode_block_kv(&key_value_pairs).unwrap();
                let pair_count = key_value_pairs.len() as u32;

                // 3. Prepend a simple header (version + count)
                let mut block_with_header = Vec::with_capacity(10 + payload.len());
                block_with_header.push(0); // Version byte
                leb128::write::unsigned(&mut block_with_header, pair_count as u64).unwrap();
                block_with_header.extend_from_slice(&payload);
                
                // 4. Send the finished block to the writer thread
                if let Err(e) = tx.send(WriterMessage::Block(pid, block_with_header)) {
                    log::error!("Failed to send block {} to writer thread: {}", pid, e);
                }
            });

        log::info!("All blocks compressed. Shutting down writer thread...");
        self.writer_tx.send(WriterMessage::Shutdown)?;
        let index_cache = self
            .writer_handle
            .take()
            .unwrap()
            .join()
            .unwrap()
            .context("Writer thread panicked")?;
        
        log::info!("Writer thread finished. Building master index...");

        // --- Build the SQLite Index ---
        let db_path = self.base_path.with_extension("db");
        let mut conn = Connection::open(db_path)?;
        create_master_index_tables(&conn)?;
        
        let tx = conn.transaction()?;
        bulk_insert_entries(&tx, &index_cache)?;
        write_metadata(&tx, &self.hpin, &self.split_prefixes)?;
        tx.commit()?;
        
        log::info!("HIBC index build complete.");
        Ok(())
    }
}

/// The main loop for the dedicated I/O writer thread.
fn writer_thread_loop(
    rx: Receiver<WriterMessage>,
    data_path: PathBuf,
) -> anyhow::Result<Vec<IndexEntry>> {
    let data_file = OpenOptions::new().append(true).create(true).open(data_path)?;
    let mut data_writer = BufWriter::new(data_file);
    let mut index_cache = Vec::new();

    for message in rx {
        match message {
            WriterMessage::Block(pid, block_data) => {
                let offset = data_writer.stream_position()?;
                data_writer.write_all(&block_data)?;
                let size_on_disk = block_data.len() as u64;

                // We can read the count back from the header we just wrote.
                let mut reader = std::io::Cursor::new(&block_data);
                reader.seek(SeekFrom::Start(1))?; // Skip version byte
                let tail_count = leb128::read::unsigned(&mut reader)? as u32;

                index_cache.push(IndexEntry {
                    pid,
                    pointer: BlockPointerInternal { offset, size: size_on_disk, tail_count },
                });
            }
            WriterMessage::Shutdown => break,
        }
    }

    data_writer.flush()?;
    Ok(index_cache)
}


// --- SQLite Helper Functions ---
fn create_master_index_tables(conn: &Connection) -> anyhow::Result<()> {
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         CREATE TABLE master_index (
             prefix_id BLOB(8) PRIMARY KEY,
             offset INTEGER NOT NULL,
             size INTEGER NOT NULL,
             tail_count INTEGER NOT NULL
         );
         CREATE TABLE metadata (
             key TEXT PRIMARY KEY,
             value TEXT NOT NULL
         );",
    )?;
    Ok(())
}

fn bulk_insert_entries(tx: &Transaction, entries: &[IndexEntry]) -> anyhow::Result<()> {
    let mut stmt = tx.prepare_cached(
        "INSERT INTO master_index (prefix_id, offset, size, tail_count) VALUES (?, ?, ?, ?)",
    )?;
    for entry in entries {
        stmt.execute((
            &entry.pid.to_be_bytes(),
            entry.pointer.offset,
            entry.pointer.size,
            entry.pointer.tail_count,
        ))?;
    }
    Ok(())
}

fn write_metadata(
    tx: &Transaction,
    hpin: &Hpin,
    split_prefixes: &HashSet<u64>,
) -> anyhow::Result<()> {
    let mut stmt = tx.prepare_cached("INSERT INTO metadata (key, value) VALUES (?, ?)")?;
    
    // TODO: This is inefficient, find a better way to get alphabet back
    let alphabet_str: String = (0..=255)
        .filter_map(|i| if hpin.parse(&[i; hpin.n()]).is_ok() { Some(i as char) } else { None })
        .collect();
    
    stmt.execute(("alphabet", alphabet_str))?;
    stmt.execute(("n", hpin.n().to_string()))?;
    stmt.execute(("m", hpin.m().to_string()))?;

    let split_str = split_prefixes.iter().map(ToString::to_string).collect::<Vec<_>>().join(",");
    stmt.execute(("split_prefixes", split_str))?;

    Ok(())
}
