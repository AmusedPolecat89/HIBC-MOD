// In src/storage/blob.rs

use anyhow::Context;
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Seek, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use bytemuck::{Pod, Zeroable};

/// A pointer to a variable-length blob of data within a BlobStore.
///
/// This is a plain-old-data struct that is cheap to copy and pass around.
#[repr(C)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Pod, Zeroable)]
pub struct BlobPointer {
    pub offset: u64,
    pub size: u64,
}

/// A handle for writing to a BlobStore.
///
/// It uses a buffered writer wrapped in an Arc<Mutex> to allow for
/// thread-safe, concurrent appends from multiple builder threads.
#[derive(Clone)]
pub struct BlobWriter {
    writer: Arc<Mutex<BufWriter<File>>>,
}

impl BlobWriter {
    /// Creates a new BlobWriter or opens an existing one for appending.
    /// The file will be created if it does not exist.
    pub fn new(path: &Path) -> anyhow::Result<Self> {
        let file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(path)
            .with_context(|| format!("Failed to open or create blob store file: {}", path.display()))?;

        Ok(Self {
            writer: Arc::new(Mutex::new(BufWriter::new(file))),
        })
    }

    /// Appends a slice of bytes to the end of the blob store.
    ///
    /// This method is thread-safe.
    ///
    /// # Returns
    /// A `BlobPointer` indicating the location of the written data.
    pub fn append(&mut self, data: &[u8]) -> anyhow::Result<BlobPointer> {
        let mut writer_guard = self.writer.lock().unwrap();
        
        let offset = writer_guard.stream_position()?;
        let size = data.len() as u64;

        writer_guard.write_all(data)?;

        Ok(BlobPointer { offset, size })
    }
    
    /// Flushes the underlying buffered writer to ensure all data is written to disk.
    /// Should be called at the end of a build process.
    pub fn flush(&self) -> anyhow::Result<()> {
        self.writer.lock().unwrap().flush()?;
        Ok(())
    }
}

/// A handle for reading from a BlobStore.
///
/// It uses a memory map for extremely fast, zero-copy reads.
pub struct BlobReader {
    mmap: Mmap,
}

impl BlobReader {
    /// Opens an existing BlobStore for reading.
    ///
    /// # Returns
    /// An error if the file does not exist or cannot be mapped.
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("Failed to open blob store file for reading: {}", path.display()))?;

        // Safety: The caller must ensure that the file is not modified by another
        // process while this mmap is active. Our architecture ensures this by
        // separating build (write) and query (read) phases.
        let mmap = unsafe { Mmap::map(&file)? };

        Ok(Self { mmap })
    }

    /// Reads a slice of bytes from the blob store using a `BlobPointer`.
    ///
    /// # Returns
    /// A byte slice `&[u8]` pointing directly into the memory map. This is a zero-copy operation.
    pub fn read(&self, pointer: BlobPointer) -> anyhow::Result<&[u8]> {
        let start = pointer.offset as usize;
        let end = start + pointer.size as usize;

        self.mmap
            .get(start..end)
            .with_context(|| "BlobPointer out of bounds")
    }
}
