// In src/storage/slab.rs

use anyhow::{anyhow, Context};
use memmap2::Mmap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// A type alias for the ID of a record within a SlabStore.
/// This is simply the 0-based index of the record.
pub type RecordId = u64;

// Magic bytes to identify a slab file, followed by the version.
const MAGIC: &[u8; 6] = b"SLABv1";
const HEADER_SIZE: u64 = 16; // 6 bytes for magic, 2 for padding, 8 for record_size

/// A handle for writing to a SlabStore.
///
/// Ensures all records are of a fixed size.
#[derive(Clone)]
pub struct SlabWriter {
    writer: Arc<Mutex<BufWriter<File>>>,
    record_size: u64,
    record_count: Arc<Mutex<u64>>,
}

impl SlabWriter {
    /// Creates a new SlabWriter. This will create a new file, overwriting any
    /// existing file at the path, and write the header.
    pub fn new(path: &Path, record_size: u64) -> anyhow::Result<Self> {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true) // Start with a fresh file
            .open(path)
            .with_context(|| format!("Failed to create slab store file: {}", path.display()))?;

        let mut writer = BufWriter::new(file);

        // Write the header
        writer.write_all(MAGIC)?;
        writer.write_all(&[0; 2])?; // 2 bytes padding for alignment
        writer.write_all(&record_size.to_be_bytes())?;
        assert_eq!(writer.stream_position()?, HEADER_SIZE);

        Ok(Self {
            writer: Arc::new(Mutex::new(writer)),
            record_size,
            record_count: Arc::new(Mutex::new(0)),
        })
    }

    /// Appends a new record to the slab.
    ///
    /// This method is thread-safe.
    ///
    /// # Errors
    /// Returns an error if the provided data's length does not match the
    /// fixed `record_size`.
    ///
    /// # Returns
    /// The `RecordId` of the newly appended record.
    pub fn append(&mut self, data: &[u8]) -> anyhow::Result<RecordId> {
        if data.len() as u64 != self.record_size {
            return Err(anyhow!(
                "Invalid record size. Expected {}, got {}",
                self.record_size,
                data.len()
            ));
        }

        let mut writer_guard = self.writer.lock().unwrap();
        writer_guard.write_all(data)?;

        let mut count_guard = self.record_count.lock().unwrap();
        let record_id = *count_guard;
        *count_guard += 1;

        Ok(record_id)
    }
    
    /// Flushes the underlying writer to disk.
    pub fn flush(&self) -> anyhow::Result<()> {
        self.writer.lock().unwrap().flush()?;
        Ok(())
    }
}

/// A handle for reading from a SlabStore.
///
/// Uses a memory map for fast random access reads.
pub struct SlabReader {
    mmap: Mmap,
    record_size: u64,
    header_offset: u64,
}

impl SlabReader {
    /// Opens an existing SlabStore for reading.
    ///
    /// Reads and validates the header to ensure it's a compatible file.
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        let mut file = File::open(path)
            .with_context(|| format!("Failed to open slab store file for reading: {}", path.display()))?;

        // Read and validate the header
        let mut header_buf = [0u8; HEADER_SIZE as usize];
        file.read_exact(&mut header_buf)?;

        if &header_buf[0..6] != MAGIC {
            return Err(anyhow!("Invalid magic bytes. Not a SLABv1 file."));
        }
        
        let record_size = u64::from_be_bytes(header_buf[8..16].try_into().unwrap());
        if record_size == 0 {
            return Err(anyhow!("Record size cannot be zero."));
        }

        // We map the *entire* file, including the header.
        file.seek(SeekFrom::Start(0))?;
        let mmap = unsafe { Mmap::map(&file)? };

        Ok(Self {
            mmap,
            record_size,
            header_offset: HEADER_SIZE,
        })
    }

    /// Reads a single record from the slab by its `RecordId`.
    ///
    /// # Returns
    /// A byte slice `&[u8]` pointing directly into the memory map.
    pub fn read(&self, id: RecordId) -> anyhow::Result<&[u8]> {
        let start = self.header_offset as usize + (id * self.record_size) as usize;
        let end = start + self.record_size as usize;

        self.mmap
            .get(start..end)
            .with_context(|| format!("RecordId {} is out of bounds", id))
    }
    
    /// Returns the number of records in the slab.
    pub fn len(&self) -> u64 {
        (self.mmap.len() as u64 - self.header_offset) / self.record_size
    }

    /// Returns true if the slab contains no records.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
