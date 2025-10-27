use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op")]
pub enum WalRecord {
    Upsert {
        id: String,
        vector: Vec<f32>,
        metadata: serde_json::Value,
        ts: u64,
    },
    Delete { id: String, ts: u64 },
}

// --- add this just below WalRecord ---
#[derive(Deserialize)]
struct LegacyUpsert {
    id: String,
    vector: Vec<f32>,
    metadata: serde_json::Value,
}

pub struct WalWriter {
    f: File,
    #[allow(dead_code)]
    path: PathBuf,
}

impl WalWriter {
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        let f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("open wal for append: {}", path.display()))?;
        Ok(Self {
            f,
            path: path.to_path_buf(),
        })
    }

    pub fn append(&mut self, rec: &WalRecord) -> anyhow::Result<()> {
        let line = serde_json::to_string(rec)? + "\n";
        self.f.write_all(line.as_bytes())?;
        self.f.flush()?; // MVP: basic durability
        Ok(())
    }

    pub fn sync(&mut self) -> anyhow::Result<()> {
        // Ensure bytes are flushed to disk (durability)
        self.f.sync_all()?;
        Ok(())
    }
}

pub struct WalReader {
    f: File,
}

impl WalReader {
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        let f = OpenOptions::new()
            .read(true)
            .open(path)
            .with_context(|| format!("open wal for read: {}", path.display()))?;
        Ok(Self { f })
    }

    pub fn read_all(&mut self) -> anyhow::Result<Vec<WalRecord>> {
        let mut out = Vec::new();
        let rdr = BufReader::new(&self.f);
        for line in rdr.lines() {
            let l = line?;
            if l.trim().is_empty() {
                continue;
            }

            // Try current tagged format first
            if let Ok(rec) = serde_json::from_str::<WalRecord>(&l) {
                out.push(rec);
                continue;
            }

            // Fallback: legacy implicit upsert line
            if let Ok(legacy) = serde_json::from_str::<LegacyUpsert>(&l) {
                out.push(WalRecord::Upsert {
                    id: legacy.id,
                    vector: legacy.vector,
                    metadata: legacy.metadata,
                    ts: 0, // or a parsed/now timestamp if you prefer
                });
                continue;
            }

            // If neither matched, surface a clear error
            anyhow::bail!("invalid WAL line (neither tagged nor legacy upsert): {}", l);
        }
        Ok(out)
    }

    pub fn read_and_collapse(&mut self) -> anyhow::Result<Vec<WalRecord>> {
        let mut records = std::collections::HashMap::new();
        let rdr = BufReader::new(&self.f);
        for line in rdr.lines() {
            let l = line?;
            if l.trim().is_empty() {
                continue;
            }
            let rec: WalRecord = serde_json::from_str(&l)?;
            let id = match &rec {
                WalRecord::Upsert { id, .. } => id,
                WalRecord::Delete { id, .. } => id,
            };
            records.insert(id.clone(), rec);
        }
        Ok(records.into_values().collect())
    }
}
