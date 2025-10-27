use std::collections::{HashMap, HashSet};

#[derive(Clone)]
pub struct MemEntry {
    pub vector: Vec<f32>,
    pub metadata: serde_json::Value,
    pub tombstone: bool,
    pub ts: u64,
}

#[derive(Default)]
pub struct MemTable {
    // id -> entry
    map: HashMap<String, MemEntry>,
    // tombstoned ids (fast check)
    tomb: HashSet<String>,
}

impl MemTable {
    pub fn new() -> Self { Self::default() }

    pub fn upsert(&mut self, id: String, vector: Vec<f32>, metadata: serde_json::Value, ts: u64) {
        self.map.insert(id.clone(), MemEntry { vector, metadata, tombstone: false, ts });
        self.tomb.remove(&id);
    }

    pub fn delete(&mut self, id: String, ts: u64) {
        if let Some(e) = self.map.get_mut(&id) {
            e.tombstone = true;
            e.ts = ts;
        } else {
            self.map.insert(id.clone(), MemEntry {
                vector: Vec::new(),
                metadata: serde_json::Value::Null,
                tombstone: true,
                ts,
            });
        }
        self.tomb.insert(id);
    }

    pub fn is_tombstoned(&self, id: &str) -> bool {
        self.tomb.contains(id) || self.map.get(id).map(|e| e.tombstone).unwrap_or(false)
    }

    pub fn get(&self, id: &str) -> Option<&MemEntry> {
        self.map.get(id)
    }

    pub fn clear(&mut self) {
        self.map.clear();
        self.tomb.clear();
    }

    pub fn approx_size_bytes(&self) -> usize {
        // A very rough estimate
        self.map.len() * (32 + 128) // id + vector
    }

    pub fn iter(&self) -> impl Iterator<Item=(&String, &MemEntry)> {
        self.map.iter()
    }

    pub fn apply_wal_records(&mut self, records: Vec<super::wal::WalRecord>) {
        for rec in records {
            match rec {
                super::wal::WalRecord::Upsert { id, vector, metadata, ts } => {
                    self.upsert(id, vector, metadata, ts);
                }
                super::wal::WalRecord::Delete { id, ts } => {
                    self.delete(id, ts);
                }
            }
        }
    }
}
