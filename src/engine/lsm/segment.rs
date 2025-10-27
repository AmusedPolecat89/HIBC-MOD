use serde::{Serialize, Deserialize};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentMeta {
    pub id: String,
    pub created: String, // ISO8601
    pub count: usize,
    pub vector_dim: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op")]
pub enum OpRecord {
    Upsert {
        id: String,
        vector: Vec<f32>,
        metadata: serde_json::Value,
        ts: u64,
    },
    Delete { id: String, ts: u64 },
}

#[derive(Clone)]
pub struct SegmentPaths {
    pub base: PathBuf,
    pub segment_json: PathBuf,
}
pub fn gen_segment_id() -> String {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis();
    format!("seg_{:016}", ts)
}

pub fn paths_for_segment(base: &Path, seg_id: &str) -> SegmentPaths {
    let seg_base = base.join("segments").join(seg_id);
    SegmentPaths {
        base: seg_base.clone(),
        segment_json: seg_base.join("segment.json"),
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StoredPayload {
    pub ts: u64,
    pub metadata: serde_json::Value,
    pub is_tombstone: bool,
}
