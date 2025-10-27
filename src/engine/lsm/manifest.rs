use serde::{Serialize, Deserialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentRef {
    pub id: String, // "seg_0001" etc.
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Manifest {
    pub active: Vec<SegmentRef>,
}

pub struct ManifestStore {
    path: PathBuf,
}

impl ManifestStore {
    pub fn new(base: &Path) -> anyhow::Result<Self> {
        Ok(Self { path: base.join("manifest.json") })
    }

    pub fn load(&self) -> anyhow::Result<Manifest> {
        if !self.path.exists() { return Ok(Manifest::default()); }
        let bytes = fs::read(&self.path)?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    pub fn store(&self, m: &Manifest) -> anyhow::Result<()> {
        let tmp = self.path.with_extension("json.tmp");
        let data = serde_json::to_vec_pretty(m)?;
        fs::write(&tmp, data)?;
        fs::rename(&tmp, &self.path)?;
        Ok(())
    }
}
