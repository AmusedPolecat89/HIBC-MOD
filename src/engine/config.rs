use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub version: u32,                    // start at 1
    pub vector_dim: usize,
    pub builder_capacity_hint: usize,

    pub doc_id_key_len: usize,
    pub id_key_len: usize,

    pub ann: AnnParams,
    pub docmap: HpinParams,
    pub idmap: HpinParams,

    pub ann_build_neighbor_k: usize,     // neighbors serialized per node
    pub lsm: Option<LsmConfig>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LsmConfig {
    pub flush_threshold_bytes: usize,
    pub wal_fsync_each_write: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnParams {
    pub m: usize,
    pub ef_construction: usize,
    pub nb_layers: Option<usize>,        // None => auto
    pub ef_search: usize,                // reserved for future query impls
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HpinParams {
    pub n: usize,                        // key length
    pub m: usize,                        // tail length
    pub alphabet: AlphabetSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum AlphabetSpec {
    Utf8 { chars: String },
    ByteRange { start: u8, end: u8 },    // inclusive
}

impl EngineConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.vector_dim > 0, "vector_dim must be > 0");
        anyhow::ensure!(self.doc_id_key_len == self.docmap.n, "doc_id_key_len must equal docmap.n");
        anyhow::ensure!(self.id_key_len == self.idmap.n, "id_key_len must equal idmap.n");
        Self::validate_hpin(&self.docmap, "docmap")?;
        Self::validate_hpin(&self.idmap, "idmap")?;
        anyhow::ensure!(self.ann.m > 0, "ann.m must be > 0");
        anyhow::ensure!(self.ann.ef_construction > 0, "ann.ef_construction must be > 0");
        anyhow::ensure!(self.ann_build_neighbor_k > 0, "ann_build_neighbor_k must be > 0");
        Ok(())
    }

    fn validate_hpin(h: &HpinParams, name: &str) -> anyhow::Result<()> {
        anyhow::ensure!(h.n > 1, "{}.n must be > 1", name);
        anyhow::ensure!(h.m > 0 && h.m < h.n, "{}.m must be in [1, n-1]", name);
        Ok(())
    }

    pub fn alphabet_bytes(spec: &AlphabetSpec) -> Vec<u8> {
        match spec {
            AlphabetSpec::Utf8 { chars } => chars.as_bytes().to_vec(),
            AlphabetSpec::ByteRange { start, end } => (*start..=*end).collect(),
        }
    }
}

impl Default for AnnParams {
    fn default() -> Self {
        Self {
            m: 24,
            ef_construction: 400,
            ef_search: 100,
            nb_layers: None,
        }
    }
}

impl Default for HpinParams {
    fn default() -> Self {
        Self {
            n: 36,
            m: 30,
            alphabet: AlphabetSpec::Utf8 {
                chars: "abcdefghijklmnopqrstuvwxyz0123456789_ ".into(),
            },
        }
    }
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            version: 1,
            vector_dim: 512,
            builder_capacity_hint: 10_000,
            doc_id_key_len: 36,
            id_key_len: 8,
            ann: Default::default(),
            ann_build_neighbor_k: 10,
            docmap: Default::default(),
            idmap: HpinParams {
                n: 8,
                m: 4,
                alphabet: AlphabetSpec::ByteRange { start: 0, end: 255 },
            },
            lsm: Some(LsmConfig {
                flush_threshold_bytes: 256 * 1024,
                wal_fsync_each_write: false,
            }),
        }
    }
}
