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
