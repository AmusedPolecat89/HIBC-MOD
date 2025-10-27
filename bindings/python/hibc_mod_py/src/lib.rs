use pyo3::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::types::PyModule;
use pyo3::Bound;

// IMPORTANT: prefix with `::` so we import the Rust crate, not the Python module named `hibc_mod`.
use ::hibc_mod::engine::engine::{DataEngine, QueryResult, Document};

#[pyclass]
struct PyDataEngine {
    inner: DataEngine,
}

#[pymethods]
impl PyDataEngine {
    /// Open an existing database at `base_path`.
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let engine = DataEngine::open(std::path::Path::new(path))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: engine })
    }

    /// Search with a vector and top-k.
    ///
    /// NOTE: we return metadata as a JSON string for robustness.
    fn search(&self, vector: Vec<f32>, k: usize) -> PyResult<Vec<PyQueryResult>> {
        if vector.len() != self.inner.config.vector_dim {
            return Err(PyValueError::new_err(format!(
                "vector length {} != config.vector_dim {}",
                vector.len(), self.inner.config.vector_dim
            )));
        }
        let k = k.min(1000);
        let rs = self.inner.search(&vector, k)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(rs.into_iter().map(PyQueryResult::from).collect())
    }

    /// Get a document by ID.
    ///
    /// NOTE: we return metadata as a JSON string for robustness.
    fn get_document_by_id(&self, doc_id: &str) -> PyResult<Option<PyDocument>> {
        let r = self.inner.get_document_by_id(doc_id)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(r.map(PyDocument::from))
    }

    /// Expose loaded config (as JSON string).
    fn config_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner.config)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

#[pyclass]
struct PyQueryResult {
    #[pyo3(get)] id: String,
    #[pyo3(get)] distance: f32,
    /// JSON string of the metadata
    #[pyo3(get)] metadata_json: String,
}
impl From<QueryResult> for PyQueryResult {
    fn from(q: QueryResult) -> Self {
        let metadata_json = serde_json::to_string(&q.metadata).unwrap_or_else(|_| "null".to_string());
        Self { id: q.id, distance: q.distance, metadata_json }
    }
}

#[pyclass]
struct PyDocument {
    #[pyo3(get)] id: String,
    /// JSON string of the metadata
    #[pyo3(get)] metadata_json: String,
}
impl From<Document> for PyDocument {
    fn from(d: Document) -> Self {
        let metadata_json = serde_json::to_string(&d.metadata).unwrap_or_else(|_| "null".to_string());
        Self { id: d.id, metadata_json }
    }
}

/// PyO3 0.22+ uses `Bound<PyModule>`
#[pymodule]
fn hibc_mod(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyDataEngine>()?;
    m.add_class::<PyQueryResult>()?;
    m.add_class::<PyDocument>()?;
    Ok(())
}
