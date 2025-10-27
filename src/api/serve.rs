use axum::{
    extract::{Path, State},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, sync::Arc};
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use crate::engine::engine::{DataEngine, QueryResult, Document};

#[derive(Clone)]
pub struct AppState {
    engine: Arc<DataEngine>,
}

#[derive(Deserialize)]
struct SearchPayload {
    vector: Vec<f32>,
    k: usize,
}

#[derive(Serialize)]
struct SearchResponse {
    results: Vec<QueryResult>,
}

pub async fn serve(db_path: std::path::PathBuf, bind: SocketAddr) -> anyhow::Result<()> {
    let engine = Arc::new(DataEngine::open(&db_path)?);
    let state = AppState { engine };

    let app = Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .route("/config", get(get_config))
        .route("/search", post(search_handler))
        .route("/document/:id", get(get_document))
        .with_state(state)
        .layer(
            CorsLayer::permissive() // adjust in prod
        )
        .layer(
            TraceLayer::new_for_http()
        );

    let listener = tokio::net::TcpListener::bind(bind).await?;
    log::info!("Listening on http://{}", listener.local_addr()?);

    let serve_fut = axum::serve(listener, app);
    // Graceful shutdown (Ctrl+C)
    let graceful = serve_fut.with_graceful_shutdown(async {
        let _ = tokio::signal::ctrl_c().await;
    });
    graceful.await?;
    Ok(())
}

async fn get_config(State(state): State<AppState>) -> Json<serde_json::Value> {
    // redact nothing for now; you can redact in the future
    Json(serde_json::to_value(&state.engine.config).unwrap())
}

async fn search_handler(
    State(state): State<AppState>,
    Json(payload): Json<SearchPayload>,
) -> Result<Json<SearchResponse>, (axum::http::StatusCode, String)> {
    // Validate inputs
    if payload.vector.len() != state.engine.config.vector_dim {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            format!("vector length {} != {}", payload.vector.len(), state.engine.config.vector_dim),
        ));
    }
    let k = payload.k.min(1000);

    state.engine
        .search(&payload.vector, k)
        .map(|results| Json(SearchResponse { results }))
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

async fn get_document(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Document>, (axum::http::StatusCode, String)> {
    match state.engine.get_document_by_id(&id) {
        Ok(Some(doc)) => Ok(Json(doc)),
        Ok(None) => Err((axum::http::StatusCode::NOT_FOUND, "not found".into())),
        Err(e) => Err((axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string())),
    }
}
