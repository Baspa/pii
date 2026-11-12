//! A privacy-safe identifier on every response.

use axum::extract::Request;
use axum::middleware::Next;
use axum::response::Response;

/// A per-request identifier, safe to log and safe to return.
///
/// Random rather than derived from anything about the request, so it can
/// never encode a property of the text. It is deliberately NOT a metric
/// label: as a unique value per request it would give every series unbounded
/// cardinality, and metrics are the one surface where that is unrecoverable.
#[derive(Clone, Debug)]
pub struct RequestId(String);

impl RequestId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().simple().to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for RequestId {
    fn default() -> Self {
        Self::new()
    }
}

/// Attaches an identifier to the request's extensions and to the response
/// header, for every route and every status.
///
/// Applied as a layer over the whole router rather than inside a handler, so
/// a response the handlers never see -- a `404`, a `405`, a body axum
/// rejected before extraction -- carries one too.
pub async fn attach(mut request: Request, next: Next) -> Response {
    let id = RequestId::new();
    request.extensions_mut().insert(id.clone());

    let mut response = next.run(request).await;
    if let Ok(value) = axum::http::HeaderValue::from_str(id.as_str()) {
        response.headers_mut().insert("x-request-id", value);
    }
    response
}
