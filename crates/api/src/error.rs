use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

/// Every rejection this service can produce.
///
/// No variant carries input text. `UnsupportedEntityType` and
/// `UnsupportedLanguage` echo a caller-supplied code, which is a parameter and
/// not content; nothing else is echoed at all.
#[derive(Debug)]
pub enum ApiError {
    UnsupportedEntityType(String),
    UnsupportedLanguage(String),
    TextTooLong,
    /// The body failed to deserialize into `AnalyzeRequest`: for example,
    /// `texts` sent as a bare string instead of an array.
    ///
    /// Carries nothing from axum's `JsonRejection`. That rejection's own
    /// message embeds the offending value verbatim (e.g. `invalid type:
    /// string "…the submitted text…", expected a sequence`), so no part of
    /// it (not a truncated excerpt, not a field name pulled out of it) may
    /// reach the response. Discarding it entirely is the only handling that
    /// is obviously safe on inspection; do not "improve" this by adding
    /// detail back.
    MalformedRequest,
    /// The service is not ready: it was configured with an artifact
    /// directory and that artifact would not verify, load, or warm up (see
    /// `crate::startup`).
    ///
    /// `503`, and never a rules-only `200`. Design section 10: *"There is
    /// no degraded rules-only mode: a partial answer is never presented as
    /// a safe one."* Temperate maps `503` to `PiiUnavailableException` and the
    /// firewall keeps failing closed, which is the behaviour a half-loaded
    /// detector must produce.
    NotReady,
    /// A detector this deployment does run failed mid-request -- an
    /// inference error, or a scan thread that did not return.
    ///
    /// Also `503`, and for the same reason `NotReady` is: the rules half
    /// may well have succeeded, and returning it alone would be exactly the
    /// partial answer presented as a safe one. Carries nothing from the
    /// underlying error; `ModelError`'s Display is name-and-count only, but
    /// this response is a wire surface and gets the fixed code regardless.
    DetectionUnavailable,
}

impl ApiError {
    /// A fixed, bounded string naming the kind of refusal.
    ///
    /// Deliberately not the value that caused it. `UnsupportedEntityType`
    /// and `UnsupportedLanguage` carry a caller-supplied code and echo it
    /// in the response body, which is safe and useful -- the caller sent
    /// it, and it goes back only to them. A metric label is a different
    /// surface: it is scraped, stored and joined for as long as the series
    /// exists, a caller can put anything in that field, and one time
    /// series per distinct value is a cardinality explosion that cannot be
    /// undone after the fact. So the label is the *kind*, and the kind is
    /// one of six strings written down here.
    ///
    /// Exhaustive by construction: adding a variant to `ApiError` fails to
    /// compile until it is named here, which is what keeps the allowed set
    /// in `tests/cardinality.rs` honest.
    pub fn reason(&self) -> &'static str {
        match self {
            Self::UnsupportedEntityType(_) => "unsupported_entity_type",
            Self::UnsupportedLanguage(_) => "unsupported_language",
            Self::TextTooLong => "text_too_long",
            Self::MalformedRequest => "malformed_request",
            Self::NotReady => "not_ready",
            Self::DetectionUnavailable => "detection_unavailable",
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, body) = match self {
            Self::UnsupportedEntityType(value) => (
                StatusCode::BAD_REQUEST,
                json!({ "error": "unsupported_entity_type", "value": value }),
            ),
            Self::UnsupportedLanguage(value) => (
                StatusCode::BAD_REQUEST,
                json!({ "error": "unsupported_language", "value": value }),
            ),
            Self::TextTooLong => (
                StatusCode::BAD_REQUEST,
                json!({ "error": "text_too_long", "limit": crate::wire::MAX_CHARACTERS }),
            ),
            Self::MalformedRequest => (
                StatusCode::BAD_REQUEST,
                json!({ "error": "malformed_request" }),
            ),
            Self::NotReady => (
                StatusCode::SERVICE_UNAVAILABLE,
                json!({ "error": "not_ready" }),
            ),
            Self::DetectionUnavailable => (
                StatusCode::SERVICE_UNAVAILABLE,
                json!({ "error": "detection_unavailable" }),
            ),
        };
        (status, Json(body)).into_response()
    }
}
