//! Every response carries an identifier, including the ones nobody planned
//! for. An identifier that appears only on success is useless for the case
//! someone actually needs to investigate, which is why the third test here
//! is the one that matters.

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use serde_json::json;
use tower::ServiceExt;

async fn request_id_of(uri: &str) -> String {
    let response = pii_detection_api::routes::rules_only_router()
        .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();
    response
        .headers()
        .get("x-request-id")
        .expect("every response carries a request id")
        .to_str()
        .unwrap()
        .to_string()
}

#[tokio::test]
async fn every_endpoint_returns_a_request_id() {
    for uri in ["/health", "/capabilities", "/metrics"] {
        let id = request_id_of(uri).await;
        assert_eq!(id.len(), 32, "{uri} returned a malformed id: {id}");
        assert!(
            id.chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        );
    }
}

#[tokio::test]
async fn two_requests_get_different_identifiers() {
    assert_ne!(
        request_id_of("/health").await,
        request_id_of("/health").await
    );
}

#[tokio::test]
async fn a_rejected_request_still_carries_a_request_id() {
    let response = pii_detection_api::routes::rules_only_router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/analyze")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({"texts": ["x"], "entities": ["BSN"], "language": "nl"}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(
        response.headers().contains_key("x-request-id"),
        "an error response must be traceable too"
    );
}
