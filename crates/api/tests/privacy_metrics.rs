//! The metric-label half of the canary.
//!
//! Its own binary because `tests/privacy.rs` installs a *global* tracing
//! subscriber and can hold only one test; see that file's module docs.

use axum::body::Body;
use axum::http::{Request, header};
use serde_json::json;
use tower::ServiceExt;

/// The metrics surface is telemetry too, and the design forbids raw input in
/// metric labels by name. A canary that only reads logs would miss it.
#[tokio::test]
async fn no_metric_label_contains_caller_input() {
    let canary = "Zorbtastic Vandelay";

    let _ = pii_detection_api::routes::rules_only_router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/analyze")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "texts": [format!("{canary} met BSN 123456782")],
                        "entities": ["NATIONAL_ID"],
                        "language": "nl"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    let rendered = pii_detection_api::telemetry::handle().render();
    assert!(
        !rendered.contains(canary),
        "the canary reached a metric label:\n{rendered}"
    );
    assert!(
        !rendered.contains("123456782"),
        "a detected value reached a metric label:\n{rendered}"
    );
}
