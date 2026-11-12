use axum::body::Body;
use axum::http::Request;
use serde_json::Value;
use tower::ServiceExt;

/// Every test in this file runs a *rules-only* deployment
/// (`rules_only_router()`) -- the configuration a container with no
/// `PII_DETECTION_ARTIFACT_DIR` serves. The model-loaded half of the same
/// questions lives in `tests/model.rs`, which is artifact-gated;
/// keeping them apart is what lets this file run on a clean checkout and
/// still mean something.
async fn get(path: &str) -> Value {
    let response = pii_detection_api::routes::rules_only_router()
        .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

#[tokio::test]
async fn capabilities_reports_only_types_a_recognizer_actually_detects() {
    let body = get("/capabilities").await;
    let types: Vec<String> = serde_json::from_value(body["entity_types"].clone()).unwrap();

    assert!(types.contains(&"EMAIL_ADDRESS".to_string()));
    assert!(types.contains(&"NATIONAL_ID".to_string()));
    // The packs detect the driving licence now -- the German Führerschein
    // and the DVLA driver number -- so a rules-only deployment advertises it.
    // It was the standing example of a type the enum carried and nothing
    // looked for; `PERSON` and `STREET_ADDRESS` are the remaining ones here.
    assert!(types.contains(&"DRIVING_LICENCE".to_string()));

    // No model loaded. Advertising either of these would tell a workspace it
    // is protected when nothing looks for it. `tests/model.rs` asserts the
    // other half: with an artifact loaded, PERSON and STREET_ADDRESS appear
    // and nothing else does.
    assert!(!types.contains(&"PERSON".to_string()));
    assert!(!types.contains(&"STREET_ADDRESS".to_string()));
}

#[tokio::test]
async fn capabilities_lists_all_twenty_four_accepted_languages() {
    let body = get("/capabilities").await;
    let accepted: Vec<String> =
        serde_json::from_value(body["languages"]["accepted"].clone()).unwrap();
    assert_eq!(accepted.len(), 24);
    // The distinction this block exists for: `accepted` answers "will you
    // take this code", not "is the model any good at it". Polish is accepted
    // by every deployment and trained by none of the shipped artifacts.
    assert!(accepted.contains(&"pl".to_string()));
}

#[tokio::test]
async fn a_rules_only_deployment_claims_no_trained_language_and_no_model_only_type() {
    // The honest state of a deployment with no model, and the one that makes
    // `model_trained` readable at all: it is empty because there is no
    // artifact to have been trained, not because the field is decorative. A
    // list of six hardcoded in Rust would appear here too, on a deployment
    // running no model whatsoever.
    let body = get("/capabilities").await;
    let trained: Vec<String> =
        serde_json::from_value(body["languages"]["model_trained"].clone()).unwrap();
    assert!(
        trained.is_empty(),
        "a deployment with no model claimed trained languages: {trained:?}"
    );

    let model_only: Vec<String> =
        serde_json::from_value(body["languages"]["model_only_entity_types"].clone()).unwrap();
    assert!(
        model_only.is_empty(),
        "a deployment with no model named model-only types: {model_only:?}"
    );
}

#[tokio::test]
async fn health_is_ready_once_the_packs_compile() {
    let body = get("/health").await;
    assert_eq!(body["ready"], true);
    // A rules-only deployment is a valid deployment, and it says so: no
    // model revision, no calibration revision, and ready anyway. A broken
    // artifact is the case that must not be ready -- see
    // `tests/readiness.rs`.
    assert!(body["model"].is_null());
    assert!(body["calibration"].is_null());
    assert!(body["rules"].is_string());
    assert!(body["revision"].is_string());
}
