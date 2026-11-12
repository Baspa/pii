//! The endpoint with a model behind it.
//!
//! Every test that needs the real artifact goes through `require_model!`,
//! which is `pii-detection-model`'s own gate (`crates/model/src/
//! test_support.rs`), re-exported rather than reimplemented so this suite
//! and that one cannot drift into two different skip conditions. Unset
//! `PII_DETECTION_ARTIFACT_DIR` prints a skip line the test harness cannot swallow
//! and returns; a variable that is set but does not resolve is a hard
//! failure, not a skip.
//!
//! The tests that need no artifact -- the rules-only and unready cases --
//! are here too and always run, because the distinction between them and
//! the loaded case is most of what this file is about.

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use pii_detection_api::routes::{detected_types, router, router_unready, rules_only_router};
use pii_detection_engine::entity::EntityType;
use pii_detection_model::require_model;
use serde_json::{Value, json};
use std::collections::BTreeSet;
use tower::ServiceExt;

async fn send(app: &axum::Router, request: Request<Body>) -> (StatusCode, String) {
    let response = app.clone().oneshot(request).await.expect("router responds");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

fn analyze_request(body: &Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/analyze")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .expect("request")
}

async fn post_raw(app: &axum::Router, body: Value) -> String {
    send(app, analyze_request(&body)).await.1
}

async fn post(app: &axum::Router, body: Value) -> Value {
    let (status, raw) = send(app, analyze_request(&body)).await;
    assert_eq!(status, StatusCode::OK, "expected 200, body: {raw}");
    serde_json::from_str(&raw).expect("json body")
}

async fn status_of(app: &axum::Router, body: Value) -> StatusCode {
    send(app, analyze_request(&body)).await.0
}

async fn get(app: &axum::Router, path: &str) -> Value {
    let request = Request::builder()
        .uri(path)
        .body(Body::empty())
        .expect("request");
    let (_, raw) = send(app, request).await;
    serde_json::from_str(&raw).expect("json body")
}

async fn get_status(app: &axum::Router, path: &str) -> StatusCode {
    let request = Request::builder()
        .uri(path)
        .body(Body::empty())
        .expect("request");
    send(app, request).await.0
}

fn any_analyze_body() -> Value {
    json!({
        "texts": ["jan@example.nl"],
        "entities": ["EMAIL_ADDRESS"],
        "language": "nl"
    })
}

#[tokio::test]
async fn capabilities_does_not_advertise_person_without_a_model() {
    // The honest state of a rules-only deployment, and it must stay honest:
    // a type advertised and not detected reads to a workspace as protection
    // it does not have. This is also what keeps pii-detection-eval reporting
    // NOT MEASURED rather than a fabricated 0% recall.
    let types = detected_types(None);
    assert!(!types.contains(&EntityType::Person));
    assert!(!types.contains(&EntityType::StreetAddress));
    // The rules half is still there -- this is not "detects nothing".
    assert!(types.contains(&EntityType::EmailAddress));
    assert!(types.contains(&EntityType::NationalId));
}

#[tokio::test]
async fn capabilities_advertises_person_and_street_address_with_a_model() {
    let model = require_model!();
    let types = detected_types(Some(&model));
    assert!(types.contains(&EntityType::Person));
    assert!(types.contains(&EntityType::StreetAddress));
    // And nothing else new: the model's ten null-mapped labels must not
    // widen what the service claims to detect.
    //
    // THIS USED TO BE SPELLED `assert!(!types.contains(&DrivingLicence))`,
    // because `DRIVING_LICENCE` was a BIO label in the graph, a null in
    // `labels.json`, and detected by nothing -- which made it the one label
    // that would catch a `detected_types` reading the graph's width instead
    // of the artifact's map. The packs now detect it, so that spelling would
    // fail for a reason unrelated to what it was asking about, and deleting
    // it would have taken the guard with it.
    //
    // Stated as the property instead, which is what it always meant and
    // which no future pack can invalidate: loading a model widens the set by
    // exactly the types the artifact's own map produces, and by nothing
    // else. A `detected_types` that read the graph would add eight more.
    let mut expected = detected_types(None);
    expected.extend([EntityType::Person, EntityType::StreetAddress]);
    assert_eq!(types, expected);
    // Loading a model adds types; it never removes one.
    for rule_type in detected_types(None) {
        assert!(types.contains(&rule_type));
    }
}

#[tokio::test]
async fn capabilities_reports_person_over_the_wire_too() {
    // `detected_types` is the function; this is the endpoint. A
    // `/capabilities` handler that answered from a different source than
    // `/analyze` validates against would let the two disagree.
    let app = router(require_model!());
    let body = get(&app, "/capabilities").await;
    let types: BTreeSet<String> =
        serde_json::from_value(body["entity_types"].clone()).expect("entity_types");
    assert!(types.contains("PERSON"));
    assert!(types.contains("STREET_ADDRESS"));
    // `DRIVING_LICENCE` is a rule type (`fa98768`), not a model type, so it is
    // present here too -- a model-backed deployment never loses a rule type,
    // it only gains `PERSON`/`STREET_ADDRESS` on top of them.
    assert!(types.contains("DRIVING_LICENCE"));
    assert!(body["versions"]["model"].is_string());
}

#[tokio::test]
async fn analyze_returns_a_person_span_at_code_point_offsets() {
    let app = router(require_model!());
    let body = post(
        &app,
        json!({
            "texts": ["Grüße an Jürgen Müller aus Köln."],
            "entities": ["PERSON"], "language": "de"
        }),
    )
    .await;
    let entity = &body["results"][0][0];
    assert_eq!(entity["entity_type"], "PERSON");
    assert_eq!(entity["start"], 9); // code points, not bytes
    assert_eq!(entity["end"], 22);
}

#[tokio::test]
async fn analyze_returns_rules_and_model_results_from_one_call() {
    // Design section 6 step 5: both sources merge into one candidate list.
    // A response carrying only one of them means the join never happened.
    let app = router(require_model!());
    let body = post(
        &app,
        json!({
            "texts": ["Jan de Vries, jan@example.nl, BSN 123456782"],
            "entities": ["PERSON", "EMAIL_ADDRESS", "NATIONAL_ID"], "language": "nl"
        }),
    )
    .await;
    let types: BTreeSet<&str> = body["results"][0]
        .as_array()
        .expect("a result list")
        .iter()
        .map(|entity| entity["entity_type"].as_str().expect("a type"))
        .collect();
    assert_eq!(
        types,
        BTreeSet::from(["PERSON", "EMAIL_ADDRESS", "NATIONAL_ID"])
    );
}

#[tokio::test]
async fn health_reports_the_artifact_revision_and_the_calibration_revision() {
    let app = router(require_model!());
    assert_eq!(get_status(&app, "/health").await, StatusCode::OK);
    let body = get(&app, "/health").await;
    assert_eq!(body["ready"], true);
    assert!(body["model"].is_string());
    assert!(body["rules"].is_string());
    assert!(body["calibration"].is_string());
    assert!(body["revision"].is_string());
}

#[tokio::test]
async fn a_service_that_has_not_warmed_up_refuses_traffic_rather_than_serving_rules_only() {
    // Design section 10: "There is no degraded rules-only mode: a partial
    // answer is never presented as a safe one." A configured-but-unready
    // model must produce 503, not a rules-only 200.
    let app = router_unready();
    assert_eq!(
        status_of(&app, any_analyze_body()).await,
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(
        get_status(&app, "/health").await,
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(get(&app, "/health").await["ready"], false);
    // And it does not publish a capability list it cannot honour either.
    assert_eq!(
        get_status(&app, "/capabilities").await,
        StatusCode::SERVICE_UNAVAILABLE
    );
}

#[tokio::test]
async fn a_rules_only_deployment_is_ready_and_serves_rules() {
    // The distinction that matters: no artifact configured is a valid
    // deployment; a broken artifact is not.
    let app = rules_only_router();
    assert_eq!(get_status(&app, "/health").await, StatusCode::OK);
    assert_eq!(get(&app, "/health").await["ready"], true);
    let body = post(&app, any_analyze_body()).await;
    assert_eq!(body["results"][0][0]["entity_type"], "EMAIL_ADDRESS");
}

#[tokio::test]
async fn asking_for_person_from_a_rules_only_deployment_is_refused_not_answered_empty() {
    // The existing rule, unchanged, and it must survive this task: a 200
    // with an empty list is a success response confirming no PII where
    // nothing ever looked.
    let app = rules_only_router();
    assert_eq!(
        status_of(
            &app,
            json!({
                "texts": ["Jan de Vries"], "entities": ["PERSON"], "language": "nl"
            })
        )
        .await,
        StatusCode::BAD_REQUEST
    );
}

#[tokio::test]
async fn the_response_carries_no_text_and_no_token_identifiers() {
    // The existing privacy tests in crates/api/tests/privacy.rs
    // cover the rules path; this extends the same discipline to the model
    // path, whose errors and spans are new surface.
    let app = router(require_model!());
    let raw = post_raw(
        &app,
        json!({
            "texts": ["Grüße an Jürgen Müller, Beethovenstraße 12"],
            "entities": ["PERSON", "STREET_ADDRESS"], "language": "de"
        }),
    )
    .await;
    assert!(!raw.contains("Jürgen"));
    assert!(!raw.contains("Müller"));
    assert!(!raw.contains("Beethovenstraße"));
    assert!(!raw.contains("input_ids"));
    assert!(!raw.contains("token"));
}

/// The stage histogram splits the two engines that run concurrently.
///
/// `tests/metrics.rs` asserts the four stages a rules-only deployment
/// records, and asserts that it records no `model` stage -- a deployment
/// that ran no model must not report a model latency of zero, which reads
/// as a fast model rather than an absent one. This is the other half: with
/// an artifact loaded, the model stage is there, so `rules` and `model` can
/// be compared in production rather than inferred from `total`.
#[tokio::test]
async fn the_model_stage_is_timed_separately_from_the_rules_stage() {
    let app = router(require_model!());
    let _ = post(
        &app,
        json!({
            "texts": ["Jan de Vries woont in Amsterdam"],
            "entities": ["PERSON"], "language": "nl"
        }),
    )
    .await;

    let rendered = pii_detection_api::telemetry::handle().render();
    for stage in ["rules", "model", "resolve", "total"] {
        assert!(
            rendered.contains(&format!(
                "pii_detection_stage_seconds_bucket{{stage=\"{stage}\""
            )),
            "stage {stage} missing from a deployment running a model:\n{rendered}"
        );
    }
}
