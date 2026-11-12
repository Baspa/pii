use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use serde_json::{Value, json};
use tower::ServiceExt;

/// A marker distinctive enough that any accidental echo of input text is
/// unmistakable in a failing assertion, regardless of which error path
/// leaked it.
const CANARY: &str = "canary-8f3c1a02";

async fn post(body: Value) -> (StatusCode, Value) {
    let response = pii_detection_api::routes::rules_only_router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/analyze")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

#[tokio::test]
async fn a_batch_returns_one_result_list_per_text_in_order() {
    let (status, body) = post(json!({
        "texts": ["Mail jane@example.com", "Niets hier", "Het BSN is 123456782"],
        "entities": ["EMAIL_ADDRESS", "NATIONAL_ID"],
        "language": "nl"
    }))
    .await;

    // `results` is the design's list-per-text, unchanged in shape and
    // order; it moved under an envelope so that the regime that produced it
    // travels with it. See `wire::AnalyzeResponse`.
    let results = &body["results"];
    assert_eq!(status, StatusCode::OK);
    assert_eq!(results.as_array().unwrap().len(), 3);
    assert_eq!(results[0][0]["entity_type"], "EMAIL_ADDRESS");
    assert_eq!(results[1].as_array().unwrap().len(), 0);
    assert_eq!(results[2][0]["entity_type"], "NATIONAL_ID");
    assert_eq!(results[2][0]["country"], "NL");
}

#[tokio::test]
async fn offsets_are_code_points_not_bytes() {
    let (_, body) = post(json!({
        "texts": ["Grüße an jane@example.com"],
        "entities": ["EMAIL_ADDRESS"],
        "language": "de"
    }))
    .await;

    // "Grüße an " is nine code points and eleven bytes.
    assert_eq!(body["results"][0][0]["start"], 9);
}

#[tokio::test]
async fn an_unknown_entity_type_is_rejected_rather_than_ignored() {
    let (status, body) = post(json!({
        "texts": ["x"], "entities": ["BSN"], "language": "nl"
    }))
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"], "unsupported_entity_type");
    assert_eq!(body["value"], "BSN");
}

#[tokio::test]
async fn a_type_the_taxonomy_knows_but_nothing_detects_is_rejected() {
    // `PERSON` is a real platform type and a real variant of `EntityType`. No
    // recognizer in this build looks for it, and `/capabilities` says so.
    // Accepting it here would answer a workspace migrating from a detector
    // that does find people with `200 OK` and an empty list: a success
    // response confirming no PII, where nothing ever looked. Design section 4.
    //
    // `DRIVING_LICENCE` used to be the third entry here and no longer is:
    // `packs/de.toml` and `packs/gb.toml` now detect it, so this deployment
    // advertises it and `/analyze` must accept it. That direction is asserted
    // in `a_newly_packed_type_is_accepted_once_a_recognizer_exists` below,
    // so the pair reads as one fact rather than as a deletion.
    for entity in ["PERSON", "STREET_ADDRESS"] {
        let (status, body) = post(json!({
            "texts": ["Jan de Vries"], "entities": [entity], "language": "nl"
        }))
        .await;

        assert_eq!(status, StatusCode::BAD_REQUEST, "{entity} was accepted");
        assert_eq!(body["error"], "unsupported_entity_type");
        assert_eq!(body["value"], entity);
    }
}

/// The other direction of the rule above, and the one that keeps it from
/// becoming a permanent refusal list: a type the taxonomy knows *and* a pack
/// now detects is accepted, and answers over the real text.
///
/// `DRIVING_LICENCE` is the type that changed sides. It was scaffolding --
/// an `EntityType` variant, a BIO label mapped to null, no recognizer and no
/// threshold -- and the German Führerschein and the DVLA driver number now
/// give it two. If a future edit removes those recognizers without removing
/// the type, this fails rather than quietly going back to `400`.
#[tokio::test]
async fn a_newly_packed_type_is_accepted_once_a_recognizer_exists() {
    let (status, body) = post(json!({
        "texts": ["Führerscheinnummer B072RRE2I55 liegt vor"],
        "entities": ["DRIVING_LICENCE"],
        "language": "de"
    }))
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["results"][0][0]["entity_type"], "DRIVING_LICENCE");
    // "Führerscheinnummer " is nineteen code points and twenty bytes.
    assert_eq!(body["results"][0][0]["start"], 19);
    assert_eq!(body["results"][0][0]["end"], 30);
}

#[tokio::test]
async fn every_type_capabilities_advertises_is_accepted_by_analyze() {
    // The other half of the contract: `/capabilities` is the single source of
    // truth, so nothing it advertises may be refused here.
    let response = pii_detection_api::routes::rules_only_router()
        .oneshot(
            Request::builder()
                .uri("/capabilities")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let capabilities: Value = serde_json::from_slice(&bytes).unwrap();
    let advertised: Vec<String> =
        serde_json::from_value(capabilities["entity_types"].clone()).unwrap();
    assert!(!advertised.is_empty());

    let (status, _) = post(json!({
        "texts": ["x"], "entities": advertised, "language": "auto"
    }))
    .await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn an_unknown_language_is_rejected() {
    let (status, body) = post(json!({
        "texts": ["x"], "entities": ["EMAIL_ADDRESS"], "language": "xx"
    }))
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"], "unsupported_language");
}

#[tokio::test]
async fn a_text_over_the_limit_is_refused_with_the_limit_named() {
    let (status, body) = post(json!({
        "texts": ["x".repeat(125_001)], "entities": ["EMAIL_ADDRESS"], "language": "auto"
    }))
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"], "text_too_long");
    assert_eq!(body["limit"], 125_000);
}

#[tokio::test]
async fn a_malformed_body_is_rejected_without_echoing_it() {
    // `texts` as a bare string, not an array: a plausible caller mistake
    // (sending one text unwrapped). axum's own JSON rejection would normally
    // quote the offending value verbatim in its message; the canary makes
    // any such leak unmistakable.
    let (status, body) = post(json!({
        "texts": format!("{CANARY} sent as a string instead of an array"),
        "entities": ["EMAIL_ADDRESS"],
        "language": "auto"
    }))
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body, json!({ "error": "malformed_request" }));
    assert!(
        !body.to_string().contains(CANARY),
        "a malformed-body error must not echo the offending value"
    );
}

#[tokio::test]
async fn no_error_body_repeats_the_input() {
    // Two distinct error paths that each once had, or plausibly could have,
    // a way for the offending value to reach the response: the length
    // limit (checked entirely in this crate) and a malformed body (checked
    // by axum's own extractor, upstream of `ApiError`). Guarding only one
    // path here previously left the other's leak undetected.
    let (_, too_long_body) = post(json!({
        "texts": ["x".repeat(125_001)], "entities": ["EMAIL_ADDRESS"], "language": "auto"
    }))
    .await;
    assert!(
        !too_long_body.to_string().contains("xxxx"),
        "a text-too-long error must not echo the text"
    );

    let (_, malformed_body) = post(json!({
        "texts": format!("{CANARY} sent as a string instead of an array"),
        "entities": ["EMAIL_ADDRESS"],
        "language": "auto"
    }))
    .await;
    assert!(
        !malformed_body.to_string().contains(CANARY),
        "a malformed-body error must not echo the text"
    );
}
