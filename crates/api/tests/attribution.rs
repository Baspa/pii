//! Phone attribution against the model: every number that could be a person's stays masked.

use axum::body::Body;
use axum::http::{Request, header};
use pii_detection_api::routes::router;
use pii_detection_engine::entity::EntityType;
use pii_detection_engine::text::Document;
use pii_detection_model::require_model;
use pii_detection_rules::matcher::Matcher;
use serde::Deserialize;
use serde_json::{Value, json};
use tower::ServiceExt;

#[derive(Deserialize)]
struct Case {
    id: String,
    lang: String,
    expect: String,
    phone: String,
    text: String,
}

fn cases(file: &str) -> Vec<Case> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmark/attribution")
        .join(file);
    serde_json::from_str(&std::fs::read_to_string(&path).expect("fixture")).expect("fixture parses")
}

/// Every char offset in `text` where `phone` starts.
fn occurrences(text: &str, phone: &str) -> Vec<usize> {
    let chars: Vec<char> = text.chars().collect();
    let phone: Vec<char> = phone.chars().collect();
    (0..=chars.len() - phone.len())
        .filter(|&at| chars[at..at + phone.len()] == phone[..])
        .collect()
}

/// Whether every occurrence of the case's number is inside a returned PHONE_NUMBER span.
async fn masked(app: &axum::Router, case: &Case) -> bool {
    let body = json!({"texts": [case.text], "entities": ["PHONE_NUMBER", "PERSON"], "language": case.lang});
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/analyze")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))
                .expect("request"),
        )
        .await
        .expect("router responds");
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    let answer: Value = serde_json::from_slice(&bytes).expect("json");
    let spans: Vec<(usize, usize)> = answer["results"][0]
        .as_array()
        .expect("results")
        .iter()
        .filter(|entity| entity["entity_type"] == "PHONE_NUMBER")
        .map(|entity| {
            (
                entity["start"].as_u64().unwrap() as usize,
                entity["end"].as_u64().unwrap() as usize,
            )
        })
        .collect();
    let phone_len = case.phone.chars().count();
    let found = occurrences(&case.text, &case.phone);
    assert!(
        !found.is_empty(),
        "{}: the phone value is not in the text",
        case.id
    );
    found.iter().all(|&at| {
        spans
            .iter()
            .any(|&(start, end)| start <= at && at + phone_len <= end)
    })
}

/// Whether the rules recognizer, built exactly as the service builds it, finds a `PHONE_NUMBER`
/// span over the case's number that clears the resolver's threshold, before attribution runs.
fn detected(matcher: &Matcher, case: &Case) -> bool {
    let document = Document::new(case.text.clone());
    let bar =
        pii_detection_resolve::rule_threshold(EntityType::PhoneNumber).expect("a phone threshold");
    let spans: Vec<(usize, usize)> = matcher
        .scan(&document)
        .into_iter()
        .filter(|candidate| {
            candidate.entity_type == EntityType::PhoneNumber && candidate.score >= bar
        })
        .map(|candidate| (candidate.span.start as usize, candidate.span.end as usize))
        .collect();
    let phone_len = case.phone.chars().count();
    occurrences(&case.text, &case.phone).iter().all(|&at| {
        spans
            .iter()
            .any(|&(start, end)| start <= at && at + phone_len <= end)
    })
}

async fn check(file: &str) {
    let app = router(require_model!());
    let matcher = Matcher::new();
    let mut leaks = Vec::new();
    let mut cleared = 0;
    let mut undetected = 0;
    let mut may_clear = 0;
    for case in cases(file) {
        let masked = masked(&app, &case).await;
        match case.expect.as_str() {
            "mask" if !masked => leaks.push(case.id),
            "may_clear" => {
                may_clear += 1;
                if !masked {
                    if detected(&matcher, &case) {
                        cleared += 1;
                    } else {
                        undetected += 1;
                    }
                }
            }
            _ => {}
        }
    }
    eprintln!("{file}: {cleared} of {may_clear} company-only numbers cleared");
    eprintln!("{file}: {undetected} of {may_clear} company-only numbers not detected");
    assert!(
        leaks.is_empty(),
        "{file}: numbers that could be a person's were cleared: {leaks:?}"
    );
}

#[tokio::test]
async fn hand_written_cases_leak_nothing() {
    check("phone-attribution-hand.json").await;
}

#[tokio::test]
async fn gretel_documents_leak_nothing() {
    check("gretel.json").await;
}
