//! Names and addresses in hand-written business text stay masked, and company, product and place
//! names stay clear.

use axum::body::Body;
use axum::http::{Request, header};
use pii_detection_api::routes::router;
use pii_detection_model::require_model;
use serde::Deserialize;
use serde_json::{Value, json};
use tower::ServiceExt;

#[derive(Deserialize)]
struct Masked {
    value: String,
}

#[derive(Deserialize)]
struct Case {
    id: String,
    lang: String,
    text: String,
    must_mask: Vec<Masked>,
    must_clear: Vec<String>,
    #[serde(default)]
    known_misses: Vec<String>,
    #[serde(default)]
    known_over_masks: Vec<String>,
}

fn cases() -> Vec<Case> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmark/business/business-hand.json");
    serde_json::from_str(&std::fs::read_to_string(&path).expect("fixture")).expect("fixture parses")
}

/// Every code-point range in `text` where `value` occurs.
fn occurrences(text: &str, value: &str) -> Vec<(usize, usize)> {
    let chars: Vec<char> = text.chars().collect();
    let value: Vec<char> = value.chars().collect();
    (0..=chars.len() - value.len())
        .filter(|&at| chars[at..at + value.len()] == value[..])
        .map(|at| (at, at + value.len()))
        .collect()
}

async fn spans(app: &axum::Router, case: &Case) -> Vec<(usize, usize)> {
    let body = json!({"texts": [case.text], "entities": ["PERSON", "STREET_ADDRESS"], "language": case.lang});
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
    answer["results"][0]
        .as_array()
        .expect("results")
        .iter()
        .map(|entity| {
            (
                entity["start"].as_u64().unwrap() as usize,
                entity["end"].as_u64().unwrap() as usize,
            )
        })
        .collect()
}

#[tokio::test]
async fn names_and_addresses_are_masked_and_business_names_are_not() {
    let app = router(require_model!());
    let mut leaks = Vec::new();
    let mut over_masks = Vec::new();
    for case in cases() {
        let found = spans(&app, &case).await;
        let chars: Vec<char> = case.text.chars().collect();
        for must in &case.must_mask {
            let places = occurrences(&case.text, &must.value);
            assert!(
                !places.is_empty(),
                "{}: {} is not in the text",
                case.id,
                must.value
            );
            let masked = places.iter().all(|&(start, end)| {
                (start..end).all(|at| {
                    chars[at].is_whitespace() || found.iter().any(|&(s, e)| s <= at && at < e)
                })
            });
            if !masked && !case.known_misses.contains(&must.value) {
                leaks.push(format!("{}: {}", case.id, must.value));
            }
        }
        for clear in &case.must_clear {
            let touched = occurrences(&case.text, clear)
                .iter()
                .any(|&(start, end)| found.iter().any(|&(s, e)| s < end && start < e));
            if touched && !case.known_over_masks.contains(clear) {
                over_masks.push(format!("{}: {}", case.id, clear));
            }
        }
    }
    assert!(
        leaks.is_empty(),
        "names or addresses left unmasked: {leaks:?}"
    );
    assert!(
        over_masks.is_empty(),
        "business names masked: {over_masks:?}"
    );
}
