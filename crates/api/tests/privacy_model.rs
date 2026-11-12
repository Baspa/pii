//! The canary run, extended to the model path.
//!
//! `tests/privacy.rs` proves the rules path logs nothing about the text it
//! analysed. The model path is new surface: it tokenizes, it batches, it
//! decodes label ids back to spans, and every one of those steps holds
//! something that must never reach a log line -- the raw text, a detected
//! value, or a token identifier, which is the text in a different alphabet.
//!
//! WHY THIS IS ITS OWN TEST BINARY. It installs a *global* tracing
//! subscriber, not a thread-local one. `tracing::subscriber::set_default` is
//! per-thread, and the model does its work on `spawn_blocking` threads, so a
//! thread-local sink would capture nothing and pass no matter what got
//! logged -- exactly the "test that claims more than it checks" this project
//! keeps finding. A global subscriber can only be installed once per
//! process, so this file holds one test and nothing else.

use axum::body::Body;
use axum::http::{Request, header};
use pii_detection_api::routes::router;
use pii_detection_model::require_model;
use serde_json::json;
use tower::ServiceExt;
use tracing_subscriber::layer::SubscriberExt;

/// A name and a street, both distinctive enough that any accidental echo is
/// unmistakable, and both types the model actually produces -- so this text
/// exercises the decode path rather than only the tokenize path.
const CANARY_NAME: &str = "Zorbtastic Vandelay";
const CANARY_STREET: &str = "Quibblewurtzstraat 41";

#[tokio::test]
async fn neither_the_logs_nor_the_response_carry_the_text_or_a_token_identifier() {
    let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
    let sink = std::sync::Arc::clone(&captured);
    let subscriber = tracing_subscriber::registry().with(
        tracing_subscriber::fmt::layer()
            .json()
            .with_writer(move || CanaryWriter(std::sync::Arc::clone(&sink))),
    );
    // Installed before `require_model!` on purpose: loading and warming up
    // the model is itself a logging path, and the artifact loader's own
    // error lines are part of what this asserts is text-free.
    tracing::subscriber::set_global_default(subscriber).expect("global subscriber");

    let app = router(require_model!());
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/analyze")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "texts": [format!("{CANARY_NAME} woont op {CANARY_STREET}, BSN 123456782")],
                        "entities": ["PERSON", "STREET_ADDRESS", "NATIONAL_ID"],
                        "language": "nl"
                    })
                    .to_string(),
                ))
                .expect("request"),
        )
        .await
        .expect("router responds");

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    let body = String::from_utf8_lossy(&body).into_owned();
    let logs = String::from_utf8_lossy(&captured.lock().expect("sink").clone()).into_owned();

    for (label, haystack) in [("the logs", &logs), ("the response body", &body)] {
        for needle in [CANARY_NAME, CANARY_STREET, "Zorbtastic", "Quibblewurtz"] {
            assert!(!haystack.contains(needle), "{label} carried analysed text");
        }
        assert!(
            !haystack.contains("123456782"),
            "{label} carried a detected value"
        );
        for needle in ["input_ids", "token_ids", "attention_mask", "get_tokens"] {
            assert!(!haystack.contains(needle), "{label} named a token field");
        }
        assert!(
            !looks_like_a_token_id_array(haystack),
            "{label} carried something shaped like a token id array"
        );
    }

    // The run has to have actually detected something, or the assertions
    // above are a tautology over an empty response.
    let parsed: serde_json::Value = serde_json::from_str(&body).expect("json body");
    let found: Vec<&str> = parsed["results"][0]
        .as_array()
        .expect("a result list")
        .iter()
        .filter_map(|entity| entity["entity_type"].as_str())
        .collect();
    assert!(
        found.contains(&"PERSON"),
        "the model found nothing, so this test proved nothing"
    );
}

/// Whether `text` contains a run of at least four comma-separated integers,
/// which is what a leaked `input_ids` vector looks like once `Debug` or
/// `Display` has had it. Deliberately shape-based rather than a search for
/// the field name: a future `tracing::debug!(?ids)` would print the numbers
/// under whatever name the local variable happened to have.
///
/// Four rather than three: `[start, end, score]` triples and version
/// strings are legitimate, and a token id vector is never that short.
fn looks_like_a_token_id_array(text: &str) -> bool {
    let bytes = text.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if !bytes[index].is_ascii_digit() {
            index += 1;
            continue;
        }
        let mut run = 0;
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            while index < bytes.len() && bytes[index].is_ascii_digit() {
                index += 1;
            }
            run += 1;
            let mut lookahead = index;
            if lookahead < bytes.len() && bytes[lookahead] == b',' {
                lookahead += 1;
                while lookahead < bytes.len() && bytes[lookahead] == b' ' {
                    lookahead += 1;
                }
                if lookahead < bytes.len() && bytes[lookahead].is_ascii_digit() {
                    index = lookahead;
                    continue;
                }
            }
            break;
        }
        if run >= 4 {
            return true;
        }
    }
    false
}

struct CanaryWriter(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

impl std::io::Write for CanaryWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if let Ok(mut sink) = self.0.lock() {
            sink.extend_from_slice(buf);
        }
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod scanner_tests {
    use super::looks_like_a_token_id_array;

    #[test]
    fn a_token_id_vector_is_caught() {
        assert!(looks_like_a_token_id_array(
            r#"{"ids":[0, 4865, 219, 76, 2]}"#
        ));
    }

    #[test]
    fn ordinary_numbers_are_not() {
        // A span, a score, a version, a timestamp: none of these is a
        // leaked token vector, and a scanner that flagged them would be
        // turned off within a week.
        assert!(!looks_like_a_token_id_array(
            r#"{"start":9,"end":22,"score":0.97,"rules":"1.0.3"}"#
        ));
        assert!(!looks_like_a_token_id_array("2026-08-22T10:00:00Z"));
    }
}
