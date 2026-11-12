//! The canary run: analyse text carrying a distinctive value and assert
//! that no log line contains it. The spec forbids raw input, detected
//! values and token text in telemetry; this is the test that keeps that
//! true for the rules path.
//!
//! WHY THIS IS ITS OWN TEST BINARY, AND WHY THE SUBSCRIBER IS GLOBAL. Two
//! reasons, and both were learned here rather than assumed.
//!
//! A thread-local sink (`set_default`) is not enough. The rules scan runs
//! on a `spawn_blocking` thread, so anything logged from inside it would
//! land on a thread that sink never sees, and the test would pass by
//! missing it -- the same trap `tests/privacy_model.rs` documents for the
//! model path.
//!
//! And a subscriber installed inside one test does not protect a callsite
//! that another test in the same binary reached first. `tracing` caches a
//! callsite's interest globally the first time it is hit: a sibling test
//! running in parallel with no subscriber installed can get there first,
//! have the callsite cached as uninteresting, and silently disable the
//! event this test exists to inspect. That is not a hypothesis -- writing
//! this test with a thread-local sink and a sibling in the same file
//! failed four runs in ten, always by capturing nothing at all. A global
//! subscriber can only be installed once per process, so this file holds
//! one test and nothing else; the metric-label half of the canary lives in
//! `tests/privacy_metrics.rs`.

use axum::body::Body;
use axum::http::{Request, header};
use serde_json::json;
use tower::ServiceExt;
use tracing_subscriber::layer::SubscriberExt;

#[tokio::test]
async fn no_log_line_contains_the_analysed_text() {
    let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
    let sink = captured.clone();

    let subscriber = tracing_subscriber::registry().with(
        tracing_subscriber::fmt::layer()
            .json()
            .with_writer(move || CanaryWriter(sink.clone())),
    );
    // Global, not thread-local: see this file's module docs.
    tracing::subscriber::set_global_default(subscriber).expect("global subscriber");

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

    let logs = String::from_utf8_lossy(&captured.lock().unwrap()).to_string();
    assert!(!logs.contains(canary), "the canary reached the logs");
    assert!(
        !logs.contains("123456782"),
        "a detected value reached the logs"
    );

    // What replaced the tripwire, and why.
    //
    // This test used to end by asserting the captured sink was *empty*:
    // the service logged nothing on the request path, so the two canary
    // assertions above could not fail, and the tripwire said so out loud
    // rather than letting them read as protection nobody had earned. It
    // was written to fail the day telemetry landed, so that nobody could
    // add logging under cover of a privacy test that could not fail.
    //
    // Telemetry has landed, so the tripwire is gone -- but its job is not.
    // The assertions below are what took it over: they prove this sink
    // captured the request path's own log event, which is what makes the
    // two assertions above load-bearing. Without them the test would go
    // quietly back to passing the moment the event stopped being emitted,
    // was routed to a different subscriber, or moved to a thread this
    // thread-local sink never sees -- which is precisely the failure the
    // tripwire existed to prevent. Keep them, and keep them positive.
    assert!(
        !logs.is_empty(),
        "the request path emitted nothing, so the canary assertions above \
         proved nothing -- see the comment here before changing this test"
    );
    assert!(
        logs.contains("\"texts\":1"),
        "the request event should record a text count:\n{logs}"
    );
    assert!(
        logs.contains("\"characters\":"),
        "the request event should record a character count:\n{logs}"
    );
    assert!(
        logs.contains("request_id"),
        "the request event should carry a request id:\n{logs}"
    );
}

struct CanaryWriter(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

impl std::io::Write for CanaryWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
