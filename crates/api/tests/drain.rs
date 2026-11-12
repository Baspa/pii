//! Proof for the design's graceful drain, and proof of the proof.
//!
//! Every other test in this crate drives the router in-process through
//! `tower::ServiceExt::oneshot` -- there is no real listener, no real
//! process, and nothing a `SIGTERM` could reach. That is the wrong shape
//! for what this file has to show: a drain is a property of `main.rs`'s
//! own signal wiring, not of `routes::router`, so
//! `sigterm_drains_an_in_flight_request_instead_of_cutting_it` is the one
//! test in this crate that spawns the actual compiled `pii-detection-api` binary
//! and delivers it a real `SIGTERM`. The two smaller tests below it
//! isolate the mechanism (`AppState::begin_drain`) from that wiring, so a
//! failure in one or the other points at the right half.
//!
//! WHY "IN FLIGHT" MEANS PAST THE READINESS GATE, NOT MID-UPLOAD. The
//! first version of this test held a request open by sending its headers
//! with a `Content-Length` promising more body than it actually sent.
//! That turned out to prove the wrong thing: `routes::analyze`'s
//! readiness check runs *after* axum has already finished extracting the
//! full body (extraction is a function-parameter `FromRequest`, which
//! axum always runs to completion before the handler body executes), so a
//! request still uploading when `SIGTERM` lands has not passed the
//! readiness gate yet, and correctly gets refused with `503` once it
//! finishes uploading -- the design says stop accepting *new*
//! work, and an upload still in progress is new work. That is not the
//! case this test needs to prove; the case that matters is a request that
//! *has* passed the gate and is doing real detection work when the signal
//! arrives.
//!
//! WHY THAT CASE USES A DETERMINISTIC STALL INSTEAD OF TIMING. Real
//! detection is fast enough -- p95 29.2 ms, see `pii_detection_api::shutdown`'s
//! module docs -- that racing a `kill -TERM` sent from another process
//! against it would be a race between two processes' schedulers, not a
//! deterministic sequence, and this project has enough history with tests
//! that pass without checking anything to know better than to write
//! another one. `routes::test_support::stall_if_requested` (compiled in
//! only under the `test-support` feature -- see `Cargo.toml`) sleeps for
//! `PII_DETECTION_TEST_STALL_MS` milliseconds *after* the readiness check and
//! before the rest of `/analyze` runs, so this test can hold a
//! legitimately-accepted request open for as long as it likes and know
//! exactly why.
//!
//! WHY ORDERING IS PROVEN BY A LOG LINE, NOT A SECOND LIVE HTTP CALL. A
//! probe of `/health` sent *after* `SIGTERM` would race axum's own accept
//! loop, which stops taking new connections once the shutdown signal
//! fires (`axum::serve`'s `with_graceful_shutdown`) -- a connection
//! attempt that loses that race gets no `503` to read, just a hang or a
//! reset, neither of which is evidence of anything. `main.rs` emits a
//! `drain_start` log line the instant it flips `ready` -- as the last
//! thing the shutdown future does before it returns control to axum at
//! all -- so this test waits to *see that line on the child's stdout*
//! before it reads the still-stalled request's response. Observing the
//! line is observing the flip; the response cannot arrive for another
//! `PII_DETECTION_TEST_STALL_MS`, which is what proves the flip happened well
//! before the drain completed, by construction rather than by luck.
//!
//! RULES-ONLY, DELIBERATELY. This file asserts nothing about the model,
//! so it runs `PII_DETECTION_RULES_ONLY=1` -- no artifact to fetch, ready in
//! milliseconds -- and therefore runs on every checkout, same as the
//! rest of this crate's non-gated tests. The one exception is
//! `begin_drain_leaves_model_and_calibration_populated_unlike_a_corrupt_artifact`,
//! which is artifact-gated for the reason its own doc comment gives.

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use pii_detection_api::routes::{router_and_state, rules_only_router_and_state};
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use tower::ServiceExt;

const ADDR: &str = "127.0.0.1:8090";

/// Comfortably longer than the round trip of spawning `kill(1)` and this
/// process observing the `drain_start` log line (low milliseconds in
/// practice), and comfortably shorter than `pii_detection_api::shutdown`'s
/// ten-second `DRAIN_WINDOW`, so the request finishes as a genuine drain
/// rather than being cut off by the window elapsing.
const STALL_MS: u64 = 2_000;

// ------------------------------------------------------- the real thing

#[tokio::test(flavor = "multi_thread")]
async fn sigterm_drains_an_in_flight_request_instead_of_cutting_it() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_pii-detection-api"))
        .env("PII_DETECTION_RULES_ONLY", "1")
        .env_remove("PII_DETECTION_ARTIFACT_DIR")
        .env("PII_DETECTION_TEST_STALL_MS", STALL_MS.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn pii-detection-api");
    let pid = child.id();

    // `Child`'s stdout is a blocking `Read`; forward it line by line to the
    // async test through a channel rather than blocking the runtime.
    let stdout = child.stdout.take().expect("piped stdout");
    let (lines_tx, mut lines_rx) = mpsc::unbounded_channel::<String>();
    let reader = std::thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    if lines_tx.send(line.trim_end().to_string()).is_err() {
                        break;
                    }
                }
            }
        }
    });

    wait_until_ready().await;

    // A complete, ordinary request. It passes the readiness gate
    // immediately and then stalls for `STALL_MS` inside the handler --
    // see the module docs for why that, and not a partial upload, is
    // what "in flight" has to mean here.
    let full_body = serde_json::json!({
        "texts": ["mail jane@example.com"],
        "entities": ["EMAIL_ADDRESS"],
        "language": "en"
    })
    .to_string();
    let request = format!(
        "POST /analyze HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        full_body.len(),
        full_body
    );
    let mut in_flight = TcpStream::connect(ADDR)
        .await
        .expect("connect for the in-flight request");
    in_flight
        .write_all(request.as_bytes())
        .await
        .expect("send the request");

    // Sanity: still ready, on a fresh connection, while the request above
    // is stalled inside the handler and the signal has not been sent yet.
    assert!(
        health_is_ready().await,
        "must still be ready before the signal is sent"
    );

    let status = Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .status()
        .expect("run kill(1)");
    assert!(status.success(), "kill -TERM must be delivered");

    // Wait to *see* the drain start. The in-flight request cannot possibly
    // have completed yet -- it is still mid-stall -- so observing this
    // line here is observing the flip strictly before the drain
    // completes, not merely at some point during it.
    let saw_drain_start = recv_matching(&mut lines_rx, "drain_start", Duration::from_secs(8)).await;
    assert!(
        saw_drain_start,
        "expected a drain_start log line while the in-flight request was still stalled"
    );

    // The stalled request must still complete -- not be cut off by the
    // `SIGTERM` -- once its stall ends. Bounded just under
    // `pii_detection_api::shutdown::DRAIN_WINDOW` (10s): if the drain window
    // itself elapsed first, the server would already have cut the
    // connection, so waiting longer here would not help.
    let mut response = Vec::new();
    tokio::time::timeout(Duration::from_secs(8), in_flight.read_to_end(&mut response))
        .await
        .expect("read the in-flight response before the drain window elapses")
        .expect("read the in-flight response");
    let response = String::from_utf8_lossy(&response);
    assert!(
        response.starts_with("HTTP/1.1 200"),
        "the in-flight request must complete, not be cut off by the SIGTERM: got {response:?}"
    );
    assert!(
        response.contains("EMAIL_ADDRESS"),
        "the completed response must carry the real detection result: got {response:?}"
    );

    // The process exits on its own, well inside the drain window.
    let exit_status = tokio::task::spawn_blocking(move || child.wait())
        .await
        .expect("join the wait task")
        .expect("wait on the child process");
    assert!(
        exit_status.success(),
        "a graceful exit is a clean exit: {exit_status:?}"
    );

    reader.join().expect("stdout reader thread must not panic");
}

async fn wait_until_ready() {
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if health_is_ready().await {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("pii-detection-api must become ready within 10s of being spawned");
}

async fn health_is_ready() -> bool {
    let Ok(mut stream) = TcpStream::connect(ADDR).await else {
        return false;
    };
    if stream
        .write_all(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
        .await
        .is_err()
    {
        return false;
    }
    let mut response = Vec::new();
    if stream.read_to_end(&mut response).await.is_err() {
        return false;
    }
    String::from_utf8_lossy(&response).starts_with("HTTP/1.1 200")
}

async fn recv_matching(
    rx: &mut mpsc::UnboundedReceiver<String>,
    needle: &str,
    timeout: Duration,
) -> bool {
    tokio::time::timeout(timeout, async {
        while let Some(line) = rx.recv().await {
            if line.contains(needle) {
                return true;
            }
        }
        false
    })
    .await
    .unwrap_or(false)
}

// -------------------------------------------------- the mechanism alone

/// `begin_drain` in isolation from the process/signal machinery the test
/// above proves is wired to it -- if this one fails, the bug is in
/// `AppState`, not in `main.rs`'s signal handling.
#[tokio::test]
async fn begin_drain_flips_readiness_immediately_and_only_one_way() {
    let (app, state) = rules_only_router_and_state();
    assert!(state.is_ready());
    assert_eq!(status(app.clone(), "/health").await, StatusCode::OK);
    assert_eq!(status(app.clone(), "/capabilities").await, StatusCode::OK);

    state.begin_drain();

    assert!(!state.is_ready());
    assert_eq!(
        status(app.clone(), "/health").await,
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(
        status(app, "/capabilities").await,
        StatusCode::SERVICE_UNAVAILABLE
    );

    // One-way: a second call does not "undrain" it.
    state.begin_drain();
    assert!(!state.is_ready());
}

/// Design section 4 fixes `Health`'s field list, so a draining deployment
/// cannot get a dedicated wire field saying so. What it keeps instead:
/// `model` and `calibration` are untouched by `begin_drain` (only `ready`
/// is), so a draining artifact-backed deployment still reads
/// `model: "<revision>"` where a genuinely unready one -- a missing or
/// corrupt artifact -- always reads `model: null`. This is what
/// `AppState::begin_drain`'s doc comment means by "not confusable with a
/// corrupt artifact" for the deployment that matters for it: the one
/// actually running a model.
///
/// Artifact-gated, unlike the rest of this file: the distinction only
/// exists to check on a deployment that has a model to keep populated.
#[tokio::test]
async fn begin_drain_leaves_model_and_calibration_populated_unlike_a_corrupt_artifact() {
    let model = pii_detection_model::require_model!();
    let (app, state) = router_and_state(model);

    let before = body(app.clone(), "/health").await;
    assert_eq!(before["ready"], true);
    assert!(before["model"].is_string());
    assert!(before["calibration"].is_string());

    state.begin_drain();

    let during = body(app, "/health").await;
    assert_eq!(during["ready"], false);
    assert_eq!(
        during["model"], before["model"],
        "the model revision must survive a drain -- it is what distinguishes \
         draining from a corrupt-artifact refusal, which always reads model: null"
    );
    assert_eq!(during["calibration"], before["calibration"]);
}

async fn status(app: axum::Router, path: &str) -> StatusCode {
    app.oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
        .await
        .unwrap()
        .status()
}

async fn body(app: axum::Router, path: &str) -> serde_json::Value {
    let response = app
        .oneshot(
            Request::builder()
                .uri(path)
                .header(header::ACCEPT, "application/json")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}
