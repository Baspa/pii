#[tokio::main]
async fn main() -> Result<(), std::io::Error> {
    if std::env::args().any(|argument| argument == "--health-check") {
        // The embedded packs are fixed at compile time, so any binary that
        // built at all reports them non-empty: that answers nothing about
        // whether the server is actually up. Probe the live listener
        // instead: connect, ask for /health, and require a 200 within the
        // timeout, so a bound-but-wedged or never-bound server fails this.
        // A 200 here requires ready: true -- see probe_health.
        return match tokio::time::timeout(std::time::Duration::from_secs(1), probe_health()).await {
            Ok(Ok(())) => Ok(()),
            _ => Err(std::io::Error::other("health probe failed")),
        };
    }

    tracing_subscriber::fmt().json().init();
    // Built before the listener binds, so nothing is ever served by a
    // process that has not yet decided whether it is ready. A broken -- or
    // unconfigured -- artifact still binds and serves 503 from every
    // endpoint, because the design refuses readiness rather than
    // exiting. Rules-only is `PII_DETECTION_RULES_ONLY=1` and nothing else; a
    // missing `PII_DETECTION_ARTIFACT_DIR` is a refusal, not a mode. See
    // `pii_detection_api::startup`'s module docs for the deployment accident that
    // ordering exists to prevent.
    let (router, app_state) = pii_detection_api::startup::router_and_state_from_env().await;
    let listener = tokio::net::TcpListener::bind(listen_address()).await?;

    // Design section 22's graceful drain. `axum::serve`'s own
    // `with_graceful_shutdown` stops the accept loop and waits out
    // already-accepted connections once its `signal` future resolves; the
    // future below flips `AppState::ready` to `false` as the *last* thing
    // it does before resolving, so that refusal is visible on `/health`
    // strictly before axum stops accepting or starts draining a single
    // connection -- see `pii_detection_api::routes::AppState::begin_drain` and
    // `pii_detection_api::shutdown`'s module docs for why this waits on
    // connections rather than on the model's inference semaphore.
    //
    // That alone waits unboundedly, which a container with a stop-timeout
    // before `SIGKILL` cannot afford, so it is raced here against
    // `drain_deadline`: a second, independent listener for the same signal
    // (see `shutdown::wait_for_shutdown_signal`'s docs) that only starts
    // counting `shutdown::DRAIN_WINDOW` once a signal actually arrives, and
    // never before. Whichever finishes first wins; if the deadline wins,
    // the still-pending `serve` future is dropped, which is this process's
    // own bounded exit rather than an orchestrator's `SIGKILL`.
    let drain_state = app_state.clone();
    let serve = axum::serve(listener, router)
        .with_graceful_shutdown(async move {
            pii_detection_api::shutdown::wait_for_shutdown_signal().await;
            tracing::warn!(
                event = "drain_start",
                window_secs = pii_detection_api::shutdown::DRAIN_WINDOW.as_secs(),
                "shutdown signal received; refusing new work and draining in-flight requests"
            );
            drain_state.begin_drain();
        })
        .into_future();
    tokio::pin!(serve);

    tokio::select! {
        result = &mut serve => result,
        _ = drain_deadline() => {
            tracing::warn!(
                event = "drain_timeout",
                window_secs = pii_detection_api::shutdown::DRAIN_WINDOW.as_secs(),
                "drain window elapsed with requests still in flight; exiting"
            );
            Ok(())
        }
    }
}

/// The address the service listens on: `PII_DETECTION_ADDRESS`, or `127.0.0.1:8090`.
/// A container sets `0.0.0.0:8090`.
fn listen_address() -> String {
    std::env::var("PII_DETECTION_ADDRESS").unwrap_or_else(|_| "127.0.0.1:8090".to_owned())
}

/// The port of [`listen_address`], which the health probe connects to on the loopback address.
fn listen_port() -> u16 {
    listen_address()
        .rsplit(':')
        .next()
        .and_then(|port| port.parse().ok())
        .unwrap_or(8090)
}

/// Resolves [`pii_detection_api::shutdown::DRAIN_WINDOW`] after a shutdown signal
/// arrives, and not a moment before -- so it never races the server's
/// ordinary lifetime, only the drain that follows a signal.
async fn drain_deadline() {
    pii_detection_api::shutdown::wait_for_shutdown_signal().await;
    tokio::time::sleep(pii_detection_api::shutdown::DRAIN_WINDOW).await;
}

/// Connects to the server's own listener and requires a 200 status line from
/// `GET /health`. Only the status line is ever inspected: the response
/// body is never read, so it can never end up in a failure message.
///
/// A 200 from `/health` now means `ready: true` and nothing less: the
/// handler answers `503` whenever readiness is false (see
/// `routes::health`). That is deliberately carried in the status line
/// rather than in a `ready` field this probe would have to parse, so the
/// probe can stay body-blind and still refuse to call a rules-only
/// fallback healthy.
async fn probe_health() -> Result<(), std::io::Error> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let mut stream = tokio::net::TcpStream::connect(("127.0.0.1", listen_port())).await?;
    stream
        .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .await?;

    let mut received = Vec::with_capacity(32);
    let mut chunk = [0u8; 32];
    while !received.windows(2).any(|pair| pair == b"\r\n") {
        let read = stream.read(&mut chunk).await?;
        if read == 0 {
            break;
        }
        received.extend_from_slice(&chunk[..read]);
        if received.len() > 256 {
            break;
        }
    }

    match received.starts_with(b"HTTP/1.1 200") {
        true => Ok(()),
        false => Err(std::io::Error::other("unexpected response")),
    }
}
