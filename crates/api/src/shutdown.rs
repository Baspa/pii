//! Turning a shutdown signal into the drain `main.rs` runs on it.
//!
//! Design section 22's graceful drain has three moving parts: stop
//! accepting new work, flip [`crate::routes::AppState`]'s readiness flag
//! to `false` *before* that refusal takes effect, and wait out whatever
//! is already in flight within a bounded window before exiting. This
//! module owns the first and third -- the signal wait, and
//! [`DRAIN_WINDOW`]'s bound -- and documents why the second is deliberately
//! not delegated to `axum::serve` alone. `main.rs` wires the three
//! together; see its module-level comment for the exact ordering.
//!
//! # What the drain actually waits on
//!
//! `axum::serve(listener, router).with_graceful_shutdown(signal)` already
//! does most of the second and third parts by itself: once `signal`
//! resolves, its accept loop stops calling `listener.accept()` (see
//! `axum`'s `serve` module), and every connection already accepted is told
//! to finish its current exchange -- via `hyper`'s own
//! `graceful_shutdown()` -- before the whole future resolves. That is a
//! wait on **connections**, not on the model.
//!
//! **It is not a wait on [`pii_detection_model::Model`]'s inference semaphore, and
//! this module does not add one either.** The semaphore is sized to
//! exactly one permit by default (`PII_DETECTION_INFERENCE_SLOTS`, see
//! `pii_detection_model::ModelConfig::from_env`'s docs), so a request that is
//! queued behind it at the moment a shutdown signal arrives is not a
//! request the drain has lost track of: it is still the body of an HTTP
//! exchange on a connection `axum::serve` is already holding open, and
//! that connection is exactly what the graceful shutdown above waits for.
//! Waiting on the semaphore *as well* would be redundant on the happy path
//! and dangerous on the unhappy one -- a permit acquire has no timeout of
//! its own, so a second, independent wait on it could hang past
//! [`DRAIN_WINDOW`] instead of being bounded by it.

use std::time::Duration;

/// How long the process waits, after a shutdown signal, for requests
/// already in flight to finish before it exits regardless of whether they
/// have.
///
/// **Ten seconds.** The evidence this is judged against:
///
/// * Measured p95 end-to-end through the built container is **29.2 ms**,
///   with a max of 152.6 ms across 3,000 requests -- one outlier, "host
///   noise, not model behaviour"
///   (`training/reports/distillation-run.md` §3, table under "Headline
///   (S5), student, gate parameters"). Ten seconds is roughly 65x that
///   single-request max and roughly 340x the p95: generous headroom even
///   for several requests queuing behind the inference semaphore's single
///   permit (see this module's docs above) at the instant a signal
///   arrives, without materially slowing a rollout.
/// * `PII_DETECTION_INFERENCE_SLOTS` defaults to one, so concurrent requests
///   serialise behind that one permit rather than each independently
///   running long -- the drain is waiting out a short queue of short
///   requests, not a set of requests that could each individually run for
///   seconds.
/// * It sits comfortably inside the stop-timeout budget an orchestrator
///   gives before `SIGKILL` -- ECS's task-level `stopTimeout` defaults to
///   30 seconds. Ten leaves several seconds of slack for process
///   scheduling and log flushing so this process's own bounded exit
///   reliably wins that race, rather than the drain being cut short by a
///   `SIGKILL` that lands before it finishes on its own.
/// * The Dockerfile's `HEALTHCHECK --start-period=90s` is a *different*
///   budget -- boot time, covering a ~135 MB digest verification, an ONNX
///   session open and a warm-up inference -- and is not evidence about
///   shutdown timing at all. It is cited here only as confirmation that
///   this deployment already reasons about container timing in
///   tens-of-seconds increments, which ten seconds fits naturally and one
///   or two would not.
///
/// Not read from an environment variable, unlike `PII_DETECTION_INTRA_THREADS` or
/// `PII_DETECTION_INFERENCE_SLOTS`: those are properties of the machine this
/// process happens to run on and are expected to be re-swept per host.
/// This number is not -- it is a property of the request-latency
/// distribution above and the orchestrator's own timeout, both recorded
/// here, and a change to either is a reason to edit this constant (and
/// its reasoning) rather than to quietly override it with a flag nobody
/// will remember to unset.
pub const DRAIN_WINDOW: Duration = Duration::from_secs(10);

/// Waits for whichever arrives first: `SIGTERM`, the signal ECS and
/// `docker stop` send, or `SIGINT` (`Ctrl+C`), for `cargo run` and manual
/// local testing. Returns once one of them fires; the caller decides what
/// "shutdown" means from there.
///
/// Tokio supports multiple independent, concurrent listeners for the same
/// signal -- each call to this function installs its own -- which is what
/// lets `main.rs` await it twice: once as the signal
/// `with_graceful_shutdown` reacts to, and once to start
/// [`DRAIN_WINDOW`]'s countdown. Both listeners fire off the same
/// underlying signal.
///
/// Installing a signal handler can fail -- in practice, only if the
/// underlying OS call itself fails, not from any condition this process
/// controls. Library code here carries no `unwrap()`/`expect()`, so a
/// failed installation is not treated as fatal: it narrows the set of
/// signals that can start a drain rather than crashing the process that
/// was trying to start one. If every installation attempted here somehow
/// failed, this function would simply never return, which is the
/// fail-safe this process has instead of a panic -- an unreachable signal
/// is a container that must be killed rather than one that drains
/// silently wrong. It has not been observed to fail on either target this
/// service ships to: Linux under Docker, and macOS in local development.
pub async fn wait_for_shutdown_signal() {
    let ctrl_c = async {
        if tokio::signal::ctrl_c().await.is_err() {
            std::future::pending::<()>().await;
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut stream) => {
                stream.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
}
