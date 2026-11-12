//! The process-wide metrics recorder, and the names of everything it holds.
//!
//! Design section 13 draws the line this module sits on: counts, latencies,
//! versions and outcomes are allowed; raw input, detected values, token
//! strings and text-bearing exception messages are not. Every label value
//! recorded through these names comes from a fixed set known at compile
//! time -- an `EntityType`, a `Country`, or one of a handful of literals --
//! and never from caller input. `tests/cardinality.rs` is what holds that
//! line; `docs/observability.md` is the list a reviewer can check instead of
//! reading the code.

use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use std::sync::OnceLock;

/// Latency buckets, in seconds.
///
/// Explicit buckets rather than the exporter's default, because the default
/// renders a Prometheus *summary* carrying per-instance quantiles, and a
/// summary cannot be aggregated across instances. The design states its
/// latency gate as a fleet p95 over a load-balanced pool, which only
/// bucketed histograms can answer.
///
/// The range is chosen around that gate: 5 ms and 10 ms resolve the fast
/// path, 25 ms and 50 ms bracket the 50 ms target for a 2,000-character
/// request, and the tail out to 2.5 s covers a 125,000-character document.
pub const LATENCY_BUCKETS: [f64; 9] = [0.005, 0.010, 0.025, 0.050, 0.100, 0.250, 0.500, 1.0, 2.5];

/// Requests, by endpoint and by how they ended.
pub const REQUESTS: &str = "pii_detection_requests_total";
/// Texts submitted, counted before any work so a refusal counts too.
pub const TEXTS: &str = "pii_detection_texts_total";
/// Characters submitted, on the same basis as [`TEXTS`].
pub const CHARACTERS: &str = "pii_detection_characters_total";
/// Per-stage latency: `validate`, `rules`, `model`, `resolve`, `total`.
pub const STAGE_SECONDS: &str = "pii_detection_stage_seconds";
/// Detections that survived resolution, by type and country.
pub const ENTITIES: &str = "pii_detection_entities_total";
/// Requests by language *mode* -- `explicit` or `auto` -- never by code.
pub const LANGUAGE: &str = "pii_detection_language_requests_total";
/// Analyses that got past validation, by whether the model was trained on the
/// language they ran for: `trained`, `untrained`, `undetermined` (`auto`, on
/// which this service performs no identification) or `rules_only`.
///
/// Separate from [`LANGUAGE`] rather than a second label on it, because the
/// two have different denominators on purpose: `LANGUAGE` is counted before
/// validation so a refusal still records the mode it asked for, and a request
/// that was refused never ran under any regime. Still never the code itself
/// -- one of four literals from `wire::ModelCoverage::as_label`.
pub const COVERAGE: &str = "pii_detection_language_coverage_total";
/// Refusals, by [`crate::error::ApiError::reason`].
pub const REJECTIONS: &str = "pii_detection_rejections_total";
/// Phone numbers cleared because a company, and no person, stood beside them. No labels: see `resolve::Resolved`.
pub const PHONES_ATTRIBUTED: &str = "pii_detection_phone_attributed_to_company_total";

/// The process-wide recorder.
///
/// Installed exactly once. `install_recorder` fails on a second call, and the
/// test binary builds a router per test, so this must be a `OnceLock` rather
/// than something `router()` does each time.
pub fn handle() -> &'static PrometheusHandle {
    static HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();
    HANDLE.get_or_init(|| {
        let handle = PrometheusBuilder::new()
            .set_buckets(&LATENCY_BUCKETS)
            .unwrap_or_else(|error| panic!("latency buckets are invalid: {error}"))
            .install_recorder()
            .unwrap_or_else(|error| panic!("metrics recorder failed to install: {error}"));

        describe_and_seed();
        handle
    })
}

/// Records the build identity and declares every metric up front.
///
/// Declaring them here means a scrape before the first request still shows
/// the series, so a dashboard is not empty-and-ambiguous on a fresh node.
fn describe_and_seed() {
    let rules = pii_detection_rules::pack::all()
        .iter()
        .map(|pack| pack.version.as_str())
        .collect::<Vec<_>>()
        .join(".");

    // Descriptions, so `/metrics` carries HELP and TYPE lines for every
    // series this service can produce and a reader of a scrape does not
    // have to come here to find out what one means.
    metrics::describe_counter!(REQUESTS, "Requests, by endpoint and by how they ended.");
    metrics::describe_counter!(TEXTS, "Texts submitted for analysis.");
    metrics::describe_counter!(CHARACTERS, "Characters submitted for analysis.");
    metrics::describe_counter!(
        REJECTIONS,
        "Refusals, by a fixed reason string -- never by the value refused."
    );
    metrics::describe_counter!(ENTITIES, "Detections that survived resolution.");
    metrics::describe_counter!(LANGUAGE, "Requests by language mode, explicit or auto.");
    metrics::describe_counter!(
        COVERAGE,
        "Analyses by whether the model was trained on the language they ran for."
    );
    metrics::describe_histogram!(
        STAGE_SECONDS,
        metrics::Unit::Seconds,
        "Per-stage request latency."
    );
    metrics::describe_counter!(
        PHONES_ATTRIBUTED,
        "Phone numbers cleared because a company, and no person, stood beside them."
    );
    // No labels to carry this into existence on a first request, so it is
    // seeded at zero here -- otherwise a fresh node's scrape omits the
    // series entirely until the first company phone number clears.
    metrics::counter!(PHONES_ATTRIBUTED).increment(0);

    metrics::gauge!(
        "pii_detection_build_info",
        "service" => env!("CARGO_PKG_VERSION"),
        "rules" => rules,
    )
    .set(1.0);
}
