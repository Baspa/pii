//! Every label value this service can emit, enumerated.
//!
//! A metric label must come from a bounded set known at compile time. This
//! test is what makes that a property of the build rather than a habit: add
//! a label sourced from caller input and it fails here, before it reaches a
//! scrape and becomes permanent. Unbounded cardinality is not a performance
//! footnote on this service -- a label carrying a detected value would be
//! one time series per distinct person's name, which is a privacy leak
//! stored for the retention period of the metrics system.
//!
//! There are no blind exemptions here. The plan this implements skipped
//! `entity_type`, `country`, `service`, `rules` and `le` as "bounded by
//! construction", which is true of the code as written and is exactly the
//! kind of assurance that stops being true without anything failing. Each
//! is checked against the thing that bounds it instead: the enum's own
//! parser, the build's own version, the configured bucket list.

use axum::body::Body;
use axum::http::{Request, header};
use pii_detection_engine::entity::{Country, EntityType};
use serde_json::json;
use tower::ServiceExt;

/// Label values that are literals in this crate's source.
const ALLOWED: &[&str] = &[
    // endpoints and the success outcome
    "analyze",
    "ok",
    // rejection reasons, which are also failure outcomes
    "unsupported_entity_type",
    "unsupported_language",
    "text_too_long",
    "malformed_request",
    "not_ready",
    "detection_unavailable",
    // stages
    "validate",
    "rules",
    "model",
    "resolve",
    "total",
    // language modes
    "explicit",
    "auto",
    // model language coverage -- `wire::ModelCoverage::as_label`
    "trained",
    "untrained",
    "undetermined",
    "rules_only",
    // absent country
    "none",
];

/// Label *names*. New names are as much of a decision as new values, and a
/// name nobody chose -- `request_id`, `text`, `value` -- is the shape this
/// failure usually takes.
const ALLOWED_NAMES: &[&str] = &[
    "endpoint",
    "outcome",
    "stage",
    "entity_type",
    "country",
    "mode",
    "coverage",
    "reason",
    "service",
    "rules",
    "le",
];

async fn post(router: axum::Router, body: serde_json::Value) {
    let _ = router
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
}

fn rules_version() -> String {
    pii_detection_rules::pack::all()
        .iter()
        .map(|pack| pack.version.as_str())
        .collect::<Vec<_>>()
        .join(".")
}

#[tokio::test]
async fn every_metric_label_value_is_from_the_allowed_set() {
    // Exercise success, rejection and refusal so all three label families
    // are present in the render.
    for body in [
        json!({"texts": ["Het BSN is 123456782, mail jan@example.com"],
               "entities": ["NATIONAL_ID", "EMAIL_ADDRESS"], "language": "nl"}),
        json!({"texts": ["x"], "entities": ["Zorbtastic"], "language": "nl"}),
        json!({"texts": ["x"], "entities": ["NATIONAL_ID"], "language": "auto"}),
    ] {
        post(pii_detection_api::routes::rules_only_router(), body).await;
    }
    post(
        pii_detection_api::routes::router_unready(),
        json!({"texts": ["x"], "entities": ["NATIONAL_ID"], "language": "nl"}),
    )
    .await;

    let rendered = pii_detection_api::telemetry::handle().render();

    let buckets: Vec<String> = pii_detection_api::telemetry::LATENCY_BUCKETS
        .iter()
        .map(|bucket| bucket.to_string())
        .chain(std::iter::once("+Inf".to_string()))
        .collect();

    let mut checked = 0usize;
    for line in rendered.lines() {
        let Some(open) = line.find('{') else { continue };
        let Some(close) = line.rfind('}') else {
            continue;
        };
        for pair in line[open + 1..close].split(',') {
            let Some((name, value)) = pair.split_once('=') else {
                continue;
            };
            let value = value.trim_matches('"');
            checked += 1;

            assert!(
                ALLOWED_NAMES.contains(&name),
                "label name {name:?} is not one this service chose to emit:\n{line}"
            );

            // A `quantile` label means the histogram rendered as a summary,
            // which is its own failure -- see `tests/metrics.rs`.
            assert_ne!(
                name, "quantile",
                "a summary series reached the render:\n{line}"
            );

            let bounded = match name {
                // Checked against the enum's own parser rather than
                // exempted: this is the label a leak would hide in, because
                // it is the one whose value describes a detection.
                "entity_type" => EntityType::from_wire(value).is_some(),
                "country" => {
                    value == "none" || serde_json::from_value::<Country>(json!(value)).is_ok()
                }
                // Compile-time facts about the build, checked against the
                // build rather than waved through.
                "service" => value == env!("CARGO_PKG_VERSION"),
                "rules" => value == rules_version(),
                // Exporter-generated, and bounded by the configured buckets.
                "le" => buckets.iter().any(|bucket| bucket == value),
                _ => ALLOWED.contains(&value),
            };

            assert!(
                bounded,
                "label {name}=\"{value}\" is not in the allowed set -- if a caller can \
                 influence it, it does not belong in a metric:\n{line}"
            );
        }
    }

    // A guard that parsed nothing would pass. These are the series the
    // requests above must have produced, so the loop above cannot be
    // vacuously true.
    for series in [
        "pii_detection_requests_total{endpoint=\"analyze\",outcome=\"ok\"}",
        "pii_detection_requests_total{endpoint=\"analyze\",outcome=\"unsupported_entity_type\"}",
        "pii_detection_rejections_total{reason=\"unsupported_entity_type\"}",
        "pii_detection_rejections_total{reason=\"not_ready\"}",
        "pii_detection_entities_total{entity_type=\"NATIONAL_ID\",country=\"NL\"}",
        "pii_detection_entities_total{entity_type=\"EMAIL_ADDRESS\",country=\"none\"}",
        "pii_detection_language_requests_total{mode=\"explicit\"}",
        "pii_detection_language_requests_total{mode=\"auto\"}",
        // A rules-only router is the only deployment these requests can
        // build without an artifact, so `rules_only` is the only coverage
        // value reachable here. The other three are exercised over HTTP in
        // `tests/language_coverage.rs`, which loads the real model.
        "pii_detection_language_coverage_total{coverage=\"rules_only\"}",
        "pii_detection_stage_seconds_bucket{stage=\"rules\"",
        "pii_detection_build_info{",
    ] {
        assert!(
            rendered.contains(series),
            "{series} is missing, so this test checked less than it claims:\n{rendered}"
        );
    }
    assert!(
        checked > 40,
        "only {checked} label pairs were checked, which is fewer than these \
         requests must produce -- the parse above is probably not reading the \
         render:\n{rendered}"
    );
}
