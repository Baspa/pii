//! What `GET /metrics` renders, and the shape it renders it in.
//!
//! The bucket assertions in `analyzing_records_counts_and_every_stage` are
//! the load-bearing ones. `metrics-exporter-prometheus` renders a histogram
//! as a Prometheus *summary* -- per-instance quantiles -- unless buckets are
//! configured, and a summary cannot be aggregated across instances. The
//! design states its latency gate as a fleet-wide p95 over a load-balanced
//! pool, so a summary would make that gate unmeasurable in production while
//! looking perfectly healthy on one node. Asserting on `_bucket{le="..."}`
//! series is what fails the day somebody drops `set_buckets`.

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use tower::ServiceExt;

async fn metrics_body() -> String {
    let response = pii_detection_api::routes::rules_only_router()
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get(header::CONTENT_TYPE).unwrap(),
        "text/plain; version=0.0.4"
    );
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    String::from_utf8(bytes.to_vec()).unwrap()
}

#[tokio::test]
async fn metrics_reports_the_build_it_is_running() {
    let body = metrics_body().await;
    assert!(
        body.contains("pii_detection_build_info"),
        "build info missing:\n{body}"
    );
    assert!(
        body.contains(&format!("service=\"{}\"", env!("CARGO_PKG_VERSION"))),
        "service version missing:\n{body}"
    );
    assert!(body.contains("rules=\""), "rules version missing:\n{body}");
}

use serde_json::json;

async fn analyze(texts: serde_json::Value) {
    let _ = pii_detection_api::routes::rules_only_router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/analyze")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({"texts": texts, "entities": ["NATIONAL_ID"], "language": "nl"})
                        .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
}

#[tokio::test]
async fn analyzing_records_counts_and_every_stage() {
    analyze(json!(["Het BSN is 123456782", "niets hier"])).await;
    let body = metrics_body().await;

    assert!(
        body.contains("pii_detection_requests_total{endpoint=\"analyze\",outcome=\"ok\"}"),
        "{body}"
    );
    assert!(body.contains("pii_detection_texts_total"), "{body}");
    assert!(body.contains("pii_detection_characters_total"), "{body}");

    // `model` is deliberately absent from this list: this is a rules-only
    // router, and a stage that did not run must not report a latency. The
    // model stage is asserted in `tests/model.rs`, where one runs.
    for stage in ["validate", "rules", "resolve", "total"] {
        assert!(
            body.contains(&format!(
                "pii_detection_stage_seconds_bucket{{stage=\"{stage}\""
            )),
            "stage {stage} missing, or rendered as a summary rather than a histogram:\n{body}"
        );
    }
    assert!(
        !body.contains("pii_detection_stage_seconds_bucket{stage=\"model\""),
        "a rules-only deployment reported a model latency:\n{body}"
    );
}

/// The rendering the fleet-wide p95 gate depends on.
///
/// `metrics-exporter-prometheus` renders a histogram as a summary unless
/// buckets are configured, and a summary carries per-instance quantiles that
/// cannot be aggregated across a load-balanced pool. This asserts on both
/// halves of that: every configured bucket boundary is present with its
/// `_bucket{le="..."}` series and a `+Inf` bucket, and no `quantile="..."`
/// series exists for the stage histogram at all.
#[tokio::test]
async fn stage_latency_renders_as_a_histogram_and_never_as_a_summary() {
    analyze(json!(["Het BSN is 123456782"])).await;
    let body = metrics_body().await;

    assert!(
        body.contains("# TYPE pii_detection_stage_seconds histogram"),
        "the stage metric is not typed as a histogram:\n{body}"
    );
    for bucket in pii_detection_api::telemetry::LATENCY_BUCKETS {
        assert!(
            body.contains(&format!("le=\"{bucket}\"")),
            "bucket {bucket} missing, so p95 cannot be interpolated across it:\n{body}"
        );
    }
    assert!(body.contains("le=\"+Inf\""), "no +Inf bucket:\n{body}");
    assert!(
        body.contains("pii_detection_stage_seconds_sum")
            && body.contains("pii_detection_stage_seconds_count"),
        "a histogram needs _sum and _count for rate() to work:\n{body}"
    );

    for line in body
        .lines()
        .filter(|line| line.starts_with("pii_detection_stage_seconds"))
    {
        assert!(
            !line.contains("quantile=\""),
            "rendered as a summary; a fleet-wide p95 would be unobtainable:\n{line}"
        );
    }
}

#[tokio::test]
async fn detected_entities_are_counted_by_type_and_country() {
    analyze(json!(["Het BSN is 123456782"])).await;
    let body = metrics_body().await;

    assert!(
        body.contains("pii_detection_entities_total{entity_type=\"NATIONAL_ID\",country=\"NL\"}"),
        "{body}"
    );
    assert!(
        body.contains("pii_detection_language_requests_total{mode=\"explicit\"}"),
        "{body}"
    );
}

#[tokio::test]
async fn an_entity_without_a_country_is_labelled_none_rather_than_omitted() {
    let _ = pii_detection_api::routes::rules_only_router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/analyze")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "texts": ["Mail jane@example.com"],
                        "entities": ["EMAIL_ADDRESS"],
                        "language": "auto"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    let body = metrics_body().await;
    assert!(
        body.contains(
            "pii_detection_entities_total{entity_type=\"EMAIL_ADDRESS\",country=\"none\"}"
        ),
        "{body}"
    );
    assert!(
        body.contains("pii_detection_language_requests_total{mode=\"auto\"}"),
        "{body}"
    );
}

#[tokio::test]
async fn a_rejection_is_counted_by_reason_and_never_by_value() {
    let canary = "Zorbtastic";
    let _ = pii_detection_api::routes::rules_only_router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/analyze")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({"texts": ["x"], "entities": [canary], "language": "nl"}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    let body = metrics_body().await;
    assert!(
        body.contains("pii_detection_rejections_total{reason=\"unsupported_entity_type\"}"),
        "{body}"
    );
    assert!(
        !body.contains(canary),
        "the rejected value must not become a label:\n{body}"
    );
    assert!(
        body.contains(
            "pii_detection_requests_total{endpoint=\"analyze\",outcome=\"unsupported_entity_type\"}"
        ),
        "the request counter should carry the same fixed reason:\n{body}"
    );
}

#[tokio::test]
async fn phone_attribution_is_described_in_metrics() {
    let body = metrics_body().await;
    assert!(
        body.contains("pii_detection_phone_attributed_to_company_total"),
        "phone attribution counter missing:\n{body}"
    );
}

/// The two refusals our four-reason list predates. Both are `503`,
/// both are states an operator is likely to be looking at the dashboard
/// during, and a refusal that incremented nothing would make a service
/// that answers nothing look like a service nobody is calling.
#[tokio::test]
async fn an_unready_deployment_counts_its_refusals_too() {
    let _ = pii_detection_api::routes::router_unready()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/analyze")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({"texts": ["x"], "entities": ["NATIONAL_ID"], "language": "nl"})
                        .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    let body = metrics_body().await;
    assert!(
        body.contains("pii_detection_rejections_total{reason=\"not_ready\"}"),
        "{body}"
    );
}
