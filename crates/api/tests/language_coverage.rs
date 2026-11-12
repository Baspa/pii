//! The API accepts 24 languages and the model was trained on six. This file
//! is what makes a caller able to tell which of the two they got.
//!
//! WHAT THE GAP WAS. `POST /analyze` validated `language` against the 24
//! official EU codes and then said nothing else about it. A request carrying
//! `"pl"` validated, ran through a model that has never seen Polish, was
//! scored against a flat default threshold, and came back with spans that
//! looked exactly like Dutch ones. The failure mode is not silence: a
//! reviewer measured Maltese -- absent from training entirely -- scoring
//! 0.54-0.58 against a 0.50 threshold. An untrained language does not fail
//! quietly, it fails confidently, which is why "the caller can see it" had to
//! become a property of the response rather than of the documentation.
//!
//! WHAT IS DELIBERATELY NOT TESTED HERE. That an untrained language detects
//! `PERSON` *well*. It does not, that is the entire point, and a test
//! asserting a quality figure for a language the benchmark does not cover
//! would be the claim the design refuses to make ("a language that
//! cannot be measured cannot be claimed"). What is tested is that the service
//! says which regime it ran under, and that the rules keep working either
//! way.
//!
//! THE ARTIFACT-GATED HALF. The regime distinction only exists where there is
//! a model, so most of this file goes through `require_model!` /
//! `require_artifact!` and prints an uncaptured skip line without one -- see
//! `crates/model/src/test_support.rs`. The rules-only half above it
//! always runs.

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use pii_detection_api::routes::{router, rules_only_router};
use pii_detection_model::{ModelConfig, require_artifact, require_model};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use tower::ServiceExt;

/// Polish prose carrying an email, an IBAN and a person's name. The
/// structured identifiers are what the rules must still find; the name is
/// what only the model produces and what an untrained language cannot vouch
/// for.
const POLISH: &str = "Prosze o kontakt: biuro@example.pl, IBAN PL61109010140000071219812874, \
                      sprawa prowadzona przez Jana Kowalskiego.";

async fn send(app: &axum::Router, request: Request<Body>) -> (StatusCode, Value) {
    let response = app.clone().oneshot(request).await.expect("router responds");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    let parsed = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, parsed)
}

async fn analyze(app: &axum::Router, body: Value) -> Value {
    let request = Request::builder()
        .method("POST")
        .uri("/analyze")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .expect("request");
    let (status, parsed) = send(app, request).await;
    assert_eq!(status, StatusCode::OK, "expected 200, body: {parsed}");
    parsed
}

async fn capabilities(app: &axum::Router) -> Value {
    let request = Request::builder()
        .uri("/capabilities")
        .body(Body::empty())
        .expect("request");
    let (status, parsed) = send(app, request).await;
    assert_eq!(status, StatusCode::OK);
    parsed
}

/// The entity types in one text's result list.
fn types_in(response: &Value, text: usize) -> Vec<String> {
    response["results"][text]
        .as_array()
        .expect("a result list")
        .iter()
        .map(|entity| entity["entity_type"].as_str().expect("a type").to_string())
        .collect()
}

/// The spans in one text's result list, as `(type, start, end)` -- enough to
/// compare two runs for identical detection without quoting any text.
fn spans_in(response: &Value, text: usize) -> Vec<(String, u64, u64)> {
    response["results"][text]
        .as_array()
        .expect("a result list")
        .iter()
        .map(|entity| {
            (
                entity["entity_type"].as_str().expect("a type").to_string(),
                entity["start"].as_u64().expect("a start"),
                entity["end"].as_u64().expect("an end"),
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// No artifact needed: these run on every checkout.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn an_unsupported_language_still_gets_rules_based_detection() {
    // The whole reason the decision was "accept, but declare" rather than
    // "refuse": the design makes language a ranking and context hint,
    // never a gate on a structured recognizer, so a Polish IBAN is matched by
    // format and validated by checksum exactly as a Dutch one is. Refusing
    // `pl` outright would throw that away.
    let app = rules_only_router();
    let response = analyze(
        &app,
        json!({"texts": [POLISH], "entities": ["EMAIL_ADDRESS", "IBAN_CODE"], "language": "pl"}),
    )
    .await;

    let found = types_in(&response, 0);
    assert!(
        found.contains(&"EMAIL_ADDRESS".to_string()),
        "the rules stopped finding an email in Polish prose: {found:?}"
    );
    assert!(
        found.contains(&"IBAN_CODE".to_string()),
        "the rules stopped finding an IBAN in Polish prose: {found:?}"
    );
}

#[tokio::test]
async fn every_response_carries_the_regime_it_ran_under() {
    // Including a rules-only deployment, which has its own regime rather than
    // a missing field: a caller must never have to distinguish "no model" from
    // "the service forgot to say".
    let app = rules_only_router();
    for language in ["nl", "pl", "auto"] {
        let response = analyze(
            &app,
            json!({"texts": ["mail jane@example.com"],
                   "entities": ["EMAIL_ADDRESS"], "language": language}),
        )
        .await;
        assert_eq!(response["language"]["requested"], language);
        assert_eq!(
            response["language"]["model_coverage"], "rules_only",
            "a deployment with no model reported something other than rules_only"
        );
        // Nothing is unverified when nothing came from a model. The types the
        // model would have produced are not advertised and not accepted --
        // `tests/model.rs` holds that half.
        assert_eq!(response["language"]["unverified_entity_types"], json!([]));
    }
}

// ---------------------------------------------------------------------------
// Artifact-gated: the regime distinction needs a model to be about.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn capabilities_separates_what_is_accepted_from_what_was_trained() {
    let app = router(require_model!());
    let body = capabilities(&app).await;

    let accepted: Vec<String> =
        serde_json::from_value(body["languages"]["accepted"].clone()).expect("accepted");
    let trained: Vec<String> =
        serde_json::from_value(body["languages"]["model_trained"].clone()).expect("model_trained");

    assert_eq!(accepted.len(), 24);
    assert!(accepted.contains(&"pl".to_string()));
    assert!(accepted.contains(&"mt".to_string()));

    // The fact a caller has to be able to read: Polish is accepted and it is
    // not trained. If these two lists were ever the same field again this
    // assertion is the one that fails.
    assert!(
        !trained.contains(&"pl".to_string()),
        "the shipped artifact claims Polish training: {trained:?}"
    );
    assert!(trained.contains(&"nl".to_string()));
    assert!(
        trained.len() < accepted.len(),
        "trained {trained:?} is not narrower than accepted"
    );

    // And what the difference costs, named rather than left to be inferred.
    let model_only: Vec<String> =
        serde_json::from_value(body["languages"]["model_only_entity_types"].clone())
            .expect("model_only_entity_types");
    assert_eq!(
        model_only,
        vec!["PERSON".to_string(), "STREET_ADDRESS".to_string()],
        "the model-only set is what an untrained language loses"
    );
}

#[tokio::test]
async fn the_trained_list_is_read_from_the_manifest_and_not_from_rust() {
    // THE MUTATION THIS EXISTS TO CATCH: a `model_trained` that returns a
    // list written in this workspace instead of the artifact's own
    // `supported_languages`. A constant spelling out the six shipped codes
    // would satisfy every other test in this file, because the shipped
    // artifact happens to declare exactly those six.
    //
    // So this loads an artifact that declares something else. The directory
    // is the real one -- the same graph, tokenizer, labels and calibration,
    // hard-linked so nothing is copied -- with one field of `manifest.json`
    // rewritten. `manifest.json` carries the digests and is not itself
    // digested, so the four files still verify unchanged; only the declared
    // coverage differs.
    let source = require_artifact!();
    let restated = ArtifactWithLanguages::new(&source, &["pl"]);

    let model =
        match pii_detection_model::Model::load_and_warm(restated.dir(), ModelConfig::default())
            .await
        {
            Ok(model) => std::sync::Arc::new(model),
            Err(error) => panic!("the restated artifact would not load: {error}"),
        };
    let app = router(model);

    let trained: Vec<String> =
        serde_json::from_value(capabilities(&app).await["languages"]["model_trained"].clone())
            .expect("model_trained");
    assert_eq!(
        trained,
        vec!["pl".to_string()],
        "`/capabilities` did not read the manifest it was given"
    );

    // And the response signal follows the manifest too, in both directions:
    // Polish is now the trained language and Dutch is not.
    let polish = analyze(
        &app,
        json!({"texts": [POLISH], "entities": ["PERSON"], "language": "pl"}),
    )
    .await;
    assert_eq!(polish["language"]["model_coverage"], "trained");
    assert_eq!(polish["language"]["unverified_entity_types"], json!([]));

    let dutch = analyze(
        &app,
        json!({"texts": [POLISH], "entities": ["PERSON"], "language": "nl"}),
    )
    .await;
    assert_eq!(
        dutch["language"]["model_coverage"], "untrained",
        "Dutch stayed trained against a manifest that no longer declares it"
    );
}

#[tokio::test]
async fn a_trained_language_says_so_and_vouches_for_every_type() {
    let app = router(require_model!());
    let response = analyze(
        &app,
        json!({"texts": ["Jan de Vries, jan@example.nl, Beethovenstraat 12 Amsterdam"],
               "entities": ["PERSON", "STREET_ADDRESS", "EMAIL_ADDRESS"], "language": "nl"}),
    )
    .await;

    assert_eq!(response["language"]["requested"], "nl");
    assert_eq!(response["language"]["model_coverage"], "trained");
    assert_eq!(
        response["language"]["unverified_entity_types"],
        json!([]),
        "a trained language must not flag its own types"
    );
    // Not a vacuous pass: the model actually produced something to vouch for.
    assert!(
        types_in(&response, 0).contains(&"PERSON".to_string()),
        "the model found no person, so this test proved nothing"
    );
}

#[tokio::test]
async fn an_accepted_but_untrained_language_declares_itself() {
    let app = router(require_model!());
    let response = analyze(
        &app,
        json!({"texts": [POLISH],
               "entities": ["PERSON", "EMAIL_ADDRESS", "IBAN_CODE"], "language": "pl"}),
    )
    .await;

    assert_eq!(response["language"]["requested"], "pl");
    assert_eq!(response["language"]["model_coverage"], "untrained");
    assert_eq!(
        response["language"]["unverified_entity_types"],
        json!(["PERSON", "STREET_ADDRESS"]),
        "an untrained language must name the types nothing backs"
    );

    // Accept, not refuse: the rules ran and found what they always find.
    let found = types_in(&response, 0);
    assert!(found.contains(&"EMAIL_ADDRESS".to_string()), "{found:?}");
    assert!(found.contains(&"IBAN_CODE".to_string()), "{found:?}");
}

#[tokio::test]
async fn the_same_text_detects_identically_and_only_the_declaration_differs() {
    // The truthfulness fix must not be a scoring change. Sending one text
    // twice, under a trained code and an untrained one, must produce the same
    // spans -- the language reaches only the calibration lookup, and
    // `calibration.json` currently has an empty `languages` block, so both
    // fall to the same default. If a future calibration gives `nl` its own
    // threshold this test is expected to be revisited; it is pinned to what
    // the shipped artifact actually does.
    let app = router(require_model!());
    let entities = json!(["PERSON", "STREET_ADDRESS", "EMAIL_ADDRESS", "IBAN_CODE"]);

    let trained = analyze(
        &app,
        json!({"texts": [POLISH], "entities": entities, "language": "nl"}),
    )
    .await;
    let untrained = analyze(
        &app,
        json!({"texts": [POLISH], "entities": entities, "language": "pl"}),
    )
    .await;

    assert_eq!(
        spans_in(&trained, 0),
        spans_in(&untrained, 0),
        "declaring the regime changed what was detected"
    );
    assert!(
        !spans_in(&trained, 0).is_empty(),
        "nothing was detected either way, so this comparison proved nothing"
    );
    assert_ne!(
        trained["language"]["model_coverage"], untrained["language"]["model_coverage"],
        "the two regimes reported the same coverage"
    );
}

#[tokio::test]
async fn auto_is_undetermined_because_nothing_identifies_the_language() {
    // `auto` is the default and the mode the benchmark itself sends. The
    // service performs no language identification at all -- there is no
    // `pii-detection-lang` crate -- so reporting `trained` here would be a claim
    // nothing checked, and reporting `untrained` would be a claim in the
    // other direction. It gets its own word, and it carries the same
    // unverified list as `untrained`, which is the fail-closed side.
    let app = router(require_model!());
    let response = analyze(
        &app,
        json!({"texts": ["Jan de Vries, jan@example.nl"],
               "entities": ["PERSON", "EMAIL_ADDRESS"], "language": "auto"}),
    )
    .await;

    assert_eq!(response["language"]["requested"], "auto");
    assert_eq!(response["language"]["model_coverage"], "undetermined");
    assert_eq!(
        response["language"]["unverified_entity_types"],
        json!(["PERSON", "STREET_ADDRESS"])
    );
}

#[tokio::test]
async fn a_request_omitting_the_language_field_is_auto_and_says_so() {
    // The wire default. A caller that never sends `language` is the one most
    // likely to also never read the documentation.
    let app = router(require_model!());
    let response = analyze(
        &app,
        json!({"texts": ["jan@example.nl"], "entities": ["EMAIL_ADDRESS"]}),
    )
    .await;
    assert_eq!(response["language"]["requested"], "auto");
    assert_eq!(response["language"]["model_coverage"], "undetermined");
}

#[tokio::test]
async fn the_coverage_metric_records_the_regime_and_never_the_code() {
    let app = router(require_model!());
    for language in ["nl", "pl", "auto"] {
        let _ = analyze(
            &app,
            json!({"texts": ["jan@example.nl"],
                   "entities": ["EMAIL_ADDRESS"], "language": language}),
        )
        .await;
    }

    let rendered = pii_detection_api::telemetry::handle().render();
    for coverage in ["trained", "untrained", "undetermined"] {
        assert!(
            rendered.contains(&format!(
                "pii_detection_language_coverage_total{{coverage=\"{coverage}\"}}"
            )),
            "coverage {coverage} missing from the render:\n{rendered}"
        );
    }
    // The discipline `routes.rs` records the language *mode* for: a
    // caller-supplied code is an unbounded label, and this metric must never
    // become one. `tests/cardinality.rs` enumerates the allowed values; this
    // asserts the specific mistake -- the code itself -- is absent.
    for code in ["coverage=\"pl\"", "coverage=\"nl\"", "coverage=\"auto\""] {
        assert!(
            !rendered.contains(code),
            "a caller-supplied language code reached a metric label: {code}"
        );
    }
}

// ---------------------------------------------------------------------------
// A real artifact directory whose manifest declares different coverage.
// ---------------------------------------------------------------------------

/// A temporary artifact directory: the source artifact's four verified files,
/// hard-linked rather than copied, plus a `manifest.json` with
/// `supported_languages` replaced.
///
/// Hard links because `model.onnx` is 107 MB and this test has no interest in
/// its bytes, only in the manifest beside them; `fs::copy` is the fallback for
/// a temp directory on another filesystem. The digests in the manifest are
/// left exactly as they were, so `Artifact::load`'s verification runs for real
/// against the real files -- this fixture weakens nothing.
struct ArtifactWithLanguages {
    dir: PathBuf,
}

impl ArtifactWithLanguages {
    fn new(source: &Path, languages: &[&str]) -> ArtifactWithLanguages {
        let dir = std::env::temp_dir().join(format!(
            "pii-detection-language-coverage-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create the restated artifact directory");

        for file in [
            "model.onnx",
            "tokenizer.json",
            "labels.json",
            "calibration.json",
        ] {
            let from = source.join(file);
            let to = dir.join(file);
            if std::fs::hard_link(&from, &to).is_err() {
                std::fs::copy(&from, &to).expect("copy an artifact file");
            }
        }

        let raw = std::fs::read_to_string(source.join("manifest.json")).expect("read the manifest");
        let mut manifest: Value = serde_json::from_str(&raw).expect("parse the manifest");
        assert!(
            manifest["supported_languages"].is_array(),
            "the source manifest has no supported_languages to restate"
        );
        manifest["supported_languages"] = json!(languages);
        std::fs::write(
            dir.join("manifest.json"),
            serde_json::to_string_pretty(&manifest).expect("serialize the manifest"),
        )
        .expect("write the restated manifest");

        ArtifactWithLanguages { dir }
    }

    fn dir(&self) -> &Path {
        &self.dir
    }
}

impl Drop for ArtifactWithLanguages {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}
