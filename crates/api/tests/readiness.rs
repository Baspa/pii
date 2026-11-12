//! Design section 8: *"The service verifies all four at start and refuses
//! readiness on a mismatch."*
//!
//! WHY THIS FILE CONSTRUCTS BROKEN ARTIFACTS INSTEAD OF ASSERTING THE HAPPY
//! PATH. A readiness test that passes whether or not the artifact loaded is
//! worse than no test: it reads as coverage and proves nothing. So every
//! test below builds a *specific* defect and requires a `503` from it, and
//! `an_intact_artifact_copy_is_ready` is the control that stops the whole
//! file from passing because the fixture builder is broken in some way that
//! makes every artifact unloadable.
//!
//! THE ASSERTION THAT MATTERS IS THE `/analyze` ONE. A service that answers
//! `503` on `/health` and `200` with rule spans on `/analyze` has invented
//! the degraded rules-only mode the design forbids -- and it would
//! look healthy to a caller that only ever posts text. Every refusal case
//! here asks for `EMAIL_ADDRESS`, which the rules alone can answer, and
//! requires `503` anyway.
//!
//! THE FIXTURES ARE CHEAP ON PURPOSE. The real graph is 118 MB; nothing here
//! copies it. Files that must stay intact are symlinked, files that must be
//! broken are written small, and where a defect has to survive the digest
//! check the manifest is re-pinned to the broken file's own SHA-256. That
//! last trick is the point of `a_graph_the_digests_agree_on_...` and
//! `a_calibration_the_digests_agree_on_...`: they separate "the digest check
//! caught it" from "the digests all agreed and the contents were still
//! wrong", which are different guards and fail in different places.
//!
//! WHICH STAGE EACH DEFECT FAILS AT MATTERS, AND IS ASSERTED. Every defect
//! in the first review round failed at *load*; none at *warm-up*, so the
//! warm-up guard could be deleted with the whole suite still green.
//! `a_graph_that_opens_but_cannot_run_refuses_readiness` closes that: it
//! asserts `Model::load` succeeds on its fixture and only `warm_up` fails,
//! before asking the router for the same `503`.

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use pii_detection_api::startup::{Deployment, router_for, router_for_artifact};
use pii_detection_model::{ModelConfig, require_artifact};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use tower::ServiceExt;

// ---------------------------------------------------------------- the tests

#[tokio::test]
async fn an_intact_artifact_copy_is_ready() {
    // The control. Every other test in this file asserts a refusal, and a
    // refusal is only evidence if the same construction can also produce a
    // service that is ready.
    let source = require_artifact!();
    let temp = tempdir();
    let artifact = artifact_copy(temp.path(), &source);

    let app = router_for_artifact(&artifact).await;
    assert_eq!(get_status(&app, "/health").await, StatusCode::OK);
    assert_eq!(get_body(&app, "/health").await["ready"], true);
    assert_eq!(
        analyze_status(&app, rules_answerable()).await,
        StatusCode::OK
    );
    // And it is the *model* deployment, not a rules-only fallback that
    // happened to be ready.
    let capabilities = get_body(&app, "/capabilities").await;
    let types: Vec<String> =
        serde_json::from_value(capabilities["entity_types"].clone()).expect("entity_types");
    assert!(types.contains(&"PERSON".to_string()));
}

#[tokio::test]
async fn a_configured_artifact_directory_that_does_not_exist_refuses_readiness() {
    // Needs no artifact of its own: the directory is the thing that is
    // missing. Runs on every checkout, artifact fetched or not.
    let temp = tempdir();
    assert_refuses(&temp.path().join("never-fetched")).await;
}

#[tokio::test]
async fn a_file_that_no_longer_matches_the_manifest_refuses_readiness() {
    let source = require_artifact!();
    let temp = tempdir();
    let artifact = artifact_copy(temp.path(), &source);
    // Replace the graph without re-pinning it. This is the corruption case:
    // the manifest still claims a digest the file no longer has.
    fs::remove_file(artifact.join("model.onnx")).expect("remove");
    fs::write(artifact.join("model.onnx"), b"corrupted").expect("write");
    assert_refuses(&artifact).await;
}

#[tokio::test]
async fn a_manifest_that_declares_the_wrong_digest_refuses_readiness() {
    let source = require_artifact!();
    let temp = tempdir();
    let artifact = artifact_copy(temp.path(), &source);
    // The mirror image of the test above: every file is byte-for-byte the
    // reviewed one, and the manifest lies about `tokenizer.json`. A loader
    // that only ever hashed `model.onnx` would sail past this.
    let mut manifest = read_manifest(&artifact);
    manifest["files"]["tokenizer.json"] = json!(format!("sha256:{}", "0".repeat(64)));
    write_manifest(&artifact, &manifest);
    assert_refuses(&artifact).await;
}

#[tokio::test]
async fn a_graph_the_digests_agree_on_but_onnx_cannot_open_refuses_readiness() {
    let source = require_artifact!();
    let temp = tempdir();
    let artifact = artifact_copy(temp.path(), &source);
    // Every digest checks out and the graph is still not a graph. This is
    // the failure the digest check cannot see, and it has to be caught
    // somewhere else -- `Session::open` -- or the service reports ready and
    // fails every request.
    let graph = artifact.join("model.onnx");
    fs::remove_file(&graph).expect("remove");
    fs::write(&graph, b"ONNX this is not").expect("write");
    repin(&artifact, "model.onnx");
    assert_refuses(&artifact).await;
}

#[tokio::test]
async fn a_calibration_the_digests_agree_on_but_the_resolver_cannot_parse_refuses_readiness() {
    let source = require_artifact!();
    let temp = tempdir();
    let artifact = artifact_copy(temp.path(), &source);
    // Valid UTF-8, valid JSON, digest-consistent -- and not a calibration.
    // `Artifact::load` deliberately does not parse this file (the resolver
    // is its one parser), so nothing before the router notices. Without a
    // guard there, the model would load, warm up, run on every request, and
    // detect nothing at all: a resolver with no calibration applies
    // `f32::INFINITY` to every model-sourced candidate. That is
    // indistinguishable from a model quality problem, which is why it
    // refuses readiness instead.
    fs::write(
        artifact.join("calibration.json"),
        br#"{"revision":"x","default":{"NOT_A_TYPE":0.5},"languages":{}}"#,
    )
    .expect("write");
    repin(&artifact, "calibration.json");
    assert_refuses(&artifact).await;
}

#[tokio::test]
async fn a_calibration_whose_thresholds_are_not_probabilities_refuses_readiness() {
    let source = require_artifact!();
    let temp = tempdir();
    let artifact = artifact_copy(temp.path(), &source);
    // A generator that emitted raw logits instead of sigmoid-ed
    // probabilities. `Calibration::from_json` refuses the range, and this
    // asserts the refusal reaches readiness rather than being logged and
    // shrugged off.
    fs::write(
        artifact.join("calibration.json"),
        br#"{"revision":"x","default":{"PERSON":7.4},"languages":{}}"#,
    )
    .expect("write");
    repin(&artifact, "calibration.json");
    assert_refuses(&artifact).await;
}

#[tokio::test]
async fn a_graph_that_opens_but_cannot_run_refuses_readiness() {
    let source = require_artifact!();
    let temp = tempdir();
    let artifact = artifact_copy(temp.path(), &source);
    // `tests/fixtures/opens-but-cannot-run.onnx` is a 458-byte graph that
    // declares exactly the three inputs `Session::open` checks for
    // (`input_ids`, `attention_mask`, `token_type_ids`) and an output whose
    // last dimension is the manifest's 25 labels -- so every check `open`
    // performs passes -- but declares those inputs as `float32` where the
    // real export declares `int64`. `Session::infer` feeds `int64` tensors,
    // and ONNX Runtime refuses the run.
    //
    // Regenerate with (onnx 1.22, `training/.venv`):
    //
    //   ins = [helper.make_tensor_value_info(n, TensorProto.FLOAT,
    //          ["batch", "seq"]) for n in
    //          ["input_ids", "attention_mask", "token_type_ids"]]
    //   out = helper.make_tensor_value_info("logits", TensorProto.FLOAT,
    //          ["batch", "seq", 25])
    //   nodes: Add(input_ids, attention_mask) -> Add(_, token_type_ids)
    //          -> Unsqueeze(_, axes=[2]) -> Mul(_, zeros(25)) -> logits
    //   model.ir_version = 9, opset 17
    //
    // It contains no weights, no vocabulary and no text -- it is four nodes
    // and a zero vector.
    let graph = artifact.join("model.onnx");
    fs::remove_file(&graph).expect("remove");
    fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/opens-but-cannot-run.onnx"),
        &graph,
    )
    .expect("copy fixture graph");
    repin(&artifact, "model.onnx");

    // The point of the fixture, asserted rather than assumed: this defect
    // is invisible to every check that runs before the warm-up inference.
    // If a future change makes `Session::open` reject it, this line fails
    // and says so, rather than the test quietly moving to testing the load
    // guard that four other tests here already cover.
    let model = pii_detection_model::Model::load(&artifact, ModelConfig::default())
        .expect("the fixture graph must OPEN: its defect is at run time, not load time");
    assert!(
        model.warm_up().await.is_err(),
        "the fixture graph must fail the warm-up inference"
    );

    // And the router refuses, which is the behaviour that matters.
    assert_refuses(&artifact).await;
}

#[tokio::test]
async fn a_dropped_artifact_variable_refuses_readiness_rather_than_serving_rules_only() {
    // `docker run -e PII_DETECTION_ARTIFACT_DIR pii-detection-api:local`, with the
    // variable unset on the host, DELETES the image's own ENV -- and
    // compose's `environment: - PII_DETECTION_ARTIFACT_DIR` and a dropped line in
    // a task definition produce the same thing. The process then sees
    // neither variable, which is exactly `Deployment::Unconfigured`.
    //
    // This used to come up ready, `model: null`, health checks green,
    // serving half the product. It must now be a refusal like every other
    // broken configuration -- including, and especially, on `/analyze`.
    assert_router_refuses(&router_for(Deployment::Unconfigured).await).await;
}

#[tokio::test]
async fn an_explicit_rules_only_opt_in_is_ready_and_says_it_has_no_model() {
    // The other half of the inversion: rules-only must stay genuinely
    // reachable, or the refusal above just removes a legitimate deployment
    // instead of making it deliberate.
    let app = router_for(Deployment::RulesOnly).await;

    assert_eq!(get_status(&app, "/health").await, StatusCode::OK);
    let health = get_body(&app, "/health").await;
    assert_eq!(health["ready"], true);
    assert_eq!(
        health["model"],
        Value::Null,
        "a rules-only deployment must report the model's absence on the wire"
    );
    assert_eq!(health["calibration"], Value::Null);

    // It serves what it can, and refuses what it cannot -- with a `400`,
    // never a `200` and an empty list.
    assert_eq!(
        analyze_status(&app, rules_answerable()).await,
        StatusCode::OK
    );
    assert_eq!(
        analyze_status(
            &app,
            json!({"texts": ["Jan de Vries"], "entities": ["PERSON"], "language": "nl"})
        )
        .await,
        StatusCode::BAD_REQUEST
    );

    let capabilities = get_body(&app, "/capabilities").await;
    let types: Vec<String> =
        serde_json::from_value(capabilities["entity_types"].clone()).expect("entity_types");
    assert!(!types.contains(&"PERSON".to_string()));
}

// ------------------------------------------------------------- the assertion

/// The refusal every failure case must produce: `503` on `/health`,
/// `ready: false` in its body, `503` on `/capabilities`, and -- the one that
/// matters -- `503` on an `/analyze` the rules alone could have answered.
async fn assert_refuses(dir: &Path) {
    assert_router_refuses(&router_for_artifact(dir).await).await;
}

/// The same refusal, asserted against an already-built router, for the
/// cases whose defect is a *configuration* rather than an artifact.
async fn assert_router_refuses(app: &axum::Router) {
    let app = app.clone();

    assert_eq!(
        get_status(&app, "/health").await,
        StatusCode::SERVICE_UNAVAILABLE,
        "/health must refuse: a probe reads the status line, not the body"
    );
    assert_eq!(get_body(&app, "/health").await["ready"], false);
    assert_eq!(
        get_status(&app, "/capabilities").await,
        StatusCode::SERVICE_UNAVAILABLE,
        "an instance whose artifact would not load must not publish a capability list"
    );
    assert_eq!(
        analyze_status(&app, rules_answerable()).await,
        StatusCode::SERVICE_UNAVAILABLE,
        "a 200 here is the degraded rules-only mode the design forbids"
    );
}

/// A request the rule packs alone can answer. Deliberately not a `PERSON`
/// request: asking for a type only the model can produce would be refused
/// even by a correct rules-only service, so it could not tell the two
/// apart.
fn rules_answerable() -> Value {
    json!({
        "texts": ["jan@example.nl"],
        "entities": ["EMAIL_ADDRESS"],
        "language": "nl"
    })
}

// ------------------------------------------------------------------ fixtures

/// The three small files copied, the two large ones symlinked. Nothing here
/// duplicates 118 MB, and every byte the loader hashes is the shipped
/// artifact's own until a test deliberately changes one.
fn artifact_copy(temp: &Path, source: &Path) -> PathBuf {
    let artifact = temp.join("artifact");
    fs::create_dir_all(&artifact).expect("create dir");
    for small in ["manifest.json", "labels.json", "calibration.json"] {
        fs::copy(source.join(small), artifact.join(small)).expect("copy");
    }
    for large in ["model.onnx", "tokenizer.json"] {
        std::os::unix::fs::symlink(source.join(large), artifact.join(large)).expect("symlink");
    }
    artifact
}

fn read_manifest(artifact: &Path) -> Value {
    let raw = fs::read_to_string(artifact.join("manifest.json")).expect("read manifest");
    serde_json::from_str(&raw).expect("manifest parses")
}

fn write_manifest(artifact: &Path, manifest: &Value) {
    fs::write(
        artifact.join("manifest.json"),
        serde_json::to_string_pretty(manifest).expect("serialize manifest"),
    )
    .expect("write manifest");
}

/// Rewrites the manifest's digest for `name` to that file's actual SHA-256,
/// so a deliberately broken file still passes verification and the *next*
/// guard is the one under test.
fn repin(artifact: &Path, name: &str) {
    let bytes = fs::read(artifact.join(name)).expect("read file");
    let digest = format!("sha256:{:x}", Sha256::digest(&bytes));
    let mut manifest = read_manifest(artifact);
    manifest["files"][name] = json!(digest);
    write_manifest(artifact, &manifest);
}

/// A bare-bones temp directory, cleaned up on drop. Same shape as
/// `pii_detection_model::artifact`'s test helper, and for the same reason: no
/// external crate for something this small.
struct TempDir(PathBuf);

impl TempDir {
    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn tempdir() -> TempDir {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let unique = format!(
        "pii-detection-api-readiness-{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    );
    let mut dir = std::env::temp_dir();
    dir.push(unique);
    fs::create_dir_all(&dir).expect("create tempdir");
    TempDir(dir)
}

// ------------------------------------------------------------- http plumbing

async fn send(app: &axum::Router, request: Request<Body>) -> (StatusCode, String) {
    let response = app.clone().oneshot(request).await.expect("router responds");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

async fn get_status(app: &axum::Router, path: &str) -> StatusCode {
    let request = Request::builder()
        .uri(path)
        .body(Body::empty())
        .expect("request");
    send(app, request).await.0
}

async fn get_body(app: &axum::Router, path: &str) -> Value {
    let request = Request::builder()
        .uri(path)
        .body(Body::empty())
        .expect("request");
    let (_, raw) = send(app, request).await;
    serde_json::from_str(&raw).expect("json body")
}

async fn analyze_status(app: &axum::Router, body: Value) -> StatusCode {
    let request = Request::builder()
        .method("POST")
        .uri("/analyze")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .expect("request");
    send(app, request).await.0
}
