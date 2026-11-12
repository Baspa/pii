//! The model seam: loads a digest-verified artifact, windows and tokenizes
//! text exactly as `training/align.py` does, runs the ONNX graph, decodes
//! BIO labels back to code-point spans, and emits the same `Candidate` type
//! the rules engine emits.
//!
//! [`Model`] is the public entry point the service wiring connects into the detection
//! flow. Everything below it -- [`artifact`], [`window`], [`decode`] and
//! [`session`] -- is a separable seam with its own tests, most of them
//! differential against Python's real output (see each module's own docs).

pub mod artifact;
pub mod cities;
pub mod decode;
pub mod refine;
pub mod session;
/// The artifact-gating condition every artifact-dependent test in this
/// workspace shares -- see the module's own docs for why it is a feature and
/// not a `#[cfg(test)]` helper. Never compiled into the shipped binary:
/// `pii-detection-api` enables `test-support` from `[dev-dependencies]` only.
#[cfg(any(test, feature = "test-support"))]
pub mod test_support;
pub mod window;

use std::path::Path;
use std::sync::Arc;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

use pii_detection_engine::candidate::Candidate;
use pii_detection_engine::text::Document;

use crate::artifact::{Artifact, ArtifactError, Labels};
use crate::window::Windower;

/// The ONNX Runtime build this binary actually links, as the runtime itself
/// reports it (branch, commit, build type) -- not `ort::MINOR_VERSION`,
/// which is only the API version the `ort` crate was compiled against and
/// does not name a release.
///
/// Public because a latency number is a statement about a runtime as much as
/// about a machine: the spec's own comparison table is stamped "ONNX Runtime
/// 1.29", and `crates/eval/src/bin/latency.rs` stamps every run it
/// writes with this. `examples/runtime_version.rs` prints it on its own.
pub fn runtime_info() -> &'static str {
    ort::info()
}

/// The `intra_threads` cap this crate falls back to when
/// `PII_DETECTION_INTRA_THREADS` is not set. **Measured on one machine, not
/// derived, and not a platform constant -- treat it as a citation, not a
/// law.**
///
/// Measured on an Apple M1 Max (10 cores, 8 performance + 2 efficiency,
/// macOS 26.5.2 arm64, ONNX Runtime 1.28.0) in
/// `training/reports/latency-investigation.md`: `available_parallelism()`
/// (10 on that machine) is past this workload's scaling knee, and once
/// intra-op spinning is on (see [`crate::session`]'s module docs) it is
/// the *worst* setting tested there -- raw-graph p50 117.8 ms at
/// `intra_threads=10` against 27.5 ms at 6 (investigation report §4.1).
/// Six wins on raw-graph p50/min, but it was *not* the best end-to-end
/// number: across the full `Model::detect` sweep (tokenise, batched
/// inference, decode, reconcile, trim) the lowest measured p95 was at
/// four threads, not six -- six's p95 was noisier on the contended host
/// the sweep ran on (report §3, §7.1). Four is therefore the shipped
/// default, not six.
///
/// **This number is a property of one CPU's core layout, not a hardware
/// constant.** It *must* be re-swept on the actual deployment instance
/// (report §7.4: Graviton has no efficiency cores and a different core
/// count) before being trusted there, and `PII_DETECTION_INTRA_THREADS` exists
/// so that re-sweep can change the running default without a code
/// change. No p95 figure from that investigation may be quoted as a gate
/// result -- see report §7.1: the host was contended throughout, and
/// p95 (unlike min/p50) did not reproduce across runs. The latency measurement re-measures
/// on a quiet host.
const MEASURED_INTRA_THREADS_CAP: usize = 4;

/// Threading and concurrency configuration for [`Model::load`].
///
/// `intra_threads` bounds how many OS threads a single ONNX Runtime
/// inference call may use; `inference_slots` sizes the semaphore that
/// bounds how many [`Model::detect`] calls may be mid-flight at once.
/// Design section 12: ONNX thread count and HTTP concurrency are tuned
/// together, not independently -- see [`ModelConfig::from_env`]'s defaults,
/// [`MEASURED_INTRA_THREADS_CAP`]'s docs for where the `intra_threads`
/// default is measured from, and the latency measurement's thread matrix, which is where
/// these get re-measured on a quiet, real deployment host rather than
/// trusted from a contended laptop.
#[derive(Debug, Clone, Copy)]
pub struct ModelConfig {
    pub intra_threads: usize,
    pub inference_slots: usize,
}

impl ModelConfig {
    /// `PII_DETECTION_INTRA_THREADS`, defaulting to
    /// `available_parallelism().min(MEASURED_INTRA_THREADS_CAP)` -- see
    /// that constant's docs for the machine and sweep this cap is cited
    /// from, and why it is a default to override, not a constant to
    /// trust on different hardware.
    ///
    /// `PII_DETECTION_INFERENCE_SLOTS`, defaulting to a flat `1` rather than any
    /// formula derived from `intra_threads`. [`crate::session`]'s module
    /// docs explain why: one `Mutex`-guarded session is shared by every
    /// call, so slots beyond 1 buy no parallel ONNX compute today, only
    /// more callers queued on tokenising and waiting for the lock -- and
    /// with intra-op spinning on, each extra slot is an extra
    /// `intra_threads` worth of cores a queued caller can burn spinning
    /// (investigation report §4.2, §8 point 5: "keep it at 1 until the
    /// session-pool question is settled"). [`session::Session::open`]
    /// still enforces `inference_slots * intra_threads <= vCPUs` whenever
    /// spinning is on, regardless of how either number was set, so an
    /// operator who does override `PII_DETECTION_INFERENCE_SLOTS` upward gets a
    /// refusal to load rather than a silent oversubscription.
    pub fn from_env() -> ModelConfig {
        let available = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1);
        let intra_threads = env_usize("PII_DETECTION_INTRA_THREADS")
            .unwrap_or_else(|| available.min(MEASURED_INTRA_THREADS_CAP))
            .max(1);
        let inference_slots = env_usize("PII_DETECTION_INFERENCE_SLOTS")
            .unwrap_or(1)
            .max(1);
        ModelConfig {
            intra_threads,
            inference_slots,
        }
    }
}

impl Default for ModelConfig {
    fn default() -> ModelConfig {
        ModelConfig::from_env()
    }
}

fn env_usize(key: &str) -> Option<usize> {
    std::env::var(key)
        .ok()?
        .parse::<usize>()
        .ok()
        .filter(|&value| value > 0)
}

/// Errors from loading or running the model. Same discipline as
/// [`ArtifactError`]: every variant carries only a fixed name or a count,
/// never text read from a file, request text, or a token identifier --
/// `no_model_error_variant_carries_text_or_token_ids` (in
/// [`session`]'s test module) holds the line.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("graph label count does not match the manifest: declared {declared}, graph {graph}")]
    LabelCountMismatch { declared: usize, graph: usize },
    #[error("graph is missing an expected input: {name}")]
    MissingInput { name: &'static str },
    #[error("tokenizer has no pad token")]
    PadTokenMissing,
    #[error("inference failed")]
    Inference,
    /// Intra-op spinning is only safe while every slot's threads still fit
    /// inside the host's vCPUs -- a spinning thread burns a core even
    /// while idle, so oversubscribing turns the measured ~30% win into a
    /// collapse (investigation report §4.1: 117.8 ms against 27.5 ms for
    /// the same graph, `intra_threads=10` vs `=6` spinning, on a 10-vCPU
    /// host). [`session::Session::open`] refuses to build a session in
    /// this state rather than silently running it.
    #[error(
        "inference_slots ({inference_slots}) x intra_threads ({intra_threads}) = {product} \
         exceeds the host's {vcpus} vCPUs with intra-op spinning enabled"
    )]
    UnsafeSpinningConfig {
        inference_slots: usize,
        intra_threads: usize,
        product: usize,
        vcpus: usize,
    },
    #[error(transparent)]
    Artifact(#[from] ArtifactError),
}

/// Fixed ASCII text with no personal data, used only to prime ONNX
/// Runtime's thread pools and lazy-initialised buffers before readiness is
/// reported (the design: "readiness passes only after a warm-up
/// inference"). Must stay free of anything resembling a real name, address
/// or identifier -- it is the one piece of "detected" text that could ever
/// safely reach a warm-up timing log, and only because it never changes.
const WARM_UP_TEXT: &str = "The quick brown fox jumps over the lazy dog.";

/// A loaded model: the ONNX session, the windower, and the artifact facts
/// [`Model::revision`] and [`Model::calibration_json`] expose to callers
/// that need them (the calibration, and `/health`).
pub struct Model {
    session: Arc<session::Session>,
    windower: Arc<Windower>,
    labels: Labels,
    revision: String,
    supported_languages: Vec<String>,
    calibration_json: String,
    semaphore: Arc<tokio::sync::Semaphore>,
    #[cfg(test)]
    in_flight_inferences: Arc<AtomicUsize>,
    #[cfg(test)]
    peak_concurrent_inferences: Arc<AtomicUsize>,
}

impl Model {
    /// Loads the artifact, the tokenizer/windower, and the ONNX session.
    /// Any of the three failing refuses the whole load: there is no
    /// partially-usable `Model`, matching the design's "refuses
    /// readiness on a mismatch" for the artifact and extending it to the
    /// graph and tokenizer this crate adds on top.
    pub fn load(dir: &Path, config: ModelConfig) -> Result<Model, ModelError> {
        let outcome = Self::load_inner(dir, config);
        if let Err(ref error) = outcome {
            // Safe to log: ModelError's Display carries only fixed names
            // and counts, the same discipline as ArtifactError -- see
            // `no_model_error_variant_carries_text_or_token_ids`.
            tracing::error!(%error, "model failed to load");
        }
        outcome
    }

    fn load_inner(dir: &Path, config: ModelConfig) -> Result<Model, ModelError> {
        let artifact = Artifact::load(dir)?;
        let windower = Windower::new(&dir.join("tokenizer.json"), artifact.manifest.windowing)
            .map_err(|_| ModelError::Inference)?;
        let session = session::Session::open(&artifact, config)?;

        Ok(Model {
            session: Arc::new(session),
            windower: Arc::new(windower),
            labels: artifact.labels,
            revision: artifact.manifest.revision,
            supported_languages: artifact.manifest.supported_languages,
            calibration_json: artifact.calibration_json,
            semaphore: Arc::new(tokio::sync::Semaphore::new(config.inference_slots.max(1))),
            #[cfg(test)]
            in_flight_inferences: Arc::new(AtomicUsize::new(0)),
            #[cfg(test)]
            peak_concurrent_inferences: Arc::new(AtomicUsize::new(0)),
        })
    }

    /// The manifest revision this model was built from.
    pub fn revision(&self) -> &str {
        &self.revision
    }

    /// The language codes this artifact's manifest declares it was trained
    /// on (`supported_languages`).
    ///
    /// Read from the artifact rather than declared in Rust, for the same
    /// reason `windowing` and `num_labels` are: a
    /// retrained model with different coverage is a new artifact directory,
    /// not a code change. A constant in `pii-detection-api` would keep claiming six
    /// languages after an artifact trained on eight -- or on two -- was
    /// deployed, and the direction that drifts in is the dangerous one:
    /// coverage claimed and not held reads to a caller as quality it does not
    /// have.
    ///
    /// `/capabilities` publishes this beside the codes the API accepts, and
    /// `/analyze` reports per response which side of it the request fell on.
    /// See `pii_detection_api::wire::model_coverage`.
    pub fn supported_languages(&self) -> &[String] {
        &self.supported_languages
    }

    /// Temperate types this artifact's `labels.json` maps a BIO label to
    /// -- what `/capabilities` may honestly add to the rules engine's set.
    /// See [`Labels::mapped_types`].
    pub fn detected_types(
        &self,
    ) -> std::collections::BTreeSet<pii_detection_engine::entity::EntityType> {
        self.labels.mapped_types()
    }

    /// The artifact's calibration thresholds, still-JSON and unparsed --
    /// `pii-detection-resolve` is the one parser for this schema.
    pub fn calibration_json(&self) -> &str {
        &self.calibration_json
    }

    /// Detects candidates in `document`: windows it, runs every window of
    /// this one document through a single batched inference call, then
    /// decodes, reconciles and trims the result.
    ///
    /// Design section 6 step 4: bounded by an inference semaphore so
    /// request concurrency cannot oversubscribe ONNX threads. The permit
    /// is acquired here, on the async side, and held for the duration of
    /// the `spawn_blocking` call below -- `acquire_owned` rather than
    /// `acquire`, because the permit has to outlive this call's borrow of
    /// `&self` to move into a `'static` blocking closure. ONNX Runtime is
    /// synchronous and CPU-bound; running it on a tokio runtime worker
    /// would starve the async executor, so tokenizing and running happen
    /// on a blocking thread instead.
    pub async fn detect(&self, document: &Document) -> Result<Vec<Candidate>, ModelError> {
        let permit = Arc::clone(&self.semaphore)
            .acquire_owned()
            .await
            .map_err(|_| ModelError::Inference)?;

        #[cfg(test)]
        {
            let now = self.in_flight_inferences.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak_concurrent_inferences
                .fetch_max(now, Ordering::SeqCst);
        }

        let session = Arc::clone(&self.session);
        let windower = Arc::clone(&self.windower);
        let labels = self.labels.clone();
        // The closure below has to be 'static, so the document is copied
        // into an owned value up front (not because the copy is otherwise
        // wanted) rather than moving the caller's borrowed `&Document`.
        let owned_document = Document::new(document.as_str().to_string());

        let outcome = tokio::task::spawn_blocking(move || {
            run_detection(&session, &windower, &labels, &owned_document)
        })
        .await
        .map_err(|_| ModelError::Inference)?;

        #[cfg(test)]
        self.in_flight_inferences.fetch_sub(1, Ordering::SeqCst);

        drop(permit);
        outcome
    }

    /// Loads an artifact and warms it up in one step, yielding the only
    /// value `pii-detection-api`'s `router` will accept. See [`WarmModel`] for why
    /// the two are welded together rather than left as two calls a caller
    /// is trusted to make in order.
    ///
    /// Async because the warm-up inference is; the load half is blocking
    /// and CPU-bound (it SHA-256s ~135 MB and opens an ONNX session), so it
    /// goes to the blocking pool here rather than on the caller's behalf.
    pub async fn load_and_warm(dir: &Path, config: ModelConfig) -> Result<WarmModel, ModelError> {
        let owned = dir.to_path_buf();
        tokio::task::spawn_blocking(move || Model::load(&owned, config))
            .await
            .map_err(|_| ModelError::Inference)??
            .warm_up()
            .await
    }

    /// Runs [`Model::detect`] against a fixed, PII-free warm-up string and
    /// reports failure rather than letting a broken artifact panic the
    /// process. Design section 12: readiness passes only after this
    /// succeeds.
    ///
    /// **Takes `self` by value and returns a [`WarmModel`].** That is the
    /// whole guard: a `Model` that has not warmed up cannot be turned into
    /// the type the router accepts, and this is the only function in the
    /// workspace that constructs one. An earlier shape -- `warm_up(&self)
    /// -> Result<(), _>` called from `pii-detection-api`'s startup, with the
    /// caller expected to branch on the error -- was a guard nothing
    /// enforced: deleting the branch and keeping only its log line left
    /// every test in the workspace passing, and `router(Some(model))`
    /// accepted any `Model` at all.
    pub async fn warm_up(self) -> Result<WarmModel, ModelError> {
        let document = Document::new(WARM_UP_TEXT.to_string());
        self.detect(&document).await?;
        Ok(WarmModel(self))
    }

    /// The high-water mark of concurrent [`Model::detect`] calls that were
    /// past the semaphore acquire (and therefore doing tokenize/batch/run/
    /// decode work) at once. Test-only: proving the semaphore's bound
    /// requires a live counter around exactly that window, and shipping
    /// the bookkeeping in production would pay two atomics per request for
    /// a number nothing reads outside a test.
    #[cfg(test)]
    pub(crate) fn test_peak_concurrent_inferences(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.peak_concurrent_inferences)
    }
}

/// A [`Model`] that has completed a warm-up inference: the design's
/// *"readiness passes only after a warm-up inference"* as a type rather
/// than as a branch someone has to remember to write.
///
/// The field is private and [`Model::warm_up`] -- which consumes the
/// `Model` and only returns `Ok` after a real inference came back -- is the
/// only constructor. `pii_detection_api::routes::router` takes a `WarmModel` and
/// nothing else, so "ready" and "warmed up" are the same fact rather than
/// two facts that agree by convention. There is deliberately no
/// `From<Model>`, no `new`, and no way back from `WarmModel` to an owned
/// `Model`.
///
/// It `Deref`s to `Model` because every *use* of a warm model -- `detect`,
/// `revision`, `detected_types`, `supported_languages`, `calibration_json`
/// -- is a `Model`
/// operation. `Deref` hands out a borrow, never ownership, so it cannot be
/// used to reconstruct the unwarmed value.
pub struct WarmModel(Model);

impl std::ops::Deref for WarmModel {
    type Target = Model;

    fn deref(&self) -> &Model {
        &self.0
    }
}

/// The synchronous half of [`Model::detect`]: window, batch-infer, decode,
/// reconcile, trim, extend a `STREET_ADDRESS` span over a dropped leading
/// prefix (`decode::extend_leading_prefix`) and over the city in the
/// address's tail position (`decode::extend_address_tail_city`), refine (`refine::refine`), convert. A free function rather
/// than a `Model` method
/// so it can run inside `spawn_blocking` from borrowed `Arc`s without
/// needing `Model` itself -- and its semaphore, and its test-only counters
/// -- to cross the blocking-thread boundary.
fn run_detection(
    session: &session::Session,
    windower: &Windower,
    labels: &Labels,
    document: &Document,
) -> Result<Vec<Candidate>, ModelError> {
    let windows = windower
        .windows(document)
        .map_err(|_| ModelError::Inference)?;

    let predictions = session.infer(&windows)?;
    let mut spans = Vec::new();
    for (index, (window, prediction)) in windows.iter().zip(predictions.iter()).enumerate() {
        spans.extend(decode::decode_window(
            window,
            index,
            &prediction.label_ids,
            &prediction.scores,
            labels,
        ));
    }
    let reconciled = decode::reconcile(spans);
    let trimmed = decode::trim_whitespace(document, reconciled);
    let prefixed = decode::extend_leading_prefix(document, trimmed);
    let with_city = decode::extend_address_tail_city(document, prefixed);
    let refined = refine::refine(document.as_str(), with_city);
    Ok(decode::to_candidates(refined))
}
