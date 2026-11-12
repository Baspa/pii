use crate::error::ApiError;
use crate::wire::{
    ACCEPTED_LANGUAGES, AUTO, AnalyzeRequest, AnalyzeResponse, Entity, LanguageCapabilities,
    LanguageRegime, MAX_CHARACTERS, model_coverage, unverified_types,
};
use axum::Extension;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::{
    Json, Router,
    routing::{get, post},
};
use pii_detection_engine::candidate::Candidate;
use pii_detection_engine::entity::EntityType;
use pii_detection_engine::text::Document;
use pii_detection_model::{Model, WarmModel};
use pii_detection_resolve::{Calibration, Resolver};
use pii_detection_rules::matcher::Matcher;
use serde::Serialize;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

/// A deterministic hold point inside `/analyze`, compiled only under the
/// `test-support` feature -- see `Cargo.toml` for how that feature is
/// wired so it never reaches the shipped binary.
///
/// `tests/drain.rs` needs to prove that a request already past the
/// readiness gate and doing real work survives a `SIGTERM`, as opposed to
/// one still uploading its body, which the drain is correctly allowed to
/// refuse (see that file's module docs). Real detection is fast enough --
/// p95 29.2 ms, see `crate::shutdown`'s module docs -- that racing a
/// `SIGTERM` against it directly would be flaky by construction: it would
/// be a race between two processes' schedulers, not a deterministic
/// sequence, and this project has enough history with tests that pass
/// without checking anything. This makes the hold a fact instead.
#[cfg(feature = "test-support")]
pub mod test_support {
    /// Sleeps for `PII_DETECTION_TEST_STALL_MS` milliseconds, read fresh on
    /// every call rather than cached once -- this only ever runs inside
    /// a short-lived test process, where a handful of environment reads
    /// costs nothing. Zero, the default and the value in every
    /// configuration except the one test that sets it, is a no-op.
    pub async fn stall_if_requested() {
        let millis = std::env::var("PII_DETECTION_TEST_STALL_MS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0);
        if millis > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(millis)).await;
        }
    }
}

/// Design section 4's exact field list. `model` and `calibration` are
/// `null` -- not omitted, not an empty string -- on a deployment with no
/// artifact configured, so "this instance has no model" is distinguishable
/// on the wire from "this instance has a model that found nothing".
#[derive(Serialize)]
pub struct Health {
    pub ready: bool,
    /// The artifact manifest's `revision`, or `null` for a rules-only
    /// deployment.
    pub model: Option<String>,
    /// The concatenated rule pack versions, the same string
    /// `/capabilities` reports under `versions.rules`.
    ///
    /// **Wire change, service `0.1.0` onwards.** This field previously
    /// carried `env!("CARGO_PKG_VERSION")` -- the *service* build version,
    /// `"0.1.0"` -- which duplicated `revision` and said nothing about the
    /// packs. It now carries the concatenated pack versions (`"1.1.1"` for
    /// the packs as shipped), so `/health.rules` and
    /// `/capabilities.versions.rules` are the same fact rather than two
    /// fields with the same name and different meanings. Both strings are
    /// opaque to a client and neither is parsed anywhere in this
    /// workspace, but a monitor that pinned the literal `"0.1.0"` will see
    /// it change; `revision` is where the service build version still
    /// lives.
    pub rules: String,
    /// `calibration.json`'s own `revision`, or `null` when no artifact is
    /// loaded. A model revision without a calibration revision would leave
    /// no way to tell which thresholds a deployment is actually applying.
    pub calibration: Option<String>,
    /// This service build's own version.
    pub revision: &'static str,
}

#[derive(Serialize)]
pub struct Capabilities {
    pub entity_types: Vec<EntityType>,
    /// What this instance accepts versus what its model was trained on, and
    /// what the difference costs. A flat list of 24 codes used to sit here,
    /// which said only the first of the three -- see [`LanguageCapabilities`].
    pub languages: LanguageCapabilities,
    pub versions: Versions,
}

#[derive(Serialize)]
pub struct Versions {
    pub rules: String,
    /// The artifact revision, or `null` for a rules-only deployment. Same
    /// reason as [`Health::model`]: a client must be able to read the
    /// model's absence rather than infer it from a missing `PERSON` in
    /// `entity_types`.
    pub model: Option<String>,
    pub service: &'static str,
}

/// The types the loaded rule packs detect, derived rather than declared,
/// and cached because they genuinely are fixed at compile time.
fn rule_types() -> &'static BTreeSet<EntityType> {
    static DETECTED: OnceLock<BTreeSet<EntityType>> = OnceLock::new();
    DETECTED.get_or_init(|| {
        let mut types: BTreeSet<EntityType> = pii_detection_rules::pack::all()
            .iter()
            .flat_map(|pack| pack.recognizer.iter().map(|r| r.entity_type))
            .collect();
        types.insert(EntityType::PhoneNumber);
        types
    })
}

/// The types this *instance* detects: the loaded packs, plus whatever the
/// loaded artifact's `labels.json` maps a BIO label to. Derived rather than
/// declared, and no longer a compile-time constant -- which is why it takes
/// the model and no longer caches behind a `OnceLock`.
///
/// A hardcoded list would drift from the packs and from the artifact, and
/// the direction it drifts in is the dangerous one: a type advertised and
/// not detected reads to a workspace as protection it does not have.
///
/// `/capabilities` reports this set and `/analyze` validates against it, so
/// the two can never disagree about what is detected. Membership in
/// `EntityType` is not the same question: the enum carries the whole platform
/// taxonomy, including `DRIVING_LICENCE`, which waits on a pack, and --
/// on a rules-only deployment -- `PERSON` and `STREET_ADDRESS`, which come
/// from the model. Accepting one of those on `/analyze` would answer a
/// workspace migrating from a detector that does find them with `200 OK`
/// and an empty list -- a success response confirming no PII where nothing
/// ever looked. Design section 4.
///
/// The model half comes from [`Model::detected_types`], which reads the
/// artifact's own label map. The artifact maps ten of its twenty-five BIO
/// labels to null (`B-CREDIT_CARD`, `B-PHONE_NUMBER`, ...) because those
/// types belong to the rules; those contribute nothing here, so loading a
/// model widens this set by exactly the types the artifact actually
/// produces.
///
/// Public so an offline harness can ask the service itself which types it
/// detects rather than restating the list. A measurement of "the rules
/// engine" that requested a hand-written set of types would measure a
/// deployment nobody runs.
pub fn detected_types(model: Option<&Model>) -> BTreeSet<EntityType> {
    let mut types = rule_types().clone();
    if let Some(model) = model {
        types.extend(model.detected_types());
    }
    types
}

/// The types this instance detects that *only* the model produces: today
/// `PERSON` and `STREET_ADDRESS`, and empty on a rules-only deployment.
///
/// This is the exact price of an accepted-but-untrained language, which is
/// why it is derived from the artifact and the packs rather than written
/// down. The rules engine keys on format and checksum, not on prose (design
/// section 8), so every type a pack detects is detected as well in Polish as
/// in Dutch; what an untrained language loses is what nothing but the model
/// looks for. A pack that later gained a `PERSON` recognizer would shrink
/// this set on its own, and a hardcoded `[Person, StreetAddress]` would go on
/// telling callers a type was unverified after it stopped being.
pub fn model_only_types(model: Option<&Model>) -> Vec<EntityType> {
    let rules = rule_types();
    detected_types(model)
        .into_iter()
        .filter(|entity_type| !rules.contains(entity_type))
        .collect()
}

/// Shared handler state: the matcher, the resolver, the optional model, and
/// the readiness flag every handler consults first.
///
/// The matcher and resolver are each built once and held behind `Arc` rather
/// than reconstructed per request. `Matcher::new()` compiles every pattern
/// in every pack and parses phone region metadata, far too expensive to
/// repeat per call.
///
/// `ready` is an `AtomicBool` rather than a plain `bool` because it is a
/// runtime fact, not a build-time one: it is decided by whether the
/// artifact verified and warmed up (see [`crate::startup`]), and later
/// flipped exactly once more by [`AppState::begin_drain`].
///
/// **The one write this atomic ever receives after construction is
/// `begin_drain`.** Design section 22's graceful drain -- stop accepting,
/// flip this flag, let in-flight requests finish -- lives in `main.rs`,
/// which holds the `AppState` returned alongside the `Router` (see
/// `router_and_state` and `crate::startup::router_and_state_from_env`)
/// so it has something to call `begin_drain` on when a shutdown signal
/// arrives. See `crate::shutdown` for the signal wait and the bounded
/// window, and `begin_drain`'s own docs for why this is a one-way flip.
#[derive(Clone)]
pub struct AppState {
    matcher: Arc<Matcher>,
    resolver: Arc<Resolver>,
    model: Option<Arc<WarmModel>>,
    /// `calibration.json`'s own `revision`, read once at construction.
    /// Held here rather than asked of the `Resolver` on each `/health`:
    /// the resolver's job is to apply thresholds, not to describe itself,
    /// and this is the only caller that wants the label.
    calibration: Option<String>,
    ready: Arc<AtomicBool>,
}

impl AppState {
    /// The state a deployment with no artifact configured serves: rules
    /// only, ready, and with a resolver holding no calibration -- which
    /// means any model-sourced candidate would fail closed, and there are
    /// none to begin with.
    ///
    /// Public so `pii-detection-eval`'s `fixture-spans` binary measures the same
    /// assembly the endpoint serves rather than a second one of its own.
    pub fn rules_only() -> AppState {
        AppState {
            matcher: Arc::new(Matcher::new()),
            resolver: Arc::new(Resolver::new()),
            model: None,
            calibration: None,
            ready: Arc::new(AtomicBool::new(true)),
        }
    }

    /// The model this state runs, if any. Lets a caller that holds an
    /// `AppState` ask [`detected_types`] the right question.
    pub fn model(&self) -> Option<&Model> {
        self.model.as_deref().map(|warm| &**warm)
    }

    /// Flips readiness to `false`. Design section 22's graceful drain
    /// calls this *before* it does anything else -- before axum's own
    /// accept loop stops taking new connections, and before its
    /// per-connection graceful shutdown begins -- so `/health`,
    /// `/capabilities` and `/analyze` already answer the same `503` an
    /// unready deployment always answered while whatever is already in
    /// flight is still being served to completion. See `crate::shutdown`
    /// for what actually waits those requests out; this method only
    /// ever turns the flag off.
    ///
    /// **One-way.** There is no matching "undrain": a service that was
    /// already refusing readiness for its own reason -- a missing or
    /// corrupt artifact, `router_unready` -- must not be waved back to
    /// `ready: true` by a shutdown signal it happens to also receive.
    /// Calling this twice, or calling it on a state that was never
    /// ready, is harmless -- it is a single relaxed store either way.
    ///
    /// The wire body cannot always distinguish *why* `ready` is `false`
    /// -- the design fixes `Health`'s field list, and a draining
    /// rules-only deployment (`model: null` even when healthy) reads
    /// identically to an unconfigured one on the JSON body alone. That
    /// is not new: an unconfigured deployment and a corrupt-artifact one
    /// already collapse to the same body today, and the wire has never
    /// carried the "why" -- logs do (`crate::startup`'s `tracing::error!`
    /// lines, and the drain's own `tracing::warn!` in `main.rs`). A
    /// deployment running an artifact stays distinguishable even on the
    /// body alone: `model` and `calibration` stay populated through a
    /// drain, because this method touches only `ready`, and both are
    /// always `null` on a refusal that was never ready to begin with.
    pub fn begin_drain(&self) {
        self.ready.store(false, Ordering::Relaxed);
    }

    /// Whether this state currently reports itself ready -- the same
    /// atomic `/health` reads, so a test can assert on the fact a caller
    /// would see without going through HTTP.
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Relaxed)
    }
}

/// The router for a deployment that is up, serving, and running the model.
///
/// **It takes a [`WarmModel`], which is the only type in the workspace that
/// cannot be constructed without a completed warm-up inference.** Design
/// section 12 says readiness passes only after one, and this is where that
/// is enforced: a `Model` that loaded but never ran cannot be passed here
/// at all, so `ready = true` and "warmed up" are the same fact rather than
/// two that agree by convention. The previous signature took any `Model`
/// and set `ready = true` for it.
///
/// The resolver is built from that artifact's *own* `calibration.json`, so
/// the thresholds a deployment applies are the thresholds its manifest
/// pinned. A rules-only deployment is [`rules_only_router`], a separate
/// function rather than a `None` here, because it is a separate decision --
/// see [`crate::startup`].
///
/// A `calibration.json` that will not parse produces an *unready* router,
/// not a ready one with no calibration. The file was digest-verified before
/// it got here, so a parse failure means the contents are wrong, and a
/// resolver with no calibration applies `f32::INFINITY` to every
/// model-sourced candidate -- the model would load, warm up, run on every
/// request, and detect nothing, which is indistinguishable from a model
/// quality problem. Refusing readiness is the same answer the design
/// gives a digest mismatch, for the same reason.
pub fn router(model: Arc<WarmModel>) -> Router {
    router_and_state(model).0
}

/// Everything [`router`] builds, plus the [`AppState`] handle a graceful
/// drain flips. `router` stays the entry point for every call site that
/// only ever serves the `Router` -- effectively all of them; `main.rs` is
/// the one caller that also needs to reach [`AppState::begin_drain`] once
/// a shutdown signal arrives, and this is its seam. See
/// [`crate::startup::router_and_state_from_env`], which threads this all
/// the way from the process environment.
pub fn router_and_state(model: Arc<WarmModel>) -> (Router, AppState) {
    let calibration = match Calibration::from_json(model.calibration_json()) {
        Ok(calibration) => calibration,
        Err(error) => {
            // Safe to log: `CalibrationError`'s Display carries only a wire
            // type name and a JSON position, never file content.
            tracing::error!(%error, "calibration refused readiness");
            return unready_router_and_state();
        }
    };

    let state = AppState {
        matcher: Arc::new(Matcher::new()),
        resolver: Arc::new(Resolver::with_calibration(calibration.clone())),
        model: Some(model),
        calibration: Some(calibration.revision().to_string()),
        ready: Arc::new(AtomicBool::new(true)),
    };
    (router_with_state(state.clone()), state)
}

/// The router a deployment that deliberately runs no model serves: the rule
/// packs, ready, and honest about it -- `/capabilities` does not list
/// `PERSON`, and `/analyze` refuses a request for it with `400` rather than
/// answering `200` and an empty list.
///
/// **Reaching this is a decision, never an omission.** [`crate::startup`]
/// builds it only for an explicit `PII_DETECTION_RULES_ONLY=1`; a deployment that
/// merely fails to name an artifact gets [`router_unready`] instead. See
/// that module's docs for the accident this ordering exists to prevent.
pub fn rules_only_router() -> Router {
    rules_only_router_and_state().0
}

/// [`rules_only_router`], plus the [`AppState`] handle. See
/// [`router_and_state`]'s docs for why this pairing exists.
pub fn rules_only_router_and_state() -> (Router, AppState) {
    let state = AppState::rules_only();
    (router_with_state(state.clone()), state)
}

/// The router a process serves when it was configured with an artifact and
/// that artifact would not verify, load, or warm up.
///
/// Every endpoint refuses: `/health` and `/capabilities` answer `503`, and
/// `/analyze` answers `503` rather than falling back to the rules. Design
/// section 10: *"There is no degraded rules-only mode: a partial answer is
/// never presented as a safe one."* A container that started, served rules
/// only, and reported itself ready would silently halve the product, and
/// would look like a model quality problem to whoever noticed the missing
/// `PERSON` spans weeks later.
///
/// The process still stays up (see [`crate::startup`]): the design
/// says the service *refuses readiness*, and a container that exited
/// instead would be restarted forever with no `/health` to explain why.
pub fn router_unready() -> Router {
    unready_router_and_state().0
}

/// [`router_unready`], plus the [`AppState`] handle. See
/// [`router_and_state`]'s docs for why this pairing exists.
pub fn unready_router_and_state() -> (Router, AppState) {
    let state = AppState {
        matcher: Arc::new(Matcher::new()),
        resolver: Arc::new(Resolver::new()),
        model: None,
        calibration: None,
        ready: Arc::new(AtomicBool::new(false)),
    };
    (router_with_state(state.clone()), state)
}

fn router_with_state(state: AppState) -> Router {
    // Before `AppState` is handed to a single handler, so the recorder is
    // installed before anything can record into it. `handle()` is a
    // `OnceLock`: the second and later routers in a test process get the
    // recorder the first one installed.
    crate::telemetry::handle();

    Router::new()
        .route("/health", get(health))
        .route("/capabilities", get(capabilities))
        .route("/metrics", get(metrics))
        .route("/analyze", post(analyze))
        .layer(axum::middleware::from_fn(crate::request_id::attach))
        .with_state(state)
}

/// The scrape endpoint.
///
/// Deliberately *not* behind the readiness gate the other three sit behind.
/// An instance that refused readiness -- a corrupt artifact, a drain in
/// progress -- is exactly the instance whose counters someone needs to
/// read, and a `/metrics` that answered `503` in that state would go dark
/// at the only moment it matters.
async fn metrics() -> impl axum::response::IntoResponse {
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        crate::telemetry::handle().render(),
    )
}

/// `503` when the service is not ready, `200` when it is, with the same body
/// either way.
///
/// The status code carries the verdict on purpose. `pii-detection-api
/// --health-check` (see `main.rs`) reads the status line and never the body,
/// so no failure detail can reach a probe's output; making readiness visible
/// in the status line is what lets that probe stay body-blind and still
/// require a warmed-up model.
async fn health(State(state): State<AppState>) -> (StatusCode, Json<Health>) {
    let ready = state.ready.load(Ordering::Relaxed);
    let body = Health {
        ready,
        model: state
            .model
            .as_ref()
            .map(|model| model.revision().to_string()),
        rules: rules_version(),
        calibration: state.calibration.clone(),
        revision: env!("CARGO_PKG_VERSION"),
    };
    let status = match ready {
        true => StatusCode::OK,
        false => StatusCode::SERVICE_UNAVAILABLE,
    };
    (status, Json(body))
}

/// What this instance detects. Refuses while unready for the same reason
/// `/analyze` does: an instance whose artifact would not load cannot
/// honestly publish a capability list at all, and publishing the rules-only
/// list would be the degraded mode the design forbids, written down
/// as a promise.
async fn capabilities(State(state): State<AppState>) -> Result<Json<Capabilities>, ApiError> {
    if !state.ready.load(Ordering::Relaxed) {
        return Err(ApiError::NotReady);
    }
    Ok(Json(Capabilities {
        entity_types: detected_types(state.model()).into_iter().collect(),
        languages: LanguageCapabilities {
            accepted: ACCEPTED_LANGUAGES,
            // From the artifact's manifest, never from a constant in this
            // crate. `Model::supported_languages` reads `supported_languages`
            // straight off the manifest that was digest-verified at load, so
            // deploying an artifact trained on a different set publishes a
            // different set here with no code change -- which is the whole
            // reason `windowing` and `num_labels` live there too.
            model_trained: state
                .model()
                .map(|model| model.supported_languages().to_vec())
                .unwrap_or_default(),
            model_only_entity_types: model_only_types(state.model()),
        },
        versions: Versions {
            rules: rules_version(),
            model: state
                .model
                .as_ref()
                .map(|model| model.revision().to_string()),
            service: env!("CARGO_PKG_VERSION"),
        },
    }))
}

fn rules_version() -> String {
    pii_detection_rules::pack::all()
        .iter()
        .map(|pack| pack.version.as_str())
        .collect::<Vec<_>>()
        .join(".")
}

/// Counts taken before any work, so a rejection logs the same shape a
/// success does. A request that is refused still tells you how much was
/// asked of the service.
///
/// The default -- all zeroes -- is what a request whose body never parsed
/// gets, and a refusal before parsing genuinely did ask for nothing: the
/// counts are not unknown, they are zero. Nothing here is derived from the
/// content of the text, only from its size.
#[derive(Default)]
struct Counted {
    texts: usize,
    characters: usize,
}

impl Counted {
    fn of(request: &AnalyzeRequest) -> Self {
        Self {
            texts: request.texts.len(),
            characters: request.texts.iter().map(|text| text.chars().count()).sum(),
        }
    }
}

/// `/analyze`, wrapped in the one log event per request that the design
/// asks for.
///
/// The handler proper is [`analyze_counted`]; this layer exists only so that
/// success and refusal emit the *same* event shape. Every field below is a
/// number, a fixed string, or the random request identifier -- no text, no
/// span, no detected value, not a truncated excerpt of one. If a field that
/// would help debugging suggests itself here, that is exactly the change
/// `tests/privacy.rs` exists to reject.
async fn analyze(
    State(state): State<AppState>,
    Extension(request_id): Extension<crate::request_id::RequestId>,
    body: Result<Json<AnalyzeRequest>, JsonRejection>,
) -> Result<Json<AnalyzeResponse>, ApiError> {
    let started = std::time::Instant::now();
    let (counted, outcome) = analyze_counted(&state, body).await;

    let (entities, label) = match &outcome {
        Ok(response) => (response.results.iter().map(Vec::len).sum::<usize>(), "ok"),
        Err(error) => {
            metrics::counter!(crate::telemetry::REJECTIONS, "reason" => error.reason())
                .increment(1);
            (0, error.reason())
        }
    };

    metrics::counter!(
        crate::telemetry::REQUESTS,
        "endpoint" => "analyze",
        "outcome" => label,
    )
    .increment(1);
    metrics::counter!(crate::telemetry::TEXTS).increment(counted.texts as u64);
    metrics::counter!(crate::telemetry::CHARACTERS).increment(counted.characters as u64);
    metrics::histogram!(crate::telemetry::STAGE_SECONDS, "stage" => "total")
        .record(started.elapsed().as_secs_f64());

    // One of five fixed strings, four of them `ModelCoverage`'s own labels
    // and `none` for a request that never ran -- the same convention
    // `count_entities` uses for an absent country, and for the same reason: a
    // field present on some events and absent on others makes a log query
    // silently miss rows. It is a property of the deployment and the
    // requested code, never of the text.
    let coverage = match &outcome {
        Ok(response) => response.language.model_coverage.as_label(),
        Err(_) => "none",
    };

    tracing::info!(
        request_id = request_id.as_str(),
        texts = counted.texts,
        characters = counted.characters,
        entities,
        duration_ms = started.elapsed().as_millis() as u64,
        outcome = label,
        coverage,
        "analyze",
    );

    outcome.map(Json)
}

/// The request, and the counts describing it, on both paths.
///
/// Returns rather than `?`-propagates so that a refusal still carries what
/// was asked of the service. The counts are taken the moment the body
/// parses and before any validation, so the numbers a rejection logs are
/// the same numbers a success would have logged.
async fn analyze_counted(
    state: &AppState,
    body: Result<Json<AnalyzeRequest>, JsonRejection>,
) -> (Counted, Result<AnalyzeResponse, ApiError>) {
    // Before the body is even parsed: an unready service has no answer to
    // give, and the one thing it must not do is give the rules-only one.
    if !state.ready.load(Ordering::Relaxed) {
        return (Counted::default(), Err(ApiError::NotReady));
    }

    // See `test_support`'s module docs: a no-op everywhere except one
    // drain test, which needs a request that has passed this gate to stay
    // in flight for a controllable, non-zero amount of time.
    #[cfg(feature = "test-support")]
    test_support::stall_if_requested().await;

    // `JsonRejection`'s own message embeds the offending value verbatim, so
    // it is discarded rather than inspected or forwarded. See
    // `ApiError::MalformedRequest`.
    let Ok(Json(request)) = body else {
        return (Counted::default(), Err(ApiError::MalformedRequest));
    };

    let counted = Counted::of(&request);
    // The *mode*, never the code. A caller chooses the language, so a
    // `language="nl"` label would be caller-influenced -- bounded here by
    // the `LANGUAGES` check, but bounded by a validation somebody could
    // later relax. Two literals cannot be relaxed into a cardinality
    // problem. Recorded before validation, so a request refused for an
    // unsupported language still says which mode was asked for.
    let mode = match request.language.as_str() {
        AUTO => "auto",
        _ => "explicit",
    };
    metrics::counter!(crate::telemetry::LANGUAGE, "mode" => mode).increment(1);

    let outcome = analyze_validated(state, request).await;
    (counted, outcome)
}

/// Validation and detection, in the order they have always run: language,
/// then entity types, then per-text length.
///
/// **The language check is unchanged and stays unchanged: all 24 codes are
/// accepted, and only a code outside them is refused.** Design section 10a:
/// rejecting a Polish request would fail a customer the rules would have
/// served correctly, because a PESEL is matched by format and validated by
/// checksum whatever language the prose around it is in. What this function
/// now also does is *say* which side of the model's training the accepted
/// code fell on -- see [`AnalyzeResponse`].
async fn analyze_validated(
    state: &AppState,
    request: AnalyzeRequest,
) -> Result<AnalyzeResponse, ApiError> {
    let validation_started = std::time::Instant::now();

    if request.language != AUTO && !ACCEPTED_LANGUAGES.contains(&request.language.as_str()) {
        return Err(ApiError::UnsupportedLanguage(request.language));
    }

    let detected = detected_types(state.model());
    let mut requested = Vec::with_capacity(request.entities.len());
    for entity in &request.entities {
        // Both halves are the same refusal: a type this deployment has no
        // recognizer and no model label for is as unsupported as a type the
        // taxonomy has never heard of. See `detected_types`.
        let parsed = EntityType::from_wire(entity)
            .filter(|entity_type| detected.contains(entity_type))
            .ok_or_else(|| ApiError::UnsupportedEntityType(entity.clone()))?;
        requested.push(parsed);
    }
    // Recorded here rather than per check: `validate` is the cost of the
    // gate as a whole, and only a request that cleared it has a duration
    // that means anything -- a refusal's timing is the rejection counter's
    // story, not this histogram's.
    metrics::histogram!(crate::telemetry::STAGE_SECONDS, "stage" => "validate")
        .record(validation_started.elapsed().as_secs_f64());

    // Decided once the code is known to be one this service accepts, and
    // recorded on its own counter rather than as a second label on
    // `pii_detection_language_requests_total`: that one is taken before validation
    // (so a refusal still says which mode was asked for) and a request that
    // never ran has no regime. Two counters with honest denominators beat one
    // whose label set changes with the outcome.
    let coverage = model_coverage(
        state.model().map(Model::supported_languages),
        &request.language,
    );
    // Four fixed literals, chosen in `ModelCoverage::as_label` and bounded by
    // that enum -- never the caller's code, which is exactly the unbounded
    // label this metric has always refused to carry.
    metrics::counter!(crate::telemetry::COVERAGE, "coverage" => coverage.as_label()).increment(1);

    let mut results = Vec::with_capacity(request.texts.len());
    for text in request.texts {
        if text.chars().count() > MAX_CHARACTERS {
            return Err(ApiError::TextTooLong);
        }
        results.push(analyze_text(state, text, &requested, &request.language).await?);
    }

    Ok(AnalyzeResponse {
        language: LanguageRegime {
            model_coverage: coverage,
            unverified_entity_types: unverified_types(coverage, &model_only_types(state.model())),
            requested: request.language,
        },
        results,
    })
}

/// The whole detection path for one text: build the document, scan it with
/// the rules and the model at the same time, merge both sources into one
/// candidate list, then resolve it against the requested types.
///
/// Extracted and public so that an offline evaluation harness runs the same
/// code the endpoint runs, rather than a second assembly of the same calls.
/// Two assemblies would be free to drift (a different requested set, a
/// forgotten resolver step, a model that only one of them runs) and the
/// drift would show up as a benchmark number no deployment can reproduce.
///
/// Offsets in the returned entities are code points, because `Document`
/// converts every recognizer's byte offsets exactly once on the way out, and
/// the model's spans are in code points from the tokenizer's offset map.
pub async fn analyze_text(
    state: &AppState,
    text: String,
    requested: &[EntityType],
    language: &str,
) -> Result<Vec<Entity>, ApiError> {
    // Design section 6 steps 3 and 4: the model runs concurrently with the
    // rules, not after them.
    //
    // Rules are synchronous and CPU-bound, so they go to the blocking pool
    // too. The reason is executor starvation, not ordering: `join!` polls
    // its futures in order, so an inline `matcher.scan()` would run *before*
    // the model future is ever polled, not behind its semaphore permit --
    // it cannot be "serialised behind the permit" because the permit has not
    // been asked for yet. What it would actually do is occupy a runtime
    // worker for the whole scan, blocking every other task assigned to that
    // worker (including, on a single-worker runtime, the model future in
    // this very `join!`). `spawn_blocking` keeps the CPU-bound half off the
    // async workers so both halves make progress at once.
    let document = Arc::new(Document::new(text));
    let (rule_candidates, model_candidates) = tokio::join!(
        scan_rules(Arc::clone(&state.matcher), Arc::clone(&document)),
        scan_model(state.model.clone(), Arc::clone(&document)),
    );

    // Step 5: one candidate list carrying span, type, country, source and
    // raw score, whichever side produced it. The resolver below cannot tell
    // -- and must not need to tell -- which entries came from where, beyond
    // the `Source` it reads to pick a threshold table.
    let mut candidates = rule_candidates?;
    candidates.extend(model_candidates?);

    let resolve_started = std::time::Instant::now();
    let resolved = state
        .resolver
        .resolve_reporting(candidates, requested, language);
    metrics::histogram!(crate::telemetry::STAGE_SECONDS, "stage" => "resolve")
        .record(resolve_started.elapsed().as_secs_f64());
    metrics::counter!(crate::telemetry::PHONES_ATTRIBUTED)
        .increment(resolved.attributed_to_company as u64);

    count_entities(&resolved.kept);
    Ok(resolved.kept.iter().map(Entity::from).collect())
}

/// Counts what survived resolution, by type and by country.
///
/// Both label values are bounded by an enum and drawn from that enum's own
/// serialization rather than from anything the caller wrote: `entity_type`
/// is one of twelve, `country` one of twenty-seven or the literal `none`.
/// The span, the score and the text are not recorded anywhere -- the count
/// is the whole of what the design allows, and a "sample" field
/// here would be a detected value on a permanent surface.
///
/// An absent country becomes the literal `none` rather than a dropped
/// label: a label present on some series of a metric and absent on others
/// makes a Prometheus query silently miss rows, and silently missing rows
/// on a PII counter is the wrong failure.
fn count_entities(kept: &[Candidate]) {
    for candidate in kept {
        // Drawn from each enum's own `Serialize`, so the label can never
        // drift from the name `/analyze` puts on the wire, and a type
        // added to the taxonomy needs no second table here. A `String` per
        // candidate rather than a cached `&'static str`: `metrics` takes
        // an owned label either way, and a process-wide cache would put a
        // lock on the path of a service whose whole point is a 50 ms p95.
        let entity_type = wire_name(candidate.entity_type).unwrap_or_else(|| "unknown".to_string());
        let country = candidate
            .country
            .and_then(wire_name)
            .unwrap_or_else(|| "none".to_string());

        metrics::counter!(
            crate::telemetry::ENTITIES,
            "entity_type" => entity_type,
            "country" => country,
        )
        .increment(1);
    }
}

/// The wire name of an enum value: the string its `Serialize` produces, and
/// nothing else. `None` for anything that does not serialize to a string,
/// which no variant of either enum does.
fn wire_name<T: Serialize>(value: T) -> Option<String> {
    match serde_json::to_value(value).ok()? {
        serde_json::Value::String(name) => Some(name),
        _ => None,
    }
}

async fn scan_rules(
    matcher: Arc<Matcher>,
    document: Arc<Document>,
) -> Result<Vec<Candidate>, ApiError> {
    // Timed inside the closure, not around the `await`: the scan is what
    // this stage is, and the wait for a blocking-pool thread belongs to
    // neither engine. `total` is where queueing shows up.
    let (candidates, elapsed) = tokio::task::spawn_blocking(move || {
        let started = std::time::Instant::now();
        let candidates = matcher.scan(&document);
        (candidates, started.elapsed())
    })
    .await
    .map_err(|_| ApiError::DetectionUnavailable)?;

    metrics::histogram!(crate::telemetry::STAGE_SECONDS, "stage" => "rules")
        .record(elapsed.as_secs_f64());
    Ok(candidates)
}

/// The model half of the join. `None` -- a rules-only deployment -- yields
/// an empty candidate list, which is the correct merge input: there is no
/// model, so it contributed nothing. That is *not* the same as a model that
/// failed, which is an error the caller turns into a `503`; a deployment
/// that has no model also does not advertise `PERSON`, so nobody can have
/// asked for one.
async fn scan_model(
    model: Option<Arc<WarmModel>>,
    document: Arc<Document>,
) -> Result<Vec<Candidate>, ApiError> {
    match model {
        // No stage is recorded here on purpose. A rules-only deployment
        // did not run a model in zero seconds -- it did not run one -- and
        // a `stage="model"` series sitting at zero would read as a fast
        // model rather than an absent one.
        None => Ok(Vec::new()),
        Some(model) => {
            // Includes the wait for the inference semaphore's single
            // permit, which is deliberate: queueing behind another
            // request is latency a caller actually experiences, and
            // hiding it here would make `total` unexplainable from its
            // parts.
            let started = std::time::Instant::now();
            let detected = model.detect(&document).await;
            metrics::histogram!(crate::telemetry::STAGE_SECONDS, "stage" => "model")
                .record(started.elapsed().as_secs_f64());
            detected.map_err(|error| {
                // Safe to log: `ModelError`'s Display carries only fixed names
                // and counts -- `no_model_error_variant_carries_text_or_token_ids`
                // holds that line.
                tracing::error!(%error, "inference failed");
                ApiError::DetectionUnavailable
            })
        }
    }
}
