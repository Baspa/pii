//! Turning the process environment into the router the process serves.
//!
//! This is where the design's *"the service verifies all four at start
//! and refuses readiness on a mismatch"* becomes a behaviour instead of a
//! sentence, and it lives in the library rather than in `main.rs` so the
//! refusal can be constructed and tested. A readiness guard that only ever
//! runs in a container nobody points at a corrupted artifact is a guard
//! nobody has seen fail; `crates/api/tests/readiness.rs` points this
//! function at a missing directory, a digest mismatch, a manifest that
//! declares the wrong digest, a graph that will not open, a graph that opens
//! and will not *run*, and a `calibration.json` that will not parse, and
//! requires a `503` from each.
//!
//! # Rules-only is opt-in. Everything else fails closed.
//!
//! [`RULES_ONLY_VARIABLE`] set to exactly `1` selects a rules-only
//! deployment. **Nothing else does** -- and in particular, an unset
//! [`ARTIFACT_DIR_VARIABLE`] does not.
//!
//! THE ACCIDENT THIS ORDERING EXISTS TO PREVENT. The obvious shape --
//! "no artifact directory configured, so serve the rules" -- fails open on
//! the one case that happens by accident rather than by decision. Every
//! *tight* case was already correct: `-e PII_DETECTION_ARTIFACT_DIR=` (empty),
//! a directory that does not exist, and a directory that is empty all
//! refuse readiness. Only *absence* was permissive, and absence is
//! producible by a standard idiom:
//!
//! ```text
//! docker run -e PII_DETECTION_ARTIFACT_DIR pii-detection-api:local   # unset on the host
//! ```
//!
//! Passing `-e NAME` with no `=value` **deletes** the image's own `ENV` when
//! the variable is unset on the host. Compose's `environment: -
//! PII_DETECTION_ARTIFACT_DIR` has the same shape, and so does a line dropped from
//! a task definition. Under the old rule that container came up
//! `ready: true`, `model: null`, health checks green, serving half the
//! product -- exactly the silent degradation the design forbids,
//! arriving through the one door left open. Under this rule a dropped
//! variable behaves like an empty one already did: `503` everywhere.
//!
//! Rules-only stays genuinely reachable, because it is a legitimate
//! deployment -- it is just a decision now rather than an omission. It also
//! wins over a configured artifact directory, so the shipped image (whose
//! `ENV PII_DETECTION_ARTIFACT_DIR` a rules-only operator cannot easily unset) can
//! be run that way with one variable.
//!
//! # The outcomes
//!
//! * `PII_DETECTION_RULES_ONLY=1` -- a rules-only deployment. Ready, serving, and
//!   honest about it: `/capabilities` does not list `PERSON`, and
//!   `/analyze` refuses a request for it with `400` rather than answering
//!   `200` and an empty list.
//! * An artifact directory that verifies, loads and warms up -- the full
//!   product. Ready only after all three, which
//!   [`Model::load_and_warm`] enforces with a type rather than a branch.
//! * An artifact directory that fails any of the three -- **unready, and
//!   still up**. Every endpoint answers `503`. The process deliberately
//!   does not exit: section 8 says the service refuses *readiness*, and a
//!   container that exited instead would be restarted forever with no
//!   `/health` to explain why.
//! * Neither variable usable -- unready, for the same reason and with the
//!   same behaviour. An unconfigured process is a misconfigured one.
//!
//! There is no outcome where a broken or missing artifact degrades to the
//! rules. Design section 10: *"There is no degraded rules-only mode: a
//! partial answer is never presented as a safe one."*

use std::ffi::{OsStr, OsString};
use std::path::PathBuf;
use std::sync::Arc;

use axum::Router;
use pii_detection_model::{Model, ModelConfig};

use crate::routes::AppState;

/// The environment variable naming the artifact directory.
pub const ARTIFACT_DIR_VARIABLE: &str = "PII_DETECTION_ARTIFACT_DIR";

/// The environment variable that opts a deployment into rules-only.
pub const RULES_ONLY_VARIABLE: &str = "PII_DETECTION_RULES_ONLY";

/// The one value of [`RULES_ONLY_VARIABLE`] that selects rules-only.
///
/// Exactly this string, not "any non-empty value" and not a list of
/// friendly spellings. `-e PII_DETECTION_RULES_ONLY` with the variable unset on
/// the host, and `-e PII_DETECTION_RULES_ONLY=` with it empty, are the same
/// accidents [`ARTIFACT_DIR_VARIABLE`] already had to be hardened against;
/// neither may select the smaller product.
pub const RULES_ONLY_VALUE: &str = "1";

/// What the environment selected. Public so the selection can be tested
/// without a process-wide `set_var`, which is both `unsafe` and racy across
/// the parallel tests in one binary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Deployment {
    /// `PII_DETECTION_RULES_ONLY=1`, an explicit decision to run without a model.
    RulesOnly,
    /// An artifact directory to verify, load and warm up.
    Artifact(PathBuf),
    /// Neither. Not a mode -- a misconfiguration, served as `503`.
    Unconfigured,
}

/// Which deployment a given pair of environment values selects, as a pure
/// function of the two so it can be asserted directly.
///
/// Order matters and is deliberate: the explicit opt-in is read first, so
/// the shipped image -- which bakes `ENV PII_DETECTION_ARTIFACT_DIR` in -- can be
/// run rules-only by setting one variable rather than by unsetting one that
/// a `Dockerfile` put there. An empty `PII_DETECTION_ARTIFACT_DIR` is *not* mapped
/// to [`Deployment::Unconfigured`]: it stays an artifact directory that
/// cannot load, which refuses readiness, and both refusals are the same
/// `503`.
pub fn deployment_for(rules_only: Option<&OsStr>, artifact_dir: Option<&OsStr>) -> Deployment {
    if rules_only.is_some_and(|value| value == OsStr::new(RULES_ONLY_VALUE)) {
        return Deployment::RulesOnly;
    }
    match artifact_dir {
        Some(dir) => Deployment::Artifact(PathBuf::from(dir)),
        None => Deployment::Unconfigured,
    }
}

/// Builds the router from the process environment. The only thing `main.rs`
/// has to do beyond binding a listener.
pub async fn router_from_env() -> Router {
    router_and_state_from_env().await.0
}

/// [`router_from_env`], plus the [`AppState`] handle a graceful drain
/// flips. `main.rs` is the one caller that needs it -- to call
/// [`AppState::begin_drain`] when a shutdown signal arrives -- and calls
/// this instead of `router_from_env`; every other caller keeps using the
/// `Router`-only entry points, which are unchanged.
pub async fn router_and_state_from_env() -> (Router, AppState) {
    let rules_only: Option<OsString> = std::env::var_os(RULES_ONLY_VARIABLE);
    let artifact_dir: Option<OsString> = std::env::var_os(ARTIFACT_DIR_VARIABLE);

    // A value that is neither absent nor the one accepted spelling is
    // almost always a misspelt opt-in (`true`, `yes`, `on`). It is treated
    // as absent -- fail closed -- and said out loud, because the failure it
    // produces otherwise ("why is my rules-only container 503?") points at
    // the wrong variable. The value itself is never logged: it is
    // configuration, not request text, but the discipline of never printing
    // an environment value is cheaper to keep than to relitigate.
    if rules_only.is_some() && rules_only.as_deref() != Some(OsStr::new(RULES_ONLY_VALUE)) {
        tracing::warn!(
            variable = RULES_ONLY_VARIABLE,
            expected = RULES_ONLY_VALUE,
            "ignored: only the exact value selects a rules-only deployment"
        );
    }

    router_and_state_for(deployment_for(
        rules_only.as_deref(),
        artifact_dir.as_deref(),
    ))
    .await
}

/// Builds the router for an already-selected [`Deployment`]. See the module
/// docs for the outcomes.
///
/// Nothing this function logs carries a path, a digest, or a byte of any
/// file: `ModelError` and `ArtifactError` are both name-and-count only
///, and the mode lines below name the mode and nothing
/// else. A log that named the artifact path would be harmless today and
/// would be the obvious place for someone to add "and here is what was in
/// it" later.
pub async fn router_for(deployment: Deployment) -> Router {
    router_and_state_for(deployment).await.0
}

/// [`router_for`], plus the [`AppState`] handle. See
/// [`router_and_state_from_env`]'s docs for why this pairing exists.
pub async fn router_and_state_for(deployment: Deployment) -> (Router, AppState) {
    let dir = match deployment {
        Deployment::RulesOnly => {
            tracing::info!(
                mode = "rules-only",
                "rules-only deployment selected explicitly; serving rule packs only"
            );
            return crate::routes::rules_only_router_and_state();
        }
        Deployment::Unconfigured => {
            tracing::error!(
                mode = "unready",
                artifact_variable = ARTIFACT_DIR_VARIABLE,
                rules_only_variable = RULES_ONLY_VARIABLE,
                "no artifact directory configured and rules-only was not selected; \
                 refusing readiness"
            );
            return crate::routes::unready_router_and_state();
        }
        Deployment::Artifact(dir) => dir,
    };

    // Load *and* warm up, as one step that cannot come apart: `load_and_warm`
    // returns a `WarmModel`, the only value `routes::router` accepts, and it
    // is only constructible from a completed warm-up inference (design
    // section 12). There is no branch here to forget, and no way to hand the
    // router a model that has never run.
    match Model::load_and_warm(&dir, ModelConfig::default()).await {
        Ok(model) => {
            tracing::info!(
                mode = "rules+model",
                revision = model.revision(),
                "artifact verified and warmed up"
            );
            crate::routes::router_and_state(Arc::new(model))
        }
        Err(error) => {
            // `Model::load` has already logged a load-stage `ModelError`;
            // a warm-up failure is only logged here.
            tracing::error!(
                mode = "unready",
                %error,
                "an artifact directory was configured but the model would not load or \
                 warm up; refusing readiness"
            );
            crate::routes::unready_router_and_state()
        }
    }
}

/// Builds the router for a given artifact directory. Kept as its own entry
/// point because that is what the readiness suite drives; `router_for` is
/// the general form.
pub async fn router_for_artifact(dir: &std::path::Path) -> Router {
    router_for(Deployment::Artifact(dir.to_path_buf())).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn os(value: &str) -> OsString {
        OsString::from(value)
    }

    /// The regression this whole inversion exists for. `docker run -e
    /// PII_DETECTION_ARTIFACT_DIR` with the variable unset on the host deletes the
    /// image's `ENV`, and the process sees exactly this: both variables
    /// absent. It must not be a mode.
    #[test]
    fn a_dropped_artifact_variable_is_unconfigured_not_rules_only() {
        assert_eq!(deployment_for(None, None), Deployment::Unconfigured);
    }

    #[test]
    fn the_exact_opt_in_value_selects_rules_only() {
        assert_eq!(
            deployment_for(Some(OsStr::new("1")), None),
            Deployment::RulesOnly
        );
    }

    /// Rules-only must stay reachable from the shipped image, whose
    /// `Dockerfile` bakes `ENV PII_DETECTION_ARTIFACT_DIR=/opt/pii-detection/model` in and
    /// which an operator cannot unset with a `-e` flag.
    #[test]
    fn the_opt_in_wins_over_a_baked_in_artifact_directory() {
        assert_eq!(
            deployment_for(
                Some(OsStr::new("1")),
                Some(OsStr::new("/opt/pii-detection/model"))
            ),
            Deployment::RulesOnly
        );
    }

    /// `-e PII_DETECTION_RULES_ONLY` (unset on the host) and `-e
    /// PII_DETECTION_RULES_ONLY=` both arrive as an empty value. The smaller
    /// product may not be selected by an accident of shell quoting.
    #[test]
    fn an_empty_or_misspelt_opt_in_does_not_select_rules_only() {
        for value in ["", "true", "yes", "0", "on", "1 ", " 1"] {
            assert_eq!(
                deployment_for(Some(&os(value)), None),
                Deployment::Unconfigured,
                "{value:?} must not select rules-only"
            );
        }
    }

    #[test]
    fn an_artifact_directory_alone_selects_the_artifact() {
        assert_eq!(
            deployment_for(None, Some(OsStr::new("/opt/pii-detection/model"))),
            Deployment::Artifact(PathBuf::from("/opt/pii-detection/model"))
        );
    }

    /// An empty `PII_DETECTION_ARTIFACT_DIR` stays an artifact directory that will
    /// fail to load, rather than becoming "unconfigured". Both refuse
    /// readiness, so the distinction is only about which log line fires --
    /// but collapsing them would quietly widen `Unconfigured` into a
    /// catch-all, which is how the original fail-open crept in.
    #[test]
    fn an_empty_artifact_directory_is_still_an_artifact_directory() {
        assert_eq!(
            deployment_for(None, Some(OsStr::new(""))),
            Deployment::Artifact(PathBuf::new())
        );
    }
}
