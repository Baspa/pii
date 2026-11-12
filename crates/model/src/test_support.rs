//! The one artifact-gating condition every artifact-dependent test in this
//! workspace shares.
//!
//! WHY THIS IS A MODULE AND NOT A COPIED HELPER. `crates/model/src/
//! session.rs` and `crates/api/tests/model.rs` both need "run this
//! test only if the real ~88 MB artifact is available, and make it obvious
//! when it is not". Two copies of that condition are free to drift, and the
//! direction they drift in is the one this project has already paid for
//! once: a documented verification command that silently skipped every
//! artifact-gated test while reporting "27 passed, 0 failed" (see
//! `.superpowers/sdd/2026-08-22-pii-detection-model/artifact-test-skip-report.md`).
//! One module, one `PII_DETECTION_ARTIFACT_DIR` resolution, one skip message.
//!
//! WHY IT IS FEATURE-GATED RATHER THAN ALWAYS COMPILED. `cfg(test)` does not
//! cross a crate boundary, so `pii-detection-api`'s integration tests cannot see
//! this crate's `#[cfg(test)]` items at all. A `test-support` feature is the
//! narrowest thing that does cross: `pii-detection-api` enables it from
//! `[dev-dependencies]` only, so `cargo build --release -p pii-detection-api` --
//! the command the Dockerfile runs -- never compiles any of this into the
//! shipped binary. `cfg(any(test, feature = "test-support"))` is what lets
//! this crate's own `#[cfg(test)]` modules use it without the feature.

use std::path::{Path, PathBuf};

/// The workspace root, derived at compile time from this crate's own
/// manifest directory rather than the test binary's *runtime* working
/// directory. `cargo test` runs a package's test binaries with the working
/// directory set to the package root (`crates/model`,
/// `crates/api`, ...), not the workspace root -- unlike `cargo run`,
/// which leaves the working directory as whatever it was invoked from. The
/// documented command (`PII_DETECTION_ARTIFACT_DIR=models/pii-detection-model-full-merged-int8
/// cargo test --workspace`, run from the workspace root) gives a path
/// relative to the workspace root, so resolving it against the *process* CWD
/// silently looked in `crates/<package>/models/...` -- which never exists --
/// and every gated test treated that as "unset" and skipped, with the run
/// still reporting "27 passed, 0 failed". See
/// `.superpowers/sdd/2026-08-22-pii-detection-model/artifact-test-skip-report.md`.
fn workspace_root() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
}

/// The three states `PII_DETECTION_ARTIFACT_DIR` can be in, kept distinct so a
/// skipped test can never be mistaken for one that actually ran:
///
/// * unset -- a legitimate skip: a developer who has not fetched the
///   artifact can still run the fast suite. [`print_uncaptured`] still puts
///   the skip in the terminal, so this is never silent.
/// * set but not a usable directory -- unambiguously a mistake (a typo, a
///   path that was never fetched, a stale env var). The macros below panic
///   rather than skip, because a skip here would hide a real configuration
///   error behind the same "27 passed" the documented command already
///   produced once.
/// * set and valid -- resolved relative to the workspace root (see
///   [`workspace_root`]) when the given path is relative, so the documented
///   command works regardless of which directory `cargo test` happens to run
///   the binary from.
pub enum ArtifactDirStatus {
    Unset,
    Invalid(PathBuf),
    Valid(PathBuf),
}

pub fn artifact_dir_status() -> ArtifactDirStatus {
    let Some(raw) = std::env::var_os("PII_DETECTION_ARTIFACT_DIR") else {
        return ArtifactDirStatus::Unset;
    };
    let raw = PathBuf::from(raw);
    let resolved = if raw.is_absolute() {
        raw
    } else {
        workspace_root().join(raw)
    };
    if resolved.is_dir() {
        ArtifactDirStatus::Valid(resolved)
    } else {
        ArtifactDirStatus::Invalid(resolved)
    }
}

/// What a skipped test prints. One string, shared, so the two suites cannot
/// tell a developer two different things about how to actually run them.
pub const SKIP_MESSAGE: &str = "SKIPPED (verified NOTHING): PII_DETECTION_ARTIFACT_DIR is not set. Run \
     `PII_DETECTION_ARTIFACT_DIR=models/pii-detection-model-full-merged-int8 cargo test --workspace` \
     from the workspace root to actually execute this test.";

/// Writes directly to the process's real stderr file descriptor, bypassing
/// the test harness's per-test output capture. Plain `eprintln!` goes
/// through `std::io::stderr()`, which the harness redirects to a per-test
/// buffer that is only flushed to the terminal for a *failed* test or under
/// `--nocapture` -- exactly how the previous `SKIPPED` message stayed
/// invisible in a normal `cargo test` run. A skip that only a special flag
/// can see is a skip that looks identical to a pass, which is the defect
/// this module exists to prevent; falls back to `eprintln!` if `/dev/stderr`
/// cannot be opened (still captured, but strictly no worse than the
/// previous behaviour).
pub fn print_uncaptured(message: &str) {
    use std::io::Write;
    match std::fs::OpenOptions::new().write(true).open("/dev/stderr") {
        Ok(mut stderr) => {
            let _ = writeln!(stderr, "{message}");
        }
        Err(_) => eprintln!("{message}"),
    }
}

/// Resolves `PII_DETECTION_ARTIFACT_DIR` to a `PathBuf`, loudly skipping the
/// calling test when it is unset and panicking when it is set to something
/// unusable. See [`ArtifactDirStatus`].
#[macro_export]
macro_rules! require_artifact {
    () => {
        match $crate::test_support::artifact_dir_status() {
            $crate::test_support::ArtifactDirStatus::Valid(dir) => dir,
            $crate::test_support::ArtifactDirStatus::Unset => {
                $crate::test_support::print_uncaptured($crate::test_support::SKIP_MESSAGE);
                return;
            }
            $crate::test_support::ArtifactDirStatus::Invalid(dir) => {
                panic!(
                    "PII_DETECTION_ARTIFACT_DIR is set but {dir:?} is not a usable directory \
                     -- fetch the artifact (see training/reports/artifact.md) or fix \
                     the path. This is a hard failure, not a skip: an explicitly set \
                     variable that does not resolve is always a mistake."
                );
            }
        }
    };
}

/// [`require_artifact!`] plus the load and the warm-up: expands to an
/// `Arc<WarmModel>`, the type `pii_detection_api::routes::router` accepts and the
/// state `pii-detection-api` calls ready. Async -- the warm-up is awaited -- so it
/// only works inside an `async fn`.
///
/// It goes through [`crate::Model::load_and_warm`] rather than doing its
/// own load-then-warm pair, so a test can never hold a model in a state
/// production cannot produce.
///
/// A load or warm-up failure panics rather than skipping. The variable was
/// set and resolved to a real directory, so anything that fails after that
/// is a broken artifact, not an absent one, and the two must never look the
/// same from the outside. `ModelError`'s `Display` carries only fixed names
/// and counts (`no_model_error_variant_carries_text_or_token_ids`), so
/// printing it in the panic message leaks nothing.
#[macro_export]
macro_rules! require_model {
    () => {{
        let dir = $crate::require_artifact!();
        match $crate::Model::load_and_warm(&dir, $crate::ModelConfig::default()).await {
            Ok(model) => std::sync::Arc::new(model),
            Err(error) => panic!(
                "PII_DETECTION_ARTIFACT_DIR resolved to a directory that would not load and \
                 warm up: {error}"
            ),
        }
    }};
}
