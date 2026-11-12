//! Derives a **passthrough artifact** from a shipped one: the same graph,
//! the same tokenizer, the same labels, and a `calibration.json` whose
//! thresholds are all `0.0`.
//!
//! # The problem it solves
//!
//! `pii-detection-eval calibrate` sweeps a threshold against a running service's
//! `/analyze` and fits the lowest value that meets its precision target.
//! But `/analyze` only ever returns candidates that already cleared
//! `calibration.json`'s *current* thresholds, and the shipped artifact
//! seeds both calibrated types at `0.50`. A sweep against it therefore
//! cannot observe a single candidate below `0.50`, and so cannot fit a
//! threshold below `0.50` -- not because `0.50` is right, but because the
//! population it measures was filtered before it arrived. The gate runner depends
//! on that sweep being able to see the whole distribution.
//!
//! # Why it is an artifact and not a switch
//!
//! The obvious fix is a runtime escape hatch -- `PII_DETECTION_CALIBRATION_PASSTHROUGH=1`,
//! say -- that makes the service skip the calibration gate. An earlier
//! draft of `calibrate.rs`'s docs promised exactly that. It was refused,
//! and this exists instead, because:
//!
//! * **There is no production code path.** Nothing in `pii-detection-api`,
//!   `pii-detection-model` or `pii-detection-resolve` knows this binary exists. There is
//!   no branch to flip, no variable to set, no header to send, and nothing
//!   for a review to have to keep watching. A fail-closed detection gate
//!   with a documented way to open it is no longer fail-closed; the gate
//!   here is untouched, and it is the *input* that changed.
//! * **It rides the unit of change the design already designates.** Design
//!   section 8 makes the artifact directory the thing that carries
//!   thresholds and the thing a manifest pins. A different set of
//!   thresholds is a different artifact. That is not a workaround, it is
//!   the schema working as written.
//! * **`/health` self-identifies it.** The generated `calibration.json`
//!   carries `revision: "passthrough-<date>"`, and `/health.calibration`
//!   reports that string verbatim. A passthrough service is therefore
//!   impossible to mistake for a production one from the outside, by
//!   anything that reads `/health` -- including a human running `curl`.
//!   The *model* revision in the manifest is deliberately left alone: the
//!   graph really is the shipped graph, and claiming otherwise would make
//!   the two revisions lie in the opposite direction.
//! * **It cannot reach an image by accident.** `.dockerignore` un-ignores
//!   exactly `models/pii-detection-model-full-merged-int8`; a passthrough
//!   directory sitting beside it is not even in the build context.
//!
//! # What it writes
//!
//! ```text
//! <out>/model.onnx        symlink -> the source artifact's own graph
//! <out>/tokenizer.json    symlink -> the source artifact's own tokenizer
//! <out>/labels.json       copied byte for byte
//! <out>/calibration.json  generated: every mapped type at 0.0
//! <out>/manifest.json     the source manifest, calibration digest re-pinned
//! ```
//!
//! The two large files are symlinked rather than copied: they are
//! byte-identical to the source by definition (`Artifact::load` verifies
//! the source's digests before anything is written, and the manifest's
//! entries for them are carried over unchanged), and copying 135 MB to say
//! so would be waste that also invites the two from drifting.
//!
//! The thresholds come from the artifact's *own* `labels.json`, through
//! `Labels::mapped_types()` -- the same function `/capabilities` uses --
//! never from a list written here. A retrained artifact that detects a
//! third type gets a passthrough that opens all three; a hardcoded pair
//! would silently keep gating the new one at `f32::INFINITY` and look like
//! a model that never fires.
//!
//! Finally the written directory is loaded back through the real
//! `Artifact::load` and `Calibration::from_json`, so this binary cannot
//! report success over a directory the service would refuse.
//!
//! # Usage
//!
//! ```text
//! cargo run --release -p pii-detection-eval --bin passthrough-artifact -- \
//!     --artifact models/pii-detection-model-full-merged-int8 \
//!     --out models/pii-detection-model-passthrough
//!
//! PII_DETECTION_ARTIFACT_DIR=models/pii-detection-model-passthrough \
//!     cargo run --release -p pii-detection-api &
//!
//! cargo run --release -p pii-detection-eval --bin calibrate -- \
//!     --service http://localhost:8090 --split calibration \
//!     --out models/pii-detection-model-full-merged-int8/calibration.json
//! ```
//!
//! Note which directory the fitted thresholds are written back to: the
//! *shipped* artifact. The passthrough directory is a measuring instrument
//! and is never itself deployed.
//!
//! It emits no text from any file it reads: the graph, the tokenizer and
//! the labels are moved or linked as bytes and never rendered, and every
//! failure below carries a path or a fixed name and nothing else.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

use pii_detection_eval::gates::wire_name;
use pii_detection_model::artifact::Artifact;
use pii_detection_resolve::Calibration;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

/// The threshold every mapped type gets. `Resolver::resolve` keeps a
/// candidate when `score >= threshold`, and every model score is a softmax
/// probability in `0.0..=1.0`, so `0.0` keeps all of them -- while still
/// being a value `Calibration::from_json` accepts as in range. There is no
/// "no threshold" spelling that would do this: an *absent* type is
/// `f32::INFINITY`, the opposite.
const PASSTHROUGH_THRESHOLD: f32 = 0.0;

const SMALL_FILE: &str = "labels.json";
const LINKED_FILES: [&str; 2] = ["model.onnx", "tokenizer.json"];
const CALIBRATION_FILE: &str = "calibration.json";
const MANIFEST_FILE: &str = "manifest.json";

// ---------------------------------------------------------------------------
// Failures
// ---------------------------------------------------------------------------

enum Failure {
    Usage,
    /// The source directory is not a loadable, digest-verified artifact.
    /// Carries the `ArtifactError`'s own rendering, which is name-and-count
    /// only and never file content.
    SourceRefused(String),
    /// The source artifact's `labels.json` maps no BIO label to any Temperate
    /// type, so a passthrough over it would open nothing.
    NothingToOpen,
    /// The output directory already exists. Never overwritten: the one
    /// mistake with real cost here is pointing `--out` at the shipped
    /// artifact and replacing its calibration in place.
    OutputExists(PathBuf),
    Unwritable(PathBuf),
    /// The manifest parsed but is not the shape a re-pin needs.
    ManifestShape,
    /// The directory this binary just wrote does not load, or its
    /// calibration does not parse. A bug here, reported rather than
    /// shipped.
    SelfCheckFailed(String),
}

impl Failure {
    fn report(&self) -> String {
        match self {
            Self::Usage => "usage: passthrough-artifact --artifact <dir> --out <dir>".to_string(),
            Self::SourceRefused(error) => {
                format!("the source artifact did not verify: {error}")
            }
            Self::NothingToOpen => {
                "the source artifact's labels.json maps no BIO label to a platform type, \
                 so a passthrough over it would open nothing"
                    .to_string()
            }
            Self::OutputExists(path) => format!(
                "{} already exists; refusing to overwrite an artifact directory",
                path.display()
            ),
            Self::Unwritable(path) => format!("cannot write {}", path.display()),
            Self::ManifestShape => {
                "the source manifest has no `files` object to re-pin".to_string()
            }
            Self::SelfCheckFailed(error) => format!(
                "the generated directory does not load as an artifact, which is a bug \
                 in this binary rather than a problem with the source: {error}"
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Arguments
// ---------------------------------------------------------------------------

struct Arguments {
    artifact: PathBuf,
    out: PathBuf,
}

fn parse_arguments(raw: &[String]) -> Result<Arguments, Failure> {
    let mut artifact = None;
    let mut out = None;
    let mut rest = raw.iter();
    while let Some(argument) = rest.next() {
        match argument.as_str() {
            "--artifact" => artifact = Some(PathBuf::from(rest.next().ok_or(Failure::Usage)?)),
            "--out" => out = Some(PathBuf::from(rest.next().ok_or(Failure::Usage)?)),
            _ => return Err(Failure::Usage),
        }
    }
    Ok(Arguments {
        artifact: artifact.ok_or(Failure::Usage)?,
        out: out.ok_or(Failure::Usage)?,
    })
}

// ---------------------------------------------------------------------------
// The calibration document
// ---------------------------------------------------------------------------

/// The passthrough `calibration.json`, rendered from the types the source
/// artifact actually maps.
///
/// `languages` is left empty deliberately. `Calibration::threshold` falls
/// back to `default` for any language with no block of its own, so one
/// `default` block opens every language at once -- and a per-language block
/// here would be six copies of the same number with six more chances to
/// leave one out.
fn calibration_document(types: &[&'static str], revision: &str, command: &str) -> Value {
    let thresholds: BTreeMap<&str, f32> = types
        .iter()
        .map(|wire| (*wire, PASSTHROUGH_THRESHOLD))
        .collect();
    json!({
        "revision": revision,
        "generated_by": command,
        "notes": [
            "PASSTHROUGH ARTIFACT -- NOT FOR DEPLOYMENT. Every threshold is 0.0, so \
             every model candidate reaches the response regardless of score. It exists \
             so `pii-detection-eval calibrate` can observe the whole score distribution, \
             including the part the shipped 0.50 seeds filter out before a sweep can \
             see it.",
            "A service running this artifact reports this revision on /health as \
             `calibration`, which is how a passthrough deployment is told apart from a \
             production one from the outside."
        ],
        "default": thresholds,
        "languages": {}
    })
}

/// `passthrough-YYYY-MM-DD` from a unix timestamp, in UTC.
///
/// The date is computed rather than pulled from a crate: this workspace has
/// no date dependency, and one function of civil-calendar arithmetic is
/// cheaper than acquiring one. Howard Hinnant's `civil_from_days`, which is
/// exact for every date this will ever see.
fn revision_for(unix_seconds: i64) -> String {
    let (year, month, day) = civil_from_days(unix_seconds.div_euclid(86_400));
    format!("passthrough-{year:04}-{month:02}-{day:02}")
}

fn civil_from_days(days_since_epoch: i64) -> (i64, u32, u32) {
    let z = days_since_epoch + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * shifted_month + 2) / 5 + 1) as u32;
    let month = match shifted_month < 10 {
        true => (shifted_month + 3) as u32,
        false => (shifted_month - 9) as u32,
    };
    match month <= 2 {
        true => (year + 1, month, day),
        false => (year, month, day),
    }
}

// ---------------------------------------------------------------------------
// The work
// ---------------------------------------------------------------------------

fn run(arguments: &Arguments, command: &str) -> Result<String, Failure> {
    // Verify the source before deriving anything from it. A passthrough
    // built from a corrupted artifact would be a corrupted artifact with a
    // reassuring name on it.
    let source = Artifact::load(&arguments.artifact)
        .map_err(|error| Failure::SourceRefused(error.to_string()))?;

    let types: Vec<&'static str> = source
        .labels
        .mapped_types()
        .into_iter()
        .map(wire_name)
        .collect();
    if types.is_empty() {
        return Err(Failure::NothingToOpen);
    }

    if arguments.out.exists() {
        return Err(Failure::OutputExists(arguments.out.clone()));
    }
    std::fs::create_dir_all(&arguments.out)
        .map_err(|_| Failure::Unwritable(arguments.out.clone()))?;

    // Absolute, so the symlinks keep resolving no matter which directory
    // the service is later started from.
    let source_dir = std::fs::canonicalize(&arguments.artifact)
        .map_err(|_| Failure::Unwritable(arguments.artifact.clone()))?;
    for linked in LINKED_FILES {
        let destination = arguments.out.join(linked);
        std::os::unix::fs::symlink(source_dir.join(linked), &destination)
            .map_err(|_| Failure::Unwritable(destination))?;
    }
    let labels = arguments.out.join(SMALL_FILE);
    std::fs::copy(source_dir.join(SMALL_FILE), &labels).map_err(|_| Failure::Unwritable(labels))?;

    let unix_seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs() as i64)
        .unwrap_or(0);
    let revision = revision_for(unix_seconds);
    let calibration = calibration_document(&types, &revision, command);
    let rendered =
        serde_json::to_string_pretty(&calibration).map_err(|_| Failure::ManifestShape)?;
    let calibration_path = arguments.out.join(CALIBRATION_FILE);
    std::fs::write(&calibration_path, format!("{rendered}\n"))
        .map_err(|_| Failure::Unwritable(calibration_path.clone()))?;

    // Re-pin. The source manifest is carried over as JSON rather than
    // through `Manifest` so that every field this crate does not model --
    // and every field a later manifest version adds -- survives untouched;
    // exactly one entry changes.
    let raw_manifest = std::fs::read_to_string(source_dir.join(MANIFEST_FILE))
        .map_err(|_| Failure::Unwritable(source_dir.join(MANIFEST_FILE)))?;
    let mut manifest: Value =
        serde_json::from_str(&raw_manifest).map_err(|_| Failure::ManifestShape)?;
    let written = std::fs::read(&calibration_path)
        .map_err(|_| Failure::Unwritable(calibration_path.clone()))?;
    let files = manifest
        .get_mut("files")
        .and_then(Value::as_object_mut)
        .ok_or(Failure::ManifestShape)?;
    files.insert(
        CALIBRATION_FILE.to_string(),
        json!(format!("sha256:{:x}", Sha256::digest(&written))),
    );
    let manifest_path = arguments.out.join(MANIFEST_FILE);
    let rendered_manifest =
        serde_json::to_string_pretty(&manifest).map_err(|_| Failure::ManifestShape)?;
    std::fs::write(&manifest_path, format!("{rendered_manifest}\n"))
        .map_err(|_| Failure::Unwritable(manifest_path))?;

    // The self-check. Both halves the service performs at startup, run
    // here: the digests must verify, and the calibration must parse.
    // Without it this binary could report success over a directory
    // `router_for_artifact` answers 503 for.
    let written_artifact = Artifact::load(&arguments.out)
        .map_err(|error| Failure::SelfCheckFailed(error.to_string()))?;
    let parsed = Calibration::from_json(&written_artifact.calibration_json)
        .map_err(|error| Failure::SelfCheckFailed(error.to_string()))?;
    if parsed.revision() != revision {
        return Err(Failure::SelfCheckFailed(
            "the written calibration revision is not the one generated".to_string(),
        ));
    }

    Ok(format!(
        "Wrote {} ({}), thresholds at {PASSTHROUGH_THRESHOLD} for: {}\n\
         NOT FOR DEPLOYMENT -- /health will report calibration \"{revision}\".",
        arguments.out.display(),
        revision,
        types.join(", ")
    ))
}

fn main() -> ExitCode {
    let command = std::env::args().collect::<Vec<_>>().join(" ");
    let raw: Vec<String> = std::env::args().skip(1).collect();

    let outcome = parse_arguments(&raw).and_then(|arguments| run(&arguments, &command));
    match outcome {
        Ok(message) => {
            println!("{message}");
            ExitCode::SUCCESS
        }
        Err(failure) => {
            eprintln!("{}", failure.report());
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn every_mapped_type_is_opened_and_nothing_else_is() {
        let document = calibration_document(&["PERSON", "STREET_ADDRESS"], "passthrough-x", "cmd");
        let default = document["default"].as_object().expect("default block");
        assert_eq!(default.len(), 2);
        for wire in ["PERSON", "STREET_ADDRESS"] {
            assert_eq!(default[wire], json!(0.0), "{wire} must be fully open");
        }
        assert!(
            document["languages"]
                .as_object()
                .expect("languages")
                .is_empty()
        );
    }

    /// The document this binary writes must be one the *service's* parser
    /// accepts, not merely valid JSON -- an out-of-range threshold or an
    /// unknown type name refuses readiness (see
    /// `crates/api/tests/readiness.rs`), so a passthrough artifact
    /// that could not load would be a 503 discovered at calibration time.
    #[test]
    fn the_generated_document_parses_as_a_calibration_and_keeps_nothing_closed() {
        let document = calibration_document(&["PERSON", "STREET_ADDRESS"], "passthrough-x", "cmd");
        let rendered = serde_json::to_string(&document).expect("render");
        let calibration = Calibration::from_json(&rendered).expect("the service must accept it");
        assert_eq!(calibration.revision(), "passthrough-x");
        for entity_type in [
            pii_detection_engine::entity::EntityType::Person,
            pii_detection_engine::entity::EntityType::StreetAddress,
        ] {
            // `nl` has no block of its own, so this also proves the
            // `default` fallback reaches an arbitrary language.
            assert_eq!(calibration.threshold(entity_type, "nl"), Some(0.0));
        }
    }

    /// The whole point of `0.0` rather than an omitted entry: a type this
    /// document does not name is `f32::INFINITY` in the resolver, which is
    /// the *opposite* of a passthrough. If a future edit "simplifies" the
    /// generated document to an empty `default` block, this fails.
    #[test]
    fn an_omitted_type_would_be_closed_not_open() {
        let empty = json!({"revision": "passthrough-x", "default": {}, "languages": {}});
        let calibration =
            Calibration::from_json(&empty.to_string()).expect("an empty calibration parses");
        assert_eq!(
            calibration.threshold(pii_detection_engine::entity::EntityType::Person, "nl"),
            None,
            "an unnamed type is uncalibrated, which the resolver treats as never matching"
        );
    }

    #[test]
    fn the_revision_is_the_utc_date_of_the_run() {
        // 2026-08-22T00:00:00Z, and one second before it.
        assert_eq!(revision_for(1_787_356_800), "passthrough-2026-08-22");
        assert_eq!(revision_for(1_787_356_799), "passthrough-2026-08-21");
        // A leap day, because the month arithmetic is where this kind of
        // function goes wrong: 2024-02-29T12:00:00Z.
        assert_eq!(revision_for(1_709_208_000), "passthrough-2024-02-29");
    }

    // ------------------------------------------------------------------
    // The generator itself, against a synthetic artifact.
    //
    // `Artifact::load` hashes the four declared files, parses `labels.json`
    // and checks its length against the manifest, and UTF-8 decodes
    // `calibration.json`. It never opens the ONNX graph or the tokenizer --
    // that is `pii-detection-model`'s `Session`/`Windower`, which this binary does
    // not use. So a complete, loadable artifact can be eleven bytes of
    // "graph", which is what makes running the real `run()` in a unit test
    // affordable at all.
    // ------------------------------------------------------------------

    struct TempDir(PathBuf);

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn tempdir() -> TempDir {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "pii-detection-passthrough-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("create tempdir");
        TempDir(dir)
    }

    fn synthetic_artifact(root: &Path) -> PathBuf {
        let dir = root.join("source");
        std::fs::create_dir_all(&dir).expect("create source");
        let files = [
            ("model.onnx", "not a graph".to_string()),
            ("tokenizer.json", "{}".to_string()),
            (
                "labels.json",
                json!({"O": null, "B-PERSON": "PERSON", "I-PERSON": "PERSON"}).to_string(),
            ),
            (
                CALIBRATION_FILE,
                json!({"revision": "seed", "default": {"PERSON": 0.5}, "languages": {}})
                    .to_string(),
            ),
        ];
        let mut digests = serde_json::Map::new();
        for (name, body) in &files {
            std::fs::write(dir.join(name), body).expect("write");
            digests.insert(
                (*name).to_string(),
                json!(format!("sha256:{:x}", Sha256::digest(body.as_bytes()))),
            );
        }
        let manifest = json!({
            "name": "synthetic",
            "revision": "synthetic-rev",
            "base_model": {"name": "n", "license": "l"},
            "architecture": "a",
            "parameters": 1,
            "tokenizer": {"name": "n", "kind": "k"},
            "windowing": {"max_length": 8, "stride": 2},
            "num_labels": 3,
            "quantization": {
                "scheme": "s", "dtype": "d", "config": "c", "fp32_sha256": "f"
            },
            "onnx_runtime_version_at_export": "0",
            "training_run": "r",
            "training_corpora": [],
            "supported_languages": ["nl"],
            "files": Value::Object(digests),
        });
        std::fs::write(
            dir.join(MANIFEST_FILE),
            serde_json::to_string_pretty(&manifest).expect("render manifest"),
        )
        .expect("write manifest");
        dir
    }

    /// The whole generator, end to end, including its self-check.
    ///
    /// The load at the end is the assertion that matters: it is the *same*
    /// `Artifact::load` the service runs at startup, so a manifest this
    /// binary forgot to re-pin, a calibration it wrote with a stale digest,
    /// or a symlink that does not resolve all show up here as a refusal
    /// rather than as a 503 discovered halfway through a calibration run.
    #[test]
    fn the_generated_directory_loads_as_an_artifact_and_opens_every_mapped_type() {
        let temp = tempdir();
        let source = synthetic_artifact(&temp.0);
        let out = temp.0.join("passthrough");

        let arguments = Arguments {
            artifact: source.clone(),
            out: out.clone(),
        };
        let message = run(&arguments, "unit test").map_err(|failure| failure.report());
        let message = match message {
            Ok(message) => message,
            Err(report) => panic!("{report}"),
        };
        assert!(message.contains("NOT FOR DEPLOYMENT"));

        // Loads through the service's own verifier, digests and all.
        let artifact = Artifact::load(&out).expect("the generated directory must load");
        assert_eq!(
            artifact.manifest.revision, "synthetic-rev",
            "the MODEL revision is the source's: the graph really is the source's graph"
        );

        let calibration =
            Calibration::from_json(&artifact.calibration_json).expect("calibration parses");
        assert!(
            calibration.revision().starts_with("passthrough-"),
            "/health.calibration is how a passthrough deployment identifies itself"
        );
        assert_eq!(
            calibration.threshold(pii_detection_engine::entity::EntityType::Person, "nl"),
            Some(0.0),
            "the source seeded PERSON at 0.5; the passthrough must open it"
        );

        // The two large files are the source's, by link rather than by copy.
        for linked in LINKED_FILES {
            let target = std::fs::read_link(out.join(linked)).expect("symlink");
            assert!(
                target.is_absolute(),
                "must resolve from any working directory"
            );
            assert_eq!(
                std::fs::read(out.join(linked)).expect("read through the link"),
                std::fs::read(source.join(linked)).expect("read source"),
            );
        }

        // Exactly one manifest entry changed.
        let read = |dir: &Path| -> Value {
            serde_json::from_str(
                &std::fs::read_to_string(dir.join(MANIFEST_FILE)).expect("read manifest"),
            )
            .expect("parse manifest")
        };
        let (mut before, mut after) = (read(&source), read(&out));
        assert_ne!(
            before["files"][CALIBRATION_FILE], after["files"][CALIBRATION_FILE],
            "the calibration digest must be re-pinned to the generated file"
        );
        before["files"][CALIBRATION_FILE] = Value::Null;
        after["files"][CALIBRATION_FILE] = Value::Null;
        assert_eq!(before, after, "nothing else in the manifest may change");
    }

    /// Pointing `--out` at a directory that exists is how someone
    /// accidentally rewrites the shipped artifact's calibration in place.
    #[test]
    fn an_existing_output_directory_is_refused_rather_than_overwritten() {
        let temp = tempdir();
        let source = synthetic_artifact(&temp.0);
        let arguments = Arguments {
            artifact: source.clone(),
            out: source,
        };
        assert!(run(&arguments, "unit test").is_err());
    }

    /// A source that does not verify must not become a passthrough that
    /// looks reviewed. One byte of the graph is enough.
    #[test]
    fn a_source_artifact_that_does_not_verify_is_refused() {
        let temp = tempdir();
        let source = synthetic_artifact(&temp.0);
        std::fs::write(source.join("model.onnx"), "not the graph it was").expect("corrupt");
        let arguments = Arguments {
            artifact: source,
            out: temp.0.join("passthrough"),
        };
        assert!(run(&arguments, "unit test").is_err());
    }

    #[test]
    fn arguments_require_both_directories() {
        assert!(parse_arguments(&[]).is_err());
        assert!(parse_arguments(&["--artifact".to_string(), "a".to_string()]).is_err());
        assert!(parse_arguments(&["--out".to_string(), "b".to_string()]).is_err());
        match parse_arguments(&[
            "--artifact".to_string(),
            "a".to_string(),
            "--out".to_string(),
            "b".to_string(),
        ]) {
            Ok(parsed) => {
                assert_eq!(parsed.artifact, Path::new("a"));
                assert_eq!(parsed.out, Path::new("b"));
            }
            Err(failure) => panic!("both flags given: {}", failure.report()),
        }
    }
}
