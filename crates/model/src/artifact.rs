//! Loads an artifact directory and verifies it before anything in it is
//! trusted.
//!
//! Design section 8: an artifact directory holds the ONNX graph, the
//! tokenizer, `labels.json` and `calibration.json`, with a manifest carrying
//! a digest for each. The service verifies all four at start and refuses
//! readiness on a mismatch. This module is that verification. It does not
//! open the ONNX graph, run the tokenizer, or interpret calibration
//! thresholds: those belong to later seams (Tasks 4, 6 and 7 respectively).
//! What it produces is a `labels.json` mapping already checked against the
//! manifest's declared label count, and a `calibration.json` that has been
//! digest-verified and decoded to UTF-8 but deliberately left unparsed, so
//! the threshold schema has exactly one parser (`pii-detection-resolve`).
//!
//! Every [`ArtifactError`] variant carries only a `&'static str` file name
//! (one of the artifact's known files, never a path) or a plain count. None
//! carries a digest, a label string, or any byte read from a file: design
//! section 13 forbids any text-bearing exception message, and an error type
//! is where that leaks first. `no_error_variant_carries_text_from_a_file`
//! asserts the shape rather than trusting it by inspection.

use pii_detection_engine::entity::EntityType;
use serde::Deserialize;
use serde::de::{MapAccess, Visitor};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

const MANIFEST_FILE: &str = "manifest.json";
const MODEL_FILE: &str = "model.onnx";
const TOKENIZER_FILE: &str = "tokenizer.json";
const LABELS_FILE: &str = "labels.json";
const CALIBRATION_FILE: &str = "calibration.json";

/// The four files the manifest must declare a digest for. Fixed by design
/// section 8's artifact contract: pii-detection-model does not merely hash unknown
/// files, it also has to know what to do with them (load the graph, feed the
/// tokenizer, decode labels, hand calibration to the resolver), so a name
/// outside this set is a manifest defect, not an extensibility point.
const REQUIRED_FILES: [&str; 4] = [MODEL_FILE, TOKENIZER_FILE, LABELS_FILE, CALIBRATION_FILE];

/// The windowing parameters the model was trained against. Carried in the
/// manifest deliberately, not as Rust constants: a retrained model with a
/// different position limit or stride is a new artifact directory, not a
/// code change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct Windowing {
    pub max_length: usize,
    pub stride: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BaseModel {
    pub name: String,
    pub license: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TokenizerInfo {
    pub name: String,
    pub kind: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Quantization {
    pub scheme: String,
    pub dtype: String,
    pub config: String,
    pub fp32_sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Corpus {
    pub name: String,
    pub license: String,
}

/// The manifest that pins an artifact directory: what it is, how it was
/// built, and a `sha256:`-prefixed digest for each file the service must
/// trust. `files` is a `BTreeMap` deliberately: verification below walks it
/// in sorted order, and the manifest's own key set is what drives which
/// files get verified.
#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub name: String,
    pub revision: String,
    pub base_model: BaseModel,
    pub architecture: String,
    pub parameters: u64,
    pub tokenizer: TokenizerInfo,
    pub windowing: Windowing,
    pub num_labels: usize,
    pub quantization: Quantization,
    pub onnx_runtime_version_at_export: String,
    pub training_run: String,
    pub training_corpora: Vec<Corpus>,
    pub supported_languages: Vec<String>,
    pub files: BTreeMap<String, String>,
}

/// `labels.json` decoded to platform types by position. Position is the model's
/// own label id (the order the file's keys appear on disk, which is the
/// order the model was trained against): not alphabetical, which is why
/// this is parsed with a manual `Visitor` rather than through
/// `serde_json::Map` (whose iteration order this workspace does not pin).
#[derive(Debug, Clone)]
pub struct Labels {
    entries: Vec<Option<EntityType>>,
}

impl Labels {
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Every platform type this label map actually produces, deduplicated --
    /// `B-PERSON` and `I-PERSON` are one type, and the ten labels
    /// `labels.json` maps to null contribute nothing.
    ///
    /// This is what `/capabilities` adds to the rules engine's own set when
    /// a model is loaded, derived from the artifact rather than declared in
    /// Rust: a retrained model with a different label set is a new artifact
    /// directory, and a hardcoded `[Person,
    /// StreetAddress]` in the API would keep advertising both after an
    /// artifact that no longer detects them was deployed.
    pub fn mapped_types(&self) -> BTreeSet<EntityType> {
        self.entries.iter().flatten().copied().collect()
    }

    /// Temperate type a model label id maps to, or `None` if that id is `O`
    /// or a BIO tag `labels.json` deliberately leaves unmapped (present and
    /// null: the only safe kind of absence; see [`ArtifactError::LabelCountMismatch`]
    /// for the unsafe kind). An out-of-range id is also `None`: nothing this
    /// crate returns can carry the label text that would explain why.
    pub fn platform_type(&self, label_id: usize) -> Option<EntityType> {
        self.entries.get(label_id).copied().flatten()
    }
}

#[cfg(test)]
/// Builds a `Labels` directly from an ordered entry list, for tests
/// elsewhere in this crate (`decode.rs`'s unit tests) that need a `Labels`
/// value without paying for a full digest-verified `Artifact::load` against
/// a tempdir fixture -- the shape `artifact.rs`'s own tests already build
/// via `valid_artifact`. `pub(crate)` and `cfg(test)`-gated: never part of
/// this crate's public API, and invisible to the differential tests in
/// `tests/parity.rs`, which load the real shipped artifact instead because
/// they exist to test agreement with it specifically.
pub(crate) fn labels_for_tests(entries: Vec<Option<EntityType>>) -> Labels {
    Labels { entries }
}

/// A verified artifact: the manifest, the directory it was loaded from, the
/// decoded label map, and calibration's raw (still-JSON) bytes as a string.
#[derive(Debug, Clone)]
pub struct Artifact {
    pub manifest: Manifest,
    pub dir: PathBuf,
    pub labels: Labels,
    /// The verified bytes of `calibration.json`, decoded to UTF-8 and
    /// nothing more. Parsing the threshold schema lives in `pii-detection-resolve`
    /// so it has exactly one parser; this crate never depends on
    /// the resolver.
    pub calibration_json: String,
    verified: Vec<String>,
}

impl Artifact {
    /// Loads and verifies an artifact directory.
    ///
    /// The order is the load contract and is deliberate:
    ///
    /// 1. Read and parse `manifest.json`.
    /// 2. For every file the manifest declares, in sorted order: read it,
    ///    hash it, compare the digest in constant time. Any of the four can
    ///    fail independently and is reported by name.
    /// 3. Only once all four have passed does anything get *interpreted*:
    ///    `labels.json` is parsed and its label count checked against the
    ///    manifest, `calibration.json` is decoded to UTF-8. A corrupt file
    ///    is reported as corrupt, never as a parse error that ran on
    ///    unverified bytes.
    pub fn load(dir: &Path) -> Result<Artifact, ArtifactError> {
        let outcome = Self::load_verified(dir);
        if let Err(ref error) = outcome {
            // Safe to log: ArtifactError's Display carries only a fixed file
            // name or a count, never a digest, a path, or file content:
            // `no_error_variant_carries_text_from_a_file` holds the line.
            tracing::error!(%error, "artifact verification refused readiness");
        }
        outcome
    }

    /// The names of the files verification actually hashed and compared,
    /// sorted. Exists so a loader that checks `model.onnx` and takes the
    /// other three on trust is distinguishable from one that verifies all
    /// four: the former would leave this short no matter which single file
    /// was mutated.
    pub fn verified_files(&self) -> Vec<String> {
        self.verified.clone()
    }

    fn load_verified(dir: &Path) -> Result<Artifact, ArtifactError> {
        let manifest_bytes =
            fs::read(dir.join(MANIFEST_FILE)).map_err(|_| ArtifactError::Missing {
                file: MANIFEST_FILE,
            })?;
        let manifest: Manifest =
            serde_json::from_slice(&manifest_bytes).map_err(|_| ArtifactError::Malformed {
                file: MANIFEST_FILE,
            })?;

        for required in REQUIRED_FILES {
            if !manifest.files.contains_key(required) {
                return Err(ArtifactError::Malformed {
                    file: MANIFEST_FILE,
                });
            }
        }

        let mut verified = Vec::with_capacity(manifest.files.len());
        let mut labels_bytes: Option<Vec<u8>> = None;
        let mut calibration_bytes: Option<Vec<u8>> = None;

        // BTreeMap::iter() yields keys in sorted order, which is the
        // "sorted order" the load contract specifies.
        for (name, digest_spec) in manifest.files.iter() {
            let file = known_file(name).ok_or(ArtifactError::Malformed {
                file: MANIFEST_FILE,
            })?;
            let bytes = fs::read(dir.join(name)).map_err(|_| ArtifactError::Missing { file })?;
            let expected_hex =
                digest_spec
                    .strip_prefix("sha256:")
                    .ok_or(ArtifactError::Malformed {
                        file: MANIFEST_FILE,
                    })?;
            let actual_hex = hex_sha256(&bytes);
            if !constant_time_eq(actual_hex.as_bytes(), expected_hex.as_bytes()) {
                return Err(ArtifactError::DigestMismatch { file });
            }
            verified.push(file.to_string());
            match file {
                LABELS_FILE => labels_bytes = Some(bytes),
                CALIBRATION_FILE => calibration_bytes = Some(bytes),
                _ => {}
            }
        }

        // Unreachable in practice: the required-file check above guarantees
        // both were present in manifest.files, and the loop above verifies
        // every entry in manifest.files before this point. Handled as
        // Missing rather than unwrapped so a future refactor that breaks
        // that invariant fails closed instead of panicking.
        let labels_bytes = labels_bytes.ok_or(ArtifactError::Missing { file: LABELS_FILE })?;
        let calibration_bytes = calibration_bytes.ok_or(ArtifactError::Missing {
            file: CALIBRATION_FILE,
        })?;

        let raw_labels: RawLabelFile = serde_json::from_slice(&labels_bytes)
            .map_err(|_| ArtifactError::Malformed { file: LABELS_FILE })?;
        let found = raw_labels.0.len();
        if found != manifest.num_labels {
            return Err(ArtifactError::LabelCountMismatch {
                declared: manifest.num_labels,
                found,
            });
        }

        let mut entries = Vec::with_capacity(found);
        for (_label, platform_type) in raw_labels.0 {
            match platform_type {
                None => entries.push(None),
                Some(wire) => {
                    let entity = EntityType::from_wire(&wire)
                        .ok_or(ArtifactError::Malformed { file: LABELS_FILE })?;
                    entries.push(Some(entity));
                }
            }
        }

        let calibration_json =
            String::from_utf8(calibration_bytes).map_err(|_| ArtifactError::Malformed {
                file: CALIBRATION_FILE,
            })?;

        Ok(Artifact {
            manifest,
            dir: dir.to_path_buf(),
            labels: Labels { entries },
            calibration_json,
            verified,
        })
    }
}

/// Maps a manifest-declared file name to the crate's own `&'static str`
/// constant for it. Never returns a `&'static str` built from the manifest's
/// own (runtime) text, so an `ArtifactError` can never carry anything read
/// from a file.
fn known_file(name: &str) -> Option<&'static str> {
    match name {
        MODEL_FILE => Some(MODEL_FILE),
        TOKENIZER_FILE => Some(TOKENIZER_FILE),
        LABELS_FILE => Some(LABELS_FILE),
        CALIBRATION_FILE => Some(CALIBRATION_FILE),
        _ => None,
    }
}

fn hex_sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Byte-for-byte equality that does not short-circuit on the first
/// difference, so how far two digests agree cannot be observed by timing.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .fold(0u8, |acc, (x, y)| acc | (x ^ y))
        == 0
}

/// `labels.json` deserialized as an ordered list of (BIO tag, platform type)
/// pairs, in file order. A plain `serde_json::Map` does not guarantee that
/// order without the `preserve_order` feature (not enabled in this
/// workspace); a hand-written `Visitor` gets it for free, because
/// `serde_json`'s deserializer always calls `visit_map` once per key in the
/// order the keys appear in the source document, regardless of what the
/// target type does with them.
struct RawLabelFile(Vec<(String, Option<String>)>);

impl<'de> Deserialize<'de> for RawLabelFile {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct RawLabelFileVisitor;

        impl<'de> Visitor<'de> for RawLabelFileVisitor {
            type Value = RawLabelFile;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a JSON object mapping a BIO label to a platform type or null")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut entries = Vec::with_capacity(map.size_hint().unwrap_or(0));
                while let Some(entry) = map.next_entry::<String, Option<String>>()? {
                    entries.push(entry);
                }
                Ok(RawLabelFile(entries))
            }
        }

        deserializer.deserialize_map(RawLabelFileVisitor)
    }
}

/// Every variant names a file (one of the crate's own `&'static str`
/// constants, never a path) or carries a plain count. Never a digest, never
/// a label string, never anything read from a file: see the module-level
/// docs and `no_error_variant_carries_text_from_a_file`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ArtifactError {
    #[error("artifact file missing: {file}")]
    Missing { file: &'static str },
    #[error("artifact file digest does not match the manifest: {file}")]
    DigestMismatch { file: &'static str },
    #[error("artifact file is malformed: {file}")]
    Malformed { file: &'static str },
    #[error("model label id has no entry: {index}")]
    UnknownLabel { index: usize },
    #[error("label count does not match the manifest: declared {declared}, found {found}")]
    LabelCountMismatch { declared: usize, found: usize },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// A minimal but valid manifest for the fixture written by
    /// `valid_artifact`. Every digest is computed from the bytes already on
    /// disk at `artifact`, so callers only ever need to touch one file (and,
    /// via `write_with_digest`, its manifest entry) to build each test case.
    fn manifest_json(artifact: &std::path::Path) -> String {
        let model_digest = hex_sha256(&fs::read(artifact.join("model.onnx")).expect("read"));
        let tokenizer_digest =
            hex_sha256(&fs::read(artifact.join("tokenizer.json")).expect("read"));
        let labels_digest = hex_sha256(&fs::read(artifact.join("labels.json")).expect("read"));
        let calibration_digest =
            hex_sha256(&fs::read(artifact.join("calibration.json")).expect("read"));

        format!(
            r#"{{
  "name": "test-fixture",
  "revision": "test-revision",
  "base_model": {{ "name": "test/base", "license": "MIT" }},
  "architecture": "BertForTokenClassification",
  "parameters": 1,
  "tokenizer": {{ "name": "test/tokenizer", "kind": "XLMRobertaTokenizerFast" }},
  "windowing": {{ "max_length": 256, "stride": 32 }},
  "num_labels": 25,
  "quantization": {{
    "scheme": "dynamic",
    "dtype": "int8",
    "config": "test",
    "fp32_sha256": "{fp32_digest}"
  }},
  "onnx_runtime_version_at_export": "1.29.0",
  "training_run": "test",
  "training_corpora": [],
  "supported_languages": ["en"],
  "files": {{
    "model.onnx": "sha256:{model_digest}",
    "tokenizer.json": "sha256:{tokenizer_digest}",
    "labels.json": "sha256:{labels_digest}",
    "calibration.json": "sha256:{calibration_digest}"
  }}
}}"#,
            fp32_digest = "0".repeat(64),
        )
    }

    /// The real shipped label map (models/pii-detection-model-full-merged-int8/labels.json):
    /// 25 BIO tags, only PERSON and STREET_ADDRESS mapped to a platform type.
    fn labels_json() -> String {
        r#"{
  "O": null,
  "B-CREDIT_CARD": null,
  "I-CREDIT_CARD": null,
  "B-DRIVING_LICENCE": null,
  "I-DRIVING_LICENCE": null,
  "B-EMAIL_ADDRESS": null,
  "I-EMAIL_ADDRESS": null,
  "B-IBAN_CODE": null,
  "I-IBAN_CODE": null,
  "B-IP_ADDRESS": null,
  "I-IP_ADDRESS": null,
  "B-NATIONAL_ID": null,
  "I-NATIONAL_ID": null,
  "B-PASSPORT_NUMBER": null,
  "I-PASSPORT_NUMBER": null,
  "B-PERSON": "PERSON",
  "I-PERSON": "PERSON",
  "B-PHONE_NUMBER": null,
  "I-PHONE_NUMBER": null,
  "B-STREET_ADDRESS": "STREET_ADDRESS",
  "I-STREET_ADDRESS": "STREET_ADDRESS",
  "B-TAX_ID": null,
  "I-TAX_ID": null,
  "B-VAT_NUMBER": null,
  "I-VAT_NUMBER": null
}"#
        .to_string()
    }

    fn calibration_json() -> String {
        r#"{
  "revision": "test-revision",
  "generated_by": "test-fixture",
  "default": {
    "PERSON": 0.5,
    "STREET_ADDRESS": 0.5
  },
  "languages": {}
}"#
        .to_string()
    }

    /// Writes `content` to `dir/file_name` and updates that file's entry in
    /// `manifest.json` to its digest, so the test exercises the
    /// parse/validate path for `file_name` rather than tripping the digest
    /// check first.
    fn write_with_digest(dir: &std::path::Path, file_name: &str, content: &str) {
        fs::write(dir.join(file_name), content.as_bytes()).expect("write file");
        let manifest_path = dir.join("manifest.json");
        let manifest_text = fs::read_to_string(&manifest_path).expect("read manifest");
        let mut manifest: serde_json::Value =
            serde_json::from_str(&manifest_text).expect("parse manifest");
        manifest["files"][file_name] =
            serde_json::Value::String(format!("sha256:{}", hex_sha256(content.as_bytes())));
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest).expect("serialize manifest"),
        )
        .expect("write manifest");
    }

    /// A bare-bones temp directory, cleaned up on drop. No external crate:
    /// this and the helpers above are test-only.
    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn tempdir() -> TempDir {
        // Tests run in parallel, in the same process, and this crate's own
        // tests proved nanosecond timestamps alone collide under that load
        // (macOS clock resolution is coarser than a nanosecond in practice).
        // A per-process atomic counter makes every call unique regardless of
        // clock resolution or thread timing.
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let unique = format!(
            "pii-detection-model-test-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        );
        let mut dir = std::env::temp_dir();
        dir.push(unique);
        fs::create_dir_all(&dir).expect("create tempdir");
        TempDir(dir)
    }

    /// Writes a minimal but structurally valid artifact directory and returns it.
    /// `model.onnx` is a placeholder byte string here: this module verifies
    /// digests and parses JSON, and never opens the graph. The session code is where the
    /// graph is loaded, and its tests use the real one.
    fn valid_artifact(dir: &std::path::Path) -> std::path::PathBuf {
        let artifact = dir.join("artifact");
        fs::create_dir_all(&artifact).expect("create dir");
        fs::write(artifact.join("model.onnx"), b"not-a-real-graph").expect("write");
        fs::write(artifact.join("tokenizer.json"), br#"{"version":"1.0"}"#).expect("write");
        fs::write(artifact.join("labels.json"), labels_json().as_bytes()).expect("write");
        fs::write(
            artifact.join("calibration.json"),
            calibration_json().as_bytes(),
        )
        .expect("write");
        fs::write(
            artifact.join("manifest.json"),
            manifest_json(&artifact).as_bytes(),
        )
        .expect("write");
        artifact
    }

    #[test]
    fn a_valid_artifact_loads() {
        let temp = tempdir();
        let dir = valid_artifact(temp.path());
        let artifact = Artifact::load(&dir).expect("valid artifact must load");
        assert_eq!(artifact.manifest.windowing.max_length, 256);
        assert_eq!(artifact.manifest.windowing.stride, 32);
        assert_eq!(artifact.manifest.num_labels, 25);
        assert_eq!(artifact.labels.len(), 25);
    }

    #[test]
    fn the_fp32_variant_s_quantization_shape_still_loads() {
        // Regression test for the defect fixed in training/package.py:
        // `package.py --variant fp32` used to write `"config": null` for
        // this block. `Quantization::config` is typed `String`, so
        // `serde_json` refused to decode it and `Artifact::load` returned
        // `Malformed { file: "manifest.json" }` for every fp32 artifact
        // ever built -- the variant switch was verified on the Python
        // side and never once loaded from Rust. package.py now writes
        // `"config": "none"`, mirroring `"scheme": "none"` right next to
        // it for the same "not applicable to this variant" case. This
        // pins that shape at the Rust boundary so it cannot regress back
        // to `null` unnoticed.
        let temp = tempdir();
        let dir = valid_artifact(temp.path());
        let manifest_path = dir.join("manifest.json");
        let manifest_text = fs::read_to_string(&manifest_path).expect("read manifest");
        let mut manifest: serde_json::Value =
            serde_json::from_str(&manifest_text).expect("parse manifest");
        manifest["quantization"] = serde_json::json!({
            "scheme": "none",
            "dtype": "float32",
            "config": "none",
            "fp32_sha256": "0".repeat(64),
        });
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest).expect("serialize manifest"),
        )
        .expect("write manifest");

        let artifact = Artifact::load(&dir).expect("fp32-shaped manifest must load");
        assert_eq!(artifact.manifest.quantization.scheme, "none");
        assert_eq!(artifact.manifest.quantization.dtype, "float32");
        assert_eq!(artifact.manifest.quantization.config, "none");
    }

    #[test]
    fn a_single_changed_byte_in_the_graph_is_refused() {
        // The whole point of the digest: "a rebuild cannot silently change
        // detection". One byte is the smallest change
        // that must still be caught.
        let temp = tempdir();
        let dir = valid_artifact(temp.path());
        fs::write(dir.join("model.onnx"), b"not-a-real-graphX").expect("write");
        assert!(matches!(
            Artifact::load(&dir),
            Err(ArtifactError::DigestMismatch { file: "model.onnx" })
        ));
    }

    #[test]
    fn a_changed_tokenizer_is_refused() {
        // A different tokenizer build gives different offsets for the same
        // text, and the symptom is spans that are subtly wrong rather than an
        // error. This is the digest that catches it.
        let temp = tempdir();
        let dir = valid_artifact(temp.path());
        fs::write(dir.join("tokenizer.json"), br#"{"version":"1.1"}"#).expect("write");
        assert!(matches!(
            Artifact::load(&dir),
            Err(ArtifactError::DigestMismatch {
                file: "tokenizer.json"
            })
        ));
    }

    #[test]
    fn a_changed_label_map_is_refused() {
        let temp = tempdir();
        let dir = valid_artifact(temp.path());
        let swapped = labels_json().replace("\"PERSON\"", "\"TAX_ID\"");
        fs::write(dir.join("labels.json"), swapped.as_bytes()).expect("write");
        assert!(matches!(
            Artifact::load(&dir),
            Err(ArtifactError::DigestMismatch {
                file: "labels.json"
            })
        ));
    }

    #[test]
    fn a_changed_calibration_is_refused() {
        let temp = tempdir();
        let dir = valid_artifact(temp.path());
        fs::write(dir.join("calibration.json"), b"{}").expect("write");
        assert!(matches!(
            Artifact::load(&dir),
            Err(ArtifactError::DigestMismatch {
                file: "calibration.json"
            })
        ));
    }

    #[test]
    fn a_missing_file_is_refused_by_name() {
        let temp = tempdir();
        let dir = valid_artifact(temp.path());
        fs::remove_file(dir.join("calibration.json")).expect("remove");
        assert!(matches!(
            Artifact::load(&dir),
            Err(ArtifactError::Missing {
                file: "calibration.json"
            })
        ));
    }

    #[test]
    fn all_four_files_are_verified_not_just_the_graph() {
        // The defect this catches: a loader that checks model.onnx and takes
        // the other three on trust would pass every test above except this
        // one, because each of those mutates a different file in isolation
        // and a partial verifier would still fail three of them. This asserts
        // the manifest's own file list is what drives verification, so adding
        // a fifth file to the manifest cannot leave it unverified.
        let temp = tempdir();
        let dir = valid_artifact(temp.path());
        let verified = Artifact::load(&dir).expect("valid").verified_files();
        assert_eq!(
            verified,
            vec![
                "calibration.json",
                "labels.json",
                "model.onnx",
                "tokenizer.json"
            ]
        );
    }

    #[test]
    fn a_label_map_that_does_not_cover_every_label_is_refused() {
        // A label the model can emit but the map does not mention would be
        // dropped silently at decode. Present-and-null is the only safe
        // absence, so a genuinely absent key is an error.
        let temp = tempdir();
        let dir = valid_artifact(temp.path());
        write_with_digest(
            &dir,
            "labels.json",
            &labels_json().replace(
                "\"B-VAT_NUMBER\": null,\n  \"I-VAT_NUMBER\": null",
                "\"B-VAT_NUMBER\": null",
            ),
        );
        assert!(matches!(
            Artifact::load(&dir),
            Err(ArtifactError::LabelCountMismatch {
                declared: 25,
                found: 24
            })
        ));
    }

    #[test]
    fn a_label_naming_a_type_the_platform_does_not_have_is_refused() {
        let temp = tempdir();
        let dir = valid_artifact(temp.path());
        write_with_digest(
            &dir,
            "labels.json",
            &labels_json().replace("\"PERSON\"", "\"BSN\""),
        );
        assert!(matches!(
            Artifact::load(&dir),
            Err(ArtifactError::Malformed {
                file: "labels.json"
            })
        ));
    }

    #[test]
    fn only_person_and_street_address_are_mapped_in_the_shipped_artifact() {
        let temp = tempdir();
        let dir = valid_artifact(temp.path());
        let artifact = Artifact::load(&dir).expect("valid");
        let mapped: std::collections::BTreeSet<_> = (0..artifact.labels.len())
            .filter_map(|i| artifact.labels.platform_type(i))
            .collect();
        assert_eq!(
            mapped,
            std::collections::BTreeSet::from([EntityType::Person, EntityType::StreetAddress])
        );
    }

    #[test]
    fn no_error_variant_carries_text_from_a_file() {
        // Design section 13 forbids any text-bearing exception message. An
        // error type is where that leaks first, so the shape is asserted
        // rather than reviewed by eye.
        let rendered = format!(
            "{} {} {} {}",
            ArtifactError::Missing {
                file: "labels.json"
            },
            ArtifactError::DigestMismatch { file: "model.onnx" },
            ArtifactError::Malformed {
                file: "calibration.json"
            },
            ArtifactError::LabelCountMismatch {
                declared: 25,
                found: 24
            },
        );
        for forbidden in ["PERSON", "sha256:", "/", "\\"] {
            assert!(
                !rendered.contains(forbidden),
                "error text leaked {forbidden:?}"
            );
        }
    }
}
