//! Differential tests against Python. The fixture is align.py's actual
//! output, committed by training/tests/test_window_parity_fixture.py --
//! not a hand-written expectation, which could only ever encode what the
//! author believed align.py does.
//!
//! Unlike `pii_detection_model::artifact`'s own tests, which build their
//! own tempdir fixtures and pass with `models/` moved aside, every test in
//! this file needs the real shipped tokenizer
//! (`models/pii-detection-model-full-merged-int8/tokenizer.json`): a differential
//! test proves nothing by comparing against a stub. Without `models/`
//! present, `cargo test -p pii-detection-model --test parity` fails at
//! `Windower::new` (`.expect("windower")` panics) rather than passing
//! silently -- deliberately: a test that reports green having verified
//! nothing is the exact failure mode this project has already logged nine
//! times over (see `.superpowers/sdd/2026-08-22-pii-detection-model/progress.md`).

use pii_detection_engine::entity::EntityType;
use pii_detection_engine::text::{CharSpan, Document};
use pii_detection_model::artifact::{Artifact, Labels, Windowing};
use pii_detection_model::decode::decode_window;
use pii_detection_model::window::{Window, WindowError, Windower};

#[derive(serde::Deserialize)]
struct Record {
    name: String,
    text: String,
    char_len: usize,
    windows: Vec<PyWindow>,
}

#[derive(serde::Deserialize)]
struct PyWindow {
    input_ids: Vec<i64>,
    attention_mask: Vec<i64>,
    spans: Vec<Option<[u32; 2]>>,
    /// Synthetic BIO ids -- see `training/tests/test_window_parity_fixture.py`'s
    /// module docs. Wire-safe `usize`: `IGNORE_INDEX` positions are already
    /// replaced with `LABEL2ID["O"]` (0) on the Python side, because
    /// `decode_window` never reads a label id at a position where `spans`
    /// is `None` in the first place.
    label_ids: Vec<usize>,
    /// `align.reconstruct_spans`'s own output for this window's
    /// `label_ids` -- what `decode_window` must reproduce exactly.
    decoded: Vec<Decoded>,
}

/// One `[start, end, entity_type]` triple from Python's `reconstruct_spans`,
/// deserialized through the plain 3-tuple JSON encodes as and converted to
/// named fields for readability at the call site.
#[derive(serde::Deserialize)]
#[serde(from = "(u32, u32, String)")]
struct Decoded {
    start: u32,
    end: u32,
    entity_type: String,
}

impl From<(u32, u32, String)> for Decoded {
    fn from((start, end, entity_type): (u32, u32, String)) -> Self {
        Decoded {
            start,
            end,
            entity_type,
        }
    }
}

fn fixture() -> Vec<Record> {
    let raw = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../training/fixtures/window_parity.json"
    ))
    .expect("parity fixture");
    serde_json::from_str(&raw).expect("parity fixture parses")
}

/// CI has no model yet: it opts out explicitly with `PII_DETECTION_SKIP_MODEL_TESTS=1`.
/// Without that, a missing model fails these tests, so a laptop never skips them by accident.
fn model_tests_skipped() -> bool {
    let skipped = std::env::var("PII_DETECTION_SKIP_MODEL_TESTS").is_ok_and(|value| value == "1");
    if skipped {
        eprintln!("skipped: PII_DETECTION_SKIP_MODEL_TESTS=1, so the shipped model is not loaded");
    }
    skipped
}

/// The shipped artifact directory. Not `pii_detection_model::artifact::Artifact`
/// -- these tests exercise the tokenizer directly and have no need for the
/// rest of the artifact (model.onnx, labels.json, calibration.json), so
/// loading it through the digest-verifying loader would only add an
/// unrelated failure mode to a test file that is already single-purpose.
fn artifact_dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../models/pii-detection-model-full-merged-int8"
    )
    .to_string()
}

fn tokenizer_path() -> std::path::PathBuf {
    std::path::Path::new(&artifact_dir()).join("tokenizer.json")
}

/// A `Windower` configured with the same windowing the shipped manifest
/// declares (256, 32) -- the values `training/tests/test_window_parity_fixture.py`
/// generated the fixture with.
fn default_windower() -> Windower {
    Windower::new(
        tokenizer_path().as_path(),
        Windowing {
            max_length: 256,
            stride: 32,
        },
    )
    .expect("windower")
}

/// The real shipped `labels.json`, decoded through `Artifact::load` --
/// unlike `artifact_dir()`'s own doc comment above, the decode parity test
/// below genuinely needs the label map (`decode_window`'s whole point is
/// consulting it), so it pays the full digest-verifying load rather than
/// parsing `labels.json` a second way.
fn shipped_labels() -> Labels {
    Artifact::load(std::path::Path::new(&artifact_dir()))
        .expect("valid shipped artifact")
        .labels
}

/// Converts a fixture window into the `pii_detection_model::window::Window` shape
/// `decode_window` consumes. `token_type_ids` is synthesized as all zeros
/// (the fixture does not carry it -- the windower always produces
/// zeros for single-sequence input, and nothing here reads it back).
/// `coverage` is recomputed the same way `window_from_encoding` computes it
/// in `src/window.rs`: the min start / max end over the window's real
/// (`Some`) spans.
fn to_window(window: &PyWindow) -> Window {
    let spans: Vec<Option<CharSpan>> = window
        .spans
        .iter()
        .map(|s| s.map(|[start, end]| CharSpan { start, end }))
        .collect();
    let mut coverage: Option<(u32, u32)> = None;
    for span in spans.iter().flatten() {
        coverage = Some(match coverage {
            None => (span.start, span.end),
            Some((min_start, max_end)) => (min_start.min(span.start), max_end.max(span.end)),
        });
    }
    let (start, end) = coverage.expect("fixture windows always have a real token");
    Window {
        input_ids: window.input_ids.clone(),
        attention_mask: window.attention_mask.clone(),
        token_type_ids: vec![0; window.input_ids.len()],
        spans,
        coverage: CharSpan { start, end },
    }
}

/// Placeholder per-token scores, uniform across the window. `decode_window`
/// takes a score per token because Rust's decode carries one and
/// `align.reconstruct_spans` does not (see `decode.rs`'s module docs); this
/// parity test exists to compare spans and types against Python's actual
/// output, never scores, so any constant value serves.
fn flat_scores(window: &PyWindow) -> Vec<f32> {
    vec![1.0; window.label_ids.len()]
}

/// `EntityType`'s wire name (`PERSON`, not `Person`) -- duplicated here
/// rather than depending on `pii-detection-eval` for one 12-line lookup table
/// (`pii-detection-model` has no dependency on `pii-detection-eval` and none of this
/// crate's other code needs one; see `pii_detection_eval::gates::wire_name`'s own
/// docs for why a compile-checked table beats a serde round trip for
/// exactly this purpose).
fn wire(entity_type: EntityType) -> &'static str {
    match entity_type {
        // The artifact's label map has no `CREDENTIAL` -- credentials come
        // from the rule packs, never from the model -- so this arm exists
        // only to keep the match exhaustive and can never be reached by a
        // decoded span.
        EntityType::Credential => "CREDENTIAL",
        EntityType::CreditCard => "CREDIT_CARD",
        EntityType::DrivingLicence => "DRIVING_LICENCE",
        EntityType::EmailAddress => "EMAIL_ADDRESS",
        EntityType::IbanCode => "IBAN_CODE",
        EntityType::IpAddress => "IP_ADDRESS",
        EntityType::NationalId => "NATIONAL_ID",
        EntityType::PassportNumber => "PASSPORT_NUMBER",
        EntityType::Person => "PERSON",
        EntityType::PhoneNumber => "PHONE_NUMBER",
        EntityType::StreetAddress => "STREET_ADDRESS",
        EntityType::TaxId => "TAX_ID",
        EntityType::VatNumber => "VAT_NUMBER",
    }
}

#[test]
fn the_rust_decoder_produces_the_same_spans_reconstruct_spans_does() {
    if model_tests_skipped() {
        return;
    }
    let labels = shipped_labels();
    for record in fixture() {
        for (index, window) in record.windows.iter().enumerate() {
            let ours = decode_window(
                &to_window(window),
                index,
                &window.label_ids,
                &flat_scores(window),
                &labels,
            );
            let theirs: Vec<(u32, u32, String)> = window
                .decoded
                .iter()
                .map(|d| (d.start, d.end, d.entity_type.clone()))
                .collect();
            let ours_tuples: Vec<(u32, u32, String)> = ours
                .iter()
                .map(|s| (s.span.start, s.span.end, wire(s.entity_type).to_string()))
                .collect();
            assert_eq!(ours_tuples, theirs, "{} window {index}", record.name);
        }
    }
}

#[test]
fn the_rust_windower_produces_the_same_windows_python_does() {
    if model_tests_skipped() {
        return;
    }
    let windower = default_windower();

    for record in fixture() {
        let document = Document::new(record.text.clone());
        assert_eq!(
            document.char_len(),
            record.char_len,
            "{}: char length",
            record.name
        );

        let ours = windower.windows(&document).expect("windows");
        assert_eq!(
            ours.len(),
            record.windows.len(),
            "{}: window count",
            record.name
        );

        for (index, (ours, theirs)) in ours.iter().zip(&record.windows).enumerate() {
            assert_eq!(
                ours.input_ids, theirs.input_ids,
                "{} window {index}: input ids",
                record.name
            );
            assert_eq!(
                ours.attention_mask, theirs.attention_mask,
                "{} window {index}: mask",
                record.name
            );
            let theirs_spans: Vec<Option<CharSpan>> = theirs
                .spans
                .iter()
                .map(|s| s.map(|[start, end]| CharSpan { start, end }))
                .collect();
            assert_eq!(
                ours.spans, theirs_spans,
                "{} window {index}: offsets",
                record.name
            );
        }
    }
}

#[test]
fn every_offset_is_a_code_point_offset_not_a_byte_offset() {
    if model_tests_skipped() {
        return;
    }
    // The defect this codebase has already paid for. On "Müller wohnt in
    // Köln" the byte length exceeds the code-point length, so a windower
    // that forwards byte offsets produces spans past the end of the
    // document and slices that start mid-word.
    let text = "Müller wohnt in Köln".to_string();
    assert!(
        text.len() > text.chars().count(),
        "the test text must be multibyte"
    );
    let document = Document::new(text);
    let windower = default_windower();
    for window in windower.windows(&document).expect("windows") {
        for span in window.spans.iter().flatten() {
            assert!(
                span.end as usize <= document.char_len(),
                "span {span:?} runs past the document's {} code points",
                document.char_len()
            );
            // Slicing is the real proof: Document::slice panics on a span
            // that is not on a code-point boundary of this document.
            let _ = document.slice(*span);
        }
    }
}

#[test]
fn the_windowing_parameters_come_from_the_manifest_not_from_a_constant() {
    if model_tests_skipped() {
        return;
    }
    // A retrained model at a different position limit must be a new
    // artifact directory, not a code change. If the
    // windower ignores its Windowing argument this test is the one that
    // notices.
    let document = Document::new("x ".repeat(4_000));
    let narrow = Windower::new(
        tokenizer_path().as_path(),
        Windowing {
            max_length: 64,
            stride: 8,
        },
    )
    .expect("windower");
    let wide = Windower::new(
        tokenizer_path().as_path(),
        Windowing {
            max_length: 256,
            stride: 32,
        },
    )
    .expect("windower");
    assert!(
        narrow.windows(&document).expect("windows").len()
            > wide.windows(&document).expect("windows").len(),
        "max_length 64 must produce more windows than 256"
    );
}

#[test]
fn a_window_with_no_real_tokens_is_an_explicit_error_not_a_fabricated_coverage() {
    if model_tests_skipped() {
        return;
    }
    // Measured directly against the shipped tokenizer: a purely-whitespace
    // string strips to input_ids == [<s>, </s>] -- zero real tokens.
    // align.py handles that fine (every label is IGNORE_INDEX, no
    // exception), but Window.coverage is a non-optional CharSpan, so there
    // is no value this crate could put there without inventing one. This
    // is exactly why training/tests/test_window_parity_fixture.py's
    // `parity_texts()` does not include a whitespace-only case: committing
    // one would make `the_rust_windower_produces_the_same_windows_python_does`
    // panic on `.expect("windows")` for a type-system mismatch, not a
    // genuine disagreement. Tested directly here instead.
    let document = Document::new("   \n\t   ".to_string());
    let windower = default_windower();
    assert_eq!(windower.windows(&document), Err(WindowError::EmptyWindow));
}
