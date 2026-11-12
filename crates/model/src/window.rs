//! Turns a `Document` into one or more overlapping tokenizer windows, in
//! exact agreement with `training/align.py`'s `tokenize_and_align` -- itself
//! proved by `crates/model/tests/parity.rs` against a fixture of
//! Python's actual output (`training/tests/test_window_parity_fixture.py`
//! regenerates and re-checks `training/fixtures/window_parity.json`).
//!
//! # What unit does the Rust tokenizer report offsets in?
//!
//! Measured, not assumed. A throwaway
//! `examples/offset_unit.rs` loaded the shipped tokenizer and ran it against
//! `"Müller wohnt in Köln"` (20 code points, 22 bytes -- the umlauts each
//! cost an extra byte). Its output:
//!
//! ```text
//! code points: 20, bytes: 22
//! "▁Müller" -> (0, 7)
//! "▁woh" -> (8, 11)
//! "nt" -> (11, 13)
//! "▁in" -> (14, 16)
//! "▁Köln" -> (17, 22)
//! ```
//!
//! `"Müller"` is 6 code points but 7 bytes; the first token's reported end
//! is 7, matching the byte length, not the code-point length. The final
//! token ends at 22 -- the text's byte length, not its 20 code points.
//! **The `tokenizers` crate reports byte offsets into the input string**,
//! which the crate's own source confirms independently of this
//! measurement: `Tokenizer::encode` (`tokenizers` 0.21.4,
//! `src/tokenizer/mod.rs`) calls `encode_single_sequence(.., 0,
//! OffsetType::Byte)`. (A separate `encode_char_offsets` method exists that
//! passes `OffsetType::Char` instead -- this module deliberately does not
//! use it, both because `encode` is the direct analogue of what Python's
//! fast tokenizer is called with, and to keep the byte-to-code-point
//! conversion on the one path this workspace already trusts.)
//!
//! So every offset from `Encoding::get_offsets()` is converted through
//! [`Document::char_span`], exactly like every other recognizer in this
//! workspace converts once on the way out (see
//! `pii-detection-engine/src/text.rs`). A `None` from that conversion -- an offset
//! that does not fall on a code-point boundary -- becomes
//! [`WindowError::OffsetNotOnBoundary`], never a silent skip: offsets from a
//! tokenizer run against the same string `document.as_str()` always land on
//! boundaries, so seeing this variant means the windower was handed a
//! `Document` built from different text than it tokenized.

use std::path::Path;

use pii_detection_engine::text::{CharSpan, Document};
use tokenizers::{Encoding, Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy};

use crate::artifact::Windowing;

/// One tokenizer window: the model's raw inputs plus the code-point span
/// each token covers.
///
/// `spans[i]` is `None` for a special token -- the same positions
/// `align.py` gives `IGNORE_INDEX` -- and `Some(code-point span)` for a
/// real one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Window {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub token_type_ids: Vec<i64>,
    pub spans: Vec<Option<CharSpan>>,
    /// The code-point range this window's real (non-special) tokens span --
    /// the Rust equivalent of `align.window_real_range`.
    pub coverage: CharSpan,
}

/// Errors windowing can produce. No variant carries a digest, a path, or
/// any text read from the document or the tokenizer -- the design
/// forbids a text-bearing error message, and this is where a token or a
/// document fragment would leak first if it were allowed to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum WindowError {
    #[error("failed to load the tokenizer")]
    TokenizerLoad,
    #[error("failed to encode the document")]
    Encode,
    #[error("a token's offset does not fall on a code-point boundary")]
    OffsetNotOnBoundary,
    #[error("a window has no real (non-special) tokens")]
    EmptyWindow,
}

/// A loaded tokenizer configured with the manifest's windowing parameters,
/// turning a `Document` into windows the way `align.py` does.
pub struct Windower {
    tokenizer: Tokenizer,
}

impl Windower {
    /// Loads the tokenizer at `tokenizer_path` and configures truncation
    /// from `windowing`: `LongestFirst`, direction `Right`, matching
    /// `align.py`'s `tokenizer(text, ..., truncation=True,
    /// max_length=max_length, stride=stride)`.
    ///
    /// Padding is left disabled. `align.py` calls the tokenizer with
    /// `padding=False` and pads later, in the collator; the Rust side pads
    /// at the tensor level too, in the session code, not here -- turning padding on
    /// in the tokenizer itself would make this layer's output diverge from
    /// the fixture the windower is proved against.
    pub fn new(tokenizer_path: &Path, windowing: Windowing) -> Result<Windower, WindowError> {
        let mut tokenizer =
            Tokenizer::from_file(tokenizer_path).map_err(|_| WindowError::TokenizerLoad)?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                direction: TruncationDirection::Right,
                max_length: windowing.max_length,
                strategy: TruncationStrategy::LongestFirst,
                stride: windowing.stride,
            }))
            .map_err(|_| WindowError::TokenizerLoad)?;
        Ok(Windower { tokenizer })
    }

    /// Windows `document`, in the same order `align.py`'s
    /// `tokenize_and_align` returns them: the first window, then each
    /// overflow (stride) window in sequence -- the direct analogue of
    /// Python's `return_overflowing_tokens=True`.
    pub fn windows(&self, document: &Document) -> Result<Vec<Window>, WindowError> {
        let encoding = self
            .tokenizer
            .encode(document.as_str(), true)
            .map_err(|_| WindowError::Encode)?;

        std::iter::once(&encoding)
            .chain(encoding.get_overflowing().iter())
            .map(|raw| window_from_encoding(document, raw))
            .collect()
    }
}

/// Builds one [`Window`] from one raw `Encoding` (either the primary
/// encoding or one of its overflow pieces -- both have the same shape:
/// special tokens included, one offset pair per token).
fn window_from_encoding(document: &Document, encoding: &Encoding) -> Result<Window, WindowError> {
    let offsets = encoding.get_offsets();
    let special = encoding.get_special_tokens_mask();

    let mut spans = Vec::with_capacity(offsets.len());
    let mut coverage: Option<(u32, u32)> = None;
    for (&(start, end), &is_special) in offsets.iter().zip(special.iter()) {
        // is_special is the authoritative signal for CLS/SEP-type tokens.
        // start == end is kept as a defensive second check only, mirroring
        // align.py's _labels_for_window exactly -- never relied on alone,
        // since a real token could in principle start at character 0.
        if is_special == 1 || start == end {
            spans.push(None);
            continue;
        }
        let span = document
            .char_span(start, end)
            .ok_or(WindowError::OffsetNotOnBoundary)?;
        coverage = Some(match coverage {
            None => (span.start, span.end),
            Some((min_start, max_end)) => (min_start.min(span.start), max_end.max(span.end)),
        });
        spans.push(Some(span));
    }

    let (start, end) = coverage.ok_or(WindowError::EmptyWindow)?;

    Ok(Window {
        input_ids: encoding.get_ids().iter().map(|&id| id as i64).collect(),
        attention_mask: encoding
            .get_attention_mask()
            .iter()
            .map(|&mask| mask as i64)
            .collect(),
        token_type_ids: encoding
            .get_type_ids()
            .iter()
            .map(|&type_id| type_id as i64)
            .collect(),
        spans,
        coverage: CharSpan { start, end },
    })
}
