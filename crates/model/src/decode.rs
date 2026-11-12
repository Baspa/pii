//! BIO decode, and window reconciliation under the spec's furthest-from-edge
//! rule.
//!
//! Three separable jobs, three functions, and the reason they are separate:
//! [`decode_window`] must agree with `training/align.py`'s
//! `reconstruct_spans` exactly, so it is testable against Python in
//! isolation (`crates/model/tests/parity.rs`, against a fixture of
//! Python's own committed output -- the same shape the windower used for the
//! windower). [`reconcile`] implements the *spec's* rule for resolving
//! windows that disagree, which `training/eval_fixture.py`'s
//! `_merge_touching_typed` does **not** implement -- see the module docs
//! below for the ruling. [`trim_whitespace`] is a Rust-only correction with
//! its own measurement. Fusing any of these would make the parity test
//! unable to isolate a disagreement in `decode_window` from a difference
//! introduced by reconciliation or trimming.
//!
//! # The ruling
//!
//! Design section 8: "Where windows disagree, the window that saw the span
//! furthest from its edge wins -- the rule the `deduplicate()` method of Temperate's earlier Azure client
//! already uses." `training/eval_fixture.py`'s `_merge_touching_typed`
//! stitches touching or overlapping same-type spans across windows instead.
//! They are not the same rule: merging joins a half-seen span from window A
//! to a half-seen span from window B; the spec's rule discards the
//! half-seen one entirely in favour of the window that saw the whole thing
//! away from its edge (see `a_half_seen_span_is_discarded_rather_than_joined_to_its_other_half`
//! below). The spec is the binding authority and matches what Temperate already
//! runs in production, so [`reconcile`] implements the spec's rule. The
//! 0.9145 fixture figure quoted elsewhere in this work was produced under
//! the merge rule; both are measured and reported (`training/reports/export.md`),
//! so a moved number is explained by this change rather than mistaken later
//! for a model regression.
//!
//! `margin`, precisely: for a span produced in a window whose real-token
//! coverage is `[c.start, c.end)`, `margin = min(span.start - c.start,
//! c.end - span.end)`. A span that touches its window's edge has margin 0.
//! Larger wins; ties break to the earlier window index, so the result is
//! deterministic. [`WindowSpan::window`] exists only to break that tie --
//! nothing else reads it.

use pii_detection_engine::candidate::{Candidate, Evidence, Source};
use pii_detection_engine::entity::EntityType;
use pii_detection_engine::text::{CharSpan, Document};

use crate::artifact::Labels;
use crate::cities;
use crate::window::Window;

/// One span a single window's decode produced: where, what, how confident,
/// and how far from that window's edge it sat -- the last of which is all
/// [`reconcile`] looks at to resolve a disagreement between windows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowSpan {
    pub span: CharSpan,
    pub entity_type: EntityType,
    pub score: f32,
    pub margin: u32,
    /// The index of the window that produced this span. Exists only to
    /// break a margin tie deterministically; nothing else reads it.
    pub window: usize,
}

/// Decodes one window's per-token label ids into character spans, mirroring
/// `align.py`'s `reconstruct_spans` token for token.
///
/// `label_ids[i]` and `scores[i]` are only ever read when `window.spans[i]`
/// is `Some` -- a special-token position is never consulted, exactly as
/// `reconstruct_spans` is driven by `lid == IGNORE_INDEX` rather than by
/// inspecting what the model predicted there. This is why a fixture-wire
/// `label_ids` value at a special position can be a placeholder (see
/// `training/tests/test_window_parity_fixture.py`'s module docs): nothing
/// here ever reads it.
///
/// Two differences from `reconstruct_spans`, and no others:
///
/// - it consults [`Labels::platform_type`], so a label id `labels.json` maps
///   to `null` behaves as `O` -- the ten structured types this model is not
///   calibrated for must never produce a span (see
///   `a_label_mapped_to_null_produces_no_span`).
/// - it carries a score: the mean over the span's tokens of the softmax
///   probability of the predicted label. Mean rather than min so that one
///   uncertain interior subtoken does not sink a well-recognised name, and
///   rather than max so that one confident token cannot carry an otherwise
///   weak span. This is Rust-only -- `align.py` has no notion of a score at
///   all -- which is exactly why the parity test compares spans and types,
///   never scores.
///
/// `reconstruct_spans` decides "B- tag, or a new type" by splitting the
/// label string on `-`. Rust's label ids carry no string; [`Labels`]
/// exposes only [`Labels::platform_type`], which collapses `B-PERSON` and
/// `I-PERSON` to the same `Some(EntityType::Person)`. `begins_type` below
/// recovers the same signal from that one method alone -- a label id
/// "begins" its type if the id one below it maps to a different type (or
/// there is no id below it) -- which holds because `labels.json` lists
/// every type's `B-` immediately followed by its `I-` (the same order
/// `align.py`'s own `LABELS` construction produces, and the order
/// `training/tests/test_export_onnx.py`'s label-ordering guard pins the
/// checkpoint to). No numeric parity is assumed; the derivation reads only
/// `platform_type`, so it does not care what the actual id values are.
pub fn decode_window(
    window: &Window,
    window_index: usize,
    label_ids: &[usize],
    scores: &[f32],
    labels: &Labels,
) -> Vec<WindowSpan> {
    // The three-way `zip` below stops at the shortest input rather than
    // erroring, so a caller that hands in a `label_ids`/`scores` shorter
    // than `window.spans` (a real possibility once the service wiring connects this to a
    // model output tensor) would silently decode a truncated window rather
    // than announcing the mismatch. `debug_assert!` rather than a runtime
    // `Result`: the contract is "one entry per token, same length as
    // `window.spans`" (this function's own doc comment and the brief's
    // interface), a caller bug rather than untrusted input, and this
    // module's error-free signature is deliberate (see the module docs).
    debug_assert_eq!(
        label_ids.len(),
        window.spans.len(),
        "label_ids must have one entry per token"
    );
    debug_assert_eq!(
        scores.len(),
        window.spans.len(),
        "scores must have one entry per token"
    );

    let mut spans = Vec::new();
    let mut open: Option<Open> = None;

    for ((span_opt, &label_id), &score) in
        window.spans.iter().zip(label_ids.iter()).zip(scores.iter())
    {
        let Some(real_span) = span_opt else {
            flush(&mut open, &mut spans, window.coverage, window_index);
            continue;
        };

        let Some(entity_type) = labels.platform_type(label_id) else {
            flush(&mut open, &mut spans, window.coverage, window_index);
            continue;
        };

        let begins = begins_type(labels, label_id, entity_type);
        let continues_open = !begins
            && open
                .as_ref()
                .is_some_and(|current| current.entity_type == entity_type);

        if continues_open {
            if let Some(current) = open.as_mut() {
                current.end = real_span.end;
                current.score_sum += score;
                current.score_count += 1;
            }
        } else {
            flush(&mut open, &mut spans, window.coverage, window_index);
            open = Some(Open {
                start: real_span.start,
                end: real_span.end,
                entity_type,
                score_sum: score,
                score_count: 1,
            });
        }
    }
    flush(&mut open, &mut spans, window.coverage, window_index);
    spans
}

/// The span currently being accumulated across consecutive same-type real
/// tokens -- `reconstruct_spans`'s `cur_start`/`cur_end`/`cur_type`, plus
/// the running sum this module adds for scoring.
struct Open {
    start: u32,
    end: u32,
    entity_type: EntityType,
    score_sum: f32,
    score_count: u32,
}

/// Closes the currently open span (if any), computing its margin from
/// `coverage` and its score as the mean of its accumulated token scores,
/// and pushes it onto `spans`. A no-op when nothing is open, exactly like
/// `reconstruct_spans`'s `flush()` when `cur_type is None`.
fn flush(
    open: &mut Option<Open>,
    spans: &mut Vec<WindowSpan>,
    coverage: CharSpan,
    window_index: usize,
) {
    let Some(current) = open.take() else {
        return;
    };
    // Every span flushed here was built only from real tokens' spans, and
    // `coverage` is defined as the min/max over those same real tokens
    // (the windower's `window_from_encoding`), so `current` always sits inside
    // `coverage`. `saturating_sub` is a defensive backstop, not a case this
    // module expects to hit.
    let margin_start = current.start.saturating_sub(coverage.start);
    let margin_end = coverage.end.saturating_sub(current.end);
    spans.push(WindowSpan {
        span: CharSpan {
            start: current.start,
            end: current.end,
        },
        entity_type: current.entity_type,
        score: current.score_sum / current.score_count as f32,
        margin: margin_start.min(margin_end),
        window: window_index,
    });
}

/// Whether `label_id` (already known to map to `entity_type`) is that
/// type's `B-` tag rather than its `I-` tag -- see this module's docs on
/// `decode_window` for why this is derived from `Labels::platform_type` alone.
fn begins_type(labels: &Labels, label_id: usize, entity_type: EntityType) -> bool {
    match label_id.checked_sub(1) {
        None => true,
        Some(previous_id) => labels.platform_type(previous_id) != Some(entity_type),
    }
}

/// Resolves spans from possibly-overlapping windows under the design's
/// rule: the window that saw a span furthest from its own edge wins.
/// Sorts by `(margin descending, window index ascending)`, then sweeps
/// keeping a span only if it overlaps nothing already kept -- the same
/// greedy shape `pii_detection_resolve::Resolver::resolve` uses for overlaps, so
/// the two read alike.
///
/// This is deliberately not `_merge_touching_typed`: a half-seen span
/// touching its window's edge is discarded outright when a whole span from
/// another window covers it, never stitched to it (see
/// `a_half_seen_span_is_discarded_rather_than_joined_to_its_other_half`).
///
/// The overlap check is type-blind, matching the spec's rule as written
/// ("the window that saw the span ... wins", not "the window that saw
/// this type of span"): a PERSON candidate and a STREET_ADDRESS candidate
/// from different windows that overlap the same characters are treated as
/// disagreement over one span, and only the higher-margin one survives.
/// `_merge_touching_typed` never merges across types, so in that specific
/// shape it would have kept both. For a masking product a dropped loser is
/// a leaked span, not merely a tidier answer -- worth knowing, though it is
/// what the design specifies (see `training/reports/export.md`'s
/// A later measurement, `nl-wp-warmtepomp-dn`, for a real case this measurement
/// found).
pub fn reconcile(spans: Vec<WindowSpan>) -> Vec<WindowSpan> {
    let mut ordered = spans;
    ordered.sort_by(|a, b| b.margin.cmp(&a.margin).then(a.window.cmp(&b.window)));

    let mut kept: Vec<WindowSpan> = Vec::new();
    for candidate in ordered {
        let overlaps = kept.iter().any(|existing| {
            candidate.span.start < existing.span.end && existing.span.start < candidate.span.end
        });
        if !overlaps {
            kept.push(candidate);
        }
    }

    kept.sort_by_key(|span| (span.span.start, span.span.end));
    kept
}

/// Trims leading and trailing whitespace code points off each span, and
/// drops a span entirely if nothing non-whitespace is left.
///
/// XLM-R's Metaspace pre-tokenizer attaches the preceding space to a token,
/// so a decoded span can start one code point early. Left uncorrected, the
/// pseudonymiser would eat a space the caller never asked to have changed.
pub fn trim_whitespace(document: &Document, spans: Vec<WindowSpan>) -> Vec<WindowSpan> {
    spans
        .into_iter()
        .filter_map(|mut window_span| {
            let mut start = window_span.span.start;
            let mut end = window_span.span.end;
            while start < end && is_whitespace_at(document, start) {
                start += 1;
            }
            while end > start && is_whitespace_at(document, end - 1) {
                end -= 1;
            }
            if start >= end {
                None
            } else {
                window_span.span = CharSpan { start, end };
                Some(window_span)
            }
        })
        .collect()
}

/// Whether the code point at `index` is whitespace. `document.slice` only
/// needs `index` and `index + 1` to be valid code-point positions in
/// `document`, which every position inside an already-decoded span is.
fn is_whitespace_at(document: &Document, index: u32) -> bool {
    document
        .slice(CharSpan {
            start: index,
            end: index + 1,
        })
        .chars()
        .next()
        .is_some_and(char::is_whitespace)
}

/// Case-insensitive street-type words and phrases that, standing whole
/// immediately before a model `STREET_ADDRESS` span, are pulled into it by
/// [`extend_leading_prefix`].
///
/// The model reliably finds the street name and building number but
/// routinely drops this leading qualifier: 19 of the 20 `STREET_ADDRESS`
/// misses the benchmark carried when this was measured (649 cases) are
/// exactly this
/// (`training/reports/street-prefixes.md`). Every entry is plain ASCII
/// lower-case, matched with an ASCII-only case fold (see
/// `leading_prefix_start`), so accented Spanish/Italian/French text
/// elsewhere in the document is never touched by the comparison.
///
/// Not exhaustive by design: only the prefixes the benchmark measured, plus
/// the handful of single-word siblings named in the task brief that carry
/// no realistic collision risk once the whole-word boundary check applies
/// (`avenida`, `corso`, `piazza`, `boulevard`, `strada`). A term left out
/// for being too collision-prone or not directly measured is named in the
/// report, not guessed at here -- a smaller correct list beats a larger
/// uncertain one.
///
/// `rue` and `avenue` were added later, once two more misses
/// (`training/reports/street-prefixes.md`'s follow-up section) measured
/// them directly: `56 rue du Pressoir` and `20 avenue du Vieux Moulin` both
/// have a model span that starts one word short of the true value (`du
/// Pressoir`, `du Vieux Moulin`), and only the bare word -- not the `rue
/// du`/`rue des` phrases already listed -- sits immediately before it. `rue`
/// and `avenue` are common French words (`rue` doubles as "regret" and the
/// herb in English prose, too), so both were left out of the first pass on
/// exactly the "unmeasured invention" grounds this doc used to describe
/// them; they are measured now. The boundary check below is what keeps
/// `rue` off the tail of `morue` (French for "cod": the character before
/// the match, `o`, is alphanumeric) the same way it already keeps `via` off
/// `previa`. A third measured miss in the same follow-up, English `Flat 2`,
/// is **not** in this list: the model emits no span at all near it (not
/// even a partial one starting at the flat number), so there is nothing for
/// this function to extend -- adding `flat` would be pure risk (an ordinary
/// English adjective) for zero recovered recall, and the same "smaller
/// correct list" principle that excluded bare `rue`/`avenue` the first time
/// excludes it now. See the follow-up section for the measurement.
const STREET_PREFIXES: &[&str] = &[
    // Spanish
    "calle",
    "avenida",
    "plaza del",
    "paseo de la",
    // Italian
    "via",
    "via dei",
    "via del",
    "via della",
    "via delle",
    "via degli",
    "via di",
    "viale dei",
    "corso",
    "piazza",
    "strada",
    // French
    "rue des",
    "rue du",
    "rue",
    "avenue",
    "boulevard",
];

/// Extends the start of every model `STREET_ADDRESS` span leftward over an
/// immediately preceding [`STREET_PREFIXES`] word or phrase, when one is
/// there. Every other span, and every other entity type, passes through
/// unchanged.
///
/// Rule-side, not a retrain: the model is right about where the street name
/// and number are and merely clips the qualifying word off the front, and a
/// prefix word list is exactly the kind of per-language vocabulary the rule
/// packs already carry for other types (see `crates/rules/packs/universal.toml`,
/// commit `0342e87`'s VAT/passport context terms for the same shape of fix).
/// This lives in `pii-detection-model` rather than `pii-detection-rules` because it
/// corrects a model span rather than proposing a new one, and the model
/// layer is the only place that still has the un-widened span to grow.
pub fn extend_leading_prefix(document: &Document, spans: Vec<WindowSpan>) -> Vec<WindowSpan> {
    spans
        .into_iter()
        .map(|mut window_span| {
            if window_span.entity_type == EntityType::StreetAddress
                && let Some(new_start) = leading_prefix_start(document, window_span.span.start)
            {
                window_span.span.start = new_start;
            }
            window_span
        })
        .collect()
}

/// The code-point offset a [`STREET_PREFIXES`] entry would start at, if the
/// text immediately before `start` -- after trimming trailing whitespace --
/// ends with one whole. `None` when nothing matches.
///
/// Works in `char`s throughout, never in bytes or in a lower-cased copy's
/// byte length, so a Unicode case fold that changes byte length elsewhere in
/// the preceding text (rare, but real for some scripts) can never desync the
/// returned offset from `document`'s own code-point indexing. Comparison
/// itself is ASCII-only (`char::to_ascii_lowercase`): every listed prefix is
/// plain ASCII, so folding anything wider is never needed and never risked.
///
/// Always searches from the true start of the document, not a fixed-size
/// window like `pii_detection_rules::matcher::neighbourhood` uses for every
/// recognizer's every match: that module clips for a performance budget
/// spent on every rule match in every document, but a `STREET_ADDRESS` span
/// is rare (at most a handful per document), so there is no such budget
/// here, and searching the whole prefix sidesteps entirely the clipped-edge
/// hazard that makes a fixed window's own boundary bookkeeping necessary
/// (see `neighbourhood`'s and `Window`'s docs).
///
/// The boundary check -- the character immediately before the matched
/// phrase must be non-alphanumeric, or the phrase must sit at the true
/// start of the document -- is what keeps `via` from also claiming the tail
/// of `previa` or `deviazione`, and `calle` from claiming the tail of
/// `callejón`: none of those words *end* in the prefix with a
/// non-alphanumeric character before it, so the suffix match already fails
/// them without any separate substring scan. Same reasoning
/// `pii_detection_rules::matcher::matches_whole_term` applies to negative-context
/// terms, applied here to a suffix instead of an interior match.
fn leading_prefix_start(document: &Document, start: u32) -> Option<u32> {
    if start == 0 {
        return None;
    }
    let preceding = document.slice(CharSpan {
        start: 0,
        end: start,
    });
    let chars: Vec<char> = preceding.trim_end().chars().collect();
    let n = chars.len();
    if n == 0 {
        return None;
    }

    STREET_PREFIXES
        .iter()
        .filter_map(|prefix| {
            let prefix_len = prefix.chars().count();
            if prefix_len > n {
                return None;
            }
            let match_start = n - prefix_len;
            let tail_matches = chars[match_start..].iter().zip(prefix.chars()).all(
                |(document_char, prefix_char)| document_char.to_ascii_lowercase() == prefix_char,
            );
            if !tail_matches {
                return None;
            }
            // `match_start == 0` means the phrase sits at the true start of
            // `preceding`, which is the true start of the document itself
            // (`preceding` always begins at document position 0) -- always
            // a boundary, never a clipped edge.
            let has_boundary = match_start == 0 || !chars[match_start - 1].is_alphanumeric();
            has_boundary.then_some((match_start, prefix_len))
        })
        // The longest matching phrase wins, so a future addition of a
        // shorter prefix that happens to be another's suffix (none exist
        // today; `STREET_PREFIXES`'s docs list every entry, and no entry is
        // a suffix of another) could never silently shadow the fuller
        // match.
        .max_by_key(|&(_, prefix_len)| prefix_len)
        .map(|(match_start, _)| match_start as u32)
}

/// Extends a model `STREET_ADDRESS` span over the city that sits in the
/// address's tail position -- after the postcode in Dutch, German, French,
/// Spanish and Italian, before it in English -- and nowhere else.
///
/// WHY A CITY IS MASKED HERE AND NOT IN OPEN TEXT. `LOCATION` is retired,
/// deliberately (`training/taxonomy.py` maps every corpus `CITY` label to
/// `None`), because masking every place name in running text destroys the
/// text's utility to buy almost nothing: a city standing on its own is not
/// personal data. A city at the end of a postal address is: `Kerkstraat 12,
/// 1016 GV Amsterdam` identifies a household, and the household is what
/// `STREET_ADDRESS` exists to protect. This function is the whole of the
/// difference between those two readings, and the difference is *position*.
///
/// THE ANCHOR IS THE MODEL'S SPAN, NOT A POSTCODE SHAPE. It grows a span the
/// model already produced, by looking exactly one single space to its right
/// (or, for the English order, to its left). Nothing scans the document for
/// a postcode-shaped token, and deliberately so: only the Dutch postcode
/// (`1016 GV`) is distinctive, while German, French, Spanish and Italian
/// postcodes are bare five-digit runs indistinguishable from an order or
/// invoice number, so a postcode-anchored rule would consult the gazetteer
/// in open text after every five-digit run in every document -- which is the
/// retired `LOCATION` failure wearing a different hat.
///
/// THE TWO ORDERS. Dutch, German, French, Spanish and Italian postal
/// addresses put the city *after* the postcode (`Turfkade 54, 3021 LJ
/// Almelo`), so the trailing rule grows the span rightward. English puts the
/// city *before* it (`27 Bramble Close, Bury St Edmunds FT2 9IO`), so the
/// leading rule grows the span leftward -- and only when the character
/// before the city, skipping spaces, is the comma that closes the street
/// clause. That comma is what distinguishes the English address order from
/// ordinary prose that merely mentions a city just before an address
/// (`onze vestiging in Almelo, Kerkstraat 12`, where the comma sits on the
/// other side of the city and the rule stays out). Both rules are tried for
/// every span, because the span itself carries no language.
///
/// TWO GUARDS, BOTH MEASURED RATHER THAN IMAGINED. The phrase must be
/// written with a capital ([`written_as_a_proper_name`]), and the text the
/// extension swallows must be clear of every other span
/// ([`free_of_other_spans`]). Both exist because an earlier version of this
/// rule without them turned `80 rue des Charmilles` into a *miss*: the model
/// emits `80` and `rue des Charmilles` as two spans, `Rue` (Somme) is a real
/// French town in the gazetteer, so `80` grew rightward over the lower-case
/// word `rue`, overlapped the neighbouring span, and lost the house number
/// to overlap resolution. See `training/reports/city-address-tails.md`.
///
/// WHAT IT CANNOT DO. It never proposes a new span, never changes a span's
/// type, and never touches a span of any other type: a `PERSON` standing
/// next to a city is left exactly as the model left it.
pub fn extend_address_tail_city(document: &Document, spans: Vec<WindowSpan>) -> Vec<WindowSpan> {
    // The spans as they stand before anything grows, so an extension can be
    // checked against its neighbours rather than against a list already
    // half-mutated by this same pass.
    let original: Vec<CharSpan> = spans.iter().map(|window_span| window_span.span).collect();

    spans
        .into_iter()
        .enumerate()
        .map(|(index, mut window_span)| {
            if window_span.entity_type == EntityType::StreetAddress {
                if let Some(new_end) = trailing_city_end(document, window_span.span.end)
                    && free_of_other_spans(&original, index, window_span.span.end, new_end)
                {
                    window_span.span.end = new_end;
                }
                if let Some(new_start) = leading_city_start(document, window_span.span.start)
                    && free_of_other_spans(&original, index, new_start, window_span.span.start)
                {
                    window_span.span.start = new_start;
                }
            }
            window_span
        })
        .collect()
}

/// Whether `[from, to)` -- the text an extension wants to swallow -- is
/// clear of every span except the one at `index`.
///
/// A span that grew into its neighbour would be handed to
/// `pii_detection_resolve::resolve` as an overlap, and overlap resolution keeps one
/// candidate and drops the other. That is how a rule meant to *add* the city
/// can end up *removing* a house number the model had already found
/// correctly, which is a recall loss disguised as a coverage improvement --
/// measured, not imagined: `training/reports/city-address-tails.md` records
/// the French cases where an earlier version of this rule did exactly that.
/// The spans reaching here never overlap (`reconcile` guarantees it), so
/// refusing to create the first overlap keeps that guarantee.
fn free_of_other_spans(spans: &[CharSpan], index: usize, from: u32, to: u32) -> bool {
    spans
        .iter()
        .enumerate()
        .all(|(other, span)| other == index || span.end <= from || to <= span.start)
}

/// Whether `phrase` is written the way a city on an address line is
/// written: with at least one capital.
///
/// This is the guard that keeps a 43,848-name list from eating the street
/// clause's own vocabulary. `Rue` (Somme), `Strada`, `Piazza` and `Hof` are
/// all real towns and all in the gazetteer, and `rue`, `strada` and `piazza`
/// are also the French and Italian words for the street the address is on --
/// written lower-case, every time, in exactly the position this rule looks
/// at. The same capital separates `Best`, `Ede`, `Como` and `Prato` the
/// cities from `best`, `ede`, `como` and `prato` the ordinary words.
///
/// It costs the rule a city written in lower case (`3021 lj almelo`), which
/// is the safe direction to fail: an address line that does not capitalise
/// its city is unusual, and the miss is visible in recall rather than
/// silent in over-masking.
fn written_as_a_proper_name(phrase: &str) -> bool {
    phrase.chars().any(char::is_uppercase)
}

/// The code point one past the city that follows the span ending at `end`,
/// or `None` when what follows is not a city.
///
/// Requires exactly one space between the span and the city -- not "some
/// whitespace": a line break or a double space is not the inside of a
/// postal address line, and accepting one would let the rule reach across a
/// sentence boundary into open text. Reads at most
/// [`cities::longest_phrase_words`] tokens, because no entry is longer, and
/// takes the longest entry that matches, so `Ascoli Piceno` is never
/// truncated to `Ascoli` (both are cities).
fn trailing_city_end(document: &Document, end: u32) -> Option<u32> {
    let length = document.char_len() as u32;
    if end >= length || character_at(document, end) != Some(' ') {
        return None;
    }
    let mut cursor = end + 1;
    let mut phrase = String::new();
    let mut best: Option<u32> = None;

    for _ in 0..cities::longest_phrase_words() {
        if cursor >= length || character_at(document, cursor).is_some_and(char::is_whitespace) {
            break;
        }
        let mut token_end = cursor;
        while token_end < length
            && !character_at(document, token_end).is_some_and(char::is_whitespace)
        {
            token_end += 1;
        }
        if !phrase.is_empty() {
            phrase.push(' ');
        }
        phrase.push_str(document.slice(CharSpan {
            start: cursor,
            end: token_end,
        }));
        if phrase.chars().count() > cities::longest_phrase_chars() {
            break;
        }
        // Sentence punctuation the city is not part of ("... 3021 LJ
        // Almelo." or "..., Almelo,") is trimmed for the lookup, and the
        // span stops before it -- masking a full stop would change text the
        // caller never asked to have changed.
        let trailing_punctuation = phrase
            .chars()
            .rev()
            .take_while(|character| !character.is_alphanumeric())
            .count();
        let kept: String = phrase
            .chars()
            .take(phrase.chars().count() - trailing_punctuation)
            .collect();
        if written_as_a_proper_name(&kept) && cities::contains(&kept) {
            best = Some(token_end - trailing_punctuation as u32);
        }
        // Only a single space continues a multi-word city name, for the
        // same reason a single space is required to reach the first token.
        if token_end >= length || character_at(document, token_end) != Some(' ') {
            break;
        }
        cursor = token_end + 1;
    }
    best
}

/// The code point the city preceding the span starting at `start` begins
/// at, or `None` when what precedes it is not a city in the English address
/// order.
///
/// Two conditions, and the second is what keeps this rule out of open text:
/// exactly one space between the city and the span, and -- skipping spaces
/// leftward from the city's own start -- a comma. In `27 Bramble Close,
/// Bury St Edmunds FT2 9IO` that comma closes the street clause, which is
/// what makes the words between it and the postcode an address line rather
/// than a sentence.
fn leading_city_start(document: &Document, start: u32) -> Option<u32> {
    if start == 0 || character_at(document, start - 1) != Some(' ') {
        return None;
    }
    let mut cursor = start - 1;
    let mut tokens: Vec<(u32, u32)> = Vec::new();
    let mut best: Option<u32> = None;

    for _ in 0..cities::longest_phrase_words() {
        if cursor == 0 || character_at(document, cursor - 1).is_some_and(char::is_whitespace) {
            break;
        }
        let token_end = cursor;
        let mut token_start = cursor;
        while token_start > 0
            && !character_at(document, token_start - 1).is_some_and(char::is_whitespace)
        {
            token_start -= 1;
        }
        tokens.push((token_start, token_end));
        let phrase = tokens
            .iter()
            .rev()
            .map(|&(from, to)| {
                document.slice(CharSpan {
                    start: from,
                    end: to,
                })
            })
            .collect::<Vec<&str>>()
            .join(" ");
        if phrase.chars().count() > cities::longest_phrase_chars() {
            break;
        }
        if written_as_a_proper_name(&phrase)
            && cities::contains(&phrase)
            && closes_a_street_clause(document, token_start)
        {
            best = Some(token_start);
        }
        if token_start == 0 || character_at(document, token_start - 1) != Some(' ') {
            break;
        }
        cursor = token_start - 1;
    }
    best
}

/// Whether the text immediately before `start`, skipping spaces, ends in
/// the comma that closes a street clause -- the English address order's own
/// punctuation, and the only signal that separates `27 Bramble Close, Bury
/// St Edmunds FT2 9IO` from a sentence that happens to name a city just
/// before an address.
fn closes_a_street_clause(document: &Document, start: u32) -> bool {
    let mut index = start;
    while index > 0 && character_at(document, index - 1) == Some(' ') {
        index -= 1;
    }
    index > 0 && character_at(document, index - 1) == Some(',')
}

/// The code point at `index`, or `None` past the end of the document.
fn character_at(document: &Document, index: u32) -> Option<char> {
    if index as usize >= document.char_len() {
        return None;
    }
    document
        .slice(CharSpan {
            start: index,
            end: index + 1,
        })
        .chars()
        .next()
}

/// Turns reconciled, trimmed spans into the [`Candidate`] shape
/// `pii-detection-resolve` consumes from every detector alike. A model span
/// carries no rule evidence: it matched no pattern and validated no
/// checksum, and claiming either would let a model span out-rank a
/// checksum-validated rule span in `pii_detection_resolve::strength_order`.
pub fn to_candidates(spans: Vec<WindowSpan>) -> Vec<Candidate> {
    spans
        .into_iter()
        .map(|window_span| Candidate {
            span: window_span.span,
            entity_type: window_span.entity_type,
            country: None,
            source: Source::Model,
            score: window_span.score,
            evidence: Evidence::default(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::labels_for_tests;

    /// The 12 Temperate types in `labels.json`'s (and `align.py`'s `PLATFORM_TYPES`
    /// and `EntityType`'s own declaration order's) order -- alphabetical by
    /// wire name. `shipped_labels` and `id` both walk this list to build
    /// and address a `Labels` that mirrors the real shipped
    /// `models/pii-detection-model-full-merged-int8/labels.json` without reading
    /// it from disk, matching `artifact.rs`'s own test convention of
    /// hardcoding that content (its `labels_json()`) rather than depending
    /// on the shipped artifact for a unit test.
    const TYPES_IN_LABEL_ORDER: [(&str, Option<EntityType>); 12] = [
        ("CREDIT_CARD", None),
        ("DRIVING_LICENCE", None),
        ("EMAIL_ADDRESS", None),
        ("IBAN_CODE", None),
        ("IP_ADDRESS", None),
        ("NATIONAL_ID", None),
        ("PASSPORT_NUMBER", None),
        ("PERSON", Some(EntityType::Person)),
        ("PHONE_NUMBER", None),
        ("STREET_ADDRESS", Some(EntityType::StreetAddress)),
        ("TAX_ID", None),
        ("VAT_NUMBER", None),
    ];

    /// A `Labels` with the real shipped artifact's 25 entries: `O`, then
    /// `B-`/`I-` per type above, only `PERSON` and `STREET_ADDRESS` mapped.
    fn shipped_labels() -> Labels {
        let mut entries = vec![None];
        for (_, mapped) in TYPES_IN_LABEL_ORDER {
            entries.push(mapped);
            entries.push(mapped);
        }
        labels_for_tests(entries)
    }

    /// The label id for a BIO tag like `"B-NATIONAL_ID"` or `"O"`, under
    /// `TYPES_IN_LABEL_ORDER`'s ordering -- the id `shipped_labels()`'s
    /// `Labels` would assign it.
    fn id(label: &str) -> usize {
        if label == "O" {
            return 0;
        }
        let (prefix, wire) = label.split_once('-').expect("a BIO label has a prefix");
        let position = TYPES_IN_LABEL_ORDER
            .iter()
            .position(|(name, _)| *name == wire)
            .expect("a known platform wire name");
        1 + position * 2 + if prefix == "B" { 0 } else { 1 }
    }

    /// A window whose real tokens have exactly `offsets`, with no special
    /// tokens -- for tests that only care about decode logic, not about
    /// windowing itself (already proven in `window.rs`/`parity.rs`).
    fn window_with_offsets(offsets: &[(u32, u32)]) -> Window {
        let spans: Vec<Option<CharSpan>> = offsets
            .iter()
            .map(|&(start, end)| Some(CharSpan { start, end }))
            .collect();
        let start = offsets.iter().map(|&(start, _)| start).min().unwrap_or(0);
        let end = offsets.iter().map(|&(_, end)| end).max().unwrap_or(0);
        Window {
            input_ids: vec![0; offsets.len()],
            attention_mask: vec![1; offsets.len()],
            token_type_ids: vec![0; offsets.len()],
            spans,
            coverage: CharSpan { start, end },
        }
    }

    /// A `WindowSpan` for window 0, for tests that don't care which window
    /// produced it.
    fn span(start: u32, end: u32, entity_type: EntityType, score: f32, margin: u32) -> WindowSpan {
        span_from_window(0, start, end, entity_type, score, margin)
    }

    fn span_from_window(
        window: usize,
        start: u32,
        end: u32,
        entity_type: EntityType,
        score: f32,
        margin: u32,
    ) -> WindowSpan {
        WindowSpan {
            span: CharSpan { start, end },
            entity_type,
            score,
            margin,
            window,
        }
    }

    #[test]
    fn a_label_mapped_to_null_produces_no_span() {
        // The ten structured types map to null in labels.json. A decoder
        // that emitted them anyway would put uncalibrated NATIONAL_ID
        // candidates in front of the 99% precision gate.
        let window = window_with_offsets(&[(0, 3), (4, 12)]);
        let spans = decode_window(
            &window,
            0,
            &[id("B-NATIONAL_ID"), id("I-NATIONAL_ID")],
            &[0.99, 0.99],
            &shipped_labels(),
        );
        assert!(spans.is_empty());
    }

    #[test]
    fn the_window_that_saw_the_span_furthest_from_its_edge_wins() {
        // Design section 8. Window A ends four code points after the span,
        // so the span sits at its edge; window B has the span in its
        // middle.
        let at_edge = span(20, 30, EntityType::Person, 0.99, 0);
        let in_middle = span(20, 30, EntityType::Person, 0.60, 118);
        let kept = reconcile(vec![at_edge, in_middle]);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].margin, 118);
        // And explicitly NOT decided by score: the edge span scored higher.
        assert!((kept[0].score - 0.60).abs() < 1e-6);
    }

    #[test]
    fn a_half_seen_span_is_discarded_rather_than_joined_to_its_other_half() {
        // The difference from eval_fixture.py's _merge_touching_typed, made
        // explicit. Window A saw "Jan de" (cut at its edge); window B saw
        // "Jan de Vries" whole. The answer is "Jan de Vries", one span, not
        // a stitched-together pair.
        let cut = span(10, 16, EntityType::Person, 0.9, 0);
        let whole = span(10, 22, EntityType::Person, 0.9, 90);
        let kept = reconcile(vec![cut, whole]);
        assert_eq!(kept.len(), 1);
        assert_eq!((kept[0].span.start, kept[0].span.end), (10, 22));
    }

    #[test]
    fn two_genuinely_different_spans_from_two_windows_are_both_kept() {
        // The rule resolves disagreement, not multiplicity. A reconciler
        // that deduplicated by type would silently delete the second
        // person in a long document, and no parity test would notice.
        let first = span(10, 22, EntityType::Person, 0.9, 90);
        let second = span(400, 411, EntityType::Person, 0.9, 90);
        assert_eq!(reconcile(vec![first, second]).len(), 2);
    }

    #[test]
    fn ties_on_margin_are_broken_deterministically_by_window_order() {
        // Same margin (40), but window 1's span is longer than window 0's:
        // a reconciler that fell back to "keep the longest" on a tie would
        // keep window 1's, not window 0's, so this both proves the window
        // tie-break and would catch that mutation, which a same-length pair
        // would not.
        let a = span_from_window(0, 10, 20, EntityType::Person, 0.5, 40);
        let b = span_from_window(1, 5, 30, EntityType::Person, 0.9, 40);
        let kept = reconcile(vec![a, b]);
        assert_eq!(kept.len(), 1);
        assert!(
            (kept[0].score - 0.5).abs() < 1e-6,
            "the earlier window wins a tie"
        );
    }

    #[test]
    fn a_leading_space_the_tokenizer_included_is_trimmed_off_the_span() {
        // XLM-R's Metaspace pre-tokenizer attaches the preceding space to a
        // token. A span that starts one code point early makes the
        // pseudonymiser eat a space, changing text the caller did not ask
        // to have changed. Whether this actually happens is measured in
        // step 5; this test pins the behaviour either way.
        let document = Document::new("Hallo Jan de Vries".to_string());
        let with_space = span(5, 18, EntityType::Person, 0.9, 40);
        let trimmed = trim_whitespace(&document, vec![with_space]);
        assert_eq!((trimmed[0].span.start, trimmed[0].span.end), (6, 18));
        assert_eq!(document.slice(trimmed[0].span), "Jan de Vries");
    }

    #[test]
    fn a_span_that_is_only_whitespace_is_dropped_rather_than_emptied() {
        let document = Document::new("a   b".to_string());
        let blank = span(1, 4, EntityType::Person, 0.9, 40);
        assert!(trim_whitespace(&document, vec![blank]).is_empty());
    }

    /// The code-point offset of `needle`'s first byte within `text`.
    /// `str::find` returns a *byte* offset; every test below plants an
    /// accented, multibyte word somewhere before the target substring
    /// specifically so a test that forgot this conversion would fail rather
    /// than pass by accident on ASCII-only fixtures.
    fn char_offset(text: &str, needle: &str) -> u32 {
        let byte_offset = text.find(needle).expect("needle must be in text");
        text[..byte_offset].chars().count() as u32
    }

    /// Runs `extend_leading_prefix` over one `STREET_ADDRESS` span covering
    /// `needle` within `text` and returns the text it now covers.
    fn extended_text(text: &str, needle: &str) -> String {
        let document = Document::new(text.to_string());
        let start = char_offset(text, needle);
        let end = start + needle.chars().count() as u32;
        let candidate = span(start, end, EntityType::StreetAddress, 0.9, 40);
        let extended = extend_leading_prefix(&document, vec![candidate]);
        assert_eq!(extended.len(), 1, "extension must not drop or add spans");
        document.slice(extended[0].span).to_string()
    }

    #[test]
    fn spanish_calle_is_pulled_into_the_span() {
        // The task brief's own example: the model finds "Mercaderes 108"
        // and drops "calle". "Envía" plants an accented, multibyte word
        // before the target so a byte/code-point offset mix-up would fail
        // this test rather than pass it by accident.
        let text = "Envía el paquete a calle Mercaderes 108 mañana";
        assert_eq!(
            extended_text(text, "Mercaderes 108"),
            "calle Mercaderes 108"
        );
    }

    #[test]
    fn spanish_plaza_del_is_pulled_in_as_one_two_word_phrase() {
        let text = "Vive en la plaza del Retiro Viejo 158 desde julio";
        assert_eq!(
            extended_text(text, "Retiro Viejo 158"),
            "plaza del Retiro Viejo 158"
        );
    }

    #[test]
    fn spanish_paseo_de_la_is_pulled_in() {
        let text = "La cita es en paseo de la Alameda 123 a mediodía";
        assert_eq!(
            extended_text(text, "Alameda 123"),
            "paseo de la Alameda 123"
        );
    }

    #[test]
    fn spanish_avenida_is_pulled_in() {
        let text = "Nos mudamos a avenida de los Almendros 109 este mes";
        assert_eq!(
            extended_text(text, "de los Almendros 109"),
            "avenida de los Almendros 109"
        );
    }

    #[test]
    fn a_calle_look_alike_directly_before_the_span_is_not_pulled_in() {
        // `callejón` sits directly against the span (only whitespace
        // between), the exact adjacency `extend_leading_prefix` inspects --
        // and still is not pulled in, because "callejón" itself does not
        // *end* in "calle" (it ends "-ejón"), so the suffix check fails
        // without needing a separate substring guard the way
        // `matches_whole_term` does.
        let text = "El paquete llegó a un callejón Mercaderes 108 ayer";
        assert_eq!(extended_text(text, "Mercaderes 108"), "Mercaderes 108");
    }

    #[test]
    fn italian_via_is_pulled_into_the_span() {
        let text = "Abito in via Palestro 156 da tre anni";
        assert_eq!(extended_text(text, "Palestro 156"), "via Palestro 156");
    }

    #[test]
    fn italian_viale_dei_is_pulled_in_as_one_two_word_phrase() {
        // "è" plants a multibyte character before the target.
        let text = "L'ufficio è in viale dei Tigli 166 al primo piano";
        assert_eq!(extended_text(text, "Tigli 166"), "viale dei Tigli 166");
    }

    #[test]
    fn italian_corso_piazza_and_strada_are_pulled_in() {
        for (prefix, text, name) in [
            (
                "corso",
                "Ci vediamo in corso Vittorio Emanuele 65 stasera",
                "Vittorio Emanuele 65",
            ),
            (
                "piazza",
                // "è" plants a multibyte character before the target.
                "Il negozio è in piazza Navona 12 in centro",
                "Navona 12",
            ),
            (
                "strada",
                "La ditta ha sede in strada Provinciale 10 fuori città",
                "Provinciale 10",
            ),
        ] {
            assert_eq!(
                extended_text(text, name),
                format!("{prefix} {name}"),
                "prefix {prefix} was not pulled in"
            );
        }
    }

    #[test]
    fn italian_via_look_alikes_are_not_pulled_in() {
        // `previa` and `deviazione` both end in letters other than "via"
        // once the whole word is considered, so the suffix check rejects
        // them the same way `matches_whole_term` rejects `order` inside
        // `border`.
        for text in [
            "Serve la previa Palestro 156",
            "Dopo la deviazione Palestro 156",
        ] {
            assert_eq!(extended_text(text, "Palestro 156"), "Palestro 156");
        }
    }

    #[test]
    fn french_rue_des_and_rue_du_are_pulled_in() {
        let text = "Nous habitons au 80 rue des Charmilles au fond";
        assert_eq!(extended_text(text, "Charmilles"), "rue des Charmilles");

        let text = "Le magasin est au 56 rue du Pressoir ce matin";
        assert_eq!(extended_text(text, "Pressoir"), "rue du Pressoir");
    }

    #[test]
    fn french_boulevard_is_pulled_in() {
        let text = "Le bureau est boulevard des Tanneurs 153 au centre";
        assert_eq!(
            extended_text(text, "des Tanneurs 153"),
            "boulevard des Tanneurs 153"
        );
    }

    #[test]
    fn french_bare_rue_and_avenue_are_pulled_in() {
        // The two real recovered misses (`training/reports/street-prefixes.md`'s
        // follow-up section), each reproduced with the benchmark's own
        // sentence. In both, only the bare word -- not the two-word `rue
        // du`/`rue des` phrases already listed -- sits immediately before
        // the model's own span: the model already includes `du Pressoir`
        // and `du Vieux Moulin`, so `rue du` and `rue des` can never match
        // here (the preceding text ends in `rue`, not `rue du`), and only
        // adding the bare word closes the gap.
        let text = "L'adresse de correspondance est 56 rue du Pressoir, 71456 Bourges.";
        assert_eq!(
            extended_text(text, "du Pressoir"),
            "rue du Pressoir",
            "fr-wb-charpente-dn"
        );

        let text = "L'adresse de correspondance est 20 avenue du Vieux Moulin, 48243 Albi.";
        assert_eq!(
            extended_text(text, "du Vieux Moulin"),
            "avenue du Vieux Moulin",
            "fr-wb-chauffage-dn"
        );
    }

    #[test]
    fn bare_rue_look_alikes_are_not_pulled_in() {
        // Four ways a real word can carry "rue" without being the standalone
        // prefix. `rueda`, `cruel` and `rueful` all fail the suffix check
        // outright, the same way `callejón` does for `calle`: none of them
        // *end* in "rue". `morue` (French for "cod") is the sharper case --
        // it does end in "rue" as a bare suffix match, so it is the boundary
        // check's job specifically, the same role `previa` plays for `via`:
        // the character before the match ("o") is alphanumeric, so the
        // whole-word requirement rejects it.
        for text in [
            "La rueda llegó Mercaderes 108 ayer",
            "Fue cruel Mercaderes 108 aquella tarde",
            "Se sintió rueful Mercaderes 108 después",
            "On sert de la morue Mercaderes 108 ce soir",
        ] {
            assert_eq!(
                extended_text(text, "Mercaderes 108"),
                "Mercaderes 108",
                "{text}"
            );
        }
    }

    #[test]
    fn a_span_at_the_very_start_of_the_document_is_left_alone() {
        // No preceding text exists at all; `leading_prefix_start` must
        // return `None` rather than underflow computing it.
        let text = "Mercaderes 108 es la dirección";
        assert_eq!(extended_text(text, "Mercaderes 108"), "Mercaderes 108");
    }

    #[test]
    fn a_person_span_is_never_extended_even_with_a_prefix_right_before_it() {
        // Only STREET_ADDRESS spans grow. A PERSON span sitting right after
        // "calle" must be left exactly where the model put it.
        let text = "Contactar con calle Mercaderes hoy";
        let document = Document::new(text.to_string());
        let start = char_offset(text, "Mercaderes");
        let end = start + "Mercaderes".chars().count() as u32;
        let person = span(start, end, EntityType::Person, 0.9, 40);
        let extended = extend_leading_prefix(&document, vec![person]);
        assert_eq!(extended[0].span, CharSpan { start, end });
    }

    /// Runs `extend_address_tail_city` over one span of `entity_type`
    /// covering `needle` within `text` and returns the text it now covers.
    fn city_extended(text: &str, needle: &str, entity_type: EntityType) -> String {
        let document = Document::new(text.to_string());
        let start = char_offset(text, needle);
        let end = start + needle.chars().count() as u32;
        let candidate = span(start, end, entity_type, 0.9, 40);
        let extended = extend_address_tail_city(&document, vec![candidate]);
        assert_eq!(extended.len(), 1, "extension must not drop or add spans");
        document.slice(extended[0].span).to_string()
    }

    /// The five languages that put the city after the postcode.
    fn address_extended(text: &str, needle: &str) -> String {
        city_extended(text, needle, EntityType::StreetAddress)
    }

    #[test]
    fn the_dutch_city_after_the_postcode_is_pulled_into_the_span() {
        let text = "Stuur het naar Turfkade 54, 3021 LJ Almelo vóór vrijdag";
        assert_eq!(address_extended(text, "3021 LJ"), "3021 LJ Almelo");
    }

    #[test]
    fn the_german_city_after_the_postcode_is_pulled_into_the_span() {
        // "Kupferstraße" plants a multibyte character before the target, so
        // a byte/code-point offset mix-up fails this test rather than
        // passing by accident on ASCII.
        let text = "Die Lieferung geht an Kupferstraße 72, 46001 Ansbach.";
        assert_eq!(address_extended(text, "46001"), "46001 Ansbach");
    }

    #[test]
    fn a_two_word_italian_city_is_pulled_in_whole() {
        let text = "Consegna in via Palestro 156, 32891 Ascoli Piceno entro maggio";
        assert_eq!(address_extended(text, "32891"), "32891 Ascoli Piceno");
    }

    #[test]
    fn the_longest_matching_city_wins_over_its_own_first_word() {
        // "Bergen" and "Bergen op Zoom" are both real Dutch cities and both
        // in the gazetteer. Stopping at the first match would leave "op
        // Zoom" in the clear beside a masked "Bergen".
        let text = "Bezorgen op Vlierlaan 176, 4611 AA Bergen op Zoom, graag";
        assert_eq!(address_extended(text, "4611 AA"), "4611 AA Bergen op Zoom");
    }

    #[test]
    fn an_apostrophe_city_is_matched_as_written() {
        let text = "Factuuradres: Kolkweg 12, 2691 AA 's-Gravenzande";
        assert_eq!(address_extended(text, "2691 AA"), "2691 AA 's-Gravenzande");
    }

    #[test]
    fn the_english_city_before_the_postcode_is_pulled_into_the_span() {
        // English is the one order where the city precedes the postcode,
        // and "Bury" alone is a city too -- so this also pins that the
        // longest phrase wins leftward, not just rightward.
        let text = "Please write to 27 Bramble Close, Bury St Edmunds FT2 9IO before June";
        assert_eq!(address_extended(text, "FT2 9IO"), "Bury St Edmunds FT2 9IO");
    }

    #[test]
    fn an_english_city_without_the_street_clause_comma_is_left_alone() {
        // Ordinary prose naming a city just before an address-shaped span.
        // The comma that closes a street clause is the only thing that
        // makes the English order an address rather than a sentence, so
        // without it the rule stays out -- this is the guard that keeps a
        // standalone city mention out of the mask.
        let text = "Our branch in Kendal LA9 4ZP is open on Saturdays";
        assert_eq!(address_extended(text, "LA9 4ZP"), "LA9 4ZP");
    }

    #[test]
    fn a_city_mentioned_before_the_street_clause_is_left_alone() {
        // "onze vestiging in Almelo, Kolkweg 12" -- the comma sits on the
        // far side of the city, so the leading rule does not fire and the
        // standalone mention survives.
        let text = "Onze vestiging in Almelo, Kolkweg 12 is verhuisd";
        assert_eq!(address_extended(text, "Kolkweg 12"), "Kolkweg 12");
    }

    #[test]
    fn sentence_punctuation_after_the_city_is_not_swallowed() {
        // Masking the full stop would change text the caller never asked to
        // have changed.
        let text = "Adres: Sluisstraat 158, 1125 RZ Hoorn. Bel gerust.";
        assert_eq!(address_extended(text, "1125 RZ"), "1125 RZ Hoorn");
    }

    #[test]
    fn a_comma_after_the_city_is_not_swallowed() {
        let text = "Adres: Sluisstraat 158, 1125 RZ Hoorn, Nederland";
        assert_eq!(address_extended(text, "1125 RZ"), "1125 RZ Hoorn");
    }

    #[test]
    fn an_ordinary_word_after_the_address_is_never_pulled_in() {
        let text = "Zie Turfkade 54, 3021 LJ bestelling 4471 voor details";
        assert_eq!(address_extended(text, "3021 LJ"), "3021 LJ");
    }

    #[test]
    fn only_a_single_space_separates_an_address_from_its_city() {
        // A line break or a double space is not the inside of an address
        // line; reaching across one would let the rule step into open text.
        for text in [
            "Adres: Turfkade 54, 3021 LJ\nAlmelo staat in het register",
            "Adres: Turfkade 54, 3021 LJ  Almelo staat in het register",
        ] {
            assert_eq!(address_extended(text, "3021 LJ"), "3021 LJ", "{text}");
        }
    }

    #[test]
    fn a_span_at_the_very_end_of_the_document_is_left_alone() {
        let text = "Adres: Turfkade 54, 3021 LJ";
        assert_eq!(address_extended(text, "3021 LJ"), "3021 LJ");
    }

    #[test]
    fn a_span_at_the_very_start_of_the_document_is_not_underflowed() {
        let text = "3021 LJ Almelo";
        assert_eq!(address_extended(text, "3021 LJ"), "3021 LJ Almelo");
    }

    #[test]
    fn no_other_entity_type_ever_grows_over_a_city() {
        // A person standing right before a city name is exactly the case
        // where a type-blind rule would mask a city that is not in address
        // position at all.
        let text = "Sprekers: Sanne Doornbos Almelo en anderen";
        assert_eq!(
            city_extended(text, "Sanne Doornbos", EntityType::Person),
            "Sanne Doornbos"
        );
    }

    #[test]
    fn a_city_standing_alone_elsewhere_in_the_document_is_untouched() {
        // The gazetteer is consulted at one position only. "Amsterdam" here
        // is nowhere near the address span, and the extension has no way to
        // reach it -- there is no document scan to reach it with.
        let text = "Amsterdam groeit. Stuur post naar Turfkade 54, 3021 LJ Almelo.";
        let document = Document::new(text.to_string());
        let start = char_offset(text, "3021 LJ");
        let end = start + "3021 LJ".chars().count() as u32;
        let extended = extend_address_tail_city(
            &document,
            vec![span(start, end, EntityType::StreetAddress, 0.9, 40)],
        );
        assert_eq!(document.slice(extended[0].span), "3021 LJ Almelo");
        assert!(extended[0].span.start > char_offset(text, "Amsterdam"));
    }

    #[test]
    fn a_lower_case_street_word_that_is_also_a_town_is_not_swallowed() {
        // `Rue` is a real commune in the Somme and is in the gazetteer;
        // `rue` is also the French word for the street this address is on.
        // The model emits the house number as its own span, so without the
        // capital guard the number's span grows over the street word.
        let text = "L'adresse de correspondance est 56 rue du Pressoir, 71456 Bourges.";
        assert_eq!(address_extended(text, "56"), "56");
    }

    #[test]
    fn bare_rue_prefix_and_the_town_gazetteer_extension_compose_correctly() {
        // `run_detection` (`lib.rs`) calls `extend_leading_prefix` before
        // `extend_address_tail_city`, always on the same spans, so this
        // pins the two functions run in that order rather than each in
        // isolation -- the composition the "Rue" hazard is actually about.
        // `Rue` is a real commune in the Somme and is in the gazetteer;
        // `rue` is now also a `STREET_PREFIXES` entry. Two model spans, `56`
        // and `du Pressoir` (the fr-wb-charpente-dn text, the model's own
        // shape): `extend_leading_prefix` first turns `du Pressoir` into
        // `rue du Pressoir` using the bare-word list, which has nothing to
        // do with the gazetteer. `extend_address_tail_city` then runs on
        // the result and must neither swallow `71456 Bourges` into the
        // street span (blocked by the comma, same guard as before) nor let
        // `56` grow rightward over the now-capitalised-nowhere lower-case
        // `rue` (blocked by `written_as_a_proper_name`, same guard as
        // `a_lower_case_street_word_that_is_also_a_town_is_not_swallowed`).
        let text = "L'adresse de correspondance est 56 rue du Pressoir, 71456 Bourges.";
        let document = Document::new(text.to_string());
        let number_start = char_offset(text, "56");
        let street_start = char_offset(text, "du Pressoir");
        let spans = vec![
            span(
                number_start,
                number_start + 2,
                EntityType::StreetAddress,
                0.9,
                40,
            ),
            span(
                street_start,
                street_start + "du Pressoir".chars().count() as u32,
                EntityType::StreetAddress,
                0.9,
                40,
            ),
        ];
        let prefixed = extend_leading_prefix(&document, spans);
        let with_city = extend_address_tail_city(&document, prefixed);
        assert_eq!(with_city.len(), 2, "extension must not drop or add spans");
        assert_eq!(document.slice(with_city[0].span), "56");
        assert_eq!(document.slice(with_city[1].span), "rue du Pressoir");
    }

    #[test]
    fn an_extension_that_would_collide_with_another_span_is_refused() {
        // Two model spans, `80` and `rue des Charmilles`, with a town name
        // (`Rue`) between them as far as the gazetteer is concerned.
        // Growing the first into the second hands `pii_detection_resolve` an
        // overlap, and overlap resolution drops one of them -- turning a
        // rule that adds the city into a rule that deletes the house
        // number. `reconcile` hands this function non-overlapping spans;
        // it must not create the first overlap itself.
        let text = "DB (adresse 80 rue des Charmilles, 84586 Quimper).";
        let document = Document::new(text.to_string());
        let number_start = char_offset(text, "80");
        let street_start = char_offset(text, "rue des Charmilles");
        let spans = vec![
            span(
                number_start,
                number_start + 2,
                EntityType::StreetAddress,
                0.9,
                40,
            ),
            span(
                street_start,
                street_start + "rue des Charmilles".chars().count() as u32,
                EntityType::StreetAddress,
                0.9,
                40,
            ),
        ];
        let extended = extend_address_tail_city(&document, spans);
        assert_eq!(document.slice(extended[0].span), "80");
        assert_eq!(document.slice(extended[1].span), "rue des Charmilles");
    }

    #[test]
    fn a_city_written_in_lower_case_is_left_alone() {
        // The safe direction to fail: an address line that does not
        // capitalise its city is unusual, and the miss shows up in recall
        // rather than silently in over-masking.
        let text = "Adres: turfkade 54, 3021 lj almelo";
        assert_eq!(address_extended(text, "3021 lj"), "3021 lj");
    }

    #[test]
    fn the_gazetteer_is_reachable_from_exactly_one_place_in_this_crate() {
        // The safety argument for shipping a 43,848-name city list is
        // entirely positional: it is consulted only from a position
        // anchored on a model STREET_ADDRESS span. That argument is only
        // true while `cities` has one caller, so this test reads the
        // crate's own sources and fails the moment a second one appears --
        // including in a helper that "just needs to check a word".
        let source_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut callers: Vec<String> = Vec::new();
        for entry in std::fs::read_dir(&source_dir).expect("the crate has a src directory") {
            let path = entry.expect("a readable directory entry").path();
            if path.extension().is_none_or(|extension| extension != "rs") {
                continue;
            }
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default()
                .to_string();
            if name == "cities.rs" {
                continue;
            }
            let source = std::fs::read_to_string(&path).expect("a readable source file");
            if source.contains("cities::") {
                callers.push(name);
            }
        }
        callers.sort();
        assert_eq!(
            callers,
            vec!["decode.rs".to_string()],
            "the city gazetteer must have exactly one caller: extend_address_tail_city"
        );
    }

    #[test]
    fn a_candidate_from_the_model_carries_no_rule_evidence() {
        // Evidence::format means "a recognizer's pattern matched". The
        // model matched no pattern and validated no checksum, and claiming
        // either would let a model span out-rank a checksum-validated rule
        // span in pii_detection_resolve::strength_order.
        let candidates = to_candidates(vec![span(0, 5, EntityType::Person, 0.9, 40)]);
        assert_eq!(candidates[0].source, Source::Model);
        assert_eq!(candidates[0].evidence, Evidence::default());
        assert_eq!(candidates[0].evidence.checksum, None);
        assert_eq!(candidates[0].country, None);
    }
}
