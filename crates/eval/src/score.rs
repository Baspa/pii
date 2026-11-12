//! Span-exact scoring of a [`Case`] against a set of detected [`Entity`]
//! spans, and the roll-up of many cases into recall and precision: overall,
//! per entity type, and per language.
//!
//! THE RULE THIS MODULE IMPLEMENTS, AND WHY IT IS NOT THE OBVIOUS ONE. An
//! earlier scorer required a `must_mask` occurrence to be covered by a
//! *single* detected span. The detector's detectors do not always emit one span per
//! value, a given name and a surname come back as two adjacent spans that
//! together mask the whole name and leak nothing, and the single-span rule
//! called that a miss. It reported recall of 0.6711 where the honest figure
//! was 0.9145. `training/fixture_scoring.py` carries the full account and the
//! corrected rule this module ports the *semantics* of (not the code: this
//! scores a different fixture, in Rust, and the two implementations must
//! never be merged into one, so that a bug in one cannot silently become a
//! bug in both):
//!
//! - a `must_mask` value counts as found only when **every** occurrence of it
//!   in the case's `text` has **every non-whitespace character** covered by
//!   the **union** of detected spans. A gap that falls entirely on
//!   whitespace between two spans is not a leak, because no information
//!   survives in a space. A value with two occurrences must be found at
//!   both: masking one and leaking the other is still a leak.
//! - a `must_clear` value counts as violated when **any** detected span
//!   overlaps **any** occurrence of it at all.
//!
//! Offsets are code points throughout, matching `Entity::start`/`Entity::end`
//! and the rest of this codebase (see `pii_detection_engine::text` and
//! `crates/eval/src/main.rs`'s header). `must_mask`/`must_clear`
//! values are matched against `Case::text` in bytes first, because that is
//! what Rust string search does, and converted to code points exactly once
//! via [`pii_detection_engine::text::Document`]: the same conversion the rest of
//! the workspace uses, not a second implementation of it.
//!
//! PER-TYPE RECALL COUNTS DECLARED TYPES, NOT DETECTED ONES, AND SAYS HOW
//! MANY IT COULD NOT COUNT. An earlier version of this module attributed a
//! `must_mask` value to the [`pii_detection_engine::entity::EntityType`] of whatever
//! detected span happened to touch it, because `Case::must_mask` was a plain
//! list of substrings and nothing in a case said what a value was supposed to
//! be. That has a defect with the wrong sign: a value **no** span touches at
//! all, the worst kind of miss, cannot be attributed that way, so it fell
//! out of its own per-type denominator, and dropping a miss from a
//! denominator *raises* the recall it is a miss of. Every gate this benchmark
//! enforces is per type, so the detector was flattered exactly where the
//! fixture is supposed to be able to fail it.
//!
//! [`crate::MustMask`] now lets a case declare the type of a value it names,
//! and [`Summary::per_type`]'s **recall** side counts declared types only: a
//! declared value lands in its declared bucket whether it was found, half
//! found, or never touched. Values with no declared type are counted in
//! [`Summary::overall`] and [`Summary::per_language`] as they always were,
//! and are reported by count in [`Summary::untyped_mask_values`], so a
//! per-type recall figure can be read for what it is (the part of the fixture
//! that says what it wants) and can never be mistaken for a complete one
//! while untyped values remain. The 128 legacy cases carry no types, so today
//! that count is large; it shrinks as cases are typed.
//!
//! The **precision** side of [`Summary::per_type`] is still keyed by detected
//! type, and has to be: a `must_clear` value has no intended type (it is not
//! personal data at all), and a spurious span's only type is the one the
//! detector gave it. `MaskVerdict::types` therefore keeps carrying the
//! detected types that touched a value, unused by the roll-up but available
//! to a caller diagnosing a type confusion.
//!
//! [`Summary::overall`] and [`Summary::per_language`] are not subject to any
//! of this: every occurrence counts there regardless of type. That makes
//! them the authoritative figures *for the set of cases they were built
//! from*, which is a real caveat, not a formality. Against a detector that
//! was never asked about some of the types the fixture declares, "every
//! occurrence counts" means every occurrence of a type nobody asked about
//! counts as a miss, and the honest headline figure for that run is over
//! the subset that was actually requested. Restricting the set is
//! [`crate::gates::drop_unrequested`]'s job, and it does it by filtering
//! per-value verdicts and re-pooling through `Summary::from`, never by
//! adjusting a number this module computed.
//!
//! MICRO AND MACRO, BOTH, FOR LANGUAGE. [`Summary::per_language`] holds one
//! [`Metrics`] per language (each language's own pooled, "micro", recall
//! and precision). [`Summary::overall`] pools every occurrence across every
//! language, which is also micro, and is dominated by whichever language has
//! the most cases. [`Summary::macro_by_language`] instead averages each
//! language's recall and precision with every language weighted equally, so
//! a language that fails badly cannot hide behind a large language that is
//! doing fine.
//!
//! `NONE` MEANS "NOTHING TO MEASURE," NOT "ZERO." [`Metrics::recall`] and
//! [`Metrics::precision`] return `Option<f64>` rather than defaulting an
//! empty denominator to `0.0` or `1.0`. A downstream gate must be
//! able to tell "found nothing" from "there was nothing to find": collapsing
//! that distinction is the same shape of mistake this module's corrected rule
//! exists to prevent, just at the aggregation stage instead of the per-value
//! stage.

use std::collections::{BTreeSet, HashMap};

use pii_detection_api::wire::Entity;
use pii_detection_engine::entity::EntityType;
use pii_detection_engine::text::Document;

use crate::Case;

/// A code-point span, half-open: `[start, end)`.
type Span = (u32, u32);

/// A detected span together with the type it was detected as.
type TypedSpan = (u32, u32, EntityType);

/// One case's outcome, as the literal values from `Case::must_mask` and
/// `Case::must_clear` that fell on each side of the scoring rule.
///
/// `masked`/`missed`/`over_masked` carry the values themselves (not indices,
/// unlike [`crate::Incoherence`]'s deliberately value-free design) because
/// the brief this type implements specifies exactly this shape, and because,
/// unlike production request data, a benchmark case's `must_mask` and
/// `must_clear` lists are fixture content that already lives in the repo in
/// plain text (`Case::text` itself). Nothing in this module logs, traces, or
/// otherwise emits an `Outcome` anywhere: a caller that does must keep the
/// same discipline the rest of this service holds for anything reachable
/// from a log line.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Outcome {
    /// `must_mask` values fully covered by the union of detected spans, at
    /// every occurrence.
    pub masked: Vec<String>,
    /// `must_mask` values with at least one uncovered non-whitespace
    /// character, in at least one occurrence: a leak.
    pub missed: Vec<String>,
    /// `must_clear` values touched by at least one detected span, anywhere:
    /// over-masking.
    pub over_masked: Vec<String>,
}

/// Scores one case against the spans a detector run produced for it.
///
/// `entities` is expected to be the `/analyze` response's entity list for
/// `case.text` (the shape `pii_detection_api::wire::Entity` already carries: a type,
/// a code-point `start`/`end`, and a score). This function only reads
/// `start`, `end` and `entity_type`: a caller with the full wire response
/// can pass its entities slice directly.
pub fn score_case(case: &Case, entities: &[Entity]) -> Outcome {
    let scored = scored_case(case, entities);
    Outcome {
        masked: scored
            .mask
            .iter()
            .filter(|verdict| verdict.found)
            .map(|verdict| verdict.value.clone())
            .collect(),
        missed: scored
            .mask
            .iter()
            .filter(|verdict| !verdict.found)
            .map(|verdict| verdict.value.clone())
            .collect(),
        over_masked: scored
            .clear
            .iter()
            .filter(|verdict| verdict.violated)
            .map(|verdict| verdict.value.clone())
            .collect(),
    }
}

/// One `must_mask` value's verdict: whether it was found, and the entity
/// type(s) attributable to it (see the module docs on why attribution is
/// best-effort and can be empty).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaskVerdict {
    pub value: String,
    pub found: bool,
    /// The type the case itself declared for this value, if it declared one.
    /// This is what [`Summary::per_type`]'s recall side counts.
    pub declared: Option<EntityType>,
    /// The distinct types of every detected span that touched this value,
    /// even partially. Empty when nothing touched it. Diagnostic only: see
    /// the module docs on why the roll-up no longer keys recall off this.
    pub types: Vec<EntityType>,
}

/// One `must_clear` value's verdict. Unlike [`MaskVerdict`], `types` is never
/// empty when `violated` is true: a violation is defined by a span
/// overlapping the value, so that span's type is always attributable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClearVerdict {
    pub value: String,
    pub violated: bool,
    pub types: Vec<EntityType>,
}

/// One case's full scoring detail: [`Outcome`]'s verdicts, kept per-value
/// with the entity types attributable to each, plus the case's language and
/// the detected spans that matched no `must_mask` occurrence at all
/// (false positives with no home in either verdict list). This is what
/// [`Summary`] is built from: [`Outcome`] alone has thrown away the
/// language and the type attribution a per-language/per-type roll-up needs.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CaseOutcome {
    pub lang: String,
    pub mask: Vec<MaskVerdict>,
    pub clear: Vec<ClearVerdict>,
    /// The type of every detected span that covers no `must_mask`
    /// occurrence in this case at all: false positives for precision that
    /// are not `must_clear` violations (nothing named them ahead of time,
    /// they simply weren't supposed to be there).
    pub spurious_spans: Vec<EntityType>,
}

/// Scores one case with the full per-value, per-type detail [`Summary`]
/// needs. [`score_case`] is a thin projection of this onto the brief's
/// [`Outcome`] shape; this is the richer form fed to [`Summary::from`].
pub fn scored_case(case: &Case, entities: &[Entity]) -> CaseOutcome {
    let chars: Vec<char> = case.text.chars().collect();
    let doc = Document::new(case.text.clone());
    let typed_spans: Vec<TypedSpan> = entities
        .iter()
        .map(|entity| (entity.start, entity.end, entity.entity_type))
        .collect();

    let mask: Vec<MaskVerdict> = case
        .must_mask
        .iter()
        .map(|value| {
            let occurrences = find_occurrences(&doc, value.as_str());
            let found = !occurrences.is_empty()
                && occurrences
                    .iter()
                    .all(|occurrence| fully_covered(&chars, *occurrence, &typed_spans));
            MaskVerdict {
                value: value.as_str().to_string(),
                found,
                declared: value.entity_type(),
                types: attributed_types(&occurrences, &typed_spans),
            }
        })
        .collect();

    let clear: Vec<ClearVerdict> = case
        .must_clear
        .iter()
        .map(|value| {
            let occurrences = find_occurrences(&doc, value);
            let violated = occurrences
                .iter()
                .any(|occurrence| overlaps_any(*occurrence, &typed_spans));
            ClearVerdict {
                value: value.clone(),
                violated,
                types: attributed_types(&occurrences, &typed_spans),
            }
        })
        .collect();

    let mask_occurrences: Vec<Span> = case
        .must_mask
        .iter()
        .flat_map(|value| find_occurrences(&doc, value.as_str()))
        .collect();

    let spurious_spans: Vec<EntityType> = typed_spans
        .iter()
        .filter(|(start, end, _)| {
            !mask_occurrences
                .iter()
                .any(|occurrence| overlaps(*occurrence, (*start, *end)))
        })
        .map(|(_, _, entity_type)| *entity_type)
        .collect();

    CaseOutcome {
        lang: case.lang.clone(),
        mask,
        clear,
        spurious_spans,
    }
}

/// Every non-overlapping code-point occurrence of `value` in `doc`, in
/// order. Rust string search is byte-based; each match's byte range is
/// converted to code points once via [`Document::char_span`], the same
/// conversion the rest of the workspace uses.
fn find_occurrences(doc: &Document, value: &str) -> Vec<Span> {
    if value.is_empty() {
        return Vec::new();
    }
    doc.as_str()
        .match_indices(value)
        .filter_map(|(byte_start, matched)| {
            let byte_end = byte_start + matched.len();
            doc.char_span(byte_start, byte_end)
                .map(|span| (span.start, span.end))
        })
        .collect()
}

fn overlaps(a: Span, b: Span) -> bool {
    a.0 < b.1 && b.0 < a.1
}

fn overlaps_any(occurrence: Span, spans: &[TypedSpan]) -> bool {
    spans
        .iter()
        .any(|(start, end, _)| overlaps(occurrence, (*start, *end)))
}

/// True when every non-whitespace character of `occurrence` is covered by
/// the union of `spans`: the corrected must_mask rule. A character is
/// itself exempt from needing coverage when it is whitespace; a gap that
/// falls only on whitespace between spans leaks nothing.
///
/// ONE PLACE THIS SCORER AND THE PYTHON ONE COULD DISAGREE. This benchmark's
/// whole premise is that the detector, the model and Azure are scored by one
/// scorer, so that the numbers are comparable, and `training/` still holds
/// the Python scorer that produced the earlier 128-case figures. The two
/// spell "whitespace" differently in exactly one place: Rust's
/// `char::is_whitespace()` is the Unicode `White_Space` property, which does
/// **not** include U+001C–U+001F (the file, group, record and unit
/// separators), while Python's `str.isspace()` does. A `must_mask` value
/// containing one of those four characters, with a detected span stopping
/// either side of it, would be a leak here and not a leak there.
///
/// It is unreachable today, and checked rather than assumed:
/// `no_case_text_contains_a_character_the_two_scorers_disagree_about` below
/// asserts every committed case contains none of the four. Documented rather than
/// reconciled because reconciling means changing one of the two scorers to
/// agree with the other, and neither is obviously the right one to bend:
/// the day a fixture needs those characters is the day to decide which.
fn fully_covered(chars: &[char], occurrence: Span, spans: &[TypedSpan]) -> bool {
    (occurrence.0..occurrence.1).all(|index| {
        let is_whitespace = chars
            .get(index as usize)
            .is_some_and(|character| character.is_whitespace());
        is_whitespace
            || spans
                .iter()
                .any(|(start, end, _)| *start <= index && index < *end)
    })
}

/// The distinct entity types of every span that overlaps any occurrence in
/// `occurrences`, even partially. Empty when nothing overlaps at all: see
/// the module docs on why that case cannot be guessed into a type.
fn attributed_types(occurrences: &[Span], spans: &[TypedSpan]) -> Vec<EntityType> {
    let mut types = BTreeSet::new();
    for occurrence in occurrences {
        for (start, end, entity_type) in spans {
            if overlaps(*occurrence, (*start, *end)) {
                types.insert(*entity_type);
            }
        }
    }
    types.into_iter().collect()
}

/// The wire name for a detected entity type (`NATIONAL_ID`, not
/// `NationalId`), matching `EntityType`'s own `SCREAMING_SNAKE_CASE` serde
/// rename (`pii_detection_engine::entity`). Written as an exhaustive match rather
/// than reusing `serde_json::to_value` so that adding an `EntityType`
/// variant without updating this function fails to compile instead of
/// falling back to a placeholder string that would silently merge that
/// type's stats into an "unknown" bucket.
fn wire_name(entity_type: EntityType) -> &'static str {
    match entity_type {
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

/// Recall and precision counts, aggregated over some slice of cases (all of
/// them, one language, or one entity type). Always raw counts, never a
/// pre-divided fraction: [`Metrics::recall`] and [`Metrics::precision`]
/// divide on demand and return `None` for an empty denominator rather than
/// asserting or defaulting to a number that would be indistinguishable from
/// a real, measured zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Metrics {
    pub mask_total: usize,
    pub mask_found: usize,
    pub clear_total: usize,
    pub clear_violated: usize,
    pub spurious_spans: usize,
}

impl Metrics {
    /// `mask_found / mask_total`, or `None` when there were no `must_mask`
    /// values to measure.
    pub fn recall(&self) -> Option<f64> {
        if self.mask_total == 0 {
            None
        } else {
            Some(self.mask_found as f64 / self.mask_total as f64)
        }
    }

    /// True positives (`mask_found`) over true positives plus false
    /// positives (`clear_violated`, the `must_clear` values touched, plus
    /// `spurious_spans`, the detected spans that covered no `must_mask`
    /// occurrence at all). `None` when both terms are zero: nothing was
    /// found and nothing was wrongly flagged, so there is nothing to divide.
    pub fn precision(&self) -> Option<f64> {
        let false_positives = self.clear_violated + self.spurious_spans;
        let denominator = self.mask_found + false_positives;
        if denominator == 0 {
            None
        } else {
            Some(self.mask_found as f64 / denominator as f64)
        }
    }

    fn merge(&mut self, other: &Metrics) {
        self.mask_total += other.mask_total;
        self.mask_found += other.mask_found;
        self.clear_total += other.clear_total;
        self.clear_violated += other.clear_violated;
        self.spurious_spans += other.spurious_spans;
    }
}

/// Recall and precision averaged across languages with every language
/// weighted equally, regardless of how many cases it has. See the module
/// docs: this exists because [`Summary::overall`]'s pooled ("micro") figure
/// is dominated by whichever language has the most cases, which can hide a
/// language that fails badly if it happens to have few.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct MacroAverage {
    pub recall: Option<f64>,
    pub precision: Option<f64>,
}

/// Recall and precision, overall, per entity type and per language, rolled
/// up from many cases' [`CaseOutcome`]s.
#[derive(Debug, Clone, Default)]
pub struct Summary {
    /// Pooled over every case and every occurrence: "micro."
    pub overall: Metrics,
    /// One [`Metrics`] per `Case::lang`, each itself pooled ("micro") within
    /// that language.
    pub per_language: HashMap<String, Metrics>,
    /// One [`Metrics`] per entity type wire name (`"NATIONAL_ID"`,
    /// `"PHONE_NUMBER"`, ...). Recall counts values whose case **declared**
    /// that type; precision counts `must_clear` violations and spurious spans
    /// by the type the **detector** returned. See the module docs for why the
    /// two sides are keyed differently.
    pub per_type: HashMap<String, Metrics>,
    /// How many `must_mask` values across every case declared no type and so
    /// could not be counted in [`Summary::per_type`]'s recall. Reported, not
    /// hidden: while this is above zero a per-type recall covers only part of
    /// the fixture, and any report quoting one must say so.
    pub untyped_mask_values: usize,
    /// Recall and precision averaged across languages, every language
    /// weighted equally.
    pub macro_by_language: MacroAverage,
}

impl From<Vec<CaseOutcome>> for Summary {
    fn from(cases: Vec<CaseOutcome>) -> Self {
        let mut overall = Metrics::default();
        let mut per_language: HashMap<String, Metrics> = HashMap::new();
        let mut per_type: HashMap<String, Metrics> = HashMap::new();
        let mut untyped_mask_values = 0usize;

        for case in &cases {
            let case_metrics = Metrics {
                mask_total: case.mask.len(),
                mask_found: case.mask.iter().filter(|verdict| verdict.found).count(),
                clear_total: case.clear.len(),
                clear_violated: case.clear.iter().filter(|verdict| verdict.violated).count(),
                spurious_spans: case.spurious_spans.len(),
            };

            overall.merge(&case_metrics);
            per_language
                .entry(case.lang.clone())
                .or_default()
                .merge(&case_metrics);

            for verdict in &case.mask {
                let Some(entity_type) = verdict.declared else {
                    untyped_mask_values += 1;
                    continue;
                };
                let bucket = per_type
                    .entry(wire_name(entity_type).to_string())
                    .or_default();
                bucket.mask_total += 1;
                if verdict.found {
                    bucket.mask_found += 1;
                }
            }
            for verdict in &case.clear {
                for entity_type in &verdict.types {
                    let bucket = per_type
                        .entry(wire_name(*entity_type).to_string())
                        .or_default();
                    bucket.clear_total += 1;
                    if verdict.violated {
                        bucket.clear_violated += 1;
                    }
                }
            }
            for entity_type in &case.spurious_spans {
                per_type
                    .entry(wire_name(*entity_type).to_string())
                    .or_default()
                    .spurious_spans += 1;
            }
        }

        // `per_language` is a `HashMap`, whose iteration order Rust
        // randomises per process. Float addition is not associative, so
        // summing `Metrics::recall()`/`precision()` in hash order made
        // `macro_by_language` vary in the last ULP between runs on
        // identical input. Sorting the keys first makes the summation
        // order -- and so the result -- deterministic. No gate reads this
        // field (per-language gates in `gates.rs` sort their own keys and
        // read `per_language` directly), so this fixes reproducibility,
        // not a verdict.
        let mut sorted_languages: Vec<&String> = per_language.keys().collect();
        sorted_languages.sort();
        let macro_by_language =
            macro_average(sorted_languages.into_iter().map(|lang| &per_language[lang]));

        Summary {
            overall,
            per_language,
            per_type,
            untyped_mask_values,
            macro_by_language,
        }
    }
}

fn macro_average<'a>(per_language: impl Iterator<Item = &'a Metrics>) -> MacroAverage {
    let mut recalls = Vec::new();
    let mut precisions = Vec::new();
    for metrics in per_language {
        if let Some(recall) = metrics.recall() {
            recalls.push(recall);
        }
        if let Some(precision) = metrics.precision() {
            precisions.push(precision);
        }
    }
    MacroAverage {
        recall: mean(&recalls),
        precision: mean(&precisions),
    }
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MustMask;

    /// A case whose `must_mask` values declare a type, for the per-type
    /// assertions. `case_with` builds the bare-string (untyped) form the 128
    /// legacy cases use; both shapes have to keep working.
    fn typed_case_with(text: &str, must_mask: &[(&str, &str)], must_clear: &[&str]) -> Case {
        let mut case = case_with(text, &[], must_clear);
        case.must_mask = must_mask
            .iter()
            .map(|(value, wire)| {
                MustMask::typed(
                    *value,
                    EntityType::from_wire(wire).expect("a known wire type in a test"),
                )
            })
            .collect();
        case
    }

    fn case_with(text: &str, must_mask: &[&str], must_clear: &[&str]) -> Case {
        Case {
            id: "test".to_string(),
            lang: "nl".to_string(),
            len: "short".to_string(),
            source: "test".to_string(),
            carrier_id: None,
            text: text.to_string(),
            must_mask: must_mask.iter().map(|s| MustMask::from(*s)).collect(),
            must_clear: must_clear.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn entity(entity_type: &str, start: u32, end: u32) -> Entity {
        Entity {
            entity_type: EntityType::from_wire(entity_type).expect("a known wire type in a test"),
            country: None,
            start,
            end,
            score: 1.0,
        }
    }

    #[test]
    fn a_must_mask_value_covered_by_a_detected_span_counts_as_found() {
        let case = case_with("Het BSN is 123456782", &["123456782"], &[]);
        let entities = vec![entity("NATIONAL_ID", 11, 20)];
        let outcome = score_case(&case, &entities);
        assert_eq!(outcome.missed.len(), 0);
        assert_eq!(outcome.masked, vec!["123456782".to_string()]);
    }

    #[test]
    fn a_span_that_only_partly_covers_the_value_does_not_count() {
        // Partial masking leaves personal data in the prompt. Half a BSN is
        // still eight digits of a BSN.
        let case = case_with("Het BSN is 123456782", &["123456782"], &[]);
        let entities = vec![entity("NATIONAL_ID", 11, 16)];
        assert_eq!(score_case(&case, &entities).missed.len(), 1);
    }

    #[test]
    fn a_must_clear_value_touched_by_any_span_counts_as_over_masked() {
        // The retired LOCATION failure: masking an address must not mask
        // "Nederland" and ruin the answer the customer asked for.
        let case = case_with("Wat is de hoofdstad van Nederland?", &[], &["Nederland"]);
        let entities = vec![entity("STREET_ADDRESS", 24, 33)];
        assert_eq!(score_case(&case, &entities).over_masked.len(), 1);
    }

    #[test]
    fn a_value_appearing_twice_must_be_masked_in_both_places() {
        // Two occurrences are two pieces of personal data. Masking one is a
        // leak, and a naive first-occurrence check would call it a pass.
        let case = case_with(
            "Bel 0612345678 of schrijf naar 0612345678",
            &["0612345678"],
            &[],
        );
        let entities = vec![entity("PHONE_NUMBER", 4, 14)];
        assert_eq!(score_case(&case, &entities).missed.len(), 1);
    }

    #[test]
    fn a_whitespace_gap_between_two_covering_spans_is_not_a_leak() {
        // The exact defect a single-span rule produced: "Anna" and
        // "Schmidt" as two adjacent spans, together covering the whole
        // name with nothing but a space between them.
        let case = case_with("Hallo Anna Schmidt, bedankt", &["Anna Schmidt"], &[]);
        let entities = vec![entity("PERSON", 6, 10), entity("PERSON", 11, 18)];
        let outcome = score_case(&case, &entities);
        assert_eq!(outcome.missed.len(), 0);
        assert_eq!(outcome.masked, vec!["Anna Schmidt".to_string()]);
    }

    #[test]
    fn recall_and_precision_are_reported_per_type_and_per_language() {
        let nl_case = typed_case_with("Het BSN is 123456782", &[("123456782", "NATIONAL_ID")], &[]);
        let nl_entities = vec![entity("NATIONAL_ID", 11, 20)];

        let mut en_case = typed_case_with(
            "Call 0612345678 now",
            &[("0612345678", "PHONE_NUMBER")],
            &[],
        );
        en_case.lang = "en".to_string();
        let en_entities = vec![entity("PHONE_NUMBER", 5, 15)];

        let summary = Summary::from(vec![
            scored_case(&nl_case, &nl_entities),
            scored_case(&en_case, &en_entities),
        ]);

        assert!(summary.per_language.contains_key("nl"));
        assert!(summary.per_language.contains_key("en"));
        assert!(summary.per_type.contains_key("NATIONAL_ID"));
        assert!(summary.per_type.contains_key("PHONE_NUMBER"));
        assert_eq!(summary.overall.recall(), Some(1.0));
        assert_eq!(summary.untyped_mask_values, 0);
    }

    #[test]
    fn a_declared_value_nothing_detected_still_counts_against_its_own_type() {
        // THE DEFECT THE DECLARED TYPE EXISTS FOR. Two NATIONAL_ID values,
        // one found and one that no span touches at all. Attributing by
        // detected span can only see the found one, so it reports NATIONAL_ID
        // recall 1.0 for a fixture the detector half failed. Declared types
        // put the miss in its own denominator, where 0.5 is the honest
        // answer.
        let found = typed_case_with("Het BSN is 123456782", &[("123456782", "NATIONAL_ID")], &[]);
        let missed = {
            let mut case =
                typed_case_with("Het BSN is 123456790", &[("123456790", "NATIONAL_ID")], &[]);
            case.id = "missed".to_string();
            case
        };

        let summary = Summary::from(vec![
            scored_case(&found, &[entity("NATIONAL_ID", 11, 20)]),
            scored_case(&missed, &[]),
        ]);

        let national_id = summary
            .per_type
            .get("NATIONAL_ID")
            .expect("declared NATIONAL_ID values are counted whether found or not");
        assert_eq!(national_id.mask_total, 2);
        assert_eq!(national_id.mask_found, 1);
        assert_eq!(national_id.recall(), Some(0.5));
        assert_eq!(summary.overall.recall(), Some(0.5));
    }

    #[test]
    fn a_declared_value_is_counted_under_what_the_case_says_not_what_was_detected() {
        // A NATIONAL_ID masked by a spa detector called PHONE_NUMBER is
        // a found value and a type confusion. It belongs to NATIONAL_ID's
        // recall -- that is the type the fixture asked for -- and must not
        // silently inflate PHONE_NUMBER's.
        let case = typed_case_with("Het BSN is 123456782", &[("123456782", "NATIONAL_ID")], &[]);
        let summary = Summary::from(vec![scored_case(&case, &[entity("PHONE_NUMBER", 11, 20)])]);

        assert_eq!(
            summary
                .per_type
                .get("NATIONAL_ID")
                .map(|metrics| (metrics.mask_total, metrics.mask_found)),
            Some((1, 1))
        );
        assert!(
            summary
                .per_type
                .get("PHONE_NUMBER")
                .is_none_or(|metrics| metrics.mask_total == 0 && metrics.mask_found == 0)
        );
    }

    #[test]
    fn untyped_values_are_reported_by_count_rather_than_guessed_into_a_bucket() {
        // The legacy shape: bare strings, no declared type. They must not
        // land in any per-type bucket -- guessing is what produced the
        // flattering figure -- and the count of them has to be visible so a
        // per-type recall is never read as covering the whole fixture.
        let case = case_with("Het BSN is 123456782", &["123456782"], &[]);
        let summary = Summary::from(vec![scored_case(&case, &[entity("NATIONAL_ID", 11, 20)])]);

        assert_eq!(summary.untyped_mask_values, 1);
        assert!(!summary.per_type.contains_key("NATIONAL_ID"));
        assert_eq!(summary.overall.recall(), Some(1.0));
    }

    #[test]
    fn a_pure_negative_case_reports_no_recall_or_precision_rather_than_zero() {
        // No must_mask, no must_clear, nothing detected: there is nothing to
        // measure, and that is a different fact than "measured and failed."
        let case = case_with("Wat is de hoofdstad van Nederland?", &[], &[]);
        let summary = Summary::from(vec![scored_case(&case, &[])]);
        assert_eq!(summary.overall.recall(), None);
        assert_eq!(summary.overall.precision(), None);
    }

    #[test]
    fn a_detected_span_covering_nothing_named_is_a_spurious_false_positive() {
        // A span that overlaps no must_mask occurrence at all is not a
        // must_clear violation (nothing named it) but it is still a false
        // positive for precision.
        let case = case_with("Wat is de hoofdstad van Nederland?", &[], &[]);
        let entities = vec![entity("PERSON", 0, 3)];
        let scored = scored_case(&case, &entities);
        assert_eq!(scored.spurious_spans, vec![EntityType::Person]);

        let summary = Summary::from(vec![scored]);
        assert_eq!(summary.overall.precision(), Some(0.0));
        let person = summary
            .per_type
            .get("PERSON")
            .expect("a spurious PERSON span is attributed to PERSON");
        assert_eq!(person.spurious_spans, 1);
    }

    #[test]
    fn a_completely_missed_untyped_value_counts_overall_but_not_per_type() {
        // No declared type and nothing detected, so there is no basis at all
        // to place it in a bucket. It must still show up in the overall and
        // per-language totals -- silently dropping it would flatter recall
        // exactly the way the retired single-span rule did -- and in the
        // untyped count, so the gap is visible rather than implied.
        let case = case_with("Het BSN is 123456782", &["123456782"], &[]);
        let summary = Summary::from(vec![scored_case(&case, &[])]);
        assert_eq!(summary.overall.recall(), Some(0.0));
        assert!(summary.per_type.is_empty());
        assert_eq!(summary.untyped_mask_values, 1);
    }

    #[test]
    fn a_partially_touched_value_is_attributed_even_though_it_is_missed() {
        let case = case_with("Het BSN is 123456782", &["123456782"], &[]);
        let entities = vec![entity("NATIONAL_ID", 11, 16)];
        let scored = scored_case(&case, &entities);
        assert_eq!(scored.mask.len(), 1);
        assert!(!scored.mask[0].found);
        assert_eq!(scored.mask[0].types, vec![EntityType::NationalId]);
    }

    #[test]
    fn macro_by_language_weighs_every_language_equally() {
        // "nl" has nine perfect cases, "en" has one total miss. Micro
        // (pooled) recall is dominated by "nl"; macro must not be.
        let nl_case = case_with("Het BSN is 123456782", &["123456782"], &[]);
        let nl_entities = vec![entity("NATIONAL_ID", 11, 20)];
        let en_case = case_with("Het BSN is 123456782", &["123456782"], &[]);

        let mut cases: Vec<CaseOutcome> = (0..9)
            .map(|_| scored_case(&nl_case, &nl_entities))
            .collect();
        let mut en_missed = scored_case(&en_case, &[]);
        en_missed.lang = "en".to_string();
        cases.push(en_missed);

        let summary = Summary::from(cases);
        assert_eq!(summary.overall.recall(), Some(0.9));
        assert_eq!(summary.macro_by_language.recall, Some(0.5));
    }

    #[test]
    fn every_wire_name_round_trips_through_from_wire() {
        for entity_type in [
            EntityType::CreditCard,
            EntityType::DrivingLicence,
            EntityType::EmailAddress,
            EntityType::IbanCode,
            EntityType::IpAddress,
            EntityType::NationalId,
            EntityType::PassportNumber,
            EntityType::Person,
            EntityType::PhoneNumber,
            EntityType::StreetAddress,
            EntityType::TaxId,
            EntityType::VatNumber,
        ] {
            let wire = wire_name(entity_type);
            assert_eq!(EntityType::from_wire(wire), Some(entity_type));
        }
    }

    // --- Multibyte regression guards -------------------------------------
    //
    // Every test above this point uses pure-ASCII case text, so none of them
    // would catch a byte-versus-code-point regression in find_occurrences or
    // fully_covered: for ASCII, byte offsets and code-point offsets are the
    // same numbers, so a bug that silently swapped one for the other would
    // pass unnoticed. These three use real multibyte content -- umlauts, a
    // cedilla, and an emoji -- placed so that a byte-offset occurrence and
    // the true code-point occurrence land in different places entirely,
    // which is what makes a regression here fail loudly instead of by luck.

    #[test]
    fn a_value_after_multibyte_and_emoji_text_is_found_at_its_true_code_point_offset() {
        // "Größe: 🎉 François meldt zich, BSN " is 34 code points but 40
        // bytes -- six of them spent on ö, the emoji, and ç. A regression
        // that used the byte offset of "123456782" (40) instead of its
        // code-point offset (34) would compare it against an entity span
        // that no longer overlaps it at all, and the value would come back
        // missed instead of masked.
        let case = case_with(
            "Größe: 🎉 François meldt zich, BSN 123456782 vandaag.",
            &["123456782"],
            &[],
        );
        let entities = vec![entity("NATIONAL_ID", 34, 43)];
        let outcome = score_case(&case, &entities);
        assert_eq!(outcome.missed.len(), 0);
        assert_eq!(outcome.masked, vec!["123456782".to_string()]);
    }

    #[test]
    fn a_multibyte_value_split_across_two_spans_by_whitespace_is_found() {
        // The "Anna Schmidt" whitespace-gap test, but with the accented
        // characters that make byte and code-point offsets actually
        // diverge: "François" (byte span 6..15) and "Müller" (byte span
        // 16..23) versus their true code-point spans, 6..14 and 15..21. A
        // byte-based fully_covered would index into the wrong characters
        // for the back half of the value and fail to see it as covered.
        let case = case_with("Hallo François Müller, bedankt", &["François Müller"], &[]);
        let entities = vec![entity("PERSON", 6, 14), entity("PERSON", 15, 21)];
        let outcome = score_case(&case, &entities);
        assert_eq!(outcome.missed.len(), 0);
        assert_eq!(outcome.masked, vec!["François Müller".to_string()]);
    }

    #[test]
    fn no_case_text_contains_a_character_the_two_scorers_disagree_about() {
        // See fully_covered's docs. Rust's char::is_whitespace() (Unicode
        // White_Space) and Python's str.isspace() agree on everything the
        // fixture actually contains, and differ on exactly these four
        // separators. While no case carries one, the Rust scorer here and
        // the Python scorer in training/ cannot disagree about whether a gap
        // between two spans is a leak -- which is the premise the whole
        // "one scorer, comparable numbers" argument rests on.
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmark/cases");
        let cases = crate::load(&dir).expect("the fixture loads");
        let disputed = ['\u{1c}', '\u{1d}', '\u{1e}', '\u{1f}'];
        for case in &cases {
            assert!(
                !case.text.chars().any(|c| disputed.contains(&c)),
                "case {} carries a character the two scorers disagree about",
                case.id
            );
        }
    }

    #[test]
    fn a_partially_covered_multibyte_value_is_still_reported_missed() {
        // Same value and spans as above, minus the "Müller" span: proves
        // the multibyte tests above are sensitive to real coverage rather
        // than passing regardless of what the entities say.
        let case = case_with("Hallo François Müller, bedankt", &["François Müller"], &[]);
        let entities = vec![entity("PERSON", 6, 14)];
        let outcome = score_case(&case, &entities);
        assert_eq!(outcome.missed, vec!["François Müller".to_string()]);
        assert_eq!(outcome.masked.len(), 0);
    }
}
