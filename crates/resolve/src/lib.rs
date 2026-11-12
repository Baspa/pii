use pii_detection_engine::candidate::{Candidate, Source};
use pii_detection_engine::entity::EntityType;
use pii_detection_engine::text::CharSpan;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

pub mod calibration;

pub use calibration::{Calibration, CalibrationError};

fn thresholds() -> &'static BTreeMap<EntityType, f32> {
    static THRESHOLDS: OnceLock<BTreeMap<EntityType, f32>> = OnceLock::new();
    THRESHOLDS.get_or_init(|| {
        toml::from_str(include_str!("../thresholds.toml"))
            .unwrap_or_else(|error| panic!("thresholds are malformed: {error}"))
    })
}

/// What a negative-context term takes off a candidate's score before it is
/// weighed against its threshold.
///
/// ### The defect this replaces
///
/// A suppressed candidate used to be deleted outright, ahead of the
/// threshold and ahead of its own checksum. That made a term list a veto
/// over evidence, which the design does not say: the resolver
/// *combines* evidence, and a passing checksum is the strongest evidence a
/// structured identifier can carry. Three separate readings found the same
/// hole from three directions -- `in order to` deleting a checksum-valid
/// BSN, `iban` deleting a checksum-valid German Steuer-ID whenever a bank
/// account sat in the same paragraph, and the phone recalibration declining
/// terms it wanted rather than trade over-masking for silent leaks. All
/// three are the same sentence: a delete cannot be outweighed, so real
/// personal data left the firewall unmasked.
///
/// ### Why a penalty and not "a checksum wins"
///
/// "A passing checksum survives suppression" is the obvious rule and it is
/// wrong, because an order number passes checksums too. `123456782` clears
/// the elfproef, `86095742719` clears the German tax-id check and
/// `4111117074292632` clears Luhn -- and the benchmark carries each of them
/// behind an order or invoice label as a `must_clear` value, because that
/// is what those digits are. A rule that let a checksum win outright would
/// mask all of them.
///
/// What separates the two readings is not the checksum, it is everything
/// else beside it. `BSN: 804985492` next to an IBAN carries a passing
/// checksum *and* the identifier's own label; `ordernummer 938570833`
/// carries a passing checksum and a label saying it is an order. So
/// suppression is weighed with the rest of the evidence rather than ranked
/// above it, which is what section 6 step 6 asks for.
///
/// ### Where the number comes from
///
/// It is derived from the packs, not chosen. Writing `b` for a
/// recognizer's `base_confidence`, `c` for `checksum_confidence`, `t` for
/// `context_confidence` and `T` for its threshold in `thresholds.toml`:
///
/// * A candidate carrying format and a checksum but no label of its own
///   must not clear its bar -- that is the order number. The tightest of
///   those is `nl_bsn` and `credit_card` at `b + c = 0.90` against
///   `T = 0.70`, so the penalty must exceed `0.20`.
/// * A candidate carrying format, a checksum *and* the identifier's own
///   label must clear it -- that is the real identifier. Those sit at
///   `b + c + t = 1.00` against `T = 0.70`, so the penalty must not
///   exceed `0.30`.
///
/// `0.25` is the centre of `(0.20, 0.30]`, which leaves the same 0.05 of
/// margin on each side. `tests/suppression.rs` in `pii-detection-rules` pins both
/// edges through the real packs and the real thresholds, so a pack whose
/// weights drift out of that window fails a test rather than silently
/// changing which of the two readings wins.
const NEGATIVE_CONTEXT_PENALTY: f32 = 0.25;

/// The score a candidate is weighed against its threshold with.
///
/// The candidate's own `score` is left alone: it is what the recognizer
/// concluded about the span, it reaches the API on every returned entity
/// (`pii_detection_api::wire`), and a negative context is a fact about the
/// *neighbourhood* rather than about the span. So the penalty applies to
/// the comparison and to nothing else -- including `strength_order`, where
/// a suppressed candidate that survived is ranked by what it is, not by
/// what stood next to it.
fn weighed_score(candidate: &Candidate) -> f32 {
    if candidate.evidence.negative_context {
        candidate.score - NEGATIVE_CONTEXT_PENALTY
    } else {
        candidate.score
    }
}

/// Whether a type has a calibrated threshold.
///
/// Exposed so `pii-detection-rules`, where the packs and the phone recognizer both
/// live, can assert at test time that every active type is calibrated. An
/// uncalibrated type already fails closed at runtime (see `resolve`), but
/// that failure should be caught by CI, not discovered by a type silently
/// never matching in production.
pub fn has_threshold(entity_type: EntityType) -> bool {
    thresholds().contains_key(&entity_type)
}

/// A rule-sourced type's threshold from `thresholds.toml`, so a test can hold a span to the bar the
/// resolver holds it to.
pub fn rule_threshold(entity_type: EntityType) -> Option<f32> {
    thresholds().get(&entity_type).copied()
}

/// `calibration` is `None` for a resolver built with [`Resolver::new`] --
/// every model-sourced candidate then fails closed (see `resolve`'s
/// threshold match), the same as an uncalibrated type in `thresholds.toml`.
/// Rule-sourced candidates never consult this field at all.
pub struct Resolver {
    calibration: Option<Calibration>,
}

impl Default for Resolver {
    fn default() -> Self {
        Self::new()
    }
}

/// What [`Resolver::resolve_reporting`] kept, and how many phone numbers it cleared as a company's.
pub struct Resolved {
    pub kept: Vec<Candidate>,
    pub attributed_to_company: usize,
}

impl Resolver {
    pub fn new() -> Self {
        Self { calibration: None }
    }

    /// A resolver whose model-sourced candidates are gated by `calibration`
    /// instead of failing closed. Rule-sourced candidates
    /// are unaffected either way -- see `resolve`'s threshold match.
    pub fn with_calibration(calibration: Calibration) -> Self {
        Self {
            calibration: Some(calibration),
        }
    }

    /// Turns raw candidates into the final answer.
    ///
    /// Order matters, and it is the design's: section 6, steps 6 to 8.
    /// Step 6 runs first and runs as one stage: checksums, negative contexts
    /// and the calibrated threshold are weighed *together* (see
    /// [`NEGATIVE_CONTEXT_PENALTY`] and [`weighed_score`]) rather than applied
    /// as a sequence of deletes, so a rejected candidate can never win an
    /// overlap against a real one and no single signal can veto the rest.
    /// Then overlaps are resolved. The filter to requested types runs last.
    ///
    /// A failed checksum is the one thing that still rejects on its own, and
    /// only where the identifier defines a checksum at all: `Some(false)` is
    /// the identifier saying it is not that identifier, which is a different
    /// claim from a word nearby saying it probably is not.
    ///
    /// Filtering first looks equivalent and is not. For
    /// `IBAN NL91 ABNA 0417 1643 00` the VAT pattern matches the sub-span
    /// `NL91 ABNA 0417`. With both types requested the checksum-validated IBAN
    /// wins the overlap and nothing is wrong. In a workspace with VAT_NUMBER
    /// enabled and IBAN_CODE not, filtering first deletes the IBAN candidate
    /// before the overlap is decided, and the caller receives a truncated span
    /// mislabelled VAT_NUMBER: the pseudonymiser substitutes a VAT placeholder
    /// and leaves `1643 00` sitting in the text, so part of an IBAN reaches the
    /// model in the clear.
    ///
    /// Filtering last, that workspace gets *nothing* for the span. The IBAN won
    /// the overlap and was then filtered out. That is deliberate and it is the
    /// right answer: the workspace did not ask for IBANs, and a mislabelled
    /// fragment is worse than silence.
    ///
    /// There is exactly one exception, and it is the last step here: when the
    /// filter removes a winner, the strongest requested candidate whose span is
    /// *identical* to that winner's is restored. For
    /// `paspoortnummer 123456782` in a workspace that asked only for
    /// PASSPORT_NUMBER, `nl_bsn` raises NATIONAL_ID over the nine digits with a
    /// passing 11-proef and the universal `passport` recognizer raises
    /// PASSPORT_NUMBER over the same nine digits on the strength of the word
    /// before them. The BSN wins the overlap, the filter removes it, and
    /// without this step a workspace that explicitly enabled passport detection
    /// receives an empty list while a Dutch passport number stays in the text.
    ///
    /// Identical, not overlapping, and the distinction is the whole rule.
    /// Identical spans mean two recognizers disagree about what one piece of
    /// text *is*; the caller asked about one of those readings, so it is
    /// answered with it, over exactly the characters the winner covered.
    /// Differing spans mean the requested candidate covers only part of what
    /// the winner covered, and handing that back is the truncated, mislabelled
    /// fragment described above, so the IBAN text stays empty for a VAT-only
    /// workspace, because the VAT candidate spans `NL91 ABNA 0417` and the
    /// winning IBAN spans `NL91 ABNA 0417 1643 00`.
    ///
    /// Restoration draws only from candidates that already survived step 6
    /// (failed checksums, and the threshold the negative-context penalty is
    /// weighed into: that stage runs first for a reason and nothing it
    /// rejected comes back) and picks between
    /// several equally-spanned candidates with the same `strength_order` the
    /// overlap used. A restored candidate cannot overlap a kept one: its span
    /// is a span that just won an overlap, and a winner overlaps no other
    /// winner.
    ///
    /// Where two requested readings share a vacated span the stronger wins,
    /// which means the label can be decided by score alone, and it is the
    /// same label the caller would have received had the unrequested
    /// recognizer never existed. `paspoortnummer 123456782` also raises
    /// PHONE_NUMBER over those same nine digits at a flat 0.9, so a workspace
    /// that enabled passports and phones but not national IDs is answered
    /// PHONE_NUMBER. The characters masked are identical either way, so
    /// nothing reaches the model in the clear; only the label differs.
    /// `requested` is deliberately not consulted as a preference order:
    /// it is the set of types a workspace enabled, and letting its order
    /// decide would let a caller force a weaker reading over a stronger one.
    /// The blunt instrument here is the phone recognizer's flat score, which
    /// is a calibration problem and belongs to the model plan.
    ///
    /// `language` reaches exactly one place: the calibration threshold
    /// lookup for a model-sourced candidate (see the threshold match
    /// below). A rule-sourced candidate never consults it -- `thresholds.toml`
    /// has no per-language axis at all.
    ///
    /// Before step 6, a phone number that belongs to a company is dropped; see [`attribute_phones`].
    pub fn resolve(
        &self,
        candidates: Vec<Candidate>,
        requested: &[EntityType],
        language: &str,
    ) -> Vec<Candidate> {
        self.resolve_reporting(candidates, requested, language).kept
    }

    /// [`Resolver::resolve`], with the count the API reports in `/metrics`.
    pub fn resolve_reporting(
        &self,
        mut candidates: Vec<Candidate>,
        requested: &[EntityType],
        language: &str,
    ) -> Resolved {
        // Without a model there is no PERSON evidence, and its absence proves nothing.
        let attributed_to_company = if self.calibration.is_some() {
            attribute_phones(&mut candidates)
        } else {
            0
        };
        let mut surviving: Vec<Candidate> = candidates
            .into_iter()
            .filter(|candidate| candidate.evidence.checksum != Some(false))
            .filter(|candidate| {
                // The split the design draws: a rule-sourced candidate
                // is gated by `thresholds.toml`, unaffected by whatever
                // `calibration.json` says; a model-sourced candidate is
                // gated by `calibration.json`, unaffected by
                // `thresholds.toml` -- keeping the two apart means this
                // struct's own model-calibration wiring can never move a
                // structured-type gate. Either way, an uncalibrated type (no
                // entry in the relevant table, or no calibration loaded at
                // all) gets `f32::INFINITY`: no score clears it, so the type
                // never matches until it is calibrated. Failing open here
                // (e.g. `unwrap_or(0.5)`) would give an unreviewed type a
                // *more* permissive bar than every calibrated one, which is
                // backwards for a fail-closed design.
                let threshold = match candidate.source {
                    Source::Model => self.calibration.as_ref().and_then(|calibration| {
                        calibration.threshold(candidate.entity_type, language)
                    }),
                    Source::Rule { .. } => thresholds().get(&candidate.entity_type).copied(),
                }
                .unwrap_or(f32::INFINITY);
                weighed_score(candidate) >= threshold
            })
            .collect();

        // A reading of a *fragment* of another reading is dropped before the
        // overlap is decided. See [`strictly_contains`] for the whole
        // argument; in short, `strength_order` ranks a passing checksum above
        // a longer span, which is right for two spans that merely overlap and
        // wrong for one that sits wholly inside another -- there, picking the
        // shorter one leaves the rest of the longer one in the clear.
        //
        // O(n^2) over the candidates of one document, the same order the
        // overlap loop below already is.
        let spans: Vec<(u32, u32)> = surviving
            .iter()
            .map(|candidate| (candidate.span.start, candidate.span.end))
            .collect();
        surviving.retain(|candidate| {
            let inner = (candidate.span.start, candidate.span.end);
            !spans.iter().any(|outer| strictly_contains(*outer, inner))
        });

        surviving.sort_by(|a, b| {
            strength_order(a, b)
                .then((a.span.start, a.span.end).cmp(&(b.span.start, b.span.end)))
                .then(a.entity_type.cmp(&b.entity_type))
        });

        let mut kept: Vec<Candidate> = Vec::new();
        // The losers, in the same strength-descending order, because a loser
        // is where a restoration comes from once the filter has run.
        let mut displaced: Vec<Candidate> = Vec::new();
        for candidate in surviving {
            let overlaps = kept.iter().any(|existing| {
                candidate.span.start < existing.span.end && existing.span.start < candidate.span.end
            });
            if overlaps {
                displaced.push(candidate);
            } else {
                kept.push(candidate);
            }
        }

        // Step 8, and last on purpose: a type the caller did not ask for still
        // takes part in deciding the overlap it belongs to, and only then
        // leaves. See the note on this method for what that costs and why the
        // cost is the right one.
        let mut vacated: BTreeSet<(u32, u32)> = BTreeSet::new();
        kept.retain(|candidate| {
            if requested.contains(&candidate.entity_type) {
                true
            } else {
                vacated.insert((candidate.span.start, candidate.span.end));
                false
            }
        });

        // The exception: a span a winner vacated goes to the strongest
        // requested candidate that claimed *exactly* that span, and to nobody
        // else. `displaced` is still strength-ordered, so the first match per
        // span is the strongest one, and keying by span restores at most one
        // candidate per vacated span.
        let mut restored: BTreeMap<(u32, u32), Candidate> = BTreeMap::new();
        for candidate in displaced {
            let span = (candidate.span.start, candidate.span.end);
            if vacated.contains(&span) && requested.contains(&candidate.entity_type) {
                restored.entry(span).or_insert(candidate);
            }
        }
        kept.extend(restored.into_values());

        kept.sort_by_key(|candidate| (candidate.span.start, candidate.span.end));
        Resolved {
            kept,
            attributed_to_company,
        }
    }
}

/// Drops a phone number a company stands beside when nothing personal does. A PERSON span anywhere
/// in the number's scope (the whole text) counts, at any score, before any threshold, including one inside the
/// company's own name.
fn attribute_phones(candidates: &mut Vec<Candidate>) -> usize {
    let people: Vec<CharSpan> = candidates
        .iter()
        .filter(|candidate| candidate.entity_type == EntityType::Person)
        .map(|candidate| candidate.span)
        .collect();
    let before = candidates.len();
    candidates.retain(|candidate| {
        let evidence = candidate.evidence;
        let attributed = candidate.entity_type == EntityType::PhoneNumber
            && evidence.company_nearby
            && !evidence.personal_signal
            && evidence.scope.is_some_and(|scope| {
                !people
                    .iter()
                    .any(|person| person.start < scope.end && scope.start < person.end)
            });
        !attributed
    });
    before - candidates.len()
}

/// Whether `outer` covers `inner` and is strictly longer than it -- one
/// recognizer reading a *fragment* of what another read.
///
/// ### The leak this closes, and why `strength_order` cannot
///
/// [`strength_order`] ranks a passing checksum above a longer span, and that
/// is right for two spans that merely overlap: neither covers the other, so
/// whichever loses leaves characters unmasked either way, and a validated
/// identifier is the better guess about which characters matter.
///
/// It is wrong when one span sits wholly inside another, and the
/// `CREDENTIAL` recognizers are what made that visible. A Slack token is
/// `xoxb-` then two eleven-digit segments then the secret, and an
/// eleven-digit run bounded by hyphens is exactly the German Steuer-ID's
/// shape -- so roughly one segment in eleven clears `german_tax_id`'s check
/// digit by chance. That candidate then *outranked* the fifty-character
/// credential it sits inside, won the overlap, and the resolver masked
/// eleven digits of a token and left the other thirty-nine characters --
/// including the secret -- in the clear. Two of the seventy-two credential
/// values in the first fixture run went out that way.
///
/// Masking the container always masks the contained span too, so preferring
/// the container cannot leak; preferring the fragment can, and did. The rule
/// is therefore: a strictly-contained reading is not a competitor, it is a
/// fragment, and it is dropped before the overlap is decided.
///
/// ### Why a filter and not another key in `strength_order`
///
/// Containment is a partial order. Folded in as a comparator key ahead of
/// the checksum it makes `strength_order` non-transitive -- `A ⊃ B`, `C`
/// disjoint from both and checksummed, gives `A < B`, `C < A`, `B < C` --
/// and `sort_by` with an inconsistent comparator produces an unspecified
/// order. Applying it as a filter over the surviving set keeps the
/// comparator a total order and is what the rule actually means: the
/// fragment never entered the competition.
///
/// ### What this deliberately does not touch
///
/// **Equal spans.** `strictly_contains` is false for two identical spans, so
/// two recognizers disagreeing about what one piece of text *is* --
/// `paspoortnummer 123456782` read as both `NATIONAL_ID` and
/// `PASSPORT_NUMBER` -- is decided by [`strength_order`] exactly as before,
/// and the equal-span restoration in [`Resolver::resolve`] still has its
/// candidates to draw from. That restoration could never have drawn a
/// strictly-contained candidate anyway: it keys on an *identical* span.
fn strictly_contains(outer: (u32, u32), inner: (u32, u32)) -> bool {
    outer.0 <= inner.0 && inner.1 <= outer.1 && (outer.1 - outer.0) > (inner.1 - inner.0)
}

/// Orders two candidates from strongest to weakest, for deciding overlaps.
///
/// A passing checksum outranks a longer span, and a longer span outranks a
/// higher score, because a validated identifier is evidence and a score is
/// an opinion. Expressed as a lexicographic `Ordering` chain (checksum,
/// then span length, then score) rather than as arithmetic summed into one
/// float, so no signal can ever overflow into the next one's territory (a
/// long enough `Person` or `StreetAddress` span, once those get
/// recognizers, could otherwise out-add a passing checksum).
fn strength_order(a: &Candidate, b: &Candidate) -> Ordering {
    let checksum_passed = |candidate: &Candidate| candidate.evidence.checksum == Some(true);
    let span_len = |candidate: &Candidate| candidate.span.end - candidate.span.start;

    checksum_passed(b)
        .cmp(&checksum_passed(a))
        .then_with(|| span_len(b).cmp(&span_len(a)))
        .then_with(|| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pii_detection_engine::candidate::{Candidate, Evidence, Source};
    use pii_detection_engine::entity::EntityType;
    use pii_detection_engine::text::CharSpan;

    fn candidate(start: u32, end: u32, entity_type: EntityType, score: f32) -> Candidate {
        Candidate {
            span: CharSpan { start, end },
            entity_type,
            country: None,
            source: Source::Rule { id: "test" },
            score,
            evidence: Evidence {
                format: true,
                ..Evidence::default()
            },
        }
    }

    /// The leak [`strictly_contains`] closes, in its own terms: a short
    /// candidate carrying a passing checksum sitting inside a long one that
    /// carries none.
    ///
    /// Both halves are asserted. Deleting the containment filter makes the
    /// first assertion fail (the fragment wins and thirty-nine of the fifty
    /// characters go out unmasked); weakening it to plain overlap makes the
    /// second fail (two candidates that merely overlap must still be decided
    /// by the checksum, which is what `strength_order` is for).
    #[test]
    fn a_checksummed_fragment_loses_to_the_span_that_contains_it() {
        let mut fragment = candidate(10, 21, EntityType::TaxId, 0.85);
        fragment.evidence.checksum = Some(true);
        let container = candidate(0, 50, EntityType::Credential, 0.80);

        let kept = Resolver::new().resolve(
            vec![fragment.clone(), container],
            &[EntityType::TaxId, EntityType::Credential],
            "auto",
        );
        assert_eq!(kept.len(), 1, "got {kept:?}");
        assert_eq!(kept[0].entity_type, EntityType::Credential);
        assert_eq!((kept[0].span.start, kept[0].span.end), (0, 50));

        // Merely overlapping, not contained: the checksum decides, exactly
        // as it did before the filter existed.
        let straddling = candidate(15, 60, EntityType::Credential, 0.80);
        let kept = Resolver::new().resolve(
            vec![fragment, straddling],
            &[EntityType::TaxId, EntityType::Credential],
            "auto",
        );
        assert_eq!(kept.len(), 1, "got {kept:?}");
        assert_eq!(kept[0].entity_type, EntityType::TaxId);
    }

    /// Two recognizers disagreeing about what one piece of text *is* are not
    /// container and fragment, and the filter must not touch them: identical
    /// spans stay in the competition, and the loser stays available to the
    /// equal-span restoration `resolve` documents.
    #[test]
    fn two_readings_of_the_same_span_are_still_decided_by_strength() {
        let mut checksummed = candidate(0, 9, EntityType::NationalId, 1.00);
        checksummed.evidence.checksum = Some(true);
        let other = candidate(0, 9, EntityType::PassportNumber, 0.70);

        let kept = Resolver::new().resolve(
            vec![checksummed, other.clone()],
            &[EntityType::NationalId, EntityType::PassportNumber],
            "auto",
        );
        assert_eq!(kept.len(), 1, "got {kept:?}");
        assert_eq!(kept[0].entity_type, EntityType::NationalId);

        // And the restoration still finds the loser when the winner's type
        // was not requested.
        let mut checksummed = candidate(0, 9, EntityType::NationalId, 1.00);
        checksummed.evidence.checksum = Some(true);
        let kept = Resolver::new().resolve(
            vec![checksummed, other],
            &[EntityType::PassportNumber],
            "auto",
        );
        assert_eq!(kept.len(), 1, "got {kept:?}");
        assert_eq!(kept[0].entity_type, EntityType::PassportNumber);
    }

    #[test]
    fn a_candidate_below_its_threshold_is_dropped() {
        let low = candidate(0, 9, EntityType::NationalId, 0.30);
        let kept = Resolver::new().resolve(vec![low], &[EntityType::NationalId], "auto");
        assert!(kept.is_empty());
    }

    /// The defect this file used to encode, from the direction it leaked.
    ///
    /// `nl_bsn`'s shape: format 0.4, a passing elfproef 0.5, its own label
    /// 0.1: a checksum-valid BSN introduced by the word `BSN`, with a
    /// negative-context term (`iban`, `in order to`) somewhere in the same
    /// clause. It used to be deleted before its threshold was ever
    /// consulted, and a real national ID went out unmasked.
    #[test]
    fn a_suppressed_candidate_with_a_checksum_and_its_own_label_survives() {
        let mut suppressed = candidate(0, 9, EntityType::NationalId, 1.00);
        suppressed.evidence.checksum = Some(true);
        suppressed.evidence.context = true;
        suppressed.evidence.negative_context = true;
        let kept = Resolver::new().resolve(vec![suppressed], &[EntityType::NationalId], "auto");
        assert_eq!(kept.len(), 1, "got {kept:?}");
    }

    /// The other half, and the half that keeps the mechanism worth having.
    /// `nl_bsn` without its own label is `0.4 + 0.5 = 0.90`: an order
    /// number whose digits happen to clear the elfproef. A passing checksum
    /// alone must not buy its way past a term that says "this is an order
    /// number", or every `must_clear` order number in the benchmark starts
    /// being masked.
    #[test]
    fn a_suppressed_candidate_carrying_only_a_checksum_is_still_dropped() {
        let mut suppressed = candidate(0, 9, EntityType::NationalId, 0.90);
        suppressed.evidence.checksum = Some(true);
        let kept_unsuppressed =
            Resolver::new().resolve(vec![suppressed.clone()], &[EntityType::NationalId], "auto");
        assert_eq!(
            kept_unsuppressed.len(),
            1,
            "the same candidate unsuppressed"
        );

        suppressed.evidence.negative_context = true;
        let kept = Resolver::new().resolve(vec![suppressed], &[EntityType::NationalId], "auto");
        assert!(kept.is_empty(), "got {kept:?}");
    }

    /// A candidate with no checksum at all: a version string beside an IP
    /// address, a tracking code read as a passport number. Nothing changed
    /// for these: the penalty carries them under their bar exactly as the
    /// delete did.
    #[test]
    fn a_suppressed_candidate_with_no_checksum_is_still_dropped() {
        let mut suppressed = candidate(0, 12, EntityType::IpAddress, 0.90);
        suppressed.evidence.negative_context = true;
        let kept = Resolver::new().resolve(vec![suppressed], &[EntityType::IpAddress], "auto");
        assert!(kept.is_empty(), "got {kept:?}");
    }

    /// The penalty is a fixed subtraction, and both edges of it are pinned
    /// here so a change to its size is a test failure rather than a quiet
    /// recalibration of every pack at once. `NATIONAL_ID` is 0.70 in
    /// `thresholds.toml`, so the break-even score for a suppressed
    /// candidate is 0.95.
    #[test]
    fn the_negative_context_penalty_is_the_size_it_says_it_is() {
        let suppressed = |score: f32| {
            let mut candidate = candidate(0, 9, EntityType::NationalId, score);
            candidate.evidence.checksum = Some(true);
            candidate.evidence.negative_context = true;
            Resolver::new()
                .resolve(vec![candidate], &[EntityType::NationalId], "auto")
                .len()
        };
        assert_eq!(suppressed(0.95), 1, "exactly at the penalised threshold");
        assert_eq!(suppressed(0.94), 0, "one cent below it");
        assert_eq!(suppressed(0.96), 1, "one cent above it");
    }

    /// The penalty decides whether a candidate is kept and nothing else. A
    /// kept candidate's score reaches the API unchanged, and the overlap
    /// order does not consult it either.
    #[test]
    fn a_surviving_suppressed_candidate_keeps_its_own_score() {
        let mut suppressed = candidate(0, 9, EntityType::NationalId, 1.00);
        suppressed.evidence.checksum = Some(true);
        suppressed.evidence.negative_context = true;
        let kept = Resolver::new().resolve(vec![suppressed], &[EntityType::NationalId], "auto");
        assert_eq!(kept.len(), 1, "got {kept:?}");
        assert_eq!(kept[0].score, 1.00);
    }

    #[test]
    fn a_type_that_was_not_requested_is_dropped() {
        let email = candidate(0, 16, EntityType::EmailAddress, 0.95);
        let kept = Resolver::new().resolve(vec![email], &[EntityType::NationalId], "auto");
        assert!(kept.is_empty());
    }

    #[test]
    fn a_type_that_was_not_requested_still_decides_the_overlap_it_belongs_to() {
        // The shape of `IBAN NL91 ABNA 0417 1643 00`: a checksum-validated
        // IBAN over the whole identifier, and a VAT candidate over a truncated
        // part of it, above its own threshold.
        let mut iban = candidate(5, 27, EntityType::IbanCode, 1.0);
        iban.evidence.checksum = Some(true);
        let vat = candidate(5, 19, EntityType::VatNumber, 0.85);

        // Requesting only VAT must not promote the fragment. The IBAN wins the
        // overlap and is then filtered out, so the answer is nothing: the
        // alternative is a truncated span labelled VAT_NUMBER, which leaves the
        // rest of the IBAN in the text the caller sends on.
        let vat_only = Resolver::new().resolve(
            vec![iban.clone(), vat.clone()],
            &[EntityType::VatNumber],
            "auto",
        );
        assert!(vat_only.is_empty(), "got {vat_only:?}");

        // Requesting both returns the whole IBAN, as it always did.
        let both = Resolver::new().resolve(
            vec![iban, vat],
            &[EntityType::IbanCode, EntityType::VatNumber],
            "auto",
        );
        assert_eq!(both.len(), 1);
        assert_eq!(both[0].entity_type, EntityType::IbanCode);
        assert_eq!(both[0].span, CharSpan { start: 5, end: 27 });
    }

    #[test]
    fn an_identical_span_a_winner_vacates_goes_to_the_requested_reading_of_it() {
        // The shape of `paspoortnummer 123456782`: a checksum-validated
        // NATIONAL_ID, and a PASSPORT_NUMBER over exactly the same characters
        // because the word before them is a passport context term.
        let mut national_id = candidate(15, 24, EntityType::NationalId, 0.9);
        national_id.evidence.checksum = Some(true);
        let passport = candidate(15, 24, EntityType::PassportNumber, 0.7);

        // Requesting only the passport: the national ID wins the overlap and
        // is filtered out, and the span goes to the reading the caller asked
        // about rather than to nobody. The two spans are the same characters,
        // so nothing is truncated and nothing is left in the text.
        let passport_only = Resolver::new().resolve(
            vec![national_id.clone(), passport.clone()],
            &[EntityType::PassportNumber],
            "auto",
        );
        assert_eq!(passport_only.len(), 1, "got {passport_only:?}");
        assert_eq!(passport_only[0].entity_type, EntityType::PassportNumber);
        assert_eq!(passport_only[0].span, CharSpan { start: 15, end: 24 });

        // Requesting both leaves the winner alone: nothing was vacated.
        let both = Resolver::new().resolve(
            vec![national_id, passport],
            &[EntityType::NationalId, EntityType::PassportNumber],
            "auto",
        );
        assert_eq!(both.len(), 1);
        assert_eq!(both[0].entity_type, EntityType::NationalId);
    }

    #[test]
    fn only_an_identical_span_is_restored_never_a_fragment_of_the_winner() {
        // A requested candidate one character short of the winner's span is
        // still a fragment: masking it would leave the rest of the winner's
        // characters in the text under a label that does not describe them.
        let mut winner = candidate(5, 27, EntityType::IbanCode, 1.0);
        winner.evidence.checksum = Some(true);
        let fragment = candidate(5, 26, EntityType::VatNumber, 0.85);

        let vat_only =
            Resolver::new().resolve(vec![winner, fragment], &[EntityType::VatNumber], "auto");
        assert!(vat_only.is_empty(), "got {vat_only:?}");
    }

    #[test]
    fn the_strongest_requested_reading_of_a_vacated_span_is_the_one_restored() {
        let mut winner = candidate(0, 9, EntityType::NationalId, 0.95);
        winner.evidence.checksum = Some(true);
        let weaker = candidate(0, 9, EntityType::PassportNumber, 0.75);
        let stronger = candidate(0, 9, EntityType::PhoneNumber, 0.9);

        // Both readings are requested and share the vacated span, so the
        // choice is made by the same `strength_order` the overlap used: no
        // checksum on either and the same length, so the higher score wins.
        let requested = &[EntityType::PassportNumber, EntityType::PhoneNumber];
        let kept = Resolver::new().resolve(
            vec![winner.clone(), weaker.clone(), stronger.clone()],
            requested,
            "auto",
        );
        assert_eq!(kept.len(), 1, "got {kept:?}");
        assert_eq!(kept[0].entity_type, EntityType::PhoneNumber);

        // And the input order does not decide it.
        let reordered = Resolver::new().resolve(vec![stronger, winner, weaker], requested, "auto");
        assert_eq!(reordered.len(), 1);
        assert_eq!(reordered[0].entity_type, EntityType::PhoneNumber);
    }

    #[test]
    fn a_candidate_an_earlier_stage_rejected_is_never_restored() {
        // Step 6 -- failed checksums, and the threshold the negative-context
        // penalty is weighed into -- runs before overlap resolution on
        // purpose. A vacated span must not be a way back in for anything it
        // rejected.
        let mut winner = candidate(0, 9, EntityType::NationalId, 0.95);
        winner.evidence.checksum = Some(true);

        // 0.70 against a 0.60 `PASSPORT_NUMBER` threshold: above the bar on
        // its own, under it once the negative-context term beside it is
        // weighed in, which is the only way this candidate is rejected at all.
        let mut suppressed = candidate(0, 9, EntityType::PassportNumber, 0.7);
        suppressed.evidence.negative_context = true;
        let mut failed_checksum = candidate(0, 9, EntityType::CreditCard, 0.9);
        failed_checksum.evidence.checksum = Some(false);
        let below_threshold = candidate(0, 9, EntityType::PhoneNumber, 0.3);

        let kept = Resolver::new().resolve(
            vec![winner, suppressed, failed_checksum, below_threshold],
            &[
                EntityType::PassportNumber,
                EntityType::CreditCard,
                EntityType::PhoneNumber,
            ],
            "auto",
        );
        assert!(kept.is_empty(), "got {kept:?}");
    }

    #[test]
    fn a_restored_candidate_takes_its_place_in_span_order_without_overlapping_a_kept_one() {
        let email = candidate(30, 46, EntityType::EmailAddress, 0.95);
        let mut national_id = candidate(15, 24, EntityType::NationalId, 0.9);
        national_id.evidence.checksum = Some(true);
        let passport = candidate(15, 24, EntityType::PassportNumber, 0.7);

        let kept = Resolver::new().resolve(
            vec![email, national_id, passport],
            &[EntityType::PassportNumber, EntityType::EmailAddress],
            "auto",
        );
        assert_eq!(kept.len(), 2, "got {kept:?}");
        assert_eq!(kept[0].entity_type, EntityType::PassportNumber);
        assert_eq!(kept[0].span, CharSpan { start: 15, end: 24 });
        assert_eq!(kept[1].entity_type, EntityType::EmailAddress);
        for pair in kept.windows(2) {
            assert!(
                pair[0].span.end <= pair[1].span.start,
                "results overlap: {kept:?}"
            );
        }
    }

    #[test]
    fn the_longer_span_wins_an_overlap() {
        let short = candidate(4, 12, EntityType::PhoneNumber, 0.9);
        let long = candidate(0, 16, EntityType::PhoneNumber, 0.9);
        let kept = Resolver::new().resolve(vec![short, long], &[EntityType::PhoneNumber], "auto");
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].span, CharSpan { start: 0, end: 16 });
    }

    #[test]
    fn a_validated_candidate_beats_an_unvalidated_one_on_the_same_span() {
        let mut validated = candidate(0, 9, EntityType::NationalId, 0.9);
        validated.evidence.checksum = Some(true);
        let generic = candidate(0, 9, EntityType::PassportNumber, 0.9);
        let kept = Resolver::new().resolve(
            vec![generic, validated],
            &[EntityType::NationalId, EntityType::PassportNumber],
            "auto",
        );
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].entity_type, EntityType::NationalId);
    }

    #[test]
    fn the_output_order_is_stable() {
        // Strength order and span order disagree here on purpose: `a` sits
        // later in the text but is stronger (higher score; same span length
        // and no checksum on either side), so the strength-descending sort
        // produces `[a, b]` before overlap resolution ever looks at spans.
        // Only the resolver's final positional re-sort turns that into span
        // order. If that re-sort were removed, `kept[0]` would be `a` and
        // this test would fail.
        let a = candidate(20, 30, EntityType::EmailAddress, 0.95);
        let b = candidate(0, 10, EntityType::EmailAddress, 0.85);
        let kept = Resolver::new().resolve(vec![a, b], &[EntityType::EmailAddress], "auto");
        assert_eq!(kept[0].span.start, 0);
        assert_eq!(kept[1].span.start, 20);
    }

    // -----------------------------------------------------------------
    // Calibration: model-sourced candidates are gated by Calibration, never
    // by thresholds.toml, and language actually reaches the lookup.
    // -----------------------------------------------------------------

    /// A minimal single-entry `Calibration`, built through `from_json`
    /// (the only constructor `Calibration` exposes) rather than a struct
    /// literal, so this helper exercises the same parse path production
    /// does. `entity_type`'s wire name comes from `EntityType`'s own
    /// `Serialize` impl rather than a second hand-written table here.
    fn calibration_with(entity_type: EntityType, language: &str, threshold: f32) -> Calibration {
        let wire = serde_json::to_value(entity_type).expect("EntityType serialises");
        let wire = wire
            .as_str()
            .expect("EntityType serialises to a string")
            .to_string();
        let raw = serde_json::json!({
            "revision": "test",
            "default": {},
            "languages": { language: { wire: threshold } }
        });
        Calibration::from_json(&raw.to_string()).expect("a well-formed calibration")
    }

    #[test]
    fn a_model_candidate_is_gated_by_calibration_not_by_thresholds_toml() {
        let resolver = Resolver::with_calibration(calibration_with(EntityType::Person, "de", 0.70));
        let mut weak = candidate(0, 12, EntityType::Person, 0.65);
        weak.source = Source::Model;
        let mut strong = weak.clone();
        strong.score = 0.75;
        assert!(
            resolver
                .resolve(vec![weak], &[EntityType::Person], "de")
                .is_empty()
        );
        assert_eq!(
            resolver
                .resolve(vec![strong], &[EntityType::Person], "de")
                .len(),
            1
        );
    }

    #[test]
    fn a_rule_candidate_still_uses_thresholds_toml_and_ignores_calibration() {
        // The structured-type gates pass today. This task must not be able to
        // move them, and a calibration entry for a rule type must therefore
        // have no effect at all.
        let resolver =
            Resolver::with_calibration(calibration_with(EntityType::NationalId, "de", 0.99));
        let mut rule = candidate(0, 9, EntityType::NationalId, 0.80); // above thresholds.toml's 0.70
        rule.source = Source::Rule { id: "nl_bsn" };
        assert_eq!(
            resolver
                .resolve(vec![rule], &[EntityType::NationalId], "de")
                .len(),
            1
        );
    }

    #[test]
    fn a_model_candidate_with_no_calibration_at_all_never_matches() {
        let resolver = Resolver::new(); // no calibration loaded
        let mut model = candidate(0, 12, EntityType::Person, 0.99);
        model.source = Source::Model;
        assert!(
            resolver
                .resolve(vec![model], &[EntityType::Person], "de")
                .is_empty()
        );
    }

    #[test]
    fn the_language_argument_actually_reaches_the_threshold_lookup() {
        // The mutation this catches: a resolve() that takes `language` and
        // never passes it on. Without this test, every per-language threshold
        // in calibration.json could be ignored and every other test here
        // would stay green.
        let resolver = Resolver::with_calibration(
            Calibration::from_json(
                r#"{
            "revision": "t", "default": { "PERSON": 0.10 },
            "languages": { "de": { "PERSON": 0.90 } }
        }"#,
            )
            .expect("parses"),
        );
        let mut model = candidate(0, 12, EntityType::Person, 0.50);
        model.source = Source::Model;
        assert_eq!(
            resolver
                .resolve(vec![model.clone()], &[EntityType::Person], "nl")
                .len(),
            1
        );
        assert!(
            resolver
                .resolve(vec![model], &[EntityType::Person], "de")
                .is_empty()
        );
    }

    // -----------------------------------------------------------------
    // The threshold comparison's boundary, at the resolution the generator
    // actually sweeps in (0.01 steps). The whole-table mutations above
    // prove the table is consulted at all; these prove the comparison at
    // the chosen value is the one the design means -- ">=", not ">" -- and
    // that a candidate one cent either side of the threshold is decided
    // correctly, not just candidates far enough away that any reasonable
    // comparison would agree.
    // -----------------------------------------------------------------

    #[test]
    fn a_candidate_exactly_at_the_threshold_is_kept() {
        let resolver = Resolver::with_calibration(calibration_with(EntityType::Person, "de", 0.70));
        let mut at = candidate(0, 12, EntityType::Person, 0.70);
        at.source = Source::Model;
        assert_eq!(
            resolver
                .resolve(vec![at], &[EntityType::Person], "de")
                .len(),
            1
        );
    }

    #[test]
    fn a_candidate_one_cent_below_the_threshold_is_dropped() {
        let resolver = Resolver::with_calibration(calibration_with(EntityType::Person, "de", 0.70));
        let mut below = candidate(0, 12, EntityType::Person, 0.69);
        below.source = Source::Model;
        assert!(
            resolver
                .resolve(vec![below], &[EntityType::Person], "de")
                .is_empty()
        );
    }

    #[test]
    fn a_candidate_one_cent_above_the_threshold_is_kept() {
        let resolver = Resolver::with_calibration(calibration_with(EntityType::Person, "de", 0.70));
        let mut above = candidate(0, 12, EntityType::Person, 0.71);
        above.source = Source::Model;
        assert_eq!(
            resolver
                .resolve(vec![above], &[EntityType::Person], "de")
                .len(),
            1
        );
    }

    // -----------------------------------------------------------------
    // Phone attribution: a company's number is cleared when nothing
    // personal stands beside it, and only when a model is loaded.
    // -----------------------------------------------------------------

    fn phone(company_nearby: bool, personal_signal: bool) -> Candidate {
        let mut found = candidate(60, 72, EntityType::PhoneNumber, 1.0);
        found.evidence.company_nearby = company_nearby;
        found.evidence.personal_signal = personal_signal;
        found.evidence.scope = Some(CharSpan { start: 0, end: 120 });
        found
    }

    fn person(start: u32, end: u32, score: f32) -> Candidate {
        Candidate {
            source: Source::Model,
            ..candidate(start, end, EntityType::Person, score)
        }
    }

    fn with_model() -> Resolver {
        Resolver::with_calibration(calibration_with(EntityType::Person, "nl", 0.70))
    }

    const PHONE: [EntityType; 2] = [EntityType::PhoneNumber, EntityType::Person];

    #[test]
    fn a_company_number_with_nobody_near_it_is_cleared_and_counted() {
        let resolved = with_model().resolve_reporting(vec![phone(true, false)], &PHONE, "nl");
        assert!(resolved.kept.is_empty());
        assert_eq!(resolved.attributed_to_company, 1);
    }

    #[test]
    fn a_personal_signal_keeps_it() {
        let resolved = with_model().resolve_reporting(vec![phone(true, true)], &PHONE, "nl");
        assert_eq!(resolved.kept.len(), 1);
        assert_eq!(resolved.attributed_to_company, 0);
    }

    #[test]
    fn no_company_keeps_it() {
        assert_eq!(
            with_model()
                .resolve(vec![phone(false, false)], &PHONE, "nl")
                .len(),
            1
        );
    }

    #[test]
    fn a_person_in_the_scope_keeps_it_even_below_the_threshold() {
        let kept =
            with_model().resolve(vec![phone(true, false), person(10, 24, 0.20)], &PHONE, "nl");
        assert!(
            kept.iter()
                .any(|found| found.entity_type == EntityType::PhoneNumber)
        );
    }

    #[test]
    fn a_person_outside_the_scope_does_not() {
        let kept = with_model().resolve(
            vec![phone(true, false), person(130, 144, 0.95)],
            &PHONE,
            "nl",
        );
        assert!(
            !kept
                .iter()
                .any(|found| found.entity_type == EntityType::PhoneNumber)
        );
    }

    #[test]
    fn without_a_model_nothing_is_cleared() {
        let resolved = Resolver::new().resolve_reporting(vec![phone(true, false)], &PHONE, "nl");
        assert_eq!(resolved.kept.len(), 1);
        assert_eq!(resolved.attributed_to_company, 0);
    }

    #[test]
    fn another_type_on_the_same_span_still_masks_it() {
        let identity = candidate(60, 72, EntityType::NationalId, 1.0);
        let kept = with_model().resolve(
            vec![phone(true, false), identity],
            &[EntityType::PhoneNumber, EntityType::NationalId],
            "nl",
        );
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].entity_type, EntityType::NationalId);
        assert_eq!((kept[0].span.start, kept[0].span.end), (60, 72));
    }

    #[test]
    fn only_a_phone_number_is_ever_attributed_to_a_company() {
        let mut email = candidate(60, 72, EntityType::EmailAddress, 1.0);
        email.evidence.company_nearby = true;
        email.evidence.personal_signal = false;
        email.evidence.scope = Some(CharSpan { start: 0, end: 120 });
        let resolved =
            with_model().resolve_reporting(vec![email], &[EntityType::EmailAddress], "nl");
        assert_eq!(resolved.kept.len(), 1);
        assert_eq!(resolved.attributed_to_company, 0);
    }

    #[test]
    fn a_phone_number_with_no_scope_is_kept() {
        let mut found = phone(true, false);
        found.evidence.scope = None;
        let resolved = with_model().resolve_reporting(vec![found], &PHONE, "nl");
        assert_eq!(resolved.kept.len(), 1);
        assert_eq!(resolved.attributed_to_company, 0);
    }

    #[test]
    fn a_person_overlapping_the_scopes_end_keeps_it() {
        let resolved = with_model().resolve_reporting(
            vec![phone(true, false), person(115, 125, 0.99)],
            &PHONE,
            "nl",
        );
        assert!(
            resolved
                .kept
                .iter()
                .any(|found| found.entity_type == EntityType::PhoneNumber)
        );
        assert_eq!(resolved.attributed_to_company, 0);
    }

    #[test]
    fn a_person_starting_exactly_at_the_windows_end_does_not_keep_it() {
        let resolved = with_model().resolve_reporting(
            vec![phone(true, false), person(120, 130, 0.99)],
            &PHONE,
            "nl",
        );
        assert!(
            !resolved
                .kept
                .iter()
                .any(|found| found.entity_type == EntityType::PhoneNumber)
        );
        assert_eq!(resolved.attributed_to_company, 1);
    }

    // A span this wide also strictly-contains the phone's, so `kept` holds the PERSON
    // reading, not a `PhoneNumber` one; this pins attribution's own decision instead.
    #[test]
    fn a_person_spanning_the_whole_window_is_not_attributed_to_a_company() {
        let resolved = with_model().resolve_reporting(
            vec![phone(true, false), person(0, 120, 0.99)],
            &PHONE,
            "nl",
        );
        assert_eq!(resolved.attributed_to_company, 0);
    }

    #[test]
    fn a_person_overlapping_the_phone_span_itself_keeps_it() {
        let resolved = with_model().resolve_reporting(
            vec![phone(true, false), person(62, 70, 0.99)],
            &PHONE,
            "nl",
        );
        assert!(
            resolved
                .kept
                .iter()
                .any(|found| found.entity_type == EntityType::PhoneNumber)
        );
        assert_eq!(resolved.attributed_to_company, 0);
    }
}
