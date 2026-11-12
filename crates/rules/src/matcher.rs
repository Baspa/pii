use crate::pack::{Recognizer, all};
use crate::validator::Validator;
use aho_corasick::{AhoCorasick, MatchKind};
use pii_detection_engine::candidate::{Candidate, Evidence, Source};
use pii_detection_engine::text::Document;
use regex::Regex;

/// How far from a match a context term still counts, in bytes.
///
/// Sixty-four is about a clause. Wider and "creditcard" at the start of a
/// paragraph lends confidence to a number at the end of it.
///
const CONTEXT_WINDOW: usize = 64;

struct Compiled {
    id: &'static str,
    recognizer: &'static Recognizer,
    patterns: Vec<Regex>,
    validator: Validator,
    context: Option<AhoCorasick>,
    negative_context: Option<AhoCorasick>,
}

pub struct Matcher {
    compiled: Vec<Compiled>,
    /// Built once. Constructing it parses region metadata, which is far too
    /// expensive to repeat per scan.
    phone: crate::phone::PhoneRecognizer,
}

impl Default for Matcher {
    fn default() -> Self {
        Self::new()
    }
}

impl Matcher {
    pub fn new() -> Self {
        let mut compiled = Vec::new();
        for pack in all() {
            for recognizer in &pack.recognizer {
                // A pattern that will not compile must not silently drop the
                // recognizer: that would leave a type undetected while the
                // workspace believes it is covered. Fail loudly at load,
                // matching `pack::all()`'s established panic-on-malformed-config
                // behaviour.
                let patterns = recognizer
                    .patterns
                    .iter()
                    .enumerate()
                    .map(|(index, pattern)| {
                        Regex::new(pattern).unwrap_or_else(|error| {
                            panic!(
                                "recognizer {} pattern {index} failed to compile: {error}",
                                recognizer.id
                            )
                        })
                    })
                    .collect();
                let terms: Vec<&str> = recognizer
                    .context_terms
                    .values()
                    .flatten()
                    .map(String::as_str)
                    .collect();
                // An unresolvable validator name must not silently degrade to
                // `Validator::None`, which accepts every candidate: a
                // checksum-guarded recognizer would start matching any string
                // of the right shape. Fail loudly at load instead.
                let validator = Validator::from_name(&recognizer.validator).unwrap_or_else(|| {
                    panic!(
                        "recognizer {} names an unknown validator: {}",
                        recognizer.id, recognizer.validator
                    )
                });
                compiled.push(Compiled {
                    // `all()` lives in a OnceLock, so its strings are 'static.
                    id: recognizer.id.as_str(),
                    recognizer,
                    patterns,
                    validator,
                    context: build_automaton(&terms),
                    negative_context: build_automaton(
                        &recognizer
                            .negative_context
                            .iter()
                            .map(String::as_str)
                            .collect::<Vec<_>>(),
                    ),
                });
            }
        }
        Self {
            compiled,
            phone: crate::phone::PhoneRecognizer::new(&crate::phone::DEFAULT_REGIONS),
        }
    }

    pub fn scan(&self, document: &Document) -> Vec<Candidate> {
        let text = document.as_str();
        let mut found = Vec::new();

        for compiled in &self.compiled {
            for pattern in &compiled.patterns {
                for hit in pattern.find_iter(text) {
                    let Some(span) = document.char_span(hit.start(), hit.end()) else {
                        continue;
                    };

                    let checksum = match compiled.validator {
                        Validator::None => None,
                        validator => Some(validator.accepts(hit.as_str())),
                    };

                    let window = neighbourhood(text, hit.start(), hit.end());
                    // Positive context stays a plain substring search on
                    // purpose. Dutch and German build compounds, so
                    // `Reisepassnummer` contains neither `reisepass` nor
                    // `passnummer` on a word boundary, and a passport is
                    // decided almost entirely by that nearby word. A term
                    // found where a boundary rule would not find one can only
                    // raise a score, and a raised score can only over-mask.
                    // The two directions are deliberately not held to the
                    // same rule; see `matches_whole_term`.
                    let context = compiled
                        .context
                        .as_ref()
                        .is_some_and(|automaton| automaton.is_match(&window.text));
                    let negative_context = compiled
                        .negative_context
                        .as_ref()
                        .is_some_and(|automaton| matches_whole_term(automaton, &window));

                    let mut score = compiled.recognizer.base_confidence;
                    if checksum == Some(true) {
                        score += compiled.recognizer.checksum_confidence;
                    }
                    if context {
                        score += compiled.recognizer.context_confidence;
                    }

                    found.push(Candidate {
                        span,
                        entity_type: compiled.recognizer.entity_type,
                        country: compiled.recognizer.country,
                        source: Source::Rule { id: compiled.id },
                        score: score.clamp(0.0, 1.0),
                        evidence: Evidence {
                            format: true,
                            checksum,
                            context,
                            negative_context,
                            ..Evidence::default()
                        },
                    });
                }
            }
        }

        found.extend(self.phone.scan(document));

        found.sort_by_key(|f| (f.span.start, f.span.end, f.entity_type));
        found
    }
}

/// `pub(crate)` so `phone` compiles its context terms exactly as a pack's
/// are compiled -- same case folding, same match kind.
pub(crate) fn build_automaton(terms: &[&str]) -> Option<AhoCorasick> {
    if terms.is_empty() {
        return None;
    }
    AhoCorasick::builder()
        .ascii_case_insensitive(true)
        // `Standard` rather than `LeftmostFirst` because
        // `matches_whole_term` needs an overlapping search, which
        // aho-corasick only offers under this kind. It also removes a way to
        // miss a suppression: under a non-overlapping search a hit rejected
        // for sitting inside a longer word consumes the text it covers, and
        // could hide a second, legitimate occurrence of another term inside
        // that same stretch.
        .match_kind(MatchKind::Standard)
        .build(terms)
        .ok()
}

/// Whether a term matches standing on its own, rather than buried inside a
/// longer word.
///
/// `AhoCorasick::is_match` was wrong for negative context: it reports any
/// substring occurrence, so the bare term `order` fired on `border`,
/// `reorder` and `recorder`, and `v.` fired on every Dutch `B.V.` The
/// resolver deletes a suppressed candidate unconditionally, ahead of the
/// threshold filter and ahead of a passing checksum, so a substring hit
/// silently under-masks: real personal data passes the firewall and reaches
/// the caller's model.
///
/// A hit counts only when the character on each side is not alphanumeric.
/// Keyed on alphanumerics rather than on ASCII word characters so non-ASCII
/// terms (`numéro de carte`, `bürgerservicenummer`) obey the same rule, and
/// so terms that begin or end in punctuation (`v.`, `ust-idnr`) still match
/// where they should.
///
/// At an edge of the window the neighbouring character is unknown, and the
/// window's own edges decide: only a true end of the text counts as a
/// boundary. Treating a clipped edge as one would let the clip invent
/// evidence: a window that begins `order …` because the text said
/// `reorder …` would suppress on nothing.
///
/// `pub(crate)` for a second caller with the same problem in the positive
/// direction: `phone`'s abbreviations (`tel`, `cell`, `tfno`) are three and
/// four characters long, so a substring search would read them out of `hotel`,
/// `excellent` and `intelligente`. Same rule, same window, one implementation.
pub(crate) fn matches_whole_term(automaton: &AhoCorasick, window: &Window) -> bool {
    automaton.find_overlapping_iter(&window.text).any(|hit| {
        let before = window.text[..hit.start()].chars().next_back();
        let after = window.text[hit.end()..].chars().next();
        let left = before.map_or(window.at_text_start, |c| !c.is_alphanumeric());
        let right = after.map_or(window.at_text_end, |c| !c.is_alphanumeric());
        left && right
    })
}

/// The lowercased text around a match, and whether each of its edges is the
/// edge of the text itself rather than a clip.
pub(crate) struct Window {
    pub(crate) text: String,
    at_text_start: bool,
    at_text_end: bool,
}

/// The text around a match, clipped to character boundaries.
///
/// `pub(crate)` so `phone` can ask the same question of the same window
/// rather than growing a second, subtly different notion of nearby.
pub(crate) fn neighbourhood(text: &str, start: usize, end: usize) -> Window {
    let mut from = start.saturating_sub(CONTEXT_WINDOW);
    while from < text.len() && !text.is_char_boundary(from) {
        from += 1;
    }
    let mut to = (end + CONTEXT_WINDOW).min(text.len());
    while to > 0 && !text.is_char_boundary(to) {
        to -= 1;
    }
    Window {
        text: text[from..to].to_lowercase(),
        at_text_start: from == 0,
        at_text_end: to == text.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pii_detection_engine::entity::EntityType;
    use pii_detection_engine::text::Document;

    fn scan(text: &str) -> Vec<pii_detection_engine::candidate::Candidate> {
        Matcher::new().scan(&Document::new(text.to_string()))
    }

    #[test]
    fn an_email_is_found_with_code_point_offsets() {
        let document = Document::new("Schrijf naar jane@example.com vandaag".to_string());
        let found = Matcher::new().scan(&document);
        let email = found
            .iter()
            .find(|c| c.entity_type == EntityType::EmailAddress)
            .expect("email");
        assert_eq!(document.slice(email.span), "jane@example.com");
    }

    #[test]
    fn offsets_survive_a_multibyte_prefix() {
        let document = Document::new("Grüße an jane@example.com".to_string());
        let found = Matcher::new().scan(&document);
        let email = found
            .iter()
            .find(|c| c.entity_type == EntityType::EmailAddress)
            .expect("email");
        assert_eq!(document.slice(email.span), "jane@example.com");
    }

    #[test]
    fn a_failed_checksum_lowers_the_score_but_does_not_delete_the_candidate() {
        let found = scan("IBAN NL91ABNA0417164301");
        let iban = found
            .iter()
            .find(|c| c.entity_type == EntityType::IbanCode)
            .expect("iban");
        assert_eq!(iban.evidence.checksum, Some(false));
        assert!(iban.score < 0.6, "score was {}", iban.score);
    }

    #[test]
    fn a_passing_checksum_raises_the_score() {
        let found = scan("IBAN NL91ABNA0417164300");
        let iban = found
            .iter()
            .find(|c| c.entity_type == EntityType::IbanCode)
            .expect("iban");
        assert_eq!(iban.evidence.checksum, Some(true));
        assert!(iban.score > 0.9, "score was {}", iban.score);
    }

    #[test]
    fn a_negative_context_term_marks_the_candidate() {
        let found = scan("Ordernummer 4111111111111111 is verzonden");
        let card = found
            .iter()
            .find(|c| c.entity_type == EntityType::CreditCard)
            .expect("card");
        assert!(card.evidence.negative_context);
    }

    #[test]
    fn a_negative_term_inside_a_longer_word_does_not_suppress() {
        // `invoice` is a negative context term for cards. Buried in
        // `invoices` it is not the label it stands for, and a candidate
        // marked here is deleted by the resolver ahead of its threshold and
        // ahead of its checksum: silently, and in the under-masking
        // direction.
        let embedded = scan("Prepaid invoices aside, 4111111111111111 was charged");
        let card = |found: &[pii_detection_engine::candidate::Candidate]| {
            found
                .iter()
                .find(|c| c.entity_type == EntityType::CreditCard)
                .expect("card")
                .evidence
                .negative_context
        };
        assert!(
            !card(&embedded),
            "a term inside a longer word must not suppress"
        );

        let alone = scan("Invoice 4111111111111111 is overdue");
        assert!(card(&alone), "the same term standing alone must suppress");
    }

    #[test]
    fn a_term_the_window_edge_cut_out_of_a_word_does_not_suppress() {
        // The window is cut at a fixed byte distance, which can land inside a
        // word and leave its tail at the edge with nothing before it. Both
        // texts here place the negative term `factuurnummer` exactly at the
        // start of the window; they differ only in whether the clip cut a
        // longer word to put it there.
        let suppressed = |text: &str| {
            scan(text)
                .iter()
                .find(|c| c.entity_type == EntityType::CreditCard)
                .expect("card")
                .evidence
                .negative_context
        };
        let term = "factuurnummer";
        let padding = " ".repeat(CONTEXT_WINDOW - term.len());

        // `Verzamelfactuurnummer` is one word: the clip lands on the `f`, and
        // the tail it exposes is evidence the clip invented.
        assert!(!suppressed(&format!(
            "Verzamel{term}{padding}4111111111111111"
        )));

        // The same geometry with the term genuinely standing alone: the
        // window starts at the start of the text, so nothing was cut.
        assert!(suppressed(&format!("{term}{padding}4111111111111111")));
    }

    #[test]
    fn a_positive_context_term_raises_the_score() {
        let with_context = scan("Mijn creditcard 4111111111111111");
        let without = scan("4111111111111111");
        let scored = |found: &[pii_detection_engine::candidate::Candidate]| {
            found
                .iter()
                .find(|c| c.entity_type == EntityType::CreditCard)
                .expect("card")
                .score
        };
        assert!(scored(&with_context) > scored(&without));
    }
}
