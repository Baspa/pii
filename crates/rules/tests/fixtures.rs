use pii_detection_engine::text::Document;
use pii_detection_resolve::Resolver;
use pii_detection_rules::matcher::Matcher;
use pii_detection_rules::pack::all;

/// Every fixture in every pack, run through the real matcher.
///
/// This is the test that makes a pack a contract rather than a description.
#[test]
fn positive_fixtures_are_found() {
    let matcher = Matcher::new();
    for pack in all() {
        for recognizer in &pack.recognizer {
            for fixture in &recognizer.positive {
                let document = Document::new(fixture.clone());
                let found = matcher.scan(&document);
                assert!(
                    found
                        .iter()
                        .any(|c| c.entity_type == recognizer.entity_type),
                    "{}: positive fixture not found",
                    recognizer.id
                );
            }
        }
    }
}

/// A raised candidate is not enough on its own: a fixture that is raised and
/// then suppressed is not detected in any sense the caller can observe.
/// Fixtures whose whole point is a negative-context term that must not decide
/// them (a BSN in a sentence containing the word `order`, an address beside
/// a Dutch `B.V.`) only lock that regression if a test looks past the
/// matcher.
///
/// The property asserted is the one that survives the resolver's own
/// calibration: **a negative-context term must not change what the resolver
/// returns for a positive fixture.** Each fixture is resolved twice, once as
/// scanned and once with `negative_context` cleared on every candidate, and
/// the two verdicts must agree.
///
/// Stated that way rather than as "the fixture must survive" because some
/// positive fixtures are *not* meant to survive the resolver alone: `vat`'s
/// are 0.55 against a 0.70 threshold and need a nearby word to clear it (see
/// `thresholds.rs`). Comparing the two runs asks only about suppression, which
/// is the thing this test is for, and it needs no copy of the threshold table
/// or of the penalty to do it.
#[test]
fn a_negative_context_term_does_not_decide_a_positive_fixture() {
    let matcher = Matcher::new();
    let resolver = Resolver::new();
    for pack in all() {
        for recognizer in &pack.recognizer {
            for fixture in &recognizer.positive {
                let document = Document::new(fixture.clone());
                let scanned = matcher.scan(&document);
                let unsuppressed: Vec<_> = scanned
                    .iter()
                    .cloned()
                    .map(|mut candidate| {
                        candidate.evidence.negative_context = false;
                        candidate
                    })
                    .collect();
                let spans = |candidates: Vec<pii_detection_engine::candidate::Candidate>| {
                    resolver
                        .resolve(candidates, &[recognizer.entity_type], "auto")
                        .into_iter()
                        .map(|c| (c.span.start, c.span.end, c.entity_type))
                        .collect::<Vec<_>>()
                };
                assert_eq!(
                    spans(scanned),
                    spans(unsuppressed),
                    "{}: a negative-context term changed the verdict on a positive fixture",
                    recognizer.id
                );
            }
        }
    }
}

#[test]
fn negative_fixtures_are_not_found() {
    let matcher = Matcher::new();
    for pack in all() {
        for recognizer in &pack.recognizer {
            for fixture in &recognizer.negative {
                let document = Document::new(fixture.clone());
                let found = matcher.scan(&document);
                let matched: Vec<_> = found
                    .iter()
                    .filter(|c| {
                        c.entity_type == recognizer.entity_type
                            && c.evidence.checksum != Some(false)
                    })
                    .collect();
                assert!(
                    matched.is_empty(),
                    "{}: negative fixture matched",
                    recognizer.id
                );
            }
        }
    }
}
