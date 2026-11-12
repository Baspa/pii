use pii_detection_engine::entity::EntityType;
use pii_detection_engine::text::Document;
use pii_detection_resolve::{Resolver, has_threshold};
use pii_detection_rules::matcher::Matcher;
use pii_detection_rules::pack::all;
use std::collections::BTreeSet;

/// The resolver fails an uncalibrated type closed at runtime
/// (`thresholds.toml` has no entry, so it gets `f32::INFINITY` and never
/// matches). That protects production, but a type nobody calibrated should
/// stop the build, not quietly stop matching, so every type any loaded pack
/// declares, plus `PhoneNumber` (which lives outside the pack structure),
/// must have a real threshold entry.
#[test]
fn every_active_entity_type_has_a_calibrated_threshold() {
    let mut active: BTreeSet<EntityType> = all()
        .iter()
        .flat_map(|pack| {
            pack.recognizer
                .iter()
                .map(|recognizer| recognizer.entity_type)
        })
        .collect();
    active.insert(EntityType::PhoneNumber);

    for entity_type in active {
        assert!(
            has_threshold(entity_type),
            "{entity_type:?} has no calibrated threshold in pii-detection-resolve/thresholds.toml"
        );
    }
}

/// VAT is the one recognizer decided by shape and a nearby word together:
/// 0.55 for the format against a 0.70 threshold, so `context_confidence` has
/// to carry it over the bar.
///
/// Both halves matter. When the format cleared the threshold alone the guard
/// was dead and the recognizer returned fragments of ordinary uppercase prose:
/// its pattern reads "PLEASE SEE" and "DECLARATION OF" as VAT numbers. If
/// the two together could not reach it, VAT detection would be off entirely
/// and nothing would say so.
#[test]
fn a_vat_number_needs_a_context_term_to_clear_its_threshold() {
    let matcher = Matcher::new();
    let resolver = Resolver::new();
    let vat = |text: &str| {
        let document = Document::new(text.to_string());
        resolver
            .resolve(matcher.scan(&document), &[EntityType::VatNumber], "auto")
            .len()
    };

    assert_eq!(vat("NL123456789B01"), 0, "shape alone must not decide");
    assert_eq!(
        vat("BTW-nummer NL123456789B01"),
        1,
        "shape plus a term must"
    );
}

/// Phone numbers are the second recognizer decided by shape and a nearby word
/// together, and the only one whose shape evidence is graded rather than
/// binary. `thresholds.toml` puts `PHONE_NUMBER` at 0.70; the weights in
/// `phone.rs` are calibrated against that number and nothing else states the
/// relationship, so it is pinned here through the real resolver rather than
/// through a constant restated in a unit test.
///
/// Each row is one rung of the ladder. If a change collapses the ladder --
/// validity alone clearing the bar again, or the top rung falling below it --
/// exactly one of these fails and says which.
///
/// This is the plain bar. Since `df77acb` the ladder is calibrated against a
/// second one -- the threshold plus what a negative-context term costs -- and
/// that half is pinned where the constant it restates can be seen
/// (`phone::tests::the_penalty_this_module_restates_is_the_one_the_resolver_applies`)
/// and, over real sentences, by `tests/suppression.rs`.
#[test]
fn a_phone_number_is_decided_by_how_it_is_written_not_by_validity_alone() {
    let matcher = Matcher::new();
    let resolver = Resolver::new();
    let phone = |text: &str| {
        let document = Document::new(text.to_string());
        resolver
            .resolve(matcher.scan(&document), &[EntityType::PhoneNumber], "auto")
            .len()
    };

    // Valid in France, valid in Germany, valid in the Netherlands. None of
    // them is written as a telephone number.
    assert_eq!(phone("123456789"), 0, "validity alone must not decide");
    assert_eq!(phone("9876543"), 0, "validity alone must not decide");
    assert_eq!(phone("65929970001"), 0, "validity alone must not decide");

    // The same digits with a word that says telephone.
    assert_eq!(phone("Telefoon 123456789"), 1, "a term must carry it over");

    // The two written forms that carry a number on their own.
    assert_eq!(
        phone("+31 20 123 4567"),
        1,
        "international form must decide"
    );
    assert_eq!(phone("020 123 4567"), 1, "national form must decide");
}
