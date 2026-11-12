use pii_detection_engine::entity::EntityType;
use pii_detection_engine::text::Document;
use pii_detection_resolve::Resolver;
use pii_detection_rules::matcher::Matcher;

/// The requested-type filter runs after overlap resolution (the design,
/// steps 7 and 8), and this is the text that shows why the order matters. The
/// VAT pattern matches the sub-span `NL91 ABNA 0417` inside a real IBAN.
///
/// Filtering first, a workspace with VAT_NUMBER enabled and IBAN_CODE not
/// would lose the IBAN candidate before the overlap was decided and receive a
/// truncated span labelled VAT_NUMBER: the pseudonymiser would replace the
/// first fourteen characters and send `1643 00` on to the model. Filtering
/// last it receives nothing, which is the design's answer and the right one.
///
/// Two independent mechanisms now stop this text: the ordering here, and the
/// VAT recognizer's base confidence, which no longer clears its threshold on
/// shape alone. `pii-detection-resolve` has the unit test that isolates the ordering
/// on its own.
#[test]
fn a_vat_only_workspace_gets_nothing_rather_than_a_fragment_of_an_iban() {
    let matcher = Matcher::new();
    let resolver = Resolver::new();
    let document = Document::new("IBAN NL91 ABNA 0417 1643 00".to_string());

    let vat_only = resolver.resolve(matcher.scan(&document), &[EntityType::VatNumber], "auto");
    assert!(
        vat_only.is_empty(),
        "a VAT-only workspace received {:?}",
        vat_only
            .iter()
            .map(|c| (c.entity_type, document.slice(c.span)))
            .collect::<Vec<_>>()
    );

    let both = resolver.resolve(
        matcher.scan(&document),
        &[EntityType::VatNumber, EntityType::IbanCode],
        "auto",
    );
    assert_eq!(both.len(), 1);
    assert_eq!(both[0].entity_type, EntityType::IbanCode);
    assert_eq!(document.slice(both[0].span), "NL91 ABNA 0417 1643 00");
}

/// The other direction of the same ordering, and the reason the resolver
/// restores an identical span. `paspoortnummer 123456782` raises NATIONAL_ID
/// over the nine digits (`nl_bsn`, 11-proef passing) and PASSPORT_NUMBER
/// over exactly the same nine digits, because `paspoortnummer` is a context
/// term for the universal passport recognizer.
///
/// The BSN wins the overlap on its checksum and the filter then removes it. A
/// workspace that explicitly enabled passport detection must not be handed an
/// empty list for that: the two recognizers disagree about what one string of
/// digits *is*, and the caller asked about one of the two readings. Both spans
/// are the same characters, so answering with the passport reading masks
/// exactly what the BSN reading would have masked.
#[test]
fn a_passport_only_workspace_gets_the_passport_reading_of_a_span_the_bsn_won() {
    let matcher = Matcher::new();
    let resolver = Resolver::new();
    let document = Document::new("paspoortnummer 123456782".to_string());

    let passport_only = resolver.resolve(
        matcher.scan(&document),
        &[EntityType::PassportNumber],
        "auto",
    );
    assert_eq!(
        passport_only.len(),
        1,
        "a passport-only workspace received {:?}",
        passport_only
            .iter()
            .map(|c| (c.entity_type, document.slice(c.span)))
            .collect::<Vec<_>>()
    );
    assert_eq!(passport_only[0].entity_type, EntityType::PassportNumber);
    assert_eq!(document.slice(passport_only[0].span), "123456782");

    // With the national ID requested too, the checksum-validated reading is
    // the answer and the passport reading stays displaced.
    let both = resolver.resolve(
        matcher.scan(&document),
        &[EntityType::PassportNumber, EntityType::NationalId],
        "auto",
    );
    assert_eq!(both.len(), 1);
    assert_eq!(both[0].entity_type, EntityType::NationalId);
    assert_eq!(document.slice(both[0].span), "123456782");
}

/// The same fragment, on a text where it actually reaches overlap resolution.
///
/// The test above cannot fail if the restoration rule is ever widened from
/// identical spans to overlapping ones: on `IBAN …` the VAT candidate scores
/// below its threshold and dies two stages before the overlap. Here the
/// leading `btw` is a VAT context term, so the candidate clears its threshold,
/// loses the overlap to the mod-97 IBAN, and lands in the pool a widened rule
/// would restore from. A VAT-only workspace must still receive nothing: the
/// VAT span is `NL91 ABNA 0417` and the winner's is the whole IBAN, so
/// returning it would mask fourteen characters, label them VAT_NUMBER and send
/// `1643 00` on to the model.
#[test]
fn a_vat_candidate_that_reaches_the_overlap_still_loses_its_fragment() {
    let matcher = Matcher::new();
    let resolver = Resolver::new();
    let document = Document::new("btw NL91 ABNA 0417 1643 00".to_string());
    let candidates = matcher.scan(&document);

    // The premise, asserted rather than assumed: a VAT candidate exists over
    // the sub-span and is strong enough to survive on its own. Without this,
    // a pack change that drops its score below threshold would leave the test
    // passing while it quietly stopped testing anything.
    let vat = candidates
        .iter()
        .find(|c| c.entity_type == EntityType::VatNumber)
        .cloned()
        .expect("the VAT pattern matches the sub-span");
    assert_eq!(document.slice(vat.span), "NL91 ABNA 0417");
    assert_eq!(
        resolver
            .resolve(vec![vat], &[EntityType::VatNumber], "auto")
            .len(),
        1,
        "the VAT candidate no longer clears its threshold, so this text no \
         longer exercises overlap resolution"
    );

    let vat_only = resolver.resolve(candidates.clone(), &[EntityType::VatNumber], "auto");
    assert!(
        vat_only.is_empty(),
        "a VAT-only workspace received {:?}",
        vat_only
            .iter()
            .map(|c| (c.entity_type, document.slice(c.span)))
            .collect::<Vec<_>>()
    );

    // And requesting both still returns the whole IBAN.
    let both = resolver.resolve(
        candidates,
        &[EntityType::VatNumber, EntityType::IbanCode],
        "auto",
    );
    assert_eq!(both.len(), 1);
    assert_eq!(both[0].entity_type, EntityType::IbanCode);
    assert_eq!(document.slice(both[0].span), "NL91 ABNA 0417 1643 00");
}
