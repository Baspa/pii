//! Negative context is weighed, not obeyed.
//!
//! The resolver used to delete a suppressed candidate outright, ahead of its
//! threshold and ahead of its checksum. Three independent readings found the
//! same hole from three directions, and each recorded it rather than fixing
//! it because it was found from inside a measuring task:
//!
//! 1. `in order to` deleted a checksum-validated Dutch BSN. The workaround
//!    was to take the bare term `order` out of `nl_bsn`'s list, which left
//!    a real `Order 123456782 shipped` unsuppressed.
//! 2. The phone recalibration declined to add negative-context terms it
//!    wanted, because a delete it could not outweigh would have traded
//!    over-masking for silent leaks.
//! 3. `de_steuer_id` lists `iban`, so a checksum-valid German Steuer-ID was
//!    deleted whenever a bank account sat in the same paragraph. The
//!    benchmark's `domain-nl-support-1` is the Dutch version of the same
//!    sentence and is a live miss: `IBAN: …` on one line, `BSN: …` on the
//!    next.
//!
//! Each of the three has a test here, run through the real packs, the real
//! matcher and the real `thresholds.toml`. Each is paired with the case that
//! must still be suppressed, because a resolver that kept everything would
//! pass the first half of every one of them.
//!
//! NO TEXT FROM THESE CASES REACHES A LOG OR AN ASSERTION MESSAGE: the
//! failure messages carry ids, types and spans, the same discipline the
//! rest of this crate's tests hold.

use pii_detection_engine::candidate::Candidate;
use pii_detection_engine::entity::EntityType;
use pii_detection_engine::text::Document;
use pii_detection_resolve::Resolver;
use pii_detection_rules::matcher::Matcher;
use pii_detection_rules::pack::all;

/// What the whole pipeline returns for one type over one text.
fn resolved(text: &str, entity_type: EntityType) -> Vec<Candidate> {
    let document = Document::new(text.to_string());
    Resolver::new().resolve(Matcher::new().scan(&document), &[entity_type], "auto")
}

/// The same, with `negative_context` forced on every candidate: the shape a
/// suppression term the packs do *not* carry would produce if it were added.
fn resolved_as_if_suppressed(text: &str, entity_type: EntityType) -> Vec<Candidate> {
    let document = Document::new(text.to_string());
    let suppressed: Vec<Candidate> = Matcher::new()
        .scan(&document)
        .into_iter()
        .map(|mut candidate| {
            if candidate.entity_type == entity_type {
                candidate.evidence.negative_context = true;
            }
            candidate
        })
        .collect();
    Resolver::new().resolve(suppressed, &[entity_type], "auto")
}

// ---------------------------------------------------------------------
// 1. A checksum-validated BSN beside a word that is not about it.
// ---------------------------------------------------------------------

/// `order` is back in `nl_bsn`'s negative context (see `packs/nl.toml`), so
/// these texts are genuinely suppressed at the matcher and genuinely
/// survive at the resolver. Without the term restored this test would assert
/// nothing at all.
#[test]
fn a_checksum_valid_bsn_survives_the_word_order_standing_beside_it() {
    let text = "In order to process this, my BSN is 123456782";
    let document = Document::new(text.to_string());
    assert!(
        Matcher::new()
            .scan(&document)
            .iter()
            .any(|c| c.entity_type == EntityType::NationalId && c.evidence.negative_context),
        "the term under test is not in the pack: this test would prove nothing"
    );

    let kept = resolved(text, EntityType::NationalId);
    assert_eq!(kept.len(), 1, "a validated BSN was suppressed: {kept:?}");
    assert_eq!(document.slice(kept[0].span).len(), 9);
}

/// `domain-nl-support-1` from the benchmark, reduced to the two lines that
/// matter. This is the live leak: an `iban` term one line above a
/// checksum-valid BSN that the caller labelled `BSN:` themselves.
#[test]
fn a_checksum_valid_bsn_survives_an_iban_on_the_line_above_it() {
    let text = "IBAN: NL90BNKA3422812463\nBSN: 804985492 (nodig voor de identiteitscontrole)";
    let document = Document::new(text.to_string());
    assert!(
        Matcher::new()
            .scan(&document)
            .iter()
            .any(|c| c.entity_type == EntityType::NationalId && c.evidence.negative_context),
        "the term under test is not in the pack: this test would prove nothing"
    );

    let kept = resolved(text, EntityType::NationalId);
    assert_eq!(kept.len(), 1, "a validated BSN was suppressed: {kept:?}");
    assert_eq!(document.slice(kept[0].span).len(), 9);
}

/// The half that keeps the mechanism worth having. `938570833` clears the
/// elfproef and is an order number, which is why the benchmark carries it as
/// a `must_clear` value. A checksum on its own must not buy past the label.
#[test]
fn an_order_number_that_passes_the_elfproef_is_still_suppressed() {
    for text in [
        "Mijn ordernummer is 938570833, besteld via de webshop afgelopen dinsdag.",
        "Order 123456782 shipped",
        "Factuurnummer 123456782 is voldaan",
    ] {
        let kept = resolved(text, EntityType::NationalId);
        assert!(kept.is_empty(), "an order number was masked: {kept:?}");
    }
}

// ---------------------------------------------------------------------
// 2. A checksum-validated Steuer-ID beside an IBAN.
// ---------------------------------------------------------------------

#[test]
fn a_checksum_valid_steuer_id_survives_an_iban_in_the_same_sentence() {
    let text = "Meine Steuer-ID lautet 86095742719, IBAN DE89370400440532013000.";
    let document = Document::new(text.to_string());
    assert!(
        Matcher::new()
            .scan(&document)
            .iter()
            .any(|c| c.entity_type == EntityType::TaxId && c.evidence.negative_context),
        "the term under test is not in the pack: this test would prove nothing"
    );

    let kept = resolved(text, EntityType::TaxId);
    assert_eq!(
        kept.len(),
        1,
        "a validated Steuer-ID was suppressed: {kept:?}"
    );
    assert_eq!(document.slice(kept[0].span).len(), 11);
}

/// `26791488099` clears the German tax-id check digit and is an invoice
/// number: `neg-de-checksum-order-1` in the benchmark, a `must_clear`.
#[test]
fn an_invoice_number_that_passes_the_tax_id_check_is_still_suppressed() {
    for text in [
        "Die Rechnungsnummer lautet 26791488099, bestellt letzte Woche im Online-Shop.",
        "Rechnungsnummer 86095742719 vom Januar",
    ] {
        let kept = resolved(text, EntityType::TaxId);
        assert!(kept.is_empty(), "an invoice number was masked: {kept:?}");
    }
}

/// The third value the penalty was sized around, and the one no test named.
/// `4111117074292632` clears Luhn and is an order number --
/// `neg-en-checksum-order-1` in the benchmark, a `must_clear`. Its pair is
/// the same recognizer's own label, which must still carry a real card over.
#[test]
fn an_order_number_that_passes_luhn_is_still_suppressed() {
    for text in [
        "Order number 4111117074292632 has shipped",
        "Factuurnummer 4111117074292632 is voldaan",
    ] {
        let document = Document::new(text.to_string());
        assert!(
            Matcher::new()
                .scan(&document)
                .iter()
                .any(|c| c.entity_type == EntityType::CreditCard && c.evidence.negative_context),
            "the term under test is not in the pack: this test would prove nothing"
        );
        let kept = resolved(text, EntityType::CreditCard);
        assert!(kept.is_empty(), "an order number was masked: {kept:?}");
    }
    let text = "Card number 4111111111111111 on file";
    let document = Document::new(text.to_string());
    let kept = resolved(text, EntityType::CreditCard);
    assert_eq!(kept.len(), 1, "a real card was not masked: {kept:?}");
    assert_eq!(document.slice(kept[0].span).len(), 16);
}

// ---------------------------------------------------------------------
// 3. The phone terms the recalibration declined to add.
// ---------------------------------------------------------------------

/// `training/reports/phone-calibration.md` names two sentences as the reason
/// no negative-context list was given to the phone recognizer: a real number
/// in a sentence *about* an invoice. **Neither term is added here and the
/// recognizer's list is still empty** -- what this asks is what would happen
/// to those sentences if one were, by marking the candidates the way a term
/// would mark them and handing them to the real resolver.
///
/// When this test was written the answer was half a no: `030 12345678`
/// reached 1.0 and survived, `020-1234567` reached 0.92 against a penalised
/// bar of 0.95 and did not -- a real Dutch telephone number argued off the
/// mask by the word `factuur`. `training/reports/phone-ladder.md` is the
/// recalibration that closed it; every row below now survives, including the
/// two trunkless languages, whose nearby word is the only evidence they have.
///
/// The other half of the mechanism is
/// `a_reference_number_stays_out_whether_or_not_a_term_suppresses_it` below:
/// a resolver that kept everything would pass this test on its own.
#[test]
fn a_number_written_as_a_number_outweighs_the_terms_the_recalibration_declined() {
    for text in [
        // The two sentences the calibration report named.
        "Rechnungsnummer 4471: Rückruf unter 030 12345678",
        "Vragen over de factuur? Bel 020-1234567",
        // Written form alone, no word that says telephone: the rung that
        // used to fall, in the two Dutch forms it falls in.
        "Factuur voldaan? Bel 020-1234567",
        "Bestelling verzonden, bel 0612345678",
        // Trunkless plans, where the nearby word is the whole of the
        // recognizer and a penalty therefore lands hardest. Written from the
        // numbering plans, not from any fixture -- the benchmark has no
        // Spanish or Italian telephone number at all.
        //
        // One number per sentence on purpose: the context window is 64
        // bytes and does not ask which number a word was written about, so
        // an invoice reference sharing a window with `Teléfono` is scored as
        // though the word labelled it. That is this recognizer's oldest
        // known cost (`training/reports/phone-es-it.md`), it is unchanged by
        // the recalibration -- 0.75 and 0.95 are both over the bar -- and it
        // is not what these rows are about.
        "Su factura está pendiente. Teléfono de contacto: 612345678",
        "Pedido enviado. Llámanos al 612 34 56 78",
        "La fattura è in scadenza. Chiamami al 3401234567",
        "Ordine spedito. Cellulare 340 123 4567",
    ] {
        assert_eq!(
            resolved(text, EntityType::PhoneNumber).len(),
            1,
            "the number must be found while no term suppresses it"
        );
        assert_eq!(
            resolved_as_if_suppressed(text, EntityType::PhoneNumber).len(),
            1,
            "a real telephone number was argued off the mask by one term"
        );
    }
}

/// And the reference numbers stay out either way: the recognizer's own
/// written-form rejection does that work, not suppression.
#[test]
fn a_reference_number_stays_out_whether_or_not_a_term_suppresses_it() {
    for text in [
        "Factuur 123456789 en order 987654321 zijn allebei geboekt op klantnummer 192837465.",
        "Belegnummer 65929970001 ist keiner Person zugeordnet.",
        "Invoice 4455667788 is overdue",
    ] {
        assert!(resolved(text, EntityType::PhoneNumber).is_empty());
        assert!(resolved_as_if_suppressed(text, EntityType::PhoneNumber).is_empty());
    }
}

// ---------------------------------------------------------------------
// The pack-to-resolver contract the penalty's size rests on.
// ---------------------------------------------------------------------

/// The negative-context penalty is one number in `pii-detection-resolve` and the
/// weights it has to sit between are declared per recognizer here. Nothing
/// else states the relationship, so it is pinned through the real resolver
/// reading the real `thresholds.toml`, exactly as `thresholds.rs` pins the
/// phone ladder.
///
/// For every checksum-guarded recognizer that carries a suppression list:
///
/// * format plus a passing checksum, and no label of its own, must **not**
///   clear the bar once a term stands beside it: that is the order number;
/// * format plus a passing checksum plus the identifier's own label **must**
///   clear it: that is the real identifier.
///
/// A pack whose confidences drift out of that window fails here and says
/// which recognizer, instead of silently changing which of the two readings
/// wins.
#[test]
fn every_suppression_list_sits_inside_the_penaltys_window() {
    let resolver = Resolver::new();
    let verdict = |recognizer: &pii_detection_rules::pack::Recognizer, score: f32| {
        let candidate = Candidate {
            span: pii_detection_engine::text::CharSpan { start: 0, end: 9 },
            entity_type: recognizer.entity_type,
            country: recognizer.country,
            source: pii_detection_engine::candidate::Source::Rule {
                id: "penalty-window",
            },
            score: score.clamp(0.0, 1.0),
            evidence: pii_detection_engine::candidate::Evidence {
                format: true,
                checksum: Some(true),
                context: false,
                negative_context: true,
                ..pii_detection_engine::candidate::Evidence::default()
            },
        };
        !resolver
            .resolve(vec![candidate], &[recognizer.entity_type], "auto")
            .is_empty()
    };

    let mut checked = 0;
    for pack in all() {
        for recognizer in &pack.recognizer {
            if recognizer.negative_context.is_empty() || recognizer.checksum_confidence == 0.0 {
                continue;
            }
            checked += 1;
            let checksum_only = recognizer.base_confidence + recognizer.checksum_confidence;
            let with_label = checksum_only + recognizer.context_confidence;
            assert!(
                !verdict(recognizer, checksum_only),
                "{}: a checksum with no label of its own outweighs its suppression list",
                recognizer.id
            );
            assert!(
                verdict(recognizer, with_label),
                "{}: a checksum plus its own label is still deleted by its suppression list",
                recognizer.id
            );
        }
    }
    assert!(checked >= 3, "only {checked} recognizers were checked");
}
