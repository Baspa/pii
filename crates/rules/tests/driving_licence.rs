//! `DRIVING_LICENCE`: two countries, two very different kinds of evidence.
//!
//! Germany's Führerscheinnummer carries a modulo-11 check digit, so it is
//! decided the way every other checksummed country identifier is: the shape
//! never decides, the check digit does, and the licence's own word is what
//! carries it over an order label standing beside it.
//!
//! The DVLA driver number carries no check character this service can
//! compute -- the algorithm is not published -- so it is decided by a
//! sixteen-character shape that encodes a range-checked date of birth. That
//! shape *does* decide on its own, which is the arrangement a documented
//! credential prefix has, and the only reason it is allowed to is that the
//! shape matches nothing at all across 23.3M characters of PII-free prose,
//! the 321 benchmark carriers and the whole case fixture.
//!
//! Both halves of each of those claims are asserted here, because a
//! recognizer that found nothing would pass the negative half of every one
//! of them.
//!
//! NO CASE TEXT AND NO DETECTED VALUE REACHES AN ASSERTION MESSAGE. The
//! values below are the published specimen numbers and values derived from
//! the published rules; failure messages carry ids, types and lengths.

use pii_detection_engine::candidate::{Candidate, Evidence, Source};
use pii_detection_engine::entity::{Country, EntityType};
use pii_detection_engine::text::{CharSpan, Document};
use pii_detection_resolve::Resolver;
use pii_detection_rules::matcher::Matcher;
use pii_detection_rules::pack::all;

/// What the whole pipeline returns for `DRIVING_LICENCE` over one text --
/// matcher and resolver together, which is what a caller of `/analyze` sees.
/// A candidate that is raised and then dropped is not a detection.
fn resolved(text: &str) -> Vec<Candidate> {
    let document = Document::new(text.to_string());
    Resolver::new().resolve(
        Matcher::new().scan(&document),
        &[EntityType::DrivingLicence],
        "auto",
    )
}

fn masked(text: &str) -> Vec<String> {
    let document = Document::new(text.to_string());
    resolved(text)
        .into_iter()
        .map(|candidate| document.slice(candidate.span).to_string())
        .collect()
}

// ---------------------------------------------------------------------
// Germany: the published specimen, and what the check digit is for.
// ---------------------------------------------------------------------

#[test]
fn the_published_specimen_is_a_german_driving_licence() {
    let document = Document::new("Führerscheinnummer B072RRE2I55 vom 4. Mai 2019".to_string());
    let found = Matcher::new().scan(&document);
    let candidate = found
        .iter()
        .find(|c| c.entity_type == EntityType::DrivingLicence)
        .expect("driving licence");
    assert_eq!(candidate.country, Some(Country::DE));
    assert_eq!(candidate.evidence.checksum, Some(true));
    assert!(candidate.evidence.context);
    assert_eq!(document.slice(candidate.span), "B072RRE2I55");
}

#[test]
fn a_german_licence_with_no_label_is_still_decided_by_its_check_digit() {
    // The check digit is the rung that decides, so a licence pasted bare
    // into a ticket is masked. If the base confidence ever rises to where
    // the shape alone clears the bar, this test still passes and
    // `the_german_shape_alone_never_decides` below is what fails.
    assert_eq!(masked("B072RRE2I55"), vec!["B072RRE2I55".to_string()]);
    assert_eq!(masked("b072rre2i55"), vec!["b072rre2i55".to_string()]);
}

#[test]
fn a_german_licence_shape_with_a_wrong_check_digit_is_not_masked() {
    // `Some(false)` rejects on its own, ahead of the threshold -- which is
    // why the ladder below can afford to let the checksum decide.
    for text in [
        "Führerscheinnummer B072RRE2I45",
        "Die Nummer B072RRE2I05 steht im Formular",
    ] {
        assert!(resolved(text).is_empty(), "a wrong check digit was masked");
    }
}

/// The collision this recognizer's shape was chosen to avoid. An eleven-digit
/// run is the German Steuer-ID's and the Belgian national register number's
/// shape, and roughly one in eleven of them clears a modulo-11 check by
/// chance. Requiring the Bundesland letter is what keeps the licence
/// recognizer off them.
#[test]
fn an_eleven_digit_run_is_never_a_driving_licence() {
    for text in [
        // The Belgian register's own published example, which clears the
        // German tax-id check too.
        "Het rijksregisternummer is 42012205181",
        "Meine Steuer-ID lautet 86095742719",
        "Steuer-ID 36574261809",
        // The same digits with the licence's own word beside them: the word
        // must not manufacture a licence out of a shape that is not one.
        "Führerscheinnummer 42012205181",
        "Der Führerschein 86095742719 liegt vor",
    ] {
        assert!(
            resolved(text).is_empty(),
            "an eleven-digit run was masked as a driving licence"
        );
    }
}

/// The same collision one step earlier, at the matcher rather than at the
/// resolver -- and the distinction is the point.
///
/// `an_eleven_digit_run_is_never_a_driving_licence` above passes even if the
/// pattern's first character is relaxed to `[A-Z0-9]`, because
/// `de_fuehrerschein` *also* requires the Bundesland letter and a candidate
/// with `checksum: Some(false)` is deleted by the resolver before anything
/// else happens. That is defence in depth and it is worth having, but it
/// means nothing above pins the pattern itself, and the pack's own comment
/// claims the pattern is what keeps this recognizer off `de_steuer_id`'s
/// shape. This asks the matcher directly: an eleven-digit run must raise **no
/// licence candidate at all**, not merely none that survives.
#[test]
fn an_eleven_digit_run_raises_no_licence_candidate_at_all() {
    let matcher = Matcher::new();
    for text in [
        "42012205181",
        "Meine Steuer-ID lautet 86095742719",
        "Führerscheinnummer 36574261809",
        "Die Referenz lautet 53537638779",
    ] {
        let document = Document::new(text.to_string());
        let raised: Vec<_> = matcher
            .scan(&document)
            .into_iter()
            .filter(|candidate| candidate.entity_type == EntityType::DrivingLicence)
            .collect();
        assert!(
            raised.is_empty(),
            "an eleven-digit run raised {} licence candidate(s) for the resolver to throw away",
            raised.len()
        );
    }
}

/// A licence shape glued into a longer token is not a licence, and the word
/// boundaries at both ends of both patterns are what say so.
///
/// This is the truncation failure arriving from the other direction: without
/// the leading `\b`, `XMORGA657054SM9IJ` yields the sixteen-character suffix
/// and the recognizer masks part of a reference code -- the same shape
/// `pii-detection-resolve`'s containment rule exists to prevent, and the same one
/// the phone recognizer's `is_free_standing` was added for.
#[test]
fn a_licence_shape_glued_into_a_longer_token_is_not_a_licence() {
    for text in [
        // A letter before the German shape, and a character after it.
        "ABB072RRE2I55",
        "B072RRE2I55X",
        "REF-B072RRE2I55X",
        // The same on both sides of the British one.
        "XMORGA657054SM9IJ",
        "MORGA657054SM9IJX",
        "PREFIXMORGA657054SM9IJ",
    ] {
        assert!(
            resolved(text).is_empty(),
            "a shape glued into a longer token was masked as a driving licence"
        );
    }
}

/// The other half: an ordinary German reference of the licence's own shape,
/// behind the word that says what it is. Only the negative-context ladder
/// can reject these -- the check digit passes.
#[test]
fn a_german_reference_shaped_like_a_licence_is_not_masked() {
    for text in [
        "Auftragsnummer B072RRE2I55 vom Januar",
        "Die Rechnungsnummer B072RRE2I55 ist beglichen",
        "Das Kennzeichen B072RRE2I55 steht im Vertrag",
        "Order number B072RRE2I55 has shipped",
    ] {
        assert!(
            resolved(text).is_empty(),
            "a reference was masked as a driving licence"
        );
    }
    // And the pair that keeps the mechanism worth having: the same
    // suppression term beside a licence that carries its own label.
    assert_eq!(
        masked("Führerscheinnummer B072RRE2I55, Rechnungsnummer 4471 liegt bei").len(),
        1,
        "a labelled licence was argued off the mask by one term"
    );
}

// ---------------------------------------------------------------------
// The United Kingdom: no check character, a shape that decides anyway.
// ---------------------------------------------------------------------

#[test]
fn a_dvla_driver_number_is_a_driving_licence() {
    let document = Document::new("Driving licence number MORGA657054SM9IJ".to_string());
    let found = Matcher::new().scan(&document);
    let candidate = found
        .iter()
        .find(|c| c.entity_type == EntityType::DrivingLicence)
        .expect("driving licence");
    // No country: `Country` is the EU-27 and the United Kingdom left it.
    assert_eq!(candidate.country, None);
    // And no checksum claimed, because none was computed.
    assert_eq!(candidate.evidence.checksum, None);
    assert_eq!(document.slice(candidate.span), "MORGA657054SM9IJ");
}

#[test]
fn a_dvla_driver_number_with_no_label_is_still_masked() {
    // The shape decides on its own. This is the rung the German recognizer
    // deliberately does not have, and the British one has to, because there
    // is nothing else it could be decided by.
    assert_eq!(
        masked("MORGA657054SM9IJ"),
        vec!["MORGA657054SM9IJ".to_string()]
    );
    // The official example from GOV.UK's own wallet documentation, which
    // pads a four-letter surname with `9` and carries letter check
    // characters.
    assert_eq!(
        masked("USER9801180MA9PC"),
        vec!["USER9801180MA9PC".to_string()]
    );
}

/// The eighteen-character form: the driver number with the two-digit licence
/// issue number written straight onto the end.
///
/// Without the optional trailing group the whole token matches *nothing*,
/// because the sixteen-character pattern's closing word boundary fails
/// against the digit after it. That is a silent total miss, not a truncation,
/// which is why it is pinned rather than left to the pattern's shape.
#[test]
fn the_issue_number_form_is_matched_whole_or_not_at_all() {
    assert_eq!(
        masked("Driving licence MORGA657054SM9IJ35"),
        vec!["MORGA657054SM9IJ35".to_string()]
    );
    // Nineteen characters is not a licence at either length, and must not
    // produce the sixteen-character prefix of one.
    assert!(resolved("MORGA657054SM9IJ357").is_empty());
}

/// The date is what carries this recognizer's precision, so every way of
/// getting it wrong has to be rejected. Each string below is one character
/// away from the specimen.
#[test]
fn a_dvla_shape_with_an_impossible_date_is_not_a_licence() {
    for text in [
        // Month 13, and month 63 -- which is 13 with the female offset.
        "MORGA613054SM9IJ",
        "MORGA663054SM9IJ",
        // Month 00, and its female form 50.
        "MORGA600054SM9IJ",
        "MORGA650054SM9IJ",
        // Day 00 and day 32.
        "MORGA657004SM9IJ",
        "MORGA657324SM9IJ",
    ] {
        assert!(
            resolved(text).is_empty(),
            "an impossible date was masked as a driving licence"
        );
    }
}

#[test]
fn a_truncated_or_extended_dvla_shape_is_not_a_licence() {
    for text in [
        // Fifteen characters: one short.
        "MORGA657054SM9I",
        // Seventeen: one long, and not the issue-number form either.
        "MORGA657054SM9IJK",
        // The surname field one character short, so the date lands in the
        // wrong place.
        "MORG657054SM9IJ",
        // Digits where the initials belong.
        "MORGA65705412 9IJ",
    ] {
        assert!(
            resolved(text).is_empty(),
            "a mis-sized shape was masked as a driving licence"
        );
    }
}

#[test]
fn an_english_reference_shaped_like_a_dvla_number_is_not_masked() {
    for text in [
        "Order number MORGA657054SM9IJ was despatched on Tuesday",
        "The tracking reference USER9801180MA9PC is still in transit",
        "Invoice MORGA657054SM9IJ remains unpaid",
        "Policy number MORGA657054SM9IJ renews in March",
    ] {
        assert!(
            resolved(text).is_empty(),
            "a reference was masked as a driving licence"
        );
    }
    // And the licence's own word puts it back, which is what makes the
    // suppression above a judgement rather than a blanket refusal.
    assert_eq!(
        masked("Driving licence MORGA657054SM9IJ, order number 4471 attached").len(),
        1,
        "a labelled licence was argued off the mask by one term"
    );
}

/// A sixteen-character Italian codice fiscale is the other structured
/// sixteen-character identifier in these packs. It must not be read as a
/// British licence, and the reason it cannot is the month letter: the DVLA
/// shape requires a digit where a codice fiscale carries one of twelve
/// letters.
#[test]
fn a_codice_fiscale_is_not_a_dvla_driver_number() {
    for text in [
        "Il codice fiscale è MRTMTT91D08F205J",
        "Codice fiscale MLLSNT82P65Z404U",
    ] {
        assert!(
            resolved(text).is_empty(),
            "a codice fiscale was masked as a driving licence"
        );
    }
}

// ---------------------------------------------------------------------
// The ladders, against the real packs and the real threshold.
// ---------------------------------------------------------------------

/// One candidate of a recognizer's own type at a given score, put through
/// the real resolver reading the real `thresholds.toml`.
fn survives(
    recognizer: &pii_detection_rules::pack::Recognizer,
    score: f32,
    checksum: Option<bool>,
    negative_context: bool,
) -> bool {
    let candidate = Candidate {
        span: CharSpan { start: 0, end: 11 },
        entity_type: recognizer.entity_type,
        country: recognizer.country,
        source: Source::Rule { id: "ladder" },
        score: score.clamp(0.0, 1.0),
        evidence: Evidence {
            format: true,
            checksum,
            context: false,
            negative_context,
            ..Evidence::default()
        },
    };
    !Resolver::new()
        .resolve(vec![candidate], &[recognizer.entity_type], "auto")
        .is_empty()
}

fn recognizer(id: &str) -> &'static pii_detection_rules::pack::Recognizer {
    all()
        .iter()
        .flat_map(|pack| pack.recognizer.iter())
        .find(|recognizer| recognizer.id == id)
        .expect("the recognizer is in a loaded pack")
}

/// The German rungs, read off the committed pack rather than restated.
///
/// Shape alone must not decide -- eleven alphanumerics is a reference number
/// as readily as a licence -- and the check digit must. Both halves matter:
/// if the base confidence ever clears the threshold the recognizer starts
/// masking every eleven-character alphanumeric token whose tenth character
/// happens to be a digit, and if shape plus checksum ever stops clearing it
/// the recognizer is off entirely and nothing else would say so.
#[test]
fn the_german_shape_alone_never_decides_and_its_check_digit_always_does() {
    let german = recognizer("de_fuehrerschein");
    assert!(
        !survives(german, german.base_confidence, None, false),
        "the German shape alone decided a driving licence"
    );
    assert!(
        survives(
            german,
            german.base_confidence + german.checksum_confidence,
            Some(true),
            false
        ),
        "a checksum-valid German licence was not decided"
    );
    // And the window the negative-context penalty requires, which
    // `tests/suppression.rs` walks for every checksummed recognizer: a
    // checksum with no label of its own loses to one word, a checksum with
    // the licence's own label does not.
    assert!(!survives(
        german,
        german.base_confidence + german.checksum_confidence,
        Some(true),
        true
    ));
    assert!(survives(
        german,
        german.base_confidence + german.checksum_confidence + german.context_confidence,
        Some(true),
        true
    ));
}

/// The British rungs. `tests/suppression.rs`'s window walk skips this
/// recognizer, because it declares no checksum confidence and that test is
/// written for checksummed recognizers; this is the same property asserted
/// where it does apply.
///
/// The shape decides alone -- it has to, there is nothing else -- one word
/// takes it back off, and the licence's own word puts it back.
#[test]
fn the_british_shape_decides_alone_and_one_word_still_argues_it_down() {
    let british = recognizer("gb_driving_licence");
    assert_eq!(
        british.checksum_confidence, 0.0,
        "this recognizer must not claim a checksum it cannot compute"
    );
    assert_eq!(british.validator, "none");

    assert!(
        survives(british, british.base_confidence, None, false),
        "the British shape must decide on its own"
    );
    assert!(
        !survives(british, british.base_confidence, None, true),
        "one word saying otherwise must take the shape back off the mask"
    );
    assert!(
        survives(
            british,
            british.base_confidence + british.context_confidence,
            None,
            true
        ),
        "the licence's own word must outweigh one suppression term"
    );
}

/// Both recognizers are `DRIVING_LICENCE`, and neither may raise the other's
/// type or fire on the other's values. The two shapes are eleven and sixteen
/// characters and there is no reason they should overlap; this is what says
/// so if a pattern is ever loosened.
#[test]
fn neither_licence_recognizer_fires_on_the_other_countrys_number() {
    let matcher = Matcher::new();
    let sources = |text: &str| {
        let document = Document::new(text.to_string());
        matcher
            .scan(&document)
            .into_iter()
            .filter(|candidate| candidate.entity_type == EntityType::DrivingLicence)
            .filter_map(|candidate| match candidate.source {
                Source::Rule { id } => Some(id),
                Source::Model => None,
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(sources("B072RRE2I55"), vec!["de_fuehrerschein"]);
    assert_eq!(sources("MORGA657054SM9IJ"), vec!["gb_driving_licence"]);
    assert_eq!(sources("MORGA657054SM9IJ35"), vec!["gb_driving_licence"]);
}

// ---------------------------------------------------------------------
// The Netherlands: no structure at all, and a ladder instead.
//
// Ten bare digits. No check character, no internal fields, nothing a
// pattern can be strict about -- which is why
// `training/reports/driving-licence.md` §3.3 excluded it. What makes it
// detectable is that the shape and a nearby licence word are worth 0.40 and
// 0.35 against a 0.70 threshold, so neither decides alone and both together
// do. Every test below asserts one half of that, and the halves are written
// separately on purpose: a recognizer that found nothing would pass all the
// negative ones.
// ---------------------------------------------------------------------

/// Neither signal reaches the bar alone. This is the whole safety argument
/// for masking a run of ten digits, asserted through the real matcher, the
/// real packs and the real `thresholds.toml`.
#[test]
fn neither_the_dutch_shape_nor_the_word_reaches_the_bar_alone() {
    // The shape, with no licence word anywhere in the text.
    for text in [
        "De referentie van de zending is 5847291036",
        "Het bedrag 5847291036 staat op de afrekening",
        "5847291036",
    ] {
        assert!(
            resolved(text).is_empty(),
            "ten digits alone were masked as a driving licence"
        );
    }
    // The word, with no ten-digit run anywhere in the text.
    for text in [
        "Het rijbewijs van de bestuurder is verlopen sinds maart",
        "Rijbewijsnummer ontbreekt in het dossier 58472 en 910",
        "Driving licence details are missing from case 123456789",
    ] {
        assert!(
            resolved(text).is_empty(),
            "a licence word with no licence-shaped run was masked"
        );
    }
    // Together.
    assert_eq!(
        masked("Het rijbewijsnummer is 5847291036"),
        vec!["5847291036".to_string()]
    );
    assert_eq!(
        masked("Driving licence number 5847291036 was checked"),
        vec!["5847291036".to_string()]
    );
}

/// The required hard negative, and the reason this ladder is calibrated
/// below the penalised bar rather than above it like the phone recognizer's.
///
/// A ten-digit order number and the word `rijbewijs` inside the same
/// 64-byte window is the ordinary shape of a car-rental invoice, and the
/// context window does not ask which number a word was written about. With
/// no check digit to argue back with, one word saying `ordernummer` has to
/// win -- 0.75 - 0.25 = 0.50 - and it does.
#[test]
fn a_ten_digit_order_number_is_not_a_licence_even_with_a_licence_word_beside_it() {
    for text in [
        "Ordernummer 5847291036 is verzonden; het rijbewijs sturen we later na",
        "Uw factuurnummer 5847291036 hoort bij de huurauto, rijbewijs bijgevoegd",
        "Het klantnummer 5847291036 staat in het rijbewijsdossier",
        "Invoice 5847291036 relates to the driving licence check",
        "Het kenteken is bekend; ordernummer 5847291036 en rijbewijs volgen",
    ] {
        assert!(
            resolved(text).is_empty(),
            "a labelled reference number was masked as a driving licence"
        );
    }
}

/// The cost of the line above, written down rather than discovered later.
///
/// A real Dutch licence whose sentence also carries an order or invoice
/// label is **not** masked. That is the trade this ladder makes and it only
/// goes one way: with no checksum there is nothing to outrank the word. The
/// German recognizer does not have this cost, because its check digit puts
/// it at 1.00 with a label of its own; this recognizer tops out at 0.75.
#[test]
fn a_dutch_licence_in_the_same_clause_as_an_order_label_is_lost() {
    assert!(
        resolved("Rijbewijsnummer 5847291036, ordernummer 4471 bijgevoegd").is_empty(),
        "this test records a known miss; if it now passes, the ladder changed"
    );
    // And the same sentence with the order label out of the 64-byte window
    // is found, which is what keeps the miss above a window effect rather
    // than a blanket refusal.
    assert_eq!(
        masked(
            "Rijbewijsnummer 5847291036 hoort bij de aanvraag die vorige maand is ingediend \
             door de afdeling verhuur in Utrecht. Het ordernummer is 4471."
        ),
        vec!["5847291036".to_string()]
    );
}

/// The adjacent shape that must never be read as a licence: nine digits with
/// a passing eleven-proof is a BSN, and a BSN is a natural person's national
/// identifier, not a document.
///
/// Asked of the **matcher**, not of the resolver: the licence recognizer
/// must raise no candidate at all over nine digits, rather than raise one
/// that something else happens to outrank.
#[test]
fn a_checksum_valid_bsn_raises_no_dutch_licence_candidate() {
    let matcher = Matcher::new();
    for text in [
        "Mijn burgerservicenummer is 123456782",
        "Rijbewijs en BSN 123456782 staan in het dossier",
        "Het rijbewijsnummer ontbreekt, het BSN 111222333 staat er wel",
        // Eleven digits: the Steuer-ID and the Belgian register number.
        "Rijbewijs bij de Steuer-ID 86095742719",
        "Rijbewijs en rijksregisternummer 42012205181",
    ] {
        let document = Document::new(text.to_string());
        let raised: Vec<_> = matcher
            .scan(&document)
            .into_iter()
            .filter(|candidate| candidate.entity_type == EntityType::DrivingLicence)
            .collect();
        assert!(
            raised.is_empty(),
            "a run that is not ten digits raised {} licence candidate(s)",
            raised.len()
        );
    }
    // And the BSN itself is still masked, as the national identifier it is.
    let document = Document::new("Mijn burgerservicenummer is 123456782".to_string());
    let kept = Resolver::new().resolve(
        Matcher::new().scan(&document),
        &[EntityType::NationalId],
        "auto",
    );
    assert_eq!(kept.len(), 1, "the BSN itself stopped being masked");
}

/// A ten-digit run glued into a longer token is a reference code, and the
/// word boundaries at both ends of the pattern are what say so.
#[test]
fn a_ten_digit_run_inside_a_longer_token_is_not_a_dutch_licence() {
    for text in [
        "Rijbewijs REF5847291036",
        "Rijbewijs 5847291036X",
        "Rijbewijs 58472910366",
        "Rijbewijs 584729103",
    ] {
        assert!(
            resolved(text).is_empty(),
            "a ten-digit fragment of a longer token was masked as a driving licence"
        );
    }
}

/// The one collision this recognizer has, pinned rather than left to be
/// discovered: a Dutch licence number beginning with a trunk zero is
/// returned as a telephone number.
///
/// `0612345678` is a valid Dutch mobile written solid, and the phone
/// recognizer scores a trunk zero 0.95 against this recognizer's 0.75, so
/// the phone candidate wins the identical-span overlap. **The same
/// characters are masked either way** -- nothing reaches the caller in the
/// clear -- and a workspace that enabled licences but not phones gets the
/// licence back through the resolver's identical-span restoration, which is
/// the second half asserted here. Only the label differs.
///
/// `inject.rs`'s Dutch generator never draws a value of this shape, and
/// names this test as the reason.
#[test]
fn a_dutch_licence_written_with_a_trunk_zero_is_masked_as_a_telephone_number() {
    let text = "Het rijbewijsnummer is 0612345678";
    let document = Document::new(text.to_string());

    // Both types enabled: the telephone reading wins.
    let both = Resolver::new().resolve(
        Matcher::new().scan(&document),
        &[EntityType::DrivingLicence, EntityType::PhoneNumber],
        "auto",
    );
    assert_eq!(both.len(), 1, "the span was masked twice or not at all");
    assert_eq!(both[0].entity_type, EntityType::PhoneNumber);
    assert_eq!(document.slice(both[0].span), "0612345678");

    // Licences only: the same characters come back, under this type.
    assert_eq!(masked(text), vec!["0612345678".to_string()]);
}

/// The suppression terms that were taken back out, and the sentence that
/// took them out.
///
/// `bsn` names a nine-digit number and `iban` names a token with letters in
/// it: neither can ever be what a ten-digit run is, so neither could ever
/// suppress a false positive -- while both sit in the ordinary Dutch form
/// that quotes a BSN or a bank account beside a licence number. This is the
/// `de_steuer_id`/`iban` defect `tests/suppression.rs` records, one
/// recognizer later.
#[test]
fn a_dutch_licence_survives_a_bsn_and_an_iban_in_the_same_sentence() {
    for text in [
        "Ter controle: mijn BSN is 252760189, het rijbewijsnummer is 7064128395",
        "Rijbewijsnummer 7064128395, IBAN NL91ABNA0417164300, beide in bijlage",
        "Het kenteken is 12-ABC-3 en het rijbewijsnummer 7064128395",
    ] {
        assert!(
            masked(text).contains(&"7064128395".to_string()),
            "a real Dutch licence was argued off the mask by a term that names another shape"
        );
    }
}

// ---------------------------------------------------------------------
// France: twelve digits with a month in the middle of them.
//
// The NEPH's third and fourth digits are the month of registration, which
// is a factor of eight against a bare twelve-digit run and the only
// constraint any source publishes. Everything else is the same ladder the
// Dutch recognizer uses.
// ---------------------------------------------------------------------

#[test]
fn neither_the_french_shape_nor_the_word_reaches_the_bar_alone() {
    for text in [
        "La référence de l'envoi est 121104123456",
        "Le montant 121104123456 figure au relevé",
        "121104123456",
    ] {
        assert!(
            resolved(text).is_empty(),
            "twelve digits alone were masked as a driving licence"
        );
    }
    for text in [
        "Le permis de conduire du chauffeur arrive à expiration en mars",
        "Le numéro NEPH manque au dossier 1211 et 04123",
        "The French driving licence details are missing from case 12110412",
    ] {
        assert!(
            resolved(text).is_empty(),
            "a licence word with no licence-shaped run was masked"
        );
    }
    assert_eq!(
        masked("Le numéro du permis de conduire est 121104123456"),
        vec!["121104123456".to_string()]
    );
    assert_eq!(
        masked("Mon NEPH est 121104123456"),
        vec!["121104123456".to_string()]
    );
}

#[test]
fn a_twelve_digit_order_number_is_not_a_licence_even_with_a_licence_word_beside_it() {
    for text in [
        "Numéro de commande 121104123456 pour le permis de conduire du chauffeur",
        "La facture 121104123456 concerne la location et le permis de conduire",
        "Order number 121104123456 relates to the driving licence on file",
        "Le code-barres 121104123456 figure sur le dossier de permis de conduire",
    ] {
        assert!(
            resolved(text).is_empty(),
            "a labelled reference number was masked as a driving licence"
        );
    }
}

/// The month is the whole of this recognizer's structural precision, so
/// both ends of the alternation have to be closed.
#[test]
fn a_neph_with_a_month_the_calendar_does_not_have_is_not_a_licence() {
    for text in [
        "Le numéro du permis de conduire est 121304123456",
        "Le numéro du permis de conduire est 120004123456",
        "Le numéro du permis de conduire est 129904123456",
        // Eleven and thirteen digits: one either side of the NEPH, and
        // thirteen is an EAN-13 barcode.
        "Le numéro du permis de conduire est 12110412345",
        "Le numéro du permis de conduire est 1211041234567",
        // Glued into a longer token.
        "Le numéro du permis de conduire est REF121104123456",
    ] {
        assert!(
            resolved(text).is_empty(),
            "a run that is not a NEPH was masked as a driving licence"
        );
    }
}

/// A French social security number is the other long digit run in French
/// correspondence. It is thirteen digits, or fifteen with its key, and must
/// never be read as a licence.
#[test]
fn a_french_nir_is_not_a_driving_licence() {
    let matcher = Matcher::new();
    for text in [
        "Le numéro de sécurité sociale est 174033000705847",
        "Permis de conduire et sécurité sociale 174033000705847 au dossier",
    ] {
        let document = Document::new(text.to_string());
        let raised: Vec<_> = matcher
            .scan(&document)
            .into_iter()
            .filter(|candidate| candidate.entity_type == EntityType::DrivingLicence)
            .collect();
        assert!(
            raised.is_empty(),
            "a NIR raised {} licence candidate(s)",
            raised.len()
        );
    }
}

/// The French half of the telephone-number collision §13.4 records.
///
/// A NEPH registered between 2000 and 2009 begins with a zero, and a
/// twelve-digit run with a leading zero is accepted by at least one of the
/// phone recognizer's eight numbering plans -- which scores a trunk prefix
/// 0.95 against this recognizer's 0.75. The same characters are masked over
/// the same span; the label is `PHONE_NUMBER`. `inject.rs`'s French
/// generator never draws such a year and names this test as the reason.
#[test]
fn a_french_licence_from_a_year_beginning_zero_is_masked_as_a_telephone_number() {
    let text = "Le numéro du permis de conduire est 051104123456";
    let document = Document::new(text.to_string());

    let both = Resolver::new().resolve(
        Matcher::new().scan(&document),
        &[EntityType::DrivingLicence, EntityType::PhoneNumber],
        "auto",
    );
    assert_eq!(both.len(), 1, "the span was masked twice or not at all");
    assert_eq!(both[0].entity_type, EntityType::PhoneNumber);
    assert_eq!(document.slice(both[0].span), "051104123456");

    // Licences only: the same characters come back, under this type.
    assert_eq!(masked(text), vec!["051104123456".to_string()]);
}

// ---------------------------------------------------------------------
// Italy: a control character nobody publishes, and two shapes that do not
// need it.
//
// §3.2 excluded Italy because the union of the three published readings --
// which is Microsoft Purview's own pattern for this identifier -- matched
// 350 times over 23.3M characters of PII-free prose, 345 of them ordinary
// ten-letter upper-case words. The two concrete readings that shipped
// instead cannot match a word at all, and the first test below is that
// claim, made against the words themselves.
// ---------------------------------------------------------------------

/// The 350-word finding, beaten, and **proved not to be vacuous in the same
/// test**.
///
/// Each word below is one Purview's pattern for the Italian driving licence
/// matches: four are from the §4 histogram itself and four more are ordinary
/// Italian words of exactly ten letters that a document about a driving
/// licence would plausibly contain in upper case. The assertion has two
/// halves, and the second is what makes the first mean anything:
///
/// 1. the shipped recognizer raises nothing for any of them, **even with
///    `patente di guida` standing beside them**, which is the arrangement
///    that would carry a licence-shaped run over the bar;
/// 2. Purview's union pattern, compiled here, matches every one of them --
///    so the words really are the population the exclusion was about, and a
///    recognizer that had simply stopped working would fail this half.
#[test]
fn the_ordinary_words_purviews_italian_pattern_matches_are_not_licences() {
    // `[A-Z][A-Z0-9][A-Z0-9]{7}[A-Z]`: one letter, one alphanumeric, seven
    // alphanumerics, one letter. Microsoft's published pattern, restated
    // here so this test can show what it does.
    let purview = regex::Regex::new(r"\b[A-Z][A-Z0-9][A-Z0-9]{7}[A-Z]\b")
        .expect("the restated pattern is a literal");
    let matcher = Matcher::new();

    for word in [
        // From the §4 histogram over `training/data-negatives`.
        "PARLAMENTO",
        "PARLAMENTS",
        "VERORDNUNG",
        "COMMISSION",
        // Ten-letter Italian words a licence document really does contain.
        "CONDUCENTE",
        "AUTOSCUOLA",
        "PRESIDENTE",
        "RILASCIATA",
    ] {
        assert!(
            purview.is_match(word),
            "this word is not in the population the exclusion was about"
        );
        for text in [
            word.to_string(),
            format!("Il numero della patente di guida è {word}"),
            format!("{word} patente di guida"),
        ] {
            let document = Document::new(text.clone());
            let raised: Vec<_> = matcher
                .scan(&document)
                .into_iter()
                .filter(|candidate| candidate.entity_type == EntityType::DrivingLicence)
                .collect();
            assert!(
                raised.is_empty(),
                "an ordinary ten-letter word raised {} licence candidate(s)",
                raised.len()
            );
        }
    }
}

#[test]
fn neither_the_italian_shape_nor_the_word_reaches_the_bar_alone() {
    for text in [
        "Il riferimento della spedizione è MI1234567A",
        "Il codice interno U1532B456K compare sul documento",
        "MI1234567A",
    ] {
        assert!(
            resolved(text).is_empty(),
            "an Italian licence shape alone was masked"
        );
    }
    for text in [
        "La patente di guida del conducente scade a marzo",
        "Il numero di patente manca alla pratica MI12345",
        "The Italian driving licence details are missing from case MI1234",
    ] {
        assert!(
            resolved(text).is_empty(),
            "a licence word with no licence-shaped run was masked"
        );
    }
    assert_eq!(
        masked("Il numero della patente di guida è MI1234567A"),
        vec!["MI1234567A".to_string()]
    );
    // The Ministry's own worked example, with the space its circular prints.
    assert_eq!(
        masked("Il duplicato della patente di guida è U1532B456 K"),
        vec!["U1532B456 K".to_string()]
    );
    assert_eq!(
        masked("Numero di patente U1532B456K"),
        vec!["U1532B456K".to_string()]
    );
}

#[test]
fn an_italian_reference_shaped_like_a_licence_is_not_masked() {
    for text in [
        "Numero d'ordine MI1234567A per la patente di guida",
        "La fattura MI1234567A riguarda il noleggio e la patente di guida",
        "Order number U1532B456K relates to the driving licence on file",
        "Il numero cliente MI1234567A è indicato sulla patente di guida",
    ] {
        assert!(
            resolved(text).is_empty(),
            "a labelled reference was masked as a driving licence"
        );
    }
}

#[test]
fn an_italian_run_of_the_wrong_length_is_not_a_licence() {
    for text in [
        // Six and eight digits where a licence has seven.
        "Numero di patente MI123456A",
        "Numero di patente MI12345678A",
        // No control letter: this is a passport's shape, not a licence's.
        "Numero di patente MI1234567",
        // Glued into a longer token, both ends.
        "Numero di patente REFMI1234567A",
        "Numero di patente MI1234567AB",
        // `U1` with the control character missing.
        "Numero di patente U1532B456",
    ] {
        assert!(
            resolved(text).is_empty(),
            "a run of the wrong length was masked as an Italian licence"
        );
    }
}

/// An Italian passport number is two letters and seven digits -- nine
/// characters, one short of the licence -- and the `passport` recognizer's
/// own pattern is wide enough to cover the licence's ten. Both are decided
/// by the word beside them, so both readings have to come out right, and
/// this asserts they do in both directions.
#[test]
fn a_passport_and_an_italian_licence_are_told_apart_by_the_word_beside_them() {
    let both = |text: &str| {
        let document = Document::new(text.to_string());
        Resolver::new()
            .resolve(
                Matcher::new().scan(&document),
                &[EntityType::DrivingLicence, EntityType::PassportNumber],
                "auto",
            )
            .into_iter()
            .map(|candidate| candidate.entity_type)
            .collect::<Vec<_>>()
    };
    assert_eq!(
        both("Il numero della patente di guida è MI1234567A"),
        vec![EntityType::DrivingLicence]
    );
    assert_eq!(
        both("Il numero di passaporto è MI1234567"),
        vec![EntityType::PassportNumber]
    );
    // The licence's shape behind the passport's word: the passport
    // recognizer's pattern reaches ten characters, so it wins on its own
    // vocabulary and the licence recognizer raises 0.40 and is dropped.
    assert_eq!(
        both("Il numero di passaporto è MI1234567A"),
        vec![EntityType::PassportNumber]
    );
}

/// The 64-byte context window is load-bearing for every context-carried
/// recognizer, and nothing else here pins its **width**.
///
/// Found by mutation: widening `matcher::CONTEXT_WINDOW` from 64 to 256
/// failed exactly one test in this crate, and in the wrong direction (a
/// licence that was meant to be lost stayed lost). The benchmark caught it
/// -- over-masking 94 to 99 and `DRIVING_LICENCE` recall 0.9884 to 0.9709 --
/// but a unit test should not need the benchmark to notice a four-fold
/// widening of the neighbourhood a bare digit run is read in.
///
/// Both halves, because a recognizer that found nothing would pass the
/// first: a licence word roughly 200 bytes from a licence-shaped run must
/// not carry it, and the same word inside the window must.
#[test]
fn a_licence_word_out_of_the_window_does_not_carry_a_licence_shaped_run() {
    // 132 characters of filler between the word and the number: inside a
    // 256-byte window, well outside a 64-byte one.
    let far = "Het rijbewijs van de bestuurder is vorige maand door de afdeling                verhuur in Utrecht gecontroleerd en akkoord bevonden voor de                hele looptijd. Het nummer is 5847291036";
    assert!(
        resolved(far).is_empty(),
        "a licence word far outside the context window carried a ten-digit run"
    );
    let near = "Het rijbewijs is gecontroleerd. Het nummer is 5847291036";
    assert_eq!(
        masked(near),
        vec!["5847291036".to_string()],
        "the same word inside the window stopped carrying it"
    );
}

// ---------------------------------------------------------------------
// The ladder these recognizers share, against the real packs, the real
// thresholds and the compile-time calibration in `pii_detection_rules::licence`.
// ---------------------------------------------------------------------

/// Every recognizer that is carried by context rather than by structure.
/// Extended as countries are added, and read by the two tests below.
const CONTEXT_CARRIED: [&str; 3] = ["nl_rijbewijs", "fr_permis_conduire", "it_patente"];

/// The drift guard between `src/licence.rs` and the committed packs.
///
/// `licence.rs` restates five numbers so the calibration can be a compile
/// error where the weights are written. Restating a number is how two files
/// come apart, so this is the test that catches it -- and it asks the real
/// resolver reading the real `thresholds.toml` for the two bars rather than
/// reading the file itself, because what matters is the number the service
/// actually applies.
#[test]
fn the_context_carried_ladder_is_the_one_the_packs_commit() {
    use pii_detection_rules::licence;

    for id in CONTEXT_CARRIED {
        let carried = recognizer(id);
        assert_eq!(
            carried.validator, "none",
            "{id} claims a validator; nothing here has a checksum to compute"
        );
        assert_eq!(
            carried.checksum_confidence, 0.0,
            "{id} claims a checksum confidence it can never earn"
        );
        assert_eq!(
            carried.base_confidence,
            licence::SHAPE,
            "{id}'s shape weight has drifted from src/licence.rs"
        );
        assert_eq!(
            carried.context_confidence,
            licence::A_WORD_NEARBY,
            "{id}'s context weight has drifted from src/licence.rs"
        );
        assert!(
            !carried.negative_context.is_empty(),
            "{id} has no suppression vocabulary, and its ladder is calibrated around having one"
        );
    }

    // The threshold `licence.rs` restates, measured through the resolver.
    let any = recognizer(CONTEXT_CARRIED[0]);
    assert!(
        !survives(any, licence::CALIBRATED_AGAINST - 0.01, None, false),
        "the resolver's DRIVING_LICENCE threshold is lower than src/licence.rs says"
    );
    assert!(
        survives(any, licence::CALIBRATED_AGAINST, None, false),
        "the resolver's DRIVING_LICENCE threshold is higher than src/licence.rs says"
    );
    // And the penalty it restates, in both directions.
    assert!(
        !survives(any, licence::PENALISED_AGAINST - 0.01, None, true),
        "the resolver's negative-context penalty is smaller than src/licence.rs says"
    );
    assert!(
        survives(any, licence::PENALISED_AGAINST, None, true),
        "the resolver's negative-context penalty is larger than src/licence.rs says"
    );
}

/// The property `src/licence.rs` asserts at compile time, asserted again
/// through the real resolver: no single rung reaches the bar, both together
/// do, and one suppression term takes them back off.
///
/// The compile-time block proves it about the constants. This proves it
/// about the numbers the service actually applies, which is not the same
/// claim.
#[test]
fn no_single_rung_of_the_context_carried_ladder_reaches_the_bar() {
    for id in CONTEXT_CARRIED {
        let carried = recognizer(id);
        assert!(
            !survives(carried, carried.base_confidence, None, false),
            "{id}: the shape alone decided a driving licence"
        );
        assert!(
            !survives(carried, carried.context_confidence, None, false),
            "{id}: a nearby word alone decided a driving licence"
        );
        assert!(
            survives(
                carried,
                carried.base_confidence + carried.context_confidence,
                None,
                false
            ),
            "{id}: the shape and a nearby word together decided nothing"
        );
        assert!(
            !survives(
                carried,
                carried.base_confidence + carried.context_confidence,
                None,
                true
            ),
            "{id}: one word saying the digits are an order number did not take them off the mask"
        );
    }
}
