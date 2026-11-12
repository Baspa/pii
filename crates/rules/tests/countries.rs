use pii_detection_engine::entity::{Country, EntityType};
use pii_detection_engine::text::Document;
use pii_detection_resolve::Resolver;
use pii_detection_rules::matcher::Matcher;

/// What the whole pipeline returns for one type over one text -- the matcher
/// and the resolver together, which is what a caller of `/analyze` sees. A
/// candidate that is raised and then dropped is not a detection.
fn resolved(
    text: &str,
    entity_type: EntityType,
) -> Vec<pii_detection_engine::candidate::Candidate> {
    let document = Document::new(text.to_string());
    Resolver::new().resolve(Matcher::new().scan(&document), &[entity_type], "auto")
}

#[test]
fn a_valid_bsn_is_a_dutch_national_id() {
    let document = Document::new("Het BSN is 123456782 en staat in het dossier".to_string());
    let found = Matcher::new().scan(&document);
    let candidate = found
        .iter()
        .find(|c| c.entity_type == EntityType::NationalId)
        .expect("national id");
    assert_eq!(candidate.country, Some(Country::NL));
    assert_eq!(candidate.evidence.checksum, Some(true));
    assert_eq!(document.slice(candidate.span), "123456782");
}

#[test]
fn a_valid_steuer_id_is_a_german_tax_id() {
    let document = Document::new("Steuer-ID 86095742719".to_string());
    let found = Matcher::new().scan(&document);
    let candidate = found
        .iter()
        .find(|c| c.entity_type == EntityType::TaxId)
        .expect("tax id");
    assert_eq!(candidate.country, Some(Country::DE));
    assert_eq!(candidate.evidence.checksum, Some(true));
}

#[test]
fn an_invoice_number_that_passes_the_checksum_is_still_marked_negative() {
    let document = Document::new("Factuurnummer 123456782 is voldaan".to_string());
    let found = Matcher::new().scan(&document);
    let candidate = found
        .iter()
        .find(|c| c.entity_type == EntityType::NationalId)
        .expect("candidate is raised, and rejected later by the resolver");
    assert!(candidate.evidence.negative_context);
}

// ---------------------------------------------------------------------
// Spain. Two identity numbers, both NATIONAL_ID, both modulo 23.
// ---------------------------------------------------------------------

#[test]
fn a_valid_dni_is_a_spanish_national_id() {
    let document = Document::new("Mi DNI es 84512377G, gracias.".to_string());
    let found = Matcher::new().scan(&document);
    let candidate = found
        .iter()
        .find(|c| c.entity_type == EntityType::NationalId)
        .expect("national id");
    assert_eq!(candidate.country, Some(Country::ES));
    assert_eq!(candidate.evidence.checksum, Some(true));
    assert_eq!(document.slice(candidate.span), "84512377G");
}

#[test]
fn a_valid_nie_is_a_spanish_national_id() {
    let document = Document::new("El número de identidad de extranjero es X7419203R.".to_string());
    let found = Matcher::new().scan(&document);
    let candidate = found
        .iter()
        .find(|c| c.entity_type == EntityType::NationalId)
        .expect("national id");
    assert_eq!(candidate.country, Some(Country::ES));
    assert_eq!(candidate.evidence.checksum, Some(true));
    assert_eq!(document.slice(candidate.span), "X7419203R");
}

#[test]
fn a_spanish_identifier_survives_the_word_factura_beside_it() {
    // The window the confidences are sized for: the identifier's own label
    // plus a passing check letter outweighs a suppression term standing in
    // the same clause, which is what a real ticket looks like. The other
    // half -- that the term still decides when the label is absent -- is
    // the test below.
    for text in [
        "Mi DNI es 84512377G y adjunto la factura del mes pasado.",
        "Su NIE, número de identidad de extranjero X7419203R, y la factura adjunta.",
    ] {
        assert_eq!(
            resolved(text, EntityType::NationalId).len(),
            1,
            "a labelled Spanish identifier was argued off the mask"
        );
    }
}

#[test]
fn a_reference_shaped_like_a_spanish_identifier_is_not_masked() {
    for text in [
        // Checksum-valid and an order number all the same.
        "Mi número de pedido es 47820913A, realizado el martes pasado.",
        "Adjunto el número de factura Z6104938F del trimestre anterior.",
        "Our warehouse quotes order number 25604817K for the pallet.",
        // Checksum-invalid: the check letter alone rejects it.
        "El documento que puse en el formulario fue 84512377A.",
    ] {
        assert!(
            resolved(text, EntityType::NationalId).is_empty(),
            "a reference was masked as a Spanish national id: {text:?}"
        );
    }
}

// ---------------------------------------------------------------------
// Italy. One identifier, the codice fiscale, and it is a TAX_ID.
// ---------------------------------------------------------------------

#[test]
fn a_valid_codice_fiscale_is_an_italian_tax_id() {
    let document = Document::new("Il codice fiscale è GRSLNE88C15H501P.".to_string());
    let found = Matcher::new().scan(&document);
    let candidate = found
        .iter()
        .find(|c| c.entity_type == EntityType::TaxId)
        .expect("tax id");
    assert_eq!(candidate.country, Some(Country::IT));
    assert_eq!(candidate.evidence.checksum, Some(true));
    assert_eq!(document.slice(candidate.span), "GRSLNE88C15H501P");
}

#[test]
fn a_codice_fiscale_survives_the_word_fattura_beside_it() {
    let text = "C.F. MRNFNC79T52F205Z. La fattura del mese scorso è ancora in sospeso.";
    assert_eq!(
        resolved(text, EntityType::TaxId).len(),
        1,
        "a labelled codice fiscale was argued off the mask"
    );
}

#[test]
fn a_reference_shaped_like_a_codice_fiscale_is_not_masked() {
    for text in [
        "Il mio numero d'ordine è CRVMRT83D48D612R, effettuato martedì scorso.",
        "Allego il numero di fattura FRRGPP70B23F839G del trimestre precedente.",
        // The specimen everybody repeats: its check character should be Q.
        "Nel modulo ho scritto RSSMRA85M01H501Z e il sistema lo ha accettato.",
    ] {
        assert!(
            resolved(text, EntityType::TaxId).is_empty(),
            "a reference was masked as an Italian tax id: {text:?}"
        );
    }
}

// ---------------------------------------------------------------------
// France. The NIR, and the Corsican departments that carry a letter.
// ---------------------------------------------------------------------

#[test]
fn a_valid_nir_is_a_french_national_id() {
    let document = Document::new("Le numéro de sécurité sociale est 174033000705847.".to_string());
    let found = Matcher::new().scan(&document);
    let candidate = found
        .iter()
        .find(|c| c.entity_type == EntityType::NationalId)
        .expect("national id");
    assert_eq!(candidate.country, Some(Country::FR));
    assert_eq!(candidate.evidence.checksum, Some(true));
    assert_eq!(document.slice(candidate.span), "174033000705847");
}

#[test]
fn a_nir_written_in_groups_is_found_whole() {
    // How it is printed on a carte Vitale, and how a customer copies it.
    let document = Document::new(
        "Voici le numéro figurant sur ma carte Vitale : 2 88 03 92 034 567 04.".to_string(),
    );
    let kept = Resolver::new().resolve(
        Matcher::new().scan(&document),
        &[EntityType::NationalId],
        "auto",
    );
    assert_eq!(kept.len(), 1, "{kept:?}");
    assert_eq!(document.slice(kept[0].span), "2 88 03 92 034 567 04");
}

#[test]
fn a_corsican_nir_is_found() {
    // 2B substitutes 18 before the modulo. Without that the number does not
    // parse at all, so the failure would be silent under-masking.
    let text = "Mon numéro de sécurité sociale est 2 90 05 2B 033 010 01.";
    assert_eq!(resolved(text, EntityType::NationalId).len(), 1);
}

#[test]
fn a_reference_shaped_like_a_nir_is_not_masked() {
    for text in [
        "Mon numéro de commande est 276081405678953, passé mardi dernier.",
        "Je joins le numéro de facture 183045906789087 du trimestre précédent.",
        // One digit off the key.
        "J'ai saisi 172067510045657 et le système l'a accepté quand même.",
    ] {
        assert!(
            resolved(text, EntityType::NationalId).is_empty(),
            "a reference was masked as a French national id: {text:?}"
        );
    }
}

// ---------------------------------------------------------------------
// Belgium. Eleven digits, two century branches, and the bis form.
// ---------------------------------------------------------------------

#[test]
fn a_valid_national_register_number_is_a_belgian_national_id() {
    let document = Document::new("Het rijksregisternummer is 78091423378.".to_string());
    let found = Matcher::new().scan(&document);
    let candidate = found
        .iter()
        .find(|c| c.entity_type == EntityType::NationalId)
        .expect("national id");
    assert_eq!(candidate.country, Some(Country::BE));
    assert_eq!(candidate.evidence.checksum, Some(true));
    assert_eq!(document.slice(candidate.span), "78091423378");
}

#[test]
fn a_punctuated_national_register_number_is_found_whole() {
    let document = Document::new(
        "Hierbij het nummer zoals het op mijn identiteitskaart staat: 88.02.19-164.61.".to_string(),
    );
    let kept = Resolver::new().resolve(
        Matcher::new().scan(&document),
        &[EntityType::NationalId],
        "auto",
    );
    assert_eq!(kept.len(), 1, "{kept:?}");
    assert_eq!(document.slice(kept[0].span), "88.02.19-164.61");
}

#[test]
fn a_belgian_number_is_found_in_french_as_well_as_dutch() {
    // Belgium is the one country here with two of the six languages, and
    // its French-speaking half is the larger one.
    let text = "Mon numéro de registre national est 02110531868, merci.";
    assert_eq!(resolved(text, EntityType::NationalId).len(), 1);
}

#[test]
fn a_reference_shaped_like_a_national_register_number_is_not_masked() {
    for text in [
        "Mijn ordernummer is 45082339117, besteld afgelopen dinsdag.",
        "Je joins le numéro de facture 71040672235 du trimestre précédent.",
        // One digit off the control.
        "Ik gaf 78091423377 op en het systeem nam het gewoon over.",
    ] {
        assert!(
            resolved(text, EntityType::NationalId).is_empty(),
            "a reference was masked as a Belgian national id: {text:?}"
        );
        // And not as a German tax id either -- the two share a pattern.
        assert!(
            resolved(text, EntityType::TaxId).is_empty(),
            "a reference was masked as a German tax id: {text:?}"
        );
    }
}
