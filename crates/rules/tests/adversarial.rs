use pii_detection_engine::entity::EntityType;
use pii_detection_engine::text::Document;
use pii_detection_resolve::Resolver;
use pii_detection_rules::matcher::Matcher;
use pii_detection_rules::pack::all;

/// Every adversarial fixture must survive the whole pipeline without producing
/// its recognizer's type. These are the strings that look like personal data
/// and are not: invoice numbers, order references, version strings.
#[test]
fn adversarial_fixtures_are_rejected() {
    let matcher = Matcher::new();
    let resolver = Resolver::new();

    for pack in all() {
        for recognizer in &pack.recognizer {
            for fixture in &recognizer.adversarial {
                let document = Document::new(fixture.clone());
                let kept =
                    resolver.resolve(matcher.scan(&document), &[recognizer.entity_type], "auto");
                assert!(
                    kept.is_empty(),
                    "{}: adversarial fixture survived: {:?}",
                    recognizer.id,
                    kept.iter().map(|c| c.entity_type).collect::<Vec<_>>()
                );
            }
        }
    }
}

/// `PhoneRecognizer` lives outside the pack structure, so the loop above
/// never exercises it: `PHONE_NUMBER` is an active, thresholded type with no
/// full-pipeline adversarial coverage otherwise.
///
/// This is still not a negative-context case, and none is planned: a valid
/// number written behind the words "order number" is still a valid number,
/// and the resolver deletes a suppressed candidate ahead of its threshold, so
/// a term list would trade over-masking for silent leaks. What decides these
/// is the written form -- see `phone::written_form` and
/// `training/reports/phone-calibration.md`. A run of digits some numbering
/// plan happens to accept, written the way a reference number is written,
/// must produce nothing.
#[test]
fn a_reference_number_shaped_like_a_valid_number_produces_no_phone_number() {
    let matcher = Matcher::new();
    let resolver = Resolver::new();

    for text in [
        // No numbering plan accepts these digits at all.
        "Order 999999999999999 shipped",
        // Every one of these *is* accepted by at least one of the eight plans
        // the recognizer consults -- `123456789` in France, `9876543` in
        // Germany, `65929970001` in the Netherlands -- which is why validity
        // alone could never have decided them.
        "Factuur 123456789 en order 987654321 zijn allebei geboekt op klantnummer 192837465.",
        "Belegnummer 65929970001 ist keiner Person zugeordnet.",
        "Order XR1234567 shipped and tracking AB9876543 is active.",
        "Invoice 4455667788 is overdue",
        "Vergadering op 01-02-2026 bevestigd",
        "Released 2026-08-21 to production",
        // Spanish and Italian references. `987654321` is a valid Spanish 9xx
        // landline and `2024001234` a valid German number, so the widened
        // context vocabulary is the only thing standing between these and the
        // bar -- and none of them says telephone.
        "El pedido 987654321 se envió ayer desde el almacén.",
        "Factura 2024001234 pendiente, referencia de cliente 192837465.",
        "L'ordine 2024001234 è stato spedito con riferimento 987654321.",
        "Partita IVA 12345678901 iscritta al registro.",
        "Riunione confermata per il 01-02-2026",
    ] {
        let document = Document::new(text.to_string());
        let kept = resolver.resolve(matcher.scan(&document), &[EntityType::PhoneNumber], "auto");
        assert!(
            kept.is_empty(),
            "phone: adversarial fixture survived: {:?}",
            kept.iter().map(|c| c.entity_type).collect::<Vec<_>>()
        );
    }
}

/// The other half of the same property, and the one that makes the half above
/// worth having: the forms real numbers take must still survive the whole
/// pipeline. A recognizer that rejects everything passes every adversarial
/// test there is.
#[test]
fn a_number_written_as_a_number_survives_the_whole_pipeline() {
    let matcher = Matcher::new();
    let resolver = Resolver::new();

    for text in [
        "Rufen Sie Herrn Mueller unter +49 30 123456 an.",
        "Bel me op +31 20 123 4567 als het dringend is.",
        "Erreichbar unter 030 12345678 zwischen neun und fuenf.",
        "Mon numero est le 01 42 68 53 00.",
        "Contact via 06 12345678.",
        "Mijn mobiel is 0612345678.",
        "Call us on 415 226 4321 during office hours.",
        // Spain and Italy, whose numbers reach the bar on a word alone. These
        // went undetected end to end until the context vocabulary was widened;
        // see `training/reports/phone-es-it.md`.
        "Mi teléfono es el 612345678, llámame cuando quieras.",
        "Tfno. de contacto: 912 34 56 78",
        "Puedes llamar al 900 123 456 en horario de oficina.",
        "Il mio cellulare è 3401234567, chiamami pure.",
        "Cell. 340 123 4567 per qualsiasi informazione.",
        "Recapito telefonico: 349 123 4567",
    ] {
        let document = Document::new(text.to_string());
        let kept = resolver.resolve(matcher.scan(&document), &[EntityType::PhoneNumber], "auto");
        assert_eq!(kept.len(), 1, "phone: {text:?} produced {kept:?}");
    }
}
