//! Spanish and Italian vocabulary for `VAT_NUMBER`, `PASSPORT_NUMBER`,
//! `CREDIT_CARD` and `IP_ADDRESS`, end to end.
//!
//! `training/reports/pii-detection-vs-azure-2026-08-23.md` measured the gap this
//! file closes: `universal.toml` carried no Spanish or Italian context terms
//! for `vat` or `passport` at all (Spanish VAT 0/14, Spanish passport 0/12,
//! Italian VAT 1/13, Italian passport 0/13, over the benchmark as it
//! then stood, 649 cases), the credit-card and BSN negative-context lists
//! were English,
//! Dutch and German only, and the ipv4 negative-context list carried
//! `version` but not `versión`. `8da345f` (the phone recognizer) is the
//! precedent this follows: written from how Spanish and Italian are
//! actually written, not from the fixture, with a dedicated set of
//! sentences pinning every term that was deliberately left out.
//!
//! `positive_fixtures_are_found`/`negative_fixtures_are_not_found`/
//! `adversarial_fixtures_are_rejected` (`tests/fixtures.rs`,
//! `tests/adversarial.rs`) already exercise every string the pack's own
//! `positive`/`negative`/`adversarial` arrays carry, which now include
//! Spanish and Italian cases too. This file is the second, independent
//! check the phone precedent added: real sentences, and -- the one hazard
//! this task was warned about twice -- sentences where a short vocabulary
//! term appears inside a longer word it must not decide.

use pii_detection_engine::entity::EntityType;
use pii_detection_engine::text::Document;
use pii_detection_resolve::Resolver;
use pii_detection_rules::matcher::Matcher;

fn kept(text: &str, entity_type: EntityType) -> usize {
    let document = Document::new(text.to_string());
    Resolver::new()
        .resolve(Matcher::new().scan(&document), &[entity_type], "auto")
        .len()
}

// --- VAT_NUMBER: Spanish and Italian ----------------------------------

/// Real phrasing, not the fixture's own carrier sentences: a support
/// email, an invoice line, a company register entry.
const VAT_FOUND: [&str; 8] = [
    "Número de IVA: ESB99887766",
    "Numero de IVA: ESA11223344",
    "El NIF de la empresa es ESB99887766, indicado en la factura adjunta.",
    "Nuestro número de IVA para la Unión Europea es ESA55667788.",
    "Partita IVA IT12345678901",
    "La nostra partita IVA è IT98765432109, iscritta al registro delle imprese.",
    "Si prega di indicare la partita IVA IT12345678901 in fattura.",
    "PARTITA IVA: IT98765432109",
];

/// Ordinary Spanish and Italian prose where a `-iva` adjective sits next to
/// a VAT-shaped string, and no full phrase does. Bare `iva` was measured
/// and deliberately left out of `context_terms` for exactly this collision
/// -- see the comment in `universal.toml`. These sentences are what "left
/// out" has to survive: an unrelated identifier next to a common Spanish or
/// Italian adjective must not become a VAT number.
const VAT_NOT_FOUND: [&str; 4] = [
    "Ofrecemos una alternativa ESB99887766 para el mismo proyecto.",
    "Fue una decisión bastante objetiva ESA11223344 según el comité.",
    "Il pacco arriva con il riferimento IT12345678101 la settimana prossima.",
    "Una proposta esclusiva IT98765432009 per i nostri clienti.",
];

#[test]
fn a_spanish_or_italian_vat_number_in_ordinary_writing_is_found() {
    for text in VAT_FOUND {
        assert_eq!(kept(text, EntityType::VatNumber), 1, "not found: {text:?}");
    }
}

#[test]
fn a_iva_suffixed_adjective_next_to_a_vat_shaped_string_decides_nothing() {
    for text in VAT_NOT_FOUND {
        assert_eq!(
            kept(text, EntityType::VatNumber),
            0,
            "over-masked: {text:?}"
        );
    }
}

// --- PASSPORT_NUMBER: Spanish and Italian -------------------------------

const PASSPORT_FOUND: [&str; 6] = [
    "Mi número de pasaporte es PX1234567.",
    "Numero de pasaporte: WA9876543",
    "El pasaporte número 123456789 fue renovado el mes pasado.",
    "Il numero di passaporto è YT1234567.",
    "Passaporto: ZQ9876543",
    "Indicare il numero di passaporto 234567891 nella richiesta.",
];

/// `pasaporte`/`passaporto` are whole, specific words -- unlike `iva` there
/// is no shorter Spanish or Italian word they sit inside -- so unlike the
/// VAT set above this is not a collision test. It pins that an unrelated
/// nine-digit reference next to ordinary Spanish/Italian business prose
/// (which is what `passport`'s pattern also matches: a bare `\d{7,9}\b`)
/// still needs one of those words nearby, and does not fire on none of
/// them being present.
const PASSPORT_NOT_FOUND: [&str; 2] = [
    "El comprador manifiesta que el pedido 192837465 ha sido recibido.",
    "La especificación del pedido 192837465 se adjunta a este correo.",
];

#[test]
fn a_spanish_or_italian_passport_number_in_ordinary_writing_is_found() {
    for text in PASSPORT_FOUND {
        assert_eq!(
            kept(text, EntityType::PassportNumber),
            1,
            "not found: {text:?}"
        );
    }
}

#[test]
fn an_order_reference_beside_unrelated_spanish_prose_is_not_a_passport() {
    for text in PASSPORT_NOT_FOUND {
        assert_eq!(
            kept(text, EntityType::PassportNumber),
            0,
            "over-masked: {text:?}"
        );
    }
}

// --- CREDIT_CARD: French, Spanish and Italian negative context ---------

/// `4111111111111111` is the same Luhn-valid number the pack's own
/// `credit_card` fixture already uses. Each sentence puts it behind an
/// order or invoice label in a language whose negative-context vocabulary
/// was missing entirely before this change -- a checksum-coincidence
/// reference number, over-masked for want of a word the pack now carries.
const CREDIT_CARD_LABELLED_AS_A_REFERENCE: [&str; 6] = [
    "Votre numéro de commande 4111111111111111 a été confirmé.",
    "Numéro de facture 4111111111111111, à régler sous 30 jours.",
    "Su número de pedido 4111111111111111 ha sido confirmado.",
    "Número de factura 4111111111111111 pendiente de pago.",
    "Il suo numero d'ordine è 4111111111111111, confermato oggi.",
    "Numero di fattura 4111111111111111 in scadenza.",
];

#[test]
fn a_luhn_valid_number_labelled_as_an_order_or_invoice_is_not_a_credit_card() {
    for text in CREDIT_CARD_LABELLED_AS_A_REFERENCE {
        assert_eq!(
            kept(text, EntityType::CreditCard),
            0,
            "over-masked: {text:?}"
        );
    }
}

/// The other half of the same property: a real card, with no order or
/// invoice label anywhere nearby, still survives.
#[test]
fn an_unlabelled_card_number_still_survives() {
    for text in [
        "Merci de régler avec la carte 4111111111111111.",
        "El pago se realizó con la tarjeta 4111111111111111.",
        "Il pagamento con la carta 4111111111111111 è andato a buon fine.",
    ] {
        assert_eq!(kept(text, EntityType::CreditCard), 1, "not found: {text:?}");
    }
}

// --- IP_ADDRESS: the Spanish and Italian spelling of "version" ---------

#[test]
fn a_spanish_or_italian_version_string_is_not_an_ip_address() {
    for text in [
        "Actualiza a la versión 10.4.2.1 hoy mismo.",
        "Aggiorna alla versione 10.4.2.1 oggi stesso.",
    ] {
        assert_eq!(
            kept(text, EntityType::IpAddress),
            0,
            "over-masked: {text:?}"
        );
    }
}

// --- NATIONAL_ID: an English tracking number over an elfproef coincidence -

/// `123456782` is the same elfproef-valid BSN the pack's own `nl_bsn`
/// fixture already uses. English tracking-number prose was the one form
/// `negative_context` had no entry for at all.
#[test]
fn an_english_tracking_number_with_a_coincidental_bsn_checksum_is_not_a_national_id() {
    assert_eq!(
        kept(
            "Your tracking number 123456782 has shipped and is on its way.",
            EntityType::NationalId,
        ),
        0,
        "over-masked"
    );
}
