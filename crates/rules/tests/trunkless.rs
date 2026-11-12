//! The two numbering plans that have no trunk prefix to offer, end to end.
//!
//! Spain writes no trunk zero at all and Italy writes none on a mobile, so a
//! Spanish `612345678` and an Italian `3401234567` reach `PHONE_NUMBER`'s 0.70
//! threshold on a nearby word or not at all -- unlike a Dutch `06`, a German
//! `030` or a French `01`, which their written form carries. That makes the
//! context vocabulary in `phone.rs` the *whole* of the recognizer for those two
//! languages.
//!
//! **The 128-case fixture cannot check any of this.** Its cases are en, nl, de
//! and fr only, so no measurement over it moves when the Spanish or Italian
//! vocabulary changes -- which is how the vocabulary came to carry three terms
//! for these two languages against seven each for Dutch and German without
//! anything failing. This file is the gate instead, and it is written from the
//! national numbering plans and from how the two languages are actually
//! written, not from any fixture.
//!
//! Three sets, and all three matter:
//!
//! 1. numbers that must be found, in the forms Spain and Italy write them;
//! 2. references that must not be, each of them a digit run some numbering
//!    plan does accept, so validity decides none of them;
//! 3. sentences where a context term appears in a *non-telephone* sense. A
//!    recognizer that finds everything passes set 1, and one that finds
//!    nothing passes sets 2 and 3; only all three together say anything.
//!
//! Measured before and after the vocabulary was widened, in
//! `training/reports/phone-es-it.md`: set 1 went from 15/44 to 44/44 while
//! sets 2 and 3 stayed clean.

use pii_detection_engine::entity::EntityType;
use pii_detection_engine::text::Document;
use pii_detection_resolve::Resolver;
use pii_detection_rules::matcher::Matcher;

/// Real Spanish and Italian numbers in ordinary writing, and the labels a form
/// puts in front of them. Every one must survive the whole pipeline.
const FOUND: [&str; 44] = [
    // es -- mobiles (6xx, 7xx), in sentences
    "Mi teléfono es el 612345678, llámame cuando quieras.",
    "Mi telefono es el 612345678.",
    "Puedes llamar al 612 34 56 78 por la tarde.",
    "Llámame al 722123456 esta noche.",
    "Llamame al 722 12 34 56 esta noche.",
    "Llámanos al 600123456 para cualquier duda.",
    "Mi móvil es 612345678.",
    "Mi movil es 612345678.",
    "Escríbeme por WhatsApp al 612345678.",
    "Mi número de teléfono de contacto es 612345678.",
    "Atención telefónica en el 912345678.",
    // es -- landlines (9xx) and freephone (900)
    "Llame al 912345678 en horario de oficina.",
    "Teléfono: 913 45 67 89",
    "Telefono: 934 12 34 56",
    "Puede llamar al 900 123 456 sin coste.",
    "Nuestro celular de guardia es 612345678.",
    // es -- form labels, which is where the abbreviations live
    "Tel. 612345678",
    "Tel: 912345678",
    "Tfno. 612345678",
    "Tfno.: 912 34 56 78",
    "Tlf: 612345678",
    "Tlfno. 612345678",
    "Telf. 912345678",
    "Móvil: 612345678",
    "Movil: 612345678",
    "TELÉFONO: 612345678",
    "MÓVIL 612345678",
    // it -- mobiles (3xx), in sentences
    "Il mio cellulare è 3401234567, chiamami pure.",
    "Il mio cellulare e 340 123 4567.",
    "Chiamami al 3491234567 dopo le sei.",
    "Chiamaci al 333 123 4567 per informazioni.",
    "Può chiamare il 3401234567 in orario di ufficio.",
    "Sono reperibile al 3491234567 tutto il giorno.",
    "Il mio recapito telefonico è 340 123 4567.",
    "Recapito telefonico: 3401234567",
    "Scrivimi su WhatsApp al 3401234567.",
    "Numero telefonico 3331234567",
    // it -- form labels
    "Cell. 3401234567",
    "Cell: 340 123 4567",
    "Cellulare 3491234567",
    "Tel. 3401234567",
    "CELL. 3401234567",
    // international form, which needed no word even before
    "Puede llamarnos al +34 612 34 56 78 cuando quiera.",
    "Il numero è +39 340 123 4567.",
];

/// References, identifiers and dates. Every digit run here is accepted by at
/// least one of the eight numbering plans -- `987654321` is a valid Spanish
/// 9xx landline, `2024001234` a valid German number, `12345678901` a valid
/// American one -- so none of them is decided by validity, and none of them
/// carries a word that says telephone.
const NOT_FOUND: [&str; 32] = [
    // es -- references, solid and grouped
    "El pedido 987654321 se envió ayer desde el almacén.",
    "Pedido 987 654 321 enviado.",
    "Número de pedido 202400123 confirmado.",
    "Factura 2024001234 pendiente de pago.",
    "Factura 202 400 123 emitida.",
    "Referencia 4455667788 del contrato.",
    "Referencia 44 55 66 77 88 del contrato.",
    "Número de cliente 192837465 en el sistema.",
    "Expediente 123456789 archivado.",
    "Albarán 987654321 recibido.",
    "Número de seguimiento 12345678901.",
    "Su reserva 2024001234 está confirmada.",
    // es -- identifiers, which carry letters and must yield no fragment
    "NIF 12345678Z de la solicitante.",
    "DNI 12345678Z verificado.",
    "CIF B12345678 registrado.",
    "IBAN ES9121000418450200051332",
    // es -- dates and zero padding
    "Reunión confirmada para el 01-02-2026.",
    "Publicado el 2026-08-21.",
    "Código 0000123456 asignado.",
    // it -- references, solid and grouped
    "L'ordine 2024001234 è stato spedito.",
    "Numero d'ordine 987654321 confermato.",
    "Ordine 987 654 321 spedito.",
    "Fattura 202400123 da saldare.",
    "Fattura 2024 001 234 emessa.",
    "Riferimento cliente 4455667788.",
    "Pratica 123456789 chiusa.",
    "Codice pratica 12345678903.",
    // it -- identifiers and dates
    "Partita IVA 12345678901 iscritta al registro.",
    "P. IVA IT12345678903 iscritta.",
    "Codice fiscale RSSMRA85M01H501Z",
    "Riunione confermata per il 01-02-2026.",
    "Documento del 2026-08-21 approvato.",
];

/// The other direction the vocabulary can fail in, and the one a wider list
/// makes likelier: a term appearing in a sense that has nothing to do with
/// telephones, with a reference number in the same clause.
///
/// Every sentence here was measured firing before the term that caused it was
/// narrowed. They are the reason `tel`, `cell`, `mob` and `mobile` carry their
/// punctuation, why `llama`, `llame`, `llamar`, `appeler`, `appelle`,
/// `chiama`, `recapito`, `cellular`, `anruf` and `rufen sie` are not in the
/// list at all, and why `reperibile` needs its `al`.
const NOT_A_TELEPHONE_SENSE: [&str; 20] = [
    // fr: `tel que` / `tel quel` is the determiner, not the abbreviation
    "Un document tel que le bon de commande 202400123",
    "Les pièces telles que la facture 4455667788",
    "Tel quel, le dossier 123456789 est clos",
    // fr: `s'appeler` is how French says "to be named"
    "Le document s'appelle 202400123",
    // nl: `tel` is the verb "count"
    "Tel de dagen af tot 123456789",
    "Ik tel 123456789 stuks in het magazijn",
    // de: calling *about* an order is not calling a number
    "Anruf wegen Bestellung 202400123",
    "Rufen Sie wegen Rechnung 202400123 zurueck",
    // es: `llamativo` is "striking", `llama` is a flame, and one calls about
    // an order as readily as one calls a number
    "Un descuento llamativo en el pedido 987654321",
    "La llama del pedido 987654321",
    "Puede llamar para el pedido 987654321",
    "La llamada de atencion sobre el pedido 987654321",
    // it: `recapitare` is to deliver, a `recapito` is an address, and
    // `reperibile` is also how a document is findable
    "Recapitare il pacco 2024001234 entro venerdì",
    "Recapito del pacco 2024001234",
    "Il documento e reperibile con il codice 202400123",
    "Il file chiamato 202400123 e corrotto",
    // en: the ordinary senses of the abbreviations
    "The mob at 4455667788 Main Street",
    "Mobile home 4455667788",
    "Paid on mobile, order 4455667788",
    "Cellular biology sample 4455667788",
];

fn phones(matcher: &Matcher, resolver: &Resolver, text: &str) -> usize {
    let document = Document::new(text.to_string());
    resolver
        .resolve(matcher.scan(&document), &[EntityType::PhoneNumber], "auto")
        .len()
}

#[test]
fn a_spanish_or_italian_number_in_ordinary_writing_is_found() {
    let (matcher, resolver) = (Matcher::new(), Resolver::new());
    for text in FOUND {
        assert_eq!(
            phones(&matcher, &resolver, text),
            1,
            "not found: {text:?}. Spain and Italy reach the threshold on a \
             nearby word or not at all -- see phone::CONTEXT_TERMS."
        );
    }
}

#[test]
fn a_spanish_or_italian_reference_is_not_found() {
    let (matcher, resolver) = (Matcher::new(), Resolver::new());
    for text in NOT_FOUND {
        assert_eq!(
            phones(&matcher, &resolver, text),
            0,
            "over-masked: {text:?}"
        );
    }
}

#[test]
fn a_context_term_in_a_non_telephone_sense_decides_nothing() {
    let (matcher, resolver) = (Matcher::new(), Resolver::new());
    for text in NOT_A_TELEPHONE_SENSE {
        assert_eq!(
            phones(&matcher, &resolver, text),
            0,
            "a term fired in a sense that is not a telephone: {text:?}"
        );
    }
}
