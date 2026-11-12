use crate::matcher::{build_automaton, matches_whole_term, neighbourhood};
use aho_corasick::AhoCorasick;
use phonenumber::country::Id;
use pii_detection_engine::candidate::{Candidate, Evidence, Source};
use pii_detection_engine::entity::EntityType;
use pii_detection_engine::text::Document;
use regex::Regex;
use std::str::FromStr;
use std::sync::OnceLock;

/// The regions a number without a country code might belong to, most likely
/// first. Mirrors `pii.phone_regions` in Temperate.
///
/// Eight plans consulted with `any` is a very wide net, and that width is why
/// the score below cannot be a constant: `9876543` is a valid German local
/// number, `123456789` a valid French one and `65929970001` a valid Dutch
/// one, so "some plan accepts it" is nearly free for any run of seven to
/// eleven digits. Validity is necessary and nowhere near sufficient, and the
/// written form is what carries the rest.
pub const DEFAULT_REGIONS: [&str; 8] = ["NL", "BE", "DE", "FR", "GB", "ES", "IT", "US"];

/// Anything shaped enough like a number to be worth asking about.
///
/// Deliberately permissive: the numbering plan decides, not this pattern. Its
/// only job is to bound how much text gets parsed, because parsing is the
/// expensive half.
fn candidates() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| {
        Regex::new(r"\+?\d[\d \-().]{5,20}\d")
            .unwrap_or_else(|error| panic!("the phone candidate pattern is a literal: {error}"))
    })
}

/// Words that say the digits beside them are a telephone number.
///
/// Matched as substrings against the same window a pack recognizer's context
/// terms are matched against, for the same reason: Dutch and German compound
/// and the Romance languages inflect, so `telefoonnummer`, `Telefonnummer`,
/// `telefónico`, `chiamami` and `llámanos` all carry the term with no word
/// boundary in sight. Every entry here is therefore a *stem*: short enough to
/// survive the endings its language adds, long enough that any word containing
/// it is still about telephones.
///
/// Nothing shorter than five characters goes in this list unless no word
/// contains it (`fax`, `gsm`, `sms`). The abbreviations people actually write
/// on a form -- `tel`, `cell`, `tfno`, `móvil` -- are three to five characters
/// and *are* contained in other words, so they live in
/// `BOUNDED_CONTEXT_TERMS` below and are matched on word boundaries instead.
///
/// ### Why this list is as wide as it is
///
/// Spain and Italy have no trunk prefix for mobiles, so a Spanish `612345678`
/// or an Italian `3401234567` carries neither of the two written-form signals
/// the ladder below rewards. A word is their only route over the bar, which
/// makes the vocabulary the *whole* of the recognizer for those two languages
/// -- and it carried three terms for them against seven each for Dutch and
/// German. That is a leak, not a preference: a real Spanish mobile in an
/// ordinary sentence went undetected. See
/// `training/reports/phone-es-it.md`.
///
/// ### What is deliberately absent
///
/// Generic contact nouns and verbs -- `contact`, `contacto`, `contattaci`,
/// `coordonnées`, `atención al cliente`. They label an email as readily as a
/// telephone number, and they sit just as naturally beside an order reference
/// (`contact us about invoice 4455667788`, which is a valid German number).
/// The specific labels that matter in practice already carry the word for the
/// thing: `teléfono de contacto` and `recapito telefonico` both match on their
/// telephone stem.
const CONTEXT_TERMS: [&str; 71] = [
    // en. `phone` covers `telephone`, `cellphone` and `smartphone`; `telephon`
    // is the stem the `-ic`/`-ique` adjectives need and English shares it
    // with French.
    "phone",
    "telephon",
    "fax",
    "sms",
    "call me",
    "call us",
    "reach me",
    "reach us",
    "text me",
    "hotline",
    "helpline",
    "landline",
    "voicemail",
    "whatsapp",
    "contact number",
    "mobile number",
    "mobile is",
    "my mobile",
    "cell number",
    "cell is",
    "my cell",
    // nl
    "telefoon",
    "mobiel",
    "gsm",
    "bereikbaar",
    "doorkies",
    "bel me",
    "bel mij",
    "bel ons",
    "bel naar",
    "bel gerust",
    "belt u",
    // de. Every entry names the number rather than the act of calling.
    // `anruf` and `rufen sie` were tried and dropped: `Anruf wegen Bestellung
    // 202400123` and `Rufen Sie wegen Rechnung 202400123 zurück` are ordinary
    // German, and both put a reference number a few words from the term.
    // German numbers carry a trunk zero and reach the bar on their written
    // form, so nothing is lost by keeping the German list to labels.
    "telefon",
    "rufnummer",
    "rückruf",
    "rueckruf",
    "durchwahl",
    "erreichbar",
    "handy",
    "festnetz",
    "mobilnummer",
    // fr. `téléphon` rather than `téléphone` so `téléphonique`, `téléphoner`
    // and `téléphonez` come with it. Bare `appeler` and `appelle` were tried
    // and dropped: French says `s'appeler` for "to be named", so `le document
    // s'appelle 202400123` carried both. The `au`/`le` that follows the verb
    // in the telephone idiom is what separates them.
    "téléphon",
    "portable",
    "appelez",
    "appeler au",
    "appeler le",
    "joignable",
    "joindre au",
    // es. Spanish moves the stress when it inflects -- `teléfono` but
    // `telefónico` -- so each spelling is its own stem, and both the accented
    // and the unaccented writing of each is here because real input carries
    // both. Bare `llama`/`llame`/`llamar` were tried and dropped: the first
    // two sit inside `llamativo` and are also the noun for a flame, and the
    // third carries `puede llamar para el pedido 987654321` -- calling *about*
    // an order rather than calling a number. The entries below follow the
    // pattern the list already used for `bel me`, `call us` and `appelez`:
    // the verb together with what is being called.
    "teléfon",
    "telefón",
    "celular",
    "llamar al",
    "llamarme al",
    "llamarnos al",
    "llámame",
    "llamame",
    "llámanos",
    "llamanos",
    "llámenos",
    "llamenos",
    "llame al",
    "llama al",
    // it. Bare `chiama` was tried and dropped: `chiamato`/`chiamata` is also
    // how Italian says "named", so `il file chiamato 202400123` carried it.
    // `recapito` went the same way -- it is a delivery address as readily as a
    // contact number, and `recapito del pacco 2024001234` is a parcel. The
    // telephone sense of both is written `recapito telefonico`, which the
    // `telefon` stem above already reads. `reperibile` needs its `al` for the
    // same reason: a document is `reperibile con il codice 202400123`, a
    // person is `reperibile al 3491234567`.
    //
    // `chiamare` is the one deliberate exception to the verb-plus-object rule
    // the lists above follow, and it is the widest term added here. `per
    // chiamare il servizio clienti indicare l'ordine 2024001234` puts a
    // reference number inside its window, and that sentence is ordinary
    // Italian rather than contrived. It stays because `per informazioni
    // chiamare il 3401234567` is how an Italian number is introduced, because
    // an Italian mobile has no written-form route over the bar at all, and
    // because the fixture's own preamble settles which way this trades: a
    // leak is a data-protection incident, over-masking is a quality problem.
    // It is the residual risk this change knowingly carries, and it is named
    // in `training/reports/phone-es-it.md` rather than left to be found.
    "chiamami",
    "chiamaci",
    "chiamatemi",
    "chiamateci",
    "chiamare",
    "cellulare",
    "cellulari",
    "reperibile al",
    "numero verde",
];

/// The abbreviations a form label uses, matched on word boundaries rather
/// than as substrings.
///
/// These are the terms the original list dropped rather than risk: `tel` sits
/// inside `hotel`, `Stellenbosch` and `intelligente`, `cell` inside
/// `excellent` and `cellar`, `mobile` inside `automobile`, `móvil` inside
/// `automóvil`. Dropping them cost Spain and Italy their commonest written
/// form -- `Tel.: 612 34 56 78`, `Cell. 340 123 4567`, `Tfno. 912345678` --
/// which is most of why those two languages leaked.
///
/// `matcher::matches_whole_term` is the same rule the negative-context lists
/// use, and it is used here for the same reason it was introduced there: a
/// term read out of the middle of a longer word is evidence nobody wrote.
/// The rule keys on alphanumerics, which is what lets a term *end* in
/// punctuation and still match: `tel.` is found in `Tel.: 612 34 56 78` and
/// not in `hotel.`
///
/// ### Why four of these carry their punctuation
///
/// A word boundary is not enough for `tel`, `cell` and `mob`, because each of
/// them is also an ordinary word standing on its own:
///
/// - French `tel que` / `tel quel` ("such as", "as is") is the determiner.
/// - Dutch `tel` is the verb "count" -- `ik tel 123456789 stuks`.
/// - English `cell` is a prison cell, a battery cell, a spreadsheet cell.
/// - English `mobile` is an adjective before it is a telephone: a mobile
///   home, a mobile app, an order paid for on mobile.
///
/// Each was measured firing on exactly those sentences with a reference
/// number in the same clause. What separates the label from the word is the
/// punctuation a form label carries, so the label is what is listed:
/// `tel.`, `tel:`, `cell.`, `cell:`, `mobile.`, `mobile:`. The cost is the
/// unpunctuated `Tel 612345678`, which is real but much the rarer writing,
/// and which the Spanish and Italian lists above still reach through
/// `teléfono`, `tfno`, `móvil` and `cellulare`. English keeps the prose forms
/// as phrases instead -- `my mobile`, `mobile is`, `mobile number` -- which
/// say telephone where the bare adjective does not.
///
/// The rest need no punctuation because they are not words in any of the six
/// languages: `tfno`, `tlfno`, `tlf` and `telf` are Spanish abbreviations,
/// `tél` is the French one -- accented, which is precisely what the French
/// determiner is not -- and `móvil`/`movil` and `mobile` name the thing.
const BOUNDED_CONTEXT_TERMS: [&str; 13] = [
    "tel.", "tel:", "cell.", "cell:", "mobile.", "mobile:", "tél", "tfno", "tlfno", "tlf", "telf",
    "móvil", "movil",
];

/// What a valid number scores on its shape alone: some numbering plan accepts
/// the digits, and nothing else is known.
///
/// Below `PHONE_NUMBER`'s 0.70 threshold on purpose, so validity alone cannot
/// decide. This is the `passport` recognizer's arrangement -- 0.1 base against
/// a 0.60 threshold, carried over the bar by a nearby word -- with a higher
/// floor because a numbering plan is stronger evidence than a run of seven to
/// nine characters.
const BASE_CONFIDENCE: f32 = 0.40;

/// A `+` and a country calling code that validates without being told which
/// country to assume. Nothing else in business prose is written this way.
///
/// 0.55 is the distance from validity to [`PENALISED_AGAINST`], which is what
/// every decisive signal below is worth. See that constant for why the three
/// are equal and what that costs.
const INTERNATIONAL_CONFIDENCE: f32 = 0.55;

/// A leading zero: the trunk prefix six of the eight regions above use for
/// national dialling, and a digit no reference number starts with by accident
/// -- `0000123456`, `0012345678` and `0001234567` are rejected by every plan
/// in the list, so a leading zero that survives validation is a trunk prefix.
///
/// Was 0.32, which decided a number only while nothing argued back. See
/// [`PENALISED_AGAINST`].
const TRUNK_CONFIDENCE: f32 = 0.55;

/// Written in separated groups rather than as one run. Reference, order and
/// customer numbers are written solid; telephone numbers are written spaced.
///
/// **This is not a rung, and after the recalibration below it decides
/// nothing at all.** It cannot carry a number on its own -- 0.60 against a
/// 0.70 threshold, deliberately, because `987 654 321` is a valid Spanish
/// landline written exactly the way a Spanish invoice reference is written --
/// and every signal it can be added to already clears the penalised bar
/// without it. What it still does is grade the `score` the caller receives
/// (`pii_detection_api::wire::Entity`), which is why it is kept rather than deleted:
/// `020 123 4567` is written more like a telephone number than `0612345678`
/// is, and the confidence says so. `the_ladder_never_lets_grouping_decide`
/// pins that it changes no outcome, so it cannot quietly become load-bearing
/// again.
const GROUPING_CONFIDENCE: f32 = 0.20;

/// A word nearby says telephone. The whole of the recognizer for the two
/// trunkless plans: a Spanish `612345678` or an Italian `3401234567` has
/// neither form signal above and is otherwise indistinguishable from a
/// customer number.
///
/// Was 0.35, which left exactly those two languages -- the ones with no
/// written-form route over the bar at all -- as the only ones a single word
/// could argue back down. See [`PENALISED_AGAINST`].
const CONTEXT_CONFIDENCE: f32 = 0.55;

/// `PHONE_NUMBER` in `crates/resolve/thresholds.toml`, restated so the
/// weights above can be checked against it where they are written.
///
/// Restating a number is normally how two files drift apart, so the real
/// relationship is pinned through the real resolver reading the real file, in
/// `tests/thresholds.rs`. That test is what catches the drift; this constant
/// is what makes the ladder below readable beside the weights it grades.
const CALIBRATED_AGAINST: f32 = 0.70;

/// `pii_detection_resolve::NEGATIVE_CONTEXT_PENALTY`, restated for the same reason
/// and pinned the same way.
///
/// The resolver used to *delete* a candidate a negative-context term stood
/// beside, ahead of the threshold. Under a delete a score could not argue,
/// so the exact score of a phone number barely mattered and this ladder was
/// only ever calibrated against `CALIBRATED_AGAINST`. The resolver now takes
/// this much off the score it weighs against the threshold and nothing else,
/// which means every rung below has two meanings that were never separated:
/// "decides when nothing argues" and "decides at all".
const NEGATIVE_CONTEXT_PENALTY: f32 = 0.25;

/// The bar a signal has to clear to decide a number against one word that
/// says otherwise -- 0.95, and the number this ladder is now calibrated to.
///
/// ### Why every decisive signal is worth exactly the same
///
/// Ask of each signal: may one nearby word take this number back off the
/// mask? For a `+31` international form, a Dutch `06`, and a Spanish
/// `612345678` behind the word `Teléfono`, the answer is the same no -- each
/// is a real telephone number, and a leak is a data-protection incident
/// while over-masking is a quality problem. So each is worth the distance
/// from validity to this bar, `0.95 - 0.40 = 0.55`, and no more, because
/// there is no third bar above this one for a larger number to clear.
///
/// The old 0.55/0.32/0.35 spread read as a gradation and was not one: no
/// threshold sat between 0.72 and 0.95, so those differences decided nothing
/// and only looked like they did. What replaces the gradation is a single
/// yes-or-no -- a signal either decides a number or does not -- and
/// `GROUPING_CONFIDENCE` is the only one that does not.
///
/// ### What this buys, stated as a property
///
/// The only combination that reaches `CALIBRATED_AGAINST` without a decisive
/// signal is validity plus grouping, and that reaches 0.60. So **every phone
/// candidate that clears its threshold clears the penalised threshold too**:
/// no negative-context term, in any pack, present or future, can change a
/// phone answer. `phone-calibration.md` argued for that position and could
/// only protect it by leaving the recognizer's negative vocabulary empty and
/// hoping nobody added one; it is now a property of the weights.
/// `no_written_form_clears_the_bar_and_then_fails_it_against_one_word` walks
/// all sixteen combinations, and `tests/suppression.rs` runs the real
/// resolver over real sentences with the flag a term would set.
const PENALISED_AGAINST: f32 = CALIBRATED_AGAINST + NEGATIVE_CONTEXT_PENALTY;

/// The calibration, as a compile error rather than a comment.
///
/// Each line is one rung: which signals, on their own, are enough to mask,
/// and -- since `df77acb` made suppression a penalty rather than a delete --
/// enough to *stay* masked with a word arguing against them.
///
/// The shape of this block is the whole design: validity is never enough,
/// each of the three strong signals is conclusive, and grouping is never
/// either, because a date and an order reference are both written in groups.
/// Promoting grouping for the plans that have no trunk prefix was considered
/// and rejected on a measurement rather than a preference: `987654321` -- one
/// of the five false positives the first calibration removed, and a customer
/// number in the fixture -- is a valid Spanish 9xx landline, and written
/// `987 654 321`, the way an invoice reference is just as often written, that
/// rule would mask it again. The vocabulary above is the cheaper half of the
/// same fix: it only fires where somebody wrote the word.
const _: () = {
    // Validity is never enough, with or without anything arguing.
    assert!(BASE_CONFIDENCE < CALIBRATED_AGAINST);
    // Each decisive signal decides, and keeps deciding against one word.
    assert!(BASE_CONFIDENCE + INTERNATIONAL_CONFIDENCE >= PENALISED_AGAINST);
    assert!(BASE_CONFIDENCE + TRUNK_CONFIDENCE >= PENALISED_AGAINST);
    assert!(BASE_CONFIDENCE + CONTEXT_CONFIDENCE >= PENALISED_AGAINST);
    // Grouping is not one of them, and is the only signal that is not --
    // which is what makes the property in `PENALISED_AGAINST` exhaustive
    // rather than a claim about the combinations somebody thought of.
    assert!(BASE_CONFIDENCE + GROUPING_CONFIDENCE < CALIBRATED_AGAINST);
    // It is still worth something, because it still grades the confidence
    // the caller is handed. At zero it would be a deleted signal pretending
    // to be a weighed one.
    assert!(GROUPING_CONFIDENCE > 0.0);
    // A penalty that is not a penalty would make every line above vacuous.
    assert!(PENALISED_AGAINST > CALIBRATED_AGAINST);
};

/// How a candidate is written, which is the evidence a numbering plan cannot
/// give.
struct WrittenForm {
    international: bool,
    trunk: bool,
    grouped: bool,
}

fn written_form(trimmed: &str) -> WrittenForm {
    let international = trimmed.starts_with('+');
    let body = trimmed.trim_start_matches('+');
    WrittenForm {
        international,
        // Only meaningful for a national number: after `+31` the `31` is a
        // country code and the trunk prefix is gone by convention.
        trunk: !international && body.starts_with('0'),
        grouped: body.chars().any(|c| !c.is_ascii_digit()),
    }
}

/// The ladder, as one function, so the property the weights are chosen for
/// can be walked over every combination rather than over the combinations a
/// test author happened to write a sentence for.
///
/// Clamped, because a number that is international *and* grouped *and*
/// labelled sums past 1.0 and a confidence above one is not a confidence.
fn ladder(form: &WrittenForm, context: bool) -> f32 {
    let mut score = BASE_CONFIDENCE;
    if form.international {
        score += INTERNATIONAL_CONFIDENCE;
    }
    if form.trunk {
        score += TRUNK_CONFIDENCE;
    }
    if form.grouped {
        score += GROUPING_CONFIDENCE;
    }
    if context {
        score += CONTEXT_CONFIDENCE;
    }
    score.clamp(0.0, 1.0)
}

/// Whether the candidate reads as a calendar date rather than a number.
///
/// Three groups, one repeated separator, and either `d?d sep m?m sep yyyy`
/// (either order for the first two, since `12-31-2025` and `31-12-2025` are
/// the same date in different conventions) or `yyyy sep m?m sep d?d`. The year
/// group must be four digits beginning 1 or 2, which is what keeps a genuine
/// five-group number like `01 42 68 53 00` and a two-group one like
/// `020-1234567` out of reach of this rule.
///
/// The cost is a national number written in exactly three groups whose last
/// group is a four-digit 1xxx or 2xxx and whose first two are at most two
/// digits each -- `06-12-2345`. No plan in `DEFAULT_REGIONS` accepts eight
/// digits in that shape, so nothing real is known to be lost, and a date is a
/// far commoner string in the texts this service sees.
fn is_date(trimmed: &str) -> bool {
    // Split by hand rather than by pattern: three groups joined by *the same*
    // separator is a backreference, which the regex crate does not support,
    // and "any three separators" would read `01-02 2026` as a date.
    let separator = match trimmed.chars().find(|c| "-./ ".contains(*c)) {
        Some(separator) => separator,
        None => return false,
    };
    let groups: Vec<&str> = trimmed.split(separator).collect();
    if groups.len() != 3
        || !groups
            .iter()
            .all(|g| !g.is_empty() && g.bytes().all(|b| b.is_ascii_digit()))
    {
        return false;
    }
    let part = |group: &str| (group.len(), group.parse::<u32>().unwrap_or(0));
    let (first_len, first) = part(groups[0]);
    let (second_len, second) = part(groups[1]);
    let (last_len, last) = part(groups[2]);
    let year = |len: usize, value: u32| len == 4 && (1000..=2999).contains(&value);
    let day_or_month = |len: usize, value: u32| len <= 2 && (1..=31).contains(&value);

    // `dd sep mm sep yyyy`, in either of the two orders the world writes it.
    if year(last_len, last)
        && day_or_month(first_len, first)
        && day_or_month(second_len, second)
        && (first <= 12 || second <= 12)
    {
        return true;
    }
    // `yyyy sep mm sep dd`.
    year(first_len, first)
        && second_len <= 2
        && (1..=12).contains(&second)
        && day_or_month(last_len, last)
}

pub struct PhoneRecognizer {
    regions: Vec<Id>,
    context: Option<AhoCorasick>,
    /// The abbreviations, held apart from `context` because they are asked a
    /// different question of the same window: whole term rather than
    /// substring.
    bounded_context: Option<AhoCorasick>,
}

impl PhoneRecognizer {
    pub fn new(regions: &[&str]) -> Self {
        Self {
            regions: regions
                .iter()
                .filter_map(|code| Id::from_str(code).ok())
                .collect(),
            context: build_automaton(&CONTEXT_TERMS),
            bounded_context: build_automaton(&BOUNDED_CONTEXT_TERMS),
        }
    }

    /// Every span this returns is a span the flat-scored version returned
    /// too. The two rejections below only remove candidates, and the flat 0.9
    /// cleared the 0.70 threshold unconditionally, so everything that used to
    /// be masked was masked -- which makes what survives here a subset of it.
    /// Over-masking therefore cannot get worse; recall is the only number that
    /// can, and `training/reports/phone-calibration.md` is where it is watched.
    pub fn scan(&self, document: &Document) -> Vec<Candidate> {
        let text = document.as_str();
        let mut found: Vec<Candidate> = Vec::new();
        let mut names_a_person: Option<bool> = None;

        for hit in candidates().find_iter(text) {
            // Trim trailing separators the greedy pattern swept up, so the
            // returned span is the number and not the punctuation after it.
            let trimmed = hit.as_str().trim_end_matches([' ', '-', '.', '(', ')']);
            if trimmed.is_empty() {
                continue;
            }
            let end = hit.start() + trimmed.len();

            // A telephone number is never written glued to a letter or
            // continued by another digit. Without this, `AB9876543` yields
            // the seven-digit fragment `9876543` -- a valid German local
            // number -- and the tracking code it came from is masked as a
            // phone number, which is the truncation failure `pii-detection-resolve`
            // warns about arriving from a different direction. A digit on
            // either side means the pattern's 22-character ceiling cut a
            // longer run, and a longer run is not a number either.
            if !is_free_standing(text, hit.start(), end) {
                continue;
            }

            if is_date(trimmed) {
                continue;
            }

            let form = written_form(trimmed);
            // For a `+` number the region hint is noise: the country code is
            // in the string. Parsing without one also means the eight-region
            // `any` cannot manufacture agreement it did not earn.
            let parsed = if form.international {
                phonenumber::parse(None, trimmed)
                    .ok()
                    .filter(phonenumber::is_valid)
            } else {
                self.regions.iter().find_map(|region| {
                    phonenumber::parse(Some(*region), trimmed)
                        .ok()
                        .filter(phonenumber::is_valid)
                })
            };
            let Some(number) = parsed else {
                continue;
            };

            let Some(span) = document.char_span(hit.start(), end) else {
                continue;
            };
            if found.iter().any(|existing| existing.span == span) {
                continue;
            }

            let window = neighbourhood(text, hit.start(), end);
            // Two searches of one window: the stems as substrings, the
            // abbreviations on word boundaries. Either one is the same
            // evidence and buys the same rung of the ladder -- `Teléfono` and
            // `Tfno.` say the same thing.
            let context = self
                .context
                .as_ref()
                .is_some_and(|automaton| automaton.is_match(&window.text))
                || self
                    .bounded_context
                    .as_ref()
                    .is_some_and(|automaton| matches_whole_term(automaton, &window));

            let score = ladder(&form, context);

            let (from, to) = crate::attribution::window(text, hit.start(), end);
            let around = &text[from..to];
            let (scope_from, scope_to) = crate::attribution::scope(text);
            let company_nearby = crate::attribution::company_mentioned(around)
                && crate::attribution::company_line_wording(around)
                && crate::attribution::introduced(around, hit.start() - from);
            // Personal if any accepting region reads it that way, not only the one `scan` scores.
            let type_is_personal = if form.international {
                international_reads_personal(&number)
            } else {
                any_region_reads_personal(&self.regions, trimmed)
            };
            let personal_signal = type_is_personal
                || crate::attribution::personal_wording(around)
                || crate::attribution::greeting(&text[scope_from..scope_to])
                || crate::attribution::named_around(around, end - from)
                || *names_a_person.get_or_insert_with(|| {
                    crate::attribution::names_a_person(&text[scope_from..scope_to])
                });

            found.push(Candidate {
                span,
                entity_type: EntityType::PhoneNumber,
                country: None,
                source: Source::Rule { id: "phone" },
                score,
                evidence: Evidence {
                    format: true,
                    // A numbering plan defines no check digit. `None` is the
                    // honest answer and keeps the resolver's
                    // `checksum != Some(false)` filter out of this.
                    checksum: None,
                    context,
                    // Deliberately never set, and the one of the three
                    // properties named in `rules-engine-fixture.md` that is
                    // left alone. Unlike a card number behind the word
                    // `ordernummer`, a number *written as a telephone number*
                    // behind the word `factuur` is still a telephone number.
                    // `training/reports/phone-calibration.md` has the
                    // measurement: the packs' whole existing negative
                    // vocabulary suppresses 2 of the 5 over-masking
                    // violations, reaching 5 needs terms invented from the
                    // fixture's own labels, and the ladder above removes all
                    // 5 without inventing anything.
                    //
                    // It is now an empty list rather than a hole. Under the
                    // recalibration (see `PENALISED_AGAINST`) a term here
                    // could not delete anything: everything that clears the
                    // threshold clears the penalised threshold too, and
                    // everything below it was already rejected. So a future
                    // list would be inert rather than dangerous -- which is
                    // the reason not to add one, in place of the fear that
                    // used to be.
                    negative_context: false,
                    company_nearby,
                    personal_signal,
                    scope: document.char_span(scope_from, scope_to),
                },
            });
        }

        // Two numbers in one text is a list of lines, not the company's one line. Only a number
        // that would be kept counts, so an invoice or register number beside it does not.
        let scopes: Vec<_> = found
            .iter()
            .filter(|candidate| candidate.score >= CALIBRATED_AGAINST)
            .map(|candidate| candidate.evidence.scope)
            .collect();
        for candidate in &mut found {
            let shared = scopes
                .iter()
                .filter(|other| **other == candidate.evidence.scope)
                .count();
            if shared > 1 {
                candidate.evidence.company_nearby = false;
            }
        }

        found.sort_by_key(|candidate| (candidate.span.start, candidate.span.end));
        found
    }
}

/// Whether the match is a token in its own right rather than a slice of a
/// longer one.
///
/// Alphanumeric on either side disqualifies it: a letter means the digits
/// belong to a reference code, a digit means the candidate pattern's ceiling
/// cut a longer run in half.
fn is_free_standing(text: &str, start: usize, end: usize) -> bool {
    let before = text[..start].chars().next_back();
    let after = text[end..].chars().next();
    before.is_none_or(|c| !c.is_alphanumeric()) && after.is_none_or(|c| !c.is_alphanumeric())
}

/// Whether any region that accepts a national-form number reads it as personal.
fn any_region_reads_personal(regions: &[Id], trimmed: &str) -> bool {
    regions.iter().any(|region| {
        phonenumber::parse(Some(*region), trimmed)
            .ok()
            .filter(phonenumber::is_valid)
            .is_some_and(|number| reads_personal(&number))
    })
}

fn international_reads_personal(number: &phonenumber::PhoneNumber) -> bool {
    reads_personal(number)
}

// Personal unless the crate's own reading, or every descriptor match found, is a published type.
fn reads_personal(number: &phonenumber::PhoneNumber) -> bool {
    use phonenumber::Type;
    let own_kind = number.number_type(&phonenumber::metadata::DATABASE);
    if own_kind != Type::Unknown && !crate::attribution::published_type(own_kind) {
        return true;
    }
    let candidates = phonenumber::metadata::DATABASE
        .by_code(&number.code().value())
        .unwrap_or_default();
    let kept_zeros = number.national().to_string();
    let dropped_zeros = number.national().value().to_string();
    let values = [kept_zeros, dropped_zeros];
    let matches: Vec<Type> = candidates
        .iter()
        .flat_map(|meta| values.iter().map(move |value| (*meta, value)))
        .flat_map(|(meta, value)| {
            descriptor_types()
                .iter()
                .map(move |&kind| (meta, value, kind))
        })
        .filter(|(meta, value, kind)| {
            meta.descriptors()
                .get(*kind)
                .is_some_and(|descriptor| descriptor.is_match(value))
        })
        .map(|(.., kind)| kind)
        .collect();
    matches.is_empty()
        || matches
            .iter()
            .any(|&kind| !crate::attribution::published_type(kind))
}

/// Every type with a descriptor of its own. A type the crate adds fails to compile until `slot`
/// places it, and the assertion keeps `ALL` in the order `slot` gives.
fn descriptor_types() -> &'static [phonenumber::Type] {
    use phonenumber::Type;
    const ALL: [Type; 15] = [
        Type::FixedLine,
        Type::Mobile,
        Type::TollFree,
        Type::PremiumRate,
        Type::SharedCost,
        Type::PersonalNumber,
        Type::Voip,
        Type::Pager,
        Type::Uan,
        Type::Emergency,
        Type::Voicemail,
        Type::ShortCode,
        Type::StandardRate,
        Type::Carrier,
        Type::NoInternational,
    ];
    /// Its index in `ALL`, or `NONE` for the two that share another's descriptor or have none.
    const NONE: usize = usize::MAX;
    const fn slot(kind: Type) -> usize {
        match kind {
            Type::FixedLine => 0,
            Type::Mobile => 1,
            Type::TollFree => 2,
            Type::PremiumRate => 3,
            Type::SharedCost => 4,
            Type::PersonalNumber => 5,
            Type::Voip => 6,
            Type::Pager => 7,
            Type::Uan => 8,
            Type::Emergency => 9,
            Type::Voicemail => 10,
            Type::ShortCode => 11,
            Type::StandardRate => 12,
            Type::Carrier => 13,
            Type::NoInternational => 14,
            Type::FixedLineOrMobile | Type::Unknown => NONE,
        }
    }
    const _: () = {
        let mut index = 0;
        while index < ALL.len() {
            assert!(slot(ALL[index]) == index);
            index += 1;
        }
    };
    &ALL
}

#[cfg(test)]
mod tests {
    use super::*;
    use pii_detection_engine::text::Document;

    fn scan(text: &str) -> Vec<pii_detection_engine::candidate::Candidate> {
        PhoneRecognizer::new(&DEFAULT_REGIONS).scan(&Document::new(text.to_string()))
    }

    /// The score the resolver would compare against `PHONE_NUMBER = 0.70`,
    /// or None where nothing was recognized at all.
    fn score(text: &str) -> Option<f32> {
        scan(text).first().map(|candidate| candidate.score)
    }

    fn kept(text: &str) -> bool {
        score(text).is_some_and(|score| score >= CALIBRATED_AGAINST)
    }

    /// The same question one word later: whether the number would still be
    /// kept with a negative-context term standing beside it, which is what
    /// the resolver now weighs (`pii_detection_resolve::NEGATIVE_CONTEXT_PENALTY`)
    /// where it used to delete.
    ///
    /// Arithmetic here, because this module's tests are about the ladder.
    /// What catches the restated constant drifting from the resolver's own is
    /// `the_penalty_this_module_restates_is_the_one_the_resolver_applies`
    /// below; what checks these sentences end to end is
    /// `tests/suppression.rs`, over the real matcher and the real resolver.
    fn kept_against_a_word(text: &str) -> bool {
        score(text).is_some_and(|score| score - NEGATIVE_CONTEXT_PENALTY >= CALIBRATED_AGAINST)
    }

    #[test]
    fn an_international_number_is_found() {
        let document = Document::new("Bel +31 6 12345678 vandaag".to_string());
        let found = PhoneRecognizer::new(&DEFAULT_REGIONS).scan(&document);
        assert_eq!(found.len(), 1);
        assert_eq!(document.slice(found[0].span), "+31 6 12345678");
    }

    #[test]
    fn a_national_number_is_read_in_the_first_region_that_accepts_it() {
        let found = scan("Bel 020-1234567 voor vragen");
        assert_eq!(found.len(), 1);
    }

    #[test]
    fn a_run_of_digits_that_is_no_numbering_plan_is_not_a_phone_number() {
        assert!(scan("Order 999999999999999 shipped").is_empty());
    }

    #[test]
    fn the_region_list_decides_whether_a_national_number_is_a_number_at_all() {
        // A Dutch national number is nothing in the North American plan, which
        // has no numbers beginning with zero. If this passes under both region
        // lists, the list is decoration and the recognizer ignores it.
        let document = Document::new("Bel 020-1234567 voor vragen".to_string());
        assert_eq!(PhoneRecognizer::new(&["NL"]).scan(&document).len(), 1);
        assert_eq!(PhoneRecognizer::new(&["US"]).scan(&document).len(), 0);
    }

    // --- the written forms a real number takes, per country -----------------
    //
    // Every one of these must clear the threshold on its own. The fixture is
    // not the gate here: these are the forms the numbering plans and national
    // conventions define, written the way the countries write them.

    #[test]
    fn an_international_number_clears_the_threshold_on_its_own() {
        for text in [
            "+31 20 123 4567",
            "+31 6 12345678",
            "+49 30 123456",
            "+49 30 1234567",
            "+33 1 42 68 53 00",
            "+32 2 511 12 34",
            "+32 495 12 34 56",
            "+44 7911 123456",
            "+34 612 34 56 78",
            "+39 06 6982",
            "+1 415 226 4321",
            "+3120 1234567",
        ] {
            assert!(kept(text), "{text:?} scored {:?}", score(text));
            assert!(
                kept_against_a_word(text),
                "{text:?} is argued out of the mask by one word: {:?}",
                score(text)
            );
        }
    }

    #[test]
    fn a_national_number_in_its_country_s_own_written_form_clears_the_threshold() {
        for text in [
            // nl: trunk zero, grouped or solid
            "020 123 4567",
            "020-1234567",
            "06 12345678",
            "06-12345678",
            "0612345678",
            // de
            "030 12345678",
            "030-12345678",
            "03012345678",
            // fr
            "01 42 68 53 00",
            "01.42.68.53.00",
            "0142685300",
            // be
            "02 511 12 34",
            "0499 12 34 56",
            // gb
            "07911 123456",
            "020 7946 0958",
        ] {
            assert!(kept(text), "{text:?} scored {:?}", score(text));
            assert!(
                kept_against_a_word(text),
                "{text:?} is argued out of the mask by one word: {:?}",
                score(text)
            );
        }
    }

    #[test]
    fn a_trunkless_national_number_needs_a_word_that_says_telephone() {
        // Spain and the United States have no trunk prefix, so a solid run of
        // their digits carries neither form signal and is indistinguishable
        // from a customer number. The word decides, exactly as it decides a
        // passport number.
        for bare in ["612345678", "4152264321"] {
            assert!(!kept(bare), "{bare:?} scored {:?}", score(bare));
            for labelled in [
                format!("Teléfono {bare}"),
                format!("phone {bare}"),
                format!("Call us on {bare}"),
            ] {
                assert!(
                    kept(&labelled),
                    "{labelled:?} scored {:?}",
                    score(&labelled)
                );
                assert!(
                    kept_against_a_word(&labelled),
                    "{labelled:?} is argued out of the mask by one word: {:?}",
                    score(&labelled)
                );
            }
        }
    }

    #[test]
    fn a_word_that_says_telephone_is_read_out_of_a_compound() {
        // `telefoonnummer` and `Telefonnummer` contain the term with no word
        // boundary; a boundary rule would miss both.
        assert!(kept("Telefoonnummer 612345678"));
        assert!(kept("Telefonnummer 612345678"));
    }

    // --- the two trunkless plans --------------------------------------------
    //
    // Spain has no trunk prefix at all and Italy has none for mobiles, so a
    // Spanish number and an Italian mobile carry neither of the two written
    // forms that decide a number on their own. The vocabulary is the whole of
    // the recognizer for them, which is why these are written from the
    // numbering plans rather than from any fixture -- no fixture has a
    // Spanish or Italian case.

    /// Every one of these is a real Spanish national number: 6xx and 7xx
    /// mobiles, 9xx and 8xx landlines, in the groupings Spain writes them in.
    const SPANISH_NATIONAL: [&str; 8] = [
        "612345678",
        "612 34 56 78",
        "722 12 34 56",
        "912345678",
        "913 45 67 89",
        "934 12 34 56",
        "900 123 456",
        "800123456",
    ];

    /// Italian mobiles, which are 3xx and carry no leading zero.
    const ITALIAN_MOBILE: [&str; 4] = ["3401234567", "340 123 4567", "349 123 4567", "3331234567"];

    #[test]
    fn a_spanish_number_is_carried_by_the_words_spanish_writing_uses() {
        for bare in SPANISH_NATIONAL {
            assert!(
                !kept(bare),
                "{bare:?} must not decide on its shape: {:?}",
                score(bare)
            );
            for written in [
                format!("Teléfono: {bare}"),
                format!("Telefono: {bare}"),
                format!("Tel. {bare}"),
                format!("Tel: {bare}"),
                format!("Tfno. {bare}"),
                format!("Tfno.: {bare}"),
                format!("Tlf: {bare}"),
                format!("Telf. {bare}"),
                format!("Móvil: {bare}"),
                format!("Movil: {bare}"),
                format!("Mi móvil es {bare}"),
                format!("Llámame al {bare}"),
                format!("Llamame al {bare}"),
                format!("Puedes llamar al {bare}"),
                format!("Llame al {bare} para más información"),
                format!("Llámanos al {bare}"),
                format!("Atención telefónica en el {bare}"),
                format!("Número de teléfono de contacto: {bare}"),
                format!("Escríbenos por WhatsApp al {bare}"),
                format!("Celular {bare}"),
            ] {
                assert!(kept(&written), "{written:?} scored {:?}", score(&written));
                assert!(
                    kept_against_a_word(&written),
                    "{written:?} is argued out of the mask by one word: {:?}",
                    score(&written)
                );
            }
        }
    }

    #[test]
    fn an_italian_number_is_carried_by_the_words_italian_writing_uses() {
        for bare in ITALIAN_MOBILE {
            assert!(
                !kept(bare),
                "{bare:?} must not decide on its shape: {:?}",
                score(bare)
            );
            for written in [
                format!("Telefono: {bare}"),
                format!("Tel. {bare}"),
                format!("Tel: {bare}"),
                format!("Cell. {bare}"),
                format!("Cell: {bare}"),
                format!("Cellulare {bare}"),
                format!("Il mio cellulare è {bare}"),
                format!("Chiamami al {bare}"),
                format!("Chiamaci al {bare}"),
                format!("Può chiamare il {bare}"),
                format!("Recapito telefonico: {bare}"),
                format!("Sono reperibile al {bare}"),
                format!("Numero telefonico {bare}"),
                format!("Scrivimi su WhatsApp al {bare}"),
            ] {
                assert!(kept(&written), "{written:?} scored {:?}", score(&written));
                assert!(
                    kept_against_a_word(&written),
                    "{written:?} is argued out of the mask by one word: {:?}",
                    score(&written)
                );
            }
        }
    }

    #[test]
    fn an_italian_landline_keeps_its_leading_zero_and_decides_on_its_own() {
        // Italy is the exception among trunkless plans: the leading zero of a
        // landline is part of the number rather than a prefix that is dropped,
        // so `06 6982` reaches the bar on its written form and needs no word.
        // Only the mobiles above depend on the vocabulary.
        for text in [
            "06 6982",
            "0612345678",
            "06 1234 5678",
            "02 1234 5678",
            "0212345678",
            "081 123 4567",
        ] {
            assert!(kept(text), "{text:?} scored {:?}", score(text));
            assert!(
                kept_against_a_word(text),
                "{text:?} is argued out of the mask by one word: {:?}",
                score(text)
            );
        }
    }

    #[test]
    fn a_spanish_or_italian_number_in_international_form_needs_no_word() {
        for text in [
            "+34 612 34 56 78",
            "+34 912 34 56 78",
            "+34 91 234 56 78",
            "+34612345678",
            "+39 340 123 4567",
            "+39 06 6982",
            "+39 02 1234 5678",
            "+393401234567",
        ] {
            assert!(kept(text), "{text:?} scored {:?}", score(text));
            assert!(
                kept_against_a_word(text),
                "{text:?} is argued out of the mask by one word: {:?}",
                score(text)
            );
        }
    }

    // --- abbreviations, spellings, and the boundaries they need -------------

    #[test]
    fn a_form_label_is_read_as_a_word_that_says_telephone() {
        // What a label looks like rather than what a sentence looks like: a
        // field name, its punctuation, and the number. Each of these is the
        // only evidence in its string.
        for text in [
            "Tel. 612345678",
            "Tel: 612345678",
            "TEL. 612345678",
            "Tel.: 612 34 56 78",
            "Tél. 612345678",
            "Tél : 612345678",
            "Tfno. 612345678",
            "Tlf. 612345678",
            "Tlfno: 612345678",
            "Telf. 612345678",
            "Cell. 3401234567",
            "Cell: 340 123 4567",
            "Mobile: 612345678",
            "Móvil: 612345678",
            "Movil - 612345678",
            "Fax 612345678",
            "SMS 612345678",
            "WhatsApp 612345678",
        ] {
            assert!(kept(text), "{text:?} scored {:?}", score(text));
            assert!(
                kept_against_a_word(text),
                "{text:?} is argued out of the mask by one word: {:?}",
                score(text)
            );
        }
    }

    #[test]
    fn an_unpunctuated_abbreviation_is_the_price_of_the_boundary_rule() {
        // Recorded rather than discovered. `tel`, `cell` and `mobile` are
        // listed with the punctuation a form label carries, because bare they
        // are also the French determiner `tel que`, the Dutch verb `tel`, an
        // English prison cell and an English adjective -- each measured firing
        // on a reference number in `tests/trunkless.rs`. The cost is the
        // label written with no punctuation at all, which is real writing and
        // is not found:
        assert!(!kept("TEL 612345678"));
        assert!(!kept("Cell 3401234567"));

        // It is a narrow gap rather than a hole, because the words these
        // abbreviate are in the list and the abbreviations Spain uses need no
        // punctuation -- none of them is a word in any supported language.
        assert!(kept("Telefono 612345678"));
        assert!(kept("Cellulare 3401234567"));
        assert!(kept("Tfno 612345678"));
        assert!(kept("Movil 612345678"));
    }

    #[test]
    fn a_short_label_buried_in_a_longer_word_says_nothing() {
        // The hazard the abbreviations were kept out of the substring list
        // for. Every one of these contains an abbreviation and none of them
        // says telephone; a substring search would carry all of them over the
        // bar. This is the same failure the negative-context lists already
        // shipped once, arriving in the opposite direction.
        for text in [
            "Hotel 612345678",
            "Hotels 612345678",
            "Stellenbosch 612345678",
            "Un servizio intelligente 612345678",
            "Intelligence 612345678",
            "Excellent 612345678",
            "Cellar 612345678",
            "Automobile 612345678",
            "Automóvil 612345678",
            "Inmovilizado 612345678",
            "Bestellen 612345678",
            "Vertellen 612345678",
            "Mobiliario 612345678",
        ] {
            assert!(!kept(text), "{text:?} scored {:?}", score(text));
        }
    }

    #[test]
    fn both_spellings_of_an_accented_word_are_read() {
        // Real input carries both, and the ones that arrive without accents
        // are not the rare case: keyboards, form validation and copy-paste all
        // strip them.
        for (accented, plain) in [
            ("Teléfono 612345678", "Telefono 612345678"),
            ("Llámame al 612345678", "Llamame al 612345678"),
            ("Llámanos al 612345678", "Llamanos al 612345678"),
            ("Móvil 612345678", "Movil 612345678"),
            ("Atención telefónica: 612345678", "Telefonica: 612345678"),
            ("Tél. 612345678", "Tel. 612345678"),
            ("Rückruf unter 612345678", "Rueckruf unter 612345678"),
        ] {
            assert!(kept(accented), "{accented:?} scored {:?}", score(accented));
            assert!(kept(plain), "{plain:?} scored {:?}", score(plain));
        }
    }

    #[test]
    fn an_uppercase_label_is_read() {
        // The window is lowercased with Unicode rules before the search, so
        // `É` folds to `é` and an all-caps form label still matches. An ASCII
        // fold alone would miss every accented one of these.
        for text in [
            "TELÉFONO: 612345678",
            "MÓVIL: 612345678",
            "TFNO: 612345678",
            "TÉL. 612345678",
            "CELL. 3401234567",
            "LLÁMAME AL 612345678",
        ] {
            assert!(kept(text), "{text:?} scored {:?}", score(text));
        }
    }

    #[test]
    fn every_supported_language_carries_a_trunkless_number_in_an_ordinary_sentence() {
        // The leak this vocabulary closed, stated as a property rather than a
        // list of terms. The detector supports six languages; a plain sentence in
        // each of them, around a number that has no written-form signal at
        // all, must reach the bar. The fixture cannot check this -- it has no
        // Spanish or Italian cases.
        for text in [
            // en
            "You can reach me on 6123456789 after five.",
            "My mobile is 6123456789, ring any time.",
            // nl
            "Mijn mobiel is 612345678, bel gerust.",
            "Ik ben bereikbaar op 612345678.",
            // de. German numbers carry a trunk zero and are decided by their
            // written form, so the German list is kept to terms that label
            // the number rather than name the act of calling: `Anruf wegen
            // Bestellung 202400123` is ordinary German too.
            "Meine Handynummer ist 612345678.",
            "Erreichbar unter 612345678 ab neun Uhr.",
            // fr
            "Mon portable est le 612345678.",
            "Vous pouvez me joindre au 612345678.",
            // es
            "Mi teléfono es el 612345678, llámame cuando quieras.",
            "Puedes llamar al 912345678 en horario de oficina.",
            // it
            "Il mio cellulare è 3401234567, chiamami pure.",
            "Sono reperibile al 3491234567 dopo le sei.",
        ] {
            assert!(kept(text), "{text:?} scored {:?}", score(text));
            assert!(
                kept_against_a_word(text),
                "{text:?} is argued out of the mask by one word: {:?}",
                score(text)
            );
        }
    }

    // --- the near-misses ----------------------------------------------------

    #[test]
    fn a_reference_number_is_not_a_phone_number() {
        // Every one of these is accepted by at least one of the eight
        // numbering plans, which is why validity alone could never decide.
        for text in [
            "Factuur 123456789 is verzonden",
            "Ordernummer 987654321",
            "Klantnummer 192837465",
            "Belegnummer 65929970001 ist keiner Person zugeordnet",
            "Customer 4455667788",
            "Reference 55512345678",
            "Dossier 8123456789",
            "Kenmerk 123456789",
        ] {
            assert!(!kept(text), "{text:?} scored {:?}", score(text));
        }
    }

    #[test]
    fn a_reference_code_with_a_letter_prefix_yields_no_fragment() {
        // The candidate pattern starts at the first digit, so without a
        // free-standing check `AB9876543` offers up `9876543` -- a valid
        // German local number -- and the code is masked through a fragment.
        for text in [
            "Order XR1234567 shipped and tracking AB9876543 is active",
            "Sendung DE1234567 unterwegs",
            "Ref AB12345678 bevestigd",
            "Batch X0612345678 ready",
        ] {
            assert!(scan(text).is_empty(), "{text:?} -> {:?}", scan(text));
        }
    }

    #[test]
    fn a_rejected_reference_code_does_not_take_a_real_number_down_with_it() {
        // The candidate pattern reads across spaces, so a rejection is also a
        // decision about the text after it. A number standing beside a code
        // that was thrown out must still be found.
        for (text, expected) in [
            ("Ref AB9876543 bel 030 12345678 vandaag", "030 12345678"),
            ("Order XR1234567, tel 020 123 4567", "020 123 4567"),
            ("Vergadering 01-02-2026, bel 020 123 4567", "020 123 4567"),
        ] {
            let document = Document::new(text.to_string());
            let found = PhoneRecognizer::new(&DEFAULT_REGIONS).scan(&document);
            assert_eq!(found.len(), 1, "{text:?} -> {found:?}");
            assert_eq!(document.slice(found[0].span), expected);
        }
    }

    #[test]
    fn a_digit_run_longer_than_any_numbering_plan_yields_no_fragment() {
        // The candidate pattern reads at most 22 characters. A longer run must
        // not be answered with the first 22 of it.
        for text in [
            "Hash 123456789012345678901234567890",
            "Sequence 06123456780612345678",
        ] {
            assert!(scan(text).is_empty(), "{text:?} -> {:?}", scan(text));
        }
    }

    #[test]
    fn a_date_is_not_a_phone_number() {
        for text in [
            "01-02-2026",
            "31.12.2025",
            "12-31-2025",
            "2026-08-21",
            "2026.08.21",
            "01 02 2026",
            "2026 08 21",
            "1-2-2026",
            "09-11-2001",
        ] {
            assert!(scan(text).is_empty(), "{text:?} -> {:?}", scan(text));
        }
    }

    #[test]
    fn a_date_stays_a_date_even_next_to_a_word_that_says_telephone() {
        // The date rule is a rejection, not a score, so context cannot argue
        // it back. `Telefonisch besproken op 01-02-2026` would otherwise reach
        // 0.40 + 0.32 + 0.20 + 0.35.
        assert!(scan("Telefonisch besproken op 01-02-2026").is_empty());
    }

    #[test]
    fn a_number_written_in_groups_a_date_never_takes_survives_the_date_rule() {
        // Two groups, five groups, and three groups whose last is not a year:
        // none of them is a date and all of them are numbers.
        assert!(kept("020-1234567"));
        assert!(kept("01 42 68 53 00"));
        assert!(kept("020 123 4567"));
    }

    #[test]
    fn an_id_sequence_is_not_a_phone_number() {
        for text in [
            "BSN 123456782",
            "Steuer-ID 65929970000",
            "IBAN NL91ABNA0417164300",
            "Kaart 4111 1111 1111 1111",
            "SSN 123-45-6789",
            "Postcode 1012 LP",
        ] {
            assert!(!kept(text), "{text:?} scored {:?}", score(text));
        }
    }

    #[test]
    fn a_spanish_or_italian_reference_number_is_not_a_phone_number() {
        // The other side of the widened vocabulary. Every one of these digit
        // runs is accepted by at least one of the eight plans -- `987654321`
        // is a valid Spanish 9xx landline, `2024001234` a valid German
        // number, `12345678901` a valid American one -- so validity decides
        // none of them, and none of them carries a word that says telephone.
        for text in [
            // es
            "Pedido 987654321 enviado",
            "Número de pedido 202400123",
            "Factura 2024001234 pendiente de pago",
            "Referencia 4455667788",
            "Número de cliente 192837465",
            "Expediente 123456789 archivado",
            "Albarán 987654321",
            "Número de seguimiento 12345678901",
            // it
            "Ordine 2024001234 spedito",
            "Numero d'ordine 987654321",
            "Fattura 202400123 da saldare",
            "Riferimento cliente 4455667788",
            "Partita IVA 12345678901",
            "Pratica 123456789 chiusa",
            "Codice pratica 12345678903",
        ] {
            assert!(!kept(text), "{text:?} scored {:?}", score(text));
        }
    }

    #[test]
    fn a_spanish_or_italian_reference_number_written_in_groups_is_still_not_a_phone_number() {
        // The rung the ladder deliberately does not add. `987 654 321` is a
        // valid Spanish landline written exactly the way a Spanish invoice
        // reference is written, so validity plus grouping cannot be allowed to
        // decide for a trunkless plan -- it would mask every one of these.
        for text in [
            "Pedido 987 654 321 enviado",
            "Factura 202 400 123",
            "Referencia 44 55 66 77 88",
            "Ordine 987 654 321",
            "Fattura 2024 001 234",
            "Riferimento 123 456 789",
        ] {
            assert!(!kept(text), "{text:?} scored {:?}", score(text));
        }
    }

    #[test]
    fn a_spanish_or_italian_identifier_yields_no_phone_number_at_all() {
        // Tax and VAT identifiers in these two countries carry letters, so
        // the free-standing rule does the work rather than the score: a
        // Spanish NIF ends in a check letter, a CIF begins with one, an
        // Italian codice fiscale is alphanumeric throughout, and both VAT
        // numbers wear their country prefix. None of them may give up the
        // digit run inside it.
        for text in [
            "NIF 12345678Z",
            "DNI 12345678Z de la solicitante",
            "CIF B12345678",
            "IBAN ES9121000418450200051332",
            "IVA IT12345678901",
            "Codice fiscale RSSMRA85M01H501Z",
            "P. IVA IT12345678903 iscritta",
        ] {
            assert!(scan(text).is_empty(), "{text:?} -> {:?}", scan(text));
        }
    }

    #[test]
    fn a_zero_padded_reference_number_is_not_a_trunk_prefixed_number() {
        for text in [
            "Order 0000123456",
            "Ref 0012345678",
            "Nummer 0001234567",
            "Code 0000000123",
        ] {
            assert!(!kept(text), "{text:?} scored {:?}", score(text));
        }
    }

    // --- the calibration itself ---------------------------------------------

    #[test]
    fn validity_alone_does_not_clear_the_threshold() {
        // The whole fix in one property. `123456789` is a valid French number
        // and `9876543` a valid German one; if either clears the bar on
        // validity alone, the flat score is back under another name.
        for bare in ["123456789", "9876543", "987654321", "65929970001"] {
            assert_eq!(
                score(bare),
                Some(BASE_CONFIDENCE),
                "{bare:?} is scored on more than its validity"
            );
            assert!(!kept(bare));
        }
    }

    #[test]
    fn the_evidence_records_what_the_score_was_built_from() {
        let international = scan("Bel +31 20 123 4567");
        assert_eq!(international.len(), 1);
        assert!(international[0].evidence.format);
        assert_eq!(international[0].evidence.checksum, None);
        // A numbering plan defines no negative context, and this recognizer
        // sets none: see the note in `scan`.
        assert!(!international[0].evidence.negative_context);

        let with_context = scan("Telefoon 612345678");
        assert_eq!(with_context.len(), 1);
        assert!(with_context[0].evidence.context);

        let without_context = scan("Klantnummer 612345678");
        assert_eq!(without_context.len(), 1);
        assert!(!without_context[0].evidence.context);
    }

    #[test]
    fn an_invoice_label_cannot_argue_down_a_number_written_as_a_number() {
        // The two sentences `phone-calibration.md` named as the reason this
        // recognizer carries no negative vocabulary. A real number beside the
        // word `factuur` is still a real number, and now it is one whichever
        // way the packs are edited: both clear the bar, and both clear it
        // again with a term standing beside them. The first is the one that
        // did not, at 0.92 against a penalised bar of 0.95.
        for text in [
            "Vragen over de factuur? Bel 020-1234567",
            "Rechnungsnummer 4471: Rückruf unter 030 12345678",
            // Trunkless, so its word is the only evidence there is -- the
            // case a penalty was always going to reach first.
            "Factura 2024001234. Teléfono de contacto: 612345678",
            "Fattura 202400123. Chiamami al 3401234567",
        ] {
            assert!(kept(text), "{text:?} scored {:?}", score(text));
            assert!(
                kept_against_a_word(text),
                "{text:?} is argued out of the mask by one word: {:?}",
                score(text)
            );
        }
    }

    // --- the two bars, and which signals reach which -------------------------

    /// The property the recalibration exists for, over every combination
    /// rather than over the ones somebody wrote a sentence for.
    ///
    /// A negative-context term used to delete a candidate outright; it now
    /// takes `NEGATIVE_CONTEXT_PENALTY` off the score the resolver weighs.
    /// That split every rung of this ladder in two -- "decides when nothing
    /// argues" and "decides at all" -- and the ladder had only ever been
    /// calibrated against the first. Under the recalibration the two are the
    /// same rung again: nothing may clear `CALIBRATED_AGAINST` and then fail
    /// it one word later.
    ///
    /// Two of the sixteen are unreachable (`written_form` never sets `trunk`
    /// on an international number) and are walked anyway: an unreachable
    /// combination that scores wrongly is a defect waiting for the day it
    /// becomes reachable.
    #[test]
    fn no_written_form_clears_the_bar_and_then_fails_it_against_one_word() {
        for bits in 0..16u8 {
            let form = WrittenForm {
                international: bits & 1 != 0,
                trunk: bits & 2 != 0,
                grouped: bits & 4 != 0,
            };
            let context = bits & 8 != 0;
            let score = ladder(&form, context);
            if score >= CALIBRATED_AGAINST {
                assert!(
                    score >= PENALISED_AGAINST,
                    "combination {bits:04b} masks at {score} and is argued back out at \
                     {}, under the {PENALISED_AGAINST} a term costs",
                    score - NEGATIVE_CONTEXT_PENALTY
                );
            }
        }
    }

    /// Which signals are rungs, stated as the four separate claims it takes
    /// to say it. A test that only checked the three positives would pass on
    /// a ladder that made everything decisive, including grouping.
    #[test]
    fn each_decisive_signal_decides_alone_and_grouping_never_does() {
        let form = |international, trunk, grouped| WrittenForm {
            international,
            trunk,
            grouped,
        };
        assert!(ladder(&form(true, false, false), false) >= PENALISED_AGAINST);
        assert!(ladder(&form(false, true, false), false) >= PENALISED_AGAINST);
        assert!(ladder(&form(false, false, false), true) >= PENALISED_AGAINST);
        assert!(ladder(&form(false, false, true), false) < CALIBRATED_AGAINST);
        assert!(ladder(&form(false, false, false), false) < CALIBRATED_AGAINST);
    }

    /// The one test in this module that runs the resolver, because it is the
    /// only place that can: it compares the penalty this module *restates*
    /// against the one `pii-detection-resolve` actually applies, and the restated
    /// constant is private to this file.
    ///
    /// Restating a number is how two files drift apart. Every other test
    /// here does arithmetic against `PENALISED_AGAINST`, so all of them
    /// would keep passing if the resolver's penalty moved and this file's
    /// copy did not; this is what fails instead. Both directions are pinned:
    /// a candidate at the bar survives the real resolver reading the real
    /// `thresholds.toml`, and one hundredth under it does not.
    #[test]
    fn the_penalty_this_module_restates_is_the_one_the_resolver_applies() {
        let suppressed = |score: f32| {
            let candidate = Candidate {
                span: pii_detection_engine::text::CharSpan { start: 0, end: 11 },
                entity_type: EntityType::PhoneNumber,
                country: None,
                source: Source::Rule {
                    id: "phone-penalty",
                },
                score,
                evidence: Evidence {
                    format: true,
                    checksum: None,
                    context: false,
                    negative_context: true,
                    ..Evidence::default()
                },
            };
            pii_detection_resolve::Resolver::new()
                .resolve(vec![candidate], &[EntityType::PhoneNumber], "auto")
                .len()
        };

        assert_eq!(
            suppressed(PENALISED_AGAINST),
            1,
            "the bar this ladder is built to is under the resolver's real one"
        );
        assert_eq!(
            suppressed(PENALISED_AGAINST - 0.01),
            0,
            "the bar this ladder is built to is over the resolver's real one"
        );
    }

    fn evidence(text: &str) -> pii_detection_engine::candidate::Evidence {
        scan(text).first().expect("a phone number").evidence
    }

    #[test]
    fn a_company_landline_is_company_and_not_personal() {
        let found = evidence(
            "de klantenservice van Noordkade Energie B.V., bereikbaar op 020 123 4567 tussen 9 en 17 uur.",
        );
        assert!(found.company_nearby);
        assert!(!found.personal_signal);
        assert!(found.scope.is_some());
    }

    #[test]
    fn a_person_named_just_outside_the_window_is_still_read_from_the_pronoun() {
        let found = evidence(
            "Beste team, Jan de Vries van Acme BV heeft gevraagd om de factuur naar Kerkstraat 12, 1016 GV Amsterdam te sturen. Zijn e-mail is jan.devries@example.nl en hij is bereikbaar op +31 20 123 4567.",
        );
        assert!(found.personal_signal);
    }

    #[test]
    fn the_scope_is_the_whole_text_even_across_a_blank_line() {
        let text = "Jan de Vries\n\nKlantenservice van Acme BV, bereikbaar op 020 123 4567.";
        let scope = evidence(text).scope.expect("a scope");
        assert_eq!(scope.start, 0);
        assert_eq!(scope.end as usize, text.chars().count());
    }

    #[test]
    fn a_company_without_wording_for_its_own_line_is_not_enough() {
        for text in [
            "Best regards,\nSarah Jones\n\nThistledown Hardware Ltd\n020 7946 0958",
            "Un saludo,\nMaría García\nFerretería Almendros S.L.\nTel. +34 915 234 567",
            "Cordiali saluti,\nGiulia\nFalegnameria Bosco S.r.l.\nTel. 02 1234 5678",
            "Het nummer van onze accountmanager bij Noordkade Energie B.V. is 020 123 4567.",
            "Home phone: 020 7946 0958. Employer: Thistledown Hardware Ltd.",
            "You can reach me at Thistledown Hardware Ltd on 020 7946 0958.",
        ] {
            let found = evidence(text);
            assert!(!found.company_nearby || found.personal_signal, "{text}");
        }
    }

    #[test]
    fn a_name_between_the_company_and_the_number_stops_it_clearing() {
        for text in [
            "Helpdesk Noordkade Energie B.V.\nLinda Jansen 020 123 4567",
            "Klantenservice Noordkade Energie B.V. (Linda) 020 123 4567",
            "Assistenza Trasporti Adriatico S.p.A., referente Marco Rossi, 02 1234 5678",
            "Your support contact at Thistledown Hardware Ltd is Tom, reachable on 020 7946 0958.",
            "Klantenservice Noordkade Energie B.V., Keizersgracht 100, 1015 AA Amsterdam, T 020 555 0100",
        ] {
            assert!(!evidence(text).company_nearby, "{text}");
        }
    }

    #[test]
    fn the_third_hunt_stays_masked() {
        for text in [
            "Best regards,\n\nSarah Jones\n\nCustomer Support | Thistledown Hardware Ltd\n020 7946 0958",
            "Best,\nTom\nSupport Thistledown Hardware Ltd\n020 7946 0958",
            "Un saluto,\nMarco\nAssistenza clienti\nTrasporti Adriatico S.p.A.\n02 1234 5678",
            "Tom's landline at Thistledown Hardware Ltd support: 020 7946 0958",
            "Helpdesk Noordkade Energie B.V. 020 123 4567 (Linda)",
            "Name;Company;Department;Phone\nTom;Thistledown Hardware Ltd;Support;020 7946 0958",
            "Customer support at Thistledown Hardware Ltd is Al, on 020 7946 0958.",
            "Kundendienst Sabine Meier e.K., Tel. 089 3456789",
        ] {
            let found = evidence(text);
            assert!(!found.company_nearby || found.personal_signal, "{text}");
        }
    }

    #[test]
    fn a_given_name_the_model_may_miss_keeps_the_number_masked() {
        for text in [
            "Tom from customer support at Thistledown Hardware Ltd on 020 7946 0958",
            "Tom\nSupport Thistledown Hardware Ltd\n020 7946 0958",
            "Uwe sitzt im Kundenservice der Isar Metallbau GmbH, unter 089 3456789.",
            "Marco Rossi lavora all'assistenza di Trasporti Adriatico S.p.A., allo 02 1234 5678.",
            "Tom Harris Ltd, customer service: 020 7946 0958",
        ] {
            let found = evidence(text);
            assert!(!found.company_nearby || found.personal_signal, "{text}");
        }
    }

    #[test]
    fn a_company_named_after_a_family_name_still_clears() {
        let found = evidence("Atención al cliente de Talleres Martín S.L.: +34 912 34 56 78.");
        assert!(found.company_nearby);
        assert!(!found.personal_signal);
    }

    #[test]
    fn an_everyday_word_the_name_lists_hold_does_not_stop_a_company_line_clearing() {
        let found = evidence(
            "Im Jahr 2024 haben wir das Glas aus Holz gerahmt. Kundenservice der Isar Metallbau GmbH, Tel. 089 3456789",
        );
        assert!(found.company_nearby);
        assert!(!found.personal_signal);
    }

    #[test]
    fn an_everyday_word_used_as_a_surname_keeps_the_number_masked() {
        for text in [
            "Die Anfrage hat Holz übernommen. Kundenservice der Isar Metallbau GmbH, Tel. 089 3456789",
            "Heute hat der Holz Dienst. Kundenservice der Isar Metallbau GmbH, Tel. 089 3456789",
        ] {
            let found = evidence(text);
            assert!(!found.company_nearby || found.personal_signal, "{text}");
        }
    }

    #[test]
    fn a_greeting_anywhere_in_the_text_marks_a_signature() {
        let found = evidence(
            "Kind regards,\nTom\nCustomer Support\nThistledown Hardware Ltd\n020 7946 0958",
        );
        assert!(found.personal_signal);
    }

    #[test]
    fn two_numbers_in_one_text_are_a_list() {
        let found = scan("Helpdesk Noordkade Energie B.V.: 020 123 4567 of 020 123 4568");
        assert_eq!(found.len(), 2);
        assert!(
            found
                .iter()
                .all(|candidate| !candidate.evidence.company_nearby)
        );
    }

    #[test]
    fn a_label_or_reachable_at_may_stand_between() {
        for text in [
            "de klantenservice van Noordkade Energie B.V., bereikbaar op 020 123 4567 tussen 9 en 17 uur.",
            "den Kundenservice der Lindenberg Haustechnik GmbH unter 030 12345678, werktags.",
            "le service client de Vallée Électronique SARL au +33 1 42 68 53 00.",
            "atención al cliente de Ferretería Almendros S.L. en el +34 912 34 56 78.",
            "Invoice from Thistledown Hardware Ltd, support 020 7946 0999.",
            "Klantenservice Noordkade Energie B.V. - Tel.: 020 123 4567",
        ] {
            assert!(evidence(text).company_nearby, "{text}");
        }
    }

    #[test]
    fn a_mobile_is_personal_whoever_stands_beside_it() {
        for text in [
            "customer service at Thistledown Hardware Ltd on +44 7911 123456, Monday to Friday.",
            "il servizio clienti di Falegnameria Bosco S.r.l. al +39 320 123 4567.",
            "klantenservice van Noordkade Energie B.V. op 06 12345678.",
        ] {
            let found = evidence(text);
            assert!(found.company_nearby, "{text}");
            assert!(found.personal_signal, "{text}");
        }
    }

    #[test]
    fn a_uk_mobile_stays_personal_even_though_germany_reads_it_as_a_fixed_line() {
        let found = evidence("Customer service at Thistledown Hardware Ltd: 07911 123456.");
        assert!(found.company_nearby);
        assert!(found.personal_signal);
    }

    #[test]
    fn an_unclassifiable_foreign_mobile_stays_personal() {
        let found = evidence("Customer service at Thistledown Hardware Ltd: 00225 07 12 34 56 78.");
        assert!(found.company_nearby);
        assert!(found.personal_signal);
    }

    #[test]
    fn a_second_unclassifiable_foreign_mobile_stays_personal() {
        let found = evidence("Customer service at Thistledown Hardware Ltd: 00242 06 612 3456.");
        assert!(found.company_nearby);
        assert!(found.personal_signal);
    }

    #[test]
    fn a_benin_number_that_matches_both_fixed_line_and_mobile_stays_personal() {
        for phone in ["+229 01 25 12 34 56", "+229 01 29 87 65 43"] {
            let found = evidence(&format!("Klantenservice Noordkade Energie B.V.: {phone}."));
            assert!(found.company_nearby, "{phone}");
            assert!(found.personal_signal, "{phone}");
        }
    }

    #[test]
    fn an_italian_number_that_only_reads_as_voicemail_with_its_zero_dropped_stays_personal() {
        for phone in ["+39 0372 8999 1074", "0372 8999 1074"] {
            let found = evidence(&format!("Klantenservice Noordkade Energie B.V.: {phone}."));
            assert!(found.company_nearby, "{phone}");
            assert!(found.personal_signal, "{phone}");
        }
    }

    #[test]
    fn the_french_business_phone_benchmark_case_stays_not_personal() {
        let found = evidence(
            "Pour toute question sur la livraison, contactez le service client de Vallée Électronique SARL au +33 1 42 68 53 00, du lundi au vendredi.",
        );
        assert!(found.company_nearby);
        assert!(!found.personal_signal);
    }

    #[test]
    fn the_spanish_business_phone_benchmark_case_stays_not_personal() {
        let found = evidence(
            "Para cualquier consulta sobre el envío, contacte con atención al cliente de Ferretería Almendros S.L. en el +34 912 34 56 78, de lunes a viernes.",
        );
        assert!(found.company_nearby);
        assert!(!found.personal_signal);
    }

    #[test]
    fn a_dutch_landline_that_only_reads_as_unknown_elsewhere_stays_publishable() {
        let found = evidence("Klantenservice: Noordkade Energie B.V., 020 123 4567.");
        assert!(found.company_nearby);
        assert!(!found.personal_signal);
    }

    #[test]
    fn a_uan_style_german_number_that_only_reads_as_unknown_elsewhere_stays_publishable() {
        let found = evidence("Kundenservice Lindenberg Haustechnik GmbH, 030 12345678");
        assert!(found.company_nearby);
        assert!(!found.personal_signal);
    }

    #[test]
    fn an_international_italian_landline_stays_publishable_despite_the_shared_vatican_code() {
        let found =
            evidence("Servizio clienti Falegnameria Bosco S.r.l., al numero +39 06 1234 5678.");
        assert!(found.company_nearby);
        assert!(!found.personal_signal);
    }

    #[test]
    fn first_person_wording_is_personal_on_a_landline() {
        let found =
            evidence("Klantenservice van Havenstad Logistiek B.V. druk? Bel mij op 010 234 5678.");
        assert!(found.personal_signal);
    }

    #[test]
    fn a_number_with_no_company_near_it_is_not_company() {
        assert!(!evidence("Het nummer is 020 765 4321.").company_nearby);
    }

    #[test]
    fn attribution_never_changes_the_score() {
        let with = scan("Noordkade Energie B.V., bereikbaar op 020 123 4567.");
        let without = scan("Bereikbaar op 020 123 4567.");
        assert_eq!(with[0].score, without[0].score);
    }

    /// Grouping is kept for the confidence it grades and for nothing else,
    /// so it is pinned as deciding nothing: flipping it never moves a
    /// candidate across either bar. If a later edit makes it load-bearing --
    /// promoting `987 654 321` for a trunkless plan, which is the edit this
    /// recognizer has refused twice -- this fails and says so.
    #[test]
    fn the_ladder_never_lets_grouping_decide() {
        for bits in 0..8u8 {
            let solid = WrittenForm {
                international: bits & 1 != 0,
                trunk: bits & 2 != 0,
                grouped: false,
            };
            let grouped = WrittenForm {
                grouped: true,
                ..solid
            };
            let context = bits & 4 != 0;
            let (a, b) = (ladder(&solid, context), ladder(&grouped, context));
            // Below the clamp it must still move the confidence -- that is
            // the whole of the job it has left. At the clamp (international
            // and trunk together already sum past 1.0) there is nowhere for
            // it to move to.
            if a < 1.0 {
                assert!(b > a, "grouping {bits:03b} changed no score at all");
            }
            assert_eq!(
                a >= CALIBRATED_AGAINST,
                b >= CALIBRATED_AGAINST,
                "grouping decided {bits:03b} at the plain bar"
            );
            assert_eq!(
                a >= PENALISED_AGAINST,
                b >= PENALISED_AGAINST,
                "grouping decided {bits:03b} at the penalised bar"
            );
        }
    }
}
