//! Whose number a phone number is, read from the text around it. The resolver
//! makes the decision; see the phone attribution spec in `docs/superpowers/specs/`.

use aho_corasick::{AhoCorasick, MatchKind};
use regex::Regex;
use std::collections::HashSet;
use std::sync::OnceLock;
use unicode_normalization::UnicodeNormalization;
use unicode_normalization::char::is_combining_mark;

const REACH: usize = 150;

/// Longest first, so `GmbH & Co. KG` wins over `GmbH` and `S.L.U.` over `S.L.`.
const SUFFIXES: [&str; 44] = [
    "GmbH & Co. KG",
    "S.L.U.",
    "S.p.A.",
    "S.r.l.",
    "S.a.s.",
    "S.n.c.",
    "V.O.F.",
    "Limited",
    "B.V.",
    "N.V.",
    "C.V.",
    "S.A.",
    "S.L.",
    "Corp.",
    "Inc.",
    "Ltd.",
    "SASU",
    "SARL",
    "EURL",
    "BVBA",
    "GmbH",
    "B.V",
    "SpA",
    "Srl",
    "SRL",
    "SLU",
    "SAS",
    "SNC",
    "OHG",
    "PLC",
    "plc",
    "LLP",
    "LLC",
    "Inc",
    "Ltd",
    "VOF",
    "BV",
    "NV",
    "CV",
    "SA",
    "SL",
    "AG",
    "KG",
    "UG",
];

const PERSONAL: [&str; 249] = [
    // Honorifics.
    "dhr.",
    "mevr.",
    "herr",
    "herrn",
    "frau",
    "mr",
    "mr.",
    "mrs",
    "mrs.",
    "ms",
    "ms.",
    "dr.",
    "m.",
    "mme",
    "mme.",
    "sr.",
    "sra.",
    "sig.",
    "sig.ra",
    // nl
    "bel mij",
    "mijn nummer",
    "mijn mobiele",
    "direct nummer",
    "privé",
    "mobiel",
    "geboortedatum",
    "woonadres",
    // de
    "ruf mich an",
    "meine nummer",
    "durchwahl",
    "privat",
    "handy",
    "geburtsdatum",
    "wohnadresse",
    // fr
    "appelez-moi",
    "mon numéro",
    "ligne directe",
    "mon portable",
    "personnel",
    "date de naissance",
    "domicile",
    // es
    "llámame",
    "mi número",
    "línea directa",
    "mi móvil",
    "personal",
    "fecha de nacimiento",
    "domicilio",
    // it
    "chiamami",
    "il mio numero",
    "diretto",
    "cellulare",
    "personale",
    "data di nascita",
    "residenza",
    // en
    "call me",
    "my number",
    "direct line",
    "mobile",
    "date of birth",
    "home address",
    "residential",
    // Third-person pronouns. Some are also an article or "there" and over-mask; that is the safe side.
    "hij",
    "hem",
    "zij",
    "ze",
    "haar",
    "er",
    "ihm",
    "ihn",
    "sein",
    "seine",
    "sie ist",
    "sie hat",
    "il",
    "elle",
    "lui",
    "él",
    "ella",
    "lei",
    "egli",
    "he",
    "him",
    "his",
    "she",
    "her",
    "hers",
    // First person, possessives and object pronouns: a person speaking of themself or of someone.
    "ik",
    "ich",
    "je",
    "yo",
    "io",
    "i",
    "i'm",
    "me",
    "mij",
    "mich",
    "mir",
    "moi",
    "mí",
    "mi",
    "my",
    "mijn",
    "mein",
    "meine",
    "mon",
    "ma",
    "mes",
    "mis",
    "mio",
    "mia",
    "diens",
    "hen",
    "deren",
    "dessen",
    "son",
    "ses",
    "su",
    "sus",
    "gli",
    "suo",
    "sua",
    "they",
    "them",
    "their",
    // Direct dial and home.
    "doorkies",
    "doorkiesnummer",
    "toestel",
    "nebenstelle",
    "apparat",
    "poste",
    "extensión",
    "interno",
    "ext",
    "ext.",
    "extension",
    "ddi",
    "dd",
    "thuis",
    "thuisnummer",
    "zu hause",
    "maison",
    "casa",
    "home",
    // Honorifics and titles.
    "mw.",
    "señor",
    "señora",
    "don",
    "doña",
    "dott.",
    "dott.ssa",
    "ing.",
    "avv.",
    "signor",
    "signora",
    "madame",
    "monsieur",
    "mlle",
    "miss",
    "prof.",
    "z. hd.",
    "rev.",
    "revd",
    "revd.",
    "mag.",
    "lic.",
    "lcdo.",
    "lcda.",
    "dña.",
    "hr.",
    // A person by role, and a record about a person.
    "collega",
    "kollege",
    "kollegin",
    "collègue",
    "compañero",
    "compañera",
    "colleague",
    "accountmanager",
    "account manager",
    "contactpersoon",
    "ansprechpartner",
    "ansprechpartnerin",
    "boekhouder",
    "buchhalter",
    "buchhalterin",
    "comptable",
    "contable",
    "contabile",
    "accountant",
    "directeur",
    "direktor",
    "directrice",
    "director",
    "direttore",
    "direttrice",
    "kandidaat",
    "applicant",
    "patiënt",
    "patient",
    "werkgever",
    "employer",
    "arbeitgeber",
    "employeur",
    "empleador",
    "datore di lavoro",
    // Roles and asks for a named person.
    "referente",
    "tecnico",
    "reperibile",
    "responsabile",
    "escalation",
    "escalatie",
    "team lead",
    "engineer",
    "technician",
    "sir",
    "mx",
    "on call",
    "fragen sie nach",
    "bereitschaft",
    "techniker",
    "sachbearbeiter",
    "sachbearbeiterin",
    "vraag naar",
    "medewerker",
    "piket",
    "klantadviseur",
    "demandez",
    "conseiller",
    "conseillère",
    "technicien",
    "pregunte por",
    "gestor",
    "técnico",
    "guardia",
    // Landline wording, and the headers of a record that names a person.
    "landline",
    "desk",
    "vaste lijn",
    "festnetz",
    "fixe",
    "fisso",
    "fijo",
    "name",
    "naam",
    "nome",
    "nom",
    "nombre",
    "vorname",
    "nachname",
    "contact_name",
];

/// Wording that says a number is the company's own line: a service desk, a switchboard, a register
/// number in a footer, or the supplier on a safety data sheet. Without one nearby nothing clears.
const COMPANY_LINE: [&str; 63] = [
    // nl
    "klantenservice",
    "klantendienst",
    "servicelijn",
    "servicedesk",
    "helpdesk",
    "receptie",
    "algemeen nummer",
    "hoofdnummer",
    "centrale",
    "kvk",
    "leverancier",
    // de
    "kundenservice",
    "kundendienst",
    "service-hotline",
    "hotline",
    "zentrale",
    "telefonzentrale",
    "empfang",
    "hrb",
    "hra",
    "handelsregister",
    "lieferant",
    "hersteller",
    // fr
    "service client",
    "service clients",
    "service clientèle",
    "assistance",
    "accueil",
    "rcs",
    "siret",
    "siren",
    "fournisseur",
    "fabricant",
    // es
    "atención al cliente",
    "servicio al cliente",
    "servicio de atención",
    "centralita",
    "recepción",
    "asistencia",
    "soporte",
    "proveedor",
    "fabricante",
    // it
    "servizio clienti",
    "assistenza",
    "centralino",
    "supporto",
    "partita iva",
    "p.iva",
    "fornitore",
    "produttore",
    // en
    "customer service",
    "customer services",
    "customer support",
    "support",
    "service desk",
    "switchboard",
    "reception",
    "main line",
    "main number",
    "company no",
    "registered in",
    "supplier",
    "manufacturer",
];

/// What may stand between the company, or its line's wording, and the number: a label, "reachable
/// at", a preposition. Anything else there, a name above all, stops the number clearing.
const CONNECTORS: [&str; 51] = [
    "t",
    "tel",
    "telefoon",
    "telefon",
    "telefonnummer",
    "tél",
    "téléphone",
    "telefono",
    "teléfono",
    "phone",
    "telephone",
    "nummer",
    "number",
    "numéro",
    "número",
    "numero",
    "bereikbaar",
    "erreichbar",
    "joignable",
    "raggiungibile",
    "disponible",
    "reachable",
    "op",
    "via",
    "unter",
    "au",
    "al",
    "allo",
    "en",
    "el",
    "on",
    "at",
    "is",
    "ist",
    "est",
    "es",
    "è",
    "de",
    "der",
    "den",
    "des",
    "van",
    "di",
    "del",
    "della",
    "of",
    "the",
    "le",
    "la",
    "il",
    "lo",
];

/// A closing greeting marks a signature, and a signature holds a person's line.
const GREETINGS: [&str; 46] = [
    "groet",
    "groeten",
    "gegroet",
    "mvg",
    "gruß",
    "grüße",
    "grüßen",
    "mfg",
    "regards",
    "cheers",
    "thanks",
    "sincerely",
    "cordialement",
    "salutations",
    "saluti",
    "cordiali",
    "grazie",
    "saludos",
    "atentamente",
    "un saludo",
    "best",
    "br",
    "warm wishes",
    "best wishes",
    "thx",
    "thank you",
    "talk soon",
    "saluto",
    "un saluto",
    "cordialmente",
    "ciao",
    "a presto",
    "buona giornata",
    "bisous",
    "bien à vous",
    "liebe grüsse",
    "grüsse",
    "gruss",
    "groetjes",
    "liefs",
    "lg",
    "vg",
    "gr",
    "hartelijke groet",
    "kind regards",
    "viele grüße",
];

/// Given names from open national and municipal data, one per line; `names/LICENSE-NAMES.md` has
/// each source and its licence.
const GIVEN_NAMES: [&str; 6] = [
    include_str!("../names/nl.txt"),
    include_str!("../names/de.txt"),
    include_str!("../names/fr.txt"),
    include_str!("../names/es.txt"),
    include_str!("../names/it.txt"),
    include_str!("../names/en.txt"),
];

/// Family names, one word per line, from the same notice.
const FAMILY_NAMES: [&str; 6] = [
    include_str!("../names/family-nl.txt"),
    include_str!("../names/family-de.txt"),
    include_str!("../names/family-fr.txt"),
    include_str!("../names/family-es.txt"),
    include_str!("../names/family-it.txt"),
    include_str!("../names/family-en.txt"),
];

/// Function words the name sources also list ("El", "Per", "Nella"); they never count as a name.
const FUNCTION_WORDS: &str = include_str!("../names/function-words.txt");

/// Everyday words and places the name lists also hold ("Jahr", "Glas", "France").
const COMMON_WORDS: &str = include_str!("../names/common-words.txt");

/// Words that open a noun phrase and never a name; "der" does, in speech ("der Müller").
const NOUN_OPENERS: [&str; 52] = [
    "das", "ein", "eine", "einem", "einen", "einer", "eines", "kein", "keine", "keinem", "keinen",
    "keiner", "keines", "dieses", "jede", "jeder", "jedes", "jedem", "jeden", "im", "ins", "ans",
    "aufs", "am", "aus", "pro", "ab", "het", "een", "uit", "the", "these", "those", "in", "un",
    "une", "ces", "au", "aux", "una", "unos", "unas", "estos", "estas", "uno", "nel", "nella",
    "nello", "negli", "nelle", "sul", "sulla",
];

/// Byte range around `start..end`: `REACH` characters each side, cut short at a blank line.
pub(crate) fn window(text: &str, start: usize, end: usize) -> (usize, usize) {
    let from = text[..start]
        .char_indices()
        .rev()
        .nth(REACH - 1)
        .map_or(0, |(index, _)| index);
    let to = text[end..]
        .char_indices()
        .nth(REACH)
        .map_or(text.len(), |(index, _)| end + index);
    let from = blank_line()
        .find_iter(&text[from..start])
        .last()
        .map_or(from, |gap| from + gap.end());
    let to = blank_line()
        .find(&text[end..to])
        .map_or(to, |gap| end + gap.start());
    (from, to)
}

/// A PERSON or a greeting anywhere in the text keeps a number masked: the paragraph was not enough,
/// because a signature converted from HTML puts its name and greeting in paragraphs of their own.
pub(crate) fn scope(text: &str) -> (usize, usize) {
    (0, text.len())
}

pub(crate) fn company_mentioned(around: &str) -> bool {
    company().is_match(around)
}

/// Whether the words right before `number_at` are the company's name or its line's wording, with
/// only connectors and punctuation between.
pub(crate) fn introduced(around: &str, number_at: usize) -> bool {
    let before = &around[..number_at];
    let mut cut = before.len();
    let mut rest = before;
    while let Some((start, word)) = last_word(rest) {
        let short = word.chars().all(|c| !c.is_alphabetic()) || word.len() <= 1;
        let lowered = word.to_lowercase();
        if !CONNECTORS.contains(&lowered.as_str())
            || (word != lowered && !short && !is_label(&lowered))
        {
            break;
        }
        cut = start;
        rest = &before[..start];
    }
    let head = before[..cut].trim_end_matches(|c: char| !c.is_alphanumeric() && c != '.');
    company()
        .find_iter(head)
        .any(|found| found.end() == head.len())
        || company_line_ends(&head.to_lowercase())
}

/// "Tom's", "Charles'", or a capitalised word in brackets straight after the number: a person.
pub(crate) fn named_around(around: &str, number_end: usize) -> bool {
    possessive().is_match(around) || bracketed_name().is_match(&around[number_end..])
}

fn possessive() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| {
        Regex::new(r"\p{Lu}\p{Ll}+(?:'s|’s|s'|s’)(?:$|[^\p{L}\p{N}])").expect("possessive pattern")
    })
}

fn bracketed_name() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| {
        Regex::new(r"^\s*[,;:-]?\s*\(\s*\p{Lu}\p{L}*").expect("bracketed name pattern")
    })
}

/// A known given or family name written as a name anywhere in the text: capitalised, in capitals
/// when it is three letters or more, or in a script without case. The model misses many bare first
/// names and surnames, so the lists decide alone. The company's own name word right before its
/// suffix, the suffix, and a company line's wording ("P.IVA") do not count, nor does a family name
/// inside the company's name ("Lindenberg Haustechnik GmbH").
pub(crate) fn names_a_person(text: &str) -> bool {
    if text.chars().any(invisible) {
        return names_a_person(&text.chars().filter(|&c| !invisible(c)).collect::<String>());
    }
    let mut skipped: Vec<(usize, usize)> = company_name()
        .captures_iter(text)
        .filter_map(|found| Some((found.name("name")?.start(), found.name("suffix")?.end())))
        .collect();
    skipped.extend(
        company_line_any_case()
            .find_overlapping_iter(text)
            .filter(|hit| {
                let before = text[..hit.start()].chars().next_back();
                let after = text[hit.end()..].chars().next();
                before.is_none_or(|c| !c.is_alphanumeric())
                    && after.is_none_or(|c| !c.is_alphanumeric())
            })
            .map(|hit| (hit.start(), hit.end())),
    );
    let listed = is_given_name;
    let family = is_family_name;
    let a_name = |word: &str| listed(word) || family(word);
    let a_noun = |at: usize, word: &str| {
        let rest = &text[at + word.len()..];
        rest.starts_with("- ")
            || rest.starts_with("-,")
            || everyday(word) && in_a_noun_phrase(text, at, at + word.len())
    };
    let named = words(text).any(|(at, word)| {
        !skipped.iter().any(|&(from, to)| from <= at && at < to)
            && ((written_as_name(word) || opens_a_line(text, at))
                && listed(word)
                && !a_noun(at, word)
                || camel_parts(word).any(|part| written_as_name(part) && listed(part)))
    });
    let company_words: Vec<(usize, usize)> = company_run()
        .captures_iter(text)
        .filter_map(|found| found.name("run").map(|run| (run.start(), run.end())))
        .chain(skipped.iter().copied())
        .collect();
    let surnamed = words(text).any(|(at, word)| {
        !company_words
            .iter()
            .any(|&(from, to)| from <= at && at < to)
            && ((written_as_name(word) || opens_a_line(text, at))
                && family(word)
                && !a_noun(at, word)
                || camel_parts(word).any(|part| written_as_name(part) && family(part)))
    });
    let any_name = a_name;
    named
        || surnamed
        || address_parts(text).any(|part| {
            part.chars().count() >= 3 && (any_name(part) || run_together(part, &any_name))
        })
        || introduced_by_name()
            .captures_iter(text)
            .filter_map(|found| found.name("name"))
            .any(|name| listed(name.as_str()))
        || initial_and_name()
            .captures_iter(text)
            .filter_map(|found| found.name("name"))
            .any(|name| {
                !skipped
                    .iter()
                    .any(|&(from, to)| from <= name.start() && name.start() < to)
            })
}

/// "im Jahr", "aus reinem Gold", "„Glas“", "AR-Glas", "Ende 2024", "New York", "Gold und Silber".
fn in_a_noun_phrase(text: &str, at: usize, end: usize) -> bool {
    let before = &text[..at];
    let after = &text[end..];
    let quoted = before.ends_with(['„', '“', '"', '‚', '‘', '«'])
        && after.starts_with(['“', '”', '"', '‘', '’', '»']);
    let joined_after = after.strip_prefix(['-', '/']).is_some_and(|rest| {
        rest.split(|c: char| !c.is_alphanumeric())
            .next()
            .is_some_and(compound_part)
    });
    let joined_before = before.strip_suffix(['-', '/']).is_some_and(|rest| {
        last_word(rest)
            .is_some_and(|(start, word)| start + word.len() == rest.len() && compound_part(word))
    });
    let mut back = before.chars().rev();
    let elided = matches!(back.next(), Some('\'' | '’'))
        && matches!(back.next(), Some('l' | 'L'))
        && back.next().is_none_or(|c| !c.is_alphabetic());
    if quoted
        || joined_after
        || joined_before
        || elided
        || year().is_match(after)
        || listed_with(before, after)
    {
        return true;
    }
    let line = before.rsplit('\n').next().unwrap_or("");
    if !line.ends_with([' ', '\t']) {
        return false;
    }
    let mut previous = line
        .split([' ', '\t'])
        .filter(|word| !word.is_empty())
        .rev();
    let (Some(first), second) = (previous.next(), previous.next()) else {
        return false;
    };
    let opener = |word: &str| NOUN_OPENERS.contains(&word.to_lowercase().as_str());
    let after_lower_case = second.is_none_or(|word| !word.starts_with(char::is_uppercase));
    if opener(first)
        || PLACE_PREFIXES.contains(&first)
        || first.eq_ignore_ascii_case("en") && after_lower_case
    {
        return true;
    }
    let length = first.chars().count();
    let determiner = [
        "unser", "euer", "eure", "ihr", "sein", "mein", "dein", "jen", "welch", "deren", "dessen",
        "gewiss", "besagt", "genannt", "erwähnt", "zeker", "certain",
    ]
    .iter()
    .any(|stem| first.starts_with(stem));
    let adjective = first.chars().all(char::is_lowercase)
        && !determiner
        && length >= 4
        && ["e", "en", "er", "es", "em"]
            .iter()
            .any(|ending| first.ends_with(ending));
    adjective
        && (second.is_some_and(|word| {
            opener(word) || PREPOSITIONS.contains(&word.to_lowercase().as_str())
        }) || first.ends_with("em")
            || first.ends_with("es"))
}

/// A compound half no double surname has: lower case, an acronym or a number ("AR-Glas").
fn compound_part(word: &str) -> bool {
    word.chars()
        .next()
        .is_some_and(|c| c.is_lowercase() || c.is_ascii_digit())
        || word.chars().count() >= 2 && !word.chars().any(char::is_lowercase)
}

/// Prepositions that open a noun phrase when an adjective follows them ("mit zunehmender Länge").
const PREPOSITIONS: [&str; 16] = [
    "mit", "von", "zu", "zum", "zur", "nach", "bei", "beim", "für", "an", "auf", "über", "unter",
    "vor", "durch", "ohne",
];

/// "Gold und Silber": joined to another everyday word, not to a surname ("Özdemir und Holz").
fn listed_with(before: &str, after: &str) -> bool {
    let other = |word: &str| {
        word.chars().next().is_some_and(char::is_uppercase)
            && word.chars().any(char::is_lowercase)
            && everyday(word.trim_end_matches('.'))
            && !GROUPS.contains(&word.to_lowercase().as_str())
    };
    let joined = |word: &str| CONJUNCTIONS.contains(&word.trim_end_matches('.'));
    let mut back = before.split([' ', '\t']).rev().skip(1);
    let ahead = after
        .strip_prefix([' ', '\t'])
        .map(|rest| rest.split([' ', '\t', ',', ';', ':', ')']));
    let behind = back.next().is_some_and(joined)
        && back
            .next()
            .is_some_and(|word| other(word.trim_end_matches(',')));
    let mut ahead = ahead.into_iter().flatten();
    behind || ahead.next().is_some_and(joined) && ahead.next().is_some_and(other)
}

const CONJUNCTIONS: [&str; 12] = [
    "und", "oder", "sowie", "u", "and", "or", "et", "ou", "y", "o", "e", "en",
];

const GROUPS: [&str; 18] = [
    "partner", "partners", "team", "kollegen", "söhne", "sohn", "tochter", "töchter", "frau",
    "mann", "kinder", "co", "familie", "family", "fils", "hijos", "figli", "sons",
];

/// "New York", "North America", "Saudi Arabia".
const PLACE_PREFIXES: [&str; 16] = [
    "New", "Los", "Las", "San", "Santa", "Saint", "Sainte", "Sankt", "St.", "North", "South",
    "East", "West", "Saudi", "Latin", "Central",
];

/// "Ende 2024", not "Holz 2024 5678".
fn year() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| {
        Regex::new(r"^[ \t]+(?:1[5-9]|20)\d\d(?:$|[^\d\s]|\s+[^\d\s])").expect("year pattern")
    })
}

/// An everyday word, unless it is one of the most frequent names ("Müller", "Sommer", "Christian").
fn everyday(word: &str) -> bool {
    let word = word.nfc().collect::<String>().to_lowercase();
    common_words().contains(&word) && !frequent_names().contains(&word)
}

fn frequent_names() -> &'static HashSet<String> {
    static NAMES: OnceLock<HashSet<String>> = OnceLock::new();
    NAMES.get_or_init(|| {
        let given = GIVEN_NAMES.iter().flat_map(|list| list.lines().take(300));
        let family = FAMILY_NAMES.iter().flat_map(|list| list.lines().take(100));
        given
            .chain(family)
            .map(|name| name.trim().nfc().collect::<String>().to_lowercase())
            .collect()
    })
}

fn common_words() -> &'static HashSet<String> {
    static WORDS: OnceLock<HashSet<String>> = OnceLock::new();
    WORDS.get_or_init(|| {
        COMMON_WORDS
            .lines()
            .map(str::trim)
            .filter(|word| !word.is_empty() && !word.starts_with('#'))
            .map(|word| word.nfc().collect::<String>().to_lowercase())
            .collect()
    })
}

/// "sarahjones": a given name of four letters or more run into a second name.
fn run_together(word: &str, any_name: &impl Fn(&str) -> bool) -> bool {
    word.char_indices()
        .map(|(at, _)| at)
        .filter(|&at| word[..at].chars().count() >= 4 && word[at..].chars().count() >= 3)
        .any(|at| is_given_name(&word[..at]) && any_name(&word[at..]))
}

/// A person introducing themself in lower case: "this is tom", "hier is kees", "c'est julie".
fn introduced_by_name() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| {
        Regex::new(
            r"(?i:\bthis is|\bit'?s|\bhere'?s|\bhier is|\bmet|\bhier ist|\bc'est|\bici|\bsoy|\baquí|\bsono|\bqui è)[ \t]+(?P<name>\p{Ll}+)",
        )
        .expect("introduced by name pattern")
    })
}

/// A zero-width or soft-hyphen character, which would split a name in two.
fn invisible(c: char) -> bool {
    matches!(c, '\u{00AD}' | '\u{200B}'..='\u{200F}' | '\u{2060}'..='\u{2064}' | '\u{FEFF}')
}

/// A word at the start of the text, a line, a bullet or a table cell: a lower-case name there.
fn opens_a_line(text: &str, at: usize) -> bool {
    let head = text[..at].trim_end_matches([' ', '\t', '-', '\u{2013}', '\u{2014}', '*', '•', '>']);
    head.is_empty() || head.ends_with(['\n', '|', ';']) || text[..at].ends_with('\t')
}

/// "SarahJ" as "Sarah" and "J".
fn camel_parts(word: &str) -> impl Iterator<Item = &str> {
    let mut rest = 0;
    let bounds: Vec<usize> = word
        .char_indices()
        .zip(word.chars().skip(1))
        .filter(|((_, current), next)| current.is_lowercase() && next.is_uppercase())
        .map(|((at, current), _)| at + current.len_utf8())
        .chain(std::iter::once(word.len()))
        .collect();
    let split = bounds.len() > 1;
    bounds.into_iter().filter(move |_| split).map(move |end| {
        let part = &word[rest..end];
        rest = end;
        part
    })
}

/// The words of an address's local part, a handle or a link's path: "sarah.jones@", "@sarah_j",
/// "calendly.com/tom-jansen".
fn address_parts(text: &str) -> impl Iterator<Item = &str> {
    address().captures_iter(text).flat_map(|found| {
        found
            .name("local")
            .or_else(|| found.name("handle"))
            .or_else(|| found.name("path"))
            .map(|part| part.as_str())
            .into_iter()
            .flat_map(|part| {
                part.split(|c: char| !c.is_alphabetic())
                    .filter(|word| !word.is_empty())
            })
    })
}

fn address() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| {
        Regex::new(r"(?P<local>[\p{L}\p{N}._+-]+)@|@(?P<handle>[\p{L}\p{N}_.-]+)|(?:https?://)?[\p{L}\p{N}-]+(?:\.[\p{L}\p{N}-]+)+/(?P<path>[^\s?#]+)")
            .expect("address pattern")
    })
}

/// An initial before a capitalised name: "T. Jones", "S.Jones".
fn initial_and_name() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| {
        Regex::new(r"(?:^|[^\p{L}\p{N}.])\p{Lu}\.[ \t]?(?P<name>\p{Lu}\p{Ll}+)")
            .expect("initial and name pattern")
    })
}

fn written_as_name(word: &str) -> bool {
    let Some(first) = word.chars().next() else {
        return false;
    };
    if !first.is_uppercase() && !first.is_lowercase() {
        return true;
    }
    first.is_uppercase() && (word.chars().any(char::is_lowercase) || word.chars().count() >= 3)
}

fn words(text: &str) -> impl Iterator<Item = (usize, &str)> {
    let mut rest = 0;
    std::iter::from_fn(move || {
        let start = rest + text[rest..].find(char::is_alphabetic)?;
        let end = text[start..]
            .find(|c: char| !c.is_alphabetic())
            .map_or(text.len(), |length| start + length);
        rest = end;
        Some((start, &text[start..end]))
    })
}

/// Lower case without accents or styled letters, so "José" or "Ｊｏｓé" finds "JOSE" in a list.
fn fold(word: &str) -> String {
    word.nfkd()
        .filter(|c| !is_combining_mark(*c))
        .flat_map(char::to_lowercase)
        .collect()
}

/// Lower case without accents or styled letters, except the umlauts on a, o and u and "ß".
fn spelled(word: &str) -> String {
    let mut out = String::new();
    for c in word
        .nfkc()
        .flat_map(char::to_lowercase)
        .collect::<String>()
        .nfd()
    {
        if is_combining_mark(c) && !(c == '\u{308}' && out.ends_with(['a', 'o', 'u'])) {
            continue;
        }
        out.push(c);
    }
    out.nfc().collect()
}

/// An entry as written and with any of its umlauts dropped or written out: "Müller", "Muller",
/// "Mueller"; "Große", "Grosse".
fn spellings(entry: &str) -> Vec<String> {
    spelled(entry)
        .chars()
        .fold(vec![String::new()], |heads, c| {
            let options: &[&str] = match c {
                'ä' => &["ä", "a", "ae"],
                'ö' => &["ö", "o", "oe"],
                'ü' => &["ü", "u", "ue"],
                'ß' => &["ß", "ss"],
                _ => &[],
            };
            heads
                .iter()
                .flat_map(|head| {
                    let options: Vec<String> = if options.is_empty() {
                        vec![c.to_string()]
                    } else {
                        options.iter().map(|option| (*option).to_string()).collect()
                    };
                    options
                        .into_iter()
                        .map(move |option| format!("{head}{option}"))
                })
                .collect()
        })
}

/// Everyday words that are also a surname as written ("Karl Bücher").
const UMLAUT_SURNAMES: [&str; 3] = ["bücher", "häuser", "brügge"];

/// A name list: the entries' own spellings, and the entries without any accent.
struct Names {
    spelled: HashSet<String>,
    folded: HashSet<String>,
}

impl Names {
    fn load(lists: &[&str]) -> Self {
        let entries: Vec<&str> = lists
            .iter()
            .flat_map(|list| list.lines())
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .collect();
        Self {
            spelled: entries.iter().flat_map(|entry| spellings(entry)).collect(),
            folded: entries.iter().map(|entry| fold(entry)).collect(),
        }
    }

    /// A word may drop an entry's accents ("Muller", "Jose") but may not add an umlaut to make an
    /// everyday word: "Länge" is not "Lange", "Größe" is not "Große".
    fn contains(&self, word: &str) -> bool {
        let spelled = spelled(word);
        if self.spelled.contains(&spelled) {
            return true;
        }
        let lowered = word.nfc().collect::<String>().to_lowercase();
        let added = spelled.contains(['ä', 'ö', 'ü'])
            && common_words().contains(&lowered)
            && !UMLAUT_SURNAMES.contains(&lowered.as_str());
        !added && self.folded.contains(&fold(word))
    }
}

fn function_words() -> &'static HashSet<String> {
    static WORDS: OnceLock<HashSet<String>> = OnceLock::new();
    WORDS.get_or_init(|| {
        FUNCTION_WORDS
            .lines()
            .map(str::trim)
            .filter(|word| !word.is_empty() && !word.starts_with('#'))
            .map(|word| word.nfkc().collect::<String>().to_lowercase())
            .collect()
    })
}

/// Accents kept: "Schon" is a function word, the surname "Schön" is not.
fn function_word(word: &str) -> bool {
    function_words().contains(&word.nfkc().collect::<String>().to_lowercase())
}

fn is_given_name(word: &str) -> bool {
    !function_word(word) && given_names().contains(word)
}

fn is_family_name(word: &str) -> bool {
    !function_word(word) && family_names().contains(word)
}

fn family_names() -> &'static Names {
    static NAMES: OnceLock<Names> = OnceLock::new();
    NAMES.get_or_init(|| Names::load(&FAMILY_NAMES))
}

/// One or two capitalised words, joined by spaces or "&", straight before a legal suffix. A word may
/// not end in a full stop, so a sentence ending in a name does not run into the company.
fn company_run() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| {
        let suffixes = SUFFIXES
            .iter()
            .map(|suffix| regex::escape(suffix))
            .collect::<Vec<_>>()
            .join("|");
        Regex::new(&format!(
            r"(?P<run>(?:\p{{Lu}}[\p{{L}}\p{{N}}&'-]*(?:\.[\p{{L}}\p{{N}}]+)*[ \t]+(?:&[ \t]+)?){{1,2}})(?:{suffixes})(?:$|[^\p{{L}}\p{{N}}])"
        ))
        .expect("company run pattern")
    })
}

fn given_names() -> &'static Names {
    static NAMES: OnceLock<Names> = OnceLock::new();
    NAMES.get_or_init(|| Names::load(&GIVEN_NAMES))
}

fn company_name() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| {
        let suffixes = SUFFIXES
            .iter()
            .map(|suffix| regex::escape(suffix))
            .collect::<Vec<_>>()
            .join("|");
        Regex::new(&format!(
            r"(?P<name>\p{{Lu}}[\p{{L}}\p{{N}}&'.-]*)[ \t]+(?P<suffix>{suffixes})(?:$|[^\p{{L}}\p{{N}}])"
        ))
        .expect("company name pattern")
    })
}

fn company_line_any_case() -> &'static AhoCorasick {
    static AUTOMATON: OnceLock<AhoCorasick> = OnceLock::new();
    AUTOMATON.get_or_init(|| {
        AhoCorasick::builder()
            .match_kind(MatchKind::Standard)
            .ascii_case_insensitive(true)
            .build(COMPANY_LINE)
            .expect("company line automaton")
    })
}

pub(crate) fn greeting(paragraph: &str) -> bool {
    whole_term(greetings(), &paragraph.to_lowercase())
}

/// A capitalised connector is a name ("Al", "Van", "Le") unless it is a label such as "Tel".
fn is_label(lowered: &str) -> bool {
    matches!(
        lowered,
        "t" | "tel"
            | "telefoon"
            | "telefon"
            | "telefonnummer"
            | "tél"
            | "téléphone"
            | "telefono"
            | "teléfono"
            | "phone"
            | "telephone"
    )
}

fn last_word(text: &str) -> Option<(usize, &str)> {
    let end = text
        .char_indices()
        .rev()
        .find(|(_, c)| c.is_alphanumeric())
        .map(|(at, c)| at + c.len_utf8())?;
    let start = text[..end]
        .char_indices()
        .rev()
        .find(|(_, c)| !c.is_alphanumeric())
        .map_or(0, |(at, c)| at + c.len_utf8());
    Some((start, &text[start..end]))
}

fn company_line_ends(lowered: &str) -> bool {
    COMPANY_LINE.iter().any(|term| {
        lowered.strip_suffix(term).is_some_and(|rest| {
            rest.chars()
                .next_back()
                .is_none_or(|c| !c.is_alphanumeric())
        })
    })
}

pub(crate) fn company_line_wording(around: &str) -> bool {
    whole_term(company_line(), &around.to_lowercase())
}

fn whole_term(automaton: &AhoCorasick, lowered: &str) -> bool {
    automaton.find_overlapping_iter(lowered).any(|hit| {
        let before = lowered[..hit.start()].chars().next_back();
        let after = lowered[hit.end()..].chars().next();
        before.is_none_or(|c| !c.is_alphanumeric()) && after.is_none_or(|c| !c.is_alphanumeric())
    })
}

pub(crate) fn personal_wording(around: &str) -> bool {
    whole_term(personal(), &around.to_lowercase()) || german_she().is_match(around)
}

/// German "she" and "her": in lower case, which the formal "Sie" and "Ihr" never are, or a capital
/// "Sie" before a verb in the third person singular, which the formal "Sie" never takes.
fn german_she() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| {
        Regex::new(r"(?:^|[^\p{L}\p{N}])(?:(?:sie|ihr|ihre|ihrem|ihren|ihrer|ihres)|Sie\s+\p{Ll}+t)(?:$|[^\p{L}\p{N}])")
            .expect("german she pattern")
    })
}

pub(crate) fn published_type(number_type: phonenumber::Type) -> bool {
    use phonenumber::Type;
    matches!(
        number_type,
        Type::FixedLine | Type::TollFree | Type::SharedCost | Type::PremiumRate | Type::Uan
    )
}

fn blank_line() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| Regex::new(r"\n[ \t\r]*\n").expect("blank line pattern"))
}

fn company() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| {
        let suffixes = SUFFIXES
            .iter()
            .map(|suffix| regex::escape(suffix))
            .collect::<Vec<_>>()
            .join("|");
        // A capitalised word, whitespace, a suffix, and no letter or digit straight after it.
        Regex::new(&format!(
            r"(?:^|[^\p{{L}}\p{{N}}])\p{{Lu}}[\p{{L}}\p{{N}}&'.-]*[ \t]+(?:{suffixes})(?:$|[^\p{{L}}\p{{N}}])"
        ))
        .expect("company pattern")
    })
}

fn greetings() -> &'static AhoCorasick {
    static AUTOMATON: OnceLock<AhoCorasick> = OnceLock::new();
    AUTOMATON.get_or_init(|| {
        AhoCorasick::builder()
            .match_kind(MatchKind::Standard)
            .build(GREETINGS)
            .expect("greeting automaton")
    })
}

fn company_line() -> &'static AhoCorasick {
    static AUTOMATON: OnceLock<AhoCorasick> = OnceLock::new();
    AUTOMATON.get_or_init(|| {
        AhoCorasick::builder()
            .match_kind(MatchKind::Standard)
            .build(COMPANY_LINE)
            .expect("company line automaton")
    })
}

fn personal() -> &'static AhoCorasick {
    static AUTOMATON: OnceLock<AhoCorasick> = OnceLock::new();
    AUTOMATON.get_or_init(|| {
        AhoCorasick::builder()
            .match_kind(MatchKind::Standard)
            .build(PERSONAL)
            .expect("personal wording automaton")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn around(text: &str, number: &str) -> String {
        let start = text.find(number).expect("number in text");
        let (from, to) = window(text, start, start + number.len());
        text[from..to].to_string()
    }

    #[test]
    fn the_window_reaches_past_a_dotted_suffix() {
        let text = "de klantenservice van Noordkade Energie B.V., bereikbaar op 020 123 4567 tussen 9 en 17 uur.";
        assert!(around(text, "020 123 4567").contains("Noordkade Energie B.V."));
    }

    #[test]
    fn the_window_crosses_a_single_newline_and_stops_at_a_blank_line() {
        let text = "Brief over iets anders.\n\nIsar Metallbau GmbH\nLeopoldstraße 10\nTel. 089 3456789\n\nMet vriendelijke groet, Jan";
        let seen = around(text, "089 3456789");
        assert!(seen.contains("Isar Metallbau GmbH"));
        assert!(!seen.contains("Brief over"));
        assert!(!seen.contains("Jan"));
    }

    #[test]
    fn the_window_stops_at_150_characters_each_side() {
        let text = format!(
            "{}Noordkade B.V. {} 020 123 4567",
            "x ".repeat(10),
            "y".repeat(160)
        );
        assert!(!around(&text, "020 123 4567").contains("Noordkade"));
    }

    #[test]
    fn a_sole_trader_is_a_person_and_not_a_company() {
        assert!(!company_mentioned("Malerbetrieb Schulz e.K."));
    }

    #[test]
    fn a_suffix_after_a_capitalised_word_is_a_company() {
        for text in [
            "Noordkade Energie B.V.",
            "Noordkade Energie BV",
            "Bakkerij Jansen V.O.F.",
            "Lindenberg Haustechnik GmbH",
            "Muster GmbH & Co. KG",
            "Muster Holding UG",
            "Vallée Électronique SARL",
            "Transports Garonne SAS",
            "Ferretería Almendros S.L.",
            "Transportes Levante S.A.",
            "Falegnameria Bosco S.r.l.",
            "Trasporti Adriatico S.p.A.",
            "Thistledown Hardware Ltd",
            "ABC Chemicals Inc.",
        ] {
            assert!(company_mentioned(text), "{text}");
        }
    }

    #[test]
    fn a_suffix_without_a_capitalised_word_before_it_is_not() {
        for text in [
            "stuur je CV naar ons",
            "esa sa es otra",
            "bv. een voorbeeld",
            "gmbh",
        ] {
            assert!(!company_mentioned(text), "{text}");
        }
    }

    #[test]
    fn first_person_private_and_record_wording_is_personal() {
        for text in [
            "Bel mij op",
            "ruf mich an",
            "Appelez-moi au",
            "Llámame al",
            "Chiamami allo",
            "Call me on",
            "my mobile is",
            "Durchwahl",
            "ligne directe",
            "Date of Birth: 08/12/1985",
            "Residential Address:",
            "Geboortedatum",
            "Dhr. Bakker",
            "Mme Laurent",
            "Sig.ra Conti",
            "Mrs Clarke",
        ] {
            assert!(personal_wording(text), "{text}");
        }
    }

    #[test]
    fn a_third_person_pronoun_is_personal() {
        for text in [
            "en hij is bereikbaar op",
            "ze is bereikbaar op",
            "erreichen Sie ihn unter",
            "Lindenberg Haustechnik GmbH: sie ist erreichbar unter",
            "Sie ist montags erreichbar unter",
            "Sie arbeitet bei der Isar Metallbau GmbH",
            "Zij werkt bij Noordkade Energie B.V.",
            "Haar werknummer bij",
            "Er ist bei der Isar Metallbau GmbH",
            "Sein Telefon bei",
            "Il travaille chez Boulangerie du Centre SARL",
            "Lei lavora presso Falegnameria Bosco S.r.l.",
            "ihre Kollegin erreichen Sie unter",
            "elle est joignable au",
            "Puede llamarla a ella al",
            "lui risponde al",
            "you can reach him on",
            "her direct number",
        ] {
            assert!(personal_wording(text), "{text}");
        }
    }

    #[test]
    fn a_business_speaking_of_itself_or_its_reader_is_not_personal() {
        for text in [
            "Wij zijn bereikbaar op",
            "Sie erreichen uns unter",
            "Nous sommes joignables au",
            "Puede llamarnos al",
            "Siamo raggiungibili al",
            "You can reach us on",
            "Ihre Fragen beantworten wir unter",
        ] {
            assert!(!personal_wording(text), "{text}");
        }
    }

    #[test]
    fn a_word_inside_a_longer_word_is_not_personal() {
        for text in ["Herrenmode Schulz", "normal service", "Mrsa screening"] {
            assert!(!personal_wording(text), "{text}");
        }
    }

    #[test]
    fn a_known_given_name_written_as_a_name_is_a_person() {
        for text in [
            "Tom from customer support at Thistledown Hardware Ltd",
            "Uwe sitzt im Kundenservice der Isar Metallbau GmbH",
            "Contatto: GIULIA, servizio clienti",
            "Atención: Jose, soporte",
            "Réponse de José au service client",
            "Hoy Martín está en la centralita",
            "Tom Harris Ltd, customer service",
            "Kundenservice: Jean-Pierre",
        ] {
            assert!(names_a_person(text), "{text}");
        }
    }

    #[test]
    fn a_given_name_in_lower_case_or_inside_a_word_is_not_a_name() {
        for text in [
            "de klantenservice van Noordkade Energie B.V.",
            "customer service at Thistledown Hardware Ltd",
            "Tomography support",
            "Muster GmbH, info@muster.de, support",
        ] {
            assert!(!names_a_person(text), "{text}");
        }
    }

    #[test]
    fn the_company_name_word_before_its_suffix_and_a_register_label_are_not_a_person() {
        for text in [
            "Atención al cliente de Talleres Martín S.L.",
            "Muster GmbH & Co. KG, Zentrale",
            "Trasporti Adriatico S.p.A., P.IVA 01234567890, centralino",
        ] {
            assert!(!names_a_person(text), "{text}");
        }
    }

    #[test]
    fn a_name_in_lower_case_at_a_line_start_in_an_address_or_after_an_initial_is_a_name() {
        for text in [
            "tom from customer support",
            "Kind of urgent\nsarah\nSupport Thistledown Hardware Ltd",
            "id;name\n7; linda",
            "sarah.jones@thistledown.co.uk, customer support",
            "@sarah_j, customer support",
            "SarahJ | Customer support",
            "T. Jones, customer service",
            "Agent 4471 (S.Jones), customer service",
            "Book a call: https://calendly.com/tom-jansen, customer service",
            "Profile: linkedin.com/in/sarahjones",
            "Book at calendly.com/tom-jansen",
            "this is tom from customer support",
            "Hallo, hier ist sabine vom Kundenservice",
            "jansen\nKlantenservice",
        ] {
            assert!(names_a_person(text), "{text}");
        }
    }

    #[test]
    fn a_known_family_name_alone_is_a_person() {
        for text in [
            "Jansen, klantenservice, Noordkade Energie B.V.",
            "Müller, Kundenservice, Isar Metallbau GmbH",
            "Rossi, servizio clienti, Trasporti Adriatico S.p.A.",
            "García, atención al cliente",
            "Dubois, service client",
            "Jones, customer service, Thistledown Hardware Ltd",
            "Ticket bearbeitet von Müller. Kundenservice Isar Metallbau GmbH",
            "Kundenservice Schulze Isar Metallbau GmbH",
            "j.jansen@noordkade.nl, klantenservice",
        ] {
            assert!(names_a_person(text), "{text}");
        }
    }

    #[test]
    fn a_family_name_inside_the_company_name_is_the_company() {
        for text in [
            "Bei Rückfragen erreichen den Kundenservice der Lindenberg Haustechnik GmbH",
            "Kundenservice der Müller Metallbau GmbH",
            "Klantenservice van Jansen & Visser B.V.",
        ] {
            assert!(!names_a_person(text), "{text}");
        }
    }

    #[test]
    fn a_name_split_by_an_invisible_character_or_written_in_styled_letters_is_a_name() {
        for text in [
            "To\u{200B}m, customer service",
            "Ｔｏｍ, customer service",
            "- tom\nSupport",
        ] {
            assert!(names_a_person(text), "{text}");
        }
    }

    #[test]
    fn a_function_word_that_a_source_lists_as_a_name_is_not_a_name() {
        for text in [
            "El servicio de atención al cliente",
            "Le service client",
            "Per informazioni, Nella sede centrale",
            "Una consulta",
            "Für Rückfragen",
            "Both lines are open",
        ] {
            assert!(!names_a_person(text), "{text}");
        }
    }

    #[test]
    fn a_name_that_only_folds_to_a_function_word_is_still_a_name() {
        for text in [
            "Schön, Kundenservice",
            "Fur, service client",
            "Côme, service client",
        ] {
            assert!(names_a_person(text), "{text}");
        }
        assert!(!names_a_person("Schon 1885 fuhr die Bahn"));
    }

    #[test]
    fn a_name_may_drop_or_write_out_its_umlauts() {
        for text in [
            "Müller, Kundenservice",
            "Mueller, Kundenservice",
            "Muller, Kundenservice",
            "MÜLLER, Kundenservice",
            "Jaeger, Kundenservice",
            "Schaefer, Kundenservice",
            "Grosse, Kundenservice",
            "Große, Kundenservice",
            "Lange, Kundenservice",
            "Kühnert, Kundenservice",
            "Bücher, Kundenservice",
            "Weiß, Kundenservice",
        ] {
            assert!(names_a_person(text), "{text}");
        }
    }

    #[test]
    fn an_everyday_word_with_an_umlaut_is_not_the_name_without_it() {
        for text in [
            "Die Länge der Leitung beträgt 20 Meter",
            "Die Größe des Lagers wächst",
            "Die Stärke der Wand reicht",
            "Beide Länder liefern Strom",
        ] {
            assert!(!names_a_person(text), "{text}");
        }
    }

    #[test]
    fn a_noun_that_is_a_family_name_still_counts() {
        assert!(names_a_person("Holz, Kundenservice"));
    }

    #[test]
    fn an_everyday_word_inside_a_noun_phrase_is_not_a_name() {
        for text in [
            "Im Jahr 2024 stieg der Umsatz. Kundenservice der Isar Metallbau GmbH",
            "Das Glas ist gehärtet, ein Holz aus Europa",
            "Tische aus Holz, Kundenservice",
            "Ende 2024 lag der Wert höher",
            "mit zunehmender Länge und reinem Gold",
            "Gold und Silber, Stahl- und Holzbau, Fern- und Nahwärme, Hand-in-Hand, AR-Glas",
            "„Glas“ bezeichnet einen Werkstoff",
            "Sie leben en France depuis 2020, service client",
            "Delivered to New York, and sold in North America",
            "l'Italia, servizio clienti in Francia",
        ] {
            assert!(!names_a_person(text), "{text}");
        }
    }

    #[test]
    fn an_everyday_word_where_a_name_stands_still_counts() {
        for text in [
            "Holz\nKundenservice Isar Metallbau GmbH",
            "Ticket erledigt von Holz. Kundenservice",
            "Die Anfrage hat Holz übernommen. Kundenservice",
            "Weitergeleitet an Glas. Kundenservice",
            "| Holz | Isar Metallbau GmbH | Kundenservice |",
            "Heute hat der Holz Dienst. Kundenservice",
            "Rücksprache mit dem Holz. Kundenservice",
            "Rücksprache mit unserem Holz. Kundenservice",
            "Ein gewisser Holz hat angerufen. Kundenservice",
            "Die Holz aus dem Kundenservice ruft zurück.",
            "Holz-Baumert, Kundenservice",
            "Meier und Holz, Kundenservice",
            "Linda en India, klantenservice",
            "Ticket 4471 Holz, Kundenservice",
        ] {
            assert!(names_a_person(text), "{text}");
        }
    }

    #[test]
    fn a_frequent_name_counts_even_inside_a_noun_phrase() {
        for text in [
            "Ein Koch aus dem Team",
            "Im Sommer, Kundenservice",
            "the Christian community",
        ] {
            assert!(names_a_person(text), "{text}");
        }
    }

    #[test]
    fn a_name_that_starts_with_a_register_label_is_still_a_name() {
        assert!(names_a_person("Ivan, servizio clienti"));
    }

    #[test]
    fn only_the_types_companies_publish_clear() {
        use phonenumber::Type;
        for published in [
            Type::FixedLine,
            Type::TollFree,
            Type::SharedCost,
            Type::PremiumRate,
            Type::Uan,
        ] {
            assert!(published_type(published), "{published:?}");
        }
        for personal in [
            Type::Mobile,
            Type::FixedLineOrMobile,
            Type::Voip,
            Type::PersonalNumber,
            Type::Pager,
            Type::Unknown,
        ] {
            assert!(!published_type(personal), "{personal:?}");
        }
    }
}
