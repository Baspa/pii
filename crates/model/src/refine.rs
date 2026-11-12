//! Cleans the model's `PERSON` and `STREET_ADDRESS` spans before the resolver sees them.
//!
//! Splits a name at a line break, joins the pieces of one address or name, drops sub-word fragments, drops address spans with
//! no street word and no number, drops one-word names with no supporting evidence, and masks the
//! other occurrences of a name that was found.

use std::sync::OnceLock;

use pii_detection_engine::entity::EntityType;
use pii_detection_engine::text::CharSpan;

use crate::decode::WindowSpan;

const GIVEN_NAMES: &str = include_str!("../gazetteer/given-names.txt");

const ADDRESS_CONNECTORS: &[&str] = &[
    "de", "del", "della", "dei", "degli", "delle", "di", "la", "las", "los", "le", "les", "du",
    "des", "d'", "van", "von", "der", "den", "am", "an", "im",
];

const NAME_PARTICLES: &[&str] = &[
    "van", "de", "der", "den", "het", "ten", "ter", "te", "von", "zu", "vom", "zur", "du", "des",
    "le", "la", "di", "da", "del", "della", "dos", "das", "do", "el", "al", "y", "e", "bin", "ben",
    "mac", "st",
];

const STREET_WORDS: &[&str] = &[
    "street",
    "st",
    "road",
    "rd",
    "lane",
    "ln",
    "avenue",
    "ave",
    "av",
    "boulevard",
    "blvd",
    "bd",
    "drive",
    "close",
    "crescent",
    "square",
    "sq",
    "place",
    "pl",
    "court",
    "ct",
    "terrace",
    "grove",
    "gardens",
    "way",
    "row",
    "walk",
    "mews",
    "flat",
    "apartment",
    "apt",
    "unit",
    "suite",
    "rue",
    "chemin",
    "impasse",
    "quai",
    "cours",
    "allée",
    "route",
    "via",
    "viale",
    "vicolo",
    "piazza",
    "piazzale",
    "corso",
    "largo",
    "strada",
    "contrada",
    "calle",
    "avenida",
    "avda",
    "paseo",
    "plaza",
    "carrer",
    "camino",
    "carretera",
    "ronda",
    "travesía",
    "rua",
    "praça",
    "postbus",
    "postfach",
    "apartado",
    "straat",
    "weg",
    "laan",
    "plein",
    "gracht",
    "kade",
    "singel",
    "dijk",
    "steeg",
    "str",
    "straße",
    "strasse",
    "gasse",
    "platz",
    "allee",
    "damm",
    "ufer",
];

const STREET_SUFFIXES: &[&str] = &[
    "straat", "weg", "laan", "plein", "gracht", "kade", "singel", "dijk", "steeg", "str.",
    "straße", "strasse", "gasse", "platz", "allee", "damm", "ufer", "chaussee", "gatan", "vägen",
    "gade", "vej", "katu",
];

const TITLES: &[&str] = &[
    "mr",
    "mrs",
    "ms",
    "miss",
    "mx",
    "dr",
    "dra",
    "prof",
    "sir",
    "dame",
    "dhr",
    "mevr",
    "mw",
    "mej",
    "meneer",
    "mevrouw",
    "heer",
    "ir",
    "ing",
    "drs",
    "herr",
    "herrn",
    "frau",
    "hr",
    "fr",
    "m",
    "mme",
    "mlle",
    "monsieur",
    "madame",
    "mademoiselle",
    "maître",
    "sig",
    "sig.ra",
    "sigra",
    "signor",
    "signore",
    "signora",
    "signorina",
    "dott",
    "dott.ssa",
    "avv",
    "sr",
    "sra",
    "srta",
    "señor",
    "señora",
    "señorita",
    "don",
    "doña",
    "dña",
    "attn",
    "t.a.v",
    "hd",
];

const GREETINGS: &[&str] = &[
    "hi",
    "hello",
    "hey",
    "dear",
    "beste",
    "hallo",
    "hoi",
    "geachte",
    "lieve",
    "liebe",
    "lieber",
    "geehrte",
    "geehrter",
    "bonjour",
    "bonsoir",
    "salut",
    "cher",
    "chère",
    "ciao",
    "caro",
    "cara",
    "gentile",
    "gentilissimo",
    "gentilissima",
    "hola",
    "estimado",
    "estimada",
    "querido",
    "querida",
    "thanks",
    "bedankt",
    "danke",
    "merci",
    "grazie",
    "gracias",
    "groeten",
    "gruß",
    "grüße",
    "regards",
    "cheers",
];

const LABELS: &[&str] = &[
    "name",
    "naam",
    "nom",
    "nome",
    "nombre",
    "contact",
    "contactpersoon",
    "ansprechpartner",
    "ansprechpartnerin",
    "from",
    "to",
    "van",
    "aan",
    "an",
    "von",
    "de",
    "da",
    "para",
    "cc",
    "bcc",
    "attn",
    "signed",
    "author",
    "auteur",
    "autor",
    "autore",
    "owner",
    "eigenaar",
    "assignee",
    "reporter",
    "client",
    "klant",
    "kunde",
    "customer",
    "cliente",
    "patient",
    "patiënt",
    "employee",
    "medewerker",
    "mitarbeiter",
    "mitarbeiterin",
    "requester",
    "aanvrager",
    "antragsteller",
    "contacto",
    "referente",
];

/// Every step, in order. `spans` are the reconciled, trimmed and extended model spans.
pub fn refine(text: &str, spans: Vec<WindowSpan>) -> Vec<WindowSpan> {
    let chars: Vec<char> = text.chars().collect();
    let spans = split_names_at_line_breaks(&chars, spans);
    let spans = merge_adjacent(&chars, spans);
    let spans = align_to_words(&chars, spans);
    let spans = merge_adjacent(&chars, spans);
    let spans = drop_unsupported_addresses(&chars, spans);
    let spans = drop_unsupported_single_names(&chars, spans);
    propagate_names(&chars, spans)
}

fn slice(chars: &[char], span: CharSpan) -> String {
    chars[span.start as usize..span.end as usize]
        .iter()
        .collect()
}

/// A name never crosses a line: the model sometimes runs a signature into the line below it
/// (`Pieter de Vries` + `Kantoor`). Each line becomes its own name, judged by the later steps.
/// An address may cross a line (`Molenstraat 18` + `5211 KL`), so only names are split.
fn split_names_at_line_breaks(chars: &[char], spans: Vec<WindowSpan>) -> Vec<WindowSpan> {
    let mut split = Vec::with_capacity(spans.len());
    for window_span in spans {
        if window_span.entity_type != EntityType::Person {
            split.push(window_span);
            continue;
        }
        let mut start = window_span.span.start as usize;
        let end = window_span.span.end as usize;
        while start < end {
            let line_end = chars[start..end]
                .iter()
                .position(|&c| c == '\n')
                .map_or(end, |offset| start + offset);
            let piece_start = start
                + chars[start..line_end]
                    .iter()
                    .take_while(|c| c.is_whitespace())
                    .count();
            let piece_end = line_end
                - chars[piece_start..line_end]
                    .iter()
                    .rev()
                    .take_while(|c| c.is_whitespace())
                    .count();
            if piece_start < piece_end {
                split.push(WindowSpan {
                    span: CharSpan {
                        start: piece_start as u32,
                        end: piece_end as u32,
                    },
                    ..window_span
                });
            }
            start = line_end + 1;
        }
    }
    split
}

fn merge_adjacent(chars: &[char], mut spans: Vec<WindowSpan>) -> Vec<WindowSpan> {
    spans.sort_by_key(|window_span| (window_span.span.start, window_span.span.end));
    let mut merged: Vec<WindowSpan> = Vec::with_capacity(spans.len());
    for window_span in spans {
        if let Some(last) = merged.last_mut()
            && last.entity_type == window_span.entity_type
            && joinable(
                chars,
                last.entity_type,
                last.span.end,
                window_span.span.start,
            )
        {
            last.span.end = last.span.end.max(window_span.span.end);
            last.score = last.score.max(window_span.score);
            continue;
        }
        merged.push(window_span);
    }
    merged
}

fn joinable(chars: &[char], entity_type: EntityType, end: u32, start: u32) -> bool {
    if start <= end {
        return true;
    }
    let gap: String = chars[end as usize..start as usize].iter().collect();
    let words: Vec<String> = gap
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect();
    let commas = gap.matches(',').count();
    let newlines = gap.matches('\n').count();
    match entity_type {
        EntityType::StreetAddress => {
            if words.is_empty() {
                commas <= 1 && newlines <= 1
            } else {
                commas == 0
                    && newlines == 0
                    && words.len() <= 2
                    && words
                        .iter()
                        .all(|word| ADDRESS_CONNECTORS.contains(&word.as_str()))
            }
        }
        _ => {
            commas == 0
                && newlines == 0
                && (gap.trim() == "-"
                    || (words.len() <= 2
                        && words
                            .iter()
                            .all(|word| NAME_PARTICLES.contains(&word.as_str()))))
        }
    }
}

/// A span cut inside a word grows to the whole word when it covers at least half of it, and is
/// dropped when it does not.
fn align_to_words(chars: &[char], spans: Vec<WindowSpan>) -> Vec<WindowSpan> {
    let inside =
        |before: usize, at: usize| chars[before].is_alphanumeric() && chars[at].is_alphanumeric();
    spans
        .into_iter()
        .filter_map(|mut window_span| {
            let start = window_span.span.start as usize;
            let end = window_span.span.end as usize;
            let mut word_start = start;
            while word_start > 0 && inside(word_start - 1, word_start) {
                word_start -= 1;
            }
            let mut word_end = end;
            while word_end < chars.len() && inside(word_end - 1, word_end) {
                word_end += 1;
            }
            if (word_start, word_end) == (start, end) {
                return Some(window_span);
            }
            if 2 * (end - start) < word_end - word_start {
                return None;
            }
            window_span.span = CharSpan {
                start: word_start as u32,
                end: word_end as u32,
            };
            Some(window_span)
        })
        .collect()
}

/// An address span needs a street word, or a number beside a word: a place name, a year or an
/// amount on its own is not an address. A piece that trails a real address in the same clause
/// (`Washington DC 20500`) stays.
fn drop_unsupported_addresses(chars: &[char], spans: Vec<WindowSpan>) -> Vec<WindowSpan> {
    let supported: Vec<bool> = spans
        .iter()
        .map(|window_span| {
            window_span.entity_type != EntityType::StreetAddress
                || is_address_like(&slice(chars, window_span.span))
        })
        .collect();
    let anchors: Vec<CharSpan> = spans
        .iter()
        .zip(&supported)
        .filter(|(window_span, supported)| {
            window_span.entity_type == EntityType::StreetAddress && **supported
        })
        .map(|(window_span, _)| window_span.span)
        .collect();
    spans
        .into_iter()
        .zip(supported)
        .filter(|(window_span, supported)| {
            *supported || trails_an_address(chars, &anchors, window_span.span)
        })
        .map(|(window_span, _)| window_span)
        .collect()
}

fn trails_an_address(chars: &[char], anchors: &[CharSpan], span: CharSpan) -> bool {
    anchors.iter().any(|anchor| {
        anchor.end <= span.start
            && span.start - anchor.end <= 32
            && !chars[anchor.end as usize..span.start as usize]
                .iter()
                .any(|c| matches!(c, '.' | ';' | ':' | '(' | ')' | '\n'))
    })
}

fn is_address_like(value: &str) -> bool {
    let tokens: Vec<String> = value
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|token| !token.is_empty())
        .map(str::to_lowercase)
        .collect();
    let street_word = tokens.iter().any(|token| {
        let bare = token.trim_matches(|c: char| !c.is_alphanumeric());
        STREET_WORDS.contains(&bare)
            || STREET_SUFFIXES.iter().any(|suffix| {
                token.ends_with(suffix) && token.chars().count() > suffix.chars().count()
            })
    });
    let number = tokens
        .iter()
        .any(|token| token.chars().any(|c| c.is_ascii_digit()));
    let word = tokens
        .iter()
        .any(|token| token.chars().filter(|c| c.is_alphabetic()).count() >= 2);
    street_word || (tokens.len() >= 2 && number && word)
}

/// A one-word name is kept only with evidence: a known given name, a title, initial, greeting or
/// field label before it, a line of its own, or a longer name in the same text that contains it.
fn drop_unsupported_single_names(chars: &[char], spans: Vec<WindowSpan>) -> Vec<WindowSpan> {
    let full_name_words: Vec<String> = spans
        .iter()
        .filter(|window_span| window_span.entity_type == EntityType::Person)
        .map(|window_span| slice(chars, window_span.span))
        .filter(|value| value.split_whitespace().count() > 1)
        .flat_map(|value| value.split_whitespace().map(bare_lower).collect::<Vec<_>>())
        .collect();
    spans
        .into_iter()
        .filter(|window_span| {
            if window_span.entity_type != EntityType::Person {
                return true;
            }
            let value = slice(chars, window_span.span);
            if value.split_whitespace().count() > 1 {
                return true;
            }
            let word = bare_lower(&value);
            is_given_name(&word)
                || full_name_words.contains(&word)
                || introduced(chars, window_span.span)
        })
        .collect()
}

fn bare_lower(word: &str) -> String {
    word.trim_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase()
}

fn is_given_name(word: &str) -> bool {
    static NAMES: OnceLock<Box<[&'static str]>> = OnceLock::new();
    let names = NAMES.get_or_init(|| {
        GIVEN_NAMES
            .lines()
            .filter(|line| !line.is_empty())
            .collect()
    });
    let folded = word.replace('\u{2019}', "'");
    names.binary_search(&folded.as_str()).is_ok()
        || folded
            .split('-')
            .all(|part| names.binary_search(&part).is_ok())
}

fn introduced(chars: &[char], span: CharSpan) -> bool {
    let start = span.start as usize;
    let end = span.end as usize;
    let line_start = chars[..start]
        .iter()
        .rposition(|&c| c == '\n')
        .map_or(0, |at| at + 1);
    let line_end = chars[end..]
        .iter()
        .position(|&c| c == '\n')
        .map_or(chars.len(), |at| end + at);
    let before: String = chars[line_start..start].iter().collect();
    let after: String = chars[end..line_end].iter().collect();
    if !before
        .chars()
        .chain(after.chars())
        .any(char::is_alphanumeric)
    {
        return true;
    }
    let before = before.trim_end();
    let (labelled, rest) = match before.strip_suffix(':') {
        Some(rest) => (true, rest.trim_end()),
        None => (false, before.trim_end_matches(',').trim_end()),
    };
    let Some(last) = rest.split_whitespace().last() else {
        return false;
    };
    if !labelled && is_initial(last) {
        return true;
    }
    let last = last
        .trim_start_matches(|c: char| !c.is_alphanumeric())
        .trim_end_matches(['.', ','])
        .to_lowercase();
    if labelled {
        LABELS.contains(&last.as_str())
    } else {
        TITLES.contains(&last.as_str()) || GREETINGS.contains(&last.as_str())
    }
}

fn is_initial(token: &str) -> bool {
    let mut chars = token.chars();
    matches!((chars.next(), chars.next(), chars.next()), (Some(letter), Some('.'), None) if letter.is_uppercase())
}

/// Masks the other occurrences of each word of a found name, written the same way, unless a
/// number sits next to it (`5 Jan 2026`).
fn propagate_names(chars: &[char], mut spans: Vec<WindowSpan>) -> Vec<WindowSpan> {
    let mut words: Vec<(Vec<char>, f32)> = Vec::new();
    for window_span in spans
        .iter()
        .filter(|window_span| window_span.entity_type == EntityType::Person)
    {
        for token in slice(chars, window_span.span).split_whitespace() {
            let token = token.trim_matches(|c: char| !c.is_alphanumeric());
            let lower = token.to_lowercase();
            if token.chars().count() < 2
                || !token.chars().next().is_some_and(char::is_uppercase)
                || NAME_PARTICLES.contains(&lower.as_str())
                || TITLES.contains(&lower.as_str())
            {
                continue;
            }
            let token: Vec<char> = token.chars().collect();
            match words.iter_mut().find(|(word, _)| *word == token) {
                Some((_, score)) => *score = score.max(window_span.score),
                None => words.push((token, window_span.score)),
            }
        }
    }
    let mut added: Vec<WindowSpan> = Vec::new();
    for (word, score) in &words {
        let length = word.len();
        if length > chars.len() {
            continue;
        }
        for at in 0..=chars.len() - length {
            if chars[at..at + length] != word[..]
                || (at > 0 && chars[at - 1].is_alphanumeric())
                || chars.get(at + length).is_some_and(|c| c.is_alphanumeric())
                || next_to_a_number(chars, at, at + length)
            {
                continue;
            }
            let span = CharSpan {
                start: at as u32,
                end: (at + length) as u32,
            };
            let taken = spans
                .iter()
                .chain(added.iter())
                .any(|other| other.span.start < span.end && span.start < other.span.end);
            if !taken {
                added.push(WindowSpan {
                    span,
                    entity_type: EntityType::Person,
                    score: *score,
                    margin: 0,
                    window: 0,
                });
            }
        }
    }
    spans.extend(added);
    spans.sort_by_key(|window_span| (window_span.span.start, window_span.span.end));
    spans
}

fn next_to_a_number(chars: &[char], start: usize, end: usize) -> bool {
    let before = chars[..start].iter().rev().find(|c| !c.is_whitespace());
    let after = chars[end..].iter().find(|c| !c.is_whitespace());
    before.is_some_and(char::is_ascii_digit) || after.is_some_and(char::is_ascii_digit)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(text: &str, value: &str, entity_type: EntityType, score: f32) -> WindowSpan {
        let byte = text.find(value).expect("value in text");
        let start = text[..byte].chars().count() as u32;
        WindowSpan {
            span: CharSpan {
                start,
                end: start + value.chars().count() as u32,
            },
            entity_type,
            score,
            margin: 0,
            window: 0,
        }
    }

    fn values(text: &str, spans: &[WindowSpan]) -> Vec<(String, EntityType)> {
        let chars: Vec<char> = text.chars().collect();
        spans
            .iter()
            .map(|window_span| (slice(&chars, window_span.span), window_span.entity_type))
            .collect()
    }

    fn address(text: &str, value: &str) -> WindowSpan {
        span(text, value, EntityType::StreetAddress, 0.9)
    }

    fn person(text: &str, value: &str) -> WindowSpan {
        span(text, value, EntityType::Person, 0.9)
    }

    #[test]
    fn the_pieces_of_one_address_become_one_span() {
        let text = "Stuur het naar Kerkstraat 12, 1016 GV Amsterdam.";
        let spans = vec![
            address(text, "Kerkstraat"),
            address(text, "12"),
            address(text, "1016 GV Amsterdam"),
        ];
        assert_eq!(
            values(text, &refine(text, spans)),
            vec![(
                "Kerkstraat 12, 1016 GV Amsterdam".to_string(),
                EntityType::StreetAddress
            )]
        );
    }

    #[test]
    fn an_address_split_on_a_connector_word_is_joined() {
        let text = "dirección paseo de la Alameda 53, 49870 Lorca";
        let spans = vec![
            address(text, "paseo"),
            address(text, "la Alameda 53, 49870 Lorca"),
        ];
        assert_eq!(
            values(text, &refine(text, spans)),
            vec![(
                "paseo de la Alameda 53, 49870 Lorca".to_string(),
                EntityType::StreetAddress
            )]
        );
    }

    #[test]
    fn two_addresses_with_words_between_stay_apart() {
        let text = "Ship to 42 Baker Street and bill to 10 Downing Street.";
        let spans = vec![
            address(text, "42 Baker Street"),
            address(text, "10 Downing Street"),
        ];
        assert_eq!(refine(text, spans).len(), 2);
    }

    #[test]
    fn a_number_or_a_place_alone_is_not_an_address() {
        let text = "In 1907 kwam Keukenhof open, 953 miljoen bezoekers later, order 4471 en T2S.";
        let spans = vec![
            address(text, "1907"),
            address(text, "Keukenhof"),
            address(text, "953"),
            address(text, "4471"),
            address(text, "T2S"),
        ];
        assert!(refine(text, spans).is_empty());
    }

    #[test]
    fn a_postcode_trailing_an_address_stays() {
        let text =
            "The invoice address is 1600 Pennsylvania Avenue, Washington DC 20500. Order 20500.";
        let spans = vec![
            address(text, "1600 Pennsylvania Avenue"),
            address(text, "20500"),
        ];
        assert_eq!(refine(text, spans).len(), 2);
    }

    #[test]
    fn a_number_before_an_address_is_dropped() {
        let text =
            "per un valore di 953 milioni di euro (indirizzo via dei Gelsi 65, 15512 Matera)";
        let spans = vec![
            address(text, "953"),
            address(text, "via dei Gelsi 65, 15512 Matera"),
        ];
        assert_eq!(
            values(text, &refine(text, spans)),
            vec![(
                "via dei Gelsi 65, 15512 Matera".to_string(),
                EntityType::StreetAddress
            )]
        );
    }

    #[test]
    fn a_street_without_a_number_is_still_an_address() {
        let text = "Ze woont aan de Prinsengracht, bij de Hauptstraße en in via Garibaldi.";
        let spans = vec![
            address(text, "Prinsengracht"),
            address(text, "Hauptstraße"),
            address(text, "via Garibaldi"),
        ];
        assert_eq!(refine(text, spans).len(), 3);
    }

    #[test]
    fn a_postcode_and_city_are_an_address() {
        let text = "Kendal LA9 4ZP and 10115 Berlin and 1016 GV";
        let spans = vec![
            address(text, "Kendal LA9 4ZP"),
            address(text, "10115 Berlin"),
            address(text, "1016 GV"),
        ];
        assert_eq!(refine(text, spans).len(), 3);
    }

    #[test]
    fn a_fragment_of_a_word_is_dropped() {
        let text = "l'éolienne Charmoy et le cycle Rankine";
        let spans = vec![person(text, "oy"), person(text, "ine")];
        assert!(refine(text, spans).is_empty());
    }

    #[test]
    fn most_of_a_word_grows_to_the_whole_word() {
        let text = "Sagen Sie es Herrn Müllers Team.";
        let spans = vec![person(text, "Müller")];
        assert_eq!(
            values(text, &refine(text, spans)),
            vec![("Müllers".to_string(), EntityType::Person)]
        );
    }

    #[test]
    fn two_halves_of_one_word_are_joined_before_the_fragment_check() {
        let text = "Approved by J. R. Whitfield today.";
        let spans = vec![person(text, "Whit"), person(text, "field")];
        assert_eq!(
            values(text, &refine(text, spans)),
            vec![("Whitfield".to_string(), EntityType::Person)]
        );
    }

    #[test]
    fn a_name_split_on_a_particle_is_joined() {
        let text = "Contact Pieter van der Berg today.";
        let spans = vec![person(text, "Pieter"), person(text, "Berg")];
        assert_eq!(
            values(text, &refine(text, spans)),
            vec![("Pieter van der Berg".to_string(), EntityType::Person)]
        );
    }

    #[test]
    fn a_name_never_crosses_a_line() {
        let text = "Groet,\nPieter de Vries\nKantoor Verhoeven Accountants B.V.";
        assert_eq!(
            values(
                text,
                &refine(text, vec![person(text, "Pieter de Vries\nKantoor")])
            ),
            vec![("Pieter de Vries".to_string(), EntityType::Person)]
        );
    }

    #[test]
    fn an_address_may_still_cross_a_line() {
        let text = "Molenstraat 18\n5211 KL 's-Hertogenbosch";
        assert_eq!(
            values(
                text,
                &refine(text, vec![address(text, "Molenstraat 18\n5211 KL")])
            ),
            vec![(
                "Molenstraat 18\n5211 KL".to_string(),
                EntityType::StreetAddress
            )]
        );
    }

    #[test]
    fn a_lone_capitalised_word_is_not_a_name() {
        let text = "Onze partner Qantas gebruikt Nirosta en Hafas.";
        let spans = vec![
            person(text, "Qantas"),
            person(text, "Nirosta"),
            person(text, "Hafas"),
        ];
        assert!(refine(text, spans).is_empty());
    }

    #[test]
    fn a_known_given_name_is_kept() {
        let text = "Sanne heeft de offerte goedgekeurd.";
        assert_eq!(refine(text, vec![person(text, "Sanne")]).len(), 1);
    }

    #[test]
    fn a_surname_after_a_title_greeting_or_label_is_kept() {
        for (text, value) in [
            ("Rufen Sie Herrn Mueller an.", "Mueller"),
            ("Gentile dott. Ferrari, le confermo.", "Ferrari"),
            ("Dear Okonkwo, thank you.", "Okonkwo"),
            ("Klant: Verhoeven", "Verhoeven"),
            ("Thanks,\nDESMOND\nFlat 2", "DESMOND"),
        ] {
            assert_eq!(refine(text, vec![person(text, value)]).len(), 1, "{text}");
        }
    }

    #[test]
    fn a_label_needs_its_colon() {
        let text = "le formulaire de Lubrizol";
        assert!(refine(text, vec![person(text, "Lubrizol")]).is_empty());
    }

    #[test]
    fn the_other_mentions_of_a_found_name_are_masked() {
        let text = "Rachel Okafor opened. Later Okafor and Rachel agreed.";
        let refined = refine(text, vec![person(text, "Rachel Okafor")]);
        assert_eq!(
            values(text, &refined),
            vec![
                ("Rachel Okafor".to_string(), EntityType::Person),
                ("Okafor".to_string(), EntityType::Person),
                ("Rachel".to_string(), EntityType::Person),
            ]
        );
    }

    #[test]
    fn a_name_word_next_to_a_number_is_not_propagated() {
        let text = "Jan de Vries signed on 5 Jan 2026.";
        let refined = refine(text, vec![person(text, "Jan de Vries")]);
        assert_eq!(refined.len(), 1);
    }

    #[test]
    fn particles_are_not_propagated() {
        let text = "Anna De Vries met De Boer.";
        let refined = refine(text, vec![person(text, "Anna De Vries")]);
        assert_eq!(
            values(text, &refined),
            vec![("Anna De Vries".to_string(), EntityType::Person)]
        );
    }

    #[test]
    fn the_given_name_list_is_sorted_for_binary_search() {
        let lines: Vec<&str> = GIVEN_NAMES.lines().collect();
        assert!(lines.windows(2).all(|pair| pair[0] < pair[1]));
    }
}
