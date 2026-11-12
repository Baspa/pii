//! The city gazetteer, and the one question it is allowed to answer.
//!
//! # Why a city list exists here at all
//!
//! This product retired `LOCATION` on purpose. Masking every place name in
//! running text destroys the utility of the text and buys almost nothing:
//! a city standing on its own is not personal data, and
//! `training/taxonomy.py` maps every corpus `CITY` label to `None` for
//! exactly that reason. Nothing here reopens that ruling.
//!
//! A city at the *end of a postal address* is a different fact.
//! `Kerkstraat 12, 1016 GV Amsterdam` identifies a household, and the
//! household is the thing `STREET_ADDRESS` exists to protect. So the
//! gazetteer answers one question and only one:
//!
//! > the model found an address ending at position N -- is what sits
//! > immediately next to it a city?
//!
//! # The constraint that makes it safe, stated as an invariant
//!
//! **Nothing in this crate may consult this module except from a position
//! already anchored on a model `STREET_ADDRESS` span.** Its only caller is
//! [`crate::decode::extend_address_tail_city`], which is handed a span the
//! model produced and looks exactly one gap-of-one-space to the right or
//! left of it. There is no scan of the document, no iteration over the
//! entries, and no other call site (a test in `decode.rs` asserts this
//! module's names appear nowhere else in the crate).
//!
//! The constraint is not decoration. Standing alone, a city list is the
//! `LOCATION` failure verbatim, and the collisions in these six languages
//! are severe: `Best` and `Ede` are Dutch towns, `Essen` is German for
//! food, `Hof` is a yard, and `Nice`, `Bath`, `Reading`, `Como`, `Prato`
//! and `León` are all at once real cities and ordinary words.
//! Positionally, every one of those collisions disappears -- nothing but a
//! city ever appears in the slot immediately after a postcode.
//!
//! # Where the data comes from
//!
//! GeoNames' `cities1000` dump, six countries (NL, DE, FR, ES, IT, GB),
//! CC BY 4.0. Source, licence, retrieval date, dump version, entry count
//! and the exact regeneration command are recorded in
//! `crates/model/gazetteer/LICENSE-GAZETTEER.md`, and the script
//! that produced `cities.txt` from the dump is committed beside it.
//!
//! # Shape
//!
//! `cities.txt` is one lower-cased name per line, sorted by UTF-8 byte
//! order -- `str`'s own `Ord` -- so [`contains`] is a binary search over a
//! borrowed index built once per process, never a linear scan per request.
//! The blob itself is `include_str!`'d, so it lives in the binary's
//! read-only data and is never read from disk at runtime, the same way the
//! rule packs are compiled in (`pii_detection_rules::pack`).

use std::sync::OnceLock;

/// One lower-cased name per line, byte-order sorted. See the module docs
/// for provenance and `gazetteer/LICENSE-GAZETTEER.md` for the licence.
const CITIES: &str = include_str!("../gazetteer/cities.txt");

/// The blob, indexed once: the lines as a sorted slice, plus the two size
/// limits a caller needs to know how far to look before giving up.
struct Gazetteer {
    /// Sorted by `str`'s `Ord`, which is byte order -- the order
    /// `build_cities.py` writes and [`contains`] binary-searches.
    entries: Box<[&'static str]>,
    /// Words in the longest entry (`kingston upon thames`, and longer).
    /// A caller reading tokens rightward or leftward stops here rather
    /// than at a number invented in the caller.
    longest_words: usize,
    /// Code points in the longest entry, for the same reason.
    longest_chars: usize,
}

fn gazetteer() -> &'static Gazetteer {
    static GAZETTEER: OnceLock<Gazetteer> = OnceLock::new();
    GAZETTEER.get_or_init(|| {
        let entries: Vec<&'static str> = CITIES.lines().filter(|line| !line.is_empty()).collect();
        let longest_words = entries
            .iter()
            .map(|entry| entry.split(' ').count())
            .max()
            .unwrap_or(0);
        let longest_chars = entries
            .iter()
            .map(|entry| entry.chars().count())
            .max()
            .unwrap_or(0);
        Gazetteer {
            entries: entries.into_boxed_slice(),
            longest_words,
            longest_chars,
        }
    })
}

/// Whether `phrase` names a city in one of the six countries.
///
/// `phrase` is compared case-insensitively, with the typographic
/// apostrophe folded onto the ASCII one (`Bo’ness` and `Bo'ness` reach the
/// same entry) -- `build_cities.py` applies the identical fold when it
/// writes the file, so the two sides cannot drift apart. Whitespace is not
/// normalised here: the caller builds the phrase token by token and joins
/// with single spaces, which is the form the file stores.
///
/// **Call this only from a position anchored on a model `STREET_ADDRESS`
/// span** -- see the module docs for why that is the whole of this
/// module's safety argument.
pub fn contains(phrase: &str) -> bool {
    if phrase.is_empty() {
        return false;
    }
    let folded = fold(phrase);
    gazetteer().entries.binary_search(&folded.as_str()).is_ok()
}

/// Words in the longest entry: how many tokens a caller need ever read
/// before it can stop looking.
pub fn longest_phrase_words() -> usize {
    gazetteer().longest_words
}

/// Code points in the longest entry: how far a caller need ever slice.
pub fn longest_phrase_chars() -> usize {
    gazetteer().longest_chars
}

/// Lower-cases and folds `’` onto `'`. Allocates, but only for a phrase a
/// caller already decided sits next to an address span -- at most a
/// handful per document, never per token of the document.
fn fold(phrase: &str) -> String {
    phrase
        .chars()
        .map(|character| {
            if character == '\u{2019}' {
                '\''
            } else {
                character
            }
        })
        .collect::<String>()
        .to_lowercase()
}

/// How many entries the compiled-in gazetteer holds. Exists so a test can
/// pin the committed file's size against
/// `gazetteer/LICENSE-GAZETTEER.md`'s recorded count -- a data file that
/// silently changed size is a data file nobody has re-licensed.
pub fn entry_count() -> usize {
    gazetteer().entries.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_committed_file_holds_the_number_of_entries_the_licence_notice_records() {
        // gazetteer/LICENSE-GAZETTEER.md records this count as part of the
        // attribution. A regenerated dump that changed it must change the
        // notice in the same commit, exactly as
        // benchmark/LICENSE-CARRIERS.md requires of the carrier counts.
        assert_eq!(entry_count(), 43_848);
    }

    #[test]
    fn the_entries_are_sorted_so_the_binary_search_is_valid() {
        // The whole lookup rests on build_cities.py having sorted by UTF-8
        // byte order rather than by a locale. A file sorted any other way
        // would not fail loudly -- it would silently miss accented names.
        let entries = &gazetteer().entries;
        assert!(entries.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn a_city_from_each_of_the_six_countries_is_found() {
        for city in [
            "amsterdam",
            "ansbach",
            "quimper",
            "teruel",
            "ascoli piceno",
            "bury st edmunds",
        ] {
            assert!(contains(city), "{city}");
        }
    }

    #[test]
    fn lookup_is_case_insensitive_and_folds_the_typographic_apostrophe() {
        assert!(contains("AMSTERDAM"));
        assert!(contains("Ascoli Piceno"));
        assert!(contains("Bo\u{2019}ness"));
        assert!(contains("Bo'ness"));
    }

    #[test]
    fn accented_and_ascii_spellings_both_resolve() {
        // build_cities.py keeps GeoNames' `name` and `asciiname` both, so
        // text that dropped the accent still reaches the entry.
        assert!(contains("Málaga"));
        assert!(contains("Malaga"));
        assert!(contains("Nîmes"));
        assert!(contains("Nimes"));
    }

    #[test]
    fn an_ordinary_word_that_is_not_a_city_is_not_found() {
        for word in ["factuur", "rechnung", "bestelling", "consegna"] {
            assert!(!contains(word), "{word}");
        }
    }

    #[test]
    fn the_collision_words_are_in_the_list_which_is_why_position_is_the_guard() {
        // Not a wish, a fact: `Best`, `Ede`, `Essen`, `Hof`, `Nice`,
        // `Bath`, `Reading`, `Como`, `Prato` and `León` are all real
        // cities AND ordinary words in these six languages. This test pins that
        // they are in the list, so nobody can later argue the gazetteer is
        // safe to consult in open text because "the collisions were
        // filtered out". They were not, and they cannot be. The only guard
        // is `decode::extend_address_tail_city`'s position rule.
        for word in [
            "best", "ede", "essen", "hof", "nice", "bath", "reading", "como", "prato", "león",
        ] {
            assert!(contains(word), "{word}");
        }
    }

    #[test]
    fn no_entry_is_short_enough_to_be_a_preposition() {
        // build_cities.py drops entries of two code points or fewer (`Ay`,
        // `Us`, `Re`): adjacent to an address those are far likelier to be
        // a function word than a village.
        assert!(
            gazetteer()
                .entries
                .iter()
                .all(|entry| entry.chars().count() > 2)
        );
    }

    #[test]
    fn the_size_limits_describe_the_data_rather_than_a_guess() {
        assert!(longest_phrase_words() >= 3, "multi-word cities exist");
        assert_eq!(
            longest_phrase_words(),
            gazetteer()
                .entries
                .iter()
                .map(|entry| entry.split(' ').count())
                .max()
                .unwrap_or(0)
        );
        assert!(longest_phrase_chars() >= "bury st edmunds".len());
    }

    #[test]
    fn an_empty_phrase_is_never_a_city() {
        assert!(!contains(""));
    }
}
