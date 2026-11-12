/// The checksum rules a pack may name.
///
/// Checksums are code, not data: a pack names one and the registry resolves it.
/// A pack naming a validator that is not here fails the pack tests rather than
/// silently accepting every candidate.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Validator {
    None,
    Elfproef,
    GermanTaxId,
    Luhn,
    IbanMod97,
    EsDni,
    EsNie,
    ItCodiceFiscale,
    FrNir,
    BeNationalRegister,
    DeFuehrerschein,
}

impl Validator {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "none" => Some(Self::None),
            "elfproef" => Some(Self::Elfproef),
            "german_tax_id" => Some(Self::GermanTaxId),
            "luhn" => Some(Self::Luhn),
            "iban_mod97" => Some(Self::IbanMod97),
            "es_dni" => Some(Self::EsDni),
            "es_nie" => Some(Self::EsNie),
            "it_codice_fiscale" => Some(Self::ItCodiceFiscale),
            "fr_nir" => Some(Self::FrNir),
            "be_national_register" => Some(Self::BeNationalRegister),
            "de_fuehrerschein" => Some(Self::DeFuehrerschein),
            _ => None,
        }
    }

    pub fn accepts(&self, value: &str) -> bool {
        match self {
            Self::None => true,
            Self::Elfproef => elfproef(value),
            Self::GermanTaxId => german_tax_id(value),
            Self::Luhn => luhn(value),
            Self::IbanMod97 => iban_mod97(value),
            Self::EsDni => es_dni(value),
            Self::EsNie => es_nie(value),
            Self::ItCodiceFiscale => it_codice_fiscale(value),
            Self::FrNir => fr_nir(value),
            Self::BeNationalRegister => be_national_register(value),
            Self::DeFuehrerschein => de_fuehrerschein(value),
        }
    }
}

fn digits(value: &str) -> Option<Vec<u32>> {
    value
        .chars()
        .filter(|c| !matches!(c, ' ' | '.' | '-' | '/'))
        .map(|c| c.to_digit(10))
        .collect()
}

/// The matched span with its separators removed and ASCII letters upper-cased.
///
/// Every country identifier below is written with optional separators (a
/// Spanish DNI as `12345678Z` or `12345678-Z`) and the matcher hands a
/// validator the exact substring the pattern matched, separators included. One
/// normalisation, shared, so two validators cannot disagree about what a
/// separator is.
fn compact(value: &str) -> String {
    value
        .chars()
        .filter(|c| !matches!(c, ' ' | '.' | '-' | '/'))
        .map(|c| c.to_ascii_uppercase())
        .collect()
}

/// The Dutch 11-proef: nine digits, the last weighted minus one.
fn elfproef(value: &str) -> bool {
    let Some(digits) = digits(value) else {
        return false;
    };
    if digits.len() != 9 {
        return false;
    }
    let mut total: i64 = 0;
    for (index, digit) in digits.iter().enumerate() {
        let weight: i64 = if index == 8 { -1 } else { 9 - index as i64 };
        total += weight * i64::from(*digit);
    }
    total % 11 == 0
}

/// The German Steuer-ID check digit, ISO 7064 MOD 11,10.
fn german_tax_id(value: &str) -> bool {
    let Some(digits) = digits(value) else {
        return false;
    };
    if digits.len() != 11 {
        return false;
    }
    let mut product = 10;
    for digit in &digits[..10] {
        let mut sum = (digit + product) % 10;
        if sum == 0 {
            sum = 10;
        }
        product = (sum * 2) % 11;
    }
    let check = (11 - product) % 10;
    check == digits[10]
}

fn luhn(value: &str) -> bool {
    let Some(digits) = digits(value) else {
        return false;
    };
    if digits.len() < 12 {
        return false;
    }
    let mut total = 0;
    for (index, digit) in digits.iter().rev().enumerate() {
        let mut value = *digit;
        if index % 2 == 1 {
            value *= 2;
            if value > 9 {
                value -= 9;
            }
        }
        total += value;
    }
    total % 10 == 0
}

/// The 23 letters a Spanish check digit is drawn from, in the order the
/// remainder indexes them.
///
/// Published by the Ministerio del Interior as the NIF/NIE control-digit
/// table. Twenty-three because the alphabet is reduced to the letters that
/// cannot be confused for one another in handwriting; the number 23 being
/// prime is what makes the check catch every single-digit error and every
/// transposition of adjacent digits (a swap moves the value by a multiple of
/// 9·10^k, and 23 divides neither 9 nor 10).
const ES_CHECK_LETTERS: [u8; 23] = *b"TRWAGMYFPDXBNJZSQVHLCKE";

/// The check letter for an eight-digit Spanish body, or `None` when the body
/// is not eight digits.
fn es_check_letter(body: &str) -> Option<u8> {
    if body.len() != 8 || !body.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let number: u32 = body.parse().ok()?;
    ES_CHECK_LETTERS.get((number % 23) as usize).copied()
}

/// Spanish DNI (and, for a natural person, the NIF that equals it): eight
/// digits and one check letter, the letter being `"TRWAGMYFPDXBNJZSQVHLCKE"`
/// indexed by the eight digits modulo 23.
///
/// Deliberately refuses the company NIF/CIF shape (a leading letter and a
/// check character that may be a digit). That identifier belongs to a legal
/// person, not a natural one, and its international form is already the
/// universal pack's VAT recognizer -- see `packs/es.toml` on why no bare CIF
/// recognizer exists.
fn es_dni(value: &str) -> bool {
    let compact = compact(value);
    if compact.len() != 9 {
        return false;
    }
    let (body, check) = compact.split_at(8);
    es_check_letter(body).is_some_and(|expected| check.as_bytes() == [expected])
}

/// Spanish NIE, the identity number issued to a foreign national: `X`, `Y` or
/// `Z`, seven digits, and one check letter.
///
/// The leading letter is replaced by its own index (`X`->0, `Y`->1, `Z`->2)
/// to form the eight-digit body the DNI rule is then applied to. The three
/// letters are the three issuing series in order, which is why the mapping is
/// positional rather than arbitrary.
fn es_nie(value: &str) -> bool {
    let compact = compact(value);
    if compact.len() != 9 {
        return false;
    }
    let series = match compact.as_bytes().first() {
        Some(b'X') => '0',
        Some(b'Y') => '1',
        Some(b'Z') => '2',
        _ => return false,
    };
    let (rest, check) = compact[1..].split_at(7);
    let body = format!("{series}{rest}");
    es_check_letter(&body).is_some_and(|expected| check.as_bytes() == [expected])
}

/// The twelve month letters of an Italian codice fiscale, in month order:
/// A is January, H is June, T is December. Six of the twenty-six letters are
/// skipped precisely so a misread month cannot land on a valid one.
const IT_MONTH_LETTERS: &str = "ABCDEHLMPRST";

/// The omocodia substitution table, indexed by the digit each letter stands
/// for: `L` is 0, `M` is 1, up to `V` for 9.
///
/// When two people would be issued the same code, the Agenzia delle Entrate
/// replaces its numeric characters with these letters, from the rightmost
/// numeric position leftwards, and recomputes the check character. Those
/// codes are issued and appear on documents, so a validator that rejected
/// them would silently under-mask the people who hold them.
const IT_OMOCODIA_LETTERS: &str = "LMNPQRSTUV";

/// The digit a codice fiscale character stands for: itself, or the digit its
/// omocodia letter substitutes for.
fn it_digit(character: char) -> Option<u32> {
    character.to_digit(10).or_else(|| {
        IT_OMOCODIA_LETTERS
            .find(character)
            .and_then(|index| u32::try_from(index).ok())
    })
}

/// The value a character contributes from an odd position (counting from
/// one), which is every even index here. Published by the Agenzia delle
/// Entrate as the CIN table; the digits and the first ten letters share a
/// row because a digit and its alphabetical twin are meant to score alike.
fn it_odd_value(character: char) -> Option<u32> {
    const DIGITS: [u32; 10] = [1, 0, 5, 7, 9, 13, 15, 17, 19, 21];
    const LETTERS: [u32; 26] = [
        1, 0, 5, 7, 9, 13, 15, 17, 19, 21, 2, 4, 18, 20, 11, 3, 6, 8, 12, 14, 16, 10, 22, 25, 24,
        23,
    ];
    match character.to_digit(10) {
        Some(digit) => DIGITS.get(digit as usize).copied(),
        None => LETTERS
            .get(usize::from(
                u8::try_from(character).ok()?.checked_sub(b'A')?,
            ))
            .copied(),
    }
}

/// The value a character contributes from an even position: a digit is
/// worth itself and a letter its place in the alphabet.
fn it_even_value(character: char) -> Option<u32> {
    match character.to_digit(10) {
        Some(digit) => Some(digit),
        None => {
            Some(u32::from(u8::try_from(character).ok()?.checked_sub(b'A')?)).filter(|v| *v < 26)
        }
    }
}

/// Italian codice fiscale for a natural person: sixteen characters, the last
/// a check letter over the other fifteen modulo 26.
///
/// The structure is checked as well as the check letter, because the code
/// encodes a birth date and half of what it encodes is constrained: the
/// ninth character is one of twelve month letters and the day is 01 to 31,
/// or 41 to 71 for a woman. Those two constraints cut the shapes a random
/// sixteen-character string can take by roughly a factor of eight before the
/// one-in-twenty-six check letter is consulted at all.
///
/// Omocodia is handled: a numeric position may carry the letter that
/// substitutes for its digit, and the day is decoded through that table
/// before it is range-checked.
fn it_codice_fiscale(value: &str) -> bool {
    let compact = compact(value);
    let characters: Vec<char> = compact.chars().collect();
    if characters.len() != 16 || !characters.iter().all(|c| c.is_ascii_alphanumeric()) {
        return false;
    }

    // Six letters of name, then a numeric-or-substituted year.
    if !characters[..6].iter().all(char::is_ascii_alphabetic) {
        return false;
    }
    let numeric_positions = [6, 7, 9, 10, 12, 13, 14];
    if !numeric_positions
        .iter()
        .all(|index| characters.get(*index).copied().and_then(it_digit).is_some())
    {
        return false;
    }
    if !IT_MONTH_LETTERS.contains(characters[8]) {
        return false;
    }
    if !characters[11].is_ascii_alphabetic() || !characters[15].is_ascii_alphabetic() {
        return false;
    }

    let (Some(tens), Some(units)) = (it_digit(characters[9]), it_digit(characters[10])) else {
        return false;
    };
    // A day is 1 to 31; a woman's carries 40 added to it.
    let day = tens * 10 + units;
    if !(1..=31).contains(&day) && !(41..=71).contains(&day) {
        return false;
    }

    let mut total = 0u32;
    for (index, character) in characters[..15].iter().enumerate() {
        let value = if index.is_multiple_of(2) {
            it_odd_value(*character)
        } else {
            it_even_value(*character)
        };
        match value {
            Some(value) => total += value,
            None => return false,
        }
    }
    let Some(expected) = u8::try_from(total % 26)
        .ok()
        .and_then(|offset| b'A'.checked_add(offset))
    else {
        return false;
    };
    characters[15] == char::from(expected)
}

/// French NIR -- the numéro de sécurité sociale: thirteen characters of sex,
/// birth date and birthplace, then a two-digit key that is 97 minus the
/// thirteen taken modulo 97.
///
/// CORSICA IS THE PART EVERY IMPLEMENTATION GETS WRONG. Departments `2A` and
/// `2B` carry a letter, and the published rule is to substitute `19` for `2A`
/// and `18` for `2B` before dividing. Skipping that does not produce a wrong
/// answer, it produces no answer at all -- the number will not parse -- so a
/// validator without it silently refuses every Corsican-born person's number,
/// which is an under-mask and therefore a leak.
///
/// WHAT THIS DELIBERATELY DOES NOT ACCEPT. The month is required to be 01 to
/// 12. The published structure also allows codes outside that range for
/// records with an incomplete birth date, and accepting them would take the
/// month from one-in-eight selectivity to almost none -- which is most of
/// what lets a fifteen-digit run sit under a gate demanding 99% precision.
/// The cost is stated rather than hidden: a NIR issued with a non-calendar
/// month is not detected by this recognizer.
fn fr_nir(value: &str) -> bool {
    let compact = compact(value);
    if compact.len() != 15 || !compact.is_ascii() {
        return false;
    }
    let (body, key) = compact.split_at(13);
    let Ok(key) = key.parse::<u64>() else {
        return false;
    };
    // The key is a complement to 97, so it never reads 00 or 98.
    if !(1..=97).contains(&key) {
        return false;
    }

    // 1 and 2 are the sexes as issued; 3, 4, 7 and 8 stand for a record
    // whose registration is provisional.
    if !matches!(
        body.as_bytes().first(),
        Some(b'1' | b'2' | b'3' | b'4' | b'7' | b'8')
    ) {
        return false;
    }
    let Some(month) = body.get(3..5).and_then(|m| m.parse::<u32>().ok()) else {
        return false;
    };
    if !(1..=12).contains(&month) {
        return false;
    }
    let Some(department) = body.get(5..7) else {
        return false;
    };
    let substituted = match department {
        "2A" => "19",
        "2B" => "18",
        // Every real department is 01 to 95, or 96 to 99 for a birth
        // overseas or abroad. `00` is not issued.
        other if other.bytes().all(|b| b.is_ascii_digit()) && other != "00" => other,
        _ => return false,
    };
    let Some(head) = body.get(..5) else {
        return false;
    };
    let Some(tail) = body.get(7..) else {
        return false;
    };
    let Ok(numeric) = format!("{head}{substituted}{tail}").parse::<u64>() else {
        return false;
    };
    97 - numeric % 97 == key
}

/// Belgian national register number (rijksregisternummer / numéro de
/// registre national / INSZ): birth date, a three-digit sequence, and a
/// two-digit control that is 97 minus the preceding nine taken modulo 97.
///
/// TWO BRANCHES, BECAUSE THE FORMAT DID NOT CHANGE FOR THE YEAR 2000. The
/// year is two digits, so `05` is 1905 or 2005 and nothing in the number
/// says which. The official instruction resolves it by prefixing a `2` to
/// the nine digits before dividing, for anyone born from 2000 onwards: a
/// number is valid if *either* branch produces the control it carries, and
/// which branch matched is what tells the century apart. Implementing only
/// the first branch would silently refuse every person born this century --
/// an under-mask, and therefore a leak, that gets worse every year.
///
/// BIS NUMBERS ARE ACCEPTED. A person with no Belgian residence -- a
/// cross-border worker, a foreign director -- is issued a bis number in the
/// same format with 20 or 40 added to the month. It carries the same
/// control and identifies a natural person exactly as the register number
/// does, so a month of 21-32 or 41-52 is accepted alongside 01-12.
fn be_national_register(value: &str) -> bool {
    let compact = compact(value);
    if compact.len() != 11 || !compact.bytes().all(|byte| byte.is_ascii_digit()) {
        return false;
    }
    let (Some(month), Some(day), Some(sequence), Some(control)) = (
        compact.get(2..4).and_then(|s| s.parse::<u32>().ok()),
        compact.get(4..6).and_then(|s| s.parse::<u32>().ok()),
        compact.get(6..9).and_then(|s| s.parse::<u32>().ok()),
        compact.get(9..).and_then(|s| s.parse::<u64>().ok()),
    ) else {
        return false;
    };
    // 00 is the month of a record whose birth date is not known; 20 and 40
    // are its bis equivalents.
    let real_month = match month {
        0 | 20 | 40 => 0,
        21..=32 => month - 20,
        41..=52 => month - 40,
        other => other,
    };
    if real_month > 12 || day > 31 {
        return false;
    }
    // The instruction issues 001 to 997 to men and 002 to 998 to women;
    // neither 000 nor 999 is assigned.
    if sequence == 0 || sequence == 999 {
        return false;
    }
    let Some(nine) = compact.get(..9).and_then(|s| s.parse::<u64>().ok()) else {
        return false;
    };
    let before_2000 = 97 - nine % 97;
    let from_2000 = 97 - (2_000_000_000 + nine) % 97;
    control == before_2000 || control == from_2000
}

/// The value a Führerscheinnummer character contributes: a digit is worth
/// itself, and a letter its place in the alphabet plus nine, so `A` is 10
/// and `Z` is 35.
///
/// The same base-36 reading the FIN (vehicle identification number) uses,
/// which is why the published description of the licence check digit calls
/// it "the modulo-11 procedure, identical to the FIN".
fn de_licence_value(character: char) -> Option<u32> {
    character.to_digit(36)
}

/// German Führerscheinnummer (field 5 of the card licence): eleven
/// characters, of which the tenth is a modulo-11 check digit over the first
/// nine.
///
/// STRUCTURE, from the published description of the numbering scheme. The
/// first four characters are the Behördenschlüssel -- one letter naming the
/// Bundesland (`A` Baden-Württemberg through `P` Thüringen, alphabetically),
/// then the district and office digits. The next five are the running
/// alphanumeric serial. The tenth is the check digit. The eleventh is the
/// Ausfertigungsnummer, which counts reissues and takes no part in the check.
///
/// THE CHECK. The nine characters of authority key and serial are read in
/// base 36 (`A` = 10 … `Z` = 35), multiplied by 9, 8, 7 … 1, summed, and
/// taken modulo eleven. A remainder of ten is written `X`. Worked on the
/// published specimen `B072RRE2I55`: 11·9 + 0·8 + 7·7 + 2·6 + 27·5 + 27·4 +
/// 14·3 + 2·2 + 18·1 = 467, and 467 mod 11 = 5.
///
/// WHY THE FIRST CHARACTER IS REQUIRED TO BE A LETTER, AND WHAT IT COSTS.
/// It is the Bundesland key, so every licence issued by one of the sixteen
/// Länder carries one. Allowing a digit there as well -- which is what
/// Microsoft Purview's own pattern for this identifier does -- makes the
/// shape `[A-Z0-9]\d\d[A-Z0-9]{6}[0-9X][A-Z0-9]`, which an eleven-digit run
/// satisfies: that is `de_steuer_id`'s and `be_national_register`'s exact
/// shape, and one eleven-digit run in eleven clears this check by chance.
/// Measured over the benchmark fixture as it stands, the permissive shape
/// raises a licence candidate over 15 committed Steuer-IDs, 1 national
/// register number and 1 order reference that must stay clear; the shape
/// with the letter required raises none, on the fixture, on the 321
/// carriers, and on 23.3M characters of PII-free prose. The cost is a
/// licence issued outside the sixteen Länder keys -- if any is -- and it is
/// stated here rather than hidden.
///
/// The eleventh character is deliberately *not* constrained. It is a reissue
/// counter that runs 1-9 and then into letters, and there is no published
/// bound on how far it runs.
fn de_fuehrerschein(value: &str) -> bool {
    let compact = compact(value);
    let characters: Vec<char> = compact.chars().collect();
    if characters.len() != 11 || !characters.iter().all(char::is_ascii_alphanumeric) {
        return false;
    }
    if !characters[0].is_ascii_alphabetic()
        || !characters[1].is_ascii_digit()
        || !characters[2].is_ascii_digit()
    {
        return false;
    }

    let mut total = 0u32;
    for (index, character) in characters[..9].iter().enumerate() {
        let Some(value) = de_licence_value(*character) else {
            return false;
        };
        let Some(weight) = u32::try_from(9 - index).ok() else {
            return false;
        };
        total += value * weight;
    }
    let expected = match total % 11 {
        10 => Some('X'),
        remainder => char::from_digit(remainder, 10),
    };
    expected.is_some_and(|expected| characters[9] == expected)
}

/// IBAN mod-97: move the first four characters to the end, map letters to
/// numbers, and the remainder must be one.
fn iban_mod97(value: &str) -> bool {
    let compact: String = value.chars().filter(|c| !c.is_whitespace()).collect();

    // Reject if any non-ASCII-alphanumeric character (prevents panic on multi-byte UTF-8
    // and ensures only valid IBAN characters are processed).
    if !compact.chars().all(|c| c.is_ascii_alphanumeric()) {
        return false;
    }

    // Real IBANs are 15 to 34 characters; count in characters, not bytes.
    let char_len = compact.chars().count();
    if !(15..=34).contains(&char_len) {
        return false;
    }

    // Now safe to index: compact contains only ASCII alphanumeric, so char boundaries align with byte boundaries.
    let rearranged = format!("{}{}", &compact[4..], &compact[..4]);
    let mut remainder: u32 = 0;
    for character in rearranged.chars() {
        let digit = character.to_digit(36);
        let Some(digit) = digit else { return false };
        if digit < 10 {
            remainder = (remainder * 10 + digit) % 97;
        } else {
            remainder = (remainder * 100 + digit) % 97;
        }
    }
    remainder == 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elfproef_accepts_a_valid_bsn_and_rejects_a_near_miss() {
        assert!(Validator::Elfproef.accepts("123456782"));
        assert!(!Validator::Elfproef.accepts("123456789"));
        assert!(!Validator::Elfproef.accepts("12345678"));
    }

    #[test]
    fn german_tax_id_check_digit() {
        assert!(Validator::GermanTaxId.accepts("86095742719"));
        assert!(Validator::GermanTaxId.accepts("36574261809"));
        assert!(!Validator::GermanTaxId.accepts("44321987654"));
    }

    #[test]
    fn luhn_accepts_a_test_card_number() {
        assert!(Validator::Luhn.accepts("4111111111111111"));
        assert!(!Validator::Luhn.accepts("4111111111111112"));
    }

    #[test]
    fn iban_mod_97_accepts_a_real_iban() {
        assert!(Validator::IbanMod97.accepts("NL91ABNA0417164300"));
        assert!(!Validator::IbanMod97.accepts("NL91ABNA0417164301"));
    }

    #[test]
    fn iban_mod_97_rejects_multi_byte_characters_without_panicking() {
        // This candidate would panic on byte-index slicing if not guarded.
        assert!(!Validator::IbanMod97.accepts("aa€aaaa"));
    }

    #[test]
    fn iban_mod_97_rejects_candidates_shorter_than_15_characters() {
        // Five-character garbage string; too short to be a valid IBAN.
        assert!(!Validator::IbanMod97.accepts("AB1234"));
    }

    /// Every adjacent-digit transposition of `value` that actually changes it.
    ///
    /// A checksum's whole job is to catch the two errors a human makes
    /// copying a number: one wrong digit, and two digits swapped. A validator
    /// that accepts a transposition of a value it just accepted is not
    /// checking the thing it claims to check, and this repository has shipped
    /// guards that claimed more than they checked before. So every checksum
    /// added here is mutated and re-asked, rather than being tested only on
    /// values it was always going to accept.
    fn adjacent_transpositions(value: &str) -> Vec<String> {
        let characters: Vec<char> = value.chars().collect();
        let mut mutated = Vec::new();
        for index in 1..characters.len() {
            if characters[index - 1] == characters[index] {
                continue;
            }
            let mut swapped = characters.clone();
            swapped.swap(index - 1, index);
            mutated.push(swapped.into_iter().collect());
        }
        mutated
    }

    /// `12345678Z` is the worked example the Ministerio del Interior's own
    /// NIF/NIE control-digit page carries (8 digits mod 23 = 14, and the
    /// fifteenth letter of `TRWAGMYFPDXBNJZSQVHLCKE` is `Z`). `00000000T` is
    /// the same table's first entry, remainder zero.
    #[test]
    fn es_dni_accepts_the_published_examples() {
        for published in ["12345678Z", "12345678-Z", "00000000T"] {
            assert!(Validator::EsDni.accepts(published), "{published}");
        }
        // Lower case is how a customer types it in a hurry.
        assert!(Validator::EsDni.accepts("12345678z"));
    }

    #[test]
    fn es_dni_rejects_a_wrong_letter_a_wrong_length_and_a_nie() {
        assert!(!Validator::EsDni.accepts("12345678A"));
        assert!(!Validator::EsDni.accepts("1234567Z"));
        assert!(!Validator::EsDni.accepts("123456789"));
        // The NIE series letter leads; this validator's body must be digits.
        assert!(!Validator::EsDni.accepts("X1234567L"));
    }

    #[test]
    fn es_dni_rejects_every_transposition_of_a_published_example() {
        for published in ["12345678Z", "00000000T"] {
            for mutated in adjacent_transpositions(published) {
                assert!(
                    !Validator::EsDni.accepts(&mutated),
                    "a transposition of a valid DNI was accepted"
                );
            }
        }
    }

    /// `X1234567L` and `X0000000T` are both published as example NIEs with
    /// their control letters: the series letter becomes `0`, so the body is
    /// `01234567` (mod 23 = 19 -> `L`) and `00000000` (mod 23 = 0 -> `T`).
    #[test]
    fn es_nie_accepts_the_published_examples() {
        for published in ["X1234567L", "X-1234567-L", "X0000000T"] {
            assert!(Validator::EsNie.accepts(published), "{published}");
        }
    }

    #[test]
    fn es_nie_rejects_a_wrong_series_letter_and_a_dni() {
        // `W` is not one of the three issuing series.
        assert!(!Validator::EsNie.accepts("W1234567L"));
        assert!(!Validator::EsNie.accepts("X1234567M"));
        assert!(!Validator::EsNie.accepts("12345678Z"));
        // `Y` and `Z` shift the body by one and two hundred million, so the
        // same digits do not keep the same letter.
        assert!(!Validator::EsNie.accepts("Y1234567L"));
        assert!(!Validator::EsNie.accepts("Z1234567L"));
    }

    #[test]
    fn es_nie_rejects_every_transposition_of_a_published_example() {
        for mutated in adjacent_transpositions("X1234567L") {
            assert!(
                !Validator::EsNie.accepts(&mutated),
                "a transposition of a valid NIE was accepted"
            );
        }
    }

    /// `MRTMTT91D08F205J` and `MLLSNT82P65Z404U` are the two complete worked
    /// examples the English Wikipedia article on the Italian fiscal code
    /// carries. `BBBTTT20H12X122H` is a third, from a published
    /// step-by-step derivation whose sum (111) and remainder (7 -> `H`) this
    /// implementation reproduces exactly.
    #[test]
    fn it_codice_fiscale_accepts_the_published_examples() {
        for published in ["MRTMTT91D08F205J", "MLLSNT82P65Z404U", "BBBTTT20H12X122H"] {
            assert!(Validator::ItCodiceFiscale.accepts(published), "{published}");
        }
        // Lower case, as a customer types it.
        assert!(Validator::ItCodiceFiscale.accepts("mrtmtt91d08f205j"));
    }

    /// THE MOST-QUOTED EXAMPLE OF AN ITALIAN FISCAL CODE IS NOT A VALID ONE.
    /// `RSSMRA85M01H501Z` -- Mario Rossi, born in Rome on 1 August 1985 --
    /// appears in article after article as *the* specimen codice fiscale.
    /// Its first fifteen characters sum to 120, and 120 modulo 26 is 16,
    /// which is `Q`, not `Z`. The same tables reproduce three published
    /// worked examples character for character, so the tables are not what
    /// is wrong.
    ///
    /// This is exactly the trap the VAT fixture fell into once already: a
    /// value everybody repeats, nobody checks, and an answer key built on it
    /// marks a correct detector wrong. It is pinned here as a rejection so
    /// that it can never be adopted as an anchor by mistake.
    #[test]
    fn it_codice_fiscale_rejects_the_most_widely_copied_specimen() {
        assert!(!Validator::ItCodiceFiscale.accepts("RSSMRA85M01H501Z"));
        assert!(Validator::ItCodiceFiscale.accepts("RSSMRA85M01H501Q"));
    }

    #[test]
    fn it_codice_fiscale_accepts_an_omocodia_variant() {
        // Derived, and labelled as derived: the published omocodia table
        // (`L` for 0 ... `V` for 9) applied to `MRTMTT91D08F205J` from the
        // right, with the check character recomputed by the published rule.
        // One substitution, then two. Their purpose is to exercise the
        // substitution path, not to anchor the check rule -- the three
        // published examples above do that.
        assert!(Validator::ItCodiceFiscale.accepts("MRTMTT91D08F20RE"));
        assert!(Validator::ItCodiceFiscale.accepts("MRTMTT91D08F2LRP"));
    }

    #[test]
    fn it_codice_fiscale_rejects_an_impossible_month_or_day() {
        // `F` is not one of the twelve month letters, and `00` and `35` are
        // not days. Each is rejected before the check character is reached,
        // which is what stops a sixteen-character word from needing to be
        // lucky only one time in twenty-six.
        assert!(!Validator::ItCodiceFiscale.accepts("MRTMTT91F08F205J"));
        assert!(!Validator::ItCodiceFiscale.accepts("MRTMTT91D00F205J"));
        assert!(!Validator::ItCodiceFiscale.accepts("MRTMTT91D35F205J"));
        // Fifteen characters, and seventeen.
        assert!(!Validator::ItCodiceFiscale.accepts("MRTMTT91D08F205"));
        assert!(!Validator::ItCodiceFiscale.accepts("MRTMTT91D08F205JJ"));
    }

    #[test]
    fn it_codice_fiscale_rejects_every_transposition_of_a_published_example() {
        // Modulo 26 over two different position tables: swapping two
        // adjacent characters swaps which table each is scored from, so a
        // transposition changes the sum unless the two happen to score the
        // same in both -- and where it does not change the sum, the
        // structure check catches it. Every transposition of all three
        // anchors is asked.
        for published in ["MRTMTT91D08F205J", "MLLSNT82P65Z404U", "BBBTTT20H12X122H"] {
            for mutated in adjacent_transpositions(published) {
                assert!(
                    !Validator::ItCodiceFiscale.accepts(&mutated),
                    "a transposition of a valid codice fiscale was accepted"
                );
            }
        }
    }

    /// Two published worked examples, each with its arithmetic printed:
    /// `1 74 03 30 007 058` (remainder 50, key 47) and `1 85 03 69 123 045`
    /// (remainder 65, key 32).
    #[test]
    fn fr_nir_accepts_the_published_examples() {
        for published in [
            "174033000705847",
            "1 74 03 30 007 058 47",
            "185036912304532",
            "1 85 03 69 123 045 32",
        ] {
            assert!(Validator::FrNir.accepts(published), "{published}");
        }
    }

    /// The Corsican case, which is the one payroll software gets wrong.
    /// `2 90 05 2B 033 010` is the published example of the substitution
    /// (`2B` becomes 18); its key is derived here by applying the published
    /// modulo-97 rule to that published body, and is 01.
    #[test]
    fn fr_nir_handles_the_corsican_departments() {
        assert!(Validator::FrNir.accepts("2 90 05 2B 033 010 01"));
        assert!(Validator::FrNir.accepts("290052B03301001"));
        // `2A` substitutes 19, not 18, so the same digits do not keep the
        // same key -- which is the whole reason the two are distinguished.
        assert!(!Validator::FrNir.accepts("290052A03301001"));
        assert!(Validator::FrNir.accepts("290052A03301071"));
        // Any other letter is not a department.
        assert!(!Validator::FrNir.accepts("290052C03301001"));
    }

    #[test]
    fn fr_nir_rejects_an_impossible_month_department_or_key() {
        // Month 13, department 00, key 00 and key 98 -- each rejected before
        // or by the modulo, none of them a number anyone was issued.
        assert!(!Validator::FrNir.accepts("174133000705847"));
        assert!(!Validator::FrNir.accepts("174030000705847"));
        assert!(!Validator::FrNir.accepts("174033000705800"));
        assert!(!Validator::FrNir.accepts("174033000705898"));
        // Sex digit 5 is not issued.
        assert!(!Validator::FrNir.accepts("574033000705847"));
        // Fourteen characters, and sixteen.
        assert!(!Validator::FrNir.accepts("17403300070584"));
        assert!(!Validator::FrNir.accepts("1740330007058477"));
    }

    #[test]
    fn fr_nir_rejects_every_transposition_of_a_published_example() {
        // Modulo 97 catches every adjacent transposition, for the same
        // reason modulo 23 does: a swap moves the value by a multiple of
        // 9·10^k, and 97 divides neither 9 nor 10 nor any digit difference.
        for published in ["174033000705847", "185036912304532"] {
            for mutated in adjacent_transpositions(published) {
                assert!(
                    !Validator::FrNir.accepts(&mutated),
                    "a transposition of a valid NIR was accepted"
                );
            }
        }
    }

    /// `42.01.22 051-81` is the worked example in the Belgian register's own
    /// instruction IT 000 (a man born 22 January 1942). `72020290081` and
    /// `00012556777` are the two the Flemish government's validation-methods
    /// page carries, one per century branch -- the second is the one an
    /// implementation missing the year-2000 prefix silently rejects.
    /// `93051822361` is the Dutch Wikipedia article's worked example.
    #[test]
    fn be_national_register_accepts_the_published_examples() {
        for published in [
            "42012205181",
            "42.01.22 051-81",
            "72020290081",
            "00012556777",
            "93051822361",
            "93.05.18-223.61",
        ] {
            assert!(
                Validator::BeNationalRegister.accepts(published),
                "{published}"
            );
        }
    }

    #[test]
    fn be_national_register_needs_both_century_branches() {
        // `00012556777` validates only once the nine digits are prefixed
        // with a 2; an implementation with the first branch alone rejects
        // it, and with it everyone born this century -- an under-mask that
        // grows every year.
        //
        // `00012556748` is the same nine digits with the *other* branch's
        // control, and it is a valid number too: it belongs to someone born
        // on 25 January 1900. Both must be accepted. That the two exist side
        // by side is exactly why the century cannot be read off the digits
        // and why the rule needs two branches rather than one guess.
        assert!(Validator::BeNationalRegister.accepts("00012556777"));
        assert!(Validator::BeNationalRegister.accepts("00012556748"));
        // A third control belongs to neither century.
        assert!(!Validator::BeNationalRegister.accepts("00012556750"));
        // And the pre-2000 example must not need the prefix.
        assert!(Validator::BeNationalRegister.accepts("72020290081"));
    }

    #[test]
    fn be_national_register_accepts_a_bis_number() {
        // A bis number is issued to a natural person with no Belgian
        // residence, in the same format with 20 or 40 added to the month.
        // Both are derived here, by applying the published +20/+40 rule and
        // the published modulo to `93051822361` above: month 05 becomes 25
        // and 45, and each recomputes its own control.
        assert!(Validator::BeNationalRegister.accepts("93251822307"));
        assert!(Validator::BeNationalRegister.accepts("93451822350"));
        // 60 added to the month is not a rule anyone published, whatever
        // some third-party library's test vector says.
        assert!(!Validator::BeNationalRegister.accepts("66641066692"));
    }

    #[test]
    fn be_national_register_rejects_an_impossible_date_or_sequence() {
        assert!(!Validator::BeNationalRegister.accepts("93151822361"));
        assert!(!Validator::BeNationalRegister.accepts("93054022361"));
        // Sequence 000 and 999 are not assigned.
        assert!(!Validator::BeNationalRegister.accepts("93051800061"));
        assert!(!Validator::BeNationalRegister.accepts("93051899961"));
        // Ten digits, and twelve.
        assert!(!Validator::BeNationalRegister.accepts("9305182236"));
        assert!(!Validator::BeNationalRegister.accepts("930518223611"));
    }

    #[test]
    fn be_national_register_rejects_every_transposition_of_a_published_example() {
        for published in ["42012205181", "72020290081", "93051822361", "00012556777"] {
            for mutated in adjacent_transpositions(published) {
                assert!(
                    !Validator::BeNationalRegister.accepts(&mutated),
                    "a transposition of a valid national register number was accepted"
                );
            }
        }
    }

    /// A NUMBER CAN SATISFY BOTH ELEVEN-DIGIT CHECKSUMS AT ONCE, AND ONE OF
    /// THE PUBLISHED EXAMPLES DOES.
    ///
    /// `de_steuer_id` and `be_national_register` share a pattern in the
    /// packs (`\b\d{11}\b`), and the two checks are independent: ISO 7064
    /// MOD 11,10 over the first ten digits, and 97 minus the first nine
    /// modulo 97. Roughly one Belgian number in ten also clears the German
    /// check by coincidence, and `42012205181` -- the Belgian register's own
    /// published example -- is one of them.
    ///
    /// This is pinned rather than fixed, because there is nothing to fix.
    /// Both recognizers raise a candidate over the same span; the resolver
    /// picks one, and the characters masked are identical either way, so
    /// nothing reaches the caller in the clear. Only the country label and
    /// the type differ. What would be a defect is a validator quietly
    /// widened to exclude the other's values, which would reject real
    /// numbers to make a tidier answer.
    #[test]
    fn a_number_can_satisfy_both_eleven_digit_checksums() {
        assert!(Validator::BeNationalRegister.accepts("42012205181"));
        assert!(Validator::GermanTaxId.accepts("42012205181"));

        // The everyday case is still that they disagree: the German
        // examples are not Belgian numbers and two of the three Belgian
        // ones are not German.
        for german in ["86095742719", "36574261809"] {
            assert!(!Validator::BeNationalRegister.accepts(german), "{german}");
        }
        for belgian in ["72020290081", "93051822361", "00012556777"] {
            assert!(!Validator::GermanTaxId.accepts(belgian), "{belgian}");
        }
    }

    // --- Germany, the driving licence ------------------------------------

    /// `B072RRE2I55` is the number printed on the Musterführerschein, the
    /// specimen card the numbering scheme is published against, and it is
    /// the only anchor this checksum has. Two independent sources print its
    /// arithmetic in full and reach the same place: 11·9 + 0·8 + 7·7 + 2·6 +
    /// 27·5 + 27·4 + 14·3 + 2·2 + 18·1 = 467, and 467 = 42·11 + 5.
    #[test]
    fn de_fuehrerschein_accepts_the_published_specimen() {
        assert!(Validator::DeFuehrerschein.accepts("B072RRE2I55"));
        // Lower case is how a customer types it into a ticket.
        assert!(Validator::DeFuehrerschein.accepts("b072rre2i55"));
    }

    /// The Ausfertigungsnummer counts reissues and takes no part in the
    /// check, so the same licence at its second and third issue must both
    /// validate. A validator that folded the eleventh character into the sum
    /// would accept exactly one of these.
    #[test]
    fn de_fuehrerschein_ignores_the_reissue_counter() {
        for reissue in ['1', '2', '9', 'A'] {
            assert!(
                Validator::DeFuehrerschein.accepts(&format!("B072RRE2I5{reissue}")),
                "reissue {reissue}"
            );
        }
    }

    #[test]
    fn de_fuehrerschein_rejects_a_wrong_check_digit_and_a_wrong_length() {
        for wrong in ['0', '1', '4', '6', '9', 'X'] {
            assert!(
                !Validator::DeFuehrerschein.accepts(&format!("B072RRE2I{wrong}5")),
                "check digit {wrong} must not be accepted"
            );
        }
        assert!(!Validator::DeFuehrerschein.accepts("B072RRE2I5"));
        assert!(!Validator::DeFuehrerschein.accepts("B072RRE2I555"));
    }

    /// The structure half, and the one that keeps this recognizer off
    /// `de_steuer_id`'s and `be_national_register`'s shape: an eleven-digit
    /// run is not a licence number however its digits add up.
    /// `42012205181` is the Belgian register's own published example and
    /// clears the German Steuer-ID check as well (see
    /// `a_number_can_satisfy_both_eleven_digit_checksums`); it must clear
    /// neither branch of this one.
    #[test]
    fn de_fuehrerschein_rejects_an_eleven_digit_run() {
        for digits in [
            "42012205181",
            "86095742719",
            "36574261809",
            "00000000000",
            // The one eleven-digit run whose tenth character would satisfy
            // the check digit if the first character were allowed to be a
            // digit. Constructed from the specimen by replacing `B` with the
            // digit of the same value modulo 11, which is `0`.
            "00725540255",
        ] {
            assert!(
                !Validator::DeFuehrerschein.accepts(digits),
                "{digits} is not a licence number"
            );
        }
        // And the two characters the authority key requires to be digits.
        assert!(!Validator::DeFuehrerschein.accepts("BA72RRE2I55"));
        assert!(!Validator::DeFuehrerschein.accepts("B0A2RRE2I55"));
    }

    /// A remainder of ten is written `X`, which no publication anchors: the
    /// specimen's remainder is five. `A631B7O25X1` is derived from the rule
    /// as published rather than quoted from anywhere, and it is here so the
    /// branch is exercised at all rather than being a line nothing reaches.
    #[test]
    fn de_fuehrerschein_writes_a_remainder_of_ten_as_x() {
        assert!(Validator::DeFuehrerschein.accepts("A631B7O25X1"));
        // `X` is the tenth character only when the remainder really is ten.
        assert!(!Validator::DeFuehrerschein.accepts("B072RRE2IX5"));
        // And the digit ten would otherwise be written as is rejected.
        assert!(!Validator::DeFuehrerschein.accepts("A631B7O2501"));
    }

    /// Every adjacent transposition of the published specimen is rejected --
    /// but **not all of them by the check digit**, and the distinction is
    /// stated rather than left to be assumed.
    ///
    /// Modulo 11 over the weights 9, 8, 7 … 1 catches a swap of two adjacent
    /// characters exactly when their base-36 values differ by something that
    /// is not a multiple of eleven: the swap moves the sum by (v₁ − v₂)·1.
    /// Seven of the specimen's eight transpositions are caught that way. The
    /// eighth swaps `B` (11) with `0` (0) -- a difference of exactly eleven,
    /// which the modulus cannot see -- and is rejected by the structure rule
    /// instead, because it puts a digit where the Bundesland letter belongs.
    ///
    /// The limitation itself is pinned in
    /// `de_fuehrerschein_does_not_catch_every_transposition` below, so that
    /// nobody reads this test as a claim the checksum is stronger than it is.
    #[test]
    fn de_fuehrerschein_rejects_every_transposition_of_the_published_specimen() {
        let mutations = adjacent_transpositions("B072RRE2I55");
        assert_eq!(mutations.len(), 8, "the specimen has eight of them");
        for mutated in &mutations {
            assert!(
                !Validator::DeFuehrerschein.accepts(mutated),
                "a transposition of the published specimen was accepted"
            );
        }
        // The one the modulus alone would have let through, named.
        assert_eq!(
            adjacent_transpositions("B072RRE2I55")[0],
            "0B72RRE2I55",
            "the swap the structure rule catches must be the one this says it is"
        );
    }

    /// What this checksum does **not** catch, written down so it cannot be
    /// discovered later as a surprise.
    ///
    /// `A631B7O25X1` is a licence number by the published rule. Swapping its
    /// seventh and eighth characters -- `O` (24) and `2` (2), a difference of
    /// twenty-two -- leaves the weighted sum unchanged modulo eleven and
    /// leaves the structure intact, so the result is accepted too. That is a
    /// property of modulo 11 over consecutive weights, not a defect in this
    /// implementation, and it is most of why `DRIVING_LICENCE` is gated as an
    /// unchecksummed type (see `pii_detection_eval::gates`) and why the pack's
    /// shape, not its check digit, carries the precision.
    #[test]
    fn de_fuehrerschein_does_not_catch_every_transposition() {
        assert!(Validator::DeFuehrerschein.accepts("A631B7O25X1"));
        assert!(
            Validator::DeFuehrerschein.accepts("A631B72O5X1"),
            "if this ever starts failing, the modulus changed and the report's \
             claim about what the check catches is stale"
        );
    }

    #[test]
    fn none_accepts_anything_because_the_format_already_decided() {
        assert!(Validator::None.accepts("whatever"));
    }

    #[test]
    fn every_name_a_pack_may_use_resolves() {
        for name in [
            "none",
            "elfproef",
            "german_tax_id",
            "luhn",
            "iban_mod97",
            "es_dni",
            "es_nie",
            "it_codice_fiscale",
            "fr_nir",
            "be_national_register",
            "de_fuehrerschein",
        ] {
            assert!(Validator::from_name(name).is_some(), "{name} must resolve");
        }
        assert!(Validator::from_name("invented").is_none());
    }
}
