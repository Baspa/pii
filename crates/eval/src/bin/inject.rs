//! Places synthetic identifiers at known positions in real, licensed carrier
//! text, and writes the result out as benchmark cases.
//!
//! WHY CARRIERS EXIST AT ALL. The held-out split used while training the model
//! came from the same generators as the training data, so it shares their
//! blind spots: it scored 0.9940 on person names and gave no warning of a
//! regression the hand-written fixture caught immediately. A benchmark built
//! by the process that built the training data certifies its own siblings.
//! The carrier text under `benchmark/carriers/` is therefore real documents
//! from outside both training corpora (`ai4privacy`, `nym`, and
//! `training/generate.py`) -- EUR-Lex legislation, Wikipedia, Wikibooks,
//! Wikivoyage -- while every identifier inside it is synthetic. Real prose
//! gives real structure; synthetic identifiers give exactly known positions.
//!
//! WHY NO CHECKSUM IS IMPLEMENTED HERE. The task this file answers offered a
//! choice between porting `training/generate.py`'s checksum generators and
//! shelling out to them. It does neither, because there is a third option
//! that is stronger than both: every checksummed value is *drawn against the
//! service's own validator*. The generator fixes the free digits at random
//! and then asks [`pii_detection_rules::validator::Validator`] which check digit it
//! accepts, so the value is correct by the definition the service itself
//! uses. A port can drift from the Rust validator (that is exactly the risk
//! `generate.py` writes its own second, independent verifier to cover), and
//! shelling out would make `cargo test` depend on a Python interpreter. There
//! is no transcription of a checksum rule anywhere in this file.
//!
//! WHICH COUNTRY IDENTIFIERS ARE INJECTED, AND WHICH ARE NOT. An identifier
//! is injected only where a country pack can detect it. The rule has not
//! changed -- a case holding a value nothing in the service looks for fails
//! its per-type gate for a scoping reason and drowns out a real regression --
//! but the set of packs has: the Dutch BSN and the German Steuer-ID are
//! joined by the packs `country_pack` below names.
//!
//! THOSE NEW IDENTIFIERS GO IN THEIR OWN CASES, NOT INTO THE EXISTING ONES.
//! `build` produces the `d0`/`d1`/`dn` cases and is untouched by them, so
//! `benchmark/cases/carriers.json` is byte-for-byte what it was; `build_country`
//! produces one extra case per carrier, written to its own file per country.
//! Two reasons, and the second is the load-bearing one. A country pack is
//! reviewed and reverted as one commit, and a per-country case file reverts
//! with it. And the committed Azure baseline was collected over the 1,150
//! cases as they stand: re-drawing the values inside them would move
//! `PERSON` and `STREET_ADDRESS` under a baseline that could not be
//! re-collected, silently. Growing the fixture instead trips
//! `eval`'s own case-count guard, which is loud.
//!
//! POSITIONS. Values go in at sentence boundaries and mid-sentence in equal
//! proportion (alternating by injection index), because a detector that only
//! works at line starts is a detector that fails on prose. Insertion never
//! rewrites the carrier -- it only splices text in, so every sentence the
//! carrier already had survives verbatim.
//!
//! Offsets are never computed here. `must_mask` names a literal substring and
//! the scorer derives spans from it, which is what makes the answer key
//! incapable of disagreeing with its own text.

use std::collections::BTreeSet;
use std::fmt;
use std::path::{Path, PathBuf};

use pii_detection_engine::entity::EntityType;
use pii_detection_eval::{Case, MustMask};
use pii_detection_rules::validator::Validator;

// ---------------------------------------------------------------------------
// Deterministic randomness
// ---------------------------------------------------------------------------

/// SplitMix64. A seeded stream, not a checksum: it decides *which* value is
/// drawn, never whether the value is valid. Written out rather than pulled
/// from a crate so that a given seed produces the same fixture on any machine
/// and any toolchain, for as long as this file is unchanged.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A number in `0..bound`. `bound` of zero yields zero rather than
    /// dividing by it.
    fn below(&mut self, bound: u64) -> u64 {
        if bound == 0 {
            0
        } else {
            self.next_u64() % bound
        }
    }

    fn digit(&mut self) -> char {
        char::from(b'0' + (self.below(10) as u8))
    }

    fn digits(&mut self, count: usize) -> String {
        (0..count).map(|_| self.digit()).collect()
    }

    fn upper(&mut self) -> char {
        char::from(b'A' + (self.below(26) as u8))
    }

    fn pick<'a, T>(&mut self, from: &'a [T]) -> Option<&'a T> {
        from.get(self.below(from.len() as u64) as usize)
    }
}

// ---------------------------------------------------------------------------
// What can be injected
// ---------------------------------------------------------------------------

/// One identifier to place in a carrier: what kind it is, and the language
/// the sentence around it should be written in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Injection {
    kind: Kind,
    lang: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Kind {
    /// A national identity number, keyed by which one. The key is a country
    /// code where the country issues exactly one (`"NL"` for the BSN), and a
    /// country code plus the identifier's name where it issues more than one
    /// (`"ES-DNI"` and `"ES-NIE"` -- Spain issues both, to nationals and to
    /// foreign residents respectively, and they are not interchangeable).
    /// A key with no generator draws nine digits that pass no country's
    /// checksum, which is deliberate; see the module docs.
    NationalId(String),
    /// A tax identification number, keyed the same way. `"DE"` is the
    /// Steuer-ID.
    TaxId(String),
    Iban(String),
    CreditCard,
    Email,
    IpAddress,
    Phone(String),
    Vat(String),
    Passport,
    /// A driving licence, keyed by which country issues it: `"DE"` for the
    /// Führerschein and `"GB"` for the DVLA driver number. Those are the two
    /// The detector detects, and the module docs' rule applies here as everywhere
    /// -- an identifier is injected only where a pack can find it.
    DrivingLicence(String),
    Person,
    StreetAddress,
    /// A machine credential, keyed by which format: `"aws"`, `"github"`,
    /// `"jwt"`, `"pem-rsa"` and the rest of [`CREDENTIAL_FORMATS`]. Every
    /// value drawn is structurally correct -- the documented prefix, the
    /// documented length, the documented alphabet -- and cryptographically
    /// worthless; see [`credential`]'s own docs for why that is not a
    /// contradiction and how it is established rather than asserted.
    Credential(String),
    /// A literal value chosen by the caller. Nothing generates it, so nothing
    /// can re-draw it when it collides with the carrier -- which is what
    /// makes it the shape a collision test is written against.
    Exact(String),
    /// A literal value the caller supplies, introduced by prose that says it
    /// is an **order number** rather than by the neutral "the reference is".
    ///
    /// [`Kind::Exact`] would do everything else this needs, and does for the
    /// German and British licence negatives, where a check digit or a
    /// sixteen-character structure is what rejects the value and the words
    /// around it are irrelevant. The Dutch, French and Italian recognizers
    /// have no checksum: the only thing that can reject a licence-shaped run
    /// is a word saying it is something else, and a hard negative introduced
    /// as "the reference is" does not carry one. This kind makes the
    /// negative's own label part of the fixture rather than a hope about
    /// where the injector happened to put it.
    OrderReference(String),
}

impl Injection {
    /// A national identification number, by the key [`Kind::NationalId`]
    /// documents. A key with no generator yields a nine-digit value that
    /// passes no country's checksum, which is deliberate: see the module
    /// docs on why unsupported countries are not injected into the fixture.
    pub fn national_id(key: &str) -> Self {
        Self::of(Kind::NationalId(key.to_uppercase()), key)
    }

    pub fn tax_id(key: &str) -> Self {
        Self::of(Kind::TaxId(key.to_uppercase()), key)
    }

    pub fn iban(country: &str) -> Self {
        Self::of(Kind::Iban(country.to_uppercase()), country)
    }

    pub fn credit_card() -> Self {
        Self::of(Kind::CreditCard, "en")
    }

    pub fn email() -> Self {
        Self::of(Kind::Email, "en")
    }

    pub fn ip_address() -> Self {
        Self::of(Kind::IpAddress, "en")
    }

    pub fn phone(country: &str) -> Self {
        Self::of(Kind::Phone(country.to_uppercase()), country)
    }

    pub fn vat(country: &str) -> Self {
        Self::of(Kind::Vat(country.to_uppercase()), country)
    }

    pub fn passport() -> Self {
        Self::of(Kind::Passport, "en")
    }

    /// A driving licence for `country`. See [`Kind::DrivingLicence`] for the
    /// keys.
    pub fn driving_licence(country: &str) -> Self {
        Self::of(Kind::DrivingLicence(country.to_uppercase()), country)
    }

    pub fn person() -> Self {
        Self::of(Kind::Person, "en")
    }

    pub fn street_address() -> Self {
        Self::of(Kind::StreetAddress, "en")
    }

    /// A machine credential of the named format. See [`CREDENTIAL_FORMATS`]
    /// for the keys and [`credential`] for what each one draws.
    pub fn credential(format: &str) -> Self {
        Self::of(Kind::Credential(format.to_string()), "en")
    }

    /// A value the caller supplies verbatim.
    pub fn exact(value: &str) -> Self {
        Self::of(Kind::Exact(value.to_string()), "en")
    }

    /// A value the caller supplies verbatim, introduced as an order number.
    /// See [`Kind::OrderReference`] for why that is a different kind rather
    /// than a different sentence.
    pub fn order_reference(value: &str) -> Self {
        Self::of(Kind::OrderReference(value.to_string()), "en")
    }

    /// The language of the sentence built around the value. Defaults to the
    /// country code lowercased when that is one of the six supported
    /// languages, and to `"en"` otherwise, so `Injection::national_id("NL")`
    /// reads as Dutch without the caller saying so twice.
    pub fn in_lang(mut self, lang: &str) -> Self {
        self.lang = lang.to_lowercase();
        self
    }

    fn of(kind: Kind, hint: &str) -> Self {
        let hint = hint.to_lowercase();
        let lang = if LANGUAGES.contains(&hint.as_str()) {
            hint
        } else {
            "en".to_string()
        };
        Self { kind, lang }
    }

    /// The type this injection declares on the case's `must_mask` entry.
    /// `Exact` declares nothing: the caller chose the string, and this file
    /// has no basis to say what it is.
    fn entity_type(&self) -> Option<EntityType> {
        Some(match self.kind {
            Kind::NationalId(_) => EntityType::NationalId,
            Kind::TaxId(_) => EntityType::TaxId,
            Kind::Iban(_) => EntityType::IbanCode,
            Kind::CreditCard => EntityType::CreditCard,
            Kind::Email => EntityType::EmailAddress,
            Kind::IpAddress => EntityType::IpAddress,
            Kind::Phone(_) => EntityType::PhoneNumber,
            Kind::Vat(_) => EntityType::VatNumber,
            Kind::Passport => EntityType::PassportNumber,
            Kind::DrivingLicence(_) => EntityType::DrivingLicence,
            Kind::Person => EntityType::Person,
            Kind::StreetAddress => EntityType::StreetAddress,
            Kind::Credential(_) => EntityType::Credential,
            Kind::Exact(_) | Kind::OrderReference(_) => return None,
        })
    }

    /// Whether re-drawing can resolve a collision with the carrier. Only a
    /// caller-supplied literal cannot be re-drawn.
    fn is_generated(&self) -> bool {
        !matches!(self.kind, Kind::Exact(_) | Kind::OrderReference(_))
    }

    /// Whether the prose around this value should call it an order number.
    /// See [`Kind::OrderReference`].
    fn is_order_reference(&self) -> bool {
        matches!(self.kind, Kind::OrderReference(_))
    }

    /// Which country identifier this is, for the kinds that name one. Used to
    /// pick the sentence built around it: a French NIR and a Belgian national
    /// register number are both `NATIONAL_ID` in French prose and are
    /// introduced by different words, so the entity type alone cannot choose.
    fn identifier_key(&self) -> Option<&str> {
        match &self.kind {
            Kind::NationalId(key) | Kind::TaxId(key) => Some(key),
            _ => None,
        }
    }

    /// Which country's driving licence this is, for the same reason
    /// [`Injection::identifier_key`] exists: `EntityType::DrivingLicence`
    /// alone cannot choose the sentence. A German Führerschein and a British
    /// driver number are introduced by different words in every one of the
    /// six languages, and a fixture whose prose names the wrong country is a
    /// fixture that is ambiguous about what it is testing.
    fn licence_country(&self) -> Option<&str> {
        match &self.kind {
            Kind::DrivingLicence(country) => Some(country),
            _ => None,
        }
    }

    /// Which credential format this is, for the same reason
    /// [`Injection::identifier_key`] exists: `EntityType::Credential` alone
    /// cannot choose the sentence. "The API key in the configuration is"
    /// is the right introduction for an `AKIA…` and the wrong one for a PEM
    /// block, and a fixture whose prose does not match its value is a
    /// fixture that is ambiguous about what it is testing.
    fn credential_format(&self) -> Option<&str> {
        match &self.kind {
            Kind::Credential(format) => Some(format),
            _ => None,
        }
    }

    fn draw(&self, rng: &mut Rng) -> String {
        match &self.kind {
            Kind::NationalId(country) => national_id(rng, country),
            Kind::TaxId(country) => tax_id(rng, country),
            Kind::Iban(country) => iban(rng, country),
            Kind::CreditCard => credit_card(rng),
            Kind::Email => email(rng),
            Kind::IpAddress => ip_address(rng),
            Kind::Phone(country) => phone(rng, country),
            Kind::Vat(country) => vat(rng, country),
            Kind::Passport => passport(rng),
            Kind::DrivingLicence(country) => driving_licence(rng, country),
            Kind::Person => person(rng, &self.lang),
            Kind::StreetAddress => street_address(rng, &self.lang),
            Kind::Credential(format) => credential(rng, format),
            Kind::Exact(value) | Kind::OrderReference(value) => value.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Value generators
// ---------------------------------------------------------------------------

/// The check digit that makes `validator` accept `prefix` followed by it, or
/// `None` when no single digit does. Nothing here knows what the checksum is:
/// the free digits are fixed first and the service's own validator decides.
fn check_digit(prefix: &str, validator: Validator) -> Option<char> {
    check_character(prefix, "0123456789", validator)
}

/// The same, over an arbitrary alphabet: several country identifiers close
/// with a check *letter* rather than a check digit.
fn check_character(prefix: &str, alphabet: &str, validator: Validator) -> Option<char> {
    alphabet
        .chars()
        .find(|candidate| validator.accepts(&format!("{prefix}{candidate}")))
}

/// The 23 letters a Spanish check character is drawn from. Written out here
/// only as the *search space* -- which of them is correct is decided by
/// `Validator::EsDni`/`EsNie`, not by this file, so no ordering is
/// transcribed and there is nothing here to get wrong.
const ES_CHECK_ALPHABET: &str = "TRWAGMYFPDXBNJZSQVHLCKE";

/// A national identity number for `key`. See [`Kind::NationalId`] on what a
/// key is and what an unknown one produces.
fn national_id(rng: &mut Rng, key: &str) -> String {
    match key {
        "NL" => nl_bsn(rng),
        "ES-DNI" => es_dni(rng),
        "ES-NIE" => es_nie(rng),
        "FR" => fr_nir(rng),
        "BE" => be_national_register(rng),
        // No pack for this key; see the module docs.
        _ => rng.digits(9),
    }
}

/// Belgian national register number: a birth date, a three-digit sequence,
/// and the two-digit control `Validator::BeNationalRegister` accepts.
///
/// The year is drawn from both centuries -- 00 to 26 as well as 27 to 99 --
/// because the control differs between them and a fixture drawn only from
/// the 1900s would never exercise the year-2000 branch. One draw in five
/// carries the bis month (+20 or +40), so the bis form is exercised too.
///
/// A number that also clears the German Steuer-ID check is kept, not
/// redrawn: those exist (see `validator.rs`), both recognizers mask exactly
/// the same characters, and a fixture that quietly excluded them would be
/// measuring an easier problem than the real one.
fn be_national_register(rng: &mut Rng) -> String {
    for _ in 0..16 {
        let year = rng.digits(2);
        let month = 1 + rng.below(12);
        let month = match rng.below(5) {
            0 => month + 20,
            1 => month + 40,
            _ => month,
        };
        let day = 1 + rng.below(28);
        let sequence = 1 + rng.below(997);
        let prefix = format!("{year}{month:02}{day:02}{sequence:03}");
        if let Some(control) = (1..=97u32).find(|candidate| {
            Validator::BeNationalRegister.accepts(&format!("{prefix}{candidate:02}"))
        }) {
            return format!("{prefix}{control:02}");
        }
    }
    "93051822361".to_string()
}

/// French NIR: sex, year, month, department, commune and order, then the
/// two-digit key `Validator::FrNir` accepts.
///
/// The department is drawn from the two-digit range and, one draw in ten
/// for each of `2A` and `2B`, from Corsica -- deliberately far above
/// Corsica's share of the population, so the fixture exercises the
/// substitution rule rather than only the arithmetic. The key is searched
/// over `01..=97`
/// rather than computed here, so nothing in this file transcribes the
/// modulo.
fn fr_nir(rng: &mut Rng) -> String {
    for _ in 0..16 {
        let sex = if rng.below(2) == 0 { '1' } else { '2' };
        let year = rng.digits(2);
        let month = 1 + rng.below(12);
        let department = match rng.below(10) {
            0 => "2A".to_string(),
            1 => "2B".to_string(),
            _ => format!("{:02}", 1 + rng.below(95)),
        };
        let commune = format!("{:03}", 1 + rng.below(990));
        let order = format!("{:03}", 1 + rng.below(999));
        let prefix = format!("{sex}{year}{month:02}{department}{commune}{order}");
        if let Some(key) = (1..=97u32)
            .find(|candidate| Validator::FrNir.accepts(&format!("{prefix}{candidate:02}")))
        {
            let value = format!("{prefix}{key:02}");
            // A fifteen-digit NIR is inside the credit-card recognizer's
            // length range, and roughly one in ten is Luhn-valid. The same
            // guard `phone` carries, for the same reason: an answer key
            // demanding NATIONAL_ID over a string the engine reads as a card
            // number tests the overlap resolver, not the country pack.
            if !digits_pass_as_a_credit_card(&value) {
                return value;
            }
        }
    }
    "174033000705847".to_string()
}

/// Dutch BSN: eight free digits, the ninth whichever one the elfproef
/// accepts. About one draw in eleven has no accepting ninth digit, so it is
/// redrawn.
fn nl_bsn(rng: &mut Rng) -> String {
    loop {
        let prefix = format!("{}{}", 1 + rng.below(9), rng.digits(7));
        if let Some(check) = check_digit(&prefix, Validator::Elfproef) {
            return format!("{prefix}{check}");
        }
    }
}

/// Spanish DNI: eight free digits and the check letter `Validator::EsDni`
/// accepts. Every eight-digit body has exactly one, so the loop returns on
/// its first pass; it is a loop rather than an unwrap because this file must
/// not panic on a fixture build, and the fixed fallback is itself a published
/// valid value.
fn es_dni(rng: &mut Rng) -> String {
    for _ in 0..8 {
        let prefix = format!("{}{}", 1 + rng.below(9), rng.digits(7));
        if let Some(check) = check_character(&prefix, ES_CHECK_ALPHABET, Validator::EsDni) {
            return format!("{prefix}{check}");
        }
    }
    "00000000T".to_string()
}

/// Spanish NIE: one of the three issuing series letters, seven free digits,
/// and the check letter `Validator::EsNie` accepts.
fn es_nie(rng: &mut Rng) -> String {
    for _ in 0..8 {
        let series = ['X', 'Y', 'Z'][rng.below(3) as usize];
        let prefix = format!("{series}{}", rng.digits(7));
        if let Some(check) = check_character(&prefix, ES_CHECK_ALPHABET, Validator::EsNie) {
            return format!("{prefix}{check}");
        }
    }
    "X0000000T".to_string()
}

/// German Steuer-ID: ten digits obeying the *structure* rule, then the
/// eleventh whichever one `german_tax_id` accepts.
///
/// THE CHECK DIGIT IS NOT THE WHOLE RULE, AND THE DIFFERENCE BIASES THE
/// BENCHMARK. `german_tax_id` -- like `training/generate.py::_steuer_id`, and
/// like the earlier version of this function -- validates only the ISO 7064
/// MOD 11,10 check digit. The BZSt specification also constrains the first
/// ten digits: exactly one digit value occurs more than once (twice, or three
/// times non-consecutively), every other value occurs at most once, and the
/// first digit is not zero. Ten free digits plus a correct check digit
/// satisfies our validator and violates the specification about 96 times in a
/// hundred.
///
/// That is not a cosmetic defect in a benchmark. A detector applying the real
/// rule would reject those values, and the answer key -- which demands they be
/// masked -- would mark it *wrong* where the detector, applying the looser rule, is
/// marked right. The whole purpose of building the fixture on carriers outside
/// the training corpora is defeated if the identifiers inside it inherit the
/// training generator's blind spot. So the structure is generated here and
/// verified in the tests against a separate transcription of the rule.
fn tax_id(rng: &mut Rng, key: &str) -> String {
    match key {
        "DE" => de_steuer_id(rng),
        "IT" => it_codice_fiscale(rng),
        // No pack for this key; see the module docs.
        _ => rng.digits(11),
    }
}

fn de_steuer_id(rng: &mut Rng) -> String {
    loop {
        let prefix = steuer_id_body(rng);
        if let Some(check) = check_digit(&prefix, Validator::GermanTaxId) {
            return format!("{prefix}{check}");
        }
    }
}

/// The letter triples an Italian codice fiscale opens with, and the
/// cadastral codes it carries.
///
/// NOTHING VALIDATES EITHER, WHICH IS WHY NEITHER IS COMPUTED. The name
/// triples are the surname and given-name codes of the Italian names this
/// file already draws people from, written out rather than derived: the
/// derivation rule (consonants first, then vowels; the first, third and
/// fourth consonant when there are four or more) is a rule
/// `Validator::ItCodiceFiscale` does not check, and this file's own history
/// says that anything the service does not check needs a second, independent
/// transcription before it can be trusted. A table costs nothing and has
/// nothing to get wrong. The cadastral codes are five real ones, published
/// by the Agenzia delle Entrate: Roma, Milano, Torino, Firenze, Napoli.
const IT_SURNAME_CODES: [&str; 8] = ["ZNT", "BTT", "MCC", "RVR", "SLV", "CTT", "FRT", "LMB"];
const IT_GIVEN_CODES: [&str; 8] = ["CHR", "LNZ", "BRC", "MLN", "SRN", "TMS", "LRI", "RCR"];
const IT_PLACE_CODES: [&str; 5] = ["H501", "F205", "L219", "D612", "F839"];
/// The twelve month letters, as a search space only -- which of them means
/// which month is `Validator::ItCodiceFiscale`'s business, not this file's.
const IT_MONTH_LETTERS: [char; 12] = ['A', 'B', 'C', 'D', 'E', 'H', 'L', 'M', 'P', 'R', 'S', 'T'];

/// Italian codice fiscale: fifteen characters of name, birth date and place,
/// then the check letter `Validator::ItCodiceFiscale` accepts.
fn it_codice_fiscale(rng: &mut Rng) -> String {
    for _ in 0..8 {
        let surname = rng.pick(&IT_SURNAME_CODES).copied().unwrap_or("RSS");
        let given = rng.pick(&IT_GIVEN_CODES).copied().unwrap_or("MRA");
        let year = rng.digits(2);
        let month = rng.pick(&IT_MONTH_LETTERS).copied().unwrap_or('A');
        // 1 to 28, plus 40 for a woman. Both halves of the range the
        // validator accepts are reachable.
        let day = 1 + rng.below(28) + if rng.below(2) == 0 { 0 } else { 40 };
        let place = rng.pick(&IT_PLACE_CODES).copied().unwrap_or("H501");
        let prefix = format!("{surname}{given}{year}{month}{day:02}{place}");
        if let Some(check) = check_character(
            &prefix,
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            Validator::ItCodiceFiscale,
        ) {
            return format!("{prefix}{check}");
        }
    }
    "MRTMTT91D08F205J".to_string()
}

/// The first ten digits of a Steuer-ID: nine distinct digit values, one of
/// them used twice, shuffled, first digit non-zero. That is exactly the
/// BZSt structure rule's "twice" case, and it implies the rule's other
/// clause for free -- with nine values across ten places, exactly one of the
/// ten digit values is absent.
fn steuer_id_body(rng: &mut Rng) -> String {
    let mut values: Vec<u32> = (0..10).collect();
    let absent = rng.below(10) as usize;
    values.remove(absent);
    let repeated = values[rng.below(9) as usize];
    values.push(repeated);

    // Fisher-Yates, so every arrangement is reachable.
    for index in (1..values.len()).rev() {
        let swap = rng.below(index as u64 + 1) as usize;
        values.swap(index, swap);
    }
    // A leading zero is not issued. Swapping rather than redrawing keeps the
    // multiset -- and so the structure rule -- intact.
    if values[0] == 0
        && let Some(position) = values.iter().position(|digit| *digit != 0)
    {
        values.swap(0, position);
    }
    values.iter().map(|digit| digit.to_string()).collect()
}

// ---------------------------------------------------------------------------
// National VAT rules, transcribed from their public specifications
//
// WHY THESE ARE NOT DRAWN AGAINST THE SERVICE'S OWN VALIDATOR, THE WAY
// EVERYTHING ABOVE IS. `universal.toml`'s VAT recogniser carries
// `validator = "none"`: it decides on shape and a nearby word, and no
// checksum. Drawing a VAT number against it would accept any string of the
// right shape, which is how the first version of this fixture came to hold
// twenty-two VAT numbers of which *none* satisfied its country's rule, and
// one -- an `ESH…F` -- that was malformed by construction, since Spanish
// entity type H takes a digit check character and never a letter.
//
// A fixture whose answer key demands the masking of values no authority would
// issue marks a stricter detector wrong and a looser one right. It is a
// measurement that flatters us, built into the instrument we intend to gate
// releases with. So each of these follows the real national rule, and each is
// checked in the tests by a verifier transcribed separately from the
// generator -- the same two-transcription discipline `training/generate.py`
// uses, and for the same reason.
// ---------------------------------------------------------------------------

/// The Luhn check digit for `prefix`.
///
/// `Validator::Luhn` cannot serve here: it refuses anything under twelve
/// digits, correctly, because a payment card is never shorter. A French SIREN
/// is nine digits and an Italian partita IVA eleven, and both use this same
/// algorithm.
fn luhn_check_digit(prefix: &[u32]) -> u32 {
    let mut total = 0u32;
    // With the check digit appended, `prefix`'s last digit sits one place
    // from the right, which is a doubled position.
    for (index, digit) in prefix.iter().rev().enumerate() {
        let mut value = *digit;
        if index.is_multiple_of(2) {
            value *= 2;
            if value > 9 {
                value -= 9;
            }
        }
        total += value;
    }
    (10 - total % 10) % 10
}

fn digit_values(rng: &mut Rng, count: usize) -> Vec<u32> {
    (0..count).map(|_| rng.below(10) as u32).collect()
}

fn joined(digits: &[u32]) -> String {
    digits.iter().map(|digit| digit.to_string()).collect()
}

/// Dutch BTW-identificatienummer, legal-entity form: `NL`, the nine-digit
/// RSIN, `B`, and a two-digit sub-number. The RSIN carries the same 11-proef
/// the BSN does -- weights 9..2 on the first eight digits, minus one on the
/// ninth, the sum divisible by eleven -- written out here rather than drawn
/// against `Validator::Elfproef`, so that the tests can use that validator as
/// an independent second opinion instead of as the definition.
fn nl_vat(rng: &mut Rng) -> String {
    loop {
        let head = digit_values(rng, 8);
        let weighted: u32 = head
            .iter()
            .enumerate()
            .map(|(index, digit)| (9 - index as u32) * digit)
            .sum();
        let check = weighted % 11;
        if check > 9 {
            continue;
        }
        let sub = 1 + rng.below(20);
        return format!("NL{}{}B{sub:02}", joined(&head), check);
    }
}

/// German USt-IdNr: `DE` and nine digits, the ninth an ISO 7064 MOD 11,10
/// check over the first eight. Same scheme as the Steuer-ID, one digit
/// shorter, and with no structure rule attached.
fn de_vat(rng: &mut Rng) -> String {
    let head = digit_values(rng, 8);
    let mut product = 10u32;
    for digit in &head {
        let mut sum = (digit + product) % 10;
        if sum == 0 {
            sum = 10;
        }
        product = (sum * 2) % 11;
    }
    let mut check = 11 - product;
    if check == 10 {
        check = 0;
    }
    format!("DE{}{check}", joined(&head))
}

/// French numéro de TVA intracommunautaire: `FR`, a two-digit key, and the
/// nine-digit SIREN. The SIREN is Luhn-valid; the key is
/// `(12 + 3 × (SIREN mod 97)) mod 97`.
fn fr_vat(rng: &mut Rng) -> String {
    let mut siren = digit_values(rng, 8);
    if siren[0] == 0 {
        siren[0] = 1 + rng.below(9) as u32;
    }
    siren.push(luhn_check_digit(&siren));
    let numeric: u64 = joined(&siren).parse().unwrap_or(0);
    let key = (12 + 3 * (numeric % 97)) % 97;
    format!("FR{key:02}{}", joined(&siren))
}

/// Spanish NIF for a legal entity (CIF): `ES`, an entity-type letter, seven
/// digits and a check character.
///
/// Only `A` (sociedad anónima) and `B` (sociedad de responsabilidad limitada)
/// are used, and both take a **digit** check character. The letter-check types
/// (`P`, `Q`, `S`, `N`, `W`, `K`, `R`) exist and are deliberately avoided: the
/// two families take different check characters, and the first version of this
/// generator emitted a letter check on a digit-check type, producing values
/// no register would hold.
fn es_vat(rng: &mut Rng) -> String {
    let kind = if rng.below(2) == 0 { 'A' } else { 'B' };
    let body = digit_values(rng, 7);
    // Odd positions (1st, 3rd, 5th, 7th -- index 0, 2, 4, 6) are doubled and
    // their digits summed; even positions are added as they are.
    let mut total = 0u32;
    for (index, digit) in body.iter().enumerate() {
        if index.is_multiple_of(2) {
            let doubled = digit * 2;
            total += doubled / 10 + doubled % 10;
        } else {
            total += digit;
        }
    }
    let check = (10 - total % 10) % 10;
    format!("ES{kind}{}{check}", joined(&body))
}

/// Italian partita IVA: `IT` and eleven digits -- a seven-digit serial, a
/// three-digit office code, and a Luhn check digit over the first ten.
fn it_vat(rng: &mut Rng) -> String {
    let mut body = digit_values(rng, 7);
    // The office code: 001 to 100 are the provincial offices.
    let office = 1 + rng.below(100);
    for digit in format!("{office:03}").chars() {
        body.push(digit.to_digit(10).unwrap_or(0));
    }
    let check = luhn_check_digit(&body);
    format!("IT{}{check}", joined(&body))
}

/// Bank identifiers that resemble the four-letter codes real IBANs carry
/// without being any bank's assigned code.
const BANK_CODES: [&str; 6] = ["BNKA", "BNKB", "BNKC", "BNKD", "BNKE", "BNKF"];

/// An IBAN whose two check digits are whichever pair `iban_mod97` accepts,
/// searched over `02..=98`.
///
/// THE RANGE IS PART OF THE RULE, NOT AN OPTIMISATION. ISO 13616 check digits
/// run 02 to 98; 00, 01 and 99 are not issuable. Two pairs in `00..=99` always
/// satisfy mod-97 -- if `k` works then so does `k + 97` whenever that is still
/// two digits, so `00`/`97`, `01`/`98` and `02`/`99` are the three colliding
/// pairs. An earlier version searched from `00` and took the first hit, which
/// produced `DE00...` for roughly one draw in fifty: a value `iban_mod97`
/// accepts and no bank could ever issue, sitting in the answer key as a value
/// a stricter detector is marked *wrong* for rejecting. Searching `02..=98`
/// leaves exactly one solution in range, which is the issuable one.
fn iban(rng: &mut Rng, country: &str) -> String {
    let bban = match country {
        "NL" => format!(
            "{}{}",
            rng.pick(&BANK_CODES).copied().unwrap_or("BNKA"),
            rng.digits(10)
        ),
        "DE" => rng.digits(18),
        "FR" => rng.digits(23),
        "ES" => rng.digits(20),
        "IT" => format!("{}{}", rng.upper(), rng.digits(22)),
        _ => format!(
            "{}{}",
            rng.pick(&BANK_CODES).copied().unwrap_or("BNKA"),
            rng.digits(14)
        ),
    };
    let country = if country.len() == 2 { country } else { "GB" };
    for check in 2..=98u32 {
        let candidate = format!("{country}{check:02}{bban}");
        if Validator::IbanMod97.accepts(&candidate) {
            return candidate;
        }
    }
    // Unreachable for a well-formed BBAN; returning the unchecked shape is
    // better than panicking in a generator, and the caller's own coherence
    // check still holds because the value is a literal substring either way.
    format!("{country}02{bban}")
}

/// Issuer identification numbers the card schemes publish for testing, so a
/// checksum-valid sixteen-digit number in this fixture cannot collide with a
/// live account.
const TEST_CARD_PREFIXES: [&str; 4] = ["411111", "555555", "400000", "550000"];

fn credit_card(rng: &mut Rng) -> String {
    let prefix = rng.pick(&TEST_CARD_PREFIXES).copied().unwrap_or("411111");
    loop {
        let body = format!("{prefix}{}", rng.digits(9));
        if let Some(check) = check_digit(&body, Validator::Luhn) {
            let solid = format!("{body}{check}");
            // Half solid, half in the four-by-four grouping people write on
            // forms: a detector that only reads one of the two shapes is a
            // detector with a hole in it.
            if rng.below(2) == 0 {
                return solid;
            }
            let grouped: Vec<String> = solid
                .as_bytes()
                .chunks(4)
                .map(|chunk| String::from_utf8_lossy(chunk).to_string())
                .collect();
            return grouped.join(" ");
        }
    }
}

/// `example.com`, `example.net` and `example.org` are reserved by RFC 2606
/// and can never be registered, so no address generated here can reach a real
/// mailbox. It also makes the email cases easier than production traffic;
/// `benchmark/carriers/README.md` records that as a known limitation.
const EMAIL_DOMAINS: [&str; 3] = ["example.com", "example.net", "example.org"];

fn email(rng: &mut Rng) -> String {
    let given = rng.pick(&GIVEN_NAMES_EN).copied().unwrap_or("alex");
    let family = rng.pick(&FAMILY_NAMES_EN).copied().unwrap_or("morgan");
    let domain = rng.pick(&EMAIL_DOMAINS).copied().unwrap_or("example.com");
    format!("{}.{}@{domain}", ascii_lower(given), ascii_lower(family))
}

/// The documentation and private ranges (RFC 5737, RFC 1918). No address
/// generated here routes to a real host.
fn ip_address(rng: &mut Rng) -> String {
    match rng.below(5) {
        0 => format!("192.0.2.{}", 1 + rng.below(253)),
        1 => format!("198.51.100.{}", 1 + rng.below(253)),
        2 => format!("203.0.113.{}", 1 + rng.below(253)),
        3 => format!(
            "10.{}.{}.{}",
            rng.below(256),
            rng.below(256),
            1 + rng.below(253)
        ),
        _ => format!("192.168.{}.{}", rng.below(256), 1 + rng.below(253)),
    }
}

/// True if every digit in `candidate` -- read left to right, everything
/// that is not a digit (the leading `+`, spaces, dashes) simply dropped --
/// forms a Luhn-valid sequence of card length. This is deliberately more
/// permissive about punctuation than `Validator::Luhn` (whose own `digits`
/// helper trips over a leading `+` and reports "not Luhn" for exactly the
/// candidates this function exists to catch): a card-scanning regex reads
/// digits wherever they fall and does not care what character precedes
/// them, and neither should this guard.
///
/// This is the fixture-side half of the `de-wv-camping-dn` incident: the detector
/// once emitted `CREDIT_CARD` over `49 151 71053098`, part of the phone
/// number `+49 151 71053098`, because that digit string is Luhn-valid at 13
/// digits. Whether a phone-shaped Luhn collision like that *should* be
/// misread as a card is a real defect in `pii-detection-rules`, out of scope here
/// (see the carrier-expansion report) -- but this benchmark's own generator
/// should not hand that detector a coin flip to lose on a case it can never
/// pass. A phone number whose digits double as a valid card number is
/// simply not a number this fixture should ever inject; draw another one.
fn digits_pass_as_a_credit_card(candidate: &str) -> bool {
    let digits_only: String = candidate.chars().filter(char::is_ascii_digit).collect();
    Validator::Luhn.accepts(&digits_only)
}

/// A telephone number in `country`'s numbering plan, checked against
/// `phonenumber` -- the same library the service's own recogniser uses, but
/// used here as ground truth about the plan rather than as the detector. A
/// number the plan does not contain is not a number the detector should be
/// asked to find, and would be a permanent, undeserved miss in the fixture.
/// A number whose digits are incidentally Luhn-valid at card length is
/// rejected for the same reason -- see `digits_pass_as_a_credit_card`.
fn phone(rng: &mut Rng, country: &str) -> String {
    let (region, international, national) = match country {
        "NL" => ("NL", "+31 6 1{}", "06-1{}"),
        "DE" => ("DE", "+49 151 {}", "0151 {}"),
        "FR" => ("FR", "+33 6 {}", "06 {}"),
        "ES" => ("ES", "+34 6{}", "6{}"),
        "IT" => ("IT", "+39 3{}", "3{}"),
        // Not the Ofcom drama range (07700 900xxx): libphonenumber marks it
        // invalid, so a detector that consults the numbering plan is right to
        // ignore it, and every one in the fixture would be a permanent
        // undeserved miss. See the README on why every injected number is
        // drawn from a live allocated range instead.
        _ => ("GB", "+44 79{}", "079{}"),
    };
    for _ in 0..64 {
        let body = match country {
            "NL" => rng.digits(7),
            "DE" => rng.digits(8),
            "FR" => format!(
                "{} {} {} {}",
                rng.digits(2),
                rng.digits(2),
                rng.digits(2),
                rng.digits(2)
            ),
            "ES" => format!(
                "{} {} {} {}",
                rng.digits(2),
                rng.digits(2),
                rng.digits(2),
                rng.digits(2)
            ),
            "IT" => format!("{} {} {}", rng.digits(2), rng.digits(3), rng.digits(4)),
            _ => format!("{} {}", rng.digits(2), rng.digits(6)),
        };
        let template = if rng.below(2) == 0 {
            international
        } else {
            national
        };
        let candidate = template.replace("{}", &body);
        if is_a_real_number(&candidate, region) && !digits_pass_as_a_credit_card(&candidate) {
            return candidate;
        }
    }
    // Fall back to a shape known to be in the plan rather than emitting one
    // that is not -- and, same as the loop above, not one whose digits are
    // incidentally Luhn-valid at card length. The DE and GB shapes below
    // were changed from an earlier revision (trailing digit 8 and 6
    // respectively) after discovering both original constants were
    // themselves Luhn-valid at 13 and 12 digits; both replacements are
    // re-verified real numbers (see the `phone_fallbacks_are_never_luhn_
    // valid_credit_card_shapes` test).
    match country {
        "NL" => "+31 6 12345678".to_string(),
        "DE" => "+49 151 12345679".to_string(),
        "FR" => "+33 6 12 34 56 78".to_string(),
        "ES" => "+34 612 34 56 78".to_string(),
        "IT" => "+39 320 123 4567".to_string(),
        _ => "+44 7911 123457".to_string(),
    }
}

fn is_a_real_number(candidate: &str, region: &str) -> bool {
    let Ok(region) = region.parse::<phonenumber::country::Id>() else {
        return false;
    };
    phonenumber::parse(Some(region), candidate)
        .map(|number| phonenumber::is_valid(&number))
        .unwrap_or(false)
}

fn vat(rng: &mut Rng, country: &str) -> String {
    match country {
        "NL" => nl_vat(rng),
        "DE" => de_vat(rng),
        "FR" => fr_vat(rng),
        "ES" => es_vat(rng),
        "IT" => it_vat(rng),
        // No United Kingdom prefix exists in the recogniser's country list.
        // An English-language carrier gets a Dutch number, for the same
        // reason it gets a Dutch BSN: an English ticket from a Dutch company
        // is the use case. The Irish rule was the other candidate and
        // was rejected -- its check character is a 23-letter alphabet over a
        // weighted sum, and getting it subtly wrong would reintroduce exactly
        // the defect this function was rewritten to remove.
        _ => nl_vat(rng),
    }
}

// ---------------------------------------------------------------------------
// Driving licences
// ---------------------------------------------------------------------------

/// The sixteen Bundesland keys a Behördenschlüssel can begin with, `A` for
/// Baden-Württemberg through `P` for Thüringen in alphabetical order.
///
/// Written out as the *draw space*, never as an answer: which of them a value
/// carries is a free choice, and the check digit over it is decided by
/// `Validator::DeFuehrerschein`, not here.
const DE_LAENDER_KEYS: &str = "ABCDEFGHIJKLMNOP";

/// The alphabet a German licence's running serial is drawn from: base 36,
/// which is exactly what the check digit reads it as.
const DE_SERIAL_ALPHABET: &str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";

/// German Führerscheinnummer: a Bundesland letter, three authority digits,
/// five serial characters, the check digit the service's own validator
/// accepts, and a reissue counter.
///
/// THE CHECK DIGIT IS NOT COMPUTED HERE. It sits at position ten of eleven,
/// so [`check_character`] -- which appends -- cannot place it. The candidate
/// is spliced into position instead and the service's own validator decides,
/// which keeps this file's promise that no checksum rule is transcribed in
/// it. `X` is in the search space because a remainder of ten is written that
/// way; whether any given body reaches it is not this function's business.
fn de_fuehrerschein(rng: &mut Rng) -> String {
    let laender: Vec<char> = DE_LAENDER_KEYS.chars().collect();
    let serial: Vec<char> = DE_SERIAL_ALPHABET.chars().collect();
    for _ in 0..8 {
        let mut body = String::new();
        body.push(rng.pick(&laender).copied().unwrap_or('A'));
        body.push_str(&rng.digits(3));
        for _ in 0..5 {
            body.push(rng.pick(&serial).copied().unwrap_or('0'));
        }
        // 1 to 9: the counter starts at one and runs into letters only for a
        // licence reissued ten times, which is not a shape worth drawing.
        let reissue = char::from(b'1' + (rng.below(9) as u8));
        let check = "0123456789X".chars().find(|candidate| {
            Validator::DeFuehrerschein.accepts(&format!("{body}{candidate}{reissue}"))
        });
        if let Some(check) = check {
            return format!("{body}{check}{reissue}");
        }
    }
    // The published specimen, which is a valid licence number by the rule
    // itself. A fallback rather than an unwrap, for the reason `es_dni`'s is.
    "B072RRE2I55".to_string()
}

/// Surnames a DVLA driver number's first five characters are built from.
/// Deliberately of several lengths, so the `9` padding rule is exercised
/// rather than assumed: `LEE` becomes `LEE99` and `MACDONALD` is cut to
/// `MACDO`.
const GB_SURNAMES: [&str; 10] = [
    "MORGAN",
    "LEE",
    "OKAFOR",
    "SHAW",
    "MACDONALD",
    "PATEL",
    "NG",
    "WHITEHOUSE",
    "AHMED",
    "BRENNAN",
];

/// The days each month has, January first. February is the short one on
/// purpose: the generator never draws past the 28th, so no drawn value can
/// depend on whether a year is a leap year and the independent verifier can
/// range-check the day against this table without a calendar.
const GB_MONTH_LENGTHS: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

/// A DVLA driver number: five characters of surname padded with `9`, the
/// decade digit of the year of birth, the month with fifty added for a woman,
/// the day, the year's unit digit, two forename initials padded with `9`, an
/// arbitrary digit, and two check characters.
///
/// **THE CHECK CHARACTERS ARE DRAWN, NOT COMPUTED, AND THAT IS NOT A
/// SHORTCUT.** The DVLA has refused a Freedom of Information request for the
/// algorithm behind characters 15 and 16 on the grounds that publishing it
/// would allow valid numbers to be generated. There is therefore no published
/// rule for this generator to satisfy and none for the verifier in
/// `national_rules` to check; both are limited to the structure, and the
/// structure is what `gb_driving_licence` detects on. A fixture value whose
/// check characters were computed by a reverse-engineered rule would be
/// asserting something nobody can verify.
fn gb_driving_licence(rng: &mut Rng) -> String {
    let surname = rng.pick(&GB_SURNAMES).copied().unwrap_or("MORGAN");
    let mut field: String = surname.chars().take(5).collect();
    while field.chars().count() < 5 {
        field.push('9');
    }

    let year = 1940 + rng.below(66) as u32;
    let month = 1 + rng.below(12) as u32;
    let days = GB_MONTH_LENGTHS
        .get((month - 1) as usize)
        .copied()
        .unwrap_or(28)
        .min(28);
    let day = 1 + rng.below(u64::from(days)) as u32;
    let female = rng.below(2) == 0;
    let encoded_month = if female { month + 50 } else { month };

    let initials: String = (0..2)
        .map(|index| {
            // One driver in eight has a single forename, which is the `9`
            // the padding rule exists for.
            if index == 1 && rng.below(8) == 0 {
                '9'
            } else {
                rng.upper()
            }
        })
        .collect();

    format!(
        "{field}{decade}{encoded_month:02}{day:02}{unit}{initials}{arbitrary}{check}",
        decade = (year / 10) % 10,
        unit = year % 10,
        // "usually a 9 but can vary", so it varies.
        arbitrary = if rng.below(4) == 0 { rng.digit() } else { '9' },
        check = (0..2).map(|_| rng.upper()).collect::<String>(),
    )
}

/// Dutch rijbewijsnummer: ten digits, and there is nothing else to say
/// about it. No check character, no fields, no published structure -- the
/// RDW defines none and Purview's own definition records `Checksum: No`.
///
/// THE FIRST DIGIT IS NEVER ZERO, AND THAT IS A PROPERTY OF THIS FIXTURE
/// RATHER THAN OF THE IDENTIFIER. A ten-digit run beginning with a trunk
/// zero is also how a Dutch telephone number is written solid
/// (`0612345678`), and the phone recognizer scores that 0.95 against this
/// one's 0.75, so such a value is masked -- the same characters, over the
/// same span -- but labelled `PHONE_NUMBER`. Drawing one would measure the
/// phone recognizer rather than this one, and the fixture would say nothing
/// about either. The behaviour itself is not hidden: it is pinned directly
/// by `pii-detection-rules/tests/driving_licence.rs::a_dutch_licence_written_with_a
/// _trunk_zero_is_masked_as_a_telephone_number` and named in
/// `training/reports/driving-licence.md` §13.4.
fn nl_rijbewijs(rng: &mut Rng) -> String {
    format!("{}{}", 1 + rng.below(9), rng.digits(9))
}

/// French NEPH -- numéro d'enregistrement préfectoral harmonisé: two digits
/// of year, two of month, two of department, two of prefecture and four of
/// registration order.
///
/// THE YEAR IS NEVER 00 TO 09, FOR THE REASON [`nl_rijbewijs`] GIVES. A NEPH
/// registered between 2000 and 2009 begins with a zero, and a twelve-digit
/// run with a leading zero is accepted by at least one of the phone
/// recognizer's eight numbering plans -- which scores a trunk prefix 0.95
/// against this recognizer's 0.75, so the value is masked as a telephone
/// number. That is a real licence this service labels wrongly and it is
/// pinned as a test rather than hidden here; drawing one into the fixture
/// would only measure the wrong recognizer. See
/// `training/reports/driving-licence.md` §14.3.
///
/// The department is drawn from 01 to 95, which is metropolitan France. The
/// overseas departments are 971 to 976 and no source specifies how three
/// digits are written into the two positions the NEPH has for them, so the
/// generator does not guess and the pattern does not range-check the field
/// at all.
fn fr_permis_conduire(rng: &mut Rng) -> String {
    format!(
        "{year:02}{month:02}{department:02}{prefecture:02}{order:04}",
        year = 10 + rng.below(90),
        month = 1 + rng.below(12),
        department = 1 + rng.below(95),
        prefecture = 1 + rng.below(99),
        order = rng.below(10_000),
    )
}

/// The two-letter codes of the Motorizzazione offices an Italian licence's
/// province field can carry.
///
/// Real province abbreviations rather than two drawn letters, so a committed
/// value is one the issuing authority could have produced. Which of them a
/// value carries is a free choice; nothing here encodes anything about the
/// holder.
const IT_PROVINCE_CODES: [&str; 12] = [
    "MI", "RM", "TO", "NA", "BO", "FI", "VE", "GE", "BA", "PA", "PD", "VR",
];

/// The alphabet an Italian licence's `U1` body is drawn from. Excludes
/// nothing: the Ministry's circular says the letters A, O, Q and I are
/// excluded from the *serial letter* it describes, and the sources
/// contradict each other about where that letter sits, so the generator does
/// not act on a rule it cannot place.
const IT_SERIAL_ALPHABET: &str = "0123456789BCDEFGHJKLMNPRSTUVWXYZ";

/// An Italian patente di guida, in one of the two published forms.
///
/// **THE CONTROL CHARACTER IS DRAWN, NOT COMPUTED, AND THAT IS NOT A
/// SHORTCUT.** No source publishes the algorithm behind it -- the Ministry's
/// own circular of 18 October 2011 introduces the character and says nothing
/// about how it is derived, and Microsoft Purview records `Checksum: No`
/// from the other side. There is therefore no published rule for this
/// generator to satisfy and none for the verifier in `national_rules` to
/// check; both are limited to the structure, which is exactly what
/// `it_patente` detects on. This is the same position
/// [`gb_driving_licence`] is in and it is stated the same way.
///
/// One value in four is the `U1` form, which is roughly how often a licence
/// is a duplicate -- a guess, and labelled as one; what matters is that both
/// published forms are exercised rather than the proportion between them.
fn it_patente(rng: &mut Rng) -> String {
    let control = rng.upper();
    if rng.below(4) == 0 {
        let serial: Vec<char> = IT_SERIAL_ALPHABET.chars().collect();
        let body: String = (0..7)
            .map(|_| rng.pick(&serial).copied().unwrap_or('0'))
            .collect();
        format!("U1{body}{control}")
    } else {
        let province = rng.pick(&IT_PROVINCE_CODES).copied().unwrap_or("MI");
        format!("{province}{}{control}", rng.digits(7))
    }
}

/// A driving licence for `country`, or -- for a country with no pack -- a
/// value that no licence recognizer accepts, which is the same deliberate
/// behaviour [`national_id`] has for an uncovered country.
fn driving_licence(rng: &mut Rng, country: &str) -> String {
    match country {
        "DE" => de_fuehrerschein(rng),
        "GB" => gb_driving_licence(rng),
        "NL" => nl_rijbewijs(rng),
        "FR" => fr_permis_conduire(rng),
        "IT" => it_patente(rng),
        _ => format!("{}{}", rng.upper(), rng.digits(9)),
    }
}

fn passport(rng: &mut Rng) -> String {
    format!(
        "{}{}{}",
        rng.upper(),
        rng.upper(),
        (0..7)
            .map(|_| if rng.below(4) == 0 {
                rng.upper()
            } else {
                rng.digit()
            })
            .collect::<String>()
    )
}

// ---------------------------------------------------------------------------
// Credentials
// ---------------------------------------------------------------------------

/// Every credential format this binary can draw, in the order the families
/// rotate through them. Grouped by the family that owns them, because a
/// family is a commit and a case file.
///
/// WHY EVERY VALUE HERE IS SAFE TO COMMIT, ESTABLISHED RATHER THAN ASSERTED.
/// A real key in a committed fixture is an incident, so the argument has to
/// be structural rather than a promise that nobody pasted one in:
///
/// 1. **Nothing is transcribed.** Every character below the prefix comes out
///    of [`Rng`], the seeded SplitMix64 stream this file already draws
///    national IDs and IBANs from. No value in this module was ever issued,
///    typed in, or copied from a vendor's documentation -- there is no path
///    by which a live key could enter, because there is no literal to enter
///    it as.
/// 2. **The draw space makes collision unreachable.** The smallest space
///    here is the AWS access key ID: sixteen base32 characters, 2^80
///    possibilities. The others are larger by tens of orders of magnitude.
/// 3. **No value was ever presented to the issuer.** These strings are
///    generated locally and written to a JSON file. Nothing in this
///    workspace calls AWS, GitHub, Stripe, OpenAI, Anthropic, Slack or
///    Google, so no draw has ever been -- or could be -- registered as a
///    credential anywhere.
/// 4. **The PEM bodies cannot function as keys even by accident.** They are
///    random base64, and random base64 does not decode to a valid ASN.1
///    SEQUENCE or an OpenSSH key blob. `openssl pkey`, `ssh-keygen` and
///    every TLS stack reject them at the parse, before any cryptography
///    happens. This is deliberately *not* a throwaway real keypair: a real
///    private key committed to a repository is a private key committed to a
///    repository, whatever its lifetime, and generating one would also make
///    `cargo run --bin inject` depend on an OpenSSL binary the rest of this
///    file is careful never to need.
const CREDENTIAL_FORMATS: [&str; 8] = [
    "aws",
    "github",
    "github-pat",
    "stripe",
    "openai",
    "anthropic",
    "slack",
    "google",
];

/// The JWT and bearer formats -- family 2.
const CREDENTIAL_TOKEN_FORMATS: [&str; 2] = ["jwt", "bearer"];

/// The PEM formats -- family 3. `certificate` is the one public artefact in
/// the set and is included because a certificate block travels with the key
/// it belongs to; see `packs/universal.toml`'s own note on the trade.
const CREDENTIAL_PEM_FORMATS: [&str; 4] = ["pem-rsa", "pem-openssh", "pem-ec", "pem-certificate"];

const BASE62: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
/// RFC 4648 section 5, the URL-safe alphabet JWTs and several vendors use.
const BASE64URL: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
/// RFC 4648 section 4, the alphabet a PEM body is wrapped in.
const BASE64: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
/// The 32 characters an AWS access key ID is built from -- `0`, `1`, `8` and
/// `9` are absent from every key anyone has published a look at, which is
/// what makes it five bits per character. The recognizer's pattern accepts
/// the wider `[A-Z0-9]` on purpose (see its comment); drawing from the
/// narrower real alphabet keeps the fixture honest about what AWS issues
/// without narrowing what the service will find.
const AWS_BASE32: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";

fn from_alphabet(rng: &mut Rng, alphabet: &str, count: usize) -> String {
    let characters: Vec<char> = alphabet.chars().collect();
    (0..count)
        .map(|_| characters[rng.below(characters.len() as u64) as usize])
        .collect()
}

/// Base64url without padding, which is what a JOSE segment is.
fn base64url_encode(bytes: &[u8]) -> String {
    let alphabet: Vec<char> = BASE64URL.chars().collect();
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let triple = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        let indices = [
            (triple >> 18) & 0x3F,
            (triple >> 12) & 0x3F,
            (triple >> 6) & 0x3F,
            triple & 0x3F,
        ];
        for (position, index) in indices.iter().enumerate() {
            if position <= chunk.len() {
                out.push(alphabet[*index as usize]);
            }
        }
    }
    out
}

/// One credential of the named format. Panics on an unknown key rather than
/// silently drawing something else: a fixture built around a format nobody
/// implemented would be a whole case file of values the service is not
/// looking for, scored as misses.
fn credential(rng: &mut Rng, format: &str) -> String {
    match format {
        // AKIA is the long-term access key ID; ASIA is the temporary one STS
        // issues. Both are twenty characters, and both are the identifying
        // half of a credential pair.
        "aws" => format!(
            "{}{}",
            if rng.below(4) == 0 { "ASIA" } else { "AKIA" },
            from_alphabet(rng, AWS_BASE32, 16)
        ),
        // The classic token kinds, all 40 characters: prefix plus 36.
        "github" => {
            let kind = ["ghp", "gho", "ghu", "ghs", "ghr"];
            format!(
                "{}_{}",
                kind[rng.below(kind.len() as u64) as usize],
                from_alphabet(rng, BASE62, 36)
            )
        }
        // The fine-grained token: 93 characters in total, which is the one
        // length GitHub states outright.
        "github-pat" => format!(
            "github_pat_{}",
            from_alphabet(
                rng,
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_",
                82
            )
        ),
        // Secret, restricted and publishable keys, in live and test mode.
        "stripe" => {
            let kind = ["sk", "rk", "pk"];
            let mode = ["live", "test"];
            format!(
                "{}_{}_{}",
                kind[rng.below(kind.len() as u64) as usize],
                mode[rng.below(mode.len() as u64) as usize],
                from_alphabet(rng, BASE62, 24)
            )
        }
        // Both OpenAI generations: the legacy `sk-` plus 48, and the
        // project-scoped shapes that have been the default since 2024.
        "openai" => match rng.below(3) {
            0 => format!("sk-{}", from_alphabet(rng, BASE62, 48)),
            1 => format!("sk-proj-{}", from_alphabet(rng, BASE64URL, 74)),
            _ => format!("sk-svcacct-{}", from_alphabet(rng, BASE64URL, 74)),
        },
        // `api03` is the console API key generation; `oat01` is the
        // Claude.ai OAuth token. The body is 95 characters, ~108 in total.
        "anthropic" => {
            let scheme = ["api03", "oat01"];
            format!(
                "sk-ant-{}-{}",
                scheme[rng.below(scheme.len() as u64) as usize],
                from_alphabet(rng, BASE64URL, 95)
            )
        }
        // Slack's segmented shape: the kind, numeric segments, then the
        // secret. `xoxp` user tokens carry three numeric segments where
        // `xoxb` bot tokens carry two.
        "slack" => {
            let kind = ["xoxb", "xoxp", "xoxa"];
            let chosen = kind[rng.below(kind.len() as u64) as usize];
            let segments = if chosen == "xoxp" { 3 } else { 2 };
            let numeric: Vec<String> = (0..segments).map(|_| rng.digits(11)).collect();
            format!(
                "{chosen}-{}-{}",
                numeric.join("-"),
                from_alphabet(rng, BASE62, 24)
            )
        }
        // 39 characters in total, the one length every published description
        // of the format agrees on.
        "google" => format!("AIza{}", from_alphabet(rng, BASE64URL, 35)),
        // A signed JWT: a real JOSE header, a real claim set, and a
        // signature segment that is random rather than computed -- so the
        // token parses as a JWT everywhere and verifies nowhere.
        //
        // THE CLAIM SET CARRIES PERSONAL DATA ON PURPOSE. `sub`, `email` and
        // `name` are the claims a real token carries, and the whole design
        // question this family answers is what the service does about them.
        // It masks the token and does not decode it, so the values below are
        // never separately reported -- and because they are base64url in the
        // case text, no other recognizer can see them either. The case is
        // what makes that checkable rather than claimed.
        "jwt" => {
            let header = base64url_encode(br#"{"alg":"HS256","typ":"JWT"}"#);
            let handle = from_alphabet(rng, "abcdefghijklmnopqrstuvwxyz", 6);
            let claims = format!(
                r#"{{"sub":"{}","name":"Sanne Doornbos","email":"{handle}@example.com","iat":1787000000}}"#,
                rng.digits(9)
            );
            format!(
                "{header}.{}.{}",
                base64url_encode(claims.as_bytes()),
                from_alphabet(rng, BASE64URL, 43)
            )
        }
        // RFC 6750's `b64token`, introduced by the scheme name. The keyword
        // is part of the drawn value because it is part of the detected
        // span: Rust's regex engine has no lookbehind, so the recognizer
        // cannot anchor on `Bearer` without covering it.
        "bearer" => format!("Bearer {}", from_alphabet(rng, BASE64URL, 40)),
        "pem-rsa" => pem_block(rng, "RSA PRIVATE KEY", 6),
        "pem-openssh" => pem_block(rng, "OPENSSH PRIVATE KEY", 7),
        "pem-ec" => pem_block(rng, "EC PRIVATE KEY", 4),
        "pem-certificate" => pem_block(rng, "CERTIFICATE", 8),
        other => panic!("no generator for credential format {other}"),
    }
}

/// A PEM block: the armour lines, and `lines` lines of 64 base64 characters
/// between them.
///
/// The body is random. That is the safety property, not a shortcut: random
/// base64 does not decode to the ASN.1 SEQUENCE an RSA or EC key is, nor to
/// the `openssh-key-v1` blob OpenSSH expects, so the block is structurally a
/// perfect PEM -- which is all the recognizer keys on -- and functionally
/// nothing at all.
fn pem_block(rng: &mut Rng, label: &str, lines: usize) -> String {
    let body: Vec<String> = (0..lines).map(|_| from_alphabet(rng, BASE64, 64)).collect();
    format!(
        "-----BEGIN {label}-----\n{}\n-----END {label}-----",
        body.join("\n")
    )
}

fn person(rng: &mut Rng, lang: &str) -> String {
    let (given, family) = name_pools(lang);
    format!(
        "{} {}",
        rng.pick(given).copied().unwrap_or("Alex"),
        rng.pick(family).copied().unwrap_or("Morgan")
    )
}

fn street_address(rng: &mut Rng, lang: &str) -> String {
    let (streets, cities) = address_pools(lang);
    let street = rng.pick(streets).copied().unwrap_or("High Street");
    let city = rng.pick(cities).copied().unwrap_or("Springfield");
    let number = 1 + rng.below(180);
    match lang {
        "nl" => {
            let postcode = format!("{}{}", 1 + rng.below(9), rng.digits(3));
            let letters = format!("{}{}", rng.upper(), rng.upper());
            format!("{street} {number}, {postcode} {letters} {city}")
        }
        "de" => format!(
            "{street} {number}, {}{} {city}",
            1 + rng.below(9),
            rng.digits(4)
        ),
        "fr" => format!(
            "{number} {street}, {}{} {city}",
            1 + rng.below(9),
            rng.digits(4)
        ),
        "es" => format!(
            "{street} {number}, {}{} {city}",
            1 + rng.below(4),
            rng.digits(4)
        ),
        "it" => format!(
            "{street} {number}, {}{} {city}",
            1 + rng.below(9),
            rng.digits(4)
        ),
        _ => {
            let outward = format!("{}{}{}", rng.upper(), rng.upper(), 1 + rng.below(9));
            let inward = format!("{}{}{}", rng.digit(), rng.upper(), rng.upper());
            format!("{number} {street}, {city} {outward} {inward}")
        }
    }
}

/// Splits a value [`street_address`] generated into the `must_mask`
/// fragments STREET_ADDRESS's own definition covers.
///
/// THE JUSTIFICATION IS THE TAXONOMY, NOT THE CHECKPOINT. `training/taxonomy.py`
/// maps four separately annotated source labels onto Temperate's one
/// STREET_ADDRESS type: `BUILDINGNUM`/`BUILDING_NUMBER`, `STREET`/`STREET_NAME`,
/// `ZIPCODE`/`ZIP_CODE`, and `SECONDARY_ADDRESS` (see that file's
/// `AI4PRIVACY_MAP`/`NYM_MAP` and its module docs on why STREET_ADDRESS is
/// built by collapsing multiple *positionally adjacent* corpus sub-labels of
/// one templated field). A `must_mask` entry per source label -- the
/// street/number clause, and the postcode clause -- is what STREET_ADDRESS
/// was actually defined to be, independent of what any particular checkpoint
/// happens to predict today; a definition citing observed model output
/// instead would be indistinguishable from fitting the fixture to whatever
/// the model currently gets right, which is the one thing a benchmark must
/// never do.
///
/// THE CITY IS PART OF THE POSTCODE CLAUSE, AND ONLY THERE. `taxonomy.py`
/// still maps every corpus `CITY` label to `None`, and `LOCATION` is still
/// retired: a city standing on its own in prose is not personal data, and
/// every standalone city mention in this fixture stays in `must_clear` where
/// it was. A city at the end of a postal address is a different fact --
/// `Turfkade 54, 3021 LJ Almelo` identifies a household -- so it belongs to
/// the address value it terminates, and to nothing else. That is the whole
/// ruling: position, not vocabulary (see
/// `training/reports/city-address-tails.md` and
/// `crates/model/src/cities.rs`'s module docs).
///
/// WHY THE SPLIT IS NOW LANGUAGE-BLIND WHEN IT WAS NOT BEFORE. Every arm of
/// [`street_address`] writes exactly one `", "`, and everything after it is
/// the postcode clause: `"{postcode} {city}"` for `nl`/`de`/`fr`/`es`/`it`,
/// `"{city} {outward} {inward}"` for `en`, whose city sits *between* the
/// street clause and the postcode rather than after it. While the city was
/// being stripped, those two orders had to be told apart, and `lang` was the
/// only way to do it. Keeping the city makes the tail one value either way,
/// so the parameter went with the code that needed it -- the two orders are
/// still real, and `crates/model/src/decode.rs`'s
/// `extend_address_tail_city` is where they are still handled.
///
/// The comma joining the two fragments is deliberately in neither list: it
/// is prose punctuation, not a source label, and the model's own spans do
/// not bridge it.
///
/// Panics if `value` does not contain `", "` -- every arm of `street_address`
/// writes exactly one, so a value without one did not come from that
/// function and this function has nothing to parse.
fn split_street_address(value: &str) -> Vec<String> {
    let (street_clause, postcode_clause) = value
        .rsplit_once(", ")
        .expect("a street_address value always carries exactly one \", \"");
    vec![street_clause.to_string(), postcode_clause.to_string()]
}

fn ascii_lower(value: &str) -> String {
    value
        .to_lowercase()
        .chars()
        .map(|character| match character {
            'á' | 'à' | 'â' | 'ä' | 'ã' | 'å' => 'a',
            'é' | 'è' | 'ê' | 'ë' => 'e',
            'í' | 'ì' | 'î' | 'ï' => 'i',
            'ó' | 'ò' | 'ô' | 'ö' | 'õ' => 'o',
            'ú' | 'ù' | 'û' | 'ü' => 'u',
            'ç' => 'c',
            'ñ' => 'n',
            'ß' => 's',
            other => other,
        })
        .filter(|character| character.is_ascii_alphanumeric() || *character == '-')
        .collect()
}

// ---------------------------------------------------------------------------
// Name and address pools
//
// Hand-written, and deliberately ordinary rather than distinctive: a name
// that identifies someone real is exactly what a fixture in the repository of
// a privacy product must not contain.
// ---------------------------------------------------------------------------

const LANGUAGES: [&str; 6] = ["nl", "de", "fr", "es", "it", "en"];

const GIVEN_NAMES_NL: [&str; 8] = [
    "Sanne", "Joost", "Marleen", "Bram", "Elske", "Ruben", "Femke", "Teun",
];
const FAMILY_NAMES_NL: [&str; 8] = [
    "Doornbos", "Veenstra", "Kolthoff", "Berkhout", "Rietveld", "Sluiter", "Wolthuis", "Verkerk",
];
const GIVEN_NAMES_DE: [&str; 8] = [
    "Annika", "Lorenz", "Britta", "Hendrik", "Kerstin", "Mattis", "Roswitha", "Falk",
];
const FAMILY_NAMES_DE: [&str; 8] = [
    "Ehrhardt",
    "Klenz",
    "Wolfrum",
    "Stellwag",
    "Brinkmann",
    "Lindhorst",
    "Osterkamp",
    "Rehberg",
];
const GIVEN_NAMES_FR: [&str; 8] = [
    "Amandine", "Cédric", "Solène", "Thibaut", "Maëlle", "Gaspard", "Océane", "Rémy",
];
const FAMILY_NAMES_FR: [&str; 8] = [
    "Delaunay",
    "Chauvet",
    "Berthelot",
    "Marceau",
    "Chapelain",
    "Quillet",
    "Ferrand",
    "Lavigne",
];
const GIVEN_NAMES_ES: [&str; 8] = [
    "Nuria", "Álvaro", "Itziar", "Rubén", "Amaia", "Nicolás", "Lucía", "Iker",
];
const FAMILY_NAMES_ES: [&str; 8] = [
    "Bermejo",
    "Villalba",
    "Colomer",
    "Quintana",
    "Zabaleta",
    "Arriaga",
    "Lucena",
    "Peñalver",
];
const GIVEN_NAMES_IT: [&str; 8] = [
    "Chiara", "Lorenzo", "Beatrice", "Emiliano", "Serena", "Tommaso", "Ilaria", "Riccardo",
];
const FAMILY_NAMES_IT: [&str; 8] = [
    "Zanetti",
    "Battaglia",
    "Meucci",
    "Roversi",
    "Salvadori",
    "Cattaneo",
    "Fioretti",
    "Lombardo",
];
const GIVEN_NAMES_EN: [&str; 8] = [
    "Harriet", "Callum", "Imogen", "Desmond", "Freya", "Rowan", "Marisa", "Gareth",
];
const FAMILY_NAMES_EN: [&str; 8] = [
    "Wainwright",
    "Ashcombe",
    "Thurlow",
    "Pemberton",
    "Halloran",
    "Redmayne",
    "Kilbride",
    "Marlowe",
];

fn name_pools(lang: &str) -> (&'static [&'static str], &'static [&'static str]) {
    match lang {
        "nl" => (&GIVEN_NAMES_NL, &FAMILY_NAMES_NL),
        "de" => (&GIVEN_NAMES_DE, &FAMILY_NAMES_DE),
        "fr" => (&GIVEN_NAMES_FR, &FAMILY_NAMES_FR),
        "es" => (&GIVEN_NAMES_ES, &FAMILY_NAMES_ES),
        "it" => (&GIVEN_NAMES_IT, &FAMILY_NAMES_IT),
        _ => (&GIVEN_NAMES_EN, &FAMILY_NAMES_EN),
    }
}

const STREETS_NL: [&str; 6] = [
    "Kolkweg",
    "Bergerhout",
    "Meidoornsingel",
    "Sluisstraat",
    "Vlierlaan",
    "Turfkade",
];
const CITIES_NL: [&str; 6] = [
    "Deventer",
    "Almelo",
    "Hoorn",
    "Sneek",
    "Roosendaal",
    "Veenendaal",
];
const STREETS_DE: [&str; 6] = [
    "Schlehenweg",
    "Kupferstraße",
    "Talmühlenweg",
    "Brunnenallee",
    "Erlenkamp",
    "Rosseggerstraße",
];
const CITIES_DE: [&str; 6] = [
    "Detmold",
    "Fulda",
    "Ansbach",
    "Neuruppin",
    "Emden",
    "Zwickau",
];
const STREETS_FR: [&str; 6] = [
    "rue des Charmilles",
    "avenue du Vieux Moulin",
    "impasse Lafontaine",
    "chemin des Aubépines",
    "rue du Pressoir",
    "boulevard des Tanneurs",
];
const CITIES_FR: [&str; 6] = [
    "Bourges", "Quimper", "Albi", "Chartres", "Vesoul", "Auxerre",
];
const STREETS_ES: [&str; 6] = [
    "calle del Olivar",
    "avenida de los Almendros",
    "plaza del Retiro Viejo",
    "calle Zurbarán",
    "paseo de la Alameda",
    "calle Mercaderes",
];
const CITIES_ES: [&str; 6] = [
    "Palencia", "Lorca", "Ourense", "Manresa", "Teruel", "Ferrol",
];
const STREETS_IT: [&str; 6] = [
    "via dei Gelsi",
    "corso Vittorio Emanuele",
    "vicolo della Fontanella",
    "via Palestro",
    "viale dei Tigli",
    "via Sant'Orsola",
];
const CITIES_IT: [&str; 6] = [
    "Rovigo",
    "Ascoli Piceno",
    "Vercelli",
    "Matera",
    "Lecco",
    "Crema",
];
const STREETS_EN: [&str; 6] = [
    "Chandler's Row",
    "Marlpit Lane",
    "Coppice Way",
    "Fen End Road",
    "Bramble Close",
    "Weaver's Yard",
];
const CITIES_EN: [&str; 6] = [
    "Shrewsbury",
    "Kendal",
    "Bury St Edmunds",
    "Alnwick",
    "Chippenham",
    "Ripon",
];

fn address_pools(lang: &str) -> (&'static [&'static str], &'static [&'static str]) {
    match lang {
        "nl" => (&STREETS_NL, &CITIES_NL),
        "de" => (&STREETS_DE, &CITIES_DE),
        "fr" => (&STREETS_FR, &CITIES_FR),
        "es" => (&STREETS_ES, &CITIES_ES),
        "it" => (&STREETS_IT, &CITIES_IT),
        _ => (&STREETS_EN, &CITIES_EN),
    }
}

// ---------------------------------------------------------------------------
// The sentence built around a value
// ---------------------------------------------------------------------------

/// The clause that introduces a value at a sentence boundary, and the short
/// label that introduces it mid-sentence. One pair per language per entity
/// type, so a value never arrives naked in a place a reader would find one.
struct Phrases {
    lead: &'static str,
    label: &'static str,
}

/// The sentence built around one *named* country identifier, where the entity
/// type alone cannot choose the words.
///
/// `NATIONAL_ID` in Spanish is either a DNI or a NIE and a Spaniard would
/// never call one by the other's name; `NATIONAL_ID` in French is either the
/// French social security number or the Belgian national register number.
/// [`phrases`] is keyed by (language, type) and cannot tell those apart, so
/// this table is consulted first and falls through to it for every key that
/// needs no distinction -- which is why `"NL"` and `"DE"` are absent, and why
/// the Dutch and German cases in `carriers.json` are unchanged by any of
/// this.
fn country_phrases(lang: &str, key: &str) -> Option<Phrases> {
    match (key, lang) {
        ("ES-DNI", "es") => Some(Phrases {
            lead: "El DNI del solicitante es",
            label: "DNI",
        }),
        ("ES-DNI", _) => Some(Phrases {
            lead: "The applicant's Spanish DNI is",
            label: "DNI",
        }),
        ("ES-NIE", "es") => Some(Phrases {
            lead: "El número de identidad de extranjero del solicitante es",
            label: "NIE",
        }),
        ("ES-NIE", _) => Some(Phrases {
            lead: "The applicant's Spanish foreigner identity number is",
            label: "NIE",
        }),
        ("IT", "it") => Some(Phrases {
            lead: "Il codice fiscale del richiedente è",
            label: "codice fiscale",
        }),
        ("IT", _) => Some(Phrases {
            lead: "The applicant's Italian codice fiscale is",
            label: "codice fiscale",
        }),
        ("FR", "fr") => Some(Phrases {
            lead: "Le numéro de sécurité sociale du demandeur est",
            label: "numéro de sécurité sociale",
        }),
        ("FR", _) => Some(Phrases {
            lead: "The applicant's French social security number is",
            label: "French social security number",
        }),
        ("BE", "nl") => Some(Phrases {
            lead: "Het rijksregisternummer van de aanvrager is",
            label: "rijksregisternummer",
        }),
        ("BE", "fr") => Some(Phrases {
            lead: "Le numéro de registre national du demandeur est",
            label: "numéro de registre national",
        }),
        ("BE", _) => Some(Phrases {
            lead: "The applicant's Belgian national number is",
            label: "Belgian national number",
        }),
        _ => None,
    }
}

/// The sentence built around one credential, where the entity type alone
/// cannot choose the words.
///
/// Consulted ahead of [`phrases`] and falling through to it for every format
/// that needs no distinction -- which is why the API-key formats are absent,
/// and why `benchmark/cases/credential-apikey.json` is byte-for-byte what it
/// was before this function existed.
/// The sentence built around an order number that is there to *not* be
/// masked -- **and that names a driving licence in the same clause**.
///
/// THIS IS THE BRIEF'S REQUIRED HARD NEGATIVE, AND IT IS BUILT INTO THE
/// SENTENCE RATHER THAN LEFT TO WHERE THE INJECTOR PUT THE VALUE. A licence
/// word anywhere inside the matcher's 64-byte window carries a
/// licence-shaped run over the bar, so the case that decides these
/// recognizers is an order number with `rijbewijs` (or `permis de conduire`,
/// or `patente di guida`) in the same window. Injected at two independent
/// slots that is left to chance: measured over the committed Dutch file, the
/// two values land 132 characters apart at the closest, so *not one* of the
/// twelve would have exercised the window at all.
///
/// Every lead and every label therefore carries both -- a term from the
/// recognizer's `negative_context` list and a term from its `context_terms`
/// -- so the window always holds both and the only thing that can decide the
/// case is which of the two weighs more. The Dutch, French and Italian
/// recognizers have no checksum, so that is the whole of what stands between
/// this fixture and an over-masked order number.
fn order_reference_phrases(lang: &str) -> Phrases {
    match lang {
        "nl" => Phrases {
            lead: "Het ordernummer in dit rijbewijsdossier is",
            label: "ordernummer bij het rijbewijs",
        },
        "de" => Phrases {
            lead: "Die Bestellnummer in dieser Führerscheinakte lautet",
            label: "Bestellnummer zum Führerschein",
        },
        "fr" => Phrases {
            lead: "Le numéro de commande de ce dossier de permis de conduire est",
            label: "numéro de commande, permis de conduire",
        },
        "es" => Phrases {
            lead: "El número de pedido de este expediente del permiso de conducir es",
            label: "número de pedido, permiso de conducir",
        },
        "it" => Phrases {
            lead: "Il numero d'ordine di questa pratica della patente di guida è",
            label: "numero d'ordine, patente di guida",
        },
        _ => Phrases {
            lead: "The order number on this driving licence file is",
            label: "order number, driving licence",
        },
    }
}

/// The sentence built around one driving licence, where the entity type
/// alone cannot choose the words.
///
/// Consulted ahead of [`phrases`] for the same reason [`country_phrases`] is:
/// a German Führerschein and a British driver number are both
/// `DRIVING_LICENCE`, they are introduced by different words, and every one
/// of the six carrier languages names the other country's licence in its own
/// way. The country is named in the prose on purpose -- a Spanish company
/// writing about a German driver writes "el permiso de conducción alemán",
/// not "el permiso de conducción" -- so a reader of the fixture can see which
/// identifier a case is about without decoding the value.
///
/// Every lead carries the licence's own word, which is what the packs'
/// context terms key on. The mid-sentence label is the bare word, because
/// that is what a parenthetical carries.
fn licence_phrases(lang: &str, country: &str) -> Option<Phrases> {
    Some(match (country, lang) {
        ("DE", "de") => Phrases {
            lead: "Die Führerscheinnummer des Antragstellers lautet",
            label: "Führerschein",
        },
        ("DE", "nl") => Phrases {
            lead: "Het Duitse rijbewijsnummer van de aanvrager is",
            label: "Duits rijbewijs",
        },
        ("DE", "fr") => Phrases {
            lead: "Le numéro du permis de conduire allemand du demandeur est",
            label: "permis de conduire allemand",
        },
        ("DE", "es") => Phrases {
            lead: "El número del permiso de conducir alemán del solicitante es",
            label: "permiso de conducir alemán",
        },
        ("DE", "it") => Phrases {
            lead: "Il numero della patente di guida tedesca del richiedente è",
            label: "patente di guida tedesca",
        },
        ("DE", _) => Phrases {
            lead: "The applicant's German driving licence number is",
            label: "German driving licence",
        },
        ("GB", "de") => Phrases {
            lead: "Die Nummer des britischen Führerscheins des Antragstellers lautet",
            label: "britischer Führerschein",
        },
        ("GB", "nl") => Phrases {
            lead: "Het Britse rijbewijsnummer van de aanvrager is",
            label: "Brits rijbewijs",
        },
        ("GB", "fr") => Phrases {
            lead: "Le numéro du permis de conduire britannique du demandeur est",
            label: "permis de conduire britannique",
        },
        ("GB", "es") => Phrases {
            lead: "El número del permiso de conducir británico del solicitante es",
            label: "permiso de conducir británico",
        },
        ("GB", "it") => Phrases {
            lead: "Il numero della patente di guida britannica del richiedente è",
            label: "patente di guida britannica",
        },
        ("GB", _) => Phrases {
            lead: "The applicant's UK driving licence number is",
            label: "driving licence",
        },
        ("NL", "nl") => Phrases {
            lead: "Het rijbewijsnummer van de aanvrager is",
            label: "rijbewijs",
        },
        ("NL", "de") => Phrases {
            lead: "Die Nummer des niederländischen Führerscheins lautet",
            label: "niederländischer Führerschein",
        },
        ("NL", "fr") => Phrases {
            lead: "Le numéro du permis de conduire néerlandais du demandeur est",
            label: "permis de conduire néerlandais",
        },
        ("NL", "es") => Phrases {
            lead: "El número del permiso de conducir neerlandés del solicitante es",
            label: "permiso de conducir neerlandés",
        },
        ("NL", "it") => Phrases {
            lead: "Il numero della patente di guida olandese del richiedente è",
            label: "patente di guida olandese",
        },
        ("NL", _) => Phrases {
            lead: "The applicant's Dutch driving licence number is",
            label: "Dutch driving licence",
        },
        ("FR", "fr") => Phrases {
            lead: "Le numéro du permis de conduire du demandeur est",
            label: "permis de conduire",
        },
        ("FR", "nl") => Phrases {
            lead: "Het Franse rijbewijsnummer van de aanvrager is",
            label: "Frans rijbewijs",
        },
        ("FR", "de") => Phrases {
            lead: "Die Nummer des französischen Führerscheins lautet",
            label: "französischer Führerschein",
        },
        ("FR", "es") => Phrases {
            lead: "El número del permiso de conducir francés del solicitante es",
            label: "permiso de conducir francés",
        },
        ("FR", "it") => Phrases {
            lead: "Il numero della patente di guida francese del richiedente è",
            label: "patente di guida francese",
        },
        ("FR", _) => Phrases {
            lead: "The applicant's French driving licence number is",
            label: "French driving licence",
        },
        ("IT", "it") => Phrases {
            lead: "Il numero della patente di guida del richiedente è",
            label: "patente di guida",
        },
        ("IT", "nl") => Phrases {
            lead: "Het Italiaanse rijbewijsnummer van de aanvrager is",
            label: "Italiaans rijbewijs",
        },
        ("IT", "de") => Phrases {
            lead: "Die Nummer des italienischen Führerscheins lautet",
            label: "italienischer Führerschein",
        },
        ("IT", "fr") => Phrases {
            lead: "Le numéro du permis de conduire italien du demandeur est",
            label: "permis de conduire italien",
        },
        ("IT", "es") => Phrases {
            lead: "El número del permiso de conducir italiano del solicitante es",
            label: "permiso de conducir italiano",
        },
        ("IT", _) => Phrases {
            lead: "The applicant's Italian driving licence number is",
            label: "Italian driving licence",
        },
        _ => return None,
    })
}

fn credential_phrases(lang: &str, format: &str) -> Option<Phrases> {
    if format == "jwt" || format == "bearer" {
        return Some(match lang {
            "nl" => Phrases {
                lead: "De autorisatieheader in de logregel bevat",
                label: "authorization",
            },
            "de" => Phrases {
                lead: "Der Autorisierungsheader in der Protokollzeile enthält",
                label: "authorization",
            },
            "fr" => Phrases {
                lead: "L'en-tête d'autorisation de la ligne de journal contient",
                label: "authorization",
            },
            "es" => Phrases {
                lead: "La cabecera de autorización de la línea de registro contiene",
                label: "authorization",
            },
            "it" => Phrases {
                lead: "L'intestazione di autorizzazione della riga di log contiene",
                label: "authorization",
            },
            _ => Phrases {
                lead: "The authorization header on the logged request carried",
                label: "authorization",
            },
        });
    }
    if format.starts_with("pem-") {
        return Some(match lang {
            "nl" => Phrases {
                lead: "Het configuratiebestand eindigt met dit sleutelmateriaal:",
                label: "credential",
            },
            "de" => Phrases {
                lead: "Die Konfigurationsdatei endet mit diesem Schlüsselmaterial:",
                label: "credential",
            },
            "fr" => Phrases {
                lead: "Le fichier de configuration se termine par ce matériel de clé :",
                label: "credential",
            },
            "es" => Phrases {
                lead: "El archivo de configuración termina con este material de clave:",
                label: "credential",
            },
            "it" => Phrases {
                lead: "Il file di configurazione termina con questo materiale di chiave:",
                label: "credential",
            },
            _ => Phrases {
                lead: "The configuration file ends with this key material:",
                label: "credential",
            },
        });
    }
    None
}

fn phrases(lang: &str, entity_type: Option<EntityType>) -> Phrases {
    let Some(entity_type) = entity_type else {
        return Phrases {
            lead: match lang {
                "nl" => "De referentie is",
                "de" => "Die Referenz lautet",
                "fr" => "La référence est",
                "es" => "La referencia es",
                "it" => "Il riferimento è",
                _ => "The reference is",
            },
            label: "ref",
        };
    };
    match (lang, entity_type) {
        ("nl", EntityType::NationalId) => Phrases {
            lead: "Het burgerservicenummer van de aanvrager is",
            label: "BSN",
        },
        ("nl", EntityType::TaxId) => Phrases {
            lead: "Het fiscaal nummer van de onderneming is",
            label: "belastingnummer",
        },
        ("nl", EntityType::IbanCode) => Phrases {
            lead: "De betaling verloopt via rekeningnummer",
            label: "IBAN",
        },
        ("nl", EntityType::CreditCard) => Phrases {
            lead: "Het kaartnummer in het dossier is",
            label: "creditcard",
        },
        ("nl", EntityType::EmailAddress) => Phrases {
            lead: "Vragen kunnen per e-mail worden gestuurd naar",
            label: "e-mail",
        },
        ("nl", EntityType::IpAddress) => Phrases {
            lead: "De aanvraag kwam binnen vanaf IP-adres",
            label: "IP-adres",
        },
        ("nl", EntityType::PhoneNumber) => Phrases {
            lead: "De contactpersoon is telefonisch bereikbaar op",
            label: "telefoon",
        },
        ("nl", EntityType::VatNumber) => Phrases {
            lead: "Het btw-nummer van de leverancier is",
            label: "btw-nummer",
        },
        ("nl", EntityType::PassportNumber) => Phrases {
            lead: "Het paspoortnummer in het dossier is",
            label: "paspoortnummer",
        },
        ("nl", EntityType::Person) => Phrases {
            lead: "Het dossier is opgesteld door",
            label: "opgesteld door",
        },
        ("nl", EntityType::StreetAddress) => Phrases {
            lead: "Het correspondentieadres is",
            label: "adres",
        },

        ("de", EntityType::NationalId) => Phrases {
            lead: "Die Ausweisnummer des Antragstellers lautet",
            label: "Ausweisnummer",
        },
        ("de", EntityType::TaxId) => Phrases {
            lead: "Die Steuer-Identifikationsnummer lautet",
            label: "Steuer-ID",
        },
        ("de", EntityType::IbanCode) => Phrases {
            lead: "Die Zahlung erfolgt auf die Kontonummer",
            label: "IBAN",
        },
        ("de", EntityType::CreditCard) => Phrases {
            lead: "Die hinterlegte Kartennummer ist",
            label: "Kreditkarte",
        },
        ("de", EntityType::EmailAddress) => Phrases {
            lead: "Rückfragen bitte per E-Mail an",
            label: "E-Mail",
        },
        ("de", EntityType::IpAddress) => Phrases {
            lead: "Der Zugriff erfolgte von der IP-Adresse",
            label: "IP-Adresse",
        },
        ("de", EntityType::PhoneNumber) => Phrases {
            lead: "Die Ansprechpartnerin ist telefonisch erreichbar unter",
            label: "Telefon",
        },
        ("de", EntityType::VatNumber) => Phrases {
            lead: "Die Umsatzsteuer-Identifikationsnummer des Lieferanten ist",
            label: "USt-IdNr",
        },
        ("de", EntityType::PassportNumber) => Phrases {
            lead: "Die Passnummer im Vorgang lautet",
            label: "Passnummer",
        },
        ("de", EntityType::Person) => Phrases {
            lead: "Der Vorgang wurde bearbeitet von",
            label: "bearbeitet von",
        },
        ("de", EntityType::StreetAddress) => Phrases {
            lead: "Die Zustellanschrift lautet",
            label: "Anschrift",
        },

        ("fr", EntityType::NationalId) => Phrases {
            lead: "Le numéro national du demandeur est",
            label: "numéro national",
        },
        ("fr", EntityType::TaxId) => Phrases {
            lead: "Le numéro fiscal de l'entreprise est",
            label: "numéro fiscal",
        },
        ("fr", EntityType::IbanCode) => Phrases {
            lead: "Le règlement est effectué sur le compte IBAN",
            label: "IBAN",
        },
        ("fr", EntityType::CreditCard) => Phrases {
            lead: "Le numéro de carte de crédit enregistré est",
            label: "carte de crédit",
        },
        ("fr", EntityType::EmailAddress) => Phrases {
            lead: "Les questions peuvent être adressées par courriel à",
            label: "courriel",
        },
        ("fr", EntityType::IpAddress) => Phrases {
            lead: "La demande provient de l'adresse IP",
            label: "adresse IP",
        },
        ("fr", EntityType::PhoneNumber) => Phrases {
            lead: "La personne à contacter est joignable par téléphone au",
            label: "téléphone",
        },
        ("fr", EntityType::VatNumber) => Phrases {
            lead: "Le numéro de TVA du fournisseur est",
            label: "numéro de TVA",
        },
        ("fr", EntityType::PassportNumber) => Phrases {
            lead: "Le numéro de passeport figurant au dossier est",
            label: "numéro de passeport",
        },
        ("fr", EntityType::Person) => Phrases {
            lead: "Le dossier a été établi par",
            label: "établi par",
        },
        ("fr", EntityType::StreetAddress) => Phrases {
            lead: "L'adresse de correspondance est",
            label: "adresse",
        },

        ("es", EntityType::NationalId) => Phrases {
            lead: "El número nacional de identidad del solicitante es",
            label: "número nacional",
        },
        ("es", EntityType::TaxId) => Phrases {
            lead: "El número de identificación fiscal de la empresa es",
            label: "NIF",
        },
        ("es", EntityType::IbanCode) => Phrases {
            lead: "El pago se realiza en la cuenta IBAN",
            label: "IBAN",
        },
        ("es", EntityType::CreditCard) => Phrases {
            lead: "El número de tarjeta de crédito registrado es",
            label: "tarjeta de crédito",
        },
        ("es", EntityType::EmailAddress) => Phrases {
            lead: "Las consultas pueden enviarse por correo electrónico a",
            label: "correo",
        },
        ("es", EntityType::IpAddress) => Phrases {
            lead: "La solicitud se recibió desde la dirección IP",
            label: "dirección IP",
        },
        ("es", EntityType::PhoneNumber) => Phrases {
            lead: "La persona de contacto está disponible por teléfono en el",
            label: "teléfono",
        },
        ("es", EntityType::VatNumber) => Phrases {
            lead: "El número de IVA del proveedor es",
            label: "número de IVA",
        },
        ("es", EntityType::PassportNumber) => Phrases {
            lead: "El número de pasaporte que consta en el expediente es",
            label: "número de pasaporte",
        },
        ("es", EntityType::Person) => Phrases {
            lead: "El expediente fue redactado por",
            label: "redactado por",
        },
        ("es", EntityType::StreetAddress) => Phrases {
            lead: "La dirección de correspondencia es",
            label: "dirección",
        },

        ("it", EntityType::NationalId) => Phrases {
            lead: "Il numero della carta d'identità del richiedente è",
            label: "carta d'identità",
        },
        ("it", EntityType::TaxId) => Phrases {
            lead: "Il codice fiscale del richiedente è",
            label: "codice fiscale",
        },
        ("it", EntityType::IbanCode) => Phrases {
            lead: "Il pagamento è effettuato sul conto IBAN",
            label: "IBAN",
        },
        ("it", EntityType::CreditCard) => Phrases {
            lead: "Il numero di carta di credito registrato è",
            label: "carta di credito",
        },
        ("it", EntityType::EmailAddress) => Phrases {
            lead: "Le domande possono essere inviate per posta elettronica a",
            label: "e-mail",
        },
        ("it", EntityType::IpAddress) => Phrases {
            lead: "La richiesta proviene dall'indirizzo IP",
            label: "indirizzo IP",
        },
        ("it", EntityType::PhoneNumber) => Phrases {
            lead: "La persona di riferimento è raggiungibile telefonicamente al",
            label: "telefono",
        },
        ("it", EntityType::VatNumber) => Phrases {
            lead: "La partita IVA del fornitore è",
            label: "partita IVA",
        },
        ("it", EntityType::PassportNumber) => Phrases {
            lead: "Il numero di passaporto indicato nel fascicolo è",
            label: "numero di passaporto",
        },
        ("it", EntityType::Person) => Phrases {
            lead: "Il fascicolo è stato redatto da",
            label: "redatto da",
        },
        ("it", EntityType::StreetAddress) => Phrases {
            lead: "L'indirizzo di corrispondenza è",
            label: "indirizzo",
        },

        (_, EntityType::NationalId) => Phrases {
            lead: "The applicant's citizen service number is",
            label: "citizen service number",
        },
        (_, EntityType::TaxId) => Phrases {
            lead: "The tax identification number on file is",
            label: "tax id",
        },
        (_, EntityType::IbanCode) => Phrases {
            lead: "Payment is made to the IBAN account number",
            label: "IBAN",
        },
        (_, EntityType::CreditCard) => Phrases {
            lead: "The credit card number on file is",
            label: "credit card",
        },
        (_, EntityType::EmailAddress) => Phrases {
            lead: "Questions may be sent by email to",
            label: "email",
        },
        (_, EntityType::IpAddress) => Phrases {
            lead: "The request came from IP address",
            label: "IP address",
        },
        (_, EntityType::PhoneNumber) => Phrases {
            lead: "The contact person can be reached by phone on",
            label: "phone",
        },
        (_, EntityType::VatNumber) => Phrases {
            lead: "The supplier's VAT number is",
            label: "VAT number",
        },
        (_, EntityType::PassportNumber) => Phrases {
            lead: "The passport number on file is",
            label: "passport number",
        },
        (_, EntityType::Person) => Phrases {
            lead: "The file was prepared by",
            label: "prepared by",
        },
        (_, EntityType::StreetAddress) => Phrases {
            lead: "The correspondence address is",
            label: "address",
        },
        // A credential has no nationality, and neither does the sentence a
        // developer writes around one: the six languages differ only in the
        // verb. The word chosen is a *credential* label, not a placeholder
        // label, so every carrier case sits on the ladder's top rung -- the
        // rung a real key in a real config paste sits on. The placeholder
        // rung is exercised by the hand-authored cases and by
        // `pii-detection-rules/tests/credentials.rs`, where the text can be written
        // to say so.
        ("nl", EntityType::Credential) => Phrases {
            lead: "De API-sleutel in de configuratie is",
            label: "api key",
        },
        ("de", EntityType::Credential) => Phrases {
            lead: "Der API-Schlüssel in der Konfiguration lautet",
            label: "api key",
        },
        ("fr", EntityType::Credential) => Phrases {
            lead: "La clé d'API figurant dans la configuration est",
            label: "api key",
        },
        ("es", EntityType::Credential) => Phrases {
            lead: "La clave de API que figura en la configuración es",
            label: "api key",
        },
        ("it", EntityType::Credential) => Phrases {
            lead: "La chiave API presente nella configurazione è",
            label: "api key",
        },
        (_, EntityType::Credential) => Phrases {
            lead: "The API key in the deployment configuration is",
            label: "api key",
        },
        (_, EntityType::DrivingLicence) => Phrases {
            lead: "The driving licence number on file is",
            label: "driving licence",
        },
    }
}

// ---------------------------------------------------------------------------
// Where a value goes
// ---------------------------------------------------------------------------

/// A place a value can be spliced in: a byte offset into the carrier, and
/// which of the two forms belongs there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Slot {
    at: usize,
    boundary: bool,
}

/// Every sentence boundary and every mid-sentence position in `text`.
///
/// A boundary slot is the start of a sentence, where a whole new sentence can
/// be spliced in front of it (plus one past the end of the text, so a
/// single-sentence carrier still has one). A mid-sentence slot is the offset
/// of a sentence's own terminating mark, where a parenthetical goes in before
/// it.
///
/// The filter is deliberately conservative about what counts as the end of a
/// sentence: the mark must follow a word of at least two characters and be
/// followed by whitespace and an upper-case letter, or by nothing at all.
/// That rejects `z. B.`, `ca. 80.000` and `art. 51`, which are the shapes
/// that would otherwise put a parenthetical inside an abbreviation.
fn slots(text: &str) -> Vec<Slot> {
    let bytes = text.as_bytes();
    let mut boundaries = Vec::new();
    let mut middles = Vec::new();

    for (index, character) in text.char_indices() {
        if !matches!(character, '.' | '!' | '?') {
            continue;
        }
        let before: Vec<char> = text[..index].chars().rev().take(2).collect();
        let ends_a_word = matches!(before.first(), Some(c) if c.is_alphanumeric() || *c == ')');
        let is_an_initial =
            matches!(before.get(1), Some(c) if c.is_whitespace()) && before.len() > 1;
        if !ends_a_word || is_an_initial {
            continue;
        }

        let rest = &text[index + character.len_utf8()..];
        let next_word = rest.trim_start();
        let gap = &rest[..rest.len() - next_word.len()];
        if !rest.is_empty() && gap.is_empty() {
            // Something is glued to the mark: a decimal point, an ellipsis,
            // a version string. Not the end of anything.
            continue;
        }
        if let Some(first) = next_word.chars().next() {
            // A paragraph break is itself proof that the sentence ended. On
            // the same line, the next thing has to look like the start of
            // one: a capital, or the bracket a numbered paragraph opens with.
            let new_paragraph = gap.contains('\n');
            let opens_a_sentence =
                first.is_uppercase() || matches!(first, '(' | '"' | '\u{201e}' | '\u{00ab}');
            if !(new_paragraph || opens_a_sentence) {
                continue;
            }
            // A whole sentence may be spliced in front of this one, but not
            // in front of a numbered paragraph: `1169-d1` put a sentence
            // between a legal act's numbering and its text, which reads as
            // nothing any register produces and made the case ambiguous
            // between a fixture artefact and a detector weakness. A fixture
            // that is ambiguous about what it is testing is not a gate.
            if !first.is_numeric() {
                boundaries.push(text.len() - next_word.len());
            }
        }
        middles.push(index);
    }

    // One boundary past the end, so a carrier of a single sentence -- or the
    // last sentence of any carrier -- can still take a whole new sentence.
    let trimmed = text.trim_end().len();
    if trimmed > 0 && matches!(bytes.get(trimmed - 1), Some(b'.') | Some(b'!') | Some(b'?')) {
        boundaries.push(trimmed);
    }

    let mut all: Vec<Slot> = boundaries
        .into_iter()
        .map(|at| Slot { at, boundary: true })
        .chain(middles.into_iter().map(|at| Slot {
            at,
            boundary: false,
        }))
        .collect();
    all.sort_by_key(|slot| (slot.at, slot.boundary));
    all
}

// ---------------------------------------------------------------------------
// Injection
// ---------------------------------------------------------------------------

/// Every way injection can refuse to build a case.
///
/// No variant carries a value or a piece of carrier text: an index into the
/// caller's own slice is enough to find the offender, and error types are
/// reachable from log lines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InjectError {
    /// The value for `injections[index]` already occurs in the carrier, and
    /// re-drawing did not find one that does not. `must_mask` would then name
    /// a substring appearing twice and the case would silently demand masking
    /// of the carrier's own text.
    AlreadyPresent { index: usize },
    /// The carrier has no sentence this can be spliced into.
    NoInsertionPoint,
}

impl fmt::Display for InjectError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlreadyPresent { index } => write!(
                formatter,
                "the value drawn for injection {index} already occurs in the carrier"
            ),
            Self::NoInsertionPoint => {
                write!(formatter, "the carrier has no sentence to inject into")
            }
        }
    }
}

impl std::error::Error for InjectError {}

/// Places every injection in `carrier` and returns the resulting case.
///
/// The infallible form, for callers whose injections are all generated: a
/// generated value that collides with the carrier is simply re-drawn, so the
/// only way to reach an error is [`Injection::exact`], whose literal the
/// caller chose. When that happens there is nothing to fall back to, so the
/// carrier comes back untouched with an empty answer key -- a case that
/// demands nothing, which is inert and scoreable, rather than one that
/// quietly demands masking of the carrier's own text. The case-building CLI
/// below never takes that path: it calls [`inject_checked`] and stops.
pub fn inject(carrier: &str, injections: &[Injection], seed: u64) -> Case {
    match inject_checked(carrier, injections, seed) {
        Ok(case) => case,
        Err(_) => bare_case(carrier, injections),
    }
}

/// Places every injection in `carrier`, or says why it cannot.
pub fn inject_checked(
    carrier: &str,
    injections: &[Injection],
    seed: u64,
) -> Result<Case, InjectError> {
    build_case(carrier, injections, seed).map(|(case, _)| case)
}

/// How many values went in at a sentence boundary and how many mid-sentence.
///
/// Reported rather than assumed. The brief requires the two positions in
/// roughly equal proportion, and the first version of this file claimed that
/// "by construction" while actually producing 170/82: every single-value case
/// alternated from index 0 and so was always a boundary. The claim was backed
/// by a test using a four-injection carrier -- the one shape where
/// alternation balances on its own, and therefore the one shape that could
/// not detect the fault. Counting the real thing and printing it is what
/// makes the claim checkable.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Split {
    pub boundary: usize,
    pub mid: usize,
}

impl Split {
    fn count(&mut self, slots: &[Slot]) {
        for slot in slots {
            if slot.boundary {
                self.boundary += 1;
            } else {
                self.mid += 1;
            }
        }
    }

    fn total(&self) -> usize {
        self.boundary + self.mid
    }

    /// The boundary share, as a fraction. `None` when nothing was placed.
    pub fn boundary_share(&self) -> Option<f64> {
        if self.total() == 0 {
            None
        } else {
            Some(self.boundary as f64 / self.total() as f64)
        }
    }
}

/// [`inject_checked`], plus the slots it chose, so a caller can count the
/// boundary/mid-sentence split instead of taking a claim about it on trust.
fn build_case(
    carrier: &str,
    injections: &[Injection],
    seed: u64,
) -> Result<(Case, Vec<Slot>), InjectError> {
    let available = slots(carrier);
    if available.is_empty() && !injections.is_empty() {
        return Err(InjectError::NoInsertionPoint);
    }

    let mut rng = Rng::new(seed);
    let mut drawn: Vec<(usize, String)> = Vec::new();
    let mut taken: BTreeSet<String> = BTreeSet::new();

    for (index, injection) in injections.iter().enumerate() {
        let attempts = if injection.is_generated() { 64 } else { 1 };
        let mut value = None;
        for _ in 0..attempts {
            let candidate = injection.draw(&mut rng);
            let free = !carrier.contains(&candidate)
                && !taken
                    .iter()
                    .any(|other| other.contains(&candidate) || candidate.contains(other.as_str()));
            if free {
                value = Some(candidate);
                break;
            }
        }
        let Some(value) = value else {
            return Err(InjectError::AlreadyPresent { index });
        };
        taken.insert(value.clone());
        drawn.push((index, value));
    }

    // Choose a slot per injection: the two kinds alternate by index, so a
    // case holding several values uses both. WHICH KIND INDEX ZERO GETS IS
    // DRAWN FROM THE SEED, and that is the whole fix for a real fault: with
    // index zero always preferring a boundary, every single-value case in the
    // fixture was a boundary case, and the corpus-wide split came out 170/82
    // rather than the roughly equal proportion the brief requires. The phase
    // comes from a stream of its own so that adding it does not perturb the
    // values themselves.
    let phase = (Rng::new(seed ^ 0x5EED_0F5E).next_u64() % 2) as usize;
    let boundaries: Vec<Slot> = available.iter().copied().filter(|s| s.boundary).collect();
    let middles: Vec<Slot> = available.iter().copied().filter(|s| !s.boundary).collect();

    let mut placements: Vec<(Slot, String)> = Vec::new();
    let mut used: BTreeSet<usize> = BTreeSet::new();
    for (index, value) in &drawn {
        let prefer_boundary = (index + phase).is_multiple_of(2);
        let first = if prefer_boundary {
            &boundaries
        } else {
            &middles
        };
        let second = if prefer_boundary {
            &middles
        } else {
            &boundaries
        };
        let slot = free_slot(first, &used, &mut rng)
            .or_else(|| free_slot(second, &used, &mut rng))
            .ok_or(InjectError::NoInsertionPoint)?;
        used.insert(slot.at);

        let injection = injections
            .get(*index)
            .ok_or(InjectError::NoInsertionPoint)?;
        let phrases = injection
            .identifier_key()
            .and_then(|key| country_phrases(&injection.lang, key))
            .or_else(|| {
                injection
                    .credential_format()
                    .and_then(|format| credential_phrases(&injection.lang, format))
            })
            .or_else(|| {
                injection
                    .licence_country()
                    .and_then(|country| licence_phrases(&injection.lang, country))
            })
            .or_else(|| {
                injection
                    .is_order_reference()
                    .then(|| order_reference_phrases(&injection.lang))
            })
            .unwrap_or_else(|| phrases(&injection.lang, injection.entity_type()));
        let rendered = if slot.boundary {
            format!("{} {}. ", phrases.lead, value)
        } else {
            format!(" ({} {})", phrases.label, value)
        };
        placements.push((slot, rendered));
    }

    // Splice from the back so the offsets ahead of each insertion stay valid.
    placements.sort_by_key(|(slot, _)| std::cmp::Reverse(slot.at));
    let mut text = carrier.to_string();
    for (slot, rendered) in &placements {
        if slot.at == text.trim_end().len() && slot.boundary {
            // Appending after the final sentence: one space, and no trailing
            // one, so the carrier does not acquire stray whitespace.
            let trimmed = text.trim_end().to_string();
            let tail = text[text.trim_end().len()..].to_string();
            text = format!("{trimmed} {}{tail}", rendered.trim_end());
        } else {
            text.insert_str(slot.at, rendered);
        }
    }

    // A STREET_ADDRESS injection's drawn value keeps the whole generated
    // string in `text` (prose is untouched) and splits into the two
    // must_mask fragments STREET_ADDRESS is defined as -- the street clause
    // and the postcode clause, the latter including the city that
    // terminates the address -- see split_street_address's own docs. No
    // city is carried out to must_clear any more: in address position the
    // city is part of the address, and a standalone city mention is not an
    // address and never reaches this code at all.
    let mut must_mask: Vec<MustMask> = Vec::new();
    for (index, value) in &drawn {
        let Some(injection) = injections.get(*index) else {
            continue;
        };
        match injection.entity_type() {
            Some(EntityType::StreetAddress) => {
                must_mask.extend(
                    split_street_address(value)
                        .into_iter()
                        .map(|fragment| MustMask::typed(fragment, EntityType::StreetAddress)),
                );
            }
            Some(entity_type) => must_mask.push(MustMask::typed(value.clone(), entity_type)),
            None => must_mask.push(MustMask::from(value.clone())),
        }
    }

    let chosen: Vec<Slot> = placements.iter().map(|(slot, _)| *slot).collect();

    Ok((
        Case {
            id: format!("inject-{seed}"),
            lang: injections
                .first()
                .map(|injection| injection.lang.clone())
                .unwrap_or_else(|| "en".to_string()),
            len: length_band(&text),
            source: "carrier".to_string(),
            carrier_id: None,
            text,
            must_mask,
            must_clear: Vec::new(),
        },
        chosen,
    ))
}

fn free_slot(from: &[Slot], used: &BTreeSet<usize>, rng: &mut Rng) -> Option<Slot> {
    let free: Vec<Slot> = from
        .iter()
        .copied()
        .filter(|slot| !used.contains(&slot.at))
        .collect();
    rng.pick(&free).copied()
}

fn bare_case(carrier: &str, injections: &[Injection]) -> Case {
    Case {
        id: "inject-rejected".to_string(),
        lang: injections
            .first()
            .map(|injection| injection.lang.clone())
            .unwrap_or_else(|| "en".to_string()),
        len: length_band(carrier),
        source: "carrier".to_string(),
        carrier_id: None,
        text: carrier.to_string(),
        must_mask: Vec::new(),
        must_clear: Vec::new(),
    }
}

/// The `len` band, in code points -- the unit every offset in this workspace
/// is counted in.
fn length_band(text: &str) -> String {
    match text.chars().count() {
        0..=200 => "short",
        201..=800 => "medium",
        _ => "long",
    }
    .to_string()
}

// ---------------------------------------------------------------------------
// The command line: carriers plus a manifest in, one case file out
// ---------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut carriers = PathBuf::from("benchmark/carriers");
    let mut out: Option<PathBuf> = None;
    let mut seed: u64 = 20260821;
    let mut country: Option<String> = None;
    let mut credentials: Option<String> = None;
    let mut licence: Option<String> = None;

    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--carriers" => carriers = PathBuf::from(args.next().ok_or("--carriers needs a path")?),
            "--out" => out = Some(PathBuf::from(args.next().ok_or("--out needs a path")?)),
            "--seed" => seed = args.next().ok_or("--seed needs a number")?.parse()?,
            "--country" => {
                country = Some(args.next().ok_or("--country needs a code")?.to_uppercase())
            }
            "--credentials" => {
                credentials = Some(
                    args.next()
                        .ok_or("--credentials needs a family name")?
                        .to_lowercase(),
                )
            }
            "--licence" => {
                licence = Some(args.next().ok_or("--licence needs a code")?.to_uppercase())
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    let modes = [country.is_some(), credentials.is_some(), licence.is_some()]
        .iter()
        .filter(|chosen| **chosen)
        .count();
    if modes > 1 {
        return Err(
            "--country, --credentials and --licence build different fixtures; pass one".into(),
        );
    }

    let (cases, split) = match (&country, &credentials, &licence) {
        (Some(code), _, _) => build_country(&carriers, code, COUNTRY_PACK_SEED)?,
        (_, Some(family), _) => build_credentials(&carriers, family, CREDENTIAL_SEED)?,
        (_, _, Some(code)) => build_licences(&carriers, code, LICENCE_SEED)?,
        _ => build(&carriers, seed)?,
    };
    let out = out.unwrap_or_else(|| match (&country, &credentials, &licence) {
        (Some(code), _, _) => PathBuf::from(format!(
            "benchmark/cases/country-{}.json",
            code.to_lowercase()
        )),
        (_, Some(family), _) => PathBuf::from(format!("benchmark/cases/credential-{family}.json")),
        (_, _, Some(code)) => PathBuf::from(format!(
            "benchmark/cases/driving-licence-{}.json",
            code.to_lowercase()
        )),
        _ => PathBuf::from("benchmark/cases/carriers.json"),
    });
    let rendered = serde_json::to_string_pretty(&cases)?;
    std::fs::write(&out, format!("{rendered}\n"))?;
    println!("wrote {} cases to {}", cases.len(), out.display());
    println!(
        "placed {} values: {} at a sentence boundary, {} mid-sentence ({:.0}% boundary)",
        split.total(),
        split.boundary,
        split.mid,
        100.0 * split.boundary_share().unwrap_or(0.0)
    );
    Ok(())
}

/// One carrier's provenance and screening record, as
/// `benchmark/carriers/manifest.json` holds it.
#[derive(Debug, serde::Deserialize)]
struct Carrier {
    id: String,
    lang: String,
    register: String,
    file: String,
    url: String,
    licence: String,
    #[serde(default)]
    replacements: Vec<Replacement>,
    #[serde(default)]
    must_clear: Vec<String>,
}

/// A proper noun the carrier already contained, replaced by a synthetic
/// equivalent during screening. The original is deliberately not recorded
/// anywhere -- writing it down would put the real personal data back in the
/// repository, which is the thing screening exists to prevent.
///
/// `type` present means the replacement is personal data and belongs in
/// `must_mask`; `type` absent means it is an **eponym** -- a surname that has
/// become the name of a device or a technical term -- and belongs in
/// `must_clear`. The distinction is the fixture's one rule about surnames,
/// and it has to be stated once rather than decided per carrier: an earlier
/// version had `en-wp-windturbine` demand two eponyms be left alone while
/// `fr-wp-eolienne` demanded an eponym be masked, so no detector could
/// satisfy both and one was always scored wrong. Masking an eponym destroys
/// a technical term and leaks nothing, which is the retired `LOCATION`
/// failure exactly, so `must_clear` is the side the rule falls on.
#[derive(Debug, serde::Deserialize)]
struct Replacement {
    value: String,
    #[serde(default, rename = "type")]
    entity_type: Option<EntityType>,
}

#[derive(Debug, serde::Deserialize)]
struct Manifest {
    carriers: Vec<Carrier>,
}

/// The entity types injected into a carrier of each language.
///
/// `NATIONAL_ID` appears only for Dutch and English carriers and `TAX_ID`
/// only for German and English ones, because the Dutch BSN and the German
/// Steuer-ID are the two country packs that exist. An English-language
/// document quoting a Dutch BSN is not a contrivance -- it is the ticket a
/// Dutch company's English-speaking customer sends.
fn country_identifiers(lang: &str) -> Vec<Injection> {
    let mut identifiers = Vec::new();
    if matches!(lang, "nl" | "en") {
        identifiers.push(Injection::national_id("NL").in_lang(lang));
    }
    if matches!(lang, "de" | "en") {
        identifiers.push(Injection::tax_id("DE").in_lang(lang));
    }
    identifiers
}

/// The types every carrier can take, whatever its language.
fn general_palette(lang: &str) -> Vec<Injection> {
    let country = match lang {
        "nl" => "NL",
        "de" => "DE",
        "fr" => "FR",
        "es" => "ES",
        "it" => "IT",
        _ => "GB",
    };
    vec![
        Injection::email().in_lang(lang),
        Injection::phone(country).in_lang(lang),
        Injection::iban(country).in_lang(lang),
        Injection::ip_address().in_lang(lang),
        Injection::credit_card().in_lang(lang),
        Injection::vat(country).in_lang(lang),
        Injection::passport().in_lang(lang),
        Injection::person().in_lang(lang),
        Injection::street_address().in_lang(lang),
    ]
}

/// What goes into a carrier at each density.
///
/// The country identifier a language has goes into **every** multi-value
/// case for that language rather than taking its turn in the rotation. Left
/// to the rotation it appeared seven times across the whole fixture, and a
/// type gated at 98% recall on seven values moves 14 points per miss: a
/// figure too coarse to gate on.
fn plan(lang: &str, position: usize, density: &str) -> Vec<Injection> {
    let identifiers = country_identifiers(lang);
    let general = general_palette(lang);
    match density {
        "d0" => Vec::new(),
        "d1" => {
            // Alternate, so the country identifier is not confined to the
            // multi-value cases and single-value recall sees it too.
            if !identifiers.is_empty() && position.is_multiple_of(2) {
                vec![identifiers[(position / 2) % identifiers.len()].clone()]
            } else {
                vec![general[position % general.len()].clone()]
            }
        }
        _ => {
            let mut chosen = identifiers;
            for step in 0..3 {
                chosen.push(general[(position + step * 3) % general.len()].clone());
            }
            chosen
        }
    }
}

// ---------------------------------------------------------------------------
// Country packs: one extra case per carrier, per country
// ---------------------------------------------------------------------------

/// The seed the committed country-pack case files were built with. Fixed
/// here rather than taken from `--seed` so that adding a country cannot
/// silently re-draw the countries already committed.
const COUNTRY_PACK_SEED: u64 = 20260827;

/// The country packs of this phase, in the order the English carriers are
/// shared out between them.
///
/// English carriers are a shared resource: a document in English quoting a
/// Spanish DNI is a real thing (it is the ticket a Spanish company's
/// English-speaking customer sends) and so is one quoting an Italian codice
/// fiscale, but one English carrier hosts at most one country-pack case, so
/// they are dealt round-robin. Fixed order, so a country's share does not
/// move when another country is added.
const COUNTRY_PACKS: [&str; 4] = ["ES", "IT", "FR", "BE"];

/// Which carriers a country pack's cases are built from, and which
/// identifiers rotate through them.
struct CountryPack {
    /// The identifier keys ([`Kind::NationalId`]'s or [`Kind::TaxId`]'s),
    /// rotated by the carrier's position among the carriers this pack uses.
    /// Spain has two because Spain issues two.
    identifiers: &'static [Identifier],
    /// The carrier languages this pack's cases are written in. A carrier
    /// belongs to exactly one pack, which is what keeps case ids unique and
    /// keeps one country from dominating the `NATIONAL_ID` denominator.
    languages: &'static [&'static str],
}

/// One injectable country identifier: its key, and which of the two
/// key-carrying kinds it is.
#[derive(Clone, Copy)]
struct Identifier {
    key: &'static str,
    tax: bool,
}

impl Identifier {
    const fn national(key: &'static str) -> Self {
        Self { key, tax: false }
    }

    const fn tax(key: &'static str) -> Self {
        Self { key, tax: true }
    }

    fn injection(&self, lang: &str) -> Injection {
        if self.tax {
            Injection::tax_id(self.key).in_lang(lang)
        } else {
            Injection::national_id(self.key).in_lang(lang)
        }
    }
}

/// A country's pack, or `None` when this build has none for it. A country
/// named in [`COUNTRY_PACKS`] but absent here simply has no cases yet -- the
/// packs land one commit at a time and the English rotation is fixed in
/// advance so that the shares already committed do not move.
fn country_pack(country: &str) -> Option<CountryPack> {
    const ES: [Identifier; 2] = [
        Identifier::national("ES-DNI"),
        Identifier::national("ES-NIE"),
    ];
    const IT: [Identifier; 1] = [Identifier::tax("IT")];
    const FR: [Identifier; 1] = [Identifier::national("FR")];
    const BE: [Identifier; 1] = [Identifier::national("BE")];
    match country {
        "ES" => Some(CountryPack {
            identifiers: &ES,
            languages: &["es"],
        }),
        "IT" => Some(CountryPack {
            identifiers: &IT,
            languages: &["it"],
        }),
        "FR" => Some(CountryPack {
            identifiers: &FR,
            languages: &["fr"],
        }),
        // Belgium's own languages are Dutch and French, and French is
        // already France's. A carrier belongs to exactly one pack, so
        // Belgium takes the Dutch carriers; Belgian French is covered by
        // hand-authored cases instead. Giving Belgium the French carriers
        // as well would have left France with none.
        "BE" => Some(CountryPack {
            identifiers: &BE,
            languages: &["nl"],
        }),
        _ => None,
    }
}

/// Every carrier this country pack builds a case from, in manifest order.
///
/// A carrier in one of the pack's own languages always belongs to it. An
/// English carrier belongs to whichever pack its turn in [`COUNTRY_PACKS`]
/// names, counting English carriers in manifest order.
fn carriers_for<'a>(manifest: &'a Manifest, country: &str, pack: &CountryPack) -> Vec<&'a Carrier> {
    let mut english_seen = 0usize;
    let mut chosen = Vec::new();
    for carrier in &manifest.carriers {
        if carrier.lang == "en" {
            let turn = COUNTRY_PACKS[english_seen % COUNTRY_PACKS.len()];
            english_seen += 1;
            if turn == country {
                chosen.push(carrier);
            }
        } else if pack.languages.contains(&carrier.lang.as_str()) {
            chosen.push(carrier);
        }
    }
    chosen
}

/// One case per carrier for one country pack: the country's identifier,
/// injected into real carrier prose, in the carrier's own language.
///
/// Shares every guard `build` applies -- provenance, screening replacements,
/// `must_clear` presence and non-overlap -- because it calls the same code
/// paths. What differs is the plan (one country identifier, not a rotation
/// of the general palette), the seed, and the case id.
fn build_country(
    dir: &Path,
    country: &str,
    seed: u64,
) -> Result<(Vec<Case>, Split), Box<dyn std::error::Error>> {
    let pack = country_pack(country)
        .ok_or_else(|| format!("{country}: no country pack is built into this binary"))?;
    let manifest: Manifest =
        serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json"))?)?;
    let mut cases = Vec::new();
    let mut split = Split::default();

    for (position, carrier) in carriers_for(&manifest, country, &pack)
        .into_iter()
        .enumerate()
    {
        check_provenance(carrier)?;
        let identifier = pack.identifiers[position % pack.identifiers.len()];
        let text = std::fs::read_to_string(dir.join(&carrier.file))?;
        let carrier_seed = seed
            .wrapping_mul(1_000_003)
            .wrapping_add(position as u64 * 7919);

        let (mut case, placed) = build_case(
            &text,
            &[identifier.injection(&carrier.lang)],
            carrier_seed ^ u64::from(country.bytes().fold(0u32, |a, b| a * 31 + u32::from(b))),
        )
        .map_err(|error| format!("{} ({country}): {error}", carrier.id))?;
        split.count(&placed);

        case.id = format!("{}-cp", carrier.id);
        case.lang = carrier.lang.clone();
        case.carrier_id = Some(carrier.id.clone());
        case.source = format!("country-pack:{}", carrier.register);
        apply_screening(carrier, &mut case)?;
        cases.push(case);
    }

    Ok((cases, split))
}

// ---------------------------------------------------------------------------
// Credential families: one extra case per fifth carrier, per family
// ---------------------------------------------------------------------------

/// The seed the committed credential case files were built with. Fixed here
/// for the same reason [`COUNTRY_PACK_SEED`] is: adding a family must not
/// silently re-draw the families already committed.
const CREDENTIAL_SEED: u64 = 20260828;

/// How the carriers are shared out between the credential families. One
/// carrier in five goes to each family, by its position in the manifest, so
/// the three families draw from disjoint carriers and each gets about 64 of
/// the 321. Fixed, so adding a family does not move the carriers an existing
/// one already used.
const CREDENTIAL_FAMILY_STRIDE: usize = 5;

/// One credential family: a case file, a commit, and a set of formats to
/// rotate through the carriers it owns.
struct CredentialFamily {
    /// Which carriers this family draws: those whose manifest position is
    /// congruent to this modulo [`CREDENTIAL_FAMILY_STRIDE`].
    slot: usize,
    /// The credential formats, rotated by the carrier's position among this
    /// family's own carriers.
    formats: &'static [&'static str],
    /// The hard negatives, rotated the same way. Every one is a shape that
    /// looks like a credential and is not, injected into the same case as
    /// the credential and carried out as `must_clear` -- so every case in
    /// this fixture measures a miss and an over-mask at once.
    negatives: &'static [&'static str],
}

/// The hard negatives every family carries. Each is a real shape from the
/// text this service sits in front of -- stack traces, CI logs, container
/// manifests, config pastes -- and each is the shape a looser pattern would
/// start masking first. The phone recognizer's over-masking was found
/// exactly this way, by writing down what must stay clear before measuring.
const CREDENTIAL_NEGATIVES: [&str; 5] = [
    "git-sha",
    "uuid",
    "base64-image",
    "aws-secret",
    "docker-digest",
];

fn credential_family(name: &str) -> Option<CredentialFamily> {
    match name {
        // Family 1: documented vendor prefixes.
        "apikey" => Some(CredentialFamily {
            slot: 0,
            formats: &CREDENTIAL_FORMATS,
            negatives: &CREDENTIAL_NEGATIVES,
        }),
        // Family 2: JWTs and bearer tokens. `not-a-jwt` joins the shared
        // negatives: three dot-separated base64url segments whose header is
        // not a JSON object, which is what the `eyJ` anchor exists to
        // exclude and the only negative that tests it.
        "jwt" => Some(CredentialFamily {
            slot: 1,
            formats: &CREDENTIAL_TOKEN_FORMATS,
            negatives: &[
                "not-a-jwt",
                "git-sha",
                "uuid",
                "base64-image",
                "docker-digest",
            ],
        }),
        // Family 3: PEM blocks. `pem-public-key` joins the shared negatives:
        // a public key is not a secret, masking one destroys a diagnosable
        // config paste, and it is one word away from the armour line the
        // recognizer keys on.
        "pem" => Some(CredentialFamily {
            slot: 2,
            formats: &CREDENTIAL_PEM_FORMATS,
            negatives: &[
                "pem-public-key",
                "git-sha",
                "uuid",
                "base64-image",
                "docker-digest",
            ],
        }),
        _ => None,
    }
}

/// One hard negative of the named kind. Drawn rather than written down, so
/// no two cases carry the same one and a detector cannot pass by learning a
/// literal.
fn credential_negative(rng: &mut Rng, kind: &str) -> String {
    let hex = |rng: &mut Rng, count: usize| from_alphabet(rng, "0123456789abcdef", count);
    match kind {
        // A git commit SHA: forty hex characters, and the single most common
        // credential-shaped string in a developer's paste.
        "git-sha" => hex(rng, 40),
        "uuid" => format!(
            "{}-{}-4{}-{}{}-{}",
            hex(rng, 8),
            hex(rng, 4),
            hex(rng, 3),
            ["8", "9", "a", "b"][rng.below(4) as usize],
            hex(rng, 3),
            hex(rng, 12)
        ),
        // A base64 image fragment: an unbroken run of the same alphabet
        // every token in this family draws from, in the one place a reader
        // would never want it masked.
        "base64-image" => format!(
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA{}",
            from_alphabet(rng, BASE64, 96)
        ),
        // The forty-character half of an AWS credential pair that has no
        // prefix, and is therefore deliberately not covered. It is here as a
        // negative precisely because covering it is the tempting mistake.
        "aws-secret" => from_alphabet(rng, BASE64, 40),
        "docker-digest" => format!("sha256:{}", hex(rng, 64)),
        // Three dot-separated base64url segments whose first does not decode
        // to a JSON object. The `eyJ` anchor is the whole reason the JWT
        // pattern is not "three base64 runs and two dots".
        "not-a-jwt" => format!(
            "{}.{}.{}",
            from_alphabet(rng, BASE64URL, 24),
            from_alphabet(rng, BASE64URL, 32),
            from_alphabet(rng, BASE64URL, 43)
        ),
        // A public key in the same armour a private key wears. One word
        // apart, and the opposite answer.
        "pem-public-key" => pem_block(rng, "PUBLIC KEY", 3),
        other => panic!("no generator for credential negative {other}"),
    }
}

/// One case per carrier this family owns: the credential, injected into real
/// carrier prose in the carrier's own language, plus one hard negative in
/// the same text.
///
/// Shares every guard [`build`] applies -- provenance, screening
/// replacements, `must_clear` presence and non-overlap -- because it calls
/// the same code paths. What differs is the plan, the seed, and the case id.
///
/// THE NEGATIVE RIDES IN AS AN INJECTION AND LEAVES AS A `must_clear`. It is
/// placed by [`build_case`] like everything else, so its position is known
/// and the carrier is spliced rather than rewritten; it is then moved out of
/// `must_mask` on the one property that distinguishes it, which is that
/// [`Kind::Exact`] declares no entity type. Building it any other way would
/// mean a second placement mechanism with its own subtly different notion of
/// where a value can go.
fn build_credentials(
    dir: &Path,
    family_name: &str,
    seed: u64,
) -> Result<(Vec<Case>, Split), Box<dyn std::error::Error>> {
    let family = credential_family(family_name)
        .ok_or_else(|| format!("{family_name}: no credential family by that name"))?;
    let manifest: Manifest =
        serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json"))?)?;
    let mut cases = Vec::new();
    let mut split = Split::default();

    let owned: Vec<&Carrier> = manifest
        .carriers
        .iter()
        .enumerate()
        .filter(|(position, _)| position % CREDENTIAL_FAMILY_STRIDE == family.slot)
        .map(|(_, carrier)| carrier)
        .collect();

    for (position, carrier) in owned.into_iter().enumerate() {
        check_provenance(carrier)?;
        let text = std::fs::read_to_string(dir.join(&carrier.file))?;
        let carrier_seed = seed
            .wrapping_mul(1_000_003)
            .wrapping_add(position as u64 * 7919)
            ^ u64::from(family.slot as u32);

        // A stream of its own for the negative, so that adding or reordering
        // the negatives cannot perturb the credential values themselves.
        let mut negative_rng = Rng::new(carrier_seed ^ 0x_C0DE_1234_5678_9ABC);
        let negative = credential_negative(
            &mut negative_rng,
            family.negatives[position % family.negatives.len()],
        );

        let plan = [
            Injection::credential(family.formats[position % family.formats.len()])
                .in_lang(&carrier.lang),
            Injection::exact(&negative).in_lang(&carrier.lang),
        ];

        let (mut case, placed) = build_case(&text, &plan, carrier_seed)
            .map_err(|error| format!("{} ({family_name}): {error}", carrier.id))?;
        split.count(&placed);

        case.id = format!("{}-cred-{family_name}", carrier.id);
        case.lang = carrier.lang.clone();
        case.carrier_id = Some(carrier.id.clone());
        case.source = format!("credential:{}", carrier.register);
        apply_screening(carrier, &mut case)?;
        move_negative_to_must_clear(&mut case, &negative)?;
        cases.push(case);
    }

    Ok((cases, split))
}

// ---------------------------------------------------------------------------
// Driving licences: one extra case per third carrier, per country
// ---------------------------------------------------------------------------

/// The seed the committed driving-licence case files were built with. Fixed
/// here for the same reason [`COUNTRY_PACK_SEED`] and [`CREDENTIAL_SEED`]
/// are: adding a country must not silently re-draw the one already
/// committed.
const LICENCE_SEED: u64 = 20260912;

/// How the carriers are shared out between the two licence countries. One
/// carrier in three goes to each, by its position in the manifest, so
/// Germany and the United Kingdom draw from disjoint carriers and each gets
/// 107 of the 321.
///
/// WHY A STRIDE RATHER THAN THE CARRIER'S OWN LANGUAGE. The country packs
/// give each country its own languages, because a Spanish DNI is quoted in
/// Spanish prose far more often than in any other. A driving licence is not
/// like that: the text this service sits in front of is cross-border
/// correspondence, and a German licence quoted in an English or Italian
/// ticket is the ordinary case rather than the exotic one. A stride puts
/// both countries in all six languages, which is also what keeps the
/// per-language recall floor meaningful for this type.
const LICENCE_STRIDE: usize = 3;

/// How the one stride slot `fa98768` left free is shared out again.
///
/// Germany took slot 0 and the United Kingdom slot 1, which left the 107
/// carriers in slot 2 unused. The countries added since -- the ones with no
/// structure to be decided by -- divide that slot between them rather than
/// changing [`LICENCE_STRIDE`], because changing the stride would move which
/// carriers Germany and the United Kingdom own and silently re-draw two
/// committed case files. Every country here therefore still appears in all
/// six carrier languages, and still draws from carriers no other licence
/// country draws.
const LICENCE_SUB_STRIDE: usize = 3;

/// One country's licence fixture: a case file, a commit, and the hard
/// negatives that ride along with it.
struct LicencePack {
    /// Which carriers this country draws: those whose manifest position is
    /// congruent to this modulo [`LICENCE_STRIDE`].
    slot: usize,
    /// For a country that shares its slot, which share of it: the carrier's
    /// position *among that slot's own carriers* modulo
    /// [`LICENCE_SUB_STRIDE`]. `None` means the country takes the whole slot,
    /// which is what Germany and the United Kingdom do and must keep doing.
    sub_slot: Option<usize>,
    /// The hard negatives, rotated by the carrier's position among this
    /// country's own carriers. Every case therefore measures a miss and an
    /// over-mask at once.
    negatives: &'static [&'static str],
}

/// Germany's hard negatives.
///
/// `eleven-digits` is the one that matters and it is the reason this fixture
/// exists at all. Relaxing `de_fuehrerschein`'s first character from a letter
/// to any alphanumeric -- which is what Microsoft Purview's own pattern for
/// this identifier does -- admits the bare eleven-digit run, and roughly one
/// run in eleven clears a modulo-11 check by chance. These are eleven-digit
/// runs that are neither a German Steuer-ID nor a Belgian national register
/// number, so nothing else in the packs may mask them either; whether any
/// given one would have cleared the relaxed licence reading is left to
/// chance rather than constructed, because constructing it would mean
/// transcribing the checksum rule into this file.
const DE_LICENCE_NEGATIVES: [&str; 3] = ["eleven-digits", "licence-bad-check", "reference-twelve"];

/// The United Kingdom's hard negatives: the DVLA shape with a date that is
/// not a date, the same shape one character short, and an ordinary
/// sixteen-character booking reference -- the commonest string of exactly
/// this length in the correspondence this service sees.
const GB_LICENCE_NEGATIVES: [&str; 3] = ["dvla-impossible-date", "dvla-short", "booking-reference"];

/// The Netherlands' hard negatives, and the first one is the whole reason
/// this country is detectable at all.
///
/// `order-ten` is a ten-digit run -- the licence's exact shape, indivisible
/// from it by any pattern -- introduced by prose that calls it an order
/// number, and injected into a carrier whose other injected value is a
/// licence with the word `rijbewijs` beside it. That is the brief's required
/// negative: an order number and a licence word in the same text, where only
/// the suppression vocabulary can tell them apart. `bsn-shaped` is nine
/// digits that fail the eleven-proof, and `eleven-digits` is the shared
/// German generator's run of eleven -- the two shapes that sit one digit
/// either side of this recognizer's, which is where a loosened length bound
/// would show up first.
const NL_LICENCE_NEGATIVES: [&str; 3] = ["order-ten", "bsn-shaped", "eleven-digits"];

/// France's hard negatives, on the same plan as the Dutch ones.
///
/// `order-twelve` is the NEPH's exact shape -- twelve digits whose third and
/// fourth are a month -- introduced as an order number in a clause that also
/// names a driving licence, so only the suppression vocabulary can reject
/// it. `neph-impossible-month` is the same twelve digits with month 13 or 00,
/// which the pattern rejects on shape, and `reference-thirteen` is thirteen
/// digits: one more than a NEPH and exactly an EAN-13 barcode, which is the
/// commonest thirteen-digit run in commercial correspondence.
const FR_LICENCE_NEGATIVES: [&str; 3] = [
    "order-twelve",
    "neph-impossible-month",
    "reference-thirteen",
];

/// The negatives whose *shape* a licence recognizer accepts, and which are
/// therefore rejected only by the words around them.
///
/// These are the ones introduced as an order number rather than as a neutral
/// reference; see [`Kind::OrderReference`]. Germany's and the United
/// Kingdom's negatives are absent on purpose -- `licence-bad-check` fails a
/// modulo-11 check and `dvla-impossible-date` fails a range-checked month,
/// so no vocabulary is load-bearing for either, and changing how they are
/// introduced would re-draw two committed files for no gain.
const LICENCE_SHAPED_NEGATIVES: [&str; 3] = ["order-ten", "order-twelve", "order-patente"];

/// Italy's hard negatives, and the middle one is the measurement this
/// country's recognizer exists to beat.
///
/// `order-patente` is the licence's own province shape introduced as an
/// order number in a clause that also names a driving licence.
/// `uppercase-word` is an ordinary ten-letter upper-case Italian or German
/// word -- the population `training/reports/driving-licence.md` §3.2 measured
/// at 345 of 350 hits for Microsoft Purview's pattern for this identifier,
/// carried into the benchmark so the finding is a case and not only a
/// paragraph. `patente-wrong-length` is the same province shape with eight
/// digits where a licence has seven.
const IT_LICENCE_NEGATIVES: [&str; 3] = ["order-patente", "uppercase-word", "patente-wrong-length"];

/// Ten-letter upper-case words Purview's Italian licence pattern matches and
/// `it_patente` cannot.
///
/// Four are from the §4 histogram over `training/data-negatives` and four
/// are ordinary Italian words of exactly ten letters that a document about a
/// driving licence plausibly contains in upper case -- which is the harder
/// case, because they sit inside this recognizer's own context window.
const IT_TEN_LETTER_WORDS: [&str; 8] = [
    "PARLAMENTO",
    "VERORDNUNG",
    "COMMISSION",
    "PARLAMENTS",
    "CONDUCENTE",
    "AUTOSCUOLA",
    "PRESIDENTE",
    "RILASCIATA",
];

fn licence_pack(country: &str) -> Option<LicencePack> {
    match country {
        "DE" => Some(LicencePack {
            slot: 0,
            sub_slot: None,
            negatives: &DE_LICENCE_NEGATIVES,
        }),
        "GB" => Some(LicencePack {
            slot: 1,
            sub_slot: None,
            negatives: &GB_LICENCE_NEGATIVES,
        }),
        "NL" => Some(LicencePack {
            slot: 2,
            sub_slot: Some(0),
            negatives: &NL_LICENCE_NEGATIVES,
        }),
        "FR" => Some(LicencePack {
            slot: 2,
            sub_slot: Some(1),
            negatives: &FR_LICENCE_NEGATIVES,
        }),
        "IT" => Some(LicencePack {
            slot: 2,
            sub_slot: Some(2),
            negatives: &IT_LICENCE_NEGATIVES,
        }),
        _ => None,
    }
}

/// One hard negative of the named kind, drawn rather than written down so no
/// two cases carry the same one.
///
/// Every branch that could collide with another recognizer asks that
/// recognizer's own validator rather than reasoning about it: the
/// eleven-digit run is re-drawn until neither `german_tax_id` nor
/// `be_national_register` accepts it, and the bad-check licence is re-drawn
/// until `de_fuehrerschein` rejects it. Nothing here restates a checksum.
fn licence_negative(rng: &mut Rng, kind: &str) -> String {
    match kind {
        "eleven-digits" => {
            for _ in 0..64 {
                let candidate = format!("{}{}", 1 + rng.below(9), rng.digits(10));
                if !Validator::GermanTaxId.accepts(&candidate)
                    && !Validator::BeNationalRegister.accepts(&candidate)
                {
                    return candidate;
                }
            }
            // Eleven digits that no committed validator accepts: the
            // Steuer-ID check fails on a body of all ones, and the Belgian
            // control is not 11 either.
            "11111111111".to_string()
        }
        "licence-bad-check" => {
            for _ in 0..64 {
                let valid = de_fuehrerschein(rng);
                let mutated: String = valid
                    .chars()
                    .enumerate()
                    .map(|(index, character)| {
                        if index == 9 {
                            // One away, wrapping, so the mutation is always a
                            // change and always still a digit.
                            char::from_digit((character.to_digit(10).unwrap_or(0) + 1) % 10, 10)
                                .unwrap_or('0')
                        } else {
                            character
                        }
                    })
                    .collect();
                if !Validator::DeFuehrerschein.accepts(&mutated) {
                    return mutated;
                }
            }
            "B072RRE2I45".to_string()
        }
        // Twelve characters where a licence has eleven: the length bound in
        // the pattern is the only thing that rejects it.
        "reference-twelve" => format!(
            "{}{}{}",
            rng.upper(),
            rng.digits(2),
            (0..9)
                .map(|_| if rng.below(3) == 0 {
                    rng.upper()
                } else {
                    rng.digit()
                })
                .collect::<String>()
        ),
        "dvla-impossible-date" => {
            let valid = gb_driving_licence(rng);
            // Month 13 in its male and female writings, and day 00 and 32.
            // Each is exactly two characters away from a real licence.
            let (at, replacement) = match rng.below(4) {
                0 => (6, "13"),
                1 => (6, "63"),
                2 => (8, "00"),
                _ => (8, "32"),
            };
            let characters: Vec<char> = valid.chars().collect();
            characters
                .iter()
                .enumerate()
                .map(|(index, character)| {
                    if index >= at && index < at + 2 {
                        replacement.chars().nth(index - at).unwrap_or(*character)
                    } else {
                        *character
                    }
                })
                .collect()
        }
        // Ten digits: the Dutch licence's exact shape, and rejected only by
        // the word `ordernummer` the injector writes in front of it. The
        // first digit is never zero for the reason [`nl_rijbewijs`] gives --
        // a leading zero would make this a Dutch telephone number, which the
        // phone recognizer masks at 0.95 and no licence vocabulary can
        // argue with, so the case would be measuring the wrong recognizer.
        "order-ten" => nl_rijbewijs(rng),
        // Nine digits that are not a BSN: one digit short of the licence and
        // one digit short of nothing else. Re-drawn until the eleven-proof
        // rejects it, by asking the validator rather than restating it, so
        // the case is not demanding that a real BSN stay unmasked.
        "bsn-shaped" => {
            for _ in 0..64 {
                let candidate = format!("{}{}", 1 + rng.below(9), rng.digits(8));
                if !Validator::Elfproef.accepts(&candidate) {
                    return candidate;
                }
            }
            "111111111".to_string()
        }
        // Twelve digits with a month in the third and fourth: the NEPH's
        // exact shape, rejected only by the word `numéro de commande` the
        // injector writes in front of it. Never a year of 00 to 09, for the
        // reason [`fr_permis_conduire`] gives.
        "order-twelve" => fr_permis_conduire(rng),
        // The same twelve digits with a month the calendar does not have.
        // Drawn rather than written down, and both impossible months are in
        // the draw space: 13 (one past December) and 00 (one before
        // January), which are the two ends the pattern's alternation has to
        // close.
        "neph-impossible-month" => {
            let valid = fr_permis_conduire(rng);
            let month = if rng.below(2) == 0 { "13" } else { "00" };
            valid
                .chars()
                .enumerate()
                .map(|(index, character)| {
                    if index == 2 || index == 3 {
                        month.chars().nth(index - 2).unwrap_or(character)
                    } else {
                        character
                    }
                })
                .collect()
        }
        // Thirteen digits: one more than a NEPH, and an EAN-13 barcode.
        "reference-thirteen" => format!("{}{}", 1 + rng.below(9), rng.digits(12)),
        // The Italian licence's province shape, rejected only by the word
        // `numero d'ordine` the injector writes in front of it.
        "order-patente" => it_patente(rng),
        // An ordinary ten-letter upper-case word: the 345-of-350 population
        // §3.2 measured for Purview's pattern for this identifier.
        "uppercase-word" => rng
            .pick(&IT_TEN_LETTER_WORDS)
            .copied()
            .unwrap_or("PARLAMENTO")
            .to_string(),
        // Eight digits where an Italian licence has seven, which is the
        // length bound in the pattern and nothing else.
        "patente-wrong-length" => format!(
            "{}{}{}",
            rng.pick(&IT_PROVINCE_CODES).copied().unwrap_or("MI"),
            rng.digits(8),
            rng.upper()
        ),
        // Fifteen characters where a driver number has sixteen.
        "dvla-short" => gb_driving_licence(rng).chars().take(15).collect(),
        // Sixteen characters, and not a driver number: letters and digits
        // alternate, so the six-digit date field never forms.
        "booking-reference" => (0..16)
            .map(|index| {
                if index % 2 == 0 {
                    rng.upper()
                } else {
                    rng.digit()
                }
            })
            .collect(),
        other => panic!("no generator for licence negative {other}"),
    }
}

/// One case per carrier this country owns: the licence, injected into real
/// carrier prose in the carrier's own language, plus one hard negative in the
/// same text.
///
/// Shares every guard [`build`] applies -- provenance, screening
/// replacements, `must_clear` presence and non-overlap -- because it calls
/// the same code paths, exactly as [`build_credentials`] does. What differs
/// is the plan, the seed, and the case id.
fn build_licences(
    dir: &Path,
    country: &str,
    seed: u64,
) -> Result<(Vec<Case>, Split), Box<dyn std::error::Error>> {
    let pack = licence_pack(country)
        .ok_or_else(|| format!("{country}: no driving-licence pack is built into this binary"))?;
    let manifest: Manifest =
        serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json"))?)?;
    let mut cases = Vec::new();
    let mut split = Split::default();

    // Two filters, applied in this order and never collapsed into one: the
    // stride picks the slot, and only then does the sub-stride divide that
    // slot's own carriers. Collapsing them into a single modulus over the
    // manifest position would move Germany's and the United Kingdom's
    // carriers and re-draw two committed files.
    let owned: Vec<&Carrier> = manifest
        .carriers
        .iter()
        .enumerate()
        .filter(|(position, _)| position % LICENCE_STRIDE == pack.slot)
        .map(|(_, carrier)| carrier)
        .enumerate()
        .filter(|(within_slot, _)| match pack.sub_slot {
            None => true,
            Some(share) => within_slot % LICENCE_SUB_STRIDE == share,
        })
        .map(|(_, carrier)| carrier)
        .collect();

    for (position, carrier) in owned.into_iter().enumerate() {
        check_provenance(carrier)?;
        let text = std::fs::read_to_string(dir.join(&carrier.file))?;
        let carrier_seed = seed
            .wrapping_mul(1_000_003)
            .wrapping_add(position as u64 * 7919)
            ^ u64::from(
                country
                    .bytes()
                    .fold(0u32, |a, b| a.wrapping_mul(31).wrapping_add(u32::from(b))),
            );

        // A stream of its own for the negative, so that adding or reordering
        // the negatives cannot perturb the licence values themselves.
        let mut negative_rng = Rng::new(carrier_seed ^ 0x_D1CE_4321_8765_BA98);
        let kind = pack.negatives[position % pack.negatives.len()];
        let negative = licence_negative(&mut negative_rng, kind);

        // A negative that is licence-shaped is introduced as an order
        // number; one that is rejected by its shape alone keeps the neutral
        // "the reference is". See [`Kind::OrderReference`] for why that is
        // the difference that matters, and [`NL_LICENCE_NEGATIVES`] for
        // which negatives are which.
        let negative_injection = if LICENCE_SHAPED_NEGATIVES.contains(&kind) {
            Injection::order_reference(&negative)
        } else {
            Injection::exact(&negative)
        };
        let plan = [
            Injection::driving_licence(country).in_lang(&carrier.lang),
            negative_injection.in_lang(&carrier.lang),
        ];

        let (mut case, placed) = build_case(&text, &plan, carrier_seed)
            .map_err(|error| format!("{} ({country}): {error}", carrier.id))?;
        split.count(&placed);

        case.id = format!("{}-dl-{}", carrier.id, country.to_lowercase());
        case.lang = carrier.lang.clone();
        case.carrier_id = Some(carrier.id.clone());
        case.source = format!("driving-licence:{}", carrier.register);
        apply_screening(carrier, &mut case)?;
        move_negative_to_must_clear(&mut case, &negative)?;
        cases.push(case);
    }

    Ok((cases, split))
}

/// Moves the injected hard negative out of `must_mask` and into
/// `must_clear`, after [`apply_screening`] has rebuilt that list from the
/// carrier's own.
///
/// Fails rather than fixes up. A negative that is not in the text, or that
/// overlaps the credential injected beside it, produces a case no detector
/// output can satisfy -- the shape `check_coherence` exists to catch, caught
/// one step earlier where the file can still not be written.
fn move_negative_to_must_clear(
    case: &mut Case,
    negative: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let before = case.must_mask.len();
    case.must_mask
        .retain(|entry| entry.entity_type().is_some() || entry.as_str() != negative);
    if case.must_mask.len() != before - 1 {
        return Err(format!(
            "{}: the injected negative was not the one untyped must_mask entry",
            case.id
        )
        .into());
    }
    if !case.text.contains(negative) {
        return Err(format!("{}: the injected negative is not in the case text", case.id).into());
    }
    if case
        .must_mask
        .iter()
        .any(|mask| mask.as_str().contains(negative) || negative.contains(mask.as_str()))
    {
        return Err(format!("{}: the injected negative overlaps a masked value", case.id).into());
    }
    case.must_clear.push(negative.to_string());
    Ok(())
}

/// A carrier whose licence and source are not on record does not go into the
/// corpus. Checked at build time rather than only in a test, so no fixture
/// can be regenerated around a missing one.
fn check_provenance(carrier: &Carrier) -> Result<(), Box<dyn std::error::Error>> {
    if carrier.url.trim().is_empty() || carrier.licence.trim().is_empty() {
        return Err(format!(
            "{}: no URL or no licence recorded; a carrier of unestablished provenance is a liability, not an asset",
            carrier.id
        )
        .into());
    }
    Ok(())
}

/// The screening replacements and `must_clear` values a carrier carries,
/// applied to one case built from it.
///
/// The screening replacements stand where the carrier's own proper nouns
/// did, in synthetic form and at a known position, so they belong in the
/// answer key of every case built from it -- on the side `Replacement`'s doc
/// comment sets out.
fn apply_screening(carrier: &Carrier, case: &mut Case) -> Result<(), Box<dyn std::error::Error>> {
    let mut clears = carrier.must_clear.clone();
    for replacement in &carrier.replacements {
        if !case.text.contains(&replacement.value) {
            return Err(format!(
                "{}: a screening replacement is not present in the carrier text",
                carrier.id
            )
            .into());
        }
        match replacement.entity_type {
            Some(entity_type) => case
                .must_mask
                .push(MustMask::typed(replacement.value.clone(), entity_type)),
            None => {
                if !clears.contains(&replacement.value) {
                    clears.push(replacement.value.clone());
                }
            }
        }
    }

    for clear in &clears {
        if !case.text.contains(clear) {
            return Err(
                format!("{}: a must_clear value is not in the case text", carrier.id).into(),
            );
        }
        if case
            .must_mask
            .iter()
            .any(|mask| mask.as_str().contains(clear.as_str()) || clear.contains(mask.as_str()))
        {
            return Err(format!(
                "{}: a must_clear value overlaps an injected value",
                carrier.id
            )
            .into());
        }
    }
    case.must_clear = clears;
    Ok(())
}

fn build(dir: &Path, seed: u64) -> Result<(Vec<Case>, Split), Box<dyn std::error::Error>> {
    let manifest: Manifest =
        serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json"))?)?;
    let mut cases = Vec::new();
    let mut split = Split::default();

    for (position, carrier) in manifest.carriers.iter().enumerate() {
        check_provenance(carrier)?;

        let path = dir.join(&carrier.file);
        let text = std::fs::read_to_string(&path)?;
        let carrier_seed = seed
            .wrapping_mul(1_000_003)
            .wrapping_add(position as u64 * 7919);

        for density in ["d0", "d1", "dn"] {
            let chosen = plan(&carrier.lang, position, density);
            let (mut case, placed) =
                build_case(&text, &chosen, carrier_seed ^ (chosen.len() as u64))
                    .map_err(|error| format!("{} ({}): {error}", carrier.id, density))?;
            split.count(&placed);

            case.id = format!("{}-{density}", carrier.id);
            case.lang = carrier.lang.clone();
            case.carrier_id = Some(carrier.id.clone());
            case.source = format!("carrier:{}", carrier.register);

            apply_screening(carrier, &mut case)?;
            cases.push(case);
        }
    }

    Ok((cases, split))
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use pii_detection_eval::check_coherence;

    #[test]
    fn an_injected_value_appears_exactly_where_the_case_says() {
        let carrier = "De overeenkomst is getekend op 3 maart.";
        let case = inject(carrier, &[Injection::national_id("NL")], 0);
        let value = &case.must_mask[0];
        assert!(case.text.contains(value.as_str()));
        // Injection must not disturb the carrier's own text.
        assert!(case.text.contains("De overeenkomst is getekend"));
    }

    #[test]
    fn an_injected_identifier_passes_its_own_checksum() {
        let case = inject("Carrier text.", &[Injection::national_id("NL")], 0);
        // Reuse the service's own validator rather than reimplementing it.
        assert!(pii_detection_rules::validator::Validator::Elfproef.accepts(&case.must_mask[0]));
    }

    #[test]
    fn injection_is_deterministic_for_a_seed() {
        let a = inject("Carrier text.", &[Injection::national_id("NL")], 7);
        let b = inject("Carrier text.", &[Injection::national_id("NL")], 7);
        assert_eq!(a.text, b.text);
    }

    #[test]
    fn a_carrier_already_containing_the_injected_value_is_rejected() {
        // Otherwise must_mask would name a substring that appears twice, and the
        // case would silently demand masking of the carrier's own text.
        let carrier = "Referentie 123456782 in het dossier.";
        assert!(inject_checked(carrier, &[Injection::exact("123456782")], 0).is_err());
    }

    // --- beyond the brief -------------------------------------------------

    #[test]
    fn a_different_seed_draws_a_different_value() {
        // Determinism is only worth having if the seed actually does
        // something: a generator that ignored it would pass the determinism
        // test above and produce one identifier for the whole fixture.
        let a = inject("Carrier text.", &[Injection::national_id("NL")], 1);
        let b = inject("Carrier text.", &[Injection::national_id("NL")], 2);
        assert_ne!(a.must_mask[0].as_str(), b.must_mask[0].as_str());
    }

    #[test]
    fn every_injected_value_declares_the_type_it_was_injected_as() {
        // The whole reason the case format grew a per-value type: a value the
        // detector never touches must still be counted against the type it
        // was supposed to be.
        let case = inject(
            "Carrier text. Second sentence here.",
            &[
                Injection::national_id("NL"),
                Injection::email(),
                Injection::iban("NL"),
            ],
            11,
        );
        let declared: Vec<Option<EntityType>> = case
            .must_mask
            .iter()
            .map(|value| value.entity_type())
            .collect();
        assert_eq!(
            declared,
            vec![
                Some(EntityType::NationalId),
                Some(EntityType::EmailAddress),
                Some(EntityType::IbanCode),
            ]
        );
    }

    #[test]
    fn an_exact_injection_declares_no_type() {
        // Nothing in this file knows what a caller-supplied literal is, and
        // guessing would put it in a per-type denominator it does not belong
        // in.
        let case = inject("Carrier text.", &[Injection::exact("ABC-123")], 0);
        assert_eq!(case.must_mask[0].entity_type(), None);
    }

    const FOUR_SENTENCES: &str =
        "First sentence here. Second sentence here. Third sentence here. Fourth one here.";

    fn positions(text: &str) -> (usize, usize) {
        (
            text.matches("Questions may be sent by email to").count(),
            text.matches(" (email ").count(),
        )
    }

    #[test]
    fn a_single_value_goes_to_either_position_depending_on_the_seed() {
        // THE SHAPE THE OLD TEST COULD NOT SEE. It used a four-injection
        // carrier, where alternating from index 0 balances on its own, and so
        // it passed while the fixture as a whole ran 170 boundary to 82
        // mid-sentence: every one-value case took index 0, and index 0 always
        // preferred a boundary. A test that cannot fail is worse than no
        // test, because it is quoted as evidence. This is the one-value
        // shape, which is where the fault lived.
        let mut boundary = 0;
        let mut mid = 0;
        for seed in 0..200 {
            let case = inject(FOUR_SENTENCES, &[Injection::email()], seed);
            let (at_boundary, in_sentence) = positions(&case.text);
            assert_eq!(at_boundary + in_sentence, 1);
            boundary += at_boundary;
            mid += in_sentence;
        }
        assert!(
            boundary >= 70 && mid >= 70,
            "one-value cases must use both positions, got {boundary} boundary / {mid} mid"
        );
    }

    #[test]
    fn a_two_value_case_uses_one_of_each_position_whichever_way_the_phase_falls() {
        // The other shape the old test missed. Two values must be one of
        // each, in either order, for every seed -- not just for the seeds
        // where the phase happens to start on a boundary.
        for seed in 0..40 {
            let case = inject(
                FOUR_SENTENCES,
                &[Injection::email(), Injection::email()],
                seed,
            );
            assert_eq!(
                positions(&case.text),
                (1, 1),
                "seed {seed} did not place one value in each position"
            );
        }
    }

    #[test]
    fn four_values_split_two_and_two() {
        for seed in 0..40 {
            let case = inject(
                FOUR_SENTENCES,
                &[
                    Injection::email(),
                    Injection::email(),
                    Injection::email(),
                    Injection::email(),
                ],
                seed,
            );
            assert_eq!(positions(&case.text), (2, 2), "seed {seed}");
        }
    }

    #[test]
    fn the_committed_fixture_splits_the_two_positions_near_equally() {
        // The claim the README and the report make, measured on the thing
        // they make it about rather than on a carrier chosen to demonstrate
        // it. `inject` prints the same figure every time it runs.
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let (_, split) = build(&root.join("benchmark/carriers"), 20260821).expect("rebuild");
        let share = split
            .boundary_share()
            .expect("the fixture places at least one value");
        assert!(
            (0.42..=0.58).contains(&share),
            "boundary share {share:.3} is not roughly equal ({} boundary / {} mid)",
            split.boundary,
            split.mid
        );
    }

    #[test]
    fn an_abbreviation_is_not_mistaken_for_the_end_of_a_sentence() {
        // "z. B." and "ca." end in a full stop and end no sentence. Splicing
        // a parenthetical into one produces text no register contains.
        let carrier = "Es gibt ca. 80.000 Anlagen. Die Zahl steigt.";
        let found = slots(carrier);
        let sentence_ends: Vec<usize> = found
            .iter()
            .filter(|slot| !slot.boundary)
            .map(|slot| slot.at)
            .collect();
        assert_eq!(sentence_ends.len(), 2, "only the two real sentences end");
        assert!(
            !sentence_ends.contains(&10),
            "'ca.' does not end a sentence"
        );
    }

    #[test]
    fn a_carrier_with_no_sentence_at_all_is_refused_rather_than_mangled() {
        assert_eq!(
            inject_checked("no sentence mark here", &[Injection::email()], 0),
            Err(InjectError::NoInsertionPoint)
        );
    }

    #[test]
    fn an_injected_iban_passes_the_services_own_iban_check() {
        for country in ["NL", "DE", "FR", "ES", "IT", "GB"] {
            for seed in 0..40 {
                let case = inject("Carrier text.", &[Injection::iban(country)], seed);
                assert!(
                    Validator::IbanMod97.accepts(&case.must_mask[0]),
                    "generated {country} IBAN must satisfy iban_mod97"
                );
            }
        }
    }

    #[test]
    fn an_injected_iban_carries_check_digits_a_bank_could_actually_issue() {
        // `iban_mod97` accepts DE00... and no bank has ever issued one: ISO
        // 13616 check digits run 02 to 98, and two pairs in 00..=99 satisfy
        // mod-97 for any given BBAN. Searching from 00 took the wrong one of
        // the two for roughly one draw in fifty, putting a value in the
        // answer key that a stricter detector is marked wrong for rejecting.
        for country in ["NL", "DE", "FR", "ES", "IT", "GB"] {
            for seed in 0..200 {
                let case = inject("Carrier text.", &[Injection::iban(country)], seed);
                let value = case.must_mask[0].as_str();
                let check: u32 = value[2..4].parse().expect("two check digits");
                assert!(
                    (2..=98).contains(&check),
                    "{country} IBAN check digits {check:02} are outside ISO 13616's 02..=98"
                );
            }
        }
    }

    #[test]
    fn an_injected_card_number_passes_the_services_own_luhn_check() {
        for seed in 0..12 {
            let case = inject("Carrier text.", &[Injection::credit_card()], seed);
            assert!(Validator::Luhn.accepts(&case.must_mask[0]));
        }
    }

    #[test]
    fn an_injected_german_tax_id_passes_the_services_own_check() {
        for seed in 0..12 {
            let case = inject("Carrier text.", &[Injection::tax_id("DE")], seed);
            assert!(Validator::GermanTaxId.accepts(&case.must_mask[0]));
        }
    }

    #[test]
    fn an_injected_spanish_identifier_passes_the_services_own_check() {
        for seed in 0..64 {
            let dni = inject("Carrier text.", &[Injection::national_id("ES-DNI")], seed);
            assert!(Validator::EsDni.accepts(&dni.must_mask[0]));
            let nie = inject("Carrier text.", &[Injection::national_id("ES-NIE")], seed);
            assert!(Validator::EsNie.accepts(&nie.must_mask[0]));
        }
    }

    #[test]
    fn the_spanish_generators_do_not_fall_back_to_one_value() {
        // Both draw in a bounded loop with a fixed valid value at the end of
        // it. If the loop never succeeded the check above would still pass,
        // on one repeated identifier, and the fixture would measure nothing.
        let drawn = |key: &str| {
            (0..64)
                .map(|seed| {
                    inject("Carrier text.", &[Injection::national_id(key)], seed).must_mask[0]
                        .as_str()
                        .to_string()
                })
                .collect::<BTreeSet<String>>()
        };
        assert!(drawn("ES-DNI").len() >= 50);
        assert!(drawn("ES-NIE").len() >= 50);
    }

    // --- national rules, checked against the rule and not against us ------
    //
    // WHY THIS MODULE IS WRITTEN TWICE OVER. Everything above asks the
    // service's own validator whether a value is good. For a checksum the
    // service implements, that is the strongest available check. For anything
    // the service does *not* check -- every VAT number, and the Steuer-ID's
    // structure -- it is no check at all, and the first version of this
    // fixture shipped 22 VAT numbers of which none satisfied its country's
    // rule and 24 Steuer-IDs that all violated the structure rule. A detector
    // applying the real rule would reject them and be marked wrong by our own
    // answer key.
    //
    // So each verifier below is transcribed from the public specification
    // separately from the generator it checks, in a different arrangement
    // where the algorithm allows one -- the same two-transcription discipline
    // `training/generate.py` applies to its own generators, and for the same
    // reason: one slip is likely, the same slip made twice independently is
    // not.
    mod national_rules {
        use super::*;

        fn digits_of(value: &str) -> Vec<u32> {
            value.chars().filter_map(|c| c.to_digit(10)).collect()
        }

        /// Luhn *validation* -- the whole string, including its check digit,
        /// summing to a multiple of ten. Deliberately not the
        /// compute-the-check-digit arrangement `luhn_check_digit` uses.
        fn luhn_holds(digits: &[u32]) -> bool {
            let mut total = 0u32;
            for (index, digit) in digits.iter().rev().enumerate() {
                let mut value = *digit;
                if !index.is_multiple_of(2) {
                    value *= 2;
                    if value > 9 {
                        value -= 9;
                    }
                }
                total += value;
            }
            total.is_multiple_of(10)
        }

        /// The Dutch 11-proef in its "the weighted sum is divisible by
        /// eleven" arrangement, rather than the generator's "the check digit
        /// is the weighted remainder" one.
        fn elfproef_holds(digits: &[u32]) -> bool {
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

        /// ISO 7064 MOD 11,10 over `digits`, the last of which is the check.
        fn mod_11_10_holds(digits: &[u32]) -> bool {
            let Some((check, body)) = digits.split_last() else {
                return false;
            };
            let product = body.iter().fold(10u32, |product, digit| {
                let mut sum = (digit + product) % 10;
                if sum == 0 {
                    sum = 10;
                }
                (sum * 2) % 11
            });
            (11 - product) % 10 == *check
        }

        fn nl_vat_holds(value: &str) -> bool {
            let bytes: Vec<char> = value.chars().collect();
            value.len() == 14
                && value.starts_with("NL")
                && bytes[11] == 'B'
                && bytes[2..11].iter().all(|c| c.is_ascii_digit())
                && bytes[12..].iter().all(|c| c.is_ascii_digit())
                && elfproef_holds(&digits_of(&value[2..11]))
        }

        fn de_vat_holds(value: &str) -> bool {
            let body = digits_of(&value[2..]);
            value.len() == 11
                && value.starts_with("DE")
                && body.len() == 9
                && mod_11_10_holds(&body)
        }

        fn fr_vat_holds(value: &str) -> bool {
            if value.len() != 13 || !value.starts_with("FR") {
                return false;
            }
            let siren = digits_of(&value[4..]);
            let Ok(key) = value[2..4].parse::<u64>() else {
                return false;
            };
            let Ok(numeric) = value[4..].parse::<u64>() else {
                return false;
            };
            // The published equivalent of (12 + 3 x (SIREN mod 97)) mod 97,
            // written the other way round: 100 is 3 modulo 97.
            siren.len() == 9 && luhn_holds(&siren) && (numeric % 97 * 100 + 12) % 97 == key
        }

        fn es_vat_holds(value: &str) -> bool {
            if value.len() != 11 || !value.starts_with("ES") {
                return false;
            }
            let kind = value.chars().nth(2).unwrap_or(' ');
            // A and B are digit-check types. A letter check on one of them is
            // the malformation the first version emitted.
            if !matches!(kind, 'A' | 'B') {
                return false;
            }
            let body = digits_of(&value[3..]);
            if body.len() != 8 {
                return false;
            }
            let mut total = 0u32;
            for (index, digit) in body[..7].iter().enumerate() {
                if index % 2 == 1 {
                    total += digit;
                } else {
                    let doubled = digit * 2;
                    total += doubled / 10 + doubled % 10;
                }
            }
            (10 - total % 10) % 10 == body[7]
        }

        fn it_vat_holds(value: &str) -> bool {
            let body = digits_of(&value[2..]);
            // Validation form: every digit including the check sums to a
            // multiple of ten.
            value.len() == 13 && value.starts_with("IT") && body.len() == 11 && luhn_holds(&body)
        }

        /// The BZSt structure rule for the first ten digits of a Steuer-ID:
        /// exactly one digit value occurs more than once (twice, or three
        /// times but not consecutively), every other value at most once, and
        /// the first digit is not zero.
        fn steuer_id_structure_holds(value: &str) -> bool {
            let digits = digits_of(value);
            if digits.len() != 11 || digits[0] == 0 {
                return false;
            }
            let body = &digits[..10];
            let mut counts = [0usize; 10];
            for digit in body {
                counts[*digit as usize] += 1;
            }
            let repeated: Vec<usize> = (0..10).filter(|d| counts[*d] > 1).collect();
            if repeated.len() != 1 {
                return false;
            }
            let value_repeated = repeated[0];
            let times = counts[value_repeated];
            if !(times == 2 || times == 3) {
                return false;
            }
            if times == 3 {
                let places: Vec<usize> = body
                    .iter()
                    .enumerate()
                    .filter(|(_, d)| **d as usize == value_repeated)
                    .map(|(index, _)| index)
                    .collect();
                if places[2] - places[0] == 2 {
                    return false;
                }
            }
            true
        }

        fn drawn(injection: Injection, seed: u64) -> String {
            inject("Carrier text.", &[injection], seed).must_mask[0]
                .as_str()
                .to_string()
        }

        /// The Spanish check letters, in the order the remainder indexes
        /// them. Transcribed here so the verifiers below can be written in
        /// the *validating* arrangement -- recover the letter's index and
        /// compare it to the remainder -- rather than the generator's, which
        /// never sees this order at all: it hands candidate letters to
        /// `Validator::EsDni` and keeps whichever one is accepted.
        const ES_LETTERS: &str = "TRWAGMYFPDXBNJZSQVHLCKE";

        /// Whether `body` (eight digits) and `letter` agree under modulo 23.
        fn es_letter_agrees(body: &str, letter: char) -> bool {
            let Some(index) = ES_LETTERS.find(letter) else {
                return false;
            };
            let Ok(number) = body.parse::<u32>() else {
                return false;
            };
            number % 23 == index as u32
        }

        fn es_dni_holds(value: &str) -> bool {
            let characters: Vec<char> = value.chars().collect();
            characters.len() == 9
                && characters[..8].iter().all(|c| c.is_ascii_digit())
                && es_letter_agrees(&value[..8], characters[8])
        }

        /// The codice fiscale check character, transcribed separately from
        /// `Validator::ItCodiceFiscale` and arranged the other way round:
        /// the two position tables are built here as one lookup string per
        /// parity and indexed by character, where the validator branches on
        /// digit-or-letter. Structure is checked too, because the generator
        /// chooses the structure and nothing else would notice if it chose
        /// an impossible one.
        fn it_codice_fiscale_holds(value: &str) -> bool {
            const ODD: [u32; 36] = [
                1, 0, 5, 7, 9, 13, 15, 17, 19, 21, // 0-9
                1, 0, 5, 7, 9, 13, 15, 17, 19, 21, 2, 4, 18, 20, 11, 3, 6, 8, 12, 14, 16, 10, 22,
                25, 24, 23, // A-Z
            ];
            let index_of = |character: char| -> Option<usize> {
                if character.is_ascii_digit() {
                    Some(character as usize - '0' as usize)
                } else if character.is_ascii_uppercase() {
                    Some(10 + character as usize - 'A' as usize)
                } else {
                    None
                }
            };
            let characters: Vec<char> = value.chars().collect();
            if characters.len() != 16 {
                return false;
            }
            if !characters[..6].iter().all(|c| c.is_ascii_uppercase()) {
                return false;
            }
            if !"ABCDEHLMPRST".contains(characters[8]) {
                return false;
            }
            let digit_of = |c: char| -> Option<u32> {
                c.to_digit(10)
                    .or_else(|| "LMNPQRSTUV".find(c).map(|i| i as u32))
            };
            let (Some(tens), Some(units)) = (digit_of(characters[9]), digit_of(characters[10]))
            else {
                return false;
            };
            let day = tens * 10 + units;
            if !(1..=31).contains(&day) && !(41..=71).contains(&day) {
                return false;
            }
            let mut total = 0u32;
            for (position, character) in characters[..15].iter().enumerate() {
                let Some(index) = index_of(*character) else {
                    return false;
                };
                // Positions are counted from one in the published rule.
                total += if (position + 1) % 2 == 1 {
                    ODD[index]
                } else if index < 10 {
                    // A digit is worth itself.
                    index as u32
                } else {
                    // A letter is worth its place in the alphabet.
                    (index - 10) as u32
                };
            }
            let expected = char::from(b'A' + (total % 26) as u8);
            characters[15] == expected
        }

        /// The NIR key, transcribed separately from `Validator::FrNir` and
        /// arranged the other way round: this adds the key to the body and
        /// asks whether the sum is a multiple of 97, where the validator
        /// subtracts the remainder from 97 and compares. The two agree only
        /// if both are right, because `(n + (97 - n mod 97)) mod 97 == 0`.
        fn fr_nir_holds(value: &str) -> bool {
            let compact: String = value.chars().filter(|c| *c != ' ').collect();
            if compact.len() != 15 {
                return false;
            }
            let substituted = compact.replacen("2A", "19", 1).replacen("2B", "18", 1);
            let Ok(whole) = substituted.parse::<u64>() else {
                return false;
            };
            let Ok(key) = substituted[13..].parse::<u64>() else {
                return false;
            };
            if !(1..=97).contains(&key) {
                return false;
            }
            let Some(month) = substituted.get(3..5).and_then(|m| m.parse::<u64>().ok()) else {
                return false;
            };
            if !(1..=12).contains(&month) {
                return false;
            }
            // The body plus its key is divisible by 97.
            (whole / 100 + key).is_multiple_of(97)
        }

        /// The Belgian control, transcribed separately from
        /// `Validator::BeNationalRegister` and arranged the other way
        /// round: this adds the control to the nine digits and asks whether
        /// the sum is a multiple of 97, where the validator subtracts the
        /// remainder from 97 and compares.
        fn be_national_register_holds(value: &str) -> bool {
            let compact: String = value.chars().filter(|c| c.is_ascii_digit()).collect();
            if compact.len() != 11 {
                return false;
            }
            let (Ok(nine), Ok(control)) =
                (compact[..9].parse::<u64>(), compact[9..].parse::<u64>())
            else {
                return false;
            };
            let Ok(month) = compact[2..4].parse::<u64>() else {
                return false;
            };
            let Ok(day) = compact[4..6].parse::<u64>() else {
                return false;
            };
            let Ok(sequence) = compact[6..9].parse::<u64>() else {
                return false;
            };
            let real_month = match month {
                0 | 20 | 40 => 0,
                21..=32 => month - 20,
                41..=52 => month - 40,
                other => other,
            };
            if real_month > 12 || day > 31 || sequence == 0 || sequence == 999 {
                return false;
            }
            if !(1..=97).contains(&control) {
                return false;
            }
            (nine + control).is_multiple_of(97)
                || (2_000_000_000 + nine + control).is_multiple_of(97)
        }

        fn es_nie_holds(value: &str) -> bool {
            let characters: Vec<char> = value.chars().collect();
            if characters.len() != 9 || !characters[1..8].iter().all(|c| c.is_ascii_digit()) {
                return false;
            }
            // The three issuing series, in order, stand for 0, 1 and 2.
            let Some(series) = "XYZ".find(characters[0]) else {
                return false;
            };
            es_letter_agrees(&format!("{series}{}", &value[1..8]), characters[8])
        }

        #[test]
        fn every_spanish_identifier_satisfies_the_published_rule() {
            for seed in 0..200 {
                assert!(es_dni_holds(&drawn(Injection::national_id("ES-DNI"), seed)));
                assert!(es_nie_holds(&drawn(Injection::national_id("ES-NIE"), seed)));
            }
        }

        #[test]
        fn every_codice_fiscale_satisfies_the_published_rule() {
            for seed in 0..200 {
                let value = drawn(Injection::tax_id("IT"), seed);
                assert!(
                    it_codice_fiscale_holds(&value),
                    "seed {seed} produced a codice fiscale the published rule rejects"
                );
            }
        }

        #[test]
        fn every_nir_satisfies_the_published_rule() {
            let mut corsican = 0;
            for seed in 0..300 {
                let value = drawn(Injection::national_id("FR"), seed);
                assert!(
                    fr_nir_holds(&value),
                    "seed {seed} produced a NIR the published rule rejects"
                );
                if value.contains('A') || value.contains('B') {
                    corsican += 1;
                }
                assert!(
                    !digits_pass_as_a_credit_card(&value),
                    "seed {seed} drew a NIR whose digits pass Luhn at card length"
                );
            }
            // The Corsican substitution is the part implementations get
            // wrong, so the fixture has to actually contain some.
            assert!(corsican >= 10, "only {corsican} Corsican NIRs in 300 draws");
        }

        #[test]
        fn every_belgian_number_satisfies_the_published_rule() {
            let mut from_2000 = 0;
            let mut bis = 0;
            for seed in 0..300 {
                let value = drawn(Injection::national_id("BE"), seed);
                assert!(
                    be_national_register_holds(&value),
                    "seed {seed} produced a national register number the published rule rejects"
                );
                let nine: u64 = value[..9].parse().expect("nine digits");
                let control: u64 = value[9..].parse().expect("two digits");
                if !(nine + control).is_multiple_of(97) {
                    from_2000 += 1;
                }
                let month: u64 = value[2..4].parse().expect("two digits");
                if month > 12 {
                    bis += 1;
                }
            }
            // Both century branches and the bis form must actually occur,
            // or the fixture tests one third of the rule.
            assert!(
                from_2000 >= 20,
                "only {from_2000} numbers from this century"
            );
            assert!(bis >= 20, "only {bis} bis numbers");
        }

        #[test]
        fn the_belgian_verifier_rejects_what_a_careless_generator_would_produce() {
            for published in ["42012205181", "72020290081", "00012556777", "93051822361"] {
                assert!(be_national_register_holds(published), "{published}");
            }
            assert!(!be_national_register_holds("93051822362"));
            assert!(!be_national_register_holds("93151822361"));
            assert!(!be_national_register_holds("93051899961"));
        }

        #[test]
        fn the_french_verifier_rejects_what_a_careless_generator_would_produce() {
            assert!(fr_nir_holds("174033000705847"));
            assert!(fr_nir_holds("185036912304532"));
            assert!(fr_nir_holds("290052B03301001"));
            // 2A substitutes 19, not 18.
            assert!(!fr_nir_holds("290052A03301001"));
            assert!(fr_nir_holds("290052A03301071"));
            // Wrong key, impossible month, key out of range.
            assert!(!fr_nir_holds("174033000705848"));
            assert!(!fr_nir_holds("174133000705847"));
            assert!(!fr_nir_holds("174033000705800"));
        }

        #[test]
        fn the_italian_verifier_rejects_what_a_careless_generator_would_produce() {
            // The published anchors, and the specimen everybody repeats and
            // nobody checks -- whose real check character is `Q`.
            assert!(it_codice_fiscale_holds("MRTMTT91D08F205J"));
            assert!(it_codice_fiscale_holds("MLLSNT82P65Z404U"));
            assert!(it_codice_fiscale_holds("BBBTTT20H12X122H"));
            assert!(!it_codice_fiscale_holds("RSSMRA85M01H501Z"));
            assert!(it_codice_fiscale_holds("RSSMRA85M01H501Q"));
            // An impossible month letter and an impossible day.
            assert!(!it_codice_fiscale_holds("MRTMTT91F08F205J"));
            assert!(!it_codice_fiscale_holds("MRTMTT91D35F205J"));
        }

        #[test]
        fn the_spanish_verifiers_reject_what_a_careless_generator_would_produce() {
            // Without this the test above could pass against a verifier that
            // accepts everything. The published examples, and near misses.
            assert!(es_dni_holds("12345678Z"));
            assert!(!es_dni_holds("12345678A"));
            assert!(!es_dni_holds("21345678Z"), "a transposition must not hold");
            assert!(es_nie_holds("X1234567L"));
            assert!(!es_nie_holds("Y1234567L"));
            assert!(!es_nie_holds("W1234567L"));
            // A DNI is not a NIE and a NIE is not a DNI.
            assert!(!es_dni_holds("X1234567L"));
            assert!(!es_nie_holds("12345678Z"));
        }

        #[test]
        fn every_vat_number_satisfies_its_own_country_rule() {
            for seed in 0..120 {
                assert!(nl_vat_holds(&drawn(Injection::vat("NL"), seed)));
                assert!(de_vat_holds(&drawn(Injection::vat("DE"), seed)));
                assert!(fr_vat_holds(&drawn(Injection::vat("FR"), seed)));
                assert!(es_vat_holds(&drawn(Injection::vat("ES"), seed)));
                assert!(it_vat_holds(&drawn(Injection::vat("IT"), seed)));
                // English-language carriers carry a Dutch number.
                assert!(nl_vat_holds(&drawn(Injection::vat("GB"), seed)));
            }
        }

        #[test]
        fn a_dutch_vat_number_also_satisfies_the_services_own_elfproef() {
            // Two implementations in two crates, written from the same public
            // rule, agreeing. Not the definition of correct here -- the
            // verifier above is -- but a useful second opinion.
            for seed in 0..40 {
                let value = drawn(Injection::vat("NL"), seed);
                assert!(Validator::Elfproef.accepts(&value[2..11]));
            }
        }

        #[test]
        fn every_steuer_id_satisfies_the_structure_rule_and_not_only_the_check_digit() {
            // The finding this exists for: ten free digits plus a correct
            // check digit satisfies `german_tax_id` and violates the BZSt
            // specification about 96 times in a hundred. The old generator
            // did exactly that, as `training/generate.py::_steuer_id` still
            // does.
            for seed in 0..200 {
                let value = drawn(Injection::tax_id("DE"), seed);
                assert!(
                    steuer_id_structure_holds(&value),
                    "seed {seed} produced a Steuer-ID the structure rule rejects"
                );
                assert!(Validator::GermanTaxId.accepts(&value));
            }
        }

        #[test]
        fn the_structure_verifier_rejects_what_the_old_generator_produced() {
            // Without this, the test above could pass against a verifier that
            // accepts everything. `13570000` repeated digits: four zeros,
            // which no Steuer-ID has.
            assert!(!steuer_id_structure_holds("13570000246"));
            // Two different digits repeated.
            assert!(!steuer_id_structure_holds("11223456780"));
            // Leading zero.
            assert!(!steuer_id_structure_holds("01234567891"));
            // A real one from the pack's own positive examples.
            assert!(steuer_id_structure_holds("86095742719"));
            assert!(steuer_id_structure_holds("36574261809"));
        }

        #[test]
        fn the_vat_verifiers_reject_what_the_old_generator_produced() {
            // The exact malformed value the review found in the first
            // fixture, plus a shape-correct Dutch number with a bad checksum.
            assert!(!es_vat_holds("ESH0912908F"));
            assert!(!nl_vat_holds("NL123456789B01"));
            assert!(!de_vat_holds("DE811569860"));
            assert!(de_vat_holds("DE811569869"));
        }

        /// The country whose VAT rule applies to a `must_mask` entry that
        /// declares **no** type, or `None` when the entry is not shaped like
        /// a VAT number at all.
        ///
        /// WHY THIS EXISTS. The guard below used to look only at entries
        /// declaring `VAT_NUMBER`. `benchmark/cases/seed.json` -- the 128
        /// legacy cases -- declares no types at all, so two `must_mask` VAT
        /// values that failed their own national check digit sat in the
        /// fixture untouched by a guard written to catch exactly them. A
        /// value is checked here on what it *is*, not on whether somebody
        /// remembered to label it.
        ///
        /// The shapes are exact-length and fully anchored, so nothing else
        /// in the fixture collides with them: a Dutch IBAN is 18 characters,
        /// not `NL` + 9 digits + `B` + 2 digits.
        fn vat_country_of_an_untyped_value(value: &str) -> Option<&'static str> {
            let characters: Vec<char> = value.chars().collect();
            let digits = |slice: &[char]| {
                !slice.is_empty() && slice.iter().all(|character| character.is_ascii_digit())
            };
            match value.get(..2) {
                Some("NL")
                    if characters.len() == 14
                        && digits(&characters[2..11])
                        && characters[11] == 'B'
                        && digits(&characters[12..]) =>
                {
                    Some("NL")
                }
                Some("DE") if characters.len() == 11 && digits(&characters[2..]) => Some("DE"),
                Some("FR") if characters.len() == 13 && digits(&characters[2..]) => Some("FR"),
                Some("ES")
                    if characters.len() == 11
                        && characters[2].is_ascii_uppercase()
                        && digits(&characters[3..]) =>
                {
                    Some("ES")
                }
                Some("IT") if characters.len() == 13 && digits(&characters[2..]) => Some("IT"),
                _ => None,
            }
        }

        #[test]
        fn every_committed_vat_value_in_the_fixture_satisfies_its_country_rule() {
            // Not the generator in isolation: the values actually committed
            // -- including the ones that declare no type, which is how two
            // invalid legacy numbers escaped this guard until the final
            // whole-branch review.
            let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmark/cases");
            let cases = pii_detection_eval::load(&dir).expect("the case directory loads");
            let mut declared = 0;
            let mut undeclared = 0;
            for case in &cases {
                for value in &case.must_mask {
                    let text = value.as_str();
                    let country = match value.entity_type() {
                        Some(EntityType::VatNumber) => {
                            declared += 1;
                            text.get(..2)
                                .expect("a declared VAT value carries a country prefix")
                        }
                        Some(_) => continue,
                        None => match vat_country_of_an_untyped_value(text) {
                            Some(country) => {
                                undeclared += 1;
                                country
                            }
                            None => continue,
                        },
                    };
                    let holds = match country {
                        "NL" => nl_vat_holds(text),
                        "DE" => de_vat_holds(text),
                        "FR" => fr_vat_holds(text),
                        "ES" => es_vat_holds(text),
                        "IT" => it_vat_holds(text),
                        _ => false,
                    };
                    assert!(holds, "a committed VAT value fails its country rule");
                }
            }
            assert!(declared >= 20, "expected the fixture to carry VAT values");
            assert!(
                undeclared >= 2,
                "seed.json's untyped VAT values must be covered by this guard, not skipped by it"
            );
        }

        // --- driving licences -----------------------------------------
        //
        // Two countries, and only one of them has a rule a verifier can
        // check arithmetic against.

        /// The base-36 value of a Führerscheinnummer character, from a table
        /// written out here rather than from `char::to_digit(36)`, which is
        /// what `Validator::DeFuehrerschein` uses. Two spellings of the same
        /// mapping, so a slip in one does not hide in the other.
        const DE_LICENCE_ALPHABET: &str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";

        fn de_licence_digit(character: char) -> Option<u32> {
            DE_LICENCE_ALPHABET
                .find(character)
                .and_then(|index| u32::try_from(index).ok())
        }

        /// The German licence check, in its *validating* arrangement: the
        /// nine weighted characters plus the complement of the check digit
        /// must come to a multiple of eleven. `Validator::DeFuehrerschein`
        /// is written the other way round -- it computes the remainder and
        /// compares a character to it -- and neither arrangement can be
        /// derived from the other by copying.
        ///
        /// The structure is checked too, because it is half of what the
        /// recognizer relies on: a Bundesland letter first, then two
        /// authority digits.
        fn de_fuehrerschein_holds(value: &str) -> bool {
            let characters: Vec<char> = value.chars().collect();
            if characters.len() != 11 || !characters.iter().all(|c| c.is_ascii_alphanumeric()) {
                return false;
            }
            if !characters[0].is_ascii_uppercase()
                || !characters[1].is_ascii_digit()
                || !characters[2].is_ascii_digit()
            {
                return false;
            }
            let mut total = 0u32;
            for (index, character) in characters[..9].iter().enumerate() {
                let Some(digit) = de_licence_digit(*character) else {
                    return false;
                };
                let Ok(weight) = u32::try_from(9 - index) else {
                    return false;
                };
                total += digit * weight;
            }
            let check = match characters[9] {
                'X' => 10,
                digit => match digit.to_digit(10) {
                    Some(digit) => digit,
                    None => return false,
                },
            };
            (total + (11 - check)).is_multiple_of(11)
        }

        /// The Dutch rijbewijsnummer's published rule, in full: ten digits.
        ///
        /// THIS VERIFIER IS WEAKER THAN THE GERMAN ONE AND THE GAP IS STATED
        /// RATHER THAN PAPERED OVER. There is no check character, no field
        /// layout and no date to decode -- the RDW publishes none and
        /// Purview records `Checksum: No` -- so "satisfies the published
        /// rule" really is only "is ten digits", and a verifier cannot be
        /// more independent of the generator than the rule is of itself.
        ///
        /// What it *can* check independently is the fixture's own rule: no
        /// committed value begins with a trunk zero. A ten-digit run
        /// starting `0` is how a Dutch telephone number is written solid, so
        /// the phone recognizer would answer for it at 0.95 against this
        /// recognizer's 0.75 -- the same characters masked, under the other
        /// label -- and the case would be measuring something else. See
        /// [`nl_rijbewijs`].
        fn nl_rijbewijs_holds(value: &str) -> bool {
            let characters: Vec<char> = value.chars().collect();
            characters.len() == 10
                && characters
                    .iter()
                    .all(|character| character.is_ascii_digit())
                && characters.first() != Some(&'0')
        }

        /// The NEPH's published rule: twelve digits whose third and fourth
        /// are a month the calendar has.
        ///
        /// Decoded rather than re-encoded -- the fields are read back out of
        /// the string and range-checked -- so this is not the generator's
        /// arithmetic run backwards. It is stricter than the pattern in one
        /// place and weaker in none: the pattern range-checks the month and
        /// nothing else, and this also checks the fixture's own rule that no
        /// value carries a year of 00 to 09 (see [`fr_permis_conduire`]) and
        /// that the department is one metropolitan France has.
        ///
        /// It says nothing about the prefecture digits or the four order
        /// digits, because nothing published constrains them.
        fn fr_neph_holds(value: &str) -> bool {
            let characters: Vec<char> = value.chars().collect();
            if characters.len() != 12 || !characters.iter().all(|c| c.is_ascii_digit()) {
                return false;
            }
            let field = |from: usize, to: usize| -> Option<u32> {
                characters
                    .get(from..to)
                    .map(|slice| slice.iter().collect::<String>())
                    .and_then(|slice| slice.parse::<u32>().ok())
            };
            let (Some(year), Some(month), Some(department)) =
                (field(0, 2), field(2, 4), field(4, 6))
            else {
                return false;
            };
            (10..=99).contains(&year) && (1..=12).contains(&month) && (1..=95).contains(&department)
        }

        /// The Italian patente di guida's published structure, in both of
        /// the two concrete readings: a two-letter province code and seven
        /// digits, or `U1` and seven alphanumerics -- and in either case a
        /// final alphabetic control character, optionally preceded by the
        /// single space the Ministry's own worked example prints.
        ///
        /// **THERE IS NO ARITHMETIC HERE AND THERE CANNOT BE.** No source
        /// publishes the control character's rule, so this verifier is
        /// limited to structure exactly as the British one is, and it says
        /// so in the same words. What it does check that the pattern does
        /// not is the fixture's own rule: the province code is one of the
        /// real Motorizzazione codes [`IT_PROVINCE_CODES`] holds, so a
        /// committed value is one the issuing authority could have produced
        /// rather than two drawn letters.
        fn it_patente_holds(value: &str) -> bool {
            let compact: String = value.chars().filter(|c| !" -".contains(*c)).collect();
            let characters: Vec<char> = compact.chars().collect();
            if characters.len() != 10 || !characters.iter().all(|c| c.is_ascii_alphanumeric()) {
                return false;
            }
            let Some(control) = characters.last() else {
                return false;
            };
            if !control.is_ascii_uppercase() {
                return false;
            }
            let head: String = characters[..2].iter().collect();
            let body = &characters[2..9];
            if head == "U1" {
                return body.iter().all(|c| c.is_ascii_alphanumeric());
            }
            IT_PROVINCE_CODES.contains(&head.as_str()) && body.iter().all(|c| c.is_ascii_digit())
        }

        /// A field of `count` characters that is upper-case letters followed
        /// by `9` padding, and never a letter after a `9`. Both the surname
        /// and the initials of a DVLA driver number obey it, and getting it
        /// backwards -- padding on the left -- is the likely slip.
        fn gb_padded_field(field: &[char], count: usize) -> bool {
            if field.len() != count {
                return false;
            }
            let mut padding = false;
            for character in field {
                match character {
                    '9' => padding = true,
                    c if c.is_ascii_uppercase() && !padding => {}
                    _ => return false,
                }
            }
            true
        }

        /// Whether `year`/`month`/`day` is a date the calendar has. The
        /// generator never draws past the 28th, so a value that needs this
        /// to know about leap years has already gone wrong.
        fn gb_real_date(year: u32, month: u32, day: u32) -> bool {
            let length = match month {
                1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
                4 | 6 | 9 | 11 => 30,
                2 if year.is_multiple_of(4)
                    && (!year.is_multiple_of(100) || year.is_multiple_of(400)) =>
                {
                    29
                }
                2 => 28,
                _ => return false,
            };
            (1..=length).contains(&day)
        }

        /// The DVLA driver number, checked against the only rule that is
        /// published: its structure.
        ///
        /// THERE IS NO ARITHMETIC HERE AND THERE CANNOT BE. Characters 15 and
        /// 16 are check characters whose algorithm the DVLA has refused to
        /// publish, so neither the generator nor this verifier can say
        /// anything about them beyond their alphabet. What *is* checked is
        /// everything else: the padding rule on both name fields, the
        /// female-month offset of fifty, and a date of birth that the
        /// calendar actually has -- decoded here rather than re-encoded, so
        /// this is not the generator's arithmetic read backwards.
        ///
        /// The decade digit leaves the century ambiguous by design, so the
        /// year is accepted if either reading is a plausible driver's: the
        /// verifier asks whether the day fits the month for the reading it
        /// picks, and picks the one in living memory.
        fn gb_driver_number_holds(value: &str) -> bool {
            let characters: Vec<char> = value.chars().collect();
            // Sixteen, or eighteen with the two-digit issue number appended.
            if !matches!(characters.len(), 16 | 18) {
                return false;
            }
            if characters.len() == 18 && !characters[16..].iter().all(|c| c.is_ascii_digit()) {
                return false;
            }
            if !gb_padded_field(&characters[..5], 5) || !gb_padded_field(&characters[11..13], 2) {
                return false;
            }
            if !characters[5..11].iter().all(|c| c.is_ascii_digit())
                || !characters[13].is_ascii_digit()
                || !characters[14..16].iter().all(|c| c.is_ascii_alphanumeric())
            {
                return false;
            }
            let number =
                |slice: &[char]| -> Option<u32> { slice.iter().collect::<String>().parse().ok() };
            let (Some(decade), Some(encoded_month), Some(day), Some(unit)) = (
                number(&characters[5..6]),
                number(&characters[6..8]),
                number(&characters[8..10]),
                number(&characters[10..11]),
            ) else {
                return false;
            };
            // Fifty added for a woman, and nothing else added ever.
            let month = if encoded_month > 50 {
                encoded_month - 50
            } else {
                encoded_month
            };
            // 19x0 + unit and 20x0 + unit are the two readings of the decade
            // digit; take whichever is a plausible driver's year of birth.
            let candidates = [1900 + decade * 10 + unit, 2000 + decade * 10 + unit];
            candidates
                .iter()
                .any(|year| (1920..=2010).contains(year) && gb_real_date(*year, month, day))
        }

        /// Which country identifier a committed `must_mask` value is shaped
        /// like, or `None`.
        ///
        /// Selects on SHAPE and never on validity, for the reason the
        /// Steuer-ID guard's own comment gives: a selector gated on the
        /// property the guard exists to assert can never see the value that
        /// violates it. The shapes are exact-length and fully anchored, and
        /// none of them collides with another identifier in the fixture -- a
        /// Dutch BSN is nine digits with no letter, a Steuer-ID eleven.
        /// The Belgian national register number and the German Steuer-ID
        /// are both eleven ASCII digits and no shape tells them apart, so
        /// this one selector consults the type the case DECLARED. That is
        /// not the property being asserted -- validity is -- so it does not
        /// reopen the hole the Steuer-ID guard's own comment describes: an
        /// eleven-digit value declared `NATIONAL_ID` is a Belgian number
        /// whatever its digits say, and is checked against the Belgian
        /// rule; an eleven-digit `TAX_ID`, or an untyped one, stays the
        /// German guard's business.
        fn country_identifier_shape(
            value: &str,
            declared: Option<EntityType>,
        ) -> Option<&'static str> {
            let characters: Vec<char> = value.chars().collect();
            let digits = |slice: &[char]| slice.iter().all(|c| c.is_ascii_digit());
            if characters.len() == 11
                && digits(&characters)
                && declared == Some(EntityType::NationalId)
            {
                return Some("BE-RRN");
            }
            match characters.len() {
                9 if digits(&characters[..8]) && characters[8].is_ascii_uppercase() => {
                    Some("ES-DNI")
                }
                9 if matches!(characters[0], 'X' | 'Y' | 'Z')
                    && digits(&characters[1..8])
                    && characters[8].is_ascii_uppercase() =>
                {
                    Some("ES-NIE")
                }
                16 if characters[..6].iter().all(|c| c.is_ascii_uppercase())
                    && characters.iter().all(|c| c.is_ascii_alphanumeric()) =>
                {
                    Some("IT-CF")
                }
                15 if digits(&characters[..5])
                    && digits(&characters[7..])
                    && (digits(&characters[5..7])
                        || matches!((characters[5], characters[6]), ('2', 'A') | ('2', 'B'))) =>
                {
                    Some("FR-NIR")
                }
                _ => None,
            }
        }

        #[test]
        fn every_committed_country_identifier_satisfies_its_national_rule() {
            // The values actually committed, not the generator in isolation.
            // The VAT sibling above exists because the first version of this
            // fixture shipped 22 VAT numbers that satisfied no national rule;
            // this is the same guard for the country packs.
            let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmark/cases");
            let cases = pii_detection_eval::load(&dir).expect("the case directory loads");
            let mut counted: std::collections::BTreeMap<&str, usize> = Default::default();
            for case in &cases {
                for value in &case.must_mask {
                    let text = value.as_str();
                    let Some(kind) = country_identifier_shape(text, value.entity_type()) else {
                        continue;
                    };
                    *counted.entry(kind).or_default() += 1;
                    let holds = match kind {
                        "ES-DNI" => es_dni_holds(text),
                        "ES-NIE" => es_nie_holds(text),
                        "IT-CF" => it_codice_fiscale_holds(text),
                        "FR-NIR" => fr_nir_holds(text),
                        "BE-RRN" => be_national_register_holds(text),
                        _ => false,
                    };
                    assert!(
                        holds,
                        "a committed {kind} value fails its published national rule"
                    );
                }
            }
            for kind in ["ES-DNI", "ES-NIE", "IT-CF", "FR-NIR", "BE-RRN"] {
                assert!(
                    counted.get(kind).copied().unwrap_or(0) >= 30,
                    "the fixture must carry enough {kind} values to gate on"
                );
            }
        }

        /// Every committed `DRIVING_LICENCE` value satisfies the published
        /// rule for the country whose shape it has.
        ///
        /// Selected by SHAPE and by the declared type, never by validity --
        /// the same discipline the tax-id guard above had to be rewritten
        /// into. Eleven characters is Germany, sixteen or eighteen is the
        /// United Kingdom, and a value of neither length under this type is
        /// a value nothing in this repository knows how to check, which is
        /// itself a failure.
        #[test]
        fn every_committed_driving_licence_satisfies_its_published_rule() {
            let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmark/cases");
            let cases = pii_detection_eval::load(&dir).expect("the case directory loads");
            let mut counted: std::collections::BTreeMap<&str, usize> = Default::default();
            for case in &cases {
                for value in &case.must_mask {
                    if value.entity_type() != Some(EntityType::DrivingLicence) {
                        continue;
                    }
                    let text = value.as_str();
                    let (kind, holds) = match text.chars().count() {
                        10 if text.chars().all(|c| c.is_ascii_digit()) => {
                            ("NL", nl_rijbewijs_holds(text))
                        }
                        12 if text.chars().all(|c| c.is_ascii_digit()) => {
                            ("FR", fr_neph_holds(text))
                        }
                        // Italy before Germany, and separated from it by
                        // the separator rather than by length: a German
                        // licence is eleven alphanumerics with no space in
                        // it, and an Italian one is eleven characters only
                        // when the ninth is the space its circular prints.
                        10 if text.chars().any(|c| c.is_ascii_alphabetic()) => {
                            ("IT", it_patente_holds(text))
                        }
                        11 if text.contains(' ') || text.contains('-') => {
                            ("IT", it_patente_holds(text))
                        }
                        11 => ("DE", de_fuehrerschein_holds(text)),
                        16 | 18 => ("GB", gb_driver_number_holds(text)),
                        other => panic!(
                            "a committed driving licence is {other} characters long, which is \
                             no covered country's shape"
                        ),
                    };
                    *counted.entry(kind).or_default() += 1;
                    assert!(
                        holds,
                        "a committed {kind} driving licence fails its published rule"
                    );
                }
            }
            for kind in ["DE", "GB", "NL", "FR", "IT"] {
                assert!(
                    counted.get(kind).copied().unwrap_or(0) >= 30,
                    "the fixture must carry enough {kind} licence values to gate on"
                );
            }
        }

        /// A licence-shaped hard negative is hard **because the text says so
        /// in words**, and this reads the words out of the shipped pack
        /// rather than trusting the generator that wrote them.
        ///
        /// The Dutch, French and Italian recognizers have no checksum. A
        /// ten- or twelve-digit `must_clear` value in one of their case
        /// files is therefore not rejected by anything about the value: it
        /// is rejected only if a term from that recognizer's own committed
        /// `negative_context` list stands inside the matcher's context
        /// window. If a future edit removed the term, or the injector
        /// stopped writing it, every one of those cases would silently
        /// become a case that passes for the wrong reason -- masked nothing
        /// because the licence word was out of range either. This asserts
        /// the case is adversarial, which is the property the file is for.
        #[test]
        fn every_context_carried_hard_negative_is_argued_down_by_a_committed_term() {
            /// The matcher's `CONTEXT_WINDOW`, restated: this test measures
            /// the same neighbourhood the recognizer reads.
            const WINDOW: usize = 64;

            let terms: Vec<String> = pii_detection_rules::pack::all()
                .iter()
                .flat_map(|pack| pack.recognizer.iter())
                .filter(|recognizer| {
                    matches!(
                        recognizer.id.as_str(),
                        "nl_rijbewijs" | "fr_permis_conduire" | "it_patente"
                    )
                })
                .flat_map(|recognizer| recognizer.negative_context.iter())
                .map(|term| term.to_lowercase())
                .collect();
            assert!(
                !terms.is_empty(),
                "no context-carried recognizer is loaded: this test would prove nothing"
            );

            let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmark/cases");
            let cases = pii_detection_eval::load(&dir).expect("the case directory loads");
            let mut checked = 0;
            for case in &cases {
                // By id, not by `source`: `source` records which register
                // the carrier came from (`driving-licence:encyclopedia`),
                // and the country is in the id that `build_licences` writes
                // (`<carrier>-dl-nl`).
                //
                // A negative counts here only when its *shape* is one the
                // recognizer accepts: a run the pattern rejects is decided
                // by its shape and needs no vocabulary, which is what the
                // French impossible-month, Dutch nine-digit and Italian
                // ten-letter-word negatives are. Only the ones the pattern
                // accepts are the context-carried case this test is about.
                let licence_shaped: fn(&str) -> bool = if case.id.ends_with("-dl-nl") {
                    |value| value.chars().count() == 10 && value.chars().all(|c| c.is_ascii_digit())
                } else if case.id.ends_with("-dl-fr") {
                    |value| {
                        value.chars().count() == 12
                            && value.chars().all(|c| c.is_ascii_digit())
                            && fr_neph_holds(value)
                    }
                } else if case.id.ends_with("-dl-it") {
                    it_patente_holds
                } else {
                    continue;
                };
                for clear in &case.must_clear {
                    if !licence_shaped(clear) {
                        continue;
                    }
                    // A trunk zero would make an all-digit run a telephone
                    // number, and the phone recognizer would answer for it
                    // whatever this vocabulary said.
                    assert!(
                        !clear.starts_with('0'),
                        "a licence-shaped hard negative begins with a trunk zero"
                    );
                    let Some(at) = case.text.find(clear.as_str()) else {
                        panic!("a hard negative is not in its own case text");
                    };
                    let from = at.saturating_sub(WINDOW);
                    let to = (at + clear.len() + WINDOW).min(case.text.len());
                    let window = case.text[..]
                        .get(from..to)
                        .unwrap_or(&case.text)
                        .to_lowercase();
                    assert!(
                        terms.iter().any(|term| window.contains(term)),
                        "a licence-shaped hard negative carries no suppression term in its own \
                         window"
                    );
                    checked += 1;
                }
            }
            assert!(
                checked >= 30,
                "only {checked} context-carried hard negatives were checked"
            );

            // And the Italian file really does carry the population §3.2
            // measured: ordinary ten-letter upper-case words, which
            // Microsoft Purview's pattern for this identifier matches and
            // `it_patente` cannot. A generator that quietly stopped drawing
            // them would leave this at zero.
            let ten_letter_words = cases
                .iter()
                .filter(|case| case.id.ends_with("-dl-it"))
                .flat_map(|case| case.must_clear.iter())
                .filter(|clear| {
                    clear.chars().count() == 10 && clear.chars().all(|c| c.is_ascii_uppercase())
                })
                .count();
            assert!(
                ten_letter_words >= 10,
                "only {ten_letter_words} ten-letter-word negatives are committed"
            );
        }

        /// The hard negatives are negatives, established rather than assumed.
        ///
        /// Every `must_clear` value in the driving-licence case files that
        /// has a *checksummed or structurally decided* licence's shape must
        /// fail that licence's published rule -- otherwise the fixture is
        /// demanding that a real identifier stay unmasked, which is the
        /// opposite mistake from the one the negatives exist to catch, and a
        /// far worse one.
        ///
        /// THE CONTEXT-CARRIED COUNTRIES ARE DELIBERATELY NOT HELD TO THAT,
        /// because for them it is not a coherent thing to ask. A Dutch
        /// licence number and a ten-digit order number are the same string;
        /// what separates them is the word beside them, and a hard negative
        /// that failed the "published rule" would only be a run of the wrong
        /// length. Their negatives are checked by
        /// `every_context_carried_hard_negative_is_argued_down_by_a_committed_term`
        /// instead, which reads the suppression vocabulary out of the
        /// shipped pack.
        ///
        /// The eleven-digit and sixteen-character negatives that are not
        /// licence-shaped at all are counted too, so that a generator which
        /// quietly stopped producing them fails here instead of leaving the
        /// files a case short.
        #[test]
        fn every_committed_driving_licence_negative_really_is_one() {
            let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmark/cases");
            let cases = pii_detection_eval::load(&dir).expect("the case directory loads");
            let mut checked = 0;
            for case in &cases {
                if !case.source.starts_with("driving-licence:") {
                    continue;
                }
                for clear in &case.must_clear {
                    match clear.chars().count() {
                        11 if clear.chars().all(|c| c.is_ascii_alphanumeric()) => {
                            assert!(
                                !de_fuehrerschein_holds(clear),
                                "a driving-licence hard negative is a valid German licence"
                            );
                            // And it must not be another country's
                            // identifier either, or the case would demand
                            // that a real Steuer-ID or national register
                            // number stay unmasked.
                            assert!(!Validator::GermanTaxId.accepts(clear));
                            assert!(!Validator::BeNationalRegister.accepts(clear));
                            checked += 1;
                        }
                        // Nine digits: one short of the Dutch licence and
                        // exactly a BSN's shape. It must not *be* a BSN, or
                        // the case would be demanding that a real national
                        // identifier stay unmasked.
                        9 if clear.chars().all(|c| c.is_ascii_digit()) => {
                            assert!(
                                !Validator::Elfproef.accepts(clear),
                                "a driving-licence hard negative is a valid BSN"
                            );
                            checked += 1;
                        }
                        15 | 16 | 18 if clear.chars().all(|c| c.is_ascii_alphanumeric()) => {
                            assert!(
                                !gb_driver_number_holds(clear),
                                "a driving-licence hard negative is a valid DVLA driver number"
                            );
                            checked += 1;
                        }
                        _ => {}
                    }
                }
            }
            assert!(
                checked >= 100,
                "only {checked} driving-licence hard negatives were checked"
            );
        }

        /// The verifiers are not vacuous: each accepts the value it is
        /// written for and rejects the mutation that matters most.
        ///
        /// A verifier that returned `false` for everything would make
        /// `every_committed_driving_licence_negative_really_is_one` pass on
        /// its own, and a verifier that returned `true` for everything would
        /// make the guard above it pass. Both directions, here.
        #[test]
        fn the_licence_verifiers_accept_the_published_values_and_reject_their_mutations() {
            // The Musterführerschein, and the DVLA example GOV.UK publishes.
            assert!(de_fuehrerschein_holds("B072RRE2I55"));
            assert!(gb_driver_number_holds("USER9801180MA9PC"));
            assert!(gb_driver_number_holds("MORGA657054SM9IJ"));
            assert!(gb_driver_number_holds("MORGA657054SM9IJ35"));

            // One away on the check digit, and eleven digits.
            assert!(!de_fuehrerschein_holds("B072RRE2I45"));
            assert!(!de_fuehrerschein_holds("42012205181"));
            // Month thirteen, day thirty-two, and a date the calendar does
            // not have -- 31 November.
            assert!(!gb_driver_number_holds("MORGA613054SM9IJ"));
            assert!(!gb_driver_number_holds("MORGA657324SM9IJ"));
            assert!(!gb_driver_number_holds("MORGA611314SM9IJ"));
            // Padding on the wrong side of the surname field.
            assert!(!gb_driver_number_holds("99LEE657054SM9IJ"));
        }

        #[test]
        fn every_committed_tax_id_in_the_fixture_satisfies_the_structure_rule_and_the_check_digit()
        {
            // THE FILTER THIS TEST USED TO CARRY. It loaded all four case
            // files and then kept only cases whose `source` starts with
            // "carrier", so the one hand-authored Steuer-ID in
            // `benchmark/cases/domain.json` -- which passes the ISO
            // 7064 check digit and violates the BZSt structure rule -- was
            // never looked at, while the VAT sibling above carried no such
            // filter. The name is the specification: this checks every
            // committed tax id, declared or not.
            let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmark/cases");
            let cases = pii_detection_eval::load(&dir).expect("the case directory loads");
            let mut declared = 0;
            let mut undeclared = 0;
            for case in &cases {
                for value in &case.must_mask {
                    let text = value.as_str();
                    // A `TAX_ID` is not necessarily a German one any more:
                    // the Italian codice fiscale is a `TAX_ID` too, and is
                    // checked by `every_committed_country_identifier_...`
                    // against its own published rule. Selecting on the
                    // German SHAPE -- eleven ASCII digits, no leading zero --
                    // keeps this guard about the identifier it is written
                    // for, and keeps it selecting on shape rather than on
                    // the property it exists to assert.
                    let german_shape = text.len() == 11
                        && text.chars().all(|character| character.is_ascii_digit())
                        && !text.starts_with('0');
                    match value.entity_type() {
                        Some(EntityType::TaxId) if german_shape => declared += 1,
                        Some(EntityType::TaxId) => continue,
                        // The untyped escape hatch, closed properly this
                        // time: select by SHAPE, not by the very property
                        // this guard exists to assert. Gating the selector
                        // on `Validator::GermanTaxId.accepts(text)` -- as an
                        // earlier version of this arm did -- meant an
                        // untyped Steuer-ID with a wrong check digit was
                        // simply invisible to the guard, since it would
                        // never be selected in the first place. A bare
                        // 11-ASCII-digit string with no leading zero is a
                        // Steuer-ID whatever it is labelled, and has to
                        // satisfy the real rule too. This mirrors the VAT
                        // sibling above, which selects on exact-length,
                        // anchored shape and never on validity.
                        None if german_shape => undeclared += 1,
                        _ => continue,
                    }
                    assert!(
                        steuer_id_structure_holds(text),
                        "a committed Steuer-ID violates the BZSt structure rule"
                    );
                    assert!(
                        Validator::GermanTaxId.accepts(text),
                        "a committed Steuer-ID fails its check digit"
                    );
                }
            }
            assert!(
                declared >= 20,
                "expected the fixture to carry declared Steuer-IDs"
            );
            assert!(
                undeclared >= 1,
                "seed.json's untyped Steuer-ID must be covered by this guard, not skipped by it"
            );
        }
    }

    #[test]
    fn an_injected_telephone_number_is_one_the_numbering_plan_contains() {
        // A number outside the plan is a value no detector should find, and
        // would sit in the fixture as a permanent undeserved miss.
        for (country, region) in [
            ("NL", "NL"),
            ("DE", "DE"),
            ("FR", "FR"),
            ("ES", "ES"),
            ("IT", "IT"),
            ("GB", "GB"),
        ] {
            let mut seen = BTreeSet::new();
            for seed in 0..8 {
                let case = inject("Carrier text.", &[Injection::phone(country)], seed);
                assert!(
                    is_a_real_number(case.must_mask[0].as_str(), region),
                    "{country} number must be in the {region} numbering plan"
                );
                seen.insert(case.must_mask[0].as_str().to_string());
            }
            // The generator falls back to one fixed number when every draw
            // is rejected by the plan. Without this, a country whose shape
            // was wrong would fill the fixture with a single repeated
            // number and still pass the validity check above.
            assert!(
                seen.len() >= 4,
                "{country} numbers must vary, not fall back to one"
            );
        }
    }

    #[test]
    fn digits_pass_as_a_credit_card_catches_the_de_wv_camping_dn_incident() {
        // The exact regression: the detector's engine emitted CREDIT_CARD over
        // `49 151 71053098`, part of the phone number `+49 151 71053098`,
        // because that digit string (4915171053098) is Luhn-valid at 13
        // digits. This is the fixture-side guard that keeps `phone` from
        // ever drawing it again -- fixing the detector's own confusion is
        // out of scope (see the carrier-expansion report).
        assert!(digits_pass_as_a_credit_card("+49 151 71053098"));
        // A leading `+` must not hide a Luhn-valid digit run: `Validator::
        // Luhn` alone would miss this (its own `digits` helper reports "not
        // Luhn" for anything containing a `+`), which is exactly why this
        // function strips non-digits itself rather than delegating whole.
        assert!(!Validator::Luhn.accepts("+49 151 71053098"));
        // A real, unrelated phone number must not be flagged.
        assert!(!digits_pass_as_a_credit_card("+31 6 12345678"));
    }

    #[test]
    fn no_generated_phone_number_is_luhn_valid_at_credit_card_length() {
        // Broad coverage across every supported country and both the
        // international and national template, over many seeds: `phone`
        // must never hand back a number whose digits alone would pass as a
        // credit card, whether drawn by the retry loop or handed out by the
        // fixed fallback.
        for country in ["NL", "DE", "FR", "ES", "IT", "GB"] {
            for seed in 0..64 {
                let case = inject("Carrier text.", &[Injection::phone(country)], seed);
                let value = case.must_mask[0].as_str();
                assert!(
                    !digits_pass_as_a_credit_card(value),
                    "{country} seed {seed} drew a phone number whose digits pass Luhn at card \
                     length"
                );
            }
        }
    }

    #[test]
    fn phone_fallbacks_are_never_luhn_valid_credit_card_shapes() {
        // The fixed fallback strings `phone` returns after 64 failed draws,
        // pinned directly rather than only reached indirectly through the
        // property test above (which may never exhaust the retry loop).
        for fallback in [
            "+31 6 12345678",
            "+49 151 12345679",
            "+33 6 12 34 56 78",
            "+34 612 34 56 78",
            "+39 320 123 4567",
            "+44 7911 123457",
        ] {
            assert!(
                !digits_pass_as_a_credit_card(fallback),
                "fallback {fallback} is Luhn-valid at card length"
            );
        }
    }

    #[test]
    fn the_committed_carrier_cases_are_coherent_and_typed() {
        // The fixture this binary produces, checked as committed: every
        // must_mask value is really a substring of its own text, every one
        // declares a type, and every case names the carrier it came from so
        // the training dataset builder can exclude it.
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmark/cases");
        let cases = pii_detection_eval::load(&dir).expect("the case directory loads");
        let carrier_cases: Vec<&Case> = cases
            .iter()
            .filter(|case| {
                case.source.starts_with("carrier") || case.source.starts_with("country-pack")
            })
            .collect();
        assert!(
            carrier_cases.len() >= 100,
            "expected the carrier cases to be committed, found {}",
            carrier_cases.len()
        );
        assert!(check_coherence(&cases).is_empty());
        for case in &carrier_cases {
            assert!(case.carrier_id.is_some(), "{} has no carrier_id", case.id);
            for value in &case.must_mask {
                assert!(
                    value.entity_type().is_some(),
                    "{} has an untyped must_mask value",
                    case.id
                );
            }
        }
    }

    #[test]
    fn the_manifest_the_files_and_the_readme_all_name_the_same_carriers() {
        // Provenance is only worth having if it cannot silently go missing.
        // A carrier file with no manifest row has no licence on record; a
        // manifest row with no file is a case that will not build; and a
        // carrier absent from the README is one nobody reading the corpus is
        // told the source of.
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmark/carriers");
        let manifest: Manifest = serde_json::from_str(
            &std::fs::read_to_string(dir.join("manifest.json")).expect("read"),
        )
        .expect("parse the manifest");
        let readme = std::fs::read_to_string(dir.join("README.md")).expect("read the README");

        let mut on_disk: BTreeSet<String> = BTreeSet::new();
        for entry in std::fs::read_dir(&dir).expect("read the carrier directory") {
            let path = entry.expect("a directory entry").path();
            if path.extension().and_then(|e| e.to_str()) == Some("txt") {
                on_disk.insert(
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .expect("a file name")
                        .to_string(),
                );
            }
        }

        let listed: BTreeSet<String> = manifest
            .carriers
            .iter()
            .map(|carrier| carrier.file.clone())
            .collect();
        assert_eq!(
            listed, on_disk,
            "every carrier file needs a manifest row and every row a file"
        );

        for carrier in &manifest.carriers {
            assert!(
                readme.contains(&carrier.id),
                "{} is not listed in benchmark/carriers/README.md",
                carrier.id
            );
        }
    }

    #[test]
    fn regenerating_the_country_pack_cases_reproduces_the_committed_files() {
        // The same contract the carrier fixture has, per country: the
        // committed file must be exactly what the committed generator
        // produces. A country named in COUNTRY_PACKS with no pack built in
        // yet has no file and is skipped -- the packs land one at a time.
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut built = 0;
        for country in COUNTRY_PACKS {
            if country_pack(country).is_none() {
                continue;
            }
            built += 1;
            let (rebuilt, _) =
                build_country(&root.join("benchmark/carriers"), country, COUNTRY_PACK_SEED)
                    .expect("rebuild succeeds");
            let path = root.join(format!(
                "benchmark/cases/country-{}.json",
                country.to_lowercase()
            ));
            let committed: Vec<Case> = serde_json::from_str(
                &std::fs::read_to_string(&path).expect("read the committed country-pack cases"),
            )
            .expect("parse the committed country-pack cases");
            assert_eq!(rebuilt, committed, "{country}");
        }
        assert!(built > 0, "no country pack was checked");
    }

    #[test]
    fn a_carrier_hosts_at_most_one_country_pack_case() {
        // Case ids are `{carrier}-cp`, so two packs sharing a carrier would
        // collide. `carriers_for` is what prevents it: a carrier's own
        // language belongs to one pack, and each English carrier takes one
        // turn in the rotation.
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let manifest: Manifest = serde_json::from_str(
            &std::fs::read_to_string(root.join("benchmark/carriers/manifest.json")).expect("read"),
        )
        .expect("parse the manifest");
        let mut claimed: BTreeSet<&str> = BTreeSet::new();
        for country in COUNTRY_PACKS {
            let Some(pack) = country_pack(country) else {
                continue;
            };
            for carrier in carriers_for(&manifest, country, &pack) {
                assert!(
                    claimed.insert(carrier.id.as_str()),
                    "{} is claimed by two country packs",
                    carrier.id
                );
            }
        }
    }

    #[test]
    fn regenerating_the_carrier_cases_reproduces_the_committed_file() {
        // The committed fixture must be exactly what the committed generator
        // produces, or the answer key and the tool that wrote it have drifted
        // and nobody can tell which one is right.
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let (rebuilt, _) =
            build(&root.join("benchmark/carriers"), 20260821).expect("rebuild succeeds");
        let committed: Vec<Case> = serde_json::from_str(
            &std::fs::read_to_string(root.join("benchmark/cases/carriers.json"))
                .expect("read the committed cases"),
        )
        .expect("parse the committed cases");
        assert_eq!(rebuilt, committed);
    }
}
