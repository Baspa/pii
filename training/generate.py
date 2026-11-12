"""Generates what neither training corpus covers.

training/reports/corpora-profile.md measured two gaps by streaming every
row of both corpora (see the corpus profile): Irish (`ga`) has 2 rows and Maltese
(`mt`) has 24, across 2.44 million examples -- effectively absent, not
thin. Neither corpus has any Belgian, Spanish, Polish, Italian or French
checksummed national identifier either, so a model trained only on the
corpora never sees a BSN or a codice fiscale in context and has no reason
to believe the shape means anything.

This module fills both gaps by *constructing* text rather than annotating
it. Every record is built by ``Builder``/``_render`` one piece at a time:
plain text is appended, and an entity's exact character span is recorded
at the moment its text is appended. Nothing here ever searches a finished
string for a substring to label. That is deliberate -- searching a
finished string is exactly how the corpora's own alignment noise happens
(a name that also occurs earlier in the sentence, an accented character
that shifts a byte offset but not a character offset), and constructing
the string avoids the whole class of bug by never needing to search at
all. Offsets are character offsets throughout: Python string slicing
(``s[start:end]``) already operates on code points, so no separate
encode/decode step is needed to get that right -- the risk is a generator
that manually re-derives offsets via ``len(s.encode())`` or similar, which
this one never does.

## The STREET_ADDRESS / city rule

Temperate retired a single LOCATION entity type after a real product failure:
one type conflated countries, cities and street addresses, so masking a
home address also masked "Netherlands" and destroyed any answer that
needed it (33% of street addresses found, 0% of postcodes, 25 country and
region names over-masked -- see training/taxonomy.py's module docstring).
STREET_ADDRESS's replacement definition is narrower on purpose: street
lines and postal codes, never cities, countries or regions.

training/taxonomy.py enforces that rule at the *label-mapping* layer: a
corpus label that names a city maps to ``None``, never to
STREET_ADDRESS. This module enforces the same rule at a different layer,
the *span-construction* layer, because a generated address can legally
contain a city in its text (real addresses do) without the city being
allowed inside the STREET_ADDRESS span. ``_address_segments`` is the one
place that risk is handled: it always emits the street line and the
postcode as two separate STREET_ADDRESS entities, with the city rendered
as plain, unlabeled text between them. Every ga/mt template routes
through it rather than building its own address text, so the exclusion is
enforced in one place instead of trusted to be repeated correctly in
every template. tests/test_generate.py tests this directly (on
``_address_segments`` in isolation) and across the generated corpus (by
scanning every record for a city name string and asserting no
STREET_ADDRESS span covers it).

## What Faker's `ga_IE` and `mt_MT` locales actually provide

The plan that produced this task said both locales were "verified" to
exist, which is true -- but locale *existence* and locale *provider
coverage* are different claims, and the second one is what training text
actually needs. Measured against this Faker install (40.36.0):

- `ga_IE` ships a `person` provider, but it is only half genuine. Its
  surnames are ~96% real Irish patronymic/matronymic forms (``Ó
  Mearlaigh``, ``Mac Conaonaigh``, ``Ó Ciaráin``) -- but only ever with
  the masculine `Ó`/`Mac` particle; there is no way to ask this provider
  for the feminine `Ní`/`Uí`/`Nic` form a woman's surname requires, so
  pairing it with a feminine first name produces an ungrammatical name.
  And its *given* names are anglicised, not Irish: `Roisin`, `Grainne`,
  `Triona` instead of `Róisín`, `Gráinne`, `Tríona` -- measured at 0% of
  given-name tokens carrying a fada across 600 records from an earlier
  version of this module that used them. (An earlier draft of this
  docstring claimed `ga_IE` ships "genuine Irish given names." That
  claim was asserted, not measured, and it was wrong -- see below for
  what replaced it.) It ships no `address`, `company`, `job` or
  `phone_number` provider at all either; calling `Faker('ga_IE').address()`
  silently falls back to Faker's generic (American-shaped) default:
  ``'391 Landy Brooks Apt. 806\\nSouth Oonagh, AK 69658'`` -- a US-style
  apartment/state/ZIP pattern with no relationship to Ireland.
- `mt_MT` ships nothing usable at all for this task: its only provider is
  `ssn`, and that provider only generates Maltese-shaped *VAT* IDs (a
  type this task is explicitly told not to generate). There is no
  `person`, `address`, `company`, `job` or `phone_number` provider for
  Maltese in this Faker version. `Faker('mt_MT').name()` silently returns
  an ordinary American name (``Ryan Gallagher``, ``Jon Cole``).

Per our own rule -- "say so rather than generating English
text labelled as Irish, [because] that would be worse than generating
nothing, because it would look like coverage" -- this module does not use
any Faker provider for `ga` or `mt` at all (an earlier version used
`ga_IE`'s person provider for `ga`; that was corrected once the anglicised
given names and masculine-only surnames were measured, not just assumed
fine because a real name pool was involved). Instead:

- `ga` PERSON entities come from a hand-curated, gender-marked pool of
  Irish given names (real Irish orthography, diacritics only where the
  word genuinely carries one) paired with a hand-checked (`Ó`/`Mac`,
  `Ní`/`Nic`) surname form -- see `GA_FIRST_NAMES_MALE`/`_FEMALE` and
  `GA_SURNAMES`.
- `ga` STREET_ADDRESS entities are built from a small hand-curated pool of
  real Irish street-name patterns, county/city names (in Irish), and
  Eircode-format postcodes, with the routing-key prefix keyed to the
  chosen city (`GA_EIRCODE_PREFIX_BY_PLACE`) so it can't name the wrong
  place -- real Eircode routing-key prefixes identify a small area rather
  than a person, combined with a random synthetic unique identifier.
- `mt` PERSON entities come from a hand-curated, gender-marked pool of
  common Maltese given names and surnames (the same kind of "real name
  pool, synthetic combination" Faker itself uses for every other locale).
- `mt` STREET_ADDRESS entities come from a hand-curated pool of real
  Maltese street names, localities, and postcode-format postcodes (real
  locality postcode prefixes with a random synthetic 4-digit suffix).

See training/reports/generated.md for the full accounting and the
verbatim examples this produced, and its "what this is not" section for
an explicit statement of the confidence level of the hand-authored Irish
and Maltese prose (formulaic and administrative in register, not
free-composed literary prose, precisely to keep grammatical risk low
given neither language is generated through a verified provider).

## Checksum-bearing national identifiers

The task brief that produced this file says these algorithms are "already
implemented and tested in crates/rules/src/validator.rs" and that
porting them here is "translation... not design". That is true for two of
the seven named identifiers -- Dutch BSN (``elfproef``) and German
Steuer-ID (``german_tax_id``) -- which are ported verbatim below and
cross-checked against tests/test_generate.py, which mirrors
validator.rs's own unit tests. It does not hold for the other five
(Belgian national number, Spanish DNI, Polish PESEL, Italian codice
fiscale, French NIR): validator.rs implements exactly five validators
(``None``, ``Elfproef``, ``GermanTaxId``, ``Luhn``, ``IbanMod97``), and no
Belgian, Spanish, Polish, Italian or French rule pack exists yet either
(only `packs/de.toml`, `packs/nl.toml` and `packs/universal.toml` do).
Those five are implemented here from each algorithm's public
specification (the mod-97 Belgian/French checks, the mod-23 Spanish check
letter, the weighted mod-10 Polish check digit, and Italy's odd/even
position-substitution check character), not ported from Rust source that
doesn't exist. Each is independently re-verified in
tests/test_generate.py by a from-scratch verifier function written
separately from the generator, rather than trusted by construction alone.
See training/reports/generated.md for the verification detail per
identifier and training/reports/task-3-report.md for why this
discrepancy from the brief's description is being reported rather than
silently worked around.

Each identifier is generated in prose in its country's own language plus
at least one other language, per the brief. The languages chosen:

- Dutch BSN: `nl`, `en`.
- German Steuer-ID: `de`, `en`.
- Belgian national number: `fr`, `en` -- the brief's own example is "sitting
  in Dutch or French prose"; this module deliberately picks French only
  (not Dutch) so that the `nl` language bucket contains *only* BSNs. That
  keeps `nl` pure for tests/test_generate.py's
  ``test_a_generated_national_id_passes_its_own_checksum``, which asserts
  every NATIONAL_ID span produced for `nl` is a 9-digit value that passes
  the elfproef weights specifically -- a Belgian national number (11
  digits, mod-97) would fail that assertion on shape alone despite being
  a perfectly valid identifier in its own right.
- Spanish DNI: `es`, `en`.
- Polish PESEL: `pl`, `en`.
- Italian codice fiscale: `it`, `en`.
- French NIR: `fr`, `en`.

`en` is used as the second language across all seven deliberately: EU
cross-border correspondence, HR and banking records about a person
frequently get written in English regardless of the person's own
identifier's home country, and English is the one language every one of
these seven countries' residents plausibly also write, unlike a
minority/regional language chosen per-country that would still exclude
most residents. Belgian gets French additionally because that is the
plan's own illustrated case.

## PLATFORM_TYPES used

Only four of the twelve platform types are produced here: ``PERSON`` and
``STREET_ADDRESS`` (the two gaps ga/mt fill), and ``NATIONAL_ID`` /
``TAX_ID`` (what the seven checksummed identifiers map to, matching how
training/taxonomy.py already maps the same *kind* of identifier from both
corpora -- e.g. ``SOCIALNUM``/``SSN`` -> NATIONAL_ID,
``TAXNUM``/``TAX_ID`` -> TAX_ID). IP_ADDRESS and VAT_NUMBER are
deliberately never produced: both are country-prefixed or exactly-shaped
and the detector's deterministic rules already outperform a model on them (see
this task's brief); training the model on types the rules already own
buys nothing and spends signal PERSON and STREET_ADDRESS need instead.
"""

from __future__ import annotations

import random
import unicodedata

from faker import Faker

from taxonomy import PLATFORM_TYPES

Entity = dict[str, object]
Record = dict[str, object]


# ---------------------------------------------------------------------------
# Deterministic seed derivation.
#
# Python's built-in ``hash()`` on strings is randomised per-process unless
# PYTHONHASHSEED is fixed in the environment -- using it here would make
# `build()` non-deterministic across runs even with the same `seed`
# argument, which is exactly what
# tests/test_generate.py::test_generation_is_deterministic_for_a_seed
# checks for. This is a small, fully deterministic string-to-int mix
# instead.
# ---------------------------------------------------------------------------


def _seed_for(seed: int, *parts: str) -> int:
    mixed = seed & 0xFFFFFFFF
    for part in parts:
        for ch in part:
            mixed = (mixed * 1_000_003 + ord(ch)) & 0xFFFFFFFF
    return mixed


# ---------------------------------------------------------------------------
# Builder: construct text and record entity spans as the string is built.
# Never annotate a finished string -- see the module docstring.
# ---------------------------------------------------------------------------


class Builder:
    def __init__(self) -> None:
        self._parts: list[str] = []
        self.entities: list[Entity] = []

    def _pos(self) -> int:
        return sum(len(p) for p in self._parts)

    def text(self, s: str) -> "Builder":
        self._parts.append(s)
        return self

    def entity(self, s: str, label: str) -> "Builder":
        if label not in PLATFORM_TYPES:
            raise ValueError(f"{label!r} is not a platform type")
        start = self._pos()
        self._parts.append(s)
        self.entities.append({"start": start, "end": self._pos(), "label": label})
        return self

    def record(self) -> Record:
        return {"text": "".join(self._parts), "entities": self.entities}


Segment = tuple[str | None, str]


def _render(segments: list[Segment]) -> Record:
    b = Builder()
    for label, s in segments:
        if label is None:
            b.text(s)
        else:
            b.entity(s, label)
    return b.record()


def _address_segments(street: str, city: str, postcode: str) -> list[Segment]:
    """The one place every ga/mt template builds an address. Two
    STREET_ADDRESS spans -- the street line and the postcode -- with the
    city as plain, unlabeled text between them. See the module docstring's
    "STREET_ADDRESS / city rule" section for why the city must never be
    inside either span.
    """
    return [
        ("STREET_ADDRESS", street),
        (None, f", {city}, "),
        ("STREET_ADDRESS", postcode),
    ]


# ---------------------------------------------------------------------------
# Checksum-bearing national identifiers.
# ---------------------------------------------------------------------------


def _bsn(rng: random.Random) -> str:
    """Dutch BSN -- the elfproef. Ported verbatim from
    crates/rules/src/validator.rs::elfproef: nine digits, weights
    [9, 8, 7, 6, 5, 4, 3, 2, -1], weighted sum divisible by 11.

    Generation solves the same equation elfproef checks: pick the first
    eight digits freely, then the ninth digit is forced (weight -1) to be
    exactly the value that makes the weighted sum a multiple of 11. If
    that forced value isn't a single digit (remainder 10), redraw -- about
    1 in 11 draws.
    """
    weights = [9, 8, 7, 6, 5, 4, 3, 2]
    while True:
        digits = [rng.randint(0, 9) for _ in range(8)]
        check = sum(w * d for w, d in zip(weights, digits)) % 11
        if check <= 9:
            digits.append(check)
            return "".join(map(str, digits))


def _verify_bsn(value: str) -> bool:
    """Independent re-check of _bsn's output, transcribed from
    validator.rs::elfproef separately from the generator above so a
    transcription slip in one is unlikely to be repeated in the other."""
    digits = [int(c) for c in value if c.isdigit()]
    if len(digits) != 9:
        return False
    weights = [9, 8, 7, 6, 5, 4, 3, 2, -1]
    return sum(w * d for w, d in zip(weights, digits)) % 11 == 0


def _steuer_id(rng: random.Random) -> str:
    """German Steuer-ID -- ISO 7064 MOD 11,10. Ported verbatim from
    crates/rules/src/validator.rs::german_tax_id: the same
    product-carry loop the Rust verifier runs is run here over ten random
    leading digits (no leading zero) to compute the eleventh, check digit
    -- so the value it produces is, by construction, whatever
    german_tax_id would compute as correct for those ten digits.
    """
    digits = [rng.randint(1, 9)] + [rng.randint(0, 9) for _ in range(9)]
    product = 10
    for d in digits:
        s = (d + product) % 10
        if s == 0:
            s = 10
        product = (s * 2) % 11
    check = (11 - product) % 10
    digits.append(check)
    return "".join(map(str, digits))


def _verify_steuer_id(value: str) -> bool:
    """Independent re-check, transcribed a second time from
    validator.rs::german_tax_id rather than reusing _steuer_id's loop."""
    digits = [int(c) for c in value if c.isdigit()]
    if len(digits) != 11:
        return False
    product = 10
    for d in digits[:10]:
        s = (d + product) % 10
        if s == 0:
            s = 10
        product = (s * 2) % 11
    check = (11 - product) % 10
    return check == digits[10]


def _belgian_national_number(rng: random.Random) -> str:
    """Belgian national register number (Rijksregisternummer / numéro de
    registre national): YY.MM.DD-SSS.CC. Not in validator.rs -- no Belgian
    rule pack exists yet. Implemented from the standard public mod-97
    specification: the check is 97 minus the remainder of the first nine
    digits (prefixed with '2' for a 2000-or-later birth year, to
    distinguish it from the equivalent 1900s date) modulo 97.
    """
    year = rng.randint(1945, 2015)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    seq = rng.randint(1, 997)
    yy = year % 100
    base9 = f"{yy:02d}{month:02d}{day:02d}{seq:03d}"
    n = int(("2" + base9) if year >= 2000 else base9)
    check = 97 - (n % 97)
    if check == 0:
        check = 97
    return f"{yy:02d}.{month:02d}.{day:02d}-{seq:03d}.{check:02d}"


def _verify_belgian_national_number(value: str) -> bool:
    """Independent re-check. The number alone doesn't self-encode which
    century its owner was born in, so -- exactly as real-world validators
    for this number do -- this tries both readings and accepts either."""
    digits = "".join(c for c in value if c.isdigit())
    if len(digits) != 11:
        return False
    base9, check = digits[:9], int(digits[9:11])
    for candidate in (base9, "2" + base9):
        n = int(candidate)
        expected = 97 - (n % 97)
        if expected == 0:
            expected = 97
        if expected == check:
            return True
    return False


_DNI_LETTERS = "TRWAGMYFPDXBNJZSQVHLCKE"


def _dni(rng: random.Random) -> str:
    """Spanish DNI: eight digits plus a check letter, the digits taken
    modulo 23 indexing a fixed 23-letter table. Not in validator.rs -- no
    Spanish rule pack exists yet. Standard public specification.
    """
    number = rng.randint(0, 99_999_999)
    letter = _DNI_LETTERS[number % 23]
    return f"{number:08d}{letter}"


def _verify_dni(value: str) -> bool:
    digits = "".join(c for c in value if c.isdigit())
    letters = "".join(c for c in value if c.isalpha())
    if len(digits) != 8 or len(letters) != 1:
        return False
    return _DNI_LETTERS[int(digits) % 23] == letters.upper()


_PESEL_WEIGHTS = [1, 3, 7, 9, 1, 3, 7, 9, 1, 3]


def _pesel(rng: random.Random) -> str:
    """Polish PESEL: eleven digits -- YYMMDD (month century-encoded),
    a four-digit serial, and a weighted mod-10 check digit over the first
    ten. Not in validator.rs -- no Polish rule pack exists yet. Standard
    public specification.
    """
    year = rng.choice([rng.randint(1940, 1999), rng.randint(2000, 2029)])
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    yy = year % 100
    encoded_month = month + (20 if year >= 2000 else 0)
    serial = rng.randint(0, 999)
    gender_digit = rng.randint(0, 9)
    ten = f"{yy:02d}{encoded_month:02d}{day:02d}{serial:03d}{gender_digit:d}"
    digits = [int(c) for c in ten]
    check = (10 - sum(w * d for w, d in zip(_PESEL_WEIGHTS, digits)) % 10) % 10
    return ten + str(check)


def _verify_pesel(value: str) -> bool:
    digits = [int(c) for c in value if c.isdigit()]
    if len(digits) != 11:
        return False
    check = (10 - sum(w * d for w, d in zip(_PESEL_WEIGHTS, digits[:10])) % 10) % 10
    return check == digits[10]


def _strip_accents(word: str) -> str:
    decomposed = unicodedata.normalize("NFKD", word)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


_CF_VOWELS = set("AEIOU")


def _cf_consonant_code(word: str, is_given_name: bool) -> str:
    """The name -> 3-letter code rule at the heart of an Italian codice
    fiscale: consonants first (in order), then vowels, then pad with 'X'.
    A given name with four or more consonants skips the second one (an
    idiosyncrasy of the real algorithm, there specifically to make first
    and last name codes derive differently)."""
    letters = [c for c in _strip_accents(word).upper() if c.isalpha() and c.isascii()]
    consonants = [c for c in letters if c not in _CF_VOWELS]
    vowels = [c for c in letters if c in _CF_VOWELS]
    if is_given_name and len(consonants) >= 4:
        chosen = [consonants[0], consonants[2], consonants[3]]
    else:
        chosen = consonants[:3]
    chosen = chosen + vowels[: max(0, 3 - len(chosen))]
    chosen = chosen + ["X"] * max(0, 3 - len(chosen))
    return "".join(chosen[:3])


_CF_MONTH_CODE = {
    1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "H",
    7: "L", 8: "M", 9: "P", 10: "R", 11: "S", 12: "T",
}

# Real Italian "codice catastale" comune codes -- public geographic
# codes, not personal data, the same kind of fact a postcode is.
IT_COMUNES: dict[str, str] = {
    "Roma": "H501", "Milano": "F205", "Napoli": "F839", "Torino": "L219",
    "Palermo": "G273", "Genova": "D969", "Bologna": "A944", "Firenze": "D612",
    "Bari": "A662", "Catania": "C351", "Venezia": "L736", "Verona": "L781",
    "Padova": "G224", "Trieste": "L424",
}

_CF_ODD = {
    "0": 1, "1": 0, "2": 5, "3": 7, "4": 9, "5": 13, "6": 15, "7": 17, "8": 19, "9": 21,
    "A": 1, "B": 0, "C": 5, "D": 7, "E": 9, "F": 13, "G": 15, "H": 17, "I": 19, "J": 21,
    "K": 2, "L": 4, "M": 18, "N": 20, "O": 11, "P": 3, "Q": 6, "R": 8, "S": 12, "T": 14,
    "U": 16, "V": 10, "W": 22, "X": 25, "Y": 24, "Z": 23,
}
_CF_EVEN = {str(d): d for d in range(10)} | {chr(ord("A") + i): i for i in range(26)}
_CF_REMAINDER_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _cf_check_letter(code15: str) -> str:
    total = sum(
        _CF_ODD[ch] if i % 2 == 0 else _CF_EVEN[ch] for i, ch in enumerate(code15)
    )
    return _CF_REMAINDER_LETTERS[total % 26]


def _codice_fiscale(rng: random.Random, full_name: str) -> str:
    """Italian codice fiscale: 6 letters derived from the surname and
    given name, 2 digits birth year, 1 letter birth month, 2 digits birth
    day (+40 for a woman), 4 characters birthplace code, 1 check letter.
    Not in validator.rs -- no Italian rule pack exists yet. The 15-body
    derivation and the odd/even check-character tables are the standard
    public specification.
    """
    parts = full_name.split()
    given = parts[0]
    surname = parts[-1] if len(parts) > 1 else parts[0]
    surname_code = _cf_consonant_code(surname, is_given_name=False)
    given_code = _cf_consonant_code(given, is_given_name=True)
    year = rng.randint(1945, 2005) % 100
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    if rng.random() < 0.5:
        day += 40
    _city, comune_code = rng.choice(list(IT_COMUNES.items()))
    code15 = (
        f"{surname_code}{given_code}{year:02d}{_CF_MONTH_CODE[month]}{day:02d}{comune_code}"
    )
    return code15 + _cf_check_letter(code15)


def _verify_codice_fiscale(value: str) -> bool:
    if len(value) != 16:
        return False
    return _cf_check_letter(value[:15]) == value[15]


def _nir(rng: random.Random) -> str:
    """French NIR / numéro de sécurité sociale (INSEE number): 13 digits
    (sex, year, month, department, commune, order) plus a 2-digit key,
    97 minus the 13-digit number modulo 97. Not in validator.rs -- no
    French rule pack exists yet. Standard public specification;
    department 20 (Corsica, letter-coded 2A/2B) is deliberately excluded
    to keep the digit encoding unambiguous.
    """
    sex = rng.choice([1, 2])
    year = rng.randint(1945, 2010) % 100
    month = rng.randint(1, 12)
    dept = rng.choice([d for d in range(1, 96) if d != 20])
    commune = rng.randint(1, 999)
    order = rng.randint(1, 999)
    base = f"{sex}{year:02d}{month:02d}{dept:02d}{commune:03d}{order:03d}"
    key = 97 - (int(base) % 97)
    # base is 13 digits: sex(1) year(2) month(2) dept(2) commune(3) order(3).
    # The display grouping below must match that partition -- an earlier
    # version sliced it 1/2/2/2/2/3/1, which still produced a
    # checksum-valid value (the checksum is computed over `base` before
    # this formatting step) but grouped the digits in a way no real NIR
    # ever would. Caught by inspecting generated.md's examples, not by a
    # test: the checksum verifier only ever sees the digits, not their
    # spacing.
    return f"{base[0]} {base[1:3]} {base[3:5]} {base[5:7]} {base[7:10]} {base[10:13]} {key:02d}"


def _verify_nir(value: str) -> bool:
    digits = "".join(c for c in value if c.isdigit())
    if len(digits) != 15:
        return False
    base, key = digits[:13], int(digits[13:15])
    return 97 - (int(base) % 97) == key


# ---------------------------------------------------------------------------
# Irish (ga): PERSON and STREET_ADDRESS both come from hand-curated pools
# below -- no Faker provider is used for `ga` at all. GA_FIRST_NAMES_MALE/
# _FEMALE and GA_SURNAMES replaced Faker's `ga_IE` person provider (see
# the module docstring's "What Faker's `ga_IE` and `mt_MT` locales
# actually provide" section for why: its given names are anglicised and
# its surnames only ever carry the masculine particle).
# ---------------------------------------------------------------------------

GA_STREET_NAMES = [
    "Sráid Uí Chonaill", "Bóthar na Trá", "Ascaill na gCrann", "Cearnóg Mhuirfean",
    "Cé na hAbhann", "Bóthar an Rí", "Sráid na hEaglaise", "Cosán na Coille",
    "Bóthar na Sionna", "Sráid an Droichid", "Ascaill Phádraig", "Bóthar an Chuain",
    "Sráid na Mainistreach", "Cearnóg Parnell", "Bóthar na gCloch",
]

# Real Eircode routing-key prefixes, one per place -- keyed to the
# locality the way the Maltese postcode logic below already is, so a
# generated Eircode can never point at the wrong city. An earlier version
# picked the routing key and the city independently and could produce
# "Gaillimh, D15" -- D15 is a Dublin routing key, not a Galway one.
GA_EIRCODE_PREFIX_BY_PLACE: dict[str, str] = {
    "Baile Átha Cliath": "D02",
    "Corcaigh": "T12",
    "Gaillimh": "H91",
    "Luimneach": "V94",
    "Port Láirge": "X91",
    "Trá Lí": "V92",
    "Sligeach": "F91",
    "Dún Dealgan": "A91",
    "Cill Chainnigh": "R95",
    "Loch Garman": "Y35",
    "Baile Átha Luain": "N37",
    "Droichead Átha": "A92",
    "An Cabhán": "H12",
    "Leitir Ceanainn": "F92",
    "Inis": "V95",
}

GA_PLACES = list(GA_EIRCODE_PREFIX_BY_PLACE.keys())


def _ga_eircode(rng: random.Random, city: str) -> str:
    prefix = GA_EIRCODE_PREFIX_BY_PLACE[city]
    suffix = "".join(rng.choice("ACDEFHKNPRTVWXY0123456789") for _ in range(4))
    return f"{prefix} {suffix}"


def _ga_street_address(rng: random.Random) -> tuple[str, str, str]:
    number = rng.randint(1, 350)
    street = f"{number} {rng.choice(GA_STREET_NAMES)}"
    city = rng.choice(GA_PLACES)
    postcode = _ga_eircode(rng, city)
    return street, city, postcode


# Hand-curated, gender-marked given names and (male, female) surname
# pairs. An earlier version drew PERSON entities from Faker's `ga_IE`
# person provider; that measurably does not work -- see the module
# docstring's "What Faker's `ga_IE` and `mt_MT` locales actually provide"
# section. In short: `ga_IE`'s given names are anglicised (`Roisin`,
# `Grainne`, `Triona` rather than `Róisín`, `Gráinne`, `Tríona` -- 0% of
# them carry a fada, measured over 600 records), so half of every PERSON
# entity was a token the model already sees millions of times in English.
# And its surnames only ever carry the masculine `Ó`/`Mac` particle --
# there is no way to ask it for `Ní`/`Uí`/`Nic`, so roughly half the
# generated records (whichever got a feminine first name) paired it with
# an ungrammatical masculine surname.
#
# Surnames below are hand-paired, not derived by applying a lenition rule
# to the male form: a rule that mutates initial consonants (Ó Ceallaigh
# -> Ní Cheallaigh, but Ó Néill -> Ní Néill, unchanged, because n doesn't
# lenite in writing) is exactly the kind of algorithmic grammar mutation
# this module has no way to verify against a live speaker, so each pair
# is stored as a literal, checked string instead -- the same reason
# MT_SURNAMES stores full surnames rather than a rule, and for the same
# reason: a hardcoded correct form beats a plausible-looking generated
# one. Like the rest of the hand-authored Irish content, this should get
# native-speaker review before being trusted as gold-standard prose.
GA_FIRST_NAMES_MALE = [
    "Cathal", "Pádraig", "Éamon", "Ruairí", "Diarmaid", "Fearghal", "Oisín",
    "Tadhg", "Cormac", "Darach", "Conchúr", "Niall", "Aodh", "Breandán",
    "Séamus", "Dónal", "Ciarán", "Ruadhán", "Colm", "Fionn",
]

GA_FIRST_NAMES_FEMALE = [
    "Gráinne", "Sinéad", "Aoife", "Órla", "Niamh", "Sorcha", "Caoimhe",
    "Saoirse", "Róisín", "Máire", "Nóra", "Bríd", "Aisling", "Muireann",
    "Clíona", "Éabha", "Neasa", "Deirdre", "Fionnuala", "Gobnait",
]

GA_SURNAMES: list[tuple[str, str]] = [
    ("Ó Murchú", "Ní Mhurchú"),
    ("Ó Ceallaigh", "Ní Cheallaigh"),
    ("Ó Súilleabháin", "Ní Shúilleabháin"),
    ("Ó Riain", "Ní Riain"),
    ("Ó Conaill", "Ní Chonaill"),
    ("Ó Néill", "Ní Néill"),
    ("Ó Broin", "Ní Bhroin"),
    ("Ó Dochartaigh", "Ní Dhochartaigh"),
    ("Ó Gallchobhair", "Ní Ghallchobhair"),
    ("Ó Baoill", "Ní Bhaoill"),
    ("Ó Cléirigh", "Ní Chléirigh"),
    ("Ó Loinsigh", "Ní Loinsigh"),
    ("Ó Braonáin", "Ní Bhraonáin"),
    ("Ó Fearghail", "Ní Fhearghail"),
    ("Mac Cárthaigh", "Nic Cárthaigh"),
    ("Mac Diarmada", "Nic Dhiarmada"),
    ("Mac Mathúna", "Nic Mhathúna"),
    ("Mac Domhnaill", "Nic Dhomhnaill"),
    ("Mac Suibhne", "Nic Shuibhne"),
]


def _ga_name(rng: random.Random) -> tuple[str, str]:
    """Returns (full name, gender) -- "m" or "f" -- so a template that
    needs to agree a greeting or particle with the name it just drew
    can, the same shape `_mt_name` returns for the same reason."""
    gender = rng.choice(("m", "f"))
    first = rng.choice(GA_FIRST_NAMES_MALE if gender == "m" else GA_FIRST_NAMES_FEMALE)
    male_surname, female_surname = rng.choice(GA_SURNAMES)
    surname = male_surname if gender == "m" else female_surname
    return f"{first} {surname}", gender


def _ga_correspondence_1(rng, fakers):
    recipient, _ = _ga_name(rng)
    sender, _ = _ga_name(rng)
    street, city, postcode = _ga_street_address(rng)
    return [
        (None, "A Chara,\n\nScríobhaim chugat thar ceann "),
        ("PERSON", recipient),
        (None, ", a bhfuil cónaí air/uirthi ag "),
        *_address_segments(street, city, postcode),
        (None, ".\n\nLe meas,\n"),
        ("PERSON", sender),
    ]


def _ga_correspondence_2(rng, fakers):
    # Previously opened "A " + PERSON + ", a chara," -- putting the name
    # directly in vocative position, which Irish inflects ("A Phóil, a
    # chara," not "A Paul, a chara,"). GA_SURNAMES above stores nominative
    # forms only (see its docstring: a hardcoded nominative form is
    # verifiably correct; a hardcoded vocative form for every name would
    # need a third hand-checked form per surname plus first-name
    # slenderization this module has no way to verify). Dropped the
    # vocative frame rather than inflect into it: the fixed "A Chara,"
    # salutation used correctly elsewhere in this module never touches
    # the entity, and the customer's name is a plain, ungoverned label
    # instead -- no case or mutation applies to a name after a colon.
    customer, _ = _ga_name(rng)
    signee, _ = _ga_name(rng)
    street, city, postcode = _ga_street_address(rng)
    return [
        (None, "A Chara,\n\nCustaiméir: "),
        ("PERSON", customer),
        (None, "\nDeimhnímid gur fuair muid do litir a seoladh ó "),
        *_address_segments(street, city, postcode),
        (None, ".\nBeimid i dteagmháil leat go luath.\n\nIs mise, le meas,\n"),
        ("PERSON", signee),
    ]


def _ga_clinical_1(rng, fakers):
    patient, _ = _ga_name(rng)
    street, city, postcode = _ga_street_address(rng)
    return [
        (None, "Taifead Cliniciúil\nOthar: "),
        ("PERSON", patient),
        (None, "\nSeoladh cláraithe: "),
        *_address_segments(street, city, postcode),
        # Was "tar éis na coinsíneachta" -- coinsíneacht is a shipping
        # consignment, not a clinical consultation or admission.
        (None, "\nStádas: fágtha faoi bhreathnóireacht tar éis an chomhairliúcháin.\n"),
    ]


def _ga_clinical_2(rng, fakers):
    patient, _ = _ga_name(rng)
    doctor, _ = _ga_name(rng)
    street, city, postcode = _ga_street_address(rng)
    return [
        (None, "Nóta ó "),
        ("PERSON", doctor),
        (None, " maidir le "),
        ("PERSON", patient),
        (None, ", atá cláraithe ag "),
        *_address_segments(street, city, postcode),
        (None, ". Moltar athbhreithniú i gceann coicíse.\n"),
    ]


def _ga_banking_1(rng, fakers):
    # Same vocative fix as _ga_correspondence_2 -- see its comment.
    customer, _ = _ga_name(rng)
    street, city, postcode = _ga_street_address(rng)
    return [
        (None, "A chustaiméir,\n\nCustaiméir: "),
        ("PERSON", customer),
        (None, "\nDeimhnímid gur nuashonraíodh do sheoladh baile chuig "),
        *_address_segments(street, city, postcode),
        (None, " ar do chuntas bainc.\n"),
    ]


def _ga_banking_2(rng, fakers):
    customer, _ = _ga_name(rng)
    street, city, postcode = _ga_street_address(rng)
    return [
        (None, "Ráiteas Cuntais\nSealbhóir cuntais: "),
        ("PERSON", customer),
        (None, "\nSeoladh billeála: "),
        *_address_segments(street, city, postcode),
        (None, "\nIarratas próiseáilte go rathúil.\n"),
    ]


def _ga_hr_1(rng, fakers):
    employee, _ = _ga_name(rng)
    street, city, postcode = _ga_street_address(rng)
    return [
        (None, "Foirm Fostaithe Nua\nAinm: "),
        ("PERSON", employee),
        (None, "\nSeoladh baile: "),
        *_address_segments(street, city, postcode),
        (None, "\nRoinn: Acmhainní Daonna\n"),
    ]


def _ga_hr_2(rng, fakers):
    employee, _ = _ga_name(rng)
    manager, _ = _ga_name(rng)
    street, city, postcode = _ga_street_address(rng)
    return [
        (None, "Cuireann "),
        ("PERSON", manager),
        (None, " in iúl gur thosaigh "),
        ("PERSON", employee),
        (None, " ag obair inniu. Seoladh cláraithe an fhostaí: "),
        *_address_segments(street, city, postcode),
        (None, ".\n"),
    ]


def _ga_logs_1(rng, fakers):
    user, _ = _ga_name(rng)
    street, city, postcode = _ga_street_address(rng)
    ts = _log_timestamp(rng)
    return [
        (None, f"{ts} level=INFO user=\""),
        ("PERSON", user),
        (None, "\" addr=\""),
        *_address_segments(street, city, postcode),
        (None, "\" action=login\n"),
    ]


def _ga_logs_2(rng, fakers):
    user, _ = _ga_name(rng)
    street, city, postcode = _ga_street_address(rng)
    ts = _log_timestamp(rng)
    return [
        (None, f"{ts} level=AUDIT actor=\""),
        ("PERSON", user),
        (None, "\" event=address_updated new_addr=\""),
        *_address_segments(street, city, postcode),
        (None, "\"\n"),
    ]


GA_TEMPLATES = [
    _ga_correspondence_1, _ga_correspondence_2,
    _ga_clinical_1, _ga_clinical_2,
    _ga_banking_1, _ga_banking_2,
    _ga_hr_1, _ga_hr_2,
    _ga_logs_1, _ga_logs_2,
]


def _log_timestamp(rng: random.Random) -> str:
    return (
        f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        f"T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}Z"
    )


# ---------------------------------------------------------------------------
# Maltese (mt): both PERSON and STREET_ADDRESS come from hand-curated
# pools -- Faker has no usable provider for this locale at all (see the
# module docstring).
# ---------------------------------------------------------------------------

MT_FIRST_NAMES_MALE = [
    "Ġużeppi", "Karmenu", "Pawlu", "Toni", "Manwel", "Marlon", "Kurt", "Andrè",
    "Cliff", "Reuben",
]

MT_FIRST_NAMES_FEMALE = [
    "Marija", "Rita", "Ċensa", "Rożanna", "Nadia", "Yana", "Denise",
    "Graziella", "Charlene", "Miriam",
]

MT_SURNAMES = [
    "Borg", "Camilleri", "Vella", "Zammit", "Micallef", "Farrugia", "Attard",
    "Grech", "Mifsud", "Cassar", "Debono", "Galea", "Sciberras", "Xuereb",
    "Buttigieg",
]

MT_STREET_NAMES = [
    "Triq ir-Repubblika", "Triq il-Merkanti", "Triq San Pawl", "Triq il-Kbira",
    "Triq il-Kostituzzjoni", "Triq Nofsinhar", "Triq il-Baħar", "Triq il-Knisja",
    "Triq is-Sultan", "Sqaq il-Kavallieri", "Triq il-Ġdida", "Triq il-Mitħna",
    "Triq San Ġorġ", "Triq il-Kunsill", "Triq iż-Żejtun",
]

# Real Maltese postal locality prefixes -- public geographic codes.
MT_POSTCODE_PREFIXES: dict[str, str] = {
    "Valletta": "VLT", "Birkirkara": "BKR", "Mosta": "MST", "Sliema": "SLM",
    "Qormi": "QRM", "Rabat": "RBT", "Żebbuġ": "ZBB", "Ħamrun": "HMR",
    "Victoria": "VCT", "Naxxar": "NXR", "Żabbar": "ZBR", "Paola": "PLA",
    "Marsaskala": "MSK", "Attard": "ATD",
}

MT_PLACES = list(MT_POSTCODE_PREFIXES.keys())


def _mt_name(rng: random.Random) -> tuple[str, str]:
    """Returns (full name, gender) -- "m" or "f" -- so a template that
    needs to agree a greeting with the name it just drew can (see
    _mt_correspondence_2/_mt_banking_1 below)."""
    gender = rng.choice(("m", "f"))
    first = rng.choice(MT_FIRST_NAMES_MALE if gender == "m" else MT_FIRST_NAMES_FEMALE)
    return f"{first} {rng.choice(MT_SURNAMES)}", gender


def _mt_street_address(rng: random.Random) -> tuple[str, str, str]:
    number = rng.randint(1, 250)
    street = f"{number}, {rng.choice(MT_STREET_NAMES)}"
    city = rng.choice(MT_PLACES)
    postcode = f"{MT_POSTCODE_PREFIXES[city]} {rng.randint(1000, 9999)}"
    return street, city, postcode


def _mt_correspondence_1(rng, fakers):
    recipient, _ = _mt_name(rng)
    sender, _ = _mt_name(rng)
    street, city, postcode = _mt_street_address(rng)
    return [
        (None, "Għażiż/Għażiża,\n\nQed nikteb dwar "),
        ("PERSON", recipient),
        (None, ", li joqgħod/toqgħod fi "),
        *_address_segments(street, city, postcode),
        (None, ".\n\nGrazzi,\n"),
        ("PERSON", sender),
    ]


def _mt_correspondence_2(rng, fakers):
    # Was unconditionally "Għażiż " -- masculine-only -- against a name
    # pool that is about half feminine, giving "Għażiż Marija Borg".
    # _mt_correspondence_1 hedges with "Għażiż/Għażiża"; here the name's
    # gender is already tracked (see _mt_name), so select the correct
    # form directly instead.
    customer, gender = _mt_name(rng)
    greeting = "Għażiż" if gender == "m" else "Għażiża"
    street, city, postcode = _mt_street_address(rng)
    return [
        (None, f"{greeting} "),
        ("PERSON", customer),
        (None, ",\n\nGrazzi tal-ittra tiegħek mibgħuta minn "),
        *_address_segments(street, city, postcode),
        (None, ". Se nikkuntattjawk dalwaqt.\n"),
    ]


def _mt_clinical_1(rng, fakers):
    patient, _ = _mt_name(rng)
    street, city, postcode = _mt_street_address(rng)
    return [
        (None, "Rapport Kliniku\nPazjent: "),
        ("PERSON", patient),
        (None, "\nIndirizz reġistrat: "),
        *_address_segments(street, city, postcode),
        (None, "\nStat: taħt osservazzjoni.\n"),
    ]


def _mt_clinical_2(rng, fakers):
    patient, _ = _mt_name(rng)
    doctor, _ = _mt_name(rng)
    street, city, postcode = _mt_street_address(rng)
    return [
        (None, "Nota mingħand Dr "),
        ("PERSON", doctor),
        (None, " dwar "),
        ("PERSON", patient),
        (None, ", irreġistrat/a f'"),
        *_address_segments(street, city, postcode),
        (None, ".\n"),
    ]


def _mt_banking_1(rng, fakers):
    # Same gender-agreement fix as _mt_correspondence_2 -- see its
    # comment.
    customer, gender = _mt_name(rng)
    greeting = "Għażiż" if gender == "m" else "Għażiża"
    street, city, postcode = _mt_street_address(rng)
    return [
        (None, f"{greeting} klijent "),
        ("PERSON", customer),
        (None, ",\n\nNikkonfermaw li l-indirizz fuq il-kont tiegħek ġie aġġornat għal "),
        *_address_segments(street, city, postcode),
        (None, ".\n"),
    ]


def _mt_banking_2(rng, fakers):
    customer, _ = _mt_name(rng)
    street, city, postcode = _mt_street_address(rng)
    return [
        (None, "Rendikont tal-Kont\nDetentur: "),
        ("PERSON", customer),
        (None, "\nIndirizz tal-fatturazzjoni: "),
        *_address_segments(street, city, postcode),
        (None, "\n"),
    ]


def _mt_hr_1(rng, fakers):
    employee, _ = _mt_name(rng)
    street, city, postcode = _mt_street_address(rng)
    return [
        (None, "Formola ta' Impjegat Ġdid\nIsem: "),
        ("PERSON", employee),
        (None, "\nIndirizz tad-dar: "),
        *_address_segments(street, city, postcode),
        (None, "\nDipartiment: Riżorsi Umani\n"),
    ]


def _mt_hr_2(rng, fakers):
    employee, _ = _mt_name(rng)
    manager, _ = _mt_name(rng)
    street, city, postcode = _mt_street_address(rng)
    return [
        (None, "Konferma minn "),
        ("PERSON", manager),
        (None, " li "),
        ("PERSON", employee),
        (None, " beda x-xogħol illum. Indirizz reġistrat: "),
        *_address_segments(street, city, postcode),
        (None, ".\n"),
    ]


def _mt_logs_1(rng, fakers):
    user, _ = _mt_name(rng)
    street, city, postcode = _mt_street_address(rng)
    ts = _log_timestamp(rng)
    return [
        (None, f"{ts} level=INFO user=\""),
        ("PERSON", user),
        (None, "\" indirizz=\""),
        *_address_segments(street, city, postcode),
        (None, "\" azzjoni=login\n"),
    ]


def _mt_logs_2(rng, fakers):
    user, _ = _mt_name(rng)
    street, city, postcode = _mt_street_address(rng)
    ts = _log_timestamp(rng)
    return [
        (None, f"{ts} level=AUDIT actor=\""),
        ("PERSON", user),
        (None, "\" event=indirizz_aġġornat indirizz_ġdid=\""),
        *_address_segments(street, city, postcode),
        (None, "\"\n"),
    ]


MT_TEMPLATES = [
    _mt_correspondence_1, _mt_correspondence_2,
    _mt_clinical_1, _mt_clinical_2,
    _mt_banking_1, _mt_banking_2,
    _mt_hr_1, _mt_hr_2,
    _mt_logs_1, _mt_logs_2,
]


# ---------------------------------------------------------------------------
# Checksum-identifier templates, per language.
# ---------------------------------------------------------------------------


def _nl_name(fk: Faker) -> str:
    return f"{fk.first_name()} {fk.last_name()}"


def _nl_bsn_1(rng, fakers):
    fk = fakers["nl_NL"]
    name = _nl_name(fk)
    return [
        (None, "Cliëntgegevens\nNaam: "),
        ("PERSON", name),
        (None, "\nBSN: "),
        ("NATIONAL_ID", _bsn(rng)),
        (None, "\nVerwerkt op aanvraag van de klant.\n"),
    ]


def _nl_bsn_2(rng, fakers):
    fk = fakers["nl_NL"]
    name = _nl_name(fk)
    signee = _nl_name(fk)
    return [
        (None, "Geachte heer/mevrouw "),
        ("PERSON", name),
        (None, ",\n\nWij bevestigen dat uw burgerservicenummer "),
        ("NATIONAL_ID", _bsn(rng)),
        (None, " is geregistreerd in ons systeem.\n\nMet vriendelijke groet,\n"),
        ("PERSON", signee),
    ]


NL_TEMPLATES = [_nl_bsn_1, _nl_bsn_2]


def _de_name(fk: Faker) -> str:
    return f"{fk.first_name()} {fk.last_name()}"


def _de_steuer_1(rng, fakers):
    fk = fakers["de_DE"]
    name = _de_name(fk)
    return [
        (None, "Personalakte\nName: "),
        ("PERSON", name),
        (None, "\nSteuer-ID: "),
        ("TAX_ID", _steuer_id(rng)),
        (None, "\nAbteilung: Personalwesen\n"),
    ]


def _de_steuer_2(rng, fakers):
    fk = fakers["de_DE"]
    name = _de_name(fk)
    return [
        (None, "Sehr geehrte/r "),
        ("PERSON", name),
        (None, ",\n\nIhre Steueridentifikationsnummer lautet "),
        ("TAX_ID", _steuer_id(rng)),
        (None, ". Bitte bestätigen Sie den Erhalt.\n"),
    ]


DE_TEMPLATES = [_de_steuer_1, _de_steuer_2]


def _fr_belgian_1(rng, fakers):
    fk = fakers["fr_BE"]
    name = f"{fk.first_name()} {fk.last_name()}"
    return [
        (None, "Dossier client\nNom : "),
        ("PERSON", name),
        (None, "\nNuméro de registre national : "),
        ("NATIONAL_ID", _belgian_national_number(rng)),
        (None, "\nAgence : Bruxelles\n"),
    ]


def _fr_belgian_2(rng, fakers):
    fk = fakers["fr_BE"]
    name = f"{fk.first_name()} {fk.last_name()}"
    return [
        (None, "Madame, Monsieur "),
        ("PERSON", name),
        (None, ",\n\nVotre numéro de registre national "),
        ("NATIONAL_ID", _belgian_national_number(rng)),
        (None, " a bien été mis à jour auprès de la commune.\n"),
    ]


def _fr_nir_1(rng, fakers):
    fk = fakers["fr_FR"]
    name = f"{fk.first_name()} {fk.last_name()}"
    return [
        (None, "Madame, Monsieur "),
        ("PERSON", name),
        (None, ",\n\nVotre numéro de sécurité sociale "),
        ("NATIONAL_ID", _nir(rng)),
        (None, " a bien été enregistré auprès de la CPAM.\n"),
    ]


def _fr_nir_2(rng, fakers):
    fk = fakers["fr_FR"]
    name = f"{fk.first_name()} {fk.last_name()}"
    return [
        (None, "Dossier patient\nNom : "),
        ("PERSON", name),
        (None, "\nNIR : "),
        ("NATIONAL_ID", _nir(rng)),
        (None, "\nService : admissions\n"),
    ]


FR_TEMPLATES = [_fr_belgian_1, _fr_belgian_2, _fr_nir_1, _fr_nir_2]


def _es_dni_1(rng, fakers):
    fk = fakers["es_ES"]
    name = f"{fk.first_name()} {fk.last_name()}"
    return [
        (None, "Formulario de Recursos Humanos\nNombre: "),
        ("PERSON", name),
        (None, "\nDNI: "),
        ("NATIONAL_ID", _dni(rng)),
        (None, "\nDepartamento: Recursos Humanos\n"),
    ]


def _es_dni_2(rng, fakers):
    fk = fakers["es_ES"]
    name = f"{fk.first_name()} {fk.last_name()}"
    return [
        (None, "Estimado/a "),
        ("PERSON", name),
        (None, ",\n\nConfirmamos que su Documento Nacional de Identidad "),
        ("NATIONAL_ID", _dni(rng)),
        (None, " ha sido verificado correctamente.\n"),
    ]


ES_TEMPLATES = [_es_dni_1, _es_dni_2]


def _pl_pesel_1(rng, fakers):
    fk = fakers["pl_PL"]
    name = f"{fk.first_name()} {fk.last_name()}"
    return [
        (None, "Karta pracownika\nImię i nazwisko: "),
        ("PERSON", name),
        (None, "\nPESEL: "),
        ("NATIONAL_ID", _pesel(rng)),
        (None, "\nDział: Kadry\n"),
    ]


def _pl_pesel_2(rng, fakers):
    fk = fakers["pl_PL"]
    name = f"{fk.first_name()} {fk.last_name()}"
    return [
        (None, "Szanowna Pani, Szanowny Panie "),
        ("PERSON", name),
        (None, ",\n\nInformujemy, że numer PESEL "),
        ("NATIONAL_ID", _pesel(rng)),
        (None, " został zweryfikowany w systemie bankowym.\n"),
    ]


PL_TEMPLATES = [_pl_pesel_1, _pl_pesel_2]


def _it_cf_1(rng, fakers):
    fk = fakers["it_IT"]
    name = f"{fk.first_name()} {fk.last_name()}"
    return [
        (None, "Registro pazienti\nPaziente: "),
        ("PERSON", name),
        (None, "\nCodice fiscale: "),
        ("TAX_ID", _codice_fiscale(rng, name)),
        (None, "\nReparto: Cardiologia\n"),
    ]


def _it_cf_2(rng, fakers):
    fk = fakers["it_IT"]
    name = f"{fk.first_name()} {fk.last_name()}"
    return [
        (None, "Gentile "),
        ("PERSON", name),
        (None, ",\n\nIl suo codice fiscale "),
        ("TAX_ID", _codice_fiscale(rng, name)),
        (None, " risulta correttamente registrato presso l'INPS.\n"),
    ]


IT_TEMPLATES = [_it_cf_1, _it_cf_2]


def _en_bsn(rng, fakers):
    fk = fakers["en_US"]
    name = f"{fk.first_name()} {fk.last_name()}"
    return [
        (None, "Cross-border HR notice\nEmployee: "),
        ("PERSON", name),
        (None, "\nDutch citizen service number (BSN): "),
        ("NATIONAL_ID", _bsn(rng)),
        (None, "\n"),
    ]


def _en_steuer(rng, fakers):
    fk = fakers["en_US"]
    name = f"{fk.first_name()} {fk.last_name()}"
    return [
        (None, "Payroll memo for "),
        ("PERSON", name),
        (None, ". German tax ID on file: "),
        ("TAX_ID", _steuer_id(rng)),
        (None, ".\n"),
    ]


def _en_belgian(rng, fakers):
    fk = fakers["en_US"]
    name = f"{fk.first_name()} {fk.last_name()}"
    return [
        (None, "Relocation file: "),
        ("PERSON", name),
        (None, ", Belgian national register number "),
        ("NATIONAL_ID", _belgian_national_number(rng)),
        (None, ", processed by the Brussels office.\n"),
    ]


def _en_dni(rng, fakers):
    fk = fakers["en_US"]
    name = f"{fk.first_name()} {fk.last_name()}"
    return [
        (None, "KYC check for "),
        ("PERSON", name),
        (None, ": Spanish DNI "),
        ("NATIONAL_ID", _dni(rng)),
        (None, " matched against the register.\n"),
    ]


def _en_pesel(rng, fakers):
    fk = fakers["en_US"]
    name = f"{fk.first_name()} {fk.last_name()}"
    return [
        (None, "Immigration case notes: "),
        ("PERSON", name),
        (None, " holds Polish PESEL "),
        ("NATIONAL_ID", _pesel(rng)),
        (None, ".\n"),
    ]


def _en_cf(rng, fakers):
    fk = fakers["en_US"]
    name = f"{fk.first_name()} {fk.last_name()}"
    return [
        (None, "Tax filing summary for "),
        ("PERSON", name),
        (None, ". Italian codice fiscale: "),
        ("TAX_ID", _codice_fiscale(rng, name)),
        (None, ".\n"),
    ]


def _en_nir(rng, fakers):
    fk = fakers["en_US"]
    name = f"{fk.first_name()} {fk.last_name()}"
    return [
        (None, "Social security transfer: "),
        ("PERSON", name),
        (None, ", French NIR "),
        ("NATIONAL_ID", _nir(rng)),
        (None, ", confirmed active.\n"),
    ]


EN_TEMPLATES = [
    _en_bsn, _en_steuer, _en_belgian, _en_dni, _en_pesel, _en_cf, _en_nir,
]


# ---------------------------------------------------------------------------
# Language registry and the public build() entry point.
# ---------------------------------------------------------------------------

LANGUAGE_LOCALES: dict[str, list[str]] = {
    "ga": [],
    "mt": [],
    "nl": ["nl_NL"],
    "de": ["de_DE"],
    "fr": ["fr_FR", "fr_BE"],
    "es": ["es_ES"],
    "pl": ["pl_PL"],
    "it": ["it_IT"],
    "en": ["en_US"],
}

LANGUAGE_TEMPLATES: dict[str, list] = {
    "ga": GA_TEMPLATES,
    "mt": MT_TEMPLATES,
    "nl": NL_TEMPLATES,
    "de": DE_TEMPLATES,
    "fr": FR_TEMPLATES,
    "es": ES_TEMPLATES,
    "pl": PL_TEMPLATES,
    "it": IT_TEMPLATES,
    "en": EN_TEMPLATES,
}


def _build_language(language: str, per_language: int, seed: int) -> list[Record]:
    locales = LANGUAGE_LOCALES[language]
    fakers: dict[str, Faker] = {}
    for locale in locales:
        fk = Faker(locale)
        fk.seed_instance(_seed_for(seed, language, locale))
        fakers[locale] = fk
    templates = LANGUAGE_TEMPLATES[language]
    rng = random.Random(_seed_for(seed, language, "templates"))
    records = []
    for _ in range(per_language):
        template_fn = rng.choice(templates)
        segments = template_fn(rng, fakers)
        records.append(_render(segments))
    return records


def build(languages: list[str], per_language: int, seed: int) -> list[Record]:
    """Generate ``per_language`` records for each language in
    ``languages``, deterministically for a given ``seed``.

    Each record is ``{"text": str, "entities": [{"start": int, "end":
    int, "label": str}]}`` with exact character offsets, built
    incrementally rather than annotated after the fact (see the module
    docstring). Every ``label`` is one of taxonomy.PLATFORM_TYPES.

    Raises KeyError for a language this module doesn't generate, the same
    fail-loudly choice training/taxonomy.py makes for an unknown corpus
    label -- silently skipping an unrecognised language would look like
    coverage that isn't there.
    """
    records: list[Record] = []
    for language in languages:
        if language not in LANGUAGE_LOCALES:
            raise KeyError(f"unknown language: {language!r}")
        records.extend(_build_language(language, per_language, seed))
    return records
