#!/usr/bin/env python3
"""Builds `cities.txt` -- the city gazetteer `crates/model/src/cities.rs`
embeds -- from a GeoNames `cities1000` dump.

WHAT THIS PRODUCES AND WHY IT IS SHAPED THIS WAY. One lower-cased name per
line, sorted by UTF-8 byte order, no duplicates, newline-separated. Byte order
because the Rust side binary-searches the very same blob with `str`'s own
`Ord`, which is byte order; a locale-aware sort here would silently break that
search for exactly the accented names it was meant to help.

SCOPE. Six countries -- NL, DE, FR, ES, IT, GB -- because those are the six
languages the benchmark's address cases are written in. GeoNames' `cities1000`
tier is "population >= 1000, plus seats of administrative divisions down to
PPLA4 regardless of recorded population", which is the smallest published tier
that still contains every place that appears as the town line of a real postal
address in those countries (the sub-1000 tier, `cities500`, adds hamlets that
are addressed via a larger neighbouring town).

WHAT IS KEPT PER ROW. `name` and `asciiname` both, deduplicated: text in the
wild spells `Malaga` as often as `Málaga`, and the ASCII form is free here.
Bilingual GeoNames names (`Donostia / San Sebastián`) are split on `/` and both
halves kept, since either half is what a real address line carries.

WHAT IS DROPPED, AND WHY. Entries of two code points or fewer (`Ay`, `Us`,
`Re`): a two-letter token adjacent to an address is far more likely to be a
preposition than a village, and dropping 22 of them costs nothing. Entries
carrying anything other than a letter, space, hyphen, apostrophe or full stop
-- parenthesised disambiguators (`Brake (Unterweser)`), comma-qualified
districts (`Basford, Stoke-on-Trent`) -- because those spellings do not appear
in running text at all.

Usage:
    curl -O https://download.geonames.org/export/dump/cities1000.zip
    unzip cities1000.zip
    python3 crates/model/gazetteer/build_cities.py cities1000.txt \
        crates/model/gazetteer/cities.txt
"""

import sys

COUNTRIES = {"NL", "DE", "FR", "ES", "IT", "GB"}
ALLOWED_PUNCTUATION = set(" -'.")


def normalise(raw: str) -> str:
    """Lower-cases and collapses whitespace, and folds the typographic
    apostrophe onto the ASCII one -- `crates/model/src/cities.rs`
    applies the identical fold to the text it looks up, so `Bo'ness` and
    `Bo’ness` reach the same entry from either side."""
    return " ".join(raw.replace("’", "'").split()).lower()


def acceptable(name: str) -> bool:
    if len(name) <= 2:
        return False
    return all(character.isalpha() or character in ALLOWED_PUNCTUATION for character in name)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    source, destination = sys.argv[1], sys.argv[2]

    names: set[str] = set()
    rows = 0
    with open(source, encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 15 or fields[8] not in COUNTRIES:
                continue
            rows += 1
            for field in (fields[1], fields[2]):
                for half in field.split("/"):
                    name = normalise(half)
                    if name and acceptable(name):
                        names.add(name)

    ordered = sorted(names, key=lambda name: name.encode("utf-8"))
    with open(destination, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(ordered))
        handle.write("\n")

    print(f"{rows} rows in {sorted(COUNTRIES)} -> {len(ordered)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
