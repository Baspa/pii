# Licensing notice for the city gazetteer

**Scope.** This notice governs, and only governs:

- `crates/model/gazetteer/cities.txt` -- the compiled-in city gazetteer
- `crates/model/gazetteer/build_cities.py` -- the script that derives it (the script itself
  is this project's own code; it is named here because it is the record of *how* the source data
  was adapted)

Everything else in this repository is unaffected and remains under the terms declared in the
workspace manifest (`license = "UNLICENSED"`). This is a scoped notice, not a relicensing of the
project -- the same shape `benchmark/LICENSE-CARRIERS.md` uses for the benchmark carriers.

## Why this file exists

`cities.txt` is derived from GeoNames' geographical database, which is licensed **CC BY 4.0**.
The licence's condition is attribution (§3(a)): the source must be credited, the licence
identified, and modifications indicated. `cities.txt` ships inside every the detector binary
(`include_str!`), so the obligation binds on distribution of the binary, not only on this
repository. Recording it here is what keeps first distribution from being a breach. There is no
ShareAlike condition in CC BY 4.0, so -- unlike the Wikimedia carriers -- nothing here obliges the
rest of the project to any particular licence.

## The terms

`crates/model/gazetteer/cities.txt` is Adapted Material of a work licensed under the
Creative Commons Attribution 4.0 International licence, and is distributed under that same
licence: <https://creativecommons.org/licenses/by/4.0/>.

**Attribution.** © GeoNames (<https://www.geonames.org/>), licensed CC BY 4.0. The dump's own
`readme.txt` states the terms verbatim: "This work is licensed under a Creative Commons
Attribution 4.0 License, see https://creativecommons.org/licenses/by/4.0/ -- The Data is provided
'as is' without warranty or any representation of accuracy, timeliness or completeness."

**Source, version and retrieval.**

| | |
|---|---|
| source file | `https://download.geonames.org/export/dump/cities1000.zip` |
| dump tier | `cities1000` -- "all cities with a population > 1000 or seats of adm div down to PPLA3", per the dump's `readme.txt` |
| version (the dump publishes no version number, so its own modification time is the version) | `Last-Modified: Mon, 24 Aug 2026 01:52:46 GMT` |
| retrieved | 2026-08-24 |
| `sha256(cities1000.zip)` | `15d54c9894d26f7c0bc81cd3ef131239805d55ecd7dd1acb957f1a68ab583e1c` |
| `sha256(cities1000.txt)`, unzipped | `b38432a9a91926cecea6b1f8b9872f1ba0af8cde163765186b5f374f2511cccc` |
| `sha256(cities.txt)`, this repository's derived file | `fc956420d807245d8a50fcaac52bdcef22c9321be91ccf785c47d2eaa0abba16` |

**Indication of changes.** `cities.txt` is not a copy of the dump. It is derived from it by
`build_cities.py`, which:

- keeps only rows whose country code is one of **NL, DE, FR, ES, IT, GB** -- 40,005 of the dump's
  170,809 rows;
- keeps, per row, only the `name` and `asciiname` columns, and splits GeoNames' bilingual names
  (`Donostia / San Sebastián`) on `/`, keeping both halves -- every other column (coordinates,
  population, feature codes, admin divisions, timezone) is discarded;
- lower-cases every name, folds the typographic apostrophe `’` onto `'`, and collapses whitespace;
- drops names of two code points or fewer, and names carrying anything other than a letter,
  space, hyphen, apostrophe or full stop (parenthesised disambiguators such as
  `Brake (Unterweser)`, comma-qualified districts such as `Basford, Stoke-on-Trent`);
- de-duplicates, and sorts by UTF-8 byte order.

The result is **43,848 entries**, 532 KB. `crates/model/src/cities.rs`'s
`the_committed_file_holds_the_number_of_entries_the_licence_notice_records` pins that count
against this notice: a regenerated dump that changes it fails the test, so the count here cannot
silently go stale -- the same discipline `benchmark/LICENSE-CARRIERS.md` applies to its per-source
carrier counts.

**Reproducing it.**

```sh
curl -O https://download.geonames.org/export/dump/cities1000.zip
unzip cities1000.zip
python3 crates/model/gazetteer/build_cities.py cities1000.txt \
    crates/model/gazetteer/cities.txt
```

GeoNames publishes a fresh dump daily, so a later retrieval will not reproduce the digests above
byte for byte. That is why the digests, the `Last-Modified` stamp and the retrieval date are all
recorded: they identify the exact dump this file was adapted from, which is what CC BY 4.0
§3(a)(1)(A)-(C) asks an adapter to make identifiable.

## Why six countries and why this tier

The six are the languages the benchmark's address cases are written in, which are the languages
The detector's address detection is measured on. The tier is the smallest GeoNames publishes that still
contains every place that appears as the town line of a real postal address in those countries:
`cities500` adds hamlets that are addressed via a larger neighbouring town, while `cities5000`
drops places that are unquestionably real postal towns: it would cut 1,006 of the 1,565 Dutch
entries with a recorded population of 1,000 or more (`Uithuizen`, `Hapert`, `Maartensdijk` and
1,003 others), and the equivalent tail in each of the other five countries. 532 KB compiled into
the binary is the price; `training/reports/city-address-tails.md` records the reasoning in full.

## What this data is used for, and what it is not used for

It is consulted **only** to decide whether the token(s) immediately adjacent to a
`STREET_ADDRESS` span the model already found are a city -- never to find a city in open text.
`crates/model/src/cities.rs`'s module docs carry that argument, and
`the_gazetteer_is_reachable_from_exactly_one_place_in_this_crate` enforces it against the crate's
own sources. `LOCATION` remains retired.
