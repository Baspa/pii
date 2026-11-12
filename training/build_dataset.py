"""Merges ai4privacy, nym and this project's own generated records into one
grouped, language-complete train/validation/test split.

## Language scope: six, not 24

Narrowed 2026-08-21 (the design; a note at the same commit,
`5fa0d64`): the model is trained, measured and gated on six languages --
en, nl, de, fr, es, it -- matching Temperate's DACH-plus-Netherlands
go-to-market, with French, Spanish and Italian as the next-largest EU
business languages. `SUPPORTED_LANGUAGES` below is the single source of
truth; every row from every source whose language is not one of the six
is dropped from all three splits, in `_tagged_rows`, before group/split/
cap logic ever sees it.

Generated Irish and Maltese records are dropped too, not merely
unsupported: their text is LLM-authored and would need native-speaker
review before training on it, and that review has not happened and has
been descoped by the human partner. Polish is dropped for the ordinary
reason -- it is outside the supported set. `generate.py` itself is
untouched; it still knows how to build `ga`/`mt`/`pl` records via
`LANGUAGE_LOCALES`. Only `GENERATE_LANGUAGES` below no longer asks it to.

Nothing about structured detection narrows: the design already
establishes that language is a ranking hint, never a gate on a
recognizer, so identifiers are matched by format and validated by
checksum regardless of the surrounding prose's language. Only where
PERSON/STREET_ADDRESS quality is learned and measured narrows.

One thing this filter must not get wrong silently: nym carries no
language column of its own, so its `language` field (used below, same as
before this narrowing) is a `py3langid` *estimate*, not ground truth --
while ai4privacy's `language` column is real ground truth. The corpus profile's
calibration (measured against ai4privacy's ground truth; see
reports/corpora-profile.md) found py3langid 91.2% accurate overall, with
measured weak spots at et/fi/sk/lv/hr/lt (69-80% accuracy). None of the
six supported languages is in that weak band -- measured calibration:
en 99.0%, es 99.7%, fr 99.5%, de 98.6%, nl 97.8%, it 96.6%, all comfortably
above the corpus-wide average. Filtering nym on this estimate is therefore
not expected to meaningfully mis-drop or mis-admit rows for the six
languages this module now keeps.

## Why grouped, not random (the property that matters most)

All three sources are template-generated: ai4privacy's rows read as
free-form prose but are produced at scale from a small set of scenario
templates per label combination; nym's rows are filled into a visibly
smaller set of fixed sentence templates (measured below); every generated
record comes from one of a few dozen hand-written template functions in
generate.py. A random split puts near-copies of the same template on both
sides of train/test, and every metric that comes out afterwards is
inflated -- the model is scored on text it effectively memorised.

## How a "group" is computed, and why not `uid`/`source_dataset`

The task brief that produced this file suggested ai4privacy's `uid`
column or `source_dataset` column as a grouping key if nym's own "template
shape" wasn't available for it. Measured directly against ai4privacy's
real schema (`datasets-server.huggingface.co/info`, not the dataset card):

  - `uid` is an int64 with no shared-prefix structure that correlates with
    template identity -- sampled 5,000 consecutive rows and found 5,000
    distinct uids in a narrow, unstructured numeric range. It is a row id,
    not a template id.
  - `source_dataset` is a handful of *region* buckets ("asia-pacific",
    "europe", ...). The same 5,000-row sample was ALL `asia-pacific`.
    Splitting on this key would put an entire world region, tens of
    thousands of rows, indivisibly on one side of the split -- which
    risks failing "every split carries every EU language" outright if a
    whole region (and the languages that live only in it) lands entirely
    in one split. Too coarse to be a *template* group, on that ground
    alone -- coarseness, not the internal ordering of rows within a
    region. (An earlier version of this paragraph additionally claimed
    ai4privacy's rows are "stored in large contiguous per-region blocks,"
    citing corpora-profile.md's similar claim about `language`. Checked
    directly against the cached corpus: 1,528,215 language *runs* across
    1,636,375 rows -- mean run length 1.07. Language changes almost every
    row; rows are not grouped contiguously by language at all. The
    `source_dataset` coarseness argument above doesn't depend on
    contiguity and still holds on its own, but the contiguity claim
    itself was wrong and has been removed rather than left to imply a
    property this corpus doesn't have. corpora-profile.md's own
    calibration section uses the same contiguity claim to justify its
    positional 1-in-20 sampling as an approximation of a per-language
    draw -- that justification does not hold either, and should be
    revisited there; see training/reports/task-4-report.md.)

Neither column measures what the property under test actually cares
about: whether two rows read as near-duplicates of the same scaffolding
text. So this module computes a group key directly from what every
source already provides -- (text, entity spans) -- rather than trusting
an unverified column in either corpus:

    _skeleton(text, spans) replaces every entity's characters with a
    single placeholder byte, keeping the surrounding literal text
    untouched. Two rows built from the same template but filled with
    different names/addresses/dates produce the *same* skeleton; two
    rows with genuinely different content don't. The skeleton is hashed
    (sha256, truncated) to a compact `group` id.

This one function is applied uniformly to all three sources -- no
per-source special case -- and it was checked against real data before
being trusted:

  - ai4privacy, 5,000-row sample: 5,000/5,000 distinct skeletons. Its
    rows are not filled into a small fixed template pool (each reads as
    independently authored prose); this grouping key correctly degrades
    to "group of one" there, which is the right behaviour when there is
    no real template to protect against -- it does not force a false
    reduction in split precision.
  - nym, 5,000-row sample: 3,574/5,000 distinct skeletons, with one
    filler-sentence template repeating 77 times. Real, measurable
    template reuse -- this is "nym's template shape" the brief asks for,
    recovered without needing an explicit template-id column (nym's
    schema is exactly `text` + `entities`, nothing else).
  - generate.py's own output (900 records, 3 languages): the skeleton
    recovers per-template clusters directly (Dutch: 2 clusters of ~150;
    English: 7 clusters of ~40-150; Irish, whose templates also embed an
    unlabelled city name as literal text -- see the STREET_ADDRESS/city
    rule below -- correctly splits into finer template+city clusters,
    which is the more conservative, still-correct behaviour for that
    case). See training/reports/dataset.md for the exact command and
    counts this claim is measured from.

## Splitting a group into train/validation/test

Every row's group is a pure function of its (text, spans) -- no source
ever needs to be seen twice to compute it. Two different assignment
mechanisms sit on top of that group id, chosen for what each source's
scale allows:

  - ai4privacy and nym (2.44M rows streamed once each): a row's split is
    `sha256(seed, group) % 10000` bucketed 90/5/5. This needs no memory
    of groups seen before -- it is a pure function computed once per row
    in the same streaming pass that reads the row -- which is what keeps
    this module's memory flat the way training/profile_corpora.py's did.
    For every one of the 24 EU languages except Irish and Maltese,
    ai4privacy alone already provides 44,900+ rows (corpora-profile.md),
    each landing in its own group with overwhelming probability (measured
    above), so probabilistic bucketing cannot plausibly miss a split for
    any of them.
  - generate.py's output (1,800 records since the 2026-08-21 language-scope
    narrowing dropped ga/mt/pl, entirely in memory, no network): every
    generated language's distinct groups are shuffled with a seed unique
    to that language and explicitly divided so validation and test each
    get at least one group (when the language has 2+ groups, which every
    generated language does -- measured minimum 2). This guarantees,
    rather than merely makes likely, that every split contains every
    generated language's text -- the guarantee this section describes was
    originally motivated by Irish and Maltese having NO other source at
    all (2,700 records/9 languages, ga/mt measured at 129 and 135 distinct
    groups under `_normalize_for_grouping`'s case/whitespace/digit
    normalisation, up from 164/166 before it), which is why
    NO_OTHER_SOURCE_LANGUAGES exists even though it is empty now that ga
    and mt are no longer generated at all (see that constant's own
    comment).

## The training-set language cap

corpora-profile.md's own combined-language table shows English at
381,404 rows -- 2.4x the next-largest language (French, 161,600) and
roughly 8x the 24-language median. Left uncapped, English would dominate
training signal for two entity types (PERSON, STREET_ADDRESS) this whole
plan exists to teach every EU language. `TRAIN_LANGUAGE_CAP` limits any
one language to 150,000 rows in the training split specifically (not
validation/test, which are much smaller and not size-dominated the same
way) -- chosen as roughly French's own total, so it reins in English
without touching any other measured language's natural allocation. Rows
over the cap are dropped from the output entirely (never moved to another
split -- that would change what the split ratios mean and would still
leave the group intact on one side only, which is what the group property
actually requires).

This is a hard bound, not a target: `build()` orders a capped language's
groups by a hash of `(seed, group)` and takes the prefix of that order
whose cumulative row count still fits the budget (see `_cap_order_key`
and the "Pass 1b" comment in `build()`). An earlier version thinned every
row independently by a fixed acceptance probability instead, which is
only a target, not a bound: nym's own repeated filler templates carry
roughly 11,000 rows in a single group, so whether a handful of those
groups happen to land on the "accept" side of the coin flip swings the
realised total considerably. Measured by replaying seeds 0-7 against that
version: English's realised train total ranged from 130,603 to 165,939 --
10% over the 150,000 cap at seed 5. The prefix-of-a-deterministic-order
approach cannot exceed the budget for any seed, by construction.

## The STREET_ADDRESS / city question this task was asked to close

taxonomy.py maps nym's STREET_ADDRESS label straight onto Temperate's
STREET_ADDRESS with no other target, on the strength of nym's label set
reading like Faker's address-provider surface (`street_address()` is
building number + street name, not `address()`, which also includes
city/state/postcode). That claim needed checking against real spans, not
label names. This module checks it while streaming every nym row anyway:
for every (STREET_ADDRESS span, CITY span) pair within the same row, it
tests whether their character ranges overlap at all. See
training/reports/dataset.md for the measured rate.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Iterator

import py3langid as langid
from datasets import load_dataset

import generate
import taxonomy

# ---------------------------------------------------------------------------
# Fixed pipeline parameters. See the module docstring for why each value was
# chosen; training/reports/dataset.md records the measured effect of each.
# ---------------------------------------------------------------------------

SPLIT_RATIOS = {"train": 0.90, "validation": 0.05, "test": 0.05}  # of buckets, by group
TRAIN_LANGUAGE_CAP = 150_000  # rows; see module docstring

# Narrowed 2026-08-21 -- the design. Single source of truth for
# which languages this module keeps; every row of every source is dropped
# in _tagged_rows if its language isn't in this set (see the module
# docstring's "Language scope" section). frozenset: this is checked in the
# hot per-row loop over ~2.44M ai4privacy/nym rows, so O(1) membership
# matters; iterate `sorted(SUPPORTED_LANGUAGES)` wherever output order
# needs to be deterministic.
SUPPORTED_LANGUAGES = frozenset({"en", "nl", "de", "fr", "es", "it"})

AI4PRIVACY_REPO = "ai4privacy/pii-masking-openpii-1.5m"
AI4PRIVACY_SPLITS = ("train", "validation")
AI4PRIVACY_EXPECTED_ROWS = 1_636_375  # measured in the corpus profile; used only to validate the local cache

NYM_REPO = "Wismut/nym-pii-multilingual-data"
NYM_SPLITS = ("train", "validation", "test")
NYM_EXPECTED_ROWS = 805_000  # measured in the corpus profile; used only to validate the local cache

# training/generate.py's own canonical invocation -- see training/reports/generated.md.
# Fixed independently of build()'s own `seed`: these 1,800 generated records
# (6 languages x 300) are an already-reviewed, already-reported artifact;
# only which SPLIT each one lands in should move when `seed` changes, not
# their content. Narrowed 2026-08-21 from 9 languages/2,700 records -- ga,
# mt and pl are still in generate.py's LANGUAGE_LOCALES, unchanged, but are
# no longer requested here: ga/mt because their text is LLM-authored and
# needs native-speaker review that has not happened and has been
# descoped, pl because it is outside the supported set (SUPPORTED_LANGUAGES
# above). See the module docstring's "Language scope" section. Every
# language listed here is already in SUPPORTED_LANGUAGES, so the runtime
# filter in _tagged_rows never actually drops a generated row -- it is
# still applied uniformly to all three sources rather than special-cased,
# so a future edit to either list can't silently drift out of sync.
GENERATE_LANGUAGES = ["nl", "de", "fr", "es", "it", "en"]
GENERATE_PER_LANGUAGE = 300
GENERATE_SEED = 2026

CACHE_DIR = Path(__file__).parent / ".cache"


# ---------------------------------------------------------------------------
# Grouping: a source-agnostic "template shape" computed from (text, spans).
# See the module docstring's "How a group is computed" section for why this
# replaced the brief's suggested uid/source_dataset keys.
# ---------------------------------------------------------------------------


def _skeleton(text: str, spans: list[tuple[int, int]]) -> str:
    """Replace every span's characters with one placeholder, keeping the
    surrounding literal text. Defensive against overlapping/out-of-order
    spans (real corpora aren't guaranteed clean): a span that starts before
    the current position is skipped rather than raising, so one malformed
    span can't crash the whole build.
    """
    ordered = sorted(spans, key=lambda s: (s[0], s[1]))
    out: list[str] = []
    pos = 0
    for start, end in ordered:
        if start < pos:
            continue
        out.append(text[pos:start])
        out.append("\x00")
        pos = end
    out.append(text[pos:])
    return "".join(out)


_GROUPING_WHITESPACE_RE = re.compile(r"\s+")
_GROUPING_DIGITS_RE = re.compile(r"\d+")


def _normalize_for_grouping(skeleton: str) -> str:
    """Collapses exactly the kind of variation nym's own augmentation
    introduces between otherwise-identical template instances -- case and
    whitespace perturbation -- before the skeleton is hashed into a group
    id. A review flagged this leakage class and cited 2,203/38,930 nym
    test rows (5.66%) as the measured rate; independently re-deriving
    that number against the real pre-fix committed output (three separate
    methodologies, all cross-checked against each other and against the
    committed row counts) instead measured 7,025/38,930 (18.05% of nym's
    test rows, 5.77% of the whole test split) -- materially higher, not
    lower, than the cited figure. Both measurements agree the leakage is
    real, non-trivial and worth fixing; see training/reports/dataset.md
    and task-4-report.md for the full methodology and the discrepancy.
    Digit runs are also collapsed -- not separately measured as a source
    of leakage the way case/whitespace were, but harmless to fold in and
    closes the same class of near-duplicate (a template whose only
    difference in the *literal* surrounding text, outside any entity
    span, is which arbitrary reference number got substituted in).
    """
    normalized = skeleton.lower()
    normalized = _GROUPING_DIGITS_RE.sub("0", normalized)
    normalized = _GROUPING_WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def _group_id(text: str, spans: list[tuple[int, int]]) -> str:
    skeleton = _skeleton(text, spans)
    normalized = _normalize_for_grouping(skeleton)
    return hashlib.sha256(normalized.encode("utf-8", "surrogatepass")).hexdigest()[:24]


def _mix(seed: int, *parts: str) -> int:
    """Deterministic string-to-int mixer -- same reasoning as
    generate._seed_for: Python's hash() on strings is randomised per
    process unless PYTHONHASHSEED is fixed, which would make this module
    non-deterministic across runs even for a fixed `seed` argument.
    """
    mixed = seed & 0xFFFFFFFF
    for part in parts:
        for ch in part:
            mixed = (mixed * 1_000_003 + ord(ch)) & 0xFFFFFFFF
    return mixed


_BUCKET_WIDTH = 10_000
_BUCKET_THRESHOLDS = (
    ("train", round(SPLIT_RATIOS["train"] * _BUCKET_WIDTH)),
    ("validation", round((SPLIT_RATIOS["train"] + SPLIT_RATIOS["validation"]) * _BUCKET_WIDTH)),
    ("test", _BUCKET_WIDTH),
)  # derived from SPLIT_RATIOS so the two never drift apart


def _bucket(group: str, seed: int) -> str:
    """Pure function of (group, seed) -- no memory of groups seen before is
    needed, which is what keeps ai4privacy/nym's single streaming pass
    O(1)-memory per row. Bucketed by SPLIT_RATIOS, by group, not by row
    (grouped splits do not promise an exact row-level ratio -- see
    dataset.md for the measured result).
    """
    digest = hashlib.sha256(f"{seed}|{group}".encode("utf-8")).digest()
    n = int.from_bytes(digest[:8], "big") % _BUCKET_WIDTH
    for split, upper in _BUCKET_THRESHOLDS:
        if n < upper:
            return split
    return "test"  # unreachable: last threshold == _BUCKET_WIDTH


# ---------------------------------------------------------------------------
# Source 1: training/generate.py's own output. Entirely in memory (2,700
# records, no network) -- see module docstring for why Irish/Maltese need a
# guaranteed rather than probabilistic split assignment.
# ---------------------------------------------------------------------------


# Empty since the 2026-08-21 language-scope narrowing dropped ga/mt from
# GENERATE_LANGUAGES (they were the only two languages ever in this set --
# corpora-profile.md: 2 and 24 rows respectively across BOTH corpora
# combined, both noise from nym's language *estimate*, i.e. truly no other
# source). Every one of the six languages generate.py is now asked to
# build already has 44,900+ backing rows from ai4privacy alone, so none of
# them needs the unconditional validation/test guarantee below -- they all
# take the clamped branch, same as nl/de/fr/es/it/en always did even
# before this narrowing. Left in place rather than deleted: if a future
# language is ever added to GENERATE_LANGUAGES with no ai4privacy/nym
# backing (the situation ga/mt were in), this is the set that protects its
# validation/test split coverage the way it protected theirs.
NO_OTHER_SOURCE_LANGUAGES: set[str] = set()


def _generated_rows_with_splits(seed: int) -> list[tuple[dict, str, str, str]]:
    """Returns a list of (record, language, group, split) for every
    generated record, deterministic for `seed` (which only controls split
    assignment -- the records themselves come from generate.py's own fixed,
    already-reviewed seed; see GENERATE_SEED above).
    """
    records = generate.build(GENERATE_LANGUAGES, GENERATE_PER_LANGUAGE, GENERATE_SEED)
    out: list[tuple[dict, str, str, str]] = []
    for idx, language in enumerate(GENERATE_LANGUAGES):
        lang_records = records[idx * GENERATE_PER_LANGUAGE : (idx + 1) * GENERATE_PER_LANGUAGE]
        groups: dict[str, list[dict]] = {}
        for record in lang_records:
            spans = [(e["start"], e["end"]) for e in record["entities"]]
            group = _group_id(record["text"], spans)
            groups.setdefault(group, []).append(record)

        group_ids = sorted(groups)  # deterministic order before shuffling
        rng = random.Random(_mix(seed, "generated-split", language))
        rng.shuffle(group_ids)
        n = len(group_ids)
        if n < 2:
            k = 0
        else:
            k = max(1, round(SPLIT_RATIOS["validation"] * n))
            if language not in NO_OTHER_SOURCE_LANGUAGES:
                # Clamp so 2*k <= n-1: train always retains at least one
                # group. With n==2 (measured: nl, de, es, pl, it all have
                # exactly 2 generated-template groups) the unclamped
                # formula put one group in test and one in validation,
                # leaving train with ZERO generated rows for that
                # language -- the checksum-valid national IDs/tax IDs
                # the data generation exists to teach would then be present only where
                # the model is evaluated, never where it learns. Measured
                # before this fix: de, es, it, nl, pl at 0 generated train
                # rows.
                k = min(k, (n - 1) // 2)
        test_groups = set(group_ids[:k])
        validation_groups = set(group_ids[k : 2 * k])

        for group, recs in groups.items():
            if group in test_groups:
                split = "test"
            elif group in validation_groups:
                split = "validation"
            else:
                split = "train"
            for record in recs:
                out.append((record, language, group, split))
    return out


# ---------------------------------------------------------------------------
# Sources 2 & 3: ai4privacy and nym. Streamed, never materialised into
# memory as a whole -- see training/profile_corpora.py for the same
# discipline applied to measurement instead of dataset construction.
#
# Each source's raw rows are also cached to a local, gitignored file
# (training/.cache/) on the first full run and reused on later ones. This
# is a deliberate, disclosed deviation from pure streaming: the two corpora
# are immutable, publicly hosted and 4.8 GB combined, and this module's own
# test suite calls build() three times in one pytest run (plus once more
# for the report). Re-fetching identical public data over the network three
# to five times in a session was judged wasteful; the cache changes nothing
# about the algorithm (still O(1) memory per row, still incremental) and is
# only ever used for a full (uncapped) run -- see `max_rows_per_source`.
# ---------------------------------------------------------------------------


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.jsonl.gz"


def _cached_or_streamed(name: str, expected_rows: int, stream_fn: Callable[[], Iterator[dict]], max_rows_per_source: int | None) -> Iterator[dict]:
    if max_rows_per_source is not None:
        n = 0
        for row in stream_fn():
            yield row
            n += 1
            if n >= max_rows_per_source:
                return
        return

    cache_path = _cache_path(name)
    if cache_path.exists():
        with gzip.open(cache_path, "rt", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)
        return

    CACHE_DIR.mkdir(exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp.gz")
    count = 0
    with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
        for row in stream_fn():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            yield row
    if count == expected_rows:
        tmp_path.rename(cache_path)
    else:
        tmp_path.unlink(missing_ok=True)


def _clean_spans(text: str, raw_spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    n = len(text)
    return [(s, e, label) for s, e, label in raw_spans if s is not None and e is not None and 0 <= s < e <= n]


def _log(msg: str) -> None:
    import sys
    import time as _time

    ts = _time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


def _stream_ai4privacy_raw() -> Iterator[dict]:
    import time

    n = 0
    t0 = time.time()
    last_report = t0
    for split in AI4PRIVACY_SPLITS:
        ds = load_dataset(AI4PRIVACY_REPO, split=split, streaming=True)
        ds = ds.select_columns(["source_text", "privacy_mask", "language"])
        for row in ds:
            text = row.get("source_text") or ""
            raw_spans = [(m["start"], m["end"], m["label"]) for m in (row.get("privacy_mask") or [])]
            spans = _clean_spans(text, raw_spans)
            yield {"text": text, "spans": spans, "language": row.get("language") or "und"}
            n += 1
            if time.time() - last_report > 20:
                _log(f"ai4privacy[{split}]: {n} rows, {time.time() - t0:.0f}s elapsed")
                last_report = time.time()
    _log(f"ai4privacy: done, {n} rows, {time.time() - t0:.0f}s elapsed")


def _stream_nym_raw() -> Iterator[dict]:
    import time

    n = 0
    t0 = time.time()
    last_report = t0
    for split in NYM_SPLITS:
        ds = load_dataset(NYM_REPO, split=split, streaming=True)
        for row in ds:
            text = row.get("text") or ""
            raw_spans = [(e["start"], e["end"], e["label"]) for e in (row.get("entities") or [])]
            spans = _clean_spans(text, raw_spans)
            language = langid.classify(text)[0] if text else "und"
            yield {"text": text, "spans": spans, "language": language}
            n += 1
            if time.time() - last_report > 20:
                _log(f"nym[{split}]: {n} rows, {time.time() - t0:.0f}s elapsed")
                last_report = time.time()
    _log(f"nym: done, {n} rows, {time.time() - t0:.0f}s elapsed")


def _spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


# ---------------------------------------------------------------------------
# Adjacent-span merging: one entity per name, not one per name *part*.
#
# ## The defect this exists to fix
#
# Both corpora label a given name and a surname as SEPARATE entities
# (ai4privacy: GIVENNAME/SURNAME; nym: GIVEN_NAME/SURNAME) and taxonomy.py
# maps both onto Temperate's single PERSON type without joining the spans. The
# 2026-08-21 checkpoint therefore learned "detect a name *part*", and on the
# hand-written fixture it detects one part and stops -- `Anna Schmidt` ->
# `Anna`, `Michael Chen` -> `Michael`, `Sarah Jones` -> nothing. Six of that
# checkpoint's thirteen fixture misses have exactly this shape. No
# post-processing can repair it: fixture_scoring.merge_adjacent_whitespace
# joins two *predicted* spans, but here the second span was never predicted.
# The fix has to be in the labels the model is shown, which is here.
#
# ## Which types merge, and why not all of them
#
# PERSON and STREET_ADDRESS are the only two platform types this taxonomy
# builds by collapsing several *positionally adjacent sub-labels of one
# field* onto one type --
#
#   PERSON          <- GIVENNAME/GIVEN_NAME + SURNAME (+ DOCTORNAME)
#   STREET_ADDRESS  <- STREET/STREET_NAME + BUILDINGNUM/BUILDING_NUMBER
#                      (+ ZIPCODE/ZIP_CODE, SECONDARY_ADDRESS)
#
# NATIONAL_ID (IDCARDNUM/SOCIALNUM, GOVERNMENT_ID/SSN), PHONE_NUMBER
# (PHONE/FAX_NUMBER) and CREDIT_CARD draw from more than one source label
# too, but those are ALTERNATE complete values, not adjacent halves of one
# field: two card numbers side by side are two cards, not one. Merging them
# would fuse two distinct identifiers into one span and teach the model a
# boundary that does not exist. Measured over data/validation.jsonl +
# data/test.jsonl (91,602 rows, the pre-merge build at commit 4925ae4),
# adjacent same-type pairs separated by whitespace alone number 53,909+
# for PERSON and 15,573+ for STREET_ADDRESS, against 3 for NATIONAL_ID and
# 2 for CREDIT_CARD -- shape-wise PERSON and STREET_ADDRESS are alike, and
# neither is a CREDIT_CARD/NATIONAL_ID-style run of alternate values.
#
# But MERGEABLE_TYPES (2026-08-22) contains only PERSON, not both. Address
# detection was not broken: 0.9877 held-out F1, zero fixture misses of this
# shape, and the merge does not even fix the one known STREET_ADDRESS defect
# (`Kerkstraat 12, 1016 GV` stays two spans regardless, because the comma
# between them is deliberately not a bridgeable gap -- see below). What
# tipped it is that a STREET_ADDRESS merge is not measurable: after merging,
# held-out F1 scores a different task than the 0.9877 baseline, and the
# fixture carries too few address cases to serve as a second signal. PERSON
# has that second signal -- six named fixture misses that either get fixed
# or do not -- STREET_ADDRESS does not. Changing something unmeasurable that
# was already working buys nothing and risks a retrain regressing address
# recall for no verifiable reason. `fixture_scoring.MERGEABLE_TYPES` is
# unchanged and still includes STREET_ADDRESS: it merges *predicted* spans
# at scoring time, after inference, and is what already gets `Kerkstraat 12`
# scored as one span today, independent of how the training labels are
# built. The two constants no longer describe the same set on purpose --
# see reports/dataset-merged-names.md for the full reasoning and the
# measurements behind it.
#
# merge_adjacent_entities itself is unchanged and still general: it takes
# `mergeable_types` as a parameter, so tests below exercise the
# STREET_ADDRESS-shaped rules (letter-bearing runs, the letterless clause)
# by passing an explicit override rather than relying on the default.
#
# ## Which gap is bridgeable: horizontal whitespace, nothing else
#
# Measured over the same 91,602 rows, tallying the literal text between
# every adjacent same-type PERSON pair (see
# training/reports/dataset-merged-names.md for the command):
#
#   ' '            53,909   whole name, one person            -> BRIDGE
#   ', '              504   *list separator*, often two people-> no
#   '\nCognome: '     327   a field label between two fields  -> no
#   '.'               153   username/e-mail local part
#                           (`Amanda`.`Salazar`@gmail.com)     -> no
#   ' and '/' und '/
#   ' y '/' et '/' e '        two people joined by "and"      -> no
#   '. '                1   a SENTENCE BOUNDARY between
#                           two different speakers             -> no
#
# The comma is the case that settles it against generalising. Read in
# context, `Asistentes: Ekansh [Abeka], [Tori] Budavary y Hatayi
# Anantharajah` is Ekansh Abeka followed by Tori Budavary -- bridging ', '
# would fuse the surname of one attendee to the given name of the next.
# The period is the second: every one of the 146 English '.'-gap pairs
# sampled was a username or e-mail local part, and the single '. ' pair was
# a sentence break between two speakers.
#
# One clause is not about the gap at all: two spans that BOTH contain no
# letter are never joined. See merge_adjacent_entities' docstring -- it is
# the CREDIT_CARD/NATIONAL_ID judgement applied at the span level, and it is
# what stops a run of bare postcodes fusing into one address.
#
# Line breaks are excluded from "whitespace" for the same reason, one step
# weaker: no PERSON pair in the measured build is separated by a bare
# newline at all (so nothing is lost), and the 205 STREET_ADDRESS pairs
# that are, are address blocks whose parts sit on separate lines. Keeping
# every merged span inside one line makes "no span crosses a line or
# sentence boundary" checkable by construction and avoids teaching the
# model to jump lines, which is where a one-per-line list would over-merge.
# This is strictly more conservative than
# fixture_scoring.merge_adjacent_whitespace (which uses `.strip()`, so it
# does bridge newlines) -- deliberately: that function is a post-processor
# that can only widen what the model emits, so the two compose correctly.
#
# ## The tussenvoegsel (`Jan de Vries`)
#
# Checked directly rather than assumed, because it decides whether this
# approach fixes the Dutch case at all: both corpora put the particle
# INSIDE the surname span. Measured over 120,001 nym rows and 200,001
# ai4privacy rows, SURNAME spans beginning with a particle -- `van Wessex`,
# `van der Velden`, `van den Brink`, `de Groot`, `ter Waarbeek`,
# `von der Ohe`, `Della Pietra`, `da Cunha` -- number 813 and 1,097
# respectively, while particles left OUTSIDE any span, sitting in the gap
# between two name spans, number 0 and 1 (and that 1 is a false positive:
# German `von der` as a real preposition in `... sagte Xiaohua von der
# Japic Gottraux Behörde ...`). So `Jan de Vries` is labelled
# `[Jan](GIVEN_NAME) [de Vries](SURNAME)` with a single space between, and
# a whitespace bridge merges it correctly. The tussenvoegsel case IS
# fixable this way.
#
# What is NOT fixed here: `J. R. Whitfield`. Neither corpus labels a bare
# initial as a name -- there is no `J.`-shaped PERSON span to merge with
# anything (measured: of 135,226 PERSON spans in validation+test, the
# 1,664 immediately followed by `.` are all sentence-final, none are
# initials). Bridging periods would import the username/e-mail damage
# above without supplying the missing signal. Teaching initials needs
# generated records that contain them; that is a generate.py change, not a
# merge rule.
# ---------------------------------------------------------------------------

MERGEABLE_TYPES: frozenset[str] = frozenset({"PERSON"})

# Every Unicode character Python calls whitespace that also breaks a line.
# `str.isspace()` minus these is "horizontal whitespace", the only gap this
# module bridges. Kept explicit rather than expressed as a regex so the
# exact set is readable and testable.
LINE_BREAK_CHARS: frozenset[str] = frozenset(
    "\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029"
)


def is_bridgeable_gap(gap: str) -> bool:
    """True if `gap` is empty or consists only of horizontal whitespace.

    Empty counts: two same-type spans that touch are one entity split by an
    annotation artifact, and leaving them apart is the same defect with a
    zero-width gap.
    """
    return all(ch.isspace() and ch not in LINE_BREAK_CHARS for ch in gap)


def has_no_letter(span_text: str) -> bool:
    """True if `span_text` contains no alphabetic character at all."""
    return not any(ch.isalpha() for ch in span_text)


def merge_adjacent_entities(
    text: str,
    entities: list[dict],
    mergeable_types: frozenset[str] = MERGEABLE_TYPES,
) -> tuple[list[dict], int]:
    """Join runs of same-type entities of a mergeable type that are
    separated only by a bridgeable gap into one entity each.

    Returns `(entities, merges)` where `merges` is how many entities were
    absorbed into a predecessor (0 when nothing merged). Entities come back
    sorted by (start, end); the input order is not relied on.

    Two spans that BOTH contain no letter are never joined, even when
    everything else about them qualifies. This is the same judgement that
    keeps CREDIT_CARD and NATIONAL_ID out of `mergeable_types`, applied one
    level down: a span with no letters carries no evidence that it is a
    *part* of a larger value rather than a complete one, so two of them side
    by side are two postcodes, not one address. Without this clause the rule
    fuses a run of 33 French postcodes separated by narrow no-break spaces
    into a single 195-character STREET_ADDRESS span (measured, in
    train.jsonl, on the build before this clause existed) -- exactly the
    "merging two genuinely different values" failure. It costs almost
    nothing: over validation+test, 15,712 of the 16,778 bridgeable
    STREET_ADDRESS pairs have letters on one side (a street name and its
    building number, in either order), 1,060 have letters on both, and only
    6 have letters on neither. For PERSON it is a no-op -- all 54,141
    bridgeable PERSON pairs have a letter on both sides.

    Overlapping entities are left exactly as they are, even of a mergeable
    type: align.py documents that it does not resolve overlap, this module
    has never produced any (measured: 0 overlapping adjacent pairs in
    91,602 validation+test rows), and silently repairing one here would
    hide a corpus regression this build should surface instead.
    """
    if len(entities) < 2:
        return sorted(entities, key=lambda e: (e["start"], e["end"])), 0

    ordered = sorted(entities, key=lambda e: (e["start"], e["end"]))
    out: list[dict] = [dict(ordered[0])]
    merges = 0
    for entity in ordered[1:]:
        prev = out[-1]
        if (
            entity["label"] == prev["label"]
            and entity["label"] in mergeable_types
            and entity["start"] >= prev["end"]
            and is_bridgeable_gap(text[prev["end"] : entity["start"]])
            and not (
                has_no_letter(text[prev["start"] : prev["end"]])
                and has_no_letter(text[entity["start"] : entity["end"]])
            )
        ):
            prev["end"] = entity["end"]
            merges += 1
        else:
            out.append(dict(entity))
    return out, merges


# ---------------------------------------------------------------------------
# The public entry point.
# ---------------------------------------------------------------------------


def _tagged_rows(seed: int, max_rows_per_source: int | None):
    """Yields (source, text, spans, language, group, split) for every row
    of every source, in a fixed order (generated, ai4privacy, nym), for
    every row whose `language` is in SUPPORTED_LANGUAGES. `spans` is
    `(start, end, label)` with the source's RAW label for ai4privacy/nym,
    or its already-platform label for generated (see generate.build()).
    `build()` calls this generator twice -- once to account, once to
    write -- see "A deliberate deviation: local caching" above for why a
    second read of ai4privacy/nym is cheap (the local cache, not a second
    network fetch) rather than expensive.

    The SUPPORTED_LANGUAGES filter sits here, one call site for all three
    sources, applied after `_cached_or_streamed` -- so the on-disk cache
    still holds every raw row of both corpora (its row count still
    validates against AI4PRIVACY_EXPECTED_ROWS/NYM_EXPECTED_ROWS, and a
    later widening of SUPPORTED_LANGUAGES needs no cache rebuild or
    re-download) and the drop happens before the cost of computing a
    group/split for a row this module is about to discard anyway.
    """
    for record, language, group, split in _generated_rows_with_splits(seed):
        if language not in SUPPORTED_LANGUAGES:
            continue
        spans = [(e["start"], e["end"], e["label"]) for e in record["entities"]]
        yield "generated", record["text"], spans, language, group, split

    for row in _cached_or_streamed("ai4privacy", AI4PRIVACY_EXPECTED_ROWS, _stream_ai4privacy_raw, max_rows_per_source):
        text, spans, language = row["text"], row["spans"], row["language"]
        if language not in SUPPORTED_LANGUAGES:
            continue
        group = _group_id(text, [(s, e) for s, e, _ in spans])
        split = _bucket(group, seed)
        yield "ai4privacy", text, spans, language, group, split

    for row in _cached_or_streamed("nym", NYM_EXPECTED_ROWS, _stream_nym_raw, max_rows_per_source):
        text, spans, language = row["text"], row["spans"], row["language"]
        if language not in SUPPORTED_LANGUAGES:
            continue
        group = _group_id(text, [(s, e) for s, e, _ in spans])
        split = _bucket(group, seed)
        yield "nym", text, spans, language, group, split


def _cap_order_key(group: str, seed: int) -> int:
    """Deterministic sort key for ordering a capped language's groups
    into a fixed, seed-dependent random order, so a hard per-language
    train budget can be enforced by taking a prefix of that order rather
    than a prefix of stream position (see build()'s cap section for why
    stream position was wrong) or a per-row acceptance probability (see
    the same section for why that was only a coarse target, not a bound).
    Pure function of (seed, group) -- never of stream position or source.
    """
    digest = hashlib.sha256(f"{seed}|cap-order|{group}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


# Sources whose exact distinct-group count is tracked (a `set()` of every
# group id seen) rather than merely counted, so the dataset can assert
# "this source's grouping key finds real duplication" directly instead of
# trusting a docstring claim -- see test_nym_groups_are_materially_fewer_than_nym_rows.
# ai4privacy is deliberately excluded: tracking its ~1.6M-entry set would
# cost ~200MB+ for a property already independently verified at ~100%
# uniqueness by sampling (see the module docstring), and no test currently
# needs ai4privacy's exact count -- only nym's (the source with the real
# template pool this property exists to detect) and generated's (tiny).
GROUP_TRACKED_SOURCES = {"nym", "generated"}


def build(out_dir: Path, seed: int, *, max_rows_per_source: int | None = None) -> dict:
    """Merge, group and split ai4privacy, nym and generate.py's output.

    Writes train.jsonl, validation.jsonl and test.jsonl to `out_dir`
    (never accumulating the merged corpus in memory) and returns
    per-split counts: rows, a language -> row-count map, a label ->
    entity-count map, a source -> row-count map and a language -> source
    -> row-count map for each split, plus the language cap actually
    applied, the nym STREET_ADDRESS/CITY overlap measurement, and
    per-source group/row counts for the sources in GROUP_TRACKED_SOURCES.

    Up to three passes over `_tagged_rows`: pass 1 (always) measures how
    many rows would naturally land in `train` per language, combined
    across sources, before any cap. Pass 1b (only when at least one
    language actually exceeds the cap) tallies that language's non-
    generated groups' row counts, so the cap can be enforced as an exact
    prefix of a deterministic group order rather than a per-row
    probability -- see the cap section below for why that distinction
    matters. Pass 2 (always) does the real write. Every extra pass reads
    ai4privacy/nym from the local cache, not the network, once it exists
    (see "A deliberate deviation: local caching" above), so the added
    cost is Python-side processing, not network time.

    `max_rows_per_source` is not part of this task's required interface --
    it exists only so this module's own development/smoke-testing can run
    against a small slice instead of the full 2.44M rows. It disables the
    local cache (a capped run is never a valid full run to cache) and
    defaults to None, which is what every required test and the real
    report-producing run use.

    Deterministic for a given `seed`: every decision that varies is a pure
    function of (seed, group id) or (seed, language), computed the same
    way regardless of process, run or stream position.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Pass 1: accounting only -- natural per-language train totals,
    # combined across sources, before any cap. ---
    natural_train_counts: Counter = Counter()
    generated_train_counts: Counter = Counter()
    for source, text, spans, language, group, split in _tagged_rows(seed, max_rows_per_source):
        if split == "train":
            natural_train_counts[language] += 1
            if source == "generated":
                generated_train_counts[language] += 1

    # generated rows are never thinned by the cap -- the data generation's whole point
    # was adding PERSON/STREET_ADDRESS/NATIONAL_ID/TAX_ID signal the two
    # large corpora under-serve, and this cap exists to fix English's
    # outsized *corpus* share, not to compete with that. The budget
    # available to ai4privacy+nym combined is what's left of the cap
    # after generated's guaranteed contribution is reserved.
    capped_languages: dict[str, int] = {}  # language -> non-generated budget
    for language, natural in natural_train_counts.items():
        non_generated = natural - generated_train_counts[language]
        budget = max(0, TRAIN_LANGUAGE_CAP - generated_train_counts[language])
        if non_generated > budget:
            capped_languages[language] = budget

    # --- Pass 1b: only for languages that actually exceed the cap
    # (in the real dataset, only English does), tally each of their
    # non-generated groups' row counts. Skipped entirely when nothing
    # is capped, and its memory cost is bounded by the capped
    # languages' own group counts, not the whole corpus.
    #
    # Why a per-group tally rather than the per-row acceptance
    # probability this replaced: nym's own repeated filler templates
    # carry roughly 11,000 rows each in a single group, so thinning
    # every row independently by a fixed probability is coarse -- a
    # capped language's realised total swings by however many of those
    # large groups happen to land on the "accept" side of the coin flip.
    # Measured by replaying seeds 0-7: English's realised train total
    # ranged from 130,603 to 165,939 -- 10% OVER the 150,000 cap at
    # seed 5. A cap that can be exceeded isn't a cap. Ordering groups by
    # a hash of (seed, group) and taking the prefix that still fits the
    # budget (below) makes it a hard bound instead: the running total
    # never exceeds budget, by construction, for any seed.
    group_row_counts: dict[str, Counter] = {language: Counter() for language in capped_languages}
    if capped_languages:
        for source, text, spans, language, group, split in _tagged_rows(seed, max_rows_per_source):
            if split == "train" and source != "generated" and language in capped_languages:
                group_row_counts[language][group] += 1

    accepted_groups: dict[str, set] = {}
    for language, budget in capped_languages.items():
        ordered = sorted(group_row_counts[language].items(), key=lambda kv: _cap_order_key(kv[0], seed))
        accepted: set = set()
        total = 0
        for group, count in ordered:
            if total + count > budget:
                break
            accepted.add(group)
            total += count
        accepted_groups[language] = accepted

    # --- Pass 2: the real write. ---
    files = {split: open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") for split in ("train", "validation", "test")}
    split_counts = {
        split: {
            "rows": 0,
            "languages": Counter(),
            "labels": Counter(),
            "sources": Counter(),
            "by_language_source": {},  # language -> Counter(source -> rows)
        }
        for split in ("train", "validation", "test")
    }
    cap_dropped: Counter = Counter()
    overlap = {"pairs_checked": 0, "overlapping": 0, "examples": []}
    group_stats = {source: {"rows": 0, "groups": set()} for source in GROUP_TRACKED_SOURCES}

    # --- Adjacent-span merge accounting (see merge_adjacent_entities). ---
    # spans_before/spans_after are keyed (label, language) so the report can
    # state a merge count AND the share of that type's spans it represents,
    # per language, rather than a bare total that hides which language the
    # change lands on. merged_spans counts the *output* spans built from two
    # or more inputs; `spans_before - spans_after` is how many inputs
    # disappeared into a predecessor. Both are reported: they differ
    # whenever a run of three or more spans collapses into one, and a
    # report showing only one of them could not tell that case apart.
    spans_before: Counter = Counter()
    spans_after: Counter = Counter()
    merged_spans: Counter = Counter()
    merged_by_source: Counter = Counter()
    # Reservoir of merged spans WITH surrounding text, per (label, language),
    # so a human can read what the rule actually did instead of trusting a
    # count. Seeded from `seed` so the same build always yields the same
    # sample.
    merge_sample_rng = random.Random(_mix(seed, "merge-sample"))
    merge_samples: dict[tuple[str, str], list[dict]] = {}
    merge_seen: Counter = Counter()
    MERGE_SAMPLE_CAP = 60
    MERGE_CONTEXT = 60

    def _observe_merge(text: str, entities: list[dict], language: str, source: str) -> list[dict]:
        """Merge `entities` and record everything the report needs about
        what changed. Returns the merged list."""
        for entity in entities:
            spans_before[(entity["label"], language)] += 1
        starts_before = {(e["start"], e["label"]) for e in entities}
        merged, n_merges = merge_adjacent_entities(text, entities)
        for entity in merged:
            spans_after[(entity["label"], language)] += 1
        if not n_merges:
            return merged
        merged_by_source[source] += 1
        # An output span is "merged" iff it absorbed something: it starts
        # where an input started (merging never moves a start) but ends
        # further right than that input did.
        ends_before = {(e["start"], e["label"]): e["end"] for e in entities}
        for entity in merged:
            key = (entity["start"], entity["label"])
            if key in starts_before and entity["end"] > ends_before[key]:
                label, start, end = entity["label"], entity["start"], entity["end"]
                merged_spans[(label, language)] += 1
                bucket_key = (label, language)
                merge_seen[bucket_key] += 1
                bucket = merge_samples.setdefault(bucket_key, [])
                sample = {
                    "language": language,
                    "label": label,
                    "source": source,
                    "before": text[max(0, start - MERGE_CONTEXT) : start],
                    "span": text[start:end],
                    "after": text[end : end + MERGE_CONTEXT],
                    "parts": [
                        text[e["start"] : e["end"]]
                        for e in entities
                        if e["label"] == label and start <= e["start"] and e["end"] <= end
                    ],
                }
                if len(bucket) < MERGE_SAMPLE_CAP:
                    bucket.append(sample)
                else:
                    j = merge_sample_rng.randrange(merge_seen[bucket_key])
                    if j < MERGE_SAMPLE_CAP:
                        bucket[j] = sample
        return merged

    def _write(split: str, text: str, entities: list[dict], language: str, source: str, group: str) -> None:
        record = {"text": text, "entities": entities, "language": language, "source": source, "group": group}
        files[split].write(json.dumps(record, ensure_ascii=False) + "\n")
        counts = split_counts[split]
        counts["rows"] += 1
        counts["languages"][language] += 1
        counts["sources"][source] += 1
        counts["by_language_source"].setdefault(language, Counter())[source] += 1
        for entity in entities:
            counts["labels"][entity["label"]] += 1
        if source in group_stats:
            group_stats[source]["rows"] += 1
            group_stats[source]["groups"].add(group)

    try:
        for source, text, spans, language, group, split in _tagged_rows(seed, max_rows_per_source):
            if source == "nym":
                # The STREET_ADDRESS/CITY span question this task was
                # asked to close, measured over EVERY nym row this pass
                # sees, using the RAW (pre-taxonomy) labels -- and,
                # deliberately, BEFORE the cap-drop check below. A row
                # the cap later excludes from train is still a real nym
                # row with real spans; measuring it here first is what
                # keeps "over the full nym stream" true. Moving this
                # after the cap check once silently narrowed it to
                # whatever survived capping (measured: pairs checked fell
                # from 107,712 to 100,073) -- a measurement claiming a
                # scope it didn't have, the same class of error as the
                # contiguity docstring claim below.
                addr_spans = [(s, e) for s, e, label in spans if label == "STREET_ADDRESS"]
                city_spans = [(s, e) for s, e, label in spans if label == "CITY"]
                for a_s, a_e in addr_spans:
                    for c_s, c_e in city_spans:
                        overlap["pairs_checked"] += 1
                        if _spans_overlap(a_s, a_e, c_s, c_e):
                            overlap["overlapping"] += 1
                            if len(overlap["examples"]) < 8:
                                overlap["examples"].append(
                                    {"address": text[a_s:a_e], "city": text[c_s:c_e], "text": text}
                                )

            # An example whose only entity was dropped by the taxonomy map
            # is kept, not discarded -- it becomes a negative, which is
            # useful. (entities may legitimately end up empty; that's not
            # a reason to skip the row.)
            if split == "train" and source != "generated" and language in accepted_groups:
                if group not in accepted_groups[language]:
                    cap_dropped[language] += 1
                    continue

            if source == "generated":
                entities = [{"start": s, "end": e, "label": label} for s, e, label in spans]
            else:
                entities = []
                for s, e, label in spans:
                    mapped = taxonomy.to_platform(source, label)
                    if mapped is not None:
                        entities.append({"start": s, "end": e, "label": mapped})

            # One entity per name, not one per name *part* -- see
            # merge_adjacent_entities' own section above for the defect this
            # closes, the measured gap tally that chose the rule, and why it
            # runs on Temperate labels (after the taxonomy map) rather than
            # on each source's own sub-labels. Applied uniformly to all
            # three sources, no per-source special case: generate.py already
            # emits a whole name as one span, so it merges almost nothing
            # there, and `merge_stats` records that rather than assuming it.
            entities = _observe_merge(text, entities, language, source)

            _write(split, text, entities, language, source, group)
    finally:
        for f in files.values():
            f.close()

    result: dict = {
        split: {
            "rows": counts["rows"],
            "languages": dict(counts["languages"]),
            "labels": dict(counts["labels"]),
            "sources": dict(counts["sources"]),
            "by_language_source": {lang: dict(src_counts) for lang, src_counts in counts["by_language_source"].items()},
        }
        for split, counts in split_counts.items()
    }
    result["caps"] = {"train_language_cap": TRAIN_LANGUAGE_CAP, "dropped_rows_by_language": dict(cap_dropped)}
    overlap["rate"] = overlap["overlapping"] / overlap["pairs_checked"] if overlap["pairs_checked"] else 0.0
    result["nym_street_address_city_overlap"] = overlap
    result["group_stats"] = {
        source: {"rows": stats["rows"], "distinct_groups": len(stats["groups"])} for source, stats in group_stats.items()
    }
    result["span_merge"] = {
        "mergeable_types": sorted(MERGEABLE_TYPES),
        "by_label_language": {
            f"{label}|{language}": {
                "spans_before": before,
                "spans_after": spans_after[(label, language)],
                "spans_absorbed": before - spans_after[(label, language)],
                "merged_spans": merged_spans[(label, language)],
            }
            for (label, language), before in sorted(spans_before.items())
        },
        "rows_with_a_merge_by_source": dict(merged_by_source),
        "samples": [s for bucket in merge_samples.values() for s in bucket],
    }
    return result


def _du(path: Path) -> int:
    """Total bytes of every file under `path`, recursively. Cheaper and
    more portable inside this module than shelling out to `du`."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} TB"  # pragma: no cover


def render_report(counts: dict, *, seed: int, out_dir: Path, command: str, elapsed: float) -> str:
    import datetime

    now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    splits = ("train", "validation", "test")

    total_rows = sum(counts[s]["rows"] for s in splits)
    rows_table = ["| split | rows | share |", "|---|---:|---:|"]
    for s in splits:
        n = counts[s]["rows"]
        rows_table.append(f"| {s} | {n:,} | {100 * n / total_rows:.2f}% |")
    rows_table.append(f"| **total** | **{total_rows:,}** | 100.00% |")

    all_sources = sorted(set().union(*[counts[s]["sources"] for s in splits]))
    split_source_table = ["| split | " + " | ".join(all_sources) + " | total |", "|---|" + "---:|" * (len(all_sources) + 1)]
    for s in splits:
        row = [counts[s]["sources"].get(src, 0) for src in all_sources]
        split_source_table.append(f"| {s} | " + " | ".join(f"{n:,}" for n in row) + f" | {sum(row):,} |")

    all_langs = sorted(set().union(*[counts[s]["languages"] for s in splits]))
    lang_table = ["| language | supported | train | validation | test | total |", "|---|:---:|---:|---:|---:|---:|"]
    for lang in all_langs:
        row = [counts[s]["languages"].get(lang, 0) for s in splits]
        lang_table.append(
            f"| {lang} | {'yes' if lang in SUPPORTED_LANGUAGES else ''} | {row[0]:,} | {row[1]:,} | {row[2]:,} | {sum(row):,} |"
        )
    # Every present language should already be in SUPPORTED_LANGUAGES --
    # _tagged_rows filters everything else out before it reaches the
    # counters this report reads from. `unsupported_present` catches a
    # regression in that filter directly (a language showing up here that
    # shouldn't); `missing_supported` catches the coverage guarantee this
    # report used to call "all 24 EU languages" before the 2026-08-21
    # narrowing -- now "all six supported languages".
    unsupported_present = [l for l in all_langs if l not in SUPPORTED_LANGUAGES]
    missing_supported = [l for l in sorted(SUPPORTED_LANGUAGES) if any(counts[s]["languages"].get(l, 0) == 0 for s in splits)]

    # language x source, train split only -- this is precisely the view that
    # would have shown findings 1 and 2 (both invisible in an
    # aggregated-across-sources report): a language with generated-source
    # rows in validation/test but none in train, or a language whose train
    # rows come from only one of the two large corpora despite both
    # naturally covering it. Every GENERATE_LANGUAGES entry is included even
    # if it ends up with zero generated train rows, specifically so that
    # regression is visible in the report rather than only in a test.
    train_by_language_source = counts["train"]["by_language_source"]
    lang_source_train_table = ["| language | generated | ai4privacy | nym |", "|---|---:|---:|---:|"]
    for lang in sorted(set(train_by_language_source) | set(GENERATE_LANGUAGES)):
        srcs = train_by_language_source.get(lang, {})
        lang_source_train_table.append(
            f"| {lang} | {srcs.get('generated', 0):,} | {srcs.get('ai4privacy', 0):,} | {srcs.get('nym', 0):,} |"
        )

    all_labels = sorted(set().union(*[counts[s]["labels"] for s in splits]))
    label_table = ["| label | train | validation | test | total |", "|---|---:|---:|---:|---:|"]
    for label in all_labels:
        row = [counts[s]["labels"].get(label, 0) for s in splits]
        label_table.append(f"| {label} | {row[0]:,} | {row[1]:,} | {row[2]:,} | {sum(row):,} |")

    caps = counts["caps"]
    if caps["dropped_rows_by_language"]:
        cap_lines = ["| language | rows dropped (over cap) |", "|---|---:|"]
        for lang, n in sorted(caps["dropped_rows_by_language"].items(), key=lambda kv: -kv[1]):
            cap_lines.append(f"| {lang} | {n:,} |")
        cap_table = "\n".join(cap_lines)
    else:
        cap_table = "No language exceeded the cap in this run."

    overlap = counts["nym_street_address_city_overlap"]
    if overlap["examples"]:
        ex_lines = []
        for ex in overlap["examples"]:
            ex_lines.append(f"- address span `{ex['address']!r}`, city span `{ex['city']!r}`, in: `{ex['text'][:200]!r}`")
        examples_md = "\n".join(ex_lines)
    else:
        examples_md = "(none found)"

    output_bytes = _du(out_dir)

    # --- Adjacent-span merge tables (see merge_adjacent_entities) ---------
    merge = counts["span_merge"]
    merge_rows = merge["by_label_language"]
    merge_totals: dict[str, dict[str, int]] = {}
    for key, m in merge_rows.items():
        label = key.split("|", 1)[0]
        t = merge_totals.setdefault(label, {"spans_before": 0, "spans_after": 0, "spans_absorbed": 0, "merged_spans": 0})
        for field in t:
            t[field] += m[field]

    merge_type_table = [
        "| type | spans before | spans after | spans absorbed | merged spans | share of this type's pre-merge spans |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in sorted(merge_totals):
        t = merge_totals[label]
        share = t["spans_absorbed"] / t["spans_before"] if t["spans_before"] else 0.0
        merge_type_table.append(
            f"| {label} | {t['spans_before']:,} | {t['spans_after']:,} | {t['spans_absorbed']:,} | {t['merged_spans']:,} | {share:.2%} |"
        )

    merge_lang_table = [
        "| type | language | spans before | spans absorbed | merged spans | share absorbed |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key in sorted(merge_rows, key=lambda k: (k.split("|")[0], k.split("|")[1])):
        m = merge_rows[key]
        if not m["spans_absorbed"]:
            continue
        label, language = key.split("|", 1)
        share = m["spans_absorbed"] / m["spans_before"] if m["spans_before"] else 0.0
        merge_lang_table.append(
            f"| {label} | {language} | {m['spans_before']:,} | {m['spans_absorbed']:,} | {m['merged_spans']:,} | {share:.2%} |"
        )

    merge_source_lines = "\n".join(
        f"- `{source}`: {n:,} rows had at least one merge"
        for source, n in sorted(merge["rows_with_a_merge_by_source"].items())
    ) or "- (no row merged anything)"

    return f"""# Dataset: merged, grouped, split

Generated {now_iso} by `training/build_dataset.py`. Command:

```
{command}
```

Wall time: {elapsed:.1f}s ({elapsed / 60:.1f} min). Output on disk: {_fmt_bytes(output_bytes)} ({out_dir}).

## Rows per split

Split by *group* (a source-agnostic template-shape hash -- see
build_dataset.py's module docstring), not by row, so these are not
expected to hit exactly {SPLIT_RATIOS['train']:.0%}/{SPLIT_RATIOS['validation']:.0%}/{SPLIT_RATIOS['test']:.0%}: a language or
template that clusters into few, large groups can pull its split's actual
row share away from the target ratio even though every individual group
still landed on exactly one side.

{chr(10).join(rows_table)}

## Rows per split, per source

Findings 1 and 2 of the review that added this table were both invisible
in a report that only aggregates across sources -- a language could be
missing entirely from `generated`'s train contribution, or a capped
language's train rows could come from only one of the two large corpora,
and the aggregated totals alone would look unremarkable either way.

{chr(10).join(split_source_table)}

## Rows per language, per split

{chr(10).join(lang_table)}

{"**All six supported languages are present in every split.**" if not missing_supported else "**MISSING from at least one split: " + ", ".join(missing_supported) + "**"}
{"" if not unsupported_present else chr(10) + "**UNSUPPORTED language(s) present -- the SUPPORTED_LANGUAGES filter did not catch everything: " + ", ".join(unsupported_present) + "**"}

## Rows per language, per source (train split only)

{chr(10).join(lang_source_train_table)}

## Entities per label, per split

Labels are platform types only (`taxonomy.PLATFORM_TYPES`) -- every source label
that maps to `None` was dropped from `entities`, not from the row; an
example whose only entity was dropped this way is kept as a negative.

{chr(10).join(label_table)}

## Training-set language cap

Cap: {caps['train_language_cap']:,} rows/language in `train` only (validation
and test are uncapped). Rows over the cap are dropped entirely, never
moved to another split -- see build_dataset.py's module docstring for why
150,000 was chosen (roughly French's own combined total, the next-largest
language after English at 381,404 combined per corpora-profile.md).

{cap_table}

## Adjacent-span merging: one entity per name, not one per name part

Both corpora label a given name and a surname as separate entities and
`taxonomy.py` maps both onto `PERSON`, so a model trained on the unmerged
data learns to detect a name *part*. This build joins runs of same-type
spans of a **mergeable type** separated only by a **bridgeable gap**:

- Mergeable types: {", ".join(f"`{t}`" for t in merge["mergeable_types"])}. `PERSON` and `STREET_ADDRESS`
  are the only two platform types built by collapsing several positionally
  adjacent sub-labels of one field, but only `PERSON` is merged here
  (2026-08-22): address detection was not broken (0.9877 held-out F1, zero
  fixture misses of this shape) and the merge does not fix the one known
  `STREET_ADDRESS` defect either (`Kerkstraat 12, 1016 GV` stays two spans
  regardless -- the comma between them is not a bridgeable gap). The
  deciding factor was that a `STREET_ADDRESS` merge is not measurable:
  `PERSON` has an external check (named fixture misses that either get
  fixed or do not), `STREET_ADDRESS` does not. `NATIONAL_ID`,
  `PHONE_NUMBER` and `CREDIT_CARD` also draw from more than one source
  label, but those are alternate *complete* values; two adjacent card
  numbers are two cards.
- Bridgeable gap: empty, or horizontal whitespace only. Never a comma
  (a list separator that would fuse one person's surname to the next
  person's given name), never a period (username/e-mail local parts, and
  sentence boundaries), never a conjunction, never a line break.

See `build_dataset.merge_adjacent_entities`' own comment block for the
measured gap tally that chose this rule and for the tussenvoegsel check.

**Comparability warning.** Merged spans make the held-out evaluation
measure a different task -- whole names instead of name parts -- so the
`PERSON` F1 from a model trained on this data is not comparable with the
figures in `reports/full-run-cuda.md` (PERSON 0.9237 on the fixture, 0.8971
on validation). `STREET_ADDRESS` is unaffected by this build and remains
comparable. A number that moves is not by itself an improvement or a
regression.

### By type

{chr(10).join(merge_type_table)}

### By type and language (rows where anything merged)

{chr(10).join(merge_lang_table)}

### Rows with at least one merge, by source

{merge_source_lines}

## nym STREET_ADDRESS / CITY span overlap (the open question from the dataset build)

Every nym row carrying both a STREET_ADDRESS span and a CITY span, checked
for character-range overlap between the two spans, over the full nym
stream ({NYM_EXPECTED_ROWS:,} rows):

- Pairs checked: {overlap['pairs_checked']:,}
- Overlapping: {overlap['overlapping']:,}
- Rate: {overlap['rate']:.4%}

{examples_md}

{"nym's STREET_ADDRESS spans never included the paired CITY span's characters in this corpus, at the rate measured above -- consistent with the dataset build's reviewer hypothesis (Faker's `street_address()`, not `address()`)." if overlap['overlapping'] == 0 else "**Non-zero overlap measured.** This is not silently fixed here -- see the examples above; whether to trim spans or drop these rows is a decision for the human partner (see the task report)."}
"""


def main() -> None:
    import argparse
    import shlex
    import sys
    import time

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, help="Directory to write train/validation/test.jsonl to")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-out", help="Path to write a dataset.md-style markdown report to")
    args = parser.parse_args()

    command = "python " + " ".join(shlex.quote(a) for a in sys.argv)

    t0 = time.time()
    counts = build(Path(args.out_dir), seed=args.seed)
    elapsed = time.time() - t0
    print(json.dumps(counts, indent=2, ensure_ascii=False))
    print(f"# elapsed: {elapsed:.1f}s", flush=True)

    if args.report_out:
        report = render_report(counts, seed=args.seed, out_dir=Path(args.out_dir), command=command, elapsed=elapsed)
        Path(args.report_out).write_text(report)
        print(f"# wrote report to {args.report_out}", flush=True)


if __name__ == "__main__":
    main()
