"""Tests for training/build_dataset.py.

These were originally written verbatim from this task's brief, each
independently calling `build_dataset.build(tmp_path, seed=0)` -- three full
passes over the 2.44M-row corpora for one task whose whole job is one pass.
That was corrected after the first real run: `built_dataset` below calls
`build()` exactly once, session-scoped, and every test reads its result
instead of re-running the pipeline. This still asserts the grouped-split
property against the full, real dataset -- not a capped slice, which would
let the straddling check pass vacuously if the cap were small enough that
only one template's rows appeared at all (a failure mode this project has
hit before: a guard that looks like it's checking something but never
exercises the risk).

The other fix here: the brief's original reading code was
`(tmp_path / f"{split}.jsonl").read_text().splitlines()`. Python's
`str.splitlines()` treats U+0085 (NEL), U+2028 (LINE SEPARATOR) and
U+2029 (PARAGRAPH SEPARATOR) as line breaks, in addition to `\n`/`\r`.
`json.dumps` does not escape any of the three (only ASCII control
characters below 0x20 are auto-escaped), and both ai4privacy and nym
contain real text with these characters. `splitlines()` then cuts a
single JSON record into two truncated fragments, and `json.loads` fails
on the first one with "Unterminated string" -- reproduced against the
real output, first byte-for-byte identical on a row starting `Sig.ra
Sivanantharajah Babicz...`. This is a reading bug, not a writing bug: the
JSON itself is one physical line (`\n`-terminated) and completely valid;
`str.splitlines()` is simply a broader definition of "line" than JSONL
uses. `for line in handle:` (a plain file-object iteration) splits only
on `\n`, which is what JSONL actually promises, so every test below reads
that way instead.
"""

import json
from pathlib import Path

import pytest

import build_dataset
import taxonomy


@pytest.fixture(scope="session")
def built_dataset(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("dataset")
    counts = build_dataset.build(out_dir, seed=0)
    return out_dir, counts


def _read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def test_no_template_shape_appears_in_two_splits(built_dataset):
    # Both corpora are template-generated. A random split puts near-copies
    # of one template on both sides, and every metric comes out inflated.
    out_dir, _ = built_dataset
    groups = {}
    for split in ("train", "validation", "test"):
        for record in _read_jsonl(out_dir / f"{split}.jsonl"):
            groups.setdefault(record["group"], set()).add(split)
    straddling = {g: s for g, s in groups.items() if len(s) > 1}
    assert not straddling, f"{len(straddling)} template groups straddle splits"


def test_every_split_carries_every_supported_language(built_dataset):
    # Narrowed 2026-08-21 from 24 EU languages to the six the detector actually
    # trains, measures and gates on -- en, nl, de, fr, es, it (design
    # section 10a). Reads build_dataset.SUPPORTED_LANGUAGES directly
    # rather than a second hardcoded list, so this test and the filter it
    # checks can't drift apart. Asserted as exact equality, not just
    # supersetting: the property under test is now both "every supported
    # language is present" (the original guarantee) and "the language
    # filter actually filtered" (nothing unsupported leaked through) --
    # either direction failing would mean the same underlying bug: a
    # split's language set no longer matches SUPPORTED_LANGUAGES.
    _, counts = built_dataset
    for split in ("train", "validation", "test"):
        present = set(counts[split]["languages"])
        assert present == set(build_dataset.SUPPORTED_LANGUAGES), (
            f"{split}: expected exactly {sorted(build_dataset.SUPPORTED_LANGUAGES)}, got {sorted(present)}"
        )


def test_labels_are_platform_types_only(built_dataset):
    from taxonomy import PLATFORM_TYPES

    out_dir, _ = built_dataset
    for record in _read_jsonl(out_dir / "train.jsonl"):
        for entity in record["entities"]:
            assert entity["label"] in PLATFORM_TYPES


# ---------------------------------------------------------------------------
# Added after a review found four real bugs measured against the first full
# run's real output -- each is a "guard that wasn't guarding" in the same
# family the module docstring's history already tracks. Each test below is
# a direct regression guard for one of them, confirmed to fail against the
# pre-fix code before the corresponding fix, per the project's own standing
# practice of never trusting a guard that hasn't been seen to fail.
# ---------------------------------------------------------------------------


def test_group_id_recognizes_same_template_different_fillers_and_different_text():
    # The property the whole task depends on: _group_id must be a real
    # function of the TEMPLATE, not a random label. Swap it for uuid4()
    # and test_no_template_shape_appears_in_two_splits would still pass --
    # every group would have exactly one member, so none could ever
    # straddle a split, and the test would provide zero protection. This
    # test exercises _group_id directly instead, offline and fast: same
    # surrounding text with different entity fillers must group together;
    # genuinely different surrounding text must not.
    a_text, a_spans = "Hello Maria, your order is ready.", [(6, 11)]
    b_text, b_spans = "Hello Wojciech, your order is ready.", [(6, 14)]
    assert build_dataset._group_id(a_text, a_spans) == build_dataset._group_id(b_text, b_spans)

    c_text, c_spans = "Your package has shipped, Maria.", [(27, 32)]
    assert build_dataset._group_id(a_text, a_spans) != build_dataset._group_id(c_text, c_spans)


def test_group_id_ignores_case_and_whitespace_perturbations():
    # nym augments its own corpus with case and whitespace perturbation of
    # the same underlying template instance. Before this test's fix, those
    # augmented copies hashed to different groups and could land on
    # opposite sides of a split -- measured against the real pre-fix
    # output: a materially double-digit percentage of nym's own test rows
    # had a case/whitespace-normalized match sitting in train (see
    # training/reports/dataset.md and task-4-report.md for the exact
    # figures and how they were measured). This is the direct,
    # offline unit-test guard for that fix.
    a = "Please restart the service and confirm the health check passes."
    b = "PLEASE  RESTART the Service and confirm the health  check passes."
    assert build_dataset._group_id(a, []) == build_dataset._group_id(b, [])


def test_nym_groups_are_materially_fewer_than_nym_rows(built_dataset):
    # An integration-level backstop for the same property, on the real
    # dataset: nym's own label is "template-generated with real,
    # measurable reuse" (see the module docstring). If _group_id's key
    # ever gets finer -- an over-eager normalisation change, a stray
    # per-row nonce accidentally folded into the hash input, etc. -- nym's
    # distinct-group count would drift back up toward its row count, and
    # nothing else in this suite would notice (a finer key still can't
    # make a group straddle two splits; it just quietly stops protecting
    # against near-duplicates, which is exactly how this task's central
    # property silently stopped mattering once already, before this round
    # of review).
    _, counts = built_dataset
    stats = counts["group_stats"]["nym"]
    assert stats["rows"] > 0, "expected nym rows in the real build"
    assert stats["distinct_groups"] < 0.5 * stats["rows"], (
        f"nym: {stats['distinct_groups']} distinct groups over {stats['rows']} rows -- "
        "not materially below row count; the grouping key may have gotten finer"
    )


def test_every_generated_language_has_train_rows(built_dataset):
    # generate.py's 2,700 records exist specifically to teach PERSON/
    # STREET_ADDRESS/NATIONAL_ID/TAX_ID patterns the two large corpora
    # under-serve. Before this fix, `n == 2`-group languages (measured:
    # de, es, it, nl, pl) put one of their two generated-template groups
    # in validation and the other in test, leaving train with zero
    # generated rows -- the exact content the data generation was built to add was
    # then present only where the model is evaluated, never where it
    # learns.
    out_dir, _ = built_dataset
    covered = {
        record["language"]
        for record in _read_jsonl(out_dir / "train.jsonl")
        if record["source"] == "generated"
    }
    missing = set(build_dataset.GENERATE_LANGUAGES) - covered
    assert not missing, f"generated languages with zero train rows: {missing}"


def test_capped_language_train_rows_come_from_more_than_one_source(built_dataset):
    # Before this fix, the language cap dropped train rows once a running
    # per-language counter (incremented in stream order: generated, then
    # ai4privacy, then nym) hit the threshold. ai4privacy streams first
    # and its own English total (204,649) alone exceeds the 150,000 cap,
    # so the cap was always fully consumed before nym's English rows were
    # ever reached -- measured: ai4privacy 149,780, nym 0, generated 220.
    # English's evaluation set (which nym supplies ~30% of) then had no
    # matching representation in its own training set. Any language that
    # actually hits the cap in the real run must show train contributions
    # from more than one non-generated source.
    _, counts = built_dataset
    capped_languages = counts["caps"]["dropped_rows_by_language"]
    assert capped_languages, "expected at least one language to hit the cap in the real run"
    for language in capped_languages:
        sources = counts["train"]["by_language_source"].get(language, {})
        contributing = {s: n for s, n in sources.items() if s != "generated" and n > 0}
        assert len(contributing) > 1, (
            f"{language}'s train rows come only from {contributing} -- "
            "the cap may be dropping by stream position again"
        )


# ---------------------------------------------------------------------------
# Adjacent-span merging. Pure-function tests: they exercise
# merge_adjacent_entities directly, offline, so each rule in it fails on its
# own line instead of somewhere inside a 2.44M-row build. The counts these
# assert against real data live in reports/merge-verification.md, produced by
# training/verify_merged_spans.py.
# ---------------------------------------------------------------------------


def _merge(text, spans, mergeable_types=None):
    # `mergeable_types` defaults to None, meaning "use build_dataset's own
    # default" (MERGEABLE_TYPES, PERSON only as of 2026-08-22) -- passing it
    # explicitly is how a test below exercises merge_adjacent_entities' own
    # general logic (the letterless clause, the letter-bearing-run
    # accumulation) for STREET_ADDRESS without relying on the default the
    # real build no longer uses for that type.
    entities = [{"start": s, "end": e, "label": label} for s, e, label in spans]
    kwargs = {} if mergeable_types is None else {"mergeable_types": mergeable_types}
    merged, n = build_dataset.merge_adjacent_entities(text, entities, **kwargs)
    return [text[e["start"] : e["end"]] for e in merged], n


def test_a_given_name_and_a_surname_become_one_person():
    # The defect this whole change exists to fix: both corpora label these
    # as two entities, taxonomy.py maps both to PERSON, and the model
    # learned to detect a name *part*.
    text = "Beste team, Anna Schmidt heeft gebeld."
    assert _merge(text, [(12, 16, "PERSON"), (17, 24, "PERSON")]) == (["Anna Schmidt"], 1)


def test_a_tussenvoegsel_inside_the_surname_span_survives_the_merge():
    # Both corpora put the particle INSIDE the surname span (measured: 731
    # occurrences inside a PERSON span across validation+test, 0 outside
    # any span -- see reports/merge-verification.md section 0), so a
    # whitespace bridge is enough to recover the whole Dutch name.
    text = "Jan de Vries van Acme BV heeft gevraagd"
    assert _merge(text, [(0, 3, "PERSON"), (4, 12, "PERSON")]) == (["Jan de Vries"], 1)


def test_a_comma_is_not_bridged_because_it_separates_two_people():
    # Measured in the real corpus: `Asistentes: Ekansh [Abeka], [Tori]
    # Budavary` is one attendee's surname followed by the next attendee's
    # given name. Bridging ', ' fuses two different people.
    text = "Asistentes: Ekansh Abeka, Tori Budavary"
    assert _merge(text, [(19, 24, "PERSON"), (26, 30, "PERSON")]) == (["Abeka", "Tori"], 0)


def test_a_conjunction_is_not_bridged():
    text = "die Namen der Paten Fayad und Amant"
    assert _merge(text, [(20, 25, "PERSON"), (30, 35, "PERSON")]) == (["Fayad", "Amant"], 0)


def test_a_period_is_not_bridged_because_it_builds_usernames_and_sentences():
    # `Amanda`.`Salazar`@gmail.com in the corpus, and `... vroeg Paramita
    # Agnie [Damev]. [Boulos] antwoordde ...` -- a sentence boundary
    # between two different speakers.
    text = "Username: Amanda.Salazar"
    assert _merge(text, [(10, 16, "PERSON"), (17, 24, "PERSON")]) == (["Amanda", "Salazar"], 0)


def test_a_line_break_is_not_bridged():
    text = "Aanwezig:\nAnna\nMichael\n"
    assert _merge(text, [(10, 14, "PERSON"), (15, 22, "PERSON")]) == (["Anna", "Michael"], 0)


def test_non_breaking_and_narrow_spaces_are_bridged():
    # French typography uses U+202F/U+00A0 between name parts; 11,378 and
    # 615 of them respectively are bridged in the real build.
    text = "Nom : Taisa Scharegg"
    assert text[11] == "\u202f"
    assert _merge(text, [(6, 11, "PERSON"), (12, 20, "PERSON")]) == (["Taisa Scharegg"], 1)


def test_street_address_is_not_merged_by_default():
    # 2026-08-22: MERGEABLE_TYPES narrowed to PERSON only. STREET_ADDRESS
    # detection was not broken (0.9877 held-out F1, zero fixture misses of
    # this shape) and merging it would not even have fixed the one known
    # STREET_ADDRESS defect (`Kerkstraat 12, 1016 GV` stays two spans
    # either way -- the comma between them is not a bridgeable gap). The
    # deciding factor: a STREET_ADDRESS merge is not measurable the way
    # PERSON is (no equivalent of the six named fixture misses), so nothing
    # here is merged by the default the real build uses. See
    # reports/dataset-merged-names.md.
    text = "naar Kerkstraat 12 sturen"
    assert _merge(text, [(5, 15, "STREET_ADDRESS"), (16, 18, "STREET_ADDRESS")]) == (
        ["Kerkstraat", "12"],
        0,
    )


def test_a_street_name_and_its_building_number_would_become_one_address_if_asked():
    # merge_adjacent_entities' own logic is unchanged and still general --
    # only the default MERGEABLE_TYPES it is called with narrowed. Passing
    # an explicit override still merges a street name and its building
    # number, proving the previous test's zero merges is a default-types
    # decision, not a change to the merge mechanism itself.
    text = "naar Kerkstraat 12 sturen"
    assert _merge(
        text,
        [(5, 15, "STREET_ADDRESS"), (16, 18, "STREET_ADDRESS")],
        mergeable_types=frozenset({"STREET_ADDRESS"}),
    ) == (["Kerkstraat 12"], 1)


def test_two_spans_with_no_letters_are_never_joined():
    # Without this clause the rule fused a run of 33 French postcodes into
    # one 195-character STREET_ADDRESS span. A span with no letters carries
    # no evidence that it is *part* of a larger value rather than a
    # complete one -- the same judgement that keeps CREDIT_CARD out of
    # MERGEABLE_TYPES, applied one level down. Exercised here with an
    # explicit STREET_ADDRESS override so the assertion still depends on
    # the letterless clause rather than passing vacuously because
    # STREET_ADDRESS is no longer a default-mergeable type at all.
    text = "69440 14113 27190"
    assert _merge(
        text,
        [(0, 5, "STREET_ADDRESS"), (6, 11, "STREET_ADDRESS"), (12, 17, "STREET_ADDRESS")],
        mergeable_types=frozenset({"STREET_ADDRESS"}),
    ) == (["69440", "14113", "27190"], 0)


def test_a_letter_bearing_run_still_absorbs_a_trailing_number():
    # ...but the accumulated span keeps its letters, so the legitimate
    # street/number/postcode run is not blocked by the clause above.
    # Exercised with an explicit STREET_ADDRESS override -- see the two
    # tests above for why the default no longer merges this type.
    text = "Musterstr 12 10115 Berlin"
    assert _merge(
        text,
        [(0, 9, "STREET_ADDRESS"), (10, 12, "STREET_ADDRESS"), (13, 18, "STREET_ADDRESS")],
        mergeable_types=frozenset({"STREET_ADDRESS"}),
    ) == (["Musterstr 12 10115"], 2)


def test_types_outside_MERGEABLE_TYPES_are_left_alone():
    # Two adjacent card numbers are two cards, not one -- the case the
    # task brief warns about.
    text = "4111111111111111 5500000000000004"
    assert _merge(text, [(0, 16, "CREDIT_CARD"), (17, 33, "CREDIT_CARD")]) == (
        ["4111111111111111", "5500000000000004"],
        0,
    )
    assert _merge(text, [(0, 16, "NATIONAL_ID"), (17, 33, "NATIONAL_ID")]) == (
        ["4111111111111111", "5500000000000004"],
        0,
    )


def test_different_types_never_merge_however_close():
    text = "Anna anna@example.com"
    assert _merge(text, [(0, 4, "PERSON"), (5, 21, "EMAIL_ADDRESS")]) == (
        ["Anna", "anna@example.com"],
        0,
    )


def test_overlapping_entities_are_left_exactly_as_they_are():
    # align.py does not resolve overlap and this builder has never produced
    # any (measured: 0 overlapping adjacent pairs in 821,530 rows).
    # Repairing one silently here would hide a corpus regression.
    text = "Anna Schmidt"
    assert _merge(text, [(0, 8, "PERSON"), (5, 12, "PERSON")]) == (["Anna Sch", "Schmidt"], 0)


def test_merging_is_idempotent():
    text = "Anna Schmidt en Michael Chen"
    spans = [(0, 4, "PERSON"), (5, 12, "PERSON"), (16, 23, "PERSON"), (24, 28, "PERSON")]
    entities = [{"start": s, "end": e, "label": label} for s, e, label in spans]
    once, first = build_dataset.merge_adjacent_entities(text, entities)
    twice, second = build_dataset.merge_adjacent_entities(text, once)
    assert once == twice
    assert (first, second) == (2, 0)
    assert [text[e["start"] : e["end"]] for e in once] == ["Anna Schmidt", "Michael Chen"]


def test_merged_labels_are_still_platform_types_and_spans_still_sorted(built_dataset):
    # Integration backstop on the real, merged dataset: whatever the merge
    # did, every entity must still be sorted, non-overlapping, in bounds
    # and of a platform type -- align.py depends on all four.
    out_dir, _ = built_dataset
    for split in ("train", "validation", "test"):
        for record in _read_jsonl(out_dir / f"{split}.jsonl"):
            entities = record["entities"]
            for a, b in zip(entities, entities[1:]):
                assert (a["start"], a["end"]) <= (b["start"], b["end"])
                assert b["start"] >= a["end"]
            for entity in entities:
                assert entity["label"] in taxonomy.PLATFORM_TYPES
                assert 0 <= entity["start"] < entity["end"] <= len(record["text"])
