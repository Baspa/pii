"""Tests for training/eval_fixture_rules.py -- the harness that scores the
Rust engine's spans, and the union of those with the model's, using the same
fixture_scoring.py rules that scored the model and Azure.

The tests worth having here are the ones guarding the two ways this
measurement could be quietly wrong rather than loudly broken:

1. OFFSET CONVENTION. Two separate JSON parsers (serde_json in Rust, json in
   Python) read the same fixture, and every number in the report depends on
   them agreeing about where each character is. A byte/code-point confusion,
   a stale spans file, or a fixture edited between the two runs all show up
   the same way: offsets that no longer point at what they pointed at. The
   guard is a per-case `char_len` comparison plus a bounds check, and these
   tests pin that it raises rather than scoring something meaningless.

2. THE UNION PROPERTY. Combined recall must never be lower than either half
   alone, and combined over-masking must never be lower either -- that is
   what "union of spans" means, and if it ever fails to hold, the union is
   not being taken. Asserted against fixture_scoring itself, not restated.
"""

from __future__ import annotations

import pytest

from eval_fixture_rules import (
    azure_typed_spans_by_id,
    overlapping_types,
    rules_spans_by_id,
    union_spans,
)
from fixture_scoring import score_case


def cases() -> list[dict]:
    return [
        {"id": "a", "lang": "nl", "text": "Bel 020-1234567", "must_mask": [], "must_clear": []},
        {"id": "b", "lang": "de", "text": "Grüße an x@y.nl", "must_mask": [], "must_clear": []},
    ]


def spans_doc(cases_out: list[dict]) -> dict:
    return {"fixture": "f.json", "command": "c", "cases": cases_out}


def test_spans_are_accepted_when_both_parsers_agree_about_every_case() -> None:
    doc = spans_doc(
        [
            {"id": "a", "char_len": 15, "spans": [{"start": 4, "end": 15}]},
            {"id": "b", "char_len": 15, "spans": []},
        ]
    )
    assert rules_spans_by_id(doc, cases()) == {"a": [(4, 15)], "b": []}


def test_a_case_whose_length_disagrees_is_refused_rather_than_scored() -> None:
    # 17 is what a byte-counting producer would report for "Grüße an x@y.nl"
    # (two two-byte characters). Every offset in that case would then be
    # shifted, and the recall it produced would be wrong in whichever
    # direction the shift happened to fall -- silently.
    doc = spans_doc(
        [
            {"id": "a", "char_len": 15, "spans": []},
            {"id": "b", "char_len": 17, "spans": []},
        ]
    )
    with pytest.raises(RuntimeError, match="the two parses disagree"):
        rules_spans_by_id(doc, cases())


def test_a_span_past_the_end_of_the_text_is_refused() -> None:
    doc = spans_doc(
        [
            {"id": "a", "char_len": 15, "spans": [{"start": 4, "end": 99}]},
            {"id": "b", "char_len": 15, "spans": []},
        ]
    )
    with pytest.raises(RuntimeError, match="outside the text"):
        rules_spans_by_id(doc, cases())


def test_a_spans_file_in_a_different_order_is_refused() -> None:
    doc = spans_doc(
        [
            {"id": "b", "char_len": 15, "spans": []},
            {"id": "a", "char_len": 15, "spans": []},
        ]
    )
    with pytest.raises(RuntimeError, match="different order"):
        rules_spans_by_id(doc, cases())


def test_a_spans_file_for_a_different_number_of_cases_is_refused() -> None:
    doc = spans_doc([{"id": "a", "char_len": 15, "spans": []}])
    with pytest.raises(RuntimeError, match="the fixture has"):
        rules_spans_by_id(doc, cases())


def test_the_union_masks_what_either_side_masks() -> None:
    left = {"a": [(0, 4)], "b": [(2, 5)]}
    right = {"a": [(6, 9)], "c": [(1, 2)]}
    assert union_spans(left, right) == {"a": [(0, 4), (6, 9)], "b": [(2, 5)], "c": [(1, 2)]}


def test_the_union_never_loses_a_value_either_side_found() -> None:
    # One case, two values: the left side covers the first, the right side
    # covers the second, and neither covers both. Combined must find both --
    # this is the property the whole "rules + model" number rests on.
    case = {
        "id": "x",
        "lang": "nl",
        "text": "Jan de Vries op 020-1234567",
        "must_mask": ["Jan de Vries", "020-1234567"],
        "must_clear": [],
    }
    left = [(0, 12)]
    right = [(16, 27)]

    def found(spans: list[tuple[int, int]]) -> list[bool]:
        return [m["found_all"] for m in score_case(case, spans, lambda _: None)["must_mask"]]

    assert found(left) == [True, False]
    assert found(right) == [False, True]
    assert found(union_spans({"x": left}, {"x": right})["x"]) == [True, True]


def test_the_union_also_inherits_every_over_masking_violation() -> None:
    # The direction nobody wants to discover late: a union can only mask
    # more, so a must_clear value either side violates is violated by the
    # union too.
    case = {
        "id": "x",
        "lang": "nl",
        "text": "Factuur 123456789 en order 987654321",
        "must_mask": [],
        "must_clear": ["123456789", "987654321"],
    }
    left = [(8, 17)]
    right = [(27, 36)]

    def violated(spans: list[tuple[int, int]]) -> list[bool]:
        return [c["violated"] for c in score_case(case, spans, lambda _: None)["must_clear"]]

    assert violated(left) == [True, False]
    assert violated(right) == [False, True]
    assert violated(union_spans({"x": left}, {"x": right})["x"]) == [True, True]


def test_overlapping_types_names_only_the_recognizers_that_touch_the_value() -> None:
    typed = [(0, 5, "PHONE_NUMBER"), (10, 20, "EMAIL_ADDRESS"), (3, 8, "PHONE_NUMBER")]
    assert overlapping_types(typed, [[4, 6]]) == ["PHONE_NUMBER"]
    assert overlapping_types(typed, [[12, 14]]) == ["EMAIL_ADDRESS"]
    assert overlapping_types(typed, [[30, 40]]) == []


def test_azure_categories_come_from_the_saved_responses_not_a_second_call() -> None:
    azure = {
        "raw_batches": [
            {
                "raw_response": {
                    "results": {
                        "documents": [
                            {
                                "id": "x",
                                "entities": [
                                    {"category": "Person", "offset": 0, "length": 12},
                                    {"category": "Email", "offset": 20, "length": 0},
                                ],
                            }
                        ]
                    }
                }
            }
        ]
    }
    # The zero-length entity is dropped: it cannot overlap anything, and
    # feeding it forward would put a category on a value nothing covered.
    assert azure_typed_spans_by_id(azure) == {"x": [(0, 12, "Person")]}


def test_azure_categories_are_empty_when_no_raw_responses_were_saved() -> None:
    assert azure_typed_spans_by_id({"summary": {}}) == {}
