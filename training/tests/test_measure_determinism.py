"""Tests for measure_determinism.py's ship-gate scoping.

`reports/student-v2-run.md` (C7) recorded a determinism criterion written
only as "0/200" as a MISS at 1/200 -- correctly, because the criterion never
said which of the six grouping pairs it meant, and one pair (any pairing of
`all`/`waste50` against `split`/`single`) did differ while `split` vs
`single` -- the only pair with a real consequence in the shipped service --
was 0/200, worst |Δlogit| exactly 0. `distillation-plan.md` S6 now settles
this as `split` vs `single`, and only that pair. `ship_gate_status` is the
part of the tool that turns the six-pair table into that single verdict
without a human having to pick the right line by hand.

The fixture data below is transcribed from `reports/determinism-student-v2.json`
and `reports/determinism-student-6l.json` (both already committed, produced
by real ONNX runs) rather than invented, so these tests pin the exact case
that actually occurred instead of a convenient stand-in for it.
"""

from __future__ import annotations

from measure_determinism import SHIP_GATE_PAIR, ship_gate_status


def _pair(argmax: list[str], spans: list[str], worst: float) -> dict:
    return {"argmax": argmax, "spans": spans, "worst_logit_delta": worst}


# Transcribed from reports/determinism-student-v2.json: the run that was
# scored a MISS at "1/200" with no pair named.
STUDENT_V2_GROUPINGS = {
    "all vs waste50": _pair([], [], 0.6031174659729004),
    "all vs split": _pair(["en-006"], ["en-006"], 1.104985237121582),
    "all vs single": _pair(["en-006"], ["en-006"], 1.104985237121582),
    "waste50 vs split": _pair(["en-006"], ["en-006"], 1.104985237121582),
    "waste50 vs single": _pair(["en-006"], ["en-006"], 1.104985237121582),
    "split vs single": _pair([], [], 0.0),
}

# Transcribed from reports/determinism-student-6l.json: the shipped student,
# 0/200 on all six pairs.
SHIPPED_GROUPINGS = {
    "all vs waste50": _pair([], [], 0.6344177722930908),
    "all vs split": _pair([], [], 0.6344177722930908),
    "all vs single": _pair([], [], 0.6344177722930908),
    "waste50 vs split": _pair([], [], 0.5540924072265625),
    "waste50 vs single": _pair([], [], 0.5540924072265625),
    "split vs single": _pair([], [], 0.0),
}


class TestShipGatePair:
    def test_the_gate_is_split_vs_single(self):
        # Pinned by name: this is the thing S6 was written down as, and a
        # future edit changing it silently should fail a test, not slide
        # through as a refactor.
        assert SHIP_GATE_PAIR == "split vs single"


class TestShipGateStatus:
    def test_student_v2_is_met_on_the_scoped_gate_despite_the_old_1_of_200(self):
        # This is the case the fix exists for: the run was recorded a MISS
        # against the unscoped criterion at 1/200, entirely on pairs that
        # are not the gate.
        gate = ship_gate_status(STUDENT_V2_GROUPINGS, num_texts=200)
        assert gate["pair"] == "split vs single"
        assert gate["spans_differ"] == 0
        assert gate["argmax_differs"] == 0
        assert gate["worst_logit_delta"] == 0.0
        assert gate["status"] == "MET"

    def test_shipped_student_is_met_too(self):
        gate = ship_gate_status(SHIPPED_GROUPINGS, num_texts=200)
        assert gate["spans_differ"] == 0
        assert gate["status"] == "MET"

    def test_a_difference_on_split_vs_single_itself_is_a_miss(self):
        # The one pair the gate actually depends on. If this one moves, the
        # docstring's own claim -- "a defect rather than a tolerance" --
        # means it must be reported as a miss, unconditionally.
        groupings = dict(STUDENT_V2_GROUPINGS)
        groupings["split vs single"] = _pair(["nl-003"], ["nl-003"], 0.5)
        gate = ship_gate_status(groupings, num_texts=200)
        assert gate["status"] == "MISS"
        assert gate["spans_differ"] == 1

    def test_a_difference_only_on_a_padding_pair_does_not_move_the_gate(self):
        # `all` and `waste50` are superseded batching behaviour; a
        # difference confined to pairs naming them must not leak into S6.
        groupings = {k: _pair([], [], 0.0) for k in STUDENT_V2_GROUPINGS}
        groupings["all vs waste50"] = _pair(["de-001"], ["de-001"], 0.9)
        gate = ship_gate_status(groupings, num_texts=200)
        assert gate["status"] == "MET"
        assert gate["spans_differ"] == 0

    def test_spans_total_is_the_corpus_size_not_a_derived_count(self):
        gate = ship_gate_status(STUDENT_V2_GROUPINGS, num_texts=200)
        assert gate["spans_total"] == 200
