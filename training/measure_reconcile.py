"""Measures whether the design's furthest-from-edge
window rule (`pii_detection_model::decode::reconcile`) cost recall against
`eval_fixture.py`'s `_merge_touching_typed`, and does whitespace trimming
change anything on real fixture text?

This is a one-off measurement script, not a test: it runs one ONNX
inference pass per fixture case, which is not something to pay on every
`pytest` invocation. Read-only against `benchmark/` -- nothing here writes
there. Results are written by hand into `training/reports/export.md`'s
A later measurement; this script is what produced them and is the way to
reproduce them.

WHY A SEPARATE RECONCILE IMPLEMENTATION HERE, NOT A CALL INTO RUST. This
script measures a Python-side question (how the two *rules* compare, so
the number this work already quotes -- 0.9145, produced under the merge
rule -- can be explained rather than silently superseded) before the service wiring
wires the Rust service up to be callable at all. `furthest_from_edge` below
mirrors `pii_detection_model::decode::reconcile` in `crates/model/src/decode.rs`
line for line (same sort key, same greedy overlap sweep) -- it is not a
second design, it is that function transcribed into Python for the one
place in this work that needs to run it before Rust is wired end to end.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

from align import IGNORE_INDEX, reconstruct_spans, tokenize_and_align, window_real_range
from eval_fixture import _merge_touching_typed, onnx_predictor
from fixture_scoring import coverage_found, find_occurrences

TOKENIZER_NAME = "FacebookAI/xlm-roberta-base"
MEASURED_TYPES = ("PERSON", "STREET_ADDRESS")


def load_cases_with_type(directory: Path) -> list[dict]:
    """Every case in every `*.json` directly under `directory`, same
    contract as `fixture_scoring.load_case_dir` (sorted-path order,
    concatenated, non-array file is an error) -- except `must_mask` stays
    as the original `{"value", "type"}` objects (or bare strings, which get
    no `type`) instead of being flattened to values. Per-type recall is the
    whole point of this measurement, so the type cannot be the thing this
    loader throws away.
    """
    cases: list[dict] = []
    for file in sorted(directory.glob("*.json")):
        parsed = json.loads(file.read_text())
        if not isinstance(parsed, list):
            raise ValueError(f"{file} is not a JSON array of cases")
        cases.extend(parsed)
    return cases


def furthest_from_edge(
    spans: list[tuple[int, int, str, int, int]],
) -> list[tuple[int, int, str]]:
    """`pii_detection_model::decode::reconcile`, transcribed: sort by (margin
    descending, window index ascending), then greedily keep a span only if
    it overlaps nothing already kept. `spans` is (start, end, type, margin,
    window_index); the return drops margin/window, same shape
    `_merge_touching_typed` and `reconstruct_spans` already use.
    """
    ordered = sorted(spans, key=lambda s: (-s[3], s[4]))
    kept: list[tuple[int, int, str, int, int]] = []
    for candidate in ordered:
        overlaps = any(
            candidate[0] < existing[1] and existing[0] < candidate[1] for existing in kept
        )
        if not overlaps:
            kept.append(candidate)
    kept.sort(key=lambda s: (s[0], s[1]))
    return [(s, e, t) for s, e, t, _margin, _window in kept]


def predict_spans_with_margin(
    text: str,
    tokenizer,
    predict,
    max_length: int,
    stride: int,
) -> list[tuple[int, int, str, int, int]]:
    """Like `eval_fixture.predict_spans`, but stops one step short of
    reconciling -- it returns every window's raw decoded spans tagged with
    that window's margin and index, so both reconcile rules can be applied
    to the exact same per-window decode."""
    windows = tokenize_and_align(text, [], tokenizer, max_length, stride)
    raw: list[tuple[int, int, str, int, int]] = []
    for window_index, w in enumerate(windows):
        pred_ids = predict(w)
        decode_ids = [
            IGNORE_INDEX if gold == IGNORE_INDEX else pred for pred, gold in zip(pred_ids, w.labels)
        ]
        coverage = window_real_range(w)
        for start, end, entity_type in reconstruct_spans(w.offset_mapping, decode_ids):
            if coverage is None:
                margin = 0
            else:
                cov_start, cov_end = coverage
                margin = min(start - cov_start, cov_end - end)
            raw.append((start, end, entity_type, margin, window_index))
    return raw


def trim_whitespace(text: str, spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """`pii_detection_model::decode::trim_whitespace`, transcribed: walk in from
    each end while the code point there is whitespace; drop the span if
    nothing is left."""
    trimmed = []
    for start, end, entity_type in spans:
        s, e = start, end
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if s < e:
            trimmed.append((s, e, entity_type))
    return trimmed


def per_type_recall(
    cases: list[dict], spans_by_case: dict[str, list[tuple[int, int, str]]]
) -> dict[str, tuple[int, int]]:
    """(found, total) must_mask occurrences per platform type in
    `MEASURED_TYPES`, found under found_all's union-coverage rule (type is
    not itself checked -- a value found by *any* span counts, matching how
    the rest of this work already scores must_mask; only which *bucket* an
    occurrence is counted into depends on the fixture's own labelled
    type)."""
    counts = {t: [0, 0] for t in MEASURED_TYPES}
    for case in cases:
        text = case["text"]
        spans = [(s, e) for s, e, _t in spans_by_case[case["id"]]]
        for entry in case.get("must_mask", []):
            if not isinstance(entry, dict):
                continue
            entity_type = entry.get("type")
            if entity_type not in MEASURED_TYPES:
                continue
            occurrences = find_occurrences(text, entry["value"])
            if not occurrences:
                continue
            found = all(coverage_found(text, o, spans) for o in occurrences)
            counts[entity_type][1] += 1
            if found:
                counts[entity_type][0] += 1
    return {t: (f, n) for t, (f, n) in counts.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", default="../models/staging/model.onnx")
    ap.add_argument("--fixture", default="../benchmark/cases")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--stride", type=int, default=32)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    predict = onnx_predictor(args.onnx)
    cases = load_cases_with_type(Path(args.fixture))

    raw_by_case: dict[str, list[tuple[int, int, str, int, int]]] = {}
    for i, case in enumerate(cases):
        raw_by_case[case["id"]] = predict_spans_with_margin(
            case["text"], tokenizer, predict, args.max_length, args.stride
        )
        if (i + 1) % 100 == 0:
            print(f"  ...{i + 1}/{len(cases)}", file=sys.stderr)

    merged_by_case = {
        case_id: _merge_touching_typed([(s, e, t) for s, e, t, _m, _w in raw])
        for case_id, raw in raw_by_case.items()
    }
    furthest_by_case = {case_id: furthest_from_edge(raw) for case_id, raw in raw_by_case.items()}

    merged_recall = per_type_recall(cases, merged_by_case)
    furthest_recall = per_type_recall(cases, furthest_by_case)

    print(f"n_cases = {len(cases)}")
    print()
    print("=== _merge_touching_typed (eval_fixture.py's rule) ===")
    for t in MEASURED_TYPES:
        found, total = merged_recall[t]
        print(f"  {t}: {found}/{total} = {found / total:.4f}" if total else f"  {t}: 0/0")
    print()
    print("=== furthest_from_edge (the design's rule) ===")
    for t in MEASURED_TYPES:
        found, total = furthest_recall[t]
        print(f"  {t}: {found}/{total} = {found / total:.4f}" if total else f"  {t}: 0/0")

    print()
    print("=== whitespace trimming, applied on top of furthest_from_edge ===")
    total_spans = 0
    changed = 0
    dropped = 0
    trimmed_by_case: dict[str, list[tuple[int, int, str]]] = {}
    for case in cases:
        text = case["text"]
        spans = furthest_by_case[case["id"]]
        trimmed = trim_whitespace(text, spans)
        trimmed_by_case[case["id"]] = trimmed
        total_spans += len(spans)
        dropped += len(spans) - len(trimmed)
        # A span "changes" if it has leading or trailing whitespace, i.e.
        # trimming would move (or, if entirely whitespace, drop) it.
        for s, e, _t in spans:
            slice_ = text[s:e]
            if slice_ != slice_.strip():
                changed += 1
    print(f"  total spans (furthest_from_edge): {total_spans}")
    print(f"  spans whitespace-trimming changed (boundary moved or dropped): {changed}")
    print(f"  of those, spans dropped entirely (nothing left after trim): {dropped}")

    trimmed_recall = per_type_recall(cases, trimmed_by_case)
    print()
    print("=== recall after whitespace trimming (furthest_from_edge + trim) ===")
    for t in MEASURED_TYPES:
        found, total = trimmed_recall[t]
        print(f"  {t}: {found}/{total} = {found / total:.4f}" if total else f"  {t}: 0/0")


if __name__ == "__main__":
    main()
