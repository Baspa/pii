"""Shared, detector-agnostic scoring for Temperate's hand-written PII eval
fixture (benchmark/cases).

Used by both eval_fixture.py (the trained checkpoint) and
eval_fixture_azure.py (Azure AI Language), so the two are scored by the
exact same rules -- the whole point of the comparison is that neither
detector gets a scoring advantage from a second, slightly different
implementation.

CORRECTED SCORING RULE (2026-08-21). An earlier version of this module
required a must_mask occurrence to be fully covered by a SINGLE detected
span. That rule was wrong: both training corpora label a given name and a
surname as separate entities (GIVENNAME/SURNAME, GIVEN_NAME/SURNAME) and
taxonomy.py maps both onto Temperate's one PERSON type without merging the
spans, so the model faithfully emits two touching PERSON spans -- "Jan" and
"de Vries" -- for a single name. Under the old rule that counted as a MISS
even though the two spans together mask the entire name and leak nothing.
See training/reports/fixture-comparison.md for the full account; that
report originally (and wrongly) put the model's recall at 0.6711 under the
single-span rule.

Scoring is type-agnostic, because the fixture names substrings, not types:
  - a must_mask value counts as found (`found_all`) only if EVERY occurrence
    of it in the text has every one of its non-whitespace characters
    covered by the UNION of detected spans. Whitespace between two adjacent
    detected spans is not a leak: no information survives in a space, so a
    gap there doesn't make the value recoverable.
  - a must_mask value additionally gets a `single_span_found` verdict: the
    OLD, strict rule (every occurrence fully covered by ONE detected span,
    after merging only touching/overlapping spans). This is kept as a
    secondary, clearly-labelled number because it measures something real
    -- whether the detector returns an entity as one span or several, which
    matters for pseudonymisation placeholder shape even though it does not
    matter for masking. It is NOT the headline recall number.
  - a must_clear value counts as violated if ANY detected span overlaps it
    at all.

Both detectors are reduced to the same shape before scoring: a list of
(start, end) character spans per case. Callers may pre-merge
whitespace-adjacent same-type spans (see merge_adjacent_whitespace below)
before calling score_case; scoring itself only relies on the union of
whatever spans it is given.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

DEFAULT_FIXTURE = str(Path(__file__).resolve().parent.parent / "benchmark" / "cases")

# Types merge_adjacent_whitespace() is allowed to join. Restricted to PERSON
# and STREET_ADDRESS on purpose -- see taxonomy.py's AI4PRIVACY_MAP and
# NYM_MAP. Those are the only two platform types built by collapsing multiple
# *positionally adjacent* corpus sub-labels of one templated field onto a
# single type:
#   PERSON          <- GIVENNAME/GIVEN_NAME + SURNAME (+ DOCTORNAME)
#   STREET_ADDRESS  <- STREET/STREET_NAME + BUILDINGNUM/BUILDING_NUMBER
#                      (+ ZIPCODE/ZIP_CODE, SECONDARY_ADDRESS)
# NATIONAL_ID and PHONE_NUMBER also draw from more than one corpus label
# (IDCARDNUM/SOCIALNUM, GOVERNMENT_ID/SSN, FAX_NUMBER/PHONE) but those are
# ALTERNATE full values for the same person, not adjacent halves of one
# templated field, so two of them sitting next to each other is exactly the
# case the task warns about: two distinct card/ID numbers separated by a
# space are more plausibly two people's data than one. Merging those would
# silently combine them.
MERGEABLE_TYPES: frozenset[str] = frozenset({"PERSON", "STREET_ADDRESS"})


def load_cases(path: str = DEFAULT_FIXTURE) -> list[dict]:
    """Cases from either fixture shape.

    A *file* is the original hand-written fixture: one JSON object with a
    "cases" key, `must_mask` entries plain strings. A *directory* is
    benchmark/cases/, the release fixture: several JSON files each holding
    a bare array, `must_mask` entries either a plain string or a
    {"value", "type"} object. Dispatching here rather than in each caller
    keeps one loader, so eval_fixture.py, eval_fixture_azure.py and
    eval_fixture_rules.py cannot drift into scoring different case sets.
    """
    p = Path(path)
    if p.is_dir():
        return load_case_dir(p)
    data = json.loads(p.read_text())
    return data["cases"]


def load_case_dir(directory: Path) -> list[dict]:
    """Every case in every `*.json` directly under `directory`.

    Deliberately the same contract as pii-detection-eval's `load` in
    crates/eval/src/lib.rs: every JSON file directly under the
    directory, read in sorted-path order, each a bare array of cases,
    concatenated. Two loaders over one fixture is already one too many;
    they must at least agree on which cases exist and in what order, or the
    Python and Rust scores are over different fixtures.

    A non-array file is an error, not a skip: silently dropping a file
    shrinks the fixture and raises every recall figure computed from what
    is left, with nothing in the output saying so.

    `must_mask` entries are flattened to their string value. Scoring here
    is type-agnostic on purpose (see the module docstring) -- it asks
    whether a substring was masked, never whether it was given the right
    label -- so the type is dropped rather than half-used.
    """
    directory = Path(directory)
    cases: list[dict] = []
    for file in sorted(directory.glob("*.json")):
        parsed = json.loads(file.read_text())
        if not isinstance(parsed, list):
            raise ValueError(f"{file} is not a JSON array of cases")
        for case in parsed:
            case = dict(case)
            case["must_mask"] = [
                entry["value"] if isinstance(entry, dict) else entry
                for entry in case.get("must_mask", [])
            ]
            case.setdefault("must_clear", [])
            cases.append(case)
    return cases


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Union overlapping or touching (start, end) character spans,
    discarding type. Used for the must_clear overlap check (type-agnostic
    by design) and for the secondary single-span figure."""
    if not spans:
        return []
    ordered = sorted(spans)
    merged = [list(ordered[0])]
    for s, e in ordered[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def merge_adjacent_whitespace(
    text: str,
    typed_spans: list[tuple[int, int, str]],
    mergeable_types: frozenset[str] = MERGEABLE_TYPES,
) -> list[tuple[int, int, str]]:
    """Merge same-type spans that are separated only by whitespace (or
    nothing at all) into one span, restricted to `mergeable_types`.

    This is the span-merging step: "Jan" (PERSON) and "de Vries" (PERSON)
    with a single space between them become one PERSON span "Jan de Vries".
    It fixes placeholder granularity for pseudonymisation, not masking
    recall -- a value already counts as found under found_all's
    union-coverage rule whether or not the covering spans are merged.

    LIMITATION, deliberately not special-cased: this can also join two
    genuinely distinct people (or addresses) that happen to appear
    adjacently with only whitespace between them (e.g. one name per line in
    a list). For masking that is harmless -- both are still masked either
    way, and restoring the original text is unaffected -- it only changes
    whether they get one placeholder or two. See
    training/reports/fixture-comparison.md for cases this does and doesn't
    affect.
    """
    if not typed_spans:
        return []
    ordered = sorted(typed_spans)
    merged = [list(ordered[0])]
    for s, e, t in ordered[1:]:
        ps, pe, pt = merged[-1]
        if t == pt and t in mergeable_types and s >= pe and text[pe:s].strip() == "":
            merged[-1][1] = max(pe, e)
        elif t == pt and s < pe:  # overlapping same-type spans always join
            merged[-1][1] = max(pe, e)
        else:
            merged.append([s, e, t])
    return [(s, e, t) for s, e, t in merged]


def find_occurrences(text: str, needle: str) -> list[tuple[int, int]]:
    """Every non-overlapping occurrence of the literal substring `needle`
    in `text`, as (start, end) character offsets."""
    if not needle:
        return []
    return [(m.start(), m.end()) for m in re.finditer(re.escape(needle), text)]


def coverage_found(text: str, occurrence: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    """True if every non-whitespace character in `occurrence` is covered by
    the union of `spans`. This is the corrected, primary must_mask rule --
    it does not require a single span to cover the whole occurrence, only
    that no non-whitespace character of it survives unmasked anywhere."""
    s, e = occurrence
    needed = {i for i in range(s, e) if not text[i].isspace()}
    if not needed:
        return True
    covered: set[int] = set()
    for ss, se in spans:
        lo, hi = max(ss, s), min(se, e)
        if lo < hi:
            covered.update(range(lo, hi))
        if needed <= covered:
            return True
    return needed <= covered


def fully_covered(occurrence: tuple[int, int], merged: list[tuple[int, int]]) -> bool:
    """The OLD, strict rule: true only if a SINGLE span in `merged` covers
    the whole occurrence. Kept as the secondary single-span figure."""
    s, e = occurrence
    return any(ms <= s and e <= me for ms, me in merged)


def overlaps_any(occurrence: tuple[int, int], merged: list[tuple[int, int]]) -> bool:
    s, e = occurrence
    return any(ms < e and me > s for ms, me in merged)


def score_case(case: dict, spans: list[tuple[int, int]], warn) -> dict:
    """`warn` is a callable(str) for the not-found-verbatim guard (each
    caller decides whether/where to surface it, without this module writing
    to stderr on behalf of both scripts).

    THE WARNING NAMES A POSITION, NEVER A VALUE. Every caller routes `warn`
    to stderr, and stderr goes to terminals, CI logs and job archives. The
    values this loop iterates over are the fixture's personal data --
    telephone numbers, national IDs, names -- so printing one to say it could
    not be found would copy it out of the fixture and into a log, which is
    the one thing a PII service must not do. The case id and the index within
    the case locate the offending entry exactly; whoever needs the value has
    the fixture open in front of them.

    `spans` is a list of (start, end) character spans for this case --
    already merged/whitespace-merged by the caller if desired, or raw. The
    union-coverage rule (found_all) gives the same answer either way;
    single_span_found and the must_clear overlap check use
    merge_spans(spans) (touching/overlapping only) internally."""
    text = case["text"]
    merged_spans = merge_spans(spans)

    mask_results = []
    for index, value in enumerate(case["must_mask"]):
        occurrences = find_occurrences(text, value)
        if not occurrences:
            warn(f"must_mask[{index}] not found verbatim in case {case['id']!r}")
        found_all = bool(occurrences) and all(coverage_found(text, o, spans) for o in occurrences)
        single_span_found = bool(occurrences) and all(fully_covered(o, merged_spans) for o in occurrences)
        mask_results.append(
            {
                "value": value,
                "occurrences": occurrences,
                "found_all": found_all,
                "single_span_found": single_span_found,
            }
        )

    clear_results = []
    for index, value in enumerate(case["must_clear"]):
        occurrences = find_occurrences(text, value)
        if not occurrences:
            warn(f"must_clear[{index}] not found verbatim in case {case['id']!r}")
        violated = any(overlaps_any(o, merged_spans) for o in occurrences)
        clear_results.append({"value": value, "occurrences": occurrences, "violated": violated})

    return {
        "id": case["id"],
        "lang": case["lang"],
        "len": case.get("len"),
        "text": text,
        "predicted_spans": [{"start": s, "end": e, "text": text[s:e]} for s, e in merged_spans],
        "must_mask": mask_results,
        "must_clear": clear_results,
        "is_pure_negative": not case["must_mask"],
    }


def summarize(results: list[dict]) -> dict:
    def mask_stats(rows: list[dict]) -> dict:
        total = sum(len(r["must_mask"]) for r in rows)
        found = sum(1 for r in rows for m in r["must_mask"] if m["found_all"])
        single_span = sum(1 for r in rows for m in r["must_mask"] if m["single_span_found"])
        return {
            "total": total,
            "found": found,
            "recall": (found / total) if total else None,
            "single_span_found": single_span,
            "single_span_recall": (single_span / total) if total else None,
        }

    def clear_stats(rows: list[dict]) -> dict:
        total = sum(len(r["must_clear"]) for r in rows)
        violated = sum(1 for r in rows for c in r["must_clear"] if c["violated"])
        return {"total": total, "violated": violated, "over_mask_rate": (violated / total) if total else None}

    overall = {"mask": mask_stats(results), "clear": clear_stats(results)}

    by_lang: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_lang[r["lang"]].append(r)
    per_language = {
        lang: {"mask": mask_stats(rows), "clear": clear_stats(rows), "n_cases": len(rows)}
        for lang, rows in sorted(by_lang.items())
    }

    negatives = [r for r in results if r["is_pure_negative"]]
    clean_negatives = sum(1 for r in negatives if not r["predicted_spans"])
    pure_negatives = {
        "n_cases": len(negatives),
        "clean": clean_negatives,
        "corrupted": len(negatives) - clean_negatives,
    }

    return {"overall": overall, "per_language": per_language, "pure_negatives": pure_negatives}


def print_summary(summary: dict) -> None:
    m = summary["overall"]["mask"]
    c = summary["overall"]["clear"]
    n = summary["pure_negatives"]
    print(f"Overall recall: {m['found']}/{m['total']} = {m['recall']:.4f}" if m["total"] else "Overall recall: n/a")
    if m["total"]:
        print(f"Overall single-span recall (secondary, strict): {m['single_span_found']}/{m['total']} = {m['single_span_recall']:.4f}")
    print(f"Overall over-masking: {c['violated']}/{c['total']} = {c['over_mask_rate']:.4f}" if c["total"] else "Overall over-masking: n/a")
    print(f"Pure negatives (nothing to mask): {n['clean']}/{n['n_cases']} clean, {n['corrupted']} corrupted")
    for lang, stats in summary["per_language"].items():
        lm, lc = stats["mask"], stats["clear"]
        r = f"{lm['recall']:.4f}" if lm["total"] else "n/a"
        sr = f"{lm['single_span_recall']:.4f}" if lm["total"] else "n/a"
        o = f"{lc['over_mask_rate']:.4f}" if lc["total"] else "n/a"
        print(
            f"  {lang}: recall={r} ({lm['found']}/{lm['total']}) single_span={sr} "
            f"over-mask={o} ({lc['violated']}/{lc['total']}) n_cases={stats['n_cases']}"
        )
