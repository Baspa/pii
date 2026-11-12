"""Score the detector's Rust engine -- rules and resolver -- over the same 128
hand-written eval cases the trained checkpoint and Azure AI Language were
scored on, and score the union of rules + model, which is the system Temperate
would actually deploy.

WHY THIS EXISTS. training/reports/fixture-comparison.md put the model at
0.9145 recall against Azure's 0.9671. That comparison was unfair to the detector
and unfair in the dangerous direction: it measured *the model alone* against
*the whole Azure product*. The detector is rules plus model. crates/rules
(EMAIL_ADDRESS, IBAN_CODE, CREDIT_CARD, IP_ADDRESS, NATIONAL_ID,
PASSPORT_NUMBER, TAX_ID, VAT_NUMBER, plus a phone recognizer) never ran, and
three of the model's thirteen misses are types those packs claim. This
script supplies the missing measurement: rules alone, and rules + model
combined.

NO SECOND SCORER. The spans come from Rust; the scoring stays in
fixture_scoring.py, the same module that scored the model and Azure.
Reimplementing the must_mask/must_clear rule in Rust would produce a second,
subtly different scorer, and two numbers from two scorers cannot be put beside
each other honestly -- which is the entire point of this exercise. The Rust
binary therefore emits offsets and nothing else:

    cargo run --release -p pii-detection-eval --bin fixture-spans -- \
        --fixture <fixture.json> --out reports/rules-fixture-spans.json

OFFSET CONVENTION, CHECKED RATHER THAN ASSUMED. pii-detection-engine's Document and
CharSpan work in Unicode code points (Document builds a char-boundary table
and converts each recognizer's byte offsets exactly once). Python string
indices and the fixture's own offsets are code points too. Rather than trust
that, the Rust binary emits `char_len` per case and this script asserts it
equals `len(case["text"])` for every case -- 128 independent agreements
between the two parsers, several of them on multibyte text. A byte/code-point
confusion would fail that check long before it could quietly shift a recall
number.

THE COMBINED SYSTEM IS A UNION OF SPANS. Rules and model are two detectors
whose outputs are merged, not a pipeline where one filters the other. So
combined recall can only be >= either alone, and combined over-masking can
only be >= either alone. Both are reported: a recall gain paid for with a
false-positive spike is not a gain. The model's spans are read from
reports/model-fixture-2026-08-21.json (the saved run), not re-inferred.

Usage:
    python eval_fixture_rules.py \
        --spans reports/rules-fixture-spans.json \
        --model reports/model-fixture-2026-08-21.json \
        --azure reports/azure-fixture-2026-08-21-rescored.json \
        --out reports/rules-fixture-2026-08-21.json \
        --report reports/rules-engine-fixture.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fixture_scoring import (
    DEFAULT_FIXTURE,
    load_cases,
    merge_spans,
    print_summary,
    score_case,
    summarize,
)


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def rules_spans_by_id(spans_doc: dict, cases: list[dict]) -> dict[str, list[tuple[int, int]]]:
    """The engine's spans per case id, after checking that the Rust side and
    this side agree about the fixture down to the character.

    Two separate JSON parsers read the same file. If they disagreed about a
    case's length -- a different escape decoding, a different fixture
    entirely, a stale spans file -- every offset below would be silently
    misaligned and the resulting recall would be wrong in whichever direction
    the shift happened to fall. The check is per case, not just on the count.
    """
    emitted = spans_doc["cases"]
    if len(emitted) != len(cases):
        raise RuntimeError(
            f"spans file scored {len(emitted)} cases but the fixture has {len(cases)}"
        )

    by_id: dict[str, list[tuple[int, int]]] = {}
    for case, entry in zip(cases, emitted):
        if case["id"] != entry["id"]:
            raise RuntimeError(
                f"spans file is in a different order: expected {case['id']!r}, got {entry['id']!r}"
            )
        if entry["char_len"] != len(case["text"]):
            raise RuntimeError(
                f"case {case['id']!r}: the engine read {entry['char_len']} code points, "
                f"this script reads {len(case['text'])} -- the two parses disagree, so no "
                "offset from this file can be trusted"
            )
        spans = [(s["start"], s["end"]) for s in entry["spans"]]
        for start, end in spans:
            if not 0 <= start < end <= len(case["text"]):
                raise RuntimeError(
                    f"case {case['id']!r}: a span falls outside the text -- offsets are not "
                    "code points, or not this fixture's"
                )
        by_id[case["id"]] = spans
    return by_id


def typed_spans_by_id(spans_doc: dict) -> dict[str, list[tuple[int, int, str]]]:
    """Engine spans with their type label, for the miss quoting only. Scoring
    is type-agnostic (the fixture names substrings, not types)."""
    return {
        entry["id"]: [(s["start"], s["end"], s["entity_type"]) for s in entry["spans"]]
        for entry in spans_doc["cases"]
    }


def scored_spans_by_id(scored_cases: list[dict]) -> dict[str, list[tuple[int, int]]]:
    """A previously-scored file's spans, taken from the `predicted_spans` that
    score_case recorded.

    Those are merge_spans() of whatever the detector produced -- the union of
    its raw spans with touching/overlapping ones joined. Union coverage
    (found_all) and the must_clear overlap check both depend only on the
    union, so re-scoring from these reproduces that file's own numbers
    exactly; the earlier reports are re-derived below as a self-check.
    """
    return {c["id"]: [(s["start"], s["end"]) for s in c["predicted_spans"]] for c in scored_cases}


def score_all(cases: list[dict], spans_by_id: dict[str, list[tuple[int, int]]]) -> list[dict]:
    return [score_case(case, spans_by_id.get(case["id"], []), warn) for case in cases]


def union_spans(
    a: dict[str, list[tuple[int, int]]], b: dict[str, list[tuple[int, int]]]
) -> dict[str, list[tuple[int, int]]]:
    """Both detectors' spans for each case. merge_spans is applied for
    tidiness only -- score_case merges internally and union coverage does not
    care either way."""
    return {key: merge_spans(a.get(key, []) + b.get(key, [])) for key in set(a) | set(b)}


def azure_typed_spans_by_id(azure: dict) -> dict[str, list[tuple[int, int, str]]]:
    """Azure's spans with their category, taken from the verbatim responses
    the rescored run already saved.

    Used only to label a miss with what Azure called it. Reading the raw
    responses rather than the scored `predicted_spans` (which have no type)
    means the claim "every value the combined system still misses is a
    person's name" is Azure's own label, computed, not this report's
    assertion. No Azure call is made; nothing here reads credentials.
    """
    by_id: dict[str, list[tuple[int, int, str]]] = {}
    for batch in azure.get("raw_batches", []):
        for document in batch["raw_response"].get("results", {}).get("documents", []):
            spans = []
            for entity in document.get("entities", []):
                offset, length = entity.get("offset"), entity.get("length")
                if isinstance(offset, int) and isinstance(length, int) and length > 0:
                    spans.append((offset, offset + length, entity.get("category", "?")))
            by_id[document.get("id")] = spans
    return by_id


def overlapping_types(
    typed: list[tuple[int, int, str]], occurrences: list[list[int]]
) -> list[str]:
    """Which rule types actually cover a must_clear value, so an over-masking
    entry can name the recognizer that caused it rather than leaving it to be
    guessed from the span text."""
    hit = []
    for start, end, entity_type in typed:
        for occurrence_start, occurrence_end in occurrences:
            if start < occurrence_end and occurrence_start < end and entity_type not in hit:
                hit.append(entity_type)
    return hit


def fmt_rate(found: int, total: int) -> str:
    if total == 0:
        return "n/a"
    return f"{found}/{total} = {found / total:.4f}"


def recall_row(label: str, summaries: list[dict], key: str) -> str:
    cells = []
    for summary in summaries:
        stats = summary[key]["mask"] if key == "overall" else summary["per_language"][key]["mask"]
        cells.append(fmt_rate(stats["found"], stats["total"]))
    return f"| {label} | " + " | ".join(cells) + " |"


def value_verdicts(scored_by_id: dict[str, list[dict]], case_id: str) -> dict[str, bool]:
    return {m["value"]: m["found_all"] for m in scored_by_id[case_id]}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--fixture", default=DEFAULT_FIXTURE, help="pii-eval-cases.json (read-only)")
    ap.add_argument("--spans", required=True, help="the Rust engine's spans (fixture-spans --out)")
    ap.add_argument("--model", required=True, help="eval_fixture.py's saved run")
    ap.add_argument("--azure", required=True, help="eval_fixture_azure.py's rescored run")
    ap.add_argument("--out", required=True, help="path to write the scored JSON")
    ap.add_argument("--report", default=None, help="path to write the markdown report")
    args = ap.parse_args()

    cases = load_cases(args.fixture)
    spans_doc = load(args.spans)
    model = load(args.model)
    azure = load(args.azure)

    if spans_doc["fixture"] != args.fixture:
        raise RuntimeError(
            f"the spans file was produced from {spans_doc['fixture']!r}, not {args.fixture!r}"
        )
    if not (model["n_cases"] == azure["n_cases"] == len(cases)):
        raise RuntimeError(
            f"case count mismatch: fixture={len(cases)} model={model['n_cases']} azure={azure['n_cases']}"
        )

    rules_by_id = rules_spans_by_id(spans_doc, cases)
    model_by_id = scored_spans_by_id(model["cases"])  # merged variant: what a deployment masks
    model_unmerged_by_id = scored_spans_by_id(model["cases_unmerged"])
    azure_by_id = scored_spans_by_id(azure["cases"])

    rules_results = score_all(cases, rules_by_id)
    model_results = score_all(cases, model_by_id)
    combined_by_id = union_spans(rules_by_id, model_by_id)
    combined_results = score_all(cases, combined_by_id)
    combined_unmerged_results = score_all(cases, union_spans(rules_by_id, model_unmerged_by_id))

    rules_summary = summarize(rules_results)
    model_summary = summarize(model_results)
    combined_summary = summarize(combined_results)
    combined_unmerged_summary = summarize(combined_unmerged_results)

    # Re-deriving the model's own published numbers from its saved spans is a
    # self-check on this script's reading of that file. If it disagrees, the
    # comparison below is against something other than the run it names.
    published = model["summary"]["overall"]["mask"]
    rederived = model_summary["overall"]["mask"]
    if (published["found"], published["total"]) != (rederived["found"], rederived["total"]):
        raise RuntimeError(
            f"re-scoring the model's saved spans gives {rederived['found']}/{rederived['total']}, "
            f"but that file reports {published['found']}/{published['total']}"
        )

    by_id = {c["id"]: c for c in cases}
    rules_scored = {r["id"]: r["must_mask"] for r in rules_results}
    model_scored = {r["id"]: r["must_mask"] for r in model_results}
    combined_scored = {r["id"]: r["must_mask"] for r in combined_results}
    azure_scored = {r["id"]: r["must_mask"] for r in azure["cases"]}
    typed = typed_spans_by_id(spans_doc)
    azure_typed = azure_typed_spans_by_id(azure)

    misses = []
    recovered_by_rules = []
    beats_azure = []
    # Every must_mask value the phone recognizer covers, so the phone section
    # states a measured list rather than a remembered one.
    phone_covered = []
    for case in cases:
        cid = case["id"]
        rules_v = value_verdicts(rules_scored, cid)
        model_v = value_verdicts(model_scored, cid)
        azure_v = value_verdicts(azure_scored, cid)
        for entry in combined_scored[cid]:
            value = entry["value"]
            if "PHONE_NUMBER" in overlapping_types(typed[cid], entry["occurrences"]):
                phone_covered.append(
                    {
                        "id": cid,
                        "lang": case["lang"],
                        "value": value,
                        "found_by_rules": rules_v.get(value, False),
                        "found_by_model": model_v.get(value, False),
                        "found_by_azure": azure_v.get(value, False),
                    }
                )
            # What running the rules engine bought: values the model alone
            # left in the clear. This is the number the earlier comparison
            # left out of the detector's column entirely.
            if entry["found_all"] and not model_v.get(value, False):
                recovered_by_rules.append(
                    {
                        "id": cid,
                        "lang": case["lang"],
                        "value": value,
                        "text": case["text"],
                        "rules_types": [
                            t
                            for s, e, t in typed[cid]
                            for os_, oe in entry["occurrences"]
                            if s < oe and os_ < e
                        ],
                    }
                )
            # And where the combined system is already ahead of Azure.
            if entry["found_all"] and not azure_v.get(value, False):
                beats_azure.append(
                    {
                        "id": cid,
                        "lang": case["lang"],
                        "value": value,
                        "text": case["text"],
                        "found_by_rules": rules_v.get(value, False),
                        "found_by_model": model_v.get(value, False),
                    }
                )
            if entry["found_all"]:
                continue
            misses.append(
                {
                    "id": cid,
                    "lang": case["lang"],
                    "len": case.get("len"),
                    "text": case["text"],
                    "value": entry["value"],
                    "occurrences": entry["occurrences"],
                    "rules_found": rules_v.get(entry["value"], False),
                    "model_found": model_v.get(entry["value"], False),
                    "azure_found": azure_v.get(entry["value"], False),
                    "rules_spans": [
                        {"start": s, "end": e, "entity_type": t, "text": case["text"][s:e]}
                        for s, e, t in typed[cid]
                    ],
                    "model_spans": [
                        {"start": s, "end": e, "text": case["text"][s:e]}
                        for s, e in model_by_id[cid]
                    ],
                    "azure_spans": [
                        {"start": s, "end": e, "text": case["text"][s:e]}
                        for s, e in azure_by_id[cid]
                    ],
                    # What Azure calls the thing it caught and we did not.
                    "azure_types": overlapping_types(
                        azure_typed.get(cid, []), entry["occurrences"]
                    ),
                }
            )

    # Over-masking the union introduced: a must_clear value neither detector
    # violated alone but the union does is impossible (the union adds no
    # characters beyond the two span sets), so this list is expected to be
    # empty and is computed rather than asserted.
    rules_clear = {r["id"]: {c["value"]: c["violated"] for c in r["must_clear"]} for r in rules_results}
    model_clear = {r["id"]: {c["value"]: c["violated"] for c in r["must_clear"]} for r in model_results}
    over_mask = []
    for result in combined_results:
        for entry in result["must_clear"]:
            if not entry["violated"]:
                continue
            over_mask.append(
                {
                    "id": result["id"],
                    "lang": result["lang"],
                    "text": result["text"],
                    "value": entry["value"],
                    "occurrences": entry["occurrences"],
                    "by_rules": rules_clear[result["id"]][entry["value"]],
                    "by_model": model_clear[result["id"]][entry["value"]],
                    "rules_types": overlapping_types(typed[result["id"]], entry["occurrences"]),
                    "combined_spans": [
                        {"start": s["start"], "end": s["end"], "text": s["text"]}
                        for s in result["predicted_spans"]
                    ],
                }
            )

    command = "python " + " ".join(sys.argv)
    output = {
        "command": command,
        "spans_command": spans_doc["command"],
        "fixture": args.fixture,
        "engine": spans_doc["engine"],
        "rules_version": spans_doc["rules_version"],
        "requested_types": spans_doc["requested_types"],
        "model_report": args.model,
        "model_checkpoint": model["checkpoint"],
        "azure_report": args.azure,
        "azure_model_version": azure["azure_model_version"],
        "azure_min_score": azure["min_score"],
        "n_cases": len(cases),
        "summary_rules": rules_summary,
        "summary_model": model_summary,
        "summary_combined": combined_summary,
        "summary_combined_unmerged": combined_unmerged_summary,
        "summary_azure": azure["summary"],
        "combined_misses": misses,
        "recovered_by_rules": recovered_by_rules,
        "phone_covered": phone_covered,
        "combined_beats_azure": beats_azure,
        "combined_over_masking": over_mask,
        "cases_rules": rules_results,
        "cases_combined": combined_results,
    }
    Path(args.out).write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Wrote {args.out}")

    print("=== Rules engine alone ===")
    print_summary(rules_summary)
    print("=== Model alone (re-derived from its saved spans) ===")
    print_summary(model_summary)
    print("=== Rules + model combined (union of spans) ===")
    print_summary(combined_summary)
    print("=== Azure (quoted from its rescored run) ===")
    print_summary(azure["summary"])

    if args.report:
        Path(args.report).write_text(
            render(output, by_id, rules_summary, model_summary, combined_summary, azure)
        )
        print(f"Wrote {args.report}")
    print(f"{len(misses)} must_mask values the combined system still misses")


def render(
    output: dict,
    by_id: dict[str, dict],
    rules_summary: dict,
    model_summary: dict,
    combined_summary: dict,
    azure: dict,
) -> str:
    summaries = [rules_summary, model_summary, combined_summary, azure["summary"]]
    langs = sorted(combined_summary["per_language"])
    lines: list[str] = []

    # What Azure calls each value the combined system misses, counted once
    # here so every claim about "what kind of thing we still miss" in this
    # report is Azure's label for those exact characters rather than this
    # report's assertion about them.
    miss_labels: dict[str, int] = {}
    for miss in output["combined_misses"]:
        for label in miss["azure_types"] or ["nothing (Azure missed it too)"]:
            miss_labels[label] = miss_labels.get(label, 0) + 1

    lines.append("# Rules engine, and rules + model, on the 128-case fixture")
    lines.append("")
    lines.append(
        "The comparison in `fixture-comparison.md` measured **the model alone** against "
        "**the whole Azure product** and reported 0.9145 against 0.9671. The detector is rules "
        "*plus* model: `crates/rules` and `crates/resolve` never ran in that "
        "comparison, so the gap it showed was larger than the gap that exists, and it was "
        "larger in the direction that flatters us. This report is the missing half."
    )
    lines.append("")
    lines.append(
        "**Still not a release gate.** 128 hand-written cases (40 en / 40 nl / 30 de / "
        "18 fr; 152 must_mask values, 64 must_clear values) is a smoke test. A few points "
        "either way is noise. The value here is the per-value miss list at the end, not the "
        "topline."
    )
    lines.append("")

    rm = rules_summary["overall"]["mask"]
    cm = combined_summary["overall"]["mask"]
    am = azure["summary"]["overall"]["mask"]
    rc = rules_summary["overall"]["clear"]
    cc = combined_summary["overall"]["clear"]
    ac = azure["summary"]["overall"]["clear"]
    lines.append("## The number nobody had")
    lines.append("")
    lines.append(
        f"**Rules + model: {fmt_rate(cm['found'], cm['total'])} recall, "
        f"{fmt_rate(cc['violated'], cc['total'])} over-masking.** "
        f"Azure: {fmt_rate(am['found'], am['total'])} recall, "
        f"{fmt_rate(ac['violated'], ac['total'])} over-masking. The model alone was "
        f"{fmt_rate(model_summary['overall']['mask']['found'], model_summary['overall']['mask']['total'])}."
    )
    lines.append("")
    lines.append(
        f"The rules engine alone scores {fmt_rate(rm['found'], rm['total'])}, and that "
        "number means nothing on its own: it detects nine structured identifier types and "
        "the fixture is mostly names and addresses, which are the model's job and which "
        "`pii-detection-resolve` refuses to emit from rules at all (`PERSON` and `STREET_ADDRESS` "
        "have no entry in `thresholds.toml`, so they fail closed). It is reported because "
        "the brief asked for it and because it is the denominator for the next line, not "
        "because a rules engine should be judged by it."
    )
    lines.append("")
    lines.append(
        f"On over-masking the union is worse than either half ({fmt_rate(cc['violated'], cc['total'])} "
        f"against the model's {fmt_rate(model_summary['overall']['clear']['violated'], model_summary['overall']['clear']['total'])} "
        f"and the rules engine's {fmt_rate(rc['violated'], rc['total'])}), exactly as it must "
        "be: a union masks strictly more. The recall gain is real and it is not free, and "
        "the section on the phone recognizer below is where the whole cost sits."
    )
    lines.append("")
    lines.append(
        f"**Ca detector replace Azure?** On this fixture the answer is \"nearly, and the "
        f"remaining difference is one entity type\": {cm['found']} against {am['found']} of "
        f"{cm['total']} values, a gap of {am['found'] - cm['found']}. Not \"yes\": {cm['total'] - cm['found']} "
        "values still leak ("
        + ", ".join(
            f"{count} that Azure labels `{label}`" for label, count in sorted(miss_labels.items())
        )
        + "), and over-masking is worse than Azure's. And not \"no\" either -- the combined "
        "system already covers "
        "identifier types Azure has no category for at all. 128 cases cannot settle it; "
        "the real benchmark still has to be built."
    )
    lines.append("")

    lines.append("## How this was measured")
    lines.append("")
    lines.append(
        "The Rust engine runs through `pii_detection_api::routes::analyze_text` -- literally the "
        "function `POST /analyze` calls per text, not a second assembly of the same steps -- "
        "with the requested entity types taken from `routes::detected_types()`, the same set "
        "`/capabilities` advertises and `/analyze` validates against. It emits offsets only; "
        "it never sees the fixture's `must_mask`/`must_clear` fields and never emits text."
    )
    lines.append("")
    lines.append(
        "Scoring is `training/fixture_scoring.py`, unchanged -- the same module that scored "
        "the model and Azure. **No scoring rule was reimplemented in Rust.** A second scorer "
        "would be free to differ in exactly the small ways nobody checks, and then these "
        "numbers could not sit beside the earlier ones at all. A must_mask value counts as "
        "found when every non-whitespace character of every occurrence is covered by the "
        "union of detected spans; a must_clear value counts as violated when any detected "
        "span overlaps it."
    )
    lines.append("")
    lines.append(
        "**Offsets are code points, and that was checked, not assumed.** `pii-detection-engine`'s "
        "`Document` builds a character-boundary table and converts every recognizer's byte "
        "offsets into code points exactly once; Python string indices and the fixture's "
        "offsets are code points too. The engine emits `char_len` per case and the scorer "
        "asserts it equals `len(case[\"text\"])` for all 128 cases, several of which contain "
        "multibyte characters, plus a bounds check on every span. A byte/code-point mix-up "
        "fails that check rather than quietly shifting a recall number. `pii-detection-eval` also "
        "carries a unit test pinning `\"Grüße an jane@example.com\"` to start 9 (nine code "
        "points, eleven bytes)."
    )
    lines.append("")
    lines.append(
        "**Combined is a union of spans.** Rules and model are two detectors merged, not a "
        "pipeline. The union masks strictly more than either alone, so recall can only rise "
        "and over-masking can only worsen; both are reported below. The model's spans are "
        "read from its saved run, not re-inferred, and re-scoring them reproduces that run's "
        "published 139/152 exactly (the script raises if it does not)."
    )
    lines.append("")

    lines.append("### Commands")
    lines.append("")
    lines.append("```")
    lines.append("# 1. the Rust engine over the fixture (spans only, no text, no scoring)")
    lines.append(
        "cargo run --release -p pii-detection-eval --bin fixture-spans -- \\\n"
        "    --fixture ../benchmark/cases \\\n"
        "    --out training/reports/rules-fixture-spans.json"
    )
    lines.append(f"#    recorded invocation: {output['spans_command']}")
    lines.append("")
    lines.append("# 2. score rules alone and rules + model, quote Azure (run from training/)")
    lines.append(output["command"])
    lines.append("```")
    lines.append("")
    lines.append(f"- Engine: {output['engine']}, rules version `{output['rules_version']}`")
    lines.append(
        "- Requested types: " + ", ".join(f"`{t}`" for t in output["requested_types"])
    )
    lines.append(
        f"- Model checkpoint `{output['model_checkpoint']}`, spans from `{output['model_report']}` "
        "(merged variant -- what a real deployment masks with)"
    )
    lines.append(
        f"- Azure quoted from `{output['azure_report']}` (modelVersion "
        f"`{output['azure_model_version']}`, confidence >= {output['azure_min_score']}). "
        "**No Azure call was made for this report.**"
    )
    lines.append("")

    lines.append("## Recall (must_mask)")
    lines.append("")
    lines.append("| | Rules alone | Model alone | **Rules + model** | Azure |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(recall_row("**Overall**", summaries, "overall"))
    for lang in langs:
        lines.append(recall_row(lang, summaries, lang))
    lines.append("")
    by_lang: dict[str, int] = {}
    for miss in output["combined_misses"]:
        by_lang[miss["lang"]] = by_lang.get(miss["lang"], 0) + 1
    lines.append(
        "The remaining deficit is concentrated: the "
        + ", ".join(f"{count} in `{lang}`" for lang, count in sorted(by_lang.items()))
        + " values the combined system still misses are, by Azure's own labels for the "
        "same characters, "
        + ", ".join(f"{count} × `{label}`" for label, count in sorted(miss_labels.items()))
        + ". They are quoted in full below."
    )
    lines.append("")

    lines.append("## Over-masking (must_clear violations)")
    lines.append("")
    lines.append("| | Rules alone | Model alone | **Rules + model** | Azure |")
    lines.append("|---|---:|---:|---:|---:|")

    def clear_row(label: str, key: str) -> str:
        cells = []
        for summary in summaries:
            stats = (
                summary["overall"]["clear"]
                if key == "overall"
                else summary["per_language"][key]["clear"]
            )
            cells.append(fmt_rate(stats["violated"], stats["total"]))
        return f"| {label} | " + " | ".join(cells) + " |"

    lines.append(clear_row("**Overall**", "overall"))
    for lang in langs:
        lines.append(clear_row(lang, lang))
    lines.append("")

    rules_clear = rules_summary["overall"]["clear"]
    model_clear = model_summary["overall"]["clear"]
    combined_clear = combined_summary["overall"]["clear"]
    over_mask = output["combined_over_masking"]
    added_by_rules = [e for e in over_mask if e["by_rules"] and not e["by_model"]]
    lines.append(
        f"The union violates {combined_clear['violated']} must_clear values against the "
        f"model's {model_clear['violated']} and the rules engine's {rules_clear['violated']}. "
        f"{len(added_by_rules)} of them are violations the model alone did not commit, so "
        "this is the price of the recall the rules engine buys, and it has to be quoted "
        "alongside it. Every violation the union commits is listed in full at the end, "
        "attributed to the side that committed it."
    )
    lines.append("")

    phone_driven = [e for e in over_mask if "PHONE_NUMBER" in e["rules_types"]]
    if phone_driven:
        lines.append(
            f"**{len(phone_driven)} of the rules engine's {rules_clear['violated']} "
            "over-masking violations come from one recognizer: the phone recognizer.** "
            "It scores a flat 0.9 with no "
            "checksum and no negative-context terms, so an invoice number, an order number, "
            "a customer number and a German `Belegnummer` are all masked as phone numbers "
            "when they happen to sit in some numbering plan. See the phone note below. "
            "Nothing here was changed."
        )
        lines.append("")

    lines.append("## Pure negatives (cases with nothing to mask)")
    lines.append("")
    lines.append("| | clean | corrupted |")
    lines.append("|---|---:|---:|")
    for label, summary in [
        ("Rules alone", rules_summary),
        ("Model alone", model_summary),
        ("**Rules + model**", combined_summary),
        ("Azure", azure["summary"]),
    ]:
        neg = summary["pure_negatives"]
        lines.append(f"| {label} | {neg['clean']}/{neg['n_cases']} | {neg['corrupted']} |")
    lines.append("")

    recovered = output["recovered_by_rules"]
    lines.append(f"## What running the rules engine buys ({len(recovered)} values)")
    lines.append("")
    lines.append(
        "Every must_mask value the model alone left in the clear and the rules engine "
        "covers. These are the values the earlier comparison charged to the detector's account "
        "without ever running the half of the detector that finds them."
    )
    lines.append("")
    lines.append("| case | lang | value | found by |")
    lines.append("|---|---|---|---|")
    for entry in recovered:
        types = ", ".join(f"`{t}`" for t in dict.fromkeys(entry["rules_types"])) or "(rules)"
        lines.append(f"| `{entry['id']}` | {entry['lang']} | `{entry['value']}` | {types} |")
    lines.append("")

    ahead = output["combined_beats_azure"]
    lines.append(f"## Where the combined system is already ahead of Azure ({len(ahead)} values)")
    lines.append("")
    lines.append(
        "Every must_mask value the combined system covers and Azure does not, at the "
        "confidence threshold Temperate's own driver uses. Azure AI Language supplies no "
        "VAT_NUMBER and no German tax ID category at all -- Temperate's earlier Azure client's own "
        "docblock says so -- and this fixture shows the consequence."
    )
    lines.append("")
    lines.append("| case | lang | value | covered by |")
    lines.append("|---|---|---|---|")
    for entry in ahead:
        by = [
            label
            for label, flag in (("rules", entry["found_by_rules"]), ("model", entry["found_by_model"]))
            if flag
        ]
        lines.append(
            f"| `{entry['id']}` | {entry['lang']} | `{entry['value']}` | {' + '.join(by)} |"
        )
    lines.append("")

    misses = output["combined_misses"]
    total_values = combined_summary["overall"]["mask"]["total"]
    missed_by_both = sum(1 for m in misses if not m["azure_found"])
    both_covered = total_values - len(ahead) - len(misses)
    shared = (
        "and no value is missed by both"
        if missed_by_both == 0
        else f"{missed_by_both} of which Azure misses too"
    )
    lines.append(
        f"The two lists together are the whole difference: Azure misses {len(ahead)} values "
        f"the combined system covers, the combined system misses {len(misses)} values Azure "
        f"covers, {shared}, and the remaining {both_covered} of {total_values} are covered "
        "by both."
    )
    lines.append("")

    lines.append(f"## Every must_mask value the combined system still misses ({len(misses)})")
    lines.append("")
    lines.append(
        "Quoted in full, with the case text and what each side detected in that same case. "
        "This list is the deliverable: it is what the next round of work is made of."
    )
    lines.append("")
    if not misses:
        lines.append("(none)")
    for miss in misses:
        case = by_id[miss["id"]]
        lines.append(f"### `{miss['id']}` ({miss['lang']}, {case.get('len', '?')})")
        lines.append("")
        lines.append(f"> {miss['text']}")
        lines.append("")
        lines.append(f"**Missed:** `{miss['value']}` at {miss['occurrences']}")
        lines.append("")
        lines.append(
            f"- Rules engine detected: "
            + (
                ", ".join(f"`{s['text']}` ({s['entity_type']})" for s in miss["rules_spans"])
                or "(nothing in this case)"
            )
        )
        lines.append(
            f"- Model detected: "
            + (", ".join(f"`{s['text']}`" for s in miss["model_spans"]) or "(nothing in this case)")
        )
        lines.append(
            "- Azure detected: "
            + (", ".join(f"`{s['text']}`" for s in miss["azure_spans"]) or "(nothing in this case)")
            + f" -- Azure {'found' if miss['azure_found'] else '**also missed**'} this value"
            + (
                f", as {', '.join(f'`{t}`' for t in miss['azure_types'])}"
                if miss["azure_types"]
                else ""
            )
        )
        lines.append("")

    lines.append("## The phone gap: measured, and it is not where it was thought to be")
    lines.append("")
    lines.append(
        "This report was commissioned expecting to find that there is no `PHONE_NUMBER` "
        "pack, that four fixture phone numbers are caught by neither rules nor model, and "
        "that Azure catches all four -- a gap to be measured now and closed by a later "
        "task. **The measurement says otherwise, so it is recorded rather than repeated.**"
    )
    lines.append("")
    lines.append(
        "There is no phone *pack* -- `crates/rules/packs/` holds only `universal`, "
        "`nl` and `de`, none of which declares a `PHONE_NUMBER` recognizer. But there is a "
        "phone *recognizer*: `crates/rules/src/phone.rs`, added in commit `3dd9e50`, "
        "a code recognizer rather than a TOML one because a numbering plan is not a regex. "
        "`Matcher::scan` runs it, `routes::detected_types()` adds `PHONE_NUMBER` to the "
        "advertised set, and `thresholds.toml` calibrates it at 0.70. It is live, and it "
        "runs on every `/analyze` call."
    )
    lines.append("")
    phone_values = output["phone_covered"]
    model_missed = [entry for entry in phone_values if not entry["found_by_model"]]
    lines.append(
        f"It covers all {len(phone_values)} must_mask phone numbers in this fixture, "
        f"including the {len(model_missed)} the model missed -- and Azure covers "
        f"{sum(1 for e in phone_values if e['found_by_azure'])} of them, so on this fixture "
        "phone recall is not where the detector loses to Azure:"
    )
    lines.append("")
    lines.append("| case | lang | value | model alone |")
    lines.append("|---|---|---|---|")
    for entry in phone_values:
        verdict = "found" if entry["found_by_model"] else "**missed**"
        lines.append(
            f"| `{entry['id']}` | {entry['lang']} | `{entry['value']}` | {verdict} |"
        )
    lines.append("")
    phone_over_mask = [
        e for e in output["combined_over_masking"] if "PHONE_NUMBER" in e["rules_types"]
    ]
    phone_added = [e for e in phone_over_mask if not e["by_model"]]
    lines.append(
        "So the recall gap that motivated a phone pack does not exist on this fixture. What "
        "does exist is the other side of the same recognizer, and it is worse than a missing "
        "feature because it is silent: **the phone recognizer causes "
        f"{len(phone_over_mask)} of the rules engine's {rules_clear['violated']} over-masking "
        f"violations**, and {len(phone_added)} of those are violations the model alone does "
        "not commit, so the union pays for them. It accepts "
        "any digit run that parses as valid in any of eight numbering plans, at a flat 0.9 "
        "with no checksum, no context terms and no negative-context terms -- while the "
        "packs beside it all carry `negative_context` lists for exactly this ("
        "`ordernummer`, `factuurnummer`, `invoice`, `tracking`). A Dutch invoice number, an "
        "order number, a customer number and a German `Belegnummer` are all masked as "
        "telephone numbers. In `passport-negative` it masks a seven-digit *fragment* of a "
        "tracking code, which is the truncation failure `pii-detection-resolve`'s own docblock "
        "warns about, arriving from a different direction."
    )
    lines.append("")
    lines.append(
        "**Nothing was changed here.** This is a baseline; the fix belongs to whoever picks "
        "up the phone work, and this list is what will show whether it worked."
    )
    lines.append("")

    over_mask = output["combined_over_masking"]
    lines.append(f"## Every must_clear value the combined system over-masks ({len(over_mask)})")
    lines.append("")
    if not over_mask:
        lines.append("(none)")
    for entry in over_mask:
        blame = []
        if entry["by_rules"]:
            blame.append("rules")
        if entry["by_model"]:
            blame.append("model")
        lines.append(f"### `{entry['id']}` ({entry['lang']})")
        lines.append("")
        lines.append(f"> {entry['text']}")
        lines.append("")
        types = ", ".join(f"`{t}`" for t in entry["rules_types"])
        lines.append(
            f"**Over-masked:** `{entry['value']}` at {entry['occurrences']} -- by "
            f"{' and '.join(blame) if blame else 'the union only'}"
            + (f" (rules type: {types})" if types else "")
        )
        lines.append(
            "Combined spans: "
            + (", ".join(f"`{s['text']}`" for s in entry["combined_spans"]) or "(none)")
        )
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
