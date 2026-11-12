"""Build the model-vs-Azure fixture comparison report from the two scored
result files eval_fixture.py and eval_fixture_azure.py produce.

Reads nothing live -- both detectors have already run and been scored
identically by fixture_scoring.py. This script only diffs the two case-level
result sets and renders training/reports/fixture-comparison.md: the overall
and per-language numbers (corrected rule, plus the merging effect on the
model), a full quote of every remaining genuine miss on each side, every
case where the two detectors disagree, and every case where both fail.

CORRECTION (2026-08-21). An earlier version of this report used a scoring
rule that required a must_mask occurrence to be covered by a SINGLE
detected span, and reported the model's recall at 0.6711. That rule was
wrong -- see fixture_scoring.py's module docstring for the full account.
Rescoring on whether every non-whitespace character is covered by the union
of detected spans (whitespace between two adjacent spans leaks nothing)
gives 0.9145. This script now renders both the corrected primary recall and
the old rule's figure (relabelled "single-span", a real but secondary
signal about placeholder granularity), and states the correction in the
generated report rather than silently replacing the number.

Usage:
    python compare_fixture.py \
        --model reports/model-fixture-2026-08-21.json \
        --azure reports/azure-fixture-2026-08-21-rescored.json \
        --out reports/fixture-comparison.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def fmt_rate(found: int, total: int) -> str:
    if total == 0:
        return "n/a"
    return f"{found}/{total} = {found / total:.4f}"


def mask_summary_table(model_um: dict, model_m: dict, azure: dict) -> str:
    """Model recall is identical merged vs. unmerged by construction (see
    fixture_scoring.merge_adjacent_whitespace's docstring): merging only
    ever bridges a whitespace gap, and whitespace was never required to be
    covered. Single-span recall is the number merging is expected to move."""
    lines = []
    lines.append("| | Model recall | Model single-span (unmerged) | Model single-span (merged) | Azure recall | Azure single-span |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    def row(label: str, um: dict, m: dict, a: dict) -> str:
        umm, mm, am = um["mask"], m["mask"], a["mask"]
        return (
            f"| {label} | {fmt_rate(mm['found'], mm['total'])} "
            f"| {fmt_rate(umm['single_span_found'], umm['total'])} "
            f"| {fmt_rate(mm['single_span_found'], mm['total'])} "
            f"| {fmt_rate(am['found'], am['total'])} "
            f"| {fmt_rate(am['single_span_found'], am['total'])} |"
        )

    lines.append(row("**Overall**", model_um["overall"], model_m["overall"], azure["overall"]))
    langs = sorted(model_m["per_language"].keys())
    for lang in langs:
        lines.append(row(lang, model_um["per_language"][lang], model_m["per_language"][lang], azure["per_language"][lang]))
    return "\n".join(lines)


def clear_summary_table(model_m: dict, azure: dict) -> str:
    lines = []
    lines.append("| | Model over-mask | Azure over-mask |")
    lines.append("|---|---:|---:|")

    def row(label: str, m: dict, a: dict) -> str:
        mc, ac = m["clear"], a["clear"]
        return f"| {label} | {fmt_rate(mc['violated'], mc['total'])} | {fmt_rate(ac['violated'], ac['total'])} |"

    lines.append(row("**Overall**", model_m["overall"], azure["overall"]))
    langs = sorted(model_m["per_language"].keys())
    for lang in langs:
        lines.append(row(lang, model_m["per_language"][lang], azure["per_language"][lang]))
    return "\n".join(lines)


def pure_negatives_section(model_summary: dict, azure_summary: dict) -> str:
    mn = model_summary["pure_negatives"]
    an = azure_summary["pure_negatives"]
    return (
        f"Of {mn['n_cases']} pure-negative cases (nothing to mask at all):\n\n"
        f"- Model left {mn['clean']}/{mn['n_cases']} completely clean, corrupted {mn['corrupted']}.\n"
        f"- Azure left {an['clean']}/{an['n_cases']} completely clean, corrupted {an['corrupted']}.\n"
    )


def case_lookup(cases: list[dict]) -> dict[str, dict]:
    return {c["id"]: c for c in cases}


def value_diffs(model_case: dict, azure_case: dict) -> dict:
    """Per-case value-level outcomes, paired by (kind, value)."""
    mask = []
    for m, a in zip(model_case["must_mask"], azure_case["must_mask"]):
        assert m["value"] == a["value"]
        mask.append({"value": m["value"], "occurrences": m["occurrences"], "model_found": m["found_all"], "azure_found": a["found_all"]})
    clear = []
    for m, a in zip(model_case["must_clear"], azure_case["must_clear"]):
        assert m["value"] == a["value"]
        clear.append({"value": m["value"], "occurrences": m["occurrences"], "model_violated": m["violated"], "azure_violated": a["violated"]})
    return {"mask": mask, "clear": clear}


def render_case(case: dict, diffs: dict) -> str:
    lines = [f"### `{case['id']}` ({case['lang']}, {case.get('len', '?')})", "", f"> {case['text']}", ""]
    if diffs["mask"]:
        lines.append("| must_mask | occurrences | model | azure |")
        lines.append("|---|---:|---|---|")
        for d in diffs["mask"]:
            model_s = "found" if d["model_found"] else "**MISSED**"
            azure_s = "found" if d["azure_found"] else "**MISSED**"
            lines.append(f"| `{d['value']}` | {len(d['occurrences'])} | {model_s} | {azure_s} |")
        lines.append("")
    if diffs["clear"]:
        lines.append("| must_clear | occurrences | model | azure |")
        lines.append("|---|---:|---|---|")
        for d in diffs["clear"]:
            model_s = "**OVER-MASKED**" if d["model_violated"] else "clear"
            azure_s = "**OVER-MASKED**" if d["azure_violated"] else "clear"
            lines.append(f"| `{d['value']}` | {len(d['occurrences'])} | {model_s} | {azure_s} |")
        lines.append("")
    model_spans = ", ".join(f"`{s['text']}`" for s in case["predicted_spans"]) or "(none)"
    lines.append(f"Model detected spans: {model_spans}")
    return "\n".join(lines)


def render_miss(case: dict, miss: dict, self_label: str, other_label: str, other_case: dict | None) -> str:
    """One genuine miss, quoted in full: the case text, the missed value and
    its occurrence offsets, and every span both detectors actually detected
    in that same case -- "what it did detect nearby". `case` is the
    detector that missed (`self_label`); `other_case` is the other
    detector's result on the same case, for context."""
    lines = [f"### `{case['id']}` ({case['lang']}, {case.get('len', '?')})", "", f"> {case['text']}", ""]
    lines.append(f"Missed: `{miss['value']}` at {miss['occurrences']}")
    self_spans = ", ".join(f"`{s['text']}`" for s in case["predicted_spans"]) or "(none)"
    lines.append(f"{self_label} detected spans in this case: {self_spans}")
    if other_case is not None:
        other_spans = ", ".join(f"`{s['text']}`" for s in other_case["predicted_spans"]) or "(none)"
        lines.append(f"{other_label} detected spans in this case: {other_spans}")
    return "\n".join(lines)


def genuine_misses(cases: list[dict], self_label: str, other_cases: dict[str, dict] | None, other_label: str) -> list[str]:
    out = []
    for c in cases:
        for m in c["must_mask"]:
            if not m["found_all"]:
                oc = other_cases.get(c["id"]) if other_cases else None
                out.append(render_miss(c, m, self_label, other_label, oc))
    return out


def find_disagreements_and_both_fail(model_cases: dict, azure_cases: dict) -> tuple[list, list]:
    disagreements = []
    both_fail = []
    for cid in sorted(model_cases):
        mc, ac = model_cases[cid], azure_cases[cid]
        diffs = value_diffs(mc, ac)
        case_disagree = {
            "mask": [d for d in diffs["mask"] if d["model_found"] != d["azure_found"]],
            "clear": [d for d in diffs["clear"] if d["model_violated"] != d["azure_violated"]],
        }
        case_both_fail = {
            "mask": [d for d in diffs["mask"] if not d["model_found"] and not d["azure_found"]],
            "clear": [d for d in diffs["clear"] if d["model_violated"] and d["azure_violated"]],
        }
        if case_disagree["mask"] or case_disagree["clear"]:
            disagreements.append((mc, case_disagree))
        if case_both_fail["mask"] or case_both_fail["clear"]:
            both_fail.append((mc, case_both_fail))
    return disagreements, both_fail


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path to eval_fixture.py's --out JSON")
    ap.add_argument("--azure", required=True, help="path to eval_fixture_azure.py's --out JSON (rescored, see --from-raw)")
    ap.add_argument("--out", required=True, help="path to write the markdown report")
    args = ap.parse_args()

    model = load(args.model)
    azure = load(args.azure)

    if model["n_cases"] != azure["n_cases"]:
        raise RuntimeError(f"case count mismatch: model={model['n_cases']} azure={azure['n_cases']}")

    model_cases = case_lookup(model["cases"])  # merged variant -- what a real deployment masks with
    azure_cases = case_lookup(azure["cases"])
    if set(model_cases) != set(azure_cases):
        raise RuntimeError("model and Azure result files scored a different set of case ids")

    disagreements, both_fail = find_disagreements_and_both_fail(model_cases, azure_cases)

    n_mask_disagree = sum(len(d["mask"]) for _, d in disagreements)
    n_clear_disagree = sum(len(d["clear"]) for _, d in disagreements)
    n_mask_both_fail = sum(len(d["mask"]) for _, d in both_fail)
    n_clear_both_fail = sum(len(d["clear"]) for _, d in both_fail)

    model_misses = genuine_misses(model["cases"], "Model", azure_cases, "Azure")
    azure_misses = genuine_misses(azure["cases"], "Azure", model_cases, "Model")

    lines = []
    lines.append("# Model vs. Azure AI Language: fixture comparison")
    lines.append("")
    lines.append(
        "**This is not a release gate.** It is a cheap early read, on 128 hand-written "
        "adversarial cases, of whether the model trained yesterday is in the right "
        "neighbourhood before building the real benchmark "
        "(`docs/superpowers/plans/2026-08-21-benchmark.md`). 128 cases is small; a few "
        "points of difference is noise, not a result."
    )
    lines.append("")
    lines.append("## Correction: an earlier version of this report said 0.6711, and that was wrong")
    lines.append("")
    lines.append(
        "The first pass through this comparison scored the model's recall at **0.6711** "
        "and attributed the gap to the model tagging two-token names as two separate "
        "`B-PERSON` spans -- e.g. `'Jan'` and `'de Vries'` for \"Jan de Vries\". That "
        "observation was correct; the scoring rule built on top of it was not. Both "
        "training corpora label a given name and a surname as *separate entities* "
        "(`GIVENNAME`/`SURNAME`, `GIVEN_NAME`/`SURNAME`) and `taxonomy.py` maps both onto "
        "Temperate's single `PERSON` type without merging the spans, so the model faithfully "
        "reproduces what it was taught: two touching `PERSON` spans that together mask the "
        "entire name and leak nothing. The old rule -- a must_mask value counts as found "
        "only if it is fully covered by *one* detected span -- scored that as a miss."
    )
    lines.append(
        "The corrected rule counts a must_mask value as found when every non-whitespace "
        "character of every occurrence is covered by the *union* of detected spans; a "
        "whitespace gap between two adjacent spans leaks nothing, so it isn't a miss. "
        "Rescoring the exact same model output this way gives **0.9145**, not 0.6711. "
        "Azure was rescored with the identical corrected rule, from its already-saved raw "
        "responses (no second API call) -- its recall did not move (0.9671 either way, "
        "since Azure already returns names as one continuous span). The old rule is kept "
        "as a secondary \"single-span\" figure, below, because it measures something real: "
        "whether a detector returns an entity as one span or several, which matters for "
        "pseudonymisation placeholder shape even though it doesn't matter for masking."
    )
    lines.append("")
    lines.append("## What this does and does not establish")
    lines.append("")
    lines.append(
        "- **Does establish:** an order-of-magnitude read on how the model's real-text recall "
        "and over-masking compare to Azure's, and to the model's own training-time numbers "
        "(0.9237 F1 on PERSON, 0.9877 on STREET_ADDRESS -- measured on held-out data from the "
        "*same synthetic generators* that produced the training data, not on hand-written text)."
    )
    lines.append(
        "- **Does not establish:** a release verdict, a statistically significant gap, or "
        "coverage of entity types/languages this fixture does not exercise. 128 cases (40 en / "
        "40 nl / 30 de / 18 fr, 152 must_mask values, 64 must_clear values) is a smoke test, not "
        "a benchmark -- individual disagreements and misses below are worth reading; the "
        "topline numbers are not worth treating as precise."
    )
    lines.append(
        "- Scoring is type-agnostic (the fixture names substrings, not entity types): a "
        "must_mask value counts as found (`found_all`) only if *every* occurrence in the text "
        "has every non-whitespace character covered by the union of detected spans; a "
        "must_clear value counts as violated if *any* detected span overlaps it at all."
    )
    lines.append(
        "- **Span merging** (new): before scoring, the model's adjacent same-type spans "
        "separated only by whitespace are merged into one span, restricted to `PERSON` and "
        "`STREET_ADDRESS`. Those are the two platform types `taxonomy.py` builds by collapsing "
        "multiple *positionally adjacent* corpus sub-labels of one templated field onto a "
        "single type (`GIVENNAME`+`SURNAME` -> `PERSON`; `STREET`+`BUILDINGNUM`(+`ZIPCODE`) -> "
        "`STREET_ADDRESS`). Other types that also draw from more than one corpus label "
        "(`NATIONAL_ID` from `IDCARDNUM`/`SOCIALNUM`; `PHONE_NUMBER` from `FAX_NUMBER`/`PHONE`) "
        "are excluded on purpose: those are *alternative* full values, not adjacent halves of "
        "one field, so two of them sitting next to each other is exactly the case the task "
        "brief warned about -- two adjacent card numbers separated by a space are more "
        "plausibly two people's data than one, and merging would silently combine them. "
        "**Limitation, not solved on purpose:** the same whitespace-only-gap rule can also "
        "join two genuinely distinct people (or addresses) that happen to appear adjacently "
        "with nothing but whitespace between them (e.g. one name per line in a list). For "
        "masking this is harmless -- both are still masked either way, and restoring the "
        "original text is unaffected -- it only changes whether they get one placeholder or "
        "two."
    )
    lines.append(
        "- Azure is scored the way Temperate's own driver would actually use it: only categories "
        "the category list of Temperate's earlier Azure client maps to a Temperate type count, filtered at "
        f"`confidenceScore >= {azure['min_score']}` (config/pii.php's `score_threshold` default, "
        "confirmed unset in the .env file). Azure's raw API returns more than this; "
        "counting all of it would score a capability Temperate's product does not ship."
    )
    lines.append("")

    lines.append("## Measurements and the commands that produced them")
    lines.append("")
    lines.append("```")
    lines.append(model["command"])
    lines.append("python eval_fixture_azure.py --out reports/azure-fixture-2026-08-21.json   # original live run, unchanged")
    lines.append(azure["command"])
    lines.append("python " + " ".join(sys.argv))
    lines.append("```")
    lines.append("")
    lines.append(f"- Model checkpoint: `{model['checkpoint']}` (max_length={model['max_length']}, stride={model['stride']}); merge_types={model['merge_types']}")
    lines.append(f"- Azure modelVersion: `{azure['azure_model_version']}`, api-version={azure['api_version']}, rescored_from=`{azure.get('rescored_from')}` (no live Azure call for this rescore; no credentials read)")
    lines.append("")

    lines.append("## Overall and per-language: must_mask")
    lines.append("")
    lines.append(mask_summary_table(model["summary_unmerged"], model["summary_merged"], azure["summary"]))
    lines.append("")
    lines.append(
        "Model recall is identical merged vs. unmerged, as expected: merging only bridges "
        "whitespace gaps, and whitespace was never required to be covered by the corrected "
        "rule. What merging moves is the single-span figure -- from 0.6711 (the old headline "
        "number) up to parity with recall, because the PERSON/STREET_ADDRESS spans that used "
        "to arrive as two or three fragments now arrive as one."
    )
    lines.append("")

    lines.append("## Overall and per-language: must_clear (over-masking)")
    lines.append("")
    lines.append(clear_summary_table(model["summary_merged"], azure["summary"]))
    lines.append("")
    lines.append("Over-masking is unaffected by the scoring correction or by span merging (merging never extends a span beyond the union it was already part of).")
    lines.append("")

    lines.append("## Pure negatives (40 cases with nothing to mask)")
    lines.append("")
    lines.append(pure_negatives_section(model["summary"], azure["summary"]))

    lines.append(f"## Model's genuine misses ({len(model_misses)} must_mask values)")
    lines.append("")
    lines.append(
        "Every must_mask value the model (merged spans) still fails to fully cover, after "
        "both fixes. Each is a model weakness, an Azure weakness, or a fixture case worth "
        "re-checking by hand; quoted in full with what the model actually detected in that "
        "same case."
    )
    lines.append("")
    if not model_misses:
        lines.append("(none)")
    else:
        for block in model_misses:
            lines.append(block)
            lines.append("")

    lines.append(f"## Azure's genuine misses ({len(azure_misses)} must_mask values)")
    lines.append("")
    lines.append("Every must_mask value Azure (at the configured confidence threshold) still fails to fully cover.")
    lines.append("")
    if not azure_misses:
        lines.append("(none)")
    else:
        for block in azure_misses:
            lines.append(block)
            lines.append("")

    lines.append(
        f"## Disagreements ({len(disagreements)} cases, {n_mask_disagree} must_mask + "
        f"{n_clear_disagree} must_clear value-level disagreements)"
    )
    lines.append("")
    lines.append(
        "Every case where the model (merged spans) and Azure reached a different verdict on "
        "the same substring, quoted in full. This is the same information as the two "
        "genuine-misses sections above, cross-referenced against each other, kept for "
        "side-by-side reading."
    )
    lines.append("")
    if not disagreements:
        lines.append("(none)")
    else:
        for case, diffs in disagreements:
            lines.append(render_case(case, diffs))
            lines.append("")

    lines.append(
        f"## Both fail ({len(both_fail)} cases, {n_mask_both_fail} must_mask + "
        f"{n_clear_both_fail} must_clear value-level shared failures)"
    )
    lines.append("")
    lines.append(
        "Cases where both detectors failed the same value the same way. These point at "
        "genuinely hard text, not a single detector's gap."
    )
    lines.append("")
    if not both_fail:
        lines.append("(none)")
    else:
        for case, diffs in both_fail:
            lines.append(render_case(case, diffs))
            lines.append("")

    Path(args.out).write_text("\n".join(lines))
    print(f"Wrote {args.out}")
    print(f"{len(model_misses)} model misses, {len(azure_misses)} azure misses")
    print(f"{len(disagreements)} disagreement cases, {len(both_fail)} both-fail cases")


if __name__ == "__main__":
    main()
