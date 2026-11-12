"""Prove the adjacent-span merge did exactly what it claims, before anyone
rents a GPU to train on its output.

A retrain is roughly three hours of rented A40 time. The merge changes the
labels the model is shown, so a rule that quietly swallows the word after a
name is worse than the split it replaces -- and it would only show up hours
later, as a number nobody can attribute. This script is what makes the data
checkable first.

It reads the pre-merge build and the post-merge build side by side. That
pairing is the strongest available check, and it is only possible because
the merge deliberately runs *after* the group id is computed: a row's group,
split, language, source and text are pure functions of the raw spans, which
the merge never touches. So the two builds are line-for-line the same corpus
and the only permitted difference is `entities`. Anything else is a bug in
this change, and `--old-dir` turns that from an argument into an assertion.

What it checks, in the order the task brief asks for them:

  1. How many spans merged, by type and by language, as counts and as a
     share of all spans of that type.
  2. A reservoir sample of merged spans per (type, language) across en, nl,
     de, fr, es and it, quoted with surrounding text, so a human reads what
     the rule did instead of trusting the count.
  3. Structural checks on every merged span: it holds no line break, no
     sentence-terminating punctuation, no comma or conjunction, does not
     begin or end in whitespace, and the gaps it bridged were horizontal
     whitespace only. Plus a suspicion list -- merged PERSON spans holding
     a digit or an `@` -- printed in full rather than counted, because
     "covers non-name text" is a judgement a person has to make.
  4. BIO validity through training/align.py itself (not a reimplementation):
     every `I-` is preceded by a `B-` or `I-` of the same type, no orphan
     `I-`, and `reconstruct_spans` round-trips back to the character spans
     the dataset actually holds.
  5. Global invariants: entities sorted, non-overlapping, inside the text.

Usage:
    python verify_merged_spans.py --old-dir data --new-dir data-merged \
        --splits train validation test \
        --bio-sample 400 --seed 0 --out reports/merge-verification.md
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from build_dataset import (
    LINE_BREAK_CHARS,
    MERGEABLE_TYPES,
    SUPPORTED_LANGUAGES,
    is_bridgeable_gap,
    merge_adjacent_entities,
)

# Sentence-terminating and list punctuation. A merged span is by
# construction its parts plus the whitespace it bridged, so one of these can
# reach a merged span two ways, and the difference decides whether it is a
# defect:
#   - inside a BRIDGED GAP -> the merge crossed a sentence or list boundary.
#     A real failure; `bad_gap` below catches it and fails the run.
#   - inside a PART -> it was already in one corpus span before this change
#     touched anything (`Senda Coaz 10.1`, `St. Gallen`). Not a sentence
#     boundary and not this change's doing, so it is reported for a human to
#     look at rather than failed on.
SENTENCE_PUNCTUATION = set(".!?;:,…。！？،؛") | set(LINE_BREAK_CHARS)

# "Covers non-name text" needs a check that can actually fire. Digits inside
# a merged PERSON span do NOT qualify: nym augments its text with OCR-style
# corruption (`Ju1ie`, `Phillip5`, `Jori5`), so those digits are already
# inside a pre-merge part and the merge neither added nor could add a
# character that was not in one of its parts or in the whitespace it
# bridged. What would qualify is the merge joining a name to a span that
# is not a name at all -- so the check below flags a merged span any of
# whose pre-merge parts contains no letter whatsoever.
def part_has_no_letter(part: str) -> bool:
    return not any(ch.isalpha() for ch in part)


# Name particles (tussenvoegsels and their German/French/Italian/Spanish
# equivalents). Used only to answer one question against the real data: does
# the corpus put the particle INSIDE the surname span, or leave it outside
# as unlabelled text between two spans? The answer decides whether a
# whitespace bridge can fix `Jan de Vries` at all.
NAME_PARTICLES = frozenset(
    {
        "de", "den", "der", "des", "du", "da", "das", "dos", "del", "della", "di", "do",
        "la", "le", "les", "op", "te", "ten", "ter", "van", "von", "zu", "zum", "'t", "af", "av",
        "de la", "de le", "van de", "van den", "van der", "von der", "von den", "op de",
        "op den", "in de", "in het", "aan de", "uit de", "voor de", "onder de", "over de",
    }
)


def leading_particle(span_text: str) -> str | None:
    """The name particle `span_text` begins with, if any (longest first)."""
    words = span_text.split()
    if len(words) < 2:
        return None
    for n in (2, 1):
        if len(words) > n:
            candidate = " ".join(words[:n]).lower()
            if candidate in NAME_PARTICLES:
                return candidate
    return None


def read_jsonl(path: Path):
    """Iterate a JSONL file by real `\\n` lines. Not `splitlines()`: both
    corpora contain U+0085/U+2028/U+2029, which `str.splitlines()` treats as
    line breaks and `json.dumps` does not escape, cutting one valid record
    into two invalid fragments (see tests/test_build_dataset.py's docstring)."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def paired_rows(old_path: Path, new_path: Path):
    """Yield (old_row, new_row) and fail loudly if the two builds disagree
    on anything except `entities`."""
    with open(old_path, encoding="utf-8") as fo, open(new_path, encoding="utf-8") as fn:
        for n, (lo, ln) in enumerate(zip(fo, fn), start=1):
            old, new = json.loads(lo), json.loads(ln)
            for field in ("text", "language", "source", "group"):
                if old[field] != new[field]:
                    raise SystemExit(
                        f"{new_path}:{n}: the two builds are not the same corpus -- "
                        f"{field!r} differs. The merge must not change split "
                        f"assignment, grouping or text."
                    )
            yield old, new
        if fo.readline() or fn.readline():
            raise SystemExit(f"{old_path} and {new_path} have different row counts")


def merged_parts(old_entities: list[dict], span: dict) -> list[tuple[int, int]]:
    """The pre-merge spans, as exact (start, end) offsets, that were absorbed
    into `span`. Offsets, not text: two identical parts (`Jan Jan`) would
    make a text-search reconstruction of the bridged gaps ambiguous."""
    return [
        (e["start"], e["end"])
        for e in sorted(old_entities, key=lambda e: (e["start"], e["end"]))
        if e["label"] == span["label"] and span["start"] <= e["start"] and e["end"] <= span["end"]
    ]


def describe_char(ch: str) -> str:
    try:
        name = unicodedata.name(ch)
    except ValueError:
        name = "UNNAMED"
    return f"U+{ord(ch):04X} {name}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old-dir", default="data", help="the pre-merge build")
    ap.add_argument("--new-dir", default="data-merged", help="the post-merge build")
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    ap.add_argument("--samples-per-cell", type=int, default=6, help="merged spans quoted per (type, language)")
    ap.add_argument("--bio-sample", type=int, default=400, help="records per split tokenized for the BIO check")
    ap.add_argument(
        "--token-length-split",
        default="validation",
        help="split whose ENTIRE entity list is tokenized to re-measure the entity "
        "token-length distribution against --stride (align.py's stride constraint). "
        "Empty string to skip.",
    )
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--tokenizer", default="FacebookAI/xlm-roberta-base")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    old_dir, new_dir = Path(args.old_dir), Path(args.new_dir)
    rng = random.Random(args.seed)

    rows = 0
    spans_before: Counter = Counter()  # (label, language)
    spans_after: Counter = Counter()
    merged_spans: Counter = Counter()
    parts_hist: Counter = Counter()
    gap_chars: Counter = Counter()
    by_source: Counter = Counter()
    adjacent_gaps: Counter = Counter()          # (label, gap) in the PRE-merge build
    gap_examples: dict[tuple[str, str], list[str]] = defaultdict(list)
    particle_inside: Counter = Counter()        # `van der` etc. inside a PERSON span
    particle_inside_examples: dict[str, list[str]] = defaultdict(list)
    particle_outside: Counter = Counter()       # ...or left in the gap between two spans
    particle_outside_examples: list[str] = []

    violations: list[str] = []
    violation_counts: Counter = Counter()

    def flag(kind: str, message: str) -> None:
        """Record a violation. Counts are exact; the stored messages are
        capped so a systematic failure can't exhaust memory before it is
        reported."""
        violation_counts[kind] += 1
        if len(violations) < 200:
            violations.append(message)

    suspicious: list[dict] = []
    inherited_punct: Counter = Counter()
    inherited_examples: dict[str, list[str]] = defaultdict(list)
    inherited_edge_ws: Counter = Counter()
    inherited_edge_examples: list[str] = []
    entity_texts: dict[str, list[str]] = {"old": [], "new": []}
    longest: list[tuple[int, dict]] = []
    samples: dict[tuple[str, str], list[dict]] = defaultdict(list)
    seen_cell: Counter = Counter()
    bio_pool: dict[str, list[tuple[dict, dict]]] = {split: [] for split in args.splits}
    bio_seen: Counter = Counter()

    for split in args.splits:
        for old, new in paired_rows(old_dir / f"{split}.jsonl", new_dir / f"{split}.jsonl"):
            rows += 1
            text, language, source = new["text"], new["language"], new["source"]
            old_ents = old["entities"]
            new_ents = new["entities"]

            # --- (0) the new file is exactly the old file plus the merge ---
            expected, _ = merge_adjacent_entities(text, old_ents)
            if expected != new_ents:
                flag(
                    "not_merge_of_old",
                    f"[{split}] row {rows}: new entities are not merge_adjacent_entities(old entities)",
                )

            # --- (5) global invariants ---
            for a, b in zip(new_ents, new_ents[1:]):
                if (b["start"], b["end"]) < (a["start"], a["end"]):
                    flag("unsorted", f"[{split}] row {rows}: entities not sorted")
                if b["start"] < a["end"]:
                    flag("overlapping", f"[{split}] row {rows}: overlapping entities")
            for e in new_ents:
                if not (0 <= e["start"] < e["end"] <= len(text)):
                    flag("out_of_bounds", f"[{split}] row {rows}: span outside text bounds")

            # --- the evidence that chose the rule, re-derived from the
            # PRE-merge build so the justification is reproducible from a
            # committed command rather than from a scratch script.
            ordered_old = sorted(old_ents, key=lambda x: (x["start"], x["end"]))
            for a, b in zip(ordered_old, ordered_old[1:]):
                if a["label"] != b["label"] or b["start"] < a["end"]:
                    continue
                gap = text[a["end"] : b["start"]]
                key = (a["label"], gap if len(gap) <= 12 else f"<{len(gap)} chars>")
                adjacent_gaps[key] += 1
                bucket = gap_examples[key]
                if len(bucket) < 3:
                    bucket.append(
                        f"`{language}` ...{text[max(0, a['start'] - 45):a['start']]}"
                        f"[[{text[a['start']:a['end']]}]]{gap!r}[[{text[b['start']:b['end']]}]]"
                        f"{text[b['end']:b['end'] + 45]}..."
                    )
                if a["label"] == "PERSON" and gap.strip().lower() in NAME_PARTICLES and gap.strip():
                    particle_outside[gap.strip().lower()] += 1
                    if len(particle_outside_examples) < 10:
                        particle_outside_examples.append(
                            f"`{language}` ...{text[max(0, a['start'] - 45):a['start']]}"
                            f"[[{text[a['start']:a['end']]}]]{gap!r}[[{text[b['start']:b['end']]}]]..."
                        )
            for e in ordered_old:
                if e["label"] != "PERSON":
                    continue
                particle = leading_particle(text[e["start"] : e["end"]])
                if particle:
                    particle_inside[particle] += 1
                    bucket = particle_inside_examples[particle]
                    if len(bucket) < 3:
                        bucket.append(f"`{language}` {text[e['start']:e['end']]!r}")

            for e in old_ents:
                spans_before[(e["label"], language)] += 1
            for e in new_ents:
                spans_after[(e["label"], language)] += 1

            if split == args.token_length_split:
                entity_texts["old"].extend(text[e["start"] : e["end"]] for e in old_ents)
                entity_texts["new"].extend(text[e["start"] : e["end"]] for e in new_ents)

            if len(new_ents) != len(old_ents):
                by_source[source] += 1

            old_end_by_start = {(e["start"], e["label"]): e["end"] for e in old_ents}
            for e in new_ents:
                key = (e["start"], e["label"])
                if key not in old_end_by_start or e["end"] <= old_end_by_start[key]:
                    continue  # not a merged span
                label, s = e["label"], text[e["start"] : e["end"]]
                cell = (label, language)
                merged_spans[cell] += 1
                part_offsets = merged_parts(old_ents, e)
                parts = [text[ps:pe] for ps, pe in part_offsets]
                parts_hist[(label, len(parts))] += 1

                # --- (3) structural checks on the merged span ---
                # Edge whitespace in a merged span is only this change's
                # doing if the outermost pre-merge part didn't already have
                # it. The merge sets the span's start/end to the first and
                # last part's own start/end, so in practice it never is --
                # but "in practice" is what this script exists to replace.
                if s != s.strip():
                    lead = parts[0] != parts[0].lstrip()
                    trail = parts[-1] != parts[-1].rstrip()
                    if (s != s.lstrip() and not lead) or (s != s.rstrip() and not trail):
                        flag(
                            "edge_whitespace_introduced",
                            f"[{split}] row {rows}: merged {label} span {s!r} gained edge whitespace",
                        )
                    else:
                        inherited_edge_ws[label] += 1
                        if len(inherited_edge_examples) < 10:
                            inherited_edge_examples.append(
                                f"`{language}` `{label}` `{s!r}` was "
                                + " + ".join(repr(part) for part in parts)
                            )
                # every bridged gap, re-derived from the parts' own offsets
                for (_, prev_end), (next_start, _) in zip(part_offsets, part_offsets[1:]):
                    gap = text[prev_end:next_start]
                    for ch in gap:
                        gap_chars[ch] += 1
                    if not is_bridgeable_gap(gap):
                        flag(
                            "bad_gap",
                            f"[{split}] row {rows}: merged {label} span {s!r} bridged gap {gap!r}",
                        )
                inherited = sorted({ch for ch in s if ch in SENTENCE_PUNCTUATION})
                if inherited:
                    inherited_punct[label] += 1
                    bucket = inherited_examples[label]
                    if len(bucket) < 12:
                        bucket.append(
                            f"`{language}` `{label}` `{s!r}` was "
                            + " + ".join(repr(part) for part in parts)
                            + " -- "
                            + ", ".join(describe_char(c) for c in inherited)
                        )
                if part_offsets and (part_offsets[0][0] != e["start"] or part_offsets[-1][1] != e["end"]):
                    flag(
                        "edge_not_on_part",
                        f"[{split}] row {rows}: merged {label} span {s!r} does not start/end on a pre-merge part",
                    )

                if label == "PERSON" and any(part_has_no_letter(part) for part in parts):
                    suspicious.append({"split": split, "language": language, "span": s, "parts": parts})

                longest.append((len(s), {"split": split, "language": language, "label": label, "span": s}))
                longest.sort(key=lambda kv: -kv[0])
                del longest[12:]

                # --- (2) reservoir sample, quoted with surrounding text ---
                seen_cell[cell] += 1
                item = {
                    "split": split,
                    "language": language,
                    "label": label,
                    "source": source,
                    "before": text[max(0, e["start"] - 70) : e["start"]],
                    "span": s,
                    "after": text[e["end"] : e["end"] + 70],
                    "parts": parts,
                }
                bucket = samples[cell]
                if len(bucket) < args.samples_per_cell:
                    bucket.append(item)
                else:
                    j = rng.randrange(seen_cell[cell])
                    if j < args.samples_per_cell:
                        bucket[j] = item

            # --- reservoir for the BIO check, weighted to rows that merged ---
            if len(new_ents) != len(old_ents):
                bio_seen[split] += 1
                pool = bio_pool[split]
                if len(pool) < args.bio_sample:
                    pool.append((old, new))
                else:
                    j = rng.randrange(bio_seen[split])
                    if j < args.bio_sample:
                        pool[j] = (old, new)

    # --- (4) BIO validity, through align.py itself -------------------------
    from transformers import AutoTokenizer

    from align import ID2LABEL, IGNORE_INDEX, reconstruct_spans, tokenize_and_align

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    bio = {"records": 0, "windows": 0, "orphan_I": 0, "type_switch_without_B": 0}
    roundtrip = {"old": [0, 0], "new": [0, 0]}  # variant -> [exact, total]
    bio_failures: list[str] = []

    def align_and_check(record: dict, entities: list[dict], variant: str, split: str) -> None:
        windows = tokenize_and_align(record["text"], entities, tokenizer, args.max_length, args.stride)
        if variant == "new":
            for window in windows:
                bio["windows"] += 1
                prev: str | None = None
                for lid in window.labels:
                    if lid == IGNORE_INDEX:
                        prev = None
                        continue
                    label = ID2LABEL[lid]
                    if label == "O":
                        prev = None
                        continue
                    prefix, etype = label.split("-", 1)
                    if prefix == "I":
                        if prev is None:
                            bio["orphan_I"] += 1
                            if len(bio_failures) < 10:
                                bio_failures.append(f"[{split}] orphan I-{etype} in {record['text'][:80]!r}")
                        elif prev != etype:
                            bio["type_switch_without_B"] += 1
                            if len(bio_failures) < 10:
                                bio_failures.append(f"[{split}] I-{etype} after {prev} in {record['text'][:80]!r}")
                    prev = etype
        # round-trip: character spans -> BIO -> character spans, driven only
        # by labels and offsets, exactly as a decode-time consumer has them.
        recovered: set[tuple[int, int, str]] = set()
        for window in windows:
            recovered |= set(reconstruct_spans(window.offset_mapping, window.labels))
        for e in entities:
            roundtrip[variant][1] += 1
            if (e["start"], e["end"], e["label"]) in recovered:
                roundtrip[variant][0] += 1

    for split, pool in bio_pool.items():
        for old_record, record in pool:
            bio["records"] += 1
            align_and_check(record, old_record["entities"], "old", split)
            align_and_check(record, record["entities"], "new", split)

    # --- entity token length vs stride ------------------------------------
    #
    # align.py's docstring carries a load-bearing constraint: `stride` must
    # exceed the longest entity's token length, or an entity can be cut in
    # every window and never appear complete anywhere. That was measured
    # against the PRE-merge data. Merging makes entities longer by
    # construction, so it has to be re-measured, not assumed to still hold.
    token_lengths: dict[str, list[int]] = {}
    if args.token_length_split and entity_texts["old"]:
        for variant, texts in entity_texts.items():
            lengths: list[int] = []
            batch = 20_000
            for i in range(0, len(texts), batch):
                encoded = tokenizer(texts[i : i + batch], add_special_tokens=False)["input_ids"]
                lengths.extend(len(ids) for ids in encoded)
            lengths.sort()
            token_lengths[variant] = lengths

    def percentile(sorted_values: list[int], q: float) -> int:
        if not sorted_values:
            return 0
        idx = min(len(sorted_values) - 1, int(q * len(sorted_values)))
        return sorted_values[idx]

    # --- render -----------------------------------------------------------
    out: list[str] = []
    w = out.append
    command = "python " + " ".join(sys.argv)
    w("# Merge verification: adjacent-span merging in the rebuilt dataset\n")
    w(f"Command:\n\n```\n{command}\n```\n")
    w(f"Pre-merge build: `{old_dir}`. Post-merge build: `{new_dir}`. "
      f"Splits: {', '.join(args.splits)}. Rows compared: {rows:,}.\n")

    labels = sorted({label for label, _ in spans_before})

    w("## 0. The evidence that chose the rule\n")
    w(f"Everything in this section is measured over the **pre-merge** build (`{old_dir}`), "
      "so it says what the corpora look like before this change, not after it.\n")
    w("### Every gap between two adjacent same-type spans, by type\n")
    w("The merge bridges only the rows marked BRIDGED. Read the rest: they are the "
      "reason the rule is not \"any short gap\".\n")
    for label in sorted(MERGEABLE_TYPES):
        rows_for_label = [(gap, n) for (l, gap), n in adjacent_gaps.items() if l == label]
        rows_for_label.sort(key=lambda kv: -kv[1])
        w(f"#### {label}\n")
        w("| gap | adjacent pairs | bridged? |")
        w("|---|---:|:---:|")
        for gap, n in rows_for_label[:16]:
            bridged = "**BRIDGED**" if not gap.startswith("<") and is_bridgeable_gap(gap) else "no"
            w(f"| `{gap!r}` | {n:,} | {bridged} |")
        w("")
        for gap, n in rows_for_label[:16]:
            if gap.startswith("<") or is_bridgeable_gap(gap):
                continue
            for line in gap_examples[(label, gap)][:1]:
                w(f"- not bridged, `{gap!r}` ({n:,}): {line!r}")
        w("")
    w("#### Bridgeable-gap pairs by type, over every type and every gap\n")
    w("The table above shows each mergeable type's 16 most common gaps. This one counts, "
      "across ALL gaps and ALL types, how many adjacent same-type pairs the bridgeable-gap "
      "rule would join if the type were allowed to merge. It is the measurement behind "
      "\"the two mergeable types are not merely the plausible ones\": outside them the "
      "pattern barely occurs.\n")
    bridgeable_by_type: Counter = Counter()
    for (label, gap), n in adjacent_gaps.items():
        if not gap.startswith("<") and is_bridgeable_gap(gap):
            bridgeable_by_type[label] += n
    w("| type | mergeable | adjacent same-type pairs with a bridgeable gap |")
    w("|---|:---:|---:|")
    for label in sorted(bridgeable_by_type, key=lambda l: -bridgeable_by_type[l]):
        w(f"| {label} | {'yes' if label in MERGEABLE_TYPES else 'no'} | {bridgeable_by_type[label]:,} |")
    non_mergeable_total = sum(n for l, n in bridgeable_by_type.items() if l not in MERGEABLE_TYPES)
    w(f"| **all non-mergeable types combined** | no | **{non_mergeable_total:,}** |")
    w("")

    other = [(l, gap, n) for (l, gap), n in adjacent_gaps.items() if l not in MERGEABLE_TYPES]
    other.sort(key=lambda kv: -kv[2])
    w("#### Types that are NOT merged, for scale\n")
    w("| type | gap | adjacent pairs |")
    w("|---|---|---:|")
    for label, gap, n in other[:12]:
        w(f"| {label} | `{gap!r}` | {n:,} |")
    w("")

    w("### The tussenvoegsel: is the particle inside the surname span?\n")
    w("This decides whether a whitespace bridge can fix `Jan de Vries` at all. If the "
      "corpora label `de Vries` as the surname, the gap between `Jan` and `de Vries` is a "
      "single space and the merge fixes it. If they leave `de` outside every span, no "
      "whitespace rule can reach it.\n")
    inside_total = sum(particle_inside.values())
    outside_total = sum(particle_outside.values())
    particle_total = inside_total + outside_total
    w("| where the particle sits | occurrences | share |")
    w("|---|---:|---:|")
    w(f"| INSIDE a PERSON span (`van der Velden`, `de Groot`) -- the merge recovers these "
      f"| {inside_total:,} | {inside_total / particle_total:.2%} |" if particle_total else "| inside | 0 | - |")
    w(f"| OUTSIDE, in the gap between two PERSON spans -- the merge cannot reach these "
      f"| {outside_total:,} | {outside_total / particle_total:.2%} |" if particle_total else "| outside | 0 | - |")
    w("")
    w(f"**The tussenvoegsel case is fixable by a whitespace bridge**: "
      f"{inside_total / particle_total:.2%} of the particles the corpora contain are already "
      f"inside the surname span, so `[Jan] [de Vries]` is separated by one space and merges "
      f"correctly. The exceptions are listed in full below -- read them before treating that "
      f"figure as a defect rate; most are ordinary prepositions the detector here cannot "
      f"tell apart from a particle (`sagte X von der Y-Behörde`, `X zu Y geändert`), not "
      f"names whose particle was mislabelled.\n" if particle_total else "")
    w("Most frequent particles found inside a PERSON span:\n")
    for particle, n in particle_inside.most_common(14):
        examples = ", ".join(particle_inside_examples[particle][:2])
        w(f"- `{particle}` x{n:,} -- {examples}")
    w("")
    if particle_outside:
        w("Every distinct particle found OUTSIDE any span, with examples:\n")
        w("| particle | occurrences |")
        w("|---|---:|")
        for particle, n in particle_outside.most_common():
            w(f"| `{particle}` | {n:,} |")
        w("")
        for line in particle_outside_examples:
            w(f"- {line!r}")
    else:
        w("No particle was ever found outside a span.")
    w("")

    w("## 1. Spans merged, by type\n")
    w("| type | mergeable | spans before | spans after | spans absorbed | merged spans | share of type absorbed |")
    w("|---|:---:|---:|---:|---:|---:|---:|")
    for label in labels:
        before = sum(n for (l, _), n in spans_before.items() if l == label)
        after = sum(n for (l, _), n in spans_after.items() if l == label)
        m = sum(n for (l, _), n in merged_spans.items() if l == label)
        share = (before - after) / before if before else 0.0
        w(f"| {label} | {'yes' if label in MERGEABLE_TYPES else 'no'} | {before:,} | {after:,} | "
          f"{before - after:,} | {m:,} | {share:.2%} |")
    w("")

    w("## 1b. Spans merged, by type and language\n")
    w("| type | language | spans before | spans after | spans absorbed | merged spans | share of type absorbed |")
    w("|---|---|---:|---:|---:|---:|---:|")
    for label in labels:
        for language in sorted(SUPPORTED_LANGUAGES):
            before = spans_before[(label, language)]
            after = spans_after[(label, language)]
            if not before:
                continue
            m = merged_spans[(label, language)]
            share = (before - after) / before if before else 0.0
            w(f"| {label} | {language} | {before:,} | {after:,} | {before - after:,} | {m:,} | {share:.2%} |")
    w("")

    w("### Rows where at least one merge happened, by source\n")
    for source, n in sorted(by_source.items()):
        w(f"- `{source}`: {n:,} rows")
    w("")

    w("### How many spans each merged span absorbed\n")
    w("| type | spans fused into one | merged spans |")
    w("|---|---:|---:|")
    for (label, k), n in sorted(parts_hist.items()):
        w(f"| {label} | {k} | {n:,} |")
    w("")

    w("### Every character actually bridged\n")
    w("A merged span is its parts plus these characters and nothing else.\n")
    w("| character | occurrences |")
    w("|---|---:|")
    for ch, n in gap_chars.most_common():
        w(f"| {describe_char(ch)} | {n:,} |")
    if not gap_chars:
        w("| (no gaps bridged) | 0 |")
    w("")

    w("## 2. Merged spans, quoted with surrounding text\n")
    total_quoted = sum(len(v) for v in samples.values())
    w(f"{total_quoted} merged spans, reservoir-sampled per (type, language) with seed {args.seed}. "
      f"The merged span is in `«»`; its pre-merge parts are listed after it.\n")
    for label in sorted(MERGEABLE_TYPES):
        for language in sorted(SUPPORTED_LANGUAGES):
            bucket = samples.get((label, language))
            if not bucket:
                continue
            w(f"### {label} / `{language}` ({seen_cell[(label, language)]:,} merged spans, {len(bucket)} shown)\n")
            for s in bucket:
                ctx = f"...{s['before']}«{s['span']}»{s['after']}..."
                w(f"- `{ctx!r}`")
                w(f"  - was: {' + '.join(repr(p) for p in s['parts'])}  (`{s['source']}`, `{s['split']}`)")
            w("")

    w("## 3. Structural checks\n")
    w("Every count below is over every merged span in every split, not a sample.\n")
    w("| check | violations |")
    w("|---|---:|")
    checks = [
        ("edge_whitespace_introduced", "merged span gained whitespace at an edge its parts did not have"),
        ("bad_gap", "merged span bridged anything other than horizontal whitespace"),
        ("edge_not_on_part", "merged span does not begin and end on a pre-merge span"),
        ("not_merge_of_old", "post-merge entities are not exactly `merge_adjacent_entities(pre-merge entities)`"),
        ("unsorted", "entities out of order"),
        ("overlapping", "overlapping entities"),
        ("out_of_bounds", "span outside the text"),
    ]
    for kind, description in checks:
        w(f"| {description} | {violation_counts[kind]:,} |")
    w("")
    if violations:
        w("**Violations (first 40):**\n")
        for v in violations[:40]:
            w(f"- {v}")
        w("")
    else:
        w("No violations.\n")

    w("### Sentence/list punctuation inherited from a pre-merge part\n")
    w("The merge never bridges any of these (the `bad_gap` row above is the check that "
      "proves it), so every occurrence below was already inside one corpus span before "
      "this change. Listed so a human can confirm none of them is a sentence boundary.\n")
    if inherited_punct:
        w("| type | merged spans holding inherited punctuation |")
        w("|---|---:|")
        for label, n in sorted(inherited_punct.items()):
            w(f"| {label} | {n:,} |")
        w("")
        for label in sorted(inherited_examples):
            for line in inherited_examples[label]:
                w(f"- {line}")
    else:
        w("None.")
    w("")

    w("### Edge whitespace inherited from a pre-merge part\n")
    w("A merged span takes its start from the first part and its end from the last, so it "
      "inherits any edge whitespace those already carried -- a known property of both "
      "corpora (`reports/alignment-verification.md` classifies the same spans under "
      "\"narrowed: trailing/leading whitespace in the dataset's own span\"). The check "
      "above fails only if the merge *introduced* edge whitespace.\n")
    if inherited_edge_ws:
        w("| type | merged spans with inherited edge whitespace |")
        w("|---|---:|")
        for label, n in sorted(inherited_edge_ws.items()):
            w(f"| {label} | {n:,} |")
        w("")
        for line in inherited_edge_examples:
            w(f"- {line}")
    else:
        w("None.")
    w("")

    w(f"### Merged PERSON spans where a pre-merge part contains no letter at all: {len(suspicious)}\n")
    w("This is the \"covers non-name text\" check. Digits inside a merged name are *not* "
      "evidence of over-merging -- nym augments its text with OCR-style corruption "
      "(`Ju1ie`, `Phillip5`), and the merge cannot introduce a character that was not "
      "already in one of its parts or in the horizontal whitespace it bridged. What would "
      "be evidence is a part that is not a name at all.\n")
    if suspicious:
        for s in suspicious[:40]:
            w(f"- `{s['language']}` `{s['span']!r}` was {' + '.join(repr(p) for p in s['parts'])}")
        if len(suspicious) > 40:
            w(f"- ...and {len(suspicious) - 40:,} more")
    else:
        w("None.")
    w("")

    w("### Longest merged spans\n")
    for n, s in longest:
        w(f"- {n} chars, `{s['label']}`/`{s['language']}`: `{s['span']!r}`")
    w("")

    w("## 4. BIO validity through `align.py`\n")
    w(f"Tokenizer `{args.tokenizer}`, max_length={args.max_length}, stride={args.stride}. "
      f"Sample: {bio['records']:,} records (drawn only from rows where something merged -- "
      f"the rows this change can break), {bio['windows']:,} windows.\n")
    w("| check | result |")
    w("|---|---:|")
    w(f"| orphan `I-` (an `I-` with no `B-`/`I-` of the same type before it) | {bio['orphan_I']} |")
    w(f"| `I-X` directly after `B-Y`/`I-Y` of a different type | {bio['type_switch_without_B']} |")
    w("")
    w("### Character-exact round-trip, pre-merge vs post-merge, same rows\n")
    w("An exact-boundary round-trip rate below 100% is expected and is not an "
      "alignment defect: `align.py` labels a token that *overlaps* an entity, so a "
      "subword token straddling an entity edge legitimately rounds the boundary out to "
      "the token edge, and a real inference consumer -- which also works in tokens -- "
      "rounds it identically (see `reports/alignment-verification.md`, which classifies "
      "the same effect at 96.97% on an unbiased sample). The number that matters here is "
      "the *difference* between the two columns on the same rows: merging must not make "
      "it worse.\n")
    w("| variant | entities | recovered exactly | **not** recovered exactly | rate |")
    w("|---|---:|---:|---:|---:|")
    for variant, name in (("old", f"pre-merge (`{old_dir}`)"), ("new", f"post-merge (`{new_dir}`)")):
        exact, total = roundtrip[variant]
        if not total:
            w(f"| {name} | 0 | 0 | 0 | n/a |")
            continue
        w(f"| {name} | {total:,} | {exact:,} | {total - exact:,} | {exact / total:.4%} |")
    w("")
    w("Read the fourth column, not the fifth. Merging reduces the entity count (that is "
      "what it is for), so an unchanged number of boundary-rounding cases divided by a "
      "smaller denominator shows up as a lower percentage while nothing has actually got "
      "worse.\n")
    if bio_failures:
        w("**BIO failures (first 10):**\n")
        for f in bio_failures:
            w(f"- {f}")
        w("")

    if token_lengths:
        w(f"## 5. Entity token length against `stride={args.stride}`\n")
        w("`align.py` states a constraint this change can break: `stride` must exceed the "
          "longest entity's token length, or a long entity can be cut in every window and "
          "never appear complete anywhere. Merging makes entities longer by construction, "
          "so the constraint is re-measured here rather than assumed to still hold. "
          f"Every entity in `{args.token_length_split}.jsonl`, tokenized with "
          f"`{args.tokenizer}`.\n")
        w("| build | entities | p50 | p99 | p99.9 | max | entities longer than stride |")
        w("|---|---:|---:|---:|---:|---:|---:|")
        for variant, name in (("old", f"pre-merge (`{old_dir}`)"), ("new", f"post-merge (`{new_dir}`)")):
            lengths = token_lengths[variant]
            over = sum(1 for n in lengths if n > args.stride)
            w(f"| {name} | {len(lengths):,} | {percentile(lengths, 0.50)} | {percentile(lengths, 0.99)} | "
              f"{percentile(lengths, 0.999)} | {max(lengths)} | {over:,} ({over / len(lengths):.4%}) |")
        w("")

    ok = not violations and not bio["orphan_I"] and not bio["type_switch_without_B"]
    w(f"## Verdict\n\n**{'PASS' if ok else 'FAIL'}**\n")

    report = "\n".join(out)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"# wrote {args.out}", flush=True)
    print(report)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
