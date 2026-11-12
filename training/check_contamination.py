"""Proves that no benchmark case has leaked into a training corpus.

Run before every training run, and again before believing any number the
benchmark produces afterwards. The check is cheap (a few minutes over the
full 821,530-row corpus) and the failure it prevents is total: a model
trained on the text it is later scored on makes every figure in
`training/reports/pii-detection-vs-azure-*.md` meaningless, and there is no way to
detect that after the fact from the scores alone.

What "contaminated" means here, and why one test is not enough
--------------------------------------------------------------
Three independent identities are checked, because a leak can happen three
different ways:

1. **By carrier id.** `benchmark/cases/carriers.json` names the document each
   case was cut from (`carrier_id`). `training/negatives/manifest.json` names
   the document each negative row was cut from. If those two id spaces
   intersect -- or if a negative document's URL, Wikimedia page title, CELEX
   id or ECB report slug matches a carrier's -- the same *document* is on
   both sides even if no sentence is byte-identical.

2. **By exact text.** Normalised (casefolded, punctuation-stripped,
   whitespace-collapsed) equality between a case's text and a training row's
   text. This is the check that catches the short cases -- `seed.json` has
   cases of seven words, which no n-gram check can see.

3. **By shared n-gram.** Any normalised word 8-gram that occurs in both a
   benchmark case (or a carrier document) and a training row. This is the
   check that matters: it catches a case that was reworded, retruncated, or
   re-cut from the same source paragraph, which is exactly how a leak
   happens in practice when two corpora are drawn from overlapping material.

   **Eight words is not, on its own, evidence of copying, and this tool does
   not pretend otherwise.** Measured on this repo's own corpus: 22 of
   `data-merged/train.jsonl`'s 729,928 rows share an 8-gram with a benchmark
   case, every one of them an `ai4privacy` row (a synthetic corpus that has
   never touched Wikipedia, EUR-Lex or the ECB), and every one of them
   sharing exactly one or two 8-grams and nothing longer. Raise n to ten and
   all 22 disappear.

   So a collision at the primary n triggers an **escalation**: the scan is
   re-run at larger n, and the verdict is decided at the escalation ceiling
   (default 12), not at 8. A stock phrase cannot survive twelve words; a
   re-cut paragraph produces dozens of overlapping shingles and survives
   twenty. The 8-gram count is still reported, as a coincidence count, so the
   distinction is visible rather than assumed away.

Privacy
-------
**No matched text is ever printed.** A collision is reported as a row index,
a case id and the *length* of what collided -- never the colliding words.
That is not squeamishness: the benchmark contains synthetic identifiers and
the training corpus contains generated ones, and this tool's output goes
into a committed report.

Usage
-----
    python check_contamination.py --data-dir data-merged --data-dir data-negatives
    python check_contamination.py --data-dir data-merged --report-out reports/contamination.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
BENCHMARK = REPO / "benchmark"
CASES_DIR = BENCHMARK / "cases"
CARRIERS_DIR = BENCHMARK / "carriers"

NGRAM = 8

_WORD_RE = re.compile(r"[^\w]+", re.UNICODE)


def normalise(text: str) -> list[str]:
    """Casefold, split on anything that is not a word character, drop empties.
    Deliberately aggressive: a leak that survives punctuation and casing
    changes is still a leak, and this is the form both sides are compared in."""
    return [w for w in _WORD_RE.sub(" ", text.casefold()).split() if w]


def shingles(words: list[str], n: int | None = None) -> set[str]:
    # `n=None` rather than `n=NGRAM`: a default argument is bound once, when
    # the function is defined, so `--ngram` would have been silently ignored
    # by every call that did not pass it explicitly.
    n = NGRAM if n is None else n
    if len(words) < n:
        return set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def load_cases() -> list[dict]:
    cases: list[dict] = []
    for path in sorted(CASES_DIR.glob("*.json")):
        for case in json.loads(path.read_text()):
            case = dict(case)
            case["_file"] = path.name
            cases.append(case)
    return cases


def load_carrier_documents() -> dict[str, str]:
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(CARRIERS_DIR.glob("*.txt"))}


def document_identity_keys(entry: dict) -> set[str]:
    """Everything that makes two manifest entries *the same document*.

    One implementation, used for both sides of the comparison. It was two
    near-identical copies until 2026-08-27, and the copies had drifted in the
    way duplicated code always does -- see the percent-encoding note below,
    a bug that had to be fixed twice to be fixed at all.

    `url::` is percent-decoded before it becomes a key. A fetcher that stores
    `.../wiki/Desalinizaci%C3%B3n` and a carrier manifest that stores
    `.../wiki/Desalinización` name one page, and comparing the raw strings
    says they are two. Seven of this repo's own carrier/negative pairs are
    exactly that, and every one of them was caught only because the separate
    `title::` key happens to unquote. A key that silently never fires is
    worse than an absent one: it makes the check look like it covers a
    dimension it does not.
    """
    import urllib.parse

    url = entry["url"]
    keys = {"id::" + entry["id"], "url::" + urllib.parse.unquote(url)}
    if "/wiki/" in url:
        title = urllib.parse.unquote(url.rsplit("/wiki/", 1)[-1]).replace("_", " ")
        keys.add(f"title::{entry['source']}::{title}")
    m = re.search(r"CELEX:(\w+)", url)
    if m:
        keys.add("celex::" + m.group(1))
    m = re.search(r"ecb\.(ar\d{4})~", url)
    if m:
        keys.add("ecb::" + m.group(1))
    return keys


def carrier_identity_keys() -> set[str]:
    """Everything that makes a document *the same document* as a carrier."""
    manifest = json.loads((CARRIERS_DIR / "manifest.json").read_text())
    return {k for c in manifest["carriers"] for k in document_identity_keys(c)}


def negative_identity_keys(manifest_path: Path) -> dict[str, set[str]]:
    """Per negative document id, the same identity keys, for intersection
    against the carriers'."""
    if not manifest_path.exists():
        return {}
    return {
        e["id"]: document_identity_keys(e)
        for e in json.loads(manifest_path.read_text()).get("documents", [])
    }


def build_index(cases: list[dict], carriers: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """(exact-normalised-text -> owner id, shingle -> owner id).

    Carrier documents go in as well as cases: the 462 carrier cases are cut
    from them, so a training row overlapping the *document* is contaminated
    even if it misses every cut case exactly.
    """
    exact: dict[str, str] = {}
    shingle_owner: dict[str, str] = {}
    for case in cases:
        words = normalise(case["text"])
        exact.setdefault(" ".join(words), case["id"])
        for sh in shingles(words):
            shingle_owner.setdefault(sh, case["id"])
    for carrier_id, text in carriers.items():
        words = normalise(text)
        for sh in shingles(words):
            shingle_owner.setdefault(sh, f"carrier:{carrier_id}")
    return exact, shingle_owner


def scan_file(
    path: Path,
    exact: dict[str, str],
    shingle_owner: dict[str, str],
    *,
    max_examples: int = 20,
) -> dict:
    hits_exact: list[tuple[int, str]] = []
    hits_shingle: list[tuple[int, str, int]] = []
    # Counted separately from the example lists, which are capped: a capped
    # example list read as a count silently reports "20" for any number of
    # collisions at or above 20, which is exactly the number a verdict should
    # never be allowed to understate.
    n_exact = 0
    n_shingle = 0
    owners = Counter()
    rows = 0
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            rows += 1
            text = json.loads(line)["text"]
            words = normalise(text)
            joined = " ".join(words)
            owner = exact.get(joined)
            if owner is not None:
                n_exact += 1
                owners[owner] += 1
                if len(hits_exact) < max_examples:
                    hits_exact.append((i, owner))
            overlap = shingles(words) & shingle_owner.keys()
            if overlap:
                n_shingle += 1
                first = next(iter(overlap))
                owner = shingle_owner[first]
                owners[owner] += 1
                if len(hits_shingle) < max_examples:
                    # length of the collision, never its content
                    hits_shingle.append((i, owner, len(overlap)))
    return {
        "file": path.name,
        "rows": rows,
        "exact_hits": n_exact,
        "shingle_hits": n_shingle,
        "exact_examples": hits_exact,
        "shingle_examples": hits_shingle,
        "owners": owners.most_common(20),
    }


def main() -> int:
    global NGRAM

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", action="append", default=None,
                    help="dataset directory containing train/validation/test.jsonl; repeatable")
    ap.add_argument("--negatives-manifest", default=str(HERE / "negatives" / "manifest.json"))
    ap.add_argument("--report-out", default=None)
    ap.add_argument("--ngram", type=int, default=NGRAM,
                    help="primary n-gram length; collisions here trigger escalation")
    ap.add_argument("--escalate-to", type=int, default=12,
                    help="n-gram length the verdict is actually decided at. A collision "
                         "surviving this length is copied text, not a stock phrase.")
    args = ap.parse_args()

    data_dirs = [Path(d) for d in (args.data_dir or ["data-merged"])]

    t0 = time.time()
    cases = load_cases()
    carriers = load_carrier_documents()
    print(f"benchmark: {len(cases)} cases from {len(sorted(CASES_DIR.glob('*.json')))} files, "
          f"{len(carriers)} carrier documents")

    # ---- identity check ---------------------------------------------------
    carrier_keys = carrier_identity_keys()
    neg_keys = negative_identity_keys(Path(args.negatives_manifest))
    id_clashes = {doc_id: sorted(keys & carrier_keys) for doc_id, keys in neg_keys.items() if keys & carrier_keys}
    print(f"negative source documents: {len(neg_keys)}")
    print(f"  document-identity clashes with a carrier (id / URL / title / CELEX / ECB report): "
          f"{len(id_clashes)}")

    # ---- text check -------------------------------------------------------
    NGRAM = args.ngram
    exact, shingle_owner = build_index(cases, carriers)
    print(f"index: {len(exact)} exact normalised case texts, {len(shingle_owner)} distinct {args.ngram}-grams")

    results = []
    for d in data_dirs:
        for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
            path = d / name
            if not path.exists():
                continue
            r = scan_file(path, exact, shingle_owner)
            r["dir"] = str(d)
            results.append(r)
            status = "CLEAN" if r["exact_hits"] == 0 and r["shingle_hits"] == 0 else "CONTAMINATED"
            print(f"  {d}/{name}: {r['rows']:,} rows -> exact {r['exact_hits']}, "
                  f"{args.ngram}-gram {r['shingle_hits']}  [{status}]")

    total_exact = sum(r["exact_hits"] for r in results)
    total_shingle = sum(r["shingle_hits"] for r in results)

    # Escalation. Only run when the primary scan found something, because
    # when it found nothing there is nothing for a longer n-gram to find --
    # every n-gram of length n+k contains an n-gram of length n.
    escalation: list[tuple[int, int]] = []
    total_escalated = 0
    if total_shingle:
        for n in range(args.ngram + 2, args.escalate_to + 1, 2):
            NGRAM = n
            exact_n, shingle_n = build_index(cases, carriers)
            found = 0
            for d in data_dirs:
                for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
                    path = d / name
                    if path.exists():
                        found += scan_file(path, exact_n, shingle_n)["shingle_hits"]
            escalation.append((n, found))
            print(f"  escalation at n={n}: {found} colliding rows")
            total_escalated = found
            if found == 0:
                break
        NGRAM = args.ngram

    clean = total_exact == 0 and total_escalated == 0 and not id_clashes
    elapsed = time.time() - t0
    print("")
    print("=" * 60)
    print(f"  cases checked: {len(cases)}   carrier documents checked: {len(carriers)}")
    print(f"  document-identity clashes: {len(id_clashes)}")
    print(f"  exact-text collisions: {total_exact}")
    print(f"  {args.ngram}-gram collisions: {total_shingle}"
          + ("  (stock-phrase coincidence: none survives escalation)"
             if total_shingle and total_escalated == 0 else ""))
    for n, found in escalation:
        print(f"  {n}-gram collisions: {found}")
    print(f"  VERDICT: {'CLEAN' if clean else 'CONTAMINATED'}  ({elapsed:.0f}s)")
    print("=" * 60)

    if args.report_out:
        out = Path(args.report_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Benchmark contamination check",
            "",
            f"Run {time.strftime('%Y-%m-%d')} by `training/check_contamination.py`. "
            f"{len(cases)} benchmark cases and {len(carriers)} carrier documents against "
            f"{', '.join(str(d) for d in data_dirs)}.",
            "",
            "Three identities are checked independently: document identity (carrier id, URL, "
            "Wikimedia title, CELEX id, ECB report slug), exact normalised text, and shared "
            f"normalised word {args.ngram}-grams. No matched text appears below, by design.",
            "",
            "| corpus | rows | exact-text collisions | " + f"{args.ngram}-gram collisions | verdict |",
            "|---|---|---|---|---|",
        ]
        for r in results:
            v = "clean" if r["exact_hits"] == 0 and r["shingle_hits"] == 0 else "**CONTAMINATED**"
            lines.append(f"| `{r['dir']}/{r['file']}` | {r['rows']:,} | {r['exact_hits']} | {r['shingle_hits']} | {v} |")
        lines += [
            "",
            f"Negative source documents checked for document identity: {len(neg_keys)}. "
            f"Clashes with a carrier: **{len(id_clashes)}**.",
            "",
            f"**Verdict: {'CLEAN' if clean else 'CONTAMINATED'}.** "
            f"{total_exact} exact-text collisions, {total_shingle} {args.ngram}-gram collisions, "
            f"{len(id_clashes)} document-identity clashes, over {len(cases)} cases. "
            f"Elapsed {elapsed:.0f}s.",
            "",
        ]
        if escalation:
            lines += [
                "## Escalation",
                "",
                f"The {args.ngram}-gram scan found {total_shingle} colliding rows, so the scan "
                "was re-run at longer n-grams. A stock phrase shared between two documents in "
                "the same language does not survive twelve words; a re-cut paragraph produces "
                "dozens of overlapping shingles and survives twenty. The verdict above is "
                "decided at the last row of this table, not at the first.",
                "",
                "| n | colliding rows |",
                "|---|---|",
                f"| {args.ngram} | {total_shingle} |",
            ]
            for n, found in escalation:
                lines.append(f"| {n} | {found} |")
            lines.append("")
        if id_clashes:
            lines.append("Clashing document ids: " + ", ".join(sorted(id_clashes)) + ".")
            lines.append("")
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"report written to {out}")

    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
