"""Measures how often a checkpoint invents personal data in prose that has
none.

This is the metric the negatives exist to move, and it is deliberately not
the benchmark. `benchmark/cases/` is scored by the Rust gate and must never
be trained against; this measures the same *behaviour* on the held-out split
of `data-negatives/`, which is out-of-domain prose the run never saw and
which no benchmark case is cut from. So it can be run as often as wanted,
including between epochs, without spending or spoiling the benchmark.

What it reports
---------------
Per checkpoint, over N held-out negative rows whose gold annotation is "no
entities at all":

- **rows with any span** -- the share of paragraphs the model hallucinates
  into. This is the headline: it is the user-visible failure (a document
  comes back with a name in it that was never there).
- **spans per 1,000 rows**, overall and per type. `PERSON` and
  `STREET_ADDRESS` are the two the run is trying to fix.
- the same, split by **language** and by **register**, because
  `reports/spurious-address-spans.md` found the hallucination concentrated in
  annual-report and encyclopedia prose and it should be watched there
  specifically.
- a **confidence histogram**, which is what makes the "raise the threshold
  instead" argument checkable rather than assertable. It was checked once
  already and rejected: at 0.99, 44 of 121 spurious spans survived while 35
  of 251 true positives were lost.

Decoding
--------
Argmax over the label set, then BIO spans, at the token level -- **not** the
service's window/reconcile/calibration path. That is the right choice for a
training-side metric: it measures the model, not the deployment, and it has
no thresholds to tune, so two checkpoints measured with it differ only by the
model. The absolute number is therefore not comparable to the benchmark's
spurious-span count, and the report says so wherever it is quoted; what is
comparable is one checkpoint against another.

Provenance
----------
`reports/student-v2-run.md` measured 8.9 against a brief that named 22.8 as
the shipped model's figure, and the two turned out to be from different
held-out splits (a corpus expansion had landed between them) -- the same
model re-measures at 49.0 on the split 8.9 actually shares. Nothing in the
tool would have caught that; it was caught by a human re-deriving the
shipped number by hand. So every result this script produces carries a
`corpus_digest` -- a sha256 over the exact rows measured, order included --
and `--compare` refuses to line up two results whose digests disagree
instead of printing a table that looks comparable and is not. A number
without a digest next to it is exactly how a real improvement gets read as
a regression.

Privacy
-------
Prints counts, offsets-free. No detected value, no token text, no row text --
not to stdout, not to the JSON, not to the report. The digest is a sha256
over row content; it is opaque by construction and never printed alongside
anything that would let it be reversed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer, BertForTokenClassification

from align import ID2LABEL, LABELS, tokenize_and_align
from train import TOKENIZER_NAME, pick_device

HERE = Path(__file__).resolve().parent


def corpus_digest(rows: list[dict]) -> str:
    """sha256 over the exact rows measured, in the order they were read.

    Deliberately over the rows actually used rather than the source files'
    bytes: `--limit` can measure a prefix of a file, and the digest exists
    to fingerprint what was measured, not what happens to sit on disk next
    to it. Two results with the same digest read the identical rows in the
    identical order; two that don't are not the same corpus, whatever their
    split name or directory path say.
    """
    h = hashlib.sha256()
    for row in rows:
        h.update(json.dumps(row, sort_keys=True, ensure_ascii=True).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def spans_from_tags(tags: list[str], scores: list[float]) -> list[tuple[str, float]]:
    """(type, mean score) per BIO span. Offsets are computed but not returned:
    nothing downstream needs them and returning them invites printing them."""
    out: list[tuple[str, float]] = []
    cur_type: str | None = None
    cur_scores: list[float] = []
    for tag, score in zip(tags, scores):
        if tag.startswith("B-"):
            if cur_type is not None:
                out.append((cur_type, sum(cur_scores) / len(cur_scores)))
            cur_type, cur_scores = tag[2:], [score]
        elif tag.startswith("I-") and cur_type == tag[2:]:
            cur_scores.append(score)
        else:
            if cur_type is not None:
                out.append((cur_type, sum(cur_scores) / len(cur_scores)))
            cur_type, cur_scores = None, []
    if cur_type is not None:
        out.append((cur_type, sum(cur_scores) / len(cur_scores)))
    return out


@torch.no_grad()
def measure(
    checkpoint: str,
    rows: list[dict],
    *,
    device: str,
    digest: str,
    batch_size: int = 32,
    max_length: int = 256,
    stride: int = 32,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    model = BertForTokenClassification.from_pretrained(checkpoint)
    model.to(device)
    model.eval()

    if list(model.config.id2label.values()) != LABELS:
        raise SystemExit(
            f"{checkpoint} does not carry this project's label set in this order -- "
            "its spans are not this taxonomy's spans."
        )

    # One row can window into several inputs; a span anywhere in any of a
    # row's windows counts as that row hallucinating.
    windows: list[tuple[int, list[int]]] = []
    for i, row in enumerate(rows):
        for w in tokenize_and_align(row["text"], [], tokenizer, max_length, stride):
            windows.append((i, w.input_ids))

    row_hits: set[int] = set()
    row_hits_by_type: dict[str, set[int]] = defaultdict(set)
    span_counts: Counter = Counter()
    per_lang_rows: Counter = Counter()
    per_lang_spans: Counter = Counter()
    per_reg_rows: Counter = Counter()
    per_reg_spans: Counter = Counter()
    score_hist: Counter = Counter()
    pad = tokenizer.pad_token_id

    order = sorted(range(len(windows)), key=lambda k: len(windows[k][1]))
    t0 = time.time()
    for start in range(0, len(order), batch_size):
        chunk = [windows[k] for k in order[start : start + batch_size]]
        width = max(len(ids) for _, ids in chunk)
        input_ids = torch.full((len(chunk), width), pad, dtype=torch.long)
        attn = torch.zeros((len(chunk), width), dtype=torch.long)
        for j, (_, ids) in enumerate(chunk):
            input_ids[j, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn[j, : len(ids)] = 1
        logits = model(input_ids=input_ids.to(device), attention_mask=attn.to(device)).logits
        probs = logits.softmax(-1)
        best = probs.max(-1)
        for j, (row_idx, ids) in enumerate(chunk):
            n = len(ids)
            tags = [ID2LABEL[int(t)] for t in best.indices[j, :n].tolist()]
            scores = best.values[j, :n].tolist()
            for span_type, score in spans_from_tags(tags, scores):
                span_counts[span_type] += 1
                row_hits.add(row_idx)
                row_hits_by_type[span_type].add(row_idx)
                lang = rows[row_idx].get("language", "?")
                reg = rows[row_idx].get("source", "?")
                per_lang_spans[lang] += 1
                per_reg_spans[reg] += 1
                score_hist[min(int(score * 20), 19)] += 1
    elapsed = time.time() - t0

    for i, row in enumerate(rows):
        per_lang_rows[row.get("language", "?")] += 1
        per_reg_rows[row.get("source", "?")] += 1

    n = len(rows)
    return {
        "checkpoint": str(checkpoint),
        "corpus_digest": digest,
        "layers": model.config.num_hidden_layers,
        "parameters": sum(p.numel() for p in model.parameters()),
        "rows": n,
        "windows": len(windows),
        "rows_with_any_span": len(row_hits),
        "rows_with_any_span_pct": 100.0 * len(row_hits) / max(n, 1),
        "rows_with_span_by_type": {t: len(v) for t, v in sorted(row_hits_by_type.items())},
        "spans": sum(span_counts.values()),
        "spans_per_1000_rows": 1000.0 * sum(span_counts.values()) / max(n, 1),
        "spans_by_type": dict(span_counts.most_common()),
        "spans_per_1000_by_language": {
            k: round(1000.0 * per_lang_spans[k] / max(per_lang_rows[k], 1), 1)
            for k in sorted(per_lang_rows)
        },
        "spans_per_1000_by_register": {
            k: round(1000.0 * per_reg_spans[k] / max(per_reg_rows[k], 1), 1)
            for k in sorted(per_reg_rows)
        },
        "score_histogram_20_buckets": [score_hist.get(b, 0) for b in range(20)],
        "elapsed_s": round(elapsed, 1),
        "device": device,
    }


def load_rows(paths: list[Path], limit: int | None) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    return rows
    return rows


def load_reports(paths: list[Path]) -> list[dict]:
    """Every per-checkpoint result across one or more `--out` files, each
    tagged with the file it came from so a refusal can name where a
    mismatched digest lives."""
    entries: list[dict] = []
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            data = [data]
        for r in data:
            r = dict(r)
            r["_report_file"] = str(path)
            entries.append(r)
    return entries


def compare_reports(entries: list[dict]) -> str:
    """The table `--compare` prints -- but only after every entry is
    confirmed to share one `corpus_digest`. Raises `SystemExit` rather than
    returning anything when they don't, or when an entry predates this field
    and so cannot be trusted to be the same corpus as the others: a
    comparison across different held-out splits is exactly how
    `reports/student-v2-run.md`'s 8.9-vs-22.8 mixup happened, and printing a
    table that looks comparable is worse than printing nothing.
    """
    if not entries:
        raise SystemExit("--compare: no results found in the given files")

    missing = [e["checkpoint"] for e in entries if not e.get("corpus_digest")]
    if missing:
        raise SystemExit(
            "refusing to compare: these results predate corpus_digest and cannot be "
            "trusted to share a corpus with the others -- re-measure them:\n"
            + "\n".join(f"  {c}" for c in missing)
        )

    digests = {e["corpus_digest"] for e in entries}
    if len(digests) > 1:
        lines = "\n".join(
            f"  {e['checkpoint']}: digest {e['corpus_digest'][:12]} "
            f"(rows={e.get('rows', '?')}) from {e['_report_file']}"
            for e in entries
        )
        raise SystemExit(
            "refusing to compare: these results were measured on "
            f"{len(digests)} different corpora, not one, so their headline numbers are "
            f"not comparable however similar they look:\n{lines}\n"
            "Re-measure them on the same split, or compare them explicitly as "
            "different-corpus numbers rather than through this table."
        )

    digest = next(iter(digests))
    lines = [f"{len(entries)} result(s), all on corpus digest {digest[:12]}:", ""]
    header = f"{'checkpoint':<40} {'rows':>8} {'spans/1000':>12} {'rows w/ any span %':>20}"
    lines.append(header)
    for e in entries:
        lines.append(
            f"{e['checkpoint']:<40} {e.get('rows', 0):>8,} "
            f"{e.get('spans_per_1000_rows', 0):>12.1f} "
            f"{e.get('rows_with_any_span_pct', 0):>19.2f}%"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", action="append", default=None,
                    help="checkpoint directory; repeatable to compare several in one table. "
                         "Required unless --compare is given.")
    ap.add_argument("--negatives-dir", default=str(HERE / "data-negatives"))
    ap.add_argument("--split", default="test", choices=["validation", "test", "both"],
                    help="which held-out negative split to measure; never `train`")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default=None, help="JSON results path")
    ap.add_argument("--compare", nargs="+", default=None, metavar="REPORT.json",
                     help="skip measuring; instead load one or more prior --out files and "
                          "print a comparison table. Refuses (exit nonzero, no table) if the "
                          "results in them do not all share one corpus_digest.")
    args = ap.parse_args()

    if args.compare:
        print(compare_reports(load_reports([Path(p) for p in args.compare])))
        return 0

    if not args.checkpoint:
        ap.error("--checkpoint is required unless --compare is given")

    device = pick_device(args.device)
    neg = Path(args.negatives_dir)
    names = ["validation.jsonl", "test.jsonl"] if args.split == "both" else [f"{args.split}.jsonl"]
    rows = load_rows([neg / n for n in names], args.limit)
    if not rows:
        raise SystemExit(f"no rows in {neg} for split {args.split}")
    if any(r.get("entities") for r in rows):
        raise SystemExit(
            "some rows in the negative split carry entities -- this metric assumes every "
            "span found is a hallucination, and that assumption has just failed."
        )
    digest = corpus_digest(rows)
    print(f"{len(rows):,} held-out negative rows from {neg} ({args.split}), "
          f"corpus digest {digest[:12]}, device {device}")

    results = []
    for ckpt in args.checkpoint:
        r = measure(ckpt, rows, device=device, digest=digest, batch_size=args.batch_size)
        results.append(r)
        print("")
        print(f"{ckpt}  ({r['layers']} layers, {r['parameters']:,} params)")
        print(f"  rows with any span: {r['rows_with_any_span']:,} / {r['rows']:,} "
              f"({r['rows_with_any_span_pct']:.2f}%)")
        print(f"  spans: {r['spans']:,} ({r['spans_per_1000_rows']:.1f} per 1,000 rows)")
        for t, n in r["spans_by_type"].items():
            print(f"    {t}: {n}")
        print("  per language (spans/1,000 rows): "
              + ", ".join(f"{k}={v}" for k, v in r["spans_per_1000_by_language"].items()))
        print("  per register (spans/1,000 rows): "
              + ", ".join(f"{k}={v}" for k, v in r["spans_per_1000_by_register"].items()))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
