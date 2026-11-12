"""Measures `PERSON` recall on the two shapes the corpus underrepresents:
names in tabular and list positions, and names in bylines.

The counterpart to `measure_hallucination.py`. That one measures whether the
model invents names where there are none; this one measures whether it finds
names that are there, in the two shapes
`training/reports/person-misses.md` traced all three remaining benchmark
misses to. **Both have to be read together**, because the two corpora this
run adds pull in opposite directions -- the negatives make the model less
willing to emit a `PERSON` span, the augmentation makes it more willing --
and a change that moves one number the right way while moving the other the
wrong way is a trade, not a win.

Neither metric touches `benchmark/cases/`. This one runs over the held-out
split of `data-positives/`, whose **templates** are held out, not merely
whose rows: no sentence pattern in the held-out split appears in training at
all (see `build_positives.HOLDOUT_TEMPLATES`). So what it measures is whether
the model learned the shape, not whether it memorised the phrasing.

Scoring
-------
Two numbers per group, both computed against the constructed gold spans:

- **full coverage** -- every non-whitespace character of the gold span is
  covered by the union of detected `PERSON` spans. This is the rule
  `crates/eval/src/score.rs` implements and the one the benchmark's
  `PERSON` 108/111 is computed with, so the two are comparable in kind.
- **partial coverage** -- some but not all of it. This is the
  `it-wp-rotatoria-dn` failure exactly: `Roversi` found, `Lorenzo` dropped.
  A model that improves partial coverage while leaving full coverage flat
  has not fixed that miss, and reporting only one number would hide it.

Decoding is argmax over the label set at the token level -- the model, not
the deployed window/reconcile/calibration path. Same choice, and same
caveat, as `measure_hallucination.py`: the absolute number is not the
benchmark's number; what is comparable is one checkpoint against another.

Privacy
-------
Counts only. No detected value, no gold value, no row text is printed or
written -- and these rows are synthetic, which is a reason to keep the habit
rather than a reason to drop it.
"""

from __future__ import annotations

import argparse
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


def person_char_mask(text_len: int, tags: list[str], offsets: list[tuple[int, int]]) -> bytearray:
    """A byte per character, 1 where some `PERSON` tag covers it. A mask
    rather than a span list because the scoring rule is union coverage: two
    adjacent spans that together cover the gold value count as covering it,
    which is exactly the BIO-discontinuity case
    `reports/spurious-address-spans.md` describes for `STREET_ADDRESS`."""
    mask = bytearray(text_len)
    for tag, (start, end) in zip(tags, offsets):
        if tag.endswith("PERSON") and tag != "O" and end > start:
            for i in range(start, min(end, text_len)):
                mask[i] = 1
    return mask


@torch.no_grad()
def measure(checkpoint: str, rows: list[dict], *, device: str, batch_size: int = 32,
            max_length: int = 256, stride: int = 32) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    model = BertForTokenClassification.from_pretrained(checkpoint)
    model.to(device)
    model.eval()
    if list(model.config.id2label.values()) != LABELS:
        raise SystemExit(f"{checkpoint} does not carry this project's label set in this order")

    # One row can window into several inputs; the coverage mask is per row and
    # is filled from every window, so a name split across a window boundary is
    # still counted covered if the two windows between them cover it.
    masks = [bytearray(len(r["text"])) for r in rows]
    windows: list[tuple[int, list[int], list[tuple[int, int]]]] = []
    for i, row in enumerate(rows):
        for w in tokenize_and_align(row["text"], row["entities"], tokenizer, max_length, stride):
            windows.append((i, w.input_ids, w.offset_mapping))

    pad = tokenizer.pad_token_id
    order = sorted(range(len(windows)), key=lambda k: len(windows[k][1]))
    t0 = time.time()
    for start in range(0, len(order), batch_size):
        chunk = [windows[k] for k in order[start : start + batch_size]]
        width = max(len(ids) for _, ids, _ in chunk)
        input_ids = torch.full((len(chunk), width), pad, dtype=torch.long)
        attn = torch.zeros((len(chunk), width), dtype=torch.long)
        for j, (_, ids, _) in enumerate(chunk):
            input_ids[j, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn[j, : len(ids)] = 1
        logits = model(input_ids=input_ids.to(device), attention_mask=attn.to(device)).logits
        pred = logits.argmax(-1)
        for j, (row_idx, ids, offsets) in enumerate(chunk):
            tags = [ID2LABEL[int(t)] for t in pred[j, : len(ids)].tolist()]
            sub = person_char_mask(len(rows[row_idx]["text"]), tags, offsets)
            row_mask = masks[row_idx]
            for k, v in enumerate(sub):
                if v:
                    row_mask[k] = 1
    elapsed = time.time() - t0

    full: Counter = Counter()
    partial: Counter = Counter()
    total: Counter = Counter()
    keys = defaultdict(list)
    for i, row in enumerate(rows):
        text, mask = row["text"], masks[i]
        for ent in row["entities"]:
            if ent["label"] != "PERSON":
                continue
            positions = [p for p in range(ent["start"], ent["end"]) if not text[p].isspace()]
            covered = sum(1 for p in positions if mask[p])
            for key in ("all", row["language"], row["source"], row.get("template", "?")):
                total[key] += 1
                if covered == len(positions):
                    full[key] += 1
                elif covered:
                    partial[key] += 1
                keys[key].append(None)

    def block(names: list[str]) -> dict:
        return {
            n: {
                "gold": total[n],
                "full": full[n],
                "partial": partial[n],
                "recall": round(full[n] / total[n], 4) if total[n] else None,
                "partial_rate": round(partial[n] / total[n], 4) if total[n] else None,
            }
            for n in sorted(names)
        }

    langs = sorted({r["language"] for r in rows})
    sources = sorted({r["source"] for r in rows})
    templates = sorted({r.get("template", "?") for r in rows})
    return {
        "checkpoint": str(checkpoint),
        "layers": model.config.num_hidden_layers,
        "rows": len(rows),
        "gold_person_spans": total["all"],
        "recall": round(full["all"] / total["all"], 4) if total["all"] else None,
        "partial_rate": round(partial["all"] / total["all"], 4) if total["all"] else None,
        "by_language": block(langs),
        "by_family": block(sources),
        "by_template": block(templates),
        "elapsed_s": round(elapsed, 1),
        "device": device,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", action="append", required=True)
    ap.add_argument("--positives-dir", default=str(HERE / "data-positives"))
    ap.add_argument("--split", default="test", choices=["validation", "test", "both"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = pick_device(args.device)
    base = Path(args.positives_dir)
    names = ["validation.jsonl", "test.jsonl"] if args.split == "both" else [f"{args.split}.jsonl"]
    rows: list[dict] = []
    for name in names:
        with (base / name).open(encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
                if args.limit and len(rows) >= args.limit:
                    break
    if not rows:
        raise SystemExit(f"no rows in {base} for split {args.split}")
    print(f"{len(rows):,} held-out augmentation rows from {base} ({args.split}), device {device}")

    results = []
    for ckpt in args.checkpoint:
        r = measure(ckpt, rows, device=device, batch_size=args.batch_size)
        results.append(r)
        print("")
        print(f"{ckpt}  ({r['layers']} layers)")
        print(f"  PERSON recall (full coverage): {r['recall']} over {r['gold_person_spans']:,} gold spans")
        print(f"  partial coverage (found part of the name, dropped the rest): {r['partial_rate']}")
        print("  by family:   " + ", ".join(f"{k}={v['recall']}" for k, v in r["by_family"].items()))
        print("  by language: " + ", ".join(f"{k}={v['recall']}" for k, v in r["by_language"].items()))
        print("  by template: " + ", ".join(f"{k}={v['recall']}" for k, v in r["by_template"].items()))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
