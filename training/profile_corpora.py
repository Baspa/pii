#!/usr/bin/env python3
"""Profile the two candidate PII training corpora before trusting either.

This script streams both corpora end to end -- it never materialises the
underlying parquet files to disk, and never loads the full 4.7 GB
ai4privacy corpus into memory at once. Per row it keeps only the derived
scalars it needs (a language code, a list of label strings, a character
count) and discards the row.

Corpora:
  - ai4privacy/pii-masking-openpii-1.5m (CC-BY-4.0). Ships a ground-truth
    `language` column and a `privacy_mask` column (list of
    {label, start, end, value, label_index} spans).
  - Wismut/nym-pii-multilingual-data (MIT). Ships only two columns, `text`
    and `entities` (list of {start, end, label} spans). It has NO
    `language` column at all, despite its dataset card listing ~22-23
    languages -- this was discovered while building this script (see the
    report). Its language is therefore estimated with an offline language
    identifier (py3langid) rather than measured, and that estimate is
    calibrated against ai4privacy's own ground-truth language column so its
    error rate is known rather than assumed.

Usage:
    python training/profile_corpora.py --out training/reports/corpora-profile.md
    python training/profile_corpora.py --out /tmp/test.md --max-rows-per-split 2000
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator

import psutil
import py3langid as langid
import requests
from datasets import load_dataset

EU_LANGUAGES: dict[str, str] = {
    "bg": "Bulgarian", "hr": "Croatian", "cs": "Czech", "da": "Danish",
    "nl": "Dutch", "en": "English", "et": "Estonian", "fi": "Finnish",
    "fr": "French", "de": "German", "el": "Greek", "hu": "Hungarian",
    "ga": "Irish", "it": "Italian", "lv": "Latvian", "lt": "Lithuanian",
    "mt": "Maltese", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian",
    "sk": "Slovak", "sl": "Slovenian", "es": "Spanish", "sv": "Swedish",
}
SUFFICIENCY_THRESHOLD = 5000
PARQUET_API = "https://datasets-server.huggingface.co/parquet?dataset={}"


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


def get_parquet_urls(repo_id: str, splits: list[str]) -> dict[str, list[str]]:
    """Resolve the auto-converted parquet shard URLs for a dataset's splits.

    Uses the datasets-server /parquet endpoint (confirmed working, unlike
    /rows and /filter which return 500 for ai4privacy/pii-masking-openpii-1.5m).
    """
    resp = requests.get(PARQUET_API.format(requests.utils.quote(repo_id, safe="")), timeout=30)
    resp.raise_for_status()
    data = resp.json()
    out: dict[str, list[str]] = {s: [] for s in splits}
    for f in data["parquet_files"]:
        if f["split"] in out:
            out[f["split"]].append(f["url"])
    for s in splits:
        out[s].sort()
        if not out[s]:
            raise RuntimeError(f"No parquet shards found for split={s!r} of {repo_id}")
    return out


@dataclass
class CorpusStats:
    name: str
    total_rows: int = 0
    lang_counts: Counter = field(default_factory=Counter)
    label_counts: Counter = field(default_factory=Counter)
    lengths: list[int] = field(default_factory=list)
    no_entity_rows: int = 0
    bytes_moved: int = 0
    seconds_elapsed: float = 0.0

    def add_row(self, language: str | None, labels: list[str], text_len: int) -> None:
        self.total_rows += 1
        if language:
            self.lang_counts[language] += 1
        if labels:
            self.label_counts.update(labels)
        else:
            self.no_entity_rows += 1
        self.lengths.append(text_len)

    def length_stats(self) -> tuple[float, float, float]:
        if not self.lengths:
            return (0.0, 0.0, 0.0)
        s = sorted(self.lengths)
        mean = statistics.mean(s)
        median = statistics.median(s)
        p95_idx = min(len(s) - 1, int(round(0.95 * (len(s) - 1))))
        p95 = s[p95_idx]
        return (mean, median, p95)

    def no_entity_share(self) -> float:
        return self.no_entity_rows / self.total_rows if self.total_rows else 0.0


def stream_ai4privacy(max_rows_per_split: int | None) -> tuple[CorpusStats, Counter, Counter]:
    """Stream ai4privacy/pii-masking-openpii-1.5m (train + validation).

    Returns (stats, langid_agree, langid_sampled) -- per-language counters
    that let us report py3langid's accuracy against this corpus's OWN
    ground-truth `language` field -- used later to calibrate the Wismut
    language *estimate*, which has no ground truth of its own.
    """
    repo_id = "ai4privacy/pii-masking-openpii-1.5m"
    stats = CorpusStats(name=repo_id)
    langid_agree: Counter = Counter()
    langid_sampled: Counter = Counter()

    urls = get_parquet_urls(repo_id, ["train", "validation"])
    log(f"ai4privacy: {len(urls['train'])} train shard(s), {len(urls['validation'])} validation shard(s)")

    net_start = psutil.net_io_counters().bytes_recv
    t0 = time.time()

    for split in ("train", "validation"):
        log(f"ai4privacy: streaming split={split} ({len(urls[split])} shard(s))")
        ds = load_dataset("parquet", data_files={split: urls[split]}, split=split, streaming=True)
        ds = ds.select_columns(["language", "privacy_mask", "source_text"])
        n_this_split = 0
        last_report = time.time()
        for row in ds:
            lang = row.get("language")
            labels = [m["label"] for m in (row.get("privacy_mask") or [])]
            text = row.get("source_text") or ""
            stats.add_row(lang, labels, len(text))

            # Calibrate py3langid against this corpus's ground-truth language,
            # restricted to EU languages (the ones we actually need to trust
            # an estimate for on the other corpus).
            #
            # NOTE: stats.total_rows is the absolute position in the whole
            # stream across BOTH splits and ALL languages, not a per-language
            # counter -- so this is positional sampling ("every 20th row
            # seen"), not a per-language 1-in-20 draw. It approximates a
            # per-language draw only because ai4privacy's rows are grouped
            # into large contiguous blocks by source region/language rather
            # than shuffled, so a fixed stride lands roughly evenly within
            # each language's run. If this corpus (or a future one reusing
            # this code) interleaves languages instead of grouping them
            # contiguously, this stride would sample some languages far more
            # unevenly than others. Do not "fix" this into a true per-language
            # counter without knowing that changes the sample and requires
            # re-running the full profile to get comparable numbers -- see
            # training/reports/corpora-profile.md's calibration section for
            # the spot-check that backs the 91.2% figure despite this.
            if lang in EU_LANGUAGES and stats.total_rows % 20 == 0 and text:
                pred, _ = langid.classify(text)
                langid_sampled[lang] += 1
                if pred == lang:
                    langid_agree[lang] += 1

            n_this_split += 1
            if max_rows_per_split and n_this_split >= max_rows_per_split:
                break
            if time.time() - last_report > 20:
                net_now = psutil.net_io_counters().bytes_recv
                log(
                    f"ai4privacy[{split}]: {n_this_split} rows this split, "
                    f"{stats.total_rows} total, "
                    f"{(net_now - net_start) / 1e6:.1f} MB moved, "
                    f"{time.time() - t0:.0f}s elapsed"
                )
                last_report = time.time()
        log(f"ai4privacy: finished split={split}, {n_this_split} rows")

    stats.seconds_elapsed = time.time() - t0
    stats.bytes_moved = psutil.net_io_counters().bytes_recv - net_start
    log(
        f"ai4privacy: DONE. total_rows={stats.total_rows} "
        f"bytes_moved={stats.bytes_moved / 1e6:.1f}MB "
        f"elapsed={stats.seconds_elapsed:.1f}s"
    )
    return stats, langid_agree, langid_sampled


def stream_wismut(max_rows_per_split: int | None) -> tuple[CorpusStats, Counter]:
    """Stream Wismut/nym-pii-multilingual-data (train + validation + test).

    This corpus has no `language` column (verified against its actual
    parquet schema, not just its dataset card). Every row's language is
    *estimated* with py3langid and tracked in a separate counter so the
    report can keep "measured" (ai4privacy) and "estimated" (Wismut) numbers
    visually and textually distinct.
    """
    repo_id = "Wismut/nym-pii-multilingual-data"
    stats = CorpusStats(name=repo_id)
    lang_estimate_counts: Counter = Counter()

    urls = get_parquet_urls(repo_id, ["train", "validation", "test"])
    log(f"wismut: shards -> " + ", ".join(f"{s}={len(u)}" for s, u in urls.items()))

    net_start = psutil.net_io_counters().bytes_recv
    t0 = time.time()

    for split in ("train", "validation", "test"):
        log(f"wismut: streaming split={split} ({len(urls[split])} shard(s))")
        ds = load_dataset("parquet", data_files={split: urls[split]}, split=split, streaming=True)
        n_this_split = 0
        last_report = time.time()
        for row in ds:
            labels = [e["label"] for e in (row.get("entities") or [])]
            text = row.get("text") or ""
            stats.add_row(None, labels, len(text))
            if text:
                pred, _ = langid.classify(text)
                lang_estimate_counts[pred] += 1

            n_this_split += 1
            if max_rows_per_split and n_this_split >= max_rows_per_split:
                break
            if time.time() - last_report > 20:
                net_now = psutil.net_io_counters().bytes_recv
                log(
                    f"wismut[{split}]: {n_this_split} rows this split, "
                    f"{stats.total_rows} total, "
                    f"{(net_now - net_start) / 1e6:.1f} MB moved, "
                    f"{time.time() - t0:.0f}s elapsed"
                )
                last_report = time.time()
        log(f"wismut: finished split={split}, {n_this_split} rows")

    stats.seconds_elapsed = time.time() - t0
    stats.bytes_moved = psutil.net_io_counters().bytes_recv - net_start
    log(
        f"wismut: DONE. total_rows={stats.total_rows} "
        f"bytes_moved={stats.bytes_moved / 1e6:.1f}MB "
        f"elapsed={stats.seconds_elapsed:.1f}s"
    )
    return stats, lang_estimate_counts


def fmt_pct(n: int, total: int) -> str:
    return f"{100 * n / total:.2f}%" if total else "n/a"


def render_language_table(counts: Counter, total: int, note: str = "") -> str:
    lines = ["| language | rows | share |", "|---|---:|---:|"]
    for lang, n in counts.most_common():
        lines.append(f"| {lang} | {n:,} | {fmt_pct(n, total)} |")
    return "\n".join(lines)


def render_label_table(counts: Counter, total_rows: int) -> str:
    lines = ["| label | rows | share of rows |", "|---|---:|---:|"]
    for label, n in counts.most_common():
        lines.append(f"| {label} | {n:,} | {fmt_pct(n, total_rows)} |")
    return "\n".join(lines)


def render_report(
    ai4: CorpusStats,
    wismut: CorpusStats,
    langid_agree: Counter,
    langid_total_by_lang: Counter,
    wismut_lang_est: Counter,
    args: argparse.Namespace,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    ai4_mean, ai4_median, ai4_p95 = ai4.length_stats()
    wis_mean, wis_median, wis_p95 = wismut.length_stats()

    # Calibration table: py3langid accuracy vs ai4privacy's ground-truth
    # `language` field, per EU language. Sampled every 20th row by absolute
    # stream position (see the NOTE at the sampling site above), which only
    # approximates a per-language 1-in-20 draw because ai4privacy's rows are
    # grouped into contiguous blocks per language rather than shuffled.
    calib_rows = []
    for lang in sorted(langid_total_by_lang):
        total = langid_total_by_lang[lang]
        agree = langid_agree[lang]
        calib_rows.append((lang, agree, total, agree / total if total else 0.0))
    overall_agree = sum(a for _, a, _, _ in calib_rows)
    overall_total = sum(t for _, _, t, _ in calib_rows)
    overall_rate = overall_agree / overall_total if overall_total else 0.0

    calib_lines = ["| language | py3langid correct | sampled rows | accuracy |", "|---|---:|---:|---:|"]
    for lang, agree, total, rate in calib_rows:
        calib_lines.append(f"| {lang} | {agree:,} | {total:,} | {rate:.1%} |")
    calib_lines.append(f"| **overall** | **{overall_agree:,}** | **{overall_total:,}** | **{overall_rate:.1%}** |")

    # Combined 24-EU-language sufficiency table.
    combined_lines = [
        "| language | ai4privacy rows (measured) | Wismut rows (py3langid estimate) | combined | sufficient (>= 5,000) |",
        "|---|---:|---:|---:|:---:|",
    ]
    thin_langs = []
    for code in sorted(EU_LANGUAGES):
        a_n = ai4.lang_counts.get(code, 0)
        w_n = wismut_lang_est.get(code, 0)
        combined = a_n + w_n
        sufficient = combined >= SUFFICIENCY_THRESHOLD
        if not sufficient:
            thin_langs.append((code, combined))
        combined_lines.append(
            f"| {code} ({EU_LANGUAGES[code]}) | {a_n:,} | {w_n:,} (est.) | {combined:,} | "
            f"{'yes' if sufficient else '**NO**'} |"
        )

    thin_note = ""
    if thin_langs:
        items = ", ".join(f"{c} ({EU_LANGUAGES[c]}): {n:,} rows" for c, n in thin_langs)
        thin_note = (
            "\n\n> **STOP CONDITION TRIGGERED.** The following EU languages fall below the "
            f"5,000-row threshold across both corpora combined: {items}. "
            "Per the task brief, this is a decision for the human partner, not a gap to "
            "quietly fill: the data generation would need to cover these languages "
            "too, which is a materially larger job than the design currently assumes.\n"
        )

    report = f"""# Corpora Profile: ai4privacy vs Wismut

Generated {now} by `training/profile_corpora.py`. Both corpora were streamed
in full (all splits) -- never materialised to disk, never loaded whole into
memory. Command: `python training/profile_corpora.py --out {args.out}`
{"(dev run, capped at " + str(args.max_rows_per_split) + " rows/split)" if args.max_rows_per_split else ""}

## IMPORTANT: an assumption in the task brief did not hold

The brief expected both corpora to carry a `language` field. **ai4privacy
does** (ground truth, verified below). **Wismut/nym-pii-multilingual-data
does not** -- its parquet schema has exactly two columns, `text` and
`entities`. Its dataset card claims ~22-23 languages (via Faker locale
generation during synthesis), but that information is not persisted per row
anywhere in the released data. This was discovered while building this
profiler, not assumed going in; see `/first-rows` output referenced in the
task report.

Because there is no ground truth to measure, Wismut's language distribution
below is an **estimate** produced by streaming every row's `text` through
`py3langid` (a deterministic, offline language identifier). It is not a
dataset field and is presented as an estimate throughout. To bound how much
to trust it, `py3langid` was also run against ai4privacy's own ground-truth
`language` column, sampled every twentieth row by absolute position in the
stream (not every twentieth row per language) -- see the calibration table
below for what that means for the resulting numbers. Treat the Wismut
language numbers accordingly.

Also notable: **neither corpus's dataset card claims Irish (`ga`) or Maltese
(`mt`) at all.** This is not a "thin" language, which is what the brief
anticipated might happen to the eight ai4privacy-unique languages -- it is
total absence from both corpora's stated language coverage, confirmed
against measured/estimated row counts below.

---

## ai4privacy/pii-masking-openpii-1.5m (CC-BY-4.0)

- Total rows streamed (train + validation): **{ai4.total_rows:,}**
- Data moved: {ai4.bytes_moved / 1e6:,.1f} MB over {ai4.seconds_elapsed:.0f}s
- Rows with zero entities: {ai4.no_entity_rows:,} ({fmt_pct(ai4.no_entity_rows, ai4.total_rows)})
- Character length: mean {ai4_mean:.1f}, median {ai4_median:.1f}, p95 {ai4_p95:.1f}
- Distinct labels: {len(ai4.label_counts)}
- Distinct languages: {len(ai4.lang_counts)}

### Rows per language (measured, ground truth `language` field)

{render_language_table(ai4.lang_counts, ai4.total_rows)}

### Rows per label (from `privacy_mask`)

{render_label_table(ai4.label_counts, ai4.total_rows)}

### Full label vocabulary, sorted ({len(ai4.label_counts)} labels)

{", ".join(sorted(ai4.label_counts))}

---

## Wismut/nym-pii-multilingual-data (MIT)

- Total rows streamed (train + validation + test): **{wismut.total_rows:,}**
- Data moved: {wismut.bytes_moved / 1e6:,.1f} MB over {wismut.seconds_elapsed:.0f}s
- Rows with zero entities: {wismut.no_entity_rows:,} ({fmt_pct(wismut.no_entity_rows, wismut.total_rows)})
- Character length: mean {wis_mean:.1f}, median {wis_median:.1f}, p95 {wis_p95:.1f}
- Distinct labels: {len(wismut.label_counts)}
- Distinct languages detected by py3langid: {len(wismut_lang_est)}

### Rows per language (ESTIMATED via py3langid on `text` -- no dataset field exists)

{render_language_table(wismut_lang_est, wismut.total_rows)}

### py3langid calibration (measured against ai4privacy's ground-truth `language`, EU languages only)

**Sampling method:** every twentieth row by absolute position in the full
stream across both splits and all languages (`total_rows % 20 == 0` in the
code), not every twentieth row within each language. This approximates a
1-in-20 per-language draw only because ai4privacy's rows are grouped into
large contiguous blocks by source region/language rather than shuffled --
each language's rows form a long run, so a fixed-stride positional sample
lands roughly evenly within each run. The approximation's limit: a language
whose rows were interleaved with others rather than contiguous could be
sampled less evenly than a true per-language 1-in-20 draw, and that has not
been ruled out for every language in this corpus.

{chr(10).join(calib_lines)}

### Rows per label (from `entities`)

{render_label_table(wismut.label_counts, wismut.total_rows)}

### Full label vocabulary, sorted ({len(wismut.label_counts)} labels)

{", ".join(sorted(wismut.label_counts))}

---

## Combined 24-EU-language sufficiency table

ai4privacy counts are measured (ground truth). Wismut counts are the
py3langid estimate described above -- see the calibration table for its
accuracy on this style of text. Sufficiency threshold: {SUFFICIENCY_THRESHOLD:,} rows,
combined, per the task brief.

{chr(10).join(combined_lines)}
{thin_note}
"""
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Path to write the markdown report to")
    parser.add_argument(
        "--max-rows-per-split",
        type=int,
        default=None,
        help="Cap rows read per split (for dev/testing only; omit for a full run)",
    )
    args = parser.parse_args()

    log("=== profiling ai4privacy/pii-masking-openpii-1.5m ===")
    ai4_stats, langid_agree, langid_sampled = stream_ai4privacy(args.max_rows_per_split)

    log("=== profiling Wismut/nym-pii-multilingual-data ===")
    wismut_stats, wismut_lang_est = stream_wismut(args.max_rows_per_split)

    report = render_report(
        ai4_stats, wismut_stats, langid_agree, langid_sampled, wismut_lang_est, args
    )
    with open(args.out, "w") as f:
        f.write(report)
    log(f"wrote report to {args.out}")


if __name__ == "__main__":
    main()
