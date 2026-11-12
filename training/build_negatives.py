"""Turns the fetched out-of-domain source documents into screened training
rows that carry **no** entities.

Reads `negatives/manifest.json` + `negatives/raw/` (written by
`fetch_negatives.py`), chunks each document into rows whose length
distribution matches the existing corpus, screens every chunk six ways,
and writes a grouped 90/5/5 split to `--out-dir` in exactly the format
`train.py` already consumes.

Why a negative row is dangerous, and what that forces
-----------------------------------------------------
A negative row asserts *there is nothing here*. If one overlooked real name
survives into the corpus, the model is explicitly taught to miss that name
-- and a taught miss in a privacy product is strictly worse than the
hallucination this corpus exists to fix. Screening is therefore not a tidy-up
pass; it is the reason this file is longer than the fetcher.

Six independent screens, and none of them is trusted alone
----------------------------------------------------------
1. **Selection.** Same finding as `benchmark/carriers/README.md`: choosing
   material about processes, materials and regulations rather than about
   people does most of the work before any screen runs. 146 of its 154
   carriers needed no intervention at all.

2. **Benchmark overlap**, and this one is not about personal data at all.
   These documents are deliberately different documents from
   `benchmark/carriers/`, but different EU acts share a great deal of
   formulaic text -- recital openings, comitology articles, the
   entry-into-force sentence -- and successive ECB reports reuse whole
   sentences. Measured on the first build: **468 of 25,140 chunks shared an
   8-word run with a benchmark case or carrier, and 104 still shared a
   12-word run**, which is verbatim benchmark text however different the
   source document is. Any chunk sharing a normalised word 8-gram with any
   of the 649 cases or any of the 154 carrier documents is dropped. This is
   the screen whose absence would invalidate every number the benchmark
   produces afterwards, so it runs first and it is the cheapest one to run.

3. **A mechanical sweep** for structured identifiers and for the shapes that
   surround a personal name -- e-mail addresses, URLs, IP addresses, IBAN
   shapes, long digit runs, telephone shapes, street-and-number patterns,
   postcode-and-city patterns, and honorifics. Any chunk that trips one is
   **dropped**, not repaired. The carriers *replaced* what they found,
   because a carrier has to keep reading like its register for the benchmark
   to be fair; a training negative has no such duty, and dropping is the only
   option that cannot itself introduce a mistake.

4. **The detector's own rules engine**, via `target/release/fixture-spans`
   (`AppState::rules_only()` -- the packs and the phone recognizer, no
   model). Any chunk in which it detects any of its nine structured types is
   dropped. This is the same check `benchmark/carriers/README.md` records
   running over all 462 carrier cases, and it is the only screen here that
   is the *deployed* detector: what it finds is what production would find.

5 and 6. **Two independent NER models** --
   `Babelscape/wikineural-multilingual-ner` (mBERT, trained on
   automatically-annotated Wikipedia) and `Davlan/xlm-roberta-base-ner-hrl`
   (XLM-R, trained on human-annotated CoNLL-style data), both covering
   exactly this project's six languages and sharing neither architecture,
   training data nor authors. Any chunk in which either finds a `PER` span is
   dropped. Independent matters three times over: neither was trained by this
   project, neither is the model being screened for, and a name both of them
   miss is a name that is genuinely hard to see. The
   teacher checkpoint is deliberately *not* used as the screen -- it is the
   model that hallucinates names into this exact register, so it would reject
   the corpus it is supposed to learn from. What the teacher does on this
   material is measured separately, as the size of the problem, by
   `measure_hallucination.py`.

`LOC` and `ORG` spans are **kept**. A place name or a company name is not
personal data under this project's taxonomy (`training/taxonomy.py` retired
`LOCATION` for exactly this reason), and dropping every chunk containing one
would delete the encyclopedia and annual-report registers entirely -- the two
registers this corpus exists to teach.

What is deliberately not recorded
---------------------------------
Which words tripped a screen is **not** written anywhere: not to the report,
not to a log line, not to the manifest. Counts by screen, by language and by
register are. That is `benchmark/carriers/README.md`'s "the originals are not
recorded" rule, and it applies with more force here because these documents
were never hand-read.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from build_dataset import _bucket

HERE = Path(__file__).resolve().parent
DEFAULT_NEGATIVES = HERE / "negatives"
DEFAULT_OUT = HERE / "data-negatives"

# Two NER models, not one, and deliberately not from the same lineage:
# WikiNEuRal is BERT-multilingual fine-tuned on an automatically-annotated
# Wikipedia corpus, `xlm-roberta-base-ner-hrl` is XLM-R fine-tuned on
# CoNLL-style human-annotated data. They were trained by different groups on
# different data with different architectures, so a name both of them miss is
# a name that is genuinely hard to see -- which is a much stronger statement
# than one model missing it.
#
# `Davlan/xlm-roberta-base-ner-hrl` ships no `tokenizer.json` and its
# sentencepiece file will not load under transformers 4.57 (`AttributeError:
# 'NoneType' object has no attribute 'endswith'`). It is XLM-R, so
# FacebookAI/xlm-roberta-base's own fast tokenizer is the right one and its
# vocabulary matches exactly -- both report vocab_size 250,002, checked
# rather than assumed. That is the same pairing argument train.py's docstring
# makes for MiniLM.
NER_MODELS = [
    ("Babelscape/wikineural-multilingual-ner", None),
    ("Davlan/xlm-roberta-base-ner-hrl", "FacebookAI/xlm-roberta-base"),
]

# Chunk length targets. Sampled from the existing corpus's own row lengths
# rather than picked: measured over 60,000 rows of data-merged/train.jsonl,
# mean 346 chars, p10 127, p50 273, p90 683, p99 1103, max 1200. A negative
# corpus of uniformly 1,500-character passages would teach "long text has no
# names" instead of "this register has no names", and length is the easiest
# spurious feature for a model to learn.
LENGTH_REFERENCE = HERE / "data-merged" / "train.jsonl"
LENGTH_REFERENCE_ROWS = 20_000
MIN_CHUNK_CHARS = 120
MAX_CHUNK_CHARS = 1200

# ---------------------------------------------------------------------------
# Screen 2: the mechanical sweep.
#
# Each pattern is named, because the report counts rejections per pattern and
# a pattern that never fires is worth knowing about (it means either the
# selection was good or the pattern is wrong, and those look identical until
# someone checks).
# ---------------------------------------------------------------------------

SWEEP_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "url": re.compile(r"https?://|\bwww\.\w"),
    "ipv4": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[ ]?[A-Z0-9]{4}(?:[ ]?[A-Z0-9]{4}){2,6}\b"),
    "long-digit-run": re.compile(r"\b\d{7,}\b"),
    "phone": re.compile(r"\+\d{1,3}[\s./-]?\d[\d\s./-]{6,}\d"),
    "honorific": re.compile(
        r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Herr|Frau|Dhr|Mevr|Mw|Mme|Mlle|Sr|Sra|Srta|"
        r"Sig|Sig\.ra|Dott|Ing|Drs|Ir)\.?\s+[A-ZÀ-Ý]"
    ),
    "street-and-number": re.compile(
        r"\b[A-ZÀ-Ý][\w'’-]{2,}"
        r"(?:straat|weg|laan|plein|straße|strasse|gasse|allee|damm|ufer)\s+\d{1,4}\b"
        r"|\b(?:rue|avenue|boulevard|calle|avenida|via|viale|piazza|corso)\s+"
        r"[A-ZÀ-Ý][\w'’-]{2,}[, ]+\d{1,4}\b"
        r"|\b\d{1,4}\s+[A-ZÀ-Ý][\w'’-]{2,}\s+"
        r"(?:Street|Road|Avenue|Lane|Drive|Boulevard)\b"
    ),
    "postcode-and-city": re.compile(
        r"\b\d{4}\s?[A-Z]{2}\s+[A-ZÀ-Ý][a-zà-ÿ]"
        r"|\b(?:D-|NL-|F-|E-|I-)?\d{5}\s+[A-ZÀ-Ý][a-zà-ÿ]{2,}"
    ),
    # A signature block: a line that is only a capitalised name-shaped run,
    # which is how EU acts close ("For the European Parliament / The President
    # / R. METSOLA").
    "signature-line": re.compile(r"(?m)^\s*(?:[A-Z]\.\s*){1,3}[A-ZÀ-Ý]{2,}\s*$"),
}

# A chunk that is mostly not letters is a flattened table or a formula dump,
# not prose. It teaches nothing about the register and is disproportionately
# likely to survive every other screen.
MIN_LETTER_RATIO = 0.55


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^\s*=+\s.*\s=+\s*$")
_BULLET_RE = re.compile(r"^\s*[-*•]\s")


def paragraphs(text: str) -> list[str]:
    """One list of prose paragraphs, whatever the publisher's line
    convention. Wikimedia's plain-text extracts separate paragraphs with a
    single newline and mark sections with `== Heading ==`; the HTML-derived
    documents separate with a blank line. Splitting on every newline handles
    both, and the heading and bullet filters drop what is not prose."""
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or _HEADING_RE.match(line) or _BULLET_RE.match(line):
            continue
        out.append(line)
    return out


def load_length_targets(path: Path, rows: int, rng: random.Random) -> list[int]:
    """The empirical row-length distribution of the corpus these negatives
    are mixed into. Falls back to a fixed spread if the corpus is absent, so
    this module still runs on a clone with no dataset built."""
    if not path.exists():
        log(f"  length reference {path} absent -- falling back to a fixed spread")
        return [130, 200, 270, 350, 480, 680, 900, 1100]
    lengths = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= rows:
                break
            lengths.append(len(json.loads(line)["text"]))
    lengths = [min(max(n, MIN_CHUNK_CHARS), MAX_CHUNK_CHARS) for n in lengths]
    rng.shuffle(lengths)
    return lengths


def chunk_document(paras: list[str], targets: list[int], rng: random.Random) -> list[str]:
    """Consecutive paragraphs accumulated up to a length drawn from the
    corpus's own distribution. Paragraph boundaries are never split mid-
    sentence: a truncated sentence is a text shape the model will never meet
    at inference and would be a second spurious feature of 'negative'."""
    chunks: list[str] = []
    buf: list[str] = []
    target = rng.choice(targets)
    for para in paras:
        if len(para) > target:
            # Split on sentence ends whenever the paragraph alone already
            # overshoots the target. Splitting only above MAX_CHUNK_CHARS --
            # the obvious rule -- leaves every 600-character EU recital as one
            # 600-character row however small the target was, and produced a
            # corpus averaging 456 characters a row against the corpus's 346.
            # Sentence splitting on `.!?` mis-splits abbreviations ("Art. 5",
            # "z.B."); for a corpus whose only annotation is "no entities
            # here" that costs nothing, which is why the cheap rule is the
            # right one here and would not be in align.py.
            pieces = re.split(r"(?<=[.!?。])\s+", para)
        else:
            pieces = [para]
        for piece in pieces:
            if not piece:
                continue
            have = sum(len(b) for b in buf) + max(len(buf) - 1, 0)
            # Flush *before* adding a piece that would overshoot the target by
            # more than it undershoots without it. Appending unconditionally
            # and checking afterwards -- the obvious loop -- overshoots by up
            # to a whole paragraph every time, and EU legal paragraphs are
            # long: it produced a corpus averaging 525 characters a row
            # against the 346 the name-bearing corpus averages. Row length is
            # the easiest spurious feature for the model to key "negative" on,
            # so this is not cosmetic.
            would_be = have + (1 if buf else 0) + len(piece)
            if buf and would_be - target > target - have:
                chunks.append(" ".join(buf).strip())
                buf = []
                target = rng.choice(targets)
                have = 0
                would_be = len(piece)
            buf.append(piece)
            if would_be >= target:
                chunks.append(" ".join(buf).strip())
                buf = []
                target = rng.choice(targets)
    if buf:
        text = " ".join(buf).strip()
        if len(text) >= MIN_CHUNK_CHARS:
            chunks.append(text)
    return [c for c in chunks if MIN_CHUNK_CHARS <= len(c) <= MAX_CHUNK_CHARS]


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------


def letter_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for ch in text if ch.isalpha() or ch.isspace()) / len(text)


def sweep(text: str) -> list[str]:
    """Names of every mechanical pattern this chunk trips. Never the match."""
    return [name for name, pat in SWEEP_PATTERNS.items() if pat.search(text)]


RULES_BINARY = HERE.parent / "target" / "release" / "fixture-spans"


def rules_engine_hits(
    rows: list[tuple[str, str]],
    langs: dict[str, str],
    binary: Path,
    workdir: Path,
) -> tuple[list[int], dict]:
    """Indices of chunks in which the detector's rules engine detects anything.

    Shells out to the already-built `fixture-spans` binary rather than
    reimplementing nine recognizers in Python: a second implementation would
    be a second, subtly different detector, and the whole value of this screen
    is that it is *the deployed one*. The binary is run, never rebuilt --
    `cargo` is deliberately not invoked from here.

    The binary emits offsets and type labels and no text, which is why its
    output can be summarised into a committed report at all.
    """
    import subprocess

    if not binary.exists():
        raise SystemExit(
            f"rules screen: {binary} is not built. Build it once with "
            f"`cargo build --release -p pii-detection-eval --bin fixture-spans`, or pass "
            f"--no-rules to skip this screen (and do not train on the result)."
        )
    workdir.mkdir(parents=True, exist_ok=True)
    fixture = workdir / "negatives-candidates.json"
    spans_out = workdir / "negatives-candidate-spans.json"
    # `analyze_text` refuses text over MAX_CHARACTERS; chunks are capped at
    # MAX_CHUNK_CHARS (1200), far below it, so no case can be refused here.
    cases = [
        {"id": str(i), "lang": langs[doc_id], "text": text}
        for i, (doc_id, text) in enumerate(rows)
    ]
    fixture.write_text(json.dumps({"cases": cases}, ensure_ascii=False), encoding="utf-8")
    subprocess.run(
        [str(binary), "--fixture", str(fixture), "--out", str(spans_out)],
        check=True, capture_output=True,
    )
    result = json.loads(spans_out.read_text())
    hits: list[int] = []
    by_type: Counter = Counter()
    for case in result["cases"]:
        if len(cases[int(case["id"])]["text"]) != case["char_len"]:
            raise SystemExit(
                "rules screen: the Rust and Python parses of a candidate disagree on its "
                "length in code points -- offsets from this run cannot be trusted."
            )
        if case["spans"]:
            hits.append(int(case["id"]))
            for span in case["spans"]:
                by_type[span["entity_type"]] += 1
    fixture.unlink(missing_ok=True)
    return hits, {
        "engine": result["engine"],
        "rules_version": result["rules_version"],
        "requested_types": result["requested_types"],
        "by_type": dict(by_type),
    }


class PersonScreen:
    """Screen 3. Wraps an independent multilingual NER model and answers one
    question per chunk: does anything in here look like a person?"""

    def __init__(self, model_name: str, device: str, batch_size: int, threshold: float,
                 tokenizer_name: str | None = None):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.torch = torch
        self.threshold = threshold
        self.batch_size = batch_size
        self.device = device
        self.name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        if self.model.config.vocab_size != self.tokenizer.vocab_size:
            raise SystemExit(
                f"{model_name} has vocab_size {self.model.config.vocab_size} but its "
                f"tokenizer has {self.tokenizer.vocab_size}. Token ids from one would index "
                "into the other's embedding table and the screen would be reading noise."
            )
        self.model.to(device)
        self.model.eval()
        id2label = self.model.config.id2label
        self.person_ids = [i for i, lab in id2label.items() if lab.endswith("PER")]
        if not self.person_ids:
            raise SystemExit(f"{model_name} has no PER label -- refusing to screen with it")

    def flags(self, texts: list[str]) -> list[bool]:
        """True where the model puts probability >= threshold on any PER tag
        at any real token. Probability, not argmax: a name the screen is only
        60% sure about is exactly the one that must not become a negative."""
        out: list[bool] = []
        torch = self.torch
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=256
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits
            probs = logits.softmax(-1)
            person = probs[:, :, self.person_ids].sum(-1)
            mask = enc["attention_mask"].bool()
            person = person.masked_fill(~mask, 0.0)
            out.extend((person.max(dim=1).values >= self.threshold).tolist())
        return out


# ---------------------------------------------------------------------------


def rescreen(out_dir: Path, neg_dir: Path, ngram: int) -> int:
    """Re-apply screens 2a and 2 to an already-built corpus, in place.

    Why this exists rather than "just rebuild". Screens 3 to 6 are the
    expensive ones -- the rules engine and two NER models over 25,000 chunks
    -- and they are also the ones whose verdict cannot change here: they
    inspect a chunk on its own, and nothing about a chunk changed. Only the
    benchmark changed. Screens 2a and 2 are pure functions of (document,
    benchmark) and (chunk text, benchmark), so re-running just those two over
    the built rows yields exactly the set a full rebuild would keep.

    That equivalence holds because screening is a conjunction: a row survives
    if it passes every screen, and the order they run in cannot change the
    set that passes all of them. This pass only ever removes rows. It never
    adds one, never edits one, and never moves one between splits -- so every
    row left behind is a row the full six-screen build already approved.

    The one thing a full rebuild would do that this does not: split
    assignment is drawn per (language, register) stratum, so removing whole
    documents from a stratum would, on a rebuild, reshuffle which of the
    *remaining* documents land in validation and test. This pass leaves every
    surviving document in the split it was already assigned. That is a
    difference in which held-out rows are held out, not in whether the corpus
    is clean, and it is the conservative choice: it changes nothing it does
    not have to.
    """
    import check_contamination as cc

    manifest = json.loads((neg_dir / "manifest.json").read_text())
    carrier_keys = cc.carrier_identity_keys()
    carrier_docs = {
        d["id"] for d in manifest["documents"]
        if cc.document_identity_keys(d) & carrier_keys
    }
    cc.NGRAM = ngram
    cases = cc.load_cases()
    carriers = cc.load_carrier_documents()
    _, shingle_owner = cc.build_index(cases, carriers)
    log(f"rescreen: {len(cases)} cases, {len(carriers)} carrier documents, {ngram}-gram")
    log(f"screen 2a (document identity): {len(carrier_docs)} source documents are benchmark carriers")

    total_before = total_after = dropped_2a = dropped_2 = 0
    for name in ("train", "validation", "test"):
        path = out_dir / f"{name}.jsonl"
        if not path.exists():
            continue
        keep: list[str] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                total_before += 1
                row = json.loads(line)
                if row.get("doc_id") in carrier_docs:
                    dropped_2a += 1
                    continue
                if cc.shingles(cc.normalise(row["text"])) & shingle_owner.keys():
                    dropped_2 += 1
                    continue
                keep.append(line)
        path.write_text("".join(keep), encoding="utf-8")
        total_after += len(keep)
        log(f"  {name}: {len(keep):,} rows kept")
    log(f"rescreen dropped {dropped_2a:,} rows by document identity and {dropped_2:,} by "
        f"{ngram}-gram overlap; {total_before:,} -> {total_after:,} rows")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--negatives-dir", default=str(DEFAULT_NEGATIVES))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--length-reference", default=str(LENGTH_REFERENCE))
    ap.add_argument("--ner-threshold", type=float, default=0.30,
                    help="max PER probability at which a chunk is dropped; deliberately "
                         "well below 0.5 -- a false drop costs one negative row, a false "
                         "keep teaches the model to miss a real name")
    ap.add_argument("--ner-batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--benchmark-ngram", type=int, default=8,
                    help="drop a chunk sharing a normalised word n-gram of this length with "
                         "any benchmark case or carrier document. Eight, not twelve: for the "
                         "contamination *report* an 8-gram collision can be a stock phrase, "
                         "but here the cost of dropping a chunk is one row out of 25,000 and "
                         "the cost of keeping a contaminated one is every benchmark number "
                         "afterwards, so the screen is set where the report's escalation "
                         "starts rather than where it ends.")
    ap.add_argument("--no-benchmark-screen", action="store_true",
                    help="skip screen 2 (for a fast structural dry run only -- the output of "
                         "such a run must never be trained on)")
    ap.add_argument("--balance", dest="balance", action="store_true", default=True,
                    help="trim each register to the smallest language's document count (default)")
    ap.add_argument("--no-balance", dest="balance", action="store_false",
                    help="keep every fetched document, imbalance and all")
    ap.add_argument("--max-rows-per-document", type=int, default=None,
                    help="cap rows taken from any one document, so a 100,000-character "
                         "regulation cannot outweigh fifty encyclopedia articles")
    ap.add_argument("--rules-binary", default=str(RULES_BINARY),
                    help="the already-built fixture-spans binary (AppState::rules_only()). "
                         "Never rebuilt from here.")
    ap.add_argument("--workdir", default=str(HERE / ".cache" / "negatives"),
                    help="scratch directory for the rules screen's fixture and span files")
    ap.add_argument("--no-rules", action="store_true",
                    help="skip screen 3 (for a fast structural dry run only -- the output "
                         "of such a run must never be trained on)")
    ap.add_argument("--no-ner", action="store_true",
                    help="skip screen 3 (for a fast structural dry run only -- the output "
                         "of such a run must never be trained on)")
    ap.add_argument("--report-out", default=None)
    ap.add_argument("--rescreen", action="store_true",
                    help="do not build: re-apply screens 2a (document identity) and 2 "
                         "(benchmark n-gram overlap) to the corpus already in --out-dir, "
                         "dropping rows in place. For when the benchmark has grown since "
                         "the corpus was built -- the case screen 2 cannot catch on its "
                         "own, because it runs at build time against the benchmark of the "
                         "day. Removes rows only; never adds, edits or re-splits one.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    neg_dir = Path(args.negatives_dir)

    if args.rescreen:
        return rescreen(Path(args.out_dir), neg_dir, args.benchmark_ngram)

    manifest = json.loads((neg_dir / "manifest.json").read_text())
    docs = manifest["documents"]
    doc_meta = {d["id"]: d for d in docs}
    raw_dir = neg_dir / "raw"

    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    log(f"{len(docs)} source documents, retrieved {manifest['retrieved']}")

    # --- screen 2a: document identity --------------------------------------
    #
    # Runs before balancing, because dropping a document changes the
    # per-(register, language) cap that balancing computes, and a cap
    # computed over documents that are about to be removed balances the
    # corpus against a shape it will not have.
    #
    # This screen did not exist until 2026-08-27. Screen 2 below
    # compares *chunk text* against the benchmark and is a pure function of
    # the benchmark as it stands at build time. That is correct and it
    # worked: when this corpus was built (2026-08-26 00:16) the benchmark
    # held 154 carriers, none of them a source document here, and screen 2
    # dropped what it should. Twenty-two hours later the benchmark grew to
    # 321 carriers and drew 22 of the new ones from Wikimedia pages this
    # corpus had already fetched. Nothing re-screened, so 943 rows cut from
    # 22 benchmark carrier documents stayed in the training corpus.
    #
    # Screening chunks by text was never enough on its own to prevent that.
    # A carrier document is off-limits *as a document*: the benchmark cuts
    # its cases from some passages and leaves others, and training on the
    # passages it happened not to cut is still training on the document the
    # model is later scored against. `check_contamination.py` has said so
    # since it was written -- it reports a document-identity clash as
    # contamination regardless of whether any text collides -- but only the
    # report enforced it, and a report is not a screen.
    if not args.no_benchmark_screen:
        import check_contamination as cc

        carrier_keys = cc.carrier_identity_keys()
        kept_docs_, carrier_docs = [], []
        for d in docs:
            (carrier_docs if cc.document_identity_keys(d) & carrier_keys else kept_docs_).append(d)
        if carrier_docs:
            log(f"screen 2a (document identity): dropped {len(carrier_docs)} source documents "
                f"that are themselves benchmark carriers")
            for d in carrier_docs:
                log(f"    {d['id']} ({d['lang']} {d['register']})")
        docs = kept_docs_
        doc_meta = {d["id"]: d for d in docs}

    targets = load_length_targets(Path(args.length_reference), LENGTH_REFERENCE_ROWS, rng)
    log(f"length targets sampled from {len(targets)} corpus rows "
        f"(min {min(targets)}, max {max(targets)})")

    # --- boilerplate: a paragraph that appears in more than one document of
    # the same publisher is site furniture (cookie banners, disclaimers), not
    # the register. Detected rather than pattern-matched, so it needs no
    # per-publisher maintenance.
    para_docs: dict[str, set[str]] = defaultdict(set)
    doc_paras: dict[str, list[str]] = {}
    for d in docs:
        path = raw_dir / f"{d['id']}.txt"
        if not path.exists():
            log(f"  {d['id']}: raw text absent -- skipped (re-run fetch_negatives.py)")
            continue
        ps = paragraphs(path.read_text(encoding="utf-8"))
        doc_paras[d["id"]] = ps
        for p in ps:
            para_docs[p].add(d["id"])
    boilerplate = {p for p, ids in para_docs.items() if len(ids) > 1}
    log(f"boilerplate paragraphs (identical text in >1 document): {len(boilerplate)}")

    # --- balance the document count per (language, register) ---------------
    #
    # Six languages must get the same amount of each register, or the corpus
    # teaches "English encyclopedia prose has no names" several times harder
    # than it teaches the same thing about Dutch -- and per-language
    # precision is a headline number in this project
    # (reports/pii-detection-vs-azure-2026-08-23.md reports all six separately).
    #
    # The legal and financial registers are already exactly balanced by
    # construction: the same eleven acts and the same two annual reports in
    # every language. Only the encyclopedia register is uneven, because
    # `langlinks` resolves a different number of subjects per language. So
    # this trims to the smallest language's count within each register rather
    # than across the corpus.
    docs_by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for d in docs:
        if d["id"] in doc_paras:
            docs_by_key[(d["register"], d["lang"])].append(d)
    balanced: list[dict] = []
    all_languages = sorted({d["lang"] for d in docs})
    for register in sorted({r for r, _ in docs_by_key}):
        per_lang = {lang: v for (r, lang), v in docs_by_key.items() if r == register}
        # A register present in five languages is worse than a register that
        # is absent: it silently teaches "this register contains no names" in
        # five languages and says nothing about the sixth, while every
        # per-register number computed afterwards reads as though it covered
        # all six. Balancing cannot catch it -- `min()` over five languages is
        # a perfectly good number -- so it is caught here instead.
        missing = [lang for lang in all_languages if lang not in per_lang]
        if missing and args.balance:
            raise SystemExit(
                f"register {register!r} has no documents in {', '.join(missing)}. "
                "Every register must reach every language or its per-language figures are "
                "about a subset of the languages. Fix the source registry in "
                "fetch_negatives.py, or pass --no-balance and accept an unbalanced corpus."
            )
        cap = min(len(v) for v in per_lang.values()) if args.balance else None
        for lang in sorted(per_lang):
            chosen = sorted(per_lang[lang], key=lambda d: d["id"])
            if cap is not None:
                # Spread rather than take the first `cap`: document ids sort
                # by subject, so the first N would be alphabetically clustered
                # subjects rather than a cross-section of them.
                step = len(chosen) / cap
                chosen = [chosen[int(i * step)] for i in range(cap)]
            balanced.extend(chosen)
        counts = {lang: len(v) for lang, v in sorted(per_lang.items())}
        log(f"  {register}: " + ", ".join(f"{k}={v}" for k, v in counts.items())
            + (f" -> capped to {cap} per language" if cap is not None and cap != max(counts.values()) else ""))
    dropped_balance = len(doc_paras) - len(balanced)
    if dropped_balance:
        log(f"balancing dropped {dropped_balance} documents")
    docs = balanced

    # --- chunk -------------------------------------------------------------
    by_doc: dict[str, list[str]] = {}
    for d in docs:
        ps = [p for p in doc_paras.get(d["id"], []) if p not in boilerplate]
        by_doc[d["id"]] = chunk_document(ps, targets, rng)
    if args.max_rows_per_document:
        for doc_id, chunks in by_doc.items():
            if len(chunks) > args.max_rows_per_document:
                # Spread across the document, not its first N paragraphs: an
                # act's opening recitals read nothing like its annexes, and
                # the annexes are the part that looks most like the CSV and
                # table shapes the model hallucinates on.
                step = len(chunks) / args.max_rows_per_document
                by_doc[doc_id] = [chunks[int(i * step)] for i in range(args.max_rows_per_document)]
    total_chunks = sum(len(v) for v in by_doc.values())
    log(f"chunked into {total_chunks:,} candidate rows from {len(by_doc)} documents")

    # --- screen 2: benchmark overlap ---------------------------------------
    dropped_benchmark = 0
    if args.no_benchmark_screen:
        log("screen 2 (benchmark overlap): SKIPPED -- this output must not be trained on")
    else:
        import check_contamination as cc

        cc.NGRAM = args.benchmark_ngram
        cases = cc.load_cases()
        carriers = cc.load_carrier_documents()
        _, shingle_owner = cc.build_index(cases, carriers)
        owners: Counter = Counter()
        for doc_id, chunks in by_doc.items():
            clean = []
            for text in chunks:
                overlap = cc.shingles(cc.normalise(text)) & shingle_owner.keys()
                if overlap:
                    dropped_benchmark += 1
                    owners[shingle_owner[next(iter(overlap))].split(":")[0]] += 1
                else:
                    clean.append(text)
            by_doc[doc_id] = clean
        log(f"screen 2 (benchmark overlap, {args.benchmark_ngram}-gram against "
            f"{len(cases)} cases and {len(carriers)} carrier documents): "
            f"dropped {dropped_benchmark:,} chunks")
        for owner, n in owners.most_common(6):
            log(f"    overlapping with {owner}: {n}")

    # --- screen 3 ----------------------------------------------------------
    kept: list[tuple[str, str]] = []  # (doc_id, text)
    sweep_counts: Counter = Counter()
    dropped_sweep = 0
    dropped_ratio = 0
    for doc_id, chunks in by_doc.items():
        for text in chunks:
            if letter_ratio(text) < MIN_LETTER_RATIO:
                dropped_ratio += 1
                continue
            tripped = sweep(text)
            if tripped:
                dropped_sweep += 1
                for name in tripped:
                    sweep_counts[name] += 1
                continue
            kept.append((doc_id, text))
    log(f"screen 3 (mechanical sweep): dropped {dropped_sweep:,} chunks "
        f"(+{dropped_ratio:,} for letter ratio < {MIN_LETTER_RATIO})")
    for name, n in sweep_counts.most_common():
        log(f"    {name}: {n}")

    # --- screen 4 ----------------------------------------------------------
    dropped_rules = 0
    rules_info: dict = {}
    if args.no_rules:
        log("screen 4 (the detector rules engine): SKIPPED -- this output must not be trained on")
    else:
        t0 = time.time()
        langs = {d["id"]: d["lang"] for d in docs}
        hits, rules_info = rules_engine_hits(kept, langs, Path(args.rules_binary), Path(args.workdir))
        dropped_rules = len(hits)
        hit_set = set(hits)
        kept = [row for i, row in enumerate(kept) if i not in hit_set]
        log(f"screen 4 (the detector rules engine, {rules_info['rules_version']}, "
            f"{len(rules_info['requested_types'])} types): dropped {dropped_rules:,} chunks "
            f"in {time.time() - t0:.0f}s")
        for t, n in sorted(rules_info["by_type"].items()):
            log(f"    {t}: {n}")

    # --- screens 5 and 6 ---------------------------------------------------
    ner_dropped: dict[str, int] = {}
    if args.no_ner:
        log("screens 5-6 (independent NER): SKIPPED -- this output must not be trained on")
    else:
        for n, (model_name, tok_name) in enumerate(NER_MODELS, start=5):
            log(f"screen {n} (independent NER): {model_name} on {device}, "
                f"PER probability threshold {args.ner_threshold}")
            screen = PersonScreen(model_name, device, args.ner_batch_size,
                                  args.ner_threshold, tok_name)
            t0 = time.time()
            flags = screen.flags([t for _, t in kept])
            ner_dropped[model_name] = sum(flags)
            kept = [row for row, flagged in zip(kept, flags) if not flagged]
            log(f"    dropped {ner_dropped[model_name]:,} chunks in {time.time() - t0:.0f}s "
                f"({len(kept):,} left)")
        # The second model's yield is the interesting number: it is the count
        # of chunks the first screen was happy with and a differently-trained
        # detector was not, which is the closest thing available to an
        # estimate of what a third would still find.
    dropped_ner = sum(ner_dropped.values())

    # --- rows and split ----------------------------------------------------
    #
    # The group is the *document*, not the row. Every chunk of one document
    # lands in one split -- stricter than the skeleton grouping
    # build_dataset.py uses for the name-bearing corpora, and necessarily so:
    # two chunks of one Wikipedia article share vocabulary, subject and style,
    # which is exactly the near-duplicate leakage grouped splitting exists to
    # prevent.
    #
    # The assignment is **stratified within (language, register)** rather than
    # a plain hash of the group. build_dataset._bucket is the right function
    # for a corpus of 700,000 groups, where 5% of the buckets is tens of
    # thousands of rows and every language turns up in every split by sheer
    # number. This corpus has ~150 groups; a plain hash left `de` out of
    # validation and `en` out of test on the first build, which would make a
    # per-language hallucination figure a figure about five languages. So each
    # (language, register) bucket contributes at least one document to `test`,
    # and one to `validation` wherever it has three or more documents to give.
    # `_bucket` is still used -- to *order* documents within a bucket, so the
    # choice of which document goes where stays a deterministic function of
    # (group, seed) and not of filesystem order.
    doc_split: dict[str, str] = {}
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    kept_docs = {doc_id for doc_id, _ in kept}
    for d in docs:
        if d["id"] in kept_docs:
            strata[(d["lang"], d["register"])].append(d["id"])
    for key, ids in sorted(strata.items()):
        ordered = sorted(ids, key=lambda i: (_bucket(f"neg-{i}", args.seed), i))
        n = len(ordered)
        n_test = max(1, round(0.05 * n))
        n_val = max(1, round(0.05 * n)) if n >= 3 else 0
        for i, doc_id in enumerate(ordered):
            if i < n_test:
                doc_split[doc_id] = "test"
            elif i < n_test + n_val:
                doc_split[doc_id] = "validation"
            else:
                doc_split[doc_id] = "train"

    splits: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    per_doc = Counter()
    for doc_id, text in kept:
        meta = doc_meta[doc_id]
        group = f"neg-{doc_id}"
        splits[doc_split[doc_id]].append(
            {
                "text": text,
                "entities": [],
                "language": meta["lang"],
                "source": f"negative:{meta['register']}",
                "group": group,
                "doc_id": doc_id,
            }
        )
        per_doc[doc_id] += 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        rng.shuffle(rows)
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_lang = Counter(r["language"] for rows in splits.values() for r in rows)
    by_reg = Counter(r["source"] for rows in splits.values() for r in rows)
    total_rows = sum(len(v) for v in splits.values())
    total_chars = sum(len(r["text"]) for rows in splits.values() for r in rows)
    docs_with_rows = sum(1 for v in per_doc.values() if v)

    log("")
    log(f"{total_rows:,} negative rows from {docs_with_rows} documents, "
        f"{total_chars:,} characters, mean {total_chars / max(total_rows, 1):.0f} chars/row")
    log("  splits: " + ", ".join(f"{k}={len(v):,}" for k, v in splits.items()))
    log("  by language: " + ", ".join(f"{k}={v:,}" for k, v in sorted(by_lang.items())))
    log("  by register: " + ", ".join(f"{k}={v:,}" for k, v in sorted(by_reg.items())))
    log(f"  written to {out_dir}")

    # Every language must be present in train and test, or a per-language
    # hallucination figure computed later is silently a figure about five
    # languages. Validation is checked but only warned about: a register with
    # two documents per language cannot fill three splits, and the ECB annual
    # reports are exactly that.
    problems, warnings = [], []
    all_langs = sorted({d["lang"] for d in docs})
    for split, rows in splits.items():
        present = {r["language"] for r in rows}
        for lang in all_langs:
            if lang not in present:
                (warnings if split == "validation" else problems).append(f"{lang} missing from {split}")
    for split in ("train", "test"):
        present = {r["source"] for r in splits[split]}
        for reg in sorted({f"negative:{d['register']}" for d in docs}):
            if reg not in present:
                problems.append(f"{reg} missing from {split}")
    if warnings:
        log("note: " + "; ".join(warnings))
    if problems:
        raise SystemExit(
            "the split does not cover every language and register in train and test: "
            + "; ".join(problems)
            + ". Refusing to write a corpus whose per-language numbers would be about "
              "a subset of the languages."
        )

    if args.report_out:
        path = Path(args.report_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Out-of-domain negatives: build and screening record",
            "",
            f"Built {time.strftime('%Y-%m-%d')} by `training/build_negatives.py`. "
            f"{len(doc_meta)} documents in `training/negatives/manifest.json`; {len(docs)} "
            f"kept after per-register language balancing; {docs_with_rows} still had rows "
            f"after screening.",
            "",
            "| stage | chunks |",
            "|---|---|",
            f"| candidates after chunking | {total_chunks:,} |",
            f"| dropped: shares a {args.benchmark_ngram}-gram with a benchmark case or carrier | {dropped_benchmark:,} |",
            f"| dropped: letter ratio < {MIN_LETTER_RATIO} (tables, formula dumps) | {dropped_ratio:,} |",
            f"| dropped: mechanical sweep | {dropped_sweep:,} |",
            f"| dropped: the detector rules engine detected a structured identifier | {dropped_rules:,} |",

            *[f"| dropped: independent NER `{m}` `PER` >= {args.ner_threshold} | {n:,} |"
              for m, n in ner_dropped.items()],
            f"| **kept** | **{total_rows:,}** |",
            "",
            "Mechanical sweep, by pattern (a chunk can trip more than one):",
            "",
            "| pattern | chunks tripped |",
            "|---|---|",
        ]
        for name in SWEEP_PATTERNS:
            lines.append(f"| `{name}` | {sweep_counts.get(name, 0):,} |")
        lines += [
            "",
            (f"Rules engine: `{rules_info.get('engine', 'skipped')}`, rules version "
             f"`{rules_info.get('rules_version', '-')}`, asked for "
             f"{len(rules_info.get('requested_types', []))} types. Detections in the chunks it "
             f"rejected, by type: "
             + (", ".join(f"`{k}` {v}" for k, v in sorted(rules_info.get("by_type", {}).items()))
                or "none")
             + "."),
            "",
            "Rows by language: " + ", ".join(f"`{k}` {v:,}" for k, v in sorted(by_lang.items())) + ".",
            "",
            "Rows by register: " + ", ".join(f"`{k}` {v:,}" for k, v in sorted(by_reg.items())) + ".",
            "",
            f"Splits, grouped by source **document** and stratified within (language, "
            f"register) so every language and register reaches `test`: "
            + ", ".join(f"{k} {len(v):,}" for k, v in splits.items()) + ".",
            "",
            "**No matched text is recorded here or anywhere else.** Counts by pattern, "
            "language and register are; the words that tripped a screen are not.",
            "",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        log(f"  report written to {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
