#!/usr/bin/env python3
"""Fine-tune Multilingual-MiniLM-L12-H384 for Temperate's PII token
classification (PERSON, STREET_ADDRESS, plus the ten other platform types for
completeness of the label space, though nine of those already have
deterministic rules in the Rust service).

Same script serves two purposes:
  - the smoke test: --smoke runs a fixed, tiny preset (a few hundred rows,
    a handful of steps, ~2 minutes) and prints a PASS/FAIL verdict --
    device actually selected, whether the model's parameters are genuinely
    on it, tokens/sec, first/last loss. Run this first on any new machine,
    especially a rented GPU, before trusting it with a real run. --limit N
    is the more general form (any row count, not just the smoke preset)
    for ad hoc subsample runs and throughput measurement.
  - the eventual full run: omit --limit and --smoke to train on the full
    dataset.

Base model and tokenizer pairing -- read before changing either:

  microsoft/Multilingual-MiniLM-L12-H384's own config.json declares
  `model_type: bert` and `tokenizer_class: XLMRobertaTokenizer`; its model
  card states outright that AutoModelForTokenClassification / AutoTokenizer
  will not work with this checkpoint. So: BertForTokenClassification for the
  model, and the tokenizer loaded from FacebookAI/xlm-roberta-base (its fast
  tokenizer -- MiniLM's own repo ships only sentencepiece.bpe.model, no
  tokenizer.json, so its *default* tokenizer is the slow one, and slow
  tokenizers do not return offset_mapping, which everything below depends
  on). Verified before relying on it, not assumed: MiniLM's checkpoint and
  xlm-roberta-base's sentencepiece.bpe.model are byte-identical (sha256
  cfc8146a...db2865 both), and the model's embedding table (250,037 rows)
  is exactly the tokenizer's vocab (250,002) plus 35 unused padding rows.
  Token ids from the xlm-roberta-base tokenizer index directly into
  MiniLM's own embedding table -- confirmed independently by an external
  review via nearest-neighbour probes on MiniLM's embedding rows decoded
  through this tokenizer (e.g. the row for "Paris" neighbours "Париж",
  "París", "پاریس"), which collapses to noise under any id-shift control.

Label alignment: BIO tags derived from the fast tokenizer's offset_mapping
against the dataset's exact character spans (training/align.py). Never by
re-tokenising entity text and searching for it -- see align.py's docstring
for why that would reintroduce the annotation noise the corpus was built to
avoid.

Long text is windowed (max_length + stride) rather than truncated, because
the Rust service windows long text at inference time. max_length defaults
to 256 (matching the service) with length-bucketed batching -- not a
smaller max_length -- as the fix for padding waste; see BucketBatchSampler
and training/reports/smoke-test.md for the measurement that motivated this.

Distillation (added for the 6-layer student; see
training/reports/distillation-plan.md)
--------------------------------------------------------------------------
`--student-layers N` builds an N-layer student by keeping evenly-spaced
encoder layers of `--init-from`'s checkpoint -- the DistilBERT/TinyBERT
initialisation, and the same truncation the 18.4 ms latency probe used.
`--teacher-checkpoint` then adds a KD term against the full teacher's
logits, so truncation is the *starting point* and distillation is the
*objective*, rather than the two being alternatives.

**The one interaction that is easy to get catastrophically wrong.** The
teacher is wrong in both directions on exactly the rows this run adds to
correct it. It hallucinates `PERSON` and `STREET_ADDRESS` on out-of-domain
prose (`reports/spurious-address-spans.md`), and it misses names in tabular
and byline positions (`reports/person-misses.md`). Distilling its logits on
either would teach the student the defect *in the very place the new data
is meant to remove it*, quietly, because the KD term would look healthy the
whole time. So the KD loss is **masked off on every row whose source starts
with one of `KD_EXCLUDED_SOURCE_PREFIXES`** (see `kd_mask` in `collate` and
`TrainingExample.kd`), where the constructed gold labels are correct and
the teacher is not. On the name-bearing corpus, where the teacher is
108/111 on `PERSON` and 130/133 on `STREET_ADDRESS`, KD is on.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import random
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Must be set before any MPS op runs. Harmless on a CPU-only run.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoTokenizer, BertForTokenClassification
from transformers.optimization import get_linear_schedule_with_warmup

from align import ID2LABEL, IGNORE_INDEX, LABEL2ID, LABELS, Window, tokenize_and_align
from taxonomy import PLATFORM_TYPES

BASE_MODEL = "microsoft/Multilingual-MiniLM-L12-H384"
TOKENIZER_NAME = "FacebookAI/xlm-roberta-base"

# --smoke's preset: a few hundred rows (~10 batches at the default
# batch-size of 32) is enough to exercise every stage of the pipeline --
# load, tokenize, model-to-device, a handful of optimizer steps, save --
# without the row count itself being the bottleneck. warmup_steps_excluded
# is cut from the default 5 down to 2 so a ~10-step run still leaves
# measured steps to compute tokens/sec from.
SMOKE_LIMIT = 300
SMOKE_WARMUP_STEPS_EXCLUDED = 2
# --limit caps the primary corpus only. Left alone, --smoke with
# --extra-data-dir loaded all 27,452 extra rows against 300 primary ones and
# turned the two-minute smoke test into an 877-step run -- which still passes,
# so the symptom is only that it takes twenty minutes on the pod's clock
# rather than two. 50 rows per extra directory keeps it fast and is still
# enough for both KD-masked sources to land in the first few batches, which is
# what the smoke test is checking. The resulting ratio is deliberately nothing
# like the real run's; --smoke has never claimed to be a scale model.
SMOKE_EXTRA_LIMIT = 50

# Sources whose gold labels are right and whose teacher logits are wrong, so
# the KD term is masked off on them (see kd_loss_fn and the module docstring):
#
#   negative:  out-of-domain prose with no personal data. The teacher
#              hallucinates names and addresses here -- that is the defect
#              this run exists to fix (reports/spurious-address-spans.md).
#   augment:   tabular and byline PERSON examples. The teacher *misses* names
#              in exactly these shapes -- all three remaining PERSON recall
#              misses are one of them (reports/person-misses.md).
#
# Both directions of the teacher's error are therefore excluded, not just the
# hallucination. Leaving `augment:` in would have the teacher teaching the
# student the miss, on the rows added to fix the miss, while the KD curve
# looked healthy the whole time.
KD_EXCLUDED_SOURCE_PREFIXES = ("negative:", "augment:")



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(requested: str) -> str:
    """`auto` resolves cuda -> mps -> cpu, in that order (a rented CUDA box
    and a laptop's MPS are never both present, but if they ever were, a
    dedicated GPU should win over Apple's unified-memory backend). An
    explicit `--device mps` (or `cuda`) on a machine without it used to be
    returned unchanged and only fail later, deep inside the training loop,
    on the first tensor .to() call -- fail at the argument instead.

    The cuda guard doubles as the defence against a common
    deployment trap: a rented GPU machine often ships a CUDA-enabled torch baked into
    its base image, but `training/pyproject.toml` only declares
    `torch>=2.4`, so a plain `pip install -e training/` on that machine is free
    to satisfy that constraint with any matching wheel -- including a
    CPU-only one -- silently replacing the template's CUDA build. Training
    would then run on the CPU while every log line still says it started,
    at a fraction of the speed, with slowness as the only symptom. Requesting --device cuda explicitly (never rely on
    --device auto silently falling back) turns that into a loud failure
    right here instead.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "--device cuda requested but torch.cuda.is_available() is False.\n"
            "On a rented GPU machine: `pip install -e training/` (or any plain pip install\n"
            "resolving training/pyproject.toml's `torch>=2.4`) can silently replace the\n"
            "template's CUDA-enabled torch with a CPU-only wheel. Check with:\n"
            "  python -c 'import torch; print(torch.__version__, torch.version.cuda)'\n"
            "If torch.version.cuda prints None, reinstall torch matching the CUDA build\n"
            "already on the box instead of installing training's dependencies over it --\n"
            "Otherwise use --device cpu or --device auto."
        )
    if requested == "mps" and not torch.backends.mps.is_available():
        raise SystemExit(
            "--device mps requested but torch.backends.mps.is_available() is "
            "False on this machine. Use --device cpu or --device auto."
        )
    return requested


def default_num_workers() -> int:
    """DataLoader worker processes, sized to the machine this runs on
    rather than hardcoded, because the two machines this script targets
    disagree: 10 cores on a laptop, 9 vCPU on the GPU machine. Reserves one
    core for the main process (the one driving the GPU) and caps at 4
    regardless of how many cores are left.

    The cap, not just the reservation, matters: PIIDataset tokenizes and
    windows every row eagerly at construction time (see its docstring), so
    a worker's __getitem__ is a plain Python list index, and collate()
    (called in the worker) is just padding into small tensors -- there is
    no per-item I/O or tokenization for extra workers to parallelize.
    What extra workers *do* buy is overlap: collation happening in a
    worker process while the GPU is still busy with the previous batch,
    which is the actual reason to raise this above 0 on a fast GPU (see
    the module-level docstring's data-loading-bound note). Past a handful
    of workers that overlap is already saturated and more workers just add
    IPC/pickling overhead for no further throughput -- standard PyTorch
    guidance for lightweight collate functions.

    Measured: os.cpu_count() is 10 on a laptop and 9 on the GPU machine
    it was trained on; this heuristic resolves both to 4.
    """
    n = os.cpu_count() or 1
    return max(0, min(4, n - 1))


def reservoir_sample_jsonl(path: Path, limit: int, seed: int) -> list[dict]:
    """Read `path` once, keep up to `limit` rows via reservoir sampling.

    Deliberately not `head(limit)`: this dataset's row order is not a global
    shuffle. Measured directly before writing this script -- the first
    20,000 rows of train.jsonl are 87.9% ai4privacy / 0% nym, and 16
    languages, dominated by zh/vi/id/ms/tl/ko/ja (>85% of rows), with every
    one of the 2,426 `generated` (Irish/Maltese) rows front-loaded into that
    same slice. A --limit smoke test built on head(N) would measure
    throughput on an unrepresentative language mix and validate alignment
    against a skewed script distribution, exactly backwards for a task whose
    stated risk is script/offset bugs in non-Latin text.
    """
    rng = random.Random(seed)
    reservoir: list[dict] = []
    with path.open() as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            if len(reservoir) < limit:
                reservoir.append(row)
            else:
                j = rng.randint(0, i)
                if j < limit:
                    reservoir[j] = row
    return reservoir


def load_rows(path: Path, limit: int | None, seed: int) -> list[dict]:
    if limit is None:
        rows = []
        with path.open() as f:
            for line in f:
                rows.append(json.loads(line))
        return rows
    return reservoir_sample_jsonl(path, limit, seed)


@dataclass
class TrainingExample:
    """Deliberately lean, unlike align.Window. A deep-sizing review of the
    original design (which stored full Window objects, including
    offset_mapping and a materialized attention_mask, in the training
    Dataset) found offset_mapping alone was 63.3% of memory per stored
    window and attention_mask another 5.9%, on a projected ~37 GB for the
    full 2.24M-row corpus. Neither is needed in the training path:
    offset_mapping exists only to map labels back to character spans, which
    training never does (only training/verify_alignment.py does, and it
    keeps the full Window); attention_mask is always `[1] * len(input_ids)`
    before padding (padding happens in collate, not here), so storing it is
    storing a value fully implied by input_ids' length. Dropping both cuts
    projected full-corpus memory from ~37 GB to ~11 GB with no change to
    what the model is trained on.
    """

    input_ids: list[int]
    labels: list[int]
    # False for a row from one of the corpora added to correct the teacher --
    # the out-of-domain negatives and the tabular/byline PERSON augmentation.
    # Their gold labels are constructed and correct; the teacher's logits on
    # them are the very defects this run is fixing, in both directions. See
    # KD_EXCLUDED_SOURCE_PREFIXES and the module docstring.
    kd: bool = True


class PIIDataset(Dataset):
    """Eagerly tokenizes and windows every row up front, storing only the
    lean TrainingExample per window (see its docstring for why). Still
    trades memory for simplicity at very large scale (a Python list per
    2M-row corpus, rather than a memory-mapped on-disk cache) -- the
    smoke-test report states the measured size so that trade stays visible.
    """

    def __init__(self, rows: list[dict], tokenizer, max_length: int, stride: int):
        self.examples: list[TrainingExample] = []
        for row in rows:
            kd = not str(row.get("source", "")).startswith(KD_EXCLUDED_SOURCE_PREFIXES)
            for w in tokenize_and_align(row["text"], row["entities"], tokenizer, max_length, stride):
                self.examples.append(
                    TrainingExample(input_ids=w.input_ids, labels=w.labels, kd=kd)
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> TrainingExample:
        return self.examples[idx]


def collate(batch, pad_token_id: int) -> dict[str, torch.Tensor]:
    """Works on anything with .input_ids/.labels (TrainingExample for
    training, align.Window for evaluation) -- attention_mask is always
    reconstructed as all-1s up to each example's own length, never read
    from the input, since that is what it always is pre-padding.

    `kd_mask` is emitted only when at least one example in the batch
    actually carries a `kd` attribute (TrainingExample does; align.Window,
    used for evaluation, does not). Emitting it unconditionally would put a
    key in the eval batches that `model(**batch)` would reject."""
    max_len = max(len(ex.input_ids) for ex in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
    for i, ex in enumerate(batch):
        n = len(ex.input_ids)
        input_ids[i, :n] = torch.tensor(ex.input_ids, dtype=torch.long)
        attention_mask[i, :n] = 1
        labels[i, :n] = torch.tensor(ex.labels, dtype=torch.long)
    out = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    if any(hasattr(ex, "kd") for ex in batch):
        out["kd_mask"] = torch.tensor(
            [bool(getattr(ex, "kd", True)) for ex in batch], dtype=torch.bool
        )
    return out


class BucketBatchSampler(Sampler):
    """Length-bucketed batching: each epoch, shuffle all window indices,
    chop into pools of `pool_size` windows, sort each pool by length, slice
    each sorted pool into batches of `batch_size`, then shuffle the
    resulting *batch* order (so training doesn't see a strict
    ascending-length curriculum, only length-homogeneous individual
    batches in a randomized order).

    Why this and not a smaller max_length: measured on this corpus's own
    20,000-row sample at batch_size=32, max_length=256 with plain shuffled
    batching pads to ~2.81x real token count; the same data with this
    sampler measured ~1.04x. That is a larger win than dropping
    max_length to 128 (~1.68x padding, and it raises the share of rows
    split across a window from ~2.6% to ~25% -- see align.py's stride
    docstring for why a higher window-split rate compounds the risk of an
    entity exceeding the stride). Bucketing attacks the actual cause
    (padding waste from mixed-length batches), not a proxy for it.
    """

    def __init__(self, lengths: list[int], batch_size: int, pool_size: int, seed: int):
        self.lengths = lengths
        self.batch_size = batch_size
        self.pool_size = max(pool_size, batch_size)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rng = random.Random(self.seed * 1_000_003 + self.epoch)
        indices = list(range(len(self.lengths)))
        rng.shuffle(indices)

        batches: list[list[int]] = []
        for pool_start in range(0, len(indices), self.pool_size):
            pool = indices[pool_start : pool_start + self.pool_size]
            pool.sort(key=lambda i: self.lengths[i])
            for b_start in range(0, len(pool), self.batch_size):
                batches.append(pool[b_start : b_start + self.batch_size])

        rng.shuffle(batches)
        yield from batches


def build_eval_examples(rows: list[dict], tokenizer, max_length: int, stride: int) -> list[tuple[str, Window]]:
    """Like PIIDataset, but keeps the full align.Window (offset_mapping
    included) paired with each row's language, since evaluation needs a
    per-language breakdown and seqeval scores whole tag sequences, not
    reconstructed spans. Evaluation sets are far smaller than the training
    set, so the memory argument that strips Window down to TrainingExample
    in PIIDataset does not apply here.
    """
    examples: list[tuple[str, Window]] = []
    for row in rows:
        for w in tokenize_and_align(row["text"], row["entities"], tokenizer, max_length, stride):
            examples.append((row["language"], w))
    return examples


@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    eval_examples: list[tuple[str, Window]],
    device: str,
    batch_size: int,
) -> dict:
    """Entity-level span precision/recall/F1 via seqeval -- not token
    accuracy, which flatters everything given how imbalanced O is against
    12 entity types. Reported overall, per platform type, and per language.
    seqeval decodes spans from BIO tag sequences itself, so this needs only
    per-window gold vs. predicted tag strings, not span reconstruction.
    """
    from seqeval.metrics import classification_report

    model.eval()
    all_gold: list[list[str]] = []
    all_pred: list[list[str]] = []
    all_lang: list[str] = []

    for start in range(0, len(eval_examples), batch_size):
        chunk = eval_examples[start : start + batch_size]
        langs = [lang for lang, _ in chunk]
        windows = [w for _, w in chunk]
        batch = collate(windows, tokenizer.pad_token_id)
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        preds = logits.argmax(-1).cpu().tolist()
        gold = batch["labels"].cpu().tolist()
        for i in range(len(windows)):
            gold_seq, pred_seq = [], []
            for gid, pid in zip(gold[i], preds[i]):
                if gid == IGNORE_INDEX:
                    continue
                gold_seq.append(ID2LABEL[gid])
                pred_seq.append(ID2LABEL[pid])
            if not gold_seq:
                continue
            all_gold.append(gold_seq)
            all_pred.append(pred_seq)
            all_lang.append(langs[i])
    model.train()

    if not all_gold:
        return {"overall": None, "per_type": {}, "per_language": {}, "n_windows": 0}

    report = classification_report(all_gold, all_pred, output_dict=True, zero_division=0)
    overall = report.get("micro avg", {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0})
    per_type = {k: v for k, v in report.items() if k in PLATFORM_TYPES}

    by_lang: dict[str, list[int]] = defaultdict(list)
    for i, lang in enumerate(all_lang):
        by_lang[lang].append(i)
    per_language = {}
    for lang, idxs in by_lang.items():
        g = [all_gold[i] for i in idxs]
        p = [all_pred[i] for i in idxs]
        lang_report = classification_report(g, p, output_dict=True, zero_division=0)
        lm = lang_report.get("micro avg", {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0})
        per_language[lang] = {**lm, "n_windows": len(idxs)}

    return {"overall": overall, "per_type": per_type, "per_language": per_language, "n_windows": len(all_gold)}


def truncate_encoder_layers(model, keep: int) -> list[int]:
    """Keep `keep` evenly-spaced encoder layers, dropping the rest.

    Evenly spaced from layer 0 (12 -> 0,2,4,6,8,10), which is the standard
    DistilBERT/TinyBERT choice and the same one
    reports/latency-investigation.md section 5 made for its 3/6/9-layer probe
    graphs -- so the 18.4 ms figure measured there is a measurement of *this*
    architecture and not of a differently-truncated one.

    **Which six, measured rather than argued.** There is a tempting argument
    for keeping the *last* layer instead (0,2,4,7,9,11): the classifier head
    was trained on layer 11's output, so a student whose head still receives
    layer-11 features should start closer to the teacher. It is wrong. Over
    1,530 windows of `data-merged`, untrained, straight after truncation:

    | layers kept | cross-entropy vs gold | KL to the teacher |
    |---|---|---|
    | `[0,2,4,6,8,10]` (this function)  | **0.3489** | **0.3443** |
    | `[0,2,4,7,9,11]` (last kept)      | 0.5383 | 0.5342 |
    | `[0,1,2,3,4,5]` (first six)       | 2.5642 | 2.5522 |
    | `[6,7,8,9,10,11]` (last six)      | 2.7112 | 2.7146 |
    | the 12-layer teacher itself       | 0.0071 | 0.0000 |

    Uniform stride beats keeping the endpoint by a wide margin, and both beat
    a contiguous half by an order of magnitude. The head adapts during
    training; the residual stream's spacing apparently does not.

    `config.num_hidden_layers` is updated too. If it were not, the saved
    checkpoint would declare 12 layers while carrying 6, and the ONNX export
    and the Rust service both read config rather than counting modules.
    """
    total = len(model.bert.encoder.layer)
    if keep >= total:
        raise SystemExit(
            f"--student-layers {keep} is not fewer than the {total} layers the "
            f"initialisation checkpoint has -- nothing to truncate."
        )
    indices = sorted({round(i * total / keep) for i in range(keep)})
    if len(indices) != keep:
        # Rounding collided: two i values landed on one layer. Building the
        # student anyway would silently produce a model shallower than asked
        # for, which the config would then faithfully record -- so the wrong
        # depth would look intentional everywhere downstream.
        raise SystemExit(
            f"--student-layers {keep} of {total} rounds to {len(indices)} distinct "
            f"layer indices ({indices}); pick a depth that divides more evenly."
        )
    kept = torch.nn.ModuleList([model.bert.encoder.layer[i] for i in indices])
    model.bert.encoder.layer = kept
    model.config.num_hidden_layers = keep
    return indices


def kd_loss_fn(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    kd_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Temperature-scaled KL(teacher || student) over the tokens that carry a
    real label, on the rows KD applies to.

    Two maskings, and both matter:

    - `labels != IGNORE_INDEX` drops padding and sub-word continuations. A KD
      term averaged over padding would be dominated by positions neither model
      is asked about, and would shrink as batches get more padded -- which is
      to say it would depend on the batch sampler.
    - `kd_mask` drops the out-of-domain negative rows entirely. The teacher
      hallucinates on them (reports/spurious-address-spans.md); matching its
      logits there would teach the student the defect this run exists to
      remove.

    Returns a 0-d tensor; zero when no position in the batch qualifies, which
    happens for a batch that is all negatives.
    """
    valid = (labels != IGNORE_INDEX) & kd_mask.unsqueeze(1)
    if not valid.any():
        return student_logits.new_zeros(())
    s = torch.log_softmax(student_logits[valid] / temperature, dim=-1)
    t = torch.softmax(teacher_logits[valid] / temperature, dim=-1)
    # `* T^2` is Hinton et al.'s scaling: soft-target gradients shrink as
    # 1/T^2, so without it --kd-temperature would silently double as a
    # learning-rate knob.
    return torch.nn.functional.kl_div(s, t, reduction="batchmean") * (temperature ** 2)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data", help="directory containing train/validation/test.jsonl")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="reservoir-sample this many training rows (smoke test). Omit for the full run over all rows.",
    )
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-grad-norm", type=float, default=1.0, help="gradient clipping norm; standard for this scale of fine-tune")
    ap.add_argument("--max-length", type=int, default=256, help="token window size; must match the service's inference windowing")
    ap.add_argument("--stride", type=int, default=32, help="overlap between consecutive windows; must exceed the longest entity's token length -- see align.py's docstring")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default="runs/smoke", help="where to save the checkpoint; never committed (see .gitignore)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"], help="auto resolves cuda -> mps -> cpu, in that order; an explicit choice fails loudly at startup (not deep in the training loop) if unavailable -- see pick_device()")
    ap.add_argument("--num-workers", type=int, default=None, help="DataLoader worker processes; default auto-sizes to the machine (see default_num_workers()) -- pass 0 to force single-process loading")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "tiny end-to-end run (forces --limit %d if not set, --epochs 1, --skip-eval, "
            "--warmup-steps-excluded <= %d) that prints a device/throughput PASS-FAIL verdict "
            "in about two minutes -- run this first on any new machine (especially a rented "
            "GPU) before committing to a real run" % (SMOKE_LIMIT, SMOKE_WARMUP_STEPS_EXCLUDED)
        ),
    )
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--log-every", type=int, default=50, help="steps between progress log lines")
    ap.add_argument("--warmup-steps-excluded", type=int, default=5, help="batches excluded from the start of throughput measurement (first-call/kernel-compile overhead)")
    ap.add_argument("--report-out", default=None, help="path to write a markdown measurement report")
    ap.add_argument("--no-save", action="store_true", help="skip writing the checkpoint to --output-dir")
    ap.add_argument("--no-bucketing", action="store_true", help="use plain shuffled batching instead of length-bucketed batching (for comparison only)")
    ap.add_argument("--bucket-pool-size", type=int, default=1024, help="pool size for length-bucketed batching; see BucketBatchSampler")
    ap.add_argument("--skip-eval", action="store_true", help="skip end-of-epoch validation (e.g. for pure throughput micro-benchmarks against a data dir with no validation.jsonl)")
    ap.add_argument("--eval-limit", type=int, default=None, help="reservoir-sample this many validation rows for end-of-epoch evaluation; omit for the full validation split")

    # --- out-of-domain negatives -------------------------------------------
    ap.add_argument(
        "--extra-data-dir", action="append", default=None,
        help="additional dataset directory (same train/validation/test.jsonl layout) mixed "
             "into training; repeatable. Used for training/data-negatives -- see "
             "build_negatives.py. Kept as a separate directory rather than merged into "
             "--data-dir on disk so the negative ratio is a run-time decision that shows up "
             "in the command line, and so the 482MB name-bearing corpus never has to be "
             "rewritten to change it.",
    )
    ap.add_argument(
        "--extra-limit", type=int, action="append", default=None,
        help="reservoir-sample this many rows from an --extra-data-dir's train split. Give it "
             "once to apply one limit to every extra directory, or once per --extra-data-dir "
             "to give each its own -- the negatives and the PERSON-shape augmentation need "
             "different counts and they pull in opposite directions, so one shared limit "
             "would make the balance between them unsettable. "
             "reports/distillation-plan.md records the ablation that chose the values.",
    )
    ap.add_argument(
        "--extra-eval", action="store_true",
        help="also mix each --extra-data-dir's validation split into end-of-epoch evaluation. "
             "Off by default: seqeval scores entity spans, and a corpus of all-O rows "
             "contributes no gold spans, so including it changes precision (spurious spans "
             "now have somewhere to come from) while leaving recall's denominator alone. "
             "That is a genuinely different metric and it is opt-in rather than silent.",
    )

    # --- distillation -------------------------------------------------------
    ap.add_argument(
        "--init-from", default=None,
        help="initialise from this checkpoint instead of the base model. For the student "
             "run this is the teacher (runs/full-merged), whose fine-tuned weights are a "
             "far better start than the un-finetuned base.",
    )
    ap.add_argument(
        "--student-layers", type=int, default=None,
        help="keep only this many encoder layers, chosen evenly spaced across the "
             "initialisation checkpoint's stack (12 -> 6 keeps 0,2,4,6,8,10). This is the "
             "same truncation the 18.4 ms latency probe used "
             "(reports/latency-investigation.md section 5); here it is the student's "
             "*initialisation*, not the whole method -- see --teacher-checkpoint.",
    )
    ap.add_argument(
        "--teacher-checkpoint", default=None,
        help="a full-depth checkpoint whose logits the student is trained to match, in "
             "addition to the gold labels. Omit for plain truncation + fine-tuning.",
    )
    ap.add_argument(
        "--kd-alpha", type=float, default=0.5,
        help="weight on the KD term; the hard-label cross-entropy gets (1 - alpha). 0.5 "
             "keeps the gold labels at full authority on the negatives (where KD is masked "
             "off entirely) while letting the teacher carry half the signal elsewhere.",
    )
    ap.add_argument(
        "--kd-temperature", type=float, default=2.0,
        help="softmax temperature for both sides of the KD term; the term is scaled by T^2 "
             "so its gradient magnitude does not depend on T.",
    )
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.smoke:
        if args.limit is None:
            args.limit = SMOKE_LIMIT
        if args.extra_data_dir and not args.extra_limit:
            args.extra_limit = [SMOKE_EXTRA_LIMIT]
        args.epochs = 1
        args.skip_eval = True
        args.warmup_steps_excluded = min(args.warmup_steps_excluded, SMOKE_WARMUP_STEPS_EXCLUDED)
        args.log_every = 1
        print(f"--smoke: forcing limit={args.limit} epochs=1 skip_eval=True "
              f"warmup_steps_excluded={args.warmup_steps_excluded} log_every=1"
              + (f" extra_limit={args.extra_limit[0]}/dir" if args.extra_data_dir else ""))

    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"device: {device} (requested: {args.device})")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    num_workers = args.num_workers if args.num_workers is not None else default_num_workers()
    print(f"num_workers: {num_workers} (requested: {args.num_workers})")

    data_dir = Path(args.data_dir)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    assert tokenizer.is_fast, f"{TOKENIZER_NAME} did not load a fast tokenizer -- offset_mapping unavailable"

    print("Loading training rows...")
    t0 = time.time()
    train_rows = load_rows(data_dir / "train.jsonl", args.limit, args.seed)
    n_primary_rows = len(train_rows)
    print(f"  {n_primary_rows} rows from {data_dir}"
          + (f" (reservoir-sampled from the full file, seed {args.seed})" if args.limit else " (full split)"))

    extra_dirs = args.extra_data_dir or []
    extra_limits = args.extra_limit or []
    if len(extra_limits) > 1 and len(extra_limits) != len(extra_dirs):
        raise SystemExit(
            f"--extra-limit given {len(extra_limits)} times for {len(extra_dirs)} "
            "--extra-data-dir values. Give it once (applies to all) or once per directory."
        )
    n_extra_rows = 0
    for i, extra in enumerate(extra_dirs):
        extra_path = Path(extra) / "train.jsonl"
        if not extra_path.exists():
            raise SystemExit(f"--extra-data-dir {extra}: no train.jsonl there")
        limit = None
        if extra_limits:
            limit = extra_limits[i] if len(extra_limits) > 1 else extra_limits[0]
        rows = load_rows(extra_path, limit, args.seed)
        n_extra_rows += len(rows)
        train_rows.extend(rows)
        print(f"  {len(rows)} rows from {extra}"
              + (f" (reservoir-sampled to {limit}, seed {args.seed})" if limit else " (full split)"))
    if n_extra_rows:
        # The ratio is printed rather than left to be recomputed from two
        # numbers in different log lines: it is the single number the
        # negatives ablation is about, and a run whose ratio is not what was
        # intended is a run whose result means nothing.
        print(f"  extra/primary ratio: {n_extra_rows / max(n_primary_rows, 1) * 100:.2f}% "
              f"({n_extra_rows} of {n_primary_rows + n_extra_rows} total rows)")
        # Interleave rather than append: appended rows would all land in the
        # same length buckets and, with bucketed batching, in the same
        # batches -- so the model would see whole batches of nothing-but-
        # negatives followed by whole batches of nothing-but-positives.
        random.Random(args.seed).shuffle(train_rows)
    load_time = time.time() - t0
    n_train_rows = len(train_rows)
    print(f"  {n_train_rows} training rows total in {load_time:.1f}s")

    print("Tokenizing and windowing...")
    t0 = time.time()
    train_ds = PIIDataset(train_rows, tokenizer, args.max_length, args.stride)
    tokenize_time = time.time() - t0
    del train_rows  # raw parsed-JSON rows are not needed past this point (finding 3)
    n_windows = len(train_ds)
    total_real_tokens = sum(sum(1 for l in ex.labels if l != IGNORE_INDEX) for ex in train_ds.examples)
    print(f"  {n_windows} windows from {n_train_rows} rows in {tokenize_time:.1f}s "
          f"({n_windows / max(n_train_rows, 1):.3f} windows/row, "
          f"{total_real_tokens / max(n_windows, 1):.1f} real (non-special) tokens/window avg)")

    eval_examples: list[tuple[str, Window]] = []
    if not args.skip_eval:
        print("Loading validation rows...")
        t0 = time.time()
        eval_rows = load_rows(data_dir / "validation.jsonl", args.eval_limit, args.seed)
        if args.extra_eval:
            for extra in (args.extra_data_dir or []):
                extra_val = Path(extra) / "validation.jsonl"
                if extra_val.exists():
                    eval_rows.extend(load_rows(extra_val, args.eval_limit, args.seed))
        eval_examples = build_eval_examples(eval_rows, tokenizer, args.max_length, args.stride)
        print(f"  {len(eval_rows)} validation rows -> {len(eval_examples)} windows in {time.time()-t0:.1f}s")

    init_from = args.init_from or BASE_MODEL
    model = BertForTokenClassification.from_pretrained(
        init_from, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )
    kept_layers: list[int] | None = None
    if args.student_layers is not None:
        n_before = len(model.bert.encoder.layer)
        kept_layers = truncate_encoder_layers(model, args.student_layers)
        print(f"student: {n_before} -> {args.student_layers} encoder layers, "
              f"keeping indices {kept_layers} of the {init_from} stack")
        if args.init_from is None:
            print("  NOTE: --student-layers without --init-from truncates the *base* model, "
                  "which throws away nothing that was ever trained. That is almost certainly "
                  "not what was meant.")
    model.to(device)

    teacher = None
    if args.teacher_checkpoint:
        teacher = BertForTokenClassification.from_pretrained(args.teacher_checkpoint)
        teacher.to(device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)
        if teacher.config.num_labels != len(LABELS):
            raise SystemExit(
                f"teacher {args.teacher_checkpoint} has {teacher.config.num_labels} labels, "
                f"this run has {len(LABELS)} -- their logits are not comparable."
            )
        if list(teacher.config.id2label.values()) != LABELS:
            raise SystemExit(
                f"teacher {args.teacher_checkpoint}'s label order differs from this run's. "
                "Distilling logits across two different label orderings would train the "
                "student to predict the wrong types, silently."
            )
        n_teacher = sum(p.numel() for p in teacher.parameters())
        print(f"teacher: {args.teacher_checkpoint}, "
              f"{teacher.config.num_hidden_layers} layers, {n_teacher:,} parameters, frozen, eval mode")
        print(f"  KD: alpha={args.kd_alpha} temperature={args.kd_temperature}; "
              f"masked off on rows whose source starts with any of "
              f"{KD_EXCLUDED_SOURCE_PREFIXES}")

    # Checked by inspecting an actual parameter's .device, not by trusting
    # that .to(device) did what it says -- this is the check that catches
    # the pod's CUDA-torch-silently-replaced-by-a-CPU-wheel failure mode
    # even if pick_device()'s pre-flight guard were ever bypassed (e.g. a
    # future caller invoking main() logic without going through argparse).
    # Unconditional, not just under --smoke: "no unwrap-equivalent silent
    # failures" applies to every run, not only the two-minute one.
    param_device = next(model.parameters()).device
    params_on_device = param_device.type == torch.device(device).type
    if not params_on_device:
        raise SystemExit(
            f"model.to({device!r}) did not move parameters there -- "
            f"next(model.parameters()).device is {param_device}. Refusing to train "
            f"silently on the wrong device."
        )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {BASE_MODEL} -> BertForTokenClassification, {n_params:,} parameters, {len(LABELS)} labels")
    print(f"  parameters confirmed on {param_device} (checked via next(model.parameters()).device, not the flag)")

    # functools.partial, not a lambda: DataLoader workers (--num-workers > 0)
    # are spawned via the operating system's default multiprocessing context, which
    # is "spawn" on macOS (this dev machine) -- spawn pickles collate_fn to
    # send it to each worker, and a lambda closure is not picklable. A
    # module-level function bound with partial is. (Linux, the GPU
    # machine, defaults to "fork" instead, which wouldn't have needed this,
    # but the fix is free and makes the script portable either way.)
    collate_fn = functools.partial(collate, pad_token_id=tokenizer.pad_token_id)
    pin_memory = device == "cuda"

    if args.no_bucketing:
        loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=num_workers,
            pin_memory=pin_memory, persistent_workers=num_workers > 0,
        )
        batch_sampler = None
        total_steps = len(loader) * args.epochs
    else:
        lengths = [len(ex.input_ids) for ex in train_ds.examples]
        batch_sampler = BucketBatchSampler(lengths, args.batch_size, args.bucket_pool_size, args.seed)
        loader = DataLoader(
            train_ds, batch_sampler=batch_sampler,
            collate_fn=collate_fn, num_workers=num_workers,
            pin_memory=pin_memory, persistent_workers=num_workers > 0,
        )
        total_steps = len(batch_sampler) * args.epochs

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    process = psutil.Process()
    peak_rss_bytes = process.memory_info().rss
    peak_mps_alloc_bytes = 0
    peak_cuda_alloc_bytes = 0

    measured_tokens = 0
    measured_padded_tokens = 0
    measured_wall_time = 0.0
    measured_steps = 0
    first_loss = None
    last_loss = None

    # Warnings captured across the whole run (mps-fallback-to-CPU notices),
    # accumulated from short-lived catch_warnings blocks entered fresh each
    # step -- see the loop below for why it's scoped that way rather than
    # wrapping the whole double loop in one block.
    caught: list[warnings.WarningMessage] = []

    eval_history: list[dict] = []

    model.train()
    global_step = 0

    train_start = time.time()
    for epoch in range(args.epochs):
        if batch_sampler is not None:
            batch_sampler.set_epoch(epoch)
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            step_start = time.time()
            optimizer.zero_grad()
            # catch_warnings(record=True) mutates warnings' global filter
            # state for its duration and is documented as not thread-safe;
            # a previous review flagged the old code (which wrapped the
            # *entire* double for-loop, including `for batch in loader`,
            # in one such block) as unsafe once --num-workers/pin_memory
            # can put a background pin-memory thread in this same process
            # during that window. Scoping the block to only the
            # synchronous, single-threaded compute below -- entered fresh
            # every step, never spanning a loader fetch -- removes that
            # exposure rather than relying on today's settings (pin_memory
            # only true for cuda) to make it a non-issue by accident.
            with warnings.catch_warnings(record=True) as step_caught:
                warnings.simplefilter("always")
                kd_mask = batch.pop("kd_mask", None)
                out = model(**batch)
                loss = out.loss
                ce_val = float(loss.detach())
                kd_val = 0.0
                if teacher is not None:
                    if kd_mask is None:
                        kd_mask = torch.ones(
                            batch["input_ids"].shape[0], dtype=torch.bool, device=device
                        )
                    with torch.no_grad():
                        teacher_logits = teacher(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                        ).logits
                    kd = kd_loss_fn(
                        out.logits, teacher_logits, batch["labels"], kd_mask,
                        args.kd_temperature,
                    )
                    kd_val = float(kd.detach())
                    loss = (1.0 - args.kd_alpha) * loss + args.kd_alpha * kd
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            caught.extend(step_caught)
            scheduler.step()
            if device == "mps":
                torch.mps.synchronize()
            elif device == "cuda":
                torch.cuda.synchronize()
            step_end = time.time()

            loss_val = loss.item()
            if first_loss is None:
                first_loss = loss_val
            last_loss = loss_val

            n_tokens = int(batch["attention_mask"].sum().item())
            n_padded_tokens = batch["input_ids"].shape[0] * batch["input_ids"].shape[1]
            if global_step >= args.warmup_steps_excluded:
                measured_tokens += n_tokens
                measured_padded_tokens += n_padded_tokens
                measured_wall_time += step_end - step_start
                measured_steps += 1

            rss = process.memory_info().rss
            peak_rss_bytes = max(peak_rss_bytes, rss)
            if device == "mps":
                peak_mps_alloc_bytes = max(peak_mps_alloc_bytes, torch.mps.current_allocated_memory())
            elif device == "cuda":
                peak_cuda_alloc_bytes = max(peak_cuda_alloc_bytes, torch.cuda.memory_allocated())

            if global_step % args.log_every == 0:
                extra = f" (ce {ce_val:.4f} kd {kd_val:.4f})" if teacher is not None else ""
                print(f"  epoch {epoch} step {global_step}/{total_steps} loss {loss_val:.4f}{extra}")
            global_step += 1

        if eval_examples:
            print(f"Evaluating after epoch {epoch}...")
            t0 = time.time()
            metrics = evaluate(model, tokenizer, eval_examples, device, args.batch_size)
            metrics["epoch"] = epoch
            metrics["eval_time_s"] = time.time() - t0
            eval_history.append(metrics)
            if metrics["overall"]:
                o = metrics["overall"]
                print(f"  epoch {epoch} eval: P={o['precision']:.4f} R={o['recall']:.4f} "
                      f"F1={o['f1-score']:.4f} (n={o['support']}, {metrics['eval_time_s']:.1f}s)")
                for t in ("PERSON", "STREET_ADDRESS"):
                    if t in metrics["per_type"]:
                        pt = metrics["per_type"][t]
                        print(f"    {t}: P={pt['precision']:.4f} R={pt['recall']:.4f} "
                              f"F1={pt['f1-score']:.4f} (n={pt['support']})")
    wall_time_total = time.time() - train_start

    mps_fallback_messages = sorted(
        {str(w.message) for w in caught if "mps" in str(w.message).lower()}
    )

    tokens_per_sec = measured_tokens / measured_wall_time if measured_wall_time > 0 else 0.0
    padding_factor = measured_padded_tokens / measured_tokens if measured_tokens > 0 else 0.0

    print()
    print(f"total wall time (incl. first {args.warmup_steps_excluded} warmup steps, excl. eval): {wall_time_total:.1f}s")
    print(f"measured steps: {measured_steps}, measured tokens (real+special, excl. pad): {measured_tokens}")
    print(f"tokens/sec (measured, warmup excluded): {tokens_per_sec:.1f}")
    print(f"padding factor (padded tokens actually computed / real+special tokens counted): {padding_factor:.3f}x")
    print(f"peak RSS: {peak_rss_bytes / 1e9:.2f} GB")
    if device == "mps":
        print(f"peak MPS allocated: {peak_mps_alloc_bytes / 1e9:.2f} GB")
    if device == "cuda":
        print(f"peak CUDA allocated: {peak_cuda_alloc_bytes / 1e9:.2f} GB")
    print(f"MPS fallback-to-CPU warnings observed: {len(mps_fallback_messages)}")
    for m in mps_fallback_messages:
        print(f"  - {m}")
    print(f"loss: first={first_loss:.4f} last={last_loss:.4f}")

    if args.smoke:
        # PASS/FAIL a human can read at a glance, tied to the objective
        # checks above rather than to whether loss visibly dropped over a
        # handful of steps (too noisy a signal over ~8 measured steps to
        # gate on) -- the point of --smoke is "does this box actually
        # train on the device it claims to," not a quality bar.
        #
        # params_on_device is included for completeness of what this
        # verdict claims to check, but a False value can never actually
        # reach this line: the assertion right after model.to(device),
        # above, already raises SystemExit the moment params_on_device is
        # False. So "VERDICT: FAIL" printed here never has that as its
        # cause -- if params ended up on the wrong device, the run stops
        # with a SystemExit and a message before it ever gets to print a
        # verdict at all, which is louder than FAIL and is the actual
        # failure signature to look for.
        smoke_pass = (
            params_on_device
            and measured_steps > 0
            and tokens_per_sec > 0
            and first_loss is not None
            and last_loss is not None
        )
        print()
        print("=" * 60)
        print("SMOKE TEST VERDICT")
        print(f"  requested device: {args.device}  ->  selected: {device}")
        if device == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print(f"  model parameters on {device}: {params_on_device} "
              f"(next(model.parameters()).device = {param_device})")
        print(f"  measured steps: {measured_steps}, tokens/sec: {tokens_per_sec:.1f}")
        print(f"  loss: first={first_loss:.4f} last={last_loss:.4f}")
        print(f"  VERDICT: {'PASS' if smoke_pass else 'FAIL'}")
        print("=" * 60)

    if not args.no_save:
        # The paired tokenizer's pad token id is 1 (XLM-R), while the base
        # checkpoint's own config.json says pad_token_id=0 (BERT
        # convention). Harmless during this training loop because
        # attention_mask is always passed explicitly, but the saved
        # checkpoint ships a config that disagrees with its own tokenizer
        # -- and the ONNX export and the Rust service will read config,
        # not re-derive it from the tokenizer. Fix it at the one place it
        # can silently drift: save time.
        model.config.pad_token_id = tokenizer.pad_token_id
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        print(f"checkpoint saved to {out_dir} (gitignored -- not committed)")

    if args.report_out:
        write_report(
            Path(args.report_out),
            args=args,
            device=device,
            n_train_rows=n_train_rows,
            n_windows=n_windows,
            total_real_tokens=total_real_tokens,
            load_time=load_time,
            tokenize_time=tokenize_time,
            n_params=n_params,
            total_steps=total_steps,
            measured_steps=measured_steps,
            measured_tokens=measured_tokens,
            measured_padded_tokens=measured_padded_tokens,
            measured_wall_time=measured_wall_time,
            tokens_per_sec=tokens_per_sec,
            padding_factor=padding_factor,
            wall_time_total=wall_time_total,
            peak_rss_bytes=peak_rss_bytes,
            peak_mps_alloc_bytes=peak_mps_alloc_bytes,
            peak_cuda_alloc_bytes=peak_cuda_alloc_bytes,
            mps_fallback_messages=mps_fallback_messages,
            first_loss=first_loss,
            last_loss=last_loss,
            eval_history=eval_history,
            num_workers=num_workers,
            kept_layers=kept_layers,
            n_extra_rows=n_extra_rows,
        )


def write_report(path: Path, **m) -> None:
    args = m["args"]
    lines = []
    lines.append("# Training smoke-test measurement\n")
    lines.append(f"Base model: `{BASE_MODEL}` (BertForTokenClassification)")
    lines.append(f"Tokenizer: `{TOKENIZER_NAME}` (fast, XLMRobertaTokenizerFast)")
    lines.append(f"Parameters: {m['n_params']:,}")
    lines.append(f"Labels: {len(LABELS)} ({', '.join(LABELS)})\n")
    lines.append("## Command\n")
    lines.append("```")
    lines.append(
        f"python train.py --limit {args.limit} --epochs {args.epochs} "
        f"--batch-size {args.batch_size} --lr {args.lr} --max-length {args.max_length} "
        f"--stride {args.stride} --seed {args.seed} --device {args.device} "
        f"--num-workers {m['num_workers']} "
        f"--output-dir {args.output_dir} --eval-limit {args.eval_limit} "
        f"--report-out {args.report_out}"
        + (" --no-bucketing" if args.no_bucketing else f" --bucket-pool-size {args.bucket_pool_size}")
        + ("".join(f" --extra-data-dir {d}" for d in (args.extra_data_dir or [])))
        + ("".join(f" --extra-limit {n}" for n in (args.extra_limit or [])))
        + (" --extra-eval" if args.extra_eval else "")
        + (f" --init-from {args.init_from}" if args.init_from else "")
        + (f" --student-layers {args.student_layers}" if args.student_layers is not None else "")
        + (f" --teacher-checkpoint {args.teacher_checkpoint}"
           f" --kd-alpha {args.kd_alpha} --kd-temperature {args.kd_temperature}"
           if args.teacher_checkpoint else "")
        + (" --smoke" if args.smoke else "")
    )
    lines.append("```\n")
    if args.student_layers is not None or args.teacher_checkpoint:
        lines.append("## Distillation\n")
        lines.append(f"- Initialised from: `{args.init_from or BASE_MODEL}`")
        lines.append(f"- Encoder layers kept: {args.student_layers if args.student_layers is not None else 'all'}"
                     + (f" (indices {m['kept_layers']}, evenly spaced)" if m.get("kept_layers") else ""))
        if args.teacher_checkpoint:
            lines.append(f"- Teacher: `{args.teacher_checkpoint}`, frozen, eval mode")
            lines.append(f"- KD alpha {args.kd_alpha}, temperature {args.kd_temperature}, "
                         f"KL over label-carrying tokens only")
            lines.append(
                "- KD masked off on rows whose `source` starts with "
                + " or ".join(f"`{p}`" for p in KD_EXCLUDED_SOURCE_PREFIXES)
                + " -- the teacher hallucinates on the negatives "
                "(reports/spurious-address-spans.md) and misses names in the augmented "
                "shapes (reports/person-misses.md), so its logits are wrong in both "
                "directions on exactly the rows added to correct it")
        else:
            lines.append("- Teacher: none (layer truncation + fine-tuning only)")
        lines.append("")
    if m.get("n_extra_rows"):
        lines.append("## Out-of-domain negatives\n")
        lines.append(f"- Extra directories: {', '.join(args.extra_data_dir or [])}")
        lines.append(f"- Negative rows mixed in: {m['n_extra_rows']:,} of {m['n_train_rows']:,} "
                     f"({m['n_extra_rows'] / max(m['n_train_rows'], 1) * 100:.2f}% of training rows)")
        lines.append("")
    lines.append("## Data\n")
    lines.append(f"- Training rows: {m['n_train_rows']} (limit={args.limit}, reservoir-sampled seed={args.seed})" if args.limit else f"- Training rows: {m['n_train_rows']} (full split)")
    lines.append(f"- Windows after tokenization (max_length={args.max_length}, stride={args.stride}): {m['n_windows']}")
    lines.append(f"- Windows/row: {m['n_windows']/max(m['n_train_rows'],1):.3f}")
    lines.append(f"- Real (non-special) tokens/window, average: {m['total_real_tokens']/max(m['n_windows'],1):.1f}")
    lines.append(f"- Row load time: {m['load_time']:.1f}s")
    lines.append(f"- Tokenize+window time: {m['tokenize_time']:.1f}s\n")
    lines.append("## Throughput\n")
    lines.append(f"- Device: {m['device']}")
    lines.append(f"- Batching: {'plain shuffled' if args.no_bucketing else f'length-bucketed (pool_size={args.bucket_pool_size})'}")
    lines.append(f"- Total steps this run: {m['total_steps']}, measured steps (warmup-excluded): {m['measured_steps']}")
    lines.append(f"- Measured tokens processed (warmup-excluded, real+special tokens per attention_mask.sum() -- excludes pad positions, which cost compute but not this count): {m['measured_tokens']}")
    lines.append(f"- Measured padded tokens actually computed (batch_size * max_len_in_batch, summed over measured steps): {m['measured_padded_tokens']}")
    lines.append(f"- **Padding factor: {m['padding_factor']:.3f}x** (padded / real+special -- 1.0x would mean no padding waste at all)")
    lines.append(f"- Measured wall time for those steps: {m['measured_wall_time']:.2f}s")
    lines.append(f"- **Tokens/sec: {m['tokens_per_sec']:.1f}**")
    lines.append(f"- Total wall time (incl. warmup steps + data loading not counted): {m['wall_time_total']:.1f}s\n")
    lines.append("## Memory\n")
    lines.append(f"- Peak process RSS: {m['peak_rss_bytes']/1e9:.2f} GB")
    if m["device"] == "mps":
        lines.append(f"- Peak MPS allocated (torch.mps.current_allocated_memory, sampled per-step): {m['peak_mps_alloc_bytes']/1e9:.2f} GB")
    if m["device"] == "cuda":
        lines.append(f"- Peak CUDA allocated (torch.cuda.memory_allocated, sampled per-step): {m['peak_cuda_alloc_bytes']/1e9:.2f} GB")
    lines.append(f"- DataLoader workers: {m['num_workers']}")
    lines.append("")
    lines.append("## MPS engagement\n")
    if m["device"] == "mps":
        if m["mps_fallback_messages"]:
            lines.append(f"PYTORCH_ENABLE_MPS_FALLBACK=1 was needed: {len(m['mps_fallback_messages'])} distinct op(s) fell back to CPU.")
            for msg in m["mps_fallback_messages"]:
                lines.append(f"- {msg}")
        else:
            lines.append("No MPS fallback-to-CPU warnings observed -- every op ran natively on MPS (PYTORCH_ENABLE_MPS_FALLBACK=1 was set but not exercised).")
    else:
        lines.append(f"Run was on {m['device']}, not MPS.")
    lines.append("")
    lines.append("## Loss\n")
    lines.append(f"- First measured step loss: {m['first_loss']:.4f}")
    lines.append(f"- Last step loss: {m['last_loss']:.4f}")
    lines.append("")

    lines.append("## Validation (entity-level span P/R/F1, seqeval)\n")
    if not m["eval_history"]:
        lines.append("Evaluation skipped (--skip-eval) or no validation.jsonl found.\n")
    else:
        for ev in m["eval_history"]:
            lines.append(f"### Epoch {ev['epoch']}\n")
            if not ev["overall"]:
                lines.append("No entities in the evaluation sample -- nothing to score.\n")
                continue
            o = ev["overall"]
            lines.append(
                f"Overall (micro): precision={o['precision']:.4f} recall={o['recall']:.4f} "
                f"f1={o['f1-score']:.4f}, over {ev['n_windows']} windows, {o['support']} gold entities "
                f"({ev['eval_time_s']:.1f}s)\n"
            )
            lines.append("**Release-gate types:**\n")
            lines.append("| type | precision | recall | f1 | support |")
            lines.append("|---|---:|---:|---:|---:|")
            for t in ("PERSON", "STREET_ADDRESS"):
                if t in ev["per_type"]:
                    pt = ev["per_type"][t]
                    lines.append(f"| {t} | {pt['precision']:.4f} | {pt['recall']:.4f} | {pt['f1-score']:.4f} | {pt['support']} |")
                else:
                    lines.append(f"| {t} | - | - | - | 0 (none in this eval sample) |")
            lines.append("")
            lines.append("**All types:**\n")
            lines.append("| type | precision | recall | f1 | support |")
            lines.append("|---|---:|---:|---:|---:|")
            for t in sorted(ev["per_type"]):
                pt = ev["per_type"][t]
                lines.append(f"| {t} | {pt['precision']:.4f} | {pt['recall']:.4f} | {pt['f1-score']:.4f} | {pt['support']} |")
            lines.append("")
            lines.append("**Per language** (sorted by support):\n")
            lines.append("| language | precision | recall | f1 | support | windows |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for lang, lm in sorted(ev["per_language"].items(), key=lambda kv: -kv[1]["support"]):
                lines.append(f"| {lang} | {lm['precision']:.4f} | {lm['recall']:.4f} | {lm['f1-score']:.4f} | {lm['support']} | {lm['n_windows']} |")
            lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"\nReport written to {path}")


if __name__ == "__main__":
    main()
