"""Export runs/full-merged to ONNX, quantise it to INT8, and provide the
machinery that proves the exported graph is still the same model.

WHY THIS EXISTS. Everything downstream of this file consumes the ONNX
graph: the pii-detection-model crate, the service wiring, and the release gates
that finally measure PERSON and STREET_ADDRESS. None of them has a PyTorch
runtime to compare against, so if the export changed behaviour, every
number produced after this point is a number about a different model and
nothing later would notice. This module is the only place in the design
where the checkpoint and the graph can be put side by side at all.

WHAT IS PROVEN, AND WHAT EACH CLAIM MEANS. Three different claims get made
about "the same model" and they are not interchangeable:

  * `compare_logits` -> logit closeness. The fp32 graph's raw logits
    against PyTorch's, elementwise. This is the strongest claim and it is
    only made about the fp32 graph, because INT8 logits differ by
    construction.
  * `prediction_agreement` -> argmax agreement, per token position. The
    fraction of real (non-special) token positions where the INT8 graph's
    argmax label equals PyTorch's. Stated over token positions rather than
    over spans on purpose: a span figure hides how many tokens moved
    inside a span that still decoded to the same answer.
  * span agreement -> what the service actually depends on. NOT measured
    here; it is measured by eval_fixture.py over benchmark/cases, once
    with --checkpoint and once with --onnx, so the two go through the same
    decoder and the same scorer.

REAL TEXT, NOT RANDOM TENSORS. `parity_texts` draws from
benchmark/carriers/ (licensed prose, six languages, long enough to force
the sliding-window path) and benchmark/cases/ (the fixture the gates run
on). Random token ids exercise embedding rows that never co-occur in real
text and can hide a quantisation error that only shows on the
distributions the model actually sees. Both trees are read-only here:
another agent owns benchmark/, and nothing in this module writes to it.

WINDOWING IS align.py's, NOT A SECOND IMPLEMENTATION. Every text is
windowed with `align.tokenize_and_align(text, [], tokenizer, MAX_LENGTH,
STRIDE)` -- the same function the trainer used and the same one the Rust
side will be tested against. With `entities=[]` its label pass marks every
real token "O" and every special/pad token IGNORE_INDEX, which is exactly
the mask needed to drop special-token positions from the comparison:
nothing downstream reads them.

VALUES THAT BELONG TO THE ARTIFACT, NOT TO CODE. MAX_LENGTH, STRIDE, the
25-label set and the quantisation config name are properties of *this*
checkpoint (see reports/full-run-merged.md and runs/full-merged/
config.json), not of the service. They are exposed as data here
(`artifact_facts()`) so the packaging's manifest can record them and a retrained
model with different values becomes a new artifact directory rather than a
code change.

THE QUANTISATION IS STATIC, AND THE CALIBRATION CORPUS IS TRAINING DATA.
A statically quantised graph's behaviour is set by the corpus its
activation ranges were measured on, so that corpus is part of the
artifact's identity and `calibration_manifest()` is written into the
manifest beside the digests. It is `training/data-merged/train.jsonl` and
it is never `benchmark/` or anything derived from it -- the benchmark is
what this artifact is scored against, and a graph configured from its own
exam is a measurement that flatters itself. That is enforced
(`_refuse_a_benchmark_corpus`), not merely documented. See
CALIBRATION_CORPUS and QUANTIZATION_CONFIG.

Usage:
    python export_onnx.py                    # export + quantise + report sizes
    python export_onnx.py --skip-fp32        # reuse the staged fp32 graph
    python export_onnx.py --dynamic          # the superseded recipe, for comparison
    python -m pytest tests/test_export_onnx.py
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from align import IGNORE_INDEX, LABELS, tokenize_and_align

_TRAINING_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TRAINING_DIR.parent

CHECKPOINT = _TRAINING_DIR / "runs" / "full-merged"
STAGING_DIR = _REPO_ROOT / "models" / "staging"
STAGING_FP32 = STAGING_DIR / "model_fp32.onnx"


def set_checkpoint(path) -> Path:
    """Repoint every checkpoint-reading function in this module at `path`.

    `--checkpoint` used to be threaded into `export_fp32()` and nowhere
    else, which is a worse failure than not supporting the flag at all.
    `artifact_facts()`, `checkpoint_labels()`, `_tokenizer()` and
    `_torch_model()` all read the module-level `CHECKPOINT`, so exporting a
    student with the flag produced the *student's* graph carrying the
    *teacher's* identity in the manifest -- and a manifest is precisely the
    thing the Rust side trusts without re-deriving it. Nothing raised, and
    the only visible symptom was a `checkpoint:` line naming a model that
    had not been exported.

    Rebinding the global rather than passing the path down every call site
    is deliberate: it is the one place the value lives, it is how the tests
    already repoint this module, and it cannot leave one reader behind the
    way threading a parameter through four functions can.

    The two `lru_cache`d singletons are invalidated here. Without that, a
    process that touched the tokenizer or the model before the switch would
    keep serving the previous checkpoint's -- the same class of silent
    staleness this function exists to remove.
    """
    global CHECKPOINT
    # Resolved, because the default is absolute and `artifact_facts()` calls
    # `CHECKPOINT.relative_to(_REPO_ROOT)`: a relative `--checkpoint
    # runs/student` would otherwise raise there rather than here.
    CHECKPOINT = Path(path).resolve()
    _tokenizer.cache_clear()
    _torch_model.cache_clear()
    return CHECKPOINT


def _resolve_int8() -> Path:
    """The INT8 graph the parity checks read.

    PII_DETECTION_INT8_OVERRIDE exists for exactly one purpose: the mutation check
    in the task brief's step 5, which corrupts a copy of the quantised
    graph and confirms `prediction_agreement` notices. A parity check that
    has never been shown failing on a real defect is not a parity check, and
    there is no other way to hand this module a deliberately broken graph
    without editing the code under test. It is read here and nowhere else,
    it is never written, and nothing in the export path consults it -- an
    override cannot cause a bad artifact to be produced, only to be judged.
    Relative paths resolve against the current directory, so the brief's
    `../models/staging/model_broken.onnx` works from training/.
    """
    override = os.environ.get("PII_DETECTION_INT8_OVERRIDE")
    if override:
        return Path(override)
    return STAGING_DIR / "model.onnx"


STAGING_INT8 = _resolve_int8()

MAX_LENGTH = 256  # must match training; see reports/full-run-merged.md
STRIDE = 32  # must match training; see align.py's docstring
TOKENIZER_NAME = "FacebookAI/xlm-roberta-base"

# The base model (microsoft/Multilingual-MiniLM-L12-H384) ships no
# tokenizer.json of its own, so training used xlm-roberta-base's fast
# tokenizer and saved it beside the checkpoint. Loading it from the
# checkpoint directory rather than from the hub keeps the export offline
# and pins the exact vocabulary the weights were trained against.
NUM_LABELS = len(LABELS)  # 25, derived from align.py rather than hardcoded

OPSET = 17

# ARM because the spec's latency proxy is ARM (Apple M1 Max standing in for
# Graviton) and the deployment target is Fargate. The name goes in the
# manifest, so an x86-tuned artifact is a second artifact directory rather
# than a code change.
#
# STATIC, not dynamic, since 2026-08-22. The export chose dynamic because
# "static quantisation needs a calibration corpus, and a calibration corpus
# of PII text is a data-handling question this work is not the place to
# open". What that deferral bought was a detector whose answer depends on
# how the windower batched: `DynamicQuantizeLinear` computes its activation
# scale at run time from whatever tensor it is handed, and the ONNX
# specification defines that scale as a *scalar* -- one number for the whole
# batch -- so the same window quantises differently depending on which
# windows share its call. Measured at the model over the 2,000-code-point
# corpus: 36 of 200 texts produced a different span set from the grouping
# alone (reports/export.md, addendum B3). For a compliance product that is a
# correctness defect, and no weight-side knob can reach it.
#
# The calibration corpus is the *training* corpus, never benchmark/ --
# see CALIBRATION_CORPUS.
QUANTIZATION_CONFIG = "arm64(is_static=True, per_channel=True, quant_format=QDQ)"

# The dynamic recipe the export shipped, kept so `quantize_int8_dynamic` can still
# build the graph the static one replaced -- the determinism harness
# measures the two side by side, and a comparison that had to reconstruct
# the old recipe from a report rather than run it would not be one.
DYNAMIC_QUANTIZATION_CONFIG = "arm64(is_static=False, per_channel=True)"

# Dynamic axis names written into the graph. A graph frozen at sequence 256
# would pad every window to 256 even for a one-line prompt -- pure wasted
# compute against a 50 ms budget. Naming them plainly also makes the graph
# self-describing: `graph_metadata` reads these names back out of the file
# rather than asserting a constant, so a frozen axis shows up as a missing
# name instead of passing silently.
BATCH_AXIS = "batch"
SEQUENCE_AXIS = "sequence"

LANGUAGES: tuple[str, ...] = ("en", "nl", "de", "fr", "es", "it")
# The live benchmark tree. Read-only here in every sense: another
# workstream authors it, nothing in this module writes to it, and every
# function that reads it takes the root as an argument so a pinned copy can
# be substituted when a figure has to stay reproducible.
BENCHMARK_DIR = _REPO_ROOT / "benchmark"

# How many of each language's parity texts are full carrier documents. The
# rest are fixture cases. Carriers run to a few thousand characters and so
# exercise the sliding-window path; fixture cases are mostly single-window
# prompts. Both paths have to be covered, because a windowing difference
# between the two runtimes would only ever show on the long ones.
CARRIERS_PER_LANGUAGE = 3

# --- the calibration corpus -------------------------------------------------
#
# WHY NOT benchmark/carriers/. Static quantisation needs real activations to
# measure, and `benchmark/carriers/` is the obvious source -- it is prose in
# the six gated languages and `models/latency-corpus.json` is already built
# from it. It is also the wrong source. `benchmark/cases/` is *built from
# those carriers*, and the 649-case benchmark is what decides whether this
# artifact ships. Calibrating there would configure the graph using the data
# it is then scored on. Quantisation calibration only collects activation
# ranges rather than learning the task, so the leak is weaker than training
# on the test set -- and that is exactly the argument that would let it in.
# This project has spent its whole span deleting measurements that flattered
# The detector; this one would be welded into the graph.
#
# The training corpus has none of that problem: the model has already seen
# it, no gate is scored against it, and it is the distribution the weights
# were fitted to, which is what an activation range should describe.
CALIBRATION_CORPUS = _TRAINING_DIR / "data-merged" / "train.jsonl"

# Only the two licensed corpora, never the 1,581 `generated` rows. The
# generated rows are a narrow template set (median 102 characters, one
# sentence shape repeated) built to probe specific types; their activations
# are not the distribution real documents produce, and at 0.2% of the corpus
# they would contribute nothing but a risk of a range set by a template.
CALIBRATION_SOURCES: tuple[str, ...] = ("ai4privacy", "nym")

# Documents per language. 8 x 6 = 48 documents, which window to roughly 150
# windows -- see `calibration_manifest()` for the exact shape of a given
# build, which is what the report quotes rather than this constant.
CALIBRATION_DOCS_PER_LANGUAGE = 8

# Individual training rows run to a median of ~260 characters, so one row is
# a single short window and would exercise neither the full-length window
# nor the short trailing one. Consecutive same-language rows are
# concatenated up to this many code points, which is the length the latency
# corpus and the gate are both specified at, and which windows to 2 or 3 --
# full 256-position windows plus a short tail, the shape the service
# actually feeds.
CALIBRATION_TARGET_CHARACTERS = 2000

# An upper bound on how many rows one document may swallow, so a language
# whose rows are unusually short cannot run off the end of the corpus.
CALIBRATION_MAX_ROWS_PER_DOCUMENT = 32

# ONNX Runtime is pinned to the CPU provider on purpose. CoreML is
# available on this machine and would be a different numerical
# implementation of the same graph, which would make the fp32 comparison a
# statement about CoreML rather than about the export. The deployment
# target is CPU-only Fargate anyway.
_PROVIDERS = ["CPUExecutionProvider"]


# --- artifact facts ---------------------------------------------------------


def checkpoint_labels(checkpoint: Path | None = None) -> list[str]:
    """The checkpoint's own label list, ordered by id, from its config.json.

    Read from the config rather than from align.py so the two can be
    compared. They are two independent statements of the same ordering:
    align.LABELS is what the decoder maps ids back through, and
    config.id2label is what the trained classifier head's output columns
    actually mean. If they ever disagree, every span decodes to the wrong
    type while every other check in this module still passes -- the graph
    is faithful, the logits are right, and the labels are wrong.
    """
    directory = CHECKPOINT if checkpoint is None else Path(checkpoint)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    id2label = config["id2label"]
    return [id2label[str(i)] for i in range(len(id2label))]


def artifact_facts() -> dict:
    """Everything about this export that a manifest has to record.

    Emitted as data rather than baked into Rust constants: a retrained
    model with a different max_length, stride or label set is a new
    artifact directory, not a code change.

    Raises if the checkpoint's config disagrees with align.LABELS about the
    label ordering. Emitting either list while they disagree would put a
    wrong ordering into the manifest, and a manifest is exactly the thing
    the Rust side will trust without re-deriving.
    """
    from_config = checkpoint_labels()
    if from_config != list(LABELS):
        mismatches = [
            i for i, (a, b) in enumerate(zip(LABELS, from_config)) if a != b
        ]
        raise ValueError(
            "label ordering disagrees between align.LABELS and the checkpoint's "
            f"config.json: {len(LABELS)} vs {len(from_config)} labels, "
            f"differing at indices {mismatches}"
        )
    return {
        "checkpoint": str(CHECKPOINT.relative_to(_REPO_ROOT)),
        "tokenizer": TOKENIZER_NAME,
        "max_length": MAX_LENGTH,
        "stride": STRIDE,
        "num_labels": NUM_LABELS,
        "labels": list(LABELS),
        "opset": OPSET,
        "quantization_config": QUANTIZATION_CONFIG,
        "dynamic_axes": [BATCH_AXIS, SEQUENCE_AXIS],
    }


# --- lazily built, expensive singletons -------------------------------------


@functools.lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(CHECKPOINT), use_fast=True)


@functools.lru_cache(maxsize=1)
def _torch_model():
    from transformers import BertForTokenClassification

    model = BertForTokenClassification.from_pretrained(str(CHECKPOINT))
    model.eval()
    return model


@functools.lru_cache(maxsize=4)
def _session(path: Path):
    import onnxruntime as ort

    if not Path(path).is_file():
        raise FileNotFoundError(
            f"no ONNX graph at {path}; run `python export_onnx.py` from training/ first"
        )
    return ort.InferenceSession(str(path), providers=_PROVIDERS)


# --- export -----------------------------------------------------------------


def _rename_dynamic_axes(model) -> None:
    """Rewrite the exporter's symbolic dimension names to BATCH_AXIS /
    SEQUENCE_AXIS everywhere they appear.

    optimum emits `batch_size` and `sequence_length`. Renaming is cosmetic
    to ONNX Runtime (a dim_param is just a symbol) but not cosmetic to the
    reader: `graph_metadata` reports the names it finds in the file, and
    the Rust session will assert the same two facts at load. Every
    occurrence is rewritten -- inputs, outputs and intermediate value_info
    -- because a dim_param is a shape *constraint* shared across tensors,
    and renaming it in only some of them would break that constraint.
    """
    renames = {"batch_size": BATCH_AXIS, "sequence_length": SEQUENCE_AXIS}
    graph = model.graph
    for value in list(graph.input) + list(graph.output) + list(graph.value_info):
        for dim in value.type.tensor_type.shape.dim:
            if dim.dim_param in renames:
                dim.dim_param = renames[dim.dim_param]


def export_fp32(checkpoint: Path, out: Path) -> Path:
    """Export `checkpoint` to a single fp32 ONNX file at `out`.

    optimum's `main_export` with task `token-classification`, which builds
    the input signature from the model config -- that is where
    token_type_ids comes from (config.type_vocab_size is 2) and it is why
    the three inputs are not hand-listed here.
    """
    import onnx
    from optimum.exporters.onnx import main_export

    checkpoint, out = Path(checkpoint), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as staging:
        staging_dir = Path(staging)
        main_export(
            model_name_or_path=str(checkpoint),
            output=staging_dir,
            task="token-classification",
            opset=OPSET,
            do_validation=False,
        )
        exported = staging_dir / "model.onnx"
        if not exported.is_file():
            raise RuntimeError(f"optimum wrote no model.onnx under {staging_dir}")
        model = onnx.load(str(exported))
        _rename_dynamic_axes(model)
        onnx.save(model, str(out))
    return out


# Derived from benchmark/carriers/ by `pii-detection-eval --bin latency
# --build-corpus`, and named here so the guard below can refuse it by name.
# It is the corpus a previous report actually proposed calibrating on
# (reports/export.md, addendum B5: "benchmark/carriers/ prose with
# synthetic identifiers spliced in is exactly what models/latency-corpus.json
# already is"). It is prose from the tree the 649-case benchmark is built
# from, so calibrating on it would set the graph's activation ranges from
# the data the graph is then scored against.
_DERIVED_FROM_BENCHMARK = (_REPO_ROOT / "models" / "latency-corpus.json",)


def _refuse_a_benchmark_corpus(corpus: Path) -> None:
    """Raise rather than calibrate on anything the benchmark is built from.

    Calibration collects activation ranges, not labels, so this is a
    weaker leak than training on the test set -- and that is exactly the
    argument that would let it through. The 649-case fixture is what
    decides whether this artifact ships; a graph configured from the
    fixture's own carriers is a graph tuned on its own exam. The
    alternative costs nothing: the training corpus is larger, is the
    distribution the weights were fitted to, and no gate is scored on it.

    A guard rather than a comment because the wrong path is a plausible
    keystroke -- `--corpus ../models/latency-corpus.json` is a shorter
    thing to type than the right answer, and nothing else downstream would
    notice.
    """
    resolved = Path(corpus).resolve()
    benchmark = BENCHMARK_DIR.resolve()
    inside_benchmark = resolved == benchmark or benchmark in resolved.parents
    if inside_benchmark or resolved in {path.resolve() for path in _DERIVED_FROM_BENCHMARK}:
        raise ValueError(
            f"refusing to calibrate on {resolved.name}: it is benchmark/ or derived from it, "
            "and the benchmark is what this artifact is scored against. "
            f"Calibrate on the training corpus ({CALIBRATION_CORPUS.name}) instead."
        )


@functools.lru_cache(maxsize=2)
def _calibration_rows(corpus: Path) -> dict[str, tuple[int, ...]]:
    """Line numbers, per language, of every row this module may calibrate on.

    One pass over the corpus keeping integers rather than text: the merged
    training split is 730k rows and holding it in memory to select 48
    documents would be a few hundred megabytes for nothing. Rows from a
    source outside CALIBRATION_SOURCES are skipped here, so a later change
    to the source list changes the selection rather than filtering it
    downstream.
    """
    corpus = Path(corpus)
    _refuse_a_benchmark_corpus(corpus)
    if not corpus.is_file():
        raise FileNotFoundError(f"no calibration corpus at {corpus}")
    rows: dict[str, list[int]] = {language: [] for language in LANGUAGES}
    with corpus.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle):
            row = json.loads(line)
            if row.get("source") not in CALIBRATION_SOURCES:
                continue
            bucket = rows.get(row.get("language"))
            if bucket is not None:
                bucket.append(number)
    missing = [language for language, bucket in rows.items() if not bucket]
    if missing:
        raise ValueError(f"calibration corpus {corpus} has no rows for {missing}")
    return {language: tuple(bucket) for language, bucket in rows.items()}


def _calibration_plan(per_language: int, corpus: Path) -> list[tuple[str, str, tuple[int, ...]]]:
    """(identity, language, line numbers) for each calibration document.

    Anchors are spread evenly across each language's rows rather than taken
    from the front: the corpus is grouped, so the first rows of a language
    are one source and one template family. From each anchor forward,
    consecutive same-language rows are concatenated until the document
    reaches CALIBRATION_TARGET_CHARACTERS. Deterministic and index-based --
    a second build selects exactly the same text, which is what makes the
    graph reproducible.
    """
    rows = _calibration_rows(corpus)
    plan: list[tuple[str, str, tuple[int, ...]]] = []
    for language in LANGUAGES:
        numbers = rows[language]
        last_anchor = max(0, len(numbers) - CALIBRATION_MAX_ROWS_PER_DOCUMENT)
        anchors = _spread(list(range(last_anchor + 1)), per_language)
        for index, anchor in enumerate(anchors):
            window = numbers[anchor : anchor + CALIBRATION_MAX_ROWS_PER_DOCUMENT]
            plan.append((f"{language}-{index:03d}", language, tuple(window)))
    return plan


def calibration_documents(
    per_language: int = CALIBRATION_DOCS_PER_LANGUAGE, corpus: Path | None = None
) -> list[tuple[str, str]]:
    """(identity, text) for each calibration document, in a fixed order.

    `identity` is a language and an index -- never text, and never a row
    id that could be joined back to a corpus record. These are what
    `calibration_manifest` publishes.
    """
    corpus = CALIBRATION_CORPUS if corpus is None else Path(corpus)
    plan = _calibration_plan(per_language, corpus)
    wanted: set[int] = set()
    for _identity, _language, numbers in plan:
        wanted.update(numbers)
    texts: dict[int, str] = {}
    with corpus.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle):
            if number in wanted:
                texts[number] = json.loads(line)["text"]
    documents: list[tuple[str, str]] = []
    for identity, _language, numbers in plan:
        parts: list[str] = []
        length = 0
        for number in numbers:
            parts.append(texts[number])
            length += len(texts[number]) + 1
            if length >= CALIBRATION_TARGET_CHARACTERS:
                break
        documents.append((identity, "\n".join(parts)))
    return documents


def calibration_windows(
    per_language: int = CALIBRATION_DOCS_PER_LANGUAGE, corpus: Path | None = None
) -> list:
    """Every calibration document's windows, through align.py's windower.

    At their **own exact length**, never right-padded. Padding is not part
    of the profile the service produces: `pii_detection_model::session::batches`
    groups only windows of equal length, so no `run` call ever carries a
    padded position (see reports/latency-gate.md's static-QDQ addendum).
    Calibrating on padded rows would widen every range with activations
    from positions the service never computes, and a wider range is a
    coarser quantisation of the positions it does.
    """
    tokenizer = _tokenizer()
    return [
        window
        for _identity, text in calibration_documents(per_language, corpus)
        for window in tokenize_and_align(text, [], tokenizer, MAX_LENGTH, STRIDE)
    ]


def calibration_manifest(
    per_language: int = CALIBRATION_DOCS_PER_LANGUAGE, corpus: Path | None = None
) -> dict:
    """Identity and shape of the calibration set: how many, which, how long.

    Carries identifiers, counts and a digest -- never text. The digest is
    SHA-256 over the documents in order, so a report quoting it can be
    checked against a rebuild even though the corpus file is not committed.
    """
    import hashlib

    corpus = CALIBRATION_CORPUS if corpus is None else Path(corpus)
    documents = calibration_documents(per_language, corpus)
    tokenizer = _tokenizer()
    digest = hashlib.sha256()
    lengths: list[int] = []
    window_lengths: list[int] = []
    windows_per_document: list[int] = []
    for _identity, text in documents:
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
        lengths.append(len(text))
        windows = tokenize_and_align(text, [], tokenizer, MAX_LENGTH, STRIDE)
        windows_per_document.append(len(windows))
        window_lengths.extend(len(window.input_ids) for window in windows)
    ordered = sorted(window_lengths)
    return {
        "corpus": str(corpus.relative_to(_REPO_ROOT)),
        "sources": list(CALIBRATION_SOURCES),
        "languages": list(LANGUAGES),
        "documents_per_language": per_language,
        "documents": len(documents),
        "identities": [identity for identity, _text in documents],
        "target_characters": CALIBRATION_TARGET_CHARACTERS,
        "characters": {"min": min(lengths), "max": max(lengths)},
        "windows": len(window_lengths),
        "windows_per_document": {
            str(count): windows_per_document.count(count)
            for count in sorted(set(windows_per_document))
        },
        "full_length_windows": sum(1 for length in window_lengths if length == MAX_LENGTH),
        "window_length": {
            "min": ordered[0],
            "median": ordered[len(ordered) // 2],
            "max": ordered[-1],
        },
        "sha256": digest.hexdigest(),
    }


def quantize_int8_static(
    fp32: Path,
    out: Path,
    per_language: int = CALIBRATION_DOCS_PER_LANGUAGE,
    corpus: Path | None = None,
) -> Path:
    """Statically quantise the fp32 graph to INT8 QDQ at `out`.

    Every activation scale becomes a graph constant, computed here from the
    calibration corpus, so nothing about the tensor handed to ONNX Runtime
    can move it. That is the whole point: the dynamic recipe recomputed
    each scale per call from the batch it was given, which made a window's
    prediction a function of its batch-mates.

    Three choices in the call below are load-bearing and none is the
    quantiser's default:

    * `op_types_to_quantize=["MatMul", "Gather"]` -- MatMul is all of the
      compute; Gather is the 250k x 384 embedding table, which is 384 MiB
      of the fp32 graph's 448 and has to be quantised for the artifact to
      stay small. Nothing else. Quantising `Add` in particular is fatal
      *and silently so*: the attention mask reaches softmax through an
      `Add` of a large negative constant, that constant never appears in
      calibration (no calibration window is padded), so its range collapses
      to zero and the mask stops masking. Measured on a graph built with
      the quantiser's default op list: padded positions attended to
      normally and the worst logit moved 13.8.
    * `OpTypesToExcludeOutputQuantization=["MatMul"]` -- inputs are
      quantised, the result is not. This is what turns each MatMul into
      `MatMulIntegerToFloat` rather than `QLinearMatMul` at session load,
      which is both faster (measured 44.6 ms against 48.9 for a 3-window
      document, interleaved) and more accurate, since the logits are never
      squeezed through 256 levels.
    * `CalibrationMethod.MinMax` -- the range of what was actually seen.
      Percentile and Entropy need every calibration tensor to be the same
      shape, which would mean padding every calibration window to 256 and
      so calibrating on positions the service never feeds.
    """
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    fp32, out = Path(fp32), Path(out)
    if not fp32.is_file():
        raise FileNotFoundError(f"no fp32 graph at {fp32}")
    out.parent.mkdir(parents=True, exist_ok=True)

    feeds = [_window_inputs(window) for window in calibration_windows(per_language, corpus)]
    if not feeds:
        raise ValueError("the calibration corpus produced no windows")

    class _Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self._remaining = iter(feeds)

        def get_next(self):
            return next(self._remaining, None)

        def rewind(self) -> None:
            self._remaining = iter(feeds)

    quantize_static(
        str(fp32),
        str(out),
        calibration_data_reader=_Reader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=["MatMul", "Gather"],
        extra_options={
            "ActivationSymmetric": False,
            "WeightSymmetric": True,
            "OpTypesToExcludeOutputQuantization": ["MatMul"],
        },
    )
    return out


def quantize_int8_dynamic(fp32: Path, out: Path) -> Path:
    """Dynamically quantise the fp32 graph to INT8 at `out`.

    `AutoQuantizationConfig.arm64(is_static=False, per_channel=True)` --
    the recipe the export shipped. **Not what ships now**: it is kept so the
    graph the static export replaced can still be built and measured
    beside it (see DYNAMIC_QUANTIZATION_CONFIG). Its activation scales are
    computed at run time, which is the batch dependence
    `measure_determinism.py` exists to catch.
    """
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    fp32, out = Path(fp32), Path(out)
    if not fp32.is_file():
        raise FileNotFoundError(f"no fp32 graph at {fp32}")
    out.parent.mkdir(parents=True, exist_ok=True)

    quantizer = ORTQuantizer(fp32)
    config = AutoQuantizationConfig.arm64(is_static=False, per_channel=True)
    with tempfile.TemporaryDirectory() as staging:
        staging_dir = Path(staging)
        quantizer.quantize(
            quantization_config=config,
            save_dir=staging_dir,
            file_suffix="quantized",
        )
        produced = sorted(staging_dir.glob("*.onnx"))
        if len(produced) != 1:
            raise RuntimeError(
                f"expected exactly one quantised graph under {staging_dir}, found {len(produced)}"
            )
        shutil.move(str(produced[0]), str(out))
    return out


def ensure_exported() -> None:
    """Build the staging artifacts if they are not already there.

    Never touches an PII_DETECTION_INT8_OVERRIDE path: the override names a graph
    the caller supplied deliberately (a corrupted copy, in the mutation
    check), and regenerating it would destroy the very defect the check
    exists to detect.
    """
    if not STAGING_FP32.is_file():
        export_fp32(CHECKPOINT, STAGING_FP32)
    default_int8 = STAGING_DIR / "model.onnx"
    if not default_int8.is_file():
        quantize_int8_static(STAGING_FP32, default_int8)


# --- graph inspection -------------------------------------------------------


def graph_metadata(path: Path) -> dict:
    """Facts read back out of an ONNX file, not asserted about it.

    `dynamic_axes` is the ordered list of distinct symbolic dimension names
    across the graph's inputs -- so a graph frozen at sequence 256 reports
    ["batch"] rather than ["batch", "sequence"], and a graph frozen at both
    reports []. `num_labels` is the logits output's last dimension, which
    must be a concrete integer; if it were symbolic this returns None and
    the caller's assertion fails, which is the correct outcome.

    `op_types` counts the graph's nodes by operator. It is here because
    weight element type and file size cannot tell a fully quantised graph
    from one quantised only in its embedding, and the difference between
    those two is the whole speed benefit -- see the comment beside it.
    """
    import onnx

    ensure_exported()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no ONNX graph at {path}")
    model = onnx.load(str(path), load_external_data=False)
    graph = model.graph

    axes: list[str] = []
    for value in graph.input:
        for dim in value.type.tensor_type.shape.dim:
            if dim.dim_param and dim.dim_param not in axes:
                axes.append(dim.dim_param)

    num_labels = None
    for value in graph.output:
        if value.name != "logits":
            continue
        dims = list(value.type.tensor_type.shape.dim)
        if dims and dims[-1].HasField("dim_value"):
            num_labels = dims[-1].dim_value

    # How many stored weight tensors there are of each element type. This is
    # part of what tells a quantised graph apart from a copy of the fp32
    # one: skipping quantisation produces a graph that agrees with PyTorch
    # on 100% of token positions and has exactly the same inputs, outputs,
    # labels and axes, so nothing about the graph's interface would notice.
    dtypes: dict[str, int] = {}
    for initializer in graph.initializer:
        name = onnx.TensorProto.DataType.Name(initializer.data_type)
        dtypes[name] = dtypes.get(name, 0) + 1

    # ...and weight dtype alone is still not enough. A graph quantised in
    # its embedding only -- op_types_to_quantize=["Gather"], all twelve
    # encoder layers left in fp32 -- carries UINT8 initializers and is under
    # half the fp32 size, so it clears both of those checks while none of
    # its *compute* is quantised. Such a graph runs at fp32 latency, which
    # is the entire reason quantisation exists, and it would reach a latency
    # measurement looking like a model problem rather than an export one.
    # Op-type counts are what distinguishes it: the real INT8 graph has 97
    # MatMulInteger and 0 MatMul, the embedding-only mutant has 0 and 97.
    op_types: dict[str, int] = {}
    for node in graph.node:
        op_types[node.op_type] = op_types.get(node.op_type, 0) + 1

    return {
        "path": str(path),
        "inputs": [value.name for value in graph.input],
        "outputs": [value.name for value in graph.output],
        "num_labels": num_labels,
        "dynamic_axes": axes,
        "opset": [(entry.domain or "ai.onnx", entry.version) for entry in model.opset_import],
        "initializer_dtypes": dtypes,
        "op_types": op_types,
        "bytes": path.stat().st_size,
    }


def static_quantisation_facts(path: Path) -> dict:
    """Whether a graph's activation scales are constants, read out of it.

    This is the fact the whole static export exists to establish, and it
    cannot be read from op-type counts alone. A graph can be fully
    quantised and still recompute its scales per call: that is exactly what
    the dynamic recipe produced, and it is why a window's answer used to
    depend on its batch-mates.

    Reported, per graph:

    * `dynamic_quantize_linear` -- nodes that derive a scale at run time.
      Must be 0. Anything else and the graph is batch-dependent however its
      weights were quantised.
    * `quantize_linear` / `with_constant_scale` -- how many activation
      quantisation nodes there are, and how many of those take their scale
      and zero point from a graph initializer or a `Constant` node rather
      than from a computed tensor. Equal counts is the claim; a
      `QuantizeLinear` fed a computed scale would be a `DynamicQuantizeLinear`
      wearing a different name.
    * `matmul` / `with_quantised_inputs` -- every MatMul's activation input
      arriving through a `DequantizeLinear`, which is what says the compute
      is quantised. In QDQ format the `MatMul` node itself survives the
      quantiser and is fused into `MatMulIntegerToFloat` at session load,
      so counting `MatMulInteger` nodes -- the check the dynamic export
      used -- would report 0 on a perfectly quantised static graph.
    """
    import onnx

    ensure_exported()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no ONNX graph at {path}")
    model = onnx.load(str(path), load_external_data=False)
    graph = model.graph

    constants = {initializer.name for initializer in graph.initializer}
    constants.update(
        output for node in graph.node if node.op_type == "Constant" for output in node.output
    )
    produced_by = {output: node for node in graph.node for output in node.output}

    quantize_linear = 0
    with_constant_scale = 0
    for node in graph.node:
        if node.op_type != "QuantizeLinear":
            continue
        quantize_linear += 1
        if all(name in constants for name in node.input[1:]):
            with_constant_scale += 1

    matmul = 0
    with_quantised_inputs = 0
    for node in graph.node:
        if node.op_type != "MatMul":
            continue
        matmul += 1
        sources = [produced_by.get(name) for name in node.input]
        if all(source is not None and source.op_type == "DequantizeLinear" for source in sources):
            with_quantised_inputs += 1

    return {
        "path": str(path),
        "dynamic_quantize_linear": sum(
            1 for node in graph.node if node.op_type == "DynamicQuantizeLinear"
        ),
        "quantize_linear": quantize_linear,
        "with_constant_scale": with_constant_scale,
        "matmul": matmul,
        "with_quantised_inputs": with_quantised_inputs,
    }


# --- parity inputs ----------------------------------------------------------


def _carrier_files(carriers_dir: Path, language: str) -> list[Path]:
    return sorted(carriers_dir.glob(f"{language}-*.txt"))


@functools.lru_cache(maxsize=4)
def _case_rows(cases_dir: Path) -> tuple[dict, ...]:
    """Every benchmark case, from every *.json directly under `cases_dir`,
    in sorted-file then in-file order -- the same order pii-detection-eval's `load`
    uses, so the two see the same fixture."""
    rows: list[dict] = []
    for path in sorted(Path(cases_dir).glob("*.json")):
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(parsed, list):
            raise ValueError(f"{path} is not a JSON array of cases")
        rows.extend(parsed)
    return tuple(rows)


def _spread(items: list, count: int) -> list:
    """`count` items spread evenly across `items`, endpoints included.

    Deterministic and index-based rather than sampled: the parity set has
    to be identical between the fp32 run, the INT8 run and any later
    re-run, or the two agreement figures would be measured over different
    text.
    """
    if count <= 0 or not items:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[0]]
    step = (len(items) - 1) / (count - 1)
    return [items[round(i * step)] for i in range(count)]


def _parity_selection(per_language: int, benchmark_dir: Path) -> list[tuple[str, str]]:
    """(identity, text) for each parity input, in a fixed order.

    `identity` is a carrier filename or a case id -- never text. It is what
    `parity_manifest` publishes so a reported agreement figure can be tied
    to the exact inputs it was measured over.
    """
    benchmark_dir = Path(benchmark_dir)
    carriers_dir, cases_dir = benchmark_dir / "carriers", benchmark_dir / "cases"
    chosen: list[tuple[str, str]] = []
    for language in LANGUAGES:
        # Spread rather than take the first few: carrier filenames encode
        # their register as a prefix (eb = annual report, el = EU law,
        # wb = handbook, wp = encyclopedia, wv = consumer guide), so the
        # first three alphabetically are three documents of one register.
        # Spreading crosses registers, which is the point of drawing on
        # this tree at all.
        files = _spread(
            _carrier_files(carriers_dir, language), min(CARRIERS_PER_LANGUAGE, per_language)
        )
        cases = sorted(
            (row for row in _case_rows(cases_dir) if row["lang"] == language),
            key=lambda row: row["id"],
        )
        picked = _spread(cases, per_language - len(files))
        if len(files) + len(picked) != per_language:
            raise ValueError(
                f"only {len(files) + len(picked)} parity texts available for "
                f"{language!r}, wanted {per_language}"
            )
        chosen.extend((path.name, path.read_text(encoding="utf-8")) for path in files)
        chosen.extend((row["id"], row["text"]) for row in picked)
    return chosen


def parity_texts(per_language: int = 10, benchmark_dir: Path | None = None) -> list[str]:
    """`per_language` real texts for each of the six supported languages.

    Per language: CARRIERS_PER_LANGUAGE carrier documents (long enough to
    window) spread across that language's carriers sorted by filename, then
    fixture cases spread evenly across that language's cases sorted by id.
    Spread, not "the first few", in both cases: carrier filenames encode
    their register as a prefix, so the first three alphabetically would be
    three documents of one register, and case ids group by prefix, so the
    first few would all be legacy prompts. Spreading crosses every carrier
    register and every case file -- short legacy prompts, carrier-derived
    long cases, domain-specific cases and pure negatives.

    `benchmark_dir` defaults to the repository's live benchmark/ tree, so
    the tests track the fixture as it grows. Pass a pinned copy when a
    figure has to be reproducible: benchmark/ is authored by a separate
    workstream and a set selected from it today is not the set selected
    from it tomorrow. `parity_manifest` records which it was.
    """
    root = BENCHMARK_DIR if benchmark_dir is None else Path(benchmark_dir)
    return [text for _identity, text in _parity_selection(per_language, root)]


def parity_manifest(per_language: int = 10, benchmark_dir: Path | None = None) -> dict:
    """Identity of the parity set: which texts, and a digest over them.

    Carries identifiers and a hash, never text -- this is written into
    reports and those are read in terminals and CI logs. The digest is over
    the texts in order, so two runs quoting the same digest measured the
    same thing even though benchmark/ has moved since.
    """
    import hashlib

    root = BENCHMARK_DIR if benchmark_dir is None else Path(benchmark_dir)
    selection = _parity_selection(per_language, root)
    digest = hashlib.sha256()
    for _identity, text in selection:
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return {
        "benchmark_dir": str(root),
        "per_language": per_language,
        "n_texts": len(selection),
        "identities": [identity for identity, _text in selection],
        "sha256": digest.hexdigest(),
    }


# --- forward passes ---------------------------------------------------------


def _window_inputs(window) -> dict[str, np.ndarray]:
    """The three model inputs for one window, as a batch of one.

    token_type_ids is all zeros: this is single-sequence token
    classification, and it is passed explicitly to both runtimes so neither
    is relying on its own default. PyTorch would have defaulted it to zeros
    anyway; ONNX has no default at all.
    """
    input_ids = np.asarray([window.input_ids], dtype=np.int64)
    return {
        "input_ids": input_ids,
        "attention_mask": np.asarray([window.attention_mask], dtype=np.int64),
        "token_type_ids": np.zeros_like(input_ids),
    }


def _torch_window_logits(model, window) -> np.ndarray:
    import torch

    feeds = _window_inputs(window)
    with torch.no_grad():
        out = model(
            input_ids=torch.from_numpy(feeds["input_ids"]),
            attention_mask=torch.from_numpy(feeds["attention_mask"]),
            token_type_ids=torch.from_numpy(feeds["token_type_ids"]),
        )
    return out.logits[0].numpy()


def _onnx_window_logits(session, window) -> np.ndarray:
    return session.run(["logits"], _window_inputs(window))[0][0]


def _real_positions(window) -> list[int]:
    """Indices of the window's real tokens. IGNORE_INDEX marks the special
    tokens align.py already identified; nothing downstream reads their
    predictions, so comparing them would pad the agreement figure with
    positions the service never looks at."""
    return [i for i, label in enumerate(window.labels) if label != IGNORE_INDEX]


def _windows(text: str) -> list:
    return tokenize_and_align(text, [], _tokenizer(), MAX_LENGTH, STRIDE)


def _stack(rows: list[np.ndarray]) -> np.ndarray:
    if not rows:
        return np.empty((0, NUM_LABELS), dtype=np.float32)
    return np.concatenate(rows, axis=0)


def compare_logits(texts: list[str]) -> list[tuple[np.ndarray, np.ndarray]]:
    """(torch_logits, onnx_logits) per text, from the **fp32** graph.

    Each array is (real_token_positions, NUM_LABELS), concatenated across
    the text's windows in window order. Special-token positions are
    excluded; see `_real_positions`.
    """
    ensure_exported()
    model = _torch_model()
    session = _session(STAGING_FP32)
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for text in texts:
        torch_rows: list[np.ndarray] = []
        onnx_rows: list[np.ndarray] = []
        for window in _windows(text):
            keep = _real_positions(window)
            torch_rows.append(_torch_window_logits(model, window)[keep])
            onnx_rows.append(_onnx_window_logits(session, window)[keep])
        pairs.append((_stack(torch_rows), _stack(onnx_rows)))
    return pairs


def agreement_counts(texts: list[str], onnx_path: Path | None = None) -> tuple[int, int]:
    """(agreeing positions, total real positions) between PyTorch's argmax
    and the INT8 graph's, over every real token position in `texts`.

    Returned as counts rather than a rate so a report can state the
    denominator -- a bare fraction says nothing about how much text it was
    measured over.
    """
    ensure_exported()
    model = _torch_model()
    session = _session(Path(onnx_path) if onnx_path is not None else STAGING_INT8)
    agree = 0
    total = 0
    for text in texts:
        for window in _windows(text):
            keep = _real_positions(window)
            if not keep:
                continue
            torch_pred = _torch_window_logits(model, window)[keep].argmax(axis=-1)
            onnx_pred = _onnx_window_logits(session, window)[keep].argmax(axis=-1)
            agree += int((torch_pred == onnx_pred).sum())
            total += len(keep)
    return agree, total


def prediction_agreement(texts: list[str]) -> float:
    """Fraction of real token positions where the INT8 graph's argmax
    equals PyTorch's.

    Argmax agreement, not logit closeness and not span agreement -- INT8
    logits differ by construction, and span agreement is measured
    separately by eval_fixture.py so it goes through the real decoder.
    """
    agree, total = agreement_counts(texts)
    if total == 0:
        raise ValueError("no real token positions to compare")
    return agree / total


# --- CLI --------------------------------------------------------------------


def _fmt_bytes(n: int) -> str:
    return f"{n:,} B ({n / 1_048_576:.1f} MiB)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--fp32-out", default=str(STAGING_FP32))
    parser.add_argument("--int8-out", default=str(STAGING_DIR / "model.onnx"))
    parser.add_argument(
        "--facts",
        default=None,
        help="path to write artifact_facts() as JSON (the packaging's manifest input)",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="build the superseded dynamic graph instead of the static one "
        "(for measuring the two side by side; NOT what ships)",
    )
    parser.add_argument(
        "--skip-fp32",
        action="store_true",
        help="reuse the fp32 graph already in staging rather than re-exporting it",
    )
    args = parser.parse_args()

    # Before anything reads a checkpoint. `export_fp32()` takes the path as
    # an argument, but the tokenizer, the torch model, the label check and
    # the manifest all read the module global -- see set_checkpoint().
    checkpoint = set_checkpoint(args.checkpoint)

    if args.skip_fp32 and Path(args.fp32_out).is_file():
        fp32 = Path(args.fp32_out)
    else:
        fp32 = export_fp32(checkpoint, Path(args.fp32_out))
    print(f"fp32: {fp32}  {_fmt_bytes(fp32.stat().st_size)}")
    if args.dynamic:
        int8 = quantize_int8_dynamic(fp32, Path(args.int8_out))
    else:
        print(json.dumps(calibration_manifest(), indent=2))
        int8 = quantize_int8_static(fp32, Path(args.int8_out))
    print(f"int8: {int8}  {_fmt_bytes(int8.stat().st_size)}")

    for path in (fp32, int8):
        meta = graph_metadata(path)
        print(
            f"  {path.name}: inputs={meta['inputs']} outputs={meta['outputs']} "
            f"num_labels={meta['num_labels']} dynamic_axes={meta['dynamic_axes']} "
            f"opset={meta['opset']}"
        )

    facts = artifact_facts()
    if args.facts:
        Path(args.facts).write_text(json.dumps(facts, indent=2), encoding="utf-8")
        print(f"wrote {args.facts}")
    else:
        json.dump(facts, sys.stdout, indent=2)
        print()


if __name__ == "__main__":
    main()
