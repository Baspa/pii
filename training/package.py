"""Packages an artifact directory for the Rust `pii-detection-model` crate: the
INT8 ONNX graph, the tokenizer, a label map and a calibration seed, plus a
manifest that pins a SHA-256 digest for each. Spec section 8: "An artifact
directory holds the ONNX graph, the tokenizer, labels.json and
calibration.json, with a manifest carrying a digest for each. The service
verifies all four at start and refuses readiness on a mismatch." This
module is the Python side of that contract; `crates/model/src/artifact.rs`
 is the Rust side reading what this writes.

WHY DIGESTS ARE COMPUTED LAST, AGAINST WHAT LANDED ON DISK. `build()` hashes
the four files *after* they are copied or written into the artifact
directory, never the source files it copied from. A manifest that hashed
the sources instead would still "pass" even if the copy step silently
truncated or corrupted a file on the way into the directory Rust actually
reads -- exactly the failure this artifact's own verification exists to
catch downstream. See training/tests/test_package.py's digest tests, which
this module's report (training/reports/artifact.md) proves by mutating the
built directory, not the code.

WHY INT8 SHIPS BY DEFAULT BUT fp32's DIGEST IS ALWAYS PINNED. The export kept
both `model_fp32.onnx` (448.5 MiB, proven faithful to the checkpoint --
see training/reports/export.md) and `model.onnx` (INT8, 112.9 MiB, 3.97x
smaller, but measurably costs span recall in de/es/fr/it). The latency measurement decides
between them once a p95 latency number exists to weigh against that cost.
`build(out_dir, variant=...)` is the lever that decision pulls: `variant`
selects which graph is copied into the artifact's own `model.onnx`, so
switching is a CLI flag (`--variant fp32`) and a rebuild, never a code
edit and never a re-export -- both graphs already exist in
`models/staging/`. `quantization.fp32_sha256` is written into the
manifest regardless of which variant ships, so the fp32 graph's identity
is on record even while INT8 is what's actually deployed, and does not
have to be reconstructed after the fact once the latency measurement makes the call.

WHY labels.json IS MOSTLY null. The checkpoint predicts all 25 labels
across all 12 platform types, but only PERSON and STREET_ADDRESS are used --
see docs/superpowers/plans/2026-08-22-pii-detection-model.md, "Decisions this
plan makes". The other ten structured types are already gated at 98%
recall / 99% precision by checksum-backed rules (spec section 11); an
uncalibrated model candidate for e.g. NATIONAL_ID carries no checksum
behind it and could only add false positives against the tightest
precision bar in the product. Present-and-null, not absent, for all 25
keys: an absent key is indistinguishable from a typo, and a typo that
silently drops a type is exactly the drift labels.json exists to prevent.
Widening this later is an artifact change, not a code change.

Usage:
    python package.py [OUT_DIR] [--variant int8|fp32]   # default: models/, int8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import export_onnx
from align import LABELS

_TRAINING_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TRAINING_DIR.parent

# The training run this artifact is built from. Exposed as a module
# constant (not inlined into build()) so tests can pin the tokenizer they
# check against to the exact copy the run was trained with, not a
# hand-picked path that could quietly drift from it -- see
# test_the_packaged_tokenizer_is_the_one_the_run_was_trained_with.
TRAINING_RUN = _TRAINING_DIR / "runs" / "full-merged"

NAME = "pii-detection-model-full-merged-int8"
# Bumped when the artifact's *content* changes, not on a schedule. The
# static-QDQ re-export replaced model.onnx with a different graph; leaving
# the revision at "2026-08-22" would have had two materially different
# detectors reporting the same string on /health, with only the manifest
# digest to tell them apart -- and /health does not publish that.
REVISION = "2026-08-22-static-qdq"

BASE_MODEL = {"name": "microsoft/Multilingual-MiniLM-L12-H384", "license": "MIT"}
ARCHITECTURE = "BertForTokenClassification"

TOKENIZER_KIND = "XLMRobertaTokenizerFast"

QUANTIZATION_SCHEME = "static"
QUANTIZATION_DTYPE = "int8"
# arm64 because the spec's own latency proxy is ARM (Apple M1 Max standing
# in for Graviton) and the deployment target is Fargate; per-channel is
# onnxruntime's more accurate (not the default per-tensor) weight
# quantisation mode; qdq is the static format, whose activation scales are
# graph constants rather than recomputed per call. See
# training/export_onnx.py's QUANTIZATION_CONFIG for the full onnxruntime
# call this summarises, and why "static" is the word that matters: the
# dynamic graph this replaced gave a window a different answer depending on
# which windows shared its ONNX call.
QUANTIZATION_CONFIG = "arm64/qdq/per_channel"

ONNX_RUNTIME_VERSION_AT_EXPORT = "1.29.0"

TRAINING_RUN_LABEL = "training/runs/full-merged"

TRAINING_CORPORA = (
    {"name": "ai4privacy/pii-masking-openpii-1.5m", "license": "CC-BY-4.0"},
    {"name": "Wismut/nym-pii-multilingual-data", "license": "MIT"},
)

SUPPORTED_LANGUAGES = ("en", "nl", "de", "fr", "es", "it")


def parameter_count(run: Path | None = None) -> int:
    """Count parameters in the checkpoint being packaged, from its own
    `model.safetensors` header.

    This used to be the constant `PARAMETERS = 117_515_545`, and that
    constant is how the shipped 6-layer student's manifest came to claim
    117,515,545 parameters -- the *12-layer teacher's* count. The student
    has 106,868,761. Nothing failed: `TRAINING_RUN` was repointed at the
    student for every other field, and this one number stayed behind,
    describing a different model in the same file.

    That is the same defect as `export_onnx`'s `--checkpoint` trap wearing
    different clothes, and it has the same shape of fix -- derive the fact
    from the checkpoint rather than restating it beside the checkpoint.
    A manifest is what the Rust side trusts without re-deriving; a number
    in it that no longer tracks its source is worse than an absent one.

    Read from the safetensors header rather than by loading the model:
    the header is JSON carrying every tensor's shape, so the count is
    exact, costs one file read, and needs no torch.
    """
    directory = TRAINING_RUN if run is None else Path(run)
    weights = directory / "model.safetensors"
    with weights.open("rb") as handle:
        header_length = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(header_length))
    return sum(
        math.prod(entry["shape"])
        for name, entry in header.items()
        if name != "__metadata__"
    )


# Only these two platform types get a model-sourced candidate. See the module
# docstring and our "Decisions this work makes" for why the other
# ten structured types map to null instead.
_MAPPED_TYPES = frozenset({"PERSON", "STREET_ADDRESS"})

# Replaced by the calibration step's calibration generator
# (crates/eval/src/bin/calibrate.rs), which writes a real,
# measured per-type/per-language calibration.json into the artifact
# directory and re-runs this module to re-digest it. These are seed
# values only, named as such in `generated_by` so nobody mistakes them
# for a measured calibration.
_SEED_CALIBRATION = {
    "revision": "seed-2026-08-22",
    "generated_by": "training/package.py (seed values; regenerate with pii-detection-eval calibrate)",
    "default": {"PERSON": 0.50, "STREET_ADDRESS": 0.50},
    "languages": {},
}

# The four files spec section 8 requires a digest for, in the order the
# manifest's "files" map is written.
FILES: tuple[str, ...] = ("model.onnx", "tokenizer.json", "labels.json", "calibration.json")


def _label_map() -> dict[str, str | None]:
    """label -> platform type, or None.

    Built by iterating `align.LABELS` directly, in its own order, rather
    than reconstructing that order a second way from taxonomy.PLATFORM_TYPES
    -- so this file's key order can never drift from the order the
    checkpoint's classifier head actually emits (config.json's id2label,
    which export_onnx.artifact_facts() has already checked agrees with
    align.LABELS before build() gets here).
    """
    mapping: dict[str, str | None] = {}
    for label in LABELS:
        if label == "O":
            mapping[label] = None
            continue
        _prefix, platform_type = label.split("-", 1)
        mapping[label] = platform_type if platform_type in _MAPPED_TYPES else None
    return mapping


def _sha256_file(path: Path) -> str:
    """Bare hex digest (no "sha256:" prefix) of a file's bytes, read in
    chunks so multi-hundred-megabyte graphs don't have to fit in memory at
    once just to be hashed."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


VARIANTS: tuple[str, ...] = ("int8", "fp32")


def build(out_dir: Path, variant: str = "int8") -> Path:
    """Build `out_dir/<NAME>/`: copy the selected graph and the tokenizer,
    write `labels.json` and a seed `calibration.json`, then hash every one
    of those four files as it actually landed on disk and write
    `manifest.json` last of all.

    `variant` selects which graph ships as `model.onnx`: "int8" (default,
    112.9 MiB, a small measured span-recall cost -- reports/export.md) or
    "fp32" (448.5 MiB, proven faithful to the checkpoint, no quantisation
    cost). This is the lever the latency measurement's latency-vs-recall decision pulls: a
    CLI flag and a rebuild, never a code edit -- `models/staging/model_fp32.onnx`
    is already pinned by digest in every manifest this function writes,
    regardless of which variant is shipped, so nothing has to be
    re-exported to make the switch either way.

    Idempotent and safe to re-run: `export_onnx.ensure_exported()` only
    (re)builds `models/staging/*.onnx` if either is missing, and every
    write here overwrites rather than appends.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")

    out_dir = Path(out_dir)
    directory = out_dir / NAME
    directory.mkdir(parents=True, exist_ok=True)

    # No-op once the export has run: models/staging/model_fp32.onnx and
    # model.onnx already exist. Present so `build()` is a complete,
    # self-sufficient entry point rather than one that silently assumes a
    # prior step was run first.
    export_onnx.ensure_exported()

    # Raises ValueError if the checkpoint's own config.json disagrees with
    # align.LABELS about label ordering -- the same guard export_onnx.py's
    # own tests exercise. A manifest built past that disagreement would
    # record a labels.json in an order the trained classifier head does not
    # actually use.
    facts = export_onnx.artifact_facts()

    source = export_onnx.STAGING_FP32 if variant == "fp32" else export_onnx.STAGING_DIR / "model.onnx"
    shutil.copyfile(source, directory / "model.onnx")
    shutil.copyfile(TRAINING_RUN / "tokenizer.json", directory / "tokenizer.json")
    _write_json(directory / "labels.json", _label_map())
    _write_json(directory / "calibration.json", _SEED_CALIBRATION)

    # Every digest computed last, from the bytes now sitting in `directory`
    # -- never from the source files above. See the module docstring.
    files = {name: "sha256:" + _sha256_file(directory / name) for name in FILES}
    # Pinned regardless of variant: it names models/staging/model_fp32.onnx,
    # not whichever file was just copied into model.onnx above -- the fp32
    # branch below happens to make the two the same file.
    fp32_sha256 = _sha256_file(export_onnx.STAGING_FP32)

    if variant == "fp32":
        quantization = {
            "scheme": "none",
            "dtype": "float32",
            # Not `None`: the Rust reader's `Quantization::config` is typed
            # `String`, and `serde_json` refuses to decode a `null` into it
            # -- `Artifact::load` returned `Malformed { file: "manifest.json" }`
            # for every fp32 artifact ever built, because this branch was
            # verified on the Python side and never once loaded from Rust.
            # "none" mirrors `scheme` immediately above, which already uses
            # the string "none" (not `None`/`null`) for exactly the same
            # "not applicable to this variant" case.
            "config": "none",
            "fp32_sha256": fp32_sha256,
        }
    else:
        quantization = {
            "scheme": QUANTIZATION_SCHEME,
            "dtype": QUANTIZATION_DTYPE,
            "config": QUANTIZATION_CONFIG,
            "fp32_sha256": fp32_sha256,
            # What the activation scales were measured on. A static graph
            # without this in its manifest is a graph whose behaviour was
            # set by a corpus nobody can name afterwards -- and the one
            # thing that must be checkable about this corpus is that it is
            # the *training* data and not benchmark/, which the artifact is
            # then scored against. Identifiers, counts and a digest only:
            # no text, and nothing that joins back to a corpus row.
            "calibration": export_onnx.calibration_manifest(),
        }

    manifest = {
        "name": NAME,
        "revision": REVISION,
        "base_model": dict(BASE_MODEL),
        "architecture": ARCHITECTURE,
        "parameters": parameter_count(),
        "tokenizer": {"name": facts["tokenizer"], "kind": TOKENIZER_KIND},
        "windowing": {"max_length": facts["max_length"], "stride": facts["stride"]},
        "num_labels": facts["num_labels"],
        "quantization": quantization,
        "onnx_runtime_version_at_export": ONNX_RUNTIME_VERSION_AT_EXPORT,
        "training_run": TRAINING_RUN_LABEL,
        "training_corpora": [dict(c) for c in TRAINING_CORPORA],
        "supported_languages": list(SUPPORTED_LANGUAGES),
        "files": files,
    }
    _write_json(directory / "manifest.json", manifest)
    return directory


def refresh_calibration(directory: Path) -> Path:
    """Re-digests `directory/manifest.json`'s `files` map against whatever
    is actually on disk in `directory`, after `pii-detection-eval calibrate` (task
    7's calibration generator, `crates/eval/src/bin/calibrate.rs`)
    has overwritten `calibration.json` in place.

    Does NOT touch `model.onnx`, `tokenizer.json` or `labels.json`, and does
    NOT regenerate `calibration.json` itself -- `calibrate` already wrote
    the measured thresholds there. A threshold change is a new artifact
    revision (module docstring, and spec section 8: "The service verifies
    all four [files] at start and refuses readiness on a mismatch"), and
    this is the minimal operation that makes the manifest agree with the one
    file that changed, without re-running `export_onnx.ensure_exported()` or
    re-copying the unchanged multi-hundred-megabyte graph -- `build()` does
    both of those and is the wrong tool for a calibration-only refresh.

    `directory` is the artifact directory itself (e.g.
    `models/pii-detection-model-full-merged-int8`), unlike `build()`'s `out_dir`
    (the directory *under which* `<name>/` is built) -- there is no `<name>/`
    to append here, the directory to refresh already exists.
    """
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in FILES:
        path = directory / name
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing; --refresh-calibration expects an already-built artifact "
                "directory (run package.py without it first)"
            )
        manifest["files"][name] = "sha256:" + _sha256_file(path)
    _write_json(manifest_path, manifest)
    return directory


# --- CLI ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "out_dir",
        nargs="?",
        type=Path,
        default=_REPO_ROOT / "models",
        help="directory under which <name>/ is built (default: the repo's models/)",
    )
    parser.add_argument(
        "--variant",
        choices=VARIANTS,
        default="int8",
        help="which graph ships as model.onnx (default: int8; see build()'s docstring)",
    )
    parser.add_argument(
        "--refresh-calibration",
        metavar="ARTIFACT_DIR",
        type=Path,
        default=None,
        help="re-digest an already-built artifact directory's manifest.json after "
        "pii-detection-eval calibrate has replaced its calibration.json, without re-exporting "
        "or re-copying model.onnx/tokenizer.json/labels.json -- see refresh_calibration()",
    )
    args = parser.parse_args(argv)
    if args.refresh_calibration is not None:
        directory = refresh_calibration(args.refresh_calibration)
        print(directory)
        return 0
    directory = build(args.out_dir, variant=args.variant)
    print(directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
