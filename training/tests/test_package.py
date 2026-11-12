import hashlib
import json

import package
from align import LABELS
from taxonomy import PLATFORM_TYPES


def test_every_file_is_digest_pinned(tmp_path):
    directory = package.build(tmp_path)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name in ("model.onnx", "tokenizer.json", "labels.json", "calibration.json"):
        actual = "sha256:" + hashlib.sha256((directory / name).read_bytes()).hexdigest()
        assert manifest["files"][name] == actual, f"{name} digest does not match"


def test_the_label_map_covers_every_label_the_checkpoint_can_emit(tmp_path):
    # A label the model can predict but the map does not mention would be
    # dropped silently at decode. Present-and-null is the only safe absence.
    directory = package.build(tmp_path)
    labels = json.loads((directory / "labels.json").read_text())
    assert list(labels.keys()) == LABELS
    assert len(labels) == 25


def test_the_label_map_names_only_platform_types_or_null(tmp_path):
    directory = package.build(tmp_path)
    labels = json.loads((directory / "labels.json").read_text())
    for value in labels.values():
        assert value is None or value in PLATFORM_TYPES


def test_only_person_and_street_address_are_mapped(tmp_path):
    # The decision in the design, asserted. Widening this is an artifact
    # change with its own review, never an accident.
    directory = package.build(tmp_path)
    labels = json.loads((directory / "labels.json").read_text())
    assert {v for v in labels.values() if v is not None} == {"PERSON", "STREET_ADDRESS"}


def test_the_packaged_tokenizer_is_the_one_the_run_was_trained_with(tmp_path):
    # A different tokenizer build produces different offsets for the same
    # text, and the failure mode is spans that are subtly wrong rather than
    # an error. Pin it by digest against the training run's own copy.
    directory = package.build(tmp_path)
    packaged = hashlib.sha256((directory / "tokenizer.json").read_bytes()).hexdigest()
    trained = hashlib.sha256(package.TRAINING_RUN.joinpath("tokenizer.json").read_bytes()).hexdigest()
    assert packaged == trained


def test_the_manifest_records_provenance_and_licences(tmp_path):
    # This is a compliance product; CC-BY-4.0 obliges attribution. An
    # artifact that cannot answer "what was this trained on" from itself
    # pushes that question onto a human who was not in the room.
    directory = package.build(tmp_path)
    manifest = json.loads((directory / "manifest.json").read_text())
    corpora = {c["name"]: c for c in manifest["training_corpora"]}
    assert corpora["ai4privacy/pii-masking-openpii-1.5m"]["license"] == "CC-BY-4.0"
    assert corpora["Wismut/nym-pii-multilingual-data"]["license"] == "MIT"
    assert manifest["base_model"]["license"] == "MIT"
    assert manifest["windowing"] == {"max_length": 256, "stride": 32}
    assert manifest["num_labels"] == 25


def test_the_parameter_count_follows_the_checkpoint_being_packaged(tmp_path):
    # The shipped 6-layer student's manifest claims 117,515,545 parameters.
    # That is the 12-LAYER TEACHER's count: `PARAMETERS` was a hand-written
    # constant, every other field followed `TRAINING_RUN` when it was
    # repointed at the student, and this one number stayed behind. Nothing
    # raised, and the artifact described a model it was not.
    #
    # Asserted as a relationship, not as a literal: pinning 106,868,761 here
    # would re-create the same defect one layer up -- a number in the tests
    # that stops tracking the checkpoint the moment a different one is
    # packaged. What must hold is that the count comes FROM the checkpoint.
    import math

    weights = package.TRAINING_RUN / "model.safetensors"
    with weights.open("rb") as handle:
        header = json.loads(handle.read(int.from_bytes(handle.read(8), "little")))
    expected = sum(
        math.prod(entry["shape"])
        for name, entry in header.items()
        if name != "__metadata__"
    )

    manifest = json.loads((package.build(tmp_path) / "manifest.json").read_text())
    assert manifest["parameters"] == expected
    assert manifest["parameters"] == package.parameter_count(package.TRAINING_RUN)


def test_the_fp32_variant_ships_the_fp32_graph_not_the_int8_one(tmp_path):
    # The fix for a reviewer finding: switching graphs must be a build
    # argument, not an edit to package.py's source. Prove the argument is
    # real by checking the bytes that actually landed in model.onnx, not
    # just that the manifest says what was asked for.
    import export_onnx

    directory = package.build(tmp_path, variant="fp32")
    manifest = json.loads((directory / "manifest.json").read_text())

    shipped = hashlib.sha256((directory / "model.onnx").read_bytes()).hexdigest()
    fp32_source = hashlib.sha256(export_onnx.STAGING_FP32.read_bytes()).hexdigest()
    int8_source = hashlib.sha256((export_onnx.STAGING_DIR / "model.onnx").read_bytes()).hexdigest()

    assert shipped == fp32_source
    assert shipped != int8_source
    assert manifest["files"]["model.onnx"] == "sha256:" + shipped
    assert manifest["quantization"]["scheme"] == "none"
    assert manifest["quantization"]["fp32_sha256"] == fp32_source


def test_an_unknown_variant_is_refused(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="unknown variant"):
        package.build(tmp_path, variant="fp16")


def _synthetic_artifact_dir(tmp_path, calibration_bytes: bytes):
    """A minimal, fake artifact directory -- fixed content for the three
    files refresh_calibration() must leave untouched, plus a manifest whose
    digests are already correct for that content, so a test can change only
    calibration.json and see only its digest move. Never calls
    export_onnx.ensure_exported() or package.build(), so it needs no real
    model checkpoint -- refresh_calibration() only ever reads bytes off disk
    and never interprets model.onnx, tokenizer.json or labels.json."""
    directory = tmp_path / "artifact"
    directory.mkdir()
    fixed = {
        "model.onnx": b"fake-model-bytes",
        "tokenizer.json": b"fake-tokenizer-bytes",
        "labels.json": b"fake-labels-bytes",
    }
    for name, content in fixed.items():
        (directory / name).write_bytes(content)
    (directory / "calibration.json").write_bytes(calibration_bytes)

    files = {name: "sha256:" + hashlib.sha256(content).hexdigest() for name, content in fixed.items()}
    files["calibration.json"] = "sha256:" + hashlib.sha256(calibration_bytes).hexdigest()
    manifest = {"name": "fake", "revision": "test", "files": files}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


def test_refresh_calibration_updates_only_the_calibration_digest(tmp_path):
    directory = _synthetic_artifact_dir(tmp_path, b'{"revision": "seed"}')
    original_manifest = json.loads((directory / "manifest.json").read_text())

    # calibrate.rs has already overwritten calibration.json in place, as if
    # a real run just measured new thresholds.
    new_calibration = b'{"revision": "measured-run-1", "default": {"PERSON": 0.61}}'
    (directory / "calibration.json").write_bytes(new_calibration)

    package.refresh_calibration(directory)
    refreshed = json.loads((directory / "manifest.json").read_text())

    expected_digest = "sha256:" + hashlib.sha256(new_calibration).hexdigest()
    assert refreshed["files"]["calibration.json"] == expected_digest
    assert refreshed["files"]["calibration.json"] != original_manifest["files"]["calibration.json"]
    # The three untouched files keep their original digests -- nothing was
    # re-exported or re-copied.
    for name in ("model.onnx", "tokenizer.json", "labels.json"):
        assert refreshed["files"][name] == original_manifest["files"][name]


def test_refresh_calibration_is_a_no_op_when_calibration_json_did_not_change(tmp_path):
    directory = _synthetic_artifact_dir(tmp_path, b'{"revision": "seed"}')
    before = json.loads((directory / "manifest.json").read_text())
    package.refresh_calibration(directory)
    after = json.loads((directory / "manifest.json").read_text())
    assert before["files"] == after["files"]


def test_refresh_calibration_refuses_a_directory_missing_a_required_file(tmp_path):
    import pytest

    directory = _synthetic_artifact_dir(tmp_path, b'{"revision": "seed"}')
    (directory / "labels.json").unlink()
    with pytest.raises(FileNotFoundError):
        package.refresh_calibration(directory)
