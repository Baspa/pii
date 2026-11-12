"""Proves the exported graph is the same model the checkpoint is.

Real inputs, not random tensors. Random token ids exercise embeddings that
never co-occur in real text and can hide a quantisation error that only
shows on the distributions the model actually sees. These texts are drawn
from benchmark/carriers/ (real licensed prose, six languages) and from
benchmark/cases/ (the fixture the release gates run on).
"""

import json

import numpy as np
import pytest

import export_onnx


@pytest.fixture(scope="module")
def texts():
    # 60 real texts: 10 per supported language, spanning short prompts and
    # multi-window carrier prose, so the comparison covers the windowing
    # path as well as the single-window one.
    return export_onnx.parity_texts(per_language=10)


def test_fp32_onnx_logits_match_pytorch_on_real_text(texts):
    # An export that silently changes behaviour is the expensive failure
    # here. This is the only place in the whole plan where PyTorch and ONNX
    # can be put side by side at all.
    for torch_logits, onnx_logits in export_onnx.compare_logits(texts):
        np.testing.assert_allclose(torch_logits, onnx_logits, atol=1e-4, rtol=1e-3)


def test_int8_predictions_agree_with_pytorch_on_real_text(texts):
    # Agreement, not logit closeness: INT8 logits differ by construction.
    # 0.995 of real token positions is the bar, and it is deliberately
    # stated over token positions rather than over spans, because a span
    # figure hides how many tokens moved inside a span that still decoded
    # to the same answer.
    agreement = export_onnx.prediction_agreement(texts)
    assert agreement >= 0.995, f"INT8 changed {(1 - agreement) * 100:.3f}% of token predictions"


def test_the_exported_graph_has_the_inputs_and_label_count_the_service_expects(tmp_path):
    # 25 labels, and all three Bert inputs. The Rust session asserts the same
    # two facts at load; asserting them here means a bad export fails in
    # Python, next to the code that produced it, rather than in a Rust
    # readiness probe with no context.
    meta = export_onnx.graph_metadata(export_onnx.STAGING_INT8)
    assert set(meta["inputs"]) == {"input_ids", "attention_mask", "token_type_ids"}
    assert meta["num_labels"] == 25
    assert meta["dynamic_axes"] == ["batch", "sequence"]


def test_the_int8_graphs_compute_is_quantised_not_just_its_weights():
    # Two defects hide here, and the three tests above see neither.
    #
    # A plain copy of the fp32 graph agrees with PyTorch on 100% of token
    # positions and has the same three inputs, the same 25 labels and the
    # same two dynamic axes, so it passes all three. Stored weight element
    # type and file size catch that one.
    #
    # They do NOT catch the second: a graph quantised in its embedding only
    # (op_types_to_quantize=["Gather"], all twelve encoder layers left in
    # fp32) carries UINT8 initializers and is well under half the fp32
    # size, so it clears both. None of its compute is quantised, so it runs
    # at fp32 latency -- and speed is the entire reason for quantising.
    # That graph would reach the latency measurement looking like a model
    # problem rather than an export one.
    #
    # The bar is stated against the fp32 graph rather than as a magic
    # number: EVERY fp32 MatMul must take its activations through a
    # DequantizeLinear, with none left reading fp32 directly. That catches
    # partial quantisation of any subset, not just this one.
    #
    # Counting `MatMulInteger` nodes -- what this test asserted while the
    # export was dynamic -- would now report 0 on a perfectly quantised
    # graph: in QDQ format the `MatMul` node survives the quantiser and is
    # fused into `MatMulIntegerToFloat` at session load, not at export.
    int8 = export_onnx.graph_metadata(export_onnx.STAGING_INT8)
    fp32 = export_onnx.graph_metadata(export_onnx.STAGING_FP32)
    quantised = export_onnx.static_quantisation_facts(export_onnx.STAGING_INT8)

    assert set(fp32["initializer_dtypes"]) == {"FLOAT"}, fp32["initializer_dtypes"]

    assert quantised["matmul"] == fp32["op_types"]["MatMul"], (quantised, fp32["op_types"])
    assert quantised["with_quantised_inputs"] == fp32["op_types"]["MatMul"], quantised
    assert int8["initializer_dtypes"].get("INT8", 0) >= 146, int8["initializer_dtypes"]
    assert int8["bytes"] < fp32["bytes"] / 2


def test_every_activation_scale_in_the_shipped_graph_is_a_constant():
    # The whole reason this export is static, asserted on the graph rather
    # than on the flag that produced it.
    #
    # `DynamicQuantizeLinear` derives its scale from the min/max of the
    # tensor it is handed, and the ONNX specification defines that scale as
    # a scalar -- one number for the entire batch. So the same window
    # quantises differently depending on which windows share its call, and
    # a document's spans depend on how the windower grouped it. Measured on
    # the graph this replaced: 36 of 200 texts at 2,000 code points
    # produced a different span set from the grouping alone.
    #
    # Two facts kill that, and both are checked: no node computes a scale,
    # and every node that applies one reads it from an initializer or a
    # Constant. The second is not redundant -- a QuantizeLinear handed a
    # computed scale would be a DynamicQuantizeLinear under another name,
    # and the first check alone would pass on it.
    facts = export_onnx.static_quantisation_facts(export_onnx.STAGING_INT8)
    assert facts["dynamic_quantize_linear"] == 0, facts
    assert facts["quantize_linear"] > 0, facts
    assert facts["with_constant_scale"] == facts["quantize_linear"], facts


def test_the_shipped_graph_gives_a_window_the_same_logits_however_it_is_batched():
    # The property the two graph-shape assertions above are evidence for,
    # measured end to end on the graph itself: a window's logits must not
    # depend on which windows share its ONNX call.
    #
    # Equal-length rows only, because that is what
    # `pii_detection_model::session::batches` produces -- it never pads. Padding a
    # row is a different question and a different answer (the fp32 softmax
    # reduces along the padded axis), which is why the batching rule
    # removes padding rather than tolerating it.
    #
    # Bit-identity, not a tolerance. A tolerance here would pass on the
    # dynamic graph, whose worst movement on this corpus was 3.7 logits.
    import onnxruntime as ort

    texts = export_onnx.parity_texts(per_language=2)
    session = ort.InferenceSession(
        str(export_onnx.STAGING_INT8), providers=["CPUExecutionProvider"]
    )

    full = [
        window
        for text in texts
        for window in export_onnx._windows(text)
        if len(window.input_ids) == export_onnx.MAX_LENGTH
    ]
    assert len(full) >= 4, f"only {len(full)} full-length windows to batch"
    group = full[:4]

    def logits(windows):
        feeds = {
            "input_ids": np.asarray([w.input_ids for w in windows], dtype=np.int64),
            "attention_mask": np.asarray([w.attention_mask for w in windows], dtype=np.int64),
        }
        feeds["token_type_ids"] = np.zeros_like(feeds["input_ids"])
        return session.run(["logits"], feeds)[0]

    alone = [logits([window])[0] for window in group]
    for size in (2, 3, 4):
        batched = logits(group[:size])
        for index in range(size):
            assert np.array_equal(alone[index], batched[index]), (
                f"window {index} moved when batched with {size - 1} others of its own length"
            )


def test_the_checkpoints_label_order_is_the_order_the_decoder_assumes():
    # align.LABELS is what reconstruct_spans maps predicted ids back
    # through; config.json's id2label is what the trained classifier head's
    # output columns actually mean. Nothing else in this file would notice
    # if they diverged -- the graph would still be faithful, the logits
    # still correct to 2.7e-05 -- and every decoded span would carry the
    # wrong type. artifact_facts() publishes this ordering into the packaging's
    # manifest, which the Rust side trusts without re-deriving it.
    assert export_onnx.checkpoint_labels() == list(export_onnx.LABELS)
    assert export_onnx.artifact_facts()["labels"] == list(export_onnx.LABELS)


def _stand_in_checkpoint(monkeypatch, tmp_path, labels):
    """Point export_onnx at a stand-in checkpoint carrying just the one
    file checkpoint_labels() reads.

    `_REPO_ROOT` is repointed as well, and that is not incidental.
    artifact_facts() calls CHECKPOINT.relative_to(_REPO_ROOT); left alone,
    a tmp_path outside the repo makes that call raise a ValueError of its
    own. A `pytest.raises(ValueError)` would then pass whether or not the
    label guard exists -- the test would be green for a reason that has
    nothing to do with what it claims to check. Repointing the root keeps
    relative_to happy, so the only thing that can raise here is the guard,
    and deleting the guard makes these tests fail with DID NOT RAISE.
    """
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"id2label": {str(i): label for i, label in enumerate(labels)}})
    )
    monkeypatch.setattr(export_onnx, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(export_onnx, "CHECKPOINT", checkpoint)
    return checkpoint


def test_a_reordered_label_map_is_refused_rather_than_published(tmp_path, monkeypatch):
    # The agreeing case above proves the two orderings match today. It does
    # NOT prove anything happens when they stop matching, and that branch is
    # the one that matters: a reordered label set decodes every span to the
    # wrong type, silently and completely, while the graph stays faithful
    # and the logits stay correct to 2.7e-05. The failure would read as a
    # model problem, not a label problem.
    #
    # Swap B-PERSON (15) and B-STREET_ADDRESS (19) -- the two types the
    # release gates actually measure, so this is the exact confusion that
    # would be most expensive and hardest to attribute.
    swapped = list(export_onnx.LABELS)
    swapped[15], swapped[19] = swapped[19], swapped[15]
    _stand_in_checkpoint(monkeypatch, tmp_path, swapped)

    assert export_onnx.checkpoint_labels() == swapped

    with pytest.raises(ValueError) as excinfo:
        export_onnx.artifact_facts()
    # The message has to locate the disagreement, not merely announce one:
    # whoever hits this needs to know which ids moved.
    assert "[15, 19]" in str(excinfo.value)


def test_the_stand_in_checkpoint_lets_artifact_facts_succeed_when_labels_agree(
    tmp_path, monkeypatch
):
    # Proves the two tests either side of this one fail for the RIGHT
    # reason. If the stand-in itself made artifact_facts() raise -- a bad
    # path, a missing key -- those tests would be green no matter what the
    # label guard did. Here the stand-in carries the correct ordering and
    # artifact_facts() must come back normally.
    _stand_in_checkpoint(monkeypatch, tmp_path, list(export_onnx.LABELS))
    assert export_onnx.artifact_facts()["labels"] == list(export_onnx.LABELS)


def test_a_label_map_of_the_wrong_length_is_refused_too(tmp_path, monkeypatch):
    # The other way the two can disagree: a checkpoint with a different
    # number of labels. zip() in the comparison stops at the shorter list,
    # so a truncated map agrees on every index it has -- the guard must
    # catch it on length rather than on content.
    truncated = list(export_onnx.LABELS)[:-2]
    _stand_in_checkpoint(monkeypatch, tmp_path, truncated)

    with pytest.raises(ValueError) as excinfo:
        export_onnx.artifact_facts()
    assert "25 vs 23 labels" in str(excinfo.value)


def test_set_checkpoint_repoints_the_manifest_and_not_only_the_graph(
    tmp_path, monkeypatch
):
    # The trap reports/distillation-run.md section 5 had to work around:
    # --checkpoint was threaded into export_fp32() alone, so exporting a
    # 6-layer student wrote the 12-layer TEACHER's identity into the
    # manifest. Nothing raised. The graph was the student's, the manifest
    # said otherwise, and the Rust side trusts the manifest without
    # re-deriving it -- so the artifact would have claimed to be a model it
    # was not, in the one file that is supposed to settle that question.
    #
    # This asserts the fix at the readers that were left behind, not at
    # export_fp32(), which always took the path as an argument.
    monkeypatch.setattr(export_onnx, "_REPO_ROOT", tmp_path)
    student = tmp_path / "runs" / "student"
    student.mkdir(parents=True)
    (student / "config.json").write_text(
        json.dumps(
            {"id2label": {str(i): l for i, l in enumerate(export_onnx.LABELS)}}
        )
    )
    monkeypatch.setattr(export_onnx, "CHECKPOINT", tmp_path / "runs" / "teacher")

    export_onnx.set_checkpoint(student)
    assert export_onnx.artifact_facts()["checkpoint"] == "runs/student"


def test_set_checkpoint_takes_a_relative_path(tmp_path, monkeypatch):
    # The CLI's natural spelling is `--checkpoint runs/student-6l`, run from
    # training/. CHECKPOINT's default is absolute and artifact_facts() calls
    # relative_to(_REPO_ROOT) on it, so a relative path stored as-is raises
    # there -- far from the flag that caused it. set_checkpoint() resolves.
    monkeypatch.setattr(export_onnx, "CHECKPOINT", export_onnx.CHECKPOINT)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs" / "student").mkdir(parents=True)
    assert export_onnx.set_checkpoint("runs/student").is_absolute()


def test_set_checkpoint_invalidates_the_cached_tokenizer_and_model(monkeypatch):
    # Both are lru_cache'd singletons. A process that touched either before
    # the switch would go on serving the previous checkpoint's -- which is
    # the same silent staleness the function exists to remove, arriving by
    # a different route. Asserted on the caches rather than by loading two
    # real checkpoints, which would need a GPU-sized fixture to say the
    # same thing.
    monkeypatch.setattr(export_onnx, "CHECKPOINT", export_onnx.CHECKPOINT)
    calls = []

    class _Recorder:
        def __init__(self, name):
            self.name = name

        def cache_clear(self):
            calls.append(self.name)

    monkeypatch.setattr(export_onnx, "_tokenizer", _Recorder("tokenizer"))
    monkeypatch.setattr(export_onnx, "_torch_model", _Recorder("model"))
    export_onnx.set_checkpoint(export_onnx.CHECKPOINT)
    assert sorted(calls) == ["model", "tokenizer"]


# --- the calibration corpus -------------------------------------------------


def test_the_calibration_corpus_is_training_data_and_never_the_benchmark():
    # The one thing about a static export that cannot be checked by looking
    # at the graph. Static quantisation sets the graph's activation ranges
    # from whatever corpus it was shown, and `benchmark/carriers/` is the
    # obvious, wrong choice: `benchmark/cases/` is built from those
    # carriers, and the 649-case benchmark is what decides whether this
    # artifact ships. A previous report proposed exactly that
    # (`models/latency-corpus.json`, which is carrier prose with
    # identifiers spliced in), which is why the refusal names it.
    assert export_onnx.CALIBRATION_CORPUS.is_relative_to(
        export_onnx._TRAINING_DIR
    ), export_onnx.CALIBRATION_CORPUS

    # The message assertion is load-bearing, not decoration. Without the
    # guard both paths below still raise a ValueError -- `json.JSONDecodeError`
    # IS a ValueError, and neither file is JSON Lines -- so a bare
    # `pytest.raises(ValueError)` goes green on the mutation. That is the
    # exact trap `reports/export.md`'s M6 records this suite falling into
    # once already.
    with pytest.raises(ValueError) as excinfo:
        export_onnx.calibration_documents(
            per_language=1, corpus=export_onnx.BENCHMARK_DIR / "carriers" / "nl-eb-ar24-balance.txt"
        )
    assert "scored against" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        export_onnx.calibration_documents(
            per_language=1, corpus=export_onnx._REPO_ROOT / "models" / "latency-corpus.json"
        )
    assert "scored against" in str(excinfo.value)


def test_the_calibration_set_covers_every_language_and_both_window_shapes():
    # The scales are one set of numbers for all six languages and every
    # window shape the service produces, so a calibration set that missed a
    # language or never saw a full-length window would set them from a
    # sample the service does not resemble. Both halves are asserted:
    # full-length windows (the 256-position path) and short ones (the
    # trailing window every multi-window document ends with).
    manifest = export_onnx.calibration_manifest()

    assert manifest["documents"] == export_onnx.CALIBRATION_DOCS_PER_LANGUAGE * len(
        export_onnx.LANGUAGES
    )
    for language in export_onnx.LANGUAGES:
        assert (
            sum(1 for identity in manifest["identities"] if identity.startswith(f"{language}-"))
            == export_onnx.CALIBRATION_DOCS_PER_LANGUAGE
        ), (language, manifest["identities"])

    assert manifest["full_length_windows"] > 0, manifest
    assert manifest["window_length"]["min"] < export_onnx.MAX_LENGTH, manifest
    assert manifest["window_length"]["max"] == export_onnx.MAX_LENGTH, manifest
    assert set(manifest["sources"]) == {"ai4privacy", "nym"}, manifest


def test_the_calibration_set_is_selected_the_same_way_every_time():
    # A graph is only reproducible if the corpus that configured it is. The
    # selection is index-based and spread across each language's rows, never
    # sampled -- so a rebuild a week later quantises against the same text,
    # and the digest a report quotes can be checked rather than trusted.
    first = export_onnx.calibration_manifest(per_language=2)
    export_onnx._calibration_rows.cache_clear()
    second = export_onnx.calibration_manifest(per_language=2)
    assert first["sha256"] == second["sha256"]
    assert first["identities"] == second["identities"]
