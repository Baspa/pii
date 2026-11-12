"""Tests for training/align.py -- the character-span <-> token-BIO
alignment that training/train.py and training/verify_alignment.py both
depend on.

This is the highest-stakes module in the training pipeline: the task brief
that produced it says outright that a one-off alignment bug "trains the
model on systematically wrong labels and produces plausible-looking
rubbish... It does not announce itself." These tests exist to make it
announce itself.

The tokenizer is real (FacebookAI/xlm-roberta-base's fast tokenizer), not a
stub -- a fake tokenizer's offset_mapping could accidentally paper over
exactly the boundary/whitespace/subword quirks this module has to handle
correctly against the real one. It is session-scoped so the suite pays the
one-time load cost once.
"""

from __future__ import annotations

import pytest
from transformers import AutoTokenizer

from align import ID2LABEL, IGNORE_INDEX, LABEL2ID, LABELS, reconstruct_spans, tokenize_and_align
from taxonomy import PLATFORM_TYPES


@pytest.fixture(scope="session")
def tokenizer():
    tok = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base", use_fast=True)
    assert tok.is_fast
    return tok


def _roundtrip(text: str, entities: list[dict], tokenizer, max_length=256, stride=32):
    """Tokenize+align, then reconstruct, then return the reconstructed
    entities as (text, label) pairs alongside the original for comparison.
    Assumes a single window (short text) -- callers needing multi-window
    behaviour test that separately.
    """
    windows = tokenize_and_align(text, entities, tokenizer, max_length, stride)
    assert len(windows) == 1
    w = windows[0]
    spans = reconstruct_spans(w.offset_mapping, w.labels)
    recon = [(text[s:e], t) for s, e, t in spans]
    orig = [(text[e["start"] : e["end"]], e["label"]) for e in entities]
    return orig, recon


class TestLabelSchema:
    def test_label_count(self):
        # O, plus B-/I- for each of the 12 platform types.
        assert len(LABELS) == 1 + 2 * len(PLATFORM_TYPES)

    def test_every_platform_type_has_b_and_i(self):
        for t in PLATFORM_TYPES:
            assert f"B-{t}" in LABEL2ID
            assert f"I-{t}" in LABEL2ID

    def test_label2id_id2label_are_inverses(self):
        for label, idx in LABEL2ID.items():
            assert ID2LABEL[idx] == label

    def test_o_is_present(self):
        assert LABEL2ID["O"] == 0


class TestRoundTrip:
    """The actual property this task exists to prove: char spans -> BIO ->
    char spans recovers the originals, exactly, for ordinary (non-edge-case)
    text."""

    @pytest.mark.parametrize(
        "text,entities",
        [
            pytest.param(
                "Contact Marie Dubois at 12 Rue de la Paix, Paris.",
                [
                    {"start": 8, "end": 20, "label": "PERSON"},
                    {"start": 24, "end": 41, "label": "STREET_ADDRESS"},
                ],
                id="latin-basic",
            ),
            pytest.param(
                "Καλησπέρα, ονομάζομαι Δημήτρης Παπαδόπουλος και μένω στην οδό Ερμού 15 σήμερα.",
                [
                    {"start": 22, "end": 43, "label": "PERSON"},
                    {"start": 62, "end": 70, "label": "STREET_ADDRESS"},
                ],
                id="greek-accented",
            ),
            pytest.param(
                "他叫做张伟，住在北京市朝阳区建国路88号。",
                [
                    {"start": 3, "end": 5, "label": "PERSON"},
                    {"start": 8, "end": 20, "label": "STREET_ADDRESS"},
                ],
                id="chinese-no-spaces",
            ),
            pytest.param(
                "Zpráva od Aoife Nic Cárthaigh, bydliště 28 Cosán na Coille.",
                [
                    {"start": 10, "end": 29, "label": "PERSON"},
                    {"start": 40, "end": 58, "label": "STREET_ADDRESS"},
                ],
                id="irish-diacritics",
            ),
        ],
    )
    def test_exact_round_trip(self, tokenizer, text, entities):
        orig, recon = _roundtrip(text, entities, tokenizer)
        assert recon == orig

    def test_no_entities_all_o(self, tokenizer):
        text = "Nothing sensitive in this sentence at all."
        windows = tokenize_and_align(text, [], tokenizer, 256, 32)
        assert len(windows) == 1
        w = windows[0]
        real_labels = [l for l in w.labels if l != IGNORE_INDEX]
        assert all(l == LABEL2ID["O"] for l in real_labels)
        assert reconstruct_spans(w.offset_mapping, w.labels) == []

    def test_special_tokens_get_ignore_index(self, tokenizer):
        text = "Marie Dubois lives here."
        windows = tokenize_and_align(text, [{"start": 0, "end": 12, "label": "PERSON"}], tokenizer, 256, 32)
        w = windows[0]
        # CLS at position 0, SEP at the last position, for XLM-R.
        assert w.labels[0] == IGNORE_INDEX
        assert w.labels[-1] == IGNORE_INDEX


class TestBenignBoundaryWidening:
    """Documents, rather than merely tolerates, the one legitimate class of
    round-trip "mismatch" the verification report found: when an entity is
    immediately followed by punctuation with no separating space, the
    tokenizer has no subword boundary between them, so the token -- and
    therefore the reconstructed span -- widens by that trailing character.
    This is a resolution limit of single-token BIO labels, not a bug in
    _labels_for_window/reconstruct_spans. (Discovered by this exact scenario
    breaking the parametrized round-trip test above on first write, with
    "Ερμού 15." -- trailing period, no space -- before the sentence was
    fixed to separate them; kept here as a permanent regression check for
    the behaviour itself.)"""

    def test_entity_immediately_followed_by_punctuation_widens(self, tokenizer):
        text = "Το ΑΦΜ του είναι 123456789."  # Greek: "Their tax ID is 123456789."
        entity_text = "123456789"
        start = text.index(entity_text)
        end = start + len(entity_text)
        entities = [{"start": start, "end": end, "label": "TAX_ID"}]

        windows = tokenize_and_align(text, entities, tokenizer, 256, 32)
        w = windows[0]
        spans = reconstruct_spans(w.offset_mapping, w.labels)
        assert len(spans) == 1
        r_start, r_end, r_type = spans[0]
        assert r_type == "TAX_ID"
        assert r_start == start  # left boundary is exact
        assert r_end >= end  # right boundary may widen, never narrow
        widened_text = text[r_start:r_end]
        assert widened_text.startswith(entity_text)
        extra = widened_text[len(entity_text):]
        assert extra == "" or all(not c.isalnum() for c in extra), (
            f"widened by non-punctuation content {extra!r} -- this would be a real bug"
        )


class TestWindowing:
    def test_short_text_single_window(self, tokenizer):
        windows = tokenize_and_align("Short text.", [], tokenizer, 256, 32)
        assert len(windows) == 1

    def test_long_text_multiple_windows(self, tokenizer):
        # Comfortably over 256 tokens.
        text = "Marie Dubois lives at 12 Rue de la Paix. " * 60
        windows = tokenize_and_align(text, [], tokenizer, max_length=256, stride=32)
        assert len(windows) > 1
        for w in windows:
            assert len(w.input_ids) <= 256

    def test_entity_within_stride_overlap_recovered_in_some_window(self, tokenizer):
        # Build long text with one clearly identifiable entity near the end,
        # confirm at least one window's offset_mapping covers it and labels
        # it correctly (proof the sliding window doesn't silently drop
        # entities beyond the first chunk).
        filler = "This is filler text to push the target past token 256. " * 40
        target = "Contact Jean Petit for details."
        text = filler + target
        target_start = len(filler) + len("Contact ")
        target_end = target_start + len("Jean Petit")
        entities = [{"start": target_start, "end": target_end, "label": "PERSON"}]

        windows = tokenize_and_align(text, entities, tokenizer, max_length=256, stride=32)
        found = False
        for w in windows:
            spans = reconstruct_spans(w.offset_mapping, w.labels)
            if (target_start, target_end, "PERSON") in spans:
                found = True
                break
        assert found, "entity near the end of a long document was not recovered in any window"


class TestOffsetMappingNeverRetokenises:
    """Guards the brief's central constraint: alignment must come from
    offset_mapping against the dataset's own spans, not from re-tokenising
    and searching for the entity text. A regression to search-based
    alignment would still pass naive tests if the entity text is unique in
    the string -- so this test uses a string where the entity text repeats,
    which only offset-based alignment gets right."""

    def test_repeated_substring_uses_correct_occurrence(self, tokenizer):
        text = "Ana Ana works with Ana Ana at the office."
        # The *second* "Ana Ana" (a repeated first name) is the annotated
        # entity; a search-based re-tokeniser would find the first
        # occurrence instead.
        second_start = text.index("Ana Ana", 1)
        second_end = second_start + len("Ana Ana")
        entities = [{"start": second_start, "end": second_end, "label": "PERSON"}]

        windows = tokenize_and_align(text, entities, tokenizer, 256, 32)
        w = windows[0]
        spans = reconstruct_spans(w.offset_mapping, w.labels)
        assert spans == [(second_start, second_end, "PERSON")]
