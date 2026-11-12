"""Character-span -> token-level BIO label alignment, shared by
training/train.py (which consumes it to build training batches) and
training/verify_alignment.py (which proves the alignment round-trips before
any training run is trusted).

Sharing this module between the two is deliberate: the verification is only
meaningful if it exercises the exact function the trainer uses to build
labels, not a second, hand-written reimplementation that could pass while
the real one is wrong.

The core rule, non-negotiable per the task brief: labels are derived from
the fast tokenizer's `offset_mapping` against the dataset's exact character
spans. Never re-tokenise entity text and search for it in the sequence --
that throws away the exact offsets and reintroduces the annotation noise
the corpus was built to avoid.

Every non-special token in a window gets a real BIO label (not just the
first subtoken of a word, the alternative some recipes use with -100 on the
rest). That choice is what makes `reconstruct_spans` below possible at all:
if only first-subtokens carried a label, there would be no signal in the
labels themselves for "this token continues the entity", and span
reconstruction would have to fall back on retokenising -- exactly what this
module exists to avoid.

Long text is windowed with a stride (`return_overflowing_tokens`) rather
than truncated, because the Rust service windows long input at inference
time; training on silently truncated documents would teach the model that
entities never appear past token max_length.
"""

from __future__ import annotations

from dataclasses import dataclass

from taxonomy import PLATFORM_TYPES

# --- Label schema -----------------------------------------------------------
#
# O, then B-<type>/I-<type> per platform type, grouped by type for readability.
# 1 + 12*2 = 25 labels.

LABELS: list[str] = ["O"]
for _t in PLATFORM_TYPES:
    LABELS.append(f"B-{_t}")
    LABELS.append(f"I-{_t}")

LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {i: label for label, i in LABEL2ID.items()}

IGNORE_INDEX = -100  # PyTorch CrossEntropyLoss's default ignore_index


@dataclass
class Window:
    input_ids: list[int]
    attention_mask: list[int]
    offset_mapping: list[tuple[int, int]]
    labels: list[int]


def tokenize_and_align(
    text: str,
    entities: list[dict],
    tokenizer,
    max_length: int,
    stride: int,
) -> list[Window]:
    """Tokenize `text` into one or more overlapping windows (sliding window
    with stride when the text exceeds max_length) and label every token
    through offset_mapping against `entities`' character spans.

    `entities` must be non-overlapping (guaranteed by the dataset builder,
    see training/build_dataset.py) -- overlap resolution is not attempted
    here.

    Constraint on `stride`: it must exceed the longest entity's token
    length, or an entity can be cut in every window and never appear
    complete anywhere. Measured directly (not assumed) over 493,405
    entities in validation.jsonl: the great majority are short (p99=11
    tokens, p999=17), but the longest is 34 tokens (a run of six UK
    postcodes joined into one STREET_ADDRESS span), and only 2 of 493,405
    exceed the current stride=32. stride=32 is therefore already slightly
    under that observed maximum -- tolerable in practice only because the
    overlapping stride window still recovers those 2 in a neighbouring
    window (see training/verify_alignment.py's window-straddle audit).
    Callers lowering `stride` from its current default should re-measure
    this, not assume it is still safe.
    """
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        return_overflowing_tokens=True,
        truncation=True,
        max_length=max_length,
        stride=stride,
        padding=False,
    )

    n_windows = len(enc["input_ids"])
    windows: list[Window] = []
    for w in range(n_windows):
        offsets = enc["offset_mapping"][w]
        special_mask = enc["special_tokens_mask"][w]
        labels = _labels_for_window(offsets, special_mask, entities)
        windows.append(
            Window(
                input_ids=enc["input_ids"][w],
                attention_mask=enc["attention_mask"][w],
                offset_mapping=offsets,
                labels=labels,
            )
        )
    return windows


def _labels_for_window(
    offsets: list[tuple[int, int]],
    special_mask: list[int],
    entities: list[dict],
) -> list[int]:
    labels: list[int] = []
    prev_entity_idx: int | None = None
    for (start, end), is_special in zip(offsets, special_mask):
        # is_special is the authoritative signal for CLS/SEP/PAD-type
        # tokens. start == end (which fast tokenizers also report as (0, 0)
        # for special tokens) is kept as a defensive second check only --
        # never relied on alone, since a real token could in principle
        # start at character 0.
        if is_special or start == end:
            labels.append(IGNORE_INDEX)
            prev_entity_idx = None
            continue

        match_idx = _find_entity(start, end, entities)
        if match_idx is None:
            labels.append(LABEL2ID["O"])
            prev_entity_idx = None
        else:
            entity_type = entities[match_idx]["label"]
            if prev_entity_idx == match_idx:
                labels.append(LABEL2ID[f"I-{entity_type}"])
            else:
                labels.append(LABEL2ID[f"B-{entity_type}"])
            prev_entity_idx = match_idx
    return labels


def _find_entity(tok_start: int, tok_end: int, entities: list[dict]) -> int | None:
    """Index of the entity whose character span overlaps [tok_start, tok_end),
    or None. Overlap, not containment: a token that straddles an entity
    boundary (mid-token split) still gets labelled -- the boundary then
    legitimately widens to the token's edge on reconstruction."""
    for idx, ent in enumerate(entities):
        if ent["start"] < tok_end and ent["end"] > tok_start:
            return idx
    return None


def reconstruct_spans(
    offsets: list[tuple[int, int]], label_ids: list[int]
) -> list[tuple[int, int, str]]:
    """Inverse of _labels_for_window: given a window's offsets and its
    per-token label ids, recover (start, end, type) character spans.

    This is the actual verification mechanism -- it must be driven only by
    the labels and offsets, exactly as a real decode-time consumer would
    have them, not by peeking at the original entities.
    """
    spans: list[tuple[int, int, str]] = []
    cur_start: int | None = None
    cur_end: int | None = None
    cur_type: str | None = None

    def flush():
        nonlocal cur_start, cur_end, cur_type
        if cur_type is not None:
            spans.append((cur_start, cur_end, cur_type))
        cur_start = cur_end = cur_type = None

    for (start, end), lid in zip(offsets, label_ids):
        if lid == IGNORE_INDEX:
            flush()
            continue
        label = ID2LABEL[lid]
        if label == "O":
            flush()
            continue
        prefix, entity_type = label.split("-", 1)
        if prefix == "B" or entity_type != cur_type:
            flush()
            cur_start, cur_end, cur_type = start, end, entity_type
        else:  # "I-" continuing the same open type
            cur_end = end
    flush()
    return spans


def window_real_range(window: Window) -> tuple[int, int] | None:
    """The character range this window actually covers, taken from the
    offsets of its non-special (real) tokens. None if the window has no
    real tokens at all (shouldn't happen in practice, guarded anyway).

    Shared by the alignment verifier's window-straddle detection and by
    evaluation code that needs to know a window's true coverage rather
    than assuming it always equals the source record's full span -- true
    only when the record fit in one window.
    """
    real = [(s, e) for (s, e), lid in zip(window.offset_mapping, window.labels) if lid != IGNORE_INDEX]
    if not real:
        return None
    return min(s for s, _ in real), max(e for _, e in real)
