"""Emits (and re-verifies) training/fixtures/window_parity.json: the exact
windows align.tokenize_and_align produces for a set of real texts, for the
Rust windower to be tested against.

Committed rather than generated at Rust test time because the Rust test
suite must not depend on Python, and because a fixture that both sides
regenerate is a fixture that can agree with itself while both are wrong.

`entities` is always `[]` here deliberately -- this fixture is about
windowing and offset agreement, not label alignment (already
covered by training/tests/test_align.py and training/verify_alignment.py).
With no entities every real token labels "O", so the only thing under test
is whether the two implementations tokenize, window and report offsets for
the same text identically.

The decoder work adds two fields per window, `label_ids` and `decoded`, for the BIO
*decoder* parity test (`decode_window` must agree with `reconstruct_spans`
exactly). These are synthetic label ids, never model predictions -- a
decoder test driven by real inference output would change every time the
model does, and would not let us deliberately construct the handful of BIO
patterns `reconstruct_spans` has defined (if unobvious) behaviour for. For
every text except `decode_bio_patterns`, `entities` is still `[]`, so the
synthetic label ids are simply the all-"O" ids `tokenize_and_align` already
produced -- decode has nothing to do and `decoded` is `[]`. For
`decode_bio_patterns`, `_DECODE_OVERRIDES` hand-assigns specific real
(non-special) token positions to specific BIO ids -- see that dict's own
comment for exactly which token exercises which of the six required
patterns.

`label_ids` is written to JSON with `IGNORE_INDEX` (-100) positions
replaced by `LABEL2ID["O"]` (0). The Rust side's `decode_window` never
consults `label_ids` at a special-token position -- it uses the window's
own `spans[i] == None` (the windower's signal for "this is a special token"),
exactly mirroring how `reconstruct_spans` is driven by `lid ==
IGNORE_INDEX` -- so the wire value there is never read and 0 is a safe,
`usize`-representable placeholder. `decoded` is computed from the *real*
array (actual `IGNORE_INDEX` at special positions), because that is what
`reconstruct_spans` itself needs to flush correctly at a window boundary.
"""

import json
import unicodedata
from pathlib import Path

from transformers import AutoTokenizer

from align import IGNORE_INDEX, LABEL2ID, reconstruct_spans, tokenize_and_align

FIXTURE = Path("fixtures/window_parity.json")
MAX_LENGTH, STRIDE = 256, 32

# Real-token-index (0-based over non-special tokens only) -> the BIO label
# to force at that position in `decode_bio_patterns`'s single window,
# overriding the all-"O" baseline `tokenize_and_align` produces for it
# (entities=[] there too -- see the module docstring on why). Computed once
# against the real tokenizer's actual offsets (not guessed): the text
# tokenizes to tokens 0 (<s>) and 21 (</s>) as the only special tokens, so
# real index r corresponds to raw token position r+1. Each entry is
# annotated with the token text it lands on and the decode pattern from the
# task brief -- plus the review round that followed it -- it exists to
# exercise:
#
#   0  "Ana"     B-PERSON           -- a single-token entity
#   2  "Jan"     B-PERSON           -- start of a multi-token entity
#   3  "de"      I-PERSON              (continues)
#   4  "V"       I-PERSON              (continues)
#   5  "ries"    I-PERSON              ("Jan de Vries" ends here)
#   7  "Maria"   B-PERSON           -- a second same-type entity, exactly
#                                       one "O" token (6, "and") after the
#                                       first
#   9  "the"     I-PERSON           -- an I- tag with no preceding B- (the
#                                       token before it, 8 "near", is "O",
#                                       so reconstruct_spans's `entity_type
#                                       != cur_type` branch opens a new span
#                                       exactly as it would for a B- tag)
#   11 "Otto"    B-PERSON           -- immediately preceded by an OPEN
#   12 "Berg"    I-PERSON              PERSON span-in-waiting from the
#                                       previous B-/I- pair (10, "wall", is
#                                       plain "O", so nothing is open when
#                                       "Otto" starts -- the span that
#                                       matters here is the one "Otto Berg"
#                                       itself opens for the next two lines)
#   13 "Rosa"    B-PERSON           -- a `B-` tag directly adjacent to an
#                                       already-open SAME-type span with NO
#                                       "O" or type change between them
#                                       (12 "Berg" is still I-PERSON,
#                                       cur_type is already PERSON): this is
#                                       the one pattern the original fixture
#                                       never exercised. `reconstruct_spans`
#                                       flushes on `prefix == "B"` alone, so
#                                       "Otto Berg" and "Rosa Klein" decode
#                                       as TWO spans, not one merged
#                                       "Otto Berg Rosa Klein". A decoder
#                                       that derives "is this a B- tag" as
#                                       "always false" (i.e. "same type
#                                       always continues") passes every
#                                       other case in this fixture and only
#                                       fails here -- see
#                                       `crates/model/src/decode.rs`'s
#                                       `begins_type` and the mutation the
#                                       review round caught it missing.
#   14 "Klein"   I-PERSON              ("Rosa Klein" ends here)
#   16 "later"   I-PERSON           -- another orphan I-, opening the span
#                                       that the next line changes type on
#   17 "signed"  I-STREET_ADDRESS   -- a type change mid-entity: prefix is
#                                       still "I-", not "B-", so this is the
#                                       `entity_type != cur_type` branch of
#                                       reconstruct_spans's flush condition,
#                                       not the `prefix == "B"` branch
#   19 "Peter"   B-PERSON           -- an entity that runs to the last real
#                                       token of the window
_DECODE_OVERRIDES: dict[int, str] = {
    0: "B-PERSON",
    2: "B-PERSON",
    3: "I-PERSON",
    4: "I-PERSON",
    5: "I-PERSON",
    7: "B-PERSON",
    9: "I-PERSON",
    11: "B-PERSON",
    12: "I-PERSON",
    13: "B-PERSON",
    14: "I-PERSON",
    16: "I-PERSON",
    17: "I-STREET_ADDRESS",
    19: "B-PERSON",
}


def _decode_label_ids(base_labels: list[int], overrides: dict[int, str] | None) -> list[int]:
    """`base_labels` (a window's real, all-"O"-or-IGNORE_INDEX labels from
    `tokenize_and_align` with `entities=[]`) with `overrides` applied by
    real-token position. `overrides=None` returns `base_labels` unchanged
    -- every text except `decode_bio_patterns` takes this path."""
    if overrides is None:
        return list(base_labels)
    ids = list(base_labels)
    real_index = -1
    for i, lid in enumerate(ids):
        if lid == IGNORE_INDEX:
            continue
        real_index += 1
        if real_index in overrides:
            ids[i] = LABEL2ID[overrides[real_index]]
    return ids


def _multilingual_paragraph() -> str:
    """One paragraph rotating through the six supported languages, written
    for this fixture (not sourced from benchmark/carriers -- that corpus's
    CC BY-SA / EUR-Lex / ECB text is deliberately kept out of the rest of
    the repository; see benchmark/LICENSE-CARRIERS.md's "Practical
    consequence" section). Plausible-looking names, numbers and addresses
    are fine here because this fixture never labels entities (see the
    module docstring) -- they just give the tokenizer realistic subword
    variety across scripts and diacritics.
    """
    return (
        "The quarterly logistics report notes steady demand across the network, "
        "and the regional office expects volumes to hold through the next cycle. "
        "Het regiokantoor in Rotterdam meldt dat de planning voor volgende week "
        "gereed is en dat de chauffeurs tijdig zijn ingepland. "
        "Die Zentrale in München bestätigt, dass die Lieferungen pünktlich "
        "eintreffen und die Lagerbestände ausreichend sind. "
        "Le bureau de Lyon confirme que les livraisons sont planifiées avec soin "
        "et que les stocks resteront suffisants ce mois-ci. "
        "La oficina de Sevilla informa de que los pedidos se están procesando "
        "sin retrasos y que el inventario está bajo control. "
        "L'ufficio di Torino conferma che le consegne procedono regolarmente "
        "e che le scorte sono sufficienti per il prossimo trimestre. "
    )


def _padded_multilingual_text(char_len: int) -> str:
    """The paragraph above, repeated and trimmed to exactly `char_len` code
    points (Python string slicing is code-point indexed, so this never
    splits a multibyte character)."""
    paragraph = _multilingual_paragraph()
    repeated = paragraph * (char_len // len(paragraph) + 1)
    return repeated[:char_len]


def _entity_straddling_window_boundary() -> str:
    """A phrase that looks like it carries a value worth masking (an IBAN),
    positioned so truncation cuts it in half in the first window and the
    stride window recovers it whole -- the case `stride` exists for
    (align.py's own docstring on `tokenize_and_align`).

    The repeat count (18) was chosen empirically, once, against the real
    tokenizer: `filler * 18` tokenizes to 252 non-special tokens, so the
    marker phrase that follows starts two tokens before the first window's
    254-real-token limit. Measured directly (not re-measured at fixture
    build time, so this case does not depend on the tokenizer producing the
    same subword count from run to run):

        window 0 (256 tokens): only tokens [253, 254] of the marker land
        inside it -- the rest is truncated.
        window 1 (71 tokens, stride=32 back from window 0's end): the
        marker's full 20 real tokens land at [31, 50] -- recovered whole.

    If a windower's stride handling disagrees with align.py's by even one
    token, this is the case most likely to show it: the marker's offsets in
    window 1 would shift or the token that survives truncation in window 0
    would differ.
    """
    filler = "The quarterly logistics report notes steady demand across the network. "
    marker = "Please debit IBAN NL91ABNA0417164300 for the outstanding balance immediately. "
    tail = (
        "Further updates will follow once the audit team completes its review "
        "of the regional depot."
    )
    return filler * 18 + marker + tail


def parity_texts() -> dict[str, str]:
    texts: dict[str, str] = {}

    texts["ascii_short_one_window"] = (
        "Please call John Smith at 555-123-4567 to confirm tomorrow's appointment."
    )

    texts["german_umlauts_multibyte"] = (
        "Herr Müller wohnt in der Bahnhofstraße 12, 50667 Köln. Seine Kollegin "
        "Frau Größe arbeitet ebenfalls dort und kümmert sich um die Übergabe."
    )

    texts["combining_diacritic_name"] = (
        "Buenas tardes, mi nombre es "
        + unicodedata.normalize("NFD", "José")
        + " y vivo en "
        + unicodedata.normalize("NFD", "Málaga")
        + " desde hace diez años."
    )

    texts["emoji_zwj_family_sequence"] = (
        "Family update: hi \U0001F468\u200D\U0001F469\u200D\U0001F467 there, "
        "we will call the office at 06-12345678 later today."
    )

    texts["right_to_left_arabic"] = (
        "مرحبا، اسمي أحمد وأسكن في شارع الملك فهد، الرياض. "
        "رقم هاتفي هو ٠٥٠١٢٣٤٥٦٧ وبريدي الإلكتروني ahmed@example.com."
    )

    texts["multiwindow_2000_chars"] = _padded_multilingual_text(2000)
    texts["multiwindow_12000_chars"] = _padded_multilingual_text(12000)

    texts["entity_straddles_window_boundary"] = _entity_straddling_window_boundary()

    texts["leading_whitespace"] = "   \n\t  Call +31 6 1234 5678 for support."

    # Purpose-built for the decode fixture, not for windowing/offset parity
    # (every other text above is): one window, so `_DECODE_OVERRIDES`'s
    # real-token indices are unambiguous, and short enough to read by eye
    # alongside the patterns it is annotated with. "Otto Berg Rosa Klein"
    # replaced an earlier "bridge" filler word: the real tokenizer splits
    # "bridge" into two overlapping-offset subtokens after "the ", which
    # would have made the B-immediately-after-open-I- case below land on a
    # confusing offset; four single-subword names give that case a clean,
    # unambiguous pair of two-token spans instead.
    texts["decode_bio_patterns"] = (
        "Ana met Jan de Vries and Maria near the wall Otto Berg Rosa Klein "
        "and later signed by Peter"
    )

    # Deliberately no "only whitespace" entry here. Measured directly: the
    # shipped tokenizer strips a purely-whitespace string to zero real
    # tokens (input_ids == [<s>, </s>], offsets == [(0, 0), (0, 0)],
    # special_tokens_mask == [1, 1]). align.py handles that fine -- it just
    # returns a Window whose every label is IGNORE_INDEX -- but Rust's
    # Window.coverage is a non-optional CharSpan (the windower brief's exact
    # struct), so a window with no real token has no coverage to report and
    # is WindowError::EmptyWindow by design, not by bug. Committing such a
    # text here would make the differential test's `.expect("windows")`
    # panic on a type-system mismatch, not a disagreement, so this
    # degenerate case is exercised directly in
    # crates/model/tests/parity.rs instead of through this fixture.

    return texts


def build():
    tok = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base", use_fast=True)
    records = []
    for name, text in parity_texts().items():
        windows = tokenize_and_align(text, [], tok, MAX_LENGTH, STRIDE)
        overrides = _DECODE_OVERRIDES if name == "decode_bio_patterns" else None
        record_windows = []
        for w in windows:
            # w.labels is entities=[]'s all-"O"-or-IGNORE_INDEX array;
            # label_ids applies this text's decode overrides (if any) on
            # top of it. decoded is reconstruct_spans -- the same function
            # Rust's decode_window has to agree with -- run on that exact
            # array, IGNORE_INDEX included, so it flushes at the window
            # boundary exactly as align.py's own callers see it.
            label_ids = _decode_label_ids(w.labels, overrides)
            record_windows.append(
                {
                    "input_ids": w.input_ids,
                    "attention_mask": w.attention_mask,
                    # None where align.py produced IGNORE_INDEX (special
                    # tokens), so the Rust side is tested on the same
                    # special-token decision, not only on the offsets.
                    "spans": [
                        None if lid == IGNORE_INDEX else [s, e]
                        for (s, e), lid in zip(w.offset_mapping, w.labels)
                    ],
                    # IGNORE_INDEX -> LABEL2ID["O"] (0) here: see the module
                    # docstring for why 0 is a safe placeholder at a
                    # special-token position on the wire.
                    "label_ids": [
                        LABEL2ID["O"] if lid == IGNORE_INDEX else lid for lid in label_ids
                    ],
                    "decoded": [
                        [s, e, t] for s, e, t in reconstruct_spans(w.offset_mapping, label_ids)
                    ],
                }
            )
        records.append(
            {
                "name": name,
                "text": text,
                "char_len": len(text),
                "windows": record_windows,
            }
        )
    return records


def test_the_committed_fixture_matches_what_align_produces_today():
    assert json.loads(FIXTURE.read_text()) == build()
