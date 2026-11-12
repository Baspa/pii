"""Does a document's detection depend on how its windows were grouped into
ONNX calls? Measures it directly, on one graph, over the latency corpus.

WHY THIS EXISTS. `reports/latency-gate.md`'s addendum A4 found that the
served detector's output moves when the windower groups a document's
windows differently, and `reports/export.md`'s per-channel addendum B3
measured it at the model: 36 of 200 texts at 2,000 code points produce a
different span set depending only on the grouping. The cause was
`DynamicQuantizeLinear`, whose activation scale the ONNX specification
defines as a scalar computed at run time from whatever tensor it is given
-- so a window batched with one set of neighbours is quantised differently
from the same window batched with another. For a compliance product that
is a correctness defect: the same document must mask the same way every
time.

This module is the measurement that decides whether a re-export fixed it.
A correct static export reads 0. Anything else is the export not having
achieved its purpose, and the number this prints is the number that goes
in the report.

THE FOUR GROUPINGS, AND WHY THESE FOUR.

  * `all`     -- every window of a document in one batch, padded to the
                 longest. What `pii-detection-model` did before the
                 trailing-window change (commit 16ef2af).
  * `waste50` -- the superseded `pii_detection_model::session::batches`:
                 consecutive windows share a call while the padding that
                 costs stays at or under 50 positions. What the service did
                 between 16ef2af and the static export.
  * `split`   -- `pii_detection_model::session::batches` as it ships now:
                 consecutive windows of exactly the same length share a
                 call, so no call ever carries a padded position.
  * `single`  -- one call per window at its exact length. What
                 `eval_fixture.py --onnx` already does, and what
                 `reports/export.md` section 5 and 6's figures were
                 measured under.

`split` and `single` differ only in how many rows share a call, never in
what any row contains -- so a graph whose scales are constants must make
them identical, and a difference between them is a defect rather than a
tolerance.

THE SHIP GATE IS `split` VS `single`, ONLY. `reports/distillation-plan.md`
S6 defines the determinism ship gate as this one pair, not "0/200 on all
six": `split` is what the service batches today and `single` is what
`eval_fixture.py --onnx` already scores against, so this is the only pair
with a live consequence on either side. The other five pairs are printed
and returned for diagnosis -- in particular, any pair crossing a padding
grouping (`all`, `waste50`) against a non-padding one is the padding effect
below, not a batching regression -- but `ship_gate_status()` and the
`ship_gate` key in the JSON output are what decide S6.
`reports/student-v2-run.md` (C7) is the run this disambiguates: it was
recorded a miss at "1/200" with no pair named, on a model that was
0/200 on the one pair that matters.

Only the grouping moves. The windowing is `align.tokenize_and_align`, the
same function the trainer used and the same one `parity.rs` pins the Rust
side against, and the decode is `align.reconstruct_spans` plus
`eval_fixture._merge_touching_typed` -- the real decoder, not a second
implementation.

WHAT IS DIFFED. Per-token argmax as well as the decoded span set. Argmax is
the finer of the two and a superset of anything downstream can show: if two
groupings agree on every token's argmax, they agree on spans, on scores, on
the resolver's verdict and on the wire output. Span sets are reported
beside it because that is what a caller actually sees.

PADDING IS A SECOND, SEPARATE EFFECT. `--padding` runs each trailing window
*alone in a batch of one*, first at its own length and then right-padded,
with no batch-mates involved at all. Padded positions are masked out of
attention and contribute exactly zero to it, but they still pass through
the embedding, every LayerNorm and every FFN. Under dynamic quantisation
their activations are inside the min/max the scale is computed from, so
padding alone moves the row. Under a static graph the scales cannot move --
but the fp32 softmax still reduces along the padded axis, so this is a
question to measure rather than to assume, and it is measured separately
from the batching question for exactly that reason.

No raw input, detected value or token text is printed. The corpus entries
carry their own ids (`nl-012`) and those are what identify a differing
text.

Usage:
    python measure_determinism.py --onnx ../models/staging/model.onnx
    python measure_determinism.py --onnx ../models/staging/model.onnx --padding
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from align import IGNORE_INDEX, ID2LABEL, reconstruct_spans, tokenize_and_align
from eval_fixture import _merge_touching_typed

_TRAINING_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TRAINING_DIR.parent

# The 2,000-code-point corpus the latency gate is measured over, built by
# `pii-detection-eval --bin latency --build-corpus` from benchmark/ read-only. It
# is used here because it is the input population the gate and the
# batching rule are both about: ~33 texts per gated language, all of them
# long enough to window to 2 or 3.
DEFAULT_CORPUS = _REPO_ROOT / "models" / "latency-corpus.json"

MAX_LENGTH = 256
STRIDE = 32

# `pii_detection_model::session::SPLIT_BREAK_EVEN_POSITIONS`. Transcribed rather
# than imported because Rust is what ships and Python is what measures;
# the two agreeing is a fact to state, not a thing to hide behind a shared
# constant. See reports/latency-gate.md's addendum A2 for the sweep that
# chose it.
SPLIT_BREAK_EVEN_POSITIONS = 50

_PROVIDERS = ["CPUExecutionProvider"]

# THE SHIP GATE'S PAIR, AND ONLY THIS PAIR.
#
# `reports/student-v2-run.md` (C7) recorded a determinism criterion written
# only as "0/200" as a MISS at 1/200 -- correctly, because nothing said
# which of the six pairs above counted, and one of them (`all`/`waste50`
# against `split`/`single`) does differ. `distillation-plan.md`'s S6 settles
# it: the gate is `split` vs `single`, and only that pair, because those are
# the only two groupings this project can ever actually compare against a
# live consequence -- `split` is what `pii_detection_model::session::batches`
# produces today, `single` is what `eval_fixture.py --onnx` already scores
# against. `all` and `waste50` are both superseded batching behaviour no
# service has run since before the static export, and a pair that mixes a
# padding grouping (`all`, `waste50`) with a non-padding one (`split`,
# `single`) is measuring the padding effect the module docstring already
# separates out via `--padding`, not a batching regression. Those five
# pairs stay in the full table below for diagnosis; they do not decide S6.
SHIP_GATE_PAIR = "split vs single"


def load_corpus(path: Path) -> list[tuple[str, str]]:
    """(id, text) for every entry in the latency corpus, in file order."""
    parsed = json.loads(Path(path).read_text(encoding="utf-8"))
    return [(entry["id"], entry["text"]) for entry in parsed["texts"]]


def plan_all(windows: list) -> list[list]:
    """One batch holding every window, padded to the longest."""
    return [list(windows)]


def plan_single(windows: list) -> list[list]:
    """One batch per window, each at its own exact length."""
    return [[window] for window in windows]


def plan_waste50(windows: list) -> list[list]:
    """The superseded `pii_detection_model::session::batches`, transcribed.

    Consecutive windows share a batch while the padding that costs stays
    at or under SPLIT_BREAK_EVEN_POSITIONS, and start a new batch when it
    does not. Kept because it is the grouping every figure in
    `reports/latency-gate.md`'s trailing-window addendum was measured
    under, and a before/after that could not run the "before" would not be
    one.
    """
    planned: list[list] = []
    rest = list(windows)
    while rest:
        longest = len(rest[0].input_ids)
        end = 1
        while end < len(rest):
            length = len(rest[end].input_ids)
            grown = max(longest, length)
            added_padding = (end + 1) * grown - end * longest - length
            if added_padding > SPLIT_BREAK_EVEN_POSITIONS:
                break
            longest = grown
            end += 1
        planned.append(rest[:end])
        rest = rest[end:]
    return planned


def plan_split(windows: list) -> list[list]:
    """`pii_detection_model::session::batches` as it ships now, transcribed.

    Consecutive windows of exactly the same length share a batch; any
    change of length starts a new one. Batches are contiguous slices in
    the windower's own order, so every window keeps the index the decoder
    depends on, and no batch ever pads a row.
    """
    planned: list[list] = []
    rest = list(windows)
    while rest:
        length = len(rest[0].input_ids)
        end = 1
        while end < len(rest) and len(rest[end].input_ids) == length:
            end += 1
        planned.append(rest[:end])
        rest = rest[end:]
    return planned


PLANNERS = {
    "all": plan_all,
    "waste50": plan_waste50,
    "split": plan_split,
    "single": plan_single,
}


def feed(windows: list, pad_to: int | None = None) -> dict[str, np.ndarray]:
    """The three model inputs for a batch, right-padded to `pad_to` or to
    the batch's longest window.

    token_type_ids is all zeros: single-sequence token classification, and
    it is passed explicitly because ONNX has no default for it.
    """
    width = pad_to or max(len(window.input_ids) for window in windows)
    input_ids = np.zeros((len(windows), width), dtype=np.int64)
    attention_mask = np.zeros((len(windows), width), dtype=np.int64)
    for row, window in enumerate(windows):
        input_ids[row, : len(window.input_ids)] = window.input_ids
        attention_mask[row, : len(window.attention_mask)] = window.attention_mask
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": np.zeros_like(input_ids),
    }


def run_plan(session, windows: list, planner) -> list[np.ndarray]:
    """Per-window logits, sliced back to each window's own real length, for
    the batching `planner` describes. The returned list is in window order
    whatever order the batches ran in."""
    out: list[np.ndarray] = []
    for batch in planner(windows):
        logits = session.run(["logits"], feed(batch))[0]
        for row, window in enumerate(batch):
            out.append(logits[row][: len(window.input_ids)])
    return out


def decode(windows: list, per_window_logits: list[np.ndarray]) -> list[tuple[int, int, str]]:
    """The span set a caller would see, through the real decoder."""
    raw: list[tuple[int, int, str]] = []
    for window, logits in zip(windows, per_window_logits):
        predicted = logits.argmax(axis=-1).tolist()
        decode_ids = [
            IGNORE_INDEX if gold == IGNORE_INDEX else pred
            for pred, gold in zip(predicted, window.labels)
        ]
        raw.extend(reconstruct_spans(window.offset_mapping, decode_ids))
    return _merge_touching_typed(raw)


def ship_gate_status(groupings: dict, num_texts: int) -> dict:
    """Pulls `SHIP_GATE_PAIR` out of the full six-pair table and turns it
    into a verdict, so a caller (human or script) never has to look at six
    lines and guess which one decides S6. See the constant's comment for
    why this is the pair and not, e.g., "0/200 on all six"."""
    pair = groupings[SHIP_GATE_PAIR]
    spans_differ = len(pair["spans"])
    return {
        "pair": SHIP_GATE_PAIR,
        "spans_differ": spans_differ,
        "spans_total": num_texts,
        "argmax_differs": len(pair["argmax"]),
        "worst_logit_delta": pair["worst_logit_delta"],
        "status": "MET" if spans_differ == 0 else "MISS",
    }


def compare_groupings(session, tokenizer, corpus: list[tuple[str, str]]) -> dict:
    """For every pair of groupings, how many texts differ -- on argmax and
    on the decoded span set -- and the worst logit gap seen."""
    names = list(PLANNERS)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]
    result = {
        f"{a} vs {b}": {"argmax": [], "spans": [], "worst_logit_delta": 0.0}
        for a, b in pairs
    }
    for identity, text in corpus:
        windows = tokenize_and_align(text, [], tokenizer, MAX_LENGTH, STRIDE)
        logits = {name: run_plan(session, windows, PLANNERS[name]) for name in names}
        spans = {name: decode(windows, logits[name]) for name in names}
        for a, b in pairs:
            key = f"{a} vs {b}"
            worst = 0.0
            moved = False
            for left, right in zip(logits[a], logits[b]):
                worst = max(worst, float(np.abs(left - right).max()))
                if not np.array_equal(left.argmax(axis=-1), right.argmax(axis=-1)):
                    moved = True
            if moved:
                result[key]["argmax"].append(identity)
            if spans[a] != spans[b]:
                result[key]["spans"].append(identity)
            result[key]["worst_logit_delta"] = max(result[key]["worst_logit_delta"], worst)
    return result


def compare_padding(session_factory, tokenizer, corpus: list[tuple[str, str]], limit: int) -> dict:
    """The second effect: one row, alone in a batch of one, at its own
    length and then right-padded to MAX_LENGTH.

    Also re-runs each window in the same session and in a freshly opened
    one, so a difference that is merely run-to-run noise cannot be reported
    as a padding effect.
    """
    session = session_factory()
    fresh = session_factory()
    tails = []
    for _identity, text in corpus:
        windows = tokenize_and_align(text, [], tokenizer, MAX_LENGTH, STRIDE)
        tail = windows[-1]
        if len(tail.input_ids) < MAX_LENGTH:
            tails.append(tail)
        if len(tails) >= limit:
            break

    same_run = 0
    same_session = 0
    same_padded = 0
    worst = 0.0
    for window in tails:
        exact = session.run(["logits"], feed([window]))[0][0][: len(window.input_ids)]
        again = session.run(["logits"], feed([window]))[0][0][: len(window.input_ids)]
        other = fresh.run(["logits"], feed([window]))[0][0][: len(window.input_ids)]
        padded = session.run(["logits"], feed([window], MAX_LENGTH))[0][0][
            : len(window.input_ids)
        ]
        same_run += int(np.array_equal(exact, again))
        same_session += int(np.array_equal(exact, other))
        same_padded += int(np.array_equal(exact, padded))
        worst = max(worst, float(np.abs(exact - padded).max()))
    return {
        "windows": len(tails),
        "identical_on_a_repeat_run": same_run,
        "identical_in_a_fresh_session": same_session,
        "identical_when_right_padded": same_padded,
        "worst_padding_logit_delta": worst,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--onnx", required=True, help="the ONNX graph to measure")
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--checkpoint", default=None, help="tokenizer source; defaults to the export's")
    parser.add_argument("--padding", action="store_true", help="also run the batch-of-one padding check")
    parser.add_argument("--padding-windows", type=int, default=40)
    parser.add_argument("--out", default=None, help="write the result as JSON here")
    args = parser.parse_args()

    import onnxruntime as ort

    import export_onnx

    # `--checkpoint` was parsed and then ignored: the tokenizer always came
    # from export_onnx's module-level CHECKPOINT, whatever was asked for.
    # Harmless so far only because every checkpoint in this project carries
    # a byte-identical tokenizer.json (it is the base model's, and is never
    # retrained) -- but a flag that silently controls nothing is the same
    # defect as export_onnx's own `--checkpoint`, which did produce a wrong
    # manifest. Wired to the same fix rather than deleted, because naming
    # the tokenizer's source is the right thing for this script to accept.
    if args.checkpoint:
        export_onnx.set_checkpoint(args.checkpoint)

    tokenizer = export_onnx._tokenizer()
    corpus = load_corpus(Path(args.corpus))
    report_checkpoint = str(export_onnx.CHECKPOINT)

    def session_factory():
        return ort.InferenceSession(args.onnx, providers=_PROVIDERS)

    session = session_factory()
    report = {
        "graph": args.onnx,
        "corpus": str(args.corpus),
        # Recorded so the result names the tokenizer it actually used rather
        # than leaving it to be inferred from the command line.
        "tokenizer_checkpoint": report_checkpoint,
        "texts": len(corpus),
        "groupings": compare_groupings(session, tokenizer, corpus),
    }
    report["ship_gate"] = ship_gate_status(report["groupings"], len(corpus))
    if args.padding:
        report["padding"] = compare_padding(
            session_factory, tokenizer, corpus, args.padding_windows
        )

    print("per-pair (diagnostic -- none of these lines alone is the ship gate):")
    for key, value in report["groupings"].items():
        print(
            f"  {key:20s} argmax differs {len(value['argmax']):3d}/{len(corpus)}   "
            f"spans differ {len(value['spans']):3d}/{len(corpus)}   "
            f"worst |dlogit| {value['worst_logit_delta']:.4g}"
        )
    gate = report["ship_gate"]
    print(
        f"\nship gate S6 ({gate['pair']}, distillation-plan.md): "
        f"{gate['spans_differ']}/{gate['spans_total']} spans differ, "
        f"argmax differs {gate['argmax_differs']}/{gate['spans_total']}, "
        f"worst |dlogit| {gate['worst_logit_delta']:.4g}  -- {gate['status']}"
    )
    if args.padding:
        pad = report["padding"]
        print(
            f"padding, batch of one: {pad['identical_when_right_padded']}/{pad['windows']} identical; "
            f"repeat run {pad['identical_on_a_repeat_run']}/{pad['windows']}; "
            f"fresh session {pad['identical_in_a_fresh_session']}/{pad['windows']}; "
            f"worst |dlogit| {pad['worst_padding_logit_delta']:.4g}"
        )
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
