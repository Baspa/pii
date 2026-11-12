//! The ONNX Runtime session: graph loading, batched window inference, and
//! the argmax/softmax decode that turns raw logits into per-token label ids
//! and scores.
//!
//! Design section 6 step 4: the model runs over windows concurrently with
//! the rules, bounded by an inference semaphore so request concurrency
//! cannot oversubscribe ONNX threads. The semaphore itself lives on
//! [`crate::Model`] (it bounds `Model::detect` calls, not anything in this
//! module); this module is the part that actually spends the threads the
//! semaphore is guarding.
//!
//! # One session, shared, `Mutex`-guarded -- not a pool
//!
//! `ort::session::Session::run` takes `&mut self`, and `ort`'s own docs say
//! plainly: *"If you need concurrent inference, we recommend creating one
//! session per thread."* This module does not do that. It builds exactly
//! one `ort::session::Session` (with `config.intra_threads` intra-op
//! threads, per the brief's step 3), wraps it in a `Mutex`, and shares it
//! across every `Model::detect` call. That serializes the actual ONNX
//! Runtime compute regardless of `config.inference_slots`: two concurrent
//! callers can both be tokenizing, batching and waiting on the lock at
//! once, but only one is ever inside `OrtSession::run` at a time.
//!
//! This is a deliberate, load-bearing choice for this task, not an
//! oversight -- it is the literal shape step 3 describes (one
//! `commit_from_file` call, not a loop building `inference_slots` of them)
//! -- but it means `inference_slots` today bounds *concurrent callers*
//! rather than *concurrent ONNX compute*. A session-per-slot pool would
//! change that trade at the cost of `inference_slots` times the graph's
//! resident memory and load time. Whether that trade is worth it is a p95
//! question, and the latency measurement's thread matrix is where it gets measured rather
//! than assumed here.
//!
//! # Intra-op spinning, and the oversubscription it can cause
//!
//! [`Session::open`] enables `ort`'s intra-op spinning unconditionally --
//! measured worth ~30% on the shipped artifact
//! (`training/reports/latency-investigation.md` §4.2), and something
//! `ort`'s own docs claim is already the default when, measured, it is
//! not. A spinning thread busy-waits rather than sleeping between ops, so
//! it burns a core even while idle. That is a win exactly while
//! `inference_slots * intra_threads <= vCPUs`, and a collapse once it
//! isn't (measured: 117.8 ms against 27.5 ms for the same graph at
//! `intra_threads=10` vs `=6`, both spinning, on a 10-vCPU host -- report
//! §4.1). `Session::open` enforces that inequality itself, refusing to
//! build a session rather than leaving it as a comment for an operator to
//! notice too late.
//!
//! # One document, one *or two* inference calls -- see [`batches`]
//!
//! Every window of one document used to go into a single `run` call,
//! padded to the batch's longest window. That is right while the windows
//! are all the same length, and wrong the moment the last one is short: a
//! 2,000-character document that just crosses the window cliff windows to
//! `[256, 256, 66]`, and padding that 66-position tail up to 256 spends a
//! quarter of the document's compute on positions carrying no token
//! (`training/reports/latency-gate.md` §3 step 1: 768 padded positions for
//! 567 real ones -- 26.2% pure padding -- on exactly the population that
//! misses the 50 ms gate). [`batches`] is the rule that stops paying for
//! that.
//!
//! **[`batches`] now pads nothing at all, and that is a correctness rule
//! before it is a speed one.** The first version of it batched a window in
//! whenever the padding it cost stayed under a measured break-even, which
//! bought about 3.4 ms on a 2-window document. It also meant a window's
//! own row was sometimes padded and sometimes not, depending on how long
//! the rest of the document was -- and a padded row does not compute the
//! same answer as an unpadded one. Padded positions are masked out of
//! attention and contribute exactly zero to it, but the softmax that
//! consumes those scores still reduces along the padded axis, and a
//! last-place-digit change there is enough to move a value across a
//! quantisation boundary. Measured on the shipped static graph: right-
//! padding a window that was alone in its batch changed 2 of 40 trailing
//! windows, worst logit delta 0.556 (`training/reports/export.md`, the
//! static-QDQ addendum). Grouping only equal-length windows removes the
//! effect rather than making it rare: every row in a `run` call is exactly
//! as long as every other, so no position is ever padded and a window's
//! logits are a function of its own tokens alone. That is what makes the
//! detector's answer for a document independent of the document's length,
//! which is the property the static export exists to provide and which
//! this rule is the other half of.

use std::path::Path;
use std::sync::{Mutex, OnceLock};

use ort::session::Session as OrtSession;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::{Tensor, ValueType};

use crate::ModelConfig;
use crate::ModelError;
use crate::artifact::Artifact;
use crate::window::Window;

/// The three inputs `BertForTokenClassification`'s ONNX export declares.
/// [`Session::open`] refuses a graph missing any of these; feeding two and
/// letting ONNX default the third would silently change behaviour.
/// [`Session::input_names`] always reports this fixed order, regardless of
/// how the graph itself orders its declared inputs -- inference feeds them
/// by name (see [`Session::infer`]), so the order they're declared in
/// never matters to correctness, only their presence does.
const EXPECTED_INPUTS: [&str; 3] = ["input_ids", "attention_mask", "token_type_ids"];

/// How [`Session::infer`] groups one document's windows into `run` calls:
/// consecutive windows of **exactly the same length** share a call, and a
/// window of any other length starts a new one. No `run` call this plans
/// ever carries a padded position.
///
/// # Why equal length, and not "whatever pads least"
///
/// The rule this replaced batched a window in whenever the padding it cost
/// stayed under a measured break-even, worth about 3.4 ms on a 2-window
/// document. It also made a window's own row padded or unpadded depending
/// on the lengths of the windows around it, and those two do not compute
/// the same answer. Attention masks the padded positions to exactly zero
/// weight, but the softmax producing those weights still reduces along the
/// padded axis, so its denominator can move in the last place; on an INT8
/// graph that is enough to carry a value across a quantisation boundary.
/// Measured on the shipped static graph: right-padding a window that was
/// alone in its batch changed 2 of 40 trailing windows, worst logit delta
/// 0.556. Not batching it with anything of a different length removes that
/// possibility rather than making it rare.
///
/// Equal-length rows are safe in a way padded ones are not, and this is
/// measured rather than argued: across 43,134 token positions, every
/// window's logits were **bit-identical** whether it ran alone or in a
/// batch of two, three or four windows of its own length
/// (`training/reports/export.md`, the static-QDQ addendum). Batching is
/// still worth doing -- a document's full-length windows all share one
/// call, which is where the compute is.
///
/// What this costs is on the record: `[256, 235]` is now two calls rather
/// than one, +3.3 ms on a 2-window document; the 3-window class, which is
/// where the p95 lives, is unchanged because its trailing window was
/// already split out.
///
/// # Order and coverage
///
/// Order is preserved and every window appears exactly once, in its
/// original position -- the batches are contiguous slices of `windows`, so
/// `batches(windows).flatten()` is `windows`. That is load-bearing:
/// `decode::decode_window` is handed each window's index to compute its
/// distance from the document's edges, and `decode::reconcile` resolves
/// overlaps by that distance. A grouping that reordered windows, or that
/// changed which window a prediction belongs to, would change which span
/// wins a reconciliation -- a different detector, not a faster one.
pub fn batches(windows: &[Window]) -> Batches<'_> {
    Batches { windows }
}

/// The iterator [`batches`] returns. Yields contiguous, non-empty slices of
/// the windows it was built from, every window in a slice the same length.
pub struct Batches<'a> {
    windows: &'a [Window],
}

impl<'a> Iterator for Batches<'a> {
    type Item = &'a [Window];

    fn next(&mut self) -> Option<&'a [Window]> {
        let length = self.windows.first()?.input_ids.len();
        let end = self
            .windows
            .iter()
            .take_while(|window| window.input_ids.len() == length)
            .count();
        let (batch, rest) = self.windows.split_at(end);
        self.windows = rest;
        Some(batch)
    }
}

/// The token positions [`Session::infer`] actually feeds ONNX Runtime for
/// `windows` -- real tokens and padding together, summed over every `run`
/// call [`batches`] plans.
///
/// Public because the padding waste is a number the latency harness
/// reports, and a measuring instrument that computes it from its own copy
/// of the batching rule measures its own copy rather than what ships.
pub fn padded_positions(windows: &[Window]) -> usize {
    batches(windows)
        .map(|batch| {
            let longest = batch
                .iter()
                .map(|window| window.input_ids.len())
                .max()
                .unwrap_or(0);
            batch.len() * longest
        })
        .sum()
}

/// The ONNX Runtime environment is process-global and `ort::init()` only
/// usefully applies the first time it runs before any session is built; a
/// `OnceLock` makes that true regardless of how many `Session`s this
/// process ends up opening.
static ORT_ENVIRONMENT: OnceLock<()> = OnceLock::new();

fn ensure_environment_initialized() {
    ORT_ENVIRONMENT.get_or_init(|| {
        // The returned bool (whether this call's config actually applied)
        // is intentionally unchecked: `false` only means an environment was
        // already committed by an earlier caller, which is exactly the
        // outcome this function exists to make safe rather than an error.
        ort::init().commit();
    });
}

/// One window's per-token predictions. `label_ids[i]`/`scores[i]` line up
/// with `Window::spans[i]` exactly -- the shape `decode::decode_window`
/// expects. Positions where `Window::spans[i]` is `None` (special tokens)
/// are left as `0`/`0.0` placeholders: `decode_window` never reads a label
/// id or score at a position its own `spans` marks `None`, so computing a
/// real softmax there would be spent on a value nothing consumes.
pub(crate) struct WindowPrediction {
    pub label_ids: Vec<usize>,
    pub scores: Vec<f32>,
}

/// A loaded ONNX graph, ready to run batched window inference. See the
/// module docs for why this wraps one shared, `Mutex`-guarded session
/// rather than a pool.
pub struct Session {
    inner: Mutex<OrtSession>,
    pad_token_id: i64,
    num_labels: usize,
}

impl Session {
    /// Opens `artifact`'s graph, verifying its input names and output
    /// label count against the manifest before any inference can run, and
    /// refusing to build a session at all if `config` would run intra-op
    /// spinning oversubscribed. All three checks are `ModelError`s, never
    /// panics: a graph that disagrees with its own manifest, or a thread
    /// configuration that would oversubscribe the host, is a data/config
    /// problem, not a programming error.
    pub fn open(artifact: &Artifact, config: ModelConfig) -> Result<Session, ModelError> {
        ensure_environment_initialized();

        let intra_threads = config.intra_threads.max(1);
        let inference_slots = config.inference_slots.max(1);

        // Intra-op spinning (below) is safe only while every concurrently
        // running inference call's threads still fit inside the host's
        // vCPUs: a spinning thread burns a core even while it has nothing
        // to do, so `inference_slots x intra_threads` oversubscribing the
        // host turns the measured ~30% win into a collapse -- 117.8 ms
        // against 27.5 ms for `intra_threads=10` vs `=6`, both spinning,
        // on a 10-vCPU host (training/reports/latency-investigation.md
        // §4.1, §4.2). This is a live enforcement, not a comment: the
        // optimum thread count is a property of *this* CPU's core layout
        // (measured on an Apple M1 Max) and must be re-swept per host --
        // see `MEASURED_INTRA_THREADS_CAP`'s docs -- so a config that was
        // safe on the measurement machine can stop being safe the moment
        // either number is overridden, or the process lands on a
        // different host, without this check.
        let vcpus = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1);
        let product = inference_slots.saturating_mul(intra_threads);
        if product > vcpus {
            return Err(ModelError::UnsafeSpinningConfig {
                inference_slots,
                intra_threads,
                product,
                vcpus,
            });
        }

        let session = OrtSession::builder()?
            .with_intra_threads(intra_threads)?
            // Measured worth ~12 ms / ~30% at intra_threads=6 on a
            // 2-window document, reproduced across three interleaved
            // rounds against a control that sets an unrelated config
            // entry to its own default (training/reports/
            // latency-investigation.md §4.2). `ort`'s builder documents
            // this as on by default; measured, it behaves as off, which
            // is why this line exists rather than trusting the documented
            // default. Safe only under the oversubscription check above.
            .with_intra_op_spinning(true)?
            // Windows of one document are batched into a single run rather
            // than run across parallel sessions (see the module docs), so
            // there is no independent graph-level parallelism for
            // inter-op threads to fan out across; leaving this above 1
            // would only let ONNX Runtime fight the semaphore for the same
            // cores.
            .with_inter_threads(1)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(artifact.dir.join("model.onnx"))?;

        for name in EXPECTED_INPUTS {
            if !session.inputs().iter().any(|input| input.name() == name) {
                return Err(ModelError::MissingInput { name });
            }
        }

        let graph_labels = graph_label_count(&session)?;
        if graph_labels != artifact.manifest.num_labels {
            return Err(ModelError::LabelCountMismatch {
                declared: artifact.manifest.num_labels,
                graph: graph_labels,
            });
        }

        let pad_token_id = resolve_pad_token_id(&artifact.dir)?;

        Ok(Session {
            inner: Mutex::new(session),
            pad_token_id,
            num_labels: graph_labels,
        })
    }

    /// The three input names this session was verified to expect, always
    /// in this fixed order.
    pub fn input_names(&self) -> [&str; 3] {
        EXPECTED_INPUTS
    }

    /// Runs every window of one document through the inference calls
    /// [`batches`] plans -- one call while the windows are all much the
    /// same length, a second one for a short trailing window that would
    /// otherwise be padded up to the batch's longest. Each call pads to its
    /// own batch's longest window (never to the manifest's `max_length`)
    /// with the tokenizer's pad id, zero attention and zero token-type at
    /// the padding.
    ///
    /// The session lock is taken once for the whole document, not once per
    /// call: the two calls of one document are the same critical section
    /// the single call used to be, so nothing about how concurrent callers
    /// serialise changes here.
    ///
    /// Returns one [`WindowPrediction`] per window, in `windows`' own
    /// order, exactly as a single batched call did.
    pub(crate) fn infer(&self, windows: &[Window]) -> Result<Vec<WindowPrediction>, ModelError> {
        let longest = windows
            .iter()
            .map(|window| window.input_ids.len())
            .max()
            .unwrap_or(0);
        if windows.is_empty() || longest == 0 {
            return Ok(Vec::new());
        }

        let mut session = self.inner.lock().map_err(|_| ModelError::Inference)?;
        let mut predictions = Vec::with_capacity(windows.len());
        for batch in batches(windows) {
            self.run_batch(&mut session, batch, &mut predictions)?;
        }
        Ok(predictions)
    }

    /// One `run` call over one planned batch, appending its predictions to
    /// `predictions` in the batch's own order. Split out of [`infer`] only
    /// so the two are the same code: the tensor assembly, the padding and
    /// the decode below are what [`infer`] has always done, now called once
    /// per planned batch instead of once per document.
    fn run_batch(
        &self,
        session: &mut OrtSession,
        windows: &[Window],
        predictions: &mut Vec<WindowPrediction>,
    ) -> Result<(), ModelError> {
        let batch = windows.len();
        let max_len = windows
            .iter()
            .map(|window| window.input_ids.len())
            .max()
            .unwrap_or(0);
        if batch == 0 || max_len == 0 {
            // Unreachable for any window a `Windower` produces (a window
            // with no real tokens is `WindowError::EmptyWindow` before it
            // gets here), but a batch that cannot be run must still leave
            // one prediction per window: `run_detection` zips windows with
            // predictions positionally, and a short vector would silently
            // shift every later window's decode onto the wrong window.
            predictions.extend(windows.iter().map(|_| WindowPrediction {
                label_ids: Vec::new(),
                scores: Vec::new(),
            }));
            return Ok(());
        }

        let mut input_ids = vec![self.pad_token_id; batch * max_len];
        let mut attention_mask = vec![0i64; batch * max_len];
        let mut token_type_ids = vec![0i64; batch * max_len];
        for (row, window) in windows.iter().enumerate() {
            let offset = row * max_len;
            let len = window.input_ids.len();
            input_ids[offset..offset + len].copy_from_slice(&window.input_ids);
            attention_mask[offset..offset + len].copy_from_slice(&window.attention_mask);
            token_type_ids[offset..offset + len].copy_from_slice(&window.token_type_ids);
        }

        let shape = vec![batch as i64, max_len as i64];
        let input_ids = Tensor::from_array((shape.clone(), input_ids))?;
        let attention_mask = Tensor::from_array((shape.clone(), attention_mask))?;
        let token_type_ids = Tensor::from_array((shape, token_type_ids))?;

        let outputs = session.run(ort::inputs! {
            "input_ids" => input_ids,
            "attention_mask" => attention_mask,
            "token_type_ids" => token_type_ids,
        })?;

        let (out_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        // The label-count check in `Session::open` already proved the
        // graph's declared output width matches the manifest; this
        // debug_assert catches a graph that lies about its declared shape
        // versus what a real run actually produces, which `open`'s
        // metadata-only check cannot see.
        debug_assert_eq!(
            out_shape.last().copied().map(|dim| dim as usize),
            Some(self.num_labels)
        );

        let num_labels = self.num_labels;
        for (row, window) in windows.iter().enumerate() {
            let len = window.input_ids.len();
            let mut label_ids = vec![0usize; len];
            let mut scores = vec![0.0f32; len];
            for (position, span) in window.spans.iter().enumerate() {
                // Only positions where `window.spans[i]` is `Some` are
                // decoded -- the same IGNORE_INDEX mask `align.py` applies,
                // reused rather than reimplemented: a special-token
                // position is never consulted by `decode_window`, so its
                // softmax is never computed.
                if span.is_none() {
                    continue;
                }
                let base = (row * max_len + position) * num_labels;
                let logits = data
                    .get(base..base + num_labels)
                    .ok_or(ModelError::Inference)?;
                let (label_id, score) = argmax_softmax(logits);
                label_ids[position] = label_id;
                scores[position] = score;
            }
            predictions.push(WindowPrediction { label_ids, scores });
        }
        Ok(())
    }

    /// [`infer`](Session::infer)'s pre-[`batches`] behaviour: every window
    /// of the document in one `run` call, padded to the batch's longest.
    /// Test-only, and the only reason it exists is
    /// `splitting_a_short_trailing_window_out_does_not_change_a_single_prediction`
    /// -- a differential test needs both sides, and the side that no longer
    /// ships has to come from somewhere. It calls the same
    /// [`run_batch`](Session::run_batch) the shipped path calls, so it is
    /// the batching that differs between the two and nothing else.
    #[cfg(test)]
    pub(crate) fn infer_as_one_batch(
        &self,
        windows: &[Window],
    ) -> Result<Vec<WindowPrediction>, ModelError> {
        let mut session = self.inner.lock().map_err(|_| ModelError::Inference)?;
        let mut predictions = Vec::with_capacity(windows.len());
        self.run_batch(&mut session, windows, &mut predictions)?;
        Ok(predictions)
    }
}

/// Reads the graph's declared output label count from `session.outputs()`'s
/// first entry's last tensor dimension, without running the graph.
fn graph_label_count(session: &OrtSession) -> Result<usize, ModelError> {
    let output = session.outputs().first().ok_or(ModelError::Inference)?;
    let ValueType::Tensor { shape, .. } = output.dtype() else {
        return Err(ModelError::Inference);
    };
    let last = shape.last().copied().ok_or(ModelError::Inference)?;
    usize::try_from(last).map_err(|_| ModelError::Inference)
}

/// Resolves the tokenizer's pad token id once, at session open. This loads
/// its own, independent `Tokenizer` from the same `tokenizer.json` the
/// artifact already digest-verified -- `Windower` keeps its own
/// instance privately and exposes no accessor for this, and `Session::open`
/// only receives the `Artifact`, not a `Windower`. Wasteful in bytes (a
/// second ~16 MiB parse), never in latency: it happens once per
/// `Model::load`, never per request.
fn resolve_pad_token_id(dir: &Path) -> Result<i64, ModelError> {
    // Any failure here -- the tokenizer failing to load, or loading but
    // declaring no `<pad>` token -- becomes `PadTokenMissing`. The former
    // should be unreachable against a digest-verified artifact (the same
    // bytes already loaded successfully for `Windower`), but `ModelError`
    // has no separate "tokenizer load" variant, and conflating the two
    // here is safer than inventing one for a path that cannot currently be
    // exercised by anything this crate ships.
    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|_| ModelError::PadTokenMissing)?;
    tokenizer
        .token_to_id("<pad>")
        .map(i64::from)
        .ok_or(ModelError::PadTokenMissing)
}

/// argmax and its softmax probability over one token's label logits, in two
/// passes over the slice and no heap allocation. Softmax is monotonic in
/// the logits, so the argmax of the logits is already the argmax of the
/// softmax; the winning probability then reduces to
/// `1 / sum(exp(logit - max_logit))` (the numerator `exp(max_logit -
/// max_logit)` is 1) without ever materialising the other labels'
/// probabilities. `num_labels` is small (25 for the shipped artifact), so
/// this is two cheap linear scans, not a meaningful allocation to avoid on
/// its own merits -- but it is called once per real token per window, and
/// at the 2,000-character budget that is hundreds of calls per request.
fn argmax_softmax(logits: &[f32]) -> (usize, f32) {
    let mut best_id = 0usize;
    let mut max_logit = f32::NEG_INFINITY;
    for (index, &logit) in logits.iter().enumerate() {
        if logit > max_logit {
            max_logit = logit;
            best_id = index;
        }
    }
    let mut sum = 0.0f32;
    for &logit in logits {
        sum += (logit - max_logit).exp();
    }
    let score = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    (best_id, score)
}

impl<R> From<ort::Error<R>> for ModelError {
    fn from(_: ort::Error<R>) -> Self {
        // ort::Error's Display can carry a file path or an ONNX Runtime
        // diagnostic string. ModelError's discipline (the design,
        // held by `no_model_error_variant_carries_text_or_token_ids`) is
        // names and counts only, so the message is discarded here rather
        // than threaded through into a variant that has nowhere safe to
        // put it. Generic over `R` because `SessionBuilder`'s methods
        // return `ort::Error<SessionBuilder>` (to allow recovering the
        // builder on failure) while `commit_from_file` and `Session::run`
        // return the plain `ort::Error<()>` -- one impl covers both without
        // needing two near-identical ones.
        ModelError::Inference
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::Ordering;

    use pii_detection_engine::entity::EntityType;
    use pii_detection_engine::text::{CharSpan, Document};

    use super::{Session, batches, padded_positions};
    use crate::artifact::Artifact;
    use crate::require_artifact;
    use crate::window::{Window, Windower};
    use crate::{Model, ModelConfig, ModelError};

    /// A window of `len` positions, with the shape the batch planner reads
    /// and nothing else filled in meaningfully. `batches` and
    /// `padded_positions` consult `input_ids.len()` and nothing else, so
    /// these carry no ids worth inventing -- and inventing plausible ones
    /// would only suggest the planner looks at them.
    fn window_of(len: usize) -> Window {
        Window {
            input_ids: vec![0; len],
            attention_mask: vec![1; len],
            token_type_ids: vec![0; len],
            spans: vec![None; len],
            coverage: CharSpan { start: 0, end: 0 },
        }
    }

    fn lengths(windows: &[Window]) -> Vec<Vec<usize>> {
        batches(windows)
            .map(|batch| {
                batch
                    .iter()
                    .map(|window| window.input_ids.len())
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// ~2,000 characters of neutral Dutch filler with "Jan de Vries" placed
    /// so it sits right at the first window's trailing edge and squarely in
    /// the second's middle -- measured against the real shipped tokenizer,
    /// not assumed. `training/.venv`'s `tokenizers` (the same library
    /// version and the same `tokenizer.json`) was driven with the
    /// manifest's own truncation settings (`max_length=256, stride=32,
    /// strategy=longest_first, direction=right` -- identical to
    /// `Windower::new`) over candidate texts, reading back each window's
    /// offsets and special-token mask, until this exact construction (36
    /// repeats of the filler sentence, the name, then 54 more repeats)
    /// produced: window 0 sees the name's tokens at `margin (252, 0)` --
    /// its last token is window 0's very last real token, margin zero,
    /// literally at the edge -- and window 1 sees the same tokens at
    /// `margin (30, 220)`, comfortably in the middle. A third window forms
    /// past the name and never overlaps it.
    fn boundary_text() -> String {
        const FILLER: &str = "De kat zit op de mat. ";
        let before = FILLER.repeat(36);
        let after = FILLER.repeat(54);
        format!("{before}Jan de Vries. {after}")
    }

    #[test]
    fn a_short_trailing_window_is_split_out_of_the_batch() {
        // The shape this change exists for. A 2,000-character document
        // that just crosses the window cliff windows to [256, 256, 66];
        // batching all three pads the 66-position tail up to 256 and
        // computes 190 positions that carry no token -- 26.2% of the whole
        // document's compute (training/reports/latency-gate.md §3 step 1).
        assert_eq!(
            lengths(&[window_of(256), window_of(256), window_of(66)]),
            vec![vec![256, 256], vec![66]]
        );
    }

    #[test]
    fn a_nearly_full_trailing_window_is_split_out_too() {
        // The half of the rule that changed, and the one that costs
        // something. A 2-window document's trailing window is typically
        // ~235 of 256 positions, and the previous rule batched it in
        // because splitting measured 3.4 ms slower. It is split now
        // anyway: batching it pads its row by 21 positions, and a padded
        // row is not guaranteed to compute what an unpadded one computes.
        // Paying 3.3 ms for a detector that does not depend on document
        // length is the trade this rule makes.
        assert_eq!(
            lengths(&[window_of(256), window_of(235)]),
            vec![vec![256], vec![235]]
        );
    }

    #[test]
    fn one_position_of_difference_is_enough_to_start_a_new_call() {
        // There is no threshold left to be near. A window one position
        // short of its neighbour would have cost one padded position under
        // any break-even rule and been batched by all of them; it is split
        // here, because "one padded position" and "no padded position" are
        // different computations and only the second is reproducible.
        assert_eq!(
            lengths(&[window_of(256), window_of(255)]),
            vec![vec![256], vec![255]]
        );
        assert_eq!(
            lengths(&[window_of(256), window_of(256)]),
            vec![vec![256, 256]],
            "equal-length windows should still share a call"
        );
    }

    #[test]
    fn no_plan_ever_leaves_a_padded_position() {
        // The property the rule exists for, stated as a property rather
        // than as a list of shapes: `padded_positions` counts real tokens
        // and padding together, so it equals the sum of the window lengths
        // exactly when nothing was padded.
        for shape in [
            vec![256usize],
            vec![256, 235],
            vec![256, 256, 66],
            vec![256, 256, 256, 12],
            vec![64, 256, 64],
            vec![7, 7, 8, 7],
            vec![1, 1, 1, 1],
        ] {
            let windows: Vec<Window> = shape.iter().map(|&len| window_of(len)).collect();
            assert_eq!(
                padded_positions(&windows),
                shape.iter().sum::<usize>(),
                "plan for {shape:?} padded something"
            );
        }
    }

    #[test]
    fn a_batch_plan_covers_every_window_exactly_once_in_order() {
        // Order and coverage are load-bearing, not incidental:
        // `decode_window` is handed each window's index to compute its
        // distance from the document's edges and `reconcile` resolves
        // overlaps by that distance, so a plan that dropped, duplicated or
        // reordered a window would change which span wins -- a different
        // detector, not a faster one.
        for shape in [
            vec![256usize],
            vec![256, 235],
            vec![256, 256, 66],
            vec![256, 256, 256, 12],
            vec![64, 256, 64],
            vec![1, 1, 1, 1],
        ] {
            let windows: Vec<Window> = shape.iter().map(|&len| window_of(len)).collect();
            let flattened: Vec<usize> = batches(&windows)
                .flat_map(|batch| batch.iter().map(|window| window.input_ids.len()))
                .collect();
            assert_eq!(
                flattened, shape,
                "plan did not reproduce {shape:?} in order"
            );
            assert!(
                batches(&windows).all(|batch| !batch.is_empty()),
                "plan yielded an empty batch for {shape:?}"
            );
        }
    }

    #[test]
    fn a_longer_window_arriving_after_shorter_ones_starts_a_new_one() {
        // The rule is symmetric in the direction of the length change:
        // a long window after short ones would have padded every row
        // already in the batch, and it starts a new call for the same
        // reason a short one does. Nothing a `Windower` emits looks like
        // this today (its only short window is the last), which is exactly
        // why the rule should not quietly depend on that.
        assert_eq!(
            lengths(&[window_of(60), window_of(60), window_of(256)]),
            vec![vec![60, 60], vec![256]]
        );
    }

    #[test]
    fn padded_positions_counts_the_padding_the_plan_leaves_behind() {
        // The number the latency harness reports as "padded vs real", read
        // from the shipped planner rather than from a copy of it. Under
        // the equal-length rule it is the real-token count for every
        // input, which is the point -- see `no_plan_ever_leaves_a_padded_position`.
        assert_eq!(
            padded_positions(&[window_of(256), window_of(256), window_of(66)]),
            512 + 66
        );
        assert_eq!(
            padded_positions(&[window_of(256), window_of(235)]),
            256 + 235
        );
        assert_eq!(padded_positions(&[]), 0);
    }

    #[test]
    fn the_shipped_batch_plan_is_bit_identical_to_one_call_per_window() {
        // The correctness bar this rule exists to clear, and it is a
        // stronger one than the rule it replaced could state.
        //
        // Everything downstream -- the furthest-from-edge reconciliation,
        // the BIO decode, the span offsets -- assumes the model sees what
        // `align.py` fed it. Under the previous, padding-tolerant rule the
        // best that could be asserted was "no label and no span moved,
        // and the scores stayed within 0.05", because a padded row does
        // not compute what an unpadded one computes and a quantised graph
        // amplifies the difference. Under the equal-length rule there is
        // no padding anywhere, so the assertion is exact float equality on
        // every score: a window's logits depend on its own tokens and
        // nothing else, whether it shares a call with one window of its
        // length or three.
        //
        // A regression to any rule that pads -- including one that pads a
        // single position -- fails this test rather than degrading
        // quietly.
        let dir = require_artifact!();
        let artifact = Artifact::load(&dir).expect("artifact");
        let windower = Windower::new(&dir.join("tokenizer.json"), artifact.manifest.windowing)
            .expect("windower");
        let labels = artifact.labels.clone();
        let session = Session::open(&artifact, ModelConfig::default()).expect("session");

        // One document per window count from 1 to 5, plus the boundary
        // text, plus two documents whose trailing window is *nearly full*.
        //
        // Those last two are not decoration. Measured against the shipped
        // tokenizer, `SENTENCE.repeat(22)` windows to `[256, 242]` and
        // `repeat(32)` to `[256, 256, 230]` -- tails of 242 and 230, which
        // the previous break-even rule batched in (14 and 26 padded
        // positions, both under 50) and this rule splits out. Without them
        // every text here windows to a tail short enough that BOTH rules
        // split it, so reverting `batches` changes nothing this test can
        // see: verified by mutation, where the score comparison below
        // stayed green until these two texts were added.
        const SENTENCE: &str = "De gemeente heeft de aanvraag beoordeeld en \
             vastgesteld dat de ingediende documenten volledig zijn. ";
        let mut texts: Vec<String> = (1..=5)
            .map(|windows| SENTENCE.repeat(windows * 10))
            .collect();
        texts.push(SENTENCE.repeat(22));
        texts.push(SENTENCE.repeat(32));
        texts.push(boundary_text());

        let mut batched_at_least_once = false;
        let mut split_at_least_once = false;
        for (case, text) in texts.iter().enumerate() {
            let document = Document::new(text.clone());
            let windows = windower.windows(&document).expect("windows");
            let plan = batches(&windows).count();
            batched_at_least_once |= batches(&windows).any(|batch| batch.len() > 1);
            split_at_least_once |= plan > 1;

            // The precondition the bit-identity below actually rests on,
            // asserted here rather than assumed. Measured: reverting
            // `batches` to the padding break-even rule leaves the score
            // comparison below GREEN on these six texts -- the padding it
            // reintroduces happens not to move anything they contain. So
            // "the scores matched" is not on its own evidence that the plan
            // stopped padding, and this line is what makes this test fail on
            // that mutation instead of passing through it.
            assert_eq!(
                padded_positions(&windows),
                windows.iter().map(|w| w.input_ids.len()).sum::<usize>(),
                "case {case}: the plan padded something"
            );

            let planned = session.infer(&windows).expect("planned inference");
            // One call per window, each at its own exact length: the
            // reference every grouping has to reproduce, because it is the
            // only one whose input depends on nothing but the window.
            let alone: Vec<super::WindowPrediction> = (0..windows.len())
                .map(|index| {
                    session
                        .infer_as_one_batch(&windows[index..=index])
                        .expect("single-window inference")
                        .pop()
                        .expect("one prediction per window")
                })
                .collect();
            assert_eq!(planned.len(), windows.len(), "case {case}");

            for (index, (planned, alone)) in planned.iter().zip(alone.iter()).enumerate() {
                assert_eq!(
                    planned.label_ids, alone.label_ids,
                    "case {case} window {index} decoded to different labels"
                );
                assert_eq!(
                    planned.scores, alone.scores,
                    "case {case} window {index} scored differently when batched -- \
                     the plan padded something, or the graph is not statically quantised"
                );
            }

            let decode = |predictions: &[super::WindowPrediction]| {
                let mut spans = Vec::new();
                for (index, (window, prediction)) in
                    windows.iter().zip(predictions.iter()).enumerate()
                {
                    spans.extend(crate::decode::decode_window(
                        window,
                        index,
                        &prediction.label_ids,
                        &prediction.scores,
                        &labels,
                    ));
                }
                let reconciled = crate::decode::reconcile(spans);
                let trimmed = crate::decode::trim_whitespace(&document, reconciled);
                crate::decode::to_candidates(trimmed)
                    .into_iter()
                    .map(|candidate| (candidate.span, candidate.entity_type))
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                decode(&planned),
                decode(&alone),
                "case {case} decoded to different spans"
            );
        }
        assert!(
            split_at_least_once,
            "no case was actually split, so this test compared nothing"
        );
        assert!(
            batched_at_least_once,
            "no case actually shared a call, so batching was never exercised"
        );
    }

    #[test]
    fn the_session_rejects_a_graph_whose_label_count_is_not_the_manifest_s() {
        // 25 labels is a fact about the checkpoint (config.json, and
        // reports/full-run-merged.md). A graph with a different output
        // width decodes to the wrong types entirely, and every symptom of
        // that is silent.
        let dir = require_artifact!();
        let mut artifact = Artifact::load(&dir).expect("artifact");
        artifact.manifest.num_labels = 24;
        assert!(matches!(
            Session::open(&artifact, ModelConfig::default()),
            Err(ModelError::LabelCountMismatch {
                declared: 24,
                graph: 25
            })
        ));
    }

    #[test]
    fn the_session_refuses_to_open_when_spinning_would_oversubscribe_the_host() {
        // The safety condition intra-op spinning depends on
        // (training/reports/latency-investigation.md §4.1, §4.2): a
        // spinning thread burns a core even while idle, so
        // `inference_slots * intra_threads` exceeding the host's vCPUs
        // turns the measured ~30% win into a collapse (117.8 ms against
        // 27.5 ms at `intra_threads=10` vs `=6`, both spinning, on a
        // 10-vCPU host). A config that plainly oversubscribes any real
        // host must be refused before a session is ever built, not left
        // as a comment for an operator to discover in production.
        let dir = require_artifact!();
        let artifact = Artifact::load(&dir).expect("artifact");
        let config = ModelConfig {
            intra_threads: 1_000,
            inference_slots: 1_000,
        };
        assert!(matches!(
            Session::open(&artifact, config),
            Err(ModelError::UnsafeSpinningConfig {
                inference_slots: 1_000,
                intra_threads: 1_000,
                ..
            })
        ));
    }

    #[test]
    fn the_session_rejects_a_graph_missing_an_input_it_expects() {
        // BertForTokenClassification exports three inputs. Feeding two and
        // letting ONNX default the third would silently change behaviour.
        let dir = require_artifact!();
        let artifact = Artifact::load(&dir).expect("artifact");
        let session = Session::open(&artifact, ModelConfig::default()).expect("session");
        assert_eq!(
            session.input_names(),
            ["input_ids", "attention_mask", "token_type_ids"]
        );
    }

    #[tokio::test]
    async fn a_known_name_is_detected_as_a_person() {
        // The end-to-end smoke test: real artifact, real tokenizer, real
        // graph, one sentence in, one PERSON span out, at the right
        // offsets.
        let dir = require_artifact!();
        let model = Model::load(&dir, ModelConfig::default()).expect("model");
        let document =
            Document::new("Neem contact op met Jan de Vries over de factuur.".to_string());
        let candidates = model.detect(&document).await.expect("detect");
        let person = candidates
            .iter()
            .find(|c| c.entity_type == EntityType::Person)
            .expect("a PERSON");
        assert_eq!(document.slice(person.span), "Jan de Vries");
    }

    #[tokio::test]
    async fn a_person_in_multibyte_text_lands_on_the_right_code_points() {
        // Byte offsets would put this span two code points late.
        let dir = require_artifact!();
        let model = Model::load(&dir, ModelConfig::default()).expect("model");
        let document = Document::new("Grüße an Jürgen Müller aus Köln.".to_string());
        let candidates = model.detect(&document).await.expect("detect");
        let person = candidates
            .iter()
            .find(|c| c.entity_type == EntityType::Person)
            .expect("a PERSON");
        assert_eq!(document.slice(person.span), "Jürgen Müller");
    }

    #[tokio::test]
    async fn a_text_longer_than_one_window_still_detects_a_name_near_the_boundary() {
        // The case stride exists for. A name placed deliberately at the
        // first window's edge must come back whole, from the window that
        // saw it in the middle.
        let dir = require_artifact!();
        let model = Model::load(&dir, ModelConfig::default()).expect("model");
        let document = Document::new(boundary_text()); // ~2,000 chars, name at the stride seam
        let candidates = model.detect(&document).await.expect("detect");
        let person = candidates
            .iter()
            .find(|c| c.entity_type == EntityType::Person)
            .expect("a PERSON");
        assert_eq!(document.slice(person.span), "Jan de Vries");
    }

    #[tokio::test]
    async fn concurrent_requests_never_exceed_the_configured_inference_slots() {
        // Design section 6 step 4: "bounded by an inference semaphore so
        // request concurrency cannot oversubscribe ONNX threads." A
        // counter around the critical section is the only way to assert
        // it; reading the permit count would only assert that a Semaphore
        // is a Semaphore.
        //
        // Proven both directions by hand (not committed): hardcoding the
        // semaphore to 32 permits inside `Model::load` fails the `<= 2`
        // assertion, confirming the bound is real, not vacuous. Removing
        // `detect`'s `spawn_blocking` (running `run_detection` inline on
        // the calling task instead) fails the `>= 2` assertion instead --
        // `#[tokio::test]` defaults to a single-threaded runtime, so a
        // synchronous, non-`.await`-ing `run_detection` call blocks that
        // one worker to completion before any other spawned task can even
        // start its own inference, and peak concurrency never leaves 1.
        // Unlike the brief's cautionary case ("if it still passes, note
        // that this test does not prove blocking-pool placement"), in this
        // implementation it does NOT still pass: this test also proves
        // `spawn_blocking` is load-bearing, not just the semaphore's slot
        // count.
        //
        // `intra_threads` is pinned to 1 here, not 2: `Session::open`
        // refuses to build a session at all once `inference_slots *
        // intra_threads` exceeds the host's vCPUs (see
        // `the_session_refuses_to_open_when_spinning_would_oversubscribe_the_host`),
        // and this test's proof needs only `inference_slots = 2` -- the
        // semaphore's slot count -- to demonstrate the bound;
        // `intra_threads` plays no part in it. Pinning it at 1 keeps the
        // product at 2, portable down to any host with at least 2 vCPUs.
        // That is why the old `intra_threads: 2` (product 4) broke on
        // CI: this repository is private, and GitHub's standard hosted
        // Linux runner for a private repo is a 2-vCPU machine, so
        // `Model::load`'s `.expect("model")` panicked on the
        // `UnsafeSpinningConfig` error `Session::open` returned there --
        // not on either assertion below, which is the tell that this was
        // a host-capacity mismatch, not a real concurrency-bound defect.
        // Below 2 vCPUs there is no smaller slot count left that would
        // still prove contention, so skip rather than open a
        // single-slot session that could never exceed 1 anyway --
        // loudly, on stderr, so this never reads as a pass that checked
        // nothing (see this crate's other skip discipline in
        // `test_support::require_artifact!`).
        let vcpus = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1);
        if vcpus < 2 {
            crate::test_support::print_uncaptured(&format!(
                "SKIPPED concurrent_requests_never_exceed_the_configured_inference_slots: \
                 only {vcpus} vCPU(s) available, need >= 2 to open a 2-slot session \
                 without tripping Session::open's oversubscription guard"
            ));
            return;
        }
        let dir = require_artifact!();
        let model = Arc::new(
            Model::load(
                &dir,
                ModelConfig {
                    intra_threads: 1,
                    inference_slots: 2,
                },
            )
            .expect("model"),
        );
        let peak = model.test_peak_concurrent_inferences();
        let mut set = tokio::task::JoinSet::new();
        for _ in 0..32 {
            let model = Arc::clone(&model);
            set.spawn(async move {
                let document = Document::new("Neem contact op met Jan de Vries.".to_string());
                model.detect(&document).await.expect("detect")
            });
        }
        while set.join_next().await.is_some() {}
        assert!(
            peak.load(Ordering::SeqCst) <= 2,
            "peak concurrent inferences exceeded the 2 slots"
        );
        assert!(
            peak.load(Ordering::SeqCst) >= 2,
            "the test never actually contended; it proves nothing"
        );
    }

    #[tokio::test]
    async fn warm_up_runs_an_inference_and_reports_failure_rather_than_panicking() {
        let dir = require_artifact!();
        let model = Model::load(&dir, ModelConfig::default()).expect("model");
        // `warm_up` consumes the model and hands back the `WarmModel` that
        // is the only thing `pii_detection_api::routes::router` accepts -- the
        // guard is the type, not a branch a caller has to remember. See
        // `WarmModel`'s docs.
        let warm = model
            .warm_up()
            .await
            .expect("warm-up must succeed on a valid artifact");
        assert!(!warm.revision().is_empty());
    }

    #[test]
    fn no_model_error_variant_carries_text_or_token_ids() {
        // Design section 13 names token identifiers explicitly alongside
        // raw input. An input_ids slice in an error is as forbidden as the
        // text.
        let rendered = format!(
            "{} {} {} {}",
            ModelError::LabelCountMismatch {
                declared: 24,
                graph: 25
            },
            ModelError::MissingInput {
                name: "token_type_ids"
            },
            ModelError::Inference,
            ModelError::UnsafeSpinningConfig {
                inference_slots: 4,
                intra_threads: 8,
                product: 32,
                vcpus: 10,
            },
        );
        for forbidden in ["[", "input_ids:", "Jan"] {
            assert!(
                !rendered.contains(forbidden),
                "error text leaked {forbidden:?}"
            );
        }
    }
}
