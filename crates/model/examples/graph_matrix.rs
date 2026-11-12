//! Raw-graph latency matrix: ONNX Runtime `Session::run` in isolation, with
//! tokenisation, decoding, the semaphore and the async path all removed.
//!
//! `latency_matrix` answers "what does a request cost". This answers "which
//! part of that is the graph, and which knob moves it" -- two questions
//! that need different instruments, because a knob that moves end-to-end
//! latency by 5 ms is uninterpretable until you know whether the graph is
//! 90% of the number or 30% of it.
//!
//! What it sweeps, all against the same real windows of the same document:
//!
//! * `intra_threads` -- the knob `ModelConfig` exposes and currently
//!   guesses at (`available_parallelism()`), never measured for this
//!   workload.
//! * `inter_threads`, with and without parallel execution -- the graph is a
//!   single sequential encoder stack, so this is expected to do nothing;
//!   "expected" is not "measured".
//! * `GraphOptimizationLevel`, all four -- to establish that Level3 is in
//!   fact the fastest and not merely the highest.
//! * **Batched versus sequential**: every window of one document in one
//!   `run` call, against one `run` call per window.
//! * **Padding to the batch's longest window versus to `max_length`** --
//!   the lever `Session::infer` already pulls, measured rather than
//!   assumed.
//! * **Sequence length and batch size independently**, so cost per padded
//!   token position and cost per window can be separated. That is what
//!   turns "the graph is slow" into a per-layer number a distillation
//!   decision can be made from.
//!
//! `PII_DETECTION_GRAPH` overrides which `.onnx` file is loaded (defaulting to the
//! artifact's own `model.onnx`), so the same harness measures the INT8
//! artifact, the fp32 graph, and any reduced-layer probe graph without a
//! code change. `PII_DETECTION_BENCH_TOKENS` sizes the document in tokens (see
//! `latency_matrix`'s docs for why tokens and not characters).
//!
//! # Privacy
//!
//! Token ids never leave the process: they are built from filler defined in
//! this file, fed straight to the graph, and only timings and shapes are
//! printed. No text, no ids and no decoded value is ever logged.
//!
//! Usage:
//!   PII_DETECTION_ARTIFACT_DIR=models/pii-detection-model-full-merged-int8 \
//!     cargo run --release -p pii-detection-model --example graph_matrix

use std::path::{Path, PathBuf};
use std::time::Instant;

use ort::session::Session as OrtSession;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use pii_detection_engine::text::Document;
use pii_detection_model::artifact::Artifact;
use pii_detection_model::window::{Window, Windower};

const WARM_UPS: usize = 20;
const THREAD_SWEEP: [usize; 6] = [1, 2, 4, 6, 8, 10];
/// Real-token targets for the `tail` section, chosen so the trailing
/// window's own length walks from very short to nearly full in both window
/// classes. Windowing at `max_length=256, stride=32` puts 254 real tokens in
/// the first window and 222 in each one after, so a 2-window document's
/// trailing window is `tokens - 220` positions long and a 3-window
/// document's is `tokens - 442`.
const TAIL_SWEEP: [usize; 10] = [286, 320, 370, 420, 468, 508, 542, 592, 642, 682];

const FILLER: &str = "De gemeente heeft de aanvraag beoordeeld en vastgesteld dat de ingediende documenten volledig zijn. Het besluit wordt binnen zes weken schriftelijk toegelicht aan de betrokken partijen. ";
const NAME: &str = "Jan de Vries. ";

fn document_of_tokens(tokenizer: &tokenizers::Tokenizer, target_tokens: usize) -> String {
    // Word by word, not paragraph by paragraph: the window count is a step
    // function of the token count (a first window holds 254 real tokens and
    // every further window adds `254 - stride`), so overshooting a target
    // by one paragraph can silently add a whole window and a third of the
    // document's compute. Growing a word at a time lands within one token
    // of the target.
    let mut text = String::from(NAME);
    let words: Vec<&str> = FILLER.split_whitespace().collect();
    let mut index = 0usize;
    loop {
        let encoded = tokenizer.encode(text.as_str(), false).expect("encode");
        if encoded.len() >= target_tokens {
            return text;
        }
        text.push_str(words[index % words.len()]);
        text.push(' ');
        index += 1;
    }
}

struct Summary {
    n: usize,
    min: f64,
    p50: f64,
    p95: f64,
    p99: f64,
    max: f64,
    mean: f64,
}

fn summarise(mut samples: Vec<f64>) -> Summary {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = samples.len();
    let pick = |p: f64| -> f64 {
        let rank = (p * n as f64).ceil().max(1.0) as usize;
        samples[rank.min(n) - 1]
    };
    let mean = samples.iter().sum::<f64>() / n as f64;
    Summary {
        n,
        min: samples[0],
        p50: pick(0.50),
        p95: pick(0.95),
        p99: pick(0.99),
        max: samples[n - 1],
        mean,
    }
}

impl Summary {
    fn row(&self, label: &str) -> String {
        format!(
            "| {label} | {n} | {min:.2} | {p50:.2} | {p95:.2} | {p99:.2} | {max:.2} | {mean:.2} |",
            n = self.n,
            min = self.min,
            p50 = self.p50,
            p95 = self.p95,
            p99 = self.p99,
            max = self.max,
            mean = self.mean,
        )
    }
}

const HEADER: &str =
    "| case | n | min | p50 | p95 | p99 | max | mean |\n|---|---|---|---|---|---|---|---|";

fn load_average() -> f64 {
    let mut averages = [0f64; 3];
    // SAFETY: `getloadavg` writes at most `n` doubles into the buffer and
    // reads nothing else; the buffer is exactly 3 doubles and 3 is passed.
    let count = unsafe { getloadavg(averages.as_mut_ptr(), 3) };
    if count >= 1 { averages[0] } else { f64::NAN }
}

unsafe extern "C" {
    fn getloadavg(loadavg: *mut f64, nelem: i32) -> i32;
}

/// Builds the three input tensors for `windows`, padded to `pad_to` (or to
/// the batch's own longest window when `pad_to` is `None`) -- the same
/// layout `Session::infer` builds, reproduced here so this harness can vary
/// the padding target that `infer` fixes. `truncate_to` shortens every row,
/// which is how a sequence-length sweep is run against real windows rather
/// than against invented ids.
fn tensors(
    windows: &[&Window],
    pad_to: Option<usize>,
    truncate_to: Option<usize>,
    pad_token_id: i64,
) -> (Tensor<i64>, Tensor<i64>, Tensor<i64>) {
    let batch = windows.len();
    let longest = windows
        .iter()
        .map(|window| {
            truncate_to
                .unwrap_or(usize::MAX)
                .min(window.input_ids.len())
        })
        .max()
        .unwrap_or(0);
    let max_len = pad_to.unwrap_or(longest).max(longest);

    let mut input_ids = vec![pad_token_id; batch * max_len];
    let mut attention_mask = vec![0i64; batch * max_len];
    let mut token_type_ids = vec![0i64; batch * max_len];
    for (row, window) in windows.iter().enumerate() {
        let offset = row * max_len;
        let len = truncate_to
            .unwrap_or(usize::MAX)
            .min(window.input_ids.len());
        input_ids[offset..offset + len].copy_from_slice(&window.input_ids[..len]);
        attention_mask[offset..offset + len].copy_from_slice(&window.attention_mask[..len]);
        token_type_ids[offset..offset + len].copy_from_slice(&window.token_type_ids[..len]);
    }
    let shape = vec![batch as i64, max_len as i64];
    (
        Tensor::from_array((shape.clone(), input_ids)).expect("input_ids"),
        Tensor::from_array((shape.clone(), attention_mask)).expect("attention_mask"),
        Tensor::from_array((shape, token_type_ids)).expect("token_type_ids"),
    )
}

fn run_once(
    session: &mut OrtSession,
    windows: &[&Window],
    pad_to: Option<usize>,
    truncate_to: Option<usize>,
    pad_id: i64,
) {
    let (input_ids, attention_mask, token_type_ids) = tensors(windows, pad_to, truncate_to, pad_id);
    let outputs = session
        .run(ort::inputs! {
            "input_ids" => input_ids,
            "attention_mask" => attention_mask,
            "token_type_ids" => token_type_ids,
        })
        .expect("run");
    let (_, data) = outputs[0].try_extract_tensor::<f32>().expect("extract");
    std::hint::black_box(data.len());
}

struct Options {
    intra: usize,
    inter: usize,
    parallel: bool,
    level: GraphOptimizationLevel,
    /// `None` leaves ONNX Runtime's default. Spinning makes intra-op
    /// worker threads busy-wait between operators, which is free on an
    /// idle machine and actively harmful on a shared one -- a spinning
    /// thread holds a core the rest of the host wants. It is a p95 knob,
    /// not a p50 knob, which is exactly why it belongs in this matrix.
    spinning: Option<bool>,
    /// Sets a session config entry that changes nothing (`with_prepacking`
    /// to its own default). The control for the spinning result: if merely
    /// *setting* an entry moved the number, the spinning finding would be
    /// an artifact of the call rather than of the setting.
    neutral_entry: bool,
}

impl Options {
    fn threads(intra: usize) -> Options {
        Options {
            intra,
            inter: 1,
            parallel: false,
            level: GraphOptimizationLevel::Level3,
            spinning: None,
            neutral_entry: false,
        }
    }
}

fn build(path: &Path, options: &Options) -> OrtSession {
    let mut builder = OrtSession::builder()
        .expect("builder")
        .with_intra_threads(options.intra)
        .expect("intra")
        .with_inter_threads(options.inter)
        .expect("inter")
        .with_parallel_execution(options.parallel)
        .expect("parallel")
        .with_optimization_level(options.level)
        .expect("level");
    if let Some(spinning) = options.spinning {
        builder = builder.with_intra_op_spinning(spinning).expect("spinning");
    }
    if options.neutral_entry {
        builder = builder.with_prepacking(true).expect("prepacking");
    }
    builder.commit_from_file(path).expect("commit")
}

fn measure(
    path: &Path,
    options: &Options,
    windows: &[&Window],
    pad_to: Option<usize>,
    truncate_to: Option<usize>,
    pad_id: i64,
    iters: usize,
) -> Summary {
    let mut session = build(path, options);
    for _ in 0..WARM_UPS {
        run_once(&mut session, windows, pad_to, truncate_to, pad_id);
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let started = Instant::now();
        run_once(&mut session, windows, pad_to, truncate_to, pad_id);
        samples.push(started.elapsed().as_secs_f64() * 1e3);
    }
    summarise(samples)
}

/// Every window run on its own -- the sequential alternative to batching
/// one document's windows into a single call.
fn measure_sequential(
    path: &Path,
    options: &Options,
    windows: &[&Window],
    pad_id: i64,
    iters: usize,
) -> Summary {
    let mut session = build(path, options);
    for _ in 0..WARM_UPS {
        for window in windows {
            run_once(&mut session, &[window], None, None, pad_id);
        }
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let started = Instant::now();
        for window in windows {
            run_once(&mut session, &[window], None, None, pad_id);
        }
        samples.push(started.elapsed().as_secs_f64() * 1e3);
    }
    summarise(samples)
}

/// Times a strategy that batches every window except a short trailing one
/// and runs that trailing window alone, at its own length. Padding a batch
/// to its longest member means a 66-token tail window is computed as 256
/// positions; splitting pays one extra call to avoid ~190 wasted positions.
/// Which side wins is a measurement, not an argument.
fn measure_split(
    path: &Path,
    options: &Options,
    windows: &[&Window],
    pad_id: i64,
    iters: usize,
) -> Summary {
    let mut session = build(path, options);
    let (head, tail) = windows.split_at(windows.len() - 1);
    for _ in 0..WARM_UPS {
        run_once(&mut session, head, None, None, pad_id);
        run_once(&mut session, tail, None, None, pad_id);
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let started = Instant::now();
        run_once(&mut session, head, None, None, pad_id);
        run_once(&mut session, tail, None, None, pad_id);
        samples.push(started.elapsed().as_secs_f64() * 1e3);
    }
    summarise(samples)
}

fn main() {
    let dir: PathBuf = std::env::var_os("PII_DETECTION_ARTIFACT_DIR")
        .map(PathBuf::from)
        .expect("set PII_DETECTION_ARTIFACT_DIR");
    let graph: PathBuf = std::env::var_os("PII_DETECTION_GRAPH")
        .map(PathBuf::from)
        .unwrap_or_else(|| dir.join("model.onnx"));
    let iters: usize = std::env::var("PII_DETECTION_BENCH_ITERS")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&value| value > 0)
        .unwrap_or(100);
    let tokens: usize = std::env::var("PII_DETECTION_BENCH_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(466);
    let sections: String = std::env::var("PII_DETECTION_SECTIONS")
        .unwrap_or_else(|_| "threads,batch,pad,scale,opt,inter".to_string());
    let want = |name: &str| sections.split(',').any(|section| section == name);

    ort::init().commit();

    let artifact = Artifact::load(&dir).expect("artifact");
    let windower =
        Windower::new(&dir.join("tokenizer.json"), artifact.manifest.windowing).expect("windower");
    let mut tokenizer =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    tokenizer.with_truncation(None).expect("no truncation");
    let pad_id = i64::from(tokenizer.token_to_id("<pad>").expect("pad id"));

    let text = document_of_tokens(&tokenizer, tokens);
    let document = Document::new(text);
    let owned = windower.windows(&document).expect("windows");
    let windows: Vec<&Window> = owned.iter().collect();
    let max_length = artifact.manifest.windowing.max_length;

    println!("# graph_matrix");
    println!();
    println!("graph: {}", graph.display());
    println!("ort build: {}", ort::info());
    println!(
        "workload: {} chars, {} real tokens, {} windows, lengths {:?}",
        document.as_str().chars().count(),
        tokenizer
            .encode(document.as_str(), false)
            .expect("encode")
            .len(),
        windows.len(),
        windows
            .iter()
            .map(|window| window.input_ids.len())
            .collect::<Vec<_>>()
    );
    println!("timed iterations per cell: {iters} (after {WARM_UPS} warm-ups)");
    println!("load average at start: {:.2}", load_average());
    println!();

    if want("threads") {
        println!("## intra_threads, whole document batched into one run");
        println!();
        println!("{HEADER}");
        for intra in THREAD_SWEEP {
            let summary = measure(
                &graph,
                &Options::threads(intra),
                &windows,
                None,
                None,
                pad_id,
                iters,
            );
            println!("{}", summary.row(&format!("intra={intra}")));
        }
        println!("load average after: {:.2}", load_average());
        println!();
    }

    if want("batch") {
        println!("## batching: one run for all windows vs one run per window");
        println!();
        println!("{HEADER}");
        for intra in THREAD_SWEEP {
            let options = Options::threads(intra);
            let batched = measure(&graph, &options, &windows, None, None, pad_id, iters);
            println!("{}", batched.row(&format!("intra={intra} batched")));
            let sequential = measure_sequential(&graph, &options, &windows, pad_id, iters);
            println!("{}", sequential.row(&format!("intra={intra} sequential")));
        }
        println!("load average after: {:.2}", load_average());
        println!();
    }

    if want("pad") {
        println!("## padding: to the batch's longest window vs to max_length={max_length}");
        println!();
        println!("{HEADER}");
        for intra in [4, 6, 8] {
            let options = Options::threads(intra);
            let longest = measure(&graph, &options, &windows, None, None, pad_id, iters);
            println!("{}", longest.row(&format!("intra={intra} pad=longest")));
            let padded = measure(
                &graph,
                &options,
                &windows,
                Some(max_length),
                None,
                pad_id,
                iters,
            );
            println!("{}", padded.row(&format!("intra={intra} pad={max_length}")));
        }
        println!("load average after: {:.2}", load_average());
        println!();
    }

    if want("scale") {
        // Cost per padded token position and cost per window, separated:
        // batch=1 at four sequence lengths, then batch 1..n at full length.
        println!("## scaling: sequence length at batch=1, then batch size at seq=256");
        println!();
        println!("{HEADER}");
        for intra in [4, 6, 8] {
            for seq in [64usize, 128, 192, 256] {
                let summary = measure(
                    &graph,
                    &Options::threads(intra),
                    &windows[..1],
                    Some(seq),
                    Some(seq),
                    pad_id,
                    iters,
                );
                println!(
                    "{}",
                    summary.row(&format!("intra={intra} batch=1 seq={seq}"))
                );
            }
            for batch in 1..=windows.len() {
                let summary = measure(
                    &graph,
                    &Options::threads(intra),
                    &windows[..batch],
                    Some(256),
                    Some(256),
                    pad_id,
                    iters,
                );
                println!(
                    "{}",
                    summary.row(&format!("intra={intra} batch={batch} seq=256"))
                );
            }
        }
        println!("load average after: {:.2}", load_average());
        println!();
    }

    if want("opt") {
        println!("## graph optimization level");
        println!();
        println!("{HEADER}");
        for intra in [6] {
            for (name, level) in [
                ("Disable", GraphOptimizationLevel::Disable),
                ("Level1", GraphOptimizationLevel::Level1),
                ("Level2", GraphOptimizationLevel::Level2),
                ("Level3", GraphOptimizationLevel::Level3),
            ] {
                let options = Options {
                    intra,
                    inter: 1,
                    parallel: false,
                    level,
                    spinning: None,
                    neutral_entry: false,
                };
                let summary = measure(&graph, &options, &windows, None, None, pad_id, iters);
                println!("{}", summary.row(&format!("intra={intra} {name}")));
            }
        }
        println!("load average after: {:.2}", load_average());
        println!();
    }

    if want("inter") {
        println!("## inter_threads and parallel execution (intra=6)");
        println!();
        println!("{HEADER}");
        for (inter, parallel) in [(1, false), (2, false), (2, true), (4, true)] {
            let options = Options {
                intra: 6,
                inter,
                parallel,
                level: GraphOptimizationLevel::Level3,
                spinning: None,
                neutral_entry: false,
            };
            let summary = measure(&graph, &options, &windows, None, None, pad_id, iters);
            println!(
                "{}",
                summary.row(&format!("inter={inter} parallel={parallel}"))
            );
        }
        println!("load average after: {:.2}", load_average());
        println!();
    }

    if want("spin") {
        println!("## intra-op spinning");
        println!();
        println!("{HEADER}");
        for intra in [4, 6, 8, 10] {
            for spinning in [true, false] {
                let options = Options {
                    intra,
                    inter: 1,
                    parallel: false,
                    level: GraphOptimizationLevel::Level3,
                    spinning: Some(spinning),
                    neutral_entry: false,
                };
                let summary = measure(&graph, &options, &windows, None, None, pad_id, iters);
                println!(
                    "{}",
                    summary.row(&format!("intra={intra} spinning={spinning}"))
                );
            }
        }
        println!("load average after: {:.2}", load_average());
        println!();
    }

    if want("split") {
        println!("## trailing short window: batched with the rest vs run alone");
        println!();
        println!(
            "window lengths: {:?}",
            windows
                .iter()
                .map(|window| window.input_ids.len())
                .collect::<Vec<_>>()
        );
        println!("{HEADER}");
        for spinning in [None, Some(true)] {
            for intra in [4, 6] {
                let options = Options {
                    intra,
                    inter: 1,
                    parallel: false,
                    level: GraphOptimizationLevel::Level3,
                    spinning,
                    neutral_entry: false,
                };
                let label = match spinning {
                    Some(true) => "spin",
                    Some(false) => "nospin",
                    None => "default",
                };
                let batched = measure(&graph, &options, &windows, None, None, pad_id, iters);
                println!(
                    "{}",
                    batched.row(&format!("intra={intra} {label} all batched"))
                );
                if windows.len() > 1 {
                    let split = measure_split(&graph, &options, &windows, pad_id, iters);
                    println!(
                        "{}",
                        split.row(&format!("intra={intra} {label} split tail"))
                    );
                }
            }
        }
        println!("load average after: {:.2}", load_average());
        println!();
    }

    if want("tail") {
        // The threshold question `Session::infer`'s batching rule needs an
        // answer to, measured at the shipped configuration rather than
        // reasoned about: splitting a trailing window out of the batch
        // avoids padding it up to the batch's longest, but it also pays a
        // second `run` call whose batch is 1, and a batch of 1 is measurably
        // less efficient per position than a batch of 2 or 3. Sweeping the
        // trailing window's own length across both the 2-window and the
        // 3-window class is what locates the crossover.
        println!(
            "## trailing window length: batched with the rest vs run alone (intra=4, spinning)"
        );
        println!();
        println!("{HEADER}");
        let options = Options {
            intra: 4,
            inter: 1,
            parallel: false,
            level: GraphOptimizationLevel::Level3,
            spinning: Some(true),
            neutral_entry: false,
        };
        for target in TAIL_SWEEP {
            let text = document_of_tokens(&tokenizer, target);
            let document = Document::new(text);
            let owned = windower.windows(&document).expect("windows");
            let windows: Vec<&Window> = owned.iter().collect();
            if windows.len() < 2 {
                continue;
            }
            let lengths: Vec<usize> = windows
                .iter()
                .map(|window| window.input_ids.len())
                .collect();
            let batched = measure(&graph, &options, &windows, None, None, pad_id, iters);
            println!("{}", batched.row(&format!("{lengths:?} batched")));
            let split = measure_split(&graph, &options, &windows, pad_id, iters);
            println!("{}", split.row(&format!("{lengths:?} split")));
            let sequential = measure_sequential(&graph, &options, &windows, pad_id, iters);
            println!("{}", sequential.row(&format!("{lengths:?} sequential")));
        }
        println!("load average after: {:.2}", load_average());
        println!();
    }

    if want("control") {
        // Three variants, interleaved across repeats so a drifting host
        // cannot be mistaken for an effect: the default (no entry set),
        // spinning explicitly on, and a config entry that sets a value to
        // its own default.
        println!("## control: default vs explicit spinning vs a no-op config entry (intra=6)");
        println!();
        println!("{HEADER}");
        for round in 1..=3 {
            for (name, spinning, neutral) in [
                ("default", None, false),
                ("spinning=true", Some(true), false),
                ("neutral entry", None, true),
            ] {
                let options = Options {
                    intra: 6,
                    inter: 1,
                    parallel: false,
                    level: GraphOptimizationLevel::Level3,
                    spinning,
                    neutral_entry: neutral,
                };
                let summary = measure(&graph, &options, &windows, None, None, pad_id, iters);
                println!("{}", summary.row(&format!("round {round} {name}")));
            }
        }
        println!("load average after: {:.2}", load_average());
    }
}
