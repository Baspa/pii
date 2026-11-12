//! End-to-end latency matrix for `Model::detect`: the number the 50 ms p95
//! gate is actually about, measured across the thread
//! settings `ModelConfig` exposes.
//!
//! # What this measures, and what it deliberately does not
//!
//! One process, one `Model`, one document at a time, `inference_slots = 1`
//! -- no HTTP layer, no concurrent callers, no rules engine running
//! alongside. That is the *floor*: whatever this reports, a served p95 can
//! only be higher. A configuration that misses the budget here cannot be
//! rescued by anything downstream, which is why this is the right
//! instrument for "can the model serve inside the budget at all".
//!
//! # Why workloads are sized in tokens, not characters
//!
//! The gate says "2,000 characters", but the graph never sees a character.
//! It sees padded token positions, and the characters-per-token ratio is a
//! property of the *text*, not of the budget: on this artifact's tokenizer,
//! `benchmark/carriers/` prose measures 4.08-4.58 characters per token
//! across the six supported languages, while a repeated four-word filler
//! sentence measures 3.14 and Chinese measures 1.16. The same "2,000
//! characters" is therefore anywhere from 437 tokens to 1,700, which is the
//! difference between two windows and seven. Sizing the workload in tokens
//! makes the table mean one thing; the report converts back to characters
//! once, using a ratio it states.
//!
//! # Honesty rules this harness follows
//!
//! * **Warm up before timing.** `WARM_UPS` untimed iterations per cell,
//!   after `Model::warm_up`. First-inference cost is allocation, page
//!   faults and lazy thread-pool spin-up; the gate is about steady state.
//! * **Report a distribution.** min / p50 / p90 / p95 / p99 / max / mean
//!   and standard deviation, by nearest-rank on the sorted samples
//!   (`index = ceil(p * n) - 1`) -- never a mean alone, which hides exactly
//!   the tail the budget names.
//! * **Say what the input was.** Every workload prints its character count,
//!   token count, window count and longest window.
//! * **Run enough iterations, and say how many.** `PII_DETECTION_BENCH_ITERS`
//!   (default 200) timed samples per cell, printed with every table.
//! * **Say what else the machine was doing.** The 1-minute load average is
//!   printed before and after every table, because a p95 measured on a
//!   contended host is a statement about the host.
//!
//! # Privacy
//!
//! Design section 13: no raw input, no detected value and no token text is
//! ever printed. Workloads are synthetic filler built in this file (or a
//! file named by `PII_DETECTION_BENCH_TEXT`, which is likewise never echoed), and
//! only counts and timings are reported.
//!
//! Usage:
//!   PII_DETECTION_ARTIFACT_DIR=models/pii-detection-model-full-merged-int8 \
//!     cargo run --release -p pii-detection-model --example latency_matrix

use std::path::PathBuf;
use std::time::Instant;

use pii_detection_engine::text::Document;
use pii_detection_model::artifact::Artifact;
use pii_detection_model::window::Windower;
use pii_detection_model::{Model, ModelConfig};

/// Untimed iterations before every timed batch.
const WARM_UPS: usize = 20;

/// The intra-op thread counts swept by default. `available_parallelism()`
/// on the machine this was written for is 10 (8 performance + 2 efficiency
/// cores), which is the current -- unmeasured -- default.
const THREAD_SWEEP: [usize; 6] = [1, 2, 4, 6, 8, 10];

/// Token counts to size documents at. 466 is 2,000 characters of Dutch
/// carrier prose at its measured 4.30 characters per token; 490 is the
/// French figure (4.08), the densest of the six supported languages; 320
/// and 640 bracket it so the shape of the curve is visible rather than
/// interpolated.
const TOKEN_SWEEP: [usize; 6] = [128, 320, 437, 466, 490, 640];

/// Neutral Dutch filler. No personal data: the two-token name exists only
/// so the decode path has something to decode, and is never printed.
const FILLER: &str = "De gemeente heeft de aanvraag beoordeeld en vastgesteld dat de ingediende documenten volledig zijn. Het besluit wordt binnen zes weken schriftelijk toegelicht aan de betrokken partijen. ";
const NAME: &str = "Jan de Vries. ";

/// Builds a document whose tokenization is as close to `target_tokens` real
/// tokens as the filler's granularity allows, by growing the text and
/// re-encoding rather than assuming a characters-per-token ratio.
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

/// Sorted samples in milliseconds, summarised. Nearest-rank percentiles:
/// `index = ceil(p * n) - 1` over the sorted vector.
struct Summary {
    n: usize,
    min: f64,
    p50: f64,
    p90: f64,
    p95: f64,
    p99: f64,
    max: f64,
    mean: f64,
    sd: f64,
}

fn summarise(mut samples: Vec<f64>) -> Summary {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = samples.len();
    let pick = |p: f64| -> f64 {
        let rank = (p * n as f64).ceil().max(1.0) as usize;
        samples[rank.min(n) - 1]
    };
    let mean = samples.iter().sum::<f64>() / n as f64;
    let variance = samples
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / (n.max(2) - 1) as f64;
    Summary {
        n,
        min: samples[0],
        p50: pick(0.50),
        p90: pick(0.90),
        p95: pick(0.95),
        p99: pick(0.99),
        max: samples[n - 1],
        mean,
        sd: variance.sqrt(),
    }
}

impl Summary {
    fn row(&self, label: &str) -> String {
        format!(
            "| {label} | {n} | {min:.1} | {p50:.1} | {p90:.1} | {p95:.1} | {p99:.1} | {max:.1} | {mean:.1} | {sd:.1} |",
            n = self.n,
            min = self.min,
            p50 = self.p50,
            p90 = self.p90,
            p95 = self.p95,
            p99 = self.p99,
            max = self.max,
            mean = self.mean,
            sd = self.sd,
        )
    }
}

const HEADER: &str = "| case | n | min | p50 | p90 | p95 | p99 | max | mean | sd |\n|---|---|---|---|---|---|---|---|---|---|";

/// The host's 1-minute load average, or `NaN` if the operating system refuses it.
/// Printed around every table: a latency distribution measured while
/// something else owns the cores is a statement about the machine, and
/// this is the cheapest way to make that visible in the output itself
/// rather than in a footnote nobody reads.
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

fn artifact_dir() -> PathBuf {
    std::env::var_os("PII_DETECTION_ARTIFACT_DIR")
        .map(PathBuf::from)
        .expect("set PII_DETECTION_ARTIFACT_DIR to the artifact directory")
}

fn iterations() -> usize {
    std::env::var("PII_DETECTION_BENCH_ITERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(200)
}

async fn time_detect(model: &Model, document: &Document, iters: usize) -> (Summary, usize) {
    for _ in 0..WARM_UPS {
        model.detect(document).await.expect("detect");
    }
    let mut samples = Vec::with_capacity(iters);
    let mut candidates = 0usize;
    for _ in 0..iters {
        let started = Instant::now();
        let found = model.detect(document).await.expect("detect");
        samples.push(started.elapsed().as_secs_f64() * 1e3);
        candidates = found.len();
    }
    (summarise(samples), candidates)
}

#[tokio::main]
async fn main() {
    let dir = artifact_dir();
    let iters = iterations();
    let artifact = Artifact::load(&dir).expect("artifact");
    let windower =
        Windower::new(&dir.join("tokenizer.json"), artifact.manifest.windowing).expect("windower");
    let mut tokenizer =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    tokenizer.with_truncation(None).expect("no truncation");

    // `PII_DETECTION_BENCH_TEXT` names a file whose text is used verbatim as a
    // seventh workload -- the cross-check that synthetic filler at N tokens
    // costs what real prose at N tokens costs. Its content is never echoed.
    let external = std::env::var_os("PII_DETECTION_BENCH_TEXT")
        .map(PathBuf::from)
        .and_then(|path| std::fs::read_to_string(path).ok());

    let mut cases: Vec<(String, String)> = TOKEN_SWEEP
        .iter()
        .map(|&tokens| {
            (
                format!("tok~{tokens}"),
                document_of_tokens(&tokenizer, tokens),
            )
        })
        .collect();
    if let Some(text) = external {
        cases.push(("external".to_string(), text));
    }

    println!("# latency_matrix");
    println!();
    println!("artifact: {}", artifact.manifest.name);
    println!("revision: {}", artifact.manifest.revision);
    println!(
        "quantization: {} {}",
        artifact.manifest.quantization.dtype, artifact.manifest.quantization.config
    );
    println!("ort build: {}", ort::info());
    println!(
        "available_parallelism: {}",
        std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(0)
    );
    println!("timed iterations per cell: {iters} (after {WARM_UPS} warm-ups)");
    println!("percentiles: nearest-rank on sorted samples, milliseconds");
    println!();

    println!("## workload shapes");
    println!();
    println!("| case | chars | real tokens | windows | padded positions | longest window |");
    println!("|---|---|---|---|---|---|");
    for (label, text) in &cases {
        let document = Document::new(text.clone());
        let windows = windower.windows(&document).expect("windows");
        let longest = windows.iter().map(|w| w.input_ids.len()).max().unwrap_or(0);
        let real = tokenizer
            .encode(text.as_str(), false)
            .expect("encode")
            .len();
        println!(
            "| {} | {} | {} | {} | {} | {} |",
            label,
            text.chars().count(),
            real,
            windows.len(),
            windows.len() * longest,
            longest
        );
    }
    println!();

    println!("## tokenisation only (Windower::windows, no inference)");
    println!();
    println!("load average before: {:.2}", load_average());
    println!("{HEADER}");
    for (label, text) in &cases {
        let document = Document::new(text.clone());
        for _ in 0..WARM_UPS {
            windower.windows(&document).expect("windows");
        }
        let mut samples = Vec::with_capacity(iters);
        for _ in 0..iters {
            let started = Instant::now();
            let windows = windower.windows(&document).expect("windows");
            samples.push(started.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box(windows);
        }
        println!("{}", summarise(samples).row(label));
    }
    println!("load average after: {:.2}", load_average());
    println!();

    // The sweep the brief calls the cheapest lever: intra_threads, which
    // today defaults to available_parallelism() by formula, never measured.
    let sweep_label =
        std::env::var("PII_DETECTION_SWEEP_CASE").unwrap_or_else(|_| "tok~466".to_string());
    let sweep_text = cases
        .iter()
        .find(|(label, _)| *label == sweep_label)
        .map(|(_, text)| text.clone())
        .expect("sweep case");
    let document = Document::new(sweep_text);
    println!("## intra_threads sweep, {sweep_label} (detect end to end)");
    println!();
    println!("load average before: {:.2}", load_average());
    println!("{HEADER}");
    for threads in THREAD_SWEEP {
        let model = Model::load(
            &dir,
            ModelConfig {
                intra_threads: threads,
                inference_slots: 1,
            },
        )
        .expect("model");
        let model = model.warm_up().await.expect("warm up");
        let (summary, candidates) = time_detect(&model, &document, iters).await;
        println!(
            "{}",
            summary.row(&format!("intra={threads} (candidates={candidates})"))
        );
    }
    println!("load average after: {:.2}", load_average());
    println!();

    let best: usize = std::env::var("PII_DETECTION_BEST_INTRA")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(6);
    println!("## every workload at intra_threads={best}");
    println!();
    println!("load average before: {:.2}", load_average());
    println!("{HEADER}");
    let model = Model::load(
        &dir,
        ModelConfig {
            intra_threads: best,
            inference_slots: 1,
        },
    )
    .expect("model");
    let model = model.warm_up().await.expect("warm up");
    for (label, text) in &cases {
        let document = Document::new(text.clone());
        let (summary, candidates) = time_detect(&model, &document, iters).await;
        println!(
            "{}",
            summary.row(&format!("{label} (candidates={candidates})"))
        );
    }
    println!("load average after: {:.2}", load_average());
}
