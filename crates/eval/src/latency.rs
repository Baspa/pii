//! The latency measurement's own types: the corpus it is measured over, the
//! distribution it produces, and the provenance stamp without which that
//! distribution cannot be written to disk.
//!
//! # Why this is a module and not three lines in a binary
//!
//! Design section 11 sets one latency gate: *"p95 under 50 ms at 2,000
//! characters on the chosen task size with the shipped artifact."* Three
//! phrases in that sentence are load-bearing, and each one is a way to
//! report a number that is true and yet not the number the gate means:
//!
//! * **"p95"**: a mean, or a p50, quoted where a tail was asked for. This
//!   project has already shipped one badly wrong figure from an untested
//!   measuring instrument (`training/fixture_scoring.py`: 0.6711 where the
//!   truth was 0.9145), so [`percentile`] is a tested function with one
//!   definition, used by every caller in this crate rather than re-derived
//!   per binary.
//! * **"at 2,000 characters"**: 2,000 *code points*, not 2,000 bytes. The
//!   German, French, Spanish and Italian carriers are multibyte; a corpus
//!   cut by byte length would be materially shorter in tokens, and every
//!   p95 taken over it optimistic. [`Corpus`] is built and checked in code
//!   points, and [`CorpusText::chars`] is a code-point count.
//! * **"on the chosen task size"**: the spec's own comparison table is
//!   stamped *"An ARM proxy for Graviton, not the target instance"*. A
//!   [`LatencyRun`] therefore cannot be serialised without a [`Hardware`]
//!   stamp: the type refuses, rather than a reviewer having to remember.
//!
//! # Nothing here prints or stores text
//!
//! [`Corpus`] holds carrier prose with fixture identifiers injected into it,
//! and it is written to a gitignored path. No function in this module, and
//! no binary that uses it, puts a corpus text, an injected value or a token
//! into a log line, an error message or a report field: the design.
//! Everything reported about a text is a count.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use serde::de::Error as _;
use serde::ser::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// The input length the design specifies the gate at, in code points.
pub const TARGET_CHARS: usize = 2_000;

/// How far a corpus text may sit from [`TARGET_CHARS`] and still count as a
/// 2,000-character text. Prose cannot be cut to an exact code-point count
/// without cutting mid-word, and a mid-word cut changes the tokenisation
/// this whole measurement is about.
pub const CHARS_TOLERANCE: usize = 50;

/// The six languages the model's quality is scoped and gated on (design
/// section 2). The corpus is drawn from these and only these: a latency
/// figure taken over languages the model is not gated on would be measuring
/// a tokenizer's behaviour on text nobody ships.
pub const CORPUS_LANGUAGES: [&str; 6] = ["nl", "de", "en", "fr", "es", "it"];

/// How many texts [`build_corpus`] produces by default: roughly 33 per
/// language.
pub const DEFAULT_CORPUS_TEXTS: usize = 200;

// ---------------------------------------------------------------------------
// Percentiles
// ---------------------------------------------------------------------------

/// Nearest-rank percentile over an already-sorted slice of milliseconds.
/// `p` is a fraction in `[0, 1]`.
///
/// Nearest rank (`index = ceil(p * n) - 1`) rather than any interpolating
/// definition, because an interpolated tail invents a value that no request
/// actually took. For `n = 100` and `p = 0.95` this is the 95th-fastest
/// sample: 95% of requests were at most this fast.
pub fn percentile(sorted_ms: &[f64], p: f64) -> Option<f64> {
    if sorted_ms.is_empty() {
        return None;
    }
    let n = sorted_ms.len();
    let rank = (p.clamp(0.0, 1.0) * n as f64).ceil().max(1.0);
    let index = (rank as usize).min(n) - 1;
    sorted_ms.get(index).copied()
}

fn sorted(mut samples: Vec<f64>) -> Vec<f64> {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    samples
}

// ---------------------------------------------------------------------------
// The hardware stamp
// ---------------------------------------------------------------------------

/// The machine a [`LatencyRun`] was taken on, and what that machine is a
/// proxy for.
///
/// A newtype around `Option<String>` with a hand-written [`Serialize`] that
/// **fails** when the stamp is absent. That is the whole point: a run
/// without its hardware cannot be written to a file, so it cannot be read
/// back by `eval` and turned into a gate verdict, so a number can never
/// reach a report without the machine that produced it. "A number without
/// its command does not go in a report", enforced by the type rather than
/// by a reviewer noticing.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Hardware(Option<String>);

/// The error a caller sees when an unstamped run is serialised.
pub const UNSTAMPED_MESSAGE: &str = "a LatencyRun without a hardware stamp cannot be serialised: every latency number is a \
     statement about a machine, and the machine goes in the file beside it (the design, \
     'An ARM proxy for Graviton, not the target instance')";

impl Hardware {
    /// Stamps a run with a machine description. Whitespace-only is the same
    /// as absent: a stamp of `" "` is not provenance.
    pub fn new(description: impl Into<String>) -> Hardware {
        let description = description.into();
        if description.trim().is_empty() {
            Hardware(None)
        } else {
            Hardware(Some(description))
        }
    }

    /// No stamp. Serialising this is an error, by design.
    pub fn absent() -> Hardware {
        Hardware(None)
    }

    pub fn as_str(&self) -> Option<&str> {
        self.0.as_deref()
    }
}

impl fmt::Display for Hardware {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0.as_deref().unwrap_or("<unstamped>"))
    }
}

impl Serialize for Hardware {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.0.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
            Some(description) => serializer.serialize_str(description),
            None => Err(S::Error::custom(UNSTAMPED_MESSAGE)),
        }
    }
}

impl<'de> Deserialize<'de> for Hardware {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Hardware, D::Error> {
        let description = String::deserialize(deserializer)?;
        if description.trim().is_empty() {
            return Err(D::Error::custom(UNSTAMPED_MESSAGE));
        }
        Ok(Hardware(Some(description)))
    }
}

// ---------------------------------------------------------------------------
// The run
// ---------------------------------------------------------------------------

/// The observed distribution of one dimension of a run, in milliseconds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Distribution {
    pub n: usize,
    pub min_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
    /// Reported **beside** the percentiles, never instead of them, so that a
    /// reader can see how far the tail is from the middle. The gate reads
    /// `p95_ms` and nothing else.
    pub mean_ms: f64,
    pub stddev_ms: f64,
    /// Counts per 10 ms bucket, keyed by the bucket's lower edge. The shape
    /// of a tail is evidence about its cause: a smooth spread is the
    /// service's own variance, while a tight mode plus a handful of far
    /// outliers is a host that stalled. A p95 quoted without it cannot
    /// distinguish the two, and this project's previous latency work was
    /// invalidated by exactly that ambiguity.
    pub histogram_10ms: BTreeMap<u64, usize>,
}

impl Distribution {
    pub fn from_samples(samples: Vec<f64>) -> Distribution {
        let samples = sorted(samples);
        let n = samples.len();
        if n == 0 {
            return Distribution {
                n: 0,
                min_ms: f64::NAN,
                p50_ms: f64::NAN,
                p95_ms: f64::NAN,
                p99_ms: f64::NAN,
                max_ms: f64::NAN,
                mean_ms: f64::NAN,
                stddev_ms: f64::NAN,
                histogram_10ms: BTreeMap::new(),
            };
        }
        let mean = samples.iter().sum::<f64>() / n as f64;
        let variance = samples
            .iter()
            .map(|sample| (sample - mean) * (sample - mean))
            .sum::<f64>()
            / n as f64;
        let mut histogram: BTreeMap<u64, usize> = BTreeMap::new();
        for sample in &samples {
            let bucket = (sample.max(0.0) / 10.0).floor() as u64 * 10;
            *histogram.entry(bucket).or_insert(0) += 1;
        }
        Distribution {
            n,
            min_ms: samples.first().copied().unwrap_or(f64::NAN),
            p50_ms: percentile(&samples, 0.50).unwrap_or(f64::NAN),
            p95_ms: percentile(&samples, 0.95).unwrap_or(f64::NAN),
            p99_ms: percentile(&samples, 0.99).unwrap_or(f64::NAN),
            max_ms: samples.last().copied().unwrap_or(f64::NAN),
            mean_ms: mean,
            stddev_ms: variance.sqrt(),
            histogram_10ms: histogram,
        }
    }
}

/// What the host was doing while the run was taken.
///
/// The one threat that invalidated every p95 in
/// `training/reports/latency-investigation.md`: `min` and `p50` reproduced
/// within 1 ms across seven runs on that host, and `p95` did not, because
/// the tail being measured was the machine's and not the model's. So the
/// load average is sampled throughout a run and travels with the number.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HostLoad {
    pub samples: usize,
    /// `None` where the host's load average could not be read at all --
    /// which is itself worth recording, because it means the run cannot say
    /// how quiet the machine was.
    pub min: Option<f64>,
    pub median: Option<f64>,
    pub max: Option<f64>,
    /// vCPUs the *measuring* host has, so `max` can be read as a fraction.
    pub host_vcpus: usize,
}

/// How the samples were taken. The gate is the end-to-end HTTP number and
/// nothing else: an in-process `Model::detect` timing hides the HTTP layer,
/// the rules engine running alongside, the semaphore, serialisation and the
/// async runtime, so it is the floor a served p95 can only be above. The
/// distinction is a field rather than a comment so that
/// `gates::evaluate_latency` can refuse an in-process run outright instead
/// of a reader having to notice which one they are holding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Transport {
    /// `POST /analyze` over HTTP, timed from a separate process, from the
    /// request being written to the response being fully read.
    Http,
    /// `Model::detect` called directly in this process. Informational.
    InProcess,
}

/// Everything about a run that is not a millisecond: what was measured, on
/// what, with which knobs. Every field here is required to stamp a run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    /// See [`Transport`]. The gate accepts [`Transport::Http`] only.
    pub transport: Transport,
    /// The machine. Serialising a run without this fails: see [`Hardware`].
    pub hardware: Hardware,
    /// What that machine stands in for, said out loud. The spec's own table
    /// says "An ARM proxy for Graviton, not the target instance"; a run that
    /// cannot say the equivalent has not retired that caveat either.
    pub proxy_for: String,
    /// `intra_threads` the service was configured with.
    pub threads: usize,
    /// `inference_slots` the service was configured with.
    pub slots: usize,
    /// Concurrent callers the harness ran. `1` is isolated latency; more is
    /// queueing latency.
    pub concurrency: usize,
    /// The ONNX Runtime the *service* reported, not the one this harness
    /// links.
    pub ort_version: String,
    /// Whether `GET /capabilities` advertised `PERSON`: that is, whether
    /// the model was actually in the loop. A p95 taken against a rules-only
    /// service is not the p95 the gate means.
    pub model_loaded: bool,
    /// Untimed requests issued before the first timed one.
    pub warmup_n: usize,
    /// The exact command that produced this run.
    pub command: String,
    pub measured_unix_secs: u64,
    pub host_load: HostLoad,
    /// Free-form caveats the run wants to carry. Never text from the corpus.
    #[serde(default)]
    pub notes: Vec<String>,
}

/// One latency measurement: a distribution, the input it was taken over, and
/// the provenance it may not be separated from.
///
/// Construct with [`LatencyRun::from_samples`], which deliberately yields an
/// **unstamped** run, then [`LatencyRun::stamp`] it. A run is born without
/// provenance and cannot be written until it has some.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatencyRun {
    /// End-to-end `POST /analyze`, from the request being written to the
    /// response being fully read, from a separate process. **This is the
    /// gate.**
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub n: usize,
    /// The median code-point length of the texts actually sent. The gate
    /// checks this against [`TARGET_CHARS`].
    pub chars: usize,
    pub chars_min: usize,
    pub chars_max: usize,
    /// The full end-to-end distribution `p50_ms`/`p95_ms`/`p99_ms` are drawn
    /// from, so no reader has to take three numbers on trust.
    pub end_to_end: Distribution,
    /// In-process `Model::detect` over the same corpus, when the harness was
    /// asked for it. Informational: it hides the HTTP layer, the rules
    /// engine, the semaphore and serialisation, so it is the floor and never
    /// the gate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_only: Option<Distribution>,
    /// The corpus's token and window shape: the arithmetic the design asked
    /// to see checked ("roughly three windows per 2,000-character text").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corpus_shape: Option<CorpusShape>,
    /// The same samples, split by how many windows the text they were taken
    /// over produced.
    ///
    /// At 2,000 code points a real carrier text falls on either side of the
    /// window cliff (`max_length=256`, `stride=32` holds 254 real tokens in
    /// the first window and 222 in each one after, so 476 real tokens is the
    /// boundary), and the two sides differ by 50% of the compute. A single
    /// p95 over a mixed population is therefore a statement about the mix as
    /// much as about the service, and which side the p95 lands in decides
    /// which escalation a miss calls for. Empty when the run was not given
    /// an artifact to window the corpus with.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub by_windows: BTreeMap<usize, Distribution>,
    pub provenance: Provenance,
}

impl LatencyRun {
    /// Builds a run from end-to-end samples in milliseconds. The returned
    /// run is **unstamped**: it has a distribution and no provenance, and
    /// serialising it fails until [`LatencyRun::stamp`] is called.
    pub fn from_samples(samples: Vec<f64>, chars: usize) -> LatencyRun {
        let end_to_end = Distribution::from_samples(samples);
        LatencyRun {
            p50_ms: end_to_end.p50_ms,
            p95_ms: end_to_end.p95_ms,
            p99_ms: end_to_end.p99_ms,
            n: end_to_end.n,
            chars,
            chars_min: chars,
            chars_max: chars,
            end_to_end,
            model_only: None,
            corpus_shape: None,
            by_windows: BTreeMap::new(),
            provenance: Provenance {
                transport: Transport::Http,
                hardware: Hardware::absent(),
                proxy_for: String::new(),
                threads: 0,
                slots: 0,
                concurrency: 0,
                ort_version: String::new(),
                model_loaded: false,
                warmup_n: 0,
                command: String::new(),
                measured_unix_secs: 0,
                host_load: HostLoad {
                    samples: 0,
                    min: None,
                    median: None,
                    max: None,
                    host_vcpus: 0,
                },
                notes: Vec::new(),
            },
        }
    }

    /// A run carrying a distribution and no provenance: exactly what
    /// [`LatencyRun::from_samples`] returns before it is stamped. Named so
    /// that the test proving such a run cannot be serialised reads as the
    /// thing it is testing.
    pub fn unstamped() -> LatencyRun {
        LatencyRun::from_samples(vec![1.0], TARGET_CHARS)
    }

    pub fn stamp(mut self, provenance: Provenance) -> LatencyRun {
        self.provenance = provenance;
        self
    }

    /// Whether this run's input length is the length the gate is specified
    /// at, within [`CHARS_TOLERANCE`].
    pub fn measured_at_gate_length(&self) -> bool {
        self.chars.abs_diff(TARGET_CHARS) <= CHARS_TOLERANCE
    }
}

// ---------------------------------------------------------------------------
// The corpus
// ---------------------------------------------------------------------------

/// One 2,000-character text. `chars` is a code-point count, always.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorpusText {
    pub id: String,
    pub lang: String,
    pub chars: usize,
    /// Identifiers injected into this text, as a count. Never the values.
    pub identifiers: usize,
    pub text: String,
}

/// The corpus's token and window shape, computed against the shipped
/// tokenizer by whoever has one loaded. Counts only.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorpusShape {
    pub texts: usize,
    pub tokens_min: usize,
    pub tokens_median: usize,
    pub tokens_max: usize,
    pub windows_min: usize,
    pub windows_median: usize,
    pub windows_max: usize,
    /// windows -> how many texts had that many. Our arithmetic says
    /// "roughly three"; this is where that claim is checked rather than
    /// repeated.
    pub windows_histogram: BTreeMap<usize, usize>,
    /// What `pii_detection_model::session::batches` actually feeds ONNX Runtime,
    /// summed over the calls it plans for one document.
    pub padded_positions_median: usize,
    /// What one batch per document, padded to that document's longest
    /// window, would feed ONNX Runtime -- the shape `Session::infer` had
    /// before the trailing-window split, kept so a run file states the
    /// padding it removed rather than only the padding it left.
    #[serde(default)]
    pub padded_positions_unsplit_median: usize,
    /// One entry per window count, so the escalation path's first step --
    /// "re-check the observed padded-token count per request against the
    /// real token count" -- can be answered from the run's own file.
    #[serde(default)]
    pub by_windows: BTreeMap<usize, WindowClass>,
}

/// What one window-count class of the corpus costs the graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WindowClass {
    pub texts: usize,
    /// Median sum of the class's actual window lengths -- what the graph
    /// would compute with no padding at all.
    pub tokens_median: usize,
    /// Median padded positions -- what `Session::infer` actually feeds
    /// ONNX, summed over every call
    /// `pii_detection_model::session::batches` plans for the document. Read from
    /// the shipped planner, never from a copy of its rule: a measuring
    /// instrument with its own copy of the batching measures its own copy.
    pub padded_positions_median: usize,
    /// Median `batch x longest window` -- what a single call per document
    /// would feed ONNX, which is what `Session::infer` did before the
    /// trailing-window split. The difference between this and
    /// `padded_positions_median` is the padding this change removed.
    #[serde(default)]
    pub padded_positions_unsplit_median: usize,
    /// The fraction of `padded_positions_median` that is pure padding. A
    /// document that just crosses the window cliff has a short trailing
    /// window, and padding it to the batch's longest is compute spent on
    /// nothing (`training/reports/latency-investigation.md` §4.3).
    pub padding_waste_fraction: f64,
}

/// A generated latency corpus. Written to a gitignored path and regenerated
/// by command; never committed, because it carries carrier prose from
/// `benchmark/carriers/` with identifiers spliced into it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Corpus {
    pub built_unix_secs: u64,
    pub command: String,
    pub target_chars: usize,
    pub tolerance_chars: usize,
    /// The identifier density the fixture itself carries, measured from
    /// `benchmark/cases/` rather than chosen, in identifiers per 1,000 code
    /// points.
    pub identifier_density_per_1000_chars: f64,
    pub texts: Vec<CorpusText>,
}

impl Corpus {
    pub fn chars_median(&self) -> usize {
        let mut lengths: Vec<usize> = self.texts.iter().map(|text| text.chars).collect();
        lengths.sort_unstable();
        lengths.get(lengths.len() / 2).copied().unwrap_or(0)
    }

    pub fn chars_min(&self) -> usize {
        self.texts.iter().map(|text| text.chars).min().unwrap_or(0)
    }

    pub fn chars_max(&self) -> usize {
        self.texts.iter().map(|text| text.chars).max().unwrap_or(0)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CorpusError {
    #[error("cannot read {path}: {kind}")]
    Read {
        path: PathBuf,
        kind: std::io::ErrorKind,
    },
    #[error("{path} is not the expected shape at line {line} column {column}")]
    Malformed {
        path: PathBuf,
        line: usize,
        column: usize,
    },
    #[error(
        "language {lang} has {available} code points of carrier prose, too few to cut \
         {wanted} text(s) of {TARGET_CHARS} code points"
    )]
    NotEnoughProse {
        lang: &'static str,
        available: usize,
        wanted: usize,
    },
    #[error(
        "built a text of {chars} code points, outside the {TARGET_CHARS} +/- \
         {CHARS_TOLERANCE} band the gate is specified at"
    )]
    OutOfBand { chars: usize },
    #[error("no carrier prose found under {path}")]
    NoCarriers { path: PathBuf },
}

fn read_to_string(path: &Path) -> Result<String, CorpusError> {
    std::fs::read_to_string(path).map_err(|error| CorpusError::Read {
        path: path.to_path_buf(),
        kind: error.kind(),
    })
}

/// A deterministic, seeded shuffle source. The corpus must be reproducible:
/// two runs of `--build-corpus` on the same inputs produce the same texts,
/// so a p95 taken a week apart is taken over the same work.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 16
    }

    fn pick<'a, T>(&mut self, items: &'a [T]) -> Option<&'a T> {
        if items.is_empty() {
            return None;
        }
        items.get(self.next() as usize % items.len())
    }
}

struct FixtureIdentifiers {
    density_per_1000_chars: f64,
    by_language: BTreeMap<String, Vec<String>>,
}

/// Reads the benchmark fixture through the crate's own loader (never a
/// second, subtly different parse of the same files) for the two things the
/// corpus needs from it: how dense identifiers are, and what an identifier of
/// a given language looks like.
fn read_fixture_identifiers(cases_dir: &Path) -> Result<FixtureIdentifiers, CorpusError> {
    let cases = crate::load(cases_dir).map_err(|_| CorpusError::Read {
        path: cases_dir.to_path_buf(),
        kind: std::io::ErrorKind::InvalidData,
    })?;

    let mut by_language: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut total_chars = 0usize;
    let mut total_values = 0usize;

    for case in &cases {
        total_chars += case.text.chars().count();
        total_values += case.must_mask.len();
        let bucket = by_language.entry(case.lang.clone()).or_default();
        for value in &case.must_mask {
            bucket.push(value.as_str().to_string());
        }
    }

    for values in by_language.values_mut() {
        values.sort();
        values.dedup();
    }

    let density_per_1000_chars = if total_chars == 0 {
        0.0
    } else {
        1_000.0 * total_values as f64 / total_chars as f64
    };

    Ok(FixtureIdentifiers {
        density_per_1000_chars,
        by_language,
    })
}

/// Reads every carrier under `carriers_dir` whose filename starts with
/// `<lang>-`, in sorted order, and returns their prose as one word list per
/// carrier. Read-only: nothing under `benchmark/` is written by this crate.
fn read_carriers(carriers_dir: &Path, lang: &str) -> Result<Vec<Vec<String>>, CorpusError> {
    let entries = std::fs::read_dir(carriers_dir).map_err(|error| CorpusError::Read {
        path: carriers_dir.to_path_buf(),
        kind: error.kind(),
    })?;
    let mut paths: Vec<PathBuf> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let is_txt = path.extension().is_some_and(|ext| ext == "txt");
        let matches_lang = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with(&format!("{lang}-")));
        if is_txt && matches_lang {
            paths.push(path);
        }
    }
    paths.sort();

    let mut carriers = Vec::new();
    for path in paths {
        let raw = read_to_string(&path)?;
        let words: Vec<String> = raw.split_whitespace().map(str::to_string).collect();
        if !words.is_empty() {
            carriers.push(words);
        }
    }
    Ok(carriers)
}

/// Builds `wanted` texts of [`TARGET_CHARS`] ± [`CHARS_TOLERANCE`] code
/// points, roughly evenly split across [`CORPUS_LANGUAGES`], from carrier
/// prose, with identifiers spliced in at the density the benchmark fixture
/// itself carries.
///
/// Deterministic given the same inputs and `wanted`.
pub fn build_corpus(
    carriers_dir: &Path,
    cases_dir: &Path,
    wanted: usize,
    command: String,
) -> Result<Corpus, CorpusError> {
    let fixture = read_fixture_identifiers(cases_dir)?;
    let mut texts = Vec::with_capacity(wanted);
    let languages = CORPUS_LANGUAGES.len();

    for (index, lang) in CORPUS_LANGUAGES.iter().enumerate() {
        // Spread the remainder over the first languages rather than
        // dropping it, so `wanted` texts is exactly `wanted` texts.
        let share = wanted / languages + usize::from(index < wanted % languages);
        if share == 0 {
            continue;
        }
        let carriers = read_carriers(carriers_dir, lang)?;
        if carriers.is_empty() {
            return Err(CorpusError::NoCarriers {
                path: carriers_dir.to_path_buf(),
            });
        }
        let available: usize = carriers
            .iter()
            .map(|words| {
                words
                    .iter()
                    .map(|word| word.chars().count() + 1)
                    .sum::<usize>()
            })
            .sum();
        if available < TARGET_CHARS {
            return Err(CorpusError::NotEnoughProse {
                lang,
                available,
                wanted: share,
            });
        }
        let empty = Vec::new();
        let identifiers = fixture.by_language.get(*lang).unwrap_or(&empty);

        // Seeded per language so adding a language cannot renumber another
        // language's texts.
        let mut rng = Lcg(0x1d10_4a00 ^ (index as u64 + 1).wrapping_mul(0x9e37_79b9));
        // A rolling cursor through the concatenated prose, so consecutive
        // texts of one language are different prose rather than the same
        // opening paragraph 33 times.
        let mut carrier_index = 0usize;
        let mut word_cursor = 0usize;

        for nth in 0..share {
            let text = build_text(
                &carriers,
                &mut carrier_index,
                &mut word_cursor,
                identifiers,
                fixture.density_per_1000_chars,
                &mut rng,
            )?;
            texts.push(CorpusText {
                id: format!("{lang}-{nth:03}"),
                lang: (*lang).to_string(),
                chars: text.0.chars().count(),
                identifiers: text.1,
                text: text.0,
            });
        }
    }

    Ok(Corpus {
        built_unix_secs: unix_secs(),
        command,
        target_chars: TARGET_CHARS,
        tolerance_chars: CHARS_TOLERANCE,
        identifier_density_per_1000_chars: fixture.density_per_1000_chars,
        texts,
    })
}

/// Cuts one text of [`TARGET_CHARS`] ± [`CHARS_TOLERANCE`] code points,
/// reserving room for the identifiers it will splice in so the finished text,
/// not the prose it started as, is the length the gate is specified at.
fn build_text(
    carriers: &[Vec<String>],
    carrier_index: &mut usize,
    word_cursor: &mut usize,
    identifiers: &[String],
    density_per_1000_chars: f64,
    rng: &mut Lcg,
) -> Result<(String, usize), CorpusError> {
    let wanted_identifiers = if identifiers.is_empty() {
        0
    } else {
        // At least one, so every text exercises the rules engine and the
        // resolver as well as the graph.
        ((density_per_1000_chars * TARGET_CHARS as f64 / 1_000.0).round() as usize).max(1)
    };
    let chosen: Vec<&str> = (0..wanted_identifiers)
        .filter_map(|_| rng.pick(identifiers).map(String::as_str))
        .collect();
    // Each identifier costs its own code points plus the space before it.
    let reserved: usize = chosen.iter().map(|value| value.chars().count() + 1).sum();
    let prose_budget = TARGET_CHARS.saturating_sub(reserved);

    let mut words: Vec<String> = Vec::new();
    let mut chars = 0usize;
    // Bounded: `carriers` is non-empty and every carrier has words, so this
    // terminates once `prose_budget` is reached; the guard is against a
    // pathological carrier set rather than the normal path.
    let mut visited = 0usize;
    let ceiling = carriers.iter().map(Vec::len).sum::<usize>() + carriers.len() + 1;
    while chars < prose_budget && visited < ceiling {
        let carrier = match carriers.get(*carrier_index % carriers.len()) {
            Some(carrier) => carrier,
            None => break,
        };
        match carrier.get(*word_cursor) {
            Some(word) => {
                let cost = word.chars().count() + usize::from(!words.is_empty());
                if chars + cost > prose_budget + CHARS_TOLERANCE {
                    // A very long word would overshoot the band; skip it
                    // rather than emit an out-of-band text.
                    *word_cursor += 1;
                    visited += 1;
                    continue;
                }
                words.push(word.clone());
                chars += cost;
                *word_cursor += 1;
            }
            None => {
                *carrier_index += 1;
                *word_cursor = 0;
            }
        }
        visited += 1;
    }

    // Splice the identifiers in at evenly spaced word boundaries, so they
    // are distributed through the windows rather than bunched in the first.
    let mut spliced = words;
    let count = chosen.len();
    for (nth, value) in chosen.iter().enumerate() {
        let position = if count == 0 {
            0
        } else {
            (spliced.len() + nth) * (nth + 1) / (count + 1)
        };
        let position = position.min(spliced.len());
        spliced.insert(position, (*value).to_string());
    }

    let text = spliced.join(" ");
    let chars = text.chars().count();
    if chars.abs_diff(TARGET_CHARS) > CHARS_TOLERANCE {
        return Err(CorpusError::OutOfBand { chars });
    }
    Ok((text, count))
}

pub fn write_corpus(path: &Path, corpus: &Corpus) -> Result<(), CorpusError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| CorpusError::Read {
            path: parent.to_path_buf(),
            kind: error.kind(),
        })?;
    }
    let json = serde_json::to_string_pretty(corpus).map_err(|error| CorpusError::Malformed {
        path: path.to_path_buf(),
        line: error.line(),
        column: error.column(),
    })?;
    std::fs::write(path, json).map_err(|error| CorpusError::Read {
        path: path.to_path_buf(),
        kind: error.kind(),
    })
}

pub fn load_corpus(path: &Path) -> Result<Corpus, CorpusError> {
    let raw = read_to_string(path)?;
    serde_json::from_str(&raw).map_err(|error| CorpusError::Malformed {
        path: path.to_path_buf(),
        line: error.line(),
        column: error.column(),
    })
}

pub fn unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

/// The 1-minute load average of the host this process runs on, or `None`
/// where it cannot be read. Used to say how quiet the measuring host was,
/// which is the difference between a quotable p95 and a story about a
/// machine.
pub fn load_average() -> Option<f64> {
    if let Ok(raw) = std::fs::read_to_string("/proc/loadavg") {
        return raw.split_whitespace().next()?.parse().ok();
    }
    let output = std::process::Command::new("/usr/sbin/sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()?;
    let text = String::from_utf8(output.stdout).ok()?;
    text.split_whitespace()
        .find(|token| token.parse::<f64>().is_ok())
        .and_then(|token| token.parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workspace_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .canonicalize()
            .unwrap_or_else(|_| PathBuf::from("."))
    }

    fn corpus() -> Corpus {
        let root = workspace_root();
        build_corpus(
            &root.join("benchmark/carriers"),
            &root.join("benchmark/cases"),
            DEFAULT_CORPUS_TEXTS,
            "cargo test".to_string(),
        )
        .expect("the committed carriers and cases must build a corpus")
    }

    #[test]
    fn p95_is_the_95th_percentile_and_not_the_mean() {
        // Ninety 1ms samples and ten 100ms ones. The mean is 10.9ms and the
        // p95 is 100ms. A harness that reported 10.9 here would make the
        // gate unfailable.
        let mut samples: Vec<f64> = vec![1.0; 90];
        samples.extend(vec![100.0; 10]);
        let run = LatencyRun::from_samples(samples, TARGET_CHARS);
        assert!((run.p95_ms - 100.0).abs() < 1e-9, "p95 was {}", run.p95_ms);
        assert!((run.p50_ms - 1.0).abs() < 1e-9, "p50 was {}", run.p50_ms);
        assert!(
            (run.end_to_end.mean_ms - 10.9).abs() < 1e-9,
            "mean was {}",
            run.end_to_end.mean_ms
        );
    }

    #[test]
    fn ninety_five_fast_and_five_slow_has_a_p95_of_one_not_a_hundred() {
        // The task-9 brief sketched this test with 95 x 1ms and 5 x 100ms
        // and asserted p95 == 100. It is off by one bucket: with exactly 5%
        // of samples slow, the 95th-fastest sample is still a fast one under
        // every non-interpolating definition, and 5.95 -- the number the
        // brief called "the mean" -- is what numpy's *interpolating* default
        // returns for the p95 of this set. Pinned here so the discrepancy is
        // recorded rather than quietly resolved in either direction.
        let mut samples: Vec<f64> = vec![1.0; 95];
        samples.extend(vec![100.0; 5]);
        let run = LatencyRun::from_samples(samples, TARGET_CHARS);
        assert!((run.p95_ms - 1.0).abs() < 1e-9, "p95 was {}", run.p95_ms);
        assert!((run.p99_ms - 100.0).abs() < 1e-9, "p99 was {}", run.p99_ms);
        assert!((run.end_to_end.mean_ms - 5.95).abs() < 1e-9);
    }

    #[test]
    fn a_run_without_its_hardware_stamp_cannot_be_serialised() {
        // "A number without its command does not go in a report." The type
        // enforces it rather than the reviewer remembering.
        assert!(serde_json::to_string(&LatencyRun::unstamped()).is_err());
    }

    #[test]
    fn a_stamped_run_round_trips() {
        let mut run = LatencyRun::from_samples(vec![10.0, 20.0, 30.0], TARGET_CHARS);
        run.provenance.hardware = Hardware::new("a machine");
        let json = serde_json::to_string(&run).expect("a stamped run serialises");
        let back: LatencyRun = serde_json::from_str(&json).expect("and reads back");
        assert_eq!(back, run);
    }

    #[test]
    fn a_whitespace_only_stamp_is_not_a_stamp() {
        let mut run = LatencyRun::from_samples(vec![10.0], TARGET_CHARS);
        run.provenance.hardware = Hardware::new("   ");
        assert!(serde_json::to_string(&run).is_err());
    }

    #[test]
    fn the_corpus_is_actually_two_thousand_code_points_not_two_thousand_bytes() {
        // German, French, Spanish and Italian carriers are multibyte. A
        // corpus built by byte length would be materially shorter in tokens
        // than the gate asks for, and every measured p95 would be
        // optimistic.
        let corpus = corpus();
        assert_eq!(corpus.texts.len(), DEFAULT_CORPUS_TEXTS);
        let mut multibyte_seen = false;
        for text in &corpus.texts {
            let chars = text.text.chars().count();
            assert_eq!(chars, text.chars, "declared length disagrees with the text");
            assert!(
                (TARGET_CHARS - CHARS_TOLERANCE..=TARGET_CHARS + CHARS_TOLERANCE).contains(&chars),
                "{} is {chars} code points",
                text.id
            );
            if text.text.len() > chars {
                multibyte_seen = true;
            }
        }
        assert!(
            multibyte_seen,
            "no corpus text was multibyte, so this test proved nothing about bytes vs code points"
        );
    }

    #[test]
    fn every_corpus_text_carries_at_least_one_identifier() {
        // A latency corpus of pure prose would measure a service that finds
        // nothing: no resolver work, no calibration lookup, no merge.
        let corpus = corpus();
        assert!(corpus.identifier_density_per_1000_chars > 0.0);
        for text in &corpus.texts {
            assert!(text.identifiers >= 1, "{} carries no identifier", text.id);
        }
    }

    #[test]
    fn the_corpus_covers_every_gated_language_and_is_deterministic() {
        let first = corpus();
        let second = corpus();
        assert_eq!(
            first.texts, second.texts,
            "two builds of the corpus disagree, so a p95 taken a week apart is taken over \
             different work"
        );
        for lang in CORPUS_LANGUAGES {
            let count = first.texts.iter().filter(|text| text.lang == lang).count();
            assert!(count >= 30, "{lang} contributed only {count} texts");
        }
    }

    #[test]
    fn percentile_of_an_empty_sample_is_none_not_zero() {
        assert!(percentile(&[], 0.95).is_none());
        let empty = Distribution::from_samples(Vec::new());
        assert_eq!(empty.n, 0);
        assert!(empty.p95_ms.is_nan());
    }
}
