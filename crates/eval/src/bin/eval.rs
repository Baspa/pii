//! The gate runner: hits a running the detector service's `/analyze` with every
//! case in the benchmark fixture, scores the result with
//! `pii_detection_eval::score` -- the exact scorer that scores the engine's own
//! `fixture-spans` binary and the Azure baseline -- evaluates it against
//! `pii_detection_eval::gates`, writes a report, and exits non-zero when a gate
//! fails. This is what turns the fixture into something that can fail a
//! build.
//!
//! WHAT THIS DOES NOT DO. It does not score (see `pii_detection_eval::score`'s
//! module docs) and it does not decide thresholds (see
//! `pii_detection_eval::gates`'s). It calls a service, hands the response to the
//! scorer, hands the scorer's output to the gate module, and turns the
//! result into a file and an exit code. Three modules, three jobs, on
//! purpose.
//!
//! THE THREE-VALUED EXIT CODE. A gate can pass, fail, or be unmeasured for
//! this run -- `PERSON` and `STREET_ADDRESS` against a deployment that does
//! not advertise them, and the p95 latency gate whenever no `--latency` run
//! was supplied (see `pii_detection_eval::gates`'s ruling 1). Collapsing
//! "regressed" and "nobody measured it" into the same exit code would make
//! a CI job either permanently red for a reason nobody can fix from a pull
//! request, or silently green over the exact gap this benchmark exists to
//! keep visible. So this binary distinguishes them at the process boundary:
//!
//! - **0** -- every gate measured, none failed. A clean pass.
//! - **1** -- at least one gate that *did* run measured a regression. This
//!   is the code a CI job should treat as blocking.
//! - **2** -- no gate failed, but at least one is `NotMeasured`. Every run
//!   against a rules-only service is in this state (`PERSON` and
//!   `STREET_ADDRESS` are structurally unproducible there), and so is every
//!   run given no `--latency`. A CI job can choose not to block a pull
//!   request on this code alone, but it must never report it as a plain
//!   success -- see `ci.yml`'s own comments on exactly how it acts on this.
//!
//! WHERE THE LATENCY GATE'S NUMBER COMES FROM, AND WHERE IT DOES NOT. This
//! binary times its own fixture requests, and reports them as
//! `fixture_request_latency`. **That is not the gate**, and it never was:
//! the fixture's cases are mostly far shorter than the 2,000 characters
//! the design specifies, so a p95 over them is a true number about a
//! different question. The gate reads `--latency <path>` -- one run of the
//! `latency` binary, over a dedicated 2,000-code-point corpus, end to end
//! over HTTP, stamped with its machine -- and nothing else. Two numbers with
//! two meanings in one report need two names.
//!
//! WHY A DECLARED TYPE CAN STILL READ NOT MEASURED. `Case::must_mask` can
//! declare a type -- `PERSON`, say -- that this deployment's `/capabilities`
//! does not list, because the fixture is shared across every detector this
//! benchmark ever measures (rules alone, rules-plus-model, Azure), not
//! rebuilt per run. Scoring blind against that would read as "asked, and
//! missed every time" (`Metrics::recall() == Some(0.0)`), which is a
//! confident false failure, not the honest "never asked" ruling 1 requires.
//! This binary asks the service what it actually supports
//! (`GET /capabilities`), requests only those types from `/analyze`, and
//! passes that same set to `pii_detection_eval::gates::drop_unrequested` before
//! evaluating -- see that function's own docs for the full account of the
//! defect it exists to prevent.
//!
//! NOTHING PRINTED OR WRITTEN CARRIES CASE TEXT OR A DETECTED VALUE -- only
//! counts, ids, types, and rates, the same discipline `baseline.rs` and
//! `fixture-spans` (`main.rs`) hold. The report file *is* new content this
//! binary produces (unlike the Azure baseline, it holds no fixture text of
//! its own), and it is not fixture content, so it does not inherit the
//! "safe to commit" reasoning `baseline.rs`'s docs give for its own output;
//! this binary's report is meant to be a CI artifact, not a committed file.
//!
//! WHAT THIS BENCHMARK DOES NOT ESTABLISH. A clean run of this binary is
//! evidence about 1,150 fixture cases built from real carrier prose with
//! synthetic, injected identifiers (see `benchmark/carriers/README.md` and
//! `benchmark/LICENSE-CARRIERS.md`) -- never real customer data, and never a
//! substitute for adjudicated disagreement on real traffic, which is future
//! work (a shadow run on real traffic, not this work). Quote a number from this
//! binary's report with that scope attached.
//!
//! Usage:
//!     cargo run -p pii-detection-eval --bin eval -- \
//!         --service http://localhost:8202 \
//!         [--cases benchmark/cases] \
//!         [--baseline benchmark/baseline/azure-2026-08-22.json] \
//!         [--latency training/reports/latency-<hardware>-<date>.json] \
//!         [--out benchmark/reports/gate-report.json] \
//!         [--timeout-secs 10]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use pii_detection_api::wire::Entity;
use pii_detection_engine::entity::{Country, EntityType};
use pii_detection_eval::gates::{self, Baseline, GateResult, Verdict};
use pii_detection_eval::latency::LatencyRun;
use pii_detection_eval::score::{Metrics, Summary, scored_case};
use pii_detection_eval::{Case, check_coherence, load};
use serde::{Deserialize, Serialize};

/// A benchmark case's declared identifiers are synthetic and its carrier
/// prose is real, licensed or public text -- never real customer
/// traffic. Carried into the report so a reader of the JSON alone, without
/// this file's own docs, still sees the scope of what a passing gate means.
/// See `benchmark/carriers/README.md` and `benchmark/LICENSE-CARRIERS.md`
/// for the full account.
const PROVENANCE: &str = "This benchmark's carrier prose is real (licensed or public-domain \
     text -- see benchmark/carriers/README.md and benchmark/LICENSE-CARRIERS.md); every \
     identifier injected into it (national IDs, IBANs, credit cards, phone numbers, VAT/tax \
     IDs, passport numbers, emails, IPs) is synthetically generated, never drawn from a real \
     person or from real customer traffic. A passing gate is evidence about this fixture, \
     not a claim about production traffic -- adjudicated production disagreement is future work \
     (a shadow run on real traffic), not this benchmark.";

const DEFAULT_TIMEOUT_SECS: u64 = 10;
const DEFAULT_CASES_DIR: &str = "benchmark/cases";
const DEFAULT_OUT_PATH: &str = "benchmark/reports/gate-report.json";

// ---------------------------------------------------------------------------
// Arguments
// ---------------------------------------------------------------------------

struct Arguments {
    service: String,
    cases: PathBuf,
    baseline: Option<PathBuf>,
    /// The JSON one run of the `latency` binary wrote. **The only input the
    /// latency gate reads.** Without it the gate is `NotMeasured`; this
    /// binary never derives the gate from its own fixture request times --
    /// see [`Report::fixture_request_latency`].
    latency: Option<PathBuf>,
    out: PathBuf,
    /// Where to write the value-free per-case detail, or `None` to write
    /// none. See [`PerCaseDto`] for why this is a separate file and a
    /// separate flag rather than a field of [`Report`].
    per_case: Option<PathBuf>,
    timeout_secs: u64,
}

fn parse_arguments(raw: &[String]) -> Result<Arguments, Failure> {
    let usage = "usage: eval --service <url> [--cases <dir>] [--baseline <path>] \
                 [--latency <path>] [--out <path>] [--per-case <path>] \
                 [--timeout-secs <n>]";

    let mut service = None;
    let mut cases = PathBuf::from(DEFAULT_CASES_DIR);
    let mut baseline = None;
    let mut latency = None;
    let mut out = PathBuf::from(DEFAULT_OUT_PATH);
    let mut per_case = None;
    let mut timeout_secs = DEFAULT_TIMEOUT_SECS;

    let mut rest = raw.iter();
    while let Some(argument) = rest.next() {
        match argument.as_str() {
            "--service" => service = Some(rest.next().ok_or(Failure::Usage(usage))?.clone()),
            "--cases" => cases = PathBuf::from(rest.next().ok_or(Failure::Usage(usage))?),
            "--baseline" => {
                baseline = Some(PathBuf::from(rest.next().ok_or(Failure::Usage(usage))?))
            }
            "--latency" => latency = Some(PathBuf::from(rest.next().ok_or(Failure::Usage(usage))?)),
            "--out" => out = PathBuf::from(rest.next().ok_or(Failure::Usage(usage))?),
            "--per-case" => {
                per_case = Some(PathBuf::from(rest.next().ok_or(Failure::Usage(usage))?))
            }
            "--timeout-secs" => {
                timeout_secs = rest
                    .next()
                    .ok_or(Failure::Usage(usage))?
                    .parse()
                    .map_err(|_| Failure::Usage(usage))?
            }
            _ => return Err(Failure::Usage(usage)),
        }
    }

    Ok(Arguments {
        service: service
            .ok_or(Failure::Usage(usage))?
            .trim_end_matches('/')
            .to_string(),
        cases,
        baseline,
        latency,
        out,
        per_case,
        timeout_secs,
    })
}

// ---------------------------------------------------------------------------
// Wire shapes for talking to a running the detector service. Deliberately
// separate from `pii_detection_api::wire`'s types: `AnalyzeRequest` there is
// `Deserialize`-only (it is the server's own inbound type) and `Entity` is
// `Serialize`-only (the server's outbound type), so a client needs its own
// small mirror of each rather than fighting those derives.
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct AnalyzeRequestOut {
    texts: Vec<String>,
    entities: Vec<String>,
    language: String,
}

#[derive(Deserialize)]
struct WireEntity {
    entity_type: EntityType,
    #[serde(default)]
    country: Option<Country>,
    start: u32,
    end: u32,
    score: f32,
}

impl From<WireEntity> for Entity {
    fn from(wire: WireEntity) -> Self {
        Entity {
            entity_type: wire.entity_type,
            country: wire.country,
            start: wire.start,
            end: wire.end,
            score: wire.score,
        }
    }
}

/// `POST /analyze`'s envelope. The spans live under `results`; the `language`
/// block beside them says which regime produced them (see
/// `pii_detection_api::wire::AnalyzeResponse`) and is deliberately not read here --
/// every request this binary sends is `language: "auto"`, so the regime is
/// the same `undetermined` for all of them and carries no per-case
/// information. Reading `results` by name rather than indexing a bare array
/// is what keeps this parse honest if the envelope ever gains a field.
#[derive(Deserialize)]
struct AnalyzeResponseIn {
    results: Vec<Vec<WireEntity>>,
}

/// Only the field this binary needs from `GET /capabilities` -- the types
/// this deployment can actually detect. `languages` and `versions` are
/// present on the real response and silently ignored here (serde's default
/// behaviour), not read for anything.
#[derive(Deserialize)]
struct CapabilitiesResponse {
    entity_types: Vec<EntityType>,
}

// ---------------------------------------------------------------------------
// Reading a committed Azure baseline file (benchmark/baseline/*.json)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct BaselineTypeMetrics {
    recall: Option<f64>,
}

#[derive(Deserialize)]
struct BaselineSummary {
    per_type: BTreeMap<String, BaselineTypeMetrics>,
}

#[derive(Deserialize)]
struct BaselineFile {
    collected_at: String,
    azure_model_version: Option<String>,
    /// How many fixture cases the baseline was collected over. The hard
    /// guard below exists because this number can silently drift away from
    /// the fixture being scored today: the baseline is a point-in-time
    /// snapshot (see `benchmark/baseline/README.md`) but `benchmark/cases`
    /// keeps growing, and growth is never uniform across languages. Compare
    /// the 349-case baseline this repository once shipped against the
    /// 649-case fixture it was quoted beside as of 2026-08-22: every one of
    /// the 300 added cases skewed toward Spanish, Italian and French, the
    /// languages the detector's rules cover worst. The same thing would have
    /// happened again on 2026-08-26, when the fixture grew to 1,150.
    /// Scoring the detector over one population against
    /// hand the `PERSON` and `STREET_ADDRESS` non-inferiority gates --
    /// `evaluate_against_baseline`'s whole reason to exist -- a verdict
    /// built from two different samples, with no error and no sign in the
    /// report that it happened.
    n_cases: usize,
    summary: BaselineSummary,
}

struct LoadedBaseline {
    baseline: Baseline,
    path: String,
    collected_at: String,
    azure_model_version: Option<String>,
}

/// Loads a committed Azure baseline and refuses to hand back anything
/// usable if it was not collected over exactly the fixture being scored
/// today (`fixture_n_cases`, i.e. `cases.len()` in `run`). See
/// `BaselineFile::n_cases`'s docs for why this has to be a hard error and
/// not a warning: a size mismatch here means the non-inferiority gates
/// would otherwise compare two different populations and report a verdict
/// as if they hadn't.
fn load_baseline(path: &Path, fixture_n_cases: usize) -> Result<LoadedBaseline, Failure> {
    let contents = std::fs::read_to_string(path)
        .map_err(|_| Failure::UnreadableBaseline(path.to_path_buf()))?;
    let parsed: BaselineFile =
        serde_json::from_str(&contents).map_err(|error| Failure::MalformedBaseline {
            line: error.line(),
            column: error.column(),
        })?;

    if parsed.n_cases != fixture_n_cases {
        return Err(Failure::BaselineCaseCountMismatch {
            baseline_n_cases: parsed.n_cases,
            fixture_n_cases,
        });
    }

    let recall_of = |type_name: &str| {
        parsed
            .summary
            .per_type
            .get(type_name)
            .and_then(|m| m.recall)
    };

    Ok(LoadedBaseline {
        baseline: Baseline {
            person_recall: recall_of(gates::wire_name(EntityType::Person)),
            street_address_recall: recall_of(gates::wire_name(EntityType::StreetAddress)),
        },
        path: path.display().to_string(),
        collected_at: parsed.collected_at,
        azure_model_version: parsed.azure_model_version,
    })
}

// ---------------------------------------------------------------------------
// Latency: informational only -- see gates::evaluate_latency for why the
// verdict never derives from this number.
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Serialize)]
struct LatencyStats {
    sample_n: usize,
    min_ms: Option<f64>,
    p95_ms: Option<f64>,
    max_ms: Option<f64>,
}

/// The percentile comes from `pii_detection_eval::latency::percentile`, not from a
/// second definition in this file. Two percentile functions in one crate are
/// two answers to "what is the p95", and this benchmark has already shipped
/// one badly wrong figure from exactly that shape of duplication.
fn latency_stats(mut latencies_ms: Vec<f64>) -> LatencyStats {
    latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    LatencyStats {
        sample_n: latencies_ms.len(),
        min_ms: latencies_ms.first().copied(),
        p95_ms: pii_detection_eval::latency::percentile(&latencies_ms, 0.95),
        max_ms: latencies_ms.last().copied(),
    }
}

// ---------------------------------------------------------------------------
// Report shape -- what this binary writes and prints
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Serialize)]
struct MetricsDto {
    mask_total: usize,
    mask_found: usize,
    clear_total: usize,
    clear_violated: usize,
    spurious_spans: usize,
    recall: Option<f64>,
    precision: Option<f64>,
}

impl From<&Metrics> for MetricsDto {
    fn from(metrics: &Metrics) -> Self {
        Self {
            mask_total: metrics.mask_total,
            mask_found: metrics.mask_found,
            clear_total: metrics.clear_total,
            clear_violated: metrics.clear_violated,
            spurious_spans: metrics.spurious_spans,
            recall: metrics.recall(),
            precision: metrics.precision(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
struct MacroAverageDto {
    recall: Option<f64>,
    precision: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct SummaryDto {
    overall: MetricsDto,
    per_language: BTreeMap<String, MetricsDto>,
    per_type: BTreeMap<String, MetricsDto>,
    /// See `pii_detection_eval::score`'s module docs, and ruling 3: printed beside
    /// every per-type figure because an untyped value drops out of its own
    /// per-type denominator, which *raises* the recall it should lower.
    untyped_mask_values: usize,
    macro_by_language: MacroAverageDto,
}

impl From<&Summary> for SummaryDto {
    fn from(summary: &Summary) -> Self {
        Self {
            overall: MetricsDto::from(&summary.overall),
            per_language: summary
                .per_language
                .iter()
                .map(|(lang, metrics)| (lang.clone(), MetricsDto::from(metrics)))
                .collect(),
            per_type: summary
                .per_type
                .iter()
                .map(|(type_name, metrics)| (type_name.clone(), MetricsDto::from(metrics)))
                .collect(),
            untyped_mask_values: summary.untyped_mask_values,
            macro_by_language: MacroAverageDto {
                recall: summary.macro_by_language.recall,
                precision: summary.macro_by_language.precision,
            },
        }
    }
}

#[derive(Debug, Serialize)]
struct BaselineDto {
    provided: bool,
    path: Option<String>,
    collected_at: Option<String>,
    azure_model_version: Option<String>,
    person_recall: Option<f64>,
    street_address_recall: Option<f64>,
}

/// The three-valued outcome documented at the top of this file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum OverallStatus {
    Pass,
    Fail,
    Incomplete,
}

impl OverallStatus {
    fn from_gates(gates: &[GateResult]) -> Self {
        if gates.iter().any(|gate| gate.verdict == Verdict::Fail) {
            Self::Fail
        } else if gates
            .iter()
            .any(|gate| gate.verdict == Verdict::NotMeasured)
        {
            Self::Incomplete
        } else {
            Self::Pass
        }
    }

    fn exit_code(self) -> u8 {
        match self {
            Self::Pass => 0,
            Self::Fail => 1,
            Self::Incomplete => 2,
        }
    }

    fn explanation(self) -> &'static str {
        match self {
            Self::Pass => "every gate ran and every gate passed",
            Self::Fail => {
                "at least one gate that ran measured a regression -- this is a blocking result"
            }
            Self::Incomplete => {
                "no gate failed, but at least one is NOT MEASURED (PERSON and STREET_ADDRESS \
                 against a service that does not advertise them; p95 latency whenever no \
                 --latency run was supplied, or the run supplied was not the measurement the \
                 gate means -- see pii_detection_eval::gates's module docs). This is not a clean \
                 pass; see this binary's own docs on why exit code 2 exists"
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Per-case detail, by position and never by value.
//
// WHY THIS EXISTS. `reports/distillation-run.md`'s first concern was that a
// run could report `STREET_ADDRESS` 132 -> 131 and be unable to say *which*
// value moved, because this binary emitted only roll-ups. A one-value
// regression that cannot be named cannot be diagnosed, and "one address, I
// don't know which" is the kind of finding that gets waved through.
//
// WHY IT IS A SEPARATE FILE AND A SEPARATE FLAG. `Report` is the release
// gate's own artifact and is read by things that expect its shape; growing
// it by one nested object per case would change what every existing consumer parses
// for a diagnostic almost every run does not want.
//
// WHY BY INDEX. `score::MaskVerdict` carries the literal `must_mask` string,
// and `score.rs`'s own docs are explicit that a caller emitting an `Outcome`
// must keep the discipline the rest of the service holds for anything
// reachable from a log line. The index into the case's declared
// `must_mask`/`must_clear` list identifies the value exactly, for anyone
// holding the fixture, while this file stays free of case text -- so it can
// be attached to a report, pasted into an issue, or committed. Nothing here
// has a field capable of holding a value in the first place, which is the
// same structural discipline `Failure` above is built on.
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct MaskVerdictDto {
    /// Position in this case's declared `must_mask` list.
    index: usize,
    /// The type the case declared, or `null` where it declared none --
    /// ruling 3's untyped values, which drop out of their own per-type
    /// denominator.
    declared: Option<EntityType>,
    /// False is a leak: at least one occurrence had an uncovered
    /// non-whitespace character.
    found: bool,
    /// Types of every detected span that touched the value, even partially.
    /// Empty when nothing touched it -- which distinguishes "found nothing
    /// here" from "found something and typed it wrong".
    attributed: Vec<EntityType>,
}

#[derive(Debug, Serialize)]
struct ClearVerdictDto {
    /// Position in this case's declared `must_clear` list.
    index: usize,
    /// True is an over-masking violation.
    violated: bool,
    types: Vec<EntityType>,
}

#[derive(Debug, Serialize)]
struct PerCaseDto {
    case_id: String,
    lang: String,
    must_mask: Vec<MaskVerdictDto>,
    must_clear: Vec<ClearVerdictDto>,
    /// Types of detected spans covering no `must_mask` occurrence at all --
    /// the false positives nothing named ahead of time.
    spurious_span_types: Vec<EntityType>,
}

impl PerCaseDto {
    fn new(case: &Case, outcome: &pii_detection_eval::score::CaseOutcome) -> Self {
        Self {
            case_id: case.id.clone(),
            lang: outcome.lang.clone(),
            must_mask: outcome
                .mask
                .iter()
                .enumerate()
                .map(|(index, verdict)| MaskVerdictDto {
                    index,
                    declared: verdict.declared,
                    found: verdict.found,
                    attributed: verdict.types.clone(),
                })
                .collect(),
            must_clear: outcome
                .clear
                .iter()
                .enumerate()
                .map(|(index, verdict)| ClearVerdictDto {
                    index,
                    violated: verdict.violated,
                    types: verdict.types.clone(),
                })
                .collect(),
            spurious_span_types: outcome.spurious_spans.clone(),
        }
    }
}

#[derive(Debug, Serialize)]
struct Report {
    command: String,
    service: String,
    cases_dir: String,
    n_cases: usize,
    /// What `GET /capabilities` reported this deployment can detect, and
    /// therefore what `/analyze` was actually asked for. Today, against a
    /// rules-only service, this never includes `PERSON` or
    /// `STREET_ADDRESS` -- see this file's module docs.
    requested_types: Vec<EntityType>,
    run_started_unix_secs: u64,
    baseline: BaselineDto,
    /// **Informational, and not the latency gate.** These are the request
    /// times of this run's own fixture calls, over a population whose cases
    /// are mostly far shorter than the 2,000 characters the design
    /// specifies. It is a true number about a different question, which is
    /// why it has a different name from the gate's input: two numbers with
    /// two meanings in one report need two names.
    fixture_request_latency: LatencyStats,
    /// The dedicated 2,000-code-point run the gate actually reads, verbatim
    /// as the `latency` binary wrote it, or `null` when `--latency` was not
    /// given. Carried in full so a reader of this report never has to take
    /// the gate's one-line `detail` on trust.
    latency_run: Option<LatencyRun>,
    /// The summary every gate in `gates` was computed from: restricted to
    /// the `must_mask` values whose declared type is in `requested_types`.
    summary: SummaryDto,
    /// The same run over every `must_mask` value the fixture declares,
    /// including the types this deployment was never asked for. Reported
    /// beside `summary` so the difference between the two is visible rather
    /// than implied -- and so nobody can quote the restricted recall without
    /// the unrestricted one being one field away.
    summary_as_declared: SummaryDto,
    gates: Vec<GateResult>,
    status: OverallStatus,
    exit_code: u8,
    status_explanation: &'static str,
    sample_size_caveat: &'static str,
    provenance: &'static str,
}

// ---------------------------------------------------------------------------
// Failures. No variant carries case text, a detected value, or a response
// body -- only paths, ids, counts and status codes, matching the discipline
// `baseline.rs::Failure` and `main.rs::Failure` document. This is
// structural, not merely tested: nothing below has a field capable of
// holding `Case::text` or a span's covered substring in the first place.
// ---------------------------------------------------------------------------

enum Failure {
    Usage(&'static str),
    UnreadableCases(PathBuf),
    Incoherent(usize),
    ClientBuildFailed,
    CapabilitiesRequest,
    CapabilitiesStatus(u16),
    CapabilitiesMalformed,
    AnalyzeRequest {
        case_id: String,
    },
    AnalyzeStatus {
        case_id: String,
        status: u16,
    },
    AnalyzeMalformed {
        case_id: String,
    },
    AnalyzeEmpty {
        case_id: String,
    },
    UnreadableLatency(PathBuf),
    MalformedLatency {
        path: PathBuf,
        line: usize,
        column: usize,
    },
    UnreadableBaseline(PathBuf),
    MalformedBaseline {
        line: usize,
        column: usize,
    },
    BaselineCaseCountMismatch {
        baseline_n_cases: usize,
        fixture_n_cases: usize,
    },
    Unwritable(PathBuf),
    Unserializable,
}

impl Failure {
    fn report(&self) -> String {
        match self {
            Self::Usage(message) => (*message).to_string(),
            Self::UnreadableCases(path) => format!("cannot read case directory {}", path.display()),
            Self::UnreadableLatency(path) => format!(
                "cannot read the --latency run at {}. It is written by `pii-detection-eval --bin \
                 latency`; see that binary's docs. A missing file is a hard failure rather \
                 than a silent NOT MEASURED, so a typo in a path cannot quietly switch the \
                 gate off.",
                path.display()
            ),
            Self::MalformedLatency { path, line, column } => format!(
                "the --latency run at {} is not the expected shape (line {line}, column \
                 {column}). A run with no hardware stamp is one way to land here: the type \
                 refuses to serialise or deserialise without one.",
                path.display()
            ),
            Self::Incoherent(count) => format!(
                "the fixture has {count} incoherent case(s); run check_coherence and fix them \
                 first (see pii_detection_eval::check_coherence)"
            ),
            Self::ClientBuildFailed => "could not build an HTTP client".to_string(),
            Self::CapabilitiesRequest => "GET /capabilities: the service was unreachable or \
                 the request timed out"
                .to_string(),
            Self::CapabilitiesStatus(status) => {
                format!("GET /capabilities: the service returned status {status}")
            }
            Self::CapabilitiesMalformed => {
                "GET /capabilities: the response was not the expected shape".to_string()
            }
            Self::AnalyzeRequest { case_id } => format!(
                "case {case_id}: POST /analyze -- the service was unreachable or the request \
                 timed out"
            ),
            Self::AnalyzeStatus { case_id, status } => {
                format!("case {case_id}: POST /analyze returned status {status}")
            }
            Self::AnalyzeMalformed { case_id } => {
                format!("case {case_id}: POST /analyze's response was not the expected shape")
            }
            Self::AnalyzeEmpty { case_id } => {
                format!("case {case_id}: POST /analyze returned no result for the one text sent")
            }
            Self::UnreadableBaseline(path) => format!("cannot read {}", path.display()),
            Self::MalformedBaseline { line, column } => {
                format!(
                    "the --baseline file is not the expected shape, at line {line} column {column}"
                )
            }
            Self::BaselineCaseCountMismatch {
                baseline_n_cases,
                fixture_n_cases,
            } => format!(
                "the --baseline file was collected over {baseline_n_cases} case(s) but the \
                 fixture being scored has {fixture_n_cases}; refusing to compare recall across \
                 two different populations -- collect a fresh baseline over the current \
                 fixture (see benchmark/baseline/README.md) or point --baseline at one that \
                 matches"
            ),
            Self::Unwritable(path) => format!("cannot write {}", path.display()),
            Self::Unserializable => "cannot serialise the report".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Talking to the service
// ---------------------------------------------------------------------------

fn fetch_capabilities(
    client: &reqwest::blocking::Client,
    service: &str,
) -> Result<BTreeSet<EntityType>, Failure> {
    let response = client
        .get(format!("{service}/capabilities"))
        .send()
        .map_err(|_| Failure::CapabilitiesRequest)?;
    if !response.status().is_success() {
        return Err(Failure::CapabilitiesStatus(response.status().as_u16()));
    }
    let parsed: CapabilitiesResponse = response
        .json()
        .map_err(|_| Failure::CapabilitiesMalformed)?;
    Ok(parsed.entity_types.into_iter().collect())
}

/// Sends one case's text to `/analyze`, timed, and returns its scored
/// outcome plus the elapsed wall time in milliseconds.
fn analyze_case(
    client: &reqwest::blocking::Client,
    service: &str,
    entities: &[String],
    case: &Case,
) -> Result<(pii_detection_eval::score::CaseOutcome, f64), Failure> {
    let body = AnalyzeRequestOut {
        texts: vec![case.text.clone()],
        entities: entities.to_vec(),
        language: "auto".to_string(),
    };

    let started = Instant::now();
    let response = client
        .post(format!("{service}/analyze"))
        .json(&body)
        .send()
        .map_err(|_| Failure::AnalyzeRequest {
            case_id: case.id.clone(),
        })?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;

    let status = response.status();
    if !status.is_success() {
        return Err(Failure::AnalyzeStatus {
            case_id: case.id.clone(),
            status: status.as_u16(),
        });
    }

    let parsed: AnalyzeResponseIn = response.json().map_err(|_| Failure::AnalyzeMalformed {
        case_id: case.id.clone(),
    })?;
    let Some(first) = parsed.results.into_iter().next() else {
        return Err(Failure::AnalyzeEmpty {
            case_id: case.id.clone(),
        });
    };

    let detected: Vec<Entity> = first.into_iter().map(Entity::from).collect();
    Ok((scored_case(case, &detected), elapsed_ms))
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn run(arguments: &Arguments, command: String) -> Result<(PathBuf, Report), Failure> {
    // Read before a single request is made. A `--latency` file that cannot
    // be read is a hard failure, never a silent `None`: silently degrading
    // to NOT MEASURED would turn a typo in a path into a gate that quietly
    // stopped running.
    let latency_run = match arguments.latency.as_deref() {
        None => None,
        Some(path) => {
            let raw = std::fs::read_to_string(path)
                .map_err(|_| Failure::UnreadableLatency(path.to_path_buf()))?;
            let run: LatencyRun =
                serde_json::from_str(&raw).map_err(|error| Failure::MalformedLatency {
                    path: path.to_path_buf(),
                    line: error.line(),
                    column: error.column(),
                })?;
            println!(
                "latency run: p95 {:.1}ms over n={} at {} code points, measured on {}",
                run.p95_ms, run.n, run.chars, run.provenance.hardware
            );
            Some(run)
        }
    };

    let cases =
        load(&arguments.cases).map_err(|_| Failure::UnreadableCases(arguments.cases.clone()))?;

    let incoherent = check_coherence(&cases);
    if !incoherent.is_empty() {
        return Err(Failure::Incoherent(incoherent.len()));
    }
    println!(
        "fixture: {} coherent cases from {}",
        cases.len(),
        arguments.cases.display()
    );

    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(arguments.timeout_secs))
        .build()
        .map_err(|_| Failure::ClientBuildFailed)?;

    let requested_types = fetch_capabilities(&client, &arguments.service)?;
    let requested_wire: Vec<String> = requested_types
        .iter()
        .map(|entity_type| gates::wire_name(*entity_type).to_string())
        .collect();
    println!(
        "service {} detects {} type(s): {}",
        arguments.service,
        requested_types.len(),
        requested_wire.join(", ")
    );

    let mut case_outcomes = Vec::with_capacity(cases.len());
    let mut latencies_ms = Vec::with_capacity(cases.len());
    for case in &cases {
        let (outcome, elapsed_ms) =
            analyze_case(&client, &arguments.service, &requested_wire, case)?;
        case_outcomes.push(outcome);
        latencies_ms.push(elapsed_ms);
    }

    // Written here rather than beside the report, so that a run asked for
    // per-case detail fails on an unwritable path immediately after the
    // cases are scored rather than after everything else has been computed.
    if let Some(path) = &arguments.per_case {
        let detail: Vec<PerCaseDto> = cases
            .iter()
            .zip(case_outcomes.iter())
            .map(|(case, outcome)| PerCaseDto::new(case, outcome))
            .collect();
        let rendered =
            serde_json::to_string_pretty(&detail).map_err(|_| Failure::Unserializable)?;
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
            && std::fs::create_dir_all(parent).is_err()
        {
            return Err(Failure::Unwritable(parent.to_path_buf()));
        }
        std::fs::write(path, format!("{rendered}\n"))
            .map_err(|_| Failure::Unwritable(path.clone()))?;
        println!("wrote {} (per-case detail, no case text)", path.display());
    }

    // Two summaries, both reported. `summary_as_declared` counts every
    // must_mask value the fixture declares, including the 94 PERSON and
    // STREET_ADDRESS values a rules-only service is never asked about and so
    // can never find. `summary_for_gates` counts only the values whose
    // declared type this deployment was actually asked for -- ruling 1: a
    // type nobody asked about must read NOT MEASURED, not "measured, 0%
    // recall", and that has to hold for the headline and per-language
    // figures as much as for the per-type table. See drop_unrequested's
    // docs. Every gate reads the second; the first is carried into the
    // report so a reader can see exactly what was set aside and why, rather
    // than having to take the smaller denominator on trust.
    let summary_as_declared = Summary::from(case_outcomes.clone());
    let summary_for_gates = gates::drop_unrequested(&case_outcomes, &requested_types);

    let structured = gates::evaluate(&summary_for_gates);

    let (baseline, baseline_dto) = match &arguments.baseline {
        Some(path) => {
            let loaded = load_baseline(path, cases.len())?;
            let dto = BaselineDto {
                provided: true,
                path: Some(loaded.path.clone()),
                collected_at: Some(loaded.collected_at.clone()),
                azure_model_version: loaded.azure_model_version.clone(),
                person_recall: loaded.baseline.person_recall,
                street_address_recall: loaded.baseline.street_address_recall,
            };
            (loaded.baseline, dto)
        }
        None => (
            Baseline::default(),
            BaselineDto {
                provided: false,
                path: None,
                collected_at: None,
                azure_model_version: None,
                person_recall: None,
                street_address_recall: None,
            },
        ),
    };
    let baseline_relative = gates::evaluate_against_baseline(&summary_for_gates, baseline);

    let fixture_request_latency = latency_stats(latencies_ms);
    let latency_gate = gates::evaluate_latency(latency_run.as_ref());

    let mut gates_all = structured.gates;
    gates_all.extend(baseline_relative.gates);
    gates_all.push(latency_gate);

    let status = OverallStatus::from_gates(&gates_all);

    let run_started_unix_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);

    let report = Report {
        command,
        service: arguments.service.clone(),
        cases_dir: arguments.cases.display().to_string(),
        n_cases: cases.len(),
        requested_types: requested_types.into_iter().collect(),
        run_started_unix_secs,
        baseline: baseline_dto,
        fixture_request_latency,
        latency_run,
        summary: SummaryDto::from(&summary_for_gates),
        summary_as_declared: SummaryDto::from(&summary_as_declared),
        gates: gates_all,
        status,
        exit_code: status.exit_code(),
        status_explanation: status.explanation(),
        sample_size_caveat: gates::SAMPLE_SIZE_CAVEAT,
        provenance: PROVENANCE,
    };

    Ok((arguments.out.clone(), report))
}

fn print_report(report: &Report) {
    println!(
        "overall: recall {} precision {} (untyped_mask_values={} -- see ruling 3, these cannot \
         appear in any per-type figure below)",
        format_ratio(report.summary.overall.recall),
        format_ratio(report.summary.overall.precision),
        report.summary.untyped_mask_values
    );
    println!(
        "  over the {} of {} declared must_mask values whose type this service was asked for; \
         over all {} it is recall {}, which counts {} value(s) of types nobody asked it to \
         find as misses and is not a measurement of this deployment",
        report.summary.overall.mask_total,
        report.summary_as_declared.overall.mask_total,
        report.summary_as_declared.overall.mask_total,
        format_ratio(report.summary_as_declared.overall.recall),
        report.summary_as_declared.overall.mask_total - report.summary.overall.mask_total
    );

    println!(
        "per-language (recall / precision / n) -- {}",
        report.sample_size_caveat
    );
    for (lang, metrics) in &report.summary.per_language {
        println!(
            "  {lang}: {} / {} (n={})",
            format_ratio(metrics.recall),
            format_ratio(metrics.precision),
            metrics.mask_total
        );
    }

    println!("per-type (recall / precision / n):");
    for (type_name, metrics) in &report.summary.per_type {
        println!(
            "  {type_name}: {} / {} (n={})",
            format_ratio(metrics.recall),
            format_ratio(metrics.precision),
            metrics.mask_total
        );
    }

    println!("gates:");
    for gate in &report.gates {
        println!("  [{:?}] {} -- {}", gate.verdict, gate.name, gate.detail);
    }

    println!(
        "fixture request latency (INFORMATIONAL -- this is NOT the p95 gate; the fixture's \
         cases are mostly far shorter than {} characters): n={} min={} p95={} max={} ms",
        gates::LATENCY_TEST_CHARS,
        report.fixture_request_latency.sample_n,
        format_ms(report.fixture_request_latency.min_ms),
        format_ms(report.fixture_request_latency.p95_ms),
        format_ms(report.fixture_request_latency.max_ms),
    );

    println!(
        "status: {:?} (exit code {}) -- {}",
        report.status, report.exit_code, report.status_explanation
    );
}

fn format_ratio(value: Option<f64>) -> String {
    match value {
        Some(value) => format!("{value:.4}"),
        None => "n/a".to_string(),
    }
}

fn format_ms(value: Option<f64>) -> String {
    match value {
        Some(value) => format!("{value:.1}"),
        None => "n/a".to_string(),
    }
}

fn main() -> ExitCode {
    let command = std::env::args().collect::<Vec<_>>().join(" ");
    let raw: Vec<String> = std::env::args().skip(1).collect();

    let arguments = match parse_arguments(&raw) {
        Ok(arguments) => arguments,
        Err(failure) => {
            eprintln!("{}", failure.report());
            return ExitCode::from(2);
        }
    };

    // Created before a single request is sent, not after: writing a whole
    // fixture's worth of results and then discovering the output directory
    // does not exist is a mistake this benchmark has already made once (see
    // benchmark/baseline/README.md's account of baseline.rs's own history).
    if let Some(parent) = arguments.out.parent()
        && !parent.as_os_str().is_empty()
        && std::fs::create_dir_all(parent).is_err()
    {
        eprintln!("{}", Failure::Unwritable(parent.to_path_buf()).report());
        return ExitCode::from(2);
    }

    match run(&arguments, command) {
        Ok((out_path, report)) => {
            let rendered = match serde_json::to_string_pretty(&report) {
                Ok(rendered) => rendered,
                Err(_) => {
                    eprintln!("{}", Failure::Unserializable.report());
                    return ExitCode::from(2);
                }
            };
            if std::fs::write(&out_path, format!("{rendered}\n")).is_err() {
                eprintln!("{}", Failure::Unwritable(out_path).report());
                return ExitCode::from(2);
            }
            println!("wrote {}", out_path.display());
            print_report(&report);
            ExitCode::from(report.exit_code)
        }
        Err(failure) => {
            eprintln!("{}", failure.report());
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The percentile function itself is tested where it lives, in
    // `pii_detection_eval::latency`. It used to be defined a second time in this
    // file, with a different rounding rule; the tests below now exercise
    // the one definition through `latency_stats`.

    #[test]
    fn latency_stats_reports_min_p95_max_from_an_unsorted_sample() {
        let stats = latency_stats(vec![5.0, 1.0, 3.0, 2.0, 4.0]);
        assert_eq!(stats.sample_n, 5);
        assert_eq!(stats.min_ms, Some(1.0));
        assert_eq!(stats.max_ms, Some(5.0));
        assert!(stats.p95_ms.is_some());
    }

    #[test]
    fn latency_stats_of_an_empty_sample_reports_nothing() {
        let stats = latency_stats(vec![]);
        assert_eq!(stats.sample_n, 0);
        assert_eq!(stats.min_ms, None);
        assert_eq!(stats.p95_ms, None);
        assert_eq!(stats.max_ms, None);
    }

    #[test]
    fn overall_status_is_fail_when_any_gate_failed_even_alongside_not_measured() {
        // Fail must win over Incomplete: a real regression is not softened
        // by the fact that some other, unrelated gate also happens to be
        // unmeasured today.
        let gates = vec![
            GateResult {
                name: "a".to_string(),
                verdict: Verdict::Fail,
                detail: String::new(),
            },
            GateResult {
                name: "b".to_string(),
                verdict: Verdict::NotMeasured,
                detail: String::new(),
            },
        ];
        assert_eq!(OverallStatus::from_gates(&gates), OverallStatus::Fail);
        assert_eq!(OverallStatus::from_gates(&gates).exit_code(), 1);
    }

    #[test]
    fn overall_status_is_incomplete_when_nothing_failed_but_something_is_unmeasured() {
        let gates = vec![
            GateResult {
                name: "a".to_string(),
                verdict: Verdict::Pass,
                detail: String::new(),
            },
            GateResult {
                name: "b".to_string(),
                verdict: Verdict::NotMeasured,
                detail: String::new(),
            },
        ];
        assert_eq!(OverallStatus::from_gates(&gates), OverallStatus::Incomplete);
        assert_eq!(OverallStatus::from_gates(&gates).exit_code(), 2);
    }

    #[test]
    fn overall_status_is_pass_only_when_every_gate_passed() {
        let gates = vec![GateResult {
            name: "a".to_string(),
            verdict: Verdict::Pass,
            detail: String::new(),
        }];
        assert_eq!(OverallStatus::from_gates(&gates), OverallStatus::Pass);
        assert_eq!(OverallStatus::from_gates(&gates).exit_code(), 0);
    }

    #[test]
    fn overall_status_of_no_gates_at_all_is_pass() {
        // An edge case worth pinning down explicitly rather than leaving to
        // whatever `any()` over an empty slice happens to do: zero gates
        // means zero failures and zero unmeasured, which is vacuously a
        // pass. This binary never actually produces zero gates (latency
        // alone always contributes one), but the function itself should not
        // rely on that.
        assert_eq!(OverallStatus::from_gates(&[]), OverallStatus::Pass);
    }

    #[test]
    fn arguments_require_a_service_url() {
        assert!(matches!(parse_arguments(&[]), Err(Failure::Usage(_))));
    }

    #[test]
    fn arguments_trim_a_trailing_slash_from_the_service_url() {
        // Matched rather than `.expect()`ed: `Failure` deliberately has no
        // `Debug`, so that no future code path can print one whole and
        // discover it was carrying something it shouldn't (see main.rs's
        // and baseline.rs's own tests for the same discipline).
        match parse_arguments(&[
            "--service".to_string(),
            "http://localhost:8202/".to_string(),
        ]) {
            Ok(arguments) => {
                assert_eq!(arguments.service, "http://localhost:8202");
                assert_eq!(arguments.cases, PathBuf::from(DEFAULT_CASES_DIR));
                assert_eq!(arguments.out, PathBuf::from(DEFAULT_OUT_PATH));
                assert!(arguments.baseline.is_none());
            }
            Err(failure) => panic!("a service URL is enough: {}", failure.report()),
        }
    }

    #[test]
    fn arguments_accept_a_baseline_path() {
        match parse_arguments(&[
            "--service".to_string(),
            "http://localhost:8202".to_string(),
            "--baseline".to_string(),
            "benchmark/baseline/azure-2026-08-22.json".to_string(),
        ]) {
            Ok(arguments) => assert_eq!(
                arguments.baseline,
                Some(PathBuf::from("benchmark/baseline/azure-2026-08-22.json"))
            ),
            Err(failure) => panic!("expected this to parse: {}", failure.report()),
        }
    }

    fn temp_baseline_file(name: &str, n_cases: usize) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("pii-detection-eval-test-baseline-{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp baseline dir");
        let path = dir.join("baseline.json");
        let body = format!(
            r#"{{"collected_at":"2026-08-22","azure_model_version":"2026-05-01","n_cases":{n_cases},"summary":{{"per_type":{{"PERSON":{{"recall":0.9}}}}}}}}"#
        );
        std::fs::write(&path, body).expect("write temp baseline file");
        path
    }

    #[test]
    fn load_baseline_succeeds_when_n_cases_matches_the_fixture() {
        // The guard's happy path: a baseline collected over exactly the
        // fixture being scored loads normally and its recall figures come
        // through.
        let path = temp_baseline_file("match", 649);
        match load_baseline(&path, 649) {
            Ok(loaded) => assert_eq!(loaded.baseline.person_recall, Some(0.9)),
            Err(failure) => panic!("matching n_cases should load: {}", failure.report()),
        }
    }

    #[test]
    fn load_baseline_refuses_a_mismatched_case_count() {
        // This is the guard the review found missing: a baseline collected
        // over 349 cases must never be silently compared against a 649-case
        // fixture. If this test is deleted or the guard in `load_baseline`
        // is removed, this must fail (see the mutation note in the report).
        let path = temp_baseline_file("mismatch", 349);
        match load_baseline(&path, 649) {
            Ok(_) => panic!(
                "a 349-vs-649 case count mismatch must never produce a usable baseline; the \
                 non-inferiority gates would silently compare two different populations"
            ),
            Err(Failure::BaselineCaseCountMismatch {
                baseline_n_cases,
                fixture_n_cases,
            }) => {
                assert_eq!(baseline_n_cases, 349);
                assert_eq!(fixture_n_cases, 649);
            }
            Err(other) => panic!(
                "expected BaselineCaseCountMismatch, got a different failure: {}",
                other.report()
            ),
        }
    }

    #[test]
    fn every_failure_message_is_non_empty() {
        // Every variant here is typed to carry only ids, paths, counts and
        // status codes -- never `Case::text` or a detected value, which is
        // enforced by the enum's field types, not by scanning strings at
        // runtime (a runtime scan can only prove a message is free of one
        // specific canary; it cannot prove a variant with a `String` field
        // holding arbitrary content was never constructed with one).
        for failure in [
            Failure::Usage("usage: ..."),
            Failure::UnreadableCases(PathBuf::from("/tmp/x")),
            Failure::Incoherent(3),
            Failure::ClientBuildFailed,
            Failure::CapabilitiesRequest,
            Failure::CapabilitiesStatus(500),
            Failure::CapabilitiesMalformed,
            Failure::AnalyzeRequest {
                case_id: "case-1".to_string(),
            },
            Failure::AnalyzeStatus {
                case_id: "case-1".to_string(),
                status: 400,
            },
            Failure::AnalyzeMalformed {
                case_id: "case-1".to_string(),
            },
            Failure::AnalyzeEmpty {
                case_id: "case-1".to_string(),
            },
            Failure::UnreadableBaseline(PathBuf::from("/tmp/b.json")),
            Failure::MalformedBaseline { line: 1, column: 2 },
            Failure::BaselineCaseCountMismatch {
                baseline_n_cases: 349,
                fixture_n_cases: 649,
            },
            Failure::Unwritable(PathBuf::from("/tmp/out.json")),
            Failure::Unserializable,
        ] {
            assert!(!failure.report().is_empty());
        }
    }
}
