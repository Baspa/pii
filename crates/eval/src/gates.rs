//! The gate: turns a [`crate::score::Summary`] into pass/fail/not-measured
//! verdicts against the design's thresholds, so a run of this
//! benchmark can fail a build rather than sit in a report nobody reads.
//!
//! THIS MODULE DOES NOT SCORE. It reads the counts [`crate::score::Summary`]
//! already computed and compares them against fixed thresholds. Building a
//! second, subtly different notion of recall or precision in here would be
//! exactly the mistake the module docs on `score` warn about: two numbers
//! from two scorers are not comparable, and this benchmark has already
//! shipped one badly misleading figure from precisely that error.
//!
//! THE FOUR RULINGS THIS MODULE HAS TO HONOUR, IN CODE:
//!
//! 1. **A type or a measurement nobody actually took reports `NotMeasured`
//!    -- never `Pass`.** [`drop_unrequested`] is how a caller keeps a type
//!    nobody asked the service to find from silently reading as "0% recall,
//!    FAIL" instead of "never attempted." It excludes such a type from
//!    `overall` and `per_language` as well as from `per_type`: a headline
//!    recall still carrying 94 guaranteed misses nobody was asked about is
//!    the same dishonesty one level up, and it is what the per-language
//!    floor is a fraction *of*.
//!
//!    **This ruling has changed for latency, and the old wording is gone
//!    deliberately.** It used to read "p95 latency reports `NotMeasured`
//!    unconditionally, because there is no `pii-detection-model` crate yet".
//!    `pii-detection-model` exists, the service runs the rules and the model
//!    together, and the latency measurement measured a real p95 at 2,000 code points through
//!    `POST /analyze` on the built container, so [`evaluate_latency`] now
//!    takes the measurement and its provenance and returns a real verdict.
//!    Leaving the old comment in place would be worse than no comment: the
//!    next reader would trust it. What it does *not* do is hand out a pass
//!    for a number that is not the number the gate means -- a run measured
//!    at the wrong input length, taken in-process instead of over HTTP, or
//!    taken against a service with no model loaded is `NotMeasured`, not a
//!    pass. `PERSON` and `STREET_ADDRESS` remain `NotMeasured` against any
//!    deployment that does not advertise them.
//!
//!    **This is not a gap the gate runner closes -- it is the mechanism working.**
//!    [`drop_unrequested`] reads `/capabilities` and asks only about what a
//!    deployment actually advertises; it never assumed `pii-detection-model` was
//!    absent, so nothing about it needed to change when the artifact
//!    started shipping. A deployment with `PII_DETECTION_RULES_ONLY=1` set, or one
//!    whose artifact directory is unset or fails digest verification, is
//!    still a *valid* deployment -- rules-only is a legitimate
//!    configuration, just not the default -- and it must keep reporting
//!    `PERSON`/`STREET_ADDRESS` as NOT MEASURED forever, not as 0% recall,
//!    for exactly as long as that deployment exists. `Pass` (or `Fail`)
//!    only appears once a model-backed deployment's `/capabilities`
//!    advertises the type and a run against it is scored.
//! 2. **`VAT_NUMBER` and `TAX_ID` are never compared to Azure.** Azure has no
//!    equivalent for either, so its recall on them is a structural 0.0, and
//!    "non-inferior to zero" is satisfied by detecting nothing at all. Both
//!    are gated on the absolute structured-type thresholds in [`evaluate`]
//!    instead -- `TAX_ID` as checksummed, `VAT_NUMBER` as unchecksummed --
//!    and [`evaluate_against_baseline`] never looks at either: its only two
//!    types are `PERSON` and `STREET_ADDRESS`, full stop.
//! 3. **[`crate::score::Summary::untyped_mask_values`] rides beside every
//!    per-type figure.** This module does not print anything itself -- the
//!    `eval` binary does -- but every [`GateResult`] whose gate reads
//!    `per_type` names its sample size (`n=...`) so a caller cannot quote a
//!    recall without a count next to it, and the `eval` binary surfaces
//!    `untyped_mask_values` beside the whole table for the same reason.
//! 4. **The carrier shortfall follows every per-language number.**
//!    [`SAMPLE_SIZE_CAVEAT`] is folded into every per-language `detail`
//!    string this module produces, not left for the caller to remember to
//!    add.
//!
//! TWO DIFFERENT MEANINGS OF "PASSED," ON PURPOSE. [`Evaluation::passed`] is
//! true when nothing it evaluated explicitly *failed* -- a [`Verdict::Fail`]
//! among the gates in scope. A [`Verdict::NotMeasured`] gate does not flip it
//! to `false`, because [`evaluate`] and [`evaluate_against_baseline`] each
//! see only a slice of a real run (a test summary with one entity type
//! populated, or a run against a rules-only service that structurally cannot
//! populate `PERSON`), and demanding full coverage at this level would make
//! `passed` uninformative on every partial input, including the very tests
//! this module ships with. Ruling 1's "must not return an overall pass while
//! any gate is unmeasured" is enforced one level up, by the `eval` binary,
//! which merges every [`GateResult`] this module produces -- structured,
//! baseline-relative, and latency -- and only calls the run a clean pass when
//! **none** of them is `NotMeasured`. Conflating the two meanings inside this
//! module would make `evaluate(&summary_with_one_type_set)` unable to ever
//! return `passed: true` in a unit test, which is the anti-pattern this
//! plan's own retrospective calls out: a test that cannot fail on the thing
//! it claims to check is worse than no test.
//!
//! Design section 11's table:
//!
//! | Class | Recall | Precision |
//! |---|---|---|
//! | Checksummed structured -- `NATIONAL_ID`, `TAX_ID`, `CREDIT_CARD`, `IBAN_CODE` | >= 98% | >= 99% |
//! | Unchecksummed structured -- `EMAIL_ADDRESS`, `IP_ADDRESS`, `PHONE_NUMBER`, `VAT_NUMBER`, `PASSPORT_NUMBER`, `CREDENTIAL` | >= 95% | >= 97% |
//! | `PERSON`, `STREET_ADDRESS` | non-inferior to Azure, 90% target | -- |
//!
//! plus: no language below 80% of overall recall, and p95 under 50ms at
//! 2,000 characters.

use std::collections::BTreeSet;

use pii_detection_engine::entity::EntityType;
use serde::Serialize;

use crate::latency::{LatencyRun, Transport};
use crate::score::{CaseOutcome, Metrics, Summary};

// ---------------------------------------------------------------------------
// Thresholds -- the design, verbatim
// ---------------------------------------------------------------------------

/// Checksummed structured types: `NATIONAL_ID`, `TAX_ID`, `CREDIT_CARD`,
/// `IBAN_CODE`. `TAX_ID` sits here per ruling 2 -- gated on its own absolute
/// bar, never on non-inferiority to an Azure category that does not exist.
pub const CHECKSUMMED_STRUCTURED: [EntityType; 4] = [
    EntityType::NationalId,
    EntityType::TaxId,
    EntityType::CreditCard,
    EntityType::IbanCode,
];

/// Unchecksummed structured types: `EMAIL_ADDRESS`, `IP_ADDRESS`,
/// `PHONE_NUMBER`, `VAT_NUMBER`, `PASSPORT_NUMBER`, `CREDENTIAL`,
/// `DRIVING_LICENCE`. `VAT_NUMBER` sits here per ruling 2, for the same
/// reason `TAX_ID` sits above.
///
/// **Why `DRIVING_LICENCE` is here even though half of it is checksummed.**
/// The type carries two recognizers and they are not alike. `de_fuehrerschein`
/// names a real check digit -- modulo 11 over the authority key and serial,
/// computed by this service and recorded in `Evidence::checksum` -- and on
/// its own would belong in the row above. `gb_driving_licence` names
/// `validator = "none"`, because the DVLA has refused to publish its check
/// character algorithm; it is decided entirely by a sixteen-character shape
/// that happens to be extremely distinctive. A per-type gate is one number
/// over one denominator, so the two are measured together, and the row a
/// mixed type belongs in is the weaker one: calling the type checksummed
/// would tell the next reader that every span behind the figure was
/// verified, and for the British half nothing was.
///
/// The German half of the check is also weaker than the name suggests.
/// Modulo 11 over consecutive weights misses an adjacent transposition of
/// two characters whose base-36 values differ by a multiple of eleven --
/// `de_fuehrerschein_does_not_catch_every_transposition` in
/// `pii-detection-rules`'s validator pins one -- which is a second reason not to
/// let this type stand in the row whose bar is justified by verification.
///
/// Nothing is given up by the placement in practice: the measured figures
/// (`training/reports/driving-licence.md`) clear the checksummed bar as
/// well as the unchecksummed one. If they ever stop doing so, 95/97 is the
/// honest bar for a type half of which is decided by shape alone.
///
/// **Why `CREDENTIAL` is here and not in [`CHECKSUMMED_STRUCTURED`].** The
/// argument for promoting it is real: `AKIA` followed by sixteen upper-case
/// alphanumerics, or a thirty-character PEM armour line, is arguably *more*
/// decisive than a modulo-97 check over an IBAN, because there is no
/// population of innocent strings with that shape for a checksum to have to
/// separate it from. The argument loses anyway, on what the class name
/// means rather than on how strong the evidence is. Every type in the
/// checksummed row names a check the service itself computes and records in
/// `Evidence::checksum`; a credential recognizer names `validator = "none"`
/// and records `checksum: None`, and the resolver's "a failed checksum
/// rejects on its own" rule therefore never fires for one. Putting a type
/// with no checksum into the row called checksummed would make the row's
/// name false, and the next reader would take a 99% precision bar as
/// evidence that something was verified.
///
/// Nothing is given up by the placement in practice. The measured figures
/// (see `training/reports/credentials.md`) clear the checksummed bar as
/// well as the unchecksummed one; if they ever stop doing so, the 95/97 row
/// is the honest bar for a type decided by shape alone, which is exactly
/// what these are.
///
/// One vendor could have earned the promotion: GitHub encodes a base62
/// CRC32 of the token body in its last six characters, so `ghp_…` is
/// genuinely checksummed. It is deliberately not implemented -- see
/// `packs/universal.toml`'s `credential_github_token` for why a
/// reverse-engineered checksum whose failure *deletes* the candidate is the
/// wrong trade for a secret.
pub const UNCHECKSUMMED_STRUCTURED: [EntityType; 7] = [
    EntityType::EmailAddress,
    EntityType::IpAddress,
    EntityType::PhoneNumber,
    EntityType::VatNumber,
    EntityType::PassportNumber,
    EntityType::Credential,
    EntityType::DrivingLicence,
];

pub const CHECKSUMMED_RECALL_MIN: f64 = 0.98;
pub const CHECKSUMMED_PRECISION_MIN: f64 = 0.99;
pub const UNCHECKSUMMED_RECALL_MIN: f64 = 0.95;
pub const UNCHECKSUMMED_PRECISION_MIN: f64 = 0.97;

/// No language's recall may fall below this fraction of the overall recall.
pub const LANGUAGE_RECALL_FLOOR_RATIO: f64 = 0.80;

/// The design's stated target for `PERSON`/`STREET_ADDRESS` recall. Not
/// itself a pass/fail line -- the gate is non-inferiority to Azure (see
/// [`evaluate_against_baseline`]) -- reported in each gate's detail purely so
/// a reader can see how far a passing (non-inferior) number still is from
/// the design's actual ambition.
pub const PERSON_STREET_ADDRESS_TARGET: f64 = 0.90;

/// The p95 latency ceiling the design sets, and the input length it
/// is measured at.
///
/// **Neither moves.** Design section 3: *"Latency must therefore be a
/// selection gate, not a post-hoc check."* A gate that is raised when the
/// measurement disappoints it is not a gate, and a gate measured at fewer
/// characters than it specifies is measuring a different question. If a run
/// misses, the remedy is in our escalation order -- confirm the
/// export, tune threads and slots, reduce work per request, then distil --
/// never here.
pub const LATENCY_P95_MS_MAX: f64 = 50.0;
pub const LATENCY_TEST_CHARS: usize = 2_000;

/// How far a measured run's input length may sit from [`LATENCY_TEST_CHARS`]
/// and still be the measurement the gate means. Mirrors
/// [`crate::latency::CHARS_TOLERANCE`]: prose cannot be cut to an exact
/// code-point count without cutting mid-word.
pub const LATENCY_CHARS_TOLERANCE: usize = crate::latency::CHARS_TOLERANCE;

/// Ruling 4, folded into every per-language gate this module produces.
///
/// The corpus carried 9 carriers per language until the 2026-08-22
/// expansion and 24 to 28 until the 2026-08-26 one; it now carries 40 to 66
/// (`en` 40, `nl` 44, `de` 47, `fr` 61, `it` 63, `es` 66), so every language
/// is at or above our floor of 40 for the first time and one
/// carrier moves a per-language figure by 1.5% to 2.5% of its own weight
/// rather than 4%, or the 11% it was at 9.
///
/// **Small is not the same as gone.** The per-language `must_mask`
/// denominators these gates divide by are 393 to 483 as of the 2026-09-12
/// `DRIVING_LICENCE` expansion (1,676 -> 2,041 cases; every `DRIVING_LICENCE`
/// case reuses an existing carrier, so the carrier counts above did not
/// move), so one value is still worth roughly a quarter of a percentage
/// point. The smallest *per-type* denominators are still `PASSPORT_NUMBER`
/// 136, `CREDIT_CARD` 139, `VAT_NUMBER` 139 and `IP_ADDRESS` 140 (all four
/// carried by all six languages, and all four unchanged in count because no
/// `DRIVING_LICENCE` case touches any of them) and are far smaller than the
/// per-language figures. The caveat names both denominators, because a
/// reader who sees only the carrier count will read a per-type figure as
/// better-estimated than it is.
///
/// This constant's own numbers are stale exactly when the fixture grows and
/// nobody remembers to re-derive them here -- it has happened three times
/// already (1,150 -> 1,452, 1,452 -> 1,676 and 1,676 -> 2,041 cases each left
/// it quoting the prior population's figures for a while). There is no
/// automated check tying this string to `benchmark/cases/`'s actual counts;
/// re-derive it by hand against the current fixture whenever case counts
/// move, the same way this revision did.
///
/// The Azure baseline is a separate document from this constant (see
/// `benchmark/baseline/README.md`) and is not repeated here: this caveat is
/// about sample size, not about which baseline file is current.
pub const SAMPLE_SIZE_CAVEAT: &str = "sample-size caveat: 40-66 carriers/language against the \
     plan's 40-60 (~1.5-2.5% per carrier, down from ~4% at 24-28 and ~11% at 9); per-language \
     must_mask denominators are 393-483, and the smallest per-type ones are 136-140 \
     (PASSPORT_NUMBER, CREDIT_CARD, VAT_NUMBER, IP_ADDRESS). Treat this figure as indicative, \
     not conclusive.";

// ---------------------------------------------------------------------------
// Verdicts
// ---------------------------------------------------------------------------

/// One gate's outcome. `NotMeasured` is not a kind of failure -- see the
/// module docs on why [`Evaluation::passed`] does not fold it in, and why the
/// `eval` binary's own overall verdict does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Pass,
    Fail,
    NotMeasured,
}

/// One gate's name, verdict, and a human-readable detail line. `detail` is
/// built from counts and rates only -- never a raw value, a case id's text,
/// or anything drawn from `Case::text` -- matching the discipline the rest of
/// this crate holds for anything that might reach a log line or a CI
/// annotation.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GateResult {
    pub name: String,
    pub verdict: Verdict,
    pub detail: String,
}

/// The result of evaluating some scope of gates -- [`evaluate`]'s structured
/// types and languages, or [`evaluate_against_baseline`]'s two
/// baseline-relative types. `passed` is true when nothing in `gates`
/// explicitly failed; see the module docs for why `NotMeasured` does not
/// affect it.
#[derive(Debug, Clone, Serialize)]
pub struct Evaluation {
    pub gates: Vec<GateResult>,
    pub passed: bool,
}

fn no_explicit_failure(gates: &[GateResult]) -> bool {
    !gates.iter().any(|gate| gate.verdict == Verdict::Fail)
}

// ---------------------------------------------------------------------------
// Structured-type and per-language gates (fast: no Azure baseline needed)
// ---------------------------------------------------------------------------

/// Evaluates every structured-type gate (checksummed and unchecksummed,
/// the design's first two rows) and the per-language floor, against
/// `summary`. Takes no baseline -- nothing in this function's scope is ever
/// compared to Azure (ruling 2) -- so it is the gate a pull request can run
/// without spending anything.
pub fn evaluate(summary: &Summary) -> Evaluation {
    let mut gates = Vec::new();

    for entity_type in CHECKSUMMED_STRUCTURED {
        gates.extend(structured_gates(
            summary,
            entity_type,
            "checksummed structured",
            CHECKSUMMED_RECALL_MIN,
            CHECKSUMMED_PRECISION_MIN,
        ));
    }
    for entity_type in UNCHECKSUMMED_STRUCTURED {
        gates.extend(structured_gates(
            summary,
            entity_type,
            "unchecksummed structured",
            UNCHECKSUMMED_RECALL_MIN,
            UNCHECKSUMMED_PRECISION_MIN,
        ));
    }
    gates.extend(language_gates(summary));

    let passed = no_explicit_failure(&gates);
    Evaluation { gates, passed }
}

fn structured_gates(
    summary: &Summary,
    entity_type: EntityType,
    class: &str,
    recall_min: f64,
    precision_min: f64,
) -> Vec<GateResult> {
    let type_name = wire_name(entity_type);
    let metrics = summary.per_type.get(type_name);
    vec![
        recall_gate(
            &format!("{type_name} recall ({class})"),
            metrics,
            recall_min,
        ),
        precision_gate(
            &format!("{type_name} precision ({class})"),
            metrics,
            precision_min,
        ),
    ]
}

fn recall_gate(name: &str, metrics: Option<&Metrics>, min: f64) -> GateResult {
    match metrics.and_then(Metrics::recall) {
        None => GateResult {
            name: name.to_string(),
            verdict: Verdict::NotMeasured,
            detail: "n=0 must_mask values with a declared type in this run".to_string(),
        },
        Some(recall) => {
            let n = metrics.map(|m| m.mask_total).unwrap_or(0);
            let verdict = if recall >= min {
                Verdict::Pass
            } else {
                Verdict::Fail
            };
            let cmp = if recall >= min { ">=" } else { "<" };
            GateResult {
                name: name.to_string(),
                verdict,
                detail: format!("recall {recall:.4} (n={n}) {cmp} {min:.4} required"),
            }
        }
    }
}

fn precision_gate(name: &str, metrics: Option<&Metrics>, min: f64) -> GateResult {
    match metrics.and_then(Metrics::precision) {
        None => GateResult {
            name: name.to_string(),
            verdict: Verdict::NotMeasured,
            detail: "n=0 true or false positives to divide in this run".to_string(),
        },
        Some(precision) => {
            let n = metrics
                .map(|m| m.mask_found + m.clear_violated + m.spurious_spans)
                .unwrap_or(0);
            let verdict = if precision >= min {
                Verdict::Pass
            } else {
                Verdict::Fail
            };
            let cmp = if precision >= min { ">=" } else { "<" };
            GateResult {
                name: name.to_string(),
                verdict,
                detail: format!("precision {precision:.4} (n={n}) {cmp} {min:.4} required"),
            }
        }
    }
}

/// One gate per language present in `summary.per_language`, sorted by name
/// for a deterministic report. A language with no `must_mask` values at all
/// reports `NotMeasured`; a language whose recall falls below
/// [`LANGUAGE_RECALL_FLOOR_RATIO`] of the overall recall fails. No languages
/// are checked at all when the overall recall itself has nothing to measure.
fn language_gates(summary: &Summary) -> Vec<GateResult> {
    let Some(overall_recall) = summary.overall.recall() else {
        return Vec::new();
    };
    let floor = overall_recall * LANGUAGE_RECALL_FLOOR_RATIO;

    let mut languages: Vec<&String> = summary.per_language.keys().collect();
    languages.sort();

    languages
        .into_iter()
        .map(|lang| {
            let metrics = &summary.per_language[lang];
            let name = format!("language floor: {lang}");
            match metrics.recall() {
                None => GateResult {
                    name,
                    verdict: Verdict::NotMeasured,
                    detail: format!("n=0 must_mask values for {lang} in this run. {SAMPLE_SIZE_CAVEAT}"),
                },
                Some(recall) => {
                    let n = metrics.mask_total;
                    let verdict = if recall >= floor { Verdict::Pass } else { Verdict::Fail };
                    let cmp = if recall >= floor { ">=" } else { "<" };
                    GateResult {
                        name,
                        verdict,
                        detail: format!(
                            "{lang} recall {recall:.4} (n={n}) {cmp} floor {floor:.4} \
                             ({ratio:.0}% of overall recall {overall_recall:.4}). {SAMPLE_SIZE_CAVEAT}",
                            ratio = LANGUAGE_RECALL_FLOOR_RATIO * 100.0
                        ),
                    }
                }
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Baseline-relative gates (PERSON / STREET_ADDRESS only -- ruling 2)
// ---------------------------------------------------------------------------

/// Azure's recall on the two types the detector gates against it. `None` when the
/// comparison run this figure would come from was not supplied -- the `eval`
/// binary's default on every pull request, since the Azure comparison is
/// scheduled-only.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct Baseline {
    pub person_recall: Option<f64>,
    pub street_address_recall: Option<f64>,
}

/// Evaluates `PERSON` and `STREET_ADDRESS` -- and *only* those two -- for
/// non-inferiority to `baseline`. Ruling 2: `VAT_NUMBER` and `TAX_ID` never
/// appear here; a caller cannot even ask this function about them, because
/// there is no parameter through which to.
pub fn evaluate_against_baseline(summary: &Summary, baseline: Baseline) -> Evaluation {
    let gates = vec![
        non_inferiority_gate(
            "PERSON",
            summary.per_type.get(wire_name(EntityType::Person)),
            baseline.person_recall,
        ),
        non_inferiority_gate(
            "STREET_ADDRESS",
            summary.per_type.get(wire_name(EntityType::StreetAddress)),
            baseline.street_address_recall,
        ),
    ];
    let passed = no_explicit_failure(&gates);
    Evaluation { gates, passed }
}

fn non_inferiority_gate(
    type_name: &str,
    metrics: Option<&Metrics>,
    azure_recall: Option<f64>,
) -> GateResult {
    let name = format!("{type_name} non-inferior to Azure");
    match (metrics.and_then(Metrics::recall), azure_recall) {
        (Some(pii_detection_recall), Some(azure_recall)) => {
            let n = metrics.map(|m| m.mask_total).unwrap_or(0);
            let verdict = if pii_detection_recall >= azure_recall {
                Verdict::Pass
            } else {
                Verdict::Fail
            };
            let cmp = if pii_detection_recall >= azure_recall {
                ">="
            } else {
                "<"
            };
            GateResult {
                name,
                verdict,
                detail: format!(
                    "pii-detection recall {pii_detection_recall:.4} (n={n}) {cmp} azure {azure_recall:.4}; \
                     design target {target:.0}% (informational, not itself a pass/fail line)",
                    target = PERSON_STREET_ADDRESS_TARGET * 100.0
                ),
            }
        }
        _ => GateResult {
            name,
            verdict: Verdict::NotMeasured,
            detail: "requires both a detector measurement and an Azure baseline figure. \
                     Either no --baseline was supplied, or this deployment does not \
                     advertise this type and so was never asked to detect it (see \
                     drop_unrequested). Which of the two it is is visible in the report's \
                     `baseline.provided` and `requested_types`."
                .to_string(),
        },
    }
}

// ---------------------------------------------------------------------------
// Latency (ruling 1: never a pass, regardless of the number)
// ---------------------------------------------------------------------------

/// The latency gate: the design's *"p95 under 50 ms at 2,000
/// characters on the chosen task size with the shipped artifact"*, evaluated
/// against a [`LatencyRun`] produced by the `latency` binary.
///
/// **Four ways to be `NotMeasured` rather than a pass**, because each one is
/// a way for a true number to be the wrong number:
///
/// * `None` -- no run was supplied. `eval` was not given `--latency`.
/// * A run whose `chars` is outside the [`LATENCY_TEST_CHARS`] band. The
///   fixture's own request times are a real measurement of a mostly-short
///   population, and quoting them here is the exact dishonesty this gate was
///   rewritten to remove.
/// * A run taken in-process rather than over HTTP. `Model::detect` timed
///   with an `Instant` hides tokenisation's caller, the HTTP layer, the
///   rules engine running alongside, the semaphore, serialisation and the
///   runtime. It is the floor a served p95 can only be above.
/// * A run against a service whose `/capabilities` did not advertise
///   `PERSON` -- that is, with no model in the loop. The gate is about
///   `rules ∪ model`; the rules alone answer in about a millisecond, and a
///   pass built on that measures the absence of the thing being gated.
///
/// Otherwise `Pass` or `Fail` against [`LATENCY_P95_MS_MAX`], with the
/// hardware stamp in `detail` -- because a latency verdict without the
/// machine that produced it is not quotable, and the design stamps its
/// own table *"An ARM proxy for Graviton, not the target instance"*.
///
/// **The boundary is `<=`.** "Under 50 ms" is read as "not over 50 ms", so a
/// run measuring exactly 50.0 passes. Written down here so nobody
/// re-litigates it from the prose later.
pub fn evaluate_latency(measured: Option<&LatencyRun>) -> GateResult {
    let name = format!("p95 latency @ {LATENCY_TEST_CHARS} chars < {LATENCY_P95_MS_MAX:.0}ms");

    let Some(run) = measured else {
        return GateResult {
            name,
            verdict: Verdict::NotMeasured,
            detail: "not measured: no latency run supplied (eval was not given --latency)"
                .to_string(),
        };
    };

    let stamp = run.provenance.hardware.to_string();
    let unmeasured = |detail: String| GateResult {
        name: name.clone(),
        verdict: Verdict::NotMeasured,
        detail,
    };

    if !run.measured_at_gate_length() {
        return unmeasured(format!(
            "not measured: the run was taken at {} characters (median of {} to {}), and the \
             gate is specified at {LATENCY_TEST_CHARS} +/- {LATENCY_CHARS_TOLERANCE}. Its p95 \
             of {:.1}ms is a true number about a different question. Machine: {stamp}.",
            run.chars, run.chars_min, run.chars_max, run.p95_ms,
        ));
    }

    if run.provenance.transport != Transport::Http {
        return unmeasured(format!(
            "not measured: the run was taken {:?}, not end to end over HTTP. Its p95 of \
             {:.1}ms is the floor a served p95 can only be above -- it excludes the HTTP \
             layer, the rules engine running alongside, the semaphore and serialisation. \
             Machine: {stamp}.",
            run.provenance.transport, run.p95_ms,
        ));
    }

    if !run.provenance.model_loaded {
        return unmeasured(format!(
            "not measured: /capabilities did not advertise PERSON, so no model was in the \
             loop. Its p95 of {:.1}ms is the rules engine alone, and the gate is about rules \
             union model. Machine: {stamp}.",
            run.p95_ms,
        ));
    }

    let over = run.p95_ms > LATENCY_P95_MS_MAX;
    let margin = (LATENCY_P95_MS_MAX - run.p95_ms).abs();
    let detail = format!(
        "p95 {:.1}ms {} {LATENCY_P95_MS_MAX:.0}ms by {margin:.1}ms over n={} end-to-end \
         POST /analyze requests at {} code points, {} warm-up requests excluded (p50 {:.1}, \
         p99 {:.1}, max {:.1}, mean {:.1}, sd {:.1}); intra_threads {}, inference_slots {}, \
         concurrency {}; {}. Machine: {stamp}. Proxy for: {}.",
        run.p95_ms,
        if over { "is OVER" } else { "is inside" },
        run.n,
        run.chars,
        run.provenance.warmup_n,
        run.end_to_end.p50_ms,
        run.end_to_end.p99_ms,
        run.end_to_end.max_ms,
        run.end_to_end.mean_ms,
        run.end_to_end.stddev_ms,
        run.provenance.threads,
        run.provenance.slots,
        run.provenance.concurrency,
        run.provenance.ort_version,
        run.provenance.proxy_for,
    );

    GateResult {
        name,
        verdict: if over { Verdict::Fail } else { Verdict::Pass },
        detail,
    }
}

// ---------------------------------------------------------------------------
// Keeping "never requested" from reading as "measured, 0% recall"
// ---------------------------------------------------------------------------

/// Whether a `must_mask` value's declared type is one this run never asked
/// the service to detect.
///
/// THE ONE DISTINCTION THE WHOLE MECHANISM TURNS ON, WRITTEN DOWN. A value
/// is excluded for what it **is** -- a declared type absent from `requested`
/// -- and never for what **happened to it**. `MaskVerdict::found` is not
/// read here and must never be: dropping values because they were missed
/// would raise every recall figure in the report by deleting exactly the
/// evidence this benchmark exists to collect. That is the
/// flattering-the-detector defect the fixture's whole design is built to
/// prevent (see `crate::check_coherence`'s docs on the same shape of
/// mistake), and it is why this predicate takes the declared type and
/// nothing else -- there is no parameter through which a caller could make
/// it depend on the outcome.
///
/// An entry with **no** declared type is kept, always. Nothing here knows
/// what it was meant to be, so nothing here can claim nobody asked for it;
/// and keeping it can only push the reported recall down, never up.
fn is_unrequested(declared: Option<EntityType>, requested: &BTreeSet<EntityType>) -> bool {
    declared.is_some_and(|entity_type| !requested.contains(&entity_type))
}

/// Rebuilds a [`Summary`] from `outcomes` with every `must_mask` value whose
/// declared type is not in `requested` left out -- of `overall` and
/// `per_language` as well as `per_type`.
///
/// THE DEFECT THIS GUARDS AGAINST. A [`Summary`] is built from every case's
/// *declared* `must_mask` types (see `crate::score`'s module docs),
/// regardless of what a detector run was actually asked to find. Against a
/// rules-only service, `PERSON` and `STREET_ADDRESS` are never in the
/// `entities` list a caller sends `/analyze` -- the service has no
/// recognizer for either and rejects the request outright if they are asked
/// for -- so every case whose fixture declares a `PERSON` value still
/// contributes `mask_total += 1`, and `mask_found` never gets a chance to
/// move, because nothing was ever asked to find it. Read directly, that is
/// indistinguishable from "the detector tried on every one of these and
/// missed": `Metrics::recall()` returns `Some(0.0)`, not `None`, and
/// [`evaluate_against_baseline`] would report a confident `Fail` rather than
/// the honest `NotMeasured` ruling 1 requires.
///
/// WHY THIS TAKES `&[CaseOutcome]` AND NOT `&Summary`. It used to take a
/// [`Summary`] and `retain` on `per_type` alone, which is all a [`Summary`]
/// makes reachable: `overall` and `per_language` are pooled counts with the
/// per-value type attribution already thrown away. The result was a report
/// whose per-type table honestly said `PERSON: not measured` while its
/// headline recall and every per-language figure still carried all 94
/// `PERSON` and `STREET_ADDRESS` values -- 18.5% of the fixture's 507
/// `must_mask` values -- as guaranteed misses. That is not a cosmetic
/// difference. It moved the reported overall recall from 0.7070 to 0.5759,
/// invited a comparison against an Azure number measured over a different
/// subset, and dragged the per-language floor (a fixed fraction of the
/// overall recall) down with it far enough to **invert** the per-language
/// ranking. The counts have to be re-pooled from the per-value verdicts, so
/// the per-value verdicts are what this function takes.
///
/// It re-pools by handing the filtered outcomes back to `Summary::from`, the
/// same roll-up every other number in this benchmark comes from. This
/// module still does not score: no threshold, ratio or coverage rule is
/// computed here, and writing a second one would be the mistake this
/// module's own docs open by warning about.
///
/// The precision side is deliberately untouched. `must_clear` violations and
/// spurious spans are keyed by the type the **detector** returned, and a
/// detector cannot return a type it was never asked for, so there is nothing
/// there to exclude.
pub fn drop_unrequested(outcomes: &[CaseOutcome], requested: &BTreeSet<EntityType>) -> Summary {
    let restricted: Vec<CaseOutcome> = outcomes
        .iter()
        .map(|outcome| CaseOutcome {
            mask: outcome
                .mask
                .iter()
                .filter(|verdict| !is_unrequested(verdict.declared, requested))
                .cloned()
                .collect(),
            ..outcome.clone()
        })
        .collect();
    Summary::from(restricted)
}

// ---------------------------------------------------------------------------
// Wire names
// ---------------------------------------------------------------------------

/// The wire name for an entity type -- `NATIONAL_ID`, not `NationalId`.
/// Written as an exhaustive match, deliberately duplicating
/// `crate::score`'s private `wire_name` (which cannot be called from here):
/// see that function's own docs for why a compile-checked table beats a
/// serde round trip for exactly this purpose. Two copies of a 12-line lookup
/// table is a far smaller risk than the scoring-rule duplication the module
/// docs warn against -- there is only one way to spell any of these names,
/// and the exhaustiveness check catches a missing arm at compile time either
/// way.
///
/// `pub` so the `eval` binary -- a separate crate from this one even though
/// it lives in the same package -- can build the `entities` list it sends
/// `/analyze` from the same table, rather than growing a third copy.
pub fn wire_name(entity_type: EntityType) -> &'static str {
    match entity_type {
        EntityType::Credential => "CREDENTIAL",
        EntityType::CreditCard => "CREDIT_CARD",
        EntityType::DrivingLicence => "DRIVING_LICENCE",
        EntityType::EmailAddress => "EMAIL_ADDRESS",
        EntityType::IbanCode => "IBAN_CODE",
        EntityType::IpAddress => "IP_ADDRESS",
        EntityType::NationalId => "NATIONAL_ID",
        EntityType::PassportNumber => "PASSPORT_NUMBER",
        EntityType::Person => "PERSON",
        EntityType::PhoneNumber => "PHONE_NUMBER",
        EntityType::StreetAddress => "STREET_ADDRESS",
        EntityType::TaxId => "TAX_ID",
        EntityType::VatNumber => "VAT_NUMBER",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::MaskVerdict;
    use std::collections::HashMap;

    // -----------------------------------------------------------------
    // Test summary builders. Metrics stores raw usize counts, not a
    // floating-point ratio, so these pick counts on a common
    // one-million denominator that land recall/precision within about
    // 1e-6 of the requested value -- nowhere near enough to cross any
    // threshold this module checks (the tightest is a 1% margin), and
    // exact enough that every assertion below is checking the real
    // comparison, not a rounding artifact.
    // -----------------------------------------------------------------

    const DENOM: usize = 1_000_000;

    fn recall_metrics(recall: f64) -> Metrics {
        Metrics {
            mask_total: DENOM,
            mask_found: (recall * DENOM as f64).round() as usize,
            clear_total: 0,
            clear_violated: 0,
            spurious_spans: 0,
        }
    }

    /// Builds `Metrics` whose `recall()` and `precision()` are each close to
    /// the requested value, by picking a false-positive count that solves
    /// `mask_found / (mask_found + fp) = precision`. Rounds the
    /// false-positive count *down* (never up), so the computed precision is
    /// always at least `precision`, never a rounding hair below it -- a
    /// boundary test (e.g. "exactly 0.99 passes") must not flake on which
    /// way `f64` rounding happened to fall.
    fn metrics_for(recall: f64, precision: f64) -> Metrics {
        let mask_found = (recall * DENOM as f64).round() as usize;
        let false_positives = if precision <= 0.0 {
            0
        } else {
            ((mask_found as f64) * (1.0 - precision) / precision).floor() as usize
        };
        Metrics {
            mask_total: DENOM,
            mask_found,
            clear_total: false_positives,
            clear_violated: false_positives,
            spurious_spans: 0,
        }
    }

    /// Builds `Metrics` whose `precision()` is *exactly* `mask_found as f64
    /// / (mask_found + false_positives) as f64` -- bit-identical to the
    /// nearest-representable-double a literal like `0.99` or `0.97` parses
    /// to, when the caller picks `mask_found`/`false_positives` so that
    /// ratio reduces to a fraction of a small power of ten (e.g. `99/100`,
    /// `97/100`). `metrics_for` above cannot do this: it floors its
    /// false-positive count so the computed precision never falls short of
    /// the nominal target, which means it also never lands exactly *on*
    /// it -- a test built from `metrics_for` can prove "at or above
    /// threshold passes" but not "exactly at the threshold passes," and an
    /// inclusive-boundary test needs the latter. `recall()` here is pinned
    /// to `1.0` (`mask_total == mask_found`) so it can never itself cause a
    /// verdict this constructor's callers are not testing.
    fn exact_precision_metrics(mask_found: usize, false_positives: usize) -> Metrics {
        Metrics {
            mask_total: mask_found,
            mask_found,
            clear_total: false_positives,
            clear_violated: false_positives,
            spurious_spans: 0,
        }
    }

    fn summary_with(entity_type: &str, recall: f64, precision: f64) -> Summary {
        let mut per_type = HashMap::new();
        per_type.insert(entity_type.to_string(), metrics_for(recall, precision));
        Summary {
            per_type,
            ..Default::default()
        }
    }

    fn summary_with_languages(languages: &[(&str, f64)], overall_recall: f64) -> Summary {
        let mut per_language = HashMap::new();
        for (lang, recall) in languages {
            per_language.insert(lang.to_string(), recall_metrics(*recall));
        }
        Summary {
            overall: recall_metrics(overall_recall),
            per_language,
            ..Default::default()
        }
    }

    fn person_recall(recall: f64) -> Summary {
        let mut per_type = HashMap::new();
        per_type.insert("PERSON".to_string(), recall_metrics(recall));
        Summary {
            per_type,
            ..Default::default()
        }
    }

    fn street_address_recall(recall: f64) -> Summary {
        let mut per_type = HashMap::new();
        per_type.insert("STREET_ADDRESS".to_string(), recall_metrics(recall));
        Summary {
            per_type,
            ..Default::default()
        }
    }

    fn azure_person_recall(recall: f64) -> Baseline {
        Baseline {
            person_recall: Some(recall),
            street_address_recall: None,
        }
    }

    fn azure_street_address_recall(recall: f64) -> Baseline {
        Baseline {
            person_recall: None,
            street_address_recall: Some(recall),
        }
    }

    // -----------------------------------------------------------------
    // The brief's three tests, verbatim.
    // -----------------------------------------------------------------

    #[test]
    fn a_checksummed_type_below_ninety_eight_percent_recall_fails_its_gate() {
        let summary = summary_with("NATIONAL_ID", 0.97, 0.99);
        assert!(!evaluate(&summary).passed);
    }

    #[test]
    fn a_language_below_eighty_percent_of_overall_recall_fails() {
        // A model that is excellent on average and blind in Italian is not a
        // model this product can ship.
        let summary = summary_with_languages(&[("nl", 0.95), ("it", 0.70)], 0.92);
        assert!(!evaluate(&summary).passed);
    }

    #[test]
    fn person_is_gated_on_non_inferiority_to_azure_not_an_absolute() {
        let summary = person_recall(0.86);
        assert!(evaluate_against_baseline(&summary, azure_person_recall(0.84)).passed);
        assert!(!evaluate_against_baseline(&summary, azure_person_recall(0.91)).passed);
    }

    // -----------------------------------------------------------------
    // Mutation-proof supplements: each of these is written so that
    // flipping the comparison direction, the threshold, or removing a
    // safeguard makes it fail. That was verified by hand for a
    // representative subset (see the task report).
    // -----------------------------------------------------------------

    #[test]
    fn a_checksummed_type_exactly_at_both_thresholds_passes() {
        // WHAT THIS DOES AND DOES NOT PROVE. The recall side is a genuine
        // equality-boundary check: recall is a direct proportion with no
        // rounding, so `metrics_for`'s 0.98 lands on the exact same double
        // `evaluate`'s 0.98 threshold does. The precision side does NOT
        // prove the same thing for precision -- `metrics_for` floors its
        // false-positive count, so the computed precision here is a hair
        // *above* 0.99, never exactly on it (see `metrics_for`'s own
        // docs). This test still shows nothing else in the summary being
        // measured does not itself count against `passed` -- see the
        // module docs on why NotMeasured does not fail it. The genuine
        // precision equality-boundary check is
        // `a_checksummed_precision_gate_at_exact_equality_with_its_threshold_passes`,
        // below.
        let summary = summary_with("IBAN_CODE", 0.98, 0.99);
        let evaluation = evaluate(&summary);
        assert!(evaluation.passed, "gates: {:?}", evaluation.gates);

        let iban_recall = evaluation
            .gates
            .iter()
            .find(|g| g.name == "IBAN_CODE recall (checksummed structured)")
            .expect("an IBAN_CODE recall gate is always produced");
        assert_eq!(iban_recall.verdict, Verdict::Pass);
    }

    #[test]
    fn a_checksummed_precision_gate_at_exact_equality_with_its_threshold_passes() {
        // 99 / 100 == 0.99 bit-for-bit in f64 (both the division and the
        // literal are correctly-rounded to the same unique nearest double
        // for the exact rational 99/100), unlike anything built through
        // `metrics_for`'s floored false-positive count. This is the genuine
        // inclusive-boundary proof `a_checksummed_type_exactly_at_both_thresholds_passes`
        // could not give for precision.
        let mut per_type = HashMap::new();
        per_type.insert("IBAN_CODE".to_string(), exact_precision_metrics(99, 1));
        let summary = Summary {
            per_type,
            ..Default::default()
        };
        let evaluation = evaluate(&summary);
        let precision = evaluation
            .gates
            .iter()
            .find(|g| g.name == "IBAN_CODE precision (checksummed structured)")
            .expect("gate present");
        assert_eq!(
            precision.verdict,
            Verdict::Pass,
            "detail: {}",
            precision.detail
        );
    }

    #[test]
    fn an_unchecksummed_precision_gate_at_exact_equality_with_its_threshold_passes() {
        // 97 / 100 == 0.97 bit-for-bit, same reasoning as the checksummed
        // version above.
        let mut per_type = HashMap::new();
        per_type.insert("PHONE_NUMBER".to_string(), exact_precision_metrics(97, 3));
        let summary = Summary {
            per_type,
            ..Default::default()
        };
        let evaluation = evaluate(&summary);
        let precision = evaluation
            .gates
            .iter()
            .find(|g| g.name == "PHONE_NUMBER precision (unchecksummed structured)")
            .expect("gate present");
        assert_eq!(
            precision.verdict,
            Verdict::Pass,
            "detail: {}",
            precision.detail
        );
    }

    #[test]
    fn person_non_inferiority_at_exact_equality_with_azure_recall_passes() {
        // The detector's measured recall (99/100, exact) and Azure's baseline
        // figure (the literal 0.99) land on the identical f64 bit pattern.
        // Non-inferiority is supposed to mean "at least as good," so equal
        // must pass -- this is the equality-boundary case
        // `person_is_gated_on_non_inferiority_to_azure_not_an_absolute`
        // (0.86 vs. 0.84/0.91, neither an exact tie) does not cover.
        let mut per_type = HashMap::new();
        per_type.insert(
            "PERSON".to_string(),
            Metrics {
                mask_total: 100,
                mask_found: 99,
                clear_total: 0,
                clear_violated: 0,
                spurious_spans: 0,
            },
        );
        let summary = Summary {
            per_type,
            ..Default::default()
        };
        let evaluation = evaluate_against_baseline(
            &summary,
            Baseline {
                person_recall: Some(0.99),
                street_address_recall: None,
            },
        );
        let person = evaluation
            .gates
            .iter()
            .find(|g| g.name == "PERSON non-inferior to Azure")
            .expect("gate present");
        assert_eq!(person.verdict, Verdict::Pass, "detail: {}", person.detail);
    }

    #[test]
    fn a_checksummed_type_one_case_under_the_recall_threshold_fails() {
        // Same shape as the boundary test, one step below it, to prove the
        // boundary test above is actually exercising the comparison and not
        // passing regardless of the input.
        let summary = summary_with("IBAN_CODE", 0.9799, 0.99);
        let evaluation = evaluate(&summary);
        assert!(!evaluation.passed);
        let iban_recall = evaluation
            .gates
            .iter()
            .find(|g| g.name == "IBAN_CODE recall (checksummed structured)")
            .expect("gate present");
        assert_eq!(iban_recall.verdict, Verdict::Fail);
    }

    #[test]
    fn a_checksummed_type_with_good_recall_but_bad_precision_still_fails() {
        let summary = summary_with("CREDIT_CARD", 0.99, 0.98);
        let evaluation = evaluate(&summary);
        assert!(!evaluation.passed);
        let precision = evaluation
            .gates
            .iter()
            .find(|g| g.name == "CREDIT_CARD precision (checksummed structured)")
            .expect("gate present");
        assert_eq!(precision.verdict, Verdict::Fail);
    }

    #[test]
    fn an_unchecksummed_type_below_its_lower_recall_bar_fails() {
        // 0.94 would pass the checksummed bar's neighbourhood but is below
        // the unchecksummed bar (95%) -- proves the two classes use
        // different thresholds, not a single shared one.
        let summary = summary_with("PHONE_NUMBER", 0.94, 0.98);
        assert!(!evaluate(&summary).passed);
    }

    #[test]
    fn an_unchecksummed_type_at_its_own_bar_passes() {
        // Same caveat as the checksummed version above: this genuinely
        // proves recall's boundary (a direct proportion, exact) but not
        // precision's (floored by `metrics_for`, so it lands a hair above
        // 0.97, never exactly on it). See
        // `an_unchecksummed_precision_gate_at_exact_equality_with_its_threshold_passes`
        // for the real precision equality-boundary check.
        let summary = summary_with("PHONE_NUMBER", 0.95, 0.97);
        let evaluation = evaluate(&summary);
        assert!(evaluation.passed, "gates: {:?}", evaluation.gates);
    }

    #[test]
    fn vat_number_and_tax_id_are_gated_on_absolute_thresholds_in_evaluate() {
        // Ruling 2, structurally: `evaluate` takes no baseline parameter at
        // all, so there is no way for VAT_NUMBER or TAX_ID to be compared
        // against Azure's structural 0.0 through this function.
        let vat = summary_with("VAT_NUMBER", 0.95, 0.97);
        assert!(
            evaluate(&vat).passed,
            "VAT_NUMBER at its own bar should pass"
        );

        let tax_id = summary_with("TAX_ID", 0.98, 0.99);
        assert!(
            evaluate(&tax_id).passed,
            "TAX_ID at its own bar should pass"
        );

        let weak_vat = summary_with("VAT_NUMBER", 0.80, 0.97);
        assert!(
            !evaluate(&weak_vat).passed,
            "a weak VAT_NUMBER recall must still fail"
        );
    }

    #[test]
    fn evaluate_against_baseline_never_produces_a_vat_or_tax_id_gate() {
        // Even if a caller tried to smuggle VAT_NUMBER/TAX_ID data into a
        // summary handed to the baseline-relative evaluator, that evaluator
        // only ever asks about PERSON and STREET_ADDRESS -- ruling 2 holds
        // structurally, not just by convention.
        let mut summary = summary_with("VAT_NUMBER", 0.95, 0.97);
        summary
            .per_type
            .insert("TAX_ID".to_string(), metrics_for(0.98, 0.99));

        let evaluation = evaluate_against_baseline(&summary, Baseline::default());
        assert!(
            evaluation
                .gates
                .iter()
                .all(|g| !g.name.contains("VAT_NUMBER") && !g.name.contains("TAX_ID"))
        );
    }

    #[test]
    fn a_language_at_exactly_the_eighty_percent_floor_passes() {
        // Overall recall pinned at 1.0 so evaluate()'s floor
        // (overall_recall * LANGUAGE_RECALL_FLOOR_RATIO) is exactly
        // LANGUAGE_RECALL_FLOOR_RATIO -- multiplying by 1.0 introduces no
        // rounding -- and nl is built at that same constant, so both sides
        // of the `>=` comparison land on the identical f64 bit pattern
        // rather than two independently-rounded approximations of "0.736".
        let summary = summary_with_languages(&[("nl", LANGUAGE_RECALL_FLOOR_RATIO)], 1.0);
        let evaluation = evaluate(&summary);
        assert!(evaluation.passed, "gates: {:?}", evaluation.gates);
    }

    #[test]
    fn every_language_gate_detail_carries_the_sample_size_caveat() {
        // Ruling 4: repeated wherever a per-language number is quoted, not
        // left implicit.
        let summary = summary_with_languages(&[("nl", 0.95), ("it", 0.70)], 0.92);
        let evaluation = evaluate(&summary);
        let language_gates: Vec<&GateResult> = evaluation
            .gates
            .iter()
            .filter(|g| g.name.starts_with("language floor"))
            .collect();
        assert_eq!(language_gates.len(), 2);
        for gate in language_gates {
            assert!(gate.detail.contains("40-66 carriers/language"));
        }
    }

    #[test]
    fn not_measured_structured_types_do_not_block_an_otherwise_clean_pass() {
        // Only NATIONAL_ID has data; the other eight structured types are
        // NotMeasured. `passed` must still be true -- see the module docs
        // on why full coverage is enforced one level up, by the eval
        // binary, not here.
        let summary = summary_with("NATIONAL_ID", 0.99, 0.995);
        let evaluation = evaluate(&summary);
        assert!(evaluation.passed);
        assert!(
            evaluation
                .gates
                .iter()
                .any(|g| g.verdict == Verdict::NotMeasured),
            "expected at least one NotMeasured gate among the eight untouched types"
        );
    }

    #[test]
    fn street_address_is_gated_on_non_inferiority_to_azure_not_an_absolute() {
        // Mirrors the brief's PERSON test for the other baseline-relative
        // type, so STREET_ADDRESS is not the untested half of the pair.
        let summary = street_address_recall(0.80);
        assert!(evaluate_against_baseline(&summary, azure_street_address_recall(0.75)).passed);
        assert!(!evaluate_against_baseline(&summary, azure_street_address_recall(0.85)).passed);
    }

    // -----------------------------------------------------------------
    // drop_unrequested. These build CaseOutcomes rather than a Summary
    // directly: the whole point of the function is that it re-pools the
    // counts from per-value verdicts, which a Summary has already thrown
    // away.
    // -----------------------------------------------------------------

    fn verdict(declared: Option<EntityType>, found: bool) -> MaskVerdict {
        MaskVerdict {
            value: "x".to_string(),
            found,
            declared,
            types: Vec::new(),
        }
    }

    fn outcome(lang: &str, mask: Vec<MaskVerdict>) -> CaseOutcome {
        CaseOutcome {
            lang: lang.to_string(),
            mask,
            clear: Vec::new(),
            spurious_spans: Vec::new(),
        }
    }

    /// Every type but `PERSON` and `STREET_ADDRESS` -- what a rules-only
    /// `GET /capabilities` reports today.
    fn rules_only_capabilities() -> BTreeSet<EntityType> {
        CHECKSUMMED_STRUCTURED
            .into_iter()
            .chain(UNCHECKSUMMED_STRUCTURED)
            .collect()
    }

    #[test]
    fn a_type_never_requested_from_the_service_reads_not_measured_not_zero_recall() {
        // THE DEFECT drop_unrequested EXISTS FOR. A rules-only service is
        // never asked to find PERSON, so every declared PERSON value in the
        // fixture is a permanent miss if scored blind -- recall 0.0, a
        // confident Fail. That is not the honest status; "never asked" is.
        let outcomes = vec![outcome(
            "nl",
            vec![verdict(Some(EntityType::Person), false)],
        )];

        // Without stripping: recall 0.0 reads as a real, measured failure.
        let unstripped = Summary::from(outcomes.clone());
        let evaluated = evaluate_against_baseline(&unstripped, azure_person_recall(0.5));
        assert!(
            evaluated.gates.iter().any(|g| g.verdict == Verdict::Fail),
            "sanity check: an unstripped 0.0 recall against a 0.5 baseline must read as Fail"
        );

        // With stripping (PERSON not in the requested set): NotMeasured.
        let stripped = drop_unrequested(&outcomes, &rules_only_capabilities());
        let evaluation = evaluate_against_baseline(&stripped, azure_person_recall(0.5));
        assert!(evaluation.gates.iter().all(|g| g.verdict != Verdict::Fail));
        assert!(
            evaluation
                .gates
                .iter()
                .any(|g| g.verdict == Verdict::NotMeasured)
        );
    }

    #[test]
    fn drop_unrequested_leaves_a_requested_type_untouched() {
        // 86 found out of 100 declared NATIONAL_ID values, every one of them
        // requested. Nothing moves.
        let mut mask = vec![verdict(Some(EntityType::NationalId), true); 86];
        mask.extend(vec![verdict(Some(EntityType::NationalId), false); 14]);
        let outcomes = vec![outcome("nl", mask)];

        let stripped = drop_unrequested(&outcomes, &rules_only_capabilities());
        assert_eq!(
            stripped
                .per_type
                .get("NATIONAL_ID")
                .and_then(Metrics::recall),
            Some(0.86)
        );
        assert_eq!(stripped.overall.recall(), Some(0.86));
    }

    #[test]
    fn drop_unrequested_corrects_overall_and_per_language_not_only_per_type() {
        // THE FINDING THIS TEST IS THE FIX FOR. The old implementation
        // retained on `per_type` alone, so a type nobody asked about
        // vanished from the per-type table and stayed in the headline and
        // per-language denominators as a guaranteed miss. Here: one Italian
        // case with one requested value (found) and three never-requested
        // ones (therefore never found). Corrected, `it` reads 1.0 on the one
        // thing that was measured; uncorrected it reads 0.25, and 0.25 is
        // not a measurement of anything.
        let outcomes = vec![outcome(
            "it",
            vec![
                verdict(Some(EntityType::PhoneNumber), true),
                verdict(Some(EntityType::Person), false),
                verdict(Some(EntityType::Person), false),
                verdict(Some(EntityType::StreetAddress), false),
            ],
        )];

        let uncorrected = Summary::from(outcomes.clone());
        assert_eq!(uncorrected.overall.recall(), Some(0.25));
        assert_eq!(
            uncorrected.per_language.get("it").and_then(Metrics::recall),
            Some(0.25)
        );

        let corrected = drop_unrequested(&outcomes, &rules_only_capabilities());
        assert_eq!(corrected.overall.recall(), Some(1.0));
        assert_eq!(
            corrected.per_language.get("it").and_then(Metrics::recall),
            Some(1.0)
        );
        assert_eq!(corrected.overall.mask_total, 1);
        assert!(!corrected.per_type.contains_key("PERSON"));
    }

    #[test]
    fn a_requested_type_that_was_simply_missed_stays_in_every_denominator() {
        // THE LINE THIS MECHANISM MUST NOT CROSS, PROVEN. Excluding a type
        // the service was never asked for is honest. Excluding one it *was*
        // asked for and missed is the flattering-the-detector defect this
        // whole benchmark exists to prevent -- and the two differ only by
        // which field the filter reads. Both values below are misses; only
        // the one nobody asked about leaves the denominator.
        let outcomes = vec![outcome(
            "de",
            vec![
                // Requested, and missed. Must survive, everywhere.
                verdict(Some(EntityType::PassportNumber), false),
                // Never requested. Must go, everywhere.
                verdict(Some(EntityType::Person), false),
            ],
        )];

        let corrected = drop_unrequested(&outcomes, &rules_only_capabilities());
        assert_eq!(corrected.overall.mask_total, 1);
        assert_eq!(corrected.overall.recall(), Some(0.0));
        assert_eq!(
            corrected.per_language.get("de").and_then(Metrics::recall),
            Some(0.0)
        );
        assert_eq!(
            corrected
                .per_type
                .get("PASSPORT_NUMBER")
                .and_then(Metrics::recall),
            Some(0.0)
        );
    }

    /// One `CaseOutcome` per `PERSON` value: `found` verdicts marked found,
    /// `missed` marked not found. Mirrors `outcome`/`verdict` above -- built
    /// for the brief's own tests, which need a specific found/missed split
    /// on a single type rather than the fixed-ratio `recall_metrics` builder
    /// `Summary`-level tests use.
    fn outcomes_with_person_values(found: usize, missed: usize) -> Vec<CaseOutcome> {
        let mut mask = vec![verdict(Some(EntityType::Person), true); found];
        mask.extend(vec![verdict(Some(EntityType::Person), false); missed]);
        vec![outcome("en", mask)]
    }

    // -----------------------------------------------------------------
    // The gate runner's own tests, verbatim from the brief: the whole point of
    // this work, asserted in one place. A model-backed run must produce a
    // real PERSON verdict, that verdict must be able to fail, and a
    // deployment with no artifact at all must still read NOT MEASURED --
    // drop_unrequested's reason for existing does not go away.
    // -----------------------------------------------------------------

    #[test]
    fn a_model_backed_run_produces_a_real_person_verdict_not_not_measured() {
        // The whole point of this work, asserted in one place.
        let outcomes = outcomes_with_person_values(/* found */ 60, /* missed */ 9);
        let requested = BTreeSet::from([EntityType::Person, EntityType::StreetAddress]);
        let summary = drop_unrequested(&outcomes, &requested);
        let evaluation = evaluate_against_baseline(&summary, azure_person_recall(0.80));
        let person = evaluation
            .gates
            .iter()
            .find(|g| g.name.starts_with("PERSON"))
            .expect("gate");
        assert_ne!(person.verdict, Verdict::NotMeasured);
        assert_eq!(person.verdict, Verdict::Pass); // 60/69 = 0.8696 >= 0.80
    }

    #[test]
    fn a_person_recall_below_azure_still_fails() {
        // The gate has to be able to fail on the day it starts running, or it
        // is decoration. 50/69 = 0.7246 < 0.80.
        let outcomes = outcomes_with_person_values(50, 19);
        let requested = BTreeSet::from([EntityType::Person]);
        let summary = drop_unrequested(&outcomes, &requested);
        assert!(!evaluate_against_baseline(&summary, azure_person_recall(0.80)).passed);
    }

    #[test]
    fn a_deployment_without_an_artifact_still_reports_not_measured() {
        // drop_unrequested's reason for existing does not go away. A
        // rules-only deployment must still say "never attempted", not "0%".
        let outcomes = outcomes_with_person_values(0, 69);
        let summary = drop_unrequested(&outcomes, &rules_only_capabilities());
        let evaluation = evaluate_against_baseline(&summary, azure_person_recall(0.80));
        assert!(
            evaluation
                .gates
                .iter()
                .any(|g| g.verdict == Verdict::NotMeasured)
        );
        assert!(evaluation.passed);
    }

    #[test]
    fn an_untyped_mask_value_is_never_dropped() {
        // 152 of the fixture's 507 must_mask values declare no type at all.
        // Nothing here knows what they were meant to be, so nothing here may
        // claim nobody asked for them. They stay counted -- which can only
        // lower the reported recall, never raise it.
        let outcomes = vec![outcome(
            "nl",
            vec![
                verdict(None, false),
                verdict(Some(EntityType::Person), false),
            ],
        )];
        let corrected = drop_unrequested(&outcomes, &rules_only_capabilities());
        assert_eq!(corrected.overall.mask_total, 1);
        assert_eq!(corrected.untyped_mask_values, 1);
        assert_eq!(corrected.overall.recall(), Some(0.0));
    }

    // -----------------------------------------------------------------
    // The latency gate. Every one of these proves the gate can produce a
    // verdict it does not want to -- in both directions.
    // -----------------------------------------------------------------

    /// A run stamped exactly as the `latency` binary stamps a real one,
    /// with the three things the gate inspects left to the caller.
    fn run_at(p95_ms: f64, chars: usize, model_loaded: bool) -> LatencyRun {
        run_with(p95_ms, chars, model_loaded, Transport::Http)
    }

    fn run_with(p95_ms: f64, chars: usize, model_loaded: bool, transport: Transport) -> LatencyRun {
        // A hundred samples all at `p95_ms`, so the p95 is exactly it.
        let mut run = LatencyRun::from_samples(vec![p95_ms; 100], chars);
        run.provenance = crate::latency::Provenance {
            transport,
            hardware: crate::latency::Hardware::new("a stated machine"),
            proxy_for: "a stated target".to_string(),
            threads: 4,
            slots: 1,
            concurrency: 1,
            ort_version: "a stated runtime".to_string(),
            model_loaded,
            warmup_n: 100,
            command: "a stated command".to_string(),
            measured_unix_secs: 0,
            host_load: crate::latency::HostLoad {
                samples: 0,
                min: None,
                median: None,
                max: None,
                host_vcpus: 0,
            },
            notes: Vec::new(),
        };
        run
    }

    #[test]
    fn a_p95_over_the_budget_fails() {
        let run = run_at(50.1, 2_000, true);
        assert_eq!(evaluate_latency(Some(&run)).verdict, Verdict::Fail);
    }

    #[test]
    fn a_p95_exactly_at_the_budget_passes() {
        // "under 50 ms" -- the boundary is stated so nobody re-litigates it
        // from the prose later.
        let run = run_at(50.0, 2_000, true);
        assert_eq!(evaluate_latency(Some(&run)).verdict, Verdict::Pass);
    }

    #[test]
    fn a_p95_under_the_budget_passes() {
        let run = run_at(37.6, 2_000, true);
        assert_eq!(evaluate_latency(Some(&run)).verdict, Verdict::Pass);
    }

    #[test]
    fn a_fast_run_measured_at_the_wrong_length_is_not_measured_rather_than_a_pass() {
        // The exact dishonesty this gate was rewritten to remove: an
        // excellent number over the whole (mostly short) fixture is not the
        // number the gate means.
        let run = run_at(4.0, 180, true);
        assert_eq!(evaluate_latency(Some(&run)).verdict, Verdict::NotMeasured);
    }

    #[test]
    fn a_run_at_the_edge_of_the_character_band_is_still_measured() {
        // The band exists because prose cannot be cut to an exact code
        // point without cutting mid-word. Both edges are inside it, and one
        // past either edge is not -- otherwise the tolerance is either
        // decoration or a loophole.
        for chars in [
            LATENCY_TEST_CHARS - LATENCY_CHARS_TOLERANCE,
            LATENCY_TEST_CHARS + LATENCY_CHARS_TOLERANCE,
        ] {
            assert_eq!(
                evaluate_latency(Some(&run_at(40.0, chars, true))).verdict,
                Verdict::Pass,
                "{chars} should be inside the band"
            );
        }
        for chars in [
            LATENCY_TEST_CHARS - LATENCY_CHARS_TOLERANCE - 1,
            LATENCY_TEST_CHARS + LATENCY_CHARS_TOLERANCE + 1,
        ] {
            assert_eq!(
                evaluate_latency(Some(&run_at(40.0, chars, true))).verdict,
                Verdict::NotMeasured,
                "{chars} should be outside the band"
            );
        }
    }

    #[test]
    fn a_fast_run_with_no_model_loaded_is_not_measured_rather_than_a_pass() {
        let run = run_at(4.0, 2_000, false);
        assert_eq!(evaluate_latency(Some(&run)).verdict, Verdict::NotMeasured);
    }

    #[test]
    fn a_fast_in_process_run_is_not_measured_rather_than_a_pass() {
        // `Model::detect` around an `Instant` is the floor a served p95 can
        // only be above. Measured in-process, this workload really is
        // faster than the HTTP number -- so this is the shape of an
        // accidental pass, not a hypothetical one.
        let run = run_with(41.0, 2_000, true, Transport::InProcess);
        assert_eq!(evaluate_latency(Some(&run)).verdict, Verdict::NotMeasured);
    }

    #[test]
    fn no_run_at_all_is_not_measured() {
        assert_eq!(evaluate_latency(None).verdict, Verdict::NotMeasured);
    }

    #[test]
    fn every_latency_detail_names_the_machine_and_the_sample_size() {
        // A latency number without its machine is not quotable, and a p95
        // without an n is not a tail. Both travel in `detail`, for the pass
        // and the fail alike.
        for p95 in [37.6, 61.8] {
            let detail = evaluate_latency(Some(&run_at(p95, 2_000, true))).detail;
            assert!(detail.contains("a stated machine"), "{detail}");
            assert!(detail.contains("n=100"), "{detail}");
            assert!(detail.contains("2000 code points"), "{detail}");
        }
    }

    #[test]
    fn every_wire_name_used_by_this_module_round_trips() {
        for entity_type in CHECKSUMMED_STRUCTURED
            .into_iter()
            .chain(UNCHECKSUMMED_STRUCTURED)
            .chain([EntityType::Person, EntityType::StreetAddress])
        {
            let wire = wire_name(entity_type);
            assert_eq!(EntityType::from_wire(wire), Some(entity_type));
        }
    }
}
