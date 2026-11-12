//! Runs Azure AI Language's Text PII API over the benchmark fixture, once,
//! and scores the result with `pii_detection_eval::score` -- the exact scorer that
//! scores the detector's own engine (`fixture-spans`). Two of the benchmark's gates
//! are non-inferiority to Azure, and a number produced by a second, subtly
//! different scorer would not be comparable to the detector's, which is the same
//! discipline `fixture-spans`'s own header documents for the engine side.
//!
//! THIS SPENDS REAL MONEY. Azure AI Language bills PII detection per 1,000-
//! character "text record" (`ceil(chars/1000)` per document, minimum 1 per
//! document), shared with NER/sentiment/key-phrase/language-detection under
//! one 5,000-text-record/month free allotment and, beyond that, list price.
//! This binary always computes and prints the estimate before doing anything
//! else, refuses outright to call Azure when the estimate exceeds
//! `--max-cost-usd` (default $25, see [`DEFAULT_MAX_COST_USD`]), and even
//! under that ceiling will not call Azure without an explicit `--confirm` --
//! a bare invocation only estimates. `--from-raw` rescopes a previous run's
//! saved `raw_batches` with today's scoring rules and never touches the
//! network or the credentials file at all, which is how this binary's own
//! scoring path can be iterated on without spending anything a second time.
//!
//! CREDENTIALS LIVE OUTSIDE THIS REPOSITORY. `AZURE_LANGUAGE_ENDPOINT` and
//! `AZURE_LANGUAGE_KEY` are read at run time out of Temperate's
//! `.env` file (`--env-path`, read-only, never written to) and held only in
//! local variables -- never logged, never traced, never written to the output
//! file, never part of an error message. `--env-path` is **required** with
//! `--confirm` and has no default: one machine's absolute path in committed
//! source is a path that is wrong everywhere else.
//!
//! THE API CONTRACT IS READ FROM, NOT GUESSED FROM, THE PLATFORM'S OWN DRIVER:
//! our earlier Azure client.
//! Specifically: `POST {endpoint}language/:analyze-text?api-version=2024-11-01`,
//! header `Ocp-Apim-Subscription-Key`, `kind=PiiEntityRecognition`,
//! `parameters.modelVersion=latest`, `parameters.stringIndexType=
//! UnicodeCodePoint` (so Azure's offsets are already code points -- the same
//! unit `pii_detection_api::wire::Entity::start`/`end` and this crate's scorer use,
//! needing no translation), `parameters.loggingOptOut=true`, at most 5
//! documents per request (a hard 400 above that, not a truncation). Every
//! benchmark case is well under Azure's 5,120-character synchronous limit
//! (`AZURE_SYNC_LIMIT` below, with margin), so unlike the PHP driver this
//! binary does not implement long-document windowing -- it refuses to run at
//! all if a future case ever needs it, rather than silently mis-scoring one.
//!
//! CATEGORIES MAP ONTO THE DETECTOR'S TYPES THROUGH [`category_map`], AND THE MAP'S
//! JOB IS TO TRANSLATE *AZURE'S* VOCABULARY INTO THE ONE THE FIXTURE IS
//! WRITTEN IN. That sentence is the whole design, and getting it wrong cost
//! this benchmark four baseline collections of a badly unfair comparison, so
//! it is written out here rather than left implicit.
//!
//! **This map used to be a transcription of the category list of Temperate's earlier Azure client
//! in Temperate -- 12 categories onto 9 types -- and that was the
//! defect.** Temperate's `PiiEntityType` has 11 cases; the detector detects 14; the
//! fixture declares 13. A category Azure returns for a class of identifier
//! *the fixture asks about* was dropped before scoring purely because
//! *Temperate's* enum had nowhere to put it, and a dropped category scores as a
//! miss. The arithmetic only runs one way: every such drop makes the
//! competitor look worse and this project look better, and four baseline
//! collections were published that way. `DRIVING_LICENCE` was the visible
//! case (Temperate has no such type at all, so all nine `*DriversLicenseNumber`
//! categories were discarded and Azure scored 3/344 on a detector it plainly
//! has), but a sweep of every category in every committed
//! `benchmark/baseline/azure-*.json` found 22 more in the same position --
//! national identification numbers, tax identification numbers, VAT numbers,
//! two passport categories and a debit-card category, all of them classes
//! this fixture declares `must_mask` values for. The old map's two
//! `NO_AZURE_EQUIVALENT` claims were false statements of fact against the
//! raw responses this repository already held: Azure returns
//! `DEValueAddedNumber`/`ITValueAddedTaxNumber`/`NLValueAddedTaxNumber` for
//! VAT numbers (it does not "return nothing at all"), and
//! `DETaxIdentificationNumber` for a German Steuer-ID (it does not only
//! "classify it as a phone number"). See `training/reports/
//! pii-detection-vs-azure-2026-08-23.md`'s ninth revision for the full measurement
//! and the size of the correction.
//!
//! So the question this map now answers for each category is **"does the
//! fixture ask for this class of identifier?"**, never "does Temperate have a
//! type for it". 38 Azure categories map onto 12 of the detector's 13 fixture
//! types. Categories that answer a question the fixture never asks -- or
//! answer it with a detector that cannot possibly be right for this corpus
//! -- are listed, with reasons, in [`deliberately_unmapped`] rather than
//! left to be inferred from their absence -- and
//! [`every_category_azure_returns_is_mapped_or_explicitly_unmapped`] fails
//! the build if a category appears in a committed raw response that is in
//! neither list. That test is the guard the old map did not have, which is
//! why the old map's drops survived four collections unnoticed.
//!
//! **`USDriversLicenseNumber` was mapped, then deliberately removed, and
//! both decisions are on the record rather than only the current one.** It
//! was first included on what it denotes, never on how well it scores: on
//! this six-language, seven-country European corpus it is a firehose --
//! 1,973 spans at a flat 0.75 confidence, most of them single digits, years
//! and ordinary words -- costing Azure far more in overall precision (0.4970
//! with it) than the three extra licence values it recovered bought in
//! recall (0.8533 with it vs. 0.8522 without). Both figures were published
//! side by side rather than one substituted for the other, because dropping
//! it on a judgement made *after* looking at the score would have been the
//! same error as the 27-category drop above, with the sign flipped. The
//! human partner has since reviewed that judgement call and decided to
//! remove it: a US driving-licence detector has nothing to find in Dutch,
//! French, German, Italian, Spanish or English prose about European
//! carriers, so its matches here are not a competitor's near-miss, they are
//! noise, and scoring them made Azure's precision look far worse than the
//! product is for a reason that has nothing to do with the product. It now
//! sits in [`deliberately_unmapped`] with that reasoning attached and the
//! honest cost recorded (`DRIVING_LICENCE` 305/344 -> 302/344, the three
//! values it alone caught), and both sets of figures still live in
//! `training/reports/pii-detection-vs-azure-2026-08-23.md` so a reader who
//! disagrees with the removal can see exactly what changed and why.
//!
//! `CREDENTIAL` is the one the detector type with no Azure category at all
//! ([`no_azure_equivalent`]), and that 0/211 was re-checked against the raw
//! responses rather than inherited: no category Azure returns, mapped or
//! unmapped, covers a single one of the fixture's 211 credential values.
//!
//! ONLY THE FOUR FIELDS THE SCORER NEEDS ARE EVER PULLED OUT OF AN AZURE
//! ENTITY -- `category`, `offset`, `length`, `confidenceScore` -- filtered
//! exactly as the `span()` method of Temperate's earlier Azure client filters them: unknown category
//! dropped, confidence below `--min-score` (default 0.5, mirroring Temperate's
//! `PII_SCORE_THRESHOLD` default) dropped. Azure's `text` field on each
//! entity is never read for scoring, only kept in the untouched raw response
//! this binary writes to disk for audit -- the same choice
//! `training/eval_fixture_azure.py` made, and safe here for the same reason:
//! every case's `text` is fixture content already committed to this
//! repository (synthetic or licensed carrier prose), so nothing new or
//! sensitive is introduced by keeping Azure's own copy of a substring of it.
//!
//! THE FIXTURE IS GROUND TRUTH; AZURE IS A COMPARATOR, NOT AN ORACLE. Every
//! case where Azure's spans disagree with the fixture's `must_mask`/
//! `must_clear` -- a miss or an over-mask -- is recorded in `disagreements`,
//! because that disagreement log is the most useful material this whole
//! benchmark produces for finding fixture errors (see the module docs on
//! `pii_detection_eval` and `check_coherence`).
//!
//! NOTHING PRINTED TO STDOUT OR STDERR CARRIES CASE TEXT OR A DETECTED VALUE
//! -- only counts, ids, types, and numbers. The output *file* does carry
//! them (fixture text and Azure's raw responses), which is what "commit
//! them" in the brief asks for; see the paragraph above on why that is safe.
//!
//! Usage:
//!     # Estimate only -- never touches the network or the credentials file.
//!     cargo run -p pii-detection-eval --bin baseline -- --date 2026-08-22
//!
//!     # The real run. --env-path names Temperate's `.env` file
//!     # and is required here; there is no default path to guess.
//!     cargo run -p pii-detection-eval --bin baseline -- --date 2026-08-22 --confirm \
//!         --env-path .env
//!
//!     # Rescore a previous run's saved raw_batches with today's scoring
//!     # rules, without calling Azure again.
//!     cargo run -p pii-detection-eval --bin baseline -- --date 2026-08-22 \
//!         --from-raw benchmark/baseline/azure-2026-08-22.json \
//!         --out benchmark/baseline/azure-2026-08-22-rescored.json

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use pii_detection_api::wire::Entity;
use pii_detection_engine::entity::EntityType;
use pii_detection_eval::score::{CaseOutcome, Metrics, Summary, scored_case};
use pii_detection_eval::{Case, check_coherence, load};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// `POST {endpoint}language/:analyze-text?api-version={API_VERSION}`, exactly
/// as Temperate's earlier Azure client calls it.
const API_VERSION: &str = "2024-11-01";

/// Azure's hard per-request document ceiling (a 400, not a truncation).
const DOCUMENTS_PER_REQUEST: usize = 5;

/// Azure's synchronous per-document character ceiling is 5,120; this binary
/// does not replicate Temperate's earlier Azure client's long-document windowing, so any
/// case at or over this (comfortably short of the true limit) fails the run
/// loudly before a single request is sent, rather than silently letting
/// Azure return that document under `results.errors` and mis-scoring it as a
/// miss.
const AZURE_SYNC_LIMIT: usize = 5_000;

/// `config/pii.php`'s `score_threshold` default in Temperate,
/// confirmed unset in the `.env` file -- so this is what production
/// actually runs with, and what makes this comparison honest: what Temperate's
/// Azure driver would actually decide to mask, not everything the raw API
/// can be made to say.
const DEFAULT_MIN_SCORE: f64 = 0.5;

/// `azure_timeout`'s default in `config/pii.php`.
const DEFAULT_TIMEOUT_SECS: u64 = 10;

/// The hard budget this binary refuses to cross without a human raising it.
/// Below this, the brief says to go ahead; above it, the decision belongs to
/// the human partner, not to this program.
const DEFAULT_MAX_COST_USD: f64 = 25.0;

/// Azure's list price for the first "Standard Text Records" volume tier
/// (0-500K text records/month) -- meter `Standard Text Records`, product
/// `Azure Language`, `tierMinimumUnits: 0`, confirmed against the public
/// Retail Prices API (`https://prices.azure.com/api/retail/prices`) on
/// 2026-08-22, and independently against the Language service's published
/// pricing page, which documents PII detection as billed on the same
/// `Standard Text Records` meter as sentiment/NER/key-phrase/language
/// detection. This is the ceiling price: those features share one 5,000
/// text-record/month free allotment, so if that allotment is unused
/// elsewhere this month the real charge is lower, never higher, than an
/// estimate computed at this rate.
const USD_PER_1K_TEXT_RECORDS: f64 = 1.00;

// There is deliberately no default `.env` path. This binary reads Azure
// credentials out of Temperate's own `.env` file at run time,
// read-only, and that repository is checked out wherever the person running
// it happens to keep it. Baking one machine's absolute path into committed
// source made `--env-path` look optional to a reader, pointed every other
// checkout at a file that does not exist there, and put one developer's home
// directory layout in the repository for no gain. `--env-path` is now
// required for a run that actually calls Azure, and unused by every run that
// does not (an estimate, or a `--from-raw` rescore).

/// Azure PII category name -> the [`EntityType`] the fixture would call that
/// class of identifier. Every category here denotes something
/// `benchmark/cases` declares `must_mask` values for; see the module docs for
/// why this is keyed off the fixture's vocabulary rather than off Temperate's
/// `PiiEntityType`, and what the old platform-shaped map cost.
///
/// The mapping is by what the category *names*, not by what it scores. A
/// span credits a `must_mask` value by covering it, whatever type it carries
/// (`pii_detection_eval::score`'s recall side is type-agnostic on purpose -- masking
/// is masking, and a leak is a leak), so the type here only decides which
/// bucket a *false positive* is attributed to in `per_type` precision. Where
/// Azure's own name and the fixture's label disagree about the same value --
/// `ESTaxIdentificationNumber` over a Spanish DNI the fixture declares
/// `NATIONAL_ID`, `USSocialSecurityNumber` over a value the fixture declares
/// `VAT_NUMBER` -- Azure's name wins here, because this table's job is to
/// say what Azure said, not to correct it.
fn category_map() -> BTreeMap<&'static str, EntityType> {
    [
        // Postal addresses, names, contact details, bank and card numbers --
        // the twelve this map carried when it was a transcription of
        // the category list of Temperate's earlier Azure client.
        ("Address", EntityType::StreetAddress),
        ("CreditCardNumber", EntityType::CreditCard),
        ("DEPassportNumber", EntityType::PassportNumber),
        ("Email", EntityType::EmailAddress),
        ("EUPassportNumber", EntityType::PassportNumber),
        ("FRPassportNumber", EntityType::PassportNumber),
        ("InternationalBankingAccountNumber", EntityType::IbanCode),
        ("IPAddress", EntityType::IpAddress),
        ("NLCitizensServiceNumber", EntityType::NationalId),
        ("Person", EntityType::Person),
        ("PhoneNumber", EntityType::PhoneNumber),
        ("USUKPassportNumber", EntityType::PassportNumber),
        // Driving licences. Nine categories, none of which Temperate's driver
        // carries, because Temperate has no `PiiEntityType::DRIVING_LICENCE`;
        // the fixture declares 344 `DRIVING_LICENCE` values, so they belong
        // here. `US`/`AU`/`JP`/`CA` are included on the same rule as the
        // rest -- they name driving licence numbers -- and not excluded for
        // firing loosely on European text; see the module docs.
        ("AUDriversLicenseNumber", EntityType::DrivingLicence),
        ("CADriversLicenseNumber", EntityType::DrivingLicence),
        ("DEDriversLicenseNumber", EntityType::DrivingLicence),
        ("EUDriversLicenseNumber", EntityType::DrivingLicence),
        ("FRDriversLicenseNumber", EntityType::DrivingLicence),
        ("ITDriversLicenseNumber", EntityType::DrivingLicence),
        ("JPDriversLicenseNumber", EntityType::DrivingLicence),
        ("UKDriversLicenseNumber", EntityType::DrivingLicence),
        // USDriversLicenseNumber is deliberately absent: see
        // deliberately_unmapped.
        // National identification numbers, including the social-security
        // and identity-card numbers that serve as one. The fixture declares
        // 334 `NATIONAL_ID` values (Dutch BSN, Spanish DNI/NIE, Belgian
        // national register number, French NIR).
        ("BENationalNumber", EntityType::NationalId),
        ("ESDNI", EntityType::NationalId),
        ("ESSocialSecurityNumber", EntityType::NationalId),
        ("EUNationalIdentificationNumber", EntityType::NationalId),
        ("EUSocialSecurityNumber", EntityType::NationalId),
        ("FRSocialSecurityNumber", EntityType::NationalId),
        ("HKIdentityCardNumber", EntityType::NationalId),
        ("USSocialSecurityNumber", EntityType::NationalId),
        // Tax identifiers. The fixture declares 197 `TAX_ID` values (German
        // Steuer-ID, Italian codice fiscale, Spanish NIF).
        ("DETaxIdentificationNumber", EntityType::TaxId),
        ("ESTaxIdentificationNumber", EntityType::TaxId),
        ("EUTaxIdentificationNumber", EntityType::TaxId),
        ("ITFiscalCode", EntityType::TaxId),
        // VAT registration numbers. The fixture declares 139 `VAT_NUMBER`
        // values.
        ("DEValueAddedNumber", EntityType::VatNumber),
        ("ITValueAddedTaxNumber", EntityType::VatNumber),
        ("NLValueAddedTaxNumber", EntityType::VatNumber),
        // Two more passport categories, and a payment card category, that
        // Temperate-shaped map happened not to list.
        ("JPPassportNumber", EntityType::PassportNumber),
        ("PLPassportNumber", EntityType::PassportNumber),
        ("EUDebitCardNumber", EntityType::CreditCard),
    ]
    .into_iter()
    .collect()
}

/// Azure categories this binary sees and deliberately does **not** score,
/// with the reason for each.
///
/// These are here so that "unmapped" is a stated decision with an argument
/// attached rather than an absence a reader has to infer -- the exact
/// failure mode that let nine driving-licence categories and thirteen
/// identifier categories fall out of the comparison unnoticed. Five of these
/// name a class the fixture asks nothing about: `must_clear` is full of
/// organisation names, dates and quantities precisely because they are *not*
/// personal data in this taxonomy, so scoring Azure on them would be the
/// mirror image of the defect this list exists to prevent -- measuring a
/// competitor against a question nobody asked it.
///
/// The sixth, `USDriversLicenseNumber`, is different in kind:
/// `DRIVING_LICENCE` *is* a type this fixture asks about, and the category
/// does denote it. It is refused anyway because a US driving-licence detector
/// has nothing to find on this fixture's six European languages and seven
/// European countries -- its 1,973 matches here are digits and ordinary
/// words, not near-misses on the right thing -- so scoring it would be the
/// mirror-image error in the other direction: making the competitor look
/// worse for a reason that has nothing to do with the product. See its own
/// entry below, the module docs, and `training/reports/
/// pii-detection-vs-azure-2026-08-23.md` for the measurement behind that call.
fn deliberately_unmapped() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "DateTime",
            "A date or time. The fixture declares no date type and lists \
             dates under must_clear.",
        ),
        (
            "Organization",
            "A company or institution name. Not personal data in this \
             taxonomy; the fixture lists organisation names under must_clear.",
        ),
        (
            "PersonType",
            "A role or job title (not a person's name, which is the separate \
             Person category this map does carry). The fixture declares no \
             such type.",
        ),
        (
            "Quantity",
            "A number with a unit. The fixture declares no such type and \
             lists quantities under must_clear.",
        ),
        ("URL", "A web address. The fixture declares no URL type."),
        (
            "USDriversLicenseNumber",
            "A US driving-licence number. Unlike the other five entries above, this one \
             does name a class the fixture asks about -- DRIVING_LICENCE -- and the \
             category was mapped for exactly that reason from 2026-09-12 through this \
             revision. It is refused now because a US detector has nothing to find in \
             this fixture's six European languages and seven European countries: at \
             2,041 cases it fires 1,973 times, flat 0.75 confidence, mostly on single \
             digits, years and ordinary words, producing 1,894 spurious spans for three \
             real licence values. Checked against the other eight *DriversLicenseNumber \
             categories: none of the three is covered by any of them, so removing this \
             one is not free -- DRIVING_LICENCE recall drops from 305/344 (0.8866) to \
             302/344 (0.8779). That is the honest price of the correction, paid rather \
             than hidden, because including the category made Azure's overall precision \
             look far worse than the product is, which is the same error as the \
             27-category drop this map's history records, with the sign flipped. Both \
             sets of figures -- with it (recall 0.8533, precision 0.4970) and without \
             (0.8522 / 0.8643) -- are in training/reports/pii-detection-vs-azure-2026-08-23.md \
             rather than only the one this map now scores.",
        ),
    ]
}

/// The detector types Azure has no equivalent for at all, and why -- reported
/// alongside the categories map so a reader of the output file never has to
/// infer their absence from silence.
///
/// **`VAT_NUMBER` and `TAX_ID` used to be on this list and are not any
/// more**: both claims were checked against the raw responses already
/// committed under `benchmark/baseline/` and both were false. Azure returns
/// `DEValueAddedNumber`, `ITValueAddedTaxNumber` and `NLValueAddedTaxNumber`
/// for VAT numbers, and `DETaxIdentificationNumber`, `EUTaxIdentification-
/// Number`, `ESTaxIdentificationNumber` and `ITFiscalCode` for tax
/// identifiers. The German Steuer-ID *is* also returned as a `PhoneNumber`
/// in some cases, which is what the original note described; what it got
/// wrong was concluding from that mislabel that there was no correct label
/// alongside it.
fn no_azure_equivalent() -> Vec<(EntityType, &'static str)> {
    vec![(
        EntityType::Credential,
        "Azure has no API-key, token or private-key category. Re-checked \
         against every category in every committed raw response, not \
         inherited: nothing Azure returns covers any of the fixture's 211 \
         CREDENTIAL values.",
    )]
}

// ---------------------------------------------------------------------------
// Cost
// ---------------------------------------------------------------------------

/// A case's length in code points -- the same unit Azure's `offset`/`length`
/// use under `stringIndexType=UnicodeCodePoint`, and the unit
/// `pii_detection_api::wire::Entity` and the scorer use throughout.
fn char_len(text: &str) -> usize {
    text.chars().count()
}

/// Azure's own definition of a text record: `ceil(chars/1000)`, minimum 1
/// per document, applied per document (never pooled across a batch).
fn text_records(chars: usize) -> usize {
    chars.div_ceil(1000).max(1)
}

#[derive(Debug, Clone, Copy, Default)]
struct Cost {
    total_characters: usize,
    text_records: usize,
    usd: f64,
}

fn estimate_cost(cases: &[Case]) -> Cost {
    let total_characters: usize = cases.iter().map(|case| char_len(&case.text)).sum();
    let text_records: usize = cases
        .iter()
        .map(|case| self::text_records(char_len(&case.text)))
        .sum();
    Cost {
        total_characters,
        text_records,
        usd: text_records as f64 / 1000.0 * USD_PER_1K_TEXT_RECORDS,
    }
}

// ---------------------------------------------------------------------------
// Wire shapes: what this binary writes, and the small slice of a previous
// run it can read back for --from-raw
// ---------------------------------------------------------------------------

/// One request's document ids and Azure's untouched JSON response, exactly
/// as sent and received. Kept for audit and for `--from-raw` rescoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawBatch {
    batch_index: usize,
    document_ids: Vec<String>,
    raw_response: Value,
}

/// The subset of a previous run's output this binary reads back for
/// `--from-raw`. Deliberately narrow: only what rescoring needs, so a wider
/// future change to [`Output`] cannot silently break `--from-raw` by
/// changing an unrelated field's shape.
#[derive(Debug, Deserialize)]
struct PriorRun {
    n_cases: usize,
    api_version: String,
    raw_batches: Vec<RawBatch>,
}

/// [`Metrics`], with recall/precision computed rather than left for a reader
/// to re-derive, and serialisable.
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
    /// See `pii_detection_eval::score`'s module docs: `must_mask` values with no
    /// declared type cannot be counted in `per_type`'s recall side. Every
    /// case in this fixture declares a type, so this is expected
    /// to be zero; reported rather than assumed so a future untyped case
    /// cannot silently shrink `per_type`'s coverage unnoticed.
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
                .map(|(wire_type, metrics)| (wire_type.clone(), MetricsDto::from(metrics)))
                .collect(),
            untyped_mask_values: summary.untyped_mask_values,
            macro_by_language: MacroAverageDto {
                recall: summary.macro_by_language.recall,
                precision: summary.macro_by_language.precision,
            },
        }
    }
}

/// One case where Azure's spans disagree with the fixture's answer key: a
/// `must_mask` value Azure left uncovered, or a `must_clear` value Azure
/// touched. `value` is a literal substring of the case's own `text`, which
/// is fixture content already committed to this repository -- see the
/// module docs on why that is safe to write here.
///
/// The fixture is ground truth; this list is not a claim that the fixture is
/// right and Azure is wrong, only a record of where the two disagree, for a
/// human to adjudicate.
#[derive(Debug, Clone, Serialize)]
struct Disagreement {
    case_id: String,
    lang: String,
    kind: DisagreementKind,
    value: String,
    /// The type the fixture declared, for a missed `must_mask` value. `None`
    /// for an over-masked `must_clear` value (which has no declared type)
    /// and for a legacy untyped `must_mask` value.
    declared_type: Option<EntityType>,
    /// The distinct types of every Azure span that touched this value, even
    /// partially. Empty for a `must_mask` value nothing touched at all.
    detected_types: Vec<EntityType>,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum DisagreementKind {
    /// A `must_mask` value with at least one uncovered non-whitespace
    /// character -- Azure would have leaked it.
    AzureMissed,
    /// A `must_clear` value Azure touched -- Azure would have over-masked
    /// it.
    AzureOverMasked,
}

fn disagreements(case_id: &str, outcome: &CaseOutcome) -> Vec<Disagreement> {
    let mut found = Vec::new();
    for verdict in &outcome.mask {
        if !verdict.found {
            found.push(Disagreement {
                case_id: case_id.to_string(),
                lang: outcome.lang.clone(),
                kind: DisagreementKind::AzureMissed,
                value: verdict.value.clone(),
                declared_type: verdict.declared,
                detected_types: verdict.types.clone(),
            });
        }
    }
    for verdict in &outcome.clear {
        if verdict.violated {
            found.push(Disagreement {
                case_id: case_id.to_string(),
                lang: outcome.lang.clone(),
                kind: DisagreementKind::AzureOverMasked,
                value: verdict.value.clone(),
                declared_type: None,
                detected_types: verdict.types.clone(),
            });
        }
    }
    found
}

#[derive(Debug, Serialize)]
struct CostDto {
    total_characters: usize,
    estimated_text_records: usize,
    estimated_cost_usd: f64,
    /// Recomputed from what was actually sent (identical to the estimate on
    /// a clean run, since every case is sent exactly once and none needs
    /// windowing). `0.0` and noted as such when `--from-raw` made no new
    /// request at all.
    actual_cost_usd: f64,
    price_usd_per_1k_text_records: f64,
    note: &'static str,
}

#[derive(Debug, Serialize)]
struct UnmappedType {
    #[serde(rename = "type")]
    entity_type: EntityType,
    reason: &'static str,
}

/// One Azure category this run saw and deliberately did not score, with the
/// reason. Written into the output file so the decision is visible to anyone
/// reading a baseline, not only to anyone reading this source file.
#[derive(Debug, Serialize)]
struct UnmappedCategory {
    category: &'static str,
    reason: &'static str,
}

/// The baseline file's shape: what the gates read. Per-type and per-language
/// recall/precision live in `summary`, not only an overall number, because a
/// per-type gate (the gates check `PERSON` on non-inferiority to Azure, among
/// others) needs its own type's figure, not the pooled one. `collected_at`
/// and `azure_model_version` are both mandatory-in-spirit: an undated
/// baseline silently becomes a comparison against a service that has since
/// changed.
#[derive(Debug, Serialize)]
struct Output {
    command: String,
    collected_at: String,
    fixture_dir: String,
    n_cases: usize,
    api_version: String,
    /// `results.modelVersion` from Azure's own response. `None` only when
    /// `--from-raw` was pointed at a run that predates this field being
    /// recorded, which no run this binary itself produces does.
    azure_model_version: Option<String>,
    min_score: f64,
    categories_map: BTreeMap<&'static str, EntityType>,
    /// Categories Azure returns that this run scored nothing for, and why.
    /// See [`deliberately_unmapped`]: stated, never silent.
    deliberately_unmapped_categories: Vec<UnmappedCategory>,
    no_azure_equivalent: Vec<UnmappedType>,
    cost: CostDto,
    summary: SummaryDto,
    disagreements: Vec<Disagreement>,
    raw_batches: Vec<RawBatch>,
}

// ---------------------------------------------------------------------------
// Talking to Azure
// ---------------------------------------------------------------------------

/// `AZURE_LANGUAGE_ENDPOINT`/`AZURE_LANGUAGE_KEY`, read out of the
/// `.env` file at `path`. Read-only; the values are held only in the
/// returned tuple and are never printed, logged, or written anywhere by this
/// binary.
fn read_env_credentials(path: &Path) -> Result<(String, String), Failure> {
    let contents = std::fs::read_to_string(path)
        .map_err(|_| Failure::UnreadableEnvFile(path.to_path_buf()))?;

    let mut endpoint = None;
    let mut key = None;
    for line in contents.lines() {
        let line = line.trim();
        if let Some(value) = line.strip_prefix("AZURE_LANGUAGE_ENDPOINT=") {
            endpoint = Some(unquote(value));
        } else if let Some(value) = line.strip_prefix("AZURE_LANGUAGE_KEY=") {
            key = Some(unquote(value));
        }
    }

    match (endpoint, key) {
        (Some(endpoint), Some(key)) if !endpoint.is_empty() && !key.is_empty() => {
            let endpoint = if endpoint.ends_with('/') {
                endpoint
            } else {
                format!("{endpoint}/")
            };
            Ok((endpoint, key))
        }
        _ => Err(Failure::MissingCredentials(path.to_path_buf())),
    }
}

fn unquote(value: &str) -> String {
    value
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .to_string()
}

/// One Azure entity's four scoring fields, and nothing else -- `text` is
/// deliberately never extracted here; see the module docs.
struct RawSpan {
    category: String,
    offset: i64,
    length: i64,
    confidence: f64,
}

fn parse_entities(document: &Value) -> Vec<RawSpan> {
    document
        .get("entities")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|entity| {
            Some(RawSpan {
                category: entity.get("category")?.as_str()?.to_string(),
                offset: entity.get("offset")?.as_i64()?,
                length: entity.get("length")?.as_i64()?,
                confidence: entity.get("confidenceScore")?.as_f64()?,
            })
        })
        .collect()
}

/// `RawSpan`s -> `Entity`s, filtered exactly as the `span()` method of Temperate's earlier Azure client
/// filters them: an unmapped category or a confidence below `min_score` is
/// dropped, never passed through.
fn azure_entities(
    spans: &[RawSpan],
    categories: &BTreeMap<&str, EntityType>,
    min_score: f64,
) -> Vec<Entity> {
    spans
        .iter()
        .filter_map(|span| {
            if span.offset < 0 || span.length <= 0 || span.confidence < min_score {
                return None;
            }
            let entity_type = *categories.get(span.category.as_str())?;
            Some(Entity {
                entity_type,
                country: None,
                start: span.offset as u32,
                end: (span.offset + span.length) as u32,
                score: span.confidence as f32,
            })
        })
        .collect()
}

/// Every way this binary can fail to produce a baseline. No variant carries
/// case text, a detected value, or an Azure response body -- only paths,
/// ids, counts and status codes, matching the discipline
/// `crates/eval/src/main.rs`'s `Failure` type documents.
#[derive(Debug)]
enum Failure {
    Usage(&'static str),
    UnreadableCases(PathBuf),
    Incoherent(usize),
    CaseTooLong {
        id: String,
        chars: usize,
    },
    BadDate(String),
    /// `--confirm` without `--env-path`. There is no default: see the note
    /// where one used to be.
    MissingEnvPath,
    UnreadableEnvFile(PathBuf),
    MissingCredentials(PathBuf),
    UnreadableRawFile(PathBuf),
    MalformedRaw {
        line: usize,
        column: usize,
    },
    RawFixtureMismatch {
        raw_n: usize,
        fixture_n: usize,
    },
    Request {
        batch_index: usize,
        status: Option<u16>,
    },
    UnreadableResponse {
        batch_index: usize,
    },
    MalformedResponse {
        batch_index: usize,
    },
    DocumentAccounting {
        batch_index: usize,
        missing: usize,
        sent: usize,
    },
    Unwritable(PathBuf),
    CostExceedsLimit {
        estimate_usd: f64,
        limit_usd: f64,
    },
}

impl Failure {
    fn report(&self) -> String {
        match self {
            Self::Usage(message) => (*message).to_string(),
            Self::UnreadableCases(path) => format!("cannot read case directory {}", path.display()),
            Self::Incoherent(count) => {
                format!(
                    "the fixture has {count} incoherent case(s); run check_coherence and fix them first"
                )
            }
            Self::CaseTooLong { id, chars } => format!(
                "case {id} is {chars} code points, at or over Azure's {AZURE_SYNC_LIMIT}-code-point \
                 synchronous limit this binary does not window around"
            ),
            Self::BadDate(value) => format!("--date {value:?} is not a YYYY-MM-DD date"),
            Self::MissingEnvPath => "--confirm needs --env-path <path>: this binary reads Azure \
                 credentials out of Temperate's `.env` file at run time and there \
                 is deliberately no default path to guess"
                .to_string(),
            Self::UnreadableEnvFile(path) => format!("cannot read {}", path.display()),
            Self::MissingCredentials(path) => format!(
                "AZURE_LANGUAGE_ENDPOINT/AZURE_LANGUAGE_KEY not found in {}",
                path.display()
            ),
            Self::UnreadableRawFile(path) => format!("cannot read {}", path.display()),
            Self::MalformedRaw { line, column } => {
                format!(
                    "the --from-raw file is not the expected shape, at line {line} column {column}"
                )
            }
            Self::RawFixtureMismatch { raw_n, fixture_n } => format!(
                "--from-raw scored {raw_n} cases but the fixture has {fixture_n}; refusing to rescore a mismatched fixture"
            ),
            Self::Request {
                batch_index,
                status,
            } => match status {
                Some(status) => format!("batch {batch_index}: Azure returned status {status}"),
                None => {
                    format!("batch {batch_index}: Azure was unreachable or the request timed out")
                }
            },
            Self::UnreadableResponse { batch_index } => {
                format!("batch {batch_index}: Azure's response body could not be read")
            }
            Self::MalformedResponse { batch_index } => {
                format!("batch {batch_index}: Azure's response was not the expected shape")
            }
            Self::DocumentAccounting {
                batch_index,
                missing,
                sent,
            } => format!(
                "batch {batch_index}: Azure accounted for only {} of {sent} sent documents",
                sent - missing
            ),
            Self::Unwritable(path) => format!("cannot write {}", path.display()),
            Self::CostExceedsLimit {
                estimate_usd,
                limit_usd,
            } => format!(
                "estimated cost ${estimate_usd:.2} exceeds the ${limit_usd:.2} limit; \
                 stopping without calling Azure. Raising the limit is a human decision."
            ),
        }
    }
}

/// `YYYY-MM-DD`, checked for shape only -- not calendar validity beyond
/// digit ranges. Good enough to catch a typo without pulling in a date
/// library for one format check.
fn valid_date(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() != 10 {
        return false;
    }
    let digits_at = |range: std::ops::Range<usize>| bytes[range].iter().all(u8::is_ascii_digit);
    bytes[4] == b'-'
        && bytes[7] == b'-'
        && digits_at(0..4)
        && digits_at(5..7)
        && digits_at(8..10)
        && matches!(
            &value[5..7],
            "01" | "02" | "03" | "04" | "05" | "06" | "07" | "08" | "09" | "10" | "11" | "12"
        )
        && matches!(value[8..10].parse::<u8>(), Ok(1..=31))
}

// ---------------------------------------------------------------------------
// Arguments
// ---------------------------------------------------------------------------

struct Arguments {
    cases: PathBuf,
    /// `None` unless `--env-path` was given. Required only by a run that
    /// actually calls Azure; see the note where the default used to be.
    env_path: Option<PathBuf>,
    date: String,
    out: Option<PathBuf>,
    min_score: f64,
    max_cost_usd: f64,
    timeout_secs: u64,
    confirm: bool,
    from_raw: Option<PathBuf>,
}

fn parse_arguments(raw: &[String]) -> Result<Arguments, Failure> {
    let mut cases = PathBuf::from("benchmark/cases");
    let mut env_path = None;
    let mut date = None;
    let mut out = None;
    let mut min_score = DEFAULT_MIN_SCORE;
    let mut max_cost_usd = DEFAULT_MAX_COST_USD;
    let mut timeout_secs = DEFAULT_TIMEOUT_SECS;
    let mut confirm = false;
    let mut from_raw = None;

    let usage = "usage: baseline --date YYYY-MM-DD [--cases <dir>] [--out <path>] \
                 [--min-score <f>] [--max-cost-usd <f>] [--timeout-secs <n>] \
                 [--from-raw <path>] [--confirm --env-path <path>]  \
                 (--env-path is required with --confirm: it names the \
                 .env file, read-only)";

    let mut rest = raw.iter();
    while let Some(argument) = rest.next() {
        match argument.as_str() {
            "--cases" => cases = PathBuf::from(rest.next().ok_or(Failure::Usage(usage))?),
            "--env-path" => {
                env_path = Some(PathBuf::from(rest.next().ok_or(Failure::Usage(usage))?))
            }
            "--date" => date = Some(rest.next().ok_or(Failure::Usage(usage))?.clone()),
            "--out" => out = Some(PathBuf::from(rest.next().ok_or(Failure::Usage(usage))?)),
            "--min-score" => {
                min_score = rest
                    .next()
                    .ok_or(Failure::Usage(usage))?
                    .parse()
                    .map_err(|_| Failure::Usage(usage))?
            }
            "--max-cost-usd" => {
                max_cost_usd = rest
                    .next()
                    .ok_or(Failure::Usage(usage))?
                    .parse()
                    .map_err(|_| Failure::Usage(usage))?
            }
            "--timeout-secs" => {
                timeout_secs = rest
                    .next()
                    .ok_or(Failure::Usage(usage))?
                    .parse()
                    .map_err(|_| Failure::Usage(usage))?
            }
            "--confirm" => confirm = true,
            "--from-raw" => {
                from_raw = Some(PathBuf::from(rest.next().ok_or(Failure::Usage(usage))?))
            }
            _ => return Err(Failure::Usage(usage)),
        }
    }

    let date = date.ok_or(Failure::Usage(usage))?;
    if !valid_date(&date) {
        return Err(Failure::BadDate(date));
    }

    Ok(Arguments {
        cases,
        env_path,
        out: out.or_else(|| {
            Some(PathBuf::from(format!(
                "benchmark/baseline/azure-{date}.json"
            )))
        }),
        date,
        min_score,
        max_cost_usd,
        timeout_secs,
        confirm,
        from_raw,
    })
}

// ---------------------------------------------------------------------------
// Sending requests
// ---------------------------------------------------------------------------

fn send_batch(
    client: &reqwest::blocking::Client,
    endpoint: &str,
    key: &str,
    batch_index: usize,
    documents: &[(&str, &str, &str)],
) -> Result<Value, Failure> {
    let payload = serde_json::json!({
        "kind": "PiiEntityRecognition",
        "parameters": {
            "modelVersion": "latest",
            "stringIndexType": "UnicodeCodePoint",
            "loggingOptOut": true,
        },
        "analysisInput": {
            "documents": documents.iter().map(|(id, language, text)| {
                serde_json::json!({"id": id, "language": language, "text": text})
            }).collect::<Vec<_>>(),
        },
    });

    let url = format!("{endpoint}language/:analyze-text?api-version={API_VERSION}");

    // A handful of retries on a rate limit or a transient server error, with
    // a short backoff, before this run fails closed. Azure only bills for a
    // document it actually analysed, so a retry after a 429/5xx does not
    // double the charge for the documents in this batch.
    let mut last_status = None;
    for attempt in 0..4 {
        if attempt > 0 {
            std::thread::sleep(Duration::from_secs(1 << attempt));
        }
        let response = client
            .post(&url)
            .header("Ocp-Apim-Subscription-Key", key)
            .json(&payload)
            .send();

        let response = match response {
            Ok(response) => response,
            Err(_) => continue,
        };

        let status = response.status();
        if status.is_success() {
            return response
                .json::<Value>()
                .map_err(|_| Failure::UnreadableResponse { batch_index });
        }
        last_status = Some(status.as_u16());
        if !(status.as_u16() == 429 || status.is_server_error()) {
            break;
        }
    }

    Err(Failure::Request {
        batch_index,
        status: last_status,
    })
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn run(arguments: &Arguments, command: String) -> Result<(PathBuf, Output), Failure> {
    let cases =
        load(&arguments.cases).map_err(|_| Failure::UnreadableCases(arguments.cases.clone()))?;

    let incoherent = check_coherence(&cases);
    if !incoherent.is_empty() {
        return Err(Failure::Incoherent(incoherent.len()));
    }

    for case in &cases {
        let chars = char_len(&case.text);
        if chars >= AZURE_SYNC_LIMIT {
            return Err(Failure::CaseTooLong {
                id: case.id.clone(),
                chars,
            });
        }
    }

    let cost = estimate_cost(&cases);
    println!(
        "fixture: {} cases, {} characters, {} estimated text records, ${:.4} estimated at ${:.2}/1K records",
        cases.len(),
        cost.total_characters,
        cost.text_records,
        cost.usd,
        USD_PER_1K_TEXT_RECORDS
    );

    if cost.usd > arguments.max_cost_usd {
        return Err(Failure::CostExceedsLimit {
            estimate_usd: cost.usd,
            limit_usd: arguments.max_cost_usd,
        });
    }

    let categories = category_map();

    let (raw_batches, api_version, from_raw) = if let Some(from_raw_path) = &arguments.from_raw {
        let contents = std::fs::read_to_string(from_raw_path)
            .map_err(|_| Failure::UnreadableRawFile(from_raw_path.clone()))?;
        let prior: PriorRun =
            serde_json::from_str(&contents).map_err(|error| Failure::MalformedRaw {
                line: error.line(),
                column: error.column(),
            })?;
        if prior.n_cases != cases.len() {
            return Err(Failure::RawFixtureMismatch {
                raw_n: prior.n_cases,
                fixture_n: cases.len(),
            });
        }
        println!(
            "rescoring {} previously saved batches from {} -- no network request made",
            prior.raw_batches.len(),
            from_raw_path.display()
        );
        (prior.raw_batches, prior.api_version, true)
    } else {
        if !arguments.confirm {
            println!(
                "estimate only -- pass --confirm to actually call Azure (cost gate: ${:.2})",
                arguments.max_cost_usd
            );
            return Err(Failure::Usage(
                "not confirmed: re-run with --confirm to spend the estimated amount above",
            ));
        }

        let env_path = arguments.env_path.as_ref().ok_or(Failure::MissingEnvPath)?;
        let (endpoint, key) = read_env_credentials(env_path)?;
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(arguments.timeout_secs))
            .build()
            .map_err(|_| Failure::Request {
                batch_index: 0,
                status: None,
            })?;

        let mut raw_batches = Vec::new();
        for (batch_index, batch) in cases.chunks(DOCUMENTS_PER_REQUEST).enumerate() {
            let documents: Vec<(&str, &str, &str)> = batch
                .iter()
                .map(|case| (case.id.as_str(), case.lang.as_str(), case.text.as_str()))
                .collect();
            let raw_response = send_batch(&client, &endpoint, &key, batch_index, &documents)?;
            raw_batches.push(RawBatch {
                batch_index,
                document_ids: batch.iter().map(|case| case.id.clone()).collect(),
                raw_response,
            });
            println!("batch {batch_index}: {} documents analysed", batch.len());
        }

        (raw_batches, API_VERSION.to_string(), false)
    };

    // entities_by_id and model_versions, accounted for exactly as
    // the `spans()` method of Temperate's earlier Azure client accounts for them: every document id sent
    // must come back in results.documents or results.errors, or this run
    // fails rather than silently reading a missing document as "found
    // nothing".
    let mut entities_by_id: BTreeMap<String, Vec<RawSpan>> = BTreeMap::new();
    let mut model_versions: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();

    for batch in &raw_batches {
        let results = batch
            .raw_response
            .get("results")
            .ok_or(Failure::MalformedResponse {
                batch_index: batch.batch_index,
            })?;

        if let Some(model_version) = results.get("modelVersion").and_then(Value::as_str) {
            model_versions.insert(model_version.to_string());
        }

        let error_ids: std::collections::BTreeSet<String> = results
            .get("errors")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|error| error.get("id").and_then(Value::as_str))
            .map(str::to_string)
            .collect();

        let documents = results.get("documents").and_then(Value::as_array).ok_or(
            Failure::MalformedResponse {
                batch_index: batch.batch_index,
            },
        )?;

        let mut answered: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for document in documents {
            let Some(id) = document.get("id").and_then(Value::as_str) else {
                return Err(Failure::MalformedResponse {
                    batch_index: batch.batch_index,
                });
            };
            answered.insert(id.to_string());
            entities_by_id.insert(id.to_string(), parse_entities(document));
        }

        let missing = batch
            .document_ids
            .iter()
            .filter(|id| !answered.contains(id.as_str()) && !error_ids.contains(id.as_str()))
            .count();
        if missing > 0 {
            return Err(Failure::DocumentAccounting {
                batch_index: batch.batch_index,
                missing,
                sent: batch.document_ids.len(),
            });
        }
    }

    if model_versions.len() > 1 {
        eprintln!(
            "warning: Azure returned {} distinct modelVersions across batches",
            model_versions.len()
        );
    }
    let azure_model_version = model_versions.into_iter().next();

    let mut case_outcomes: Vec<(String, CaseOutcome)> = Vec::with_capacity(cases.len());
    for case in &cases {
        let spans = entities_by_id
            .get(&case.id)
            .map(Vec::as_slice)
            .unwrap_or_default();
        let entities = azure_entities(spans, &categories, arguments.min_score);
        case_outcomes.push((case.id.clone(), scored_case(case, &entities)));
    }

    let case_disagreements: Vec<Disagreement> = case_outcomes
        .iter()
        .flat_map(|(case_id, outcome)| disagreements(case_id, outcome))
        .collect();

    let outcomes_only: Vec<CaseOutcome> = case_outcomes
        .into_iter()
        .map(|(_, outcome)| outcome)
        .collect();
    let summary = Summary::from(outcomes_only);

    let actual_cost_usd = if from_raw { 0.0 } else { cost.usd };
    let cost_note = if from_raw {
        "rescored from a previous run's saved raw_batches; no new Azure request was made, so \
         actual_cost_usd is 0.0"
    } else {
        "identical to the estimate: every case was sent exactly once, none needed windowing, \
         and Azure only bills for a document it actually analysed"
    };

    let no_azure_equivalent = no_azure_equivalent()
        .into_iter()
        .map(|(entity_type, reason)| UnmappedType {
            entity_type,
            reason,
        })
        .collect();

    let deliberately_unmapped_categories = deliberately_unmapped()
        .into_iter()
        .map(|(category, reason)| UnmappedCategory { category, reason })
        .collect();

    let out_path = arguments.out.clone().unwrap_or_else(|| {
        PathBuf::from(format!("benchmark/baseline/azure-{}.json", arguments.date))
    });

    let output = Output {
        command,
        collected_at: arguments.date.clone(),
        fixture_dir: arguments.cases.display().to_string(),
        n_cases: cases.len(),
        api_version,
        azure_model_version,
        min_score: arguments.min_score,
        categories_map: categories,
        deliberately_unmapped_categories,
        no_azure_equivalent,
        cost: CostDto {
            total_characters: cost.total_characters,
            estimated_text_records: cost.text_records,
            estimated_cost_usd: cost.usd,
            actual_cost_usd,
            price_usd_per_1k_text_records: USD_PER_1K_TEXT_RECORDS,
            note: cost_note,
        },
        summary: SummaryDto::from(&summary),
        disagreements: case_disagreements,
        raw_batches,
    };

    Ok((out_path, output))
}

fn main() -> ExitCode {
    let command = std::env::args().collect::<Vec<_>>().join(" ");
    let raw: Vec<String> = std::env::args().skip(1).collect();

    let arguments = match parse_arguments(&raw) {
        Ok(arguments) => arguments,
        Err(failure) => {
            eprintln!("{}", failure.report());
            return ExitCode::FAILURE;
        }
    };

    // Created before a single request is sent, not just before the final
    // write: a missing output directory is free to fail on, and finding out
    // about it only after paying Azure for 349 documents is not the kind of
    // mistake this binary gets to make twice.
    if let Some(parent) = arguments.out.as_deref().and_then(Path::parent)
        && !parent.as_os_str().is_empty()
        && std::fs::create_dir_all(parent).is_err()
    {
        eprintln!("{}", Failure::Unwritable(parent.to_path_buf()).report());
        return ExitCode::FAILURE;
    }

    match run(&arguments, command) {
        Ok((out_path, output)) => {
            let rendered = match serde_json::to_string_pretty(&output) {
                Ok(rendered) => rendered,
                Err(_) => {
                    eprintln!("cannot serialise the output");
                    return ExitCode::FAILURE;
                }
            };
            if std::fs::write(&out_path, format!("{rendered}\n")).is_err() {
                eprintln!("{}", Failure::Unwritable(out_path).report());
                return ExitCode::FAILURE;
            }
            println!("wrote {}", out_path.display());
            println!(
                "azure_model_version: {}",
                output
                    .azure_model_version
                    .as_deref()
                    .unwrap_or("(none reported)")
            );
            println!(
                "overall: recall {} precision {}",
                format_ratio(output.summary.overall.recall),
                format_ratio(output.summary.overall.precision)
            );
            println!(
                "actual cost: ${:.4} (estimated ${:.4})",
                output.cost.actual_cost_usd, output.cost.estimated_cost_usd
            );
            println!("disagreements recorded: {}", output.disagreements.len());
            ExitCode::SUCCESS
        }
        Err(failure) => {
            eprintln!("{}", failure.report());
            ExitCode::FAILURE
        }
    }
}

fn format_ratio(value: Option<f64>) -> String {
    match value {
        Some(value) => format!("{value:.4}"),
        None => "n/a".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_records_rounds_up_with_a_minimum_of_one() {
        assert_eq!(text_records(0), 1);
        assert_eq!(text_records(1), 1);
        assert_eq!(text_records(1000), 1);
        assert_eq!(text_records(1001), 2);
        assert_eq!(text_records(2308), 3);
    }

    #[test]
    fn estimate_cost_matches_the_fixture_s_own_arithmetic() {
        let cases = vec![
            case_with("short", "a".repeat(500)),
            case_with("long", "a".repeat(1200)),
        ];
        let cost = estimate_cost(&cases);
        assert_eq!(cost.total_characters, 1700);
        assert_eq!(cost.text_records, 1 + 2);
        assert!((cost.usd - 0.003).abs() < 1e-9);
    }

    #[test]
    fn category_map_covers_thirty_eight_azure_categories_onto_twelve_types() {
        let map = category_map();
        assert_eq!(map.len(), 38);
        let distinct: std::collections::BTreeSet<EntityType> = map.values().copied().collect();
        // Every type the fixture declares except CREDENTIAL, which Azure has
        // no category for -- see `no_azure_equivalent`.
        assert_eq!(distinct.len(), 12);
        assert!(!distinct.contains(&EntityType::Credential));
    }

    #[test]
    fn no_azure_equivalent_names_exactly_credential() {
        let types: Vec<EntityType> = no_azure_equivalent().into_iter().map(|(t, _)| t).collect();
        assert_eq!(types, vec![EntityType::Credential]);
    }

    #[test]
    fn a_category_is_never_both_mapped_and_deliberately_unmapped() {
        let map = category_map();
        for (category, _) in deliberately_unmapped() {
            assert!(
                !map.contains_key(category),
                "a category cannot be both scored and deliberately unscored"
            );
        }
    }

    /// Every distinct `category` in every committed baseline's raw response.
    /// Reads the files Azure was actually paid for, not a hand-kept list of
    /// what it is believed to return.
    fn categories_in_committed_baselines() -> BTreeMap<String, BTreeSet<String>> {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmark/baseline");
        let mut found: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let entries = std::fs::read_dir(&dir).expect("the baseline directory is readable");
        for entry in entries {
            let path = entry
                .expect("a baseline directory entry is readable")
                .path();
            let is_baseline = path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("azure-") && name.ends_with(".json"));
            if !is_baseline {
                continue;
            }
            let contents = std::fs::read_to_string(&path).expect("a baseline file is readable");
            let parsed: Value = serde_json::from_str(&contents).expect("a baseline file is JSON");
            let batches = parsed
                .get("raw_batches")
                .and_then(Value::as_array)
                .expect("a baseline file carries raw_batches");
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default()
                .to_string();
            let seen = found.entry(name).or_default();
            for batch in batches {
                let documents = batch
                    .pointer("/raw_response/results/documents")
                    .and_then(Value::as_array)
                    .expect("a raw batch carries results.documents");
                for document in documents {
                    for span in parse_entities(document) {
                        seen.insert(span.category);
                    }
                }
            }
        }
        assert!(
            !found.is_empty(),
            "expected at least one committed baseline to sweep"
        );
        found
    }

    /// THE GUARD THIS BINARY DID NOT HAVE, AND WHOSE ABSENCE IS WHY THE
    /// COMPARISON UNDERSTATED AZURE THROUGH FOUR BASELINE COLLECTIONS.
    ///
    /// `azure_entities` drops a category it does not recognise. That is the
    /// correct behaviour -- Temperate's own driver does the same -- but it is
    /// *silent*, and a silent drop scores exactly like a miss. Nine
    /// `*DriversLicenseNumber` categories, eight national-identifier
    /// categories, four tax categories, three VAT categories, two passport
    /// categories and a debit-card category were discarded that way, all of
    /// them classes `benchmark/cases` declares `must_mask` values for, and
    /// every one of those drops moved the published comparison in this
    /// project's own favour. Nothing failed while that was true.
    ///
    /// So: every category Azure has ever returned into a committed baseline
    /// must be either scored (`category_map`) or refused on the record
    /// (`deliberately_unmapped`). A category in neither list fails here, on
    /// the run that first collects it, instead of quietly costing the
    /// competitor recall for another four collections.
    #[test]
    fn every_category_azure_returns_is_mapped_or_explicitly_unmapped() {
        let map = category_map();
        let unmapped: BTreeSet<&str> = deliberately_unmapped()
            .into_iter()
            .map(|(category, _)| category)
            .collect();

        let mut orphans: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for (file, categories) in categories_in_committed_baselines() {
            for category in categories {
                if !map.contains_key(category.as_str()) && !unmapped.contains(category.as_str()) {
                    orphans.entry(category).or_default().insert(file.clone());
                }
            }
        }

        assert!(
            orphans.is_empty(),
            "Azure returns {} category/categories that this binary silently discards, which \
             scores identically to Azure missing the value. Add each to category_map (if the \
             fixture declares values of that class) or to deliberately_unmapped with a reason \
             (if it does not): {:?}",
            orphans.len(),
            orphans
        );
    }

    /// The other half of the same guarantee, applied only where it can be:
    /// `deliberately_unmapped` is a list of categories Azure *does* return
    /// and this binary refuses to score, so an entry naming a category Azure
    /// has never returned is an unchecked claim about Azure's behaviour --
    /// exactly what `NO_AZURE_EQUIVALENT`'s VAT and TAX_ID notes were, for
    /// long enough that a published report repeated them as fact.
    ///
    /// `category_map` is deliberately **not** held to this. Two of its
    /// entries (`FRPassportNumber`, `USUKPassportNumber`) came from Temperate's
    /// own driver and have never appeared in a committed baseline; carrying
    /// a mapping for a category this corpus happens not to elicit costs
    /// nothing and refusing one would be the drop-by-default behaviour this
    /// whole change exists to remove.
    #[test]
    fn every_deliberately_unmapped_category_is_one_azure_actually_returns() {
        let observed: BTreeSet<String> = categories_in_committed_baselines()
            .into_values()
            .flatten()
            .collect();

        let invented: Vec<&str> = deliberately_unmapped()
            .into_iter()
            .map(|(name, _)| name)
            .filter(|category| !observed.contains(*category))
            .collect();

        assert!(
            invented.is_empty(),
            "these categories are refused on the record but appear in no committed baseline, so \
             nothing checks the claim being made about them: {invented:?}"
        );
    }

    #[test]
    fn azure_entities_drops_unmapped_categories_and_low_confidence() {
        let categories = category_map();
        let spans = vec![
            RawSpan {
                category: "Person".to_string(),
                offset: 0,
                length: 4,
                confidence: 0.9,
            },
            RawSpan {
                category: "PersonType".to_string(),
                offset: 10,
                length: 5,
                confidence: 0.9,
            },
            RawSpan {
                category: "Person".to_string(),
                offset: 20,
                length: 4,
                confidence: 0.2,
            },
        ];
        let entities = azure_entities(&spans, &categories, 0.5);
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].entity_type, EntityType::Person);
        assert_eq!(entities[0].start, 0);
        assert_eq!(entities[0].end, 4);
    }

    #[test]
    fn azure_entities_drops_a_non_positive_length_or_negative_offset() {
        let categories = category_map();
        let spans = vec![
            RawSpan {
                category: "Person".to_string(),
                offset: 0,
                length: 0,
                confidence: 0.9,
            },
            RawSpan {
                category: "Person".to_string(),
                offset: -1,
                length: 4,
                confidence: 0.9,
            },
        ];
        assert!(azure_entities(&spans, &categories, 0.5).is_empty());
    }

    #[test]
    fn valid_date_accepts_and_rejects() {
        assert!(valid_date("2026-08-22"));
        assert!(!valid_date("2026-8-22"));
        assert!(!valid_date("2026/08/22"));
        assert!(!valid_date("not-a-date"));
        assert!(!valid_date("2026-13-01"));
        assert!(!valid_date("2026-08-99"));
    }

    #[test]
    fn read_env_credentials_parses_quoted_and_unquoted_values() {
        let dir = std::env::temp_dir().join("pii-detection-baseline-test-env");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join(".env");
        std::fs::write(
            &path,
            "AZURE_LANGUAGE_ENDPOINT=\"https://example.cognitiveservices.azure.com\"\nAZURE_LANGUAGE_KEY=abc123\n",
        )
        .expect("write test env file");

        let (endpoint, key) = read_env_credentials(&path).expect("credentials parse");
        assert_eq!(endpoint, "https://example.cognitiveservices.azure.com/");
        assert_eq!(key, "abc123");
    }

    #[test]
    fn read_env_credentials_fails_closed_when_a_value_is_missing() {
        let dir = std::env::temp_dir().join("pii-detection-baseline-test-env-missing");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join(".env");
        std::fs::write(&path, "AZURE_LANGUAGE_ENDPOINT=https://example.com\n")
            .expect("write test env file");

        match read_env_credentials(&path) {
            Err(Failure::MissingCredentials(_)) => {}
            other => panic!("expected MissingCredentials, got {other:?}"),
        }
    }

    use std::collections::BTreeSet;

    /// Every `Failure` variant, as an exhaustive match. Adding a variant to
    /// `Failure` without adding an arm here fails to *compile*. That is as
    /// far as this function's exhaustiveness reaches, though: it says
    /// nothing about whether the variant is actually *exercised* by the
    /// `failures` array in the test below, which is a plain array an entry
    /// can be silently deleted from without any compile-time signal. The
    /// `FAILURE_VARIANT_COUNT` assertion in that test is what catches a
    /// shrunk array; this match on its own does not.
    fn variant_name(failure: &Failure) -> &'static str {
        match failure {
            Failure::Usage(_) => "Usage",
            Failure::UnreadableCases(_) => "UnreadableCases",
            Failure::Incoherent(_) => "Incoherent",
            Failure::CaseTooLong { .. } => "CaseTooLong",
            Failure::BadDate(_) => "BadDate",
            Failure::MissingEnvPath => "MissingEnvPath",
            Failure::UnreadableEnvFile(_) => "UnreadableEnvFile",
            Failure::MissingCredentials(_) => "MissingCredentials",
            Failure::UnreadableRawFile(_) => "UnreadableRawFile",
            Failure::MalformedRaw { .. } => "MalformedRaw",
            Failure::RawFixtureMismatch { .. } => "RawFixtureMismatch",
            Failure::Request { .. } => "Request",
            Failure::UnreadableResponse { .. } => "UnreadableResponse",
            Failure::MalformedResponse { .. } => "MalformedResponse",
            Failure::DocumentAccounting { .. } => "DocumentAccounting",
            Failure::Unwritable(_) => "Unwritable",
            Failure::CostExceedsLimit { .. } => "CostExceedsLimit",
        }
    }

    #[test]
    fn every_failure_message_is_free_of_case_content() {
        // WHAT THIS USED TO ASSERT, AND WHY THAT WAS NOT ENOUGH. It rendered
        // every variant and checked the message was non-empty -- which is
        // true of a message that quotes a whole case. The property the name
        // claims is that no rendered message carries fixture content, so
        // that is what it now checks, against the real fixture's own
        // `must_mask` and `must_clear` values: the exact strings this
        // repository treats as the things a message must never contain.
        //
        // It stays a real test rather than a tautology because the
        // constructed inputs below are the worst case each variant can hold
        // (a case id, a path, counts), and because `variant_name`'s
        // exhaustive match makes a new variant a compile error here.
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmark/cases");
        let cases = load(&dir).expect("the case directory loads");
        // Values shorter than eight characters are dropped: the seed
        // fixture's `must_clear` list includes `18` and `24` (a percentage
        // and a quantity that must not be masked), which collide with any
        // message that happens to contain those digits and say nothing about
        // leaked case content either way.
        let forbidden: Vec<&str> = cases
            .iter()
            .flat_map(|case| {
                case.must_mask
                    .iter()
                    .map(|value| value.as_str())
                    .chain(case.must_clear.iter().map(String::as_str))
            })
            .filter(|value| value.len() >= 8)
            .collect();
        assert!(
            forbidden.len() > 300,
            "expected the fixture to supply plenty of strings to check against"
        );

        let failures = [
            Failure::Usage("usage: ..."),
            Failure::UnreadableCases(PathBuf::from("/tmp/x")),
            Failure::Incoherent(3),
            Failure::CaseTooLong {
                id: "domain-de-contract-1".to_string(),
                chars: 6000,
            },
            Failure::BadDate("nope".to_string()),
            Failure::MissingEnvPath,
            Failure::UnreadableEnvFile(PathBuf::from("/tmp/.env")),
            Failure::MissingCredentials(PathBuf::from("/tmp/.env")),
            Failure::UnreadableRawFile(PathBuf::from("/tmp/raw.json")),
            Failure::MalformedRaw { line: 1, column: 2 },
            Failure::RawFixtureMismatch {
                raw_n: 1,
                fixture_n: 2,
            },
            Failure::Request {
                batch_index: 0,
                status: Some(429),
            },
            Failure::UnreadableResponse { batch_index: 0 },
            Failure::MalformedResponse { batch_index: 0 },
            Failure::DocumentAccounting {
                batch_index: 0,
                missing: 1,
                sent: 5,
            },
            Failure::Unwritable(PathBuf::from("/tmp/out.json")),
            Failure::CostExceedsLimit {
                estimate_usd: 30.0,
                limit_usd: 25.0,
            },
        ];

        let covered: BTreeSet<&str> = failures.iter().map(variant_name).collect();
        assert_eq!(
            covered.len(),
            failures.len(),
            "every Failure variant must appear here exactly once"
        );
        // `variant_name`'s exhaustive match is only compile-enforced against
        // the enum -- it says nothing about the `failures` array above.
        // Deleting an entry from `failures` (say, `Failure::MissingEnvPath`)
        // shrinks `covered` and `failures` together, so the equality check
        // above stays green with a variant silently unexercised. Pinning
        // `covered.len()` to the enum's real variant count closes that gap:
        // update `FAILURE_VARIANT_COUNT` -- and add the missing entry to
        // `failures` -- whenever a variant is added to or removed from
        // `Failure`, or this test fails.
        const FAILURE_VARIANT_COUNT: usize = 17;
        assert_eq!(
            covered.len(),
            FAILURE_VARIANT_COUNT,
            "every Failure variant must be exercised by `failures`, not just appear once"
        );

        for failure in &failures {
            let name = variant_name(failure);
            let message = failure.report();
            assert!(!message.is_empty(), "{name} renders an empty message");
            for value in &forbidden {
                assert!(
                    !message.contains(value),
                    "{name}'s message carries a value out of the fixture"
                );
            }
        }
    }
    #[test]
    fn disagreements_reports_a_missed_mask_value_and_an_over_masked_clear_value() {
        use pii_detection_eval::MustMask;
        use pii_detection_eval::score::scored_case;

        let case = Case {
            id: "t-1".to_string(),
            lang: "nl".to_string(),
            len: "short".to_string(),
            source: "test".to_string(),
            carrier_id: None,
            text: "Bel Jan de Vries op Nederland vandaag".to_string(),
            must_mask: vec![MustMask::typed("Jan de Vries", EntityType::Person)],
            must_clear: vec!["Nederland".to_string()],
        };

        // No spans detected at all: the mask value is missed, and nothing
        // touches the clear value, so only one disagreement is expected.
        let outcome = scored_case(&case, &[]);
        let found = disagreements(&case.id, &outcome);
        assert_eq!(found.len(), 1);
        assert!(matches!(found[0].kind, DisagreementKind::AzureMissed));
        assert_eq!(found[0].value, "Jan de Vries");

        // A span over "Nederland": the clear value is now violated too.
        let entities = vec![Entity {
            entity_type: EntityType::StreetAddress,
            country: None,
            start: 21,
            end: 30,
            score: 0.9,
        }];
        let outcome = scored_case(&case, &entities);
        let found = disagreements(&case.id, &outcome);
        assert_eq!(found.len(), 2);
        assert!(
            found
                .iter()
                .any(|d| matches!(d.kind, DisagreementKind::AzureOverMasked))
        );
    }

    fn case_with(id: &str, text: String) -> Case {
        Case {
            id: id.to_string(),
            lang: "en".to_string(),
            len: "short".to_string(),
            source: "test".to_string(),
            carrier_id: None,
            text,
            must_mask: Vec::new(),
            must_clear: Vec::new(),
        }
    }
}
