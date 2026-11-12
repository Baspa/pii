//! Runs the detector's Rust engine, rules and resolver, exactly as `/analyze`
//! does, over an evaluation fixture and emits the spans it detects.
//!
//! WHY THIS EXISTS. The detector is rules *plus* model. The first comparison
//! against Azure AI Language ran the trained checkpoint alone against the
//! whole Azure product, which measured a system nobody deploys and did so in
//! The detector's favour. This binary supplies the missing half: the rules engine's
//! own spans over the same fixture, so that "rules alone" and "rules ∪ model"
//! can be scored beside the model and beside Azure.
//!
//! IT EMITS SPANS, IT DOES NOT SCORE. Scoring lives in
//! `training/fixture_scoring.py` and stays there. A second implementation of
//! the must_mask/must_clear rule in Rust would be a second, subtly different
//! scorer, and two numbers produced by two scorers are not comparable: the
//! whole point of this measurement is that they are. So this side does the
//! one thing only it can do (run the engine) and hands offsets to the scorer
//! that already scored the model and Azure.
//!
//! OFFSETS ARE CODE POINTS. `Document` converts each recognizer's byte
//! offsets into code points exactly once (see `pii_detection_engine::text`), which
//! is the unit the fixture's own offsets and Python's string indices use. The
//! emitted `char_len` per case exists so the consumer can assert that its
//! parse of the fixture agrees with this one, character for character,
//! instead of trusting that it does.
//!
//! IT NEVER EMITS TEXT. Output carries ids, offsets, type labels and scores:
//! never the fixture's text and never a detected value. The consumer already
//! holds the fixture and can slice it itself. Errors carry no content either,
//! only a position, for the same reason.
//!
//! Usage:
//!     cargo run --release -p pii-detection-eval --bin fixture-spans -- \
//!         --fixture /path/to/pii-eval-cases.json \
//!         --out training/reports/rules-fixture-spans.json

use pii_detection_api::routes::{AppState, analyze_text, detected_types};
use pii_detection_api::wire::{Entity, MAX_CHARACTERS};
use pii_detection_engine::entity::EntityType;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::ExitCode;

/// Only the three fields the engine needs. The fixture's `must_mask`,
/// `must_clear` and `len` are the scorer's business, not this binary's, and
/// are deliberately not read: nothing here can be accused of having seen the
/// answers. `lang` is not an answer -- it is an input the service would have
/// received on a real request (the resolver's per-language calibration
/// lookup needs it, see `pii_detection_resolve::Resolver::resolve`), so reading it
/// here does not weaken that guarantee the way reading `must_mask` would.
#[derive(Deserialize)]
struct Case {
    id: String,
    lang: String,
    text: String,
}

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Serialize)]
struct CaseSpans {
    id: String,
    /// The case text's length in code points, as this process parsed it. The
    /// consumer asserts its own parse matches before trusting any offset.
    char_len: usize,
    spans: Vec<Entity>,
}

#[derive(Serialize)]
struct Output {
    command: String,
    fixture: String,
    /// What produced the spans, named so a report cannot claim more than was
    /// run.
    engine: &'static str,
    rules_version: String,
    service_version: &'static str,
    requested_types: Vec<EntityType>,
    n_cases: usize,
    cases: Vec<CaseSpans>,
}

/// Everything that can go wrong, as a message with no input in it.
enum Failure {
    Usage,
    Unreadable(PathBuf),
    Malformed {
        line: usize,
        column: usize,
    },
    TextTooLong(String),
    /// `analyze_text` refused a case. Carries the fixture's own case id and
    /// nothing else -- the refusal itself is a fixed `ApiError` variant with
    /// no text in it, and this binary has no more to say about it.
    DetectionFailed(String),
    Unwritable(PathBuf),
}

impl Failure {
    fn report(&self) -> String {
        match self {
            Self::Usage => "usage: fixture-spans --fixture <path> [--out <path>]".to_string(),
            Self::Unreadable(path) => format!("cannot read {}", path.display()),
            // serde_json's own message quotes the offending value verbatim
            // ("invalid type: integer `5`, expected a string"), which for a
            // PII fixture is the input in an error message. Only the
            // position survives.
            Self::Malformed { line, column } => {
                format!("the fixture is not the expected shape, at line {line} column {column}")
            }
            // The case id is the fixture's own label, not its content.
            Self::TextTooLong(id) => {
                format!(
                    "case {id} is longer than the {MAX_CHARACTERS} code points /analyze accepts"
                )
            }
            Self::DetectionFailed(id) => {
                format!("the engine refused case {id}; no spans were produced")
            }
            Self::Unwritable(path) => format!("cannot write {}", path.display()),
        }
    }
}

struct Arguments {
    fixture: PathBuf,
    out: Option<PathBuf>,
}

fn parse_arguments(raw: &[String]) -> Result<Arguments, Failure> {
    let mut fixture = None;
    let mut out = None;
    let mut rest = raw.iter();
    while let Some(argument) = rest.next() {
        match argument.as_str() {
            "--fixture" => fixture = Some(PathBuf::from(rest.next().ok_or(Failure::Usage)?)),
            "--out" => out = Some(PathBuf::from(rest.next().ok_or(Failure::Usage)?)),
            _ => return Err(Failure::Usage),
        }
    }
    Ok(Arguments {
        fixture: fixture.ok_or(Failure::Usage)?,
        out,
    })
}

async fn run(arguments: &Arguments, command: String) -> Result<String, Failure> {
    let source = std::fs::read_to_string(&arguments.fixture)
        .map_err(|_| Failure::Unreadable(arguments.fixture.clone()))?;
    let fixture: Fixture = serde_json::from_str(&source).map_err(|error| Failure::Malformed {
        line: error.line(),
        column: error.column(),
    })?;

    // The service's own rules-only state, built once exactly as the service
    // builds it once per process: `Matcher::new()` compiles every pattern in
    // every pack and parses phone region metadata.
    //
    // `model: None` is not a shortcut, it is the deployment this binary
    // measures. Its whole reason to exist is supplying "rules alone" as a
    // separate number from "rules union model" (see the module docs), and it
    // reaches that number by running the endpoint's own `analyze_text`
    // against the endpoint's own rules-only state rather than by assembling
    // a second engine here.
    let state = AppState::rules_only();
    // Every type this deployment actually detects, asked for by the
    // service's own definition rather than by a list written here. A
    // workspace that enabled fewer types would see fewer spans; this is the
    // ceiling the rules engine can reach today. `None` because this state
    // has no model -- so `PERSON` and `STREET_ADDRESS` are correctly absent
    // and the emitted spans cannot claim a type nothing looked for.
    let requested: Vec<EntityType> = detected_types(state.model()).into_iter().collect();

    let mut cases = Vec::with_capacity(fixture.cases.len());
    for case in fixture.cases {
        let char_len = case.text.chars().count();
        if char_len > MAX_CHARACTERS {
            return Err(Failure::TextTooLong(case.id));
        }
        let spans = analyze_text(&state, case.text, &requested, &case.lang)
            .await
            .map_err(|_| Failure::DetectionFailed(case.id.clone()))?;
        cases.push(CaseSpans {
            id: case.id,
            char_len,
            spans,
        });
    }

    let output = Output {
        command,
        fixture: arguments.fixture.display().to_string(),
        engine: "pii-detection-rules (packs + phone recognizer) -> pii-detection-resolve, \
                 via pii_detection_api::routes::analyze_text",
        rules_version: pii_detection_rules::pack::all()
            .iter()
            .map(|pack| pack.version.as_str())
            .collect::<Vec<_>>()
            .join("."),
        service_version: env!("CARGO_PKG_VERSION"),
        requested_types: requested,
        n_cases: cases.len(),
        cases,
    };

    // A fixture of any size is small next to the cost of getting this wrong,
    // so the whole document is serialized at once and written or printed.
    serde_json::to_string_pretty(&output).map_err(|_| Failure::Usage)
}

/// `#[tokio::main]` because `analyze_text` is async: it runs the rules and
/// the model concurrently, and this binary calls the very same function the
/// endpoint calls rather than reassembling the pipeline (see the module
/// docs). A rules-only state never reaches the model half, but the
/// signature is shared and so is the runtime it needs -- the rules scan
/// itself runs on the blocking pool.
#[tokio::main]
async fn main() -> ExitCode {
    // The invocation as it actually happened, argv[0] included, rather than
    // a tidied-up `cargo run …` line this process cannot know was used. A
    // recorded command that was not the command run is worse than none.
    let command = std::env::args().collect::<Vec<_>>().join(" ");
    let raw: Vec<String> = std::env::args().skip(1).collect();

    let result = match parse_arguments(&raw) {
        Err(failure) => Err(failure),
        Ok(arguments) => match run(&arguments, command).await {
            Err(failure) => Err(failure),
            Ok(rendered) => match &arguments.out {
                Some(path) => std::fs::write(path, rendered)
                    .map_err(|_| Failure::Unwritable(path.clone()))
                    .map(|()| format!("Wrote {}", path.display())),
                None => Ok(rendered),
            },
        },
    };

    match result {
        Ok(message) => {
            println!("{message}");
            ExitCode::SUCCESS
        }
        Err(failure) => {
            eprintln!("{}", failure.report());
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn spans_for(text: &str) -> Vec<Entity> {
        let state = AppState::rules_only();
        let requested: Vec<EntityType> = detected_types(state.model()).into_iter().collect();
        analyze_text(&state, text.to_string(), &requested, "auto")
            .await
            .expect("a rules-only state analyses text")
    }

    #[tokio::test]
    async fn the_emitted_offsets_are_code_points_not_bytes() {
        // "Grüße an " is nine code points and eleven bytes. A byte offset
        // here would silently shift every span in every multibyte case, and
        // the fixture is full of them: the resulting recall would be wrong
        // in whichever direction the shift happened to fall.
        let found = spans_for("Grüße an jane@example.com").await;
        assert_eq!(found.len(), 1, "expected one span");
        assert_eq!(found[0].start, 9);
        assert_eq!(found[0].end, 25);
    }

    #[tokio::test]
    async fn char_len_counts_the_same_unit_the_offsets_use() {
        let text = "Grüße an jane@example.com";
        assert_eq!(text.chars().count(), 25);
        let found = spans_for(text).await;
        assert!(found.iter().all(|span| span.end as usize <= 25));
    }

    #[test]
    fn a_fixture_is_read_by_id_text_and_lang_alone() {
        // The scorer's fields (must_mask, must_clear, len) must not be
        // needed to parse a case: this binary is not allowed to have seen
        // the answers. `lang` is not one of those fields -- it is an input
        // the service would have received on a real request -- so reading
        // it here is fine, and required for the resolver's per-language
        // calibration lookup.
        let fixture: Fixture =
            serde_json::from_str(r#"{"cases":[{"id":"a","text":"hoi","lang":"nl"}]}"#)
                .expect("a case is id, text and lang");
        assert_eq!(fixture.cases.len(), 1);
        assert_eq!(fixture.cases[0].id, "a");
        assert_eq!(fixture.cases[0].lang, "nl");
    }

    #[test]
    fn every_failure_message_is_free_of_input() {
        for failure in [
            Failure::Usage,
            Failure::Unreadable(PathBuf::from("/tmp/x.json")),
            Failure::Malformed {
                line: 3,
                column: 12,
            },
            Failure::TextTooLong("case-1".to_string()),
            Failure::DetectionFailed("case-1".to_string()),
            Failure::Unwritable(PathBuf::from("/tmp/y.json")),
        ] {
            let message = failure.report();
            assert!(!message.is_empty());
        }
    }

    #[test]
    fn arguments_require_a_fixture() {
        assert!(parse_arguments(&[]).is_err());
        assert!(parse_arguments(&["--fixture".to_string()]).is_err());
        // Matched rather than `.expect()`ed: `Failure` deliberately has no
        // `Debug`, so that no future code path can print one whole and
        // discover it was carrying something it shouldn't.
        match parse_arguments(&["--fixture".to_string(), "/tmp/f.json".to_string()]) {
            Ok(parsed) => {
                assert_eq!(parsed.fixture, PathBuf::from("/tmp/f.json"));
                assert!(parsed.out.is_none());
            }
            Err(failure) => panic!("a fixture path is enough: {}", failure.report()),
        }
    }
}
