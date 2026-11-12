//! The benchmark's case format, its loader, and the guard that keeps the
//! fixture honest.
//!
//! A benchmark case names two things about a piece of text: substrings that
//! MUST end up masked (`must_mask`, a miss is a leak) and substrings that
//! must NOT be touched (`must_clear`, a hit is over-masking). Both lists are
//! defined as literal substrings of the case's own `text` on purpose: a span
//! derived from a substring search can never disagree with the text it was
//! drawn from, where a hand-annotated offset can and eventually will.
//!
//! That guarantee only holds if the substring is actually there. A typo in a
//! `must_mask` value (one digit off, a missing accent) makes the case
//! silently unscoreable: the scorer never finds the span to look for, the
//! case contributes nothing to recall, and a real leak in that case would
//! never register as one. That is worse than an obviously broken fixture,
//! because it looks fine. [`check_coherence`] is the guard against exactly
//! that: it re-derives, for every case, whether its own annotations are
//! actually substrings of its own text, and reports every failure rather
//! than the first so a fixture author sees the whole list in one run.
//!
//! Benchmark cases must never leak into training. Every case carries
//! `source` (where it came from) and `carrier_id` (which real document, if
//! any, produced it) precisely so a training dataset builder can filter this
//! set out by those two fields rather than by guessing.
//!
//! WHY A `must_mask` ENTRY MAY CARRY A TYPE. `must_mask` started as a plain
//! list of substrings, and a substring alone says nothing about what kind of
//! identifier it is. [`score`]'s per-entity-type breakdown therefore had to
//! attribute a value to the type of whatever detected span happened to touch
//! it -- which means a value that **no** span touches at all, the worst kind
//! of miss, could not be attributed and fell out of its own per-type
//! denominator. Removing a miss from a denominator *raises* the recall it is
//! a miss of: the detector was flattered exactly where a fixture is supposed
//! to be able to fail it, and every gate the benchmark enforces is per type.
//!
//! So a `must_mask` entry is now **either** a bare JSON string (type not
//! declared -- the legacy form, and still valid) **or** an object
//! `{"value": "...", "type": "NATIONAL_ID"}`. Anything that knows what it put
//! in the text -- the `inject` binary, a hand-authored case -- declares the
//! type, and [`score::Summary::per_type`] counts declared types only, with
//! [`score::Summary::untyped_mask_values`] reporting how many values it could
//! not count so a per-type figure can never be mistaken for a complete one.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs;
use std::ops::Deref;
use std::path::{Path, PathBuf};

use pii_detection_engine::entity::EntityType;
use serde::{Deserialize, Serialize};

pub mod gates;
pub mod latency;
pub mod score;

/// One `must_mask` entry: a literal substring of the case's `text`, and
/// optionally the [`EntityType`] it is supposed to be detected as.
///
/// Serialises as the shape it was written in -- [`MustMask::Untyped`] is a
/// bare JSON string, [`MustMask::Typed`] is `{"value": ..., "type": ...}` --
/// so the legacy fixture files, whose entries are all bare strings, round
/// trip unchanged. `type` is the wire name (`"NATIONAL_ID"`, not
/// `"NationalId"`), matching `pii_detection_engine::entity::EntityType`'s own serde
/// rename and the `/analyze` response; a name the engine does not know fails
/// the whole file to load rather than being silently kept as an unrecognised
/// string, because a type nothing can ever match would count as a permanent
/// miss in its own bucket.
///
/// Declaring the type is what lets a value that no detected span touched at
/// all still be counted against the type it was supposed to be. See the
/// module docs for the defect that motivates it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MustMask {
    /// A value whose intended entity type is not declared. Per-type metrics
    /// cannot count it; overall and per-language metrics still do.
    Untyped(String),
    /// A value with its intended entity type declared.
    Typed {
        value: String,
        #[serde(rename = "type")]
        entity_type: EntityType,
    },
}

impl MustMask {
    /// The literal substring this entry names.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Untyped(value) => value,
            Self::Typed { value, .. } => value,
        }
    }

    /// The declared entity type, or `None` when this entry is the bare-string
    /// form.
    pub fn entity_type(&self) -> Option<EntityType> {
        match self {
            Self::Untyped(_) => None,
            Self::Typed { entity_type, .. } => Some(*entity_type),
        }
    }

    /// A typed entry.
    pub fn typed(value: impl Into<String>, entity_type: EntityType) -> Self {
        Self::Typed {
            value: value.into(),
            entity_type,
        }
    }
}

/// So a `&MustMask` can be handed straight to anything taking a `&str` --
/// `pii_detection_rules::validator::Validator::accepts`, `str::contains` -- without
/// every call site spelling out `.as_str()`. An entry *is* its substring;
/// the type is an annotation on it.
impl Deref for MustMask {
    type Target = str;

    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl From<&str> for MustMask {
    fn from(value: &str) -> Self {
        Self::Untyped(value.to_string())
    }
}

impl From<String> for MustMask {
    fn from(value: String) -> Self {
        Self::Untyped(value)
    }
}

/// One benchmark case: a piece of text plus what a detector run over it is
/// required to do.
///
/// `must_mask` and `must_clear` are literal substrings of `text`, checked by
/// [`check_coherence`] rather than trusted. `source` names where the case
/// came from (e.g. `"legacy"`, `"carrier"`, `"hand-authored"`) and
/// `carrier_id` names which carrier document it was drawn from, if any:
/// together they are what lets the training dataset builder exclude every
/// benchmark case, and what keeps a carrier's cases groupable when a set is
/// split.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Case {
    pub id: String,
    pub lang: String,
    pub len: String,
    pub source: String,
    #[serde(default)]
    pub carrier_id: Option<String>,
    pub text: String,
    #[serde(default)]
    pub must_mask: Vec<MustMask>,
    #[serde(default)]
    pub must_clear: Vec<String>,
}

/// Every way a case directory can fail to load.
///
/// No variant carries file content or case text: only a path, and for a
/// parse failure, a position in the file. serde_json's own parse error
/// message quotes the offending value verbatim, so that message is
/// discarded; only its line and column survive.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("cannot read case directory {0}")]
    UnreadableDir(PathBuf),
    /// One entry of the case directory could not be described by the OS at
    /// all, so nothing here can tell whether it was a case file. Carries the
    /// directory, not the entry: there is no name to carry.
    #[error("cannot read an entry of case directory {0}")]
    UnreadableDirEntry(PathBuf),
    #[error("cannot read case file {0}")]
    UnreadableFile(PathBuf),
    #[error("case file {path} is not valid JSON (line {line}, column {column})")]
    Malformed {
        path: PathBuf,
        line: usize,
        column: usize,
    },
}

/// Loads every case in `dir`.
///
/// Each `*.json` file directly under `dir` must be a JSON array of [`Case`]
/// objects; every such file is read, in a fixed (sorted-by-path) order, and
/// their cases concatenated. This does not check coherence: call
/// [`check_coherence`] on the result to find out whether the cases it
/// returns are actually scoreable.
pub fn load(dir: &Path) -> Result<Vec<Case>, LoadError> {
    let entries = fs::read_dir(dir).map_err(|_| LoadError::UnreadableDir(dir.to_path_buf()))?;

    let mut paths: Vec<PathBuf> = Vec::new();
    for entry in entries {
        // Surfaced, not skipped. An entry the OS refuses to describe is not
        // an entry this function can prove is *not* a case file, and a
        // silently smaller fixture raises every recall figure computed from
        // what is left with nothing in the output saying so -- the same
        // shape of defect as a `must_mask` value that is not in its own
        // text. Fail the load instead.
        let entry = entry.map_err(|_| LoadError::UnreadableDirEntry(dir.to_path_buf()))?;
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|ext| ext.to_str()) == Some("json") {
            paths.push(path);
        }
    }
    paths.sort();

    let mut cases = Vec::new();
    for path in paths {
        let raw = fs::read_to_string(&path).map_err(|_| LoadError::UnreadableFile(path.clone()))?;
        let mut parsed: Vec<Case> =
            serde_json::from_str(&raw).map_err(|error| LoadError::Malformed {
                path: path.clone(),
                line: error.line(),
                column: error.column(),
            })?;
        cases.append(&mut parsed);
    }

    Ok(cases)
}

/// One structural problem with one case, found without running a detector.
///
/// Never carries the offending value itself: a `must_mask` or `must_clear`
/// entry is a verbatim excerpt of the case's `text`, and excerpts of case
/// text do not belong in anything this type might reach: a report, a log
/// line, a CI annotation. `case_id` plus the index inside [`Reason`] is
/// enough for an author holding the fixture file to find the exact entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Incoherence {
    pub case_id: String,
    pub reason: Reason,
}

/// Why one case failed the coherence check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reason {
    /// `must_mask[index]` is not a substring of the case's `text`.
    MustMaskAbsentFromText { index: usize },
    /// `must_clear[index]` is not a substring of the case's `text`.
    MustClearAbsentFromText { index: usize },
    /// `must_mask[index]` also appears in `must_clear`: the case asks for
    /// the same substring to be both masked and left alone, which no
    /// detector output can ever satisfy.
    ValueInBothLists { index: usize },
    /// This case's `id` is shared with an earlier case in the slice, so the
    /// two cannot be told apart by id alone.
    DuplicateId,
}

/// Every structural problem found in `cases`.
///
/// Checks, per case: that every `must_mask` and every `must_clear` value is
/// actually a substring of that case's `text`; that no value appears in both
/// lists; and, across the whole slice, that no `id` repeats. Returns every
/// problem found rather than stopping at the first, so one run tells a
/// fixture author everything that needs fixing. An empty result means the
/// cases are scoreable, not that they are otherwise correct.
pub fn check_coherence(cases: &[Case]) -> Vec<Incoherence> {
    let mut found = Vec::new();
    let mut seen_ids: HashSet<&str> = HashSet::new();

    for case in cases {
        if !seen_ids.insert(case.id.as_str()) {
            found.push(Incoherence {
                case_id: case.id.clone(),
                reason: Reason::DuplicateId,
            });
        }

        for (index, value) in case.must_mask.iter().enumerate() {
            if !case.text.contains(value.as_str()) {
                found.push(Incoherence {
                    case_id: case.id.clone(),
                    reason: Reason::MustMaskAbsentFromText { index },
                });
            }
            if case.must_clear.iter().any(|clear| clear == value.as_str()) {
                found.push(Incoherence {
                    case_id: case.id.clone(),
                    reason: Reason::ValueInBothLists { index },
                });
            }
        }

        for (index, value) in case.must_clear.iter().enumerate() {
            if !case.text.contains(value.as_str()) {
                found.push(Incoherence {
                    case_id: case.id.clone(),
                    reason: Reason::MustClearAbsentFromText { index },
                });
            }
        }
    }

    found
}

/// One string that one case requires masked and a *different* case requires
/// left alone.
///
/// Carries the two case ids and nothing else, for the same reason
/// [`Incoherence`] does: the colliding string is a verbatim excerpt of two
/// cases' text, and excerpts of case text do not belong in a report, a log
/// line or a CI annotation. An author holding the fixture has everything
/// needed to find it from the two ids.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CrossCaseCollision {
    /// The case whose `must_mask` names the string.
    pub mask_case_id: String,
    /// The case whose `must_clear` names the same string.
    pub clear_case_id: String,
}

/// Every string this fixture requires masked in one case and left alone in
/// another.
///
/// WHY THIS IS A SEPARATE FUNCTION AND NOT PART OF [`check_coherence`].
/// [`check_coherence`] is per case, by construction: it re-derives whether a
/// case's own annotations are substrings of that case's own text. A
/// collision between two *different* cases is structurally invisible to it,
/// and, unlike everything [`check_coherence`] reports, a collision is not
/// automatically a defect. `benchmark/carriers/README.md`'s one rule about
/// surnames says a string cannot be in `must_mask` and `must_clear` **at
/// once**, and gives two synthetic names to a carrier that used one surname
/// for both a person and the device named after it. That rule is about one
/// document: masking is decided per request, so one detector run can mask a
/// string in one case and leave it in another, and asking it to is asking
/// for context sensitivity, not for the impossible.
///
/// So this reports, and a caller decides. The fixture's own decision, which
/// collisions are argued for and which are accidents, is pinned by
/// `the_fixtures_only_cross_case_collisions_are_the_argued_ones` in this
/// module's tests, which is the check that makes the *next* collision a test
/// failure rather than something a reviewer has to notice by hand.
///
/// EXACT STRINGS ONLY, DELIBERATELY. A `must_mask` value that merely
/// *contains* another case's `must_clear` value is not reported: the seed
/// fixture's `must_clear` lists include two-character values like `18` and
/// `24` (a percentage and a quantity that must not be masked), which are
/// substrings of dozens of unrelated IBANs and phone numbers in other files.
/// Reporting those would bury the four real collisions in 44 items of noise
/// and the check would be ignored, which is how a guard stops guarding.
pub fn check_cross_case_collisions(cases: &[Case]) -> Vec<CrossCaseCollision> {
    let mut masked_by: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    let mut cleared_by: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();

    for case in cases {
        for value in &case.must_mask {
            masked_by
                .entry(value.as_str())
                .or_default()
                .insert(case.id.as_str());
        }
        for value in &case.must_clear {
            cleared_by
                .entry(value.as_str())
                .or_default()
                .insert(case.id.as_str());
        }
    }

    let mut found = Vec::new();
    for (value, mask_ids) in &masked_by {
        let Some(clear_ids) = cleared_by.get(value) else {
            continue;
        };
        for mask_id in mask_ids {
            for clear_id in clear_ids {
                // A value in both lists of the *same* case is
                // `Reason::ValueInBothLists`, which check_coherence already
                // reports and which really is unsatisfiable. Not repeated
                // here.
                if mask_id == clear_id {
                    continue;
                }
                found.push(CrossCaseCollision {
                    mask_case_id: (*mask_id).to_string(),
                    clear_case_id: (*clear_id).to_string(),
                });
            }
        }
    }
    found.sort();
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case(id: &str, text: &str, must_mask: &[&str], must_clear: &[&str]) -> Case {
        Case {
            id: id.to_string(),
            lang: "nl".to_string(),
            len: "short".to_string(),
            source: "test".to_string(),
            carrier_id: None,
            text: text.to_string(),
            must_mask: must_mask.iter().map(|s| MustMask::from(*s)).collect(),
            must_clear: must_clear.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn a_must_mask_value_absent_from_its_own_text_is_incoherent() {
        // The defect this guard exists for: a typo in a must_mask value makes the
        // case silently unscoreable, which removes a leak from the score and
        // flatters the detector. A fixture that cannot be wrong is not a gate.
        let cases = vec![case("typo-1", "Het BSN is 123456782", &["123456789"], &[])];
        let found = check_coherence(&cases);
        assert_eq!(found.len(), 1);
        assert!(matches!(
            found[0].reason,
            Reason::MustMaskAbsentFromText { .. }
        ));
    }

    #[test]
    fn a_must_clear_value_absent_from_its_own_text_is_incoherent() {
        let cases = vec![case("typo-2", "Wat is de hoofdstad?", &[], &["Nederland"])];
        assert_eq!(check_coherence(&cases).len(), 1);
    }

    #[test]
    fn a_value_in_both_lists_is_incoherent() {
        // Asking for the same substring to be masked and left alone is a case
        // that can never pass, and it would sit in the fixture unnoticed.
        let cases = vec![case("both-1", "Jan de Vries", &["Jan"], &["Jan"])];
        assert_eq!(check_coherence(&cases).len(), 1);
    }

    #[test]
    fn duplicate_ids_are_incoherent() {
        let cases = vec![
            case("dup", "Het BSN is 123456782", &["123456782"], &[]),
            case("dup", "Ander geval", &[], &[]),
        ];
        assert!(!check_coherence(&cases).is_empty());
    }

    #[test]
    fn a_coherent_fixture_reports_nothing() {
        let cases = vec![case(
            "ok-1",
            "Het BSN is 123456782",
            &["123456782"],
            &["BSN"],
        )];
        assert!(check_coherence(&cases).is_empty());
    }

    fn temp_case_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("pii-detection-eval-test-{name}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create temp case dir");
        dir
    }

    #[test]
    fn load_reads_every_json_file_in_the_directory() {
        let dir = temp_case_dir("multi-file");
        fs::write(
            dir.join("a.json"),
            r#"[{"id":"a-1","lang":"nl","len":"short","source":"test","text":"hoi","must_mask":[],"must_clear":[]}]"#,
        )
        .expect("write a.json");
        fs::write(
            dir.join("b.json"),
            r#"[{"id":"b-1","lang":"en","len":"short","source":"test","text":"hi","must_mask":[],"must_clear":[]}]"#,
        )
        .expect("write b.json");

        let cases = load(&dir).expect("load succeeds");
        let mut ids: Vec<&str> = cases.iter().map(|c| c.id.as_str()).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec!["a-1", "b-1"]);
    }

    #[test]
    fn load_defaults_a_missing_carrier_id_to_none() {
        let dir = temp_case_dir("no-carrier-id");
        fs::write(
            dir.join("seed.json"),
            r#"[{"id":"x-1","lang":"nl","len":"short","source":"legacy","text":"hoi","must_mask":[],"must_clear":[]}]"#,
        )
        .expect("write seed.json");

        let cases = load(&dir).expect("load succeeds");
        assert_eq!(cases.len(), 1);
        assert!(cases[0].carrier_id.is_none());
    }

    #[test]
    fn a_bare_string_must_mask_entry_still_loads_and_declares_no_type() {
        // The 128 legacy cases are all this shape. They must keep loading
        // unchanged, and must not acquire a guessed type by loading.
        let dir = temp_case_dir("untyped-must-mask");
        fs::write(
            dir.join("legacy.json"),
            r#"[{"id":"l-1","lang":"nl","len":"short","source":"legacy","text":"Het BSN is 123456782","must_mask":["123456782"],"must_clear":[]}]"#,
        )
        .expect("write legacy.json");

        let cases = load(&dir).expect("load succeeds");
        assert_eq!(cases[0].must_mask[0].as_str(), "123456782");
        assert_eq!(cases[0].must_mask[0].entity_type(), None);
        assert!(check_coherence(&cases).is_empty());
    }

    #[test]
    fn a_typed_must_mask_entry_carries_its_declared_type() {
        let dir = temp_case_dir("typed-must-mask");
        fs::write(
            dir.join("carriers.json"),
            r#"[{"id":"c-1","lang":"nl","len":"short","source":"carrier","carrier_id":"nl-wp-x","text":"Het BSN is 123456782","must_mask":[{"value":"123456782","type":"NATIONAL_ID"}],"must_clear":[]}]"#,
        )
        .expect("write carriers.json");

        let cases = load(&dir).expect("load succeeds");
        assert_eq!(cases[0].must_mask[0].as_str(), "123456782");
        assert_eq!(
            cases[0].must_mask[0].entity_type(),
            Some(EntityType::NationalId)
        );
        assert!(check_coherence(&cases).is_empty());
    }

    #[test]
    fn a_must_mask_type_the_engine_does_not_know_fails_the_load() {
        // A type nothing can ever be detected as would sit in its own
        // per-type bucket as a permanent miss, which is a silently wrong
        // gate rather than a loud one. Fail the file instead.
        let dir = temp_case_dir("unknown-must-mask-type");
        fs::write(
            dir.join("bad.json"),
            r#"[{"id":"b-1","lang":"nl","len":"short","source":"carrier","text":"x","must_mask":[{"value":"x","type":"BSN"}],"must_clear":[]}]"#,
        )
        .expect("write bad.json");

        match load(&dir) {
            Err(LoadError::Malformed { .. }) => {}
            Err(other) => panic!("expected a Malformed error, got: {other}"),
            Ok(_) => panic!("expected an unknown entity type to fail the load"),
        }
    }

    #[test]
    fn a_typed_entry_serialises_as_an_object_and_an_untyped_one_as_a_string() {
        let typed = MustMask::typed("123456782", EntityType::NationalId);
        assert_eq!(
            serde_json::to_value(&typed).expect("serialise"),
            serde_json::json!({"value": "123456782", "type": "NATIONAL_ID"})
        );
        let untyped = MustMask::from("123456782");
        assert_eq!(
            serde_json::to_value(&untyped).expect("serialise"),
            serde_json::json!("123456782")
        );
    }

    #[test]
    fn a_typed_entry_is_still_checked_against_its_own_text() {
        // The coherence guard must not lose its grip on the typed form: a
        // typed value absent from the text is exactly as unscoreable as an
        // untyped one.
        let mut case = case("typed-typo", "Het BSN is 123456782", &[], &[]);
        case.must_mask = vec![MustMask::typed("123456789", EntityType::NationalId)];
        let found = check_coherence(&[case]);
        assert_eq!(found.len(), 1);
        assert!(matches!(
            found[0].reason,
            Reason::MustMaskAbsentFromText { .. }
        ));
    }

    #[test]
    fn load_reports_a_missing_directory() {
        let dir = std::env::temp_dir().join("pii-detection-eval-test-does-not-exist");
        let _ = fs::remove_dir_all(&dir);
        assert!(matches!(load(&dir), Err(LoadError::UnreadableDir(_))));
    }

    #[test]
    fn load_reports_malformed_json_by_position_only() {
        let dir = temp_case_dir("malformed");
        fs::write(dir.join("bad.json"), "{not valid json").expect("write bad.json");

        match load(&dir) {
            Err(LoadError::Malformed { .. }) => {}
            Err(other) => panic!("expected a Malformed error, got: {other}"),
            Ok(_) => panic!("expected malformed JSON to fail to load"),
        }
    }

    /// The guard applied to the actual seed fixture, so a future edit to
    /// `benchmark/cases/seed.json` that breaks coherence fails CI instead of
    /// silently flattering whatever detector is scored against it next.
    #[test]
    fn the_seed_fixture_is_coherent() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmark/cases");
        let cases = load(&dir).expect("the seed fixture loads");
        assert!(
            !cases.is_empty(),
            "expected the seed fixture to contain cases"
        );

        let incoherent = check_coherence(&cases);
        assert!(
            incoherent.is_empty(),
            "found {} incoherent case(s) in the seed fixture",
            incoherent.len()
        );
    }

    /// The fixture is exactly these files, of exactly these sizes.
    ///
    /// WHY A BARE COUNT IS WORTH A TEST. Every recall figure this benchmark
    /// produces is a ratio whose denominator is built from whatever cases
    /// happened to load. A case file dropped from `benchmark/cases/`, a JSON
    /// array truncated by a bad merge, a directory entry the loader skipped
    /// -- each removes `must_mask` values from the denominator, and removing
    /// misses from a denominator *raises* the score. Every other guard on
    /// this fixture checks the cases that are present; nothing until this
    /// checked that they are all still there.
    ///
    /// WHY PER FILE AND NOT ONE TOTAL. The total alone was satisfied by a
    /// swap -- one file shrinking while another grew -- which is exactly
    /// what adding a country pack does to a careless edit. Each row is the
    /// size one file is meant to be; change a row deliberately when that
    /// file deliberately changes size, and add a row when a file is added.
    /// The four original files are the 1,150-case benchmark every committed
    /// figure before the country packs was measured on.
    #[test]
    fn the_fixture_is_exactly_the_files_it_is_meant_to_be() {
        const EXPECTED: &[(&str, usize)] = &[
            ("domain.json", 19),
            ("carriers.json", 963),
            ("credential-apikey-hand.json", 12),
            ("credential-apikey.json", 65),
            ("credential-jwt-hand.json", 9),
            ("credential-jwt.json", 64),
            ("credential-pem-hand.json", 10),
            ("credential-pem.json", 64),
            ("country-be-hand.json", 7),
            ("country-be.json", 54),
            ("country-es-hand.json", 8),
            ("country-es.json", 76),
            ("country-fr-hand.json", 7),
            ("country-fr.json", 71),
            ("country-it-hand.json", 6),
            ("country-it.json", 73),
            ("driving-licence-de.json", 107),
            ("driving-licence-gb.json", 107),
            ("driving-licence-hand.json", 14),
            ("driving-licence-fr-hand.json", 10),
            ("driving-licence-fr.json", 36),
            ("driving-licence-it-hand.json", 10),
            ("driving-licence-it.json", 35),
            ("driving-licence-nl-hand.json", 10),
            ("driving-licence-nl.json", 36),
            ("negatives.json", 40),
            ("seed.json", 128),
        ];

        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmark/cases");
        let mut found: BTreeMap<String, usize> = BTreeMap::new();
        for entry in fs::read_dir(&dir).expect("the case directory reads") {
            let path = entry.expect("a directory entry").path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("json") {
                continue;
            }
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .expect("a file name")
                .to_string();
            let cases: Vec<Case> =
                serde_json::from_str(&fs::read_to_string(&path).expect("read")).expect("parse");
            found.insert(name, cases.len());
        }

        let expected: BTreeMap<String, usize> = EXPECTED
            .iter()
            .map(|(name, count)| ((*name).to_string(), *count))
            .collect();
        assert_eq!(
            found, expected,
            "the fixture changed size; a silently smaller fixture raises every recall figure              computed from it"
        );

        let total: usize = expected.values().sum();
        assert_eq!(load(&dir).expect("the fixture loads").len(), total);
    }

    /// Finds the most recently dated `training/reports/contamination-<date>.md`,
    /// skipping any file that does not match the pattern the report writer
    /// itself produces (`check_contamination.py`'s `--report-out`).
    fn latest_contamination_report_path() -> PathBuf {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../training/reports");
        let mut best: Option<(String, PathBuf)> = None;
        for entry in fs::read_dir(&dir).expect("training/reports reads") {
            let path = entry.expect("a directory entry").path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let Some(date) = name
                .strip_prefix("contamination-")
                .and_then(|rest| rest.strip_suffix(".md"))
            else {
                continue;
            };
            let is_date = date.len() == 10
                && date.bytes().enumerate().all(|(i, b)| {
                    if i == 4 || i == 7 {
                        b == b'-'
                    } else {
                        b.is_ascii_digit()
                    }
                });
            if !is_date {
                continue;
            }
            if best
                .as_ref()
                .is_none_or(|(best_date, _)| date > best_date.as_str())
            {
                best = Some((date.to_string(), path));
            }
        }
        best.expect("at least one dated contamination report is committed under training/reports/")
            .1
    }

    /// `check_contamination.py`'s report writer stamps the exact sentence
    /// `f"{len(cases)} benchmark cases and {len(carriers)} carrier documents against "`
    /// as the first paragraph's opening words -- see that file's `main()`. This
    /// pulls the number back out without a regex dependency: find the fixed
    /// anchor text and read the whitespace-delimited token immediately before it.
    fn stamped_case_count(report: &str) -> Option<usize> {
        let idx = report.find(" benchmark cases")?;
        let token = report[..idx].rsplit(char::is_whitespace).next()?;
        token.replace(',', "").parse().ok()
    }

    /// Three independent guards --
    /// `fetch_negatives.py`'s carrier check, `build_negatives.py`'s screen 2,
    /// and `check_contamination.py`'s verdict -- all compare against the
    /// benchmark *as it stands when they run*, and growing
    /// `benchmark/cases`/`benchmark/carriers` invalidates all three at once
    /// without touching a single line any of them would flag. A committed
    /// report reading `VERDICT: CLEAN` is a claim about the fixture size it
    /// was run against, not about whatever fixture exists when someone reads
    /// it later -- exactly the gap that let 22 negative-corpus documents sit
    /// alongside their own benchmark carriers for a day.
    ///
    /// This closes it structurally rather than procedurally: the fixture's
    /// live case count (via [`load`], the same loader every gate uses) must
    /// equal the case count the most recently committed contamination report
    /// stamped on itself. Unlike a runbook step, this cannot be skipped by
    /// forgetting -- it is `cargo test --workspace`, which every push and
    /// every PR already runs in CI's `test` job (`.github/workflows/ci.yml`),
    /// not merely the weekly `benchmark-gate-full` job. Adding a country pack
    /// (or any other case file) without re-running
    /// `training/check_contamination.py` and committing the fresh dated
    /// report fails this test, and a failing test fails CI on every commit,
    /// including a direct push to `main`.
    #[test]
    fn the_committed_contamination_report_matches_the_live_fixture_case_count() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmark/cases");
        let live_n_cases = load(&dir).expect("the fixture loads").len();

        let report_path = latest_contamination_report_path();
        let report = fs::read_to_string(&report_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", report_path.display()));
        let stamped_n_cases = stamped_case_count(&report).unwrap_or_else(|| {
            panic!(
                "could not find a stamped case count (\"<N> benchmark cases\") in {}",
                report_path.display()
            )
        });

        assert_eq!(
            stamped_n_cases,
            live_n_cases,
            "benchmark/cases now holds {live_n_cases} case(s) but the most recently committed \
             contamination report ({}) was stamped against {stamped_n_cases}. Growing the \
             benchmark can pull negative-corpus documents into it without touching training \
             data at all. Re-run \
             `training/check_contamination.py --data-dir data-merged --data-dir data-negatives \
             --data-dir data-positives --report-out training/reports/contamination-<today>.md`, \
             confirm VERDICT: CLEAN, and commit the fresh report before landing this change.",
            report_path.display(),
        );
    }

    #[test]
    fn a_string_masked_in_one_case_and_cleared_in_another_is_reported() {
        let cases = vec![
            case(
                "mask-side",
                "Bel 030 12345678 vandaag",
                &["030 12345678"],
                &[],
            ),
            case(
                "clear-side",
                "Onze klantenservice: 030 12345678",
                &[],
                &["030 12345678"],
            ),
        ];
        let found = check_cross_case_collisions(&cases);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].mask_case_id, "mask-side");
        assert_eq!(found[0].clear_case_id, "clear-side");
    }

    #[test]
    fn a_cross_case_collision_report_carries_no_case_content() {
        // The same discipline `Incoherence` keeps: two ids, and nothing a
        // report or a CI annotation could print that came out of a case's
        // text. This is structural -- there is no field to put it in -- and
        // this test exists so a future field cannot be added without it
        // failing.
        let cases = vec![
            case("m", "Bel 030 12345678", &["030 12345678"], &[]),
            case("c", "Service: 030 12345678", &[], &["030 12345678"]),
        ];
        let rendered = format!("{:?}", check_cross_case_collisions(&cases));
        assert!(!rendered.contains("030 12345678"), "{rendered}");
    }

    #[test]
    fn a_value_in_both_lists_of_one_case_is_not_reported_as_a_cross_case_collision() {
        // That is `Reason::ValueInBothLists`, which really is unsatisfiable
        // and which check_coherence already reports. Reporting it twice, in
        // two vocabularies, would make both reports harder to read.
        let cases = vec![case("both", "Jan de Vries", &["Jan"], &["Jan"])];
        assert!(check_cross_case_collisions(&cases).is_empty());
        assert_eq!(check_coherence(&cases).len(), 1);
    }

    #[test]
    fn a_must_clear_value_that_is_merely_a_substring_of_a_must_mask_value_is_not_a_collision() {
        // `18` is a percentage the seed fixture requires left alone, and a
        // substring of dozens of unrelated IBANs. See
        // check_cross_case_collisions's docs on why exact strings only.
        let cases = vec![
            case(
                "iban",
                "IBAN NL86BNKB4188073131",
                &["NL86BNKB4188073131"],
                &[],
            ),
            case("percent", "Een stijging van 18 procent", &[], &["18"]),
        ];
        assert!(check_cross_case_collisions(&cases).is_empty());
    }

    /// THE FOUR COLLISIONS THIS FIXTURE ARGUES FOR, AND THE ARGUMENT.
    ///
    /// Five (id, id) pairs over four strings: three business phone numbers
    /// and one passport-shaped order number. Each is a value one case
    /// requires masked and another requires left alone.
    ///
    /// They are kept, not removed, and the argument is that **the bar they
    /// set is context sensitivity, which is a bar this product has to clear
    /// anyway, and which none of them raises by so much as one inch.** Take
    /// `neg-de-business-phone-1`: a supplier's published customer-service
    /// number inside a "you can reach us on ..." sentence must not be
    /// masked. A detector that keys on the digits alone fails that negative
    /// whatever digits it carries -- giving the negative a *different*
    /// number would leave it exactly as hard, and would only hide that the
    /// same string is a customer's callback number in `phone-de-1`. The
    /// collision is not the source of the difficulty; it is the clearest
    /// available statement of it.
    ///
    /// `benchmark/carriers/README.md`'s one rule about surnames is not in
    /// tension with this: that rule is about one document, where a string
    /// genuinely cannot be masked and unmasked at once. Masking is decided
    /// per request, so two cases can disagree about one string and both be
    /// satisfiable -- see `check_cross_case_collisions`'s docs.
    ///
    /// What is honestly true, and stated rather than buried: `phone.rs` has
    /// no negative-context suppression today, so the three phone pairs are
    /// unwinnable by the current engine and show up in the benchmark as
    /// `PHONE_NUMBER` precision failing. That is the gate doing its job.
    /// The passport pair is already winnable and already won: the passport
    /// recognizer requires a context term, and `Order XR1234567` carries
    /// none.
    ///
    /// A collision not on this list fails this test. That is the point of
    /// the list: the next one gets caught rather than reasoned about after
    /// the fact.
    #[test]
    fn the_fixtures_only_cross_case_collisions_are_the_argued_ones() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmark/cases");
        let cases = load(&dir).expect("the fixture loads");

        let argued: Vec<CrossCaseCollision> = [
            // "+44 7911 123456" -- a customer's phone in a webhook payload
            // vs. a shop's published service line.
            ("domain-en-json-1", "neg-en-business-phone-1"),
            // "+39 320 123 4567" -- a customer row in a CSV vs. a joinery's
            // published service line.
            ("domain-it-csv-1", "neg-it-business-phone-1"),
            // "030 12345678" -- a callback number in a support ticket, and
            // the same number as a caller's own line, vs. a supplier's
            // published service line.
            ("phone-de-1", "neg-de-business-phone-1"),
            ("support-de-1", "neg-de-business-phone-1"),
            // "XR1234567" -- a passport number under "Passport no." vs. an
            // order number under "Order". Already winnable, already won.
            ("passport-en-1", "passport-negative"),
        ]
        .into_iter()
        .map(|(mask, clear)| CrossCaseCollision {
            mask_case_id: mask.to_string(),
            clear_case_id: clear.to_string(),
        })
        .collect();
        let mut argued = argued;
        argued.sort();

        assert_eq!(
            check_cross_case_collisions(&cases),
            argued,
            "the fixture's cross-case collisions changed; each one is a value required masked              in one case and left alone in another, and each needs an argument in this test's              docs before it is added here"
        );
    }
}
