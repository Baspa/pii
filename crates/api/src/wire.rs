use pii_detection_engine::candidate::Candidate;
use pii_detection_engine::entity::{Country, EntityType};
use serde::{Deserialize, Serialize};

/// The longest single text the service will analyse, in code points.
/// Mirrors `pii.max_characters` in Temperate, and applies per text, not per batch.
pub const MAX_CHARACTERS: usize = 125_000;

/// Temperate's extension to the 24 codes: "I do not know which language this
/// is, work it out". Nothing in this workspace works it out -- there is no
/// `pii-detection-lang` crate, so `auto` reaches the resolver as the literal string
/// and matches no calibration block. See [`ModelCoverage::Undetermined`].
pub const AUTO: &str = "auto";

/// The 24 official EU languages, plus Temperate's extension `auto`.
///
/// **This is the list the API *accepts*, and it is not the list the model was
/// *trained* on.** The two were the same field until they were separated:
/// `/analyze` validated against these 24 and said nothing more, so a request
/// for one of the eighteen the model has never seen was answered exactly like
/// a request for one of the six it has. The trained list is not here and must
/// never be written here -- it belongs to the artifact's own manifest
/// (`supported_languages`), for the same reason `max_length`, `stride` and the
/// label count do: a retrained model with different coverage is a new artifact
/// directory, not an edit to this file. See [`model_coverage`] and
/// [`LanguageCapabilities`].
pub const ACCEPTED_LANGUAGES: [&str; 24] = [
    "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "ga", "hr", "hu", "it", "lt", "lv",
    "mt", "nl", "pl", "pt", "ro", "sk", "sl", "sv",
];

#[derive(Deserialize)]
pub struct AnalyzeRequest {
    pub texts: Vec<String>,
    pub entities: Vec<String>,
    #[serde(default = "default_language")]
    pub language: String,
}

fn default_language() -> String {
    AUTO.to_string()
}

#[derive(Serialize)]
pub struct Entity {
    pub entity_type: EntityType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<Country>,
    pub start: u32,
    pub end: u32,
    pub score: f32,
}

impl From<&Candidate> for Entity {
    fn from(candidate: &Candidate) -> Self {
        Self {
            entity_type: candidate.entity_type,
            country: candidate.country,
            start: candidate.span.start,
            end: candidate.span.end,
            score: candidate.score,
        }
    }
}

/// Whether the model behind an answer was trained on the language that answer
/// was produced for.
///
/// The rules engine is language-independent -- an IBAN, a card, an email, an
/// IP and every country pack key on format and checksum, not on prose -- so
/// these four values say nothing about the structured types. What they are
/// about is `PERSON` and `STREET_ADDRESS`, which come only from the model, and
/// which an untrained language produces *confidently* rather than not at all:
/// a reviewer measured Maltese, absent from training entirely, scoring
/// 0.54-0.58 against a 0.50 threshold. A caller could not previously see the
/// difference, and a high score is exactly the wrong signal to leave alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelCoverage {
    /// An explicit code the artifact's manifest lists in `supported_languages`.
    /// Every type this deployment advertises is backed by training.
    Trained,
    /// An explicit code the API accepts and the artifact was never trained on.
    /// The rules ran at full quality; the model ran on a language it has not
    /// seen, so `PERSON` and `STREET_ADDRESS` are whatever the multilingual
    /// base transfers -- not nothing, and not promised.
    Untrained,
    /// `auto`. The service performs no language identification, so it cannot
    /// claim the text was in a trained language and does not pretend to: this
    /// is `Untrained`'s honest twin, not a third quality level, and it carries
    /// the same unverified type list.
    Undetermined,
    /// A rules-only deployment. There is no model, so nothing is unverified --
    /// `/capabilities` does not advertise `PERSON` or `STREET_ADDRESS` and
    /// `/analyze` refuses a request for either.
    RulesOnly,
}

impl ModelCoverage {
    /// Every variant, in `index` order. Used by this module's own tests to
    /// check the wire name and the metric label agree for all of them, which
    /// is the pair that would otherwise be free to drift.
    pub const ALL: [ModelCoverage; 4] = [
        ModelCoverage::Trained,
        ModelCoverage::Untrained,
        ModelCoverage::Undetermined,
        ModelCoverage::RulesOnly,
    ];

    /// The metric label for this regime: a fixed string, one of four, chosen
    /// here and never derived from caller input.
    ///
    /// Exhaustive by construction, the same discipline as
    /// [`crate::error::ApiError::reason`]: a fifth variant fails to compile
    /// until it is named here. It must serialize to the same string it
    /// reports on the wire -- `label_matches_the_wire_name` holds that, so a
    /// dashboard and a response body cannot end up using two vocabularies for
    /// one fact.
    pub fn as_label(self) -> &'static str {
        match self {
            Self::Trained => "trained",
            Self::Untrained => "untrained",
            Self::Undetermined => "undetermined",
            Self::RulesOnly => "rules_only",
        }
    }

    /// This variant's position in [`ModelCoverage::ALL`]. Exists only so
    /// `all_lists_every_variant` can prove that constant is complete: adding a
    /// variant forces an arm here, and the new arm's index then fails that
    /// test until `ALL` carries it too.
    #[cfg(test)]
    fn index(self) -> usize {
        match self {
            Self::Trained => 0,
            Self::Untrained => 1,
            Self::Undetermined => 2,
            Self::RulesOnly => 3,
        }
    }

    /// Whether this regime leaves the model-only types unbacked by training.
    fn leaves_types_unverified(self) -> bool {
        match self {
            Self::Untrained | Self::Undetermined => true,
            Self::Trained | Self::RulesOnly => false,
        }
    }
}

/// The regime one `/analyze` call ran under, returned beside the spans.
///
/// **This is the signal, and it is in the body rather than a header on
/// purpose.** The requirement is that a caller who never reads the
/// documentation cannot mistake a Polish answer for a Dutch one, and a header
/// is exactly what such a caller ignores. A body field cannot be ignored by a
/// client that parses the body at all.
#[derive(Serialize)]
pub struct LanguageRegime {
    /// The code the caller sent, echoed back: one of [`ACCEPTED_LANGUAGES`] or
    /// [`AUTO`], because validation ran before this was built. Safe on the
    /// wire for the same reason `ApiError::UnsupportedLanguage` echoes a code
    /// -- it is a parameter the caller chose, not content, and it goes back
    /// only to them.
    pub requested: String,
    pub model_coverage: ModelCoverage,
    /// The types in this response that are *not* backed by training for
    /// `requested` -- empty unless `model_coverage` says otherwise.
    ///
    /// Derived from the artifact (the model-sourced types minus the ones a
    /// rule pack also detects), never listed in Rust, so an artifact whose
    /// `labels.json` maps a different set names a different set here.
    ///
    /// Deliberately *not* intersected with the types the caller requested:
    /// this describes what the deployment can and cannot vouch for in this
    /// language, which is a fact a caller can cache, rather than a per-request
    /// answer that changes shape with the entity list.
    pub unverified_entity_types: Vec<EntityType>,
}

/// `/analyze`'s body.
///
/// **Wire change, service `0.1.0` onwards.** This endpoint previously returned
/// the bare `[[{entity}, ...], ...]` of the design, with nothing
/// anywhere in the response saying which language regime produced it. The
/// spans now live under `results`, unchanged in shape and order, beside the
/// [`LanguageRegime`] that produced them. The break is the point: a caller
/// that goes on indexing the top level as an array fails loudly instead of
/// reading an untrained answer as a trained one.
#[derive(Serialize)]
pub struct AnalyzeResponse {
    pub language: LanguageRegime,
    /// One list per input text, in order -- the design, unchanged.
    pub results: Vec<Vec<Entity>>,
}

/// `/capabilities`' `languages` block.
///
/// **Wire change, service `0.1.0` onwards.** This field was the flat array of
/// [`ACCEPTED_LANGUAGES`], which answered "which codes will you accept" and
/// left "which of them is the model actually good at" unanswerable from the
/// endpoint that exists to answer capability questions.
#[derive(Serialize)]
pub struct LanguageCapabilities {
    /// The 24 codes `/analyze` accepts. `auto` is a mode rather than a
    /// language and is not listed here; it is always accepted.
    pub accepted: [&'static str; 24],
    /// The codes the loaded artifact's manifest declares it was trained on,
    /// read from `supported_languages` -- empty on a rules-only deployment,
    /// which has no model to have been trained.
    pub model_trained: Vec<String>,
    /// The types only the model produces, so a caller can read off what a code
    /// in `accepted` but not in `model_trained` costs them: those types, and
    /// nothing else.
    pub model_only_entity_types: Vec<EntityType>,
}

/// Which regime a request for `language` runs under on a deployment whose
/// model declares `model_languages` (`None` for a rules-only deployment).
///
/// A pure function over the manifest's own list, so the decision can be tested
/// without an 88 MB artifact and so no second copy of the rule can appear in
/// the handler.
pub fn model_coverage(model_languages: Option<&[String]>, language: &str) -> ModelCoverage {
    match model_languages {
        None => ModelCoverage::RulesOnly,
        Some(_) if language == AUTO => ModelCoverage::Undetermined,
        Some(languages) => match languages.iter().any(|trained| trained == language) {
            true => ModelCoverage::Trained,
            false => ModelCoverage::Untrained,
        },
    }
}

/// The types to name in [`LanguageRegime::unverified_entity_types`] for a
/// given regime: the model-only set when the model was not trained for this
/// request's language, and nothing at all when it was.
pub fn unverified_types(coverage: ModelCoverage, model_only: &[EntityType]) -> Vec<EntityType> {
    match coverage.leaves_types_unverified() {
        true => model_only.to_vec(),
        false => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn six() -> Vec<String> {
        ["en", "nl", "de", "fr", "es", "it"]
            .iter()
            .map(|code| code.to_string())
            .collect()
    }

    #[test]
    fn all_lists_every_variant() {
        for (position, coverage) in ModelCoverage::ALL.iter().enumerate() {
            assert_eq!(coverage.index(), position);
        }
    }

    #[test]
    fn label_matches_the_wire_name() {
        // A metric label and a response field naming the same fact with two
        // different strings is a dashboard that cannot be joined to a
        // reproduction.
        for coverage in ModelCoverage::ALL {
            let serialized = serde_json::to_value(coverage).expect("a string");
            assert_eq!(serialized, serde_json::json!(coverage.as_label()));
        }
    }

    #[test]
    fn a_trained_language_is_trained_and_an_accepted_one_is_not() {
        let six = six();
        assert_eq!(
            model_coverage(Some(&six), "nl"),
            ModelCoverage::Trained,
            "nl is in the manifest list"
        );
        // Every one of the eighteen the API accepts and the model never saw.
        for code in ACCEPTED_LANGUAGES {
            let expected = match six.iter().any(|trained| trained == code) {
                true => ModelCoverage::Trained,
                false => ModelCoverage::Untrained,
            };
            assert_eq!(model_coverage(Some(&six), code), expected, "code {code}");
        }
        assert_eq!(model_coverage(Some(&six), "pl"), ModelCoverage::Untrained);
        assert_eq!(model_coverage(Some(&six), "mt"), ModelCoverage::Untrained);
    }

    #[test]
    fn auto_is_undetermined_rather_than_trained() {
        // The service resolves no language, so `auto` cannot be reported as
        // trained even on a deployment whose model covers six languages --
        // that would be a claim nothing checked.
        assert_eq!(
            model_coverage(Some(&six()), AUTO),
            ModelCoverage::Undetermined
        );
    }

    #[test]
    fn a_deployment_with_no_model_is_rules_only_whatever_was_asked() {
        for code in ["nl", "pl", AUTO] {
            assert_eq!(model_coverage(None, code), ModelCoverage::RulesOnly);
        }
    }

    #[test]
    fn the_manifest_list_is_the_only_thing_that_decides() {
        // The mutation this catches: a `model_coverage` that consults a list
        // written in Rust instead of the one it was handed. A manifest
        // declaring Polish makes Polish trained, and one declaring nothing
        // makes every code untrained -- neither is true of any constant in
        // this crate.
        let polish = vec!["pl".to_string()];
        assert_eq!(model_coverage(Some(&polish), "pl"), ModelCoverage::Trained);
        assert_eq!(
            model_coverage(Some(&polish), "nl"),
            ModelCoverage::Untrained
        );
        assert_eq!(model_coverage(Some(&[]), "nl"), ModelCoverage::Untrained);
    }

    #[test]
    fn only_an_unbacked_regime_names_unverified_types() {
        let model_only = [EntityType::Person, EntityType::StreetAddress];
        assert!(unverified_types(ModelCoverage::Trained, &model_only).is_empty());
        assert!(unverified_types(ModelCoverage::RulesOnly, &model_only).is_empty());
        assert_eq!(
            unverified_types(ModelCoverage::Untrained, &model_only),
            model_only.to_vec()
        );
        assert_eq!(
            unverified_types(ModelCoverage::Undetermined, &model_only),
            model_only.to_vec()
        );
    }
}
