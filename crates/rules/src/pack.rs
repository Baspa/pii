use pii_detection_engine::entity::{Country, EntityType};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::sync::OnceLock;

#[derive(Deserialize, Debug)]
pub struct Recognizer {
    pub id: String,
    pub entity_type: EntityType,
    #[serde(default)]
    pub country: Option<Country>,
    pub patterns: Vec<String>,
    pub validator: String,
    pub base_confidence: f32,
    #[serde(default)]
    pub checksum_confidence: f32,
    #[serde(default)]
    pub context_confidence: f32,
    #[serde(default)]
    pub context_terms: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    pub negative_context: Vec<String>,
    pub positive: Vec<String>,
    pub negative: Vec<String>,
    pub adversarial: Vec<String>,
}

#[derive(Deserialize, Debug)]
pub struct Pack {
    pub version: String,
    pub recognizer: Vec<Recognizer>,
}

/// Every pack, embedded at compile time.
///
/// Embedded rather than read from disk so a container cannot be started with
/// packs that differ from the ones the image was tested with. Detection
/// behaviour is part of the artifact, exactly as the model is.
const SOURCES: &[(&str, &str)] = &[
    ("universal", include_str!("../packs/universal.toml")),
    ("nl", include_str!("../packs/nl.toml")),
    ("de", include_str!("../packs/de.toml")),
    ("es", include_str!("../packs/es.toml")),
    ("it", include_str!("../packs/it.toml")),
    ("fr", include_str!("../packs/fr.toml")),
    ("be", include_str!("../packs/be.toml")),
    // Not a member state, and deliberately last: see the head of `gb.toml`
    // for why the United Kingdom has a pack and why its one recognizer
    // declares no `country`.
    ("gb", include_str!("../packs/gb.toml")),
];

pub fn all() -> &'static [Pack] {
    static PACKS: OnceLock<Vec<Pack>> = OnceLock::new();
    PACKS.get_or_init(|| {
        SOURCES
            .iter()
            .map(|(name, source)| {
                toml::from_str(source)
                    .unwrap_or_else(|error| panic!("pack {name} is malformed: {error}"))
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_universal_pack_parses() {
        let packs = all();
        assert!(!packs.is_empty(), "at least the universal pack must load");
        assert!(
            packs
                .iter()
                .any(|pack| pack.recognizer.iter().any(|r| r.id == "email"))
        );
    }

    #[test]
    fn every_validator_named_by_a_pack_exists() {
        for pack in all() {
            for recognizer in &pack.recognizer {
                assert!(
                    crate::validator::Validator::from_name(&recognizer.validator).is_some(),
                    "{} names an unknown validator: {}",
                    recognizer.id,
                    recognizer.validator
                );
            }
        }
    }

    #[test]
    fn every_recognizer_carries_all_three_fixture_kinds() {
        for pack in all() {
            for recognizer in &pack.recognizer {
                assert!(
                    !recognizer.positive.is_empty(),
                    "{}: no positive fixture",
                    recognizer.id
                );
                assert!(
                    !recognizer.negative.is_empty(),
                    "{}: no negative fixture",
                    recognizer.id
                );
                assert!(
                    !recognizer.adversarial.is_empty(),
                    "{}: no adversarial fixture",
                    recognizer.id
                );
            }
        }
    }

    #[test]
    fn recognizer_ids_are_unique_across_every_pack() {
        let mut seen = std::collections::BTreeSet::new();
        for pack in all() {
            for recognizer in &pack.recognizer {
                assert!(
                    seen.insert(recognizer.id.clone()),
                    "duplicate id: {}",
                    recognizer.id
                );
            }
        }
    }
}
