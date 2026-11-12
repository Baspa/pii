use crate::entity::{Country, EntityType};
use crate::text::CharSpan;

/// Which detector produced a candidate. Provenance stays internal: it reaches
/// metrics as a count and never reaches the API.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Source {
    Rule { id: &'static str },
    Model,
}

/// What was true about a candidate when it was found.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Evidence {
    pub format: bool,
    /// None when the identifier defines no checksum.
    pub checksum: Option<bool>,
    pub context: bool,
    pub negative_context: bool,
    /// Phone numbers only: a legal-entity name and wording that names the company's own line (a
    /// service desk, a switchboard, a register number) stand within 150 characters of the number.
    pub company_nearby: bool,
    /// Phone numbers only: the number type or the words within 150 characters point to a person.
    pub personal_signal: bool,
    /// Phone numbers only: the text a PERSON span keeps the number masked from; today the whole text.
    pub scope: Option<CharSpan>,
}

#[derive(Clone, Debug)]
pub struct Candidate {
    pub span: CharSpan,
    pub entity_type: EntityType,
    pub country: Option<Country>,
    pub source: Source,
    pub score: f32,
    pub evidence: Evidence,
}
