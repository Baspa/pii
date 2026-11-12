use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EntityType {
    /// A machine credential: a vendor-prefixed API key, a JWT or bearer
    /// token, a PEM private key or certificate block.
    ///
    /// **The one type in this enum Temperate's own `PiiEntityType` does not
    /// carry** (see `PLATFORM_TYPES` in this module's tests, which is that
    /// enum's mirror and deliberately does not list it). That asymmetry is
    /// safe in exactly one direction and it is this one: `/analyze` returns
    /// only the types a caller *named* in its request, so a caller whose
    /// taxonomy has no `CREDENTIAL` never asks for one and never receives
    /// one. Detection is inert until a caller opts in, which is what makes
    /// shipping the recognizers ahead of the consumer safe rather than a
    /// silent contract break.
    ///
    /// It is not personal data in the GDPR sense the rest of this enum is
    /// about, and it is here anyway: a leaked API key is exploitable the
    /// moment it leaves Temperate, in a way a leaked name is not, and
    /// this service already sits in the request path of the logs, config
    /// fragments and stack traces credentials hide in.
    Credential,
    CreditCard,
    DrivingLicence,
    EmailAddress,
    IbanCode,
    IpAddress,
    NationalId,
    PassportNumber,
    Person,
    PhoneNumber,
    StreetAddress,
    TaxId,
    VatNumber,
}

impl EntityType {
    pub fn from_wire(value: &str) -> Option<Self> {
        serde_json::from_value(serde_json::Value::String(value.to_string())).ok()
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
#[allow(clippy::upper_case_acronyms)]
pub enum Country {
    AT,
    BE,
    BG,
    CY,
    CZ,
    DE,
    DK,
    EE,
    ES,
    FI,
    FR,
    GR,
    HR,
    HU,
    IE,
    IT,
    LT,
    LU,
    LV,
    MT,
    NL,
    PL,
    PT,
    RO,
    SE,
    SI,
    SK,
}

impl Country {
    pub const ALL: [Country; 27] = [
        Country::AT,
        Country::BE,
        Country::BG,
        Country::CY,
        Country::CZ,
        Country::DE,
        Country::DK,
        Country::EE,
        Country::ES,
        Country::FI,
        Country::FR,
        Country::GR,
        Country::HR,
        Country::HU,
        Country::IE,
        Country::IT,
        Country::LT,
        Country::LU,
        Country::LV,
        Country::MT,
        Country::NL,
        Country::PL,
        Country::PT,
        Country::RO,
        Country::SE,
        Country::SI,
        Country::SK,
    ];
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact strings Temperate's PiiEntityType carries after the taxonomy
    /// migration. If this list and the enum disagree, one side is masking
    /// something the other never asked for.
    ///
    /// `CREDENTIAL` is deliberately absent: Temperate's enum does not carry it,
    /// and writing it here would make this constant a wish rather than a
    /// mirror. See [`EntityType::Credential`] for why a detector-only type is
    /// safe, and `a_type_the_platform_does_not_carry_is_not_silently_added_here`
    /// below for the guard that keeps the two facts from drifting.
    const PLATFORM_TYPES: [&str; 12] = [
        "CREDIT_CARD",
        "DRIVING_LICENCE",
        "EMAIL_ADDRESS",
        "IBAN_CODE",
        "IP_ADDRESS",
        "NATIONAL_ID",
        "PASSPORT_NUMBER",
        "PERSON",
        "PHONE_NUMBER",
        "STREET_ADDRESS",
        "TAX_ID",
        "VAT_NUMBER",
    ];

    #[test]
    fn every_platform_type_round_trips() {
        for wire in PLATFORM_TYPES {
            let parsed = EntityType::from_wire(wire).expect("known type");
            assert_eq!(
                serde_json::to_value(parsed).unwrap(),
                serde_json::json!(wire)
            );
        }
    }

    /// `CREDENTIAL` round trips on the detector's wire and is *not* one of the
    /// twelve strings Temperate accepts. Both halves are asserted, because the
    /// dangerous edit is adding `"CREDENTIAL"` to `PLATFORM_TYPES` above to
    /// make some future list look complete: `taxonomy.py` mirrors that
    /// constant by hand across a language boundary with no compiler, and a
    /// thirteenth string there would tell the training pipeline the service
    /// can receive a type it cannot.
    #[test]
    fn a_type_the_platform_does_not_carry_is_not_silently_added_here() {
        assert_eq!(
            EntityType::from_wire("CREDENTIAL"),
            Some(EntityType::Credential)
        );
        assert_eq!(
            serde_json::to_value(EntityType::Credential).unwrap(),
            serde_json::json!("CREDENTIAL")
        );
        assert!(
            !PLATFORM_TYPES.contains(&"CREDENTIAL"),
            "PLATFORM_TYPES mirrors Temperate's enum, which has no CREDENTIAL"
        );
    }

    #[test]
    fn an_unknown_type_is_rejected_rather_than_guessed() {
        assert_eq!(EntityType::from_wire("BSN"), None);
        assert_eq!(EntityType::from_wire("LOCATION"), None);
    }

    #[test]
    fn there_are_twenty_seven_member_states() {
        assert_eq!(Country::ALL.len(), 27);
        assert_eq!(
            serde_json::to_value(Country::NL).unwrap(),
            serde_json::json!("NL")
        );
    }
}
