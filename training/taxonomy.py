"""Maps ai4privacy's and Wismut/nym's corpus labels onto Temperate's PII
taxonomy.

Source of truth for *which labels exist*: training/reports/corpora-profile.md,
built in the corpus profile by streaming every row of both corpora -- not the dataset
cards, which list examples rather than the full label set. Both maps below
are checked for completeness against that report by
training/tests/test_taxonomy.py, so a label present in the report but
missing here (or vice versa) fails the test suite rather than silently
drifting.

Source of truth for *what Temperate can receive*: crates/engine/src/entity.rs.
PLATFORM_TYPES below is a hand-copied mirror of that Rust enum. There is no
compiler across this language boundary -- a wrong string here means the
model predicts a label the Rust service silently can't parse
(EntityType::from_wire returns None and the entity is dropped). Keep it in
sync by hand and let the test that compares it to entity.rs's own
const guard the two from drifting apart.

The decision that shapes every STREET_ADDRESS/None call in this file: Temperate
retired a single LOCATION type (see
database/migrations/2026_08_09_000002_retire_the_location_pii_entity_type.php
in Temperate) because it conflated countries, cities and street
addresses -- masking a home address also masked "Netherlands" and destroyed
any answer that needed it. Measured against evaluation cases, that type
found 33% of street addresses and 0% of postcodes while over-masking 25
country and region names. STREET_ADDRESS's replacement definition is
narrower on purpose: "Street addresses and postal codes. Not cities or
countries, so a question about a place still gets an answer." Every label in
this file that names a street, building number or postcode maps to
STREET_ADDRESS; every label that names a city, state/region or country maps
to None. Reintroducing that conflation here would rebuild the failure the
service already paid to remove, in a place no Rust test can see.
"""

from __future__ import annotations

# The 12 types Temperate's PiiEntityType enum accepts, in the exact order and
# spelling of crates/engine/src/entity.rs's own `PLATFORM_TYPES` test
# constant.
PLATFORM_TYPES: list[str] = [
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
]


# ai4privacy/pii-masking-openpii-1.5m -- 43 labels.
#
# The task brief that produced this file said "19 labels", carried over from
# the dataset card's summary table. training/reports/corpora-profile.md
# streamed every row of both splits (1,636,375 rows) and found 43 distinct
# strings actually present in the `privacy_mask` field -- see its "Full
# label vocabulary, sorted" line. This file follows the measured 43, per
# this task's own instruction to build from the report rather than the
# card; see training/reports/taxonomy-map.md and task-2-report.md for the
# discrepancy.
AI4PRIVACY_MAP: dict[str, str | None] = {
    "ACCOUNTNUM": None,  # generic bank account number, not IBAN-formatted; no catch-all account-number type in the detector's taxonomy
    "AGE": None,  # not in the detector's taxonomy; an age alone doesn't identify anyone
    "AIRPORTCODE": None,  # not personal data; not in the detector's taxonomy
    "ALLERGIES": None,  # health data; not in the detector's taxonomy
    "AMOUNT": None,  # a monetary amount is not personal data; not in the detector's taxonomy
    "BANKNAME": None,  # organisation name, not personal data; not in the detector's taxonomy
    "BUILDINGNUM": "STREET_ADDRESS",  # street-level detail, exactly what STREET_ADDRESS covers
    "CITY": None,  # a city, not a street address -- the LOCATION conflation this taxonomy exists to avoid
    "COUNTRY": None,  # a country, not a street address -- same reason as CITY
    "CREDITCARDNUMBER": "CREDIT_CARD",
    "CURRENCY": None,  # a currency code/name is not personal data; not in the detector's taxonomy
    "DATE": None,  # not in the detector's taxonomy; no DATE_OF_BIRTH or generic date type among the 12
    "DOCTORNAME": "PERSON",  # a professional's name is still a person's name
    "DRIVERLICENSENUM": "DRIVING_LICENCE",
    "EMAIL": "EMAIL_ADDRESS",
    "GENDER": None,  # not in the detector's taxonomy; gender alone doesn't identify anyone
    "GIVENNAME": "PERSON",
    "HEIGHT": None,  # physical measurement, not identifying alone; not in the detector's taxonomy
    "HOSPITALNAME": None,  # organisation/facility name, not personal data; not in the detector's taxonomy
    "IDCARDNUM": "NATIONAL_ID",
    "IPV4": "IP_ADDRESS",
    "JOBTITLE": None,  # not personal data; not in the detector's taxonomy
    "JOBTYPE": None,  # not personal data; not in the detector's taxonomy
    "MACADDRESS": None,  # hardware identifier in a different format from IP_ADDRESS; not in the detector's taxonomy
    "MEDICATION": None,  # health data; not in the detector's taxonomy
    "ORGANISATION": None,  # organisation name, not personal data; not in the detector's taxonomy
    "PASSPORTNUM": "PASSPORT_NUMBER",
    "PASSWORD": None,  # a credential, not classic PII; not in the detector's taxonomy
    "POLICYNUM": None,  # insurance policy number; no generic policy/account-number type in the detector's taxonomy
    "SALARY": None,  # not personal data; not in the detector's taxonomy
    "SEX": None,  # not in the detector's taxonomy; sex alone doesn't identify anyone
    "SOCIALNUM": "NATIONAL_ID",  # social security / national insurance number
    "STREET": "STREET_ADDRESS",
    "SURNAME": "PERSON",
    "TAXNUM": "TAX_ID",
    "TELEPHONENUM": "PHONE_NUMBER",
    "TIME": None,  # a clock time is not personal data; not in the detector's taxonomy
    "TIMEZONE": None,  # not personal data; not in the detector's taxonomy
    "TITLE": None,  # an honorific (Mr., Dr.) is not independently identifying; folding it into PERSON risks over-masking generic titles
    "URL": None,  # not personal data by itself; not in the detector's taxonomy
    "USERNAME": None,  # an online handle, not a PERSON name; not in the detector's taxonomy
    "WEIGHT": None,  # physical measurement, not identifying alone; not in the detector's taxonomy
    "ZIPCODE": "STREET_ADDRESS",  # a postal code, explicitly covered by STREET_ADDRESS's definition
}


# Wismut/nym-pii-multilingual-data -- 40 labels (measured; see
# training/reports/corpora-profile.md "Full label vocabulary, sorted" under
# the Wismut section). This count matches the task brief.
NYM_MAP: dict[str, str | None] = {
    "ACCOUNT_NUMBER": None,  # generic bank account number, not IBAN-formatted; no catch-all account-number type in the detector's taxonomy
    "AGE": None,  # not in the detector's taxonomy; an age alone doesn't identify anyone
    "API_KEY": None,  # a credential/secret, not classic PII; not in the detector's taxonomy
    "BUILDING_NUMBER": "STREET_ADDRESS",  # street-level detail, exactly what STREET_ADDRESS covers
    "CITY": None,  # a city, not a street address -- the LOCATION conflation this taxonomy exists to avoid
    "COMPANY_NAME": None,  # organisation name, not personal data; not in the detector's taxonomy
    "COUNTRY": None,  # a country, not a street address -- same reason as CITY
    "CREDIT_DEBIT_CARD": "CREDIT_CARD",
    "CUSTOMER_ID": None,  # a business-internal identifier, not a government-issued ID; not in the detector's taxonomy
    "CVV": None,  # a card-verification code, not a card number; different format from CREDIT_CARD's PAN, not in the detector's taxonomy
    "DATE": None,  # not in the detector's taxonomy; no generic date type among the 12
    "DATE_OF_BIRTH": None,  # not in the detector's taxonomy; no date-of-birth type among the 12
    "DRIVERS_LICENSE": "DRIVING_LICENCE",
    "EMAIL": "EMAIL_ADDRESS",
    "EMPLOYEE_ID": None,  # an employer-assigned identifier, not a government-issued ID; not in the detector's taxonomy
    "FAX_NUMBER": "PHONE_NUMBER",  # a fax line is a telephone number in the same format Temperate already detects as PHONE_NUMBER
    "GENDER": None,  # not in the detector's taxonomy; gender alone doesn't identify anyone
    "GIVEN_NAME": "PERSON",
    "GOVERNMENT_ID": "NATIONAL_ID",  # a government-issued identity number, distinct from the separately-labelled PASSPORT and DRIVERS_LICENSE
    "IBAN": "IBAN_CODE",
    "LICENSE_PLATE": None,  # a vehicle registration mark, conceptually distinct from a driving licence; not in the detector's taxonomy
    "MAC_ADDRESS": None,  # hardware identifier in a different format from IP_ADDRESS; not in the detector's taxonomy
    "MEDICAL_RECORD_NUMBER": None,  # health record identifier; not in the detector's taxonomy
    "PASSPORT": "PASSPORT_NUMBER",
    "PASSWORD": None,  # a credential, not classic PII; not in the detector's taxonomy
    "PHONE": "PHONE_NUMBER",
    "PIN": None,  # a short numeric secret, not identifying alone; not in the detector's taxonomy
    "ROUTING_NUMBER": None,  # identifies a bank branch, not a person, and isn't IBAN-formatted; not in the detector's taxonomy
    "SECONDARY_ADDRESS": "STREET_ADDRESS",  # apartment/unit/suite detail that supplements a street address
    "SSN": "NATIONAL_ID",
    "STATE": None,  # a region name, the same over-masking risk the retired LOCATION type caused for country and region names
    "STREET_ADDRESS": "STREET_ADDRESS",
    "STREET_NAME": "STREET_ADDRESS",
    "SURNAME": "PERSON",
    "SWIFT_BIC": None,  # a bank/branch identifier code, not an IBAN; not in the detector's taxonomy
    "TAX_ID": "TAX_ID",
    "TIME": None,  # a clock time is not personal data; not in the detector's taxonomy
    "URL": None,  # not personal data by itself; not in the detector's taxonomy
    "USERNAME": None,  # an online handle, not a PERSON name; not in the detector's taxonomy
    "ZIP_CODE": "STREET_ADDRESS",  # a postal code, explicitly covered by STREET_ADDRESS's definition
}


_SOURCES: dict[str, dict[str, str | None]] = {
    "ai4privacy": AI4PRIVACY_MAP,
    "nym": NYM_MAP,
}


def to_platform(source: str, label: str) -> str | None:
    """Map one corpus label to its Temperate entity type, or None if it is
    deliberately dropped (never trained as that entity type).

    Raises KeyError -- rather than returning None -- for a `source` this
    module doesn't know, or a `label` absent from that source's map. A
    corpus revision that adds a label should fail the build loudly, not
    have the new label silently trained as background (O).
    """
    try:
        mapping = _SOURCES[source]
    except KeyError:
        raise KeyError(f"unknown corpus source: {source!r}") from None
    return mapping[label]
