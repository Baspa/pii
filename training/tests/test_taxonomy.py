"""Tests for training/taxonomy.py -- the map from ai4privacy's and nym's
corpus labels onto Temperate's 12-type PII taxonomy.

The PLATFORM_TYPES test guards a cross-language contract with no compiler
behind it: get one string wrong and the model predicts a label the Rust
service can't parse, and nothing fails loudly until production.

The two "map is complete" tests replace a test in the original task brief,
`test_nothing_is_mapped_by_omission`, which read:

    assert all(v is not None or k in mapping for k, v in mapping.items())

`k` is drawn from `mapping.items()`, so `k in mapping` is always true and the
test could never fail -- a tautology, not a completeness check. What it was
trying to say -- "the map must cover every label the corpus actually has" --
can only be checked against something outside the map itself: the label
vocabularies training/reports/corpora-profile.md measured by streaming every
row of both corpora in the corpus profile. So these tests read that report and diff it
against AI4PRIVACY_MAP / NYM_MAP, in both directions: a label the report has
and the map lacks (an omission), and a label the map has and the report
doesn't (a stale or invented entry). Reading the report keeps the test
honest against the corpus profile's measurement instead of a hand-copied list that could
silently drift from it.
"""

import re
from pathlib import Path

import pytest

from taxonomy import AI4PRIVACY_MAP, PLATFORM_TYPES, NYM_MAP, to_platform

# Copied from crates/engine/src/entity.rs. A mismatch here means the
# model predicts a label the service cannot receive.
EXPECTED = [
    "CREDIT_CARD", "DRIVING_LICENCE", "EMAIL_ADDRESS", "IBAN_CODE",
    "IP_ADDRESS", "NATIONAL_ID", "PASSPORT_NUMBER", "PERSON",
    "PHONE_NUMBER", "STREET_ADDRESS", "TAX_ID", "VAT_NUMBER",
]

REPORT_PATH = Path(__file__).resolve().parent.parent / "reports" / "corpora-profile.md"


def _measured_vocabulary(section_heading: str) -> set[str]:
    """Read the full, sorted label vocabulary the corpus profile measured for one corpus
    out of training/reports/corpora-profile.md.

    The report is prose with an embedded data line, not a structured
    format, so this scrapes a specific heading ("Full label vocabulary,
    sorted (N labels)") and takes the comma-separated line beneath it. That
    is a genuine, if narrow, coupling to the report's wording -- see the
    note in task-2-report.md about how awkward this is to parse and why it
    is still the right call.
    """
    text = REPORT_PATH.read_text()
    start = text.index(section_heading)
    section = text[start:]
    next_heading = re.search(r"\n## ", section[1:])
    if next_heading:
        section = section[: next_heading.start() + 1]
    match = re.search(
        r"Full label vocabulary, sorted \(\d+ labels\)\s*\n\s*\n(.+)", section
    )
    if not match:
        raise AssertionError(
            f"could not find a 'Full label vocabulary, sorted' line under "
            f"{section_heading!r} in {REPORT_PATH} -- its format changed "
            "since this parser was written"
        )
    labels_line = match.group(1).splitlines()[0]
    return {label.strip() for label in labels_line.split(",")}


AI4PRIVACY_MEASURED_LABELS = _measured_vocabulary("## ai4privacy/pii-masking-openpii-1.5m")
NYM_MEASURED_LABELS = _measured_vocabulary("## Wismut/nym-pii-multilingual-data")


def test_the_taxonomy_matches_the_service():
    assert PLATFORM_TYPES == EXPECTED


@pytest.mark.parametrize("mapping", [AI4PRIVACY_MAP, NYM_MAP])
def test_every_target_is_a_real_platform_type_or_an_explicit_drop(mapping):
    for source_label, target in mapping.items():
        assert target is None or target in PLATFORM_TYPES, (
            f"{source_label} maps to {target}, which the service cannot receive"
        )


def test_ai4privacy_map_covers_exactly_the_measured_vocabulary():
    # No label the corpus profile measured is missing, and no entry exists for a label
    # the corpus profile never saw -- checked against the report, not a copied list.
    assert set(AI4PRIVACY_MAP.keys()) == AI4PRIVACY_MEASURED_LABELS


def test_nym_map_covers_exactly_the_measured_vocabulary():
    assert set(NYM_MAP.keys()) == NYM_MEASURED_LABELS


def test_person_is_assembled_from_name_parts():
    # Both corpora split names into components; Temperate has one PERSON type.
    assert to_platform("ai4privacy", "GIVENNAME") == "PERSON"
    assert to_platform("ai4privacy", "SURNAME") == "PERSON"
    assert to_platform("nym", "GIVEN_NAME") == "PERSON"
    assert to_platform("nym", "SURNAME") == "PERSON"


def test_a_country_is_not_a_street_address():
    # The failure that retired LOCATION from the service: masking an address
    # must not mask "Netherlands" and ruin an answer that needed it.
    assert to_platform("nym", "COUNTRY") is None
    assert to_platform("ai4privacy", "CITY") is None


def test_a_region_is_not_a_street_address_either():
    # The retired LOCATION type over-masked 25 country *and region* names.
    # nym's STATE label is exactly that shape of region name.
    assert to_platform("nym", "STATE") is None


def test_street_level_detail_does_map_to_street_address():
    # Street names, building numbers and postcodes are exactly what
    # STREET_ADDRESS is for.
    assert to_platform("ai4privacy", "STREET") == "STREET_ADDRESS"
    assert to_platform("ai4privacy", "BUILDINGNUM") == "STREET_ADDRESS"
    assert to_platform("ai4privacy", "ZIPCODE") == "STREET_ADDRESS"
    assert to_platform("nym", "STREET_NAME") == "STREET_ADDRESS"
    assert to_platform("nym", "BUILDING_NUMBER") == "STREET_ADDRESS"
    assert to_platform("nym", "ZIP_CODE") == "STREET_ADDRESS"


def test_an_unknown_label_raises_rather_than_silently_dropping():
    with pytest.raises(KeyError):
        to_platform("nym", "INVENTED_LABEL")
    with pytest.raises(KeyError):
        to_platform("ai4privacy", "INVENTED_LABEL")


def test_an_unknown_source_raises():
    with pytest.raises(KeyError):
        to_platform("some_other_corpus", "GIVEN_NAME")
