#!/usr/bin/env python3
"""Builds `given-names.txt` from Faker's person providers.

One lower-cased given name per line, sorted by UTF-8 byte order so the Rust side can
binary-search it. Single words of letters, hyphens and apostrophes only, two code points or more.

Usage:
    pip install faker==40.40.0
    python3 crates/model/gazetteer/build_given_names.py crates/model/gazetteer/given-names.txt
"""

import importlib
import sys

LOCALES = [
    "bg_BG", "cs_CZ", "da_DK", "de_AT", "de_CH", "de_DE", "el_GR", "en_GB", "en_IE", "en_IN", "en_US",
    "es_ES", "et_EE", "fi_FI", "fr_BE", "fr_CH", "fr_FR", "ga_IE", "hr_HR", "hu_HU", "it_IT", "lt_LT",
    "lv_LV", "nl_BE", "nl_NL", "pl_PL", "pt_PT", "ro_RO", "sk_SK", "sl_SI", "sv_SE", "tr_TR",
]
ATTRIBUTES = ["first_names", "first_names_male", "first_names_female", "first_names_nonbinary"]


def names(locale):
    try:
        provider = importlib.import_module(f"faker.providers.person.{locale}").Provider
    except ModuleNotFoundError:
        return set()
    found = set()
    for attribute in ATTRIBUTES:
        value = getattr(provider, attribute, None)
        if value:
            found |= set(value.keys() if isinstance(value, dict) else value)
    return found


def keep(name):
    return len(name) >= 2 and all(c.isalpha() or c in "-'" for c in name)


def main(out):
    collected = set()
    for locale in LOCALES:
        collected |= {name.strip().lower().replace("’", "'") for name in names(locale)}
    lines = sorted((name for name in collected if keep(name)), key=lambda name: name.encode("utf-8"))
    with open(out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
