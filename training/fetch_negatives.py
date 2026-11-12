"""Fetches the out-of-domain **negative** source documents: real, licensed
prose in six languages that contains no personal data.

Why this exists
---------------
`training/reports/spurious-address-spans.md` diagnosed 83 of the detector's 121
spurious `STREET_ADDRESS` spans (and the same mechanism drives most of
`PERSON`'s 0.6506 precision) as a *training-distribution gap*, not a decode
defect: both training corpora (`ai4privacy`, `nym`) and this project's own
`generate.py` are name-bearing by construction, so the model has never seen
a paragraph of prose that simply contains nobody. Meeting encyclopedia,
legal, financial or technical text it fills it with names and addresses.

This module fetches material to close that gap. `build_negatives.py` turns
what lands here into training rows with an empty `entities` list.

Why not `benchmark/carriers/`
-----------------------------
The 649-case benchmark is *built from* those 154 documents. Training on them
would make every subsequent measurement of this model worthless. So this
module reuses the carriers' **method** -- the same publishers, the same
licence analysis, the same per-document provenance record -- on **different
material**, and refuses at startup to fetch anything a carrier already used
(see `_assert_disjoint_from_carriers`).

Sources and licences, established rather than assumed
-----------------------------------------------------
The inbound position for each publisher is the same one
`benchmark/carriers/README.md` establishes; it is restated per document in
`negatives/manifest.json` so this corpus stands on its own record.

- **Wikimedia projects** (Wikipedia, Wikibooks, Wikivoyage, and the
  Wikipedias' Help namespaces), CC BY-SA 4.0. Text contributed since June
  2009 is CC BY-SA 4.0; the dual-GFDL option covers only pre-2009 imports, so
  CC BY-SA 4.0 is what is relied on here. Attribution is satisfied by the
  article URL, whose history page lists the contributors; the manifest
  carries that URL *and* the `oldid` of the exact revision fetched, for every
  document. Source: Wikimedia Foundation Terms of Use section 7,
  <https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use>.

  Since 2026-08-25 that is **verified rather than inherited**: every
  Wikimedia host is asked for `meta=siteinfo&siprop=rightsinfo` -- the
  project's own statement of its content licence -- before anything is
  fetched from it, and a project whose answer is not CC BY-SA 4.0 aborts the
  run (`wikimedia_licence()`). The carriers' licence analysis covered
  Wikipedia and Wikibooks; extending it to Wikivoyage and to a Help namespace
  by assuming "it is Wikimedia, so it is CC BY-SA" would have been an
  assumption where this corpus needs a record.

- **EUR-Lex / Publications Office of the EU**, Commission Decision
  2011/833/EU: reuse authorised for commercial and non-commercial purposes,
  provided the source is acknowledged and changes are indicated. (c) European
  Union, 1998-2026. Changes made here: extraction of the paragraph text from
  the XHTML rendering, removal of footnote reference markers, and chunking.
  Source: <https://eur-lex.europa.eu/content/legal-notice/legal-notice.html>.
  Three registers come through this one endpoint and this one licence:
  legislation (`EURLEX_CELEX`), European Parliament resolutions
  (`EP_RESOLUTIONS`) and Commission communications
  (`COMMISSION_COMMUNICATIONS`). They are separated because they read nothing
  alike, not because their provenance differs.

  **Access route note, and it is a real change since 2026-08-22.**
  `eur-lex.europa.eu` now sits behind an AWS WAF JS challenge -- every
  request from this environment returns `HTTP 202` with
  `x-amzn-waf-action: challenge` and a zero-length body, including the
  homepage. The carriers were fetched on 2026-08-21/22, before it was
  turned on. The same documents are served by the Publications Office's
  own Cellar REST endpoint, `publications.europa.eu/resource/celex/{CELEX}`
  with `Accept: application/xhtml+xml` and an ISO-639-3 `Accept-Language`,
  which is unaffected. That is a different *route to the same corpus under
  the same licence*, not a different corpus: the licence above is the
  Publications Office's own notice for the material it publishes on
  EUR-Lex. The manifest records both the Cellar URL actually fetched and
  the human-readable EUR-Lex URL for the same CELEX id.

- **European Central Bank**, ECB website copyright notice: reuse permitted
  provided the information appears accurately, the ECB is cited as the
  source, and any modification is disclosed (the fourth condition, about
  documents sold to buyers, is not triggered -- nothing here is sold). The
  carve-out for named-author documents such as working papers does not apply
  to the Annual Report, which is an institutional publication. Source:
  <https://www.ecb.europa.eu/services/disclaimer/html/index.en.html>.
  Commission Decision 2011/833/EU does **not** apply to the ECB and is not
  relied on for these documents.

What is committed and what is not
---------------------------------
`negatives/manifest.json` (provenance, licence, retrieval date, revision id,
sha256 and character count per document) is committed. The raw text and the
built rows are **not** -- `training/data/` and `training/data-merged/` are
already gitignored for the same reason, and every document here is
re-fetchable from the recorded `oldid`/CELEX id, with the sha256 to prove
the re-fetch matched.

Privacy
-------
No log line, error message or manifest field in this module ever carries
document text. Identity is a document id, a title, or a URL -- never a
sentence, never a matched value. That is the same rule
`benchmark/carriers/README.md` states as "the originals are not recorded".
"""

from __future__ import annotations

import argparse
import hashlib
import html as html_mod
import json
import re
import sys
import time
import urllib.parse
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CARRIER_MANIFEST = REPO / "benchmark" / "carriers" / "manifest.json"
DEFAULT_OUT = HERE / "negatives"

LANGUAGES = ["en", "nl", "de", "fr", "es", "it"]

# ISO-639-3, which is what the Publications Office's Cellar endpoint wants in
# Accept-Language. ISO-639-1 there silently returns the English expression.
ISO3 = {"en": "eng", "nl": "nld", "de": "deu", "fr": "fra", "es": "spa", "it": "ita"}

USER_AGENT = (
    "pii-detection-negatives/1.0 (privacy-detection training corpus; "
    "contact via github.com/temperate-works/pii) python-urllib"
)

WIKIMEDIA_LICENCE = "CC BY-SA 4.0"
WIKIMEDIA_LICENCE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
WIKIMEDIA_TERMS_URL = "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use"

EURLEX_LICENCE = (
    "(c) European Union 1998-2026; reuse authorised under Commission Decision "
    "2011/833/EU, source acknowledged and changes indicated"
)
EURLEX_LICENCE_URL = "https://eur-lex.europa.eu/content/legal-notice/legal-notice.html"

ECB_LICENCE = (
    "(c) European Central Bank; reuse permitted under the ECB website copyright "
    "notice, source cited and modifications disclosed"
)
ECB_LICENCE_URL = "https://www.ecb.europa.eu/services/disclaimer/html/index.en.html"


# ---------------------------------------------------------------------------
# The source registry. Deliberately literal and in-file, so what was fetched
# is reviewable in a diff rather than discovered by running the script.
# ---------------------------------------------------------------------------

# Wikipedia subjects, named once in English and resolved into the other five
# languages through the API's own `langlinks` -- so every language gets the
# *same subject*, and a per-language difference downstream cannot be an
# artefact of the six corpora being about different things. That is the same
# argument benchmark/carriers/README.md makes for taking the same four ECB
# sections in every language.
#
# Selection rule, and it is the one the carriers README found did most of the
# screening work: processes, materials and natural phenomena -- never a
# biography, never a "history of" article, never an article whose subject is a
# person or an organisation. Screening still runs over every chunk (see
# build_negatives.py); this only makes it cheap.
WIKIPEDIA_SUBJECTS = [
    "Anodizing", "Aquifer", "Ball bearing", "Blast furnace", "Capacitor",
    "Centrifugal pump", "Corrosion", "Desalination", "Diesel engine",
    "Distillation", "Electric battery", "Electroplating", "Erosion",
    "Evaporation", "Fermentation", "Galvanization", "Gear", "Glacier",
    "Glass", "Groundwater", "Humidity", "Injection moulding", "Irrigation",
    "Kiln", "Laser", "Limestone", "Lubrication", "Monsoon", "Nitrogen cycle",
    "Paper", "Pasteurization", "Photosynthesis", "Plate tectonics", "Plywood",
    "Polymer", "Radar", "Refrigeration", "Natural rubber", "Sedimentary rock",
    "Semiconductor", "Soap", "Soil", "Sonar", "Surface tension", "Tide",
    "Transformer", "Tunnel", "Gas turbine", "Ultrasound", "Volcano",
    "Weathering", "Welding", "Wheat", "Yeast", "Zinc", "Heat exchanger",
    "Hydraulics", "Rainwater harvesting", "Sand", "Steel",
    # --- added 2026-08-25 -------------------------------------------------
    # Same selection rule, but *aimed*. `reports/distillation-run.md` found
    # that the 6-layer student resolved 13 of the 34 hallucination violations
    # and grew five new ones on sibling documents -- a French wind-turbine
    # article, an Italian cycling guide, a Spanish asphalt article. Those are
    # the immediate subject neighbours of the documents that were already
    # failing (`en-wp-windturbine`, `es-wp-asfalto`, `es-wp-bomba-de-calor`,
    # `de-wp-kreisverkehr`, `nl-wp-schutsluis`, `en-wp-ventilation`), and all
    # of them are benchmark carriers, so the fix cannot be those documents --
    # it has to be *different documents from the same neighbourhoods*.
    #
    # So these subjects deliberately populate those neighbourhoods: wind
    # energy, road surfacing, thermodynamic cycles, HVAC, hydraulic
    # engineering and road junctions. Several carry the exact hazard the
    # failures are made of -- an eponymous technical term whose surname has
    # no personal referent in the text ("the Rankine cycle", "the Darrieus
    # rotor", "a Pelton wheel"), which is the shape `Marnay`, `Lehtinen`,
    # `Ciclo Rankine` and `Hafas` all take.
    "Wind power", "Darrieus wind turbine", "Savonius wind turbine", "Wind farm",
    "Windcatcher", "Pelton wheel", "Francis turbine", "Kaplan turbine",
    "Rankine cycle", "Carnot cycle", "Brayton cycle", "Stirling engine",
    "Air conditioning", "Thermal insulation", "Refrigerant", "Compressor",
    "Road surface", "Asphalt concrete", "Bitumen", "Kerb",
    "Roundabout", "Interchange (road)", "Traffic sign", "Level crossing",
    "Weir", "Canal lock", "Dam", "Levee", "Polder", "Pumping station",
    "Aqueduct (water supply)", "Sluice",
    "Ventilation (architecture)", "Air filter", "Duct (flow)", "Chimney",
    "Cooling tower", "Boiler", "Steam turbine", "Condenser (heat transfer)",
]

# Wikibooks: enumerated by book prefix through `list=allpages`, because
# Wikibooks has no reliable interlanguage links and its books differ per
# project. Books chosen on the same rule as above, and none of them is a book
# any carrier came from (asserted at startup).
WIKIBOOKS_PREFIXES = {
    "en": [
        "Cellular Automata/",
        "Chemical Sciences: A Manual for CBSE XI/",
        "Control Systems/",
        "Engineering Acoustics/",
        "Introduction to Chemical Engineering Processes/",
        "Materials Science/",
        "Structural Biochemistry/",
    ],
    # nl.wikibooks is small (492 pages in namespace 0 whose titles start with
    # A alone) and organises its books differently from the other projects.
    # These four prefixes were found by listing the project rather than
    # guessed: an earlier set ("Natuurkunde/", "Scheikunde/",
    # "Klimaatverandering/", "Elektriciteit/") returned two pages in total,
    # which would have capped the technical register at two documents per
    # language once per-register balancing trimmed every language to the
    # smallest.
    "nl": [
        "Elektronica/",
        "Wiskunde/",
        "Analytische chemie/",
        "Aquaristiek/",
        "Programmeren in Java/",
    ],
    "de": [
        "Anorganische Chemie für Schüler/",
        "Druckversion Physikalische Grundlagen der Nuklearmedizin/",
        "Metalltechnik/",
        "Tabellensammlung Chemie/",
    ],
    "fr": [
        "Chimie générale/",
        "Électronique/",
        "Photographie/",
        "Tribologie/",
    ],
    "es": [
        "Electrónica/",
        "Física/",
        "Ingeniería de alimentos/",
        "Química/",
    ],
    "it": [
        "Chimica organica/",
        "Elettrotecnica/",
        "Fisica classica/",
        "Scienze della Terra/",
    ],
}

# --- added 2026-08-25 ------------------------------------------------------
# More books per project, because the instructional/technical register is the
# thinnest in the corpus (2,142 rows against legal's 10,929) and a Wikibooks
# page is short: 150 documents produced 14 rows each. Raising the per-language
# cap alone would not help -- the previous run exhausted these prefixes -- so
# the prefixes themselves have to grow.
EXTRA_WIKIBOOKS_PREFIXES = {
    "en": [
        "Fundamentals of Transportation/",
        "Introduction to Inorganic Chemistry/",
        "Metabolomics/",
        "Practical Electronics/",
        "Sensory Systems/",
        "Solar System/",
        "Basic Physics of Nuclear Medicine/",
        "General Biology/",
        "Geometry/",
        "Analytical Chemistry/",
    ],
    "nl": [
        "Natuurkunde/",
        "Scheikunde/",
        "Statistiek/",
        "Biologie/",
        "Klimaat/",
        "Sterrenkunde/",
        "Informatica/",
        "Elektriciteit/",
    ],
    "de": [
        "Physikalische Grundlagen der Nuklearmedizin/",
        "Werkstoffkunde/",
        "Organische Chemie für Schüler/",
        "Mathematik/",
        "Biochemie und Pathobiochemie/",
        "Klettern/",
        "Computerhardware/",
    ],
    "fr": [
        "Mécanique/",
        "Mathématiques avec Python et Ruby/",
        "Le Système solaire/",
        "Programmation/",
        "Biochimie/",
        "Géologie/",
    ],
    "es": [
        "Impactos ambientales/",
        "Matemáticas/",
        "Botánica/",
        "Termodinámica/",
        "Mecánica de fluidos/",
        "Programación en C/",
    ],
    "it": [
        "Chimica per il liceo/",
        "Meccanica razionale/",
        "Geologia/",
        "Termodinamica/",
        "Ottica/",
        "Astronomia/",
    ],
}

# --- Wikivoyage: the travel-guide register, added 2026-08-25 ---------------
#
# This register is in `benchmark/carriers/` (26 documents) and was **not** in
# the negative corpus at all, which is a hole the failure list points straight
# at: `it-wv-cicloturismo`, `nl-wv-west-nederland` and `fr-wv-train-europe`
# between them account for 11 of the 34 hallucination violations, and a new
# Italian cycling-guide violation appeared after the retrain. The model has
# never seen travel-guide prose that contains nobody.
#
# Topics rather than destinations. A destination guide is mostly listings --
# an address, a phone number and an opening time per line -- which the
# mechanical sweep would drop wholesale and which is not the prose register
# the hallucination happens in. A travel *topic* is continuous prose about a
# process or an activity, which is the same selection rule the rest of this
# registry follows.
#
# Named once in English and resolved through `langlinks`, like the Wikipedia
# subjects. Anything that resolves to a page a carrier used is dropped
# automatically by `drop_carrier_documents` -- the carriers own `Camping`,
# `Bus travel`, `Cicloturismo`, `Trenes`, `Voyager en train` and eighteen
# others, and it is cheaper to let the guard catch those than to maintain a
# second copy of the carrier list here.
WIKIVOYAGE_TOPICS = [
    "Hiking", "Cycling", "Mountaineering", "Rock climbing", "Skiing",
    "Scuba diving", "Snorkeling", "Sailing", "Kayaking", "Surfing",
    "Fishing", "Birdwatching", "Caving", "Geocaching", "Horse riding",
    "Beaches", "National parks", "Volcanoes", "Deserts", "Glaciers",
    "Cold weather", "Hot weather", "Altitude sickness", "Sunburn",
    "Motion sickness", "Jet lag", "Travel insurance", "Travel health",
    "Water", "Electrical systems", "Internet access", "Telephone service",
    "Weights and measures", "Time zones", "Weather", "Climate change",
    "Air travel", "Ferries", "Driving", "Car rental", "Cargo ships",
    "Cruising", "Hitchhiking", "Walking in urban areas", "Public transport",
    "Architecture", "Castles", "Cathedrals", "Museums", "Industrial tourism",
    "Mining tourism", "Wine", "Beer", "Coffee", "Tea", "Cheese",
    "Vegetarian and vegan food", "Street food", "Photography",
    "Stargazing", "Wildlife", "Ecotourism", "Winter sports", "Nordic skiing",
]

# --- Software / product documentation, added 2026-08-25 --------------------
#
# The Help namespace of each of the six Wikipedias: software documentation
# written natively in each language, not translated, so the six-language
# balance is real rather than a translation artefact. It is a genuinely
# different surface form from every register already here -- imperative
# instructions, interface labels, option names, worked examples -- and it is
# the register the task list calls "technical documentation" and "product
# manuals".
#
# Enumerated by `list=allpages` in namespace 12 with a size floor, not by a
# hand-written list of titles: the Help namespace is named differently in
# every project (`Hilfe:`, `Aide:`, `Ayuda:`, `Aiuto:`) and a numeric
# namespace is the one identifier all six share. `apminsize` is what keeps
# this to documentation and away from the one-line redirect stubs that make
# up much of the namespace.
HELP_NAMESPACE = 12
HELP_MIN_PAGE_BYTES = 6000

# EUR-Lex, by CELEX id. Regulations and directives with technical subject
# matter, never case law: a judgment names the parties, their lawyers and the
# judges, which is exactly the personal data this corpus must not contain.
# None of these is one of the four CELEX ids the carriers used (asserted at
# startup).
# (Regulation (EC) No 1907/2006, REACH, was tried and is a 404 from Cellar --
# only its consolidated expressions are addressable by that CELEX id there.
# It is left out rather than worked around: eleven acts is already more legal
# text than this corpus can use.)
EURLEX_CELEX = [
    ("32009R1223", "Regulation (EC) No 1223/2009 on cosmetic products"),
    ("32017R0745", "Regulation (EU) 2017/745 on medical devices"),
    ("32012L0027", "Directive 2012/27/EU on energy efficiency"),
    ("32018R1999", "Regulation (EU) 2018/1999 on the governance of the energy union"),
    ("32013R0575", "Regulation (EU) No 575/2013 on prudential requirements (CRR)"),
    ("32014R0651", "Regulation (EU) No 651/2014 (General Block Exemption)"),
    ("32019R1020", "Regulation (EU) 2019/1020 on market surveillance"),
    ("32020R0852", "Regulation (EU) 2020/852 (Taxonomy Regulation)"),
    ("32021R1119", "Regulation (EU) 2021/1119 (European Climate Law)"),
    ("32009L0125", "Directive 2009/125/EC on ecodesign requirements"),
    ("32011R0305", "Regulation (EU) No 305/2011 on construction products"),
    # --- added 2026-08-25, same rule: technical subject matter, never case
    # law. Every title below was read off the English expression of the
    # document itself rather than recalled (see reports/negatives-expansion.md).
    ("32016R0679", "Regulation (EU) 2016/679 (General Data Protection Regulation)"),
    ("32019R0631", "Regulation (EU) 2019/631 setting CO2 emission performance standards for new passenger cars and light commercial vehicles"),
    ("32018R0858", "Regulation (EU) 2018/858 on the approval and market surveillance of motor vehicles"),
    ("32017R1369", "Regulation (EU) 2017/1369 setting a framework for energy labelling"),
    ("32014R0517", "Regulation (EU) No 517/2014 on fluorinated greenhouse gases"),
    ("32011L0065", "Directive 2011/65/EU on the restriction of the use of certain hazardous substances in electrical and electronic equipment"),
    ("32012L0019", "Directive 2012/19/EU on waste electrical and electronic equipment (WEEE)"),
    ("32010L0075", "Directive 2010/75/EU on industrial emissions (integrated pollution prevention and control)"),
    ("32013R0952", "Regulation (EU) No 952/2013 laying down the Union Customs Code"),
]

# --- Parliamentary proceedings, added 2026-08-25 ---------------------------
#
# European Parliament resolutions ("texts adopted"), by CELEX id, through the
# same Cellar endpoint and under the same Commission Decision 2011/833/EU as
# the acts above. A different register from legislation and a very distinct
# surface form: no articles and no recitals, but a preamble of "having regard
# to ..." citations and then a numbered operative part of hortatory sentences
# -- "Calls on the Commission to ...", "Notes with concern that ...".
#
# **Selection here carries more weight than anywhere else in this registry.**
# The Parliament also adopts resolutions on individual human-rights cases and
# on country situations, and those name people by design -- `52021IP0148` is
# titled "Assassination of Daphne Caruana Galizia ...". A resolution of that
# kind is the single worst thing that could enter a corpus whose every row
# asserts "there is nobody here". So the rule is narrower than for the acts:
# **technical and policy subjects only, and the title is read before the id
# is written down.** Four candidates were fetched, read and rejected on
# exactly this ground (0148 assassination, 0511 forced labour in a named
# factory, 0294 Afghanistan, 0042 corruption and human rights).
EP_RESOLUTIONS = [
    ("52020IP0198", "European Parliament resolution of 10 July 2020 on a comprehensive European approach to energy storage"),
    ("52020IP0305", "European Parliament resolution of 13 November 2020 on the Sustainable Europe Investment Plan"),
    ("52021IP0040", "European Parliament resolution of 10 February 2021 on the New Circular Economy Action Plan"),
    ("52021IP0073", "European Parliament resolution of 10 March 2021 on corporate due diligence and corporate accountability"),
    ("52021IP0107", "European Parliament resolution of 25 March 2021 on the implementation of the Ambient Air Quality Directives"),
    ("52021IP0240", "European Parliament resolution of 19 May 2021 on a European strategy for energy system integration"),
    ("52021IP0277", "European Parliament resolution of 9 June 2021 on the EU Biodiversity Strategy for 2030"),
    ("52021IP0338", "European Parliament resolution of 7 July 2021 on the impact on the fishing sector of offshore windfarms and other renewable energy systems"),
    ("52021IP0425", "European Parliament resolution of 20 October 2021 on a farm to fork strategy"),
    ("52022IP0140", "European Parliament resolution of 3 May 2022 on artificial intelligence in a digital age"),
    ("52022IP0347", "European Parliament resolution of 5 October 2022 on the EU's response to the increase in energy prices in Europe"),
]

# --- Policy communications, added 2026-08-25 -------------------------------
#
# Commission communications, by CELEX id, same endpoint and same licence.
# A third distinct EU register: neither an act's articles nor a resolution's
# numbered demands, but a strategy document -- narrative sections, targets,
# figures in running text, and the heaviest concentration of capitalised
# programme and initiative names anywhere in this corpus. That last property
# is the reason to want it: a capitalised multi-word name with no personal
# referent is precisely the shape the model invents `PERSON` and
# `STREET_ADDRESS` spans on.
#
# Not every communication is addressable this way: nine of fifteen candidates
# answered HTTP 300 (several manifestations, no single one negotiable) and
# are simply left out.
COMMISSION_COMMUNICATIONS = [
    ("52020DC0299", "Communication: Powering a climate-neutral economy -- an EU Strategy for Energy System Integration"),
    ("52020DC0301", "Communication: A hydrogen strategy for a climate-neutral Europe"),
    ("52020DC0562", "Communication: Stepping up Europe's 2030 climate ambition"),
    ("52021DC0082", "Communication: Forging a climate-resilient Europe -- the new EU Strategy on Adaptation to Climate Change"),
    ("52021DC0350", "Communication: Updating the 2020 New Industrial Strategy"),
    ("52021DC0660", "Communication: Tackling rising energy prices -- a toolbox for action and support"),
]

# ECB annual reports. The carriers took four sections of the **2024** report;
# these are the whole of the **2023** and **2022** reports, which are
# different documents by different drafting years. Asserted disjoint from the
# carriers' `ar2024` at startup.
ECB_REPORTS = [
    ("ar2021", "14d7439b2d", "ECB Annual Report 2021", "press/annual-reports-financial-statements/annual"),
    ("ar2022", "8ae51d163b", "ECB Annual Report 2022", "press/annual-reports-financial-statements/annual"),
    ("ar2023", "d033c21ac2", "ECB Annual Report 2023", "press/annual-reports-financial-statements/annual"),
    ("ar2025", "b7f898b33d", "ECB Annual Report 2025", "press/annual-reports-financial-statements/annual"),
    # Convergence Reports, added so the financial register has enough
    # documents per language to fill three splits with one to spare. Same
    # licence position as the Annual Report: an institutional publication with
    # no named author, so the ECB notice's working-paper carve-out does not
    # apply.
    ("cr202206", "e0fe4e1874", "ECB Convergence Report June 2022", "pub/convergence"),
    ("cr202406", "475c2172bc", "ECB Convergence Report June 2024", "pub/convergence"),
    ("cr202606", "30e75ae1c2", "ECB Convergence Report June 2026", "pub/convergence"),
]


@dataclass
class Document:
    """One fetched source document. `text` is never logged or put in the
    manifest -- only its sha256 and length are."""

    doc_id: str
    lang: str
    register: str
    source: str
    url: str
    licence: str
    licence_url: str
    work: str
    text: str
    revision: str | None = None
    fetch_url: str | None = None
    notes: list[str] = field(default_factory=list)
    cached: bool = False  # loaded from disk on a re-run, not fetched now

    def manifest_entry(self, retrieved: str) -> dict:
        return {
            "id": self.doc_id,
            "lang": self.lang,
            "register": self.register,
            "source": self.source,
            "url": self.url,
            "fetch_url": self.fetch_url or self.url,
            "licence": self.licence,
            "licence_url": self.licence_url,
            "retrieved": retrieved,
            "work": self.work,
            "revision": self.revision,
            "chars": len(self.text),
            "sha256": hashlib.sha256(self.text.encode("utf-8")).hexdigest(),
            "notes": self.notes,
        }


def log(msg: str) -> None:
    print(msg, flush=True)


# Re-run state. A fetch of this size is long enough (and Wikimedia's rate
# limiting flaky enough) that "run it again to fill the gaps" has to be the
# cheap operation, not a full refetch. `--refetch` disables it.
_EXISTING: dict[str, dict] = {}
_RAW_DIR: Path | None = None

# Checkpoint state. The Wikimedia half of a full fetch is roughly 1,500
# rate-limited requests and about two hours, and until 2026-08-25 **none of it
# was written to disk until the last one returned**: raw text and manifest
# were both emitted after every fetcher had finished. An interruption at
# minute 110 therefore cost 110 minutes, which is precisely what turned the
# previous run's throttling into a truncated corpus rather than a slow one.
#
# A document is now written the moment it is fetched, and the manifest is
# rewritten every `_CHECKPOINT_EVERY` documents. `_cached()` needs the raw
# file *and* an agreeing manifest entry, so both have to land for a document
# to count as done -- which is what makes an interrupted run resume instead of
# restart.
_MANIFEST_PATH: Path | None = None
_PRIOR_ENTRIES: dict[str, dict] = {}
_EMITTED: dict[str, dict] = {}
_RETRIEVED = ""
_CHECKPOINT_EVERY = 25
_SINCE_CHECKPOINT = 0
_CARRIER_KEYS: set[str] | None = None
_DROPPED_AS_CARRIER: list[str] = []


def _cached(doc_id: str) -> Document | None:
    """The already-fetched document, if its manifest entry and raw text are
    both on disk and agree on the sha256. A raw file that no longer hashes to
    what the manifest recorded is refetched rather than trusted."""
    entry = _EXISTING.get(doc_id)
    if entry is None or _RAW_DIR is None:
        return None
    path = _RAW_DIR / f"{doc_id}.txt"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != entry.get("sha256"):
        return None
    return Document(
        doc_id=doc_id,
        lang=entry["lang"],
        register=entry["register"],
        source=entry["source"],
        url=entry["url"],
        licence=entry["licence"],
        licence_url=entry["licence_url"],
        work=entry["work"],
        text=text,
        revision=entry.get("revision"),
        fetch_url=entry.get("fetch_url"),
        notes=list(entry.get("notes", [])),
        cached=True,
    )


def _write_manifest(entries: dict[str, dict]) -> None:
    """The manifest, from whatever entries exist right now."""
    if _MANIFEST_PATH is None:
        return
    ordered = sorted(entries.values(), key=lambda e: (e["lang"], e["register"], e["id"]))
    _MANIFEST_PATH.write_text(
        json.dumps(
            {
                "note": (
                    "Provenance and licence record for every out-of-domain negative source "
                    "document. Written by training/fetch_negatives.py; consumed by "
                    "training/build_negatives.py. No field here contains document text -- "
                    "identity is an id, a title, a URL, a revision and a sha256. Every "
                    "document was disjoint from benchmark/carriers/ when it was fetched, "
                    "asserted here at fetch time. That is a claim about the benchmark of "
                    "the day and nothing more: on 2026-08-26 the benchmark grew from 154 "
                    "carriers to 321 and drew 22 of the new ones from pages already in "
                    "this manifest, which no fetch-time assertion could have foreseen. "
                    "Disjointness is therefore re-established at build time by screen 2a "
                    "in build_negatives.py and verified by check_contamination.py, both "
                    "against the benchmark as it stands then. Growing the benchmark "
                    "invalidates this claim until check_contamination.py is re-run."
                ),
                "retrieved": _RETRIEVED,
                "languages": LANGUAGES,
                "documents": ordered,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def emit(doc: Document) -> Document | None:
    """Persist one fetched document immediately, or refuse it.

    Refusal comes first and it is the important half: a document whose URL or
    Wikimedia page title belongs to a benchmark carrier must never reach
    `raw/`, because from that moment a later build could pick it up. The
    end-of-run `drop_carrier_documents()` sweep stays as a second check, but
    it can no longer be the *only* one now that documents are written as they
    arrive.
    """
    global _SINCE_CHECKPOINT
    keys = _CARRIER_KEYS if _CARRIER_KEYS is not None else carrier_keys()
    if doc.url in keys or f"{doc.source}::{doc.work}" in keys:
        _DROPPED_AS_CARRIER.append(doc.doc_id)
        log(f"  {doc.doc_id}: a benchmark carrier uses this document -- not written")
        return None
    if _RAW_DIR is not None:
        (_RAW_DIR / f"{doc.doc_id}.txt").write_text(doc.text, encoding="utf-8")
    entry = doc.manifest_entry(_RETRIEVED)
    if doc.cached and doc.doc_id in _PRIOR_ENTRIES:
        # A cache hit was not retrieved today and the manifest must not claim
        # it was.
        entry["retrieved"] = _PRIOR_ENTRIES[doc.doc_id].get("retrieved", _RETRIEVED)
    _EMITTED[doc.doc_id] = entry
    _SINCE_CHECKPOINT += 1
    if _SINCE_CHECKPOINT >= _CHECKPOINT_EVERY:
        _SINCE_CHECKPOINT = 0
        _write_manifest({**_PRIOR_ENTRIES, **_EMITTED})
    return doc


def _append(docs: list[Document], doc: Document) -> bool:
    """Persist a document and keep it, or drop it. Every fetcher goes through
    here, so there is one place where a document becomes part of the corpus."""
    kept = emit(doc)
    if kept is not None:
        docs.append(kept)
    return kept is not None


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

_LAST_REQUEST: dict[str, float] = {}
# Wikimedia rate-limits anonymous API traffic, and it does so by *returning
# an error*, not by blocking -- at 0.4 s between requests a third of a
# 312-page fetch failed transiently and every one of them succeeded on a
# later single retry. One second between requests, with a longer backoff, is
# what makes this run once rather than three times.
#
# **This is no longer the mechanism that governs the run, and guessing at it
# is what cost the 2026-08-24 fetch its encyclopedia register.** Wikimedia's
# limiter is an Envoy token bucket that answers `HTTP 429`,
# `x-envoy-ratelimited: true` and -- the part that matters -- a
# **`Retry-After` header saying exactly how long the bucket needs**. Measured
# from this environment on 2026-08-25: about eleven requests are allowed, then
# `Retry-After` counts down from ~26 s to the moment the window resets.
#
# Guessing at that with a fixed interval is strictly worse than reading it:
#   - 3.0 s fixed interval, no Retry-After handling: 20 requests -> 11 served,
#     9 refused, 64 s. The 9 refusals are *lost documents*, which is exactly
#     how the previous run ended up with 55 English encyclopedia articles and
#     12 Dutch ones.
#   - 0.4 s interval, sleeping exactly what Retry-After asks: 40 requests ->
#     40 served, 0 refused, 187 s = 12.9 documents a minute.
# Same politeness, nothing lost, and it is the server's own number rather
# than a constant that has to be re-tuned every time the limiter changes.
MIN_INTERVAL_S = 0.4
# 429 is not a failure, it is an instruction, so waiting on one must not
# consume the retry budget that exists for genuine transient failures. This
# caps how long one request may spend waiting for a bucket to refill.
MAX_RATELIMIT_WAITS = 40
MAX_RETRY_AFTER_S = 120.0
_RATELIMIT_WAITS: dict[str, int] = {}


def _get(url: str, *, accept: str | None = None, accept_language: str | None = None,
         retries: int = 8, timeout: int = 120) -> bytes:
    """One polite GET. Rate-limited per host, retried on transient failure.

    The URL is logged on failure; the response body never is.
    """
    global MIN_INTERVAL_S
    host = urllib.parse.urlparse(url).netloc
    last = _LAST_REQUEST.get(host, 0.0)
    wait = MIN_INTERVAL_S - (time.time() - last)
    if wait > 0:
        time.sleep(wait)

    headers = {"User-Agent": USER_AGENT}
    if accept:
        headers["Accept"] = accept
    if accept_language:
        headers["Accept-Language"] = accept_language

    err: Exception | None = None
    attempt = 0
    rate_waits = 0
    while attempt < retries:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 202:
                    # EUR-Lex's WAF challenge answers 202 with an empty body.
                    # Treat it as the failure it is instead of returning b"".
                    raise RuntimeError(f"HTTP 202 with no body (WAF challenge) for {url}")
                body = resp.read()
            _LAST_REQUEST[host] = time.time()
            if not body:
                raise RuntimeError(f"empty body for {url}")
            return body
        except urllib.error.HTTPError as exc:
            # A 404 is an answer, not a hiccup. Retrying one eight times with
            # exponential backoff costs two minutes per document and tells us
            # nothing we did not know on the first attempt -- which is exactly
            # what a wrong CELEX id did on the first run of this fetcher.
            err = exc
            _LAST_REQUEST[host] = time.time()
            if exc.code in (429, 503) and exc.headers.get("Retry-After"):
                # The server said how long to wait. Wait that long, and do
                # **not** count it as an attempt: a throttle is an instruction
                # to slow down, not a failure to give up on, and treating the
                # two the same is what silently truncated the previous fetch's
                # non-English encyclopedia coverage. Only a host that keeps
                # throttling past MAX_RATELIMIT_WAITS is treated as broken.
                try:
                    delay = float(exc.headers["Retry-After"])
                except ValueError:  # HTTP-date form; the servers here use seconds
                    delay = 30.0
                rate_waits += 1
                _RATELIMIT_WAITS[host] = _RATELIMIT_WAITS.get(host, 0) + 1
                if rate_waits > MAX_RATELIMIT_WAITS:
                    break
                time.sleep(min(MAX_RETRY_AFTER_S, max(delay, 1.0)) + 0.5)
                continue
            attempt += 1
            if exc.code != 429 and 400 <= exc.code < 500:
                break
            if exc.code == 429:
                # Throttled with no Retry-After to read. Fall back to raising
                # the floor for every subsequent request to this host, so the
                # run slows down once rather than failing repeatedly.
                MIN_INTERVAL_S = min(10.0, max(MIN_INTERVAL_S, 1.0) * 1.5)
                time.sleep(min(20.0, 5.0 * 2 ** attempt))
            elif attempt < retries:
                time.sleep(min(30.0, 2.0 * 2 ** attempt))
        except Exception as exc:  # noqa: BLE001 - retried, then re-raised
            err = exc
            _LAST_REQUEST[host] = time.time()
            attempt += 1
            if attempt < retries:
                time.sleep(min(30.0, 2.0 * 2 ** attempt))
    raise RuntimeError(
        f"GET failed after {attempt} attempts and {rate_waits} rate-limit waits "
        f"({type(err).__name__}: {err}): {url}"
    ) from err


def _api(project_host: str, params: dict) -> dict:
    params = {**params, "format": "json", "formatversion": "2"}
    url = f"https://{project_host}/w/api.php?" + urllib.parse.urlencode(params)
    return json.loads(_get(url).decode("utf-8"))


# ---------------------------------------------------------------------------
# Licence verification
# ---------------------------------------------------------------------------

_RIGHTS_CACHE: dict[str, tuple[str, str]] = {}


def wikimedia_licence(host: str) -> tuple[str, str]:
    """The licence *the project itself declares*, and a refusal if it is not
    the one this corpus's licence analysis rests on.

    The four registers fetched before this existed inherited their licence
    from `benchmark/carriers/README.md`'s analysis of Wikipedia and Wikibooks.
    That analysis does not automatically extend to Wikivoyage or to a Help
    namespace, and "it is a Wikimedia project, so it is CC BY-SA" is an
    assumption, not a record. `meta=siteinfo&siprop=rightsinfo` is each
    project's own statement of its licence, so one cheap request per host
    turns the assumption into a fetched fact -- and makes a project that ever
    changes licence fail the run instead of quietly entering the corpus.

    One request per host, cached for the process.
    """
    if host in _RIGHTS_CACHE:
        return _RIGHTS_CACHE[host]
    info = _api(host, {"action": "query", "meta": "siteinfo", "siprop": "rightsinfo"})
    rights = info.get("query", {}).get("rightsinfo", {})
    url, text = rights.get("url", ""), rights.get("text", "")
    # deed.nl / deed.de and friends are the same licence in a localised
    # wrapper; the licence identity is the by-sa/4.0 path.
    if "/licenses/by-sa/4.0" not in url:
        raise SystemExit(
            f"{host} declares its content licence as {text!r} ({url}), not CC BY-SA 4.0. "
            "This corpus's licence record says CC BY-SA 4.0 for every Wikimedia project it "
            "draws on; refusing to fetch from a project whose own rightsinfo disagrees."
        )
    _RIGHTS_CACHE[host] = (WIKIMEDIA_LICENCE, WIKIMEDIA_LICENCE_URL)
    log(f"  licence check {host}: {text} <{url}>")
    return _RIGHTS_CACHE[host]


# ---------------------------------------------------------------------------
# HTML -> paragraphs
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"(?s)<[^>]+>")
_SCRIPT_RE = re.compile(r"(?is)<(script|style|noscript)\b.*?</\1>")
_TABLE_RE = re.compile(r"(?is)<table\b.*?</table>")
_P_RE = re.compile(r"(?is)<p\b[^>]*>(.*?)</p>")
# EUR-Lex renders footnote references as `( 25 )` -- inner spaces are what
# distinguish them from an act's own `(1)` paragraph numbering. Same removal
# the carriers made, recorded there and here as a disclosed modification.
_FOOTNOTE_RE = re.compile(r"\(\s+\d+\s+\)")


def _detag(fragment: str) -> str:
    text = html_mod.unescape(_TAG_RE.sub(" ", fragment))
    text = text.replace(" ", " ").replace("​", "")
    return re.sub(r"[ \t]+", " ", text).strip()


def paragraphs_from_html(raw: bytes, *, keep_table_paragraphs: bool = False,
                         min_chars: int = 0) -> list[str]:
    """Every <p> outside <script>/<style>, detagged. Tables are dropped rather
    than flattened: a flattened table is not prose and would teach the model a
    register it will not meet as text.

    `keep_table_paragraphs` exists for one specific and verified reason. The
    Official Journal renders a numbered list -- an EP resolution's operative
    paragraphs, "1. Calls on the Commission to ..." -- as a *table*, one row
    per point, with the number in the first cell and the sentence in the
    second, each wrapped in its own `<p>`. Dropping tables therefore deletes
    the body of the document and keeps only its preamble. Measured on
    `52022IP0140` (AI in a digital age): 4,696 characters with tables dropped
    against 16,787 with their paragraphs kept -- and what the extra 12,000
    characters are is running prose, not table cells.

    `min_chars` is what keeps that safe: the numbering cells ("1.", "(a)")
    are paragraphs too, and a floor of 80 characters admits the sentences and
    rejects the cells. Both are off by default, so every document fetched
    before this existed still extracts byte-identically -- checked by the
    sha256 already in the manifest.
    """
    doc = raw.decode("utf-8", errors="replace")
    doc = _SCRIPT_RE.sub(" ", doc)
    if not keep_table_paragraphs:
        doc = _TABLE_RE.sub(" ", doc)
    out = []
    for frag in _P_RE.findall(doc):
        text = _FOOTNOTE_RE.sub("", _detag(frag))
        if text and len(text) >= min_chars:
            out.append(text)
    return out


# ---------------------------------------------------------------------------
# Fetchers, one per publisher
# ---------------------------------------------------------------------------


def resolve_langlinks(subjects: list[str], langs: list[str],
                      host: str = "en.wikipedia.org") -> dict[str, dict[str, str]]:
    """English subject -> {lang: title}. Uses the English project's own
    langlinks, so each language gets its own project's article on the same
    subject rather than a guessed translation. `host` selects the project --
    en.wikipedia.org for the encyclopedia register, en.wikivoyage.org for the
    travel-guide one."""
    wanted = [lang for lang in langs if lang != "en"]
    resolved: dict[str, dict[str, str]] = {}
    # One request per subject, asking for *every* language at once.
    #
    # Two earlier shapes of this loop were both wrong, in opposite ways, and
    # the second failure is why this is one-title-at-a-time rather than
    # batched:
    #
    # - 20 titles, no `lllang`: `lllimit` caps the langlinks returned across
    #   the whole batch, and 20 articles carry a few thousand interlanguage
    #   links between them. The query silently truncated and returned a
    #   `continue` token, which resolved 52 English articles and only 12
    #   Dutch ones -- a shortfall that reads as "nl.wikipedia has fewer
    #   articles" and is really a paging bug.
    # - 20 titles with `lllang` pinned, one request per language: correct,
    #   but a 20-title langlinks query is an *expensive* API query and
    #   Wikimedia rate-limits it much harder than a cheap one. Every attempt
    #   returned HTTP 429 while single-title queries from the same process,
    #   seconds apart, returned 200.
    #
    # One title carries roughly 100-200 langlinks, comfortably inside
    # `lllimit=500`, so a single cheap request per subject is both complete
    # and permitted. The `continue` guard below stays: it is what turned the
    # first failure from a silent shortfall into a loud one.
    for subject in subjects:
        data = _api(
            host,
            {
                "action": "query",
                "prop": "langlinks",
                "titles": subject,
                "lllimit": "500",
                "redirects": "1",
            },
        )
        query = data.get("query", {})
        pages = query.get("pages", [])
        if not pages or "missing" in pages[0]:
            log(f"  langlinks: no {host} article for {subject!r} -- skipped")
            continue
        page = pages[0]
        mapping = {"en": page["title"]}
        for link in page.get("langlinks", []):
            if link["lang"] in wanted:
                mapping[link["lang"]] = link["title"]
        if "continue" in data:
            raise SystemExit(
                f"langlinks paged for a single title ({subject!r}) -- refusing to "
                "silently resolve a partial language set"
            )
        resolved[subject] = mapping
    return resolved


def fetch_wiki_page(host: str, title: str) -> tuple[str, str | None]:
    """Plain-text extract plus the revision id it came from."""
    data = _api(
        host,
        {
            "action": "query",
            "prop": "extracts|revisions",
            "explaintext": "1",
            "exsectionformat": "plain",
            "rvprop": "ids",
            "titles": title,
            "redirects": "1",
        },
    )
    pages = data.get("query", {}).get("pages", [])
    if not pages or "missing" in pages[0]:
        raise KeyError(f"missing page {host}:{title}")
    page = pages[0]
    revid = str(page["revisions"][0]["revid"]) if page.get("revisions") else None
    return page.get("extract", ""), revid


def list_book_pages(host: str, prefix: str, limit: int) -> list[str]:
    data = _api(
        host,
        {
            "action": "query",
            "list": "allpages",
            "apprefix": prefix,
            "apnamespace": "0",
            "aplimit": str(min(limit, 200)),
        },
    )
    return [p["title"] for p in data.get("query", {}).get("allpages", [])]


def _slug(text: str, maxlen: int = 46) -> str:
    s = urllib.parse.unquote(text).lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:maxlen].strip("-") or "doc"


def fetch_wikipedia(langs: list[str], subjects: list[str], per_language: int) -> list[Document]:
    log(f"wikipedia: resolving {len(subjects)} subjects into {len(langs)} languages")
    for lang in langs:
        wikimedia_licence(f"{lang}.wikipedia.org")
    mapping = resolve_langlinks(subjects, langs)
    docs: list[Document] = []
    counts = dict.fromkeys(langs, 0)
    for subject in subjects:
        titles = mapping.get(subject, {})
        for lang in langs:
            if counts[lang] >= per_language or lang not in titles:
                continue
            host = f"{lang}.wikipedia.org"
            doc_id = f"{lang}-wp-{_slug(subject)}"
            hit = _cached(doc_id)
            if hit is not None:
                # The count follows the *kept* document, not the fetched one:
                # `_append` refuses anything a carrier uses, and a per-language
                # quota that counted refusals would silently shorten the
                # languages whose titles happen to collide with a carrier.
                counts[lang] += _append(docs, hit)
                continue
            try:
                text, revid = fetch_wiki_page(host, titles[lang])
            except Exception as exc:  # noqa: BLE001
                log(f"  {host}: {subject!r} unavailable ({type(exc).__name__}) -- skipped")
                continue
            if len(text) < 1200:
                continue
            licence, licence_url = wikimedia_licence(host)
            counts[lang] += _append(
                docs,
                Document(
                    doc_id=doc_id,
                    lang=lang,
                    register="encyclopedia",
                    source=host,
                    url=f"https://{host}/wiki/" + urllib.parse.quote(titles[lang].replace(" ", "_")),
                    licence=licence,
                    licence_url=licence_url,
                    work=titles[lang],
                    text=text,
                    revision=revid,
                    notes=[f"subject resolved from en:{subject} via langlinks"],
                )
            )
    log("  " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    return docs


def interleave(listings: list[list[str]]) -> list[str]:
    """Round-robin over several ordered lists: the first item of each, then
    the second of each, and so on.

    This is what stops a per-language quota being filled from whichever book
    happens to be listed first. Taking each book to exhaustion in turn filled
    a 60-page English quota from two books that share a subject and a
    vocabulary; interleaving spends the same quota across every book the
    project has, which is the difference between more rows and more variety.
    """
    out: list[str] = []
    for i in range(max((len(t) for t in listings), default=0)):
        for titles in listings:
            if i < len(titles):
                out.append(titles[i])
    return out


def fetch_wikibooks(langs: list[str], prefixes: dict[str, list[str]], per_language: int) -> list[Document]:
    docs: list[Document] = []
    for lang in langs:
        host = f"{lang}.wikibooks.org"
        licence, licence_url = wikimedia_licence(host)
        kept = 0
        # Round-robin across the books, not one book at a time.
        #
        # Taking each prefix to exhaustion before starting the next means a
        # 60-page quota is filled from the first two books that happen to be
        # long enough -- 60 pages of *Cellular Automata* and *Control
        # Systems*, which share a subject, a vocabulary and an author. That is
        # volume, not variety, and variety is the whole point of this
        # expansion. Interleaving spends the same quota across every book the
        # project has.
        listings: list[list[str]] = []
        for prefix in prefixes.get(lang, []):
            try:
                listings.append(list_book_pages(host, prefix, 200))
            except Exception as exc:  # noqa: BLE001
                log(f"  {host}: prefix {prefix!r} unavailable ({type(exc).__name__})")
        for title in interleave(listings):
            if kept >= per_language:
                break
            doc_id = f"{lang}-wb-{_slug(title)}"
            hit = _cached(doc_id)
            if hit is not None:
                kept += _append(docs, hit)
                continue
            try:
                text, revid = fetch_wiki_page(host, title)
            except Exception:  # noqa: BLE001
                continue
            if len(text) < 1200:
                continue
            kept += _append(
                docs,
                Document(
                    doc_id=doc_id,
                    lang=lang,
                    register="instructional-technical",
                    source=host,
                    url=f"https://{host}/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
                    licence=licence,
                    licence_url=licence_url,
                    work=title,
                    text=text,
                    revision=revid,
                )
            )
        log(f"  {host}: {kept} pages")
    return docs


def fetch_wikivoyage(langs: list[str], topics: list[str], per_language: int) -> list[Document]:
    """Travel-guide prose, the register the negative corpus was missing
    entirely while eleven of the benchmark's thirty-four hallucination
    violations sat in it.

    Same langlinks mechanism as `fetch_wikipedia`, against the Wikivoyage
    projects instead. Pages a carrier already uses are not filtered here --
    `drop_carrier_documents` does that once the per-language titles exist,
    which is the only point at which they are known.
    """
    log(f"wikivoyage: resolving {len(topics)} topics into {len(langs)} languages")
    for lang in langs:
        wikimedia_licence(f"{lang}.wikivoyage.org")
    mapping = resolve_langlinks(topics, langs, host="en.wikivoyage.org")
    docs: list[Document] = []
    counts = dict.fromkeys(langs, 0)
    for topic in topics:
        titles = mapping.get(topic, {})
        for lang in langs:
            if counts[lang] >= per_language or lang not in titles:
                continue
            host = f"{lang}.wikivoyage.org"
            doc_id = f"{lang}-wv-{_slug(topic)}"
            hit = _cached(doc_id)
            if hit is not None:
                # The count follows the *kept* document, not the fetched one:
                # `_append` refuses anything a carrier uses, and a per-language
                # quota that counted refusals would silently shorten the
                # languages whose titles happen to collide with a carrier.
                counts[lang] += _append(docs, hit)
                continue
            try:
                text, revid = fetch_wiki_page(host, titles[lang])
            except Exception as exc:  # noqa: BLE001
                log(f"  {host}: {topic!r} unavailable ({type(exc).__name__}) -- skipped")
                continue
            if len(text) < 1200:
                continue
            licence, licence_url = wikimedia_licence(host)
            counts[lang] += _append(
                docs,
                Document(
                    doc_id=doc_id,
                    lang=lang,
                    register="travel-guide",
                    source=host,
                    url=f"https://{host}/wiki/" + urllib.parse.quote(titles[lang].replace(" ", "_")),
                    licence=licence,
                    licence_url=licence_url,
                    work=titles[lang],
                    text=text,
                    revision=revid,
                    notes=[f"topic resolved from en:{topic} via langlinks",
                           "licence verified against the project's own meta=siteinfo rightsinfo"],
                )
            )
    log("  langlinks pass: " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    # Top-up pass, and the reason it exists is worth stating.
    #
    # The langlinks mechanism gives every language the *same topic*, which is
    # what makes a per-language comparison meaningful. It also assumes every
    # project has the topic, and the smaller Wikivoyage editions do not:
    # measured on this run, twenty-two English topics had resolved before the
    # Dutch project produced its first document.
    #
    # A register that reaches five languages is the one thing this corpus must
    # not contain -- `build_negatives.py` refuses to build one, because a
    # per-language figure computed over five of six languages reads as though
    # it covered all six. So where a project simply does not carry the shared
    # topics, it contributes what it does carry: its own mainspace, listed
    # directly, with a size floor. Those documents are marked in the manifest
    # as `listed from the project directly`, so the corpus records which
    # language got which treatment rather than blurring the two.
    topup_wikivoyage(langs, per_language, counts, docs)
    log("  " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    return docs


def topup_wikivoyage(langs: list[str], per_language: int,
                     counts: dict[str, int], docs: list[Document]) -> list[Document]:
    short = [lang for lang in langs if counts[lang] < per_language]
    for lang in short:
        host = f"{lang}.wikivoyage.org"
        licence, licence_url = wikimedia_licence(host)
        try:
            data = _api(
                host,
                {
                    "action": "query", "list": "allpages", "apnamespace": "0",
                    "apfilterredir": "nonredirects", "apminsize": "8000",
                    "aplimit": "500",
                },
            )
        except Exception as exc:  # noqa: BLE001
            log(f"  {host}: mainspace listing unavailable ({type(exc).__name__})")
            continue
        titles_all = [p["title"] for p in data.get("query", {}).get("allpages", [])]
        want = per_language - counts[lang]
        if len(titles_all) > want:
            step = len(titles_all) / want
            titles_all = [titles_all[int(i * step)] for i in range(want)]
        for title in titles_all:
            if counts[lang] >= per_language:
                break
            doc_id = f"{lang}-wv-own-{_slug(title)}"
            hit = _cached(doc_id)
            if hit is not None:
                counts[lang] += _append(docs, hit)
                continue
            try:
                text, revid = fetch_wiki_page(host, title)
            except Exception:  # noqa: BLE001
                continue
            if len(text) < 1200:
                continue
            counts[lang] += _append(
                docs,
                Document(
                    doc_id=doc_id,
                    lang=lang,
                    register="travel-guide",
                    source=host,
                    url=f"https://{host}/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
                    licence=licence,
                    licence_url=licence_url,
                    work=title,
                    text=text,
                    revision=revid,
                    notes=["listed from the project directly, not resolved from an English "
                           "topic via langlinks: this project does not carry enough of the "
                           "shared topics to reach the per-language target",
                           "licence verified against the project's own meta=siteinfo rightsinfo"],
                ),
            )
        log(f"  {host}: topped up to {counts[lang]} from its own mainspace")
    return docs


def fetch_help_namespace(langs: list[str], per_language: int) -> list[Document]:
    """Software documentation: the Help namespace of each of the six
    Wikipedias, written natively in each language.

    Namespace 12 by number rather than by name, because the six projects call
    it `Help:`, `Hulp:`/`Help:`, `Hilfe:`, `Aide:`, `Ayuda:` and `Aiuto:`, and
    the number is the one identifier they share. `apminsize` keeps this to
    documentation: the namespace is mostly short redirect stubs, and a
    six-kilobyte floor is what separates a manual page from a shortcut.
    """
    docs: list[Document] = []
    for lang in langs:
        host = f"{lang}.wikipedia.org"
        licence, licence_url = wikimedia_licence(host)
        try:
            data = _api(
                host,
                {
                    "action": "query",
                    "list": "allpages",
                    "apnamespace": str(HELP_NAMESPACE),
                    "apfilterredir": "nonredirects",
                    "apminsize": str(HELP_MIN_PAGE_BYTES),
                    "aplimit": "500",
                },
            )
        except Exception as exc:  # noqa: BLE001
            log(f"  {host}: Help namespace unavailable ({type(exc).__name__})")
            continue
        titles = [p["title"] for p in data.get("query", {}).get("allpages", [])]
        # Spread across the alphabet rather than take the first N: `allpages`
        # is alphabetical, and the first fifty titles of any Help namespace
        # are a cluster of pages about the same few topics.
        if len(titles) > per_language:
            step = len(titles) / per_language
            titles = [titles[int(i * step)] for i in range(per_language)]
        kept = 0
        for title in titles:
            doc_id = f"{lang}-wh-{_slug(title)}"
            hit = _cached(doc_id)
            if hit is not None:
                kept += _append(docs, hit)
                continue
            try:
                text, revid = fetch_wiki_page(host, title)
            except Exception:  # noqa: BLE001
                continue
            if len(text) < 1200:
                continue
            kept += _append(
                docs,
                Document(
                    doc_id=doc_id,
                    lang=lang,
                    register="technical-documentation",
                    source=host,
                    url=f"https://{host}/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
                    licence=licence,
                    licence_url=licence_url,
                    work=title,
                    text=text,
                    revision=revid,
                    notes=["Help namespace (12); licence verified against the project's own "
                           "meta=siteinfo rightsinfo"],
                )
            )
        log(f"  {host}: {kept} Help pages")
    return docs


def fetch_eurlex(langs: list[str], celex: list[tuple[str, str]], *,
                 register: str = "legal-regulatory", id_prefix: str = "el",
                 keep_table_paragraphs: bool = False,
                 min_paragraph_chars: int = 0) -> list[Document]:
    """Documents addressed by CELEX id through the Cellar endpoint.

    One function for three registers, because they differ only in which
    documents are asked for and how the Official Journal chose to lay them
    out -- the endpoint, the licence, the modifications and the provenance
    record are identical, and a second copy of this would be a second place
    for the licence text to drift.
    """
    docs: list[Document] = []
    for celex_id, work in celex:
        for lang in langs:
            doc_id = f"{lang}-{id_prefix}-{celex_id.lower()}"
            hit = _cached(doc_id)
            if hit is not None:
                _append(docs, hit)
                continue
            fetch_url = f"https://publications.europa.eu/resource/celex/{celex_id}"
            try:
                raw = _get(
                    fetch_url,
                    accept="application/xhtml+xml",
                    accept_language=ISO3[lang],
                )
            except Exception as exc:  # noqa: BLE001
                log(f"  eur-lex {celex_id} [{lang}] unavailable ({type(exc).__name__}) -- skipped")
                continue
            paras = paragraphs_from_html(
                raw,
                keep_table_paragraphs=keep_table_paragraphs,
                min_chars=min_paragraph_chars,
            )
            text = "\n\n".join(paras)
            if len(text) < 2000:
                log(f"  eur-lex {celex_id} [{lang}]: only {len(text)} chars extracted -- skipped")
                continue
            _append(
                docs,
                Document(
                    doc_id=doc_id,
                    lang=lang,
                    register=register,
                    source="eur-lex.europa.eu",
                    url=f"https://eur-lex.europa.eu/legal-content/{lang.upper()}/TXT/HTML/?uri=CELEX:{celex_id}",
                    fetch_url=fetch_url,
                    licence=EURLEX_LICENCE,
                    licence_url=EURLEX_LICENCE_URL,
                    work=work,
                    text=text,
                    revision=celex_id,
                    notes=[
                        "fetched via the Publications Office Cellar REST endpoint; "
                        "eur-lex.europa.eu itself returns HTTP 202 x-amzn-waf-action: challenge "
                        "from this environment",
                        "modifications: paragraph text extracted from the XHTML rendering, "
                        "footnote reference markers removed, "
                        + ("paragraphs inside tables kept and paragraphs shorter than "
                           f"{min_paragraph_chars} characters dropped (the Official Journal "
                           "renders a numbered operative part as a table, one point per row)"
                           if keep_table_paragraphs else "tables dropped"),
                    ],
                )
            )
            log(f"  eur-lex {celex_id} [{lang}]: {len(text)} chars, {len(paras)} paragraphs")
    return docs


def fetch_ecb(langs: list[str], reports: list[tuple[str, str, str, str]]) -> list[Document]:
    docs: list[Document] = []
    for slug, digest, work, path in reports:
        for lang in langs:
            doc_id = f"{lang}-eb-{slug}"
            hit = _cached(doc_id)
            if hit is not None:
                _append(docs, hit)
                continue
            url = f"https://www.ecb.europa.eu/{path}/html/ecb.{slug}~{digest}.{lang}.html"
            try:
                raw = _get(url)
            except Exception as exc:  # noqa: BLE001
                log(f"  ecb {slug} [{lang}] unavailable ({type(exc).__name__}) -- skipped")
                continue
            paras = [p for p in paragraphs_from_html(raw) if len(p) >= 120]
            text = "\n\n".join(paras)
            if len(text) < 2000:
                log(f"  ecb {slug} [{lang}]: only {len(text)} chars extracted -- skipped")
                continue
            _append(
                docs,
                Document(
                    doc_id=doc_id,
                    lang=lang,
                    register="business-annual-report",
                    source="ecb.europa.eu",
                    url=url,
                    licence=ECB_LICENCE,
                    licence_url=ECB_LICENCE_URL,
                    work=work,
                    text=text,
                    revision=slug,
                    notes=[
                        "modifications disclosed per the ECB notice: paragraph text extracted "
                        "from the HTML rendering, navigation and tables dropped, chunked"
                    ],
                )
            )
            log(f"  ecb {slug} [{lang}]: {len(text)} chars, {len(paras)} paragraphs")
    return docs


# ---------------------------------------------------------------------------
# The disjointness guard
# ---------------------------------------------------------------------------


def carrier_keys() -> set[str]:
    """Every identifier that would make a negative document the same document
    as a carrier: the carrier's URL, its Wikimedia page title, its CELEX id,
    its ECB report slug."""
    manifest = json.loads(CARRIER_MANIFEST.read_text())
    keys: set[str] = set()
    for c in manifest["carriers"]:
        url = c["url"]
        keys.add(url)
        if "/wiki/" in url:
            keys.add(f"{c['source']}::" + urllib.parse.unquote(url.rsplit("/wiki/", 1)[-1]).replace("_", " "))
        m = re.search(r"CELEX:(\w+)", url)
        if m:
            keys.add("celex::" + m.group(1))
        m = re.search(r"ecb\.(ar\d{4})~", url)
        if m:
            keys.add("ecb::" + m.group(1))
    return keys


def _assert_disjoint_from_carriers() -> None:
    """Refuses to run if the registry above names anything a carrier used.
    Cheap, and it is the whole reason this corpus can be trained on."""
    keys = carrier_keys()
    clashes = []
    for registry in (EURLEX_CELEX, EP_RESOLUTIONS, COMMISSION_COMMUNICATIONS):
        for celex_id, _ in registry:
            if "celex::" + celex_id in keys:
                clashes.append(f"CELEX {celex_id}")
    for slug, _, _, _ in ECB_REPORTS:
        if "ecb::" + slug in keys:
            clashes.append(f"ECB {slug}")
    if clashes:
        raise SystemExit(
            "fetch_negatives: the source registry names material the benchmark "
            "carriers already use, which would contaminate every measurement: "
            + ", ".join(clashes)
        )
    # The same CELEX id in two registries would be the same document fetched
    # twice under two ids, which splits one document across `train` and
    # `test` -- the exact leakage the document-level grouping exists to stop.
    seen: dict[str, str] = {}
    for name, registry in (("EURLEX_CELEX", EURLEX_CELEX),
                           ("EP_RESOLUTIONS", EP_RESOLUTIONS),
                           ("COMMISSION_COMMUNICATIONS", COMMISSION_COMMUNICATIONS)):
        for celex_id, _ in registry:
            if celex_id in seen:
                raise SystemExit(
                    f"fetch_negatives: CELEX {celex_id} appears in both {seen[celex_id]} and "
                    f"{name}. One document under two ids would be split across train and test."
                )
            seen[celex_id] = name


def drop_carrier_documents(docs: list[Document]) -> tuple[list[Document], list[str]]:
    """Second pass, after the titles are actually known: drops any fetched
    document whose URL or Wikimedia title matches a carrier. The registry
    guard above cannot catch this for Wikipedia, because the per-language
    titles only exist after `langlinks` resolves them."""
    keys = carrier_keys()
    kept, dropped = [], []
    for d in docs:
        title_key = f"{d.source}::{d.work}"
        if d.url in keys or title_key in keys:
            dropped.append(d.doc_id)
        else:
            kept.append(d)
    return kept, dropped


# ---------------------------------------------------------------------------


REGISTER_NAMES = {
    "encyclopedia": "encyclopedia",
    "instructional-technical": "instructional / technical",
    "legal-regulatory": "legal / regulatory",
    "business-annual-report": "business / annual report",
    "travel-guide": "travel guide",
    "technical-documentation": "technical documentation",
    "parliamentary": "parliamentary proceedings",
    "policy-communication": "policy communication",
}


def write_readme(path: Path, manifest: dict) -> None:
    """The human-readable form of manifest.json, generated from it so the two
    cannot drift. `benchmark/carriers/README.md` is the model; this is the
    same record for the material that must **not** be in the benchmark."""
    docs = manifest["documents"]
    by_lang: dict[str, list[dict]] = {}
    for d in docs:
        by_lang.setdefault(d["lang"], []).append(d)

    lines: list[str] = [
        "# Out-of-domain negatives -- real licensed prose containing no personal data",
        "",
        "*Generated by `training/fetch_negatives.py`; do not edit by hand. The machine-readable",
        "form is `manifest.json` in this directory, and this file is written from it, so the two",
        "cannot disagree.*",
        "",
        "Each document here is a passage of real, publicly licensed prose in a register the detector",
        "currently hallucinates into. `training/build_negatives.py` chunks and screens them into",
        "training rows whose `entities` list is empty. The point is stated in one sentence in",
        "`training/reports/spurious-address-spans.md`: both training corpora are name-bearing, so",
        "**the model has never seen prose that contains no personal data**, and meeting",
        "encyclopedia, legal, financial or technical text it fills it with names and addresses.",
        "",
        "## These are not the benchmark's carriers, and that is the whole point",
        "",
        "`benchmark/carriers/` holds 154 documents from these same publishers. The benchmark's 649",
        "cases are **built from those documents**, so training on them would make every subsequent",
        "measurement of this model worthless -- the model would be scored on text it had memorised.",
        "",
        "So this corpus reuses the carriers' *method* -- same publishers, same licence analysis,",
        "same per-document provenance record -- on **different material**. Three separate",
        "mechanisms enforce that:",
        "",
        "1. `fetch_negatives._assert_disjoint_from_carriers()` refuses to start if the source",
        "   registry names a CELEX id or an ECB report a carrier already uses.",
        "2. `fetch_negatives.drop_carrier_documents()` drops any *fetched* document whose URL or",
        "   Wikimedia page title matches a carrier -- needed because the per-language Wikipedia",
        "   titles only exist after `langlinks` resolves them.",
        "3. `training/check_contamination.py` checks all 649 cases against the built rows, by",
        "   document identity (carrier id, URL, title, CELEX id, ECB report slug) **and** by text.",
        "",
        "## Licences, established rather than assumed",
        "",
        "The inbound position per publisher, restated from `benchmark/carriers/README.md` so this",
        "corpus stands on its own record. It is recorded per document in `manifest.json`.",
        "",
        "**Wikimedia projects** (Wikipedia, Wikibooks, Wikivoyage, and the Wikipedias' Help",
        "namespaces). Text contributed since June 2009 is CC BY-SA 4.0; the dual-GFDL option covers",
        "only pre-2009 imports, so **CC BY-SA 4.0 is the licence relied on**. Attribution is",
        "satisfied by the article URL, whose history page lists the contributors; every row below",
        "carries that URL and the `oldid` of the exact revision fetched. Source: Wikimedia",
        "Foundation Terms of Use section 7,",
        "<https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use>.",
        "",
        "> **This is checked, not assumed.** Before fetching anything from a Wikimedia host,",
        "> `fetch_negatives.wikimedia_licence()` asks that project for",
        "> `meta=siteinfo&siprop=rightsinfo` -- its own declaration of its content licence -- and",
        "> aborts the run if the answer is not CC BY-SA 4.0. The carriers' licence analysis covered",
        "> Wikipedia and Wikibooks; Wikivoyage and the Help namespaces were added on 2026-08-25,",
        "> and \"it is a Wikimedia project, so it is CC BY-SA\" is an assumption where this corpus",
        "> needs a record. All eighteen hosts answered CC BY-SA 4.0.",
        "",
        "**EUR-Lex / Publications Office of the EU.** Reuse authorised for commercial and",
        "non-commercial purposes under Commission Decision 2011/833/EU, provided the source is",
        "acknowledged and changes are indicated. (c) European Union, 1998-2026. The changes made",
        "here are: extraction of the paragraph text from the XHTML rendering, removal of footnote",
        "reference markers, dropping of tables, and chunking. Source:",
        "<https://eur-lex.europa.eu/content/legal-notice/legal-notice.html>.",
        "",
        "> **Access route, and it is a real change since the carriers were fetched on 2026-08-22.**",
        "> `eur-lex.europa.eu` now sits behind an AWS WAF JS challenge: every request from this",
        "> environment returns `HTTP 202` with `x-amzn-waf-action: challenge` and a zero-length",
        "> body, the homepage included. The same documents are served by the Publications Office's",
        "> own Cellar REST endpoint, `publications.europa.eu/resource/celex/{CELEX}` with",
        "> `Accept: application/xhtml+xml` and an ISO-639-3 `Accept-Language`, which is unaffected.",
        "> That is a different route to the same corpus under the same licence, not a different",
        "> corpus. Both URLs are recorded per document: `fetch_url` is what was actually",
        "> requested, `url` is the human-readable EUR-Lex address for the same CELEX id.",
        "",
        "**European Central Bank.** Reuse permitted under the ECB website copyright notice on four",
        "conditions: the information must appear accurately, the ECB must be cited as the source,",
        "any modification must be disclosed, and -- where the information is incorporated into a",
        "document that is sold -- buyers must be told it is available free of charge. The first",
        "three are met (source URL per document, and the modifications are the same extraction and",
        "chunking as above, recorded here and in `manifest.json`); the fourth is not triggered,",
        "since nothing here is sold. The notice's carve-out for named-author documents such as",
        "working papers does not apply: Annual Reports and Convergence Reports are institutional",
        "publications with no named author. Source:",
        "<https://www.ecb.europa.eu/services/disclaimer/html/index.en.html>. Commission Decision",
        "2011/833/EU does **not** apply to the ECB and is not relied on for these documents.",
        "",
        "## Screening: how it was checked that none of this is personal data",
        "",
        "A negative row asserts *there is nothing here*. One overlooked real name teaches the model",
        "to miss that name, which in a privacy product is worse than the hallucination this corpus",
        "exists to fix. `training/build_negatives.py` runs five independent screens over every",
        "chunk -- two of them independently-trained NER models -- on top of the selection rule that",
        "precedes them, and **drops** rather than repairs anything that trips one. The carriers",
        "*replaced* what they found, because a carrier has to keep reading like its register for",
        "the benchmark to be fair, and a training negative has no such duty.",
        "`training/reports/negatives.md` carries the counts from the build that produced the",
        "shipped corpus.",
        "",
        "**The words that tripped a screen are not recorded** -- not here, not in the manifest, not",
        "in any log line. That is `benchmark/carriers/README.md`'s \"the originals are not",
        "recorded\" rule, and it applies with more force here because these documents were never",
        "hand-read.",
        "",
        "## What is committed",
        "",
        "This file and `manifest.json`. **Not** `raw/` (the fetched text) and **not**",
        "`training/data-negatives/` (the built rows) -- same reasoning as `training/data/` and",
        "`training/data-merged/`, which are gitignored for being large and derived. Every document",
        "is re-fetchable from the revision id recorded below, and the `sha256` in `manifest.json`",
        "proves a re-fetch matched.",
        "",
        "## The documents",
        "",
        f"Retrieved {manifest['retrieved']}. `chars` is the document length as fetched, before",
        "chunking.",
        "",
    ]

    from collections import Counter

    reg_counts = Counter((d["lang"], d["register"]) for d in docs)
    langs = sorted(by_lang)
    registers = sorted({d["register"] for d in docs})
    lines += ["| register | " + " | ".join(f"`{l}`" for l in langs) + " | total |",
              "|---" * (len(langs) + 2) + "|"]
    for reg in registers:
        row = [reg_counts.get((l, reg), 0) for l in langs]
        lines.append(f"| {REGISTER_NAMES.get(reg, reg)} | " + " | ".join(str(n) for n in row)
                     + f" | {sum(row)} |")
    total_row = [sum(reg_counts.get((l, r), 0) for r in registers) for l in langs]
    lines.append("| **total** | " + " | ".join(f"**{n}**" for n in total_row)
                 + f" | **{sum(total_row)}** |")
    lines.append("")

    for lang in langs:
        entries = sorted(by_lang[lang], key=lambda d: (d["register"], d["id"]))
        lines += [
            f"### `{lang}` -- {len(entries)} documents",
            "",
            "| id | register | source | chars | rev / CELEX | licence | retrieved | URL |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for d in entries:
            lines.append(
                f"| `{d['id']}` | {REGISTER_NAMES.get(d['register'], d['register'])} "
                f"| {d['source']} | {d['chars']} | {d.get('revision') or '--'} "
                f"| {d['licence']} | {d['retrieved']} | <{d['url']}> |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    global MIN_INTERVAL_S, _RAW_DIR, _EXISTING
    global _MANIFEST_PATH, _PRIOR_ENTRIES, _RETRIEVED, _CARRIER_KEYS

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--retrieved", default=time.strftime("%Y-%m-%d"),
                    help="retrieval date recorded in the manifest (ISO)")
    ap.add_argument("--wikipedia-per-language", type=int, default=110)
    ap.add_argument("--wikibooks-per-language", type=int, default=60)
    ap.add_argument("--wikivoyage-per-language", type=int, default=60)
    ap.add_argument("--help-pages-per-language", type=int, default=55)
    ap.add_argument(
        "--only",
        choices=["wikipedia", "wikibooks", "wikivoyage", "wikivoyage-own", "help",
                 "eurlex", "ep", "commission", "ecb"],
        action="append",
    )
    ap.add_argument("--min-interval", type=float, default=MIN_INTERVAL_S,
                    help="seconds between requests to one host (raised automatically on HTTP 429)")
    ap.add_argument("--refetch", action="store_true",
                    help="ignore anything already on disk and fetch every document again")
    args = ap.parse_args()

    _assert_disjoint_from_carriers()

    MIN_INTERVAL_S = args.min_interval

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    _RAW_DIR = raw_dir
    manifest_path = out_dir / "manifest.json"
    prior_entries: dict[str, dict] = {}
    if manifest_path.exists():
        prior_entries = {e["id"]: e for e in json.loads(manifest_path.read_text()).get("documents", [])}
    _EXISTING = {} if args.refetch else prior_entries

    # Checkpoint state, so an interrupted fetch resumes rather than restarts.
    _MANIFEST_PATH = manifest_path
    _PRIOR_ENTRIES = {} if args.refetch else dict(prior_entries)
    _RETRIEVED = args.retrieved
    _CARRIER_KEYS = carrier_keys()

    only = set(args.only or ["wikipedia", "wikibooks", "wikivoyage", "help",
                             "eurlex", "ep", "commission", "ecb"])
    docs: list[Document] = []
    t0 = time.time()
    if "wikipedia" in only:
        docs += fetch_wikipedia(LANGUAGES, WIKIPEDIA_SUBJECTS, args.wikipedia_per_language)
    if "wikibooks" in only:
        log("wikibooks:")
        prefixes = {
            lang: WIKIBOOKS_PREFIXES.get(lang, []) + EXTRA_WIKIBOOKS_PREFIXES.get(lang, [])
            for lang in LANGUAGES
        }
        docs += fetch_wikibooks(LANGUAGES, prefixes, args.wikibooks_per_language)
    if "wikivoyage" in only:
        docs += fetch_wikivoyage(LANGUAGES, WIKIVOYAGE_TOPICS, args.wikivoyage_per_language)
    if "wikivoyage-own" in only:
        # The top-up alone, without repeating the langlinks resolution -- which
        # costs one request per topic and is pure waste when the shared topics
        # are already fetched and the only gap left is a project that does not
        # carry them.
        log("wikivoyage top-up (own mainspace, no langlinks):")
        have: dict[str, int] = dict.fromkeys(LANGUAGES, 0)
        for entry in prior_entries.values():
            if entry.get("register") == "travel-guide":
                have[entry["lang"]] = have.get(entry["lang"], 0) + 1
        log("  already held: " + ", ".join(f"{k}={v}" for k, v in have.items()))
        docs += topup_wikivoyage(LANGUAGES, args.wikivoyage_per_language, have, [])
    if "help" in only:
        log("wikipedia Help namespace:")
        docs += fetch_help_namespace(LANGUAGES, args.help_pages_per_language)
    if "eurlex" in only:
        log("eur-lex (via publications.europa.eu Cellar):")
        docs += fetch_eurlex(LANGUAGES, EURLEX_CELEX)
    if "ep" in only:
        log("european parliament resolutions (via publications.europa.eu Cellar):")
        docs += fetch_eurlex(
            LANGUAGES, EP_RESOLUTIONS, register="parliamentary", id_prefix="ep",
            keep_table_paragraphs=True, min_paragraph_chars=80,
        )
    if "commission" in only:
        log("commission communications (via publications.europa.eu Cellar):")
        docs += fetch_eurlex(
            LANGUAGES, COMMISSION_COMMUNICATIONS, register="policy-communication",
            id_prefix="ec",
        )
    if "ecb" in only:
        log("ecb:")
        docs += fetch_ecb(LANGUAGES, ECB_REPORTS)

    # Second carrier sweep. `emit()` already refuses a carrier document at the
    # moment it is fetched -- it has to, now that documents are written as they
    # arrive rather than at the end -- so this should find nothing. It stays
    # because "should find nothing" is a claim worth having a check behind.
    docs, dropped = drop_carrier_documents(docs)
    if dropped:
        log(f"dropped {len(dropped)} fetched documents that a carrier already uses: {', '.join(sorted(dropped))}")
        for doc_id in dropped:
            _EMITTED.pop(doc_id, None)
            _PRIOR_ENTRIES.pop(doc_id, None)
            (raw_dir / f"{doc_id}.txt").unlink(missing_ok=True)
    if _DROPPED_AS_CARRIER:
        log(f"emit() refused {len(_DROPPED_AS_CARRIER)} documents a carrier already uses: "
            f"{', '.join(sorted(_DROPPED_AS_CARRIER))}")

    # Entries for documents this invocation did not touch are carried
    # forward, so `--only` fills a gap instead of truncating the manifest to
    # whatever the last partial run happened to fetch.
    entries = {**_PRIOR_ENTRIES, **_EMITTED}
    _write_manifest(entries)

    manifest = json.loads(manifest_path.read_text())
    write_readme(out_dir / "README.md", manifest)

    ordered = manifest["documents"]
    total_chars = sum(e["chars"] for e in ordered)
    by_reg: dict[str, int] = {}
    by_lang: dict[str, int] = {}
    for e in ordered:
        by_reg[e["register"]] = by_reg.get(e["register"], 0) + 1
        by_lang[e["lang"]] = by_lang.get(e["lang"], 0) + 1
    log("")
    log(f"{len(ordered)} documents, {total_chars:,} characters, in {time.time() - t0:.0f}s")
    waits = sum(_RATELIMIT_WAITS.values())
    if waits:
        log(f"  rate-limit waits honoured (HTTP 429 Retry-After): {waits} across "
            + ", ".join(f"{h}={n}" for h, n in sorted(_RATELIMIT_WAITS.items()) if n))
    log("  by register: " + ", ".join(f"{k}={v}" for k, v in sorted(by_reg.items())))
    log("  by language: " + ", ".join(f"{k}={v}" for k, v in sorted(by_lang.items())))
    log(f"  manifest: {manifest_path}")
    log(f"  README:   {out_dir / 'README.md'}")
    log(f"  raw text: {raw_dir} (gitignored -- re-fetchable from the recorded revision ids)")


if __name__ == "__main__":
    sys.exit(main())
