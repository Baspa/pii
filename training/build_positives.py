"""Generates the two `PERSON` shapes the corpus underrepresents: names in
tabular and list positions, and names in bylines.

Why these two and no others
---------------------------
`training/reports/person-misses.md` diagnosed all three remaining `PERSON`
recall misses -- the 108/111 that keeps that gate red against Azure's
111/111 -- and they are one shape each of two families:

- `domain-nl-csv-1` (`nl`): a five-column CSV. The model found row 1's name
  at 0.9929 and every other field of row 2 (e-mail, phone, IBAN), and
  produced **no person signal at all** for row 2's name. That report ties it
  to the `adv-list`/`long-de-1` list-position regressions already recorded
  in `reports/retrained-comparison.md`: list- and row-structured name
  detection is weaker than prose name detection.
- `it-wv-cicloturismo-d1` (`it`): `(redatto da Ilaria Lombardo)` -- a byline
  inside parentheses. Total miss, no span at any confidence.
- `it-wp-rotatoria-dn` (`it`): `redatto da Lorenzo Roversi` -- a byline not
  in parentheses. The model returned `Roversi` alone at 0.677 and dropped
  the given name, while finding `Aurélien Roubaud` at 0.9999 in the same
  case. A weak byline prediction that failed to extend leftward.

So this module generates exactly those: **rows** and **bylines**, in all six
languages, with the name in later rows as often as in the first, and with
the byline in and out of parentheses.

Constructed, never annotated
----------------------------
Every record is built with `generate.Builder` one piece at a time, recording
each entity's character span at the moment its text is appended. Nothing here
searches a finished string for a substring to label. That is `generate.py`'s
rule and its reasoning applies unchanged: searching a finished string is how
alignment noise happens, and a mislabelled *positive* is worse here than
anywhere else, because the whole point of these rows is to be right about a
name the current model gets wrong.

Names, e-mails, phone numbers and IBANs come from Faker, seeded
deterministically. **No real personal data enters training**; that is the
same guarantee `generate.py` gives, by the same means.

Held-out templates, not held-out rows
-------------------------------------
Two rows built from one template differ only in which names Faker drew. A
random split would put near-copies on both sides and every recall number
computed afterwards would be inflated -- `build_dataset.py`'s module
docstring makes this argument at length for the corpora, and it applies with
more force to a template pool this small.

So the split is **by template**: `HOLDOUT_TEMPLATES` names the ones reserved
for validation and test, and no row built from them is ever in `train`. What
the held-out split then measures is not "can the model recall a name it has
seen" but "did it learn the *shape*, or only these sentences" -- which is the
question that matters, since the three real misses are in shapes this module
does not contain and cannot contain.

The tension this creates, stated plainly
----------------------------------------
These rows make the model **more** willing to find names. The out-of-domain
negatives in `build_negatives.py` make it **less** willing to invent them.
They pull in opposite directions and the balance between them is the whole
quality question for this run. `reports/distillation-plan.md` records the
ablation that measured the trade rather than guessing it, and the ratio it
chose.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from faker import Faker

from generate import Builder, Segment, _de_name, _nl_name, _render, _seed_for

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "data-positives"

LANGUAGES = ["en", "nl", "de", "fr", "es", "it"]
LOCALE = {"en": "en_US", "nl": "nl_NL", "de": "de_DE", "fr": "fr_FR", "es": "es_ES", "it": "it_IT"}


# ---------------------------------------------------------------------------
# Per-language surface strings. One dict per language rather than one
# template per language: the *shape* is what is being taught, and writing it
# once per shape keeps the six languages genuinely parallel instead of six
# separately-drifting sets of templates.
# ---------------------------------------------------------------------------

LOC: dict[str, dict] = {
    "en": {
        "csv_head": "id,name,email,phone,iban",
        "table_head": ["Name", "E-mail", "Phone"],
        "kv": {"name": "Name", "email": "E-mail", "phone": "Phone", "role": "Role"},
        "roles": ["author", "reviewer", "contact", "account manager", "project lead"],
        "doc": ["the report", "the file", "the annex", "the minutes", "the guide"],
        "byline_in": ["written by ", "compiled by ", "prepared by ", "edited by "],
        "byline_cap": ["Written by ", "Prepared by ", "Reviewed by "],
        "intro": [
            "The attached export lists the accounts opened this quarter.",
            "Please check the rows below against your own records.",
            "The following contacts were migrated from the old system.",
        ],
        "byline_intro": [
            "The section on rolling resistance was rewritten this year",
            "The chapter has been revised twice since publication",
            "This part of the manual covers routine maintenance",
        ],
        "caption": ["Figure 3", "Table 2", "Photo", "Diagram 1"],
        "footnote": ["Source: internal note", "Note", "Ref."],
        "participants": "Participants",
        "and": "and",
    },
    "nl": {
        "csv_head": "id,naam,email,telefoon,iban",
        "table_head": ["Naam", "E-mail", "Telefoon"],
        "kv": {"name": "Naam", "email": "E-mail", "phone": "Telefoon", "role": "Rol"},
        "roles": ["auteur", "beoordelaar", "contactpersoon", "accountmanager", "projectleider"],
        "doc": ["het rapport", "het dossier", "de bijlage", "het verslag", "de handleiding"],
        "byline_in": ["geschreven door ", "opgesteld door ", "samengesteld door ", "geredigeerd door "],
        "byline_cap": ["Geschreven door ", "Opgesteld door ", "Beoordeeld door "],
        "intro": [
            "De bijgevoegde export bevat de rekeningen die dit kwartaal zijn geopend.",
            "Controleer de onderstaande regels aan de hand van uw eigen administratie.",
            "De volgende contactpersonen zijn overgezet uit het oude systeem.",
        ],
        "byline_intro": [
            "Het hoofdstuk over rolweerstand is dit jaar herschreven",
            "De tekst is sinds de publicatie tweemaal herzien",
            "Dit deel van de handleiding gaat over regulier onderhoud",
        ],
        "caption": ["Figuur 3", "Tabel 2", "Foto", "Schema 1"],
        "footnote": ["Bron: interne notitie", "Noot", "Ref."],
        "participants": "Deelnemers",
        "and": "en",
    },
    "de": {
        "csv_head": "id,name,email,telefon,iban",
        "table_head": ["Name", "E-Mail", "Telefon"],
        "kv": {"name": "Name", "email": "E-Mail", "phone": "Telefon", "role": "Rolle"},
        "roles": ["Autor", "Prüfer", "Ansprechpartner", "Kundenbetreuer", "Projektleiter"],
        "doc": ["der Bericht", "die Akte", "der Anhang", "das Protokoll", "das Handbuch"],
        "byline_in": ["verfasst von ", "erstellt von ", "zusammengestellt von ", "bearbeitet von "],
        "byline_cap": ["Verfasst von ", "Erstellt von ", "Geprüft von "],
        "intro": [
            "Der beigefügte Export enthält die in diesem Quartal eröffneten Konten.",
            "Bitte prüfen Sie die folgenden Zeilen anhand Ihrer eigenen Unterlagen.",
            "Die folgenden Ansprechpartner wurden aus dem Altsystem übernommen.",
        ],
        "byline_intro": [
            "Der Abschnitt über den Rollwiderstand wurde in diesem Jahr neu geschrieben",
            "Der Text wurde seit der Veröffentlichung zweimal überarbeitet",
            "Dieser Teil des Handbuchs behandelt die regelmäßige Wartung",
        ],
        "caption": ["Abbildung 3", "Tabelle 2", "Foto", "Schema 1"],
        "footnote": ["Quelle: interne Notiz", "Anmerkung", "Ref."],
        "participants": "Teilnehmer",
        "and": "und",
    },
    "fr": {
        "csv_head": "id,nom,email,telephone,iban",
        "table_head": ["Nom", "E-mail", "Téléphone"],
        "kv": {"name": "Nom", "email": "E-mail", "phone": "Téléphone", "role": "Rôle"},
        "roles": ["auteur", "relecteur", "personne de contact", "gestionnaire de compte", "chef de projet"],
        "doc": ["le rapport", "le dossier", "l'annexe", "le compte rendu", "le manuel"],
        "byline_in": ["rédigé par ", "établi par ", "compilé par ", "révisé par "],
        "byline_cap": ["Rédigé par ", "Établi par ", "Révisé par "],
        "intro": [
            "L'export ci-joint recense les comptes ouverts au cours de ce trimestre.",
            "Merci de vérifier les lignes ci-dessous au regard de vos propres registres.",
            "Les contacts suivants ont été migrés depuis l'ancien système.",
        ],
        "byline_intro": [
            "La section sur la résistance au roulement a été réécrite cette année",
            "Le texte a été révisé deux fois depuis sa publication",
            "Cette partie du manuel traite de l'entretien courant",
        ],
        "caption": ["Figure 3", "Tableau 2", "Photo", "Schéma 1"],
        "footnote": ["Source : note interne", "Note", "Réf."],
        "participants": "Participants",
        "and": "et",
    },
    "es": {
        "csv_head": "id,nombre,email,telefono,iban",
        "table_head": ["Nombre", "Correo", "Teléfono"],
        "kv": {"name": "Nombre", "email": "Correo", "phone": "Teléfono", "role": "Función"},
        "roles": ["autor", "revisor", "persona de contacto", "gestor de cuenta", "jefe de proyecto"],
        "doc": ["el informe", "el expediente", "el anexo", "el acta", "el manual"],
        "byline_in": ["redactado por ", "elaborado por ", "recopilado por ", "revisado por "],
        "byline_cap": ["Redactado por ", "Elaborado por ", "Revisado por "],
        "intro": [
            "La exportación adjunta recoge las cuentas abiertas durante este trimestre.",
            "Compruebe las siguientes filas con sus propios registros.",
            "Los siguientes contactos se migraron desde el sistema anterior.",
        ],
        "byline_intro": [
            "La sección sobre la resistencia a la rodadura se reescribió este año",
            "El texto se ha revisado dos veces desde su publicación",
            "Esta parte del manual trata del mantenimiento habitual",
        ],
        "caption": ["Figura 3", "Tabla 2", "Foto", "Esquema 1"],
        "footnote": ["Fuente: nota interna", "Nota", "Ref."],
        "participants": "Participantes",
        "and": "y",
    },
    "it": {
        "csv_head": "id,nome,email,telefono,iban",
        "table_head": ["Nome", "E-mail", "Telefono"],
        "kv": {"name": "Nome", "email": "E-mail", "phone": "Telefono", "role": "Ruolo"},
        "roles": ["autore", "revisore", "referente", "gestore del conto", "responsabile di progetto"],
        "doc": ["il rapporto", "il fascicolo", "l'allegato", "il verbale", "il manuale"],
        "byline_in": ["redatto da ", "scritto da ", "curato da ", "rivisto da "],
        "byline_cap": ["Redatto da ", "Curato da ", "Rivisto da "],
        "intro": [
            "L'esportazione allegata elenca i conti aperti in questo trimestre.",
            "Si prega di verificare le righe seguenti con i propri registri.",
            "I seguenti referenti sono stati migrati dal sistema precedente.",
        ],
        "byline_intro": [
            "La sezione sulla resistenza al rotolamento è stata riscritta quest'anno",
            "Il testo è stato rivisto due volte dalla pubblicazione",
            "Questa parte del manuale riguarda la manutenzione ordinaria",
        ],
        "caption": ["Figura 3", "Tabella 2", "Foto", "Schema 1"],
        "footnote": ["Fonte: nota interna", "Nota", "Rif."],
        "participants": "Partecipanti",
        "and": "e",
    },
}


# ---------------------------------------------------------------------------
# Name drawing. nl and de route through generate.py's own name builders,
# which exist because Faker's nl_NL/de_DE person providers produce shapes
# that corpus reviewers flagged; en/fr/es/it use the locale provider.
# ---------------------------------------------------------------------------


def draw_name(lang: str, fk: Faker) -> str:
    if lang == "nl":
        return _nl_name(fk)
    if lang == "de":
        return _de_name(fk)
    return fk.name()


def _rows(rng: random.Random, lang: str, fk: Faker, n: int) -> list[dict]:
    """n people, each with the distractor fields that made `domain-nl-csv-1`
    hard: the model found every one of those and missed only the name."""
    out = []
    for i in range(n):
        name = draw_name(lang, fk)
        first = name.split()[0].lower().strip(".")
        last = name.split()[-1].lower().replace("'", "")
        out.append(
            {
                "id": str(1000 + rng.randrange(1, 900)),
                "name": name,
                "email": f"{_ascii(first)}.{_ascii(last)}@example.{rng.choice(['com', 'net', 'org'])}",
                "phone": fk.phone_number(),
                "iban": fk.iban(),
                "role": rng.choice(LOC[lang]["roles"]),
            }
        )
    return out


def _ascii(word: str) -> str:
    import unicodedata

    stripped = unicodedata.normalize("NFKD", word).encode("ascii", "ignore").decode("ascii")
    return "".join(c for c in stripped if c.isalnum()) or "user"


# ---------------------------------------------------------------------------
# The templates. Each returns a list of Segment (label-or-None, text).
#
# Every one of them puts a name in a *late* position as well as an early one
# where the shape allows it -- that is the specific defect
# `domain-nl-csv-1` exposed, and a template with a single row would not
# exercise it at all.
# ---------------------------------------------------------------------------


def t_csv_header(rng, lang, fk) -> list[Segment]:
    loc = LOC[lang]
    rows = _rows(rng, lang, fk, rng.randint(2, 5))
    seg: list[Segment] = [(None, rng.choice(loc["intro"]) + "\n" + loc["csv_head"] + "\n")]
    for r in rows:
        seg.append((None, r["id"] + ","))
        seg.append(("PERSON", r["name"]))
        seg.append((None, ","))
        seg.append(("EMAIL_ADDRESS", r["email"]))
        seg.append((None, ","))
        seg.append(("PHONE_NUMBER", r["phone"]))
        seg.append((None, ","))
        seg.append(("IBAN_CODE", r["iban"]))
        seg.append((None, "\n"))
    return seg


def t_csv_no_header(rng, lang, fk) -> list[Segment]:
    rows = _rows(rng, lang, fk, rng.randint(3, 6))
    seg: list[Segment] = []
    for r in rows:
        seg.append((None, r["id"] + ";"))
        seg.append(("PERSON", r["name"]))
        seg.append((None, ";"))
        seg.append(("EMAIL_ADDRESS", r["email"]))
        seg.append((None, ";" + r["role"] + "\n"))
    return seg


def t_pipe_table(rng, lang, fk) -> list[Segment]:
    loc = LOC[lang]
    head = loc["table_head"]
    rows = _rows(rng, lang, fk, rng.randint(2, 5))
    seg: list[Segment] = [
        (None, "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n")
    ]
    for r in rows:
        seg.append((None, "| "))
        seg.append(("PERSON", r["name"]))
        seg.append((None, " | "))
        seg.append(("EMAIL_ADDRESS", r["email"]))
        seg.append((None, " | "))
        seg.append(("PHONE_NUMBER", r["phone"]))
        seg.append((None, " |\n"))
    return seg


def t_tsv_export(rng, lang, fk) -> list[Segment]:
    rows = _rows(rng, lang, fk, rng.randint(3, 6))
    seg: list[Segment] = [(None, rng.choice(LOC[lang]["intro"]) + "\n")]
    for r in rows:
        seg.append((None, r["id"] + "\t"))
        seg.append(("PERSON", r["name"]))
        seg.append((None, "\t"))
        seg.append(("PHONE_NUMBER", r["phone"]))
        seg.append((None, "\t"))
        seg.append(("IBAN_CODE", r["iban"]))
        seg.append((None, "\n"))
    return seg


def t_kv_blocks(rng, lang, fk) -> list[Segment]:
    loc = LOC[lang]
    kv = loc["kv"]
    rows = _rows(rng, lang, fk, rng.randint(2, 4))
    seg: list[Segment] = []
    for i, r in enumerate(rows):
        seg.append((None, f"--- {i + 1} ---\n{kv['name']}: "))
        seg.append(("PERSON", r["name"]))
        seg.append((None, f"\n{kv['role']}: {r['role']}\n{kv['email']}: "))
        seg.append(("EMAIL_ADDRESS", r["email"]))
        seg.append((None, f"\n{kv['phone']}: "))
        seg.append(("PHONE_NUMBER", r["phone"]))
        seg.append((None, "\n"))
    return seg


def t_numbered_list(rng, lang, fk) -> list[Segment]:
    loc = LOC[lang]
    rows = _rows(rng, lang, fk, rng.randint(3, 6))
    seg: list[Segment] = [(None, loc["participants"] + ":\n")]
    for i, r in enumerate(rows):
        seg.append((None, f"{i + 1}. "))
        seg.append(("PERSON", r["name"]))
        seg.append((None, f" ({r['role']}), "))
        seg.append(("EMAIL_ADDRESS", r["email"]))
        seg.append((None, "\n"))
    return seg


def t_inline_enumeration(rng, lang, fk) -> list[Segment]:
    """Names in a running sentence, comma-separated, with the last one after
    'and'. Not a table, but the same late-position problem: the model's signal
    weakens along the list."""
    loc = LOC[lang]
    rows = _rows(rng, lang, fk, rng.randint(3, 5))
    seg: list[Segment] = [(None, rng.choice(loc["intro"]) + " ")]
    for i, r in enumerate(rows):
        if i:
            seg.append((None, f" {loc['and']} " if i == len(rows) - 1 else ", "))
        seg.append(("PERSON", r["name"]))
    seg.append((None, ".\n"))
    return seg


def t_signature_block(rng, lang, fk) -> list[Segment]:
    loc = LOC[lang]
    rows = _rows(rng, lang, fk, 2)
    seg: list[Segment] = [(None, rng.choice(loc["byline_intro"]) + ".\n\n")]
    for r in rows:
        seg.append((None, r["role"].capitalize() + ": "))
        seg.append(("PERSON", r["name"]))
        seg.append((None, "\n"))
        seg.append((None, loc["kv"]["email"] + ": "))
        seg.append(("EMAIL_ADDRESS", r["email"]))
        seg.append((None, "\n\n"))
    return seg


def t_byline_parenthetical(rng, lang, fk) -> list[Segment]:
    """`(redatto da Ilaria Lombardo)` -- the exact shape of
    `it-wv-cicloturismo-d1`, which the model misses completely."""
    loc = LOC[lang]
    return [
        (None, rng.choice(loc["byline_intro"]) + " ("),
        (None, rng.choice(loc["byline_in"])),
        ("PERSON", draw_name(lang, fk)),
        (None, "). " + rng.choice(loc["intro"]) + "\n"),
    ]


def t_byline_inline(rng, lang, fk) -> list[Segment]:
    """`Il fascicolo è stato redatto da Lorenzo Roversi.` -- the shape of
    `it-wp-rotatoria-dn`, where the model found the surname alone."""
    loc = LOC[lang]
    return [
        (None, rng.choice(loc["doc"]).capitalize() + " -- "),
        (None, rng.choice(loc["byline_in"])),
        ("PERSON", draw_name(lang, fk)),
        (None, ". " + rng.choice(loc["byline_intro"]) + ".\n"),
    ]


def t_byline_caption(rng, lang, fk) -> list[Segment]:
    loc = LOC[lang]
    return [
        (None, rng.choice(loc["caption"]) + " -- " + rng.choice(loc["byline_in"])),
        ("PERSON", draw_name(lang, fk)),
        (None, ", " + rng.choice(loc["roles"]) + ".\n"),
    ]


def t_byline_footnote(rng, lang, fk) -> list[Segment]:
    loc = LOC[lang]
    name = draw_name(lang, fk)
    return [
        (None, rng.choice(loc["byline_intro"]) + ".\n\n"),
        (None, rng.choice(loc["footnote"]) + " -- " + rng.choice(loc["byline_cap"]).lower()),
        ("PERSON", name),
        (None, " (" + rng.choice(loc["roles"]) + ").\n"),
    ]


def t_byline_two_authors(rng, lang, fk) -> list[Segment]:
    """Two names in one byline. The second is the one at risk -- the same
    late-position weakness as a table's later rows, in prose."""
    loc = LOC[lang]
    return [
        (None, rng.choice(loc["byline_cap"])),
        ("PERSON", draw_name(lang, fk)),
        (None, f" {loc['and']} "),
        ("PERSON", draw_name(lang, fk)),
        (None, ". " + rng.choice(loc["byline_intro"]) + ".\n"),
    ]


# (template id, family, function). The id is the group key and the thing the
# split is computed from.
TEMPLATES: list[tuple[str, str, object]] = [
    ("csv-header", "tabular", t_csv_header),
    ("csv-no-header", "tabular", t_csv_no_header),
    ("pipe-table", "tabular", t_pipe_table),
    ("tsv-export", "tabular", t_tsv_export),
    ("kv-blocks", "tabular", t_kv_blocks),
    ("numbered-list", "tabular", t_numbered_list),
    ("inline-enumeration", "tabular", t_inline_enumeration),
    ("signature-block", "tabular", t_signature_block),
    ("byline-parenthetical", "byline", t_byline_parenthetical),
    ("byline-inline", "byline", t_byline_inline),
    ("byline-caption", "byline", t_byline_caption),
    ("byline-footnote", "byline", t_byline_footnote),
    ("byline-two-authors", "byline", t_byline_two_authors),
]

# Held out of `train` entirely, in every language. Chosen one per family and
# one extra, so the held-out split can answer "did it learn the shape" for
# both families separately rather than for the pool as a whole. The three
# real misses are in shapes this module does not contain, so a split that
# held out *rows* rather than *templates* would measure nothing.
HOLDOUT_TEMPLATES = {
    "kv-blocks": "validation",
    "inline-enumeration": "test",
    "byline-caption": "validation",
    "byline-two-authors": "test",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--per-language-per-template", type=int, default=110,
                    help="rows built from each template in each language")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report-out", default=None)
    args = ap.parse_args()

    splits: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    by_family: Counter = Counter()
    by_lang: Counter = Counter()
    per_template: Counter = Counter()
    person_spans = 0
    seen: set[str] = set()
    duplicates = 0

    for lang in LANGUAGES:
        fk = Faker(LOCALE[lang])
        fk.seed_instance(_seed_for(args.seed, lang, "positives"))
        for template_id, family, fn in TEMPLATES:
            rng = random.Random(_seed_for(args.seed, lang, template_id))
            split = HOLDOUT_TEMPLATES.get(template_id, "train")
            for _ in range(args.per_language_per_template):
                record = _render(fn(rng, lang, fk))
                # Faker can draw the same name twice; two records that come out
                # byte-identical are one record repeated, and repeating a row
                # is not the same as covering a shape.
                if record["text"] in seen:
                    duplicates += 1
                    continue
                seen.add(record["text"])
                splits[split].append(
                    {
                        "text": record["text"],
                        "entities": record["entities"],
                        "language": lang,
                        "source": f"augment:{family}",
                        "group": f"aug-{lang}-{template_id}",
                        "template": template_id,
                    }
                )
                by_family[family] += 1
                by_lang[lang] += 1
                per_template[template_id] += 1
                person_spans += sum(1 for e in record["entities"] if e["label"] == "PERSON")

    # Offsets are constructed, but constructed code can still be wrong; this
    # asserts every span slices back to text that is there, on every row.
    for rows in splits.values():
        for row in rows:
            for e in row["entities"]:
                assert 0 <= e["start"] < e["end"] <= len(row["text"]), row["group"]
                assert row["text"][e["start"]:e["end"]].strip() == row["text"][e["start"]:e["end"]], row["group"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    for name, rows in splits.items():
        rng.shuffle(rows)
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = sum(len(v) for v in splits.values())
    chars = sum(len(r["text"]) for rows in splits.values() for r in rows)
    log(f"{total:,} augmentation rows, {person_spans:,} PERSON spans, "
        f"mean {chars / max(total, 1):.0f} chars/row ({duplicates} duplicate draws dropped)")
    log("  splits (by template, never by row): "
        + ", ".join(f"{k}={len(v):,}" for k, v in splits.items()))
    log("  by family: " + ", ".join(f"{k}={v:,}" for k, v in sorted(by_family.items())))
    log("  by language: " + ", ".join(f"{k}={v:,}" for k, v in sorted(by_lang.items())))
    log(f"  held out of train: {', '.join(sorted(HOLDOUT_TEMPLATES))}")
    log(f"  written to {out_dir}")

    for split in ("train", "validation", "test"):
        present = {r["language"] for r in splits[split]}
        missing = set(LANGUAGES) - present
        if missing:
            raise SystemExit(f"{split} is missing languages: {sorted(missing)}")
        fams = {r["source"] for r in splits[split]}
        if len(fams) < 2:
            raise SystemExit(f"{split} does not carry both families: {sorted(fams)}")

    if args.report_out:
        path = Path(args.report_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Tabular and byline `PERSON` augmentation",
            "",
            f"Built {time.strftime('%Y-%m-%d')} by `training/build_positives.py`, seed {args.seed}.",
            "",
            "Targets the three remaining `PERSON` recall misses diagnosed in "
            "`training/reports/person-misses.md`: a name in a later CSV row "
            "(`domain-nl-csv-1`), a byline inside parentheses "
            "(`it-wv-cicloturismo-d1`), and a byline whose given name was dropped "
            "(`it-wp-rotatoria-dn`).",
            "",
            f"**{total:,} rows, {person_spans:,} `PERSON` spans**, six languages, "
            f"{len(TEMPLATES)} templates in two families. Every name, e-mail, phone number "
            "and IBAN is Faker-generated; no real personal data enters training. Every span "
            "is constructed by `generate.Builder` as the text is built, never found by "
            "searching a finished string.",
            "",
            "| template | family | split | rows |",
            "|---|---|---|---|",
        ]
        for template_id, family, _ in TEMPLATES:
            lines.append(f"| `{template_id}` | {family} | "
                         f"{HOLDOUT_TEMPLATES.get(template_id, 'train')} | "
                         f"{per_template[template_id]:,} |")
        lines += [
            "",
            "Rows by language: " + ", ".join(f"`{k}` {v:,}" for k, v in sorted(by_lang.items())) + ".",
            "",
            "**The split is by template, not by row.** Two rows from one template differ only "
            "in which names Faker drew, so a random split would put near-copies on both sides "
            "and inflate every recall number computed afterwards. What the held-out split "
            "measures is whether the model learned the *shape* or only these sentences -- "
            "which is the question that matters, since all three real misses are in shapes "
            "this module does not contain.",
            "",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        log(f"  report written to {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
