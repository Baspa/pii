# /// script
# requires-python = ">=3.10"
# dependencies = ["pypdf>=5"]
# ///
"""Redact personal data from PDFs or text files with a local Temperate PII service.

    uv run scripts/redact.py payslip-1.pdf payslip-2.pdf --out redacted/

Writes <name>.redacted.txt per input and mapping.private.json (placeholder -> original).
The same value gets the same placeholder in every file, so documents stay comparable.
"""
import argparse
import json
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

ENTITY_TYPES = [
    "PERSON", "STREET_ADDRESS", "IBAN_CODE", "CREDIT_CARD", "NATIONAL_ID", "TAX_ID",
    "VAT_NUMBER", "PASSPORT_NUMBER", "DRIVING_LICENCE", "PHONE_NUMBER", "EMAIL_ADDRESS",
    "IP_ADDRESS", "CREDENTIAL",
]

# Temperate PII has no type for the first two, and misses the last when it stands apart from a street.
EXTRA_PATTERNS = {
    "DATE_OF_BIRTH": re.compile(
        r"(?:geboortedatum|geb\.?\s*datum|date of birth|birth date|geburtsdatum)\s*:?\s*"
        r"(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})",
        re.IGNORECASE,
    ),
    "EMPLOYEE_NUMBER": re.compile(
        r"(?:personeelsn(?:umme)?r\.?|werknemersn(?:umme)?r\.?|medewerkersn(?:umme)?r\.?"
        r"|employee\s*(?:number|no\.?|id))\s*:?\s*([A-Z0-9][A-Z0-9-]{2,})",
        re.IGNORECASE,
    ),
    "POSTCODE": re.compile(r"(\b\d{4} ?[A-Z]{2}\b(?:[ \t]{1,3}[A-Z][\w'-]+(?: [A-Z][\w'-]+)?)?)"),
}
INITIALS_BEFORE = re.compile(r"(?:\b[A-Z]\. ?)+ ?$")


def read_text(path: Path) -> str:
    if path.suffix.lower() != ".pdf":
        return path.read_text()
    from pypdf import PdfReader

    pages = [page.extract_text(extraction_mode="layout") for page in PdfReader(path).pages]
    if not "".join(pages).strip():
        sys.exit(f"{path}: no text layer, this looks like a scan and needs OCR first")
    return "\n".join(pages)


def detect(texts: list[str], language: str, service: str) -> list[list[dict]]:
    body = json.dumps({"texts": texts, "entities": ENTITY_TYPES, "language": language}).encode()
    request = urllib.request.Request(
        f"{service}/analyze", data=body, headers={"content-type": "application/json"}
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)["results"]


def extra_spans(text: str) -> list[dict]:
    return [
        {"entity_type": entity_type, "start": match.start(1), "end": match.end(1), "score": 1.0}
        for entity_type, pattern in EXTRA_PATTERNS.items()
        for match in pattern.finditer(text)
    ]


def with_initials(text: str, span: dict) -> dict:
    if span["entity_type"] != "PERSON":
        return span
    initials = INITIALS_BEFORE.search(text[max(0, span["start"] - 20):span["start"]])
    return {**span, "start": span["start"] - len(initials.group(0))} if initials else span


def without_overlaps(spans: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for span in sorted(spans, key=lambda s: (s["start"], s["start"] - s["end"])):
        if not kept or span["start"] >= kept[-1]["end"]:
            kept.append(span)
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("redacted"))
    parser.add_argument("--language", default="nl")
    parser.add_argument("--service", default="http://127.0.0.1:8090")
    parser.add_argument("--show-values", action="store_true", help="print the originals next to their placeholders")
    args = parser.parse_args()

    texts = [read_text(path) for path in args.files]
    results = detect(texts, args.language, args.service)

    placeholders: dict[tuple[str, str], str] = {}
    counters: Counter[str] = Counter()
    mapping: dict[str, str] = {}
    args.out.mkdir(parents=True, exist_ok=True)

    for path, text, found in zip(args.files, texts, results):
        spans = without_overlaps([with_initials(text, span) for span in found] + extra_spans(text))
        redacted = text
        for span in reversed(spans):
            value = text[span["start"]:span["end"]]
            key = (span["entity_type"], re.sub(r"\W", "", value.lower()))
            if key not in placeholders:
                counters[span["entity_type"]] += 1
                placeholders[key] = f"[{span['entity_type']}_{counters[span['entity_type']]}]"
                mapping[placeholders[key]] = value
            span["placeholder"] = placeholders[key]
            redacted = redacted[:span["start"]] + span["placeholder"] + redacted[span["end"]:]

        (args.out / f"{path.stem}.redacted.txt").write_text(redacted)
        print(f"\n{path.name}: {len(spans)} values masked")
        for span in spans:
            shown = f"  {text[span['start']:span['end']]!r}" if args.show_values else ""
            print(f"  {span['placeholder']:<22} score {span['score']:.2f}{shown}")

    (args.out / "mapping.private.json").write_text(json.dumps(mapping, indent=2, ensure_ascii=False))
    print(f"\nRedacted text in {args.out}/. Read it before you send it anywhere; mapping.private.json stays local.")


if __name__ == "__main__":
    main()
