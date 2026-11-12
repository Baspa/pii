"""Run Azure AI Language's Text PII API over the same 128 hand-written eval
cases eval_fixture.py runs the trained checkpoint over, and score it with
the exact same rules (see fixture_scoring.py) -- the other half of the first
honest comparison between the two, on text neither has seen.

THIS IS NOT A RELEASE GATE, either. 128 cases is small; a few points of
difference between the model and Azure here is noise, not a result. See
eval_fixture.py's docstring for the full context.

API CONTRACT. Read from, not guessed from, our earlier Azure client.
Specifically:
  - POST {endpoint}language/:analyze-text?api-version=2024-11-01
  - header Ocp-Apim-Subscription-Key: <key>
  - kind=PiiEntityRecognition, parameters.modelVersion=latest,
    parameters.stringIndexType=UnicodeCodePoint (Python string indices are
    already Unicode code points, so this needs no offset translation),
    parameters.loggingOptOut=true
  - at most DOCUMENTS_PER_REQUEST=5 documents per request (a hard 400 above
    that, not a truncation)
  - the fixture's "lang" field is sent as each document's "language"
  - every document id sent must come back or be listed in results.errors;
    a document silently missing from both must never be read as "found
    nothing", which is exactly how the composite driver could fail open

CATEGORIES below mirrors the category list of Temperate's earlier Azure client exactly: the Azure
PII endpoint returns categories beyond what Temperate's driver acts on (its own
docblock lists two -- VAT_NUMBER and DE_TAX_ID -- Azure cannot supply at
all), and counting every category Azure happens to return would score a
capability Temperate's product never ships. MIN_SCORE=0.5 mirrors config/pii.php's
`score_threshold` default (confirmed unset in the .env file, so this
is what production actually runs with). Both choices make this the honest
comparison -- what Temperate's Azure driver would actually decide to mask --
not "everything the raw API can be made to say."

CREDENTIALS. Read directly out of the .env file at call time and held
only in local variables. Never printed, never logged, never written to a
file this repository tracks, never included in the saved response JSON.

RESCORING WITHOUT CALLING AZURE AGAIN. --from-raw <path> loads a previous
run's "raw_batches" (the verbatim Azure API responses, already saved to
disk) and rescores them with the current rules in fixture_scoring.py,
without reading credentials or making any network request. This is how
this script was re-run after fixture_scoring.py's must_mask rule was
corrected from "fully covered by one span" (wrong -- see
fixture_scoring.py's docstring) to "every non-whitespace character covered
by the union of spans": Azure's own saved responses were rescored, not
re-fetched, both because a second live call is unnecessary and because it
would risk a credential touching a log or file it shouldn't.

Usage:
    python eval_fixture_azure.py \
        --fixture ../benchmark/cases \
        --env-path .env \
        --out reports/azure-fixture-2026-08-21.json

    python eval_fixture_azure.py \
        --from-raw reports/azure-fixture-2026-08-21.json \
        --out reports/azure-fixture-2026-08-21-rescored.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import requests

from fixture_scoring import DEFAULT_FIXTURE, load_cases, print_summary, score_case, summarize

DEFAULT_ENV_PATH = ".env"
API_VERSION = "2024-11-01"
DOCUMENTS_PER_REQUEST = 5
WINDOW_SIZE = 5_000  # Azure's synchronous per-document character ceiling; see Temperate's earlier Azure client
DEFAULT_MIN_SCORE = 0.5  # config/pii.php's score_threshold default

# Mirrors the category list of Temperate's earlier Azure client exactly (see docstring above).
CATEGORIES = {
    "Address": "STREET_ADDRESS",
    "CreditCardNumber": "CREDIT_CARD",
    "DEPassportNumber": "PASSPORT_NUMBER",
    "Email": "EMAIL_ADDRESS",
    "EUPassportNumber": "PASSPORT_NUMBER",
    "FRPassportNumber": "PASSPORT_NUMBER",
    "InternationalBankingAccountNumber": "IBAN_CODE",
    "IPAddress": "IP_ADDRESS",
    "NLCitizensServiceNumber": "BSN",
    "Person": "PERSON",
    "PhoneNumber": "PHONE_NUMBER",
    "USUKPassportNumber": "PASSPORT_NUMBER",
}


def read_env_credentials(env_path: str) -> tuple[str, str]:
    """AZURE_LANGUAGE_ENDPOINT and AZURE_LANGUAGE_KEY, read directly out of
    the .env file. Read-only; nothing here is ever printed, logged
    or written back out."""
    endpoint = key = None
    for line in Path(env_path).read_text().splitlines():
        line = line.strip()
        if line.startswith("AZURE_LANGUAGE_ENDPOINT="):
            endpoint = line.split("=", 1)[1].strip().strip('"').strip("'")
        elif line.startswith("AZURE_LANGUAGE_KEY="):
            key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not endpoint or not key:
        raise RuntimeError(f"AZURE_LANGUAGE_ENDPOINT/AZURE_LANGUAGE_KEY not found in {env_path}")
    if not endpoint.endswith("/"):
        endpoint += "/"
    return endpoint, key


def chunk(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def send_batch(endpoint: str, key: str, documents: list[dict], timeout: int) -> dict:
    payload = {
        "kind": "PiiEntityRecognition",
        "parameters": {
            "modelVersion": "latest",
            "stringIndexType": "UnicodeCodePoint",
            "loggingOptOut": True,
        },
        "analysisInput": {"documents": documents},
    }
    resp = requests.post(
        endpoint + f"language/:analyze-text?api-version={API_VERSION}",
        headers={"Ocp-Apim-Subscription-Key": key},
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def entity_spans(entities: list[dict], min_score: float) -> list[tuple[int, int]]:
    spans = []
    for e in entities:
        category = e.get("category")
        offset = e.get("offset")
        length = e.get("length")
        score = e.get("confidenceScore")
        if category not in CATEGORIES:
            continue
        if not isinstance(offset, int) or not isinstance(length, int) or length <= 0:
            continue
        if not isinstance(score, (int, float)) or score < min_score:
            continue
        spans.append((offset, offset + length))
    return spans


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def process_raw_batches(raw_batches: list[dict], warn) -> tuple[dict[str, list[dict]], set[str]]:
    """Turn a list of {"batch_index", "document_ids", "raw_response"} dicts
    -- the exact shape both a fresh API run and a previously-saved file use
    -- into (entities_by_id, model_versions). Shared by the live-call path
    and the --from-raw rescoring path so the two can never silently
    interpret the same response shape differently."""
    entities_by_id: dict[str, list[dict]] = {}
    model_versions: set[str] = set()

    for batch in raw_batches:
        i = batch["batch_index"]
        body = batch["raw_response"]
        results = body.get("results", {})
        model_version = results.get("modelVersion")
        if model_version:
            model_versions.add(model_version)

        errors = results.get("errors", [])
        error_ids = {e.get("id") for e in errors}
        if errors:
            warn(f"batch {i}: Azure returned errors for document ids {sorted(error_ids)}: {errors}")

        sent_ids = set(batch["document_ids"])
        answered_ids = set()
        for doc in results.get("documents", []):
            doc_id = doc.get("id")
            answered_ids.add(doc_id)
            entities_by_id[doc_id] = doc.get("entities", [])

        missing = sent_ids - answered_ids - error_ids
        if missing:
            raise RuntimeError(f"batch {i}: Azure never answered document ids {sorted(missing)} (not in documents or errors)")

    return entities_by_id, model_versions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", default=DEFAULT_FIXTURE, help="path to pii-eval-cases.json (read-only, never modified)")
    ap.add_argument("--env-path", default=DEFAULT_ENV_PATH, help=".env file to read Azure credentials from at run time (ignored with --from-raw)")
    ap.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE, help="minimum confidenceScore to act on (mirrors config/pii.php's score_threshold)")
    ap.add_argument("--timeout", type=int, default=10, help="per-request timeout in seconds (mirrors azure_timeout's default in config/pii.php); ignored with --from-raw")
    ap.add_argument("--from-raw", default=None, help="rescore a previous run's saved raw_batches instead of calling Azure again -- no credentials read, no network request")
    ap.add_argument("--out", required=True, help="path to write raw Azure responses + scored results (training/reports/azure-fixture-<date>.json)")
    args = ap.parse_args()

    cases = load_cases(args.fixture)

    if args.from_raw:
        prior = json.loads(Path(args.from_raw).read_text())
        if prior.get("n_cases") != len(cases):
            raise RuntimeError(
                f"--from-raw {args.from_raw!r} scored {prior.get('n_cases')} cases but "
                f"{args.fixture!r} has {len(cases)}; refusing to rescore a mismatched fixture"
            )
        raw_batches = prior["raw_batches"]
        api_version = prior.get("api_version", API_VERSION)
    else:
        endpoint, key = read_env_credentials(args.env_path)

        for case in cases:
            if len(case["text"]) > WINDOW_SIZE:
                raise RuntimeError(
                    f"case {case['id']!r} is {len(case['text'])} chars, over Azure's "
                    f"{WINDOW_SIZE}-char synchronous limit; this script does not "
                    "replicate Temperate's earlier Azure client's long-document windowing"
                )

        documents = [{"id": c["id"], "language": c["lang"], "text": c["text"]} for c in cases]
        batches = chunk(documents, DOCUMENTS_PER_REQUEST)

        raw_batches = []
        for i, batch in enumerate(batches):
            body = send_batch(endpoint, key, batch, args.timeout)
            raw_batches.append({"batch_index": i, "document_ids": [d["id"] for d in batch], "raw_response": body})
        api_version = API_VERSION

    entities_by_id, model_versions = process_raw_batches(raw_batches, warn)

    if len(model_versions) > 1:
        warn(f"Azure returned more than one modelVersion across batches: {sorted(model_versions)}")
    azure_model_version = sorted(model_versions)[0] if model_versions else None

    case_results = []
    for case in cases:
        raw_entities = entities_by_id.get(case["id"], [])
        spans = entity_spans(raw_entities, args.min_score)
        case_results.append(score_case(case, spans, warn))

    summary = summarize(case_results)

    total_characters = sum(len(c["text"]) for c in cases)
    total_documents = len(cases)
    estimated_text_records = sum(max(1, math.ceil(len(c["text"]) / 1000)) for c in cases)

    command = "python " + " ".join(sys.argv)
    output = {
        "command": command,
        "fixture": args.fixture,
        "api_version": api_version,
        "rescored_from": args.from_raw,
        "azure_model_version": azure_model_version,
        "min_score": args.min_score,
        "categories_map": CATEGORIES,
        "cost": {
            "total_documents": total_documents,
            "total_characters": total_characters,
            "estimated_text_records": estimated_text_records,
            "note": (
                "Azure bills PII detection per 1,000-character text record "
                "(ceil(chars/1000) per document, minimum 1). Azure AI Language's "
                "free tier covers 5,000 text records/month shared across "
                "sentiment/key-phrase/NER/PII; this run's "
                f"{estimated_text_records} records are within that allowance if "
                "unused elsewhere this month. List price beyond the free tier is "
                "on Microsoft's public pricing page "
                "(https://azure.microsoft.com/pricing/details/language/), which "
                "renders per-region/currency via JS and would not resolve to a "
                "single confirmed figure here -- reported as records, not a "
                "guessed dollar amount."
            ),
        },
        "n_cases": len(cases),
        "summary": summary,
        "cases": case_results,
        "raw_batches": raw_batches,
    }

    Path(args.out).write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Wrote {args.out}")
    print(f"Azure modelVersion: {azure_model_version}")
    print(f"Documents sent: {total_documents}, characters: {total_characters}, estimated text records: {estimated_text_records}")
    print_summary(summary)


if __name__ == "__main__":
    main()
