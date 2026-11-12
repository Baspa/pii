# Licensing notice for the benchmark carrier corpus

**Scope.** This notice governs, and only governs:

- `benchmark/carriers/**` -- the carrier text (`*.txt`), `manifest.json`, `README.md`
- `benchmark/cases/carriers.json` -- the cases derived from that text

Everything else in this repository is unaffected and remains under the terms declared in the
workspace manifest (`license = "UNLICENSED"`). This is a scoped notice, not a relicensing of
the project.

## Why this file exists

243 of the 321 carriers are text from Wikimedia projects, licensed **CC BY-SA 4.0**. The text
here is not a verbatim copy: it has been excerpted, screened (proper nouns substituted) and
had synthetic identifiers injected into it. Under CC BY-SA 4.0 §3(b) that makes it Adapted
Material, and distributing Adapted Material carries a ShareAlike obligation -- the adaptation
must itself be offered under the same licence, or a BY-SA-compatible one.

Attribution alone, which the project already provides per carrier, satisfies §3(a) but not
§3(b). The obligation binds on distribution; recording it here is what keeps first
distribution from being a breach. For a compliance product, the argument for getting this
right is the product's own argument.

## The terms

### Wikimedia-derived carriers -- CC BY-SA 4.0

The following files, and the portions of `benchmark/cases/carriers.json` derived from them,
are Adapted Material of works licensed under the Creative Commons
Attribution-ShareAlike 4.0 International licence, and are distributed under that same
licence: <https://creativecommons.org/licenses/by-sa/4.0/>.

- every `benchmark/carriers/*-wp-*.txt` (Wikipedia, 123 files)
- every `benchmark/carriers/*-wb-*.txt` (Wikibooks, 63 files)
- every `benchmark/carriers/*-wv-*.txt` (Wikivoyage, 57 files)

These counts are part of the notice, not decoration: a carrier added without updating them
leaves a CC BY-SA file distributed under a notice that does not list it. `manifest.json` is
the authority on which file came from where, and a carrier addition that changes any count
here must change it in the same commit.

**Attribution.** `benchmark/carriers/manifest.json` and `benchmark/carriers/README.md` carry,
for every one of these files, the URL of the source article. Each such URL reaches the
article's history page, which lists its contributors -- the form of attribution the Wikimedia
Foundation Terms of Use §7 name as sufficient.

**Indication of changes.** Every file was excerpted from a longer article; some had proper
nouns substituted during screening; some had MediaWiki maintenance tags (`[citation needed]`
and its per-language equivalents) and bracketed reference markers removed with the extraction;
the carriers added on 2026-08-26 additionally had the zero-width characters that stand where a
reference superscript was (`U+200B` and friends) removed, so that no `must_mask` substring can
contain a character a reader cannot see; all had synthetic identifiers injected. The per-carrier change record is
`benchmark/carriers/README.md` (screening) and `benchmark/cases/carriers.json` (injection).
The substituted originals are deliberately not recorded -- see the README for why. For carriers
added on 2026-08-22 and 2026-08-26, `manifest.json` also records the `oldid` of the source
revision, so the adapted work identifies the exact version it adapts. On 2026-11-11 every em dash in the carrier texts and in the cases made from them was replaced by an en dash, one character for one.

### EUR-Lex carriers -- Commission Decision 2011/833/EU

`benchmark/carriers/*-el-*.txt` (42 files) are excerpts of documents published on EUR-Lex.
© European Union, 1998–2026. Reuse is authorised for commercial and non-commercial purposes
under Commission Decision 2011/833/EU, provided the source is acknowledged and any changes
are indicated. Source and changes are recorded in the same two places as above. Decision
2011/833/EU imposes no ShareAlike obligation.

The 18 carriers added on 2026-08-26 were fetched from the Publications Office's Cellar
endpoint (`https://publications.europa.eu/resource/celex/{CELEX}`, content-negotiated per
language) rather than from `eur-lex.europa.eu`, which now serves a JavaScript challenge to a
plain HTTP client. Same documents, same publisher, same legal notice; `manifest.json` records
the canonical EUR-Lex URL and `benchmark/carriers/README.md` records the endpoint the bytes
came from.

### European Central Bank carriers -- the ECB's own reuse notice

`benchmark/carriers/*-eb-*.txt` (36 files) are excerpts of the **ECB Annual Report 2024**,
published by the European Central Bank in HTML in each of the six languages.
© European Central Bank, 2025. The ECB's website copyright notice permits reuse of information
published on its site under four conditions: the information must appear accurately, the ECB
must be cited as the source, any modification must be disclosed, and -- where the information
is incorporated into a document that is sold, in any medium -- the person publishing it must
tell buyers, before they pay any subscription or fee and again each time they access the
information, that it is available free of charge on the ECB's site:
<https://www.ecb.europa.eu/services/disclaimer/html/index.en.html>.

The fourth condition (the sold-document disclosure) is not triggered by anything this project
does today: this corpus is not sold, and nothing here incorporates ECB information into a
document offered for sale. Recorded here so the transcription is complete, not because it
currently binds.

Source and modifications are recorded in the same two places as above: `manifest.json` names
the URL and the section number of every excerpt, and `carriers.json` records every injected
value. The modifications are excerpting a contiguous run of paragraphs, removing footnote
reference markers (the ECB renders them `[77]`, in square brackets), and injecting synthetic
identifiers.

Commission Decision 2011/833/EU is **not** the instrument here. It governs Commission
documents; the ECB publishes under its own notice, and an earlier draft of
`benchmark/carriers/README.md` named the wrong one. The ECB notice imposes no ShareAlike
obligation. Its one carve-out -- documents carrying named authors, such as working papers,
which need prior written authorisation -- does not reach the Annual Report, which is an
institutional publication with no named author.

### The injected content

The synthetic identifiers, the sentences built around them, the `must_mask`/`must_clear`
annotations and the tooling in `crates/eval/` are this project's own work and are not
CC BY-SA material. Where they are inseparably combined with CC BY-SA text in the same file
(`carriers.json`, the `*.txt` files above), the file as a whole is offered under CC BY-SA 4.0,
which is the outcome the ShareAlike term requires.

## Practical consequence

If this corpus is distributed -- vendored into another repository, published, or shipped to a
customer -- the recipient receives the files listed above under CC BY-SA 4.0 and must be told
so. Nothing here requires the rest of the detector to be licensed that way; the two are separate
files and separately licensed. Keeping them separable is why the carrier text lives in its own
directory rather than being inlined into source.
