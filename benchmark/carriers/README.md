# Carriers -- real licensed text with synthetic identifiers injected at known positions

Each `*.txt` here is a passage of real, publicly licensed prose. `manifest.json` records
its provenance and its screening record; `crates/eval/src/bin/inject.rs` reads both
and writes `benchmark/cases/carriers.json`. Nothing in this directory is hand-annotated:
the only personal data in a carrier case is what the injector put there, plus the synthetic
names screening substituted in, and both are known exactly because they were written in.

**Licensing of this directory is not the repository's default.** 243 of the 321 carriers are
CC BY-SA 4.0 material that this project adapted, which carries a ShareAlike obligation on
distribution. See [`../LICENSE-CARRIERS.md`](../LICENSE-CARRIERS.md), which scopes the terms
to `benchmark/carriers/**` and `benchmark/cases/carriers.json` and leaves the rest of the detector
alone.

## Why real text, and why synthetic identifiers

The held-out split used while training the model came from the same generators as the
training data, so it shares their blind spots. It scored 0.9940 on person names and gave no
warning of a regression the hand-written fixture caught immediately: a benchmark built by the
process that built the training data certifies its own siblings. So the carrier text is real
documents from outside **both** training corpora (`ai4privacy`, `nym`, and this project's own
`training/generate.py`), and the identifiers inside it are synthetic. Real prose gives real
structure; synthetic identifiers give exactly known positions.

That independence has to hold for the identifiers too, not only for the text. See
["Every identifier satisfies its real national rule"](#every-identifier-satisfies-its-real-national-rule)
 --  an earlier version of this corpus inherited the training generator's own blind spot and
would have marked a stricter detector wrong.

## READ THIS BEFORE QUOTING A PER-LANGUAGE NUMBER FROM THIS BENCHMARK

The plan asks for **40–60 carriers per language**. This directory now holds **40 to 66 per
language, 321 in total**: every language is at or above our floor for the first time,
and the three thinnest are above its ceiling. It held 9 per language until 2026-08-22 and
24 to 28 until 2026-08-26.

| | plan | delivered | gap |
|---|---|---|---|
| carriers per language | 40–60 | 40 (`en`), 44 (`nl`), 47 (`de`), 61 (`fr`), 63 (`it`), 66 (`es`) | none; `fr`/`it`/`es` sit above the band |
| carriers in total | 240–360 | 321 | none |
| registers per language | 4 | 5 | none; business is present |
| languages | 6 | 6 | none |

**Any per-language recall or precision figure computed from these cases rests on 40 to 66
documents.** A single unlucky carrier moves a per-language number by **1.5% to 2.5%** of its
own weight -- 1/66 for `es`, 1/40 for `en`. It was ~4% (1/24 to 1/28) before the 2026-08-26
expansion and ~11% (1/9) before the one on 2026-08-22. Say so wherever such a figure is
quoted, including in the gate report. `pii_detection_eval::gates::SAMPLE_SIZE_CAVEAT` carries
the same sentence and is folded into every per-language gate `evaluate` produces.

**What is still small is the per-type denominator, not the per-language one, and closing the
first did not close the second.** Nine of the eleven declared types are carried by every
language, but `NATIONAL_ID` (n=118) appears only in Dutch and English carriers and `TAX_ID`
(n=120) only in German and English ones, because the Dutch BSN and the German Steuer-ID are
the two country packs that exist. One value is 0.8% of either. A per-type gate that turns on a
single value is exactly as fragile as it was before this expansion; only the per-language
gates got sturdier.

### What the expansion bought, in the denominators that matter

`must_mask` values per language across the whole fixture, counting every declared value of
every one of the eleven types the deployed service advertises -- the denominator every
per-language recall figure in the gate report is divided by:

| language | 154 carriers (2026-08-22) | 321 carriers (2026-08-26) | |
|---|---|---|---|
| `de` | 184 | **308** | ×1.67 |
| `en` | 218 | **320** | ×1.47 |
| `es` | 138 | **306** | ×2.22 |
| `fr` | 152 | **308** | ×2.03 |
| `it` | 145 | **301** | ×2.08 |
| `nl` | 198 | **306** | ×1.55 |
| total | 1,035 | **1,849** | ×1.79 |

**These are not the numbers the 2026-08-22 table quoted, and the difference is the counting
rule, not the fixture.** That table counted only the nine types a rules-only deployment was
asked for plus the untyped legacy values (413 → 790); the deployed service is model-backed and
advertises all eleven, so the honest denominator today includes `PERSON` and `STREET_ADDRESS`.
The two tables are not comparable and the older one is not reproduced here.

Spanish and Italian were the point again, as they were in August. At n=138 a two-point quality
change is three values; at n=306 it is six, and the figure no longer moves a point on one
document. Every language is now above 300, which was the target this expansion was
commissioned against, and the tightest is `it` at 301.

Per type, whole fixture: `CREDIT_CARD` 67→139, `EMAIL_ADDRESS` 85→155, `IBAN_CODE` 72→144,
`IP_ADDRESS` 70→140, `NATIONAL_ID` 68→118, `PASSPORT_NUMBER` 66→136, `PERSON` 111→182,
`PHONE_NUMBER` 78→149, `STREET_ADDRESS` 133→275, `TAX_ID` 65→120, `VAT_NUMBER` 68→139, plus
152 untyped legacy values that did not change. `must_clear` went 1,106 → 2,108.

**This time the Azure baseline grew with the fixture.**
`benchmark/baseline/azure-2026-08-26.json` was collected over all 1,150 cases for $2.17, so
both sides of every non-inferiority gate describe the same population and `eval`'s
`load_baseline` size guard passes rather than refusing the comparison. The superseded
`azure-2026-08-23.json` (649 cases) and `azure-2026-08-22.json` (349 cases) are kept for
provenance and must not be quoted as the current baseline.

## Which registers are here

The plan names four registers: legal, encyclopedia, technical documentation, and **business**
 --  company filings and annual reports. All four are now present. A fifth the design mentions,
public support-forum threads, is still absent and still argued for below.

| plan's register | delivered | |
|---|---|---|
| EUR-Lex judgments and regulations | legal / regulatory, 7 per language (42 total) | as asked |
| Wikipedia articles on mundane subjects | encyclopedia, 13–27 per language (123 total) | as asked |
| technical documentation | instructional / technical (Wikibooks), 7–13 per language (63 total) | as asked |
| public company filings and annual reports | **business / annual report (ECB Annual Report 2024), 6 per language (36 total)** | **added 2026-08-22, widened 2026-08-26** |
| -- | practical travel guide (Wikivoyage), 7–14 per language (57 total) | extra, kept |
| public support-forum threads (chat) | absent | see limitations |

The encyclopedia register grew fastest and is now 38% of the corpus, up from 34%. That is a
real shift in the corpus's centre of gravity and it is recorded rather than smoothed over:
Wikipedia is the only source with enough impersonal, licence-clean articles in all six
languages to close a per-language denominator on this scale, and the alternative -- taking many
more slices from the same handful of EUR-Lex acts and one annual report -- would have bought
document count without buying document diversity.

### The business register, and what it still does not cover

The previous version of this corpus substituted travel guide for business and disclosed the
substitution. The substitution is now filled: **36 carriers, six per language, are sections
of the European Central Bank's Annual Report 2024**, published by the ECB in HTML in all six
languages. Six sections, the same six in every language, so a per-language difference in
this register cannot be an artefact of the text differing:

| id suffix | section | added |
|---|---|---|
| `eb-ar24-balance` | 2.2 -- normalising the Eurosystem balance sheet | 2026-08-22 |
| `eb-ar24-target` | 4.1 -- TARGET Services | 2026-08-22 |
| `eb-ar24-borrowing` | 5.2 -- administration of EU borrowing and lending operations | 2026-08-22 |
| `eb-ar24-cash` | 6.1 -- cash circulation and handling | 2026-08-22 |
| `eb-ar24-stability` | 3.1 -- the financial stability environment in 2024 | 2026-08-26 |
| `eb-ar24-cashusage` | 6.2 -- cash usage by consumers and companies | 2026-08-26 |

**Section 12, "Meet our people", was never a candidate and is worth naming as the one that was
not taken.** It is five staff profiles under five real names -- the exact shape a carrier must
not have -- and it sits in the same document as the six sections that were. Selection did the
work here, as it did everywhere else in this corpus.

This is genuine annual-report prose: balance-sheet items, outstanding nominal amounts,
year-on-year percentages, instrument acronyms, reporting dates. It is **not** the whole of
what the design meant by the business register. The earlier disclosure said business documents
have "contact blocks, tabular fragments, registration and VAT lines in headers and footers,
addresses in signature position rather than in prose", and an annual report's narrative
sections have none of those. What is closed is the register's *language*; what is still open
is its *layout*. A carrier corpus cannot close the layout gap on its own -- the shapes it
names are letterhead and signature blocks, which is the hand-written cases' hand-authored territory, and
which is now the honest statement of what remains rather than a claim the gap is gone.

**Why the ECB and not the European Court of Auditors.** The earlier note named both. The ECA's
annual report is published in the Official Journal, but its EUR-Lex rendering could not be
located by CELEX within this task, while the ECB publishes its Annual Report as HTML in every
official language under one URL pattern. The ECB was taken because it was reachable, not
because it is better; the ECA remains a good source for a later addition.

**A correction to the earlier note.** It said the ECA and ECB annual reports were reusable
"under Decision 2011/833/EU". That is wrong for the ECB: Commission Decision 2011/833/EU
governs Commission documents, and the ECB publishes under its own copyright notice. The
notice is permissive and is recorded per carrier -- see the next section -- but recording the
wrong instrument in a compliance product is exactly the kind of error this corpus exists not
to make, so it is corrected rather than quietly reused.

## Licences, established rather than assumed

A carrier whose licence could not be established did not go in. Outbound obligations are in
[`../LICENSE-CARRIERS.md`](../LICENSE-CARRIERS.md); what follows is the inbound position.

**Wikimedia projects** (Wikipedia, Wikibooks, Wikivoyage), 243 carriers. Text contributed
since June 2009 is licensed CC BY-SA 4.0; the dual GFDL option covers only text imported from
before that change, so **CC BY-SA 4.0 is the licence relied on here** and the one recorded per
carrier in `manifest.json`. Attribution is satisfied by a URL to the article, whose history
page lists the contributors; `manifest.json` carries that URL for every carrier, and for every
carrier added on 2026-08-22 or 2026-08-26 it also carries the `oldid` of the exact revision the
text was taken from. The 2026-08-26 carriers were retrieved through the MediaWiki action API
(`action=query&prop=extracts|revisions&explaintext=1`), which returns the plain-text rendering
and the current revision id in one response, so the recorded `oldid` is the revision the text
actually came from rather than one looked up separately afterwards. Source: Wikimedia Foundation Terms of Use §7,
<https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use>, read 2026-08-21.

**EUR-Lex**, 42 carriers. Reuse of documents published on EUR-Lex is authorised for commercial
and non-commercial purposes under Commission Decision 2011/833/EU, provided the source is
acknowledged and any changes are indicated. © European Union, 1998–2026. The changes made here
are: extraction of a contiguous run of paragraphs from the HTML rendering, removal of footnote
reference markers (EUR-Lex renders them `( 25 )`, with inner spaces, which is what
distinguishes them from an act's own `(1)` paragraph numbering), and the injections listed in
`benchmark/cases/carriers.json`. Source:
<https://eur-lex.europa.eu/content/legal-notice/legal-notice.html>, read 2026-08-21.

**How the 2026-08-26 EUR-Lex carriers were retrieved, since it is not the URL in the
manifest.** `eur-lex.europa.eu` now serves a JavaScript challenge to a plain HTTP client, so
the text was fetched from the Publications Office's own Cellar endpoint for the same CELEX
document -- `https://publications.europa.eu/resource/celex/{CELEX}`, content-negotiated with
`Accept: text/html` and `Accept-Language: {nld,deu,fra,spa,ita,eng}` -- which returns the same
official rendering under the same legal notice. The manifest records the canonical EUR-Lex URL
because that is the address a reader should be sent to; this paragraph records the address the
bytes actually came from, because the two are not the same and pretending otherwise would be a
provenance claim nobody could reproduce. Each carrier is one numbered article, located by its
own heading, so the six language versions are the same article rather than the same offset.

**European Central Bank**, 36 carriers. The ECB's website copyright notice permits reuse of
information published on the site on four conditions: the information must appear accurately,
the ECB must be cited as the source, **any modification must be disclosed**, and -- where the
information is incorporated into a document that is sold, in any medium -- the publisher must
tell buyers, before they pay any subscription or fee and again each time they access the
information, that it is available free of charge on the ECB's site. The first three are met
here -- the source URL and the section number are recorded per carrier in `manifest.json`, the
modifications are the same three as for EUR-Lex (contiguous extraction, footnote marker
removal -- the ECB renders them `[77]`, in square brackets -- and injection), and they are
recorded in the same two places. The fourth is not triggered:
this corpus is not sold, and nothing here incorporates ECB information into a document offered
for sale; it is recorded so the transcription is complete, not because it currently binds. The
notice's one carve-out is for documents carrying named authors, such as working papers, which
need prior written
authorisation; the Annual Report is an institutional publication with no named author and is
not one of those. Source:
<https://www.ecb.europa.eu/services/disclaimer/html/index.en.html>, read 2026-08-22.
Decision 2011/833/EU does **not** apply to the ECB and is not relied on for these carriers.

### Two further modifications, disclosed because they are modifications

Both apply to the 167 carriers added on 2026-08-26 and to no earlier one, and both are removals
of characters that are not part of the prose:

- **Maintenance tags.** `[cita requerida]`, `[réf. nécessaire]`, `[citation needed]` and their
  equivalents are editorial annotations the plain-text rendering inlines into the sentence.
  They are stripped, on the same reasoning as a footnote marker: they are not something the
  article says.
- **Invisible characters.** The plain-text rendering carries the zero-width characters
  (`U+200B` and friends) that stand where a reference superscript was. They are stripped
  because a carrier must not contain a character a reader cannot see while an offset still
  counts it -- a `must_mask` value is a literal substring, and an invisible character inside one
  would make the answer key disagree with anything a human checked by eye.

Neither removal changes a word. The extractor additionally *refuses* a passage rather than
repairing it when it finds a coordinate pair left by a stripped listing template, Wikivoyage
listing furniture (`(updated Jun 2018)`, a numbered listing marker at line start), or a
sentence boundary the plain-text renderer swallowed -- a full stop with no space after it,
followed by a capital. Three candidates were dropped on exactly those grounds and one more had
its first window refused for a swallowed boundary before its second was rejected for personal
names; see the screening log below.

## Screening: proper nouns that were already in the text

The answer key knows only what was injected. A real name left in a carrier would be found by
The detector and scored as a false positive -- the benchmark would punish the model for being
right, and the gates would tighten against a fiction. The stronger reason is the second one:
a fixture full of real people's names would put real personal data in the repository of a
privacy product.

So every carrier was read in full, and every personal name in one was **replaced** with a
synthetic equivalent of the same shape and register -- never deleted, because removing a name
leaves a sentence that no longer reads like the register it came from, which is the one thing
carriers are for.

### The 2026-08-22 expansion: selection did the work, and here is the log

The 100 carriers added on 2026-08-22 needed **zero** replacements. That is not because the
screen was skipped. Every candidate was swept mechanically (emails, URLs, IP addresses, long
digit runs, IBAN shapes, honorifics, name particles, street-and-number patterns, postcodes,
every capitalised token that was not sentence-initial) and then read in full, and **ten
candidates were rejected or re-cut** because of what the screen found:

| candidate | what the screen found | what was done |
|---|---|---|
| `fr.wikivoyage` *Voyager en autocar en Europe* | a company e-mail address, a telephone number, a street address and two postcodes in a listing block | rejected; replaced by *Prendre la route* |
| `nl.wikibooks` *Linux Systeembeheer/Webservers* | a private IPv4 address and a named individual | rejected; replaced by *Opslagapparaten beheren* |
| `nl.wikipedia` *Optische vezel* | five named scientists in one paragraph | rejected; replaced by *Warmte-isolatie* |
| `fr.wikipedia` *Béton de ciment* | two named inventors | rejected; replaced by *Assainissement* |
| `it.wikipedia` *Tegola* | a named manufacturer-founder | rejected; replaced by *Isolamento termico*, then *Trattamento delle acque reflue* |
| `it.wikipedia` *Calcestruzzo* | a named Roman author | rejected; replaced by *Malta (materiale)* |
| `en.wikipedia` *Lock (water navigation)* | two named historical figures | rejected; replaced by *Thermal insulation*, then *Ventilation (architecture)* |
| `nl.wikipedia` *Asfalt* | two named chemists in the history paragraphs | **re-cut**: the excerpt window was moved past them, to the material-composition paragraphs |
| `en.wikipedia` *Asphalt concrete* | a named Babylonian king | **re-cut**: excerpt shortened to the three paragraphs before him |
| `es.wikipedia` *Esclusa* | a named Ptolemaic ruler | **re-cut**: excerpt shortened to the three paragraphs before him |

Four further candidates were dropped for extraction artefacts rather than for personal data  -- 
two `it.wikibooks` chapters that began mid-formula, and two articles whose paragraphs broke
around stripped TeX. That is why two rows above name a second replacement. The extractor now
refuses any paragraph that starts on punctuation or contains a `\displaystyle` block, so that
class fails loudly instead of being read past.

Re-cutting is a third option beside replace and delete, and it has its own cost, stated here
rather than left implicit: **it biases the corpus away from historical and attributive prose**,
which is precisely the prose personal names live in. The corpus is therefore more
structurally impersonal than a random sample of the same sources would be, and its zero
false-positive result (below) is correspondingly easier than production traffic. Replacement,
not re-cutting, is the right tool when the sentence itself is worth keeping; re-cutting was
used only where a neighbouring, equally representative run of the same document existed.

### The 2026-08-26 expansion: the same screen, at 167 carriers, and here is that log

The 167 carriers added on 2026-08-26 needed **zero** replacements, for the same reason as the
100 before them: selection, not luck. Every candidate window was swept mechanically for e-mail
addresses, URLs, IPv4 literals, digit runs of seven or more, IBAN shapes, honorifics followed
by a capital, name particles followed by a capital, street-and-number patterns, postcode
shapes and international dialling patterns; a window with any of those was discarded outright.
Every surviving window's non-sentence-initial capitalised tokens were listed, and **every one
of the 167 carriers was then read in full before it went in**.

That read is what caught the cases the mechanical sweep could not. Eight candidates were
rejected or moved after it:

| candidate | what the read found | what was done |
|---|---|---|
| `de.wikipedia` *Polyethylen* | first window carried a swallowed sentence boundary; the next one names six chemists across the discovery paragraphs | rejected; replaced by *Torf* |
| `es.wikipedia` *Aluminio* | three named chemists, including the two the process is called after | rejected; replaced by *Reciclaje* |
| `fr.wikipedia` *Fermentation* | a named 19th-century chemist | rejected; replaced by *Cuivre* |
| `de.wikivoyage` *Frachtschiffreise* | a proverb attributed to a named historical figure | **re-cut** to the neighbouring run (cabin accommodation), which carries none |
| `es.wikivoyage` *Registros geográficos* | coordinate pairs left behind by stripped listing templates | rejected; replaced by *Frutas y verduras* |
| `en.wikivoyage` *Algal bloom* | numbered listing entries and `(updated Jun 2018)` markers | rejected; replaced by *Airline alliances* |
| `it.wikibooks` *Impiantazione ionica* | a sentence boundary the plain-text renderer had swallowed | rejected; replaced by *Il vuoto* |
| `es.wikibooks` *Almacenamiento óptico* (and two more) | the same chapter as an existing carrier under the book's new title | rejected as duplicates of committed text |

Only **two** of the 167 carriers use anything other than the first clean window of their source
document, which is a much smaller re-cutting bias than the 2026-08-22 round carried. The two
are `de-wv-frachtschiffreise` and `es-wp-biogas`.

**A rule this round applied that the last one did not.** Where the first clean window of an
article contained a personal name, the article was *dropped* rather than advanced to a later
window. Advancing was tried first and was worse: a Wikipedia article that names one chemist in
its opening usually names five in the paragraph after, so the second window is more
biographical than the first, not less. The three rejections at the top of the table are exactly
that discovery. This is a deliberate divergence from "prefer replacement", and its cost is the
same bias re-cutting has -- recorded as limitation 10 rather than argued away.

### The one rule about surnames

A surname can appear in a carrier in two ways and they are scored differently:

- **A personal name** -- someone being referred to as a person -- is `must_mask` with type
  `PERSON`. It is personal data and the detector must find it.
- **An eponym** -- a surname that has become the name of a device, a place or a technical
  term -- is `must_clear`. Masking it destroys a technical term and leaks nothing, which is
  the retired `LOCATION` failure exactly.

The rule is stated once and applied everywhere because an earlier version of this corpus did
not have one: `en-wp-windturbine` demanded two eponymous rotor names be left alone while
`fr-wp-eolienne` demanded an eponymous turbine name be masked. No detector can satisfy both,
so one of the two was always scored wrong however good the detector was.

Where a carrier used the *same* surname for both -- a person and the device named after them  -- 
the two are given different synthetic names, because one string cannot be in `must_mask`
and `must_clear` at once. `fr-wp-eolienne` is the case: the inventor and the turbine now
carry different names. It reads as a person who invented a device named for somewhere else,
which is odd but not impossible, and it is the price of the fixture stating one rule.

In `manifest.json` a replacement with a `type` becomes a `must_mask` value; a replacement
without one becomes a `must_clear` value. `inject` routes them; it does not decide.

#### Eponyms are named, not replaced, from 2026-08-22 onward

The first 54 carriers *replaced* their eponyms as well as their personal names -- `Darrieus`
became `Marnay`, `Savonius` became `Lehtinen` -- as a side effect of one uniform "substitute
every proper noun" pass. The four eponyms in the 100 new carriers are kept as they are and
named in `must_clear` instead:

| carrier | eponym in `must_clear` |
|---|---|
| `en-wb-bike-cables` | `Bowden cable` |
| `es-wp-bomba-de-calor` | `Ciclo Rankine` |
| `fr-wp-pompe-a-chaleur` | `effet Joule` |
| `nl-wp-warmte-isolatie` | `de wetten van Fourier` |

The argument, because a divergence from the earlier treatment needs one. Replacement exists to
solve two problems. The first is scoring: a name the detector finds and the answer key does not
mention is counted a false positive. Naming the eponym in `must_clear` solves that completely
and more directly -- the key now says "leave this alone", the detector leaving it alone is
scored correct, and masking it is scored a `must_clear` violation, which is the behaviour that
actually deserves to be marked wrong. The second is not putting real personal data in the
repository, and `Bowden cable`, `Ciclo Rankine`, `effet Joule` and `de wetten van Fourier` are
unit and law names, not references to a living person. Set against that, replacing them turns
correct technical prose into nonsense and destroys the exact class of value `must_clear` exists
to protect. No contradiction with the one rule is introduced: nothing in this corpus asks for
an eponym to be masked.

### What that rule does *not* say: the same string in two different cases

"One string cannot be in `must_mask` and `must_clear` at once" is about **one document**.
Masking is decided per request, so two different cases can disagree about one string and a
detector can satisfy both -- by reading the context, which is the whole job.

The fixture does exactly that, four times, across `seed.json`, `domain.json` and
`negatives.json`: three business phone numbers (`+44 7911 123456`, `+39 320 123 4567`,
`030 12345678`) that a customer gives as their own in one case and a company publishes as its
service line in another, and one passport-shaped string (`XR1234567`) that is a passport
number under "Passport no." and an order number under "Order". They are kept deliberately.
Giving the negatives different digits would leave them exactly as hard -- a detector that keys
on digits alone fails a published-service-line negative whatever digits it carries -- and would
only hide that the fixture is asking for context sensitivity. Today the engine clears the
passport pair (the passport recognizer needs a context term, and "Order" is not one) and fails
all three phone pairs (`phone.rs` has no negative-context suppression), which is why
`PHONE_NUMBER` precision reads 0.8539.

`pii_detection_eval::check_coherence` is per case and cannot see any of this. `pii_detection_eval::check_cross_case_collisions`
can, and `the_fixtures_only_cross_case_collisions_are_the_argued_ones` pins the list to exactly
these five (id, id) pairs, so a fifth collision arriving by accident is a test failure rather
than something a reviewer has to spot. **The 600 `must_clear` values added on 2026-08-22
introduced none**: the allowlist is unchanged and the test still passes as written.

### The originals are not recorded

**The names that were replaced are deliberately not written down anywhere in this
repository.** Recording them would put the personal data back, which is what screening exists
to prevent. The count is recorded; the value is not.

### Replacement count per carrier

Carrier selection did most of the work -- regulations rather than judgments, technical
documentation rather than support threads full of signatures, encyclopedia articles about
places and processes rather than about living people, institutional annual reports rather than
the "meet our people" section of one:

- **313 of 321 carriers needed no replacement at all.**
- **8 carriers needed one or two**, 13 substitutions in total -- all eight of them among the
  original 54.
- No carrier needed three or more. A carrier that needs ten is the wrong carrier, and on
  2026-08-22 seven candidates were rejected on exactly that ground.

| carrier | replacements | → `must_mask` | → `must_clear` | what was replaced |
|---|---|---|---|---|
| `de-wp-kreisverkehr` | 2 | 2 | 0 | two named engineers, in a sentence about the first roundabouts |
| `fr-wp-eolienne` | 2 | 1 | 1 | an inventor's name, and the eponymous turbine named after them |
| `es-wp-compostaje` | 1 | 1 | 0 | a named chemist |
| `it-wp-rotatoria` | 1 | 1 | 0 | a named urbanist |
| `it-wv-treno` | 2 | 2 | 0 | an author and a character, in an allusion to a novel |
| `en-wp-windturbine` | 2 | 0 | 2 | two eponymous vertical-axis rotor names |
| `en-wb-road-safety` | 1 | 1 | 0 | a named cyclist, in an account of an 1842 collision |
| `en-wv-rail-tips` | 2 | 2 | 0 | an author and a character, in an allusion to a novel |

Replacement makes the affected sentences factually wrong -- the roundabout article now credits
people who do not exist, and the wind turbine article names rotor types nobody invented. That
is correct for a fixture and would not be for a publication; nothing here is republished as
fact.

**Verification, re-run over the whole set on 2026-08-26.** the detector's rules engine, asked for all
nine types it detects, was run over all 963 carrier cases -- the same `/analyze` entry point the
service exposes, in a container started with `PII_DETECTION_RULES_ONLY=1`. It detected **zero** spans
the answer key does not name and violated **zero** of the **1,956** `must_clear` values, across
963 cases including the 314 that contain no identifier at all. The corresponding figures over
the 154-carrier and 54-carrier corpora were also zero, so neither the 100 carriers added on
2026-08-22 nor the 167 added on 2026-08-26 left behind a false positive of any kind. Whatever
else is true of the screening, it left nothing behind that this detector can see.

That run did surface one thing worth naming, which is not a screening result:
**`fr-wp-mur-soutenement-dn`'s injected `IP_ADDRESS` is not found** (1,184 of 1,185 carrier
`must_mask` values, rules-only). It is a recogniser scoring behaviour the larger fixture
exposed, not a fixture defect -- the value is a valid RFC 5737 documentation address introduced
by the same French context phrase every other French carrier uses. It is left unfixed here on
this project's standing rule: changing the thing being measured from inside the instrument that
measures it is the wrong order, and `crates/rules/` is outside this task's boundary.

## The carriers

Retrieved 2026-08-21 (carriers 1–54), 2026-08-22 (carriers 55–154) and 2026-08-26
(carriers 155–321). `chars` is the carrier length before injection; `rev` is the Wikimedia
`oldid` the text was taken from, recorded from carrier 55 onward.

### Dutch (`nl`) -- 44 carriers

| id | register | source | chars | repl. | rev | URL |
|---|---|---|---|---|---|---|
| `nl-eb-ar24-balance` | business / annual report | ecb.europa.eu | 1629 | 0 | -- | <https://www.ecb.europa.eu/press/annual-reports-financial-statements/annual/html/ecb.ar2024~8402d8191f.nl.html> |
| `nl-eb-ar24-borrowing` | business / annual report | ecb.europa.eu | 1580 | 0 | -- | <https://www.ecb.europa.eu/press/annual-reports-financial-statements/annual/html/ecb.ar2024~8402d8191f.nl.html> |
| `nl-eb-ar24-cash` | business / annual report | ecb.europa.eu | 1372 | 0 | -- | <https://www.ecb.europa.eu/press/annual-reports-financial-statements/annual/html/ecb.ar2024~8402d8191f.nl.html> |
| `nl-eb-ar24-cashusage` | business / annual report | ecb.europa.eu | 1747 | 0 | -- | <https://www.ecb.europa.eu/press/annual-reports-financial-statements/annual/html/ecb.ar2024~8402d8191f.nl.html> |
| `nl-eb-ar24-stability` | business / annual report | ecb.europa.eu | 1294 | 0 | -- | <https://www.ecb.europa.eu/press/annual-reports-financial-statements/annual/html/ecb.ar2024~8402d8191f.nl.html> |
| `nl-eb-ar24-target` | business / annual report | ecb.europa.eu | 1520 | 0 | -- | <https://www.ecb.europa.eu/press/annual-reports-financial-statements/annual/html/ecb.ar2024~8402d8191f.nl.html> |
| `nl-wp-aquacultuur` | encyclopedia | nl.wikipedia.org | 1815 | 0 | 68649079 | <https://nl.wikipedia.org/wiki/Aquacultuur> |
| `nl-wp-asfalt` | encyclopedia | nl.wikipedia.org | 1438 | 0 | 71759835 | <https://nl.wikipedia.org/wiki/Asfalt> |
| `nl-wp-beton` | encyclopedia | nl.wikipedia.org | 1401 | 0 | 71534662 | <https://nl.wikipedia.org/wiki/Beton> |
| `nl-wp-grind` | encyclopedia | nl.wikipedia.org | 1320 | 0 | 71835513 | <https://nl.wikipedia.org/wiki/Grind> |
| `nl-wp-kalksteen` | encyclopedia | nl.wikipedia.org | 1278 | 0 | 70440542 | <https://nl.wikipedia.org/wiki/Kalksteen> |
| `nl-wp-koper-element` | encyclopedia | nl.wikipedia.org | 1493 | 0 | 71836089 | <https://nl.wikipedia.org/wiki/Koper_(element)> |
| `nl-wp-lijm` | encyclopedia | nl.wikipedia.org | 1152 | 0 | 71713666 | <https://nl.wikipedia.org/wiki/Lijm> |
| `nl-wp-rotonde` | encyclopedia | nl.wikipedia.org | 1452 | 0 | -- | <https://nl.wikipedia.org/wiki/Rotonde_(verkeer)> |
| `nl-wp-schutsluis` | encyclopedia | nl.wikipedia.org | 1641 | 0 | 70965885 | <https://nl.wikipedia.org/wiki/Schutsluis> |
| `nl-wp-verf` | encyclopedia | nl.wikipedia.org | 1755 | 0 | 71003470 | <https://nl.wikipedia.org/wiki/Verf> |
| `nl-wp-verkeerslicht` | encyclopedia | nl.wikipedia.org | 1483 | 0 | 71760645 | <https://nl.wikipedia.org/wiki/Verkeerslicht> |
| `nl-wp-warmte-isolatie` | encyclopedia | nl.wikipedia.org | 1767 | 0 | 71281823 | <https://nl.wikipedia.org/wiki/Warmte-isolatie> |
| `nl-wp-warmtepomp` | encyclopedia | nl.wikipedia.org | 1481 | 0 | 71119102 | <https://nl.wikipedia.org/wiki/Warmtepomp> |
| `nl-wp-waterschap` | encyclopedia | nl.wikipedia.org | 1558 | 0 | -- | <https://nl.wikipedia.org/wiki/Waterschap> |
| `nl-wp-zonnepaneel` | encyclopedia | nl.wikipedia.org | 1557 | 0 | -- | <https://nl.wikipedia.org/wiki/Zonnepaneel> |
| `nl-wb-beheer-onderhoud` | instructional / technical | nl.wikibooks.org | 1575 | 0 | -- | <https://nl.wikibooks.org/wiki/Beheer_en_Onderhoud> |
| `nl-wb-computerbussen` | instructional / technical | nl.wikibooks.org | 1406 | 0 | 424836 | <https://nl.wikibooks.org/wiki/Computersystemen/Computerbussen> |
| `nl-wb-computerparken-opvolgen-beheren` | instructional / technical | nl.wikibooks.org | 1549 | 0 | 422796 | <https://nl.wikibooks.org/wiki/Linux_Systeembeheer/Computerparken_opvolgen_en_beheren> |
| `nl-wb-linux-firewalls` | instructional / technical | nl.wikibooks.org | 1635 | 0 | 424385 | <https://nl.wikibooks.org/wiki/Linux_Systeembeheer/Firewalls> |
| `nl-wb-linux-installeren` | instructional / technical | nl.wikibooks.org | 1256 | 0 | 422785 | <https://nl.wikibooks.org/wiki/Linux_Systeembeheer/Linux_installeren> |
| `nl-wb-linux-opslag` | instructional / technical | nl.wikibooks.org | 1509 | 0 | 422788 | <https://nl.wikibooks.org/wiki/Linux_Systeembeheer/Opslagapparaten_beheren> |
| `nl-wb-processor` | instructional / technical | nl.wikibooks.org | 1462 | 0 | 422874 | <https://nl.wikibooks.org/wiki/Computersystemen/Processor> |
| `nl-wb-servervirtualisatie` | instructional / technical | nl.wikibooks.org | 1174 | 0 | -- | <https://nl.wikibooks.org/wiki/Linux_Systeembeheer/Servervirtualisatie> |
| `nl-wb-tp-connectoren-plaatsen` | instructional / technical | nl.wikibooks.org | 1590 | 0 | 414154 | <https://nl.wikibooks.org/wiki/Computernetwerken_voor_gevorderden/Netwerkbekabeling/TP-connectoren_plaatsen> |
| `nl-el-0024-procurement` | legal / regulatory | eur-lex.europa.eu | 1708 | 0 | -- | <https://eur-lex.europa.eu/legal-content/NL/TXT/HTML/?uri=CELEX:32014L0024> |
| `nl-el-0031-energy` | legal / regulatory | eur-lex.europa.eu | 1594 | 0 | -- | <https://eur-lex.europa.eu/legal-content/NL/TXT/HTML/?uri=CELEX:32010L0031> |
| `nl-el-0060-water` | legal / regulatory | eur-lex.europa.eu | 1502 | 0 | -- | <https://eur-lex.europa.eu/legal-content/NL/TXT/HTML/?uri=CELEX:32000L0060> |
| `nl-el-0098-waste` | legal / regulatory | eur-lex.europa.eu | 1712 | 0 | -- | <https://eur-lex.europa.eu/legal-content/NL/TXT/HTML/?uri=CELEX:32008L0098> |
| `nl-el-1169-food` | legal / regulatory | eur-lex.europa.eu | 1723 | 0 | -- | <https://eur-lex.europa.eu/legal-content/NL/TXT/HTML/?uri=CELEX:32011R1169> |
| `nl-el-1999-adaptation` | legal / regulatory | eur-lex.europa.eu | 1540 | 0 | -- | <https://eur-lex.europa.eu/legal-content/NL/TXT/HTML/?uri=CELEX:31999L0031> |
| `nl-el-1999-landfill` | legal / regulatory | eur-lex.europa.eu | 1508 | 0 | -- | <https://eur-lex.europa.eu/legal-content/NL/TXT/HTML/?uri=CELEX:31999L0031> |
| `nl-wv-geld` | practical travel guide | nl.wikivoyage.org | 1271 | 0 | 157925 | <https://nl.wikivoyage.org/wiki/Geld> |
| `nl-wv-oplichterspraktijken` | practical travel guide | nl.wikivoyage.org | 1425 | 0 | 143120 | <https://nl.wikivoyage.org/wiki/Oplichterspraktijken> |
| `nl-wv-reisrecht` | practical travel guide | nl.wikivoyage.org | 1501 | 0 | 133800 | <https://nl.wikivoyage.org/wiki/Reisrecht> |
| `nl-wv-roken` | practical travel guide | nl.wikivoyage.org | 1893 | 0 | 134756 | <https://nl.wikivoyage.org/wiki/Roken> |
| `nl-wv-treinreizen` | practical travel guide | nl.wikivoyage.org | 1352 | 0 | -- | <https://nl.wikivoyage.org/wiki/Treinreizen_in_Europa> |
| `nl-wv-west-nederland` | practical travel guide | nl.wikivoyage.org | 1569 | 0 | -- | <https://nl.wikivoyage.org/wiki/West-Nederland> |
| `nl-wv-zakkenrollers` | practical travel guide | nl.wikivoyage.org | 1442 | 0 | 157926 | <https://nl.wikivoyage.org/wiki/Zakkenrollers> |

Registers: business / annual report 6, encyclopedia 15, instructional / technical 9, legal / regulatory 7, practical travel guide 7.

### German (`de`) -- 47 carriers

| id | register | source | chars | repl. | rev | URL |
|---|---|---|---|---|---|---|
| `de-eb-ar24-balance` | business / annual report | ecb.europa.eu | 1571 | 0 | -- | <https://www.ecb.europa.eu/press/annual-reports-financial-statements/annual/html/ecb.ar2024~8402d8191f.de.html> |
| `de-eb-ar24-borrowing` | business / annual report | ecb.europa.eu | 1482 | 0 | -- | <https://www.ecb.europa.eu/press/annual-reports-financial-statements/annual/html/ecb.ar2024~8402d8191f.de.html> |
| `de-eb-ar24-cash` | business / annual report | ecb.europa.eu | 1378 | 0 | -- | <https://www.ecb.europa.eu/press/annual-reports-financial-statements/annual/html/ecb.ar2024~8402d8191f.de.html> |
| `de-eb-ar24-cashusage` | business / annual report | ecb.europa.eu | 1827 | 0 | -- | <https://www.ecb.europa.eu/press/annual-reports-financial-statements/annual/html/ecb.ar2024~8402d8191f.de.html> |
| `de-eb-ar24-stability` | business / annual report | ecb.europa.eu | 1271 | 0 | -- | <https://www.ecb.europa.eu/press/annual-reports-financial-statements/annual/html/ecb.ar2024~8402d8191f.de.html> |
| `de-eb-ar24-target` | business / annual report | ecb.europa.eu | 1635 | 0 | -- | <https://www.ecb.europa.eu/press/annual-reports-financial-statements/annual/html/ecb.ar2024~8402d8191f.de.html> |
| `de-wp-asphalt` | encyclopedia | de.wikipedia.org | 1605 | 0 | 269351411 | <https://de.wikipedia.org/wiki/Asphalt> |
| `de-wp-beton` | encyclopedia | de.wikipedia.org | 1728 | 0 | 269239527 | <https://de.wikipedia.org/wiki/Beton> |
| `de-wp-biogas` | encyclopedia | de.wikipedia.org | 1733 | 0 | 269408089 | <https://de.wikipedia.org/wiki/Biogas> |
| `de-wp-containertransport` | encyclopedia | de.wikipedia.org | 1706 | 0 | 266864324 | <https://de.wikipedia.org/wiki/Containertransport> |
| `de-wp-fernwaerme` | encyclopedia | de.wikipedia.org | 1570 | 0 | -- | <https://de.wikipedia.org/wiki/Fernw%C3%A4rme> |
| `de-wp-glas` | encyclopedia | de.wikipedia.org | 1638 | 0 | 269521623 | <https://de.wikipedia.org/wiki/Glas> |
| `de-wp-klimaanlage` | encyclopedia | de.wikipedia.org | 1693 | 0 | 269579259 | <https://de.wikipedia.org/wiki/Klimaanlage> |
| `de-wp-kreisverkehr` | encyclopedia | de.wikipedia.org | 1548 | 2 | -- | <https://de.wikipedia.org/wiki/Kreisverkehr> |
| `de-wp-lichtwellenleiter` | encyclopedia | de.wikipedia.org | 1795 | 0 | 269301801 | <https://de.wikipedia.org/wiki/Lichtwellenleiter> |
| `de-wp-meerwasserentsalzung` | encyclopedia | de.wikipedia.org | 1832 | 0 | 266890972 | <https://de.wikipedia.org/wiki/Meerwasserentsalzung> |
| `de-wp-mehrscheiben-isolierglas` | encyclopedia | de.wikipedia.org | 1737 | 0 | 269126080 | <https://de.wikipedia.org/wiki/Mehrscheiben-Isolierglas> |
| `de-wp-rohrleitung` | encyclopedia | de.wikipedia.org | 1778 | 0 | 266376238 | <https://de.wikipedia.org/wiki/Rohrleitung> |
| `de-wp-schleuse` | encyclopedia | de.wikipedia.org | 1679 | 0 | 269064156 | <https://de.wikipedia.org/wiki/Schleuse> |
| `de-wp-stromnetz` | encyclopedia | de.wikipedia.org | 1910 | 0 | 264642188 | <https://de.wikipedia.org/wiki/Stromnetz> |
| `de-wp-torf` | encyclopedia | de.wikipedia.org | 1898 | 0 | 268284183 | <https://de.wikipedia.org/wiki/Torf> |
| `de-wp-verzinken` | encyclopedia | de.wikipedia.org | 1814 | 0 | 269035250 | <https://de.wikipedia.org/wiki/Verzinken> |
| `de-wp-waermepumpe` | encyclopedia | de.wikipedia.org | 1489 | 0 | 269702072 | <https://de.wikipedia.org/wiki/W%C3%A4rmepumpe> |
| `de-wp-windkraftanlage` | encyclopedia | de.wikipedia.org | 1542 | 0 | -- | <https://de.wikipedia.org/wiki/Windkraftanlage> |
| `de-wb-440-elektrotechnik` | instructional / technical | de.wikibooks.org | 1448 | 0 | 1067718 | <https://de.wikibooks.org/wiki/Einführung_in_die_Systemplanung_(Gebäudetechnik)/_440_Elektrotechnik> |
| `de-wb-460-forderanlagen` | instructional / technical | de.wikibooks.org | 1724 | 0 | 1067700 | <https://de.wikibooks.org/wiki/Einführung_in_die_Systemplanung_(Gebäudetechnik)/_460_Förderanlagen> |
| `de-wb-480-gebaude-anlagentechnik` | instructional / technical | de.wikibooks.org | 1433 | 0 | 1067712 | <https://de.wikibooks.org/wiki/Einführung_in_die_Systemplanung_(Gebäudetechnik)/_480_Gebäude-_und_Anlagentechnik> |
| `de-wb-ursachen-wirkungen` | instructional / technical | de.wikibooks.org | 1899 | 0 | 998562 | <https://de.wikibooks.org/wiki/Optimierung_von_Kleinheizungsanlagen/_Ursachen_und_Wirkungen> |
| `de-wb-werkstoffkunde-einfuehrung` | instructional / technical | de.wikibooks.org | 1542 | 0 | -- | <https://de.wikibooks.org/wiki/Werkstoffkunde_Metall/_Einf%C3%BChrung> |
| `de-wb-werkstoffkunde-legierung` | instructional / technical | de.wikibooks.org | 1817 | 0 | 1079271 | <https://de.wikibooks.org/wiki/Werkstoffkunde_Metall/_Innerer_Aufbau/_Legierung> |
| `de-wb-werkstoffkunde-metallurgie` | instructional / technical | de.wikibooks.org | 1550 | 0 | 1027970 | <https://de.wikibooks.org/wiki/Werkstoffkunde_Metall/_Eisen_und_Stahl/_Metallurgie> |
| `de-wb-werkstoffkunde-struktur` | instructional / technical | de.wikibooks.org | 1597 | 0 | -- | <https://de.wikibooks.org/wiki/Werkstoffkunde_Metall/_Innerer_Aufbau/_Struktur> |
| `de-el-0024-procurement` | legal / regulatory | eur-lex.europa.eu | 1775 | 0 | -- | <https://eur-lex.europa.eu/legal-content/DE/TXT/HTML/?uri=CELEX:32014L0024> |
| `de-el-0031-energy` | legal / regulatory | eur-lex.europa.eu | 1545 | 0 | -- | <https://eur-lex.europa.eu/legal-content/DE/TXT/HTML/?uri=CELEX:32010L0031> |
| `de-el-0060-water` | legal / regulatory | eur-lex.europa.eu | 1489 | 0 | -- | <https://eur-lex.europa.eu/legal-content/DE/TXT/HTML/?uri=CELEX:32000L0060> |
| `de-el-0098-waste` | legal / regulatory | eur-lex.europa.eu | 1888 | 0 | -- | <https://eur-lex.europa.eu/legal-content/DE/TXT/HTML/?uri=CELEX:32008L0098> |
| `de-el-1169-food` | legal / regulatory | eur-lex.europa.eu | 1684 | 0 | -- | <https://eur-lex.europa.eu/legal-content/DE/TXT/HTML/?uri=CELEX:32011R1169> |
| `de-el-1999-adaptation` | legal / regulatory | eur-lex.europa.eu | 1449 | 0 | -- | <https://eur-lex.europa.eu/legal-content/DE/TXT/HTML/?uri=CELEX:31999L0031> |
| `de-el-1999-landfill` | legal / regulatory | eur-lex.europa.eu | 1558 | 0 | -- | <https://eur-lex.europa.eu/legal-content/DE/TXT/HTML/?uri=CELEX:31999L0031> |
| `de-wv-camping` | practical travel guide | de.wikivoyage.org | 1894 | 0 | 1744111 | <https://de.wikivoyage.org/wiki/Camping> |
| `de-wv-frachtschiffreise` | practical travel guide | de.wikivoyage.org | 1614 | 0 | 1788444 | <https://de.wikivoyage.org/wiki/Frachtschiffreise> |
| `de-wv-ostseefahren` | practical travel guide | de.wikivoyage.org | 1860 | 0 | 1777694 | <https://de.wikivoyage.org/wiki/Ostseefähren> |
| `de-wv-reisen-mit-dem-auto` | practical travel guide | de.wikivoyage.org | 1869 | 0 | 1711998 | <https://de.wikivoyage.org/wiki/Reisen_mit_dem_Auto> |
| `de-wv-reisen-mit-dem-camper` | practical travel guide | de.wikivoyage.org | 1773 | 0 | 1657119 | <https://de.wikivoyage.org/wiki/Reisen_mit_dem_Camper> |
| `de-wv-schiff` | practical travel guide | de.wikivoyage.org | 1872 | 0 | 1748600 | <https://de.wikivoyage.org/wiki/Reisen_mit_dem_Schiff> |
| `de-wv-wandern` | practical travel guide | de.wikivoyage.org | 763 | 0 | -- | <https://de.wikivoyage.org/wiki/Wandern_in_Deutschland> |
| `de-wv-zug` | practical travel guide | de.wikivoyage.org | 1409 | 0 | -- | <https://de.wikivoyage.org/wiki/Reisen_mit_dem_Zug> |

Registers: business / annual report 6, encyclopedia 18, instructional / technical 8, legal / regulatory 7, practical travel guide 8.

### French (`fr`) -- 61 carriers

| id | register | source | chars | repl. | rev | URL |
|---|---|---|---|---|---|---|
| `fr-eb-ar24-balance` | business / annual report | ecb.europa.eu | 1785 | 0 | -- | <https://www.ecb.europa.eu/press/annual-reports-financial-statements/annual/html/ecb.ar2024~8402d8191f.fr.html> |
| `fr-eb-ar24-borrowing` | business / annual report | ecb.europa.eu | 1750 | 0 | -- | <https://www.ecb.europa.eu/press/annual-reports-financial-statements/annual/html/ecb.ar2024~8402d8191f.fr.html> |
| `fr-eb-ar24-cash` | business / annual report | ecb.europa.eu | 1450 | 0 | -- | <https://www.ecb.europa.eu/press/annual-reports-financial-statements/annual/html/ecb.ar2024~8402d8191f.fr.html> |
| `fr-eb-ar24-cashusage` | business / annual report | ecb.europa.eu | 1933 | 0 | -- | <https://www.ecb.europa.eu/press/annual-reports-financial-statements/annual/html/ecb.ar2024~8402d8191f.fr.html> |
| `fr-eb-ar24-stability` | business / annual report | ecb.europa.eu | 1376 | 0 | -- | <https://www.ecb.europa.eu/press/annual-reports-financial-statements/annual/html/ecb.ar2024~8402d8191f.fr.html> |
| `fr-eb-ar24-target` | business / annual report | ecb.europa.eu | 1654 | 0 | -- | <https://www.ecb.europa.eu/press/annual-reports-financial-statements/annual/html/ecb.ar2024~8402d8191f.fr.html> |
| `fr-wp-acier` | encyclopedia | fr.wikipedia.org | 1585 | 0 | 236968733 | <https://fr.wikipedia.org/wiki/Acier> |
| `fr-wp-aluminium` | encyclopedia | fr.wikipedia.org | 1906 | 0 | 238247064 | <https://fr.wikipedia.org/wiki/Aluminium> |
| `fr-wp-aquaculture` | encyclopedia | fr.wikipedia.org | 1428 | 0 | 238580916 | <https://fr.wikipedia.org/wiki/Aquaculture> |
| `fr-wp-ascenseur` | encyclopedia | fr.wikipedia.org | 1893 | 0 | 237315588 | <https://fr.wikipedia.org/wiki/Ascenseur> |
| `fr-wp-assainissement` | encyclopedia | fr.wikipedia.org | 1495 | 0 | 238253367 | <https://fr.wikipedia.org/wiki/Assainissement> |
| `fr-wp-carton-materiau` | encyclopedia | fr.wikipedia.org | 1156 | 0 | 219928330 | <https://fr.wikipedia.org/wiki/Carton_(matériau)> |
| `fr-wp-charbon-bois` | encyclopedia | fr.wikipedia.org | 1706 | 0 | 235574061 | <https://fr.wikipedia.org/wiki/Charbon_de_bois> |
| `fr-wp-compostage` | encyclopedia | fr.wikipedia.org | 1419 | 0 | -- | <https://fr.wikipedia.org/wiki/Compostage_(biologie)> |
| `fr-wp-coton` | encyclopedia | fr.wikipedia.org | 1530 | 0 | 238482710 | <https://fr.wikipedia.org/wiki/Coton> |
| `fr-wp-cuivre` | encyclopedia | fr.wikipedia.org | 1884 | 0 | 238024168 | <https://fr.wikipedia.org/wiki/Cuivre> |
| `fr-wp-ecluse` | encyclopedia | fr.wikipedia.org | 1836 | 0 | 236853729 | <https://fr.wikipedia.org/wiki/%C3%89cluse> |
| `fr-wp-enrobe` | encyclopedia | fr.wikipedia.org | 1776 | 0 | 225270939 | <https://fr.wikipedia.org/wiki/Enrob%C3%A9> |
| `fr-wp-entrepot` | encyclopedia | fr.wikipedia.org | 1338 | 0 | 237174806 | <https://fr.wikipedia.org/wiki/Entrepôt> |
| `fr-wp-eolienne` | encyclopedia | fr.wikipedia.org | 1541 | 2 | -- | <https://fr.wikipedia.org/wiki/%C3%89olienne> |
| `fr-wp-fibre-optique` | encyclopedia | fr.wikipedia.org | 1846 | 0 | 238815445 | <https://fr.wikipedia.org/wiki/Fibre_optique> |
| `fr-wp-giratoire` | encyclopedia | fr.wikipedia.org | 1530 | 0 | -- | <https://fr.wikipedia.org/wiki/Carrefour_giratoire> |
| `fr-wp-gravier-granulat` | encyclopedia | fr.wikipedia.org | 1831 | 0 | 233761400 | <https://fr.wikipedia.org/wiki/Gravier_(granulat)> |
| `fr-wp-isolation` | encyclopedia | fr.wikipedia.org | 1599 | 0 | 238733538 | <https://fr.wikipedia.org/wiki/Isolation_thermique> |
| `fr-wp-mur-soutenement` | encyclopedia | fr.wikipedia.org | 1330 | 0 | 230765502 | <https://fr.wikipedia.org/wiki/Mur_de_soutènement> |
| `fr-wp-palette-manutention` | encyclopedia | fr.wikipedia.org | 1209 | 0 | 236940005 | <https://fr.wikipedia.org/wiki/Palette_de_manutention> |
| `fr-wp-papier` | encyclopedia | fr.wikipedia.org | 1565 | 0 | 237786213 | <https://fr.wikipedia.org/wiki/Papier> |
| `fr-wp-passage-a-niveau` | encyclopedia | fr.wikipedia.org | 1684 | 0 | 236992063 | <https://fr.wikipedia.org/wiki/Passage_à_niveau> |
| `fr-wp-polychlorure-vinyle` | encyclopedia | fr.wikipedia.org | 1605 | 0 | 235348140 | <https://fr.wikipedia.org/wiki/Polychlorure_de_vinyle> |
| `fr-wp-polyethylene` | encyclopedia | fr.wikipedia.org | 1329 | 0 | 236940053 | <https://fr.wikipedia.org/wiki/Polyéthylène> |
| `fr-wp-pompe-a-chaleur` | encyclopedia | fr.wikipedia.org | 1736 | 0 | 238239253 | <https://fr.wikipedia.org/wiki/Pompe_%C3%A0_chaleur> |
| `fr-wp-tramway` | encyclopedia | fr.wikipedia.org | 1309 | 0 | 236602261 | <https://fr.wikipedia.org/wiki/Tramway> |
| `fr-wp-vent` | encyclopedia | fr.wikipedia.org | 1821 | 0 | 237525443 | <https://fr.wikipedia.org/wiki/Vent> |
| `fr-wb-charpente` | instructional / technical | fr.wikibooks.org | 1461 | 0 | -- | <https://fr.wikibooks.org/wiki/Construire_sa_maison/La_charpente> |
| `fr-wb-chauffage` | instructional / technical | fr.wikibooks.org | 1849 | 0 | 531897 | <https://fr.wikibooks.org/wiki/Construire_sa_maison/Le_chauffage> |
| `fr-wb-cloisons` | instructional / technical | fr.wikibooks.org | 1307 | 0 | 501306 | <https://fr.wikibooks.org/wiki/Construire_sa_maison/Les_cloisons> |
| `fr-wb-couverture` | instructional / technical | fr.wikibooks.org | 1827 | 0 | 437549 | <https://fr.wikibooks.org/wiki/Construire_sa_maison/La_couverture> |
| `fr-wb-declencheurs-souples-telecommandes` | instructional / technical | fr.wikibooks.org | 1763 | 0 | 634213 | <https://fr.wikibooks.org/wiki/Photographie/Accessoires/Déclencheurs_souples,_télécommandes,_minuteries> |
| `fr-wb-installation-electrique` | instructional / technical | fr.wikibooks.org | 1467 | 0 | -- | <https://fr.wikibooks.org/wiki/Construire_sa_maison/Installation_%C3%A9lectrique> |
| `fr-wb-installation-sanitaire` | instructional / technical | fr.wikibooks.org | 1565 | 0 | 681757 | <https://fr.wikibooks.org/wiki/Construire_sa_maison/Installation_sanitaire> |
| `fr-wb-maconnerie` | instructional / technical | fr.wikibooks.org | 1723 | 0 | 660707 | <https://fr.wikibooks.org/wiki/Construire_sa_maison/La_ma%C3%A7onnerie> |
| `fr-wb-mise-hors-d-eau` | instructional / technical | fr.wikibooks.org | 1447 | 0 | 351612 | <https://fr.wikibooks.org/wiki/Construire_sa_maison/La_mise_hors_d'eau> |
| `fr-wb-realisation-des-sols` | instructional / technical | fr.wikibooks.org | 1406 | 0 | 258889 | <https://fr.wikibooks.org/wiki/Construire_sa_maison/Réalisation_des_sols> |
| `fr-wb-scanners` | instructional / technical | fr.wikibooks.org | 1482 | 0 | 657513 | <https://fr.wikibooks.org/wiki/Photographie/Accessoires/Scanners> |
| `fr-wb-terrassement` | instructional / technical | fr.wikibooks.org | 1284 | 0 | 683278 | <https://fr.wikibooks.org/wiki/Construire_sa_maison/Le_terrassement> |
| `fr-wb-ventilation` | instructional / technical | fr.wikibooks.org | 1438 | 0 | 674903 | <https://fr.wikibooks.org/wiki/Construire_sa_maison/La_ventilation> |
| `fr-el-0024-procurement` | legal / regulatory | eur-lex.europa.eu | 1798 | 0 | -- | <https://eur-lex.europa.eu/legal-content/FR/TXT/HTML/?uri=CELEX:32014L0024> |
| `fr-el-0031-energy` | legal / regulatory | eur-lex.europa.eu | 1691 | 0 | -- | <https://eur-lex.europa.eu/legal-content/FR/TXT/HTML/?uri=CELEX:32010L0031> |
| `fr-el-0060-water` | legal / regulatory | eur-lex.europa.eu | 1474 | 0 | -- | <https://eur-lex.europa.eu/legal-content/FR/TXT/HTML/?uri=CELEX:32000L0060> |
| `fr-el-0098-waste` | legal / regulatory | eur-lex.europa.eu | 1804 | 0 | -- | <https://eur-lex.europa.eu/legal-content/FR/TXT/HTML/?uri=CELEX:32008L0098> |
| `fr-el-1169-food` | legal / regulatory | eur-lex.europa.eu | 1453 | 0 | -- | <https://eur-lex.europa.eu/legal-content/FR/TXT/HTML/?uri=CELEX:32011R1169> |
| `fr-el-1999-adaptation` | legal / regulatory | eur-lex.europa.eu | 1663 | 0 | -- | <https://eur-lex.europa.eu/legal-content/FR/TXT/HTML/?uri=CELEX:31999L0031> |
| `fr-el-1999-landfill` | legal / regulatory | eur-lex.europa.eu | 1494 | 0 | -- | <https://eur-lex.europa.eu/legal-content/FR/TXT/HTML/?uri=CELEX:31999L0031> |
| `fr-wv-arnaques-courantes` | practical travel guide | fr.wikivoyage.org | 1187 | 0 | 582730 | <https://fr.wikivoyage.org/wiki/Arnaques_courantes> |
| `fr-wv-echange-maisons` | practical travel guide | fr.wikivoyage.org | 1427 | 0 | 649880 | <https://fr.wikivoyage.org/wiki/Échange_de_maisons> |
| `fr-wv-fuseaux-horaires` | practical travel guide | fr.wikivoyage.org | 1709 | 0 | 596878 | <https://fr.wikivoyage.org/wiki/Fuseaux_horaires> |
| `fr-wv-randonnee-pedestre` | practical travel guide | fr.wikivoyage.org | 1769 | 0 | 628316 | <https://fr.wikivoyage.org/wiki/Randonnée_pédestre> |
| `fr-wv-route` | practical travel guide | fr.wikivoyage.org | 1655 | 0 | 598917 | <https://fr.wikivoyage.org/wiki/Prendre_la_route> |
| `fr-wv-train` | practical travel guide | fr.wikivoyage.org | 1431 | 0 | -- | <https://fr.wikivoyage.org/wiki/Voyager_en_train> |
| `fr-wv-train-europe` | practical travel guide | fr.wikivoyage.org | 1581 | 0 | -- | <https://fr.wikivoyage.org/wiki/Voyager_en_train_en_Europe> |
| `fr-wv-voyager-a-velo` | practical travel guide | fr.wikivoyage.org | 1219 | 0 | 605316 | <https://fr.wikivoyage.org/wiki/Voyager_à_vélo> |

Registers: business / annual report 6, encyclopedia 27, instructional / technical 13, legal / regulatory 7, practical travel guide 8.

### Spanish (`es`) -- 66 carriers

| id | register | source | chars | repl. | rev | URL |
|---|---|---|---|---|---|---|
| `es-eb-ar24-balance` | business / annual report | ecb.europa.eu | 1417 | 0 | -- | <https://www.ecb.europa.eu/press/annual-reports-financial-statements/annual/html/ecb.ar2024~8402d8191f.es.html> |
| `es-eb-ar24-borrowing` | business / annual report | ecb.europa.eu | 1532 | 0 | -- | <https://www.ecb.europa.eu/press/annual-reports-financial-statements/annual/html/ecb.ar2024~8402d8191f.es.html> |
| `es-eb-ar24-cash` | business / annual report | ecb.europa.eu | 1420 | 0 | -- | <https://www.ecb.europa.eu/press/annual-reports-financial-statements/annual/html/ecb.ar2024~8402d8191f.es.html> |
| `es-eb-ar24-cashusage` | business / annual report | ecb.europa.eu | 1872 | 0 | -- | <https://www.ecb.europa.eu/press/annual-reports-financial-statements/annual/html/ecb.ar2024~8402d8191f.es.html> |
| `es-eb-ar24-stability` | business / annual report | ecb.europa.eu | 1382 | 0 | -- | <https://www.ecb.europa.eu/press/annual-reports-financial-statements/annual/html/ecb.ar2024~8402d8191f.es.html> |
| `es-eb-ar24-target` | business / annual report | ecb.europa.eu | 1655 | 0 | -- | <https://www.ecb.europa.eu/press/annual-reports-financial-statements/annual/html/ecb.ar2024~8402d8191f.es.html> |
| `es-wp-asfalto` | encyclopedia | es.wikipedia.org | 1689 | 0 | 174647730 | <https://es.wikipedia.org/wiki/Asfalto> |
| `es-wp-biogas` | encyclopedia | es.wikipedia.org | 1617 | 0 | 174892955 | <https://es.wikipedia.org/wiki/Biogás> |
| `es-wp-bomba-de-calor` | encyclopedia | es.wikipedia.org | 1781 | 0 | 174279339 | <https://es.wikipedia.org/wiki/Bomba_de_calor> |
| `es-wp-chimenea` | encyclopedia | es.wikipedia.org | 1591 | 0 | 173745264 | <https://es.wikipedia.org/wiki/Chimenea> |
| `es-wp-compostaje` | encyclopedia | es.wikipedia.org | 1402 | 1 | -- | <https://es.wikipedia.org/wiki/Compost> |
| `es-wp-contenedor` | encyclopedia | es.wikipedia.org | 1410 | 0 | 168626670 | <https://es.wikipedia.org/wiki/Contenedor> |
| `es-wp-desalinizacion` | encyclopedia | es.wikipedia.org | 1806 | 0 | 174497314 | <https://es.wikipedia.org/wiki/Desalinización> |
| `es-wp-drenaje` | encyclopedia | es.wikipedia.org | 1452 | 0 | 174476978 | <https://es.wikipedia.org/wiki/Drenaje> |
| `es-wp-elaboracion-cerveza` | encyclopedia | es.wikipedia.org | 1554 | 0 | 166419606 | <https://es.wikipedia.org/wiki/Elaboración_de_cerveza> |
| `es-wp-esclusa` | encyclopedia | es.wikipedia.org | 1062 | 0 | 168474785 | <https://es.wikipedia.org/wiki/Esclusa> |
| `es-wp-fibra-optica` | encyclopedia | es.wikipedia.org | 1809 | 0 | 173982094 | <https://es.wikipedia.org/wiki/Fibra_%C3%B3ptica> |
| `es-wp-fotovoltaica` | encyclopedia | es.wikipedia.org | 1428 | 0 | -- | <https://es.wikipedia.org/wiki/Energ%C3%ADa_solar_fotovoltaica> |
| `es-wp-harina` | encyclopedia | es.wikipedia.org | 1959 | 0 | 173298567 | <https://es.wikipedia.org/wiki/Harina> |
| `es-wp-hormigon` | encyclopedia | es.wikipedia.org | 1110 | 0 | 174945262 | <https://es.wikipedia.org/wiki/Hormig%C3%B3n> |
| `es-wp-lampara-led` | encyclopedia | es.wikipedia.org | 1471 | 0 | 174723665 | <https://es.wikipedia.org/wiki/Lámpara_led> |
| `es-wp-moldeo-por-inyeccion` | encyclopedia | es.wikipedia.org | 1517 | 0 | 174239331 | <https://es.wikipedia.org/wiki/Moldeo_por_inyección> |
| `es-wp-mortero` | encyclopedia | es.wikipedia.org | 1629 | 0 | 170761410 | <https://es.wikipedia.org/wiki/Mortero_%28construcci%C3%B3n%29> |
| `es-wp-pale` | encyclopedia | es.wikipedia.org | 1628 | 0 | 171496697 | <https://es.wikipedia.org/wiki/Palé> |
| `es-wp-pan` | encyclopedia | es.wikipedia.org | 1354 | 0 | 174888830 | <https://es.wikipedia.org/wiki/Pan> |
| `es-wp-porcelana` | encyclopedia | es.wikipedia.org | 1461 | 0 | 174798334 | <https://es.wikipedia.org/wiki/Porcelana> |
| `es-wp-queso` | encyclopedia | es.wikipedia.org | 1812 | 0 | 174836155 | <https://es.wikipedia.org/wiki/Queso> |
| `es-wp-reciclaje` | encyclopedia | es.wikipedia.org | 1307 | 0 | 174966503 | <https://es.wikipedia.org/wiki/Reciclaje> |
| `es-wp-red-electrica` | encyclopedia | es.wikipedia.org | 1922 | 0 | 174710516 | <https://es.wikipedia.org/wiki/Red_eléctrica> |
| `es-wp-rotonda` | encyclopedia | es.wikipedia.org | 1572 | 0 | -- | <https://es.wikipedia.org/wiki/Rotonda> |
| `es-wp-tuberia` | encyclopedia | es.wikipedia.org | 1853 | 0 | 172116366 | <https://es.wikipedia.org/wiki/Tuber%C3%ADa> |
| `es-wp-ventana` | encyclopedia | es.wikipedia.org | 1440 | 0 | 174956977 | <https://es.wikipedia.org/wiki/Ventana> |
| `es-wb-almacenamiento-electronico` | instructional / technical | es.wikibooks.org | 1971 | 0 | 423629 | <https://es.wikibooks.org/wiki/Mantenimiento_y_Montaje_de_Equipos_Informáticos/Tema_3/Almacenamiento_electrónico> |
| `es-wb-almacenamiento-optico` | instructional / technical | es.wikibooks.org | 1859 | 0 | 408821 | <https://es.wikibooks.org/wiki/Mantenimiento_y_Montaje_de_Equipos_Inform%C3%A1ticos/Tema_3/Almacenamiento_%C3%B3ptico> |
| `es-wb-buses` | instructional / technical | es.wikibooks.org | 1612 | 0 | 416014 | <https://es.wikibooks.org/wiki/Mantenimiento_y_Montaje_de_Equipos_Informáticos/Tema_2/Los_Buses> |
| `es-wb-chipset` | instructional / technical | es.wikibooks.org | 1254 | 0 | 400272 | <https://es.wikibooks.org/wiki/Mantenimiento_y_Montaje_de_Equipos_Informáticos/Tema_2/El_chipset> |
| `es-wb-conectores` | instructional / technical | es.wikibooks.org | 1496 | 0 | 427612 | <https://es.wikibooks.org/wiki/Mantenimiento_y_Montaje_de_Equipos_Informáticos/Tema_2/Conectores> |
| `es-wb-copias-seguridad-o-respaldo` | instructional / technical | es.wikibooks.org | 1914 | 0 | 396149 | <https://es.wikibooks.org/wiki/Mantenimiento_y_Montaje_de_Equipos_Informáticos/Tema_8/Copias_de_seguridad_o_Respaldo_de_ficheros> |
| `es-wb-mantenimiento-correctivo` | instructional / technical | es.wikibooks.org | 1488 | 0 | -- | <https://es.wikibooks.org/wiki/Mantenimiento_y_Montaje_de_Equipos_Inform%C3%A1ticos/Tema_7/Mantenimiento_correctivo> |
| `es-wb-mantenimiento-preventivo` | instructional / technical | es.wikibooks.org | 1550 | 0 | -- | <https://es.wikibooks.org/wiki/Mantenimiento_y_Montaje_de_Equipos_Inform%C3%A1ticos/Tema_7/Mantenimiento_preventivo> |
| `es-wb-memoria-ram` | instructional / technical | es.wikibooks.org | 1685 | 0 | 431089 | <https://es.wikibooks.org/wiki/Mantenimiento_y_Montaje_de_Equipos_Inform%C3%A1ticos/Tema_2/La_memoria_R.A.M.> |
| `es-wb-placa-base` | instructional / technical | es.wikibooks.org | 1498 | 0 | 427428 | <https://es.wikibooks.org/wiki/Mantenimiento_y_Montaje_de_Equipos_Inform%C3%A1ticos/Tema_2/La_placa_base> |
| `es-wb-s-a-i` | instructional / technical | es.wikibooks.org | 1414 | 0 | 425665 | <https://es.wikibooks.org/wiki/Mantenimiento_y_Montaje_de_Equipos_Informáticos/Tema_5/S.A.I.> |
| `es-wb-tarjeta-grafica` | instructional / technical | es.wikibooks.org | 1996 | 0 | 425618 | <https://es.wikibooks.org/wiki/Mantenimiento_y_Montaje_de_Equipos_Informáticos/Tema_2/La_tarjeta_gráfica> |
| `es-wb-vocabulario` | instructional / technical | es.wikibooks.org | 1994 | 0 | 425614 | <https://es.wikibooks.org/wiki/Mantenimiento_y_Montaje_de_Equipos_Informáticos/Tema_1/Vocabulario> |
| `es-el-0024-procurement` | legal / regulatory | eur-lex.europa.eu | 1745 | 0 | -- | <https://eur-lex.europa.eu/legal-content/ES/TXT/HTML/?uri=CELEX:32014L0024> |
| `es-el-0031-energy` | legal / regulatory | eur-lex.europa.eu | 2130 | 0 | -- | <https://eur-lex.europa.eu/legal-content/ES/TXT/HTML/?uri=CELEX:32010L0031> |
| `es-el-0060-water` | legal / regulatory | eur-lex.europa.eu | 1481 | 0 | -- | <https://eur-lex.europa.eu/legal-content/ES/TXT/HTML/?uri=CELEX:32000L0060> |
| `es-el-0098-waste` | legal / regulatory | eur-lex.europa.eu | 1889 | 0 | -- | <https://eur-lex.europa.eu/legal-content/ES/TXT/HTML/?uri=CELEX:32008L0098> |
| `es-el-1169-food` | legal / regulatory | eur-lex.europa.eu | 1812 | 0 | -- | <https://eur-lex.europa.eu/legal-content/ES/TXT/HTML/?uri=CELEX:32011R1169> |
| `es-el-1999-adaptation` | legal / regulatory | eur-lex.europa.eu | 1679 | 0 | -- | <https://eur-lex.europa.eu/legal-content/ES/TXT/HTML/?uri=CELEX:31999L0031> |
| `es-el-1999-landfill` | legal / regulatory | eur-lex.europa.eu | 1680 | 0 | -- | <https://eur-lex.europa.eu/legal-content/ES/TXT/HTML/?uri=CELEX:31999L0031> |
| `es-wv-astronomia` | practical travel guide | es.wikivoyage.org | 1866 | 0 | 207591 | <https://es.wikivoyage.org/wiki/Astronomía> |
| `es-wv-dinero` | practical travel guide | es.wikivoyage.org | 1568 | 0 | 251448 | <https://es.wikivoyage.org/wiki/Dinero> |
| `es-wv-frutas-verduras` | practical travel guide | es.wikivoyage.org | 1420 | 0 | 190276 | <https://es.wikivoyage.org/wiki/Frutas_y_verduras> |
| `es-wv-inundacion-relampago` | practical travel guide | es.wikivoyage.org | 1203 | 0 | 190172 | <https://es.wikivoyage.org/wiki/Inundación_relámpago> |
| `es-wv-mal-montana` | practical travel guide | es.wikivoyage.org | 1870 | 0 | 185765 | <https://es.wikivoyage.org/wiki/Mal_de_montaña> |
| `es-wv-ninos` | practical travel guide | es.wikivoyage.org | 1536 | 0 | -- | <https://es.wikivoyage.org/wiki/Viajar_con_ni%C3%B1os> |
| `es-wv-parques-nacionales-canada` | practical travel guide | es.wikivoyage.org | 1615 | 0 | 239711 | <https://es.wikivoyage.org/wiki/Parques_nacionales_de_Canadá> |
| `es-wv-sistemas-electricos` | practical travel guide | es.wikivoyage.org | 1921 | 0 | 251958 | <https://es.wikivoyage.org/wiki/Sistemas_eléctricos> |
| `es-wv-tour-guiado` | practical travel guide | es.wikivoyage.org | 1938 | 0 | 241038 | <https://es.wikivoyage.org/wiki/Tour_guiado> |
| `es-wv-transbordador` | practical travel guide | es.wikivoyage.org | 1725 | 0 | 222042 | <https://es.wikivoyage.org/wiki/Transbordador> |
| `es-wv-trenes` | practical travel guide | es.wikivoyage.org | 1412 | 0 | -- | <https://es.wikivoyage.org/wiki/Trenes> |
| `es-wv-viajar-solo` | practical travel guide | es.wikivoyage.org | 1844 | 0 | 185031 | <https://es.wikivoyage.org/wiki/Viajar_solo> |
| `es-wv-viajes-funerarios` | practical travel guide | es.wikivoyage.org | 1910 | 0 | 189739 | <https://es.wikivoyage.org/wiki/Viajes_funerarios> |
| `es-wv-vida-silvestre-africa` | practical travel guide | es.wikivoyage.org | 1750 | 0 | 189740 | <https://es.wikivoyage.org/wiki/Vida_silvestre_en_África> |

Registers: business / annual report 6, encyclopedia 26, instructional / technical 13, legal / regulatory 7, practical travel guide 14.

### Italian (`it`) -- 63 carriers

| id | register | source | chars | repl. | rev | URL |
|---|---|---|---|---|---|---|
| `it-eb-ar24-balance` | business / annual report | ecb.europa.eu | 1849 | 0 | -- | <https://www.ecb.europa.eu/press/annual-reports-financial-statements/annual/html/ecb.ar2024~8402d8191f.it.html> |
| `it-eb-ar24-borrowing` | business / annual report | ecb.europa.eu | 1550 | 0 | -- | <https://www.ecb.europa.eu/press/annual-reports-financial-statements/annual/html/ecb.ar2024~8402d8191f.it.html> |
| `it-eb-ar24-cash` | business / annual report | ecb.europa.eu | 1488 | 0 | -- | <https://www.ecb.europa.eu/press/annual-reports-financial-statements/annual/html/ecb.ar2024~8402d8191f.it.html> |
| `it-eb-ar24-cashusage` | business / annual report | ecb.europa.eu | 1863 | 0 | -- | <https://www.ecb.europa.eu/press/annual-reports-financial-statements/annual/html/ecb.ar2024~8402d8191f.it.html> |
| `it-eb-ar24-stability` | business / annual report | ecb.europa.eu | 1388 | 0 | -- | <https://www.ecb.europa.eu/press/annual-reports-financial-statements/annual/html/ecb.ar2024~8402d8191f.it.html> |
| `it-eb-ar24-target` | business / annual report | ecb.europa.eu | 1635 | 0 | -- | <https://www.ecb.europa.eu/press/annual-reports-financial-statements/annual/html/ecb.ar2024~8402d8191f.it.html> |
| `it-wp-acciaio-inossidabile` | encyclopedia | it.wikipedia.org | 1914 | 0 | 151199686 | <https://it.wikipedia.org/wiki/Acciaio_inossidabile> |
| `it-wp-acque-reflue` | encyclopedia | it.wikipedia.org | 1752 | 0 | 147725220 | <https://it.wikipedia.org/wiki/Trattamento_delle_acque_reflue> |
| `it-wp-alluminio` | encyclopedia | it.wikipedia.org | 1301 | 0 | 151957427 | <https://it.wikipedia.org/wiki/Alluminio> |
| `it-wp-asfalto` | encyclopedia | it.wikipedia.org | 1726 | 0 | 151677557 | <https://it.wikipedia.org/wiki/Asfalto> |
| `it-wp-chiusa` | encyclopedia | it.wikipedia.org | 1685 | 0 | 139828938 | <https://it.wikipedia.org/wiki/Chiusa_%28ingegneria%29> |
| `it-wp-compostaggio` | encyclopedia | it.wikipedia.org | 1590 | 0 | -- | <https://it.wikipedia.org/wiki/Compost> |
| `it-wp-containerizzazione` | encyclopedia | it.wikipedia.org | 1495 | 0 | 147800114 | <https://it.wikipedia.org/wiki/Containerizzazione> |
| `it-wp-discarica` | encyclopedia | it.wikipedia.org | 1395 | 0 | 149959543 | <https://it.wikipedia.org/wiki/Discarica> |
| `it-wp-eolico` | encyclopedia | it.wikipedia.org | 1247 | 0 | -- | <https://it.wikipedia.org/wiki/Energia_eolica> |
| `it-wp-fibra-ottica` | encyclopedia | it.wikipedia.org | 1285 | 0 | 150937664 | <https://it.wikipedia.org/wiki/Fibra_ottica> |
| `it-wp-gesso-materiale` | encyclopedia | it.wikipedia.org | 1181 | 0 | 151261916 | <https://it.wikipedia.org/wiki/Gesso_(materiale)> |
| `it-wp-granito` | encyclopedia | it.wikipedia.org | 1746 | 0 | 151025951 | <https://it.wikipedia.org/wiki/Granito> |
| `it-wp-lana` | encyclopedia | it.wikipedia.org | 1274 | 0 | 149116573 | <https://it.wikipedia.org/wiki/Lana> |
| `it-wp-malta` | encyclopedia | it.wikipedia.org | 1688 | 0 | 144452444 | <https://it.wikipedia.org/wiki/Malta_%28materiale%29> |
| `it-wp-muratura` | encyclopedia | it.wikipedia.org | 1763 | 0 | 150619987 | <https://it.wikipedia.org/wiki/Muratura> |
| `it-wp-osmosi-inversa` | encyclopedia | it.wikipedia.org | 1960 | 0 | 142497504 | <https://it.wikipedia.org/wiki/Osmosi_inversa> |
| `it-wp-pallet` | encyclopedia | it.wikipedia.org | 1881 | 0 | 152139570 | <https://it.wikipedia.org/wiki/Pallet> |
| `it-wp-pista-aviazione` | encyclopedia | it.wikipedia.org | 1698 | 0 | 150787334 | <https://it.wikipedia.org/wiki/Pista_(aviazione)> |
| `it-wp-pompa-di-calore` | encyclopedia | it.wikipedia.org | 1564 | 0 | 151102187 | <https://it.wikipedia.org/wiki/Pompa_di_calore> |
| `it-wp-rotatoria` | encyclopedia | it.wikipedia.org | 1534 | 1 | -- | <https://it.wikipedia.org/wiki/Rotatoria> |
| `it-wp-sabbia` | encyclopedia | it.wikipedia.org | 1697 | 0 | 149712985 | <https://it.wikipedia.org/wiki/Sabbia> |
| `it-wp-tintura` | encyclopedia | it.wikipedia.org | 1828 | 0 | 151672313 | <https://it.wikipedia.org/wiki/Tintura> |
| `it-wp-vetro` | encyclopedia | it.wikipedia.org | 1642 | 0 | 150760942 | <https://it.wikipedia.org/wiki/Vetro> |
| `it-wp-vetro-camera` | encyclopedia | it.wikipedia.org | 1956 | 0 | 150900180 | <https://it.wikipedia.org/wiki/Vetro_camera> |
| `it-wb-acquario` | instructional / technical | it.wikibooks.org | 1511 | 0 | -- | <https://it.wikibooks.org/wiki/Costruire_un_acquario> |
| `it-wb-afm` | instructional / technical | it.wikibooks.org | 1656 | 0 | 490407 | <https://it.wikibooks.org/wiki/Micro_e_nanotecnologia/Microtecnologia/Tecniche_diagnostiche/Scanning_probe/AFM> |
| `it-wb-dry-etching` | instructional / technical | it.wikibooks.org | 1706 | 0 | 484592 | <https://it.wikibooks.org/wiki/Micro_e_nanotecnologia/Microtecnologia/Etching/Dry_Etching> |
| `it-wb-evaporazione` | instructional / technical | it.wikibooks.org | 1739 | 0 | 491481 | <https://it.wikibooks.org/wiki/Micro_e_nanotecnologia/Microtecnologia/Film_sottili/Evaporazione> |
| `it-wb-film-dielettrici` | instructional / technical | it.wikibooks.org | 1604 | 0 | 484714 | <https://it.wikibooks.org/wiki/Micro_e_nanotecnologia/Microtecnologia/Film_sottili/Film_dielettrici> |
| `it-wb-mems` | instructional / technical | it.wikibooks.org | 1300 | 0 | 474525 | <https://it.wikibooks.org/wiki/Micro_e_nanotecnologia/Microtecnologia/Esempi/MEMS> |
| `it-wb-pompe-da-vuoto` | instructional / technical | it.wikibooks.org | 1428 | 0 | -- | <https://it.wikibooks.org/wiki/Micro_e_nanotecnologia/Microtecnologia/Il_vuoto/Pompe_da_vuoto> |
| `it-wb-sem` | instructional / technical | it.wikibooks.org | 1913 | 0 | 484852 | <https://it.wikibooks.org/wiki/Micro_e_nanotecnologia/Microtecnologia/Tecniche_diagnostiche/Interazione_con_fasci_di_elettroni/SEM> |
| `it-wb-snom` | instructional / technical | it.wikibooks.org | 1545 | 0 | 484993 | <https://it.wikibooks.org/wiki/Micro_e_nanotecnologia/Microtecnologia/Tecniche_diagnostiche/Scanning_probe/SNOM> |
| `it-wb-sputtering` | instructional / technical | it.wikibooks.org | 1591 | 0 | 484127 | <https://it.wikibooks.org/wiki/Micro_e_nanotecnologia/Microtecnologia/Film_sottili/Sputtering> |
| `it-wb-tem` | instructional / technical | it.wikibooks.org | 1813 | 0 | 484853 | <https://it.wikibooks.org/wiki/Micro_e_nanotecnologia/Microtecnologia/Tecniche_diagnostiche/Interazione_con_fasci_di_elettroni/TEM> |
| `it-wb-vuoto` | instructional / technical | it.wikibooks.org | 1820 | 0 | 485888 | <https://it.wikibooks.org/wiki/Micro_e_nanotecnologia/Microtecnologia/Il_vuoto> |
| `it-wb-wet-etch` | instructional / technical | it.wikibooks.org | 1883 | 0 | 484534 | <https://it.wikibooks.org/wiki/Micro_e_nanotecnologia/Microtecnologia/Etching/Wet_etch> |
| `it-el-0024-procurement` | legal / regulatory | eur-lex.europa.eu | 1830 | 0 | -- | <https://eur-lex.europa.eu/legal-content/IT/TXT/HTML/?uri=CELEX:32014L0024> |
| `it-el-0031-energy` | legal / regulatory | eur-lex.europa.eu | 1933 | 0 | -- | <https://eur-lex.europa.eu/legal-content/IT/TXT/HTML/?uri=CELEX:32010L0031> |
| `it-el-0060-water` | legal / regulatory | eur-lex.europa.eu | 1527 | 0 | -- | <https://eur-lex.europa.eu/legal-content/IT/TXT/HTML/?uri=CELEX:32000L0060> |
| `it-el-0098-waste` | legal / regulatory | eur-lex.europa.eu | 1804 | 0 | -- | <https://eur-lex.europa.eu/legal-content/IT/TXT/HTML/?uri=CELEX:32008L0098> |
| `it-el-1169-food` | legal / regulatory | eur-lex.europa.eu | 1623 | 0 | -- | <https://eur-lex.europa.eu/legal-content/IT/TXT/HTML/?uri=CELEX:32011R1169> |
| `it-el-1999-adaptation` | legal / regulatory | eur-lex.europa.eu | 1619 | 0 | -- | <https://eur-lex.europa.eu/legal-content/IT/TXT/HTML/?uri=CELEX:31999L0031> |
| `it-el-1999-landfill` | legal / regulatory | eur-lex.europa.eu | 1497 | 0 | -- | <https://eur-lex.europa.eu/legal-content/IT/TXT/HTML/?uri=CELEX:31999L0031> |
| `it-wv-aggregatori` | practical travel guide | it.wikivoyage.org | 1853 | 0 | 921726 | <https://it.wikivoyage.org/wiki/Aggregatori> |
| `it-wv-all-aeroporto` | practical travel guide | it.wikivoyage.org | 1350 | 0 | 889644 | <https://it.wikivoyage.org/wiki/All'aeroporto> |
| `it-wv-arrivo-in-aereo` | practical travel guide | it.wikivoyage.org | 1366 | 0 | 805072 | <https://it.wikivoyage.org/wiki/Arrivo_in_aereo> |
| `it-wv-autobus` | practical travel guide | it.wikivoyage.org | 1655 | 0 | 907643 | <https://it.wikivoyage.org/wiki/Viaggiare_in_autobus> |
| `it-wv-bicicletta` | practical travel guide | it.wikivoyage.org | 1602 | 0 | 936974 | <https://it.wikivoyage.org/wiki/Andare_in_bicicletta> |
| `it-wv-cicloturismo` | practical travel guide | it.wikivoyage.org | 1578 | 0 | -- | <https://it.wikivoyage.org/wiki/Cicloturismo> |
| `it-wv-crimini` | practical travel guide | it.wikivoyage.org | 1508 | 0 | 875849 | <https://it.wikivoyage.org/wiki/Crimini> |
| `it-wv-documenti` | practical travel guide | it.wikivoyage.org | 1735 | 0 | 828594 | <https://it.wikivoyage.org/wiki/Documenti> |
| `it-wv-guidare-in-europa` | practical travel guide | it.wikivoyage.org | 1624 | 0 | 907756 | <https://it.wikivoyage.org/wiki/Guidare_in_Europa> |
| `it-wv-guidare-in-islanda` | practical travel guide | it.wikivoyage.org | 1596 | 0 | 906169 | <https://it.wikivoyage.org/wiki/Guidare_in_Islanda> |
| `it-wv-siti-archeologici` | practical travel guide | it.wikivoyage.org | 1940 | 0 | 901823 | <https://it.wikivoyage.org/wiki/Siti_archeologici> |
| `it-wv-treno` | practical travel guide | it.wikivoyage.org | 1381 | 2 | -- | <https://it.wikivoyage.org/wiki/Viaggiare_in_treno> |
| `it-wv-viaggiare-in-barca` | practical travel guide | it.wikivoyage.org | 1393 | 0 | 901890 | <https://it.wikivoyage.org/wiki/Viaggiare_in_barca> |

Registers: business / annual report 6, encyclopedia 24, instructional / technical 13, legal / regulatory 7, practical travel guide 13.

### English (`en`) -- 40 carriers

| id | register | source | chars | repl. | rev | URL |
|---|---|---|---|---|---|---|
| `en-eb-ar24-balance` | business / annual report | ecb.europa.eu | 1438 | 0 | -- | <https://www.ecb.europa.eu/press/annual-reports-financial-statements/annual/html/ecb.ar2024~8402d8191f.en.html> |
| `en-eb-ar24-borrowing` | business / annual report | ecb.europa.eu | 1334 | 0 | -- | <https://www.ecb.europa.eu/press/annual-reports-financial-statements/annual/html/ecb.ar2024~8402d8191f.en.html> |
| `en-eb-ar24-cash` | business / annual report | ecb.europa.eu | 1804 | 0 | -- | <https://www.ecb.europa.eu/press/annual-reports-financial-statements/annual/html/ecb.ar2024~8402d8191f.en.html> |
| `en-eb-ar24-cashusage` | business / annual report | ecb.europa.eu | 1503 | 0 | -- | <https://www.ecb.europa.eu/press/annual-reports-financial-statements/annual/html/ecb.ar2024~8402d8191f.en.html> |
| `en-eb-ar24-stability` | business / annual report | ecb.europa.eu | 1271 | 0 | -- | <https://www.ecb.europa.eu/press/annual-reports-financial-statements/annual/html/ecb.ar2024~8402d8191f.en.html> |
| `en-eb-ar24-target` | business / annual report | ecb.europa.eu | 1774 | 0 | -- | <https://www.ecb.europa.eu/press/annual-reports-financial-statements/annual/html/ecb.ar2024~8402d8191f.en.html> |
| `en-wp-asphalt` | encyclopedia | en.wikipedia.org | 1307 | 0 | 1363587735 | <https://en.wikipedia.org/wiki/Asphalt_concrete> |
| `en-wp-compost` | encyclopedia | en.wikipedia.org | 1494 | 0 | -- | <https://en.wikipedia.org/wiki/Compost> |
| `en-wp-concrete` | encyclopedia | en.wikipedia.org | 1868 | 0 | 1369945274 | <https://en.wikipedia.org/wiki/Concrete> |
| `en-wp-cotton` | encyclopedia | en.wikipedia.org | 1728 | 0 | 1371300439 | <https://en.wikipedia.org/wiki/Cotton> |
| `en-wp-heatpump` | encyclopedia | en.wikipedia.org | 1524 | 0 | 1370570611 | <https://en.wikipedia.org/wiki/Heat_pump> |
| `en-wp-intermodal-container` | encyclopedia | en.wikipedia.org | 1761 | 0 | 1368810122 | <https://en.wikipedia.org/wiki/Intermodal_container> |
| `en-wp-optical-fiber` | encyclopedia | en.wikipedia.org | 1540 | 0 | 1366583241 | <https://en.wikipedia.org/wiki/Optical_fiber> |
| `en-wp-permafrost` | encyclopedia | en.wikipedia.org | 1275 | 0 | 1369948943 | <https://en.wikipedia.org/wiki/Permafrost> |
| `en-wp-polyvinyl-chloride` | encyclopedia | en.wikipedia.org | 1441 | 0 | 1371252910 | <https://en.wikipedia.org/wiki/Polyvinyl_chloride> |
| `en-wp-roundabout` | encyclopedia | en.wikipedia.org | 1378 | 0 | -- | <https://en.wikipedia.org/wiki/Roundabout> |
| `en-wp-stainless-steel` | encyclopedia | en.wikipedia.org | 1681 | 0 | 1369190039 | <https://en.wikipedia.org/wiki/Stainless_steel> |
| `en-wp-ventilation` | encyclopedia | en.wikipedia.org | 1836 | 0 | 1370484743 | <https://en.wikipedia.org/wiki/Ventilation_%28architecture%29> |
| `en-wp-windturbine` | encyclopedia | en.wikipedia.org | 1506 | 2 | -- | <https://en.wikipedia.org/wiki/Wind_turbine> |
| `en-wb-bike-cables` | instructional / technical | en.wikibooks.org | 1761 | 0 | 3831498 | <https://en.wikibooks.org/wiki/Bicycles/Maintenance_and_Repair/Cables_and_Housings> |
| `en-wb-bike-saddles` | instructional / technical | en.wikibooks.org | 1877 | 0 | 3538944 | <https://en.wikibooks.org/wiki/Bicycles/Maintenance_and_Repair/Saddles_and_Seatposts> |
| `en-wb-composite-materials` | instructional / technical | en.wikibooks.org | 1574 | 0 | 1683817 | <https://en.wikibooks.org/wiki/Materials_Science/Materials/Composite_Materials> |
| `en-wb-introduction` | instructional / technical | en.wikibooks.org | 1602 | 0 | 3659374 | <https://en.wikibooks.org/wiki/Materials_Science/Introduction> |
| `en-wb-material-processing` | instructional / technical | en.wikibooks.org | 1952 | 0 | 4094815 | <https://en.wikibooks.org/wiki/Materials_Science/Material_Processing> |
| `en-wb-rear-derailleur` | instructional / technical | en.wikibooks.org | 1456 | 0 | -- | <https://en.wikibooks.org/wiki/Bicycles/Maintenance_and_Repair/Derailleurs/Rear_Derailleur_Adjustment> |
| `en-wb-road-safety` | instructional / technical | en.wikibooks.org | 1594 | 1 | -- | <https://en.wikibooks.org/wiki/Bicycles/Road_Safety> |
| `en-el-0024-procurement` | legal / regulatory | eur-lex.europa.eu | 1596 | 0 | -- | <https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32014L0024> |
| `en-el-0031-energy` | legal / regulatory | eur-lex.europa.eu | 1471 | 0 | -- | <https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32010L0031> |
| `en-el-0060-water` | legal / regulatory | eur-lex.europa.eu | 1293 | 0 | -- | <https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32000L0060> |
| `en-el-0098-waste` | legal / regulatory | eur-lex.europa.eu | 1639 | 0 | -- | <https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32008L0098> |
| `en-el-1169-food` | legal / regulatory | eur-lex.europa.eu | 1523 | 0 | -- | <https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32011R1169> |
| `en-el-1999-adaptation` | legal / regulatory | eur-lex.europa.eu | 1527 | 0 | -- | <https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:31999L0031> |
| `en-el-1999-landfill` | legal / regulatory | eur-lex.europa.eu | 1158 | 0 | -- | <https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:31999L0031> |
| `en-wv-airline-alliances` | practical travel guide | en.wikivoyage.org | 1481 | 0 | 5309322 | <https://en.wikivoyage.org/wiki/Airline_alliances> |
| `en-wv-altitude-sickness` | practical travel guide | en.wikivoyage.org | 1698 | 0 | 5330576 | <https://en.wikivoyage.org/wiki/Altitude_sickness> |
| `en-wv-american-cuisine` | practical travel guide | en.wikivoyage.org | 1850 | 0 | 5299483 | <https://en.wikivoyage.org/wiki/American_cuisine> |
| `en-wv-bus-travel` | practical travel guide | en.wikivoyage.org | 1648 | 0 | 5222349 | <https://en.wikivoyage.org/wiki/Bus_travel> |
| `en-wv-camping` | practical travel guide | en.wikivoyage.org | 1608 | 0 | 5291330 | <https://en.wikivoyage.org/wiki/Camping> |
| `en-wv-rail-enthusiasts` | practical travel guide | en.wikivoyage.org | 1533 | 0 | -- | <https://en.wikivoyage.org/wiki/Travel_for_rail_enthusiasts> |
| `en-wv-rail-tips` | practical travel guide | en.wikivoyage.org | 1474 | 2 | -- | <https://en.wikivoyage.org/wiki/Tips_for_rail_travel> |

Registers: business / annual report 6, encyclopedia 13, instructional / technical 7, legal / regulatory 7, practical travel guide 7.


Register spread per language: 7 legal/regulatory, 6 business/annual report, 13–27 encyclopedia,
7–13 instructional/technical, 7–14 practical travel guide.

The EUR-Lex carriers are the same seven (act, article) pairs in all six languages  -- 
Regulation (EU) No 1169/2011, Directive 2010/31/EU, Directive 2008/98/EC art. 11,
Directive 2014/24/EU art. 42, Directive 2000/60/EC art. 11, and Directive 1999/31/EC
art. 5 and art. 16 -- and the ECB carriers are the same six sections of one report. That is
deliberate: every language version is a genuine human translation of the same text rather than
a generated one, so a per-language difference in those two registers cannot be an artefact of
the text differing. It is also why those two registers are held equal across languages while
the Wikimedia ones are not: nothing forces `nl.wikipedia` and `es.wikipedia` to have articles
on the same subjects, and pretending otherwise would mean rejecting good carriers to preserve
a symmetry the sources do not have.

## How the cases are built

```
cargo run -p pii-detection-eval --bin inject
```

Reads `manifest.json` and the `.txt` files, writes `benchmark/cases/carriers.json`, and
prints the boundary/mid-sentence split it produced. It is deterministic: a test in
`inject.rs` rebuilds the fixture and asserts it equals the committed file byte for byte, so
the answer key and the tool that wrote it cannot drift apart unnoticed.

**Carrier order in `manifest.json` is load-bearing.** `inject` rotates the injection plan by a
carrier's position in the list and derives its seed from that position, so appending carriers
leaves every earlier carrier's cases byte-identical and reordering silently rewrites all of
them. Both expansions appended: the 162 cases built from the original 54 carriers and the 462
built from the first 154 are unchanged byte for byte, checked directly against the committed
file each time. Append; do not reorder.

Three cases per carrier, 963 in total:

| id suffix | identifiers | what it is for |
|---|---|---|
| `-d0` | none injected | a pure negative. A detector that masks nothing scores 100% precision on a fixture where every case contains PII; these are what stop that. |
| `-d1` | one | the ordinary case. |
| `-dn` | three to five | several identifiers in one document, which is what a real ticket or contract looks like. |

The 314 `-d0` cases from carriers whose screening added no `must_mask` value are true pure
negatives: empty `must_mask`. The other 7 carry the replaced personal names, so they are
ordinary cases rather than negatives. (`en-wp-windturbine` had two replacements and is still a
pure negative: both were eponyms, so both went to `must_clear`.)

### Where the values land

Values go in at sentence boundaries and mid-sentence, because a detector that only works at
line starts is a detector that fails on prose. The two kinds alternate by index within a
case, and **which kind index zero gets is drawn from the seed** -- without that, every
one-value case would be a boundary case, which is exactly what the first version of this
corpus did: it ran **170 boundary to 82 mid-sentence** while claiming equality "by
construction", backed by a test using the one carrier shape where the fault is invisible.

**The committed fixture is 714 boundary to 741 mid-sentence -- 49% boundary.** `inject` prints
that figure on every run and a test asserts it stays between 42% and 58%, measured on the
fixture itself rather than on a demonstration carrier.

Insertion only splices -- it never rewrites -- so every sentence the carrier already had
survives verbatim.

### Every identifier satisfies its real national rule

Every checksummed value is drawn *against a rule*, never invented and hoped over. Where the
service implements the rule, the value is drawn against
`pii_detection_rules::validator::Validator` itself: the free digits are fixed at random and the
check digit is whichever one the validator accepts. Nothing transcribes a checksum, so
nothing can drift from one. That covers the Dutch BSN, the German Steuer-ID check digit,
card numbers and IBANs.

**Where the service does not implement the rule, drawing against the service is no check at
all, and that is a way for the benchmark to flatter us.** The first version of this corpus
did exactly that and the consequences were:

- **0 of 22 VAT numbers** satisfied their country's rule. `universal.toml`'s VAT recogniser
  carries `validator = "none"`, so any string of the right shape passed. One value,
  `ESH0912908F`, was malformed by construction: Spanish entity type `H` takes a digit check
  character and never a letter.
- **All 24 Steuer-IDs** satisfied the check digit and violated the BZSt *structure* rule  -- 
  which is precisely `training/generate.py::_steuer_id`'s blind spot, inherited because both
  validate the same partial rule.
- About one IBAN in fifty carried check digits `00` or `01`, which `iban_mod97` accepts and
  ISO 13616 does not issue.

A detector applying the real rule would reject all of those and be marked **wrong** by our own
answer key, while the detector's looser rule is marked right. That is a measurement that flatters
us, built into the instrument meant to gate releases -- the exact thing the carrier corpus
exists to remove.

Every identifier satisfies its real rule:

| identifier | rule applied | how it is checked |
|---|---|---|
| Dutch BSN | 11-proef | drawn against `Validator::Elfproef` |
| German Steuer-ID | ISO 7064 MOD 11,10 **and** the BZSt structure rule | check digit drawn against `Validator::GermanTaxId`; structure generated, verified by an independent transcription |
| IBAN | mod-97, check digits in **02–98** | drawn against `Validator::IbanMod97`, range asserted |
| Card number | Luhn, published test BINs only | drawn against `Validator::Luhn` |
| Dutch BTW | 11-proef on the RSIN, `B`, sub-number | transcribed rule, cross-checked against `Validator::Elfproef` |
| German USt-IdNr | ISO 7064 MOD 11,10 over eight digits | transcribed rule, independent verifier |
| French TVA | Luhn-valid SIREN, key = (12 + 3·(SIREN mod 97)) mod 97 | transcribed rule, independent verifier |
| Spanish NIF (CIF) | digit-check entity types `A`/`B` only | transcribed rule, independent verifier |
| Italian partita IVA | Luhn over eleven digits | transcribed rule, independent verifier |
| Telephone number | the national numbering plan | drawn against `phonenumber` |

Each verifier is transcribed from the public specification **separately from the generator it
checks** -- the same two-transcription discipline `training/generate.py` applies to its own
generators, and for the same reason: one slip is likely, the same slip made twice
independently is not. Two further tests check the committed fixture rather than the
generator, and two more check that the verifiers reject the values the old generator
produced, so a verifier that accepted everything would fail. Those fixture-wide tests cover
the 321-carrier corpus, not only the original 54: they read `carriers.json` as committed.

### Which country identifiers are injected -- and the blind spot that leaves

Only the **Dutch BSN** (`NATIONAL_ID`) and the **German Steuer-ID** (`TAX_ID`), because those
are the two country packs that exist. Both also appear in English-language carriers, which is
not a contrivance: an English ticket quoting a Dutch BSN is the use case. English
carriers carry a Dutch VAT number for the same reason -- no United Kingdom prefix exists in
the recogniser's country list.

**The French NIR, the Spanish DNI and the Italian codice fiscale are not in this fixture, and
Azure AI Language detects all three.** They are left out because nothing in the detector can detect
them today, so cases containing them would fail the per-type gate for a scoping reason and
drown out a real regression. That makes this a blind spot the benchmark *cannot see*: a
decision to turn Azure off cannot be supported by these cases as far as those three
identifiers are concerned. `training/generate.py` already has generators for all three  -- 
they would need the same national-rule review the VAT generators just had before being used
here. The expansion did not change this, and tripling the Spanish and Italian carriers
without adding the DNI or the codice fiscale makes it slightly more conspicuous, not less.

## What the expanded corpus measures, against the shipped service

Run 2026-08-26 against the shipped model-backed service (`rules 1.1.1`, model
`2026-08-24-student-6l-static-qdq`, all eleven types), the two fixtures scored back to back
against the same container:

| | 649 cases (154 carriers) | 1,150 cases (321 carriers) |
|---|---|---|
| overall recall | 0.9942 (1029/1035) | **0.9941 (1838/1849)** |
| overall precision | 0.8532 | **0.8625** |
| `must_clear` violated | 32 / 1,106 (2.9%) | **52 / 2,108 (2.5%)** |
| spurious spans | 145 | **241** |
| gates | 23 pass / 3 fail / 1 not measured | **23 pass / 3 fail / 1 not measured** |

**No gate verdict moved.** The same three fail -- `PHONE_NUMBER` precision and the two
non-inferiority gates -- and the p95 latency gate is unmeasured for the same reason as before
(no `--latency` run supplied). That is the honest headline: this expansion bought better
estimates, not a different verdict.

Underneath it, three things did move, and they move in different directions:

- **`PHONE_NUMBER` precision rose 0.8667 → 0.9255 without a single error going away.** The
  error count is 12 both times (6 `must_clear` violations, 6 spurious spans); the true-positive
  denominator went 90 → 161. The gate still fails at the 0.97 line. Read this as a
  denominator effect and nothing else -- exactly the class of artefact the 2026-08-22 expansion
  produced twice in the other direction.
- **`PERSON` recall fell 0.9910 (110/111) → 0.9780 (178/182)**: one miss became four, on
  natural-prose carriers the smaller corpus did not contain. Azure's own `PERSON` recall fell
  too, 1.0000 (111/111) → 0.9945 (181/182), so the non-inferiority bar moved with it -- but the
  gap widened from 0.9 points to 1.7. This is behaviour, not denominator.
- **`STREET_ADDRESS` precision rose 0.5282 → 0.6004** while its recall rose 0.9850 → 0.9891.
  The model still over-fires on out-of-domain prose (159 of the 241 spurious spans are
  `STREET_ADDRESS`), but the new carriers are proportionally less bad for it than the old ones.

Per-language recall, before and after, is the measurement this expansion existed to improve:

| language | 649 cases | 1,150 cases | |
|---|---|---|---|
| `nl` | 1.0000 (n=198) | 1.0000 (n=306) | flat |
| `de` | 0.9946 (n=184) | 0.9968 (n=308) | up |
| `en` | 0.9908 (n=218) | 0.9938 (n=320) | up |
| `fr` | 1.0000 (n=152) | **0.9968 (n=308)** | **down** |
| `es` | 1.0000 (n=138) | **0.9935 (n=306)** | **down** |
| `it` | 0.9793 (n=145) | 0.9834 (n=301) | up |

**Two languages got worse and that is the expansion working.** `fr` and `es` both read a
perfect 1.0000 on the 649-case fixture. They are not perfect; they were unsampled. At n=138 a
single miss would have been 0.7 points and there were none to find, which is a statement about
the sample, not about the detector. At n=306 the misses exist and are counted. A number that
only ever improves when you measure it more carefully is a number nobody should trust.

The rules-only half of the service, scored over the 963 carrier cases alone, remains at
**0 spurious spans and 0 `must_clear` violations on 1,956 values** and finds 1,184 of 1,185
carrier `must_mask` values -- see the screening verification above for the single miss and why
it is not fixed here.

## Known limitations

1. **The per-type denominators are still small, even though the per-language ones no longer
   are.** The carrier floor the design asked for is met (40–66 per language); `NATIONAL_ID`
   (n=118) and `TAX_ID` (n=120) are still carried by two languages each, and a per-type gate
   still turns on a value or two. Stated at the top and restated here because closing the
   headline shortfall makes it easy to stop saying.
2. **The business register covers annual-report prose, not company-filing layout.** Contact
   blocks, letterheads, VAT lines in footers and addresses in signature position are still
   untested by any carrier. See the business-register section for why a carrier corpus is the
   wrong instrument for that and what is.
3. **The Azure baseline is one collection, on one day, of a service that changes under a
   fixed `modelVersion="latest"` request.** `azure-2026-08-26.json` matches this fixture case
   for case, so the size mismatch that used to sit here is gone -- but it is still a
   point-in-time snapshot, and both its date and its reported `azure_model_version` have to be
   quoted with any number taken from it.
4. **`PERSON` and `STREET_ADDRESS` are model-produced, and the model is the shipped 6-layer
   student.** Both non-inferiority gates fail against Azure. Neither is a scoping artefact any
   more; they are measured results.
5. **Every carrier is long** (763–2130 characters). Short-prompt behaviour is covered by the
   128 legacy cases and by the hand-written cases, not here.
6. **No chat register.** Public support threads were rejected on purpose: their licensing is
   rarely establishable and they are full of real signatures, the two things a carrier must
   not have. Chat register is the hand-written cases' hand-authored set.
7. **Email addresses use `example.com`/`.net`/`.org`** (RFC 2606, unregisterable). No address
   here can reach a real mailbox, and the cases are correspondingly easier than production
   traffic.
8. **Telephone numbers are drawn from live allocated ranges.** The reserved drama ranges are
   marked invalid by `phonenumber`, so a detector that consults the numbering plan is right to
   ignore them and every one would be a fixture-permanent miss. A generated number could in
   principle coincide with a real subscriber's; card numbers avoid the same problem because
   the schemes publish test BINs, and those are used.
9. **Wikimedia carriers can change under their URLs.** The text is a snapshot and the recorded
   URL points at the live article. Carriers added on 2026-08-22 and 2026-08-26 record `oldid`,
   the revision the text was taken from; the original 54 do not, and cannot be given one
   retrospectively without re-deriving which revision they came from.
10. **The corpus is more structurally impersonal than its sources are, and the 2026-08-26
    expansion made that worse, not better.** Ten candidates were rejected or re-cut in the
    2026-08-22 screen and eight more in the 2026-08-26 one, and the later round adopted an
    explicitly stricter rule: an article whose first clean window names a person is dropped
    rather than re-cut, because its later windows are usually more biographical, not less.
    That rule buys a cleaner answer key and pays for it in representativeness. **The
    consequence is concrete:** of the 165 `PERSON` values in the carrier cases, **30** stand in
    prose that was written about a person -- the ten screening replacements, each appearing in
    its carrier's three cases, and all ten from the original 54 carriers. The other 135 were
    injected by the generator into a sentence built to carry them. The zero-false-positive
    result and the `PERSON` recall figure are both measured on what survived that filter, and
    production traffic will not have been filtered.
11. **Three of the six languages are above our 40–60 band rather than inside it**
    (`fr` 61, `it` 63, `es` 66) because the thinnest languages were prioritised twice running.
    The corpus is deliberately unbalanced in the direction of the languages the detector is
    weakest in; a macro-average across languages weights those three more heavily in document
    terms than `en` at 40. Per-language figures are computed per language and are unaffected;
    a whole-fixture figure is not.

## Adding carriers

1. Establish the licence first. No licence, no carrier. If it is ShareAlike, check that
   [`../LICENSE-CARRIERS.md`](../LICENSE-CARRIERS.md) already covers it -- and if your addition
   changes the per-source file counts, update that file's counts in the same commit.
2. Prefer structurally impersonal text: regulations over judgments, technical documentation
   over support threads, articles about places and processes over articles about people.
3. Read the whole thing. Replace every personal name, address and telephone number with a
   synthetic equivalent and record the replacement in `manifest.json` -- with a `type` if it is
   a person, without one if it is an eponym you have a reason to substitute rather than name.
   Do not record the original. If the document needs more than two replacements, take a
   different document.
4. **Append** the `manifest.json` row; never reorder existing ones. Add the `.txt`, record the
   `oldid` for a Wikimedia source, then re-run `inject`. The determinism test will fail until
   the regenerated fixture is committed alongside.
5. Update the counts this file and `../LICENSE-CARRIERS.md` state, the fixture-size guard in
   `crates/eval/src/lib.rs`, and `pii_detection_eval::gates::SAMPLE_SIZE_CAVEAT` -- the
   per-carrier influence it quotes is `1 / (carriers per language)` and goes stale silently.
6. If the carrier needs a new kind of identifier, give its national rule the same treatment
   the VAT rules got: generate against the real rule, and verify with a transcription written
   separately from the generator.
7. **Re-run `training/check_contamination.py` and commit the fresh dated report.** Growing
   `benchmark/carriers/`/`benchmark/cases/` can silently pull a negative-corpus document into
   the benchmark it was already screened against, which is exactly what happened on
   2026-08-26: a report that was true when stamped was wrong by the time it was read. This
   step is not optional and not just a note: `crates/eval/src/lib.rs`'s
   `the_committed_contamination_report_matches_the_live_fixture_case_count` test compares the
   fixture's live case count against the number the latest `training/reports/contamination-
   <date>.md` stamped on itself, and fails `cargo test --workspace` -- which every push and PR
   runs in CI, not only the weekly Azure job -- the moment the two disagree. There is no `--out`
   flag that skips it and no fixture edit that goes unnoticed by it.

   ```sh
   cd training && .venv/bin/python check_contamination.py \
       --data-dir data-merged --data-dir data-negatives --data-dir data-positives \
       --report-out reports/contamination-<today>.md
   ```

   It must print `VERDICT: CLEAN`. If it prints `CONTAMINATED`, read the escalation table in
   the report before touching anything -- see `training/negatives/README.md` on why an 8-gram collision that dies at n=10 is not the
   same finding as one that survives to n=20.
