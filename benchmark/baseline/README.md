# The Azure baseline

`azure-<date>.json` is Azure AI Language's Text PII API run once over every case in
`benchmark/cases/`, scored by `pii_detection_eval::score` -- the exact scorer that scores the detector's
own engine (`fixture-spans`). This is the comparator the two non-inferiority gates read.
It is produced and rescored by `crates/eval/src/bin/baseline.rs`; that file's own
header documents the API contract, the category mapping, and the cost gate in full. This
README is about the *data*, not the tool.

## RESCORED IN PLACE 2026-09-12 (ninth revision): the category map was discarding 27 categories Azure returns

`azure-2026-09-12.json`'s `summary` and `disagreements` were **rescored in place** from its own
unchanged `raw_batches` after `category_map` in `crates/eval/src/bin/baseline.rs` was
corrected. No Azure call was made and nothing was spent; `command`, `cost` and `raw_batches` are
untouched and still describe the original $3.9470 `--confirm` run -- the same treatment, for the
same reason, as the 2026-08-24 in-place rescore of `azure-2026-08-23.json` recorded further down.

**What was wrong.** `category_map` was a transcription of the category list of Temperate's earlier Azure client from the
Temperate repository -- 12 categories -- and `azure_entities` silently drops any category not in it.
Temperate's `PiiEntityType` has 11 cases; this fixture declares 13 types. So a category Azure returns
for a class *the fixture asks about* was dropped before scoring because *a different system's*
enum had nowhere to put it, and a dropped category scores exactly like a missed value. 27 such
categories were being discarded -- nine `*DriversLicenseNumber`, eight national-identifier, four
tax, three VAT, two passport, one debit-card. Every one of those drops made Azure look worse.
Twelve of them are present in `azure-2026-08-22.json`, the first baseline ever collected here, so
**every published comparison in this repository has understated Azure.**

**What it cost Azure, at 2,041 cases:** overall recall 0.5632 -> **0.8533** (789 values),
`DRIVING_LICENCE` 0.0087 -> 0.8866, `TAX_ID` 0.0000 -> 1.0000, `NATIONAL_ID` 0.3563 -> 0.9551,
`VAT_NUMBER` 0.0000 -> 0.4820, `PASSPORT_NUMBER` 0.4926 -> 0.6250. Precision moves the other way,
0.9017 -> 0.4970, and over-masking 76 -> 236 of 4,307, because a category that is not scored is
not charged for its false positives either. The detector's own figures are unchanged (recall 0.9945,
over-masking 94/4,307, checked live, not assumed) and **no gate verdict moved**: the two
Azure-relative gates read only `PERSON` and `STREET_ADDRESS` recall, which no newly mapped
category touches (204/205 and 274/275, before and after). Full account, including the per-category
sweep and the one disclosed judgement call (`USDriversLicenseNumber`), is
`training/reports/pii-detection-vs-azure-2026-08-23.md`'s ninth revision.

**Two claims this README and `baseline.rs` both made are now known to be false** and have been
removed rather than softened: Azure does *not* "return nothing at all for VAT numbers" (it returns
`DEValueAddedNumber`, `ITValueAddedTaxNumber`, `NLValueAddedTaxNumber`), and it does *not* only
classify a German Steuer-ID as a phone number (it also returns `DETaxIdentificationNumber`).
`no_azure_equivalent` now names `CREDENTIAL` alone -- re-verified against every category in every
committed raw response, mapped and unmapped: Azure covers 0 of the 211 `CREDENTIAL` values.

**The guard that was missing now exists.** `every_category_azure_returns_is_mapped_or_explicitly_unmapped`
in `baseline.rs` reads every `azure-*.json` in this directory, extracts every distinct `category`
from Azure's raw responses, and fails unless each is either mapped or listed in
`deliberately_unmapped` with a reason. A category Azure starts returning fails the build on the
collection that first sees it. Each baseline file now also carries
`deliberately_unmapped_categories`, so a reader of the data can see what was refused and why
without reading the source.

## Why this file has to be dated and versioned

Azure's model changes under a fixed `modelVersion="latest"` request without this repository
doing anything. A baseline that does not say which Azure model produced it, and when, would
silently become a comparison against a service that has since changed. Every file here
carries `collected_at` (the run date) and `azure_model_version` (Azure's own reported string,
`results.modelVersion`) for exactly that reason. **Quote both whenever you quote a number from
this file.**

## RESOLVED 2026-09-12: the fixture grew again (`DRIVING_LICENCE`), and this file now matches it

`benchmark/cases/` grew to **2,041** cases (`DRIVING_LICENCE` -- Germany and Britain first,
commit `fa98768`, then the Netherlands, France and Italy via a context-carried ladder, commits
`19f3f41`/`d9a2206`/`3efd378`, see `training/reports/driving-licence.md`), which left the
baseline below stale at 1,676 and `load_baseline`'s size guard refusing every comparison against
the full fixture:

> the --baseline file was collected over 1676 case(s) but the fixture being scored has 2041;
> refusing to compare recall across two different populations

**That was the guard working, not a regression** -- the same guard, doing the same job, that
refused a 349-vs-649 mismatch, a 1,150-vs-1,452 mismatch and a 1,452-vs-1,676 mismatch before it
(re-verified this time too: pointing `eval` at this superseded file against the 2,041-case
fixture reproduces the refusal verbatim). It is resolved the same way every one of those was: by
collecting Azure over the population that now exists, not by relaxing the guard.
`azure-2026-09-12.json` (below) is that collection, estimated at $3.9470 (against a ~$4
expectation set before the run and an $8.00 human-set stop threshold) and confirmed at exactly
that estimate.

**[Ninth-revision note] The paragraph below is the eighth revision's account, kept as the record
of what was published. It measured Azure's licence coverage correctly and then left the 3/344
figure in the tables anyway. Those nine categories -- it is nine, not eight, `CA` is the one it
missed -- are in `category_map` as of 2026-09-12 and `DRIVING_LICENCE` reads 305/344 (0.8866). See
the top of this file.**

**Azure detects almost none of the 344 `DRIVING_LICENCE` values through the category map its own
driver uses -- 3/344, all incidental `PhoneNumber` overlap -- but this is not the same finding as
`CREDENTIAL`'s, and reporting it that way would be false.** Checked past the map, against Azure's
raw, unfiltered response, Azure carries eight dedicated per-country `*DriversLicenseNumber`
categories (`DE`, `UK`, `FR`, `US`, `EU`, `AU`, `JP`, `IT`) that this map does not carry --
because Temperate's own driver has no `PiiEntityType::DRIVING_LICENCE` to map any of them onto -- and
those categories fully cover **305 of 344 (88.7%)** of the fixture's declared licence values,
confidence 0.65-0.75 throughout. Broken out by the licence's own issuing country: Germany 110/111
and Britain 110/111, each through a single, correctly-named category; France 38/41; the
Netherlands 41/41, entirely through the generic `EUDriversLicenseNumber` despite no
country-specific category existing; Italy 3/40, the worst of the five. The specific hypothesis
this task set out to check -- that `NLCitizensServiceNumber`, which incidentally caught 49 of 58
Belgian national numbers at an earlier revision, might also fire on a Dutch licence on shape
alone (a BSN is 9 digits, a Dutch licence 10) -- was checked and does not hold: zero of 41. This
is a **found, not fixed, and not fixable from this side**: closing it means a platform taxonomy
change, out of scope for this repository and forbidden outright by this task's own constraint
against modifying that repository.

A second, independent finding fell out of the same raw-response check: Azure's `PhoneNumber`
category over-masks 53 `must_clear` values at 2,041 cases, up from 23 at 1,676 -- all 30 new
violations inside `driving-licence-*.json` cases specifically, the same digit-string-mislabelled
pattern this README's own "PHONE_NUMBER precision" section below already named, now generalising
further.

`PERSON` and `STREET_ADDRESS` -- the two types the gates check on non-inferiority -- move by
denominator only and not at all, respectively: `PERSON` recall ticked up (191/195 = 0.9795 ->
201/205 = 0.9805, the ten new occurrences are existing carriers' own screening replacements,
found by both sides) and `STREET_ADDRESS` did not move at all (272/275 = 0.9891 exactly, before
and after -- no `DRIVING_LICENCE` case injects an address). Full per-type, per-language and
gate-verdict detail, before and after, is `training/reports/pii-detection-vs-azure-2026-08-23.md`'s
newest revision.

## RESOLVED 2026-08-28: the fixture grew again (`CREDENTIAL`), and this file now matches it

`benchmark/cases/` grew to **1,676** cases on 2026-08-27 (the `CREDENTIAL` entity type -- API
keys, JWTs and PEM blocks, see `training/reports/credentials.md`), which left the baseline
below stale at 1,452 and `load_baseline`'s size guard refusing every comparison against the
full fixture:

> the --baseline file was collected over 1452 case(s) but the fixture being scored has 1676;
> refusing to compare recall across two different populations

**That was the guard working, not a regression** -- the same guard, doing the same job, that
refused a 349-vs-649 mismatch and a 1,150-vs-1,452 mismatch before it. It is resolved the same
way both of those were: by collecting Azure over the population that now exists, not by
relaxing the guard. `azure-2026-08-28.json` (below) is that collection, estimated at $3.2470
(against a $3.2540 expectation set before the run and a $7.00 human-set stop threshold) and
confirmed at exactly that estimate.

**Azure detects none of the 211 `CREDENTIAL` values, measured rather than assumed.** Azure AI
Language has no credential entity type, so `mask_found` for `CREDENTIAL` is 0/211 (recall
0.0) through the scorer's normal path -- every mapped category, at or above `min_score`. Going
further than the scorer does, and checking Azure's raw, unfiltered response against every
`CREDENTIAL` span regardless of category or confidence: 207 of 211 have **no** Azure entity of
any kind touching them at all, and the other 4 have a 3-to-15-character incidental overlap
(confidence 0.50-0.64) tagged `PersonType` -- Azure's occupation/role-noun category, not a
category Temperate's earlier Azure client maps to anything, and not a span that would ever reach
Temperate's masking. Zero of 211 are covered under any category by any margin. This is a finding,
not a gap: Azure was never going to have a detector for a format it has no entity type for, and
the honest report is the measured 0/211 and what "detected" would even mean here, not a
guessed zero.

`PERSON` and `STREET_ADDRESS` -- the two types the gates check on non-inferiority -- barely moved,
because no `CREDENTIAL` case's carrier document is new prose: `PERSON` recall ticked up
(189/190 = 0.9947 -> 194/195 = 0.9949, the five new occurrences are existing carriers' own
screening replacements, found by both sides) and `STREET_ADDRESS` did not move at all
(274/275 = 0.9964 exactly, before and after -- no `CREDENTIAL` case injects a new address).
Full per-type, per-language and gate-verdict detail, before and after, is
`training/reports/pii-detection-vs-azure-2026-08-23.md`'s newest revision.

## RESOLVED 2026-08-27: the fixture grew, and this file now matches it

`benchmark/cases/` grew to **1,452** cases on 2026-08-27 (four country packs -- ES, IT, FR, BE,
see `training/reports/country-packs-phase-1.md`), which for a few hours left the baseline below
stale at 1,150 and `load_baseline`'s size guard refusing every comparison against the full
fixture:

> the --baseline file was collected over 1150 case(s) but the fixture being scored has 1452;
> refusing to compare recall across two different populations

**That was the guard working, not a regression** -- exactly the same guard, doing the same job,
that refused a 349-vs-649 mismatch once before (`training/reports/pii-detection-vs-azure-2026-08-23.md`,
"Why this exists"). It is resolved the same way that one was: by collecting Azure over the
population that now exists, not by relaxing the guard. `azure-2026-08-27.json` (below) is that
collection.

## The current baseline

| | |
|---|---|
| file | `azure-2026-09-12.json` |
| collected_at | 2026-09-12 |
| azure_model_version | `2026-05-01` |
| api_version | `2024-11-01` |
| min_score | 0.5 (mirrors `PII_SCORE_THRESHOLD`'s default in Temperate) |
| n_cases | **2,041** -- the full fixture, matched exactly to what the detector is scored on |
| total_characters | 3,018,945 |
| estimated / actual cost | $3.9470 / $3.9470, one `--confirm` run, 409 batches, no retry |
| overall recall / precision | **0.8533 / 0.4970** (rescored in place 2026-09-12; published as 0.5632 / 0.9017 before the category-map correction -- see the top of this file) |
| over-masking | 236 / 4,307 (published as 76 / 4,307) |
| categories scored / refused on the record | 39 mapped, 5 `deliberately_unmapped_categories` |

Collected the day the fixture reached 2,041 (the `DRIVING_LICENCE` ladder finished landing;
`fa98768` alone had already taken it to 1,904 without a re-collection). Re-collecting rather than
leaving the 1,676-case file in place was a deliberate $3.95, estimated before spending and
confirmed against the $8.00 human-set ceiling for this task before `--confirm` was ever passed:
`load_baseline`'s size guard would otherwise have kept refusing the comparison outright and both
non-inferiority gates would have kept reading exit code 2 (not `NotMeasured` -- no report is even
written on a case-count mismatch, see `eval.rs`'s own module docs on the distinction), which is a
worse outcome than paying for the measurement. `main()`'s output-directory-before-first-request
ordering (documented below the cost section) was re-confirmed by reading the code before this run,
not merely trusted, and held: no double billing.

**[Ninth-revision note] The paragraph below describes this file as it was scored on collection and
is superseded by the rescore at the top of this README.** Its "Azure finds **3**" is 305 through
the corrected map, and its 0.6423 -> 0.5632 fall is an artifact of the map, not of the population.
It is left in place as the record of what was published.

**Azure's own figures moved again, in the same direction and for the same structural reason as
the 1,452->1,676 expansion, this time driven by a type Azure has *categories* for but that
Temperate's own driver never asks it about.** Overall recall fell 0.6423 -> 0.5632 and precision fell
0.9382 -> 0.9017 -- both driven by population change, not by Azure changing (`azure_model_version`
is `2026-05-01` throughout). The 365 new `DRIVING_LICENCE` cases add 344 `must_mask` values;
The detector finds 340 of them (`training/reports/driving-licence.md`), Azure finds **3** under any
mapped category (all incidental `PhoneNumber` overlap) but, checked against its raw unfiltered
response, eight dedicated per-country `*DriversLicenseNumber` categories fully cover 305 of the
344 -- see "RESOLVED 2026-09-12" above for the full measurement and the per-country breakdown.
`PERSON` moved by denominator only and `STREET_ADDRESS` did not move at all, because the only new
carrier prose either type could touch is the ten reused `PERSON` screening replacements the new
cases inherit. Full per-type, per-language and gate-verdict detail, before and after, is
`training/reports/pii-detection-vs-azure-2026-08-23.md`'s newest revision.

### Superseded: `azure-2026-08-28.json`, the 1,676-case baseline

Kept, unmodified, for provenance. It covered 1,676 of what was then a 1,676-case fixture and was
a real, paid-for run; it is not the current baseline and must not be quoted as one. Its own
figures -- recall 0.6423, precision 0.9382, over-masking 43/3,297 -- describe a strict subset of
the present fixture (every case file except `driving-licence-*.json`).

Collected the day after the fixture went 1,452 -> 1,676 (`CREDENTIAL` -- API keys, JWTs and PEM
blocks, `training/reports/credentials.md`). Azure's own figures moved from the 1,452-case
baseline for the same structural reason described above for this file's own successor: overall
recall fell 0.7046 -> 0.6423 and precision fell 0.9415 -> 0.9382, both because the population
changed and not because Azure did. Azure detected zero of the 211 `CREDENTIAL` values under any
mapped category, and checked against its raw unfiltered response, touched only 4 of 211 at all --
a few characters each, tagged `PersonType`, nowhere near a full span.

### Superseded: `azure-2026-08-27.json`, the 1,452-case baseline

Kept, unmodified, for provenance. It covered 1,452 of what was then a 1,452-case fixture and was
a real, paid-for run; it is not the current baseline and must not be quoted as one. Its own
figures -- recall 0.7046, precision 0.9415, over-masking 41/2,682 -- describe a strict subset of
the present fixture (every case file except `credential-*.json`).

Collected the same day the fixture went 1,150 -> 1,452 (four country packs -- ES, IT, FR, BE,
`training/reports/country-packs-phase-1.md`). Azure's own figures moved from the 1,150-case
baseline for the same structural reason described above for this file's own successor: overall
recall fell 0.7869 -> 0.7046 and precision rose 0.9497 -> 0.9415, both because the population
changed and not because Azure did. Of the four new identifiers, Azure detected **zero** of the
Spanish DNI (0/40), Spanish NIE (0/40), French NIR (0/75) and Italian codice fiscale (0/76, no
`TAX_ID` category at all) -- but **49 of 58 (0.8448) Belgian national register numbers**,
incidentally, through the same `NLCitizensServiceNumber` category Temperate's earlier Azure client maps
only to the Dutch BSN.

### Superseded: `azure-2026-08-26.json`, the 1,150-case baseline

Kept, unmodified, for provenance. It covered 1,150 of what was then a 1,150-case fixture and was
a real, paid-for run; it is not the current baseline and must not be quoted as one. Its own
figures -- recall 0.7869, precision 0.9497, `n_cases` 1,150 -- describe a strict subset of the
present fixture (the four original case files: `domain.json`, `carriers.json`,
`negatives.json`, `seed.json`).

Collected the same day the carrier corpus went 154 -> 321 and the fixture 649 -> 1,150
(`training/reports/benchmark-expansion.md`). **Azure's own figures moved from the 649-case
baseline for the same reason described above for this file's own successor:** overall recall
fell 0.7940 -> 0.7869 and precision rose 0.9235 -> 0.9497, both because the population changed
and not because Azure did. `PERSON` recall fell 1.0000 (111/111) to 0.9945 (181/182), which moved
the non-inferiority bar the detector is measured against.

### Superseded: `azure-2026-08-23.json`, the 649-case baseline

Kept, unmodified, for provenance. It covered 649 of what was then a 649-case fixture and was a
real, paid-for run; it is not the current baseline and must not be quoted as one. Its own
figures -- recall 0.7940, precision 0.9235, over-masking 30/1,106 -- describe a strict subset
of the present fixture.

**`summary` and `disagreements` were rescored in place on 2026-08-24**, from this file's own
unchanged `raw_batches`, after commit `eaac5fa` moved the address definition (the city that
terminates a postal address became part of the `STREET_ADDRESS` value, not a separate
`must_clear` mention) -- see `training/reports/city-address-tails.md` and this file's own
`STREET_ADDRESS` entry below. No Azure call was made; `command`, `cost` and `raw_batches` are
untouched and still describe the original `--confirm` run and its $1.13. The two figures above
were `0.7950 / 0.8598` before that rescore.

Per-type and per-language recall and precision, disagreements, and Azure's untouched raw
responses are all in the file; see "The file's shape" below for where. The full like-for-like
comparison against the detector's shipped static-QDQ artifact -- per type, per language, and
over-masking on both sides -- is `training/reports/pii-detection-vs-azure-2026-08-23.md`, not
duplicated here.

## Superseded: `azure-2026-08-22.json`, the 349-case baseline

Kept, unmodified, for provenance -- it was a real, paid-for run and its own history (the
directory-creation bug that doubled its cost, the pre-correction `disagreements` values) is
worth keeping legible. **Do not quote it as the current baseline.** It covered 349 of what was,
at the time, a 349-case fixture; the fixture grew to 649 cases the same day
(`benchmark/carriers/README.md`), and this file was never re-collected at that size while the
mismatch stood. `eval.rs`'s `load_baseline` guard refused to compare it against the 649-case
fixture for exactly as long as that mismatch existed -- both non-inferiority gates read
`NotMeasured` under it, not because no model was deployed (`pii-detection-model` shipped before this
gap was closed) but because the guard does not compare two different populations no matter what
is asking it to. `azure-2026-08-23.json` closed that gap and `azure-2026-08-26.json` has since replaced it;
this file no longer needs to be read as "the baseline," only as the earliest one.

## The comparison to make, and the one not to make

**This section used to warn against comparing Azure's rules-only-era recall to the detector's
headline, because the detector's deployed service had no `PERSON`/`STREET_ADDRESS` recognizer and the
denominators didn't match (507 declared values vs. 413 both sides were asked about).** That
warning no longer applies to the current baseline: the shipped service is model-backed,
advertises every type this 2,041-case fixture declares that Temperate itself has a slot for
(`CREDENTIAL` included; `DRIVING_LICENCE` is the one type the fixture now declares that Temperate's
own taxonomy has no slot for at all, see "Where this pulls its numbers from" below), and
`azure-2026-09-12.json`'s `n_cases` matches it exactly, so `summary` and `summary_as_declared`
agree (2,720 == 2,720) and no `drop_unrequested` correction is needed to make the headline honest
any more.

**Do not, however, read the detector's higher overall recall (0.9945 vs Azure's 0.8533) as "the detector has
won."** Azure leads both types this design actually gates on non-inferiority (`PERSON` 0.9951 vs
0.9805, `STREET_ADDRESS` 0.9964 vs 0.9891) and ties or leads on `IP_ADDRESS`, `TAX_ID`,
`EMAIL_ADDRESS` and `IBAN_CODE`. **This paragraph used to say Azure led precision and
over-masking as well, and that is no longer true -- but for a reason that is a correction to this
repository, not a change in Azure**: the same category-map defect that understated Azure's recall
also spared it the false positives of 27 categories it was never scored on. Corrected, the detector
leads precision 0.8461 vs 0.4970 and over-masking 94/4,307 vs 236/4,307, and one Azure category
(`USDriversLicenseNumber`, degenerate on European text) accounts for most of that gap -- without
it Azure reads 0.8643 and 145/4,307. Read the ninth revision's "The one judgement call" before
quoting either.

The detector's remaining recall lead is concentrated in `CREDENTIAL` (Azure has no such category at
all -- measured, not assumed: 0/211 even against every category in Azure's raw response, mapped
and unmapped), `VAT_NUMBER` (0.4820), `PASSPORT_NUMBER` (0.6250) and the 39 `DRIVING_LICENCE`
values Azure's licence categories miss, 37 of them Italian. The full per-type, per-language and
over-masking breakdown for the current fixture is
`training/reports/pii-detection-vs-azure-2026-08-23.md`'s "Ninth revision" section; the 1,676-case one
is that same report's "Seventh revision" section; the 1,452-case one is its "Sixth revision"
section; the 1,150-case one is `training/reports/benchmark-expansion.md`; the 649-case one is
that same report's earlier revisions. Read one of them before quoting either overall number
alone.

## Where this pulls its numbers from, and where it doesn't

**Rewritten 2026-09-12 (ninth revision). What this section used to say is recorded at the top of
this file and was wrong in a way that cost Azure 29 points of recall across every published
comparison.** It described `category_map` as a transcription of
the category list of our earlier Azure client -- 12 categories onto
9 types -- and listed `VAT_NUMBER` and `TAX_ID` as types Azure has no equivalent for. Both of
those claims were false against raw responses this directory already held.

`category_map` translates **Azure's category names into the entity types the fixture declares**.
Temperate's `PiiEntityType` is a third vocabulary belonging to one consumer, and it is not the
authority here: the question for each category is "does the fixture ask about this class of
identifier?", never "does Temperate have a type for it?". 39 Azure categories now map onto 12 of the
fixture's 13 types.

One the detector type declared in this fixture has **no** Azure equivalent at all:

- **`CREDENTIAL`** -- Azure AI Language has no API-key, token or private-key category. Measured,
  not assumed, and re-verified at the ninth revision against **every** category in every committed
  raw response, mapped and unmapped alike: 0 of 211 values covered.

It shows up in `per_type` as `recall: 0.0`, `precision: null` -- not a defect in the measurement,
the honest report of a capability Azure does not have. **A non-inferiority gate must not be
pointed at it**, because "non-inferior to zero" is not a bar the detector failing to clear would mean
anything. (`gates.rs`'s ruling 2 still excludes `VAT_NUMBER` and `TAX_ID` from Azure-relative
gating on the same grounds, and that ruling should be revisited now that Azure's real figures for
both are known: `TAX_ID` 1.0000, `VAT_NUMBER` 0.4820. Not changed here -- a gate ruling is not
something to move as a side effect of a scoring fix.)

Five categories Azure returns are **deliberately not scored**, listed with a reason in
`deliberately_unmapped` in `baseline.rs` and written into every baseline file as
`deliberately_unmapped_categories`: `PersonType` (a role or job title, distinct from the `Person`
category, which is mapped), `Organization`, `DateTime`, `Quantity` and `URL`. The fixture declares
no type for any of them and lists organisation names, dates and quantities under `must_clear`
precisely because they are not personal data in this taxonomy. Scoring them would charge Azure for
answering a question nobody asked: measured, it takes Azure's over-masking from 236 to 720 of
4,307 and its precision from 0.4970 to 0.2294, for 2 additional values of recall.

**Nothing is dropped silently any more.** A category that is in neither list fails
`every_category_azure_returns_is_mapped_or_explicitly_unmapped` in `baseline.rs`, on the
collection that first returns it.

`PERSON` -- the type the gates actually check on non-inferiority -- has a real Azure equivalent
and 205 declared occurrences in this 2,041-case fixture: recall 0.9951 (204/205), precision
0.8571. It read a clean 1.0000 (111/111) over the 649-case fixture, 0.9945 (181/182) over the
1,150-case one, 0.9947 (189/190) over the 1,452-case one and 0.9949 (194/195) over the
1,676-case one; each expansion's carrier cases inherit their carriers' own `PERSON` screening
replacements, found by Azure too, so the figure barely moves.

## A result worth flagging before anyone gates on it: PHONE_NUMBER precision

`PHONE_NUMBER` recall is 0.9732, in line with the rest of the fixture, but precision has fallen
to **0.5777** at 2,041 cases, from 0.7592 at 1,676 -- the largest single-revision drop this
finding has recorded. Azure's phone-number category fires on more spans than the fixture names as
phone numbers: 53 `must_clear` violations plus 53 spurious spans against 145 true positives, up
from 23 and 23 respectively, which had themselves been unchanged since the 1,452-case baseline
(no `CREDENTIAL` case touches a phone number). **This time the cause is named, not just
described: all 30 new violations are inside `driving-licence-*.json` cases specifically** (checked
by case id against Azure's raw response), whose own generated hard negatives -- order numbers,
control-character-bearing digit runs, near-miss-length licence numbers -- are exactly the shape
this paragraph already predicted would generalise. It was 0.6759 over the 649-case fixture on the
same error-count shape; the rate fell as the denominator grew through the 1,150-, 1,452- and
1,676-case populations (each expansion's carrier prose adds more phone-number-shaped negatives,
not because Azure's phone recognizer changed -- `azure_model_version` is `2026-05-01` throughout)
and had held flat for one revision (`CREDENTIAL` added none), before this one's `DRIVING_LICENCE`
negatives moved it further than any prior expansion. Some of this is very likely the same
`PhoneNumber`-mislabels-digit-strings pattern Temperate's earlier Azure client documents for the German
Steuer-ID, generalised to other long digit runs (national ID numbers, passport numbers, and now
order numbers and licence numbers) that this fixture's `must_clear` lists were never written to
anticipate. This is exactly the kind of Azure-vs-fixture disagreement `disagreements` exists to
surface (see below) -- read it before trusting `PHONE_NUMBER` precision at face value in either
direction.

## Sample size, honestly

2,041 cases is not a large fixture, and per-language and per-type denominators inside it are
smaller still -- `de` carries 393 declared `must_mask` values (still the smallest language; it
grew from 340 when `DRIVING_LICENCE` landed, and every other language grew more, 449-483),
`PASSPORT_NUMBER` 136, `CREDIT_CARD` 139 and `VAT_NUMBER` 139 (`IP_ADDRESS` at 140 the next
smallest -- all four unchanged in count and still the smallest structured types with an Azure
comparison at all, because no `DRIVING_LICENCE` case touches any of them; `TAX_ID` 197 and
`NATIONAL_ID` 334 remain well clear of them). `CREDENTIAL` at 211 has no Azure comparison at all
-- Azure's genuine 0/211 on it is exactly why not (see "RESOLVED 2026-08-28" above and the
ninth-revision re-verification at the top of this file). **`DRIVING_LICENCE` at 344 used to be
listed here as a second such type, on a structural 0.0087 that was an artifact of the category
map rather than of Azure; it is a real comparison at 305/344 (0.8866) and belongs in the same
class as `NATIONAL_ID` and `TAX_ID`.** A single case moving
from found to missed moves a small bucket's recall by most of a percentage point. Say so wherever
a per-type or per-language figure from this file is quoted, the same discipline
`benchmark/carriers/README.md` asks for its own sample size. **`pii_detection_eval::gates::
SAMPLE_SIZE_CAVEAT` (a hardcoded string constant in `gates.rs`) has been re-derived at every
expansion since the fifth revision found it stale twice in a row; it is re-derived again as of
this revision**: the constant and its doc comment now read 393-483 per-language (was 340-416) and
136-140 (`PASSPORT_NUMBER`, `CREDIT_CARD`, `VAT_NUMBER`, `IP_ADDRESS`, unchanged) per-type,
matching the fixture this file was collected over. The carrier-count figure ("40-66
carriers/language") did **not** need updating: every `DRIVING_LICENCE` case reuses an existing
carrier id, confirmed by diffing distinct `carrier_id` sets with and without
`driving-licence-*.json` included -- identical per language, the same check the seventh revision
ran for `CREDENTIAL`. There is still no automated check tying the constant to the fixture's actual
counts, so it can go stale again the next time `benchmark/cases` grows -- noted in the constant's
own doc comment, not fixed structurally.

## The file's shape

```
{
  "command": "...",                 // the exact invocation that produced this file
  "collected_at": "2026-08-22",     // the run date -- quote this
  "fixture_dir": "benchmark/cases",
  "n_cases": 349,
  "api_version": "2024-11-01",
  "azure_model_version": "2026-05-01",  // Azure's own results.modelVersion -- quote this
  "min_score": 0.5,
  "categories_map": { "Person": "PERSON", ... },   // Azure category -> the fixture's EntityType
  "deliberately_unmapped_categories": [ { "category": "Organization", "reason": "..." }, ... ],
  "no_azure_equivalent": [ { "type": "CREDENTIAL", "reason": "..." } ],
  "cost": { "total_characters": ..., "estimated_text_records": ..., "estimated_cost_usd": ...,
            "actual_cost_usd": ..., "price_usd_per_1k_text_records": 1.0, "note": "..." },
  "summary": {
    "overall": { "mask_total": ..., "mask_found": ..., "clear_total": ..., "clear_violated": ...,
                 "spurious_spans": ..., "recall": ..., "precision": ... },
    "per_language": { "nl": { ...same shape... }, "de": { ... }, ... },
    "per_type": { "PERSON": { ...same shape... }, "PHONE_NUMBER": { ... }, ... },
    "untyped_mask_values": 152,     // see pii_detection_eval::score's module docs -- counted overall
                                     // and per-language, not per-type
    "macro_by_language": { "recall": ..., "precision": ... }  // every language weighted equally
  },
  "disagreements": [
    { "case_id": "...", "lang": "...", "kind": "azure_missed" | "azure_over_masked",
      "value": "...", "declared_type": "PERSON" | null, "detected_types": [...] },
    ...
  ],
  "raw_batches": [ { "batch_index": 0, "document_ids": [...], "raw_response": { ... } }, ... ]
}
```

`recall`/`precision` are `null`, never `0.0`, when there was nothing to measure (an empty
denominator) -- see `pii_detection_eval::score`'s module docs on why that distinction is load-bearing
for a gate.

## `disagreements`: the fixture is ground truth, Azure is a comparator

Every case where Azure's spans disagree with the fixture's answer key -- a `must_mask` value
Azure left uncovered (`azure_missed`), or a `must_clear` value Azure touched
(`azure_over_masked`) -- is recorded in `disagreements`, one entry per value. **The fixture
wins by default**; this list is not a claim that the fixture is right and Azure is wrong, only
a record of where the two disagree, for a human to adjudicate. It is, per the benchmark plan,
the most valuable material this whole exercise produces for finding fixture errors: a
disagreement can mean Azure genuinely missed something, or it can mean a `must_mask`/
`must_clear` entry is itself wrong. Read it that way before "fixing" either side.

### Five `disagreements` values in the superseded 349-case file quote the fixture as it was on 2026-08-22

This subsection is about `azure-2026-08-22.json` (see "Superseded" above), not the current
baseline: `azure-2026-08-23.json` was collected after every correction below, over the fixture
as it stands today, so it carries no equivalent caveat.

The final review found four committed identifiers that a detector applying the real national
rule would correctly reject -- a Steuer-ID violating the BZSt structure rule, two VAT numbers
failing their national check digit, and the legacy Steuer-ID pair in `seed.json` -- and they
were replaced. Azure was run before that, over the text as it then stood, and **this file has
not been re-collected or hand-edited**: it is the artifact of the run that was paid for. Five
`disagreements` entries (`domain-de-contract-1`, `taxid-de-1`, `taxid-de-negative`, `vat-de-1`,
`vat-nl-1`) and the matching `raw_batches` documents therefore quote the pre-correction values.

Nothing in `summary` moves. Every replacement is an in-place, same-length digit substitution,
and every one of the five values was a type Azure has no equivalent for (or, for
`taxid-de-negative`, a value Azure mislabels as a phone number either way). Re-running
`--from-raw` against the corrected fixture reproduces `summary` identically **up to
floating-point associativity** -- `macro_by_language`'s recall and precision are summed over a
`HashMap`, whose iteration order Rust randomises per process, so those two fields alone can
differ in the last ULP between runs regardless of fixture content; every other count, recall,
and precision matches exactly, and only the five `value` strings change. Rescoring the
corrected and pre-correction fixtures side by side confirms this: the summaries are identical
in substance, so the five substitutions are provably summary-neutral. That was checked before
writing this paragraph. Re-collecting the baseline to refresh five strings would spend $0.51
and replace a real run's provenance with a rescore's.

`value` is a literal substring of the case's own `text`, and `raw_batches` carries Azure's
complete, unmodified responses, including the `text` field on every detected entity. Both are
safe to commit: every case's `text` is fixture content already committed to this repository
(synthetic or licensed carrier prose, see `benchmark/carriers/README.md`), so nothing new or
sensitive is introduced by keeping Azure's own copy of a substring of it. This is the same
choice `training/eval_fixture_azure.py` made for the earlier 128-case fixture. It does **not**
license committing raw production traffic anywhere -- see the global constraint on case text in
logs, which is about a running service's own telemetry, not about evaluation artifacts built
from fixture content that was already public in this repository.

## What this run actually cost

Azure AI Language bills PII detection per 1,000-character "text record"
(`ceil(chars/1000)` per document, minimum 1 per document), on the same `Standard Text Records`
meter as sentiment/NER/key-phrase/language-detection, at $1.00 per 1,000 text records for the
first volume tier (confirmed against the public Retail Prices API and the Language service's
own pricing page on 2026-08-22) -- and free for the first 5,000 text records/month shared
across those features.

**The current run (`azure-2026-09-12.json`, 2,041 cases / 3,018,945 characters): $3.9470
estimated, $3.9470 actual, in a single `--confirm` invocation, 409 batches, no retry, no data
loss.** The previous run (`azure-2026-08-28.json`, 1,676 cases / 2,472,146 characters) cost
$3.2470 against a $3.2470 estimate, 336 batches; before that (`azure-2026-08-27.json`, 1,452
cases / 2,087,003 characters) $2.7540 against a $2.7540 estimate, 291 batches; before that
(`azure-2026-08-26.json`, 1,150 cases / 1,634,673 characters) $2.1720 against a $2.1720 estimate,
230 batches; the one before that (`azure-2026-08-23.json`, 649 cases / 802,900 characters) cost
$1.1300 against a $1.1300 estimate, 130 batches.
`main()` in `baseline.rs` was re-read before every one of these runs, this one included, to
confirm the fix below still holds -- the output directory is created before the first network
call, not after the last response -- and it did every time: this is the outcome that fix was
for, observed, not merely hoped for. Detail in
`training/reports/pii-detection-vs-azure-2026-08-23.md`'s "Eighth revision" section.

**The superseded run (`azure-2026-08-22.json`, 349 cases / 285,529 characters) cost $1.0240, not
its own $0.5120 estimate**, because the first attempt at that run completed all 349 documents
successfully and then failed to write its own output: the target directory,
`benchmark/baseline/`, did not exist yet, and `std::fs::write` does not create parent
directories. That data was lost, and a second, otherwise identical run was needed to produce
`azure-2026-08-22.json`. `baseline.rs` now creates the output directory before sending a single
request rather than after collecting every response, so this specific mistake cannot repeat --
and the current run above is the confirmation that it didn't. Recorded here rather than quietly
folded into "the estimate was close enough," because the brief that governs this task asks for
exactly this kind of honesty about a cost model: the *per-run* estimate matched the *per-run*
actual exactly ($0.5120 both times, per that file's own `cost.actual_cost_usd`), and the gap was
entirely operational, not a pricing error.

## Reproducing or rescoring

```sh
# Estimate only -- never touches the network or Temperate's credentials.
cargo run -p pii-detection-eval --bin baseline -- --date <YYYY-MM-DD>

# The real run (spends money; see the cost gate above). --env-path names the .env
# file that holds the Azure credentials, read-only, and is required here: there is deliberately no
# default, because one machine's absolute path is wrong on every other machine.
cargo run -p pii-detection-eval --bin baseline -- --date <YYYY-MM-DD> --confirm \
    --env-path <path-to-temperate-checkout>/.env

# Rescore a previously saved run with today's scoring rules, without calling Azure again --
# any committed file works as --from-raw's input, matched against the fixture size it itself
# was collected over (the current one, 2,041, or a superseded one, 1,676, 1,452, 1,150, 649 or
# 349).
cargo run -p pii-detection-eval --bin baseline -- --date <YYYY-MM-DD> \
    --from-raw benchmark/baseline/azure-2026-09-12.json \
    --out benchmark/baseline/azure-2026-09-12-rescored.json
```

`--from-raw` is how `pii_detection_eval::score` changing (a corrected rule, a new type) gets applied
to Azure's already-paid-for responses without spending anything a second time -- the same
reason `training/eval_fixture_azure.py` grew the equivalent flag for the 128-case fixture.
