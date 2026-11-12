# Corpora Profile: ai4privacy vs Wismut

Generated 2026-08-20 16:53 UTC by `training/profile_corpora.py`. Both corpora were streamed
in full (all splits) -- never materialised to disk, never loaded whole into
memory. Command: `python training/profile_corpora.py --out services/pii-detection/training/reports/corpora-profile.md`


## IMPORTANT: an assumption in the task brief did not hold

The brief expected both corpora to carry a `language` field. **ai4privacy
does** (ground truth, verified below). **Wismut/nym-pii-multilingual-data
does not** -- its parquet schema has exactly two columns, `text` and
`entities`. Its dataset card claims ~22-23 languages (via Faker locale
generation during synthesis), but that information is not persisted per row
anywhere in the released data. This was discovered while building this
profiler, not assumed going in; see `/first-rows` output referenced in the
task report.

Because there is no ground truth to measure, Wismut's language distribution
below is an **estimate** produced by streaming every row's `text` through
`py3langid` (a deterministic, offline language identifier). It is not a
dataset field and is presented as an estimate throughout. To bound how much
to trust it, `py3langid` was also run against ai4privacy's own ground-truth
`language` column, sampled every twentieth row by absolute position in the
stream (not every twentieth row per language) -- see the calibration table
below for what that means for the resulting numbers. Treat the Wismut
language numbers accordingly.

Also notable: **neither corpus's dataset card claims Irish (`ga`) or Maltese
(`mt`) at all.** This is not a "thin" language, which is what the brief
anticipated might happen to the eight ai4privacy-unique languages -- it is
total absence from both corpora's stated language coverage, confirmed
against measured/estimated row counts below.

## Headline finding: the eight ai4privacy-unique EU languages are not thin

This is the risk this task was built to check. The task brief's specific
concern was that Bulgarian, Estonian, Croatian, Hungarian, Lithuanian,
Latvian, Slovak, and Slovenian -- the eight EU languages that come from
ai4privacy alone, per the dataset cards -- might be thin. Measured (ground
truth `language` column, not estimated):

| language | rows |
|---|---:|
| bg (Bulgarian) | 46,801 |
| lt (Lithuanian) | 45,823 |
| et (Estonian) | 45,777 |
| sk (Slovak) | 45,739 |
| lv (Latvian) | 45,636 |
| hu (Hungarian) | 45,345 |
| sl (Slovenian) | 44,993 |
| hr (Croatian) | 44,935 |

All eight sit in a tight 44,900-46,800 row band -- essentially balanced with
every other EU language ai4privacy carries, and 9-10x above the 5,000-row
sufficiency threshold. This risk came back clean.

---

## ai4privacy/pii-masking-openpii-1.5m (CC-BY-4.0)

- Total rows streamed (train + validation): **1,636,375**
- Data moved: 4,489.6 MB over 1467s
- Rows with zero entities: 2,300 (0.14%)
- Character length: mean 334.5, median 263.0, p95 831.0
- Distinct labels: 43
- Distinct languages: 30

### Rows per language (measured, ground truth `language` field)

| language | rows | share |
|---|---:|---:|
| en | 204,649 | 12.51% |
| fr | 131,173 | 8.02% |
| de | 120,105 | 7.34% |
| es | 90,254 | 5.52% |
| it | 89,304 | 5.46% |
| bg | 46,801 | 2.86% |
| pl | 46,621 | 2.85% |
| cs | 46,473 | 2.84% |
| lt | 45,823 | 2.80% |
| et | 45,777 | 2.80% |
| sv | 45,774 | 2.80% |
| sk | 45,739 | 2.80% |
| lv | 45,636 | 2.79% |
| hu | 45,345 | 2.77% |
| ro | 45,322 | 2.77% |
| fi | 45,313 | 2.77% |
| el | 45,178 | 2.76% |
| da | 45,062 | 2.75% |
| sl | 44,993 | 2.75% |
| hr | 44,935 | 2.75% |
| sr | 44,817 | 2.74% |
| nl | 44,460 | 2.72% |
| pt | 43,819 | 2.68% |
| ko | 26,498 | 1.62% |
| vi | 26,442 | 1.62% |
| zh | 26,272 | 1.61% |
| id | 26,167 | 1.60% |
| ms | 26,144 | 1.60% |
| ja | 25,902 | 1.58% |
| tl | 25,577 | 1.56% |

### Rows per label (from `privacy_mask`)

| label | rows | share of rows |
|---|---:|---:|
| DATE | 1,398,279 | 85.45% |
| GIVENNAME | 1,396,075 | 85.32% |
| SURNAME | 1,200,212 | 73.35% |
| EMAIL | 881,036 | 53.84% |
| CITY | 824,793 | 50.40% |
| TITLE | 770,782 | 47.10% |
| TELEPHONENUM | 660,834 | 40.38% |
| AGE | 588,562 | 35.97% |
| STREET | 559,329 | 34.18% |
| BUILDINGNUM | 549,425 | 33.58% |
| ZIPCODE | 506,109 | 30.93% |
| IDCARDNUM | 387,266 | 23.67% |
| CREDITCARDNUMBER | 374,409 | 22.88% |
| DRIVERLICENSENUM | 328,121 | 20.05% |
| GENDER | 319,615 | 19.53% |
| TAXNUM | 308,680 | 18.86% |
| SEX | 292,600 | 17.88% |
| SOCIALNUM | 281,424 | 17.20% |
| PASSPORTNUM | 239,123 | 14.61% |
| TIME | 19,145 | 1.17% |
| ORGANISATION | 329 | 0.02% |
| AMOUNT | 309 | 0.02% |
| URL | 261 | 0.02% |
| COUNTRY | 213 | 0.01% |
| CURRENCY | 100 | 0.01% |
| USERNAME | 100 | 0.01% |
| BANKNAME | 78 | 0.00% |
| ACCOUNTNUM | 46 | 0.00% |
| JOBTITLE | 23 | 0.00% |
| IPV4 | 19 | 0.00% |
| PASSWORD | 18 | 0.00% |
| WEIGHT | 12 | 0.00% |
| TIMEZONE | 10 | 0.00% |
| SALARY | 10 | 0.00% |
| ALLERGIES | 10 | 0.00% |
| MEDICATION | 4 | 0.00% |
| POLICYNUM | 2 | 0.00% |
| HEIGHT | 2 | 0.00% |
| DOCTORNAME | 1 | 0.00% |
| AIRPORTCODE | 1 | 0.00% |
| MACADDRESS | 1 | 0.00% |
| HOSPITALNAME | 1 | 0.00% |
| JOBTYPE | 1 | 0.00% |

### Full label vocabulary, sorted (43 labels)

ACCOUNTNUM, AGE, AIRPORTCODE, ALLERGIES, AMOUNT, BANKNAME, BUILDINGNUM, CITY, COUNTRY, CREDITCARDNUMBER, CURRENCY, DATE, DOCTORNAME, DRIVERLICENSENUM, EMAIL, GENDER, GIVENNAME, HEIGHT, HOSPITALNAME, IDCARDNUM, IPV4, JOBTITLE, JOBTYPE, MACADDRESS, MEDICATION, ORGANISATION, PASSPORTNUM, PASSWORD, POLICYNUM, SALARY, SEX, SOCIALNUM, STREET, SURNAME, TAXNUM, TELEPHONENUM, TIME, TIMEZONE, TITLE, URL, USERNAME, WEIGHT, ZIPCODE

---

## Wismut/nym-pii-multilingual-data (MIT)

- Total rows streamed (train + validation + test): **805,000**
- Data moved: 339.1 MB over 127s
- Rows with zero entities: 105,536 (13.11%)
- Character length: mean 247.3, median 211.0, p95 554.0
- Distinct labels: 40
- Distinct languages detected by py3langid: 68

### Rows per language (ESTIMATED via py3langid on `text` -- no dataset field exists)

| language | rows | share |
|---|---:|---:|
| en | 176,755 | 21.96% |
| ar | 40,323 | 5.01% |
| zh | 38,691 | 4.81% |
| es | 38,532 | 4.79% |
| it | 36,817 | 4.57% |
| ko | 33,512 | 4.16% |
| ja | 33,253 | 4.13% |
| fi | 32,808 | 4.08% |
| da | 32,221 | 4.00% |
| tr | 31,019 | 3.85% |
| de | 30,899 | 3.84% |
| fr | 30,427 | 3.78% |
| ro | 29,549 | 3.67% |
| nl | 29,317 | 3.64% |
| cs | 28,781 | 3.58% |
| ru | 26,171 | 3.25% |
| uk | 24,887 | 3.09% |
| pt | 23,279 | 2.89% |
| el | 22,096 | 2.74% |
| sv | 21,160 | 2.63% |
| pl | 19,595 | 2.43% |
| hi | 17,205 | 2.14% |
| mr | 2,397 | 0.30% |
| no | 1,648 | 0.20% |
| gl | 656 | 0.08% |
| nb | 588 | 0.07% |
| ne | 430 | 0.05% |
| nn | 277 | 0.03% |
| af | 256 | 0.03% |
| sk | 235 | 0.03% |
| sl | 181 | 0.02% |
| ur | 166 | 0.02% |
| fa | 159 | 0.02% |
| rw | 102 | 0.01% |
| se | 81 | 0.01% |
| bg | 77 | 0.01% |
| lb | 47 | 0.01% |
| br | 36 | 0.00% |
| et | 30 | 0.00% |
| eu | 27 | 0.00% |
| id | 25 | 0.00% |
| oc | 25 | 0.00% |
| ca | 24 | 0.00% |
| mt | 24 | 0.00% |
| mg | 20 | 0.00% |
| ms | 19 | 0.00% |
| eo | 19 | 0.00% |
| cy | 19 | 0.00% |
| tl | 18 | 0.00% |
| az | 15 | 0.00% |
| is | 14 | 0.00% |
| lt | 13 | 0.00% |
| la | 10 | 0.00% |
| an | 10 | 0.00% |
| sw | 9 | 0.00% |
| hu | 9 | 0.00% |
| jv | 7 | 0.00% |
| hr | 5 | 0.00% |
| xh | 4 | 0.00% |
| ht | 4 | 0.00% |
| sq | 3 | 0.00% |
| zu | 3 | 0.00% |
| qu | 2 | 0.00% |
| vo | 2 | 0.00% |
| ga | 2 | 0.00% |
| wa | 2 | 0.00% |
| bs | 2 | 0.00% |
| sr | 1 | 0.00% |

### py3langid calibration (measured against ai4privacy's ground-truth `language`, EU languages only)

**Sampling method:** every twentieth row by absolute position in the full
stream across both splits and all languages (`total_rows % 20 == 0` in the
code), not every twentieth row within each language. This approximates a
1-in-20 per-language draw only because ai4privacy's rows are grouped into
large contiguous blocks by source region/language rather than shuffled --
each language's rows form a long run, so a fixed-stride positional sample
lands roughly evenly within each run. The approximation's limit: a language
whose rows were interleaved with others rather than contiguous could be
sampled less evenly than a true per-language 1-in-20 draw, and that has not
been ruled out for every language in this corpus. Spot-checking the sample
sizes below against a fair 1-in-20 draw for German, French, English and
Estonian found them within 1-5%, which is reassuring but not a proof for
every language.

| language | py3langid correct | sampled rows | accuracy |
|---|---:|---:|---:|
| bg | 2,173 | 2,337 | 93.0% |
| cs | 2,187 | 2,386 | 91.7% |
| da | 1,783 | 2,192 | 81.3% |
| de | 5,843 | 5,923 | 98.6% |
| el | 2,117 | 2,312 | 91.6% |
| en | 10,315 | 10,421 | 99.0% |
| es | 4,449 | 4,463 | 99.7% |
| et | 1,651 | 2,381 | 69.3% |
| fi | 1,646 | 2,211 | 74.4% |
| fr | 6,652 | 6,683 | 99.5% |
| hr | 1,760 | 2,257 | 78.0% |
| hu | 1,834 | 2,168 | 84.6% |
| it | 4,278 | 4,428 | 96.6% |
| lt | 1,809 | 2,262 | 80.0% |
| lv | 1,729 | 2,280 | 75.8% |
| nl | 2,224 | 2,273 | 97.8% |
| pl | 2,024 | 2,344 | 86.3% |
| pt | 2,146 | 2,168 | 99.0% |
| ro | 2,036 | 2,217 | 91.8% |
| sk | 1,661 | 2,216 | 75.0% |
| sl | 1,907 | 2,291 | 83.2% |
| sv | 2,060 | 2,270 | 90.7% |
| **overall** | **64,284** | **70,483** | **91.2%** |

**Read the near-zero rows in the Wismut language table above as "the
estimator could not confidently place these rows," not as "the corpus
lacks these languages."** Wismut has no language ground truth at all, so
these numbers cannot be verified either way from the data alone. What can
be checked is whether the near-zero outcomes track with `py3langid`'s own
weak spots, and they do: `bg` (77 est. rows) calibrates at 93.0% accuracy on
ai4privacy, but `lv` (0 est. rows), `hr` (5 est. rows), `et` (30 est. rows),
`sk` (235 est. rows) and `lt` (13 est. rows) all sit in the 69-80% accuracy
band -- the weakest of all 22 EU languages calibrated. `mt` and `ga` have no
calibration data at all (ai4privacy carries neither). Low estimated counts
for `lv`, `hr`, `et`, `sk`, `lt`, `hu`, and `sl` are consistent with a
short-text identifier struggling on synthetic, entity-heavy strings in
languages it is already measurably weaker at -- not with those languages
being absent from Wismut. This does not change the sufficiency conclusion:
ai4privacy alone carries all eight of the ai4privacy-unique languages at
44,900+ rows regardless of what Wismut contributes.

### Rows per label (from `entities`)

| label | rows | share of rows |
|---|---:|---:|
| GIVEN_NAME | 445,727 | 55.37% |
| SURNAME | 431,239 | 53.57% |
| DATE | 298,352 | 37.06% |
| PHONE | 240,928 | 29.93% |
| CITY | 197,528 | 24.54% |
| DATE_OF_BIRTH | 177,128 | 22.00% |
| EMAIL | 171,000 | 21.24% |
| CUSTOMER_ID | 160,352 | 19.92% |
| STREET_ADDRESS | 139,160 | 17.29% |
| COMPANY_NAME | 128,904 | 16.01% |
| ZIP_CODE | 107,584 | 13.36% |
| GOVERNMENT_ID | 93,864 | 11.66% |
| ACCOUNT_NUMBER | 89,920 | 11.17% |
| BUILDING_NUMBER | 86,959 | 10.80% |
| USERNAME | 84,232 | 10.46% |
| TIME | 80,480 | 10.00% |
| PASSPORT | 72,400 | 8.99% |
| TAX_ID | 71,272 | 8.85% |
| EMPLOYEE_ID | 70,984 | 8.82% |
| SSN | 69,784 | 8.67% |
| IBAN | 62,096 | 7.71% |
| CREDIT_DEBIT_CARD | 61,520 | 7.64% |
| COUNTRY | 60,768 | 7.55% |
| DRIVERS_LICENSE | 54,952 | 6.83% |
| STREET_NAME | 50,712 | 6.30% |
| STATE | 49,784 | 6.18% |
| MEDICAL_RECORD_NUMBER | 49,576 | 6.16% |
| SECONDARY_ADDRESS | 34,648 | 4.30% |
| CVV | 33,352 | 4.14% |
| LICENSE_PLATE | 32,008 | 3.98% |
| PASSWORD | 31,800 | 3.95% |
| GENDER | 30,955 | 3.85% |
| API_KEY | 28,248 | 3.51% |
| AGE | 27,648 | 3.43% |
| SWIFT_BIC | 27,112 | 3.37% |
| MAC_ADDRESS | 24,336 | 3.02% |
| ROUTING_NUMBER | 24,224 | 3.01% |
| URL | 21,688 | 2.69% |
| PIN | 14,920 | 1.85% |
| FAX_NUMBER | 14,704 | 1.83% |

### Full label vocabulary, sorted (40 labels)

ACCOUNT_NUMBER, AGE, API_KEY, BUILDING_NUMBER, CITY, COMPANY_NAME, COUNTRY, CREDIT_DEBIT_CARD, CUSTOMER_ID, CVV, DATE, DATE_OF_BIRTH, DRIVERS_LICENSE, EMAIL, EMPLOYEE_ID, FAX_NUMBER, GENDER, GIVEN_NAME, GOVERNMENT_ID, IBAN, LICENSE_PLATE, MAC_ADDRESS, MEDICAL_RECORD_NUMBER, PASSPORT, PASSWORD, PHONE, PIN, ROUTING_NUMBER, SECONDARY_ADDRESS, SSN, STATE, STREET_ADDRESS, STREET_NAME, SURNAME, SWIFT_BIC, TAX_ID, TIME, URL, USERNAME, ZIP_CODE

---

## Combined 24-EU-language sufficiency table

ai4privacy counts are measured (ground truth). Wismut counts are the
py3langid estimate described above -- see the calibration table for its
accuracy on this style of text. Sufficiency threshold: 5,000 rows,
combined, per the task brief.

| language | ai4privacy rows (measured) | Wismut rows (py3langid estimate) | combined | sufficient (>= 5,000) |
|---|---:|---:|---:|:---:|
| bg (Bulgarian) | 46,801 | 77 (est.) | 46,878 | yes |
| cs (Czech) | 46,473 | 28,781 (est.) | 75,254 | yes |
| da (Danish) | 45,062 | 32,221 (est.) | 77,283 | yes |
| de (German) | 120,105 | 30,899 (est.) | 151,004 | yes |
| el (Greek) | 45,178 | 22,096 (est.) | 67,274 | yes |
| en (English) | 204,649 | 176,755 (est.) | 381,404 | yes |
| es (Spanish) | 90,254 | 38,532 (est.) | 128,786 | yes |
| et (Estonian) | 45,777 | 30 (est.) | 45,807 | yes |
| fi (Finnish) | 45,313 | 32,808 (est.) | 78,121 | yes |
| fr (French) | 131,173 | 30,427 (est.) | 161,600 | yes |
| ga (Irish) | 0 | 2 (est.) | 2 | **NO** |
| hr (Croatian) | 44,935 | 5 (est.) | 44,940 | yes |
| hu (Hungarian) | 45,345 | 9 (est.) | 45,354 | yes |
| it (Italian) | 89,304 | 36,817 (est.) | 126,121 | yes |
| lt (Lithuanian) | 45,823 | 13 (est.) | 45,836 | yes |
| lv (Latvian) | 45,636 | 0 (est.) | 45,636 | yes |
| mt (Maltese) | 0 | 24 (est.) | 24 | **NO** |
| nl (Dutch) | 44,460 | 29,317 (est.) | 73,777 | yes |
| pl (Polish) | 46,621 | 19,595 (est.) | 66,216 | yes |
| pt (Portuguese) | 43,819 | 23,279 (est.) | 67,098 | yes |
| ro (Romanian) | 45,322 | 29,549 (est.) | 74,871 | yes |
| sk (Slovak) | 45,739 | 235 (est.) | 45,974 | yes |
| sl (Slovenian) | 44,993 | 181 (est.) | 45,174 | yes |
| sv (Swedish) | 45,774 | 21,160 (est.) | 66,934 | yes |


> **STOP CONDITION TRIGGERED.** The following EU languages fall below the 5,000-row threshold across both corpora combined: ga (Irish): 2 rows, mt (Maltese): 24 rows. Per the task brief, this is a decision for the human partner, not a gap to quietly fill: Task 3 (data generation) would need to cover these languages too, which is a materially larger job than the plan currently assumes.
>
> **Ruling (human partner, 2026-08-20): proceed, plan unchanged.** The stop
> condition was worded too broadly -- it should have excluded exactly Irish
> and Maltese, because Task 3 of this plan already exists to generate those
> two: Faker carries `ga_IE` and `mt_MT` locales, both verified. The
> condition fired on precisely the gap the plan already accounts for, so it
> does not block anything. Every other EU language clears the threshold
> comfortably from ai4privacy alone (see the headline finding above), which
> is the result that mattered for whether the rest of this plan is safe to
> build on.

