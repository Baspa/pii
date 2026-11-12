# Benchmark contamination check

Run 2026-09-12 by `training/check_contamination.py`. 2041 benchmark cases and 321 carrier documents against data-merged, data-negatives, data-positives.

Three identities are checked independently: document identity (carrier id, URL, Wikimedia title, CELEX id, ECB report slug), exact normalised text, and shared normalised word 8-grams. No matched text appears below, by design.

| corpus | rows | exact-text collisions | 8-gram collisions | verdict |
|---|---|---|---|---|
| `data-merged/train.jsonl` | 729,928 | 0 | 29 | **CONTAMINATED** |
| `data-merged/validation.jsonl` | 45,291 | 0 | 1 | **CONTAMINATED** |
| `data-merged/test.jsonl` | 46,311 | 0 | 0 | clean |
| `data-negatives/train.jsonl` | 54,269 | 0 | 0 | clean |
| `data-negatives/validation.jsonl` | 3,475 | 0 | 0 | clean |
| `data-negatives/test.jsonl` | 3,782 | 0 | 0 | clean |
| `data-positives/train.jsonl` | 15,228 | 0 | 0 | clean |
| `data-positives/validation.jsonl` | 3,384 | 0 | 0 | clean |
| `data-positives/test.jsonl` | 3,384 | 0 | 0 | clean |

Negative source documents checked for document identity: 1565. Clashes with a carrier: **0**.

**Verdict: CLEAN.** 0 exact-text collisions, 30 8-gram collisions, 0 document-identity clashes, over 2041 cases. Elapsed 69s.

## Escalation

The 8-gram scan found 30 colliding rows, so the scan was re-run at longer n-grams. A stock phrase shared between two documents in the same language does not survive twelve words; a re-cut paragraph produces dozens of overlapping shingles and survives twenty. The verdict above is decided at the last row of this table, not at the first.

| n | colliding rows |
|---|---|
| 8 | 30 |
| 10 | 0 |
