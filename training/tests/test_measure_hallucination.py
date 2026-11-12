"""Tests for measure_hallucination.py's corpus digest and its refusal to
compare results measured on different corpora.

`reports/student-v2-run.md` quoted 8.9 spans/1,000 rows against a brief that
named the shipped model's 22.8 -- and the two were measured on different
held-out negative splits (a corpus expansion had landed between them). The
same shipped model re-measures at 49.0 on the split 8.9 actually shares.
Nothing in the tool caught that; a human re-derived the number by hand. These
tests pin the fix: every result carries a `corpus_digest`, and
`compare_reports` refuses to line up two results whose digests disagree.

No model or tokenizer is loaded here -- `corpus_digest` and `compare_reports`
are pure functions over rows and JSON, and that is exactly what lets this
suite catch a regression in them without paying for a GPU run. A real,
model-backed end-to-end run (two small corpora through the actual CLI, one
refusing to compare and one succeeding) was also run by hand against
`runs/student-6l` as part of implementing this; it is not repeated here
because it would make this suite pay checkpoint-load cost on every run for
no more coverage than the pure-function tests below already give the guard
logic itself.
"""

from __future__ import annotations

import pytest

from measure_hallucination import compare_reports, corpus_digest, load_reports


def _row(text: str, source: str = "encyclopedia", language: str = "en") -> dict:
    return {"text": text, "source": source, "language": language, "entities": []}


class TestCorpusDigest:
    def test_same_rows_same_digest(self):
        a = [_row("a"), _row("b")]
        b = [_row("a"), _row("b")]
        assert corpus_digest(a) == corpus_digest(b)

    def test_different_text_different_digest(self):
        a = [_row("a"), _row("b")]
        b = [_row("a"), _row("c")]
        assert corpus_digest(a) != corpus_digest(b)

    def test_order_is_part_of_the_digest(self):
        # A grouping change or a shuffled split file is a different
        # measurement, not the same corpus in a different hat.
        rows = [_row("a"), _row("b")]
        reordered = [_row("b"), _row("a")]
        assert corpus_digest(rows) != corpus_digest(reordered)

    def test_row_count_changes_the_digest(self):
        # This is the exact shape of the bug: `--limit` truncating the file
        # into a different population than a previous run measured.
        rows = [_row("a"), _row("b"), _row("c")]
        assert corpus_digest(rows) != corpus_digest(rows[:2])

    def test_a_field_change_changes_the_digest(self):
        # source/language participate too, not just the prose.
        a = [_row("a", source="encyclopedia")]
        b = [_row("a", source="travel-guide")]
        assert corpus_digest(a) != corpus_digest(b)

    def test_digest_is_hex_sha256(self):
        digest = corpus_digest([_row("a")])
        assert len(digest) == 64
        int(digest, 16)  # ValueError if not hex

    def test_empty_corpus_is_stable(self):
        assert corpus_digest([]) == corpus_digest([])


def _result(checkpoint: str, digest, report_file: str, **extra) -> dict:
    base = {
        "checkpoint": checkpoint,
        "corpus_digest": digest,
        "rows": 100,
        "spans_per_1000_rows": 10.0,
        "rows_with_any_span_pct": 5.0,
    }
    base.update(extra)
    base["_report_file"] = report_file
    return base


class TestCompareReports:
    def test_refuses_two_different_digests(self):
        # The exact incident: a shipped-model number and a candidate number
        # that look like they belong on the same axis and do not.
        entries = [
            _result("shipped", "a" * 64, "shipped.json"),
            _result("v2", "b" * 64, "v2.json"),
        ]
        with pytest.raises(SystemExit, match="different corpora"):
            compare_reports(entries)

    def test_refusal_names_both_digests_and_files_not_row_content(self):
        entries = [
            _result("shipped", "a" * 64, "shipped.json"),
            _result("v2", "b" * 64, "v2.json"),
        ]
        with pytest.raises(SystemExit) as excinfo:
            compare_reports(entries)
        message = str(excinfo.value)
        assert "shipped.json" in message
        assert "v2.json" in message
        assert "a" * 12 in message
        assert "b" * 12 in message

    def test_same_digest_succeeds_and_lists_every_checkpoint(self):
        digest = "c" * 64
        entries = [
            _result("shipped", digest, "run.json"),
            _result("v2", digest, "run.json"),
        ]
        table = compare_reports(entries)
        assert "shipped" in table
        assert "v2" in table
        assert digest[:12] in table

    def test_a_single_result_is_trivially_comparable_to_itself(self):
        table = compare_reports([_result("only", "d" * 64, "run.json")])
        assert "only" in table

    def test_missing_digest_refuses_rather_than_assuming_it_matches(self):
        # A result written before this field existed cannot be trusted to
        # be the same corpus as a fresh one, even if nothing else disagrees.
        legacy = _result("legacy", None, "old.json")
        del legacy["corpus_digest"]
        fresh = _result("v2", "e" * 64, "new.json")
        with pytest.raises(SystemExit, match="predate corpus_digest"):
            compare_reports([legacy, fresh])

    def test_no_entries_refuses(self):
        with pytest.raises(SystemExit):
            compare_reports([])


class TestLoadReports:
    def test_tags_each_entry_with_its_source_file(self, tmp_path):
        import json

        path = tmp_path / "report.json"
        path.write_text(json.dumps([_result("ckpt", "f" * 64, "unused")]))
        entries = load_reports([path])
        assert entries[0]["_report_file"] == str(path)

    def test_flattens_across_multiple_files(self, tmp_path):
        import json

        p1 = tmp_path / "a.json"
        p2 = tmp_path / "b.json"
        p1.write_text(json.dumps([_result("one", "a" * 64, "unused")]))
        p2.write_text(json.dumps([_result("two", "a" * 64, "unused")]))
        entries = load_reports([p1, p2])
        assert {e["checkpoint"] for e in entries} == {"one", "two"}
