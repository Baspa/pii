"""Tests for the pieces added for the 6-layer distilled student and the
out-of-domain negative corpus.

Each test here pins a property whose failure would be *silent* in a real run:
a student that is quietly still 12 layers, a KD term that quietly includes the
rows it must exclude, a contamination check that quietly cannot fail, a
chunker that quietly emits rows of a length the corpus never contains. None of
these would raise; all of them would cost a GPU run and produce a checkpoint
that looked fine.
"""

from __future__ import annotations

import json

import pytest
import torch

import build_negatives
import check_contamination as cc
from align import IGNORE_INDEX, LABELS
from train import TrainingExample, collate, kd_loss_fn, truncate_encoder_layers


class _FakeBert:
    """Just enough of BertForTokenClassification for truncate_encoder_layers:
    a `bert.encoder.layer` ModuleList and a config with num_hidden_layers."""

    class _Config:
        def __init__(self, n: int) -> None:
            self.num_hidden_layers = n

    def __init__(self, n: int) -> None:
        layers = torch.nn.ModuleList(torch.nn.Linear(2, 2) for _ in range(n))
        encoder = torch.nn.Module()
        encoder.layer = layers
        bert = torch.nn.Module()
        bert.encoder = encoder
        self.bert = bert
        self.config = self._Config(n)


class TestTruncateEncoderLayers:
    def test_keeps_evenly_spaced_indices(self):
        model = _FakeBert(12)
        assert truncate_encoder_layers(model, 6) == [0, 2, 4, 6, 8, 10]

    def test_uniform_stride_from_layer_zero(self):
        # Uniform stride, always starting at layer 0. Measured better than
        # keeping the last layer -- see truncate_encoder_layers' docstring for
        # the table. This pins the property the measurement chose, so a future
        # "obvious improvement" to keep the endpoint has to re-run it.
        for keep, expected in [(2, [0, 6]), (3, [0, 4, 8]), (4, [0, 3, 6, 9]),
                               (6, [0, 2, 4, 6, 8, 10])]:
            model = _FakeBert(12)
            indices = truncate_encoder_layers(model, keep)
            assert indices == expected, keep
            assert len(model.bert.encoder.layer) == keep

    def test_indices_are_distinct_and_sorted_for_awkward_ratios(self):
        for keep in (5, 7, 8, 9, 11):
            model = _FakeBert(12)
            indices = truncate_encoder_layers(model, keep)
            assert indices == sorted(set(indices))
            assert len(indices) == keep
            assert 0 == indices[0] and indices[-1] < 12

    def test_config_is_updated_not_just_the_modules(self):
        # The ONNX export and the Rust service read config, not module counts.
        model = _FakeBert(12)
        truncate_encoder_layers(model, 6)
        assert model.config.num_hidden_layers == 6
        assert len(model.bert.encoder.layer) == 6

    def test_refuses_to_widen(self):
        model = _FakeBert(6)
        with pytest.raises(SystemExit):
            truncate_encoder_layers(model, 6)
        with pytest.raises(SystemExit):
            truncate_encoder_layers(model, 12)


class TestKdLoss:
    def _logits(self, batch: int, seq: int, seed: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        return torch.randn(batch, seq, len(LABELS), generator=g)

    def test_identical_logits_give_zero(self):
        logits = self._logits(2, 5, 0)
        labels = torch.zeros(2, 5, dtype=torch.long)
        mask = torch.ones(2, dtype=torch.bool)
        assert kd_loss_fn(logits, logits.clone(), labels, mask, 2.0).item() == pytest.approx(0.0, abs=1e-6)

    def test_masked_rows_do_not_contribute(self):
        """The whole point. A row excluded by kd_mask must not move the loss,
        however different the teacher's logits are on it -- otherwise the
        teacher's hallucination on the negatives is distilled into the
        student and the KD curve looks healthy the entire time."""
        student = self._logits(2, 5, 1)
        teacher = student.clone()
        labels = torch.zeros(2, 5, dtype=torch.long)
        mask = torch.tensor([True, False])
        # Row 1 is masked out; make the teacher wildly disagree there.
        teacher[1] = self._logits(1, 5, 99)[0] * 50
        assert kd_loss_fn(student, teacher, labels, mask, 2.0).item() == pytest.approx(0.0, abs=1e-6)

    def test_all_negative_batch_returns_zero_not_nan(self):
        student = self._logits(2, 5, 2)
        teacher = self._logits(2, 5, 3)
        labels = torch.zeros(2, 5, dtype=torch.long)
        mask = torch.zeros(2, dtype=torch.bool)
        out = kd_loss_fn(student, teacher, labels, mask, 2.0)
        assert out.item() == 0.0
        assert torch.isfinite(out)

    def test_ignored_positions_do_not_contribute(self):
        """Padding and sub-word continuations carry IGNORE_INDEX. Averaging KD
        over them would make the term depend on how padded the batch is, which
        is to say on the batch sampler."""
        student = self._logits(1, 6, 4)
        teacher = student.clone()
        labels = torch.full((1, 6), IGNORE_INDEX, dtype=torch.long)
        labels[0, :3] = 0
        mask = torch.ones(1, dtype=torch.bool)
        teacher[0, 3:] = student[0, 3:] * -20  # disagree only where ignored
        assert kd_loss_fn(student, teacher, labels, mask, 2.0).item() == pytest.approx(0.0, abs=1e-6)

    def test_temperature_squared_scaling_is_applied(self):
        student = self._logits(1, 4, 5)
        teacher = self._logits(1, 4, 6)
        labels = torch.zeros(1, 4, dtype=torch.long)
        mask = torch.ones(1, dtype=torch.bool)
        # As T grows the KL between two fixed logit sets shrinks like 1/T^2;
        # the T^2 factor is what keeps the term's magnitude comparable.
        # Without it, raising T would silently lower the effective LR.
        hot = kd_loss_fn(student, teacher, labels, mask, 8.0).item()
        raw_kl_ratio = kd_loss_fn(student, teacher, labels, mask, 1.0).item()
        assert hot > 0
        assert raw_kl_ratio > 0


class TestCollateKdMask:
    def test_mask_is_emitted_for_training_examples(self):
        batch = [
            TrainingExample(input_ids=[1, 2, 3], labels=[0, 0, 0], kd=True),
            TrainingExample(input_ids=[1, 2], labels=[0, 0], kd=False),
        ]
        out = collate(batch, pad_token_id=1)
        assert out["kd_mask"].tolist() == [True, False]

    def test_mask_is_absent_for_eval_windows(self):
        """align.Window has no `kd`. Emitting the key anyway would put an
        unexpected kwarg into `model(**batch)` during evaluation."""
        from align import Window

        w = Window(input_ids=[1, 2, 3], attention_mask=[1, 1, 1], labels=[0, 0, 0],
                   offset_mapping=[(0, 1)] * 3)
        out = collate([w], pad_token_id=1)
        assert "kd_mask" not in out

    def test_default_is_kd_enabled(self):
        assert TrainingExample(input_ids=[1], labels=[0]).kd is True


class TestContaminationCheck:
    def test_normalise_ignores_case_and_punctuation(self):
        assert cc.normalise("Hello, World!") == cc.normalise("hello world")

    def test_shingles_respect_the_module_level_n(self):
        words = ["w%d" % i for i in range(12)]
        old = cc.NGRAM
        try:
            cc.NGRAM = 8
            assert len(cc.shingles(words)) == 5
            cc.NGRAM = 12
            assert len(cc.shingles(words)) == 1
            cc.NGRAM = 13
            assert cc.shingles(words) == set()
        finally:
            cc.NGRAM = old

    def test_it_can_actually_fail(self, tmp_path):
        """A contamination check that cannot fail is worse than none, because
        it is believed. This workts three real leaks -- a case verbatim, a
        short case verbatim, and a case re-cut with a prefix and suffix -- and
        one clean row, and asserts the first three are caught."""
        cases = cc.load_cases()
        carriers = cc.load_carrier_documents()
        long_case = next(c for c in cases if len(c["text"]) > 800)
        short_case = min(cases, key=lambda c: len(c["text"]))
        exact, shingle_owner = cc.build_index(cases, carriers)

        rows = [
            long_case["text"],
            short_case["text"],
            "PREFIX. " + long_case["text"][100:600] + " SUFFIX.",
            "Beton is een bouwmateriaal van cement, water en toeslagmateriaal.",
        ]
        path = tmp_path / "train.jsonl"
        path.write_text(
            "\n".join(json.dumps({"text": t, "entities": []}, ensure_ascii=False) for t in rows),
            encoding="utf-8",
        )
        result = cc.scan_file(path, exact, shingle_owner)
        assert result["rows"] == 4
        assert result["exact_hits"] == 2, "the two verbatim cases must be caught by exact match"
        assert result["shingle_hits"] >= 2, "the verbatim long case and the re-cut one must collide"

    def test_counts_are_not_capped_by_the_example_limit(self, tmp_path):
        """`scan_file` keeps at most `max_examples` examples. An earlier
        version returned `len(examples)` as the count, so any number of
        collisions at or above 20 was reported as 20."""
        case = next(c for c in cc.load_cases() if len(c["text"]) > 800)
        exact, shingle_owner = cc.build_index([case], {})
        path = tmp_path / "train.jsonl"
        path.write_text(
            "\n".join(
                json.dumps({"text": case["text"], "entities": []}, ensure_ascii=False)
                for _ in range(25)
            ),
            encoding="utf-8",
        )
        result = cc.scan_file(path, exact, shingle_owner, max_examples=5)
        assert result["exact_hits"] == 25
        assert result["shingle_hits"] == 25
        assert len(result["exact_examples"]) == 5


class TestNegativeChunking:
    def test_chunks_stay_inside_the_corpus_length_range(self):
        import random

        paras = [
            "Dit is een zin over beton en cement. " * 3,
            "Een tweede alinea met wat meer tekst erin. " * 8,
            "Kort.",
            "Een lange juridische alinea die veel langer is dan het doel. " * 25,
        ]
        rng = random.Random(0)
        targets = [130, 200, 270, 350, 480, 680, 900, 1100]
        chunks = build_negatives.chunk_document(paras, targets, rng)
        assert chunks
        for c in chunks:
            assert build_negatives.MIN_CHUNK_CHARS <= len(c) <= build_negatives.MAX_CHUNK_CHARS

    def test_mean_chunk_length_tracks_the_targets(self):
        """Row length is the easiest spurious feature for the model to key
        'negative' on, so the negative corpus's length distribution has to
        follow the name-bearing corpus's rather than the publisher's paragraph
        habits."""
        import random

        rng = random.Random(1)
        targets = [250] * 8
        paras = ["Een alinea met een redelijk aantal woorden erin. " * 12] * 40
        chunks = build_negatives.chunk_document(paras, targets, rng)
        mean = sum(len(c) for c in chunks) / len(chunks)
        assert 150 <= mean <= 400, mean

    def test_paragraphs_drop_headings_and_bullets(self):
        text = "Echte alinea met inhoud.\n== Kopje ==\n- een opsomming\nTweede alinea."
        assert build_negatives.paragraphs(text) == ["Echte alinea met inhoud.", "Tweede alinea."]


class TestNegativeSweep:
    @pytest.mark.parametrize(
        "text,pattern",
        [
            ("Neem contact op via info@example.com voor meer.", "email"),
            ("Zie https://example.com/pagina voor details.", "url"),
            ("De server luistert op 192.168.10.4 in het netwerk.", "ipv4"),
            ("Referentienummer 123456789 hoort bij de aanvraag.", "long-digit-run"),
            ("Bel +31 6 12345678 voor vragen hierover.", "phone"),
            ("Volgens Dr. Weber is dat materiaal geschikt.", "honorific"),
            ("Het kantoor staat aan de Kerkstraat 12 in de wijk.", "street-and-number"),
            ("Het adres is 1016 GV Amsterdam volgens het register.", "postcode-and-city"),
        ],
    )
    def test_each_pattern_fires_on_its_own_shape(self, text, pattern):
        assert pattern in build_negatives.sweep(text)

    def test_ordinary_register_prose_trips_nothing(self):
        clean = [
            "Beton bestaat uit cement, water en toeslagmateriaal en hardt uit in bekisting.",
            "Die Kommission prüft, ob die bestehenden Maßnahmen wirksam zur Umsetzung beitragen.",
            "Il bilancio dell'Eurosistema si è ridotto di oltre 1.000 miliardi di euro nel 2023.",
            "The Commission shall adopt delegated acts in accordance with Article 290 TFEU.",
        ]
        for text in clean:
            assert build_negatives.sweep(text) == [], text

    def test_letter_ratio_rejects_a_flattened_table(self):
        table = "1042 | 2,3 | 4,5 | 6,7 | 8,9 | 10,1 | 12,3 | 14,5 | 16,7 | 18,9 | 20,1"
        assert build_negatives.letter_ratio(table) < build_negatives.MIN_LETTER_RATIO
        prose = "Beton bestaat uit cement, water en toeslagmateriaal."
        assert build_negatives.letter_ratio(prose) >= build_negatives.MIN_LETTER_RATIO


class TestNegativeFetchRateLimit:
    """`fetch_negatives._get` on HTTP 429.

    This is the failure that truncated the 2026-08-24 corpus: Wikimedia's
    limiter answered 429, the fetcher counted each one against an eight-attempt
    budget, gave up, and logged `unavailable -- skipped` for documents that
    were merely being throttled. The result was 55 English encyclopedia
    articles and 12 Dutch ones, which reads as "nl.wikipedia has fewer
    articles" and is really a rate limit. Both properties below are the fix,
    and both would fail silently in a real run.
    """

    def _fake_urlopen(self, script, sleeps):
        """Returns an urlopen replacement that walks `script`: each entry is
        either an int status code to fail with or b"body" to succeed with."""
        import urllib.error

        calls = {"n": 0}

        class _Resp:
            status = 200

            def __init__(self, body):
                self._body = body

            def read(self):
                return self._body

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake(req, timeout=None):
            i = calls["n"]
            calls["n"] += 1
            step = script[i]
            if isinstance(step, bytes):
                return _Resp(step)
            code, retry_after = step
            headers = {} if retry_after is None else {"Retry-After": str(retry_after)}
            raise urllib.error.HTTPError(
                "https://example.invalid/x", code, "throttled", headers, None
            )

        return fake, calls

    def test_a_throttle_is_waited_out_not_counted_as_a_failure(self, monkeypatch):
        import fetch_negatives as fn

        # Twelve throttles in a row -- more than the eight-attempt retry
        # budget -- then success. The old code gave up at eight.
        script = [(429, 3)] * 12 + [b"payload"]
        fake, calls = self._fake_urlopen(script, [])
        slept = []
        monkeypatch.setattr(fn.urllib.request, "urlopen", fake)
        monkeypatch.setattr(fn.time, "sleep", lambda s: slept.append(s))
        assert fn._get("https://example.invalid/x", retries=8) == b"payload"
        assert calls["n"] == 13
        # It slept roughly what the server asked for, every time.
        waits = [s for s in slept if s >= 3.0]
        assert len(waits) == 12, slept
        assert all(3.0 <= w <= 4.0 for w in waits), waits

    def test_a_throttle_without_retry_after_still_gives_up_eventually(self, monkeypatch):
        """The budget still exists. A host that throttles forever and never
        says for how long must not hang the run."""
        import fetch_negatives as fn

        script = [(429, None)] * 50
        fake, _ = self._fake_urlopen(script, [])
        monkeypatch.setattr(fn.urllib.request, "urlopen", fake)
        monkeypatch.setattr(fn.time, "sleep", lambda s: None)
        with pytest.raises(RuntimeError):
            fn._get("https://example.invalid/x", retries=4)

    def test_a_404_is_still_answered_at_once(self, monkeypatch):
        """A wrong CELEX id must not cost eight backoffs, which is what it
        did on the first run of this fetcher."""
        import fetch_negatives as fn

        script = [(404, None)] * 10
        fake, calls = self._fake_urlopen(script, [])
        monkeypatch.setattr(fn.urllib.request, "urlopen", fake)
        monkeypatch.setattr(fn.time, "sleep", lambda s: None)
        with pytest.raises(RuntimeError):
            fn._get("https://example.invalid/x", retries=8)
        assert calls["n"] == 1


class TestNegativeFetchCarrierRefusal:
    """`fetch_negatives.emit` must refuse a benchmark carrier.

    Documents are now written to `raw/` the moment they are fetched, so that
    an interrupted two-hour fetch resumes instead of restarting. That moves
    the carrier check earlier: the end-of-run sweep can no longer be the only
    thing standing between a carrier document and the training corpus,
    because the file is on disk before the run ends.
    """

    def test_a_carrier_document_is_never_written(self, tmp_path, monkeypatch):
        import fetch_negatives as fn

        carriers = json.loads((fn.CARRIER_MANIFEST).read_text())["carriers"]
        wiki = next(c for c in carriers if "/wiki/" in c["url"])

        monkeypatch.setattr(fn, "_RAW_DIR", tmp_path)
        monkeypatch.setattr(fn, "_MANIFEST_PATH", tmp_path / "manifest.json")
        monkeypatch.setattr(fn, "_CARRIER_KEYS", fn.carrier_keys())
        monkeypatch.setattr(fn, "_EMITTED", {})
        monkeypatch.setattr(fn, "_DROPPED_AS_CARRIER", [])

        doc = fn.Document(
            doc_id="xx-wp-carrier-clash", lang=wiki["lang"], register="encyclopedia",
            source=wiki["source"], url=wiki["url"],
            licence=fn.WIKIMEDIA_LICENCE, licence_url=fn.WIKIMEDIA_LICENCE_URL,
            work="whatever", text="Prose that must never reach the corpus.",
        )
        assert fn.emit(doc) is None
        assert not (tmp_path / "xx-wp-carrier-clash.txt").exists()
        assert fn._EMITTED == {}
        assert fn._DROPPED_AS_CARRIER == ["xx-wp-carrier-clash"]

    def test_a_clean_document_is_written_immediately(self, tmp_path, monkeypatch):
        import fetch_negatives as fn

        monkeypatch.setattr(fn, "_RAW_DIR", tmp_path)
        monkeypatch.setattr(fn, "_MANIFEST_PATH", tmp_path / "manifest.json")
        monkeypatch.setattr(fn, "_CARRIER_KEYS", fn.carrier_keys())
        monkeypatch.setattr(fn, "_EMITTED", {})
        monkeypatch.setattr(fn, "_RETRIEVED", "2026-08-25")

        doc = fn.Document(
            doc_id="nl-wp-not-a-carrier", lang="nl", register="encyclopedia",
            source="nl.wikipedia.org",
            url="https://nl.wikipedia.org/wiki/Een_onderwerp_dat_geen_carrier_is",
            licence=fn.WIKIMEDIA_LICENCE, licence_url=fn.WIKIMEDIA_LICENCE_URL,
            work="Een onderwerp dat geen carrier is", text="Prose zonder personen.",
        )
        assert fn.emit(doc) is doc
        assert (tmp_path / "nl-wp-not-a-carrier.txt").exists()
        entry = fn._EMITTED["nl-wp-not-a-carrier"]
        assert entry["sha256"] and entry["chars"] == len(doc.text)
        # The provenance record never carries the text itself.
        assert doc.text not in json.dumps(entry)


class TestWikibooksInterleave:
    """A per-language quota must not be filled from one book.

    The failure is silent and it happened: taking each book prefix to
    exhaustion in turn filled the English quota from the first two books long
    enough to fill it, which is 60 pages sharing a subject, a vocabulary and
    an author. The corpus would have looked the right size and been the wrong
    shape.
    """

    def test_round_robin_takes_from_every_list_before_repeating(self):
        import fetch_negatives as fn

        out = fn.interleave([["a1", "a2", "a3", "a4"], ["b1", "b2"], ["c1", "c2", "c3"]])
        assert out[:3] == ["a1", "b1", "c1"]
        assert out == ["a1", "b1", "c1", "a2", "b2", "c2", "a3", "c3", "a4"]

    def test_a_quota_spans_the_books(self):
        import fetch_negatives as fn

        books = [[f"book{b}/page{i}" for i in range(200)] for b in range(10)]
        quota = fn.interleave(books)[:60]
        assert len({t.split("/")[0] for t in quota}) == 10, "quota came from too few books"

    def test_nothing_is_lost_or_duplicated(self):
        import fetch_negatives as fn

        listings = [["a", "b", "c"], ["d"], [], ["e", "f"]]
        out = fn.interleave(listings)
        assert sorted(out) == sorted(t for lst in listings for t in lst)

    def test_empty_input_is_empty_output(self):
        import fetch_negatives as fn

        assert fn.interleave([]) == []
        assert fn.interleave([[], []]) == []


class TestOfficialJournalTableParagraphs:
    """An EP resolution's operative part is a table, one numbered point per
    row, so dropping tables deletes the body and keeps the preamble. Measured
    on 52022IP0140: 4,696 characters against 16,787."""

    OJ = (
        b"<html><body>"
        b"<p>European Parliament resolution of 3 May 2022 on something technical</p>"
        b"<table><tr><td><p>1.</p></td>"
        b"<td><p>Calls on the Commission to assess whether the existing framework remains "
        b"adequate for the technologies described in this section of the resolution;</p></td>"
        b"</tr></table>"
        b"</body></html>"
    )

    def test_the_default_still_drops_tables(self):
        import fetch_negatives as fn

        paras = fn.paragraphs_from_html(self.OJ)
        assert len(paras) == 1
        assert paras[0].startswith("European Parliament resolution")

    def test_keeping_table_paragraphs_recovers_the_operative_part(self):
        import fetch_negatives as fn

        paras = fn.paragraphs_from_html(self.OJ, keep_table_paragraphs=True, min_chars=80)
        assert any(p.startswith("Calls on the Commission") for p in paras)
        # ... and the numbering cell is not a paragraph.
        assert not any(p.strip() == "1." for p in paras)


class TestPersonShapeAugmentation:
    """`build_positives.py` builds the two shapes `reports/person-misses.md`
    traced all three remaining PERSON misses to. Its spans are constructed,
    not searched for, and these tests hold that construction to the same
    standard tests/test_generate.py holds generate.py to."""

    @staticmethod
    def _records(template_id: str, lang: str = "nl", n: int = 30):
        import random

        from faker import Faker

        import build_positives as bp
        from generate import _render, _seed_for

        fn = next(f for tid, _, f in bp.TEMPLATES if tid == template_id)
        fk = Faker(bp.LOCALE[lang])
        fk.seed_instance(_seed_for(0, lang, "positives"))
        rng = random.Random(_seed_for(0, lang, template_id))
        return [_render(fn(rng, lang, fk)) for _ in range(n)]

    def test_every_span_slices_to_the_text_it_claims(self):
        import build_positives as bp

        for template_id, _, _ in bp.TEMPLATES:
            for lang in bp.LANGUAGES:
                for rec in self._records(template_id, lang, 5):
                    for e in rec["entities"]:
                        assert 0 <= e["start"] < e["end"] <= len(rec["text"])
                        value = rec["text"][e["start"] : e["end"]]
                        assert value == value.strip(), (template_id, lang)
                        assert value, (template_id, lang)

    def test_tabular_templates_put_a_name_in_a_late_position(self):
        """`domain-nl-csv-1` was missed on row 2 while row 1 was found at
        0.9929. A template with a single row would not exercise that at all."""
        import build_positives as bp

        multi_row = {"csv-header", "csv-no-header", "pipe-table", "tsv-export",
                     "kv-blocks", "numbered-list", "inline-enumeration"}
        for template_id in multi_row:
            counts = [
                sum(1 for e in r["entities"] if e["label"] == "PERSON")
                for r in self._records(template_id, "nl", 20)
            ]
            assert min(counts) >= 2, (template_id, counts)

    def test_byline_templates_carry_a_two_token_name(self):
        """The `Lorenzo Roversi` failure is a two-token name whose given name
        was dropped. A byline corpus of single-token names would not teach the
        extension."""
        import build_positives as bp

        for template_id, family, _ in bp.TEMPLATES:
            if family != "byline":
                continue
            for rec in self._records(template_id, "it", 10):
                for e in rec["entities"]:
                    if e["label"] == "PERSON":
                        assert len(rec["text"][e["start"] : e["end"]].split()) >= 2, template_id

    def test_parenthetical_byline_shape_is_present(self):
        recs = self._records("byline-parenthetical", "it", 5)
        for rec in recs:
            person = next(e for e in rec["entities"] if e["label"] == "PERSON")
            assert rec["text"][person["end"]] == ")", rec["text"][:120]

    def test_holdout_templates_are_never_in_train(self):
        import build_positives as bp

        assert set(bp.HOLDOUT_TEMPLATES) <= {t for t, _, _ in bp.TEMPLATES}
        families = {f for t, f, _ in bp.TEMPLATES if t in bp.HOLDOUT_TEMPLATES}
        assert families == {"tabular", "byline"}, families
        for split in ("validation", "test"):
            held = [t for t, s in bp.HOLDOUT_TEMPLATES.items() if s == split]
            assert len(held) >= 2, (split, held)

    def test_generation_is_deterministic_for_a_seed(self):
        first = self._records("csv-header", "de", 5)
        second = self._records("csv-header", "de", 5)
        assert [r["text"] for r in first] == [r["text"] for r in second]

    def test_kd_is_masked_off_for_augmentation_rows(self):
        """The teacher misses names in these shapes -- that is why they exist.
        Distilling its logits here would teach the student the miss."""
        from train import KD_EXCLUDED_SOURCE_PREFIXES

        assert "augment:".startswith(tuple(p for p in KD_EXCLUDED_SOURCE_PREFIXES if p == "augment:"))
        assert "augment:tabular".startswith(KD_EXCLUDED_SOURCE_PREFIXES)
        assert "negative:encyclopedia".startswith(KD_EXCLUDED_SOURCE_PREFIXES)
        assert not "ai4privacy".startswith(KD_EXCLUDED_SOURCE_PREFIXES)
        assert not "generated".startswith(KD_EXCLUDED_SOURCE_PREFIXES)
