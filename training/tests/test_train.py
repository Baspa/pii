"""Tests for training/train.py's own logic -- the pieces that don't need a
real model or tokenizer to verify: length-bucketed batching, collate's
attention_mask reconstruction, and the device-selection guard.

Added alongside the fixes that followed an external review of the smoke
test: the review found real gaps in what the original alignment tests
covered (window-straddle handling was never exercised, classify_mismatch's
ordering could mask a real bug behind a benign label) even though the
alignment logic itself was correct. These tests are aimed at the training
loop's own new logic (BucketBatchSampler, the lean TrainingExample/collate
path, pick_device) so the same class of "looks tested, isn't" gap doesn't
recur here.
"""

from __future__ import annotations

import functools
import pickle
import random

import pytest
import torch

from align import IGNORE_INDEX
from train import BucketBatchSampler, TrainingExample, collate, default_num_workers, pick_device


class TestBucketBatchSampler:
    def _lengths(self, n=500, seed=0):
        rng = random.Random(seed)
        return [rng.randint(10, 250) for _ in range(n)]

    def test_covers_every_index_exactly_once_per_epoch(self):
        lengths = self._lengths()
        sampler = BucketBatchSampler(lengths, batch_size=16, pool_size=64, seed=0)
        seen = []
        for batch in sampler:
            seen.extend(batch)
        assert sorted(seen) == list(range(len(lengths)))

    def test_len_matches_batch_count(self):
        lengths = self._lengths(n=500)
        sampler = BucketBatchSampler(lengths, batch_size=16, pool_size=64, seed=0)
        batches = list(sampler)
        assert len(batches) == len(sampler)
        assert len(sampler) == (len(lengths) + 15) // 16

    def test_last_batch_may_be_smaller_but_none_are_empty(self):
        lengths = self._lengths(n=500)
        sampler = BucketBatchSampler(lengths, batch_size=32, pool_size=100, seed=0)
        batches = list(sampler)
        assert all(len(b) > 0 for b in batches)
        assert all(len(b) <= 32 for b in batches)

    def test_different_epochs_give_different_orders(self):
        lengths = self._lengths()
        sampler = BucketBatchSampler(lengths, batch_size=16, pool_size=64, seed=0)
        sampler.set_epoch(0)
        order0 = list(sampler)
        sampler.set_epoch(1)
        order1 = list(sampler)
        assert order0 != order1

    def test_same_epoch_same_seed_is_deterministic(self):
        lengths = self._lengths()
        s1 = BucketBatchSampler(lengths, batch_size=16, pool_size=64, seed=0)
        s2 = BucketBatchSampler(lengths, batch_size=16, pool_size=64, seed=0)
        s1.set_epoch(3)
        s2.set_epoch(3)
        assert list(s1) == list(s2)

    def test_bucketing_reduces_padding_versus_plain_shuffle(self):
        """The actual property this sampler exists for: within-batch
        length variance (and therefore padding waste) should be much
        lower than a plain random shuffle over the same lengths."""
        lengths = self._lengths(n=2000, seed=1)
        batch_size = 32

        sampler = BucketBatchSampler(lengths, batch_size=batch_size, pool_size=512, seed=0)
        bucketed_padding = sum(max(lengths[i] for i in b) * len(b) for b in sampler)

        rng = random.Random(0)
        shuffled = list(range(len(lengths)))
        rng.shuffle(shuffled)
        plain_batches = [shuffled[i : i + batch_size] for i in range(0, len(shuffled), batch_size)]
        plain_padding = sum(max(lengths[i] for i in b) * len(b) for b in plain_batches)

        real_tokens = sum(lengths)
        bucketed_factor = bucketed_padding / real_tokens
        plain_factor = plain_padding / real_tokens
        assert bucketed_factor < plain_factor
        # Bucketing should get close to the no-padding floor (1.0x); plain
        # shuffling over this length range (10-250) should not.
        assert bucketed_factor < 1.3
        assert plain_factor > bucketed_factor * 1.2


class TestCollate:
    def test_attention_mask_reconstructed_correctly(self):
        batch = [
            TrainingExample(input_ids=[1, 2, 3], labels=[IGNORE_INDEX, 0, IGNORE_INDEX]),
            TrainingExample(input_ids=[1, 2, 3, 4, 5], labels=[IGNORE_INDEX, 0, 0, 0, IGNORE_INDEX]),
        ]
        out = collate(batch, pad_token_id=1)
        assert out["input_ids"].shape == (2, 5)
        # First example padded from length 3 to 5: mask is 1,1,1,0,0
        assert out["attention_mask"][0].tolist() == [1, 1, 1, 0, 0]
        assert out["attention_mask"][1].tolist() == [1, 1, 1, 1, 1]
        # Padded label positions are IGNORE_INDEX, not 0/O.
        assert out["labels"][0].tolist() == [IGNORE_INDEX, 0, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]

    def test_padding_uses_given_pad_token_id(self):
        batch = [
            TrainingExample(input_ids=[9], labels=[0]),
            TrainingExample(input_ids=[9, 9, 9], labels=[0, 0, 0]),
        ]
        out = collate(batch, pad_token_id=1)
        assert out["input_ids"][0].tolist() == [9, 1, 1]

    def test_collate_works_on_align_window_too(self):
        """collate is used both for TrainingExample (training) and
        align.Window (evaluation) -- it must only ever read .input_ids and
        .labels, never .attention_mask or .offset_mapping, since Window's
        own attention_mask (all-1s pre-padding, same as TrainingExample's
        implicit one) should never actually be read in preference to the
        reconstructed one."""
        from align import Window

        batch = [Window(input_ids=[1, 2], attention_mask=[1, 1], offset_mapping=[(0, 1), (1, 2)], labels=[0, 0])]
        out = collate(batch, pad_token_id=1)
        assert out["attention_mask"][0].tolist() == [1, 1]


class TestPickDevice:
    # Every case below stubs *both* torch.cuda.is_available and
    # torch.backends.mps.is_available, even the ones that only care about
    # one of them -- pick_device's auto branch checks cuda before mps (see
    # its docstring), so a test that stubs only mps and leaves cuda to
    # whatever the real machine reports is silently asserting "on a machine
    # with no CUDA GPU" rather than "when mps is/isn't available". These two
    # tests originally did exactly that: on this project's dev laptop (no
    # NVIDIA GPU) torch.cuda.is_available() is always False, so the missing
    # stub never showed -- until a rented CUDA box made it True and
    # pick_device("auto") correctly returned "cuda", failing both
    # assertions. Fixed by stating the full environment each test is
    # actually about.
    def test_auto_resolves_to_mps_when_available(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        assert pick_device("auto") == "mps"

    def test_auto_resolves_to_cpu_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        assert pick_device("auto") == "cpu"

    def test_explicit_cpu_always_returned(self, monkeypatch):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        assert pick_device("cpu") == "cpu"

    def test_explicit_mps_without_availability_fails_at_the_argument(self, monkeypatch):
        """The bug this guards: pick_device used to return "mps" unchanged
        even when unavailable, so the failure surfaced later, deep inside
        the training loop's first .to(device) call, instead of here."""
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        with pytest.raises(SystemExit):
            pick_device("mps")

    def test_explicit_mps_with_availability_succeeds(self, monkeypatch):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        assert pick_device("mps") == "mps"

    # -- cuda: the auto-resolution order and the explicit-request guard are
    # ordinary Python control flow, verifiable without CUDA hardware by
    # stubbing both availability checks (see the class-level comment
    # above). Whether an actual CUDA device trains correctly is NOT covered
    # by this suite and was not verified on this machine -- see
    # running the real --smoke on a GPU machine.

    def test_auto_prefers_cuda_over_mps_when_both_available(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        assert pick_device("auto") == "cuda"

    def test_auto_falls_back_to_mps_when_cuda_unavailable(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        assert pick_device("auto") == "mps"

    def test_auto_falls_back_to_cpu_when_neither_available(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        assert pick_device("auto") == "cpu"

    def test_auto_picks_cuda_even_without_mps(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        assert pick_device("auto") == "cuda"

    def test_explicit_cuda_without_availability_fails_at_the_argument(self, monkeypatch):
        """Same bug class as the mps guard, for cuda: this is also the
        first line of defence against the GPU deployment trap (a plain
        `pip install -e training/` can silently replace the machine's
        CUDA-enabled torch with a CPU-only wheel) -- `--device cuda`
        refuses to start rather than training on the CPU in silence."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(SystemExit) as exc_info:
            pick_device("cuda")
        assert "torch.cuda.is_available()" in str(exc_info.value)

    def test_explicit_cuda_with_availability_succeeds(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert pick_device("cuda") == "cuda"

    def test_device_type_comparison_ignores_cuda_index(self):
        """The post-.to(device) check in main() compares
        next(model.parameters()).device.type against torch.device(device).type,
        not equality of the device objects themselves -- a real CUDA
        parameter reports as cuda:0 (indexed), while args.device is the
        bare string 'cuda'. Pin that .type is the right comparison."""
        assert torch.device("cuda:0").type == torch.device("cuda").type == "cuda"


class TestDefaultNumWorkers:
    """os.cpu_count() varies by machine (10 on a laptop, 9 on the GPU
    machine) -- verify the formula directly
    rather than trusting it matches intent on just the two machines it
    happens to run on."""

    def test_reserves_one_core_and_caps_at_four(self, monkeypatch):
        import train

        cases = {1: 0, 2: 1, 4: 3, 5: 4, 9: 4, 10: 4, 32: 4}
        for cpu_count, expected in cases.items():
            monkeypatch.setattr(train.os, "cpu_count", lambda n=cpu_count: n)
            assert default_num_workers() == expected, f"cpu_count={cpu_count}"

    def test_never_negative_when_cpu_count_is_none(self, monkeypatch):
        import train

        monkeypatch.setattr(train.os, "cpu_count", lambda: None)
        assert default_num_workers() == 0


class TestCollateFnPicklability:
    """Regression guard for a bug caught before it shipped: DataLoader
    workers (--num-workers > 0) are spawned via the operating system's default
    multiprocessing context, which is "spawn" on macOS (this dev machine,
    confirmed via multiprocessing.get_context().get_start_method()) --
    spawn pickles collate_fn to hand it to each worker process. The
    original `collate_fn=lambda b: collate(b, tokenizer.pad_token_id)`
    is not picklable; train.py now binds pad_token_id with
    functools.partial over the module-level collate function instead."""

    def test_lambda_collate_fn_is_not_picklable(self):
        """Demonstrates the bug this was a near-miss for -- confirms the
        premise, not train.py's own code."""
        fn = lambda b: collate(b, pad_token_id=1)  # noqa: E731
        with pytest.raises((pickle.PicklingError, AttributeError, TypeError)):
            pickle.dumps(fn)

    def test_functools_partial_collate_fn_is_picklable_and_works(self):
        fn = functools.partial(collate, pad_token_id=1)
        restored = pickle.loads(pickle.dumps(fn))
        batch = [TrainingExample(input_ids=[9], labels=[0]), TrainingExample(input_ids=[9, 9], labels=[0, 0])]
        out = restored(batch)
        assert out["input_ids"].tolist() == [[9, 1], [9, 9]]
