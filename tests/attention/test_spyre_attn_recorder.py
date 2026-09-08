# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the attention graph recorder.

CPU-only: these check that recording compiles a graph for every variant the
bucketer enumerates, and that a subsequent dispatch compiles nothing more. The
count comes from Dynamo's own ``unique_graphs`` counter, since Dynamo is what
decides whether a dispatch reuses a graph. The kernels run on CPU here (no
Spyre), which is enough to exercise dummy-arg construction and the guards.
"""

from unittest.mock import MagicMock

import pytest
import torch
from torch._dynamo.utils import counters
from vllm.config import CompilationMode, get_current_vllm_config

from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionImpl,
    SpyrePagedKVCache,
)
from spyre_inference.v1.attention.spyre_attn_bucketer import (
    SpyreAttnBucket,
    SpyreAttnBucketer,
)

pytestmark = pytest.mark.attention

NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_SIZE = 64
BLOCK_SIZE = 64
NUM_PAGES = 8


def compiles() -> int:
    """Graphs Dynamo has compiled so far, process-wide."""
    return counters["stats"]["unique_graphs"]


@pytest.fixture()
def impl(default_vllm_config):
    # Dynamo caches on the kernel's code object, shared by every impl in the
    # process, so an earlier test's graphs would hide a recorder that compiled none.
    torch._dynamo.reset()
    # The fixture's bare CompilationConfig leaves mode unset, which resolves to
    # eager. __init__ reads the mode to pick its kernel, so set it before building.
    get_current_vllm_config().compilation_config.mode = CompilationMode.STOCK_TORCH_COMPILE
    return SpyreAttentionImpl(
        num_heads=NUM_HEADS,
        head_size=HEAD_SIZE,
        scale=1.0 / (HEAD_SIZE**0.5),
        num_kv_heads=NUM_KV_HEADS,
        alibi_slopes=None,
        sliding_window=None,
    )


@pytest.fixture()
def kv_cache():
    shape = (NUM_PAGES, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    return SpyrePagedKVCache(
        k_pages=torch.zeros(shape, dtype=torch.float16),
        v_pages=torch.zeros(shape, dtype=torch.float16),
    )


def make_bucketer(max_model_len=256, max_num_batched_tokens=64):
    config = MagicMock()
    config.cache_config.block_size = BLOCK_SIZE
    config.model_config.max_model_len = max_model_len
    config.scheduler_config.max_num_batched_tokens = max_num_batched_tokens
    return SpyreAttnBucketer(config)


def _recordable(bucketer) -> list[SpyreAttnBucket]:
    return [v for v in bucketer.variants() if v.num_blocks <= NUM_PAGES]


def _dispatch(impl, kv_cache, num_blocks, padded_query_len):
    """Invoke the kernel the way a batch of this shape would."""
    impl._record_one(
        SpyreAttnBucket(num_blocks, padded_query_len),
        *kv_cache,
        BLOCK_SIZE,
        torch.device("cpu"),
    )


class TestRecordGraphs:
    def test_records_every_enumerated_variant(self, impl, kv_cache):
        bucketer = make_bucketer()

        recorded = impl.record_graphs(torch.device("cpu"), bucketer, kv_cache)

        assert recorded == len(_recordable(bucketer)) > 0

    def test_dispatch_after_recording_compiles_nothing(self, impl, kv_cache):
        """The acceptance criterion: no request compiles a new variant.

        Rounds sizes the way production does, so a drift between the two copies
        of the rules shows up here.
        """
        bucketer = make_bucketer()
        before = compiles()
        impl.record_graphs(torch.device("cpu"), bucketer, kv_cache)
        assert compiles() > before, "recording compiled nothing"

        snapshot = compiles()
        for kv_len in (1, 60, 64, 200, 256):
            for query_len in (1, 5, 32, 64):
                if query_len > kv_len:
                    continue
                padded_query_len = bucketer.find_query_bucket(query_len)
                num_blocks = bucketer._round_up(
                    (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE, bucketer.num_blocks_buckets
                )
                assert padded_query_len is not None and num_blocks is not None
                if num_blocks > NUM_PAGES:
                    continue
                _dispatch(impl, kv_cache, num_blocks, padded_query_len)

        assert compiles() == snapshot

    def test_re_recording_compiles_nothing(self, impl, kv_cache):
        bucketer = make_bucketer()
        first = impl.record_graphs(torch.device("cpu"), bucketer, kv_cache)

        snapshot = compiles()
        assert impl.record_graphs(torch.device("cpu"), bucketer, kv_cache) == first
        assert compiles() == snapshot

    def test_skips_variants_exceeding_the_page_allocation(self, impl, kv_cache):
        """Buckets sized from max_model_len can outrun a small KV cache."""
        bucketer = make_bucketer(max_model_len=4096)

        recorded = impl.record_graphs(torch.device("cpu"), bucketer, kv_cache)

        assert 0 < recorded == len(_recordable(bucketer)) < len(bucketer.variants())

    def test_real_metadata_dispatch_compiles_nothing(self, impl, kv_cache):
        """The acceptance criterion, driven from real builder metadata.

        Unlike ``test_dispatch_after_recording_compiles_nothing``, this builds
        metadata for unbucketed kv_lens through ``SpyreAttentionMetadataBuilder``
        and dispatches on the block counts ``build()`` actually produced.
        """
        from tests.attention.test_spyre_attn import _padded_mask_metadata

        # Built from the live config, not make_bucketer's narrower stand-in, so
        # this bucketer and the builder's derive from the same config.
        bucketer = SpyreAttnBucketer(get_current_vllm_config())
        impl.record_graphs(torch.device("cpu"), bucketer, kv_cache)

        snapshot = compiles()
        for query_len, kv_len in [(1, 1), (1, 65), (1, 200), (7, 65), (32, 300), (33, 300)]:
            metadata = _padded_mask_metadata(
                [(query_len, kv_len)],
                block_size=BLOCK_SIZE,
                num_query_heads=NUM_HEADS,
                num_kv_heads=NUM_KV_HEADS,
                head_size=HEAD_SIZE,
                max_num_blocks=NUM_PAGES,
            )
            assert metadata.padded_num_blocks is not None
            num_blocks = metadata.padded_num_blocks[0]
            assert num_blocks in bucketer.num_blocks_buckets, (
                f"kv_len={kv_len} produced an unrecorded block count {num_blocks}"
            )
            _dispatch(impl, kv_cache, num_blocks, metadata.aligned_max_query_len)

        assert compiles() == snapshot

    def test_eager_records_nothing(self, impl, kv_cache):
        impl._compile_attn = False
        snapshot = compiles()
        assert impl.record_graphs(torch.device("cpu"), make_bucketer(), kv_cache) == 0
        assert compiles() == snapshot

    def test_a_failing_variant_does_not_abort_the_pass(self, impl, kv_cache, monkeypatch):
        """One bad variant must not take down engine startup."""
        bucketer = make_bucketer()
        calls = {"n": 0}
        real = impl._record_one

        def flaky(bucket, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("synthetic lowering failure")
            return real(bucket, *args, **kwargs)

        monkeypatch.setattr(impl, "_record_one", flaky)
        recorded = impl.record_graphs(torch.device("cpu"), bucketer, kv_cache)

        assert recorded == calls["n"] - 1 == len(_recordable(bucketer)) - 1


class TestRecompileLimit:
    def test_limit_is_raised_during_recording_and_restored(self, impl, kv_cache):
        """Dynamo's accumulated limit is global, so more buckets than it allows would
        otherwise stop compiling partway through and fall back to eager."""
        bucketer = make_bucketer()
        before = torch._dynamo.config.accumulated_recompile_limit
        seen = []

        real = impl._record_one

        def spy(*args, **kwargs):
            seen.append(torch._dynamo.config.accumulated_recompile_limit)
            return real(*args, **kwargs)

        impl._record_one = spy
        impl.record_graphs(torch.device("cpu"), bucketer, kv_cache)

        assert seen and min(seen) >= len(bucketer.variants())
        assert torch._dynamo.config.accumulated_recompile_limit == before

    def test_limit_is_restored_even_when_recording_raises(self, impl, kv_cache, monkeypatch):
        before = torch._dynamo.config.accumulated_recompile_limit
        monkeypatch.setattr(
            impl, "_record_all", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        with pytest.raises(RuntimeError):
            impl.record_graphs(torch.device("cpu"), make_bucketer(), kv_cache)
        assert torch._dynamo.config.accumulated_recompile_limit == before
