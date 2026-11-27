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

"""The declared-width encoder (#1525). CPU-only, except the last test, which needs a card."""

import time
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from spyre_testing_plugin.pytest_plugin import spyre_available
from vllm.model_executor.layers.pooler.seqwise.methods import CLSPool, MeanPool

from spyre_inference import envs
from spyre_inference.models import _dynamic_width_encoder as dwe
from spyre_inference.v1.attention.backends.spyre_encoder_attn import encoder_key_pad_mask
from spyre_inference.v1.worker.spyre.pool.pooling_runner import SpyrePoolingRunner

VOCAB, PAD_ID, EPS = 96, 1, 1e-5


@dataclass(frozen=True)
class _Dims:
    hidden: int
    heads: int
    intermediate: int

    @property
    def scale(self) -> float:
        return (self.hidden // self.heads) ** -0.5


SMALL = _Dims(hidden=128, heads=2, intermediate=256)
GRANITE_125M = _Dims(hidden=768, heads=12, intermediate=3072)


def _layer_params(gen, dims, dtype=torch.float32):
    """HF-order weights ``[out, in]``, as the checkpoint holds them."""
    h, i = dims.hidden, dims.intermediate

    def w(*shape):
        return (torch.randn(*shape, generator=gen) * 0.05).to(dtype)

    ones, zeros = torch.ones(h, dtype=dtype), torch.zeros(h, dtype=dtype)
    return [
        w(h, h), w(h),  # q
        w(h, h), w(h),  # k
        w(h, h), w(h),  # v
        w(h, h), w(h),  # attention output
        ones + w(h), zeros + w(h),  # attention layer norm
        w(i, h), w(i),  # up
        w(h, i), w(h),  # down
        ones + w(h), zeros + w(h),  # output layer norm
    ]  # fmt: skip


def _transposed(params):
    """The runner's form: each linear weight ``[in, out]``."""
    linear = {0, 2, 4, 6, 10, 12}
    return [p.t().contiguous() if i in linear else p for i, p in enumerate(params)]


def _reference_layer(x, params, mask, heads):
    """HF ``BertLayer`` over ``[B, L, H]`` with an additive ``[B, 1, 1, L]`` key pad."""
    wq, bq, wk, bk, wv, bv, wo, bo, ln1_w, ln1_b, w1, b1, w2, b2, ln2_w, ln2_b = params
    b, length, hidden = x.shape
    d = hidden // heads

    def split(t):
        return t.reshape(b, length, heads, d).transpose(1, 2)

    q, k, v = split(F.linear(x, wq, bq)), split(F.linear(x, wk, bk)), split(F.linear(x, wv, bv))
    context = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=d**-0.5)
    context = context.transpose(1, 2).reshape(b, length, hidden)
    h = F.layer_norm(x + F.linear(context, wo, bo), (hidden,), ln1_w, ln1_b, EPS)
    ff = F.linear(F.gelu(F.linear(h, w1, b1)), w2, b2)
    return F.layer_norm(h + ff, (hidden,), ln2_w, ln2_b, EPS)


def _reference_cls(ids, lens, table, layers, heads):
    """The CLS row of each sequence through the whole stack, in float32 on the host."""
    word, pos_type, ln_w, ln_b = (t.float() for t in table)
    x = F.layer_norm(word[ids.long()] + pos_type, (word.shape[1],), ln_w, ln_b, EPS)
    mask = encoder_key_pad_mask(ids.shape[1], lens, torch.float32)
    for params in layers:
        x = _reference_layer(x, [p.float() for p in params], mask, heads)
    return x[:, 0]


def _batch(gen, length, max_width):
    """Ids and lengths of ``max_width`` sequences: the first full, the rest shorter."""
    lens = [length, *torch.randint(1, length + 1, (max_width - 1,), generator=gen).tolist()]
    ids = torch.randint(2, VOCAB, (max_width, length), generator=gen, dtype=torch.int32)
    for s, n in enumerate(lens):
        ids[s, n:] = PAD_ID
    return ids, lens


def _table(gen, length, hidden, dtype=torch.float32):
    """Word table, position-plus-segment rows and the embedding layer norm."""

    def w(*shape):
        return (torch.randn(*shape, generator=gen) * 0.05).to(dtype)

    ones, zeros = torch.ones(hidden, dtype=dtype), torch.zeros(hidden, dtype=dtype)
    return [w(VOCAB, hidden), w(length, hidden), ones, zeros]


def test_layer_tile_is_the_bert_layer_on_transposed_weights():
    gen = torch.Generator().manual_seed(0)
    length = 64
    params = _layer_params(gen, SMALL)
    x = torch.randn(1, length, SMALL.hidden, generator=gen)
    mask = encoder_key_pad_mask(length, [37], torch.float32)

    got = dwe.layer_tile(x, mask, _transposed(params), SMALL.heads, SMALL.scale, EPS)

    want = _reference_layer(x, params, mask, SMALL.heads)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


@dataclass
class _Slice:
    dim: int
    extent: int


def _python_for_each_tile(body, operands, *, dims, tile_size, out_dim):
    """``for_each_tile``'s mapping form as a Python loop: tiles out, joined on ``out_dim``."""

    def step(d):
        return (d.dim, d.extent) if isinstance(d, _Slice) else (d, tile_size)

    (trips,) = {o.shape[step(d)[0]] // step(d)[1] for o, d in zip(operands, dims) if d is not None}
    outs = []
    for i in range(trips):
        tiles = [
            o if d is None else o.narrow(step(d)[0], i * step(d)[1], step(d)[1])
            for o, d in zip(operands, dims)
        ]
        outs.append(body(None, tiles)[1])
    return None, torch.cat(outs, out_dim)


@pytest.fixture()
def python_loops(monkeypatch):
    from torch_spyre._inductor import wsr

    monkeypatch.setattr(wsr, "for_each_tile", _python_for_each_tile, raising=False)
    monkeypatch.setattr(wsr, "Slice", _Slice, raising=False)


def test_loops_pool_every_width_like_the_grid(python_loops):
    gen = torch.Generator().manual_seed(1)
    length, max_width = 64, 4
    ids, lens = _batch(gen, length, max_width)
    table = _table(gen, length, SMALL.hidden)
    layers = [_layer_params(gen, SMALL) for _ in range(2)]
    embed = dwe.embed_loop(length, max_width, EPS)
    layer = dwe.layer_loop(length, max_width, SMALL.heads, SMALL.scale, EPS)
    cls = dwe.cls_loop(length, max_width)
    reference = _reference_cls(ids, lens, table, layers, SMALL.heads)

    for width in range(1, max_width + 1):
        mask = encoder_key_pad_mask(length, lens[:width], torch.float32).reshape(width, length)
        x = embed(ids[:width], *table)
        for params in layers:
            x = layer(x, mask, *_transposed(params))
        out = cls(x)
        assert out.shape == (width * dwe.CLS_ROWS, SMALL.hidden)
        torch.testing.assert_close(
            dwe.cls_rows(out, width), reference[:width], atol=1e-4, rtol=1e-4
        )


def _encoder(widths, ladder):
    encoder = object.__new__(dwe.DynamicWidthEncoder)
    encoder.widths = widths
    encoder.ladder = ladder
    encoder.pad_token_id = PAD_ID
    return encoder


@pytest.mark.parametrize(
    ("num_seqs", "max_len", "extent"),
    [
        (4, 512, 512),
        (1, 300, 512),
        # 512 would hold it, but its shortest ladder length is 256, which stays static.
        (2, 200, None),
        (5, 512, None),
        (2, 1024, 1024),
        (3, 1024, None),
        (1, 2048, None),
    ],
)
def test_extent_is_the_shortest_ladder_length_holding_the_batch(num_seqs, max_len, extent):
    encoder = _encoder({512: 4, 1024: 2}, (64, 128, 256, 512, 1024))
    assert encoder.extent_for(num_seqs, max_len) == extent


def test_input_ids_are_right_padded_rows(monkeypatch):
    encoder = _encoder({64: 4}, (64,))
    monkeypatch.setattr(encoder, "_declared", lambda host, length: host)
    plan = dwe.DynamicWidthPlan(extent=64, query_lens=[3, 2], starts=[0, 3], mask=torch.zeros(1))
    packed = torch.tensor([10, 11, 12, 20, 21, 0, 0, 0], dtype=torch.int32)

    encoder.set_input_ids(plan, packed)

    assert plan.input_ids is not None
    assert plan.input_ids.dtype == torch.int32
    assert plan.input_ids.shape == (2, 64)
    assert plan.input_ids[:, :4].tolist() == [[10, 11, 12, PAD_ID], [20, 21, PAD_ID, PAD_ID]]
    assert (plan.input_ids[:, 4:] == PAD_ID).all()


def test_cls_rows_take_the_first_of_each_sequence():
    rows = torch.arange(3 * dwe.CLS_ROWS * 2, dtype=torch.float32).reshape(3 * dwe.CLS_ROWS, 2)
    want = rows[:: dwe.CLS_ROWS]
    assert torch.equal(dwe.cls_rows(rows, 3), want)


def test_only_cls_pooled_tasks_take_the_loops():
    class _Pooler(nn.Module):
        def __init__(self, pooling):
            super().__init__()
            self.pooling = pooling

    dispatch = SimpleNamespace(
        poolers_by_task={
            "embed": _Pooler(CLSPool()),
            "classify": _Pooler(MeanPool()),
            "token_embed": nn.Identity(),
        }
    )
    assert dwe.cls_pooled_tasks(dispatch) == {"embed"}
    assert dwe.cls_pooled_tasks(None) == frozenset()


def test_pools_only_needs_every_request_in_the_tasks():
    runner = object.__new__(SpyrePoolingRunner)
    runner.pooling_params = {
        0: SimpleNamespace(task="embed"),
        2: SimpleNamespace(task="token_embed"),
    }
    assert runner.pools_only(SimpleNamespace(idx_mapping_np=np.array([0])), {"embed"})
    assert not runner.pools_only(SimpleNamespace(idx_mapping_np=np.array([0, 2])), {"embed"})
    # A request without pooling params, such as a dummy one, keeps the static path.
    assert not runner.pools_only(SimpleNamespace(idx_mapping_np=np.array([1])), {"embed"})


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("SPYRE_ENCODER_DYNAMIC_WIDTH_MIN_LEN", raising=False)
    envs.clear_env_cache()
    try:
        assert (
            dwe.build(None, nn.Module(), torch.device("cpu"), torch.float16, pad_token_id=1) is None
        )
    finally:
        envs.clear_env_cache()


@pytest.mark.parametrize(("dims", "length", "max_width"), [(SMALL, 64, 8), (GRANITE_125M, 512, 4)])
def test_declared_loops_compile_once_per_length_on_the_card(dims, length, max_width):
    """The production loops, upload and compile wrapper on Spyre, against the host grid."""
    if not spyre_available():
        pytest.skip("Spyre device not available")
    from torch._dynamo.utils import counters
    from torch_spyre._inductor import config as ts_config

    device, dtype = torch.device("spyre"), torch.float16
    gen = torch.Generator().manual_seed(2)
    ids, lens = _batch(gen, length, max_width)
    table_cpu = _table(gen, length, dims.hidden, dtype)
    layers_cpu = [_layer_params(gen, dims, dtype) for _ in range(2)]
    reference = _reference_cls(ids, lens, table_cpu, layers_cpu, dims.heads)
    table = [t.to(device) for t in table_cpu]
    layers = [[p.to(device) for p in _transposed(params)] for params in layers_cpu]
    mask_cpu = encoder_key_pad_mask(length, lens, dtype).reshape(max_width, length)

    torch._dynamo.reset()
    counters.clear()
    embed = dwe.compile_loop(dwe.embed_loop(length, max_width, EPS), "test embed")
    layer = dwe.compile_loop(
        dwe.layer_loop(length, max_width, dims.heads, dims.scale, EPS), "test layer"
    )
    cls = dwe.compile_loop(dwe.cls_loop(length, max_width), "test cls")
    with ts_config.patch(min_reserved_dim_size=1):
        # One sequence first, as warmup compiles it, then the widest, then the rest.
        for width in dict.fromkeys([1, max_width, *range(2, max_width)]):
            start = time.perf_counter()
            x = embed(dwe.upload_declared(ids[:width], max_width, device), *table)
            mask = dwe.upload_declared(mask_cpu[:width], max_width, device)
            for params in layers:
                x = layer(x, mask, *params)
            out = dwe.cls_rows(cls(x), width).float()
            print(f"\n[{length}] width {width}: {time.perf_counter() - start:.2f}s")
            torch.testing.assert_close(out, reference[:width], atol=1e-1, rtol=1e-1)
    assert counters["stats"]["unique_graphs"] == 3, "embed, layer and cls should compile once each"
