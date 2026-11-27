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

"""CPU-only. The tiled BERT layer and embedding against their untiled forms.

On CPU ``for_each_row_tile`` is a host loop, so this checks the per-trip math and
operand plumbing, not torch-spyre's loop lowering.
"""

import pytest
import torch
import torch.nn as nn
from transformers import BertConfig
from vllm.model_executor.models.bert import BertEmbedding, BertLayer
from vllm.utils.torch_utils import set_default_torch_dtype

from spyre_inference.models import _bert_tiled as bert_tiled
from spyre_inference.models._bert_tiled import (
    SpyreTiledBertLayer,
    _layer_refusal,
    install_tiled_encoder,
    tiled_embedding,
)
from spyre_inference.models._retype import retype
from spyre_inference.models._token_type import SpyreTokenTypeEmbedding
from spyre_inference.models.bert import SpyreBertEmbedding, split_bert_qkv
from spyre_inference.v1.attention.backends.spyre_encoder_attn import (
    _encoder_rect_kernel,
    encoder_grid,
    encoder_key_pad_mask,
)
from spyre_inference.v1.worker.encoder_dynamic import RectRange

EXTENT, HIDDEN, HEADS = 64, 128, 2
_CONFIG = BertConfig(
    hidden_size=HIDDEN,
    num_attention_heads=HEADS,
    intermediate_size=256,
    hidden_act="gelu",
    layer_norm_eps=1e-12,
    vocab_size=97,
    max_position_embeddings=EXTENT + 2,
    type_vocab_size=2,
)


@pytest.fixture(autouse=True)
def _clean_grid():
    encoder_grid().publish(None)
    yield
    encoder_grid().publish(None)


class _GridAttention(nn.Module):
    """The untiled reference: attention over the whole ``[width, extent]`` grid."""

    def __init__(self, width: int):
        super().__init__()
        self.width = width

    def forward(self, q, k, v):
        d = HIDDEN // HEADS
        a = _encoder_rect_kernel(
            q.view(-1, HEADS, d),
            k.view(-1, HEADS, d),
            v.view(-1, HEADS, d),
            encoder_grid().mask,
            d**-0.5,
            self.width,
            EXTENT,
            HEADS,
            HEADS,
            d,
        )
        return a.reshape(-1, HIDDEN)


def _randomize(module: nn.Module, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    for p in module.parameters():
        p.data.copy_(torch.randn(p.shape, generator=g) * 0.1)


@torch.no_grad()
def _layer() -> BertLayer:
    with set_default_torch_dtype(torch.float32):
        layer = BertLayer(_CONFIG)
    _randomize(layer, 0)
    for module in layer.modules():
        method = getattr(module, "quant_method", None)
        if method is not None and hasattr(module, "weight"):
            method.process_weights_after_loading(module)
    split_bert_qkv(nn.Sequential(layer))
    return layer


@torch.no_grad()
@pytest.mark.parametrize(
    ("seqs_per_trip", "lens"),
    [(1, [64]), (1, [64, 10, 1]), (2, [64, 33]), (2, [5, 64, 64, 17])],
    ids=["one", "three_ragged", "trip_of_two", "two_trips"],
)
def test_tiled_layer_matches_the_grid_layer(
    default_vllm_config, tp_group, monkeypatch, seqs_per_trip, lens
):
    trips = []
    real = bert_tiled.for_each_row_tile

    def counting(body, operands, dims, tile_size):
        trips.append(operands[0].shape[0] // tile_size)
        return real(body, operands, dims, tile_size)

    monkeypatch.setattr(bert_tiled, "for_each_row_tile", counting)
    layer = _layer()
    rect = RectRange(EXTENT, seqs_per_trip, max_width=4)
    width = rect.width_for(len(lens))
    kv_lens = lens + [1] * (width - len(lens))
    x = torch.randn(width * EXTENT, HIDDEN, generator=torch.Generator().manual_seed(1))

    layer.attention.self.attn = _GridAttention(width)
    encoder_grid().publish(encoder_key_pad_mask(EXTENT, kv_lens, torch.float32))
    want = layer(x)

    assert _layer_refusal(layer) is None
    tiled = retype(layer, SpyreTiledBertLayer)
    tiled.spyre_tile_pads = (False, False, False)
    row_mask = encoder_key_pad_mask(EXTENT, kv_lens, torch.float32).reshape(-1)
    encoder_grid().publish(None, row_mask, rect)
    got = tiled(x)
    assert trips == [width // seqs_per_trip]

    for seq, length in enumerate(lens):
        rows = slice(seq * EXTENT, seq * EXTENT + length)
        torch.testing.assert_close(got[rows], want[rows], rtol=1e-4, atol=1e-4)


@torch.no_grad()
def test_tiled_layer_runs_upstream_off_a_declared_step(default_vllm_config, tp_group):
    layer = _layer()
    layer.attention.self.attn = _GridAttention(2)
    x = torch.randn(2 * EXTENT, HIDDEN, generator=torch.Generator().manual_seed(1))
    encoder_grid().publish(encoder_key_pad_mask(EXTENT, [EXTENT, 3], torch.float32))
    want = layer(x)
    tiled = retype(layer, SpyreTiledBertLayer)
    torch.testing.assert_close(tiled(x), want, rtol=0, atol=0)


def test_a_fused_qkv_layer_is_refused(default_vllm_config, tp_group):
    with set_default_torch_dtype(torch.float32):
        layer = BertLayer(_CONFIG)
    assert _layer_refusal(layer) == "the QKV projection is not split"


def test_install_refuses_a_model_without_a_spyre_embedding(default_vllm_config, tp_group):
    model = nn.Module()
    model.layer = _layer()
    assert not install_tiled_encoder(model)
    assert not getattr(model, "spyre_dynamic_width", False)


@torch.no_grad()
@pytest.mark.parametrize("with_segments", [False, True], ids=["segment_0", "segments"])
def test_tiled_embedding_matches_the_untiled_one(default_vllm_config, tp_group, with_segments):
    with set_default_torch_dtype(torch.float32):
        emb = retype(BertEmbedding(_CONFIG), SpyreBertEmbedding)
    _randomize(emb, 2)
    rows = 3 * EXTENT
    g = torch.Generator().manual_seed(4)
    ids = torch.randint(0, _CONFIG.vocab_size, (rows,), generator=g)
    pos = torch.arange(EXTENT).repeat(3)
    segments = torch.randint(0, 2, (rows,), generator=g) if with_segments else None

    untiled = SpyreTokenTypeEmbedding._compiled_forward.__wrapped__  # ty: ignore
    want = untiled(emb, ids, pos, segments if with_segments else torch.zeros_like(ids))
    got = tiled_embedding(emb, ids, pos, segments, RectRange(EXTENT, 1, max_width=4))
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
