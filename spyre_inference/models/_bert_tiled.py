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

"""A BERT encoder whose rows are a declared batch width (``SPYRE_ENCODER_DYNAMIC_WIDTH``).

On a declared-width step every op on the grid's rows runs inside one loop over whole
sequences, since torch-spyre compiles a declared dim only that way: the embedding in
row tiles, each layer in trips of ``S`` sequences, attention within the trip. Weights go
in as loop operands, as in every other ``for_each_tile`` body here. Off such a step
(ragged, or a length with a static rectangle) each module runs its upstream forward.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.models.bert import BertEncoder, BertLayer, BertSelfAttention

from spyre_inference.custom_ops.linear import SpyreUnquantizedLinearMethod, spyre_linear_t
from spyre_inference.models._retype import retype
from spyre_inference.v1.attention.backends.spyre_encoder_attn import (
    ENCODER_LEN_ALIGNMENT,
    _encoder_rect_kernel,
    encoder_grid,
)
from spyre_inference.v1.worker.encoder_dynamic import (
    RectRange,
    check_rows,
    for_each_row_tile,
    mark_reserved_rows,
)

if TYPE_CHECKING:
    from spyre_inference.models._token_type import SpyreTokenTypeEmbedding

logger = init_logger(__name__)

# Rows per trip of the embedding loop. It only has to divide every legal row count.
_EMBED_TILE_ROWS = 512


def _plain_linear_refusal(name: str, dense: nn.Module) -> str | None:
    method = getattr(dense, "quant_method", None)
    if (
        not isinstance(method, SpyreUnquantizedLinearMethod)
        or getattr(dense, "spyre_row_padding", 0)
        or getattr(dense, "bias", None) is None
        or getattr(dense, "skip_bias_add", False)
    ):
        return f"{name} is not an unpadded, biased Spyre linear"
    return None


def _layer_refusal(layer: nn.Module) -> str | None:
    if type(layer) is not BertLayer:
        return f"{type(layer).__name__} is not a BertLayer"
    sa = layer.attention.self
    if not isinstance(sa, BertSelfAttention):
        return "self-attention is not a BertSelfAttention"
    proj = sa.qkv_proj
    if not all(
        hasattr(proj, f"{n}_weight_t") and getattr(proj, f"{n}_bias", None) is not None
        for n in "qkv"
    ):
        return "the QKV projection is not split"
    if sa.head_dim % ENCODER_LEN_ALIGNMENT:
        return f"head_dim {sa.head_dim} is not a whole stick"
    for name, dense in (
        ("attention.output.dense", layer.attention.output.dense),
        ("intermediate.dense", layer.intermediate.dense),
        ("output.dense", layer.output.dense),
    ):
        refusal = _plain_linear_refusal(name, dense)
        if refusal:
            return refusal
    return None


def _pads(dense: nn.Module) -> bool:
    return bool(dense.quant_method._pad_short_rows(dense))


class SpyreTiledBertLayer(BertLayer):
    """``BertLayer`` running a declared-width step as one loop over its sequences."""

    spyre_tile_pads: tuple[bool, bool, bool]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        grid = encoder_grid()
        rect, row_mask = grid.rect_range, grid.row_mask
        if rect is None or row_mask is None:
            return super().forward(hidden_states)
        check_rows(hidden_states.shape[0], rect)

        sa = self.attention.self
        proj = sa.qkv_proj
        out1, inter, out2 = self.attention.output, self.intermediate, self.output
        ln1, ln2, act = out1.LayerNorm, out2.LayerNorm, inter.intermediate_act_fn
        extent, seqs = rect.extent, rect.seqs_per_trip
        heads, kv_heads, head_dim = sa.num_heads, sa.num_kv_heads, sa.head_dim
        scale = sa.scaling
        pad1, pad2, pad3 = self.spyre_tile_pads

        def body(h, m, qw, qb, kw, kb, vw, vb, aw, ab, n1w, n1b, iw, ib, ow, ob, n2w, n2b):
            attn = _encoder_rect_kernel(
                spyre_linear_t(h, qw, qb).view(-1, heads, head_dim),
                spyre_linear_t(h, kw, kb).view(-1, kv_heads, head_dim),
                spyre_linear_t(h, vw, vb).view(-1, kv_heads, head_dim),
                m.view(seqs, 1, 1, extent),
                scale,
                seqs,
                extent,
                heads,
                kv_heads,
                head_dim,
            ).reshape(-1, heads * head_dim)
            x = spyre_linear_t(attn, aw, ab, pad_rows=pad1) + h
            x = F.layer_norm(x, ln1.normalized_shape, n1w, n1b, ln1.eps)
            y = act(spyre_linear_t(x, iw, ib, pad_rows=pad2))
            y = spyre_linear_t(y, ow, ob, pad_rows=pad3) + x
            return F.layer_norm(y, ln2.normalized_shape, n2w, n2b, ln2.eps)

        weights = (
            proj.q_weight_t,
            proj.q_bias,
            proj.k_weight_t,
            proj.k_bias,
            proj.v_weight_t,
            proj.v_bias,
            out1.dense.weight,
            out1.dense.bias,
            ln1.weight,
            ln1.bias,
            inter.dense.weight,
            inter.dense.bias,
            out2.dense.weight,
            out2.dense.bias,
            ln2.weight,
            ln2.bias,
        )
        return for_each_row_tile(
            body,
            (hidden_states, row_mask, *weights),
            (0, 0, *([None] * len(weights))),
            rect.granule,
        )


class SpyreMarkingBertEncoder(BertEncoder):
    """``BertEncoder`` re-marking the declared rows before every compiled block."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layer:
            mark_reserved_rows(hidden_states)
            hidden_states = layer(hidden_states)
        return hidden_states


def tiled_embedding(
    emb: SpyreTokenTypeEmbedding,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    token_type_ids: torch.Tensor | None,
    rect: RectRange,
) -> torch.Tensor:
    """The embedding sum and LayerNorm over declared rows, ``_EMBED_TILE_ROWS`` at a time.

    Without segment ids every row is segment 0, so its table row is added directly.
    """
    check_rows(input_ids.shape[0], rect)
    ln = emb.LayerNorm
    word, pos, segment = (
        emb.word_embeddings.weight,
        emb.position_embeddings.weight,
        emb.token_type_embeddings.weight,
    )
    tile = min(rect.granule, _EMBED_TILE_ROWS)

    def norm(e, lw, lb):
        return F.layer_norm(e, ln.normalized_shape, lw, lb, ln.eps)

    if token_type_ids is None:

        def body(ids, p, ww, pw, sw, lw, lb):
            return norm(F.embedding(ids, ww) + sw.narrow(0, 0, 1) + F.embedding(p, pw), lw, lb)

        operands = (input_ids, position_ids, word, pos, segment, ln.weight, ln.bias)
        dims = (0, 0, None, None, None, None, None)
    else:

        def body(ids, p, s, ww, pw, sw, lw, lb):
            return norm(F.embedding(ids, ww) + F.embedding(s, sw) + F.embedding(p, pw), lw, lb)

        operands = (input_ids, position_ids, token_type_ids, word, pos, segment, ln.weight, ln.bias)
        dims = (0, 0, 0, None, None, None, None, None)
    return for_each_row_tile(body, operands, dims, tile)


def install_tiled_encoder(model: nn.Module) -> bool:
    """Retype the BERT encoder of ``model`` for declared-width steps; False if any part
    cannot run tiled, which leaves the model on per-rectangle graphs."""
    from spyre_inference.models._token_type import SpyreTokenTypeEmbedding

    if getattr(model, "spyre_dynamic_width", False):
        return True
    encoders = [m for m in model.modules() if type(m) is BertEncoder]
    layers = [layer for encoder in encoders for layer in encoder.layer]
    embeddings = [m for m in model.modules() if isinstance(m, SpyreTokenTypeEmbedding)]
    refusal: str | None = None
    if get_tensor_model_parallel_world_size() > 1:
        refusal = "tensor parallelism is above 1"
    elif not layers or len(embeddings) != 1:
        refusal = "no single BERT encoder with a Spyre token-type embedding"
    else:
        refusal = next((r for r in map(_layer_refusal, layers) if r), None)
    if refusal:
        logger.warning(
            "SPYRE_ENCODER_DYNAMIC_WIDTH is set but %s keeps one graph per rectangle: %s.",
            type(model).__name__,
            refusal,
        )
        return False

    for layer in layers:
        tiled = retype(layer, SpyreTiledBertLayer)
        tiled.spyre_tile_pads = (
            _pads(tiled.attention.output.dense),
            _pads(tiled.intermediate.dense),
            _pads(tiled.output.dense),
        )
    for encoder in encoders:
        retype(encoder, SpyreMarkingBertEncoder)
    embeddings[0].spyre_tiled = True
    model.spyre_dynamic_width = True
    logger.info(
        "Encoder of %s runs declared-width rectangles: one block graph per length over %d layers.",
        type(model).__name__,
        len(layers),
    )
    return True
