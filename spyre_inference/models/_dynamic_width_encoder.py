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

"""An encoder step compiled once per sequence length, its batch width declared (#1525).

The static path compiles each BERT layer once per ``(length, width)`` rectangle. Here
the token ids and key-pad mask are uploaded as ``[width, length]`` with dim 0 declared
to torch-spyre (``.to(..., dynamic=...)``), so one graph per length serves every width
up to that length's widest rectangle. torch-spyre refuses a plain op over a declared
dimension, so the embedding, each layer and the CLS pick are each one ``for_each_tile``
loop with one sequence per trip and the weights as untiled operands.

What the device loop accepts, as measured on the card
(``tests/probes/test_encoder_dynamic_width_probe.py``):

* every operand stepped per sequence has its sequence axis outermost
  (``row_outermost_layout``), and the mask is ``[width, length]`` rather than the
  rank-4 ``[width, 1, 1, length]``;
* each operand's per-trip address step divides the loop step, which is one sequence of
  the layer output. The CLS loop therefore keeps ``CLS_ROWS`` rows of each sequence,
  and the pooler takes the first of each;
* layer norm is the arithmetic ``_layer_norm_kernel``, attention is SDPA, and no body
  returns its input unchanged.

Serving: ``SpyreEncoderOnlyModelState`` builds a ``DynamicWidthPlan`` for a step whose
requests are all CLS-pooled, whole prompts, and fit a declared length; this module
publishes it, and ``install``'s ``BertModel.forward`` runs the loops instead of the
embedding and the layer blocks. The pooling runner reads ``CLS_ROWS`` rows per request.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import CompilationMode
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.pooler.seqwise.methods import CLSPool
from vllm.model_executor.models.bert import BertLayer, BertModel

from spyre_inference import envs
from spyre_inference.custom_ops.layer_norm import _layer_norm_kernel
from spyre_inference.custom_ops.linear import SpyreUnquantizedLinearMethod
from spyre_inference.custom_ops.utils import convert, row_outermost_layout
from spyre_inference.models._token_type import SpyreTokenTypeEmbedding
from spyre_inference.models.roberta import roberta_position_delta
from spyre_inference.v1.attention.backends.spyre_encoder_attn import encoder_key_pad_mask
from spyre_inference.v1.worker import compile_guard
from spyre_inference.v1.worker.compile_granularity import compile_granularity
from spyre_inference.v1.worker.spyre_shape_bucketer import encoder_shape_tables

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.tasks import PoolingTask

logger = init_logger(__name__)

# The CLS loop steps a whole sequence of the layer output per trip, and every operand's
# per-trip step must divide it. One output row is one 128-byte stick; eight are 1024
# bytes, which divides every length the ladder declares.
CLS_ROWS = 8

_LINEAR_NAMES = ("q", "k", "v")


@dataclass(eq=False)
class DynamicWidthPlan:
    """One step on the declared-width path: ``len(query_lens)`` sequences of ``extent``."""

    extent: int
    query_lens: list[int]
    starts: list[int]
    """Each request's first row in the packed body."""
    mask: torch.Tensor
    """``[width, extent]`` additive key pad on the device, dim 0 declared."""
    input_ids: torch.Tensor | None = None
    """``[width, extent]`` int32 token ids on the device, dim 0 declared. Set by
    ``prepare_inputs``, which runs after the plan is built."""


class _PublishedPlan:
    """This step's plan for the ``BertModel.forward`` override; ``None`` off the path."""

    def __init__(self) -> None:
        self.plan: DynamicWidthPlan | None = None


_published = _PublishedPlan()


def publish_dynamic_width_plan(plan: DynamicWidthPlan | None) -> None:
    _published.plan = plan


def published_dynamic_width_plan() -> DynamicWidthPlan | None:
    return _published.plan


def _check_rows(n: Any, granule: int, max_width: int) -> None:
    torch._check(n >= granule)
    torch._check(n <= max_width * granule)
    torch._check(n % granule == 0)


def embed_loop(length: int, max_width: int, eps: float) -> Callable[..., torch.Tensor]:
    """``[B, L]`` ids to ``[B * L, H]``: word + (position + segment 0), normed."""
    from torch_spyre._inductor.wsr import for_each_tile

    def embed(ids, word, pos_type, ln_w, ln_b):
        _check_rows(ids.shape[0], 1, max_width)

        def body(_carry, tiles):
            ids_tile, word, pos_type, ln_w, ln_b = tiles
            x = word.index_select(0, ids_tile.reshape(length)) + pos_type
            return None, _layer_norm_kernel(x, ln_w, ln_b, eps)

        operands = (ids, word, pos_type, ln_w, ln_b)
        dims = (0, None, None, None, None)
        return for_each_tile(body, operands, dims=dims, tile_size=1, out_dim=0)[1]

    return embed


def layer_tile(
    x: torch.Tensor,
    mask: torch.Tensor,
    weights: Sequence[torch.Tensor],
    num_heads: int,
    scale: float,
    eps: float,
) -> torch.Tensor:
    """One post-norm BERT layer over ``x`` of ``[1, L, H]``, weights ``[in, out]``.

    ``mask`` is the ``[1, 1, 1, L]`` additive key pad.
    """
    wq, bq, wk, bk, wv, bv, wo, bo, ln1_w, ln1_b, w1, b1, w2, b2, ln2_w, ln2_b = weights
    _, length, hidden = x.shape
    head_size = hidden // num_heads

    def heads(t):
        return t.reshape(1, length, num_heads, head_size).transpose(1, 2)

    q = heads(torch.matmul(x, wq) + bq)
    k = heads(torch.matmul(x, wk) + bk)
    v = heads(torch.matmul(x, wv) + bv)
    # SDPA, as ``_encoder_sdpa_kernel`` does: a hand-written q @ k^T over these head
    # views has no layout inside a for_each_tile body.
    context = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
    context = context.transpose(1, 2).reshape(1, length, hidden)
    h = _layer_norm_kernel(x + torch.matmul(context, wo) + bo, ln1_w, ln1_b, eps)
    ff = torch.matmul(F.gelu(torch.matmul(h, w1) + b1), w2) + b2
    return _layer_norm_kernel(h + ff, ln2_w, ln2_b, eps)


def layer_loop(
    length: int, max_width: int, num_heads: int, scale: float, eps: float
) -> Callable[..., torch.Tensor]:
    """``[B * L, H]`` rows and a ``[B, L]`` mask through one layer, a sequence per trip."""
    from torch_spyre._inductor.wsr import Slice, for_each_tile

    def layer(rows, mask, *weights):
        _check_rows(rows.shape[0], length, max_width)
        hidden = rows.shape[1]

        def body(_carry, tiles):
            tile, mask_tile, *weights = tiles
            x = layer_tile(
                tile.reshape(1, length, hidden),
                mask_tile.reshape(1, 1, 1, length),
                weights,
                num_heads,
                scale,
                eps,
            )
            return None, x.reshape(length, hidden)

        dims = (0, Slice(0, 1)) + (None,) * len(weights)
        operands = (rows, mask, *weights)
        return for_each_tile(body, operands, dims=dims, tile_size=length, out_dim=0)[1]

    return layer


def cls_loop(length: int, max_width: int) -> Callable[..., torch.Tensor]:
    """``[B * L, H]`` to ``[B * CLS_ROWS, H]``: each sequence's first rows, CLS first."""
    from torch_spyre._inductor.wsr import for_each_tile

    def cls(rows):
        _check_rows(rows.shape[0], length, max_width)

        def body(_carry, tiles):
            (tile,) = tiles
            return None, tile[0:CLS_ROWS].clone()

        return for_each_tile(body, (rows,), dims=(0,), tile_size=length, out_dim=0)[1]

    return cls


def cls_rows(hidden_states: torch.Tensor, num_seqs: int) -> torch.Tensor:
    """The host ``[num_seqs, H]`` CLS rows of the CLS loop's ``[num_seqs * CLS_ROWS, H]``."""
    host = convert(hidden_states, "cpu")
    return host.reshape(num_seqs, CLS_ROWS, host.shape[-1])[:, 0].contiguous()


def upload_declared(host: torch.Tensor, width: int, device: torch.device) -> torch.Tensor:
    """``host`` ``[n, L]`` on ``device`` with dim 0 declared ``1..width``, rows outermost."""
    n, length = host.shape
    # The upload reserves the widest width; ``resize_`` then sets this step's.
    padded = host if n == width else torch.cat([host, host.new_zeros(width - n, length)])
    out = convert(
        padded,
        device,
        device_layout=row_outermost_layout((width, length), host.dtype),
        dynamic={0: {"min": 1, "max": width, "granularity": 1}},
    )
    return out.resize_((n, length))


@dataclass
class _Loops:
    embed: Callable[..., torch.Tensor]
    layer: Callable[..., torch.Tensor]
    cls: Callable[..., torch.Tensor]


def compile_loop(fn: Callable[..., torch.Tensor], label: str) -> Callable[..., torch.Tensor]:
    from torch.fx.experimental import _config as fx_config

    compile_guard.watch(fn, label)
    compiled = torch.compile(fn, dynamic=False, fullgraph=True)

    def run(*args: torch.Tensor) -> torch.Tensor:
        # Dynamo traces a size of 1 as a constant, so a trace at another width would
        # guard the count at 2 or more; size-oblivious tracing leaves only the checks.
        with fx_config.patch(backed_size_oblivious=True):
            return compiled(*args)

    return run


class DynamicWidthEncoder:
    """The declared-width encoder of one ``BertModel``: loops, widths and uploads."""

    def __init__(
        self,
        bert: BertModel,
        widths: dict[int, int],
        ladder: Sequence[int],
        cls_tasks: frozenset[PoolingTask],
        *,
        device: torch.device,
        dtype: torch.dtype,
        pad_token_id: int,
        position_offset: int,
    ) -> None:
        self.bert = bert
        self.widths = dict(sorted(widths.items()))
        """Declared lengths and each one's widest width."""
        self.ladder = tuple(sorted(ladder))
        """Every encoder length, declared or not."""
        self.cls_tasks = cls_tasks
        self.device = device
        self.dtype = dtype
        self.pad_token_id = pad_token_id
        self.position_offset = position_offset
        config = bert.config
        sa = cast(BertLayer, bert.encoder.layer[0]).attention.self
        emb_eps = float(cast(nn.LayerNorm, self._embeddings.LayerNorm).eps)
        self._loops = {
            length: _Loops(
                embed=compile_loop(
                    embed_loop(length, width, emb_eps),
                    "declared-width encoder embedding",
                ),
                layer=compile_loop(
                    layer_loop(length, width, sa.num_heads, sa.scaling, config.layer_norm_eps),
                    "declared-width encoder layer",
                ),
                cls=compile_loop(cls_loop(length, width), "declared-width encoder CLS rows"),
            )
            for length, width in self.widths.items()
        }
        self._pos_type: dict[int, torch.Tensor] = {}

    @property
    def lengths(self) -> tuple[int, ...]:
        return tuple(self.widths)

    @property
    def _embeddings(self) -> Any:
        return self.bert.embeddings

    def extent_for(self, num_seqs: int, max_len: int) -> int | None:
        """The declared length serving this batch, or None when it takes the static path.

        Only the shortest ladder length holding ``max_len`` counts: padding a short batch
        up to a longer declared length would trade the static rectangle for more rows.
        """
        extent = next((length for length in self.ladder if length >= max_len), None)
        if extent is None or extent not in self.widths or num_seqs > self.widths[extent]:
            return None
        return extent

    def _declared(self, host: torch.Tensor, length: int) -> torch.Tensor:
        return upload_declared(host, self.widths[length], self.device)

    def plan(self, extent: int, starts: list[int], query_lens: list[int]) -> DynamicWidthPlan:
        mask = encoder_key_pad_mask(extent, query_lens, self.dtype).reshape(-1, extent)
        return DynamicWidthPlan(
            extent=extent,
            query_lens=query_lens,
            starts=starts,
            mask=self._declared(mask, extent),
        )

    def set_input_ids(self, plan: DynamicWidthPlan, packed_ids: torch.Tensor) -> None:
        """Lay the packed body's ids out as ``[width, extent]``, right-padded."""
        extent = plan.extent
        # Host Python, as ``unpad_encoder_grid``: each CPU aten op on the per-step
        # path lengthens the eager guard chain (#981).
        flat = packed_ids[: plan.starts[-1] + plan.query_lens[-1]].tolist()
        rows = [
            flat[start : start + n] + [self.pad_token_id] * (extent - n)
            for start, n in zip(plan.starts, plan.query_lens, strict=True)
        ]
        plan.input_ids = self._declared(torch.tensor(rows, dtype=torch.int32), extent)

    def _pos_type_rows(self, length: int) -> torch.Tensor:
        """``[length, H]`` position rows of slots ``0..length-1`` plus segment 0, on device."""
        rows = self._pos_type.get(length)
        if rows is None:
            emb = self._embeddings
            pos = convert(emb.position_embeddings.weight.data, "cpu")
            seg = convert(emb.token_type_embeddings.weight.data, "cpu")
            offset = self.position_offset
            host = pos[offset : offset + length].float() + seg[0].float()
            rows = convert(host.to(self.dtype), self.device)
            self._pos_type[length] = rows
        return rows

    def drop_position_rows(self) -> None:
        """Forget the position rows, which copy the tables, after a weight reload."""
        self._pos_type.clear()

    def _layer_weights(self, layer: BertLayer) -> tuple[torch.Tensor, ...]:
        """The 16 operands of ``layer_tile``; the linears hold ``[in, out]`` weights."""
        attn, inter, out = layer.attention, layer.intermediate, layer.output
        proj = attn.self.qkv_proj
        qkv = [
            getattr(proj, f"{n}{part}") for n in _LINEAR_NAMES for part in ("_weight_t", "_bias")
        ]
        weights = (
            *qkv,
            attn.output.dense.weight,
            attn.output.dense.bias,
            attn.output.LayerNorm.weight,
            attn.output.LayerNorm.bias,
            inter.dense.weight,
            inter.dense.bias,
            out.dense.weight,
            out.dense.bias,
            out.LayerNorm.weight,
            out.LayerNorm.bias,
        )
        return cast(tuple[torch.Tensor, ...], weights)

    @torch.inference_mode()
    def run(self, plan: DynamicWidthPlan) -> torch.Tensor:
        """``[n * CLS_ROWS, H]`` on the device; row ``i * CLS_ROWS`` is sequence ``i``'s CLS."""
        assert plan.input_ids is not None, "prepare_inputs sets the ids before the forward"
        loops = self._loops[plan.extent]
        emb = self._embeddings
        x = loops.embed(
            plan.input_ids,
            emb.word_embeddings.weight,
            self._pos_type_rows(plan.extent),
            emb.LayerNorm.weight,
            emb.LayerNorm.bias,
        )
        for layer in self.bert.encoder.layer:
            x = loops.layer(x, plan.mask, *self._layer_weights(cast(BertLayer, layer)))
        return loops.cls(x)

    def warm(self) -> None:
        """Compile every length, then run it at its widest width, so neither compiles later."""
        for length, width in self.widths.items():
            for n in dict.fromkeys((1, width)):
                plan = self.plan(length, [i * length for i in range(n)], [length] * n)
                ids = torch.full((n * length,), self.pad_token_id, dtype=torch.int32)
                self.set_input_ids(plan, ids)
                convert(self.run(plan), "cpu")


def install(encoder: DynamicWidthEncoder) -> None:
    """Route ``BertModel.forward`` through the loops while a plan is published.

    An instance attribute: ``support_torch_compile``'s ``__call__`` calls
    ``self.forward`` when the model is not compiled whole, which ``ineligible_reason``
    requires.
    """
    bert = encoder.bert
    original = bert.forward

    def forward(
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        plan = published_dynamic_width_plan()
        if plan is None:
            return original(input_ids, positions, intermediate_tensors, inputs_embeds)
        return encoder.run(plan)

    bert.forward = forward  # ty: ignore[invalid-assignment]


def cls_pooled_tasks(pooler: nn.Module | None) -> frozenset[PoolingTask]:
    """The tasks whose pooler takes each sequence's first row."""
    poolers = getattr(pooler, "poolers_by_task", None) or {}
    return frozenset(
        task for task, sub in poolers.items() if isinstance(getattr(sub, "pooling", None), CLSPool)
    )


def _unquantized(linear: nn.Module) -> bool:
    method = getattr(linear, "quant_method", None)
    return isinstance(method, SpyreUnquantizedLinearMethod) and not getattr(
        linear, "spyre_row_padding", 0
    )


def ineligible_reason(vllm_config: VllmConfig, model: nn.Module) -> str | None:
    """Why this model cannot run the declared-width path, or None when it can."""
    model_config = vllm_config.model_config
    if model_config.runner_type != "pooling":
        return "not a pooling model"
    if (
        vllm_config.compilation_config.mode != CompilationMode.STOCK_TORCH_COMPILE
        or model_config.enforce_eager
        or compile_granularity() != "block"
    ):
        return "it needs the per-block STOCK_TORCH_COMPILE mode"
    if get_tensor_model_parallel_world_size() > 1:
        return "it runs at TP=1 only"
    if getattr(model, "supports_multimodal", False):
        return "multimodal encoders run packed"
    berts = [m for m in model.modules() if isinstance(m, BertModel)]
    if len(berts) != 1:
        return f"it needs exactly one BertModel, found {len(berts)}"
    bert = berts[0]
    config = bert.config
    if not isinstance(bert.embeddings, SpyreTokenTypeEmbedding):
        return f"{type(bert.embeddings).__name__} is not a Spyre absolute-position embedding"
    if getattr(config, "type_vocab_size", 1) != 1:
        return "it has more than one segment type"
    if getattr(config, "hidden_act", None) != "gelu":
        return f"hidden_act {getattr(config, 'hidden_act', None)!r} is not gelu"
    hidden = int(config.hidden_size)
    for layer in cast(list[Any], list(bert.encoder.layer)):
        sa = layer.attention.self
        if sa.head_dim % 64 or hidden % 64:
            return f"head size {sa.head_dim} and hidden size {hidden} need whole sticks"
        if getattr(sa.qkv_proj, "q_weight_t", None) is None or sa.qkv_proj.q_bias is None:
            return "its QKV projection is not split with biases"
        dense = (layer.attention.output.dense, layer.intermediate.dense, layer.output.dense)
        if not all(_unquantized(d) and d.bias is not None for d in dense):
            return "a layer linear is quantized, padded or has no bias"
    if not cls_pooled_tasks(getattr(model, "pooler", None)):
        return "no task pools the CLS row"
    return None


def _declared_dims_supported() -> bool:
    from torch_spyre._inductor import pass_utils

    return hasattr(pass_utils, "declared_dims")


def build(
    vllm_config: VllmConfig,
    model: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    *,
    pad_token_id: int,
) -> DynamicWidthEncoder | None:
    """The declared-width encoder for ``model``, or None when it is off or cannot run."""
    min_len = envs.SPYRE_ENCODER_DYNAMIC_WIDTH_MIN_LEN
    if min_len <= 0:
        return None
    reason = ineligible_reason(vllm_config, model)
    if reason is None and not _declared_dims_supported():
        reason = "this torch-spyre cannot loop over a declared dimension (torch-spyre#6066)"
    if reason is not None:
        logger.warning("SPYRE_ENCODER_DYNAMIC_WIDTH_MIN_LEN=%d is ignored: %s.", min_len, reason)
        return None

    from torch_spyre._inductor import config as ts_config

    # The width is the only declared size, so it sets these torch-spyre limits unless
    # the user exported them, as decode's page count does.
    if "MIN_RESERVED_DIM_SIZE" not in os.environ:
        ts_config.min_reserved_dim_size = 1
    max_num_seqs = int(vllm_config.scheduler_config.max_num_seqs)
    tables = encoder_shape_tables(vllm_config)
    widths: dict[int, int] = {}
    for length, batch in tables.rectangles:
        if length >= min_len and length % CLS_ROWS == 0:
            widths[length] = max(widths.get(length, 0), min(batch, max_num_seqs))
    if widths and "MAX_BUCKETS" not in os.environ:
        ts_config.max_buckets = max(ts_config.max_buckets, *widths.values())
    # With min == max the width is a constant, which fails to trace.
    widths = {
        length: width
        for length, width in widths.items()
        if ts_config.min_reserved_dim_size < width <= ts_config.max_buckets
    }
    if not widths:
        logger.warning(
            "SPYRE_ENCODER_DYNAMIC_WIDTH_MIN_LEN=%d is ignored: no rectangle at or above it "
            "holds two or more sequences within MAX_BUCKETS=%d.",
            min_len,
            ts_config.max_buckets,
        )
        return None
    bert = next(m for m in model.modules() if isinstance(m, BertModel))
    encoder = DynamicWidthEncoder(
        bert,
        widths,
        tables.lengths,
        cls_pooled_tasks(cast(nn.Module, model.pooler)),
        device=device,
        dtype=dtype,
        pad_token_id=pad_token_id,
        position_offset=roberta_position_delta(vllm_config.model_config.hf_config),
    )
    logger.info(
        "Encoder lengths %s compile once each with the batch width declared, for tasks %s.",
        {length: f"1..{width}" for length, width in encoder.widths.items()},
        sorted(encoder.cls_tasks),
    )
    return encoder
