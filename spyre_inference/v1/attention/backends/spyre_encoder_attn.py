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

"""Encoder-only (bidirectional) self-attention for Spyre without a KV cache.

Selected by ``TorchSpyrePlatform.get_attn_backend_cls`` for ENCODER/ENCODER_ONLY
layers. Operates on direct Q/K/V tensors rather than the paged KV-cache path.

Blocked flash, modelled on the decoder's ``_create_compilable_page_attn``. The
packed ``[T, H, D]`` list goes to the kernel whole and request boundaries ride
in int32 row-index tables, so a Spyre card never has to do offset arithmetic on
*shapes* — offsets are data. The kernel gathers its own sequence in-graph,
walks KV in ``ENCODER_BLOCK_SIZE``-token blocks carrying a running softmax
max/sum, and scatters the result back with ``index_copy_``. Nothing is sliced
or copied on the host, no ``(B, L)`` grid or ``[B, 1, L, L]`` mask is
materialised, and the only compile axis is the block count.

Two torch-spyre bugs shape the design. A compiled region reads its arguments
from offset 0 and ignores ``storage_offset`` (#3770), so a sequence cannot be
sliced outside the graph; and a gather that selects its whole source faults the
card (#4033), so an identity gather is skipped instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from vllm.v1.attention.backend import AttentionLayer

from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionBackend,
    SpyreAttentionImpl,
    SpyreAttentionMetadata,
    SpyrePagedKVCache,
    _maybe_compile,
)

# KV block width, in tokens. One Spyre stick of fp16.
ENCODER_BLOCK_SIZE = 64


def _blocks_for(length: int) -> int:
    """Block count covering ``length`` tokens, rounded up to a power of two.

    Encoder self-attention has ``q_len == kv_len``, so this single number fixes
    both the query extent (``num_blocks * ENCODER_BLOCK_SIZE``) and the KV loop
    trip count — the kernel cache has one shape axis, not two. Rounding to a
    power of two keeps that axis to a handful of buckets, matching the ladder
    ``_powers_of_two_up_to`` gives decoder attention.
    """
    blocks = max(1, (length + ENCODER_BLOCK_SIZE - 1) // ENCODER_BLOCK_SIZE)
    bucket = 1
    while bucket < blocks:
        bucket *= 2
    return bucket


def _create_encoder_block_kernel(
    num_blocks: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    *,
    needs_gather: bool = True,
    store_mode: str = "none",
):
    """Blocked online-softmax attention over one sequence of the packed list.

    Dynamo unrolls the KV loop because ``num_blocks``, ``needs_gather`` and
    ``store_mode`` are closure constants.

    ``head_size`` need not be stick-aligned: the kernel never slices a matmul
    operand, so Inductor's ``insert_bmm_padding`` can pad the contraction
    dimension itself.
    """
    num_queries_per_kv = num_heads // num_kv_heads
    padded_len = num_blocks * ENCODER_BLOCK_SIZE

    def specialized_encoder_block_attn_kernel(
        query,
        key,
        value,
        row_index,
        mask_tiles,
        scale,
        out=None,
    ):
        """
        Expected shapes:
            query: [num_tokens, num_heads, head_size], the whole batch's query
            key/value: [num_tokens, num_kv_heads, head_size], likewise
            row_index: [num_blocks * ENCODER_BLOCK_SIZE] int32 device tensor of
                this sequence's absolute rows. Lanes past its length repeat the
                last real row, so the gather never reads another request's
                tokens and the mask discards the duplicates.
            mask_tiles: num_blocks additive tiles, each
                [num_kv_heads, num_queries_per_kv, 1, ENCODER_BLOCK_SIZE]. The
                query axis is 1 because an encoder mask depends only on the KV
                column: every query row, real or padding, attends to exactly the
                real keys. That is what makes the padding rows exact duplicates
                of the last real row, which in turn makes the duplicate-index
                store below harmless.
            out: with store_mode="index", the caller's buffer to write into.

        Returns [padded_len, num_heads, head_size], or ``out`` when this kernel
        stored the result itself.
        """
        # A compiled region reads a view from offset 0, ignoring storage_offset
        # (torch-spyre#3770), so rows are gathered here rather than sliced
        # outside. A gather selecting its whole source instead faults the device
        # (RAS ComputeHardwareError 0x7b1b, torch-spyre#4033), hence needs_gather.
        #
        # One gather per tensor, then index the blocks at trace time: two
        # multi-element index_selects on the *same* tensor exhaust torch-spyre's
        # layout candidates (see _create_compilable_bucketed_decode_attn).
        if needs_gather:
            q_rows = query.index_select(0, row_index)
            k_rows = key.index_select(0, row_index)
            v_rows = value.index_select(0, row_index)
        else:
            q_rows, k_rows, v_rows = query, key, value

        q = (
            q_rows.unsqueeze(0)
            .transpose(1, 2)
            .reshape(num_kv_heads, num_queries_per_kv, padded_len, head_size)
        )
        k_blocks = k_rows.reshape(num_blocks, ENCODER_BLOCK_SIZE, num_kv_heads, head_size)
        v_blocks = v_rows.reshape(num_blocks, ENCODER_BLOCK_SIZE, num_kv_heads, head_size)

        tile_max = None
        tile_sum = None
        tile_output = None

        for i in range(num_blocks):
            # Token-major block to head-major for the matmuls; permutes on device.
            k_block = k_blocks[i].permute(1, 0, 2).unsqueeze(1)
            v_block = v_blocks[i].permute(1, 0, 2).unsqueeze(1)

            scores = torch.matmul(q, k_block.transpose(-2, -1)) * scale
            scores = scores + mask_tiles[i]
            scores_max = torch.amax(scores, dim=-1, keepdim=True)

            if i == 0:
                tile_max = scores_max
                tile_probs = torch.exp(scores - tile_max)
                tile_output = torch.matmul(tile_probs, v_block)
                tile_sum = tile_probs.sum(dim=-1, keepdim=True)
            else:
                # i > 0 only reachable after the i == 0 branch initialized these.
                assert tile_max is not None
                assert tile_sum is not None
                assert tile_output is not None
                new_max = torch.maximum(tile_max, scores_max)
                rescale = torch.exp(tile_max - new_max)
                tile_output = tile_output * rescale
                tile_sum = tile_sum * rescale
                tile_probs = torch.exp(scores - new_max)
                tile_output += torch.matmul(tile_probs, v_block)
                tile_sum = tile_sum + tile_probs.sum(dim=-1, keepdim=True)
                tile_max = new_max

        assert tile_output is not None and tile_sum is not None
        attn = tile_output / tile_sum
        attn = attn.reshape(1, num_heads, padded_len, head_size).transpose(1, 2)
        attn = attn.reshape(padded_len, num_heads, head_size)
        if store_mode == "index":
            # `out` and `query` are both indexed by absolute token row. Storing
            # the full padded extent keeps query_len out of the closure; the
            # padding rows repeat the sequence's last row and, per the mask note
            # above, carry the same value, so index_copy_'s undefined write
            # order for duplicate indices is harmless.
            assert out is not None
            out.index_copy_(0, row_index, attn)
            return out
        return attn

    return specialized_encoder_block_attn_kernel


@dataclass
class EncoderSeqPlan:
    """Everything the kernel needs for one request, built once per step."""

    start: int
    query_len: int
    num_blocks: int
    needs_gather: bool
    row_table: torch.Tensor
    mask_tiles: list[torch.Tensor]


def encoder_index_dtype(device: torch.device) -> torch.dtype:
    """int32 on Spyre, which has no int64 and whose compiled ``index_copy_``
    takes it; int64 everywhere else, where eager ``index_copy_`` rejects int32.
    """
    return torch.int32 if device.type == "spyre" else torch.int64


def encoder_row_table(start: int, query_len: int, extent: int, dtype: torch.dtype) -> torch.Tensor:
    """One sequence's absolute rows, pad lanes clamped to its last real row.

    ``extent`` is a multiple of ``ENCODER_BLOCK_SIZE`` and therefore of
    ``INT32_ELEMS_PER_STICK``, so the table is stick-aligned with no extra pad.
    """
    return torch.arange(extent, dtype=dtype).clamp(max=query_len - 1) + start


def _const_tile(
    masked: bool,
    num_kv_heads: int,
    num_queries_per_kv: int,
    dtype: torch.dtype,
    device: torch.device,
    cache: dict | None,
) -> torch.Tensor:
    """Shared all-zero or all-masked tile. Read-only: every block aliases it."""
    cache_key = (masked, num_kv_heads, num_queries_per_kv, dtype, str(device))
    tile = None if cache is None else cache.get(cache_key)
    if tile is None:
        fill = torch.finfo(dtype).min if masked else 0.0
        host = torch.full(
            (num_kv_heads, num_queries_per_kv, 1, ENCODER_BLOCK_SIZE), fill, dtype=dtype
        )
        tile = convert(host, device)
        if cache is not None:
            cache[cache_key] = tile
    return tile


def encoder_mask_tiles(
    num_blocks: int,
    kv_len: int,
    num_kv_heads: int,
    num_queries_per_kv: int,
    dtype: torch.dtype,
    device: torch.device,
    cache: dict | None = None,
) -> list[torch.Tensor]:
    """One additive tile per KV block. Only the boundary block is per-sequence.

    This is where the mask's ``L``-squared growth disappears: interior blocks
    are entirely real keys and beyond-the-end blocks entirely padding, so both
    take a constant tile that ``cache`` hands out by reference.
    """
    tiles: list[torch.Tensor] = []
    for i in range(num_blocks):
        lo = i * ENCODER_BLOCK_SIZE
        if lo + ENCODER_BLOCK_SIZE <= kv_len:
            tiles.append(_const_tile(False, num_kv_heads, num_queries_per_kv, dtype, device, cache))
        elif lo >= kv_len:
            tiles.append(_const_tile(True, num_kv_heads, num_queries_per_kv, dtype, device, cache))
        else:
            host = torch.full(
                (num_kv_heads, num_queries_per_kv, 1, ENCODER_BLOCK_SIZE),
                torch.finfo(dtype).min,
                dtype=dtype,
            )
            host[..., : kv_len - lo] = 0
            tiles.append(convert(host, device))
    return tiles


def dense_sdpa_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_lens: list[int],
    scale: float,
) -> torch.Tensor:
    """Per-sequence eager SDPA on the packed list (probe / unit-test reference)."""
    outs: list[torch.Tensor] = []
    start = 0
    for length in query_lens:
        q = query[start : start + length]
        k = key[start : start + length]
        v = value[start : start + length]
        qh = q.unsqueeze(0).transpose(1, 2)
        kh = k.unsqueeze(0).transpose(1, 2)
        vh = v.unsqueeze(0).transpose(1, 2)
        kwargs: dict = {"is_causal": False, "scale": scale}
        if q.shape[1] != k.shape[1]:
            kwargs["enable_gqa"] = True
        out = F.scaled_dot_product_attention(qh, kh, vh, **kwargs)
        outs.append(out.transpose(1, 2).squeeze(0))
        start += length
    return torch.cat(outs, dim=0)


class SpyreEncoderAttentionImpl(SpyreAttentionImpl):
    """Bidirectional encoder self-attention (no KV cache).

    The platform selects this impl for ENCODER/ENCODER_ONLY layers (see
    ``TorchSpyrePlatform.get_attn_backend_cls``). Forward stays inside the
    opaque ``unified_attention`` op and dispatches the blocked kernel per
    request, with no host-side slicing of activations.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._block_fns: dict[tuple, object] = {}
        # Interior and fully-masked tiles are shape-only, so one device copy
        # each serves every sequence, layer and step.
        self._const_tiles: dict[tuple, torch.Tensor] = {}
        self._warmed_buffers: set[int] = set()

    def _get_block_fn(
        self,
        num_blocks: int,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        *,
        needs_gather: bool,
        store_mode: str,
    ):
        key = (num_blocks, num_heads, num_kv_heads, head_size, needs_gather, store_mode)
        fn = self._block_fns.get(key)
        if fn is None:
            fn = _maybe_compile(
                _create_encoder_block_kernel(
                    num_blocks,
                    num_heads,
                    num_kv_heads,
                    head_size,
                    needs_gather=needs_gather,
                    store_mode=store_mode,
                ),
                self._compile_attn,
            )
            self._block_fns[key] = fn
        return fn

    def _warm_block_fns(
        self,
        buffer_rows: int,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Compile every kernel a batch of this token count can ask for.

        A request's block count follows its own length, not the body bucket, so
        the warmup dummies do not span the ladder on their own — one long
        request mid-serve would otherwise stall the server compiling. Doing it
        here, on the first forward for each body size, keeps that cost inside
        warmup, where the dummy runs already visit every body size.

        Scratch contents are irrelevant; only the shapes reach the cache key.
        """
        if buffer_rows in self._warmed_buffers:
            return
        self._warmed_buffers.add(buffer_rows)

        num_queries_per_kv = num_heads // num_kv_heads
        query = convert(torch.zeros((buffer_rows, num_heads, head_size), dtype=dtype), device)
        key = convert(torch.zeros((buffer_rows, num_kv_heads, head_size), dtype=dtype), device)
        value = convert(torch.zeros((buffer_rows, num_kv_heads, head_size), dtype=dtype), device)
        out = convert(torch.zeros((buffer_rows, num_heads, head_size), dtype=dtype), device)

        num_blocks = 1
        while num_blocks * ENCODER_BLOCK_SIZE <= buffer_rows:
            extent = num_blocks * ENCODER_BLOCK_SIZE
            rows = convert(
                encoder_row_table(0, extent, extent, encoder_index_dtype(device)), device
            )
            tiles = encoder_mask_tiles(
                num_blocks,
                extent,
                num_kv_heads,
                num_queries_per_kv,
                dtype,
                device,
                self._const_tiles,
            )
            # A sequence only skips the gather when it fills the buffer exactly,
            # which pins the block count; every other case gathers.
            for needs_gather in (True, False):
                if not needs_gather and extent != buffer_rows:
                    continue
                attn_fn = self._get_block_fn(
                    num_blocks,
                    num_heads,
                    num_kv_heads,
                    head_size,
                    needs_gather=needs_gather,
                    store_mode="index",
                )
                attn_fn(query, key, value, rows, tiles, self.scale, out=out)
            num_blocks *= 2

    def _build_plans(
        self,
        attn_metadata: SpyreAttentionMetadata,
        query: torch.Tensor,
        num_kv_heads: int,
        num_queries_per_kv: int,
    ) -> list[EncoderSeqPlan]:
        """Row tables and mask tiles for every request in the step."""
        query_start_loc = attn_metadata.query_start_loc.cpu().tolist()
        seq_lens = attn_metadata.seq_lens.cpu().tolist()
        # The body may 1D-pad past num_actual_tokens; those rows are not a request.
        num_tokens = attn_metadata.num_actual_tokens
        buffer_rows = query.shape[0]
        device = query.device
        index_dtype = encoder_index_dtype(device)

        plans: list[EncoderSeqPlan] = []
        for seq_idx in range(attn_metadata.num_seqs):
            start = int(query_start_loc[seq_idx])
            query_len = int(query_start_loc[seq_idx + 1]) - start
            if start >= num_tokens or query_len <= 0:
                continue
            query_len = min(query_len, num_tokens - start)
            kv_len = min(int(seq_lens[seq_idx]), query_len)

            num_blocks = _blocks_for(max(query_len, kv_len))
            extent = num_blocks * ENCODER_BLOCK_SIZE
            plans.append(
                EncoderSeqPlan(
                    start=start,
                    query_len=query_len,
                    num_blocks=num_blocks,
                    needs_gather=not (start == 0 and query_len == extent and buffer_rows == extent),
                    row_table=convert(
                        encoder_row_table(start, query_len, extent, index_dtype), device
                    ),
                    mask_tiles=encoder_mask_tiles(
                        num_blocks,
                        kv_len,
                        num_kv_heads,
                        num_queries_per_kv,
                        query.dtype,
                        device,
                        self._const_tiles,
                    ),
                )
            )
        return plans

    def forward(  # ty: ignore[invalid-method-override]
        self,
        layer: AttentionLayer,
        query: torch.Tensor,  # [num_tokens, num_heads, head_size]
        key: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
        value: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
        kv_cache: SpyrePagedKVCache,
        attn_metadata: SpyreAttentionMetadata,
        output: torch.Tensor,  # [num_tokens, num_heads, head_size]
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del layer, kv_cache, output_scale, output_block_scale
        if attn_metadata is None:
            return output

        num_heads = query.shape[1]
        num_kv_heads = key.shape[1]
        head_size = query.shape[2]

        # Everything runs where the result lands. A real step already has all
        # four on the same device; unit tests hand in host activations.
        if query.device != output.device:
            query = convert(query, output.device)
            key = convert(key, output.device)
            value = convert(value, output.device)

        # Folds the per-layer eager store into the attention jobplan. Re-checked
        # per call: vLLM hands out a fresh buffer per layer.
        fused_store_ok = (
            self._compile_attn
            and output.dtype == query.dtype
            # A compiled kernel reads its arguments from offset 0: torch-spyre#3770.
            and output.storage_offset() == 0
            and output.is_contiguous()
        )
        store_mode = "index" if fused_store_ok else "none"

        if store_mode == "index":
            self._warm_block_fns(
                query.shape[0], num_heads, num_kv_heads, head_size, query.dtype, query.device
            )

        # Built once per step; the whole encoder stack shares one build.
        if attn_metadata.encoder_seq_plans is None:
            attn_metadata.encoder_seq_plans = self._build_plans(
                attn_metadata, query, num_kv_heads, num_heads // num_kv_heads
            )

        for plan in attn_metadata.encoder_seq_plans:
            attn_fn = self._get_block_fn(
                plan.num_blocks,
                num_heads,
                num_kv_heads,
                head_size,
                needs_gather=plan.needs_gather,
                store_mode=store_mode,
            )
            result = attn_fn(
                query,
                key,
                value,
                plan.row_table,
                plan.mask_tiles,
                self.scale,
                out=output if store_mode == "index" else None,
            )
            if store_mode == "none":
                output[plan.start : plan.start + plan.query_len] = result[: plan.query_len]

        return output


class SpyreEncoderAttentionBackend(SpyreAttentionBackend):
    """Encoder-only (no KV cache) variant of the Spyre backend."""

    # These layers have no KV cache, but vLLM still hands encoder-only specs a
    # zero-filled slot mapping, so upstream must skip `unified_kv_cache_update` entirely.
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_impl_cls() -> type[SpyreEncoderAttentionImpl]:
        return SpyreEncoderAttentionImpl
