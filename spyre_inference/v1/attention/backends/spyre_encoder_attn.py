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

Ragged→dense packing scatters real tokens into a zeroed ``[B, L]`` grid
(compiled ``index_copy_``, same kernel style as decoder KV write). Unpack
is still ``index_select``. Pad slots stay zeros (never written). ``B=1``
with ``T == L`` skips scatter/gather: the body is already dense and the
attention mask hides pad slots (62-token prompts on an L=64 bucket). Dest
indices and the attention mask are built on the first layer of a step and
reused (decoder ``page_index_tables`` pattern).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from vllm.config import get_current_vllm_config
from vllm.v1.attention.backend import AttentionLayer

from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionBackend,
    SpyreAttentionImpl,
    SpyreAttentionMetadata,
    SpyrePagedKVCache,
)
from spyre_inference.v1.pool import select_rows
from spyre_inference.v1.worker.spyre_shape_bucketer import (
    default_encoder_len_buckets,
    pick_encoder_attention_shape,
    pooling_warmup_shapes,
)

# Pad seq length *and* head dim to the Spyre stick (64 fp16 elements).
# L-aligned keeps P·V's K stick-aligned; D-aligned keeps QKᵀ's K stick-aligned
# so Inductor never enters insert_bmm_padding (torch-spyre KeyError: 'val' on
# FX nodes missing meta["val"] when padding MiniLM's head_size=32).
ENCODER_SEQ_ALIGNMENT = 64


def _align_up(n: int, align: int = ENCODER_SEQ_ALIGNMENT) -> int:
    return max(align, (n + align - 1) // align * align)


def host_pack_indices(
    q_starts: list[int],
    lengths: list[int],
    aligned_len: int,
    pad_row: int,
) -> torch.Tensor:
    """Build ``[B, L]`` int64 row indices; pad slots point at ``pad_row``."""
    batch = len(q_starts)
    indices = torch.full((batch, aligned_len), pad_row, dtype=torch.int64)
    for s, (start, length) in enumerate(zip(q_starts, lengths)):
        if length > 0:
            indices[s, :length] = torch.arange(start, start + length, dtype=torch.int64)
    return indices


def host_scatter_pack_dest(
    q_starts: list[int],
    lengths: list[int],
    aligned_len: int,
    num_src_rows: int,
    dummy_row: int,
) -> torch.Tensor:
    """``[num_src_rows]`` packed-row dest for each source token.

    Real tokens write ``s * L + pos``. Body-pad / dummy seqs write ``dummy_row``
    (an extra workspace row, not an attention slot).
    """
    dest = torch.full((num_src_rows,), dummy_row, dtype=torch.int64)
    for s, (start, length) in enumerate(zip(q_starts, lengths)):
        if length > 0:
            dest[start : start + length] = torch.arange(
                s * aligned_len, s * aligned_len + length, dtype=torch.int64
            )
    return dest


def host_unpack_indices(
    q_starts: list[int],
    query_lens: list[int],
    aligned_len: int,
    num_tokens: int,
) -> torch.Tensor:
    """Build ``[T]`` int64 indices from flat padded ``[B*L]`` back to tokens.

    ``num_tokens`` may exceed the real count; unfilled entries stay ``0``
    (a safe row to read — nothing downstream reads those output rows).
    """
    indices = torch.zeros(num_tokens, dtype=torch.int64)
    for s, (start, length) in enumerate(zip(q_starts, query_lens)):
        if length > 0:
            base = s * aligned_len
            indices[start : start + length] = torch.arange(base, base + length, dtype=torch.int64)
    return indices


def _pad_head_dim_to_stick(flat: torch.Tensor, head_size_padded: int) -> torch.Tensor:
    """Pad last dim to a stick. MiniLM ``[T,H,32]`` cannot ``F.pad`` on Spyre."""
    head_size = flat.shape[-1]
    if head_size == head_size_padded:
        return flat
    device = flat.device
    if device.type == "spyre":
        flat = convert(flat, "cpu")
    flat = F.pad(flat, (0, head_size_padded - head_size))
    if device.type == "spyre":
        flat = convert(flat, device)
    return flat


def _is_identity_row_map(indices: torch.Tensor, num_rows: int) -> bool:
    """True when ``indices`` is ``0 .. num_rows-1`` (no pad gather). Host only."""
    if indices.numel() != num_rows or indices.device.type != "cpu":
        return False
    flat = indices.reshape(-1)
    return bool(torch.equal(flat, torch.arange(num_rows, dtype=flat.dtype)))


def _is_b1_dense_body(batch: int, num_src: int, aligned_len: int) -> bool:
    """Single sequence already shaped ``[L, H, D]`` (token bucket == SDPA L)."""
    return batch == 1 and num_src == aligned_len


def _index_copy_kernel(dst: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Tiny mutation, compiled alone — do not fuse with SDPA."""
    dst.index_copy_(0, index, src)
    return dst


_compiled_index_copy: object | None = None


def _index_copy(dst: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Eager on CPU; compiled ``index_copy_`` on Spyre (eager falls back / rejects)."""
    global _compiled_index_copy
    if dst.device.type != "spyre":
        return _index_copy_kernel(dst, index, src)
    if _compiled_index_copy is None:
        _compiled_index_copy = torch.compile(_index_copy_kernel, dynamic=False)
    return _compiled_index_copy(dst, index, src)


def gather_pack(
    flat: torch.Tensor,
    pack_indices: torch.Tensor,
    head_size_padded: int,
) -> torch.Tensor:
    """Reference pack: ``F.pad`` zero row + ``index_select`` of ``B×L``.

    Kept for tests. Serve uses ``scatter_pack``. ``B=1`` with identity rows
    (``T == L`` and no pad slots) skips ``index_select``.
    """
    batch, aligned_len = pack_indices.shape
    _t, num_heads, _d = flat.shape
    flat = _pad_head_dim_to_stick(flat, head_size_padded)
    if batch == 1 and _is_identity_row_map(pack_indices, flat.shape[0]):
        packed = flat.unsqueeze(0)
        return packed.permute(0, 2, 1, 3).contiguous()
    flat_ext = F.pad(flat, (0, 0, 0, 0, 0, 1))
    gathered = select_rows(flat_ext, pack_indices)  # [B*L, H, Dp]
    packed = gathered.view(batch, aligned_len, num_heads, head_size_padded)
    return packed.permute(0, 2, 1, 3).contiguous()


def scatter_pack(
    flat: torch.Tensor,
    dest_idx: torch.Tensor,
    batch: int,
    aligned_len: int,
    head_size_padded: int,
) -> torch.Tensor:
    """Pack varlen ``[T, H, D]`` → ``[B, H, L, Dp]`` via compiled ``index_copy_``.

    ``dest_idx`` is ``[T]`` packed-row ids (host or device). Pad slots are
    implicit zeros. A dest past ``B×L`` is a dummy row for body-bucket padding.

    ``B=1`` with ``T == L`` skips ``index_copy_``: the runner already padded
    the body to the SDPA length. The attention mask hides leftover pad tokens.
    ``B > 1`` still scatters.
    """
    _t, num_heads, _d = flat.shape
    flat = _pad_head_dim_to_stick(flat, head_size_padded)
    if _is_b1_dense_body(batch, flat.shape[0], aligned_len):
        packed = flat.unsqueeze(0)
        return packed.permute(0, 2, 1, 3).contiguous()
    packed_rows = batch * aligned_len
    # Dummy dest is always ``B*L``. Extra row avoids ``dest.max()`` (device sync).
    workspace = torch.zeros(
        packed_rows + 1,
        num_heads,
        head_size_padded,
        dtype=flat.dtype,
        device=flat.device,
    )
    # Spyre has no int64. H2D dest as int32 and never .to(int64) on device
    # (that CPU-detours and used to scramble B>1 dests). Eager CPU wants int64.
    if dest_idx.device.type == "spyre":
        idx = dest_idx
    elif flat.device.type == "spyre":
        idx = convert(dest_idx.to(torch.int32), flat.device)
    else:
        idx = dest_idx.to(device=flat.device, dtype=torch.int64)
    _index_copy(workspace, idx, flat)
    packed = workspace[:packed_rows].view(batch, aligned_len, num_heads, head_size_padded)
    return packed.permute(0, 2, 1, 3).contiguous()


def gather_unpack(
    attn_out: torch.Tensor,
    unpack_indices: torch.Tensor,
    head_size: int,
) -> torch.Tensor:
    """Unpack padded ``[B, H, L, Dp]`` → flat ``[T, H, D]`` via ``index_select``.

    Identity ``B=1`` (``T == B×L``) is a reshape; pad / multi-seq still gather.
    """
    batch, num_heads, aligned_len, head_size_padded = attn_out.shape
    tokens = attn_out.permute(0, 2, 1, 3).contiguous()
    flat_padded = tokens.reshape(batch * aligned_len, num_heads, head_size_padded)
    if _is_identity_row_map(unpack_indices, flat_padded.shape[0]) or _is_b1_dense_body(
        batch, unpack_indices.shape[0], aligned_len
    ):
        gathered = flat_padded
    else:
        gathered = select_rows(flat_padded, unpack_indices)
    if gathered.shape[-1] == head_size:
        return gathered
    # Crop is a slice, not pad. D=32 is half a stick; do it on CPU.
    if gathered.device.type == "spyre":
        gathered = convert(gathered, "cpu")
    return gathered[..., :head_size].contiguous()


def _indices_for_device(indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move unpack indices onto ``device`` once. Spyre index_select is int32."""
    if device.type == "spyre":
        cpu = indices if indices.device.type == "cpu" else indices.cpu()
        return convert(cpu.to(torch.int32), device)
    return indices.to(device=device, dtype=torch.long)


def _dest_for_device(indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move scatter dest onto ``device`` once. Spyre compiled index_copy_ is int32."""
    if device.type == "spyre":
        cpu = indices if indices.device.type == "cpu" else indices.cpu()
        return convert(cpu.to(torch.int32), device)
    return indices.to(device=device, dtype=torch.long)


def _ensure_encoder_pack(
    attn_metadata: SpyreAttentionMetadata,
    *,
    padded_tokens: int,
    n: int,
    query: torch.Tensor,
    target_device: torch.device,
    cached_encoder_shapes: list[tuple[int, int]],
    cached_max_num_seqs: int,
    cached_max_model_len: int,
    cached_max_num_batched_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build scatter dest + unpack + mask once per step; later layers reuse them."""
    if attn_metadata.encoder_q_pack_idx is not None:
        assert attn_metadata.encoder_kv_pack_idx is not None
        assert attn_metadata.encoder_unpack_idx is not None
        assert attn_metadata.encoder_attn_mask is not None
        return (
            attn_metadata.encoder_q_pack_idx,
            attn_metadata.encoder_kv_pack_idx,
            attn_metadata.encoder_unpack_idx,
            attn_metadata.encoder_attn_mask,
        )

    qsl = attn_metadata.query_start_loc.cpu()
    q_starts = qsl[:-1].tolist()
    query_lens = torch.diff(qsl).tolist()
    kv_lens = attn_metadata.seq_lens.cpu().tolist()
    num_seqs = attn_metadata.num_seqs
    max_len = max(query_lens, default=0)
    pair = pick_encoder_attention_shape(
        num_seqs,
        max_len,
        cached_encoder_shapes,
        cached_max_num_seqs,
        cached_max_model_len,
        cached_max_num_batched_tokens,
    )
    batch_bucket, aligned_len = pair if pair is not None else (num_seqs, _align_up(max_len))
    orig_q_starts = q_starts
    orig_query_lens = query_lens
    if batch_bucket > num_seqs:
        q_starts = q_starts + [n] * (batch_bucket - num_seqs)
        query_lens = query_lens + [0] * (batch_bucket - num_seqs)
        kv_lens = kv_lens + [0] * (batch_bucket - num_seqs)

    dummy_row = batch_bucket * aligned_len
    kv_pack_lens = [min(q, k) for q, k in zip(query_lens, kv_lens)]
    q_dest = host_scatter_pack_dest(q_starts, query_lens, aligned_len, padded_tokens, dummy_row)
    kv_dest = host_scatter_pack_dest(q_starts, kv_pack_lens, aligned_len, padded_tokens, dummy_row)
    unpack_idx = host_unpack_indices(orig_q_starts, orig_query_lens, aligned_len, padded_tokens)
    mask = build_attention_mask(
        batch_bucket,
        aligned_len,
        query_lens,
        kv_lens,
        dtype=query.dtype,
        device=target_device,
    )

    attn_metadata.encoder_q_pack_idx = _dest_for_device(q_dest, target_device)
    attn_metadata.encoder_kv_pack_idx = _dest_for_device(kv_dest, target_device)
    attn_metadata.encoder_unpack_idx = _indices_for_device(unpack_idx, target_device)
    attn_metadata.encoder_attn_mask = mask
    return (
        attn_metadata.encoder_q_pack_idx,
        attn_metadata.encoder_kv_pack_idx,
        attn_metadata.encoder_unpack_idx,
        attn_metadata.encoder_attn_mask,
    )


def build_attention_mask(
    num_seqs: int,
    aligned_len: int,
    query_lens: list[int],
    kv_lens: list[int],
    dtype: torch.dtype,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Additive mask ``[B, 1, L, L]``: 0 where attend, ``-inf`` elsewhere.

    Built on the host (vectorized ``lt`` + nested ``where``), then ``convert``'d.
    On-device materialization is not stick-safe: Spyre cannot produce bool from
    int32 ``lt``, and cannot broadcast ``where`` of ``[B,1,L,1]`` × ``[B,1,1,L]``
    into ``[B,1,L,L]`` (no stick-scatter).
    """
    if device is None:
        device = torch.device("cpu")
    elif not isinstance(device, torch.device):
        device = torch.device(device)
    if num_seqs != len(query_lens):
        raise ValueError(f"num_seqs={num_seqs} != len(query_lens)={len(query_lens)}")

    q_len = torch.tensor(query_lens, dtype=torch.int32)
    kv_len = torch.tensor(
        [min(q, k) for q, k in zip(query_lens, kv_lens)],
        dtype=torch.int32,
    )
    q_pos = torch.arange(aligned_len, dtype=torch.int32)
    kv_pos = torch.arange(aligned_len, dtype=torch.int32)
    zeros = torch.zeros((), dtype=dtype)
    neg_inf = torch.tensor(torch.finfo(dtype).min, dtype=dtype)

    q_ok = (q_pos.unsqueeze(0) < q_len.unsqueeze(1)).unsqueeze(1).unsqueeze(-1)
    k_ok = (kv_pos.unsqueeze(0) < kv_len.unsqueeze(1)).unsqueeze(1).unsqueeze(2)
    mask = torch.where(q_ok, torch.where(k_ok, zeros, neg_inf), neg_inf)
    if device.type == "spyre":
        return convert(mask, device)
    return mask.to(device)


class SpyreEncoderAttentionImpl(SpyreAttentionImpl):
    """Bidirectional encoder self-attention (no KV cache).

    The platform selects this impl for ENCODER/ENCODER_ONLY layers (see
    ``TorchSpyrePlatform.get_attn_backend_cls``), so there is no per-call
    ``attn_type`` branch. Setup is shared with the paged decoder impl; forward
    packs with scatter (``index_copy_``, Spyre dest is int32), then SDPA
    and gather-unpack. ``B=1`` and ``T == L`` skip pack/unpack.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # get_current_vllm_config() only works at construction time; forward()
        # runs through a custom-op boundary that loses the context.
        cfg = get_current_vllm_config()
        self._cached_max_num_seqs = cfg.scheduler_config.max_num_seqs
        self._cached_max_model_len = cfg.model_config.max_model_len
        self._cached_max_num_batched_tokens = cfg.scheduler_config.max_num_batched_tokens
        self._cached_encoder_shapes = pooling_warmup_shapes(
            max_num_seqs=self._cached_max_num_seqs,
            max_model_len=self._cached_max_model_len,
            max_num_batched_tokens=self._cached_max_num_batched_tokens,
            len_ladder=default_encoder_len_buckets(self._cached_max_model_len),
        )

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

        # query/key/value/output are padded to the runner's warmed body-bucket
        # size, not num_actual_tokens. Keep that shape or index_select
        # recompiles per request.
        n = attn_metadata.num_actual_tokens
        padded_tokens = query.shape[0]
        target_device = output.device
        num_heads = query.shape[1]
        num_kv_heads = key.shape[1]
        head_size = query.shape[2]
        head_size_padded = _align_up(head_size)
        scale = self.scale

        q_dest, kv_dest, unpack_idx, mask = _ensure_encoder_pack(
            attn_metadata,
            padded_tokens=padded_tokens,
            n=n,
            query=query,
            target_device=target_device,
            cached_encoder_shapes=self._cached_encoder_shapes,
            cached_max_num_seqs=self._cached_max_num_seqs,
            cached_max_model_len=self._cached_max_model_len,
            cached_max_num_batched_tokens=self._cached_max_num_batched_tokens,
        )

        if query.device.type != target_device.type:
            query = convert(query, target_device.type)
            key = convert(key, target_device.type)
            value = convert(value, target_device.type)

        batch, _, aligned_len, _ = mask.shape
        q_batched = scatter_pack(query, q_dest, batch, aligned_len, head_size_padded)
        k_batched = scatter_pack(key, kv_dest, batch, aligned_len, head_size_padded)
        v_batched = scatter_pack(value, kv_dest, batch, aligned_len, head_size_padded)

        sdpa_kwargs: dict = {"is_causal": False, "scale": scale}
        if num_kv_heads != num_heads:
            sdpa_kwargs["enable_gqa"] = True

        attn_out = F.scaled_dot_product_attention(
            q_batched, k_batched, v_batched, attn_mask=mask, **sdpa_kwargs
        )

        result = gather_unpack(attn_out, unpack_idx, head_size)
        if result.dtype != output.dtype:
            result = convert(result, dtype=output.dtype)

        # MiniLM D=32: flatten to [T, H*D] (384 = 6 sticks) so the write is aligned.
        use_flat_write = target_device.type == "spyre" and head_size % ENCODER_SEQ_ALIGNMENT != 0
        if use_flat_write:
            if result.device.type == "spyre":
                result = convert(result, "cpu")
            src = convert(
                result.reshape(padded_tokens, -1).contiguous(), target_device.type, output.dtype
            )
            output.reshape(padded_tokens, -1).copy_(src)
        else:
            if result.device.type != output.device.type:
                result = convert(result, output.device)
            output.copy_(result)

        return output


class SpyreEncoderAttentionBackend(SpyreAttentionBackend):
    """Encoder-only (no KV cache) variant of the Spyre backend."""

    # These layers have no KV cache, but vLLM still hands encoder-only specs a
    # zero-filled slot mapping, so upstream must skip `unified_kv_cache_update` entirely.
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_impl_cls() -> type[SpyreEncoderAttentionImpl]:
        return SpyreEncoderAttentionImpl
