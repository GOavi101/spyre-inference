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

Ragged→dense packing uses compiled ``index_copy_`` on Spyre into a
slot-major workspace (decoder KV / torch-spyre#3705) so ``view(B, L, …)``
is address-order-preserving. Default-layout ``view`` after ``index_copy_``
scrambles B>1 (real-slot cosine ~0.07). Body-pad dests write an extra
dummy row (not slot 0 — that is CLS). Unpack is ``index_select``. ``B=1``
with ``T == L`` and a full prompt compiles permute+SDPA (no ``attn_mask``).
Any live pad uses packed QK: compile matmul only, eager pad add, compile P·V
(Inductor ``matmul + mask`` → ``F.sdpa`` drops the mask; BGE cosine
~0.46). Dest/mask use ``min(qsl, seq_lens, num_actual_tokens)``.
Dest/unpack stay on host when ``T == L``. Pack tensors are built once
per step.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
import torch.nn.functional as F
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionLayer

from spyre_inference import envs
from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionBackend,
    SpyreAttentionImpl,
    SpyreAttentionMetadata,
    SpyrePagedKVCache,
    _maybe_compile,
    slot_major_kv_layout,
)
from spyre_inference.v1.pool import select_rows
from spyre_inference.v1.worker.spyre_shape_bucketer import (
    default_encoder_len_buckets,
    next_bucket,
    pick_encoder_attention_shape,
    pooling_warmup_shapes,
)

logger = init_logger(__name__)

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


def _content_query_lens(
    qsl_lens: list[int],
    kv_lens: list[int],
    num_actual_tokens: int | None = None,
) -> list[int]:
    """Per-seq counts for dest/mask. ``query_start_loc`` can include 1D body pad.

    When B=1 qsl and ``seq_lens`` are both the body bucket, ``num_actual_tokens``
    (unpadded scheduled count) still marks pad: ``min(qsl, seq, n)``.

    B>1 does not redistribute. The runner 1D-pads the concatenated body
    (suffix after the last sequence); per-seq padded qsl cannot occur. Shrinking
    the tail would turn real ``[5, 12]`` reported as ``[32, 32]`` into ``[17, 0]``.
    """
    out = [min(int(q), int(k)) for q, k in zip(qsl_lens, kv_lens)]
    if not out or num_actual_tokens is None:
        return out
    n = int(num_actual_tokens)
    if len(out) == 1:
        out[0] = min(out[0], n)
        return out
    excess = sum(out) - n
    if excess > 0:
        raise AssertionError(
            f"B>1 query_start_loc/seq_lens include {excess} pad tokens "
            f"(lens={out}, num_actual_tokens={n}). Per-seq padded qsl is "
            "unsupported; 1D body pad is a suffix after the last sequence."
        )
    return out


def dummy_pack_row(lengths: list[int], aligned_len: int) -> int:
    """First pad slot in the ``[B, L]`` grid, or ``0`` when the grid is full.

    Tests only. Serve writes body-pad to an extra workspace row (``B×L``),
    never ``0``: a full grid plus a 1D body bucket ``> B×L`` would otherwise
    overwrite CLS.
    """
    for s, length in enumerate(lengths):
        if length < aligned_len:
            return s * aligned_len + length
    return 0


def host_scatter_pack_dest(
    q_starts: list[int],
    lengths: list[int],
    aligned_len: int,
    num_src_rows: int,
    dummy_row: int,
) -> torch.Tensor:
    """``[num_src_rows]`` packed-row dest for each source token.

    Real tokens write ``s * L + pos``. Body-pad rows write ``dummy_row``
    (serve: extra workspace row ``B×L``, not a real token).
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


def _build_seq_row_index(start: int, real_len: int, aligned_len: int) -> torch.Tensor:
    """Absolute row indices for one sequence, into the flat batch tensor.

    Positions past ``real_len`` clamp to the last real row rather than a
    sentinel, mirroring the decoder's ``_build_query_row_tables``. A compiled
    kernel reads via ``index_select`` here, never a plain slice, since a
    compiled region reads a view from offset 0 regardless of storage_offset
    (torch-spyre#3770).
    """
    pos = torch.arange(aligned_len, dtype=torch.int64)
    return start + torch.minimum(pos, torch.tensor(real_len - 1, dtype=torch.int64))


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


def _is_b1_fused_sdpa(batch: int, padded_tokens: int, aligned_len: int, real_len: int) -> bool:
    """Compiled permute+SDPA is only legal when the mask is dense (no live pad)."""
    return _is_b1_dense_body(batch, padded_tokens, aligned_len) and real_len == aligned_len


def _index_copy_kernel(dst: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Tiny mutation, compiled alone — do not fuse with SDPA."""
    dst.index_copy_(0, index, src)
    return dst


_CompiledFn = Callable[..., torch.Tensor]

_compiled_kernels: dict[Callable[..., torch.Tensor], _CompiledFn] = {}


def _compile_if_spyre(kernel: _CompiledFn, device_type: str) -> _CompiledFn:
    """Compile ``kernel`` once on Spyre. CPU always runs ``kernel``.

    Memo is keyed on the Python function, so packed QK, P·V, index_copy_, and
    the two B=1 SDPA kernels each compile once without caller write-back.
    """
    if device_type != "spyre":
        return kernel
    compiled = _compiled_kernels.get(kernel)
    if compiled is None:
        compiled = cast(_CompiledFn, torch.compile(kernel, dynamic=False))
        _compiled_kernels[kernel] = compiled
    return compiled


def _b1_sdpa_kernel(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """``[T,H,D]`` → SDPA ``[1,H,T,D]`` → ``[T,H,D]``. No mask, no eager ``contiguous``.

    Full-bucket fused path only (``real_len == L``). A zeros ``attn_mask`` is
    still an H2D + per-layer add; Spyre compiled SDPA also drops a live pad
    mask, so pad stays on the packed path.
    """
    q = query.unsqueeze(0).transpose(1, 2)
    k = key.unsqueeze(0).transpose(1, 2)
    v = value.unsqueeze(0).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=scale)
    t, h, d = query.shape
    return out.transpose(1, 2).reshape(t, h, d)


def _b1_sdpa_kernel_gqa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    q = query.unsqueeze(0).transpose(1, 2)
    k = key.unsqueeze(0).transpose(1, 2)
    v = value.unsqueeze(0).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=scale, enable_gqa=True)
    t, h, d = query.shape
    return out.transpose(1, 2).reshape(t, h, d)


def host_key_pad_mask(mask: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    """CPU ``[B,1,L,L]`` → dense ``[B*KV, 1, L, L]`` for eager ``scores + mask``.

    Same key-pad row is repeated across KV and query-L. ``[B*KV, 1, 1, L]``
    was tried (decoder ``mask_by_block``). Decode gets away with that shape
    because Q=1; encoder scores are ``[BH, G, L, L]``. Eager Spyre add does
    not broadcast ``1 → L`` on the query axis (no stick-scatter). Prefill
    tiles are already ``[Q, block]``. Densify on the host once per step.
    """
    key = mask[:, :, :1, :]
    batch, _, _, length = key.shape
    return (
        key.expand(batch, num_kv_heads, length, length)
        .reshape(batch * num_kv_heads, 1, length, length)
        .contiguous()
    )


def _packed_qk_matmul(query: torch.Tensor, key: torch.Tensor, scale: float) -> torch.Tensor:
    """``[B, H, L, D]`` → scores ``[B*Hkv, G, L, L]``. Mask stays out of this graph."""
    batch, hq, length, dim = query.shape
    hkv = key.shape[1]
    g = hq // hkv
    q = query.reshape(batch, hkv, g, length, dim).reshape(batch * hkv, g, length, dim)
    k = key.reshape(batch * hkv, 1, length, dim)
    return torch.matmul(q, k.transpose(-2, -1)) * scale


def _packed_pv(scores: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    batch, hkv, length, dim = value.shape
    g = scores.shape[1]
    v = value.reshape(batch * hkv, 1, length, dim)
    scores_max = torch.amax(scores, dim=-1, keepdim=True)
    # Dummy seqs (batch_bucket > num_seqs) have all-inf key_pad, so
    # scores - scores_max is NaN. Decoder documents the same hazard where an
    # in-graph store would publish it (spyre_attn mask_bs_bb[num_seqs:, 0]).
    # Safe here: unpack only gathers orig_query_lens rows, never those seqs.
    probs = torch.exp(scores - scores_max)
    out = torch.matmul(probs, v) / probs.sum(dim=-1, keepdim=True)
    return out.reshape(batch, hkv * g, length, dim)


def _packed_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Scatter-path attention. Compile QK and P·V separately; pad add is eager.

    Compiling ``matmul + mask`` lets Inductor rewrite to ``F.sdpa``, which drops
    ``attn_mask`` on Spyre (BGE cosine ~0.46).
    """
    device_type = query.device.type
    qk = _compile_if_spyre(_packed_qk_matmul, device_type)
    pv = _compile_if_spyre(_packed_pv, device_type)
    scores = qk(query, key, scale)
    if mask.shape != scores.shape:
        # GQA: ``[B*KV, 1, L, L]`` → ``[B*KV, G, L, L]``. Query-L is already dense.
        mask = mask.expand_as(scores).contiguous()
    return pv(scores + mask, value)


def _b1_dense_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    enable_gqa: bool,
    head_size_padded: int,
    head_size: int,
) -> torch.Tensor:
    """B=1 ``T==L`` full-bucket attention. Dest/unpack/mask unused. Compiled permute+SDPA."""
    query = _pad_head_dim_to_stick(query, head_size_padded)
    key = _pad_head_dim_to_stick(key, head_size_padded)
    value = _pad_head_dim_to_stick(value, head_size_padded)
    kernel = _compile_if_spyre(
        _b1_sdpa_kernel_gqa if enable_gqa else _b1_sdpa_kernel,
        query.device.type,
    )
    result = kernel(query, key, value, scale)
    if result.shape[-1] == head_size:
        return result
    if result.device.type == "spyre":
        result = convert(result, "cpu")
    return result[..., :head_size].contiguous()


def _index_copy(dst: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Eager on CPU; compiled ``index_copy_`` on Spyre (eager falls back / rejects)."""
    compiled = _compile_if_spyre(_index_copy_kernel, dst.device.type)
    return compiled(dst, index, src)


def _dest_on_flat_device(dest_idx: torch.Tensor, flat: torch.Tensor) -> torch.Tensor:
    """Spyre compiled index_copy_ wants int32; eager CPU wants int64."""
    if dest_idx.device.type == "spyre":
        return dest_idx
    if flat.device.type == "spyre":
        return convert(dest_idx.to(torch.int32), flat.device)
    return dest_idx.to(device=flat.device, dtype=torch.int64)


def _zeros_slot_major(
    rows: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """``[rows, H, D]`` zeros. Spyre pins rows at device position 0.

    ``empty`` + ``zero_`` stays on device. Pass size as one tuple —
    ``empty_with_layout`` rejects ``empty(B, L, H, D, device_layout=)``.
    """
    if device.type != "spyre":
        return torch.zeros(rows, num_heads, head_size, dtype=dtype, device=device)
    layout = slot_major_kv_layout(rows, num_heads, head_size, dtype)
    return torch.empty(  # ty: ignore[no-matching-overload]
        (rows, num_heads, head_size),
        dtype=dtype,
        device=device,
        device_layout=layout,
    ).zero_()


def _cached_encoder_workspace(
    attn_metadata: SpyreAttentionMetadata,
    attr: str,
    rows: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Reuse the step's slot-major scratch; allocate on first pack."""
    workspace = getattr(attn_metadata, attr)
    if (
        workspace is not None
        and workspace.shape == (rows, num_heads, head_size)
        and workspace.dtype == dtype
        and workspace.device.type == device.type
    ):
        return workspace
    workspace = _zeros_slot_major(rows, num_heads, head_size, dtype, device)
    setattr(attn_metadata, attr, workspace)
    return workspace


def gather_pack(
    flat: torch.Tensor,
    pack_indices: torch.Tensor,
    head_size_padded: int,
) -> torch.Tensor:
    """Pack via ``F.pad`` zero row + ``index_select`` of ``B×L``.

    Reference / tests. Serve uses ``scatter_pack``. ``B=1`` identity skips
    ``index_select``.
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
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack varlen ``[T, H, D]`` → ``[B, H, L, Dp]`` via compiled ``index_copy_``.

    ``dest_idx`` is ``[T]`` packed-row ids (host or device). Pad slots stay
    zeros. Body-pad dests write an extra dummy row (``B×L``), not CLS.

    ``B=1`` with ``T == L`` skips ``index_copy_``: the runner already padded
    the body to the SDPA length. The attention mask hides leftover pad tokens.

    Spyre workspace is slot-major so ``view(B, L, …)`` splits an outermost
    dim (decoder KV). Default-layout ``view`` after ``index_copy_`` scrambles
    ``B>1``. Serve caches ``workspace`` on the step and ``zero_``s it here so
    pad slots from the previous pack do not leak.

    ``permute.contiguous`` does **not** copy when ``H == 1`` (size-1 dim is
    ignored by ``is_contiguous``). If the result still aliases ``workspace``,
    clone so a later pack into the same buffer cannot clobber this tensor.
    """
    _t, num_heads, _d = flat.shape
    flat = _pad_head_dim_to_stick(flat, head_size_padded)
    # Fused QKV views are strided (BGE ``stride=(2304, 64, 1)``). Compiled
    # ``index_copy_`` from that layout writes the wrong rows.
    if not flat.is_contiguous():
        flat = flat.contiguous()
    if _is_b1_dense_body(batch, flat.shape[0], aligned_len):
        packed = flat.unsqueeze(0)
        return packed.permute(0, 2, 1, 3).contiguous()
    packed_rows = batch * aligned_len
    rows = packed_rows + 1
    # Extra row is the dummy dest. Slot-major prefix view is 2c on hardware.
    if (
        workspace is None
        or workspace.shape != (rows, num_heads, head_size_padded)
        or workspace.dtype != flat.dtype
        or workspace.device.type != flat.device.type
    ):
        workspace = _zeros_slot_major(
            rows,
            num_heads,
            head_size_padded,
            flat.dtype,
            flat.device,
        )
    else:
        workspace.zero_()
    _index_copy(workspace, _dest_on_flat_device(dest_idx, flat), flat)
    packed = workspace[:packed_rows].view(batch, aligned_len, num_heads, head_size_padded)
    packed = packed.permute(0, 2, 1, 3).contiguous()
    if packed.untyped_storage().data_ptr() == workspace.untyped_storage().data_ptr():
        packed = packed.clone()
    return packed


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
    if gathered.device.type == "spyre":
        gathered = convert(gathered, "cpu")
    return gathered[..., :head_size].contiguous()


def _create_compilable_encoder_attn(
    head_size_padded: int,
    enable_gqa: bool,
):
    """Factory for one pack→SDPA kernel, closed over per-layer constants.

    Mirrors the decoder's ``_create_compilable_page_attn``: everything that
    never varies per call for a given attention layer (head size, GQA) is a
    closure constant, not a runtime argument. Fusing pack and SDPA into one
    function lets ``_maybe_compile`` wrap the whole sequence in a single
    ``torch.compile``, so Dynamo checks guards once per call instead of once
    per inner op — the latter is what makes eager per-op dispatch on Spyre
    recompile on content changes alone (spyre-inference#775 follow-up).

    ``gather_unpack`` stays outside this kernel and runs eager, called
    separately by ``forward()``: compiling it together with SDPA in one graph
    hits a torch-spyre layout-propagation limit ("Incompatible host_size and
    dim_order", `torch_spyre/_inductor/propagate_layouts.py`) regardless of
    mask or GQA — isolated experimentally, not yet root-caused inside
    torch-spyre. Pack+SDPA alone compiles cleanly; that's most of the benefit,
    since it collapses 3 gather_pack calls plus SDPA from 4+ separately
    guard-checked eager dispatches down to 1.
    """

    def encoder_attn(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        q_pack_idx: torch.Tensor,
        kv_pack_idx: torch.Tensor,
        mask: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        q_batched = gather_pack(query, q_pack_idx, head_size_padded)
        k_batched = gather_pack(key, kv_pack_idx, head_size_padded)
        v_batched = gather_pack(value, kv_pack_idx, head_size_padded)

        sdpa_kwargs: dict = {"is_causal": False, "scale": scale}
        if enable_gqa:
            sdpa_kwargs["enable_gqa"] = True
        return F.scaled_dot_product_attention(
            q_batched, k_batched, v_batched, attn_mask=mask, **sdpa_kwargs
        )

    return encoder_attn


def _indices_for_device(indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move pack dest / unpack indices onto ``device`` once.

    Spyre ``index_select`` and compiled ``index_copy_`` want int32; CPU wants int64.
    """
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
    num_kv_heads: int,
    target_device: torch.device,
    cached_encoder_shapes: list[tuple[int, int]],
    cached_max_num_seqs: int,
    cached_max_model_len: int,
    cached_max_num_batched_tokens: int,
) -> None:
    """Build scatter dest + unpack + key-pad once per step; later layers reuse them."""
    if attn_metadata.encoder_pack_batch is not None:
        return

    qsl = attn_metadata.query_start_loc.cpu()
    q_starts = qsl[:-1].tolist()
    qsl_lens = torch.diff(qsl).tolist()
    kv_lens = attn_metadata.seq_lens.cpu().tolist()
    num_seqs = attn_metadata.num_seqs
    q_starts = q_starts[:num_seqs]
    qsl_lens = qsl_lens[:num_seqs]
    kv_lens = kv_lens[:num_seqs]
    # Pick (B, L) from padded qsl/seq so identity T==L still matches the body
    # bucket. Cap dest/mask with num_actual_tokens after that (BGE: both
    # cu_seqlens and seq_lens can be the bucket).
    max_len = max(_content_query_lens(qsl_lens, kv_lens), default=0)
    pair = pick_encoder_attention_shape(
        num_seqs,
        max_len,
        cached_encoder_shapes,
        cached_max_num_seqs,
        cached_max_model_len,
        cached_max_num_batched_tokens,
    )
    if pair is not None:
        batch_bucket, aligned_len = pair
    else:
        batch_bucket, aligned_len = num_seqs, _align_up(max_len)
        logger.warning_once(
            "No warmed encoder attention shape covers num_seqs=%d, "
            "max_query_len=%d; falling back to (%d, %d), which triggers a "
            "runtime recompile. Widen --max-num-batched-tokens or lower "
            "--max-num-seqs so every batch bucket has a warmed cell.",
            num_seqs,
            max_len,
            batch_bucket,
            aligned_len,
        )
    query_lens = _content_query_lens(qsl_lens, kv_lens, num_actual_tokens=n)
    orig_q_starts = q_starts
    orig_query_lens = query_lens
    fused = _is_b1_fused_sdpa(
        batch_bucket,
        padded_tokens,
        aligned_len,
        orig_query_lens[0] if orig_query_lens else 0,
    )
    if fused:
        # No dest, unpack, or mask. Compiled SDPA has no attn_mask.
        attn_metadata.encoder_pack_batch = batch_bucket
        attn_metadata.encoder_pack_len = aligned_len
        attn_metadata.encoder_fused_sdpa = True
        return
    if batch_bucket > num_seqs:
        q_starts = q_starts + [n] * (batch_bucket - num_seqs)
        query_lens = query_lens + [0] * (batch_bucket - num_seqs)
        kv_lens = kv_lens + [0] * (batch_bucket - num_seqs)

    dummy_row = batch_bucket * aligned_len
    kv_pack_lens = [min(q, k) for q, k in zip(query_lens, kv_lens)]
    q_dest = host_scatter_pack_dest(q_starts, query_lens, aligned_len, padded_tokens, dummy_row)
    kv_dest = host_scatter_pack_dest(q_starts, kv_pack_lens, aligned_len, padded_tokens, dummy_row)
    unpack_idx = host_unpack_indices(orig_q_starts, orig_query_lens, aligned_len, padded_tokens)
    mask_cpu = build_attention_mask(
        batch_bucket,
        aligned_len,
        query_lens,
        kv_lens,
        dtype=query.dtype,
        device=torch.device("cpu"),
    )
    key_pad = host_key_pad_mask(mask_cpu, num_kv_heads)
    # Do not H2D ``mask_cpu`` ([B, 1, L, L]). Forward only needs (B, L) plus
    # this key-pad; at B=8, L=512 the unused copy is ~4 MB fp16 per step.
    if target_device.type == "spyre":
        key_pad = convert(key_pad, target_device)
    else:
        key_pad = key_pad.to(target_device)

    # B=1 T==L with live pad still identity-packs; dest/unpack stay on host.
    b1_dense = _is_b1_dense_body(batch_bucket, padded_tokens, aligned_len)
    if b1_dense:
        attn_metadata.encoder_q_pack_idx = q_dest
        attn_metadata.encoder_kv_pack_idx = kv_dest
        attn_metadata.encoder_unpack_idx = unpack_idx
    else:
        attn_metadata.encoder_q_pack_idx = _indices_for_device(q_dest, target_device)
        attn_metadata.encoder_kv_pack_idx = _indices_for_device(kv_dest, target_device)
        attn_metadata.encoder_unpack_idx = _indices_for_device(unpack_idx, target_device)
    attn_metadata.encoder_pack_batch = batch_bucket
    attn_metadata.encoder_pack_len = aligned_len
    attn_metadata.encoder_fused_sdpa = False
    attn_metadata.encoder_key_pad_mask = key_pad


def _create_compilable_encoder_seq_attn(
    head_size_padded: int,
    enable_gqa: bool,
    store: bool = False,
):
    """Factory for one sequence's pack->SDPA kernel, mirroring the decoder's
    ``_create_compilable_page_attn``: per-sequence rather than per-batch, so
    the only shape axis that varies is ``aligned_len`` (the cache key in
    ``_get_encoder_seq_attn_fn``) instead of the dense grid's ``(B, L)`` pair.

    Every tensor here is sized on ``aligned_len`` (a bucket), never on the real
    prompt length: under ``dynamic=False`` a real-length-dependent shape would
    add a Dynamo specialization per distinct prompt length, and Dynamo walks the
    resulting guard chain linearly on every later call, so throughput decays as
    more distinct lengths are seen.

    With ``store``, the write-back is fused here as ``index_copy_`` over the
    full ``aligned_len`` extent, matching the decoder's ``store_mode="index"``.
    That keeps the real length out of the caller's write too, and it must be
    fused rather than eager: eager ``index_copy_`` rejects an int32 index and
    silently falls back to CPU with an int64 one (see
    ``SpyreAttentionImpl._reshape_fn``). Rows past the real length duplicate the
    last real row's index, and the kv-only mask (see
    ``build_seq_attention_mask``) makes their values identical to that row's, so
    ``index_copy_``'s undefined write order for duplicate indices is harmless.
    """

    def encoder_seq_attn(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        q_row_idx: torch.Tensor,
        kv_row_idx: torch.Tensor,
        mask: torch.Tensor,
        scale: float,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # query/key/value are the whole padded batch tensor; gather (via
        # select_rows, which picks int32+convert on Spyre vs. int64 on CPU)
        # reads this sequence's own rows rather than relying on a slice's
        # storage_offset (torch-spyre#3770).
        q = _pad_head_dim_to_stick(select_rows(query, q_row_idx), head_size_padded)
        k = _pad_head_dim_to_stick(select_rows(key, kv_row_idx), head_size_padded)
        v = _pad_head_dim_to_stick(select_rows(value, kv_row_idx), head_size_padded)
        q = q.unsqueeze(0).transpose(1, 2)  # [1, H, L, Dp]
        k = k.unsqueeze(0).transpose(1, 2)
        v = v.unsqueeze(0).transpose(1, 2)

        sdpa_kwargs: dict = {"is_causal": False, "scale": scale}
        if enable_gqa:
            sdpa_kwargs["enable_gqa"] = True
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, **sdpa_kwargs)
        # [1, H, L, Dp] -> [L, H, Dp]. The trailing .reshape (not .view) forces
        # a contiguous copy after the transpose, mirroring the decoder kernel's
        # identical reshape/transpose/reshape.
        num_heads_out = attn_out.shape[1]
        head_dim_out = attn_out.shape[-1]
        result = attn_out.transpose(1, 2).reshape(-1, num_heads_out, head_dim_out)
        if store:
            assert out is not None
            out.index_copy_(0, q_row_idx, result)
            return out
        return result

    return encoder_seq_attn


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


def build_seq_attention_mask(
    aligned_len: int,
    kv_len: int,
    dtype: torch.dtype,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Single-sequence additive mask ``[1, 1, L, L]``, keyed on KV validity only.

    Deliberately does NOT mask padded *query* rows, unlike
    ``build_attention_mask``. Padded query positions gather the last real query
    row (``_build_seq_row_index`` clamps them), so leaving their row unmasked
    makes their output identical to the last real row's — the invariant the
    fused ``index_copy_`` store relies on, and the same one the decoder's mask
    tiles maintain. Masking them instead would leave them a uniform average of
    V (``finfo.min`` everywhere softmaxes to uniform, not NaN), which the store
    would then scatter over the last real row.

    Real query rows are unaffected: for those, ``build_attention_mask`` already
    reduces to exactly this KV-validity row.

    Materialized as a full ``[1, 1, L, L]`` on the host, not a broadcastable
    ``[1, 1, 1, L]``: Spyre cannot broadcast-scatter a mask across the query
    axis on device (see ``build_attention_mask``).
    """
    if device is None:
        device = torch.device("cpu")
    elif not isinstance(device, torch.device):
        device = torch.device(device)

    kv_pos = torch.arange(aligned_len, dtype=torch.int32)
    k_ok = kv_pos < kv_len
    row = torch.where(
        k_ok,
        torch.zeros((), dtype=dtype),
        torch.tensor(torch.finfo(dtype).min, dtype=dtype),
    )
    mask = row.unsqueeze(0).expand(aligned_len, aligned_len).contiguous()
    mask = mask.unsqueeze(0).unsqueeze(0)
    if device.type == "spyre":
        return convert(mask, device)
    return mask.to(device)


class SpyreEncoderAttentionImpl(SpyreAttentionImpl):
    """Bidirectional encoder self-attention (no KV cache).

    The platform selects this impl for ENCODER/ENCODER_ONLY layers (see
    ``TorchSpyrePlatform.get_attn_backend_cls``), so there is no per-call
    ``attn_type`` branch. Setup is shared with the paged decoder impl; forward
    packs Q/K/V with scatter into a slot-major workspace (decoder KV
    layout; extra dummy row for body-pad), then attention and gather-unpack.
    ``B=1`` ``T == L`` with a full prompt compiles permute+SDPA; live pad
    uses packed QK (matmul and P·V compiled apart so Inductor cannot fuse
    to ``F.sdpa``).
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
            len_bucket=default_encoder_len_buckets(self._cached_max_model_len),
        )
        # One compiled pack→SDPA kernel per (batch_bucket, aligned_len),
        # mirroring the decoder's self._attn_fns/_decode_fns.
        self._encoder_attn_fns: dict[tuple[int, int], object] = {}
        # Per-sequence loop path (default, see envs.SPYRE_BUCKETED_ENCODE):
        # one compiled kernel per (aligned_len, store), mirroring the decoder's
        # default per-sequence self._attn_fns cache.
        self._encoder_seq_attn_fns: dict[tuple[int, bool], object] = {}

    def _get_encoder_seq_attn_fn(
        self,
        aligned_len: int,
        head_size_padded: int,
        enable_gqa: bool,
        store: bool,
    ):
        key = (aligned_len, store)
        if key not in self._encoder_seq_attn_fns:
            self._encoder_seq_attn_fns[key] = _maybe_compile(
                _create_compilable_encoder_seq_attn(head_size_padded, enable_gqa, store=store),
                self._compile_attn,
            )
        return self._encoder_seq_attn_fns[key]

    def _get_encoder_attn_fn(
        self,
        batch_bucket: int,
        aligned_len: int,
        head_size_padded: int,
        enable_gqa: bool,
    ):
        key = (batch_bucket, aligned_len)
        if key not in self._encoder_attn_fns:
            self._encoder_attn_fns[key] = _maybe_compile(
                _create_compilable_encoder_attn(head_size_padded, enable_gqa),
                self._compile_attn,
            )
        return self._encoder_attn_fns[key]

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
        enable_gqa = num_kv_heads != num_heads

        if query.device.type != target_device.type:
            query = convert(query, target_device.type)
            key = convert(key, target_device.type)
            value = convert(value, target_device.type)

        use_flat_write = target_device.type == "spyre" and head_size % ENCODER_SEQ_ALIGNMENT != 0

        qsl = attn_metadata.query_start_loc.cpu()
        q_starts = qsl[:-1].tolist()
        qsl_lens = torch.diff(qsl).tolist()
        kv_lens = attn_metadata.seq_lens.cpu().tolist()
        num_seqs = attn_metadata.num_seqs
        q_starts = q_starts[:num_seqs]
        qsl_lens = qsl_lens[:num_seqs]
        kv_lens = kv_lens[:num_seqs]
        query_lens = _content_query_lens(qsl_lens, kv_lens, num_actual_tokens=n)

        if not envs.SPYRE_BUCKETED_ENCODE:
            len_ladder = default_encoder_len_buckets(self._cached_max_model_len)
            fused_store_ok = (
                self._compile_attn
                and head_size == head_size_padded
                and output.dtype == query.dtype
                and output.storage_offset() == 0
                and output.is_contiguous()
            )
            if self._compile_attn and not fused_store_ok:
                logger.warning_once(
                    "Encoder attention is writing back eagerly despite being "
                    "compiled (head_size=%d/%d, dtype=%s/%s, offset=%d, "
                    "contiguous=%s). The eager write is sized on each request's "
                    "real prompt length, so every distinct length adds a "
                    "torch.compile specialization and throughput decays as more "
                    "lengths are seen.",
                    head_size,
                    head_size_padded,
                    query.dtype,
                    output.dtype,
                    output.storage_offset(),
                    output.is_contiguous(),
                )

            for seq_idx in range(num_seqs):
                q_start = q_starts[seq_idx]
                real_len = query_lens[seq_idx]
                kv_real_len = min(real_len, kv_lens[seq_idx])
                aligned_len = next_bucket(real_len, len_ladder)
                q_row_idx = convert(
                    _build_seq_row_index(q_start, real_len, aligned_len).to(torch.int32),
                    target_device,
                )
                if kv_real_len == real_len:
                    kv_row_idx = q_row_idx
                else:
                    kv_row_idx = convert(
                        _build_seq_row_index(q_start, kv_real_len, aligned_len).to(torch.int32),
                        target_device,
                    )
                mask = build_seq_attention_mask(
                    aligned_len,
                    kv_real_len,
                    dtype=query.dtype,
                    device=target_device,
                )
                encoder_seq_attn_fn = self._get_encoder_seq_attn_fn(
                    aligned_len, head_size_padded, enable_gqa, store=fused_store_ok
                )
                attn_out = encoder_seq_attn_fn(
                    query,
                    key,
                    value,
                    q_row_idx,
                    kv_row_idx,
                    mask,
                    scale,
                    out=output if fused_store_ok else None,
                )
                if fused_store_ok:
                    continue

                if attn_out.shape[-1] != head_size:
                    if attn_out.device.type == "spyre":
                        attn_out = convert(attn_out, "cpu")
                    attn_out = attn_out[..., :head_size].contiguous()
                result = attn_out[:real_len]
                if result.dtype != output.dtype:
                    result = convert(result, dtype=output.dtype)

                if use_flat_write:
                    if result.device.type == "spyre":
                        result = convert(result, "cpu")
                    src = convert(
                        result.reshape(real_len, -1).contiguous(),
                        target_device.type,
                        output.dtype,
                    )
                    output[q_start : q_start + real_len].reshape(real_len, -1).copy_(src)
                else:
                    if result.device.type != output.device.type:
                        result = convert(result, output.device)
                    output[q_start : q_start + real_len] = result

            return output

        _ensure_encoder_pack(
            attn_metadata,
            padded_tokens=padded_tokens,
            n=n,
            query=query,
            num_kv_heads=num_kv_heads,
            target_device=target_device,
            cached_encoder_shapes=self._cached_encoder_shapes,
            cached_max_num_seqs=self._cached_max_num_seqs,
            cached_max_model_len=self._cached_max_model_len,
            cached_max_num_batched_tokens=self._cached_max_num_batched_tokens,
        )

        if attn_metadata.encoder_fused_sdpa:
            result = _b1_dense_attention(
                query,
                key,
                value,
                scale,
                enable_gqa,
                head_size_padded,
                head_size,
            )
        else:
            batch = attn_metadata.encoder_pack_batch
            aligned_len = attn_metadata.encoder_pack_len
            q_pack = attn_metadata.encoder_q_pack_idx
            kv_pack = attn_metadata.encoder_kv_pack_idx
            unpack_idx = attn_metadata.encoder_unpack_idx
            key_pad_mask = attn_metadata.encoder_key_pad_mask
            assert batch is not None and aligned_len is not None
            assert q_pack is not None and kv_pack is not None and unpack_idx is not None
            assert key_pad_mask is not None
            rows = batch * aligned_len + 1
            q_ws = _cached_encoder_workspace(
                attn_metadata,
                "encoder_q_workspace",
                rows,
                num_heads,
                head_size_padded,
                query.dtype,
                query.device,
            )
            kv_ws = _cached_encoder_workspace(
                attn_metadata,
                "encoder_kv_workspace",
                rows,
                num_kv_heads,
                head_size_padded,
                key.dtype,
                key.device,
            )
            v_ws = _cached_encoder_workspace(
                attn_metadata,
                "encoder_v_workspace",
                rows,
                num_kv_heads,
                head_size_padded,
                value.dtype,
                value.device,
            )
            q_batched = scatter_pack(
                query, q_pack, batch, aligned_len, head_size_padded, workspace=q_ws
            )
            k_batched = scatter_pack(
                key, kv_pack, batch, aligned_len, head_size_padded, workspace=kv_ws
            )
            v_batched = scatter_pack(
                value, kv_pack, batch, aligned_len, head_size_padded, workspace=v_ws
            )
            attn_out = _packed_masked_attention(
                q_batched,
                k_batched,
                v_batched,
                key_pad_mask,
                scale,
            )
            result = gather_unpack(attn_out, unpack_idx, head_size)
        if result.dtype != output.dtype:
            result = convert(result, dtype=output.dtype)

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
