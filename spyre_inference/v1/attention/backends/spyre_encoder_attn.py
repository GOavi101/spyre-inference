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

Varlen flash: each request is a contiguous slice of the packed ``[T, H, D]``
list (``query_start_loc``). A compiled online-softmax kernel reads that slice
(padded to a stick-aligned length) and never materialises a dense ``(B, L)``
grid or ``[B, 1, L, L]`` mask. Stays behind ``unified_attention``.

Slices are cut on the host and sent H2D so the kernel always sees the same
tensor provenance, whatever the batch packing looks like. See ``_seq_slice``.
"""

from __future__ import annotations

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

# Pad seq length *and* head dim to the Spyre stick (64 fp16 elements).
# D-aligned keeps QK^T stick-aligned so Inductor never enters
# insert_bmm_padding (MiniLM head_size=32).
ENCODER_SEQ_ALIGNMENT = 64


def _align_up(n: int, align: int = ENCODER_SEQ_ALIGNMENT) -> int:
    return max(align, (n + align - 1) // align * align)


def host_key_valid(aligned_len: int, kv_len: int, dtype: torch.dtype) -> torch.Tensor:
    """1D key mask ``[L]``: 1.0 real, 0.0 pad. No ``-inf`` (Spyre fp16 drops it)."""
    valid = torch.zeros(aligned_len, dtype=dtype)
    if kv_len > 0:
        valid[: min(kv_len, aligned_len)] = 1
    return valid


# Older name used by tests / probes that have not been recopied.
host_key_bias = host_key_valid


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


def pad_seq_to_aligned(seq: torch.Tensor, aligned_len: int) -> torch.Tensor:
    """Zero-pad a contiguous sequence ``[n, ...]`` to ``[aligned_len, ...]``."""
    n = seq.shape[0]
    if n == aligned_len:
        return seq
    if n > aligned_len:
        raise ValueError(f"seq length {n} exceeds aligned_len {aligned_len}")
    pad = seq.new_zeros((aligned_len - n, *seq.shape[1:]))
    return torch.cat([seq, pad], dim=0)


def _index_copy_kernel(dst: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Tiny mutation, compiled alone — do not fuse with the flash matmuls."""
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


def _create_encoder_flash_kernel(num_heads: int, num_kv_heads: int):
    """Aligned ``L×L`` GEMM plus same-shape ``[H,L,L]`` 0/1 (no 1D broadcast)."""
    repeat = num_heads // num_kv_heads if num_heads != num_kv_heads else 1
    gqa = repeat > 1
    pad_logit = -1.0e4

    def _kernel(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        keep: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        qh = query.permute(1, 0, 2) * scale
        kh = key.permute(1, 0, 2)
        vh = value.permute(1, 0, 2)
        if gqa:
            kh = kh.repeat_interleave(repeat, dim=0)
            vh = vh.repeat_interleave(repeat, dim=0)
        scores = torch.matmul(qh, kh.transpose(-1, -2))
        scores = scores + (1.0 - keep) * pad_logit
        scores_max = scores.amax(dim=-1)
        probs = torch.exp(scores - scores_max.unsqueeze(-1)) * keep
        tiny = torch.finfo(query.dtype).tiny
        den = probs.sum(dim=-1).clamp(min=tiny).unsqueeze(-1)
        out = torch.matmul(probs, vh) / den
        return out.permute(1, 0, 2).contiguous()

    return _kernel


def host_score_keep(
    num_heads: int, aligned_len: int, kv_len: int, dtype: torch.dtype
) -> torch.Tensor:
    """``[H, L, L]`` 1.0 on real key columns, 0.0 on pad. Built on CPU."""
    keep = torch.zeros(num_heads, aligned_len, aligned_len, dtype=dtype)
    if kv_len > 0:
        keep[:, :, : min(kv_len, aligned_len)] = 1
    return keep


def encoder_flash_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_len: int,
    scale: float,
    *,
    compile_enabled: bool = False,
    cache: dict[tuple[int, int, int, int], object] | None = None,
) -> torch.Tensor:
    """Softmax attention on one stick-aligned sequence ``[L, H, D]``.

    Builds a same-rank ``[H, L, L]`` keep tensor on the host (Spyre slice writes
    are corrupt) and passes it in so the add is not a 1D broadcast.
    """
    aligned_len, num_heads, head_size = query.shape
    num_kv_heads = key.shape[1]
    keep = host_score_keep(num_heads, aligned_len, int(kv_len), query.dtype)
    if query.device.type == "spyre":
        keep = convert(keep, query.device)
    else:
        keep = keep.to(device=query.device)
    key_id = (aligned_len, num_heads, num_kv_heads, head_size)
    fn: object | None = None if cache is None else cache.get(key_id)
    if fn is None:
        fn = _maybe_compile(
            _create_encoder_flash_kernel(num_heads, num_kv_heads),
            compile_enabled,
        )
        if cache is not None:
            cache[key_id] = fn
    return fn(query, key, value, keep, scale)  # ty: ignore[invalid-call]


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
    opaque ``unified_attention`` op: varlen flash on the packed list, no
    ``(B, L)`` gather grid.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._flash_fns: dict[tuple[int, int, int, int], object] = {}

    def _seq_slice(
        self,
        host_flat: torch.Tensor,
        start: int,
        length: int,
        aligned_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """One padded sequence ``[L, H, D]``, host-sliced then sent H2D.

        Every sequence must reach the kernel the same way. A device-side slice
        that happens to span the whole packed batch (one request filling its
        token bucket) is a no-op, so the kernel would receive the projection
        output itself, and Spyre cannot restickify that layout for the QK^T
        operand. A fresh H2D tensor lets the compiler pick the layout, which is
        what the probe validated.
        """
        seq = pad_seq_to_aligned(host_flat[start : start + length], aligned_len)
        return convert(seq, device)

    def _store_seq(
        self,
        output: torch.Tensor,
        start: int,
        length: int,
        result: torch.Tensor,
    ) -> None:
        if length <= 0:
            return
        src = result[:length]
        if output.device.type != "spyre" and start == 0 and length == output.shape[0]:
            output.copy_(src)
            return
        dest = torch.arange(start, start + length, dtype=torch.int64, device=output.device)
        if src.device.type != output.device.type:
            src = convert(src, output.device)
        _index_copy(output, dest, src)

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

        # Body may 1D-pad past num_actual_tokens; extra rows are not sequences.
        n = attn_metadata.num_actual_tokens
        scale = self.scale

        qsl = attn_metadata.query_start_loc.cpu()
        q_starts = qsl[:-1].tolist()
        query_lens = torch.diff(qsl).tolist()
        kv_lens = attn_metadata.seq_lens.cpu().tolist()

        head_size = query.shape[2]
        head_size_padded = _align_up(head_size)

        target_device = output.device
        # Slicing happens on the host: a Spyre slice at start>0 reads corrupt
        # rows, and slicing on device leaves whole-batch sequences unmaterialised
        # (see _seq_slice). One D2H per tensor here, one H2D per sequence below.
        q_host = _pad_head_dim_to_stick(convert(query, "cpu"), head_size_padded)
        k_host = _pad_head_dim_to_stick(convert(key, "cpu"), head_size_padded)
        v_host = _pad_head_dim_to_stick(convert(value, "cpu"), head_size_padded)

        for start, q_len, kv_len in zip(q_starts, query_lens, kv_lens):
            if start >= n or q_len <= 0:
                continue
            q_len = min(int(q_len), n - int(start))
            kv_len = min(int(kv_len), q_len)
            aligned_len = _align_up(max(q_len, kv_len, 1))
            q_seq = self._seq_slice(q_host, int(start), q_len, aligned_len, target_device)
            k_seq = self._seq_slice(k_host, int(start), kv_len, aligned_len, target_device)
            v_seq = self._seq_slice(v_host, int(start), kv_len, aligned_len, target_device)
            attn = encoder_flash_sdpa(
                q_seq,
                k_seq,
                v_seq,
                kv_len,
                scale,
                compile_enabled=self._compile_attn,
                cache=self._flash_fns,
            )
            if attn.shape[-1] != head_size:
                if attn.device.type == "spyre":
                    attn = convert(attn, "cpu")
                attn = attn[..., :head_size].contiguous()
            self._store_seq(output, int(start), q_len, attn)

        return output


class SpyreEncoderAttentionBackend(SpyreAttentionBackend):
    """Encoder-only (no KV cache) variant of the Spyre backend."""

    # These layers have no KV cache, but vLLM still hands encoder-only specs a
    # zero-filled slot mapping, so upstream must skip `unified_kv_cache_update` entirely.
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_impl_cls() -> type[SpyreEncoderAttentionImpl]:
        return SpyreEncoderAttentionImpl
