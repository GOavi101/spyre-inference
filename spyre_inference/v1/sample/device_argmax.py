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

"""On-device top-1 selection without ``argmax`` / ``topk``.

Spyre constraints:
    - ``aten.argmax`` is a CPU fallback (copies full logits).
    - ``topk`` emits fp16 indices (wrong past 2048) and is limited to k <= 4.

Split the index into fp16-exact digits and reduce with native ops
(``amax``, ``eq``, ``where``, ``neg``)::

    index = hi * RADIX + lo

Tie-breaking matches ``torch.argmax`` (lowest index). Used by Stage 2
pure-greedy sampling in ``TorchSpyreModelRunner``.
"""

from functools import lru_cache

import torch
import torch._dynamo as dynamo

# Digits must stay below the largest integer fp16 represents exactly.
_RADIX = 128
_FP16_EXACT_INT_MAX = 2048


def _digit_constants(
    padded_vocab: int,
    valid_vocab: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Build resident per-column constants for one logits shape.

    Returns:
        ``(hi_pos, lo_pos, valid_mask, hi_sentinel, lo_sentinel, mask_fill)``.
        Sentinels and fills are device tensors because Spyre registers only
        ``aten.where.self`` (not the Scalar overloads).
    """
    hi_sentinel = float((padded_vocab - 1) // _RADIX + 1)
    if dtype == torch.float16 and hi_sentinel > _FP16_EXACT_INT_MAX:
        raise ValueError(
            f"vocab {padded_vocab} needs a high digit up to {hi_sentinel:.0f}, "
            f"which fp16 cannot represent exactly (limit {_FP16_EXACT_INT_MAX}). "
            f"Raise _RADIX above {_RADIX} to rebalance the two digits."
        )

    def scalar(value: float) -> torch.Tensor:
        return torch.full((1, 1), value, dtype=dtype).to(device)

    # arange is a CPU fallback on Spyre; build once on host, keep on device.
    positions = torch.arange(padded_vocab, dtype=torch.int64)
    hi_pos = (positions // _RADIX).to(dtype).unsqueeze(0).to(device)
    lo_pos = (positions % _RADIX).to(dtype).unsqueeze(0).to(device)

    valid_mask = None
    if valid_vocab < padded_vocab:
        valid_mask = (positions < valid_vocab).unsqueeze(0).to(device)

    return (
        hi_pos,
        lo_pos,
        valid_mask,
        scalar(hi_sentinel),
        scalar(float(_RADIX)),
        scalar(torch.finfo(dtype).min),
    )


@lru_cache(maxsize=8)
def _cached_constants(
    padded_vocab: int,
    valid_vocab: int,
    device_str: str,
    dtype: torch.dtype,
):
    return _digit_constants(padded_vocab, valid_vocab, torch.device(device_str), dtype)


@dynamo.disable
def _get_constants(logits: torch.Tensor, valid_vocab: int | None):
    """Fetch cached constants outside Dynamo's trace.

    Dynamo would otherwise capture int64 construction and lower it with
    ``spyre::to_dtype_cpu`` (no CPU kernel).
    """
    padded_vocab = logits.shape[-1]
    if valid_vocab is None:
        valid_vocab = padded_vocab
    if not 0 < valid_vocab <= padded_vocab:
        raise ValueError(f"valid_vocab {valid_vocab} outside (0, {padded_vocab}]")
    return _cached_constants(padded_vocab, valid_vocab, str(logits.device), logits.dtype)


def _reduce(
    logits: torch.Tensor,
    hi_pos: torch.Tensor,
    lo_pos: torch.Tensor,
    valid_mask: torch.Tensor | None,
    hi_sentinel: torch.Tensor,
    lo_sentinel: torch.Tensor,
    mask_fill: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-tensor reduction that torch.compile can lower to Spyre."""
    if valid_mask is not None:
        logits = torch.where(valid_mask, logits, mask_fill)

    # amax returns an element of the input, so == is exact (argmax tie set).
    is_max = logits == logits.amax(dim=-1, keepdim=True)

    # min(x) = -max(-x); amin is not in Spyre's eager op registry.
    hi_sel = torch.where(is_max, hi_pos, hi_sentinel)
    hi_min = -(-hi_sel).amax(dim=-1, keepdim=True)

    lo_sel = torch.where(hi_sel == hi_min, lo_pos, lo_sentinel)
    lo_min = -(-lo_sel).amax(dim=-1, keepdim=True)

    return hi_min, lo_min


def argmax_digits(
    logits: torch.Tensor,
    valid_vocab: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce ``[batch, vocab]`` logits to the two digits of the top-1 index.

    Args:
        logits: ``[batch, vocab]`` scores on any device.
        valid_vocab: Real vocabulary size when ``logits`` carries LM-head
            alignment padding (zero columns that would otherwise win).

    Returns:
        ``(hi, lo)``, each ``[batch, 1]`` in the input dtype on the input device.
    """
    if logits.ndim != 2:
        raise ValueError(f"expected [batch, vocab] logits, got shape {tuple(logits.shape)}")

    return _reduce(logits, *_get_constants(logits, valid_vocab))


def combine_digits(hi: torch.Tensor, lo: torch.Tensor) -> torch.Tensor:
    """Recombine digits into ``[batch]`` int64 token ids (host)."""
    return (hi.to(torch.int64) * _RADIX + lo.to(torch.int64)).squeeze(-1)


def greedy_token_ids(
    logits: torch.Tensor,
    valid_vocab: int | None = None,
) -> torch.Tensor:
    """Reduce on the logits' device; return ``[batch]`` token ids on CPU."""
    hi, lo = argmax_digits(logits, valid_vocab=valid_vocab)
    return combine_digits(hi.cpu(), lo.cpu())
