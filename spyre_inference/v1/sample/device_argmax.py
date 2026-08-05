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

"""Top-1 selection over the vocab dim without `argmax` or `topk`.

Greedy decoding currently copies the whole ``[batch, vocab]`` logits tensor to
the host so upstream vLLM can call ``logits.argmax(dim=-1)`` there. Neither of
the obvious on-device replacements works at the current torch-spyre pin:

* ``aten.argmax.default`` is registered as a CPU fallback, so it copies the
  logits anyway.
* ``topk`` is limited to ``k <= 4``, runs single-core, and -- decisively --
  ``lower_topkindex`` builds its output with ``dst_dtype=x.get_dtype()``, so
  fp16 logits yield an fp16 *index* buffer. fp16 only represents integers
  exactly up to 2048 (spacing is 16 near a 32k vocab), so the index comes back
  rounded to the wrong token.

The real constraint is representational, not algorithmic: the reduction is
fine, but a vocab index does not fit exactly in fp16. So split the index into
two digits that individually stay inside the exact range::

    index = hi * RADIX + lo        hi < ceil(V / RADIX), lo < RADIX

and reduce each digit separately using only ops that torch-spyre runs natively
(``amax``, ``eq``, ``where``, ``neg``). Device-to-host traffic drops from
``[batch, vocab]`` to ``[batch, 2]``; the two digits are recombined in int64 on
the host, where the arithmetic is exact.

Tie-breaking matches ``torch.argmax`` (lowest index wins), which matters
because fp16 ties across a large vocab are common.

This module is the reduction used by Stage 2. ``TorchSpyreModelRunner`` calls
:func:`greedy_token_ids` for pure-greedy batches (see
:mod:`spyre_inference.v1.sample.pure_greedy`). Non-greedy / logprobs / penalties
/ masks / grammar still need full logits on the host. Under TP>1 a future
``get_top_tokens``-style path can avoid the full-vocab all-gather.
"""

from functools import lru_cache

import torch
import torch._dynamo as dynamo

# Both digits must stay below the largest integer fp16 represents exactly.
_RADIX = 128
_FP16_EXACT_INT_MAX = 2048


def _digit_constants(
    padded_vocab: int,
    valid_vocab: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, float, float]:
    """Build the resident per-column constants for one logits shape.

    Returns ``(hi_pos, lo_pos, valid_mask, hi_sentinel, lo_sentinel)``. The
    position tensors are ``[1, padded_vocab]`` so they broadcast over the batch.
    ``valid_mask`` is None when there is no padding to exclude.
    """
    hi_sentinel = float((padded_vocab - 1) // _RADIX + 1)
    lo_sentinel = float(_RADIX)

    if dtype == torch.float16 and hi_sentinel > _FP16_EXACT_INT_MAX:
        raise ValueError(
            f"vocab {padded_vocab} needs a high digit up to {hi_sentinel:.0f}, "
            f"which fp16 cannot represent exactly (limit {_FP16_EXACT_INT_MAX}). "
            f"Raise _RADIX above {_RADIX} to rebalance the two digits."
        )

    # arange is a CPU fallback on Spyre, so build on the host once and keep the
    # result resident on the device; this is setup cost, not per-step cost.
    positions = torch.arange(padded_vocab, dtype=torch.int64)
    hi_pos = (positions // _RADIX).to(dtype).unsqueeze(0).to(device)
    lo_pos = (positions % _RADIX).to(dtype).unsqueeze(0).to(device)

    valid_mask = None
    if valid_vocab < padded_vocab:
        valid_mask = (positions < valid_vocab).unsqueeze(0).to(device)

    return hi_pos, lo_pos, valid_mask, hi_sentinel, lo_sentinel


@lru_cache(maxsize=8)
def _cached_constants(
    padded_vocab: int,
    valid_vocab: int,
    device_str: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, float, float]:
    return _digit_constants(padded_vocab, valid_vocab, torch.device(device_str), dtype)


@dynamo.disable
def _get_constants(logits, valid_vocab):
    """Fetch cached constants outside of Dynamo's trace.

    Dynamo traces through lru_cache and captures the int64 tensor
    construction, which Inductor then lowers with ``spyre::to_dtype_cpu``
    -- an op with no CPU kernel. Wrapping the lookup in ``@dynamo.disable``
    makes the tensors opaque graph inputs instead.
    """
    padded_vocab = logits.shape[-1]
    if valid_vocab is None:
        valid_vocab = padded_vocab
    if not 0 < valid_vocab <= padded_vocab:
        raise ValueError(f"valid_vocab {valid_vocab} outside (0, {padded_vocab}]")
    return _cached_constants(padded_vocab, valid_vocab, str(logits.device), logits.dtype)


def _reduce(logits, hi_pos, lo_pos, valid_mask, hi_sentinel, lo_sentinel):
    """Pure-tensor reduction that torch.compile can trace and lower to Spyre."""
    if valid_mask is not None:
        logits = torch.where(valid_mask, logits, torch.finfo(logits.dtype).min)

    is_max = logits == logits.amax(dim=-1, keepdim=True)

    # min(x) = -max(-x); amin is not in Spyre's eager op registry.
    hi_sel = torch.where(is_max, hi_pos, hi_sentinel)
    hi_min = (-(-hi_sel).amax(dim=-1, keepdim=True))

    lo_sel = torch.where(hi_sel == hi_min, lo_pos, lo_sentinel)
    lo_min = (-(-lo_sel).amax(dim=-1, keepdim=True))

    return hi_min, lo_min


def argmax_digits(
    logits: torch.Tensor,
    valid_vocab: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce ``[batch, vocab]`` logits to the two digits of the top-1 index.

    Args:
        logits: ``[batch, vocab]`` scores, on any device.
        valid_vocab: real vocabulary size when ``logits`` carries alignment
            padding. ``SpyreParallelLMHead`` pads its weight rows, and the
            resulting padding columns are exactly 0.0 -- they would win the
            reduction whenever every real logit is negative, so they are
            excluded here rather than by an on-device slice.

    Returns:
        ``(hi, lo)``, each ``[batch, 1]`` in the input dtype and on the input
        device. Recombine with :func:`combine_digits`.
    """
    if logits.ndim != 2:
        raise ValueError(f"expected [batch, vocab] logits, got shape {tuple(logits.shape)}")

    hi_pos, lo_pos, valid_mask, hi_sentinel, lo_sentinel = _get_constants(logits, valid_vocab)

    return _reduce(logits, hi_pos, lo_pos, valid_mask, hi_sentinel, lo_sentinel)


def combine_digits(hi: torch.Tensor, lo: torch.Tensor) -> torch.Tensor:
    """Recombine the digits into ``[batch]`` int64 token ids.

    Intended to run on the host, where int64 arithmetic is exact.
    """
    return (hi.to(torch.int64) * _RADIX + lo.to(torch.int64)).squeeze(-1)


def greedy_token_ids(
    logits: torch.Tensor,
    valid_vocab: int | None = None,
) -> torch.Tensor:
    """Full path: reduce on the logits' device, return ``[batch]`` ids on CPU.

    Only the two digits cross the device boundary.
    """
    hi, lo = argmax_digits(logits, valid_vocab=valid_vocab)
    return combine_digits(hi.cpu(), lo.cpu())
