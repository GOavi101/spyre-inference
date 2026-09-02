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

"""Spyre shape bucketer for compilation warmup and runtime dispatch.

Body (1D, decoder and pooling): sorted ``compile_sizes`` token counts; pad the
packed batch to the nearest bucket ``>=`` actual ``num_tokens``. Linear / LN
compile on ``[T, …]``.

Encoder attention is varlen flash on that packed list (``query_start_loc``).
There is no ``(B, L)`` attention grid and no rewrite of body ``T``.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

# Spyre stick (64 fp16 elements), and the encoder attention KV block width.
ENCODER_SEQ_ALIGNMENT = 64


def default_encoder_len_buckets(max_model_len: int) -> list[int]:
    """Stick-aligned prompt-length buckets from 64 up to ``max_model_len``.

    Powers of two through the last value that still fits, then ``max_model_len``
    rounded down to a stick if that is not already on the ladder.
    """
    cap = max(1, int(max_model_len))
    buckets: list[int] = []
    size = ENCODER_SEQ_ALIGNMENT
    while size < cap:
        buckets.append(size)
        size *= 2
    aligned_cap = max(ENCODER_SEQ_ALIGNMENT, (cap // ENCODER_SEQ_ALIGNMENT) * ENCODER_SEQ_ALIGNMENT)
    if aligned_cap <= cap and aligned_cap not in buckets:
        buckets.append(aligned_cap)
    return buckets or [ENCODER_SEQ_ALIGNMENT]


def _align_up(n: int, align: int = ENCODER_SEQ_ALIGNMENT) -> int:
    return max(align, (n + align - 1) // align * align)


def next_bucket(n: int, buckets: list[int]) -> int:
    """Smallest bucket ``>= n``. If ``n`` exceeds the ladder, stick-align ``n``."""
    if n < 1:
        n = 1
    ordered = sorted({b for b in buckets if b > 0})
    for bucket in ordered:
        if bucket >= n:
            return bucket
    return _align_up(n)


def len_buckets(
    max_model_len: int,
    compile_sizes: Sequence[int] | None = None,
) -> list[int]:
    """Attention ``L`` ladder from ``max_model_len``.

    Optional ``compile_sizes`` overrides ``L`` in tests. Platform
    ``compile_sizes`` are body token counts and must not be passed here.
    """
    if compile_sizes:
        aligned = sorted({_align_up(int(v)) for v in compile_sizes if int(v) > 0})
        fitted = [v for v in aligned if v <= max_model_len]
        if fitted:
            return fitted
    return default_encoder_len_buckets(max_model_len)


def batch_buckets(max_num_seqs: int) -> list[int]:
    """Powers of two in ``[1, max_num_seqs]``, plus ``max_num_seqs`` itself.

    Same ladder as decoder attention (``_powers_of_two_up_to``): clip with
    ``--max-num-seqs``, no extra env var.
    """
    cap = max(1, max_num_seqs)
    out: list[int] = []
    size = 1
    while size < cap:
        out.append(size)
        size *= 2
    if cap not in out:
        out.append(cap)
    return out


def encoder_len_bucket(max_len: int, buckets: list[int] | None = None) -> int:
    """Nearest stick-aligned length ``>= max_len`` (flash tile / body ladder)."""
    return next_bucket(max(max_len, 1), buckets or [])


def encoder_batch_bucket(num_seqs: int, max_num_seqs: int) -> int:
    """Nearest power-of-two batch ``>= num_seqs`` (≤ ``max_num_seqs``)."""
    cap = max(1, max_num_seqs)
    n = min(max(num_seqs, 1), cap)
    return min(next_bucket(n, batch_buckets(cap)), cap)


@dataclass(frozen=True)
class SpyreBucketDescriptor:
    """Descriptor for a 1D (decoder) compilation bucket."""

    actual_num_tokens: int
    padded_num_tokens: int


class SpyreShapeBucketer:
    """Dispatches runtime batches to pre-compiled 1D body token buckets."""

    def __init__(self, vllm_config: VllmConfig) -> None:
        compilation_config = vllm_config.compilation_config
        sizes: list[int] = [int(s) for s in (compilation_config.compile_sizes or [])]
        self._bucket_sizes = sorted(sizes)
        self._max_bucket_size = self._bucket_sizes[-1] if self._bucket_sizes else 0
        self._is_warmed_up = False
        logger.info(
            "SpyreShapeBucketer initialized with %d body token buckets: min=%d, max=%d",
            len(self._bucket_sizes),
            self._bucket_sizes[0] if self._bucket_sizes else 0,
            self._max_bucket_size,
        )

    @classmethod
    def for_pooling(cls, vllm_config: VllmConfig) -> SpyreShapeBucketer | None:
        """Pooling bucketer: 1D body ``compile_sizes`` only (flash is varlen)."""
        model_config = vllm_config.model_config
        if getattr(model_config, "runner_type", None) != "pooling":
            return None
        compile_sizes = [int(s) for s in (vllm_config.compilation_config.compile_sizes or [])]
        if not compile_sizes:
            return None
        return cls(vllm_config)

    @property
    def bucket_sizes(self) -> list[int]:
        return self._bucket_sizes

    @property
    def max_bucket_size(self) -> int:
        return self._max_bucket_size

    @property
    def is_warmed_up(self) -> bool:
        return self._is_warmed_up

    def mark_warmed_up(self) -> None:
        self._is_warmed_up = True

    def find_bucket(self, num_tokens: int) -> int | None:
        """Find the smallest 1D bucket size >= num_tokens.

        Returns None if num_tokens exceeds the largest compiled bucket.
        The caller (execute_model) handles the None case by running the
        forward pass without bucket padding, which may trigger Dynamo
        recompilation for the unseen shape.
        """
        idx = bisect.bisect_left(self._bucket_sizes, num_tokens)
        if idx < len(self._bucket_sizes):
            return self._bucket_sizes[idx]
        return None

    def dispatch(self, num_tokens: int) -> SpyreBucketDescriptor | None:
        """Compute padded batch descriptor for the given token count.

        Returns None if no suitable bucket exists.
        """
        padded = self.find_bucket(num_tokens)
        if padded is None:
            return None
        return SpyreBucketDescriptor(
            actual_num_tokens=num_tokens,
            padded_num_tokens=padded,
        )
