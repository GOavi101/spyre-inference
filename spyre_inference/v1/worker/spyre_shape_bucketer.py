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

Decoder (1D): sorted ``compile_sizes`` token counts; pad the packed batch to
the nearest bucket ``>=`` actual ``num_tokens``.

Pooling / encoder (2D): warmed ``(B, L)`` cells with ``T = B × L``. SDPA
compiles on ``[B, H, L, D]``; Linear / LN compile on ``[T, …]``. Dispatch
picks a warmed cell that covers ``(num_seqs, max_query_len)``. Do not reuse
the decoder's 1D token ladder for encoder attention.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass

from vllm.config import VllmConfig
from vllm.logger import init_logger

from spyre_inference.v1.encoder_buckets import pooling_warmup_shapes

logger = init_logger(__name__)


@dataclass(frozen=True)
class SpyreBucketDescriptor:
    """Descriptor for a 1D (decoder) compilation bucket."""

    actual_num_tokens: int
    padded_num_tokens: int


@dataclass(frozen=True)
class EncoderBucketDescriptor:
    """Descriptor for a 2D encoder ``(B, L)`` compilation bucket."""

    batch_bucket: int
    len_bucket: int
    actual_num_seqs: int
    actual_max_len: int

    @property
    def padded_num_tokens(self) -> int:
        return self.batch_bucket * self.len_bucket


class SpyreShapeBucketer:
    """Dispatches runtime batches to pre-compiled bucket sizes.

    1D (``compile_sizes``): nearest token count ``>=`` actual ``num_tokens``.
    2D (``encoder_shapes``): nearest warmed ``(B, L)`` covering the batch.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        encoder_shapes: Sequence[tuple[int, int]] | None = None,
    ) -> None:
        if encoder_shapes is not None:
            self._encoder_shapes: list[tuple[int, int]] = list(encoder_shapes)
            self._bucket_sizes: list[int] = sorted(
                {batch * length for batch, length in self._encoder_shapes}
            )
        else:
            self._encoder_shapes = []
            compilation_config = vllm_config.compilation_config
            sizes: list[int] = [int(s) for s in (compilation_config.compile_sizes or [])]
            self._bucket_sizes = sorted(sizes)
        self._max_bucket_size = self._bucket_sizes[-1] if self._bucket_sizes else 0
        self._is_warmed_up = False

        if self._encoder_shapes:
            logger.info(
                "SpyreShapeBucketer initialized with %d encoder (B, L) shapes: %s",
                len(self._encoder_shapes),
                self._encoder_shapes,
            )
        else:
            logger.info(
                "SpyreShapeBucketer initialized with %d bucket sizes: min=%d, max=%d",
                len(self._bucket_sizes),
                self._bucket_sizes[0] if self._bucket_sizes else 0,
                self._max_bucket_size,
            )

    @classmethod
    def for_pooling(cls, vllm_config: VllmConfig) -> SpyreShapeBucketer | None:
        """Build a 2D encoder bucketer from the pooling warmup ladder, or None."""
        model_config = vllm_config.model_config
        if getattr(model_config, "runner_type", None) != "pooling":
            return None
        scheduler = vllm_config.scheduler_config
        shapes = pooling_warmup_shapes(
            max_num_seqs=scheduler.max_num_seqs,
            max_model_len=model_config.max_model_len,
            max_num_batched_tokens=scheduler.max_num_batched_tokens,
        )
        if not shapes:
            return None
        return cls(vllm_config, encoder_shapes=shapes)

    @property
    def bucket_sizes(self) -> list[int]:
        return self._bucket_sizes

    @property
    def encoder_shapes(self) -> list[tuple[int, int]]:
        return list(self._encoder_shapes)

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

    def find_encoder_bucket(
        self,
        num_seqs: int,
        max_query_len: int,
        max_num_seqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
    ) -> tuple[int, int] | None:
        """Smallest warmed ``(B, L)`` that covers the batch, or None.

        Only cells in ``encoder_shapes`` are eligible, so runtime pad lands on
        a graph warmup already compiled. Prefer smallest ``T = B × L``, then
        smallest ``B``, then smallest ``L``.
        """
        if num_seqs < 1 or max_query_len < 1 or not self._encoder_shapes:
            return None
        candidates = [
            (batch, length)
            for batch, length in self._encoder_shapes
            if batch >= num_seqs
            and length >= max_query_len
            and batch <= max_num_seqs
            and length <= max_model_len
            and batch * length <= max_num_batched_tokens
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda pair: (pair[0] * pair[1], pair[0], pair[1]))

    def dispatch_encoder(
        self,
        num_seqs: int,
        max_query_len: int,
        max_num_seqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
    ) -> EncoderBucketDescriptor | None:
        """Pad descriptor for encoder SDPA, or None if no warmed cell fits."""
        pair = self.find_encoder_bucket(
            num_seqs,
            max_query_len,
            max_num_seqs,
            max_model_len,
            max_num_batched_tokens,
        )
        if pair is None:
            return None
        batch_bucket, len_bucket = pair
        return EncoderBucketDescriptor(
            batch_bucket=batch_bucket,
            len_bucket=len_bucket,
            actual_num_seqs=num_seqs,
            actual_max_len=max_query_len,
        )
