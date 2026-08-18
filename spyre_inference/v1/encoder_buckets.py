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

"""Encoder compile-shape buckets.

Spyre ``torch.compile(dynamic=False)`` specializes the encoder SDPA batch
``[B, H, L, D]``. Without a ladder, every new ``max_len`` (after stick align)
or ``num_seqs`` compiles a new graph (~60s). Pad L and B up to the next
configured bucket so a 30-token, 3-seq request reuses the warmed ``(4, 64)``
graph.

Env:
    SPYRE_ENCODER_BUCKET_LENS          CSV of prompt-length buckets
                                       (default ``64,128,256,512,1024,2048``).
                                       Each value is rounded up to a multiple
                                       of 64 (Spyre stick).
    SPYRE_ENCODER_BUCKET_BATCH_SIZES   CSV of batch buckets. Default: ``1, 2,
                                       4, …, max_num_seqs``.
"""

from __future__ import annotations

import os

# Stick size; every length bucket must be a multiple of this.
ENCODER_SEQ_ALIGNMENT = 64

_DEFAULT_LEN_BUCKETS = (64, 128, 256, 512, 1024, 2048)


def parse_csv_ints(env_name: str, default: list[int]) -> list[int]:
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return list(default)
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    return values or list(default)


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


def len_buckets() -> list[int]:
    """Configured length ladder, each entry stick-aligned, strictly increasing."""
    raw = parse_csv_ints("SPYRE_ENCODER_BUCKET_LENS", list(_DEFAULT_LEN_BUCKETS))
    aligned = sorted({_align_up(v) for v in raw if v > 0})
    return aligned or [_DEFAULT_LEN_BUCKETS[0]]


def batch_buckets(max_num_seqs: int) -> list[int]:
    """Configured batch ladder, clipped to ``[1, max_num_seqs]``."""
    cap = max(1, max_num_seqs)
    env = parse_csv_ints("SPYRE_ENCODER_BUCKET_BATCH_SIZES", [])
    if env:
        values = sorted({b for b in env if 1 <= b <= cap})
        return values or [cap]
    out: list[int] = []
    size = 1
    while size < cap:
        out.append(size)
        size *= 2
    if cap not in out:
        out.append(cap)
    return out


def encoder_len_bucket(max_len: int) -> int:
    """Nearest length bucket for encoder SDPA ``L`` (always ≥ stick size)."""
    return next_bucket(max(max_len, 1), len_buckets())


def encoder_batch_bucket(num_seqs: int, max_num_seqs: int) -> int:
    """Nearest batch bucket for encoder SDPA ``B`` (≤ ``max_num_seqs``)."""
    cap = max(1, max_num_seqs)
    n = min(max(num_seqs, 1), cap)
    return min(next_bucket(n, batch_buckets(cap)), cap)


def pooling_warmup_shapes(
    max_num_seqs: int,
    max_model_len: int,
    max_num_batched_tokens: int,
) -> list[tuple[int, int]]:
    """``(batch_size, prompt_len)`` pairs to dummy at serve start."""
    shapes: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for batch_size in batch_buckets(max_num_seqs):
        for prompt_len in len_buckets():
            if prompt_len > max_model_len:
                continue
            num_tokens = batch_size * prompt_len
            if num_tokens > max_num_batched_tokens:
                continue
            key = (batch_size, prompt_len)
            if key in seen:
                continue
            seen.add(key)
            shapes.append(key)
    return shapes
