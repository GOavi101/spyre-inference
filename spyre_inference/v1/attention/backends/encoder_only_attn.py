# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spyre override of encoder-only attention.

Mirrors the structure of
``vllm/model_executor/layers/attention/encoder_only_attention.py``:

* :func:`create_spyre_encoder_only_backend` wraps the Spyre attention backend
  with a builder that sets ``causal=False``, producing a backend that performs
  bidirectional (padding-only masked) attention.

* :class:`SpyreEncoderOnlyAttention` is the drop-in ``Attention`` subclass for
  encoder-only layers on Spyre.  It selects the wrapped backend and returns
  ``None`` from ``get_kv_cache_spec()`` so that no KV cache is allocated for
  these layers.

The heavy lifting (skipping KV-cache write/gather, reshaping K/V from raw
input tokens) lives in ``SpyreAttentionImpl.forward()`` inside ``spyre_attn.py``.
It detects encoder-only execution at runtime via
``self.attn_type == AttentionType.ENCODER_ONLY``.
"""

import functools
from copy import copy

import torch

from vllm.config import CacheConfig
from vllm.config.vllm import VllmConfig
from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backend import (
    AttentionMetadata,
    AttentionType,
    CommonAttentionMetadata,
    subclass_attention_backend,
)
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.kv_cache_interface import KVCacheSpec


@functools.lru_cache
def create_spyre_encoder_only_backend(
    underlying_attn_backend,
):
    """Wrap *underlying_attn_backend* so that its builder always sets causal=False.

    This is the Spyre equivalent of
    ``vllm.model_executor.layers.attention.encoder_only_attention
    .create_encoder_only_attention_backend``.
    """
    prefix = "SpyreEncoderOnly_"
    underlying_builder = underlying_attn_backend.get_builder_cls()

    class SpyreEncoderOnlyBuilder(underlying_builder):  # type: ignore[valid-type]
        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
            fast_build: bool = False,
        ) -> AttentionMetadata:
            new_meta = copy(common_attn_metadata)
            new_meta.causal = False
            return super().build(common_prefix_len, new_meta, fast_build)

    return subclass_attention_backend(
        name_prefix=prefix,
        attention_backend_cls=underlying_attn_backend,
        builder_cls=SpyreEncoderOnlyBuilder,
    )


class SpyreEncoderOnlyAttention(Attention):
    """Encoder-only attention for Spyre — bidirectional, no KV cache.

    Mirrors ``vllm.model_executor.layers.attention.encoder_only_attention
    .EncoderOnlyAttention`` but explicitly selects the Spyre attention backend
    so that the Spyre-specific forward path in ``SpyreAttentionImpl`` is used.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        cache_config: CacheConfig | None = None,
        attn_type: str | None = None,
        **kwargs,
    ):
        dtype = torch.get_default_dtype()
        kv_cache_dtype = cache_config.cache_dtype if cache_config is not None else "auto"

        underlying_backend = get_attn_backend(
            head_size,
            dtype,
            kv_cache_dtype,
            attn_type=AttentionType.ENCODER_ONLY,
        )
        attn_backend = create_spyre_encoder_only_backend(underlying_backend)

        if attn_type is not None:
            assert attn_type == AttentionType.ENCODER_ONLY, (
                "SpyreEncoderOnlyAttention only supports AttentionType.ENCODER_ONLY"
            )

        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            cache_config=cache_config,
            attn_backend=attn_backend,
            attn_type=AttentionType.ENCODER_ONLY,
            **kwargs,
        )

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        # Encoder-only layers do not need a KV cache.
        return None

