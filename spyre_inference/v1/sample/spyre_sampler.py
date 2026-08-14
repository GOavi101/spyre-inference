# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Spyre sampler with a pre-generated exponential-noise pool.

Host-side stopgap: sampling still runs on **CPU** (host logits); torch-spyre
is only the model device. The pool avoids slow per-step
``Tensor.exponential_()`` on platforms such as s390x when doing Gumbel-max
sampling (``argmax(probs / q)``, ``q ~ Exp(1)``).

This is **not** on-device / ``torch.compile`` sampling — prefer that path
when ready. Opt-in via ``SPYRE_USE_NOISE_POOL=1`` (see ``spyre_inference.envs``).
When disabled, behaviour matches upstream vLLM's ``Sampler``.
"""

from __future__ import annotations

import platform
import random

import torch
from vllm.config import VllmConfig
from vllm.config.model import LogprobsMode
from vllm.logger import init_logger
from vllm.v1.sample.ops.topk_topp_sampler import (
    TopKTopPSampler,
    apply_top_k_top_p,
    random_sample,
)
from vllm.v1.sample.sampler import Sampler

import spyre_inference.envs as envs

logger = init_logger(__name__)


def get_noise_pool_dtype() -> torch.dtype:
    """Pool dtype from ``SPYRE_NOISE_POOL_DTYPE``, else platform default."""
    override = envs.SPYRE_NOISE_POOL_DTYPE
    if override in ("float16", "fp16", "half"):
        return torch.float16
    if override in ("float32", "fp32", "float"):
        return torch.float32
    if platform.machine() in ("s390x", "ppc64le"):
        return torch.float32
    return torch.float16


class ExponentialNoisePool:
    """Fixed Exp(1) buffer filled once; random slices returned per draw."""

    def __init__(
        self,
        numel: int,
        dtype: torch.dtype,
        device: torch.device | str = "cpu",
        seed: int = 0,
    ) -> None:
        if numel <= 0:
            raise ValueError(f"Noise pool size must be positive, got {numel}")
        self.numel = numel
        self.dtype = dtype
        self.device = torch.device(device)
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)
        self.pool = torch.empty(numel, dtype=dtype, device=self.device)
        self.pool.exponential_(generator=gen)
        # Avoid exact zeros from float16 underflow (breaks probs/q).
        self.pool.clamp_(min=torch.finfo(dtype).tiny)
        self._offset_rng = random.Random(seed)

    def draw(self, shape: torch.Size) -> torch.Tensor:
        """Return a fresh Exp(1) tensor of ``shape`` sliced from the pool."""
        n = 1
        for dim in shape:
            n *= int(dim)
        if n > self.numel:
            raise ValueError(
                f"Noise pool too small: need {n} elements for shape {tuple(shape)} "
                f"but pool holds {self.numel}. Increase SPYRE_NOISE_POOL_MULTIPLIER."
            )
        offset = self._offset_rng.randint(0, self.numel - n)
        logger.debug(
            "Noise pool draw: shape=%s (%d elems) from offset %d/%d",
            tuple(shape),
            n,
            offset,
            self.numel,
        )
        return self.pool[offset : offset + n].view(shape)

    def draw_row(self, width: int, generator: torch.Generator) -> torch.Tensor:
        """Return one row; offset chosen by the per-request ``generator``."""
        if width > self.numel:
            raise ValueError(
                f"Noise pool too small: need {width} elements for a row but pool "
                f"holds {self.numel}. Increase SPYRE_NOISE_POOL_MULTIPLIER."
            )
        offset = int(
            torch.randint(
                0, self.numel - width + 1, (1,), generator=generator, device=self.device
            ).item()
        )
        return self.pool[offset : offset + width]


def pooled_random_sample(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
    pool: ExponentialNoisePool,
) -> torch.Tensor:
    """Drop-in for vLLM ``random_sample`` backed by ``pool``.

    Seeded rows select a deterministic pool offset (reproducible for a given
    pool + seed, but not bit-identical to upstream seeded ``exponential_()``).
    Oversized batches fall back to on-the-fly ``random_sample``.
    """
    n = probs.shape.numel()
    if n > pool.numel:
        logger.warning_once(
            "Sampling batch needs %d noise elements but the pool holds only %d; "
            "falling back to on-the-fly exponential_() for oversized steps. "
            "Increase SPYRE_NOISE_POOL_MULTIPLIER to avoid this.",
            n,
            pool.numel,
        )
        return random_sample(probs, generators)
    q = pool.draw(probs.shape)
    if generators:
        # draw() returns a view into the shared pool; clone before overwrite.
        q = q.clone()
        width = probs.shape[1]
        for i, generator in generators.items():
            q[i] = pool.draw_row(width, generator)
    return probs.div_(q).argmax(dim=-1).view(-1)


class SpyreTopKTopPSampler(TopKTopPSampler):
    """``TopKTopPSampler`` that can draw Gumbel noise from a pool."""

    def __init__(
        self,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        noise_pool: ExponentialNoisePool | None = None,
    ) -> None:
        super().__init__(logprobs_mode)
        self.noise_pool = noise_pool
        self.forward = self.forward_native
        logger.info(
            "SpyreTopKTopPSampler initialized: noise_pool=%s",
            "on" if self.noise_pool is not None else "off",
        )

    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = apply_top_k_top_p(logits, k, p)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)
        probs = logits.softmax(dim=-1, dtype=torch.float32)

        if self.noise_pool is None:
            sampled = random_sample(probs, generators)
        else:
            sampled = pooled_random_sample(probs, generators, self.noise_pool)
        return sampled, logits_to_return


class SpyreSampler(Sampler):
    """Upstream ``Sampler`` with ``SpyreTopKTopPSampler`` for random draws."""

    def __init__(
        self,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        noise_pool: ExponentialNoisePool | None = None,
    ) -> None:
        super().__init__(logprobs_mode)
        self.topk_topp_sampler = SpyreTopKTopPSampler(logprobs_mode, noise_pool=noise_pool)


def build_spyre_sampler(vllm_config: VllmConfig) -> SpyreSampler:
    """Build sampler; optionally back random sampling with a noise pool.

    When ``SPYRE_USE_NOISE_POOL`` is off, returned sampler matches upstream
    behaviour (pool path disabled inside ``SpyreTopKTopPSampler``).
    """
    noise_pool: ExponentialNoisePool | None = None
    if envs.SPYRE_USE_NOISE_POOL:
        vocab_size = vllm_config.model_config.get_vocab_size()
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        multiplier = envs.SPYRE_NOISE_POOL_MULTIPLIER
        numel = multiplier * max_num_seqs * vocab_size
        dtype = get_noise_pool_dtype()
        logger.info(
            "Building exponential-noise pool: %d elements "
            "(%d x max_num_seqs=%d x vocab_size=%d), dtype=%s (~%.2f GiB)",
            numel,
            multiplier,
            max_num_seqs,
            vocab_size,
            dtype,
            numel * torch.empty((), dtype=dtype).element_size() / (1024**3),
        )
        noise_pool = ExponentialNoisePool(numel=numel, dtype=dtype, device="cpu")
    return SpyreSampler(
        logprobs_mode=vllm_config.model_config.logprobs_mode,
        noise_pool=noise_pool,
    )
