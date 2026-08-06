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

"""Spyre OOT replacement for LogitsProcessor.

Spyre constraints:
    - Last-dim vocab slice is unreliable on Spyre; keep padded width on-device
      and trim via valid_vocab (Stage 2) or to_host_logits (host sampler).
    - In-place ``logits *= scale`` fails on non-contiguous Spyre tensors; use
      out-of-place mul instead (Granite sets logits_scaling, so scale != 1).

References:
    - Upstream LogitsProcessor: vllm/model_executor/layers/logits_processor.py
      (pinned at vllm v0.26.0; re-check on pin bumps)
"""

import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

from .utils import convert

logger = init_logger(__name__)

# When set, log host-path logits D2H size once. Pure-greedy Stage 2 skips D2H.
_SAMPLER_PROFILING = os.environ.get("SPYRE_SAMPLER_PROFILING", "0") == "1"


@LogitsProcessor.register_oot(name="LogitsProcessor")
class SpyreLogitsProcessor(LogitsProcessor):
    """Out-of-tree (OOT) LogitsProcessor implementation for IBM's Spyre."""

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Gather logits; leave Spyre tensors unpadded.

        Mirrors upstream apply+gather. Does not call ``super()._get_logits``
        because upstream always slices ``[..., :org_vocab_size]``.
        """
        logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)
        logits = super()._gather_logits(logits)
        if logits is None:
            return None

        if logits.device.type == "spyre":
            return logits

        return logits[..., : self.org_vocab_size]

    def forward(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Apply soft_cap / scale; keep Spyre logits on-device.

        Same control flow as upstream, but scale is out-of-place
        (``logits * self.scale`` instead of ``logits *= self.scale``).
        """
        if self.logits_as_input:
            logits = hidden_states
        else:
            logits = self._get_logits(hidden_states, lm_head, embedding_bias)

        if logits is None:
            return None

        if self.soft_cap is not None:
            logits = torch.tanh(logits / self.soft_cap) * self.soft_cap

        if self.scale != 1.0:
            logits = logits * self.scale

        return logits

    @staticmethod
    def to_host_logits(logits: torch.Tensor, org_vocab_size: int) -> torch.Tensor:
        """Move logits to CPU and trim to ``org_vocab_size`` for the host sampler."""
        if _SAMPLER_PROFILING:
            logger.info_once(
                "Sampler logits D2H: shape=%s dtype=%s bytes=%d per step",
                tuple(logits.shape),
                logits.dtype,
                logits.numel() * logits.element_size(),
            )
        with torch.profiler.record_function("spyre_sampler::logits_d2h"):
            host = convert(logits, device="cpu")

        if host.shape[-1] > org_vocab_size:
            host = host[..., :org_vocab_size]
        return host
