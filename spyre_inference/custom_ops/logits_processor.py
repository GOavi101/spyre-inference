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

import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

from .utils import convert

logger = init_logger(__name__)

# When set, logs the size of a host-path logits D2H once. The runner converts
# to CPU only when the host sampler is required; pure-greedy Stage 2 skips it.
_SAMPLER_PROFILING = os.environ.get("SPYRE_SAMPLER_PROFILING", "0") == "1"


@LogitsProcessor.register_oot(name="LogitsProcessor")
class SpyreLogitsProcessor(LogitsProcessor):
    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Gather logits and leave Spyre tensors unpadded.

        Mirrors upstream ``LogitsProcessor._get_logits`` as of vllm v0.26.0
        (``apply`` + gather + optional vocab slice). ``super()._get_logits`` is
        intentionally not used: upstream always does
        ``logits[..., :org_vocab_size]``, and last-dim slices on Spyre are
        unreliable (see ``test_spyre_last_dim_slice``). Stage 2 excludes
        padding via ``valid_vocab`` instead; the host path trims after D2H in
        :meth:`to_host_logits`.

        Re-check this body when bumping the vLLM pin.
        """
        # Keep in sync with upstream apply+gather; only the slice differs on Spyre.
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
        """Apply soft_cap / scale without in-place ops on Spyre.

        Mirrors upstream ``LogitsProcessor.forward`` as of vllm v0.26.0.
        Semantic change vs upstream: scale is ``logits * self.scale``
        (out-of-place) instead of ``logits *= self.scale``. In-place mul on a
        non-contiguous Spyre tensor fails
        (``test_spyre_inplace_mul_noncontiguous``). Granite sets
        ``logits_scaling`` (e.g. 16 → scale=1/16), so that path always runs;
        staying on-device here is what lets Stage 2 see Spyre logits.

        Re-check when bumping the vLLM pin.
        """
        if self.logits_as_input:
            logits = hidden_states
        else:
            logits = self._get_logits(hidden_states, lm_head, embedding_bias)

        if logits is None:
            return None

        if self.soft_cap is not None:
            # Same math as upstream's three-step soft_cap; fused form is fine.
            logits = torch.tanh(logits / self.soft_cap) * self.soft_cap

        if self.scale != 1.0:
            logits = logits * self.scale

        return logits

    @staticmethod
    def to_host_logits(logits: torch.Tensor, org_vocab_size: int) -> torch.Tensor:
        """D2H + org-vocab trim for the upstream CPU sampler."""
        if _SAMPLER_PROFILING:
            logger.info_once(
                "Sampler logits D2H: shape=%s dtype=%s bytes=%d per step",
                tuple(logits.shape),
                logits.dtype,
                logits.numel() * logits.element_size(),
            )

        # record_function is cheap when no profiler is active.
        with torch.profiler.record_function("spyre_sampler::logits_d2h"):
            host = convert(logits, device="cpu")

        if host.shape[-1] > org_vocab_size:
            host = host[..., :org_vocab_size]
        return host
