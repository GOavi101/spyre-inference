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
        logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)
        logits = super()._gather_logits(logits)
        if logits is None:
            return None

        # Keep the full padded width on-device rather than slicing the last dim
        # on Spyre; the Stage 2 reduction excludes the padding via
        # valid_vocab=org_vocab_size instead. Host path slices after D2H.
        if logits.device.type == "spyre":
            return logits

        return logits[..., : self.org_vocab_size]

    def forward(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Same as upstream, but keep Spyre logits on-device and scale out-of-place.

        Upstream does ``logits *= self.scale``. Granite sets ``logits_scaling``
        (e.g. 16 → scale=1/16), so that path always runs. In-place mul on a
        non-contiguous Spyre tensor is a known torch-spyre failure
        (``test_spyre_inplace_mul_noncontiguous``), and forcing a D2H here just
        because scale != 1 would disable Stage 2 for every Granite model.

        Out-of-place ``logits * scale`` / soft_cap are Spyre-native (mul, div,
        tanh) and leave logits on the device for the runner to either reduce
        on-device or D2H for the host sampler.
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
        """D2H + org-vocab trim for the upstream CPU sampler."""
        if _SAMPLER_PROFILING:
            logger.info_once(
                "Sampler logits D2H: shape=%s dtype=%s bytes=%d per step",
                tuple(logits.shape),
                logits.dtype,
                logits.numel() * logits.element_size(),
            )
            with torch.profiler.record_function("spyre_sampler::logits_d2h"):
                host = convert(logits, device="cpu")
        else:
            host = convert(logits, device="cpu")

        if host.shape[-1] > org_vocab_size:
            host = host[..., :org_vocab_size]
        return host
