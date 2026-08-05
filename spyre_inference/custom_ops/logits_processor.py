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

from .utils import convert

logger = init_logger(__name__)

# When set, wraps the logits D2H in a torch.profiler.record_function span and
# logs the transfer size once. Sampling runs on the host, so this transfer is
# the cost that an on-device greedy/top-1 path would remove -- measuring it is
# what justifies (or deprioritizes) that work.
_SAMPLER_PROFILING = os.environ.get("SPYRE_SAMPLER_PROFILING", "0") == "1"


@LogitsProcessor.register_oot(name="LogitsProcessor")
class SpyreLogitsProcessor(LogitsProcessor):
    def _gather_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Gather TP-sharded logits on Spyre, then move the result to CPU."""
        gathered = super()._gather_logits(logits)

        if not _SAMPLER_PROFILING:
            return convert(gathered, device="cpu")

        logger.info_once(
            "Sampler logits D2H: shape=%s dtype=%s bytes=%d per step",
            tuple(gathered.shape),
            gathered.dtype,
            gathered.numel() * gathered.element_size(),
        )
        with torch.profiler.record_function("spyre_sampler::logits_d2h"):
            return convert(gathered, device="cpu")
