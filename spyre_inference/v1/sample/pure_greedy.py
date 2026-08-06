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

"""When a batch can use on-device greedy (skip full-vocab logits D2H)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.sample.metadata import SamplingMetadata


def _non_argmax_procs_inactive(sampling_metadata: SamplingMetadata) -> bool:
    """False if any non-argmax-invariant processor is active or unknown."""
    for proc in sampling_metadata.logitsprocs.non_argmax_invariant:
        biases = getattr(proc, "biases", None)
        if biases is not None:
            if biases:
                return False
            continue
        min_toks = getattr(proc, "min_toks", None)
        if min_toks is not None:
            if min_toks:
                return False
            continue
        return False  # unknown type → host
    return True


def is_pure_greedy(
    sampling_metadata: SamplingMetadata,
    *,
    has_grammar: bool = False,
    has_spec: bool = False,
) -> bool:
    """True when the batch needs only top-1 (no host logits features)."""
    if has_grammar or has_spec:
        return False
    if not sampling_metadata.all_greedy:
        return False
    if not sampling_metadata.no_penalties:
        return False
    if sampling_metadata.max_num_logprobs is not None:
        return False
    if sampling_metadata.logprob_token_ids:
        return False
    if sampling_metadata.allowed_token_ids_mask is not None:
        return False
    if sampling_metadata.bad_words_token_ids:
        return False
    holder = sampling_metadata.thinking_budget_state_holder
    if holder is not None and holder.has_tracked_requests():
        return False
    return _non_argmax_procs_inactive(sampling_metadata)
