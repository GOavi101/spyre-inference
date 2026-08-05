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

"""Unit tests for Stage 2 pure-greedy eligibility."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from spyre_inference.v1.sample.pure_greedy import is_pure_greedy
from vllm.v1.sample.logits_processor.state import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata


def _metadata(**overrides) -> SamplingMetadata:
    base = dict(
        temperature=None,
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(1),
        presence_penalties=torch.zeros(1),
        repetition_penalties=torch.ones(1),
        output_token_ids=[[]],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
        logprob_token_ids=None,
        spec_token_ids=None,
        thinking_budget_state_holder=None,
    )
    base.update(overrides)
    return SamplingMetadata(**base)


def test_pure_greedy_default_batch_is_eligible():
    assert is_pure_greedy(_metadata())


def test_rejects_non_greedy():
    assert not is_pure_greedy(_metadata(all_greedy=False))


def test_rejects_penalties():
    assert not is_pure_greedy(_metadata(no_penalties=False))


def test_rejects_logprobs():
    assert not is_pure_greedy(_metadata(max_num_logprobs=5))
    assert not is_pure_greedy(_metadata(logprob_token_ids={0: [1, 2]}))


def test_rejects_allowed_token_mask():
    mask = torch.ones(1, 8, dtype=torch.bool)
    assert not is_pure_greedy(_metadata(allowed_token_ids_mask=mask))


def test_rejects_bad_words():
    assert not is_pure_greedy(_metadata(bad_words_token_ids={0: [[1, 2]]}))


def test_rejects_grammar_and_spec():
    md = _metadata()
    assert not is_pure_greedy(md, has_grammar=True)
    assert not is_pure_greedy(md, has_spec=True)


def test_rejects_active_logit_bias():
    proc = SimpleNamespace(biases={0: {3: 1.5}}, is_argmax_invariant=lambda: False)
    md = _metadata(logitsprocs=LogitsProcessors([proc]))
    assert not is_pure_greedy(md)


def test_allows_inactive_logit_bias_processor():
    proc = SimpleNamespace(biases={}, is_argmax_invariant=lambda: False)
    md = _metadata(logitsprocs=LogitsProcessors([proc]))
    assert is_pure_greedy(md)


def test_rejects_unknown_non_argmax_processor():
    proc = SimpleNamespace(is_argmax_invariant=lambda: False)
    md = _metadata(logitsprocs=LogitsProcessors([proc]))
    assert not is_pure_greedy(md)


def test_rejects_active_thinking_budget():
    holder = SimpleNamespace(has_tracked_requests=lambda: True)
    assert not is_pure_greedy(_metadata(thinking_budget_state_holder=holder))
