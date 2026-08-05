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

"""End-to-end sampling coverage for the Spyre path (sampler stage 1).

Sampling runs on the host: SpyreLogitsProcessor moves the gathered logits to
CPU and upstream vLLM's Sampler picks tokens there. The rest of the suite only
ever asks for greedy decoding (`temperature=0.0`) or the default params, so the
filtered/random paths were previously unexercised on Spyre. These tests cover
them.

Why the non-greedy paths are worth pinning:

- `TorchSpyrePlatform` reports `PlatformEnum.OOT`, so `current_platform.is_cpu()`
  is False and `TopKTopPSampler` selects `forward_native` (not `forward_cpu`).
  That is the pure-PyTorch path, and it is the one Spyre actually runs.
- Inside `forward_native`, `apply_top_k_top_p` switches to a **Triton** kernel
  when `HAS_TRITON and logits.shape[0] >= 8`. On a Spyre host Triton normally
  reports no active driver and is disabled, but the batch>=8 top-k/top-p case is
  the one shape where that assumption is load-bearing, so it is covered below.

These run the real model, so they are slow and marked `uses_subprocess` like
tests/test_vllm_spyre_next.py.
"""

import pytest

from vllm import LLM, RequestOutput, SamplingParams
from vllm.config import AttentionConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum

MODEL = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"


@pytest.fixture(scope="module")
def llm():
    return LLM(
        MODEL,
        max_model_len=128,
        max_num_seqs=8,
        attention_config=AttentionConfig(backend=AttentionBackendEnum["CUSTOM"]),
    )


def _texts(outputs: list[RequestOutput]) -> list[str]:
    return [o.outputs[0].text for o in outputs]


@pytest.mark.uses_subprocess
def test_greedy_is_deterministic(llm):
    """temperature=0.0 takes the argmax path and must repeat exactly."""
    params = SamplingParams(max_tokens=8, temperature=0.0)

    first = _texts(llm.generate(prompts="Hello World", sampling_params=params))
    second = _texts(llm.generate(prompts="Hello World", sampling_params=params))

    assert first[0]
    assert first == second


@pytest.mark.uses_subprocess
def test_seeded_random_sampling_is_reproducible(llm):
    """A seeded random draw must reproduce; this exercises the host RNG path."""
    params = SamplingParams(max_tokens=8, temperature=0.8, top_p=0.95, seed=1234)

    first = _texts(llm.generate(prompts="Hello World", sampling_params=params))
    second = _texts(llm.generate(prompts="Hello World", sampling_params=params))

    assert first[0]
    assert first == second


@pytest.mark.uses_subprocess
@pytest.mark.parametrize(
    "params",
    [
        pytest.param(SamplingParams(max_tokens=8, temperature=1.0), id="temperature"),
        pytest.param(
            SamplingParams(max_tokens=8, temperature=0.8, top_k=20), id="top_k"
        ),
        pytest.param(
            SamplingParams(max_tokens=8, temperature=0.8, top_p=0.9), id="top_p"
        ),
        pytest.param(
            SamplingParams(max_tokens=8, temperature=0.8, top_k=50, top_p=0.95),
            id="top_k_top_p",
        ),
        pytest.param(
            SamplingParams(max_tokens=8, temperature=0.8, min_p=0.1), id="min_p"
        ),
        pytest.param(
            SamplingParams(max_tokens=8, temperature=0.0, repetition_penalty=1.2),
            id="repetition_penalty",
        ),
    ],
)
def test_sampling_params_produce_output(llm, params):
    """Each filter/penalty combination must run without falling over."""
    outputs = llm.generate(prompts="Hello World", sampling_params=params)
    assert _texts(outputs)[0]


@pytest.mark.uses_subprocess
def test_top_k_top_p_at_batch_eight(llm):
    """Batch of 8 with top-k/top-p: the shape that can reach the Triton branch.

    `apply_top_k_top_p` uses the Triton kernel when `logits.shape[0] >= 8`, so
    this is the smallest batch that exercises that decision on Spyre.
    """
    prompts = [f"Count to {i}" for i in range(8)]
    params = SamplingParams(max_tokens=8, temperature=0.8, top_k=50, top_p=0.95)

    outputs = llm.generate(prompts=prompts, sampling_params=params)

    assert len(outputs) == 8
    assert all(_texts(outputs))


@pytest.mark.uses_subprocess
def test_mixed_greedy_and_random_batch(llm):
    """Per-request params in one batch: Sampler handles a mixed batch in one call."""
    prompts = ["Hello World", "Count to three"]
    params = [
        SamplingParams(max_tokens=8, temperature=0.0),
        SamplingParams(max_tokens=16, temperature=0.8, top_p=0.9, seed=7),
    ]

    outputs = llm.generate(prompts=prompts, sampling_params=params)

    assert len(outputs) == 2
    assert all(_texts(outputs))


@pytest.mark.uses_subprocess
def test_logprobs_are_returned(llm):
    """logprobs go through Sampler.gather_logprobs on the CPU logits."""
    params = SamplingParams(max_tokens=4, temperature=0.0, logprobs=5)

    completion = llm.generate(prompts="Hello World", sampling_params=params)[0].outputs[0]

    assert completion.logprobs is not None
    assert len(completion.logprobs) == len(completion.token_ids)
