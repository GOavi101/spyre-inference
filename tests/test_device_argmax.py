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

"""Correctness of the two-digit top-1 reduction in v1/sample/device_argmax.py.

The index arithmetic and tie-breaking are device-independent, so the bulk of
this suite runs on CPU and is meaningful without a Spyre card. The final tests
re-run the same reduction on-device (eager and compiled) and skip when no card
is present.
"""

import pytest
import torch

from spyre_inference.v1.sample.device_argmax import (
    _RADIX,
    argmax_digits,
    combine_digits,
    greedy_token_ids,
)

# Padded to a multiple of 64 * 32, matching SpyreParallelLMHead's alignment.
GRANITE_VOCAB, GRANITE_PADDED = 49152, 49152
LLAMA_VOCAB, LLAMA_PADDED = 128256, 129024


def reference(logits: torch.Tensor, valid_vocab: int | None = None) -> torch.Tensor:
    """torch.argmax over the unpadded region: the behaviour we must match."""
    if valid_vocab is not None:
        logits = logits[:, :valid_vocab]
    return logits.argmax(dim=-1)


@pytest.mark.parametrize("vocab", [256, 1024, 2048, 32000, GRANITE_VOCAB, LLAMA_VOCAB])
@pytest.mark.parametrize("batch", [1, 8])
def test_matches_torch_argmax(vocab: int, batch: int):
    """Random fp16 logits at realistic vocab sizes."""
    torch.manual_seed(0)
    logits = torch.randn(batch, vocab, dtype=torch.float16)

    got = greedy_token_ids(logits)

    torch.testing.assert_close(got, reference(logits), atol=0, rtol=0)


def test_index_above_fp16_exact_range():
    """The case a naive fp16 index gets wrong.

    An index beyond 2048 is not exactly representable in fp16 -- near 32000 the
    spacing is 16 -- which is precisely why topk's fp16 index buffer cannot be
    used. The split-digit form must return it exactly.
    """
    vocab = 32000
    logits = torch.full((1, vocab), -10.0, dtype=torch.float16)
    for target in (2049, 12345, 31999):
        logits.fill_(-10.0)
        logits[0, target] = 1.0

        assert greedy_token_ids(logits).item() == target


def test_ties_take_lowest_index():
    """Duplicate maxima must resolve to the first, like torch.argmax.

    fp16 has few mantissa bits, so exact ties across a large vocab are common
    rather than pathological.
    """
    logits = torch.full((1, 4096), -1.0, dtype=torch.float16)
    tied = [100, 200, 3000, 4095]
    for i in tied:
        logits[0, i] = 5.0

    assert greedy_token_ids(logits).item() == min(tied)
    torch.testing.assert_close(greedy_token_ids(logits), reference(logits), atol=0, rtol=0)


def test_ties_within_and_across_high_digits():
    """Ties that share a high digit exercise the second reduction stage."""
    logits = torch.full((2, 2048), -1.0, dtype=torch.float16)
    # Row 0: same high digit, different low digits -> stage two decides.
    logits[0, _RADIX + 5] = 3.0
    logits[0, _RADIX + 61] = 3.0
    # Row 1: different high digits -> stage one decides.
    logits[1, 5 * _RADIX + 7] = 3.0
    logits[1, 9 * _RADIX + 1] = 3.0

    torch.testing.assert_close(greedy_token_ids(logits), reference(logits), atol=0, rtol=0)


def test_padding_columns_are_excluded():
    """Zero padding columns must not win when every real logit is negative.

    SpyreParallelLMHead pads its weight rows, and the padding rows are zeros, so
    the padded logit columns are exactly 0.0. Without masking they would beat
    any negative real logit.
    """
    valid, padded = 3000, 4096
    logits = torch.full((4, padded), -5.0, dtype=torch.float16)
    logits[:, valid:] = 0.0
    logits[:, 1234] = -1.0  # best real token, still negative

    got = greedy_token_ids(logits, valid_vocab=valid)

    assert torch.equal(got, torch.full((4,), 1234, dtype=torch.int64))


def test_padding_is_a_noop_when_unpadded():
    """Passing valid_vocab == vocab must not change the result."""
    torch.manual_seed(1)
    logits = torch.randn(4, 2048, dtype=torch.float16)

    torch.testing.assert_close(
        greedy_token_ids(logits, valid_vocab=2048),
        greedy_token_ids(logits),
        atol=0,
        rtol=0,
    )


def test_digits_are_small_enough_for_fp16():
    """Neither digit may leave fp16's exact-integer range; that is the premise."""
    logits = torch.randn(8, LLAMA_PADDED, dtype=torch.float16)

    hi, lo = argmax_digits(logits)

    assert hi.max().item() < 2048
    assert lo.max().item() < _RADIX
    torch.testing.assert_close(combine_digits(hi, lo), reference(logits), atol=0, rtol=0)


def test_float32_logits():
    """The reduction is dtype-agnostic; fp32 must work too."""
    torch.manual_seed(2)
    logits = torch.randn(4, 32000, dtype=torch.float32)

    torch.testing.assert_close(greedy_token_ids(logits), reference(logits), atol=0, rtol=0)


def test_rejects_non_2d_input():
    with pytest.raises(ValueError, match=r"\[batch, vocab\]"):
        argmax_digits(torch.randn(2, 3, 4, dtype=torch.float16))


def test_rejects_out_of_range_valid_vocab():
    with pytest.raises(ValueError, match="outside"):
        argmax_digits(torch.randn(2, 128, dtype=torch.float16), valid_vocab=129)


def test_rejects_vocab_too_large_for_fp16():
    """Beyond ~262k the high digit itself leaves fp16's exact range."""
    huge = (2048 * _RADIX) + _RADIX
    with pytest.raises(ValueError, match="fp16 cannot represent"):
        argmax_digits(torch.zeros(1, huge, dtype=torch.float16))


# ---------------------------------------------------------------------------
# On-device execution (requires a Spyre card)
# ---------------------------------------------------------------------------


@pytest.fixture()
def spyre_device():
    from spyre_testing_plugin.pytest_plugin import spyre_available

    if not spyre_available():
        pytest.skip("Spyre device not available")
    return torch.device("spyre")


@pytest.mark.parametrize("mode", ["eager", "compile"])
def test_on_spyre_matches_cpu_reference(spyre_device, mode):
    """The same reduction, run on-device, must agree with torch.argmax on CPU.

    This is the probe that decides whether the amax/amin/eq/where decomposition
    actually holds up on hardware -- every op it uses is documented as
    Spyre-native, but none has been exercised at [batch, vocab] shapes.
    """
    torch.manual_seed(3)
    logits_cpu = torch.randn(8, 32000, dtype=torch.float16)
    logits = logits_cpu.to(spyre_device)

    fn = argmax_digits
    if mode == "compile":
        fn = torch.compile(argmax_digits, dynamic=False)

    hi, lo = fn(logits)

    got = combine_digits(hi.cpu(), lo.cpu())
    torch.testing.assert_close(got, reference(logits_cpu), atol=0, rtol=0)


def test_on_spyre_does_not_fall_back_to_cpu(spyre_device):
    """The point of the decomposition is avoiding argmax's CPU round-trip.

    A FallbackWarning here means some op in the chain is not actually native and
    the full logits tensor is crossing the bus anyway -- which would defeat the
    whole exercise.
    """
    import warnings

    from torch_spyre.ops.fallbacks import FallbackWarning

    logits = torch.randn(8, 32000, dtype=torch.float16, device=spyre_device)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FallbackWarning)
        argmax_digits(logits)

    fallbacks = [str(w.message) for w in caught if issubclass(w.category, FallbackWarning)]
    assert not fallbacks, fallbacks
