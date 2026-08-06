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

"""
Test device_argmax correctness against torch.argmax (CPU + optional Spyre).
"""

import pytest
import torch

from spyre_inference.v1.sample.device_argmax import (
    _RADIX,
    _get_constants,
    _reduce,
    argmax_digits,
    combine_digits,
    greedy_token_ids,
)

# Padded to a multiple of 64 * 32, matching SpyreParallelLMHead's alignment.
GRANITE_VOCAB = 49152
LLAMA_VOCAB, LLAMA_PADDED = 128256, 129024


def reference(logits: torch.Tensor, valid_vocab: int | None = None) -> torch.Tensor:
    """torch.argmax over the unpadded region."""
    if valid_vocab is not None:
        logits = logits[:, :valid_vocab]
    return logits.argmax(dim=-1)


@pytest.mark.parametrize("vocab", [256, 1024, 2048, 32000, GRANITE_VOCAB, LLAMA_VOCAB])
@pytest.mark.parametrize("batch", [1, 8])
def test_matches_torch_argmax(vocab: int, batch: int):
    """Random fp16 logits at realistic vocab sizes match torch.argmax."""
    torch.manual_seed(0)
    logits = torch.randn(batch, vocab, dtype=torch.float16)

    got = greedy_token_ids(logits)

    torch.testing.assert_close(got, reference(logits), atol=0, rtol=0)


def test_index_above_fp16_exact_range():
    """Indices beyond 2048 must round-trip exactly (fp16 index spacing fails)."""
    vocab = 32000
    logits = torch.full((1, vocab), -10.0, dtype=torch.float16)
    for target in (2049, 12345, 31999):
        logits.fill_(-10.0)
        logits[0, target] = 1.0

        assert greedy_token_ids(logits).item() == target


def test_ties_take_lowest_index():
    """Duplicate maxima resolve to the first index, like torch.argmax."""
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
    """Zero padding columns must not win when every real logit is negative."""
    valid, padded = 3000, 4096
    logits = torch.full((4, padded), -5.0, dtype=torch.float16)
    logits[:, valid:] = 0.0
    logits[:, 1234] = -1.0

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
    """Neither digit may leave fp16's exact-integer range."""
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


def test_every_where_operand_is_a_device_tensor():
    """Sentinels and mask fills must be device tensors (aten.where.self only)."""
    logits = torch.randn(2, 4096, dtype=torch.float16)
    hi_pos, lo_pos, valid_mask, *scalars = _get_constants(logits, 4000)

    assert valid_mask is not None, "expected the padded case to build a mask"
    for const in (hi_pos, lo_pos, valid_mask, *scalars):
        assert isinstance(const, torch.Tensor)
        assert const.device == logits.device


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


@pytest.fixture()
def spyre_device():
    """Claim a Spyre card, or skip when hardware/runtime is not usable.

    Soft-gate with ``spyre_hardware_present`` first: ``spyre_available()``'s
    ``randn`` probe can SIGABRT and kill the pytest process.
    """
    import os

    from spyre_testing_plugin.vfio_reaper import spyre_hardware_present

    if not spyre_hardware_present():
        pytest.skip("Spyre hardware not present (no /dev/vfio or AIU_WORLD_SIZE)")

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")

    try:
        import torch_spyre  # noqa: F401

        torch.spyre.set_device(0)
        # empty+fill avoids the randn path that aborted in spyre__normal_.
        torch.empty(1, dtype=torch.float16, device="spyre").fill_(0)
    except Exception as exc:
        pytest.skip(f"Spyre device not usable: {exc}")

    return torch.device("spyre")


@pytest.mark.parametrize("mode", ["eager", "compile"])
def test_on_spyre_matches_cpu_reference(spyre_device, mode):
    """On-device reduction matches torch.argmax on CPU."""
    torch.manual_seed(3)
    logits_cpu = torch.randn(8, 32000, dtype=torch.float16)
    logits = logits_cpu.to(spyre_device)

    if mode == "compile":
        consts = _get_constants(logits, None)
        fn = torch.compile(_reduce, dynamic=False)
        hi, lo = fn(logits, *consts)
    else:
        hi, lo = argmax_digits(logits)

    got = combine_digits(hi.cpu(), lo.cpu())
    torch.testing.assert_close(got, reference(logits_cpu), atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["eager", "compile"])
def test_on_spyre_masks_padding_columns(spyre_device, mode):
    """valid_vocab masking works on-device (production path)."""
    valid, padded = 32000, 32064
    logits_cpu = torch.full((4, padded), -5.0, dtype=torch.float16)
    logits_cpu[:, valid:] = 0.0
    logits_cpu[:, 12345] = -1.0
    logits = logits_cpu.to(spyre_device)

    if mode == "compile":
        consts = _get_constants(logits, valid)
        hi, lo = torch.compile(_reduce, dynamic=False)(logits, *consts)
    else:
        hi, lo = argmax_digits(logits, valid_vocab=valid)

    got = combine_digits(hi.cpu(), lo.cpu())
    assert torch.equal(got, torch.full((4,), 12345, dtype=torch.int64))


def test_on_spyre_does_not_fall_back_to_cpu(spyre_device):
    """No FallbackWarning: the reduction must stay native on Spyre."""
    import warnings

    from torch_spyre.ops.fallbacks import FallbackWarning

    logits = torch.randn(8, 32000, dtype=torch.float16).to(spyre_device)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FallbackWarning)
        argmax_digits(logits)

    fallbacks = [str(w.message) for w in caught if issubclass(w.category, FallbackWarning)]
    assert not fallbacks, fallbacks
