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

"""Packed prefill: selected token-expert pairs, then the all-expert loop on overflow.

The slot tables and the batched expert math are checked on CPU. The compiled card path is
skipped when this process has no Spyre device.
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from spyre_inference.moe import (
    SpyreMoERecipe,
    SpyreUnquantizedFusedMoEMethod,
    _ExpertBiases,
    _moe_packed_prefill,
    _pack_assignment_tables,
    _packed_capacity,
    _packed_stick_token_chunk,
    _routing_weights,
)

_RECIPES = [
    ("gelu_tanh", "full_softmax"),
    ("silu", "full_softmax"),
    ("silu", "topk_softmax"),
]


def test_stick_token_chunk_stays_under_one_stick(monkeypatch):
    """64 tokens would be one fp16 stick of weights, which cannot fan out over hidden."""
    assert _packed_stick_token_chunk(8, 2816, 64) == 32
    from spyre_inference import moe as moe_module

    monkeypatch.setattr(moe_module, "_PACKED_REDUCE_ELEMENTS", 512)
    # Already under a stick: top_k * hidden = 256 and the budget reduces two tokens.
    assert _packed_stick_token_chunk(2, 128, 64) == 2


def test_packed_capacity_follows_the_grouped_mm_rule():
    """512 and 1024 Gemma-4 prefills both reserve 128 slots; a short batch reserves one stick."""
    assert _packed_capacity(512 * 8, 128) == 128
    assert _packed_capacity(1024 * 8, 128) == 128
    assert _packed_capacity(16, 4) == 64


def test_pack_assignment_tables_place_each_row_and_leave_a_zero_sentinel():
    indices = torch.tensor([[0, 2], [2, 1]], dtype=torch.float16)
    src, dst = _pack_assignment_tables(indices, experts=4, capacity=2)
    assert src.dtype == torch.int32 and dst.dtype == torch.int32
    assert src.tolist() == [0, 4, 3, 4, 1, 2, 4, 4]
    assert dst.tolist() == [0, 4, 5, 2]


def test_pack_assignment_tables_reject_an_overfull_expert_and_a_bad_id():
    crowded = torch.zeros(3, 2, dtype=torch.int64)
    assert _pack_assignment_tables(crowded, experts=4, capacity=1) is None
    with pytest.raises(ValueError, match=r"expert ids"):
        _pack_assignment_tables(torch.tensor([[0, 4]]), experts=4, capacity=2)


def _weights(experts, hidden, inter, dtype, seed, bias):
    generator = torch.Generator().manual_seed(seed)
    gate = torch.randn(experts, hidden, inter, dtype=dtype, generator=generator).mul_(0.1)
    up = torch.randn(experts, hidden, inter, dtype=dtype, generator=generator).mul_(0.1)
    down = torch.randn(experts, inter, hidden, dtype=dtype, generator=generator).mul_(0.1)
    biases = None
    if bias:
        gate_bias = torch.randn(experts, 1, inter, dtype=dtype, generator=generator).mul_(0.1)
        up_bias = torch.randn(experts, 1, inter, dtype=dtype, generator=generator).mul_(0.1)
        down_bias = torch.randn(experts, 1, hidden, dtype=dtype, generator=generator).mul_(0.1)
        biases = _ExpertBiases(gate_bias, up_bias, down_bias, down_bias)
    return gate, up, down, biases


def _reference(x, logits, gate, up, down, top_k, routing, activation, biases):
    weights, indices = _routing_weights(logits, top_k, routing, logits.dtype)
    output = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32)
    for token in range(x.shape[0]):
        row = x[token].float()
        for slot in range(top_k):
            expert = int(indices[token, slot])
            gate_out = row @ gate[expert].float()
            up_out = row @ up[expert].float()
            if biases is not None:
                gate_out = gate_out + biases.gate[expert, 0].float()
                up_out = up_out + biases.up[expert, 0].float()
            activated = _activate(gate_out, up_out, activation)
            expert_out = activated @ down[expert].float()
            if biases is not None:
                expert_out = expert_out + biases.down[expert, 0].float()
            output[token] += expert_out * float(weights[token, slot])
    return output


def _activate(gate_out, up_out, activation):
    if activation == "gelu_tanh":
        return F.gelu(gate_out, approximate="tanh") * up_out
    return F.silu(gate_out) * up_out


def _layer(activation, routing, gate, up, down, biases, top_k):
    return SimpleNamespace(
        spyre_moe_recipe=SpyreMoERecipe(activation, routing),
        spyre_moe_gate=gate,
        spyre_moe_up=up,
        spyre_moe_down=down,
        spyre_moe_biases=biases,
        spyre_moe_route_dtype=gate.dtype,
        spyre_moe_regions={},
        top_k=top_k,
    )


def _install_eager_regions(monkeypatch):
    from spyre_inference import moe as moe_module

    monkeypatch.setattr(moe_module, "_region", lambda layer, name, fn: fn)
    monkeypatch.setattr(moe_module, "_compiler_scopes", lambda: (nullcontext(), nullcontext()))
    monkeypatch.setattr(moe_module, "_name_persistent_dims", lambda *args: None)
    monkeypatch.setattr(moe_module, "_reset_named_dims", lambda: None)


@pytest.mark.parametrize(("activation", "routing"), _RECIPES)
@pytest.mark.parametrize("bias", [False, True], ids=["no_bias", "bias"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
def test_packed_prefill_matches_the_dense_reference(monkeypatch, activation, routing, bias, dtype):
    _install_eager_regions(monkeypatch)
    experts, hidden, inter, top_k, tokens = 4, 8, 6, 2, 5
    gate, up, down, biases = _weights(experts, hidden, inter, dtype, seed=3, bias=bias)
    generator = torch.Generator().manual_seed(11)
    x = torch.randn(tokens, hidden, dtype=dtype, generator=generator)
    logits = torch.randn(tokens, experts, dtype=dtype, generator=generator)
    layer = _layer(activation, routing, gate, up, down, biases, top_k)

    actual = _moe_packed_prefill(layer, x, logits)

    expected = _reference(x, logits, gate, up, down, top_k, routing, activation, biases)
    tolerance = 2e-2 if dtype == torch.float16 else 1e-5
    torch.testing.assert_close(actual.float(), expected, atol=tolerance, rtol=tolerance)
    assert layer.spyre_moe_packed_overflows == 0


def test_packed_slot_sum_matches_across_chunk_boundaries(monkeypatch):
    """A short last chunk has to land on the same rows as one sum over the whole prefill."""
    from spyre_inference import moe as moe_module

    _install_eager_regions(monkeypatch)
    # top_k * hidden = 16, so the budget of 32 reduces two tokens, then two, then one.
    monkeypatch.setattr(moe_module, "_PACKED_REDUCE_ELEMENTS", 32)
    experts, hidden, inter, top_k, tokens = 4, 8, 6, 2, 5
    gate, up, down, biases = _weights(experts, hidden, inter, torch.float32, seed=3, bias=False)
    generator = torch.Generator().manual_seed(11)
    x = torch.randn(tokens, hidden, generator=generator)
    logits = torch.randn(tokens, experts, generator=generator)
    layer = _layer("gelu_tanh", "full_softmax", gate, up, down, biases, top_k)

    actual = _moe_packed_prefill(layer, x, logits)

    expected = _reference(x, logits, gate, up, down, top_k, "full_softmax", "gelu_tanh", None)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_packed_prefill_gathers_hidden_as_separate_sticks(monkeypatch):
    """Hidden wider than one stick is gathered as [rows, nstick, stick], same values."""
    _install_eager_regions(monkeypatch)
    experts, hidden, inter, top_k, tokens = 4, 128, 32, 2, 5
    gate, up, down, biases = _weights(experts, hidden, inter, torch.float16, seed=3, bias=False)
    generator = torch.Generator().manual_seed(11)
    x = torch.randn(tokens, hidden, dtype=torch.float16, generator=generator)
    logits = torch.randn(tokens, experts, dtype=torch.float16, generator=generator)
    layer = _layer("gelu_tanh", "full_softmax", gate, up, down, biases, top_k)

    actual = _moe_packed_prefill(layer, x, logits)

    expected = _reference(x, logits, gate, up, down, top_k, "full_softmax", "gelu_tanh", None)
    torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=2e-2)


def test_packed_stick_sum_matches_across_chunk_boundaries(monkeypatch):
    """A stick-split gather still sums a short last token chunk onto the right rows."""
    from spyre_inference import moe as moe_module

    _install_eager_regions(monkeypatch)
    # top_k * hidden = 256, so a budget of 512 reduces two tokens and leaves a tail of one.
    monkeypatch.setattr(moe_module, "_PACKED_REDUCE_ELEMENTS", 512)
    experts, hidden, inter, top_k, tokens = 4, 128, 32, 2, 5
    gate, up, down, biases = _weights(experts, hidden, inter, torch.float16, seed=3, bias=False)
    generator = torch.Generator().manual_seed(11)
    x = torch.randn(tokens, hidden, dtype=torch.float16, generator=generator)
    logits = torch.randn(tokens, experts, dtype=torch.float16, generator=generator)
    layer = _layer("gelu_tanh", "full_softmax", gate, up, down, biases, top_k)

    actual = _moe_packed_prefill(layer, x, logits)

    expected = _reference(x, logits, gate, up, down, top_k, "full_softmax", "gelu_tanh", None)
    torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=2e-2)


def test_apply_uses_packed_prefill_when_the_flag_is_on(monkeypatch):
    monkeypatch.setenv("SPYRE_MOE_PACKED_PREFILL", "1")
    monkeypatch.setenv("SPYRE_MOE_GATHERED_MAX_TOKENS", "1")
    _install_eager_regions(monkeypatch)
    gate, up, down, biases = _weights(4, 8, 6, torch.float32, seed=5, bias=False)
    x = torch.randn(4, 8)
    logits = torch.randn(4, 4)
    layer = _layer("gelu_tanh", "full_softmax", gate, up, down, biases, top_k=2)
    method = object.__new__(SpyreUnquantizedFusedMoEMethod)

    actual = method.apply_monolithic(layer, x, logits)

    expected = _reference(x, logits, gate, up, down, 2, "full_softmax", "gelu_tanh", None)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_overflow_hands_the_all_expert_loop_a_dense_route(monkeypatch):
    """The fallback scatters on the host. Device scatter of a route row does not lower."""
    from spyre_inference import moe as moe_module

    _install_eager_regions(monkeypatch)
    monkeypatch.setattr(moe_module, "_packed_capacity", lambda assignments, experts: 1)
    seen = {}

    def experts(layer, x, route):
        seen["route"] = route.detach().cpu()
        return torch.zeros_like(x)

    monkeypatch.setattr(moe_module, "_experts", experts)
    experts_n, tokens, top_k = 4, 6, 2
    gate, up, down, biases = _weights(experts_n, 8, 6, torch.float32, seed=1, bias=False)
    logits = torch.full((tokens, experts_n), -5.0)
    logits[:, 0] = 5.0
    layer = _layer("silu", "topk_softmax", gate, up, down, biases, top_k)
    x = torch.randn(tokens, 8)

    output = _moe_packed_prefill(layer, x, logits)

    weights, indices = _routing_weights(logits, top_k, "topk_softmax", logits.dtype)
    expected = torch.zeros(tokens, experts_n)
    expected.scatter_(-1, indices, weights)
    torch.testing.assert_close(seen["route"].squeeze(-1), expected)
    assert output.shape == x.shape
    assert layer.spyre_moe_packed_overflows == 1


def _card_available() -> bool:
    spyre = getattr(torch, "spyre", None)
    return bool(spyre is not None and spyre.is_available())


@pytest.mark.skipif(not _card_available(), reason="needs a Spyre card")
@pytest.mark.parametrize(("activation", "routing"), _RECIPES)
@pytest.mark.parametrize("bias", [False, True], ids=["no_bias", "bias"])
def test_packed_prefill_matches_reference_on_card(activation, routing, bias):
    from torch_spyre._C import get_elem_in_stick

    from spyre_inference.moe import _to_spyre_expert_weight

    dtype = torch.float16
    stick = get_elem_in_stick(dtype)
    experts, hidden, inter, top_k, tokens = 8, stick, stick, 4, 32
    gate, up, down, biases = _weights(experts, hidden, inter, dtype, seed=9, bias=bias)
    generator = torch.Generator().manual_seed(13)
    x = torch.randn(tokens, hidden, dtype=dtype, generator=generator).mul_(0.05)
    logits = torch.randn(tokens, experts, dtype=dtype, generator=generator)
    device_biases = None
    if biases is not None:
        device_biases = _ExpertBiases(
            _to_spyre_expert_weight(biases.gate, ()),
            _to_spyre_expert_weight(biases.up, ()),
            _to_spyre_expert_weight(biases.down, ()),
            _to_spyre_expert_weight(biases.down, ()),
        )
    layer = _layer(
        activation,
        routing,
        _to_spyre_expert_weight(gate, ()),
        _to_spyre_expert_weight(up, ()),
        _to_spyre_expert_weight(down, (), kernel_order=True),
        device_biases,
        top_k,
    )

    actual = _moe_packed_prefill(layer, x.to("spyre"), logits.to("spyre"))

    expected = _reference(x, logits, gate, up, down, top_k, routing, activation, biases)
    torch.testing.assert_close(actual.cpu().float(), expected, atol=2e-2, rtol=2e-2)
    assert "packed" in layer.spyre_moe_regions
    assert layer.spyre_moe_packed_overflows == 0


@pytest.mark.skipif(not _card_available(), reason="needs a Spyre card")
def test_packed_overflow_on_card_matches_reference(monkeypatch):
    from torch_spyre._C import get_elem_in_stick

    from spyre_inference import moe as moe_module
    from spyre_inference.moe import _to_spyre_expert_weight

    monkeypatch.setattr(moe_module, "_packed_capacity", lambda assignments, experts: 1)
    dtype = torch.float16
    stick = get_elem_in_stick(dtype)
    experts, hidden, inter, top_k, tokens = 4, stick, stick, 2, 8
    gate, up, down, _ = _weights(experts, hidden, inter, dtype, seed=2, bias=False)
    x = torch.randn(tokens, hidden, dtype=dtype).mul_(0.05)
    logits = torch.full((tokens, experts), -5.0, dtype=dtype)
    logits[:, 0] = 5.0
    layer = _layer(
        "gelu_tanh",
        "full_softmax",
        _to_spyre_expert_weight(gate, ()),
        _to_spyre_expert_weight(up, ()),
        _to_spyre_expert_weight(down, (), kernel_order=True),
        None,
        top_k,
    )

    actual = _moe_packed_prefill(layer, x.to("spyre"), logits.to("spyre"))

    expected = _reference(x, logits, gate, up, down, top_k, "full_softmax", "gelu_tanh", None)
    torch.testing.assert_close(actual.cpu().float(), expected, atol=2e-2, rtol=2e-2)
    assert layer.spyre_moe_packed_overflows == 1
    assert "experts" in layer.spyre_moe_regions
